#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <khronos/common/common_types.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <nlohmann/json.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

namespace {

using Json = nlohmann::json;
using khronos::DynamicSceneGraph;
using khronos::KhronosObjectAttributes;
using khronos::NodeSymbol;
using khronos::Point;
using khronos::Points;

std::filesystem::path bridge_root;

std::string resolveBridgePath(const std::string& value) {
  const std::filesystem::path path(value);
  return (path.is_absolute() ? path : bridge_root / path).lexically_normal().string();
}

Json loadJson(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open JSON: " + path);
  }
  Json payload;
  input >> payload;
  return payload;
}

void requireExactFields(const Json& value,
                        const std::set<std::string>& expected,
                        const std::string& label) {
  if (!value.is_object() || value.size() != expected.size()) {
    throw std::runtime_error(label + " fields are invalid");
  }
  for (const auto& field : expected) {
    if (!value.contains(field)) {
      throw std::runtime_error(label + " fields are invalid");
    }
  }
}

struct ExplicitTemporalState {
  uint64_t geometry_epoch;
  bool readout_valid;
};

void validateTemporalConsistency(const Json& manifest) {
  const Json& record = manifest.at("temporal_consistency_json");
  requireExactFields(record, {"path", "sha256", "byte_count"},
                     "temporal consistency record");
  const Json payload =
      loadJson(resolveBridgePath(record.at("path").get<std::string>()));
  requireExactFields(payload, {"schema_version", "samples", "lifecycle_events"},
                     "temporal consistency payload");
  if (payload.at("schema_version").get<uint32_t>() != 1 ||
      !payload.at("samples").is_array() ||
      !payload.at("lifecycle_events").is_array()) {
    throw std::runtime_error("temporal consistency payload is invalid");
  }

  using TemporalKey = std::pair<std::string, uint64_t>;
  std::map<TemporalKey, ExplicitTemporalState> sample_states;
  const std::set<std::string> sample_fields{
      "frame_index",       "timestamp_ns",     "entity_id",
      "centroid_xyz",      "observation_count", "dynamic_state",
      "motion_confidence", "geometry_epoch", "readout_valid"};
  for (const auto& sample : payload.at("samples")) {
    requireExactFields(sample, sample_fields, "temporal consistency sample");
    if (!sample.at("entity_id").is_string() ||
        !sample.at("frame_index").is_number_unsigned() ||
        !sample.at("geometry_epoch").is_number_unsigned() ||
        !sample.at("readout_valid").is_boolean()) {
      throw std::runtime_error("temporal consistency sample is invalid");
    }
    const std::string entity_id = sample.at("entity_id").get<std::string>();
    const uint64_t frame_index = sample.at("frame_index").get<uint64_t>();
    const ExplicitTemporalState sample_state{
        sample.at("geometry_epoch").get<uint64_t>(),
        sample.at("readout_valid").get<bool>()};
    if (entity_id.empty() ||
        !sample_states.emplace(TemporalKey{entity_id, frame_index}, sample_state)
             .second) {
      throw std::runtime_error("duplicate or invalid temporal consistency sample");
    }
  }

  const std::set<std::string> event_fields{
      "frame_index", "timestamp_ns", "entity_id",      "before",
      "after",       "evidence",     "geometry_epoch", "readout_valid"};
  std::set<TemporalKey> event_keys;
  for (const auto& event : payload.at("lifecycle_events")) {
    requireExactFields(event, event_fields, "temporal consistency lifecycle event");
    if (!event.at("entity_id").is_string() ||
        !event.at("frame_index").is_number_unsigned() ||
        !event.at("geometry_epoch").is_number_unsigned() ||
        !event.at("readout_valid").is_boolean()) {
      throw std::runtime_error("temporal consistency lifecycle event is invalid");
    }
    const std::string entity_id = event.at("entity_id").get<std::string>();
    const uint64_t frame_index = event.at("frame_index").get<uint64_t>();
    const uint64_t event_epoch = event.at("geometry_epoch").get<uint64_t>();
    const bool event_readout_valid = event.at("readout_valid").get<bool>();
    const TemporalKey key{entity_id, frame_index};
    if (entity_id.empty() || !event_keys.emplace(key).second) {
      throw std::runtime_error(
          "duplicate or invalid temporal consistency lifecycle event");
    }
    const auto sample_it = sample_states.find(key);
    if (sample_it == sample_states.end()) {
      continue;
    }
    const ExplicitTemporalState& sample_state = sample_it->second;
    if (sample_state.geometry_epoch != event_epoch ||
        sample_state.readout_valid != event_readout_valid) {
      throw std::runtime_error("sample/event temporal state conflict");
    }
  }
}

Points loadPoints(const std::string& path) {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  if (pcl::io::loadPLYFile(path, cloud) != 0) {
    throw std::runtime_error("failed to load PLY: " + path);
  }
  Points points;
  points.reserve(cloud.size());
  for (const auto& point : cloud) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      throw std::runtime_error("PLY contains non-finite points: " + path);
    }
    points.emplace_back(point.x, point.y, point.z);
  }
  return points;
}

Point centroid(const Points& points) {
  Point result = Point::Zero();
  for (const auto& point : points) {
    result += point;
  }
  return result / static_cast<float>(points.size());
}

void loadTrajectory(const Json& entry,
                    uint64_t query_timestamp_ns,
                    KhronosObjectAttributes& attributes) {
  const Json trajectory =
      loadJson(resolveBridgePath(entry.at("trajectory_json").get<std::string>()));
  const bool eligible = entry.at("dynamic_track_eligible").get<bool>();
  const Json state = entry.at("dynamic_state_at_query");
  const bool explicit_dynamic = state.is_string() && state.get<std::string>() == "dynamic";
  if (eligible != explicit_dynamic) {
    throw std::runtime_error("bridge dynamic eligibility disagrees with explicit state");
  }
  if (!eligible) {
    if (!trajectory.empty()) {
      throw std::runtime_error("ineligible object has a synthesized trajectory");
    }
    return;
  }
  if (trajectory.empty() ||
      trajectory.back().at("dynamic_state").get<std::string>() != "dynamic") {
    throw std::runtime_error("eligible trajectory does not end in explicit dynamic state");
  }

  uint64_t previous = 0;
  for (const auto& sample : trajectory) {
    const uint64_t timestamp_ns = sample.at("timestamp_ns").get<uint64_t>();
    if (timestamp_ns <= previous || timestamp_ns > query_timestamp_ns) {
      throw std::runtime_error("trajectory timestamps are not causal and increasing");
    }
    const auto xyz = sample.at("centroid_xyz");
    if (xyz.size() != 3) {
      throw std::runtime_error("trajectory centroid is not xyz");
    }
    attributes.trajectory_timestamps.push_back(timestamp_ns);
    attributes.trajectory_positions.emplace_back(
        xyz.at(0).get<float>(), xyz.at(1).get<float>(), xyz.at(2).get<float>());
    previous = timestamp_ns;
  }
}

DynamicSceneGraph::Ptr buildGraph(const Json& checkpoint) {
  const std::map<std::string, spark_dsg::LayerKey> layers{
      {khronos::DsgLayers::OBJECTS, 2},
      {khronos::DsgLayers::PLACES, 3},
      {khronos::DsgLayers::ROOMS, 4},
      {khronos::DsgLayers::BUILDINGS, 5},
  };
  auto graph = DynamicSceneGraph::fromNames(layers);
  const uint64_t timestamp_ns = checkpoint.at("timestamp_ns").get<uint64_t>();

  const Points background =
      loadPoints(resolveBridgePath(checkpoint.at("background_ply").get<std::string>()));
  auto mesh = std::make_shared<spark_dsg::Mesh>();
  mesh->resizeVertices(background.size());
  for (size_t index = 0; index < background.size(); ++index) {
    mesh->setPos(index, background[index]);
  }
  graph->setMesh(mesh);

  for (const auto& entry : checkpoint.at("objects")) {
    const Points points =
        loadPoints(resolveBridgePath(entry.at("points_ply").get<std::string>()));
    if (points.empty()) {
      throw std::runtime_error("current object has no points");
    }
    const Point center = centroid(points);
    auto attributes = std::make_unique<KhronosObjectAttributes>();
    attributes->name = entry.at("entity_id").get<std::string>();
    attributes->semantic_label = entry.at("semantic_label").get<uint32_t>();
    attributes->position = center.cast<double>();
    attributes->bounding_box = spark_dsg::BoundingBox(points);
    attributes->bounding_box.world_P_center = center;
    attributes->first_observed_ns =
        entry.at("first_observed_ns").get<std::vector<uint64_t>>();
    attributes->last_observed_ns =
        entry.at("last_observed_ns").get<std::vector<uint64_t>>();
    if (attributes->first_observed_ns.empty() ||
        attributes->first_observed_ns.size() !=
            attributes->last_observed_ns.size()) {
      throw std::runtime_error("presence interval endpoint vectors are invalid");
    }
    if (!std::is_sorted(attributes->first_observed_ns.begin(),
                        attributes->first_observed_ns.end()) ||
        !std::is_sorted(attributes->last_observed_ns.begin(),
                        attributes->last_observed_ns.end())) {
      throw std::runtime_error("presence interval endpoints are not sorted");
    }
    attributes->mesh.resizeVertices(points.size());
    for (size_t point_index = 0; point_index < points.size(); ++point_index) {
      attributes->mesh.setPos(point_index, points[point_index] - center);
    }
    loadTrajectory(entry, timestamp_ns, *attributes);
    graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                       NodeSymbol('O', entry.at("node_index").get<size_t>()),
                       std::move(attributes));
  }
  return graph;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " bridge_manifest.json output_directory\n";
    return 2;
  }
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  try {
    const std::filesystem::path manifest_path = std::filesystem::absolute(argv[1]);
    bridge_root = manifest_path.parent_path();
    const Json manifest = loadJson(manifest_path.string());
    if (manifest.value("dataset", "") != "TESSE-CD") {
      throw std::runtime_error("bridge manifest is not TESSE-CD");
    }
    if (manifest.value("method", "") != "OVIV2") {
      throw std::runtime_error("bridge manifest is not OVIV2");
    }
    if (manifest.value("mode", "") != "temporal_checkpoints") {
      throw std::runtime_error("bridge manifest is not temporal_checkpoints");
    }
    validateTemporalConsistency(manifest);
    const std::filesystem::path output(argv[2]);
    if (std::filesystem::exists(output)) {
      throw std::runtime_error("output directory already exists");
    }
    std::filesystem::create_directories(output);

    hydra::PipelineConfig global_config;
    hydra::GlobalInfo::reset();
    hydra::GlobalInfo::init(global_config);
    khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
    uint64_t previous_timestamp_ns = 0;
    const auto& expected_timestamps = manifest.at("query_timestamps_ns");
    const auto& checkpoints = manifest.at("checkpoints");
    if (checkpoints.size() != expected_timestamps.size()) {
      throw std::runtime_error("checkpoint and query timestamp counts disagree");
    }
    size_t checkpoint_index = 0;
    for (const auto& checkpoint : checkpoints) {
      const uint64_t timestamp_ns = checkpoint.at("timestamp_ns").get<uint64_t>();
      if (timestamp_ns <= previous_timestamp_ns ||
          timestamp_ns != expected_timestamps.at(checkpoint_index).get<uint64_t>()) {
        throw std::runtime_error("checkpoint timestamps are not strictly increasing");
      }
      map.update(buildGraph(checkpoint), timestamp_ns);
      previous_timestamp_ns = timestamp_ns;
      ++checkpoint_index;
    }
    map.finalize();
    if (!map.save((output / "final.4dmap").string())) {
      throw std::runtime_error("failed to save final.4dmap");
    }
    std::ofstream timestamps_output(output / "map_timestamps.json");
    timestamps_output << Json(map.stamps()).dump(2) << "\n";
    std::ofstream log(output / "experiment_log.txt");
    log << "[FLAG] [Experiment Finished Cleanly] \n";
    std::ofstream provenance(output / "bridge_manifest_path.txt");
    provenance << std::filesystem::absolute(argv[1]).string() << "\n";
    std::cout << "Wrote " << map.numTimeSteps() << " causal states to " << output
              << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  return 0;
}
