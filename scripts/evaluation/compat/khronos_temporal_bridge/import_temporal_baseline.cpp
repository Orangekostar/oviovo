#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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
  if (!eligible) {
    if (!trajectory.empty()) {
      throw std::runtime_error("ineligible object has a synthesized trajectory");
    }
    return;
  }
  if (trajectory.size() < 2) {
    throw std::runtime_error("eligible dynamic track has fewer than two native samples");
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
