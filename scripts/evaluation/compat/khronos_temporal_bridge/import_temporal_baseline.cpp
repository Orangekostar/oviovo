#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
int bridge_root_fd = -1;

std::vector<std::string> relativeComponents(const std::string& value) {
  const std::filesystem::path path(value);
  if (value.empty() || path.is_absolute()) {
    throw std::runtime_error("declared artifact path must be relative");
  }
  std::vector<std::string> components;
  for (const auto& component : path) {
    const std::string item = component.string();
    if (item.empty() || item == "." || item == "..") {
      throw std::runtime_error(
          "declared artifact path contains a forbidden component");
    }
    components.push_back(item);
  }
  if (components.empty()) {
    throw std::runtime_error("declared artifact path must be relative");
  }
  return components;
}

std::string readDescriptorOnce(int descriptor, const std::string& label) {
  struct stat status {};
  if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
    if (label == "declared artifact") {
      throw std::runtime_error("declared artifact is not a regular file");
    }
    throw std::runtime_error(label + " is not a regular file");
  }
  std::string bytes;
  std::array<char, 65536> buffer{};
  while (true) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count < 0) {
      throw std::runtime_error("failed to read " + label);
    }
    if (count == 0) {
      break;
    }
    bytes.append(buffer.data(), static_cast<size_t>(count));
  }
  return bytes;
}

std::string readRelativeFileOnce(const std::string& value,
                                 const std::string& label) {
  const auto components = relativeComponents(value);
  int directory = dup(bridge_root_fd);
  if (directory < 0) {
    throw std::runtime_error("failed to duplicate bridge root descriptor");
  }
  try {
    for (size_t index = 0; index + 1 < components.size(); ++index) {
      const int next = openat(directory,
                              components[index].c_str(),
                              O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (next < 0) {
        throw std::runtime_error(
            "declared artifact path contains a symlink or invalid directory");
      }
      close(directory);
      directory = next;
    }
    const int descriptor = openat(directory,
                                  components.back().c_str(),
                                  O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) {
      throw std::runtime_error(
          "declared artifact path contains a symlink or missing file");
    }
    close(directory);
    directory = -1;
    try {
      const std::string bytes = readDescriptorOnce(descriptor, label);
      close(descriptor);
      return bytes;
    } catch (...) {
      close(descriptor);
      throw;
    }
  } catch (...) {
    if (directory >= 0) {
      close(directory);
    }
    throw;
  }
}

std::string resolveBridgePath(const std::string& value) {
  const std::string ignored = readRelativeFileOnce(value, "declared artifact");
  static_cast<void>(ignored);
  return (bridge_root / std::filesystem::path(value)).string();
}

Json parseStrictJson(const std::string& bytes, const std::string& label) {
  std::vector<std::set<std::string>> object_keys;
  const auto callback = [&object_keys](int,
                                       Json::parse_event_t event,
                                       Json& parsed) {
    if (event == Json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == Json::parse_event_t::key) {
      if (object_keys.empty() ||
          !object_keys.back().insert(parsed.get<std::string>()).second) {
        throw std::runtime_error("duplicate JSON key");
      }
    } else if (event == Json::parse_event_t::object_end) {
      object_keys.pop_back();
    }
    return true;
  };
  try {
    return Json::parse(bytes, callback);
  } catch (const std::runtime_error&) {
    throw;
  } catch (const std::exception& error) {
    throw std::runtime_error(label + " is invalid JSON: " + error.what());
  }
}

uint32_t rotateRight(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

std::string sha256Hex(std::string_view input) {
  static constexpr std::array<uint32_t, 64> constants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<uint32_t, 8> hash{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                               0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                               0x1f83d9abU, 0x5be0cd19U};
  std::vector<uint8_t> message(input.begin(), input.end());
  const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while (message.size() % 64U != 56U) {
    message.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<uint8_t>(bit_length >> shift));
  }
  for (size_t offset = 0; offset < message.size(); offset += 64U) {
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index) {
      const size_t start = offset + index * 4U;
      words[index] = (static_cast<uint32_t>(message[start]) << 24U) |
                     (static_cast<uint32_t>(message[start + 1]) << 16U) |
                     (static_cast<uint32_t>(message[start + 2]) << 8U) |
                     static_cast<uint32_t>(message[start + 3]);
    }
    for (size_t index = 16; index < words.size(); ++index) {
      const uint32_t s0 = rotateRight(words[index - 15], 7U) ^
                          rotateRight(words[index - 15], 18U) ^
                          (words[index - 15] >> 3U);
      const uint32_t s1 = rotateRight(words[index - 2], 17U) ^
                          rotateRight(words[index - 2], 19U) ^
                          (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint32_t a = hash[0];
    uint32_t b = hash[1];
    uint32_t c = hash[2];
    uint32_t d = hash[3];
    uint32_t e = hash[4];
    uint32_t f = hash[5];
    uint32_t g = hash[6];
    uint32_t h = hash[7];
    for (size_t index = 0; index < words.size(); ++index) {
      const uint32_t sigma1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^
                              rotateRight(e, 25U);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + sigma1 + choice + constants[index] + words[index];
      const uint32_t sigma0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^
                              rotateRight(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const uint32_t value : hash) {
    output << std::setw(8) << value;
  }
  return output.str();
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

Json readDeclaredJson(const Json& record) {
  requireExactFields(record, {"path", "sha256", "byte_count"},
                     "temporal consistency record");
  if (!record.at("path").is_string() || !record.at("sha256").is_string() ||
      !record.at("byte_count").is_number_unsigned()) {
    throw std::runtime_error("temporal consistency record is invalid");
  }
  const std::string bytes =
      readRelativeFileOnce(record.at("path").get<std::string>(),
                           "declared artifact");
  if (bytes.size() != record.at("byte_count").get<uint64_t>()) {
    throw std::runtime_error("declared artifact byte count mismatch");
  }
  if (sha256Hex(bytes) != record.at("sha256").get<std::string>()) {
    throw std::runtime_error("declared artifact SHA256 mismatch");
  }
  return parseStrictJson(bytes, "declared artifact");
}

uint64_t requireUnsigned(const Json& value, const std::string& label) {
  if (!value.is_number_unsigned()) {
    throw std::runtime_error(label + " must be an unsigned integer");
  }
  return value.get<uint64_t>();
}

bool isFiniteNumber(const Json& value) {
  return value.is_number() && !value.is_boolean() &&
         std::isfinite(value.get<double>());
}

void validateTemporalConsistency(const Json& manifest) {
  const Json& record = manifest.at("temporal_consistency_json");
  const Json payload = readDeclaredJson(record);
  requireExactFields(payload,
                     {"schema_version", "frame_coverage", "query_timestamps_ns",
                      "temporal_audit_counts", "samples", "lifecycle_events"},
                     "temporal consistency payload");
  if (!payload.at("schema_version").is_number_unsigned() ||
      payload.at("schema_version").get<uint64_t>() != 1 ||
      !payload.at("frame_coverage").is_array() ||
      !payload.at("query_timestamps_ns").is_array() ||
      !payload.at("temporal_audit_counts").is_object() ||
      !payload.at("samples").is_array() ||
      !payload.at("lifecycle_events").is_array()) {
    throw std::runtime_error("temporal consistency payload is invalid");
  }

  const Json& coverage = payload.at("frame_coverage");
  std::vector<uint64_t> coverage_timestamps;
  std::vector<uint64_t> expected_sample_counts;
  std::vector<uint64_t> expected_event_counts;
  const std::set<std::string> coverage_fields{
      "frame_index", "timestamp_ns", "record_count", "event_count"};
  uint64_t previous_coverage_timestamp = 0;
  for (size_t frame = 0; frame < coverage.size(); ++frame) {
    const Json& item = coverage.at(frame);
    requireExactFields(item, coverage_fields, "temporal consistency coverage");
    const uint64_t frame_index =
        requireUnsigned(item.at("frame_index"), "coverage frame index");
    const uint64_t timestamp =
        requireUnsigned(item.at("timestamp_ns"), "coverage timestamp");
    if (frame_index != frame || timestamp <= previous_coverage_timestamp) {
      throw std::runtime_error("temporal consistency coverage is noncausal");
    }
    coverage_timestamps.push_back(timestamp);
    expected_sample_counts.push_back(
        requireUnsigned(item.at("record_count"), "coverage record count"));
    expected_event_counts.push_back(
        requireUnsigned(item.at("event_count"), "coverage event count"));
    previous_coverage_timestamp = timestamp;
  }
  if (coverage.empty()) {
    throw std::runtime_error("temporal consistency coverage is empty");
  }

  const Json& sidecar_queries = payload.at("query_timestamps_ns");
  const Json& manifest_queries = manifest.at("query_timestamps_ns");
  if (!manifest_queries.is_array() || sidecar_queries != manifest_queries ||
      sidecar_queries.empty()) {
    throw std::runtime_error("temporal consistency query timestamps mismatch");
  }
  uint64_t previous_query = 0;
  for (const auto& query : sidecar_queries) {
    const uint64_t timestamp = requireUnsigned(query, "query timestamp");
    if (timestamp <= previous_query) {
      throw std::runtime_error("temporal consistency query timestamps are noncausal");
    }
    previous_query = timestamp;
  }

  using TemporalKey = std::pair<std::string, uint64_t>;
  std::map<TemporalKey, ExplicitTemporalState> sample_states;
  std::vector<uint64_t> sample_counts(coverage.size(), 0);
  std::optional<std::pair<uint64_t, std::string>> previous_sample_key;
  uint64_t static_count = 0;
  uint64_t dynamic_count = 0;
  uint64_t unknown_count = 0;
  uint64_t invalid_readout_count = 0;
  std::set<std::pair<std::string, uint64_t>> geometry_epochs;
  const std::set<std::string> sample_fields{
      "frame_index",       "timestamp_ns",     "entity_id",
      "centroid_xyz",      "observation_count", "dynamic_state",
      "motion_confidence", "geometry_epoch", "readout_valid"};
  for (const auto& sample : payload.at("samples")) {
    requireExactFields(sample, sample_fields, "temporal consistency sample");
    if (!sample.at("entity_id").is_string() ||
        !sample.at("timestamp_ns").is_number_unsigned() ||
        !sample.at("observation_count").is_number_unsigned() ||
        !sample.at("dynamic_state").is_string() ||
        !sample.at("centroid_xyz").is_array() ||
        sample.at("centroid_xyz").size() != 3 ||
        !isFiniteNumber(sample.at("motion_confidence")) ||
        sample.at("motion_confidence").get<double>() < 0.0 ||
        sample.at("motion_confidence").get<double>() > 1.0 ||
        !sample.at("readout_valid").is_boolean()) {
      throw std::runtime_error("temporal consistency sample is invalid");
    }
    const std::string entity_id = sample.at("entity_id").get<std::string>();
    const uint64_t frame_index =
        requireUnsigned(sample.at("frame_index"), "sample frame index");
    const uint64_t timestamp =
        requireUnsigned(sample.at("timestamp_ns"), "sample timestamp");
    const uint64_t observation_count =
        requireUnsigned(sample.at("observation_count"), "observation count");
    const std::string dynamic_state =
        sample.at("dynamic_state").get<std::string>();
    const auto& xyz = sample.at("centroid_xyz");
    const ExplicitTemporalState sample_state{
        requireUnsigned(sample.at("geometry_epoch"), "sample geometry epoch"),
        sample.at("readout_valid").get<bool>()};
    const std::pair<uint64_t, std::string> canonical_key{frame_index, entity_id};
    if (entity_id.empty() || frame_index >= coverage.size() ||
        timestamp != coverage_timestamps.at(frame_index) || observation_count == 0 ||
        (dynamic_state != "static" && dynamic_state != "dynamic" &&
         dynamic_state != "unknown") ||
        !isFiniteNumber(xyz.at(0)) || !isFiniteNumber(xyz.at(1)) ||
        !isFiniteNumber(xyz.at(2)) ||
        (previous_sample_key && canonical_key <= *previous_sample_key) ||
        !sample_states.emplace(TemporalKey{entity_id, frame_index}, sample_state)
             .second) {
      throw std::runtime_error("temporal consistency samples are not canonical");
    }
    previous_sample_key = canonical_key;
    ++sample_counts.at(frame_index);
    static_count += dynamic_state == "static";
    dynamic_count += dynamic_state == "dynamic";
    unknown_count += dynamic_state == "unknown";
    invalid_readout_count += !sample_state.readout_valid;
    geometry_epochs.emplace(entity_id, sample_state.geometry_epoch);
  }
  if (sample_counts != expected_sample_counts) {
    throw std::runtime_error("temporal consistency sample coverage mismatch");
  }

  const std::set<std::string> event_fields{
      "frame_index", "timestamp_ns", "entity_id",      "before",
      "after",       "evidence",     "geometry_epoch", "readout_valid"};
  std::set<TemporalKey> event_keys;
  std::vector<uint64_t> event_counts(coverage.size(), 0);
  std::optional<std::pair<uint64_t, std::string>> previous_event_key;
  for (const auto& event : payload.at("lifecycle_events")) {
    requireExactFields(event, event_fields, "temporal consistency lifecycle event");
    if (!event.at("entity_id").is_string() ||
        !event.at("timestamp_ns").is_number_unsigned() ||
        !event.at("before").is_string() || !event.at("after").is_string() ||
        !event.at("evidence").is_string() ||
        !event.at("readout_valid").is_boolean()) {
      throw std::runtime_error("temporal consistency lifecycle event is invalid");
    }
    const std::string entity_id = event.at("entity_id").get<std::string>();
    const uint64_t frame_index =
        requireUnsigned(event.at("frame_index"), "event frame index");
    const uint64_t timestamp =
        requireUnsigned(event.at("timestamp_ns"), "event timestamp");
    const uint64_t event_epoch =
        requireUnsigned(event.at("geometry_epoch"), "event geometry epoch");
    const bool event_readout_valid = event.at("readout_valid").get<bool>();
    const std::string before = event.at("before").get<std::string>();
    const std::string after = event.at("after").get<std::string>();
    const std::string evidence = event.at("evidence").get<std::string>();
    const TemporalKey key{entity_id, frame_index};
    const std::pair<uint64_t, std::string> canonical_key{frame_index, entity_id};
    const std::set<std::string> lifecycle_states{"active", "uncertain", "dormant"};
    const std::set<std::string> evidence_values{
        "present", "visible_absent", "occluded", "out_of_view", "depth_unknown"};
    if (entity_id.empty() || frame_index >= coverage.size() ||
        timestamp != coverage_timestamps.at(frame_index) ||
        !lifecycle_states.count(before) || !lifecycle_states.count(after) ||
        !evidence_values.count(evidence) ||
        (previous_event_key && canonical_key <= *previous_event_key) ||
        !event_keys.emplace(key).second) {
      throw std::runtime_error(
          "temporal consistency lifecycle events are not canonical");
    }
    previous_event_key = canonical_key;
    ++event_counts.at(frame_index);
    geometry_epochs.emplace(entity_id, event_epoch);
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
  if (event_counts != expected_event_counts) {
    throw std::runtime_error("temporal consistency lifecycle coverage mismatch");
  }

  const Json& sidecar_audit = payload.at("temporal_audit_counts");
  const Json& manifest_audit = manifest.at("temporal_audit_counts");
  const std::set<std::string> audit_fields{
      "static_sample_count",       "dynamic_sample_count",
      "unknown_sample_count",      "missing_frame_count",
      "lifecycle_transition_count", "geometry_epoch_count",
      "invalid_readout_sample_count"};
  requireExactFields(sidecar_audit, audit_fields, "temporal consistency audit");
  requireExactFields(manifest_audit, audit_fields, "bridge manifest audit");
  const std::map<std::string, uint64_t> computed_audit{
      {"static_sample_count", static_count},
      {"dynamic_sample_count", dynamic_count},
      {"unknown_sample_count", unknown_count},
      {"missing_frame_count", 0},
      {"lifecycle_transition_count", event_keys.size()},
      {"geometry_epoch_count", geometry_epochs.size()},
      {"invalid_readout_sample_count", invalid_readout_count}};
  for (const auto& [field, expected] : computed_audit) {
    if (requireUnsigned(sidecar_audit.at(field), "sidecar audit count") != expected ||
        requireUnsigned(manifest_audit.at(field), "manifest audit count") != expected) {
      throw std::runtime_error("temporal consistency audit mismatch");
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
  const std::string trajectory_path =
      entry.at("trajectory_json").get<std::string>();
  const std::string trajectory_bytes =
      readRelativeFileOnce(trajectory_path, "bridge trajectory");
  const Json trajectory =
      parseStrictJson(trajectory_bytes, "bridge trajectory");
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

struct ValidatedBridgeInputs {
  std::vector<DynamicSceneGraph::Ptr> graphs;
  std::vector<uint64_t> timestamps;
};

ValidatedBridgeInputs validateBridgeInputs(const Json& manifest) {
  validateTemporalConsistency(manifest);
  const auto& expected_timestamps = manifest.at("query_timestamps_ns");
  const auto& checkpoints = manifest.at("checkpoints");
  if (!expected_timestamps.is_array() || !checkpoints.is_array() ||
      checkpoints.size() != expected_timestamps.size()) {
    throw std::runtime_error("checkpoint and query timestamp counts disagree");
  }
  ValidatedBridgeInputs result;
  uint64_t previous_timestamp_ns = 0;
  for (size_t index = 0; index < checkpoints.size(); ++index) {
    const Json& checkpoint = checkpoints.at(index);
    const uint64_t timestamp_ns =
        requireUnsigned(checkpoint.at("timestamp_ns"), "checkpoint timestamp");
    if (timestamp_ns <= previous_timestamp_ns ||
        timestamp_ns != requireUnsigned(expected_timestamps.at(index),
                                        "query timestamp")) {
      throw std::runtime_error("checkpoint timestamps are not strictly increasing");
    }
    result.graphs.push_back(buildGraph(checkpoint));
    result.timestamps.push_back(timestamp_ns);
    previous_timestamp_ns = timestamp_ns;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " bridge_manifest.json output_directory\n";
    return 2;
  }
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  std::filesystem::path output;
  bool output_created = false;
  try {
    const std::filesystem::path manifest_path = std::filesystem::absolute(argv[1]);
    bridge_root = manifest_path.parent_path();
    bridge_root_fd = open(bridge_root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (bridge_root_fd < 0) {
      throw std::runtime_error("failed to open bridge root directory");
    }
    const int manifest_descriptor =
        open(manifest_path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (manifest_descriptor < 0) {
      throw std::runtime_error("bridge manifest must not be a symlink");
    }
    std::string manifest_bytes;
    try {
      manifest_bytes = readDescriptorOnce(manifest_descriptor, "bridge manifest");
      close(manifest_descriptor);
    } catch (...) {
      close(manifest_descriptor);
      throw;
    }
    const Json manifest = parseStrictJson(manifest_bytes, "bridge manifest");
    if (manifest.value("dataset", "") != "TESSE-CD") {
      throw std::runtime_error("bridge manifest is not TESSE-CD");
    }
    if (manifest.value("method", "") != "OVIV2") {
      throw std::runtime_error("bridge manifest is not OVIV2");
    }
    if (manifest.value("mode", "") != "temporal_checkpoints") {
      throw std::runtime_error("bridge manifest is not temporal_checkpoints");
    }
    const ValidatedBridgeInputs validated = validateBridgeInputs(manifest);
    output = std::filesystem::path(argv[2]);
    if (!std::filesystem::create_directory(output)) {
      throw std::runtime_error("output directory already exists");
    }
    output_created = true;

    hydra::PipelineConfig global_config;
    hydra::GlobalInfo::reset();
    hydra::GlobalInfo::init(global_config);
    khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
    for (size_t index = 0; index < validated.graphs.size(); ++index) {
      map.update(validated.graphs.at(index), validated.timestamps.at(index));
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
    if (output_created) {
      std::filesystem::remove_all(output);
    }
    if (bridge_root_fd >= 0) {
      close(bridge_root_fd);
      bridge_root_fd = -1;
    }
    std::cerr << error.what() << std::endl;
    return 1;
  }
  if (bridge_root_fd >= 0) {
    close(bridge_root_fd);
    bridge_root_fd = -1;
  }
  return 0;
}
