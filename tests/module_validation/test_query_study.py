"""Actual capture adapter and streaming replay keep unknown evidence inaccessible."""

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from src.static_ovmap.module_validation.assets import sha256_file
from src.static_ovmap.module_validation.native_capture import (
    RegionRequest,
    _array_digest,
)
from src.static_ovmap.module_validation.query_state import (
    AcquisitionPayload,
    FeatureStore,
)
from src.static_ovmap.module_validation.query_study import (
    CapturedFrames,
    replay_captured,
)
from tests.module_validation.test_query_gain_policy import _candidate
from tests.module_validation.test_query_lineage import _snapshot


def test_capture_reader_builds_exact_native_requests_without_reading_final_or_future(tmp_path):
    target = np.ones((40, 40), bool)
    Image.fromarray(np.ones((40, 40), np.uint16)).save(tmp_path / "owners.png")
    Image.fromarray(np.zeros((40, 40), np.uint16)).save(tmp_path / "local.png")
    np.savez(tmp_path / "depth.npz", depth_m=np.ones((40, 40), np.float32))
    (tmp_path / "state.json").write_text(json.dumps({"native_state": _snapshot({11: 1})}))
    request = RegionRequest("scene-a", 10, "owner:1", ("segment:11", "owner:1"), "map",
        _array_digest(target), (0, 0, 39, 39), _array_digest(target), 1600,
        "native_global_bbox_union_exclusive_upper_v1", 0, "a" * 64)
    frame = {"scene_id": "scene-a", "frame_id": 10, "requests": [request.to_dict()],
        "pose_c2w": np.eye(4).tolist(), "intrinsics": np.eye(3).tolist(),
        "global_owner_path": "owners.png", "global_owner_sha256": sha256_file(tmp_path / "owners.png"),
        "panoptic_path": "local.png", "panoptic_sha256": sha256_file(tmp_path / "local.png"),
        "depth_path": "depth.npz", "depth_sha256": sha256_file(tmp_path / "depth.npz"),
        "native_state": {"path": "state.json", "sha256": sha256_file(tmp_path / "state.json")}}
    manifest = {"scene_id": "scene-a", "scheduled_frame_ids": [0, 10, 20],
        "frames": [frame, {"frame_id": 20, "native_state": "FORBIDDEN_FUTURE"}],
        "surface": "FORBIDDEN_FINAL", "alias_table": "FORBIDDEN_FINAL"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    frames = CapturedFrames(path)
    assert frames.load(0) is None
    current = frames.load(1)
    assert current["candidates"][0].request_id == request.request_id
    assert current["candidates"][0].majority_entity_zero
    assert current["candidates"][0].frame_index == 1
    assert current["candidates"][0].overlap_pixels == 1600
    # ScanNet scene0534 frame1064 exposed a positive raycast owner whose
    # final-export confidence gate gave owner zero. Never use that gate for Q.
    snapshot = _snapshot({11: 0})
    snapshot["association"] = {"label_mapping_count_threshold_factor": .1}
    (tmp_path / "state.json").write_text(json.dumps({"native_state": snapshot}))
    frame["native_state"]["sha256"] = sha256_file(tmp_path / "state.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="threshold differs"):
        CapturedFrames(path).load(1)
    snapshot["label_instances"][0]["raycast_instance_label"] = 1
    (tmp_path / "state.json").write_text(json.dumps({"native_state": snapshot}))
    frame["native_state"]["sha256"] = sha256_file(tmp_path / "state.json")
    path.write_text(json.dumps(manifest))
    current = CapturedFrames(path).load(1)
    assert current["snapshot"]["label_instances"][0]["instance_label"] == 1
    assert current["candidates"][0].request_id == request.request_id
    assert json.loads((tmp_path / "state.json").read_text())["native_state"]["label_instances"][0]["instance_label"] == 0


def test_captured_replay_sentinel_barrier_and_current_merge():
    class Frames:
        scene_id = "scene-a"
        schedule = (0, 1, 2)

        def load(self, index):
            if index == 1:
                return None
            candidates = [replace(_candidate(owner, f"{index}-{owner}", overlap), frame_index=index)
                          for owner, overlap in ((1, 12), (2, 10))] if index == 0 else []
            return {"candidates": candidates, "snapshot": _snapshot({11: 1, 22: 2} if index == 0 else {11: 2, 22: 2})}

    def run(unpaid):
        calls = []
        store = FeatureStore(lambda candidate: calls.append(candidate.request_id) or AcquisitionPayload(np.array([1., 0.]), 6, .1),
                             initial_cache={"0-2": AcquisitionPayload(unpaid, 6, 0.)})
        result = replay_captured(Frames(), "Q_AREA", store, np.eye(2), budget=3)
        return result, calls

    normal, calls = run(np.array([0., 1.]))
    sentinel, _ = run(np.array([1e30, -1e30]))
    assert calls == ["0-1"]
    assert normal["decisions"] == sentinel["decisions"]
    assert normal["state"].logical_ledger.attempts == 1
    assert [row.request_id for row in normal["state"].object_state(2).features] == ["0-1"]
    assert normal["decisions"][1]["quota"] == 1
    assert normal["decisions"][2]["quota"] == 2
