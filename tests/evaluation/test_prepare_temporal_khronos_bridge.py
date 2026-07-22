from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.evaluation.prepare_temporal_khronos_bridge as bridge_module
from scripts.evaluation.prepare_temporal_khronos_bridge import (
    prepare_temporal_bridge,
    validate_temporal_bridge_manifest,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


@pytest.fixture(autouse=True)
def _bind_fixture_label_space(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_contents = (
        "label_names:\n  - {label: 0, name: Unknown}\n  - {label: 5, name: Chair}\n",
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
    )
    monkeypatch.setattr(
        bridge_module,
        "OFFICIAL_LABEL_SPACE_SHA256",
        {
            "apartment": frozenset(
                hashlib.sha256(content.encode("utf-8")).hexdigest()
                for content in fixture_contents
            ),
            "office": frozenset(),
        },
        raising=False,
    )


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _entity(entity_id: str, x: float) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([[x, 0.0, 1.0], [x + 0.1, 0.0, 1.0]], dtype=np.float32),
        semantic_embedding=None,
        semantic_label="Chair",
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=x,
        metadata={"entity_type": "object"},
    )


def _write_temporal_fixture(root: Path) -> Path:
    root.mkdir()
    schedule = root / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_id": "tesse_cd_causal_schedule_v2",
                "dataset": "TESSE-CD",
                "method_predictions_used": False,
                "parameters": {"frame_indexing": "zero_based"},
                "scenes": {
                    "apartment": {
                        "frame_count": 5,
                        "entries": [
                            {"frame_index": 0, "timestamp_ns": 100},
                            {"frame_index": 2, "timestamp_ns": 300},
                            {"frame_index": 4, "timestamp_ns": 500},
                        ],
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoints = [
        (100, [_entity("r-reappear", 1), _entity("sparse", 2), _entity("z-first", 3)]),
        (300, [_entity("a-later", 4), _entity("sparse", 5), _entity("z-first", 6)]),
        (500, [_entity("a-later", 7), _entity("r-reappear", 8), _entity("sparse", 9)]),
    ]
    checkpoint_records = []
    for frame, (timestamp, entities) in enumerate(checkpoints):
        checkpoint_root = root / "checkpoints" / f"{frame:08d}"
        snapshot = MapSnapshot(
            method="OVIV2",
            scene_id="apartment",
            timestamp=float(timestamp),
            entities=entities,
            background_xyz=np.asarray([[frame, 0.0, 0.0]], dtype=np.float32),
            scope="current",
        )
        paths = write_map_snapshot(snapshot, checkpoint_root)
        checkpoint_records.append(
            {
                "frame_index": frame * 2,
                "timestamp_ns": timestamp,
                "consumed_through_frame": frame * 2,
                "consumed_through_frame_exclusive": frame * 2 + 1,
                "snapshot": _record(paths["snapshot"], relative_to=root),
                "entities": _record(paths["entities"], relative_to=root),
            }
        )

    trajectories = root / "trajectories.jsonl"
    trajectory_rows = [
        (0, 100, "r-reappear", [1.0, 0.0, 1.0]),
        (0, 100, "sparse", [2.0, 0.0, 1.0]),
        (0, 100, "z-first", [3.0, 0.0, 1.0]),
        (1, 200, "z-first", [3.5, 0.0, 1.0]),
        (2, 300, "a-later", [4.0, 0.0, 1.0]),
        (2, 300, "z-first", [6.0, 0.0, 1.0]),
        (3, 400, "a-later", [5.5, 0.0, 1.0]),
        (4, 500, "a-later", [7.0, 0.0, 1.0]),
        (4, 500, "r-reappear", [8.0, 0.0, 1.0]),
    ]
    trajectories.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "entity_id": entity_id,
                    "centroid_xyz": centroid,
                },
                sort_keys=True,
            )
            + "\n"
            for frame, timestamp, entity_id, centroid in trajectory_rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoints",
        "method": "OVIV2",
        "scene": "apartment",
        "sources": {"schedule": _record(schedule, relative_to=root)},
        "checkpoints": checkpoint_records,
        "entity_lifecycles": [
            {
                "entity_id": "r-reappear",
                "semantic_label": "Chair",
                "entity_type": "object",
                "presence_intervals": [
                    {
                        "first_frame_index": 0,
                        "first_timestamp_ns": 100,
                        "last_frame_index": 0,
                        "last_timestamp_ns": 100,
                    },
                    {
                        "first_frame_index": 4,
                        "first_timestamp_ns": 500,
                        "last_frame_index": 4,
                        "last_timestamp_ns": 500,
                    },
                ],
            },
            {
                "entity_id": "sparse",
                "semantic_label": "Chair",
                "entity_type": "object",
                "presence_intervals": [
                    {
                        "first_frame_index": 0,
                        "first_timestamp_ns": 100,
                        "last_frame_index": 4,
                        "last_timestamp_ns": 500,
                    }
                ],
            },
            {
                "entity_id": "z-first",
                "semantic_label": "Chair",
                "entity_type": "object",
                "presence_intervals": [
                    {
                        "first_frame_index": 0,
                        "first_timestamp_ns": 100,
                        "last_frame_index": 2,
                        "last_timestamp_ns": 300,
                    }
                ],
            },
            {
                "entity_id": "a-later",
                "semantic_label": "Chair",
                "entity_type": "object",
                "presence_intervals": [
                    {
                        "first_frame_index": 2,
                        "first_timestamp_ns": 300,
                        "last_frame_index": 4,
                        "last_timestamp_ns": 500,
                    }
                ],
            },
        ],
        "trajectories": _record(trajectories, relative_to=root),
    }
    manifest_path = root / "temporal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_prepares_stable_symbols_intervals_and_causal_native_tracks(
    tmp_path: Path,
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names:\n  - {label: 0, name: Unknown}\n  - {label: 5, name: Chair}\n",
        encoding="utf-8",
    )

    manifest_path = prepare_temporal_bridge(
        temporal, labels, tmp_path / "bridge"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "temporal_checkpoints"
    assert manifest["dataset"] == "TESSE-CD"
    assert manifest["method"] == "OVIV2"
    assert manifest["display_mode"] == "online"
    assert manifest["query_timestamps_ns"] == [100, 300, 500]
    symbols = {
        entry["entity_id"]: entry["node_symbol"]
        for entry in manifest["symbol_assignments"]
    }
    assert symbols == {
        "r-reappear": "O0",
        "sparse": "O1",
        "z-first": "O2",
        "a-later": "O3",
    }
    assignments = {
        entry["entity_id"]: entry for entry in manifest["symbol_assignments"]
    }
    assert assignments["r-reappear"]["first_observed_ns"] == [100, 500]
    assert assignments["r-reappear"]["last_observed_ns"] == [300]
    assert assignments["r-reappear"]["presence_intervals"] == [
        {"start_ns": 100, "end_ns_exclusive": 300},
        {"start_ns": 500, "end_ns_exclusive": None},
    ]
    assert assignments["z-first"]["last_observed_ns"] == [500]
    assert assignments["a-later"]["first_observed_ns"] == [300]

    first_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][0]["objects"]
    }
    assert first_objects["r-reappear"]["node_symbol"] == "O0"
    assert first_objects["r-reappear"]["trajectory_sample_count"] == 0
    middle_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][1]["objects"]
    }
    assert "r-reappear" not in middle_objects
    assert middle_objects["z-first"]["trajectory_sample_count"] == 3
    assert middle_objects["a-later"]["trajectory_sample_count"] == 0
    final_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][2]["objects"]
    }
    assert final_objects["r-reappear"]["node_symbol"] == "O0"
    assert final_objects["r-reappear"]["first_observed_ns"] == [100, 500]
    assert final_objects["r-reappear"]["last_observed_ns"] == [300]
    assert final_objects["sparse"]["dynamic_track_eligible"] is False
    assert final_objects["sparse"]["trajectory_sample_count"] == 0

    for checkpoint in manifest["checkpoints"]:
        query = checkpoint["timestamp_ns"]
        for obj in checkpoint["objects"]:
            assert not Path(obj["trajectory_json"]).is_absolute()
            assert not Path(obj["points_ply"]).is_absolute()
            trajectory = json.loads(
                (manifest_path.parent / obj["trajectory_json"]).read_text(
                    encoding="utf-8"
                )
            )
            assert all(sample["timestamp_ns"] <= query for sample in trajectory)
            for key in ("points_ply", "trajectory_json"):
                path = manifest_path.parent / obj[key]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == obj[
                    key.replace("_ply", "_sha256").replace("_json", "_sha256")
                ]
        assert not Path(checkpoint["background_ply"]).is_absolute()
        background = manifest_path.parent / checkpoint["background_ply"]
        assert hashlib.sha256(background.read_bytes()).hexdigest() == checkpoint[
            "background_sha256"
        ]
    validate_temporal_bridge_manifest(manifest_path)


def test_rejects_checkpoint_schedule_mismatch(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    schedule_path = temporal.parent / payload["sources"]["schedule"]["path"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["scenes"]["apartment"]["entries"][1]["frame_index"] = 1
    schedule_path.write_text(json.dumps(schedule, sort_keys=True) + "\n", encoding="utf-8")
    payload["sources"]["schedule"] = _record(
        schedule_path, relative_to=temporal.parent
    )
    temporal.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schedule"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_rejects_nonofficial_scene_label_space(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="official apartment label space"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_bridge_manifest_rejects_hash_changes(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(
        temporal, labels, tmp_path / "bridge"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points = (
        manifest_path.parent
        / manifest["checkpoints"][0]["objects"][0]["points_ply"]
    )
    changed = bytearray(points.read_bytes())
    changed[-1] ^= 1
    points.write_bytes(changed)

    with pytest.raises(ValueError, match="SHA256"):
        validate_temporal_bridge_manifest(manifest_path)


@pytest.mark.parametrize(("field", "value"), [("dataset", "other"), ("method", "DUALMAP")])
def test_bridge_validator_rejects_identity_tampering(
    tmp_path: Path, field: str, value: str
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        validate_temporal_bridge_manifest(manifest_path)


def test_bridge_validator_rejects_artifact_path_escape(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = manifest["checkpoints"][0]["objects"][0]
    source = manifest_path.parent / obj["points_ply"]
    escaped = manifest_path.parent.parent / "escaped.ply"
    escaped.write_bytes(source.read_bytes())
    obj["points_ply"] = "../escaped.ply"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inside bridge"):
        validate_temporal_bridge_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("node_symbol", "O99"), ("semantic_label", 99)],
)
def test_bridge_validator_rejects_object_assignment_tampering(
    tmp_path: Path, field: str, value: object
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"][0]["objects"][0][field] = value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assignment"):
        validate_temporal_bridge_manifest(manifest_path)


def test_rejects_temporal_trajectory_later_than_query(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / manifest["trajectories"]["path"]
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[3])
    row["timestamp_ns"] = 301
    rows[3] = json.dumps(row, sort_keys=True)
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["trajectories"] = _record(trajectory, relative_to=temporal.parent)
    temporal.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="later than query"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_rejects_unknown_trajectory_fields(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / manifest["trajectories"]["path"]
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["future_state_path"] = "/forbidden"
    rows[0] = json.dumps(row, sort_keys=True)
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["trajectories"] = _record(trajectory, relative_to=temporal.parent)
    temporal.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trajectory fields"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_validator_rejects_single_sample_declared_as_dynamic(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(
        temporal, labels, tmp_path / "bridge"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = next(
        item
        for item in manifest["checkpoints"][1]["objects"]
        if item["entity_id"] == "z-first"
    )
    trajectory = manifest_path.parent / obj["trajectory_json"]
    samples = json.loads(trajectory.read_text(encoding="utf-8"))[:1]
    trajectory.write_text(json.dumps(samples) + "\n", encoding="utf-8")
    obj["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    obj["trajectory_byte_count"] = trajectory.stat().st_size
    obj["trajectory_sample_count"] = 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="at least two native samples"):
        validate_temporal_bridge_manifest(manifest_path)


def test_rejects_symlinked_schedule_source(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    schedule = temporal.parent / payload["sources"]["schedule"]["path"]
    linked = temporal.parent / "schedule-link.json"
    linked.symlink_to(schedule.name)
    payload["sources"]["schedule"] = _record(
        linked, relative_to=temporal.parent
    )
    temporal.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="symlink"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_source_drift_during_write_leaves_no_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / payload["trajectories"]["path"]
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    original = bridge_module._write_binary_ply
    changed = False

    def mutate_after_write(path: Path, points: np.ndarray) -> None:
        nonlocal changed
        original(path, points)
        if not changed:
            trajectory.write_bytes(trajectory.read_bytes() + b"\n")
            changed = True

    monkeypatch.setattr(bridge_module, "_write_binary_ply", mutate_after_write)
    output = tmp_path / "bridge"

    with pytest.raises(ValueError, match="changed while reading"):
        prepare_temporal_bridge(temporal, labels, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset", "not-tesse", "requires TESSE-CD"),
        ("method", "DUALMAP", "frozen OVIV2 artifact"),
        ("mode", "offline", "requires causal checkpoints"),
    ],
)
def test_rejects_non_oviv2_causal_source_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    payload[field] = value
    temporal.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
