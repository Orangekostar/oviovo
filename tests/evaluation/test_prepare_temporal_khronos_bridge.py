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


_FIXTURE_SCHEDULE_BYTES = (
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
    + "\n"
).encode("utf-8")


@pytest.fixture(autouse=True)
def _bind_fixture_label_space(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_contents = (
        "label_names:\n  - {label: 0, name: Unknown}\n  - {label: 5, name: Chair}\n",
        "label_names:\n  - {label: 0, name: Unknown}\n"
        "  - {label: 5, name: Chair}\n  - {label: 7, name: Table}\n",
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
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
    monkeypatch.setattr(
        bridge_module,
        "OFFICIAL_SCHEDULE_SHA256",
        frozenset({hashlib.sha256(_FIXTURE_SCHEDULE_BYTES).hexdigest()}),
        raising=False,
    )


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _rewrite_consistency(
    manifest_path: Path,
    mutate: Any,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    consistency_path = (
        manifest_path.parent / manifest["temporal_consistency_json"]["path"]
    )
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    mutate(consistency)
    consistency_path.write_text(
        json.dumps(consistency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["temporal_consistency_json"] = _record(
        consistency_path, relative_to=manifest_path.parent
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepared_bridge(tmp_path: Path) -> Path:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
        encoding="utf-8",
    )
    return prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def _entity(entity_id: str, x: float, *, label: str = "Chair") -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([[x, 0.0, 1.0], [x + 0.1, 0.0, 1.0]], dtype=np.float32),
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=x,
        metadata={"entity_type": "object"},
    )


def _official_is_present(obj: dict[str, Any], query: int) -> bool:
    appeared = max(
        timestamp for timestamp in obj["first_observed_ns"] if timestamp <= query
    )
    disappeared = max(
        (
            timestamp
            for timestamp in obj["last_observed_ns"]
            if timestamp <= query
        ),
        default=-1,
    )
    return appeared > disappeared


def _write_temporal_fixture(root: Path) -> Path:
    root.mkdir()
    schedule = root / "schedule.json"
    schedule.write_bytes(_FIXTURE_SCHEDULE_BYTES)
    checkpoints = [
        (100, [_entity("r-reappear", 1), _entity("sparse", 2), _entity("z-first", 3)]),
        (
            300,
            [
                _entity("a-later", 4),
                _entity("sparse", 5, label="Table"),
                _entity("z-first", 6),
            ],
        ),
        (
            500,
            [
                _entity("a-later", 7),
                _entity("r-reappear", 8),
                _entity("sparse", 9, label="Table"),
            ],
        ),
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
        (0, 100, "r-reappear", [1.0, 0.0, 1.0], "dynamic", True),
        (0, 100, "sparse", [2.0, 0.0, 1.0], "static", True),
        (0, 100, "z-first", [3.0, 0.0, 1.0], "static", True),
        (1, 200, "z-first", [3.5, 0.0, 1.0], "dynamic", True),
        (2, 300, "a-later", [4.0, 0.0, 1.0], "dynamic", True),
        (2, 300, "z-first", [6.0, 0.0, 1.0], "dynamic", True),
        (3, 400, "a-later", [5.5, 0.0, 1.0], "dynamic", True),
        (4, 500, "a-later", [7.0, 0.0, 1.0], "static", True),
        (4, 500, "r-reappear", [8.0, 0.0, 1.0], "dynamic", True),
    ]
    trajectories.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "entity_id": entity_id,
                    "centroid_xyz": centroid,
                    "observation_count": index + 1,
                    "dynamic_state": dynamic_state,
                    "motion_confidence": 0.9,
                    "geometry_epoch": 0,
                    "readout_valid": readout_valid,
                },
                sort_keys=True,
            )
            + "\n"
            for index, (
                frame,
                timestamp,
                entity_id,
                centroid,
                dynamic_state,
                readout_valid,
            ) in enumerate(trajectory_rows)
        ),
        encoding="utf-8",
    )
    lifecycle = root / "lifecycle_transitions.jsonl"
    lifecycle_rows = [
        (1, 200, "r-reappear", False),
        (3, 400, "z-first", False),
        (4, 500, "r-reappear", True),
    ]
    lifecycle.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "entity_id": entity_id,
                    "before": "active" if readout_valid else "active",
                    "after": "active" if readout_valid else "uncertain",
                    "evidence": "present" if readout_valid else "visible_absent",
                    "geometry_epoch": 0,
                    "readout_valid": readout_valid,
                },
                sort_keys=True,
            )
            + "\n"
            for frame, timestamp, entity_id, readout_valid in lifecycle_rows
        ),
        encoding="utf-8",
    )
    coverage = root / "temporal_frame_coverage.jsonl"
    coverage.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "record_count": sum(row[0] == frame for row in trajectory_rows),
                    "event_count": sum(row[0] == frame for row in lifecycle_rows),
                },
                sort_keys=True,
            )
            + "\n"
            for frame, timestamp in enumerate((100, 200, 300, 400, 500))
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoints",
        "method": "OVIV2",
        "scene": "apartment",
        "sources": {
            "schedule": _record(schedule, relative_to=root),
            "frame_coverage": _record(coverage, relative_to=root),
            "lifecycle_transitions": _record(lifecycle, relative_to=root),
        },
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
        "frame_coverage": _record(coverage, relative_to=root),
        "lifecycle_transitions": _record(lifecycle, relative_to=root),
        "temporal_export_schema_version": 1,
        "temporal_audit_counts": {
            "static_sample_count": 3,
            "dynamic_sample_count": 6,
            "unknown_sample_count": 0,
            "missing_frame_count": 0,
            "lifecycle_transition_count": 3,
            "geometry_epoch_count": 4,
            "invalid_readout_sample_count": 0,
        },
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
        "label_names:\n  - {label: 0, name: Unknown}\n"
        "  - {label: 5, name: Chair}\n  - {label: 7, name: Table}\n",
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
    consistency_record = manifest["temporal_consistency_json"]
    consistency_path = manifest_path.parent / consistency_record["path"]
    assert consistency_record == _record(
        consistency_path, relative_to=manifest_path.parent
    )
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    assert any(
        sample["frame_index"] == 4
        and sample["entity_id"] == "r-reappear"
        and sample["geometry_epoch"] == 0
        for sample in consistency["samples"]
    )
    assert any(
        event["frame_index"] == 4
        and event["entity_id"] == "r-reappear"
        and event["geometry_epoch"] == 0
        for event in consistency["lifecycle_events"]
    )
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
    assert assignments["r-reappear"]["presence_intervals"] == [
        {"start_ns": 100, "end_ns_exclusive": 200},
        {"start_ns": 500, "end_ns_exclusive": None},
    ]
    assert assignments["r-reappear"]["last_observed_ns"] == [200, None]
    assert assignments["z-first"]["last_observed_ns"] == [400]
    assert assignments["a-later"]["first_observed_ns"] == [300]
    assert assignments["sparse"]["node_symbol"] == "O1"
    assert assignments["sparse"]["semantic_label_name"] == "Chair"
    assert assignments["sparse"]["semantic_label"] == 5
    assert assignments["sparse"]["semantic_observations"] == [
        {
            "timestamp_ns": 100,
            "semantic_label_name": "Chair",
            "semantic_label": 5,
            "label_matched": True,
        },
        {
            "timestamp_ns": 300,
            "semantic_label_name": "Table",
            "semantic_label": 7,
            "label_matched": True,
        },
        {
            "timestamp_ns": 500,
            "semantic_label_name": "Table",
            "semantic_label": 7,
            "label_matched": True,
        },
    ]

    first_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][0]["objects"]
    }
    assert first_objects["r-reappear"]["node_symbol"] == "O0"
    assert first_objects["r-reappear"]["trajectory_sample_count"] == 1
    assert first_objects["r-reappear"]["dynamic_track_eligible"] is True
    middle_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][1]["objects"]
    }
    assert "r-reappear" not in middle_objects
    assert middle_objects["z-first"]["trajectory_sample_count"] == 3
    assert middle_objects["a-later"]["trajectory_sample_count"] == 1
    assert middle_objects["a-later"]["dynamic_track_eligible"] is True
    assert middle_objects["sparse"]["node_symbol"] == "O1"
    assert middle_objects["sparse"]["semantic_label_name"] == "Table"
    assert middle_objects["sparse"]["semantic_label"] == 7
    final_objects = {
        entry["entity_id"]: entry for entry in manifest["checkpoints"][2]["objects"]
    }
    assert final_objects["r-reappear"]["node_symbol"] == "O0"
    assert final_objects["r-reappear"]["first_observed_ns"] == [100, 500]
    assert final_objects["r-reappear"]["last_observed_ns"] == [200, 501]
    assert final_objects["sparse"]["dynamic_track_eligible"] is False
    assert final_objects["sparse"]["trajectory_sample_count"] == 0
    assert final_objects["a-later"]["dynamic_track_eligible"] is False
    assert final_objects["a-later"]["trajectory_sample_count"] == 0

    for checkpoint in manifest["checkpoints"]:
        query = checkpoint["timestamp_ns"]
        frame = checkpoint["frame_index"]
        for obj in checkpoint["objects"]:
            assert len(obj["first_observed_ns"]) == len(obj["last_observed_ns"])
            assert obj["first_observed_ns"]
            assert obj["last_observed_ns"][-1] == query + 1
            assert _official_is_present(obj, query)
            assert not Path(obj["trajectory_json"]).is_absolute()
            assert not Path(obj["points_ply"]).is_absolute()
            trajectory = json.loads(
                (manifest_path.parent / obj["trajectory_json"]).read_text(
                    encoding="utf-8"
                )
            )
            assert all(sample["timestamp_ns"] <= query for sample in trajectory)
            assert all(sample["frame_index"] <= frame for sample in trajectory)
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


def test_bridge_never_promotes_static_track_from_sample_count(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory_path = temporal.parent / manifest["trajectories"]["path"]
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["entity_id"] == "z-first":
            row["dynamic_state"] = "static"
    trajectory_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest["trajectories"] = _record(
        trajectory_path, relative_to=temporal.parent
    )
    manifest["temporal_audit_counts"]["static_sample_count"] += 2
    manifest["temporal_audit_counts"]["dynamic_sample_count"] -= 2
    temporal.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
        encoding="utf-8",
    )

    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    bridge = json.loads(manifest_path.read_text(encoding="utf-8"))
    middle = next(
        item
        for item in bridge["checkpoints"][1]["objects"]
        if item["entity_id"] == "z-first"
    )
    assert middle["dynamic_track_eligible"] is False
    assert middle["trajectory_sample_count"] == 0


def test_bridge_rejects_missing_explicit_frame_coverage(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    manifest["sources"].pop("frame_coverage")
    manifest.pop("frame_coverage")
    temporal.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame coverage"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_bridge_rejects_sample_event_geometry_epoch_conflict(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    lifecycle_path = temporal.parent / manifest["lifecycle_transitions"]["path"]
    rows = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
    ]
    overlapping = next(
        row
        for row in rows
        if row["frame_index"] == 4 and row["entity_id"] == "r-reappear"
    )
    overlapping["geometry_epoch"] = 99
    lifecycle_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    lifecycle_record = _record(lifecycle_path, relative_to=temporal.parent)
    manifest["lifecycle_transitions"] = lifecycle_record
    manifest["sources"]["lifecycle_transitions"] = lifecycle_record
    manifest["temporal_audit_counts"]["geometry_epoch_count"] = 5
    temporal.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="geometry epoch"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


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


def test_rejects_self_consistent_noncanonical_schedule(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    schedule_path = temporal.parent / payload["sources"]["schedule"]["path"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["parameters"]["unreviewed_variant"] = True
    schedule_path.write_text(
        json.dumps(schedule, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload["sources"]["schedule"] = _record(
        schedule_path, relative_to=temporal.parent
    )
    temporal.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="official TESSE-CD causal schedule"):
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


def test_bridge_manifest_rejects_rehashed_sample_event_epoch_conflict(
    tmp_path: Path,
) -> None:
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
    consistency_path = (
        manifest_path.parent / manifest["temporal_consistency_json"]["path"]
    )
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    overlapping = next(
        event
        for event in consistency["lifecycle_events"]
        if event["frame_index"] == 4 and event["entity_id"] == "r-reappear"
    )
    overlapping["geometry_epoch"] = 99
    consistency_path.write_text(
        json.dumps(consistency, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["temporal_consistency_json"] = _record(
        consistency_path, relative_to=manifest_path.parent
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="geometry epoch"):
        validate_temporal_bridge_manifest(manifest_path)


def test_consistency_sidecar_contains_complete_canonical_context(
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_bridge(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    consistency_path = (
        manifest_path.parent / manifest["temporal_consistency_json"]["path"]
    )
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))

    assert set(consistency) == {
        "schema_version",
        "frame_coverage",
        "query_timestamps_ns",
        "temporal_audit_counts",
        "samples",
        "lifecycle_events",
    }
    assert consistency["frame_coverage"][-1] == {
        "frame_index": 4,
        "timestamp_ns": 500,
        "record_count": 2,
        "event_count": 1,
    }
    assert consistency["query_timestamps_ns"] == [100, 300, 500]
    assert consistency["temporal_audit_counts"] == manifest[
        "temporal_audit_counts"
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["samples"].clear(),
        lambda payload: payload["lifecycle_events"].clear(),
        lambda payload: payload["samples"].reverse(),
        lambda payload: payload.update(unexpected=[]),
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload["frame_coverage"].reverse(),
        lambda payload: payload["frame_coverage"][0].update(record_count=99),
        lambda payload: payload["query_timestamps_ns"].reverse(),
        lambda payload: payload["temporal_audit_counts"].update(
            static_sample_count=99
        ),
        lambda payload: payload["samples"][0].update(motion_confidence=0.5),
        lambda payload: payload["samples"][0].update(
            centroid_xyz=["invalid", 0.0, 1.0]
        ),
        lambda payload: payload["samples"][0].update(
            centroid_xyz=[True, 0.0, 1.0]
        ),
        lambda payload: payload["samples"][0].update(dynamic_state="moving"),
        lambda payload: payload["samples"][0].update(timestamp_ns=True),
        lambda payload: payload["lifecycle_events"][0].update(before="missing"),
        lambda payload: payload["lifecycle_events"].append(
            dict(payload["lifecycle_events"][0])
        ),
    ],
)
def test_bridge_manifest_rejects_rehashed_consistency_replica_tampering(
    tmp_path: Path,
    mutate: Any,
) -> None:
    manifest_path = _prepared_bridge(tmp_path)
    _rewrite_consistency(manifest_path, mutate)

    with pytest.raises(ValueError, match="consistency|coverage|audit|source"):
        validate_temporal_bridge_manifest(manifest_path)


def test_bridge_manifest_rejects_symlinked_consistency_sidecar(
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_bridge(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    consistency_path = (
        manifest_path.parent / manifest["temporal_consistency_json"]["path"]
    )
    target = consistency_path.with_name("temporal-consistency-target.json")
    consistency_path.rename(target)
    consistency_path.symlink_to(target.name)

    with pytest.raises(ValueError, match="symlink"):
        validate_temporal_bridge_manifest(manifest_path)


def test_consistency_replica_normalizes_integer_and_float_centroid_numbers(
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_bridge(tmp_path)

    def use_integer_coordinate(payload: dict[str, Any]) -> None:
        coordinate = payload["samples"][0]["centroid_xyz"][0]
        assert coordinate == 1.0
        payload["samples"][0]["centroid_xyz"][0] = int(coordinate)

    _rewrite_consistency(manifest_path, use_integer_coordinate)
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda assignment: assignment["semantic_observations"].reverse(),
        lambda assignment: assignment["semantic_observations"].pop(),
        lambda assignment: assignment["semantic_observations"][0].update(
            semantic_label_name="Table", semantic_label=7
        ),
    ],
)
def test_bridge_validator_rejects_semantic_history_tampering(
    tmp_path: Path, mutate: Any
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment = next(
        item for item in manifest["symbol_assignments"] if item["entity_id"] == "sparse"
    )
    mutate(assignment)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semantic history"):
        validate_temporal_bridge_manifest(manifest_path)


def test_bridge_rejects_stable_id_type_conflict(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    entities_path = temporal.parent / manifest["checkpoints"][2]["entities"]["path"]
    records = [
        json.loads(line)
        for line in entities_path.read_text(encoding="utf-8").splitlines()
    ]
    sparse = next(record for record in records if record["entity_id"] == "sparse")
    sparse["metadata"]["entity_type"] = "region"
    entities_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest["checkpoints"][2]["entities"] = _record(
        entities_path, relative_to=temporal.parent
    )
    temporal.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="type conflict"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_rejects_temporal_trajectory_later_than_query(tmp_path: Path) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / manifest["trajectories"]["path"]
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[3])
    row["timestamp_ns"] = 501
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

    with pytest.raises(ValueError, match="frame coverage"):
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


def test_rejects_bool_centroid_in_resigned_source_trajectory(
    tmp_path: Path,
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / manifest["trajectories"]["path"]
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    sample = json.loads(rows[0])
    assert sample["centroid_xyz"][0] == 1.0
    sample["centroid_xyz"][0] = True
    rows[0] = json.dumps(sample, sort_keys=True)
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["trajectories"] = _record(
        trajectory, relative_to=temporal.parent
    )
    temporal.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="centroid"):
        prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")


def test_validator_rejects_bool_centroid_in_rehashed_bridge_trajectory(
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_bridge(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = manifest["checkpoints"][0]["objects"][0]
    trajectory = manifest_path.parent / obj["trajectory_json"]
    samples = json.loads(trajectory.read_text(encoding="utf-8"))
    assert samples[0]["centroid_xyz"][0] == 1.0
    samples[0]["centroid_xyz"][0] = True
    trajectory.write_text(
        json.dumps(samples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    obj["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    obj["trajectory_byte_count"] = trajectory.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trajectory sample"):
        validate_temporal_bridge_manifest(manifest_path)


@pytest.mark.parametrize("first_coordinate", [1, 1.0])
def test_accepts_integer_or_float_centroid_numbers(
    tmp_path: Path,
    first_coordinate: int | float,
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    manifest = json.loads(temporal.read_text(encoding="utf-8"))
    trajectory = temporal.parent / manifest["trajectories"]["path"]
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    sample = json.loads(rows[0])
    sample["centroid_xyz"][0] = first_coordinate
    rows[0] = json.dumps(sample, sort_keys=True)
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["trajectories"] = _record(
        trajectory, relative_to=temporal.parent
    )
    temporal.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )

    output = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    validate_temporal_bridge_manifest(output)


def test_validator_rejects_trajectory_whose_last_sample_is_not_dynamic(
    tmp_path: Path,
) -> None:
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

    with pytest.raises(ValueError, match="explicit|dynamic state"):
        validate_temporal_bridge_manifest(manifest_path)


def test_validator_rejects_implicit_state_in_earlier_trajectory_sample(
    tmp_path: Path,
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}, "
        "{label: 7, name: Table}]\n",
        encoding="utf-8",
    )
    manifest_path = prepare_temporal_bridge(temporal, labels, tmp_path / "bridge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = next(
        item
        for item in manifest["checkpoints"][1]["objects"]
        if item["entity_id"] == "z-first"
    )
    trajectory = manifest_path.parent / obj["trajectory_json"]
    samples = json.loads(trajectory.read_text(encoding="utf-8"))
    samples[0].pop("dynamic_state")
    trajectory.write_text(json.dumps(samples) + "\n", encoding="utf-8")
    obj["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    obj["trajectory_byte_count"] = trajectory.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fields"):
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


def test_publication_race_cleans_staging_and_reserved_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporal = _write_temporal_fixture(tmp_path / "temporal")
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "label_names: [{label: 0, name: Unknown}, {label: 5, name: Chair}]\n",
        encoding="utf-8",
    )
    staging_paths: list[Path] = []
    original_mkdtemp = bridge_module.tempfile.mkdtemp

    def capture_staging(*args: Any, **kwargs: Any) -> str:
        staging = original_mkdtemp(*args, **kwargs)
        staging_paths.append(Path(staging))
        return staging

    def fail_publication(source: Path, destination: Path) -> None:
        raise FileExistsError(f"publication race at {destination}")

    monkeypatch.setattr(bridge_module.tempfile, "mkdtemp", capture_staging)
    monkeypatch.setattr(bridge_module.os, "rename", fail_publication)
    output = tmp_path / "bridge"

    with pytest.raises(RuntimeError, match="publication-uncertain"):
        prepare_temporal_bridge(temporal, labels, output)

    assert staging_paths and all(not path.exists() for path in staging_paths)
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
