from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.evaluation import export_tesse_temporal_artifact as exporter_module
from scripts.evaluation.export_tesse_temporal_artifact import (
    export_temporal_artifact,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SCHEDULE = (
    ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
)
COMMON_V2_MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd_common_v2.json"
OFFICIAL_SCHEDULE_SHA256 = (
    "fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"
)


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _entity(
    entity_id: str,
    label: str,
    x: float,
    *,
    entity_type: str = "object",
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([[x, 0.0, 1.0]], dtype=np.float32),
        semantic_embedding=None,
        semantic_label=label,
        semantic_score=1.0,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=x,
        metadata={"entity_type": entity_type},
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_fixture(
    root: Path,
    *,
    panoptic: bool = False,
    method: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    root.mkdir()
    scene = "apartment"
    checkpoints = [(0, 100), (2, 300), (4, 500)]
    schedule_path = root / "schedule.json"
    _write_json(
        schedule_path,
        {
            "schema_version": 2,
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "dataset": "TESSE-CD",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                scene: {
                    "frame_count": 5,
                    "entries": [
                        {"frame_index": frame, "timestamp_ns": timestamp}
                        for frame, timestamp in checkpoints
                    ],
                }
            },
        },
    )

    trajectory_path = root / "trajectories.jsonl"
    if panoptic:
        trajectory_rows = [
            {
                "frame_index": frame,
                "source_timestamp_ns": timestamp,
                "entities": [
                    {
                        "native_submap_id": 41,
                        "centroid_xyz": [float(frame), 0.0, 1.0],
                    }
                ],
            }
            for frame, timestamp in enumerate((100, 200, 300, 400, 500))
        ]
    else:
        trajectory_rows = [
            {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "entity_id": entity_id,
                "centroid_xyz": [float(frame), 0.0, 1.0],
                "observation_count": frame + 1,
                "dynamic_state": "static",
                "motion_confidence": 0.0,
                "geometry_epoch": 0,
                "readout_valid": True,
            }
            for frame, timestamp, entity_id in (
                (0, 100, "entity-a"),
                (0, 100, "entity-b"),
                (1, 200, "entity-b"),
                (2, 300, "entity-b"),
                (4, 500, "entity-a"),
            )
        ]
    trajectory_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in trajectory_rows
        ),
        encoding="utf-8",
    )
    lifecycle_path = root / "lifecycle_transitions.jsonl"
    lifecycle_rows = [] if panoptic else [
        {
            "frame_index": frame,
            "timestamp_ns": timestamp,
            "entity_id": entity_id,
            "before": before,
            "after": after,
            "evidence": evidence,
            "geometry_epoch": 0,
            "readout_valid": readout_valid,
        }
        for frame, timestamp, entity_id, before, after, evidence, readout_valid in (
            (1, 200, "entity-a", "active", "uncertain", "visible_absent", False),
            (3, 400, "entity-b", "active", "uncertain", "visible_absent", False),
            (4, 500, "entity-a", "uncertain", "active", "present", True),
        )
    ]
    lifecycle_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in lifecycle_rows),
        encoding="utf-8",
    )
    records_by_frame: dict[int, int] = {}
    for row in trajectory_rows:
        frame = int(row["frame_index"])
        records_by_frame[frame] = records_by_frame.get(frame, 0) + (
            len(row["entities"]) if "entities" in row else 1
        )
    events_by_frame: dict[int, int] = {}
    for row in lifecycle_rows:
        frame = int(row["frame_index"])
        events_by_frame[frame] = events_by_frame.get(frame, 0) + 1
    coverage_path = root / "temporal_frame_coverage.jsonl"
    coverage_path.write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "record_count": records_by_frame.get(frame, 0),
                    "event_count": events_by_frame.get(frame, 0),
                },
                sort_keys=True,
            )
            + "\n"
            for frame, timestamp in enumerate((100, 200, 300, 400, 500))
        ),
        encoding="utf-8",
    )

    checkpoint_records = []
    artifact_entities = {
        0: (
            [_entity("panoptic:41", "chair", 0.0)]
            if panoptic
            else [_entity("entity-a", "chair", 0.0), _entity("entity-b", "table", 0.0)]
        ),
        2: (
            [_entity("panoptic:41", "chair", 2.0)]
            if panoptic
            else [_entity("entity-b", "table", 2.0)]
        ),
        4: (
            [_entity("panoptic:41", "chair", 4.0)]
            if panoptic
            else [_entity("entity-a", "chair", 4.0)]
        ),
    }
    checkpoint_status_records = []
    for frame, timestamp in checkpoints:
        checkpoint_root = root / "checkpoints" / str(timestamp)
        checkpoint_root.mkdir(parents=True)
        status_path = checkpoint_root / "checkpoint_status.json"
        status_payload = (
            {
                "schema_version": 1,
                "frame_index": frame,
                "source_timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "panmap_file": f"frame_{frame:08d}.panmap",
            }
            if panoptic
            else {
                "schema_version": 1,
                "status": "PASS",
                "checkpoint_frame": frame,
                "timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
            }
        )
        _write_json(status_path, status_payload)
        checkpoint_status_records.append(_record(status_path))

        snapshot = MapSnapshot(
            method=method
            or ("Panoptic Mapping + shared masks" if panoptic else "DualMap"),
            scene_id=scene,
            timestamp=float(timestamp),
            entities=artifact_entities[frame],
            background_xyz=None,
            scope="current",
        )
        artifact_paths = write_map_snapshot(snapshot, checkpoint_root / "artifact")
        checkpoint_records.append(
            {
                "frame_index": frame,
                "timestamp_ns": timestamp,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "checkpoint_status": _record(status_path),
                "snapshot": _record(artifact_paths["snapshot"]),
                "entities": _record(artifact_paths["entities"]),
            }
        )

    capture_status_path = root / "run_status.json"
    _write_json(
        capture_status_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "scene": scene,
            "mode": "causal_checkpoints",
            "scheduled_frame_indices": [frame for frame, _ in checkpoints],
            "captured_frame_indices": [frame for frame, _ in checkpoints],
            "schedule": _record(schedule_path),
            "trajectories": _record(trajectory_path),
            "frame_coverage": _record(coverage_path),
            "lifecycle_transitions": _record(lifecycle_path),
            "checkpoint_statuses": checkpoint_status_records,
        },
    )
    index_payload = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "mode": "causal_checkpoint_exports",
        "method": "PANOPTIC_SHARED" if panoptic else "DUALMAP",
        "scene": scene,
        "schedule": _record(schedule_path),
        "capture_status": _record(capture_status_path),
        "trajectories": _record(trajectory_path),
        "frame_coverage": _record(coverage_path),
        "lifecycle_transitions": _record(lifecycle_path),
        "checkpoints": checkpoint_records,
    }
    index_path = root / "source_index.json"
    _write_json(index_path, index_payload)
    return index_path, index_payload


def _rewrite_index(index_path: Path, payload: dict[str, Any]) -> None:
    _write_json(index_path, payload)


def _make_v2_formal_source(root: Path, *, evidence: str = "7") -> Path:
    index_path, index = _build_fixture(root, method="OVIV2")
    index["method"] = "OVIV2"
    runtime_diagnostics = root / "runtime_diagnostics.json"
    _write_json(
        runtime_diagnostics,
        {
            "schema_version": 1,
            "execution_profile": "a2",
            "processed_frame_count": 5,
            "counters": {"absence": 1, "eligible_reid": 2},
        },
    )

    def relative(record: dict[str, Any]) -> dict[str, Any]:
        return {
            **record,
            "path": Path(record["path"]).relative_to(root).as_posix(),
        }

    index["schedule"] = relative(index["schedule"])
    index["trajectories"] = relative(index["trajectories"])
    index["frame_coverage"] = relative(index["frame_coverage"])
    index["lifecycle_transitions"] = relative(index["lifecycle_transitions"])
    index["runtime_diagnostics"] = relative(_record(runtime_diagnostics))
    for checkpoint in index["checkpoints"]:
        for role in ("checkpoint_status", "snapshot", "entities"):
            checkpoint[role] = relative(checkpoint[role])
    capture_path = Path(index["capture_status"]["path"])
    capture = json.loads(capture_path.read_text())
    capture["schedule"] = index["schedule"]
    capture["trajectories"] = index["trajectories"]
    capture["frame_coverage"] = index["frame_coverage"]
    capture["lifecycle_transitions"] = index["lifecycle_transitions"]
    capture["checkpoint_statuses"] = [
        checkpoint["checkpoint_status"] for checkpoint in index["checkpoints"]
    ]
    _write_json(capture_path, capture)
    index["capture_status"] = relative(_record(capture_path))
    frozen = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v2",
        "protocol_id": "oviv2-tessecd-v2",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": "apartment",
        "freeze_manifest": {"sha256": "1" * 64, "byte_count": 1},
        "repository": {"commit": "2" * 40, "tree": "3" * 40},
        "config": {"sha256": "4" * 64, "byte_count": 1},
        "algorithm_hash": "5" * 64,
        "input_bindings_sha256": "6" * 64,
        "formal_evidence_sha256": evidence * 64,
    }
    status = os.stat(root, follow_symlinks=False)
    execution = {
        "schema_version": 1,
        "run_slot": "apartment_run1",
        "output_root": str(root.resolve()),
        "root_device": status.st_dev,
        "root_inode": status.st_ino,
    }
    execution["execution_id"] = hashlib.sha256(
        json.dumps(
            execution, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    index.update({"frozen_run_identity": frozen})
    _write_json(index_path, index)
    inventory = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    _write_json(
        root / "run_manifest.json",
        {
            "schema_version": 2,
            "protocol_id": "oviv2-tessecd-v2",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": "apartment",
            "mode": "dual_readout_causal_checkpoints",
            "algorithm_hash": frozen["algorithm_hash"],
            "processed_frame_count": 5,
            "covered_frame_count": 5,
            "trajectory_frame_count": 4,
            "first_frame_index": 0,
            "last_frame_index": 4,
            "temporal_export_schema_version": 1,
            "scheduled_frame_indices": [0, 2, 4],
            "captured_frame_indices": [0, 2, 4],
            "config": {"sha256": "8" * 64, "byte_count": 1},
            "normalized_run_config": {
                "path": "source_index.json",
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "byte_count": index_path.stat().st_size,
            },
            "schedule": {"sha256": "9" * 64, "byte_count": 1},
            "target_manifest": {"sha256": "a" * 64, "byte_count": 1},
            "source_bindings": {},
            "input_sha256": "b" * 64,
            "code_commit": "c" * 40,
            "checkpoints": [],
            "occlusion_checkpoint_index": {
                "path": "source_index.json",
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "byte_count": index_path.stat().st_size,
            },
            "artifact_inventory": inventory,
            "source_index": {
                "path": "source_index.json",
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "byte_count": index_path.stat().st_size,
            },
            "frozen_run_identity": frozen,
        },
    )
    _write_json(
        root / "execution_receipt.json",
        {
            "schema_version": 1,
            "provenance": {},
            "environment": {},
            "frozen_run_identity": frozen,
            "run_execution": execution,
        },
    )
    return index_path


def _artifact_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _rewrite_trajectories(
    index_path: Path,
    payload: dict[str, Any],
    content: str,
) -> None:
    trajectory_path = Path(payload["trajectories"]["path"])
    trajectory_path.write_text(content, encoding="utf-8")
    payload["trajectories"] = _record(trajectory_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["trajectories"] = payload["trajectories"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)


def _rewrite_lifecycle_transitions(
    index_path: Path,
    payload: dict[str, Any],
    content: str,
) -> None:
    lifecycle_path = Path(payload["lifecycle_transitions"]["path"])
    lifecycle_path.write_text(content, encoding="utf-8")
    payload["lifecycle_transitions"] = _record(lifecycle_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["lifecycle_transitions"] = payload["lifecycle_transitions"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)


def _rewrite_checkpoint_entities(
    index_path: Path,
    payload: dict[str, Any],
    *,
    checkpoint: int,
    content: str,
) -> None:
    entities_path = Path(payload["checkpoints"][checkpoint]["entities"]["path"])
    entities_path.write_text(content, encoding="utf-8")
    payload["checkpoints"][checkpoint]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)


def _rewrite_schedule(
    index_path: Path,
    payload: dict[str, Any],
    schedule: dict[str, Any],
) -> None:
    schedule_path = Path(payload["schedule"]["path"])
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)


def test_cli_help_runs_from_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/export_tesse_temporal_artifact.py"),
            "--help",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source-index" in completed.stdout


def test_exports_presence_intervals_and_byte_identical_repeat(tmp_path: Path) -> None:
    index_path, source_index = _build_fixture(tmp_path / "source")

    first = export_temporal_artifact(index_path, tmp_path / "first")
    second = export_temporal_artifact(index_path, tmp_path / "second")

    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["mode"] == "causal_checkpoints"
    assert manifest["scene"] == "apartment"
    assert manifest["sources"]["trajectories"] == manifest["trajectories"]
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [0, 2, 4]
    assert [item["timestamp_ns"] for item in manifest["checkpoints"]] == [100, 300, 500]
    assert [
        item["consumed_through_frame_exclusive"]
        for item in manifest["checkpoints"]
    ] == [1, 3, 5]
    lifecycles = {item["entity_id"]: item for item in manifest["entity_lifecycles"]}
    assert lifecycles["entity-a"]["presence_intervals"] == [
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
    ]
    assert lifecycles["entity-b"]["presence_intervals"] == [
        {
            "first_frame_index": 0,
            "first_timestamp_ns": 100,
            "last_frame_index": 2,
            "last_timestamp_ns": 300,
        }
    ]
    for checkpoint in manifest["checkpoints"]:
        for key in ("snapshot", "entities"):
            path = first.parent / checkpoint[key]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint[key]["sha256"]
            assert path.stat().st_size == checkpoint[key]["byte_count"]
    first_files = {
        path.relative_to(first.parent): path.read_bytes()
        for path in first.parent.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.parent): path.read_bytes()
        for path in second.parent.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert first.read_bytes().endswith(b"\n")
    assert not first.read_bytes().endswith(b"\n\n")
    assert b"\n " not in first.read_bytes()
    exported_schedule = first.parent / manifest["sources"]["schedule"]["path"]
    source_schedule = Path(source_index["schedule"]["path"])
    assert exported_schedule.read_bytes() == source_schedule.read_bytes()
    for source in (
        manifest["sources"]["source_index"],
        manifest["sources"]["schedule"],
        manifest["sources"]["capture_status"],
        manifest["sources"]["trajectories"],
        *manifest["sources"]["checkpoint_statuses"],
    ):
        assert not Path(source["path"]).is_absolute()
        copied = first.parent / source["path"]
        assert copied.is_file()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == source["sha256"]
        assert copied.stat().st_size == source["byte_count"]


def test_rejects_missing_explicit_frame_coverage(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload.pop("frame_coverage")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture.pop("frame_coverage")
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="frame coverage"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_trajectory_without_explicit_dynamic_state(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0].pop("dynamic_state")
    _rewrite_trajectories(
        index_path,
        payload,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )

    with pytest.raises(ValueError, match="fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_sample_event_geometry_epoch_conflict(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    lifecycle_path = Path(payload["lifecycle_transitions"]["path"])
    rows = [
        json.loads(line)
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
    ]
    overlapping = next(
        row
        for row in rows
        if row["frame_index"] == 4 and row["entity_id"] == "entity-a"
    )
    overlapping["geometry_epoch"] = 99
    _rewrite_lifecycle_transitions(
        index_path,
        payload,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )

    with pytest.raises(ValueError, match="geometry epoch"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_exports_formal_v2_source_without_requiring_execution_in_occlusion_index(
    tmp_path: Path,
) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")

    manifest_path = export_temporal_artifact(index_path, tmp_path / "output")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["frozen_run_identity"]["freeze_id"] == "oviv2-tessecd-v2"
    assert "run_execution" not in manifest
    diagnostics = (
        manifest_path.parent / manifest["sources"]["runtime_diagnostics"]["path"]
    )
    assert diagnostics.read_bytes() == (
        index_path.parent / "runtime_diagnostics.json"
    ).read_bytes()
    sidecar = json.loads(
        (manifest_path.parent / "sidecars/source_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["runtime_diagnostics"] == {
        "path": "runtime_diagnostics.json",
        "sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        "byte_count": diagnostics.stat().st_size,
    }


def test_exports_nonformal_v2_runtime_diagnostics_binding(tmp_path: Path) -> None:
    index_path, index = _build_fixture(tmp_path / "source", method="OVIV2")
    index["method"] = "OVIV2"
    diagnostics = index_path.parent / "runtime_diagnostics.json"
    _write_json(diagnostics, {"schema_version": 1, "counters": {"absence": 1}})
    index["runtime_diagnostics"] = {
        **_record(diagnostics),
        "path": "runtime_diagnostics.json",
    }
    _rewrite_index(index_path, index)

    manifest_path = export_temporal_artifact(index_path, tmp_path / "output")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources"]["runtime_diagnostics"]["path"] == (
        "sidecars/runtime_diagnostics.json"
    )


def test_formal_v2_rejects_runtime_diagnostics_content_drift(tmp_path: Path) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    (index_path.parent / "runtime_diagnostics.json").write_text(
        '{"changed":true}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="runtime diagnostics.*mismatch"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_formal_v2_rejects_runtime_diagnostics_record_alias(tmp_path: Path) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["runtime_diagnostics"] = index["trajectories"]
    _rewrite_index(index_path, index)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_index"] = {
        "path": "source_index.json",
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "byte_count": index_path.stat().st_size,
    }
    manifest["artifact_inventory"].remove("runtime_diagnostics.json")
    (root / "runtime_diagnostics.json").unlink()
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="runtime diagnostics path"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_formal_v2_exports_are_identical_across_execution_roots(tmp_path: Path) -> None:
    first_index = _make_v2_formal_source(tmp_path / "source-a")
    second_index = _make_v2_formal_source(tmp_path / "source-b")

    first = export_temporal_artifact(first_index, tmp_path / "output-a")
    second = export_temporal_artifact(second_index, tmp_path / "output-b")

    assert _artifact_files(first.parent) == _artifact_files(second.parent)
    assert all(
        b"run_execution" not in content
        for content in _artifact_files(first.parent).values()
    )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("receipt_frozen", "identity"),
        ("source_execution", "execution"),
        ("inventory_missing", "inventory"),
        ("inventory_extra", "inventory"),
        ("extra_sidecar", "inventory"),
    ],
)
def test_formal_v2_rejects_authority_and_inventory_mutations(
    tmp_path: Path, mutation: str, message: str
) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    root = index_path.parent
    if mutation == "receipt_frozen":
        path = root / "execution_receipt.json"
        payload = json.loads(path.read_text())
        payload["frozen_run_identity"]["formal_evidence_sha256"] = "0" * 64
        _write_json(path, payload)
    elif mutation == "source_execution":
        path = root / "execution_receipt.json"
        payload = json.loads(path.read_text())
        payload["run_execution"]["root_inode"] += 1
        _write_json(path, payload)
    elif mutation == "inventory_missing":
        path = root / "run_manifest.json"
        payload = json.loads(path.read_text())
        payload["artifact_inventory"].remove("source_index.json")
        _write_json(path, payload)
    elif mutation == "inventory_extra":
        path = root / "run_manifest.json"
        payload = json.loads(path.read_text())
        payload["artifact_inventory"].append("missing.json")
        payload["artifact_inventory"].sort()
        _write_json(path, payload)
    else:
        _write_json(root / "unexpected.json", {})

    with pytest.raises(ValueError, match=message):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_formal_v2_rejects_cross_run_source_index_substitution(tmp_path: Path) -> None:
    first = _make_v2_formal_source(tmp_path / "first", evidence="8")
    second = _make_v2_formal_source(tmp_path / "second")
    second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="identity"):
        export_temporal_artifact(second, tmp_path / "output")


def test_formal_v2_rejects_coherent_sidecar_and_index_tampering(
    tmp_path: Path,
) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    root = index_path.parent
    index = json.loads(index_path.read_text())
    trajectory = root / index["trajectories"]["path"]
    trajectory.write_text(trajectory.read_text() + " \n", encoding="utf-8")
    index["trajectories"] = relative_record = {
        "path": index["trajectories"]["path"],
        "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "byte_count": trajectory.stat().st_size,
    }
    capture_path = root / index["capture_status"]["path"]
    capture = json.loads(capture_path.read_text())
    capture["trajectories"] = relative_record
    _write_json(capture_path, capture)
    index["capture_status"] = {
        "path": index["capture_status"]["path"],
        "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "byte_count": capture_path.stat().st_size,
    }
    _write_json(index_path, index)

    with pytest.raises(ValueError, match="source index.*authority|source index.*mismatch"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_formal_v2_rejects_symlinked_sidecar(tmp_path: Path) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    trajectory = index_path.parent / "trajectories.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(trajectory.read_bytes())
    trajectory.unlink()
    trajectory.symlink_to(replacement)

    with pytest.raises(ValueError, match="symlink"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_formal_v2_rejects_source_root_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _make_v2_formal_source(tmp_path / "source")
    root = index_path.parent
    original = exporter_module._read_json
    replaced = False

    def replace_after_schedule(source: object, *, label: str):
        nonlocal replaced
        payload = original(source, label=label)
        if label == "schedule" and not replaced:
            replacement = tmp_path / "replacement"
            parked = tmp_path / "parked"
            shutil.copytree(root, replacement)
            root.rename(parked)
            replacement.rename(root)
            replaced = True
        return payload

    monkeypatch.setattr(exporter_module, "_read_json", replace_after_schedule)
    with pytest.raises(ValueError, match="changed|identity"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_artifact_is_byte_identical_across_distinct_source_roots(
    tmp_path: Path,
) -> None:
    first_index, _ = _build_fixture(tmp_path / "source-a")
    second_index, second_payload = _build_fixture(tmp_path / "source-b")
    for checkpoint_index, checkpoint in enumerate(second_payload["checkpoints"]):
        entities_path = Path(checkpoint["entities"]["path"])
        records = [
            json.loads(line)
            for line in entities_path.read_text(encoding="utf-8").splitlines()
        ]
        differently_formatted = "".join(
            json.dumps(dict(reversed(tuple(record.items())))) + "\n"
            for record in records
        )
        _rewrite_checkpoint_entities(
            second_index,
            second_payload,
            checkpoint=checkpoint_index,
            content=differently_formatted,
        )

    first_manifest = export_temporal_artifact(first_index, tmp_path / "output-a")
    second_manifest = export_temporal_artifact(second_index, tmp_path / "output-b")

    first_files = _artifact_files(first_manifest.parent)
    second_files = _artifact_files(second_manifest.parent)
    assert first_files == second_files
    forbidden = (
        str(first_index.parent.resolve()).encode(),
        str(second_index.parent.resolve()).encode(),
    )
    assert all(
        source_root not in content
        for content in first_files.values()
        for source_root in forbidden
    )
    for relative_path, content in first_files.items():
        if relative_path.suffix == ".jsonl":
            for line in content.splitlines():
                payload = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {value}")
                    ),
                )
                assert line == json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode(), relative_path
        if relative_path.suffix != ".json" or relative_path == Path(
            "sidecars/schedule.json"
        ):
            continue
        payload = json.loads(
            content,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
        expected = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        assert content == expected, relative_path


def test_official_schedule_preserves_checked_bytes_and_common_v2_hash(
    tmp_path: Path,
) -> None:
    from scripts.evaluation.evaluate_tesse_cd_common_v2 import _load_schedule

    common_v2 = json.loads(COMMON_V2_MANIFEST.read_text(encoding="utf-8"))
    assert common_v2["schedule"]["sha256"] == OFFICIAL_SCHEDULE_SHA256
    assert hashlib.sha256(OFFICIAL_SCHEDULE.read_bytes()).hexdigest() == (
        OFFICIAL_SCHEDULE_SHA256
    )
    source = exporter_module._direct_source(OFFICIAL_SCHEDULE, label="schedule")
    payload = exporter_module._read_json(source, label="schedule")
    output = tmp_path / "schedule.json"

    exporter_module._write_schedule_sidecar(
        source,
        payload,
        selected_scene="apartment",
        output=output,
    )

    assert output.read_bytes() == OFFICIAL_SCHEDULE.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == OFFICIAL_SCHEDULE_SHA256
    assert exporter_module._output_record(output, output=tmp_path)["sha256"] == (
        common_v2["schedule"]["sha256"]
    )
    events, common = _load_schedule(output, scene="apartment")
    assert events
    assert common


def test_rejects_unknown_trajectory_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["cache_path"] = "/tmp/not-part-of-the-contract"
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_trajectories(index_path, payload, "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="trajectory .* fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonfinite_trajectory_json(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["centroid_xyz"][0] = float("nan")
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_trajectories(index_path, payload, "\n".join(rows) + "\n")

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unknown_entity_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    rows = entities_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["cache_path"] = "/tmp/not-part-of-the-contract"
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_checkpoint_entities(
        index_path,
        payload,
        checkpoint=0,
        content="\n".join(rows) + "\n",
    )

    with pytest.raises(ValueError, match="entity .* fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonfinite_entity_json(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    rows = entities_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["semantic_score"] = float("nan")
    rows[0] = json.dumps(row, sort_keys=True)
    _rewrite_checkpoint_entities(
        index_path,
        payload,
        checkpoint=0,
        content="\n".join(rows) + "\n",
    )

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_copied_json_sidecar(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schema_version"] = nonfinite
    capture_path.write_text(
        json.dumps(capture, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="non-finite JSON"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unknown_source_index_path_field(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["unreviewed_cache_path"] = "/tmp/not-part-of-the-contract"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="source index fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_unregistered_index_method(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["method"] = "OVIOVO"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="unsupported.*method"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_index_method_that_does_not_match_snapshot(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["method"] = "OVIV2"
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="method.*match"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_wrong_schedule_schema_version(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["schema_version"] = 3
    _rewrite_schedule(index_path, payload, schedule)

    with pytest.raises(ValueError, match="schedule identity"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_wrong_capture_schema_version(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schema_version"] = 2
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="capture status"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_duplicate_capture_json_keys(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture.pop("schema_version")
    capture_path.write_text(
        '{"schema_version":1,"schema_version":1,'
        + json.dumps(capture, sort_keys=True)[1:],
        encoding="utf-8",
    )
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_index", -1, "non-negative"),
        ("timestamp_ns", 0, "positive"),
        ("centroid_xyz", [True, 0.0, 1.0], "numeric"),
        ("centroid_xyz", ["0.0", 0.0, 1.0], "numeric"),
        ("centroid_xyz", [10**400, 0.0, 1.0], "numeric"),
    ],
)
def test_rejects_invalid_trajectory_scalar_types_and_domains(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    row = json.loads(trajectory_path.read_text(encoding="utf-8").splitlines()[0])
    row[field] = value
    _rewrite_trajectories(index_path, payload, json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=message):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_invalid_trajectory_entity_after_last_checkpoint(
    tmp_path: Path,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    content = trajectory_path.read_text(encoding="utf-8")
    trailing = {
        "frame_index": 6,
        "timestamp_ns": 600,
        "entity_id": "entity-a",
        "centroid_xyz": [False, 0.0, 1.0],
        "observation_count": 1,
        "dynamic_state": "static",
        "motion_confidence": 0.0,
        "geometry_epoch": 0,
        "readout_valid": True,
    }
    _rewrite_trajectories(
        index_path,
        payload,
        content + json.dumps(trailing) + "\n",
    )

    with pytest.raises(ValueError, match="numeric"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_panoptic_trajectory_without_explicit_dynamic_state(
    tmp_path: Path,
) -> None:
    index_path, _ = _build_fixture(tmp_path / "panoptic", panoptic=True)

    with pytest.raises(ValueError, match="fields"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_missing_schedule_checkpoint_without_final_fallback(
    tmp_path: Path,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"].pop(1)
    _rewrite_index(index_path, payload)
    final = index_path.parent / "final"
    final.mkdir()
    (final / "snapshot.npz").write_bytes(b"must-not-be-read")

    with pytest.raises(ValueError, match="exactly cover schedule"):
        export_temporal_artifact(index_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consumed_through_frame", 1),
        ("consumed_through_frame_exclusive", 2),
    ],
)
def test_rejects_checkpoint_that_does_not_end_exactly_at_t(
    tmp_path: Path, field: str, value: int
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"][1][field] = value
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="freeze boundary"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_checkpoint_without_exclusive_boundary(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    payload["checkpoints"][1].pop("consumed_through_frame_exclusive")
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="freeze boundary"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_nonincreasing_checkpoint_timestamps(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["scenes"]["apartment"]["entries"][1]["timestamp_ns"] = 100
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="strictly increasing"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_checkpoint_outside_declared_frame_count(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    schedule_path = Path(payload["schedule"]["path"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["scenes"]["apartment"]["frame_count"] = 4
    _write_json(schedule_path, schedule)
    payload["schedule"] = _record(schedule_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["schedule"] = payload["schedule"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="outside declared frame count"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_allows_stable_id_semantic_evolution_and_keeps_first_lifecycle_label(
    tmp_path: Path,
) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][2]["entities"]["path"])
    record = json.loads(entities_path.read_text(encoding="utf-8"))
    record["semantic_label"] = "sofa"
    entities_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    payload["checkpoints"][2]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)

    manifest_path = export_temporal_artifact(index_path, tmp_path / "output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lifecycle = next(
        item
        for item in manifest["entity_lifecycles"]
        if item["entity_id"] == "entity-a"
    )
    assert lifecycle["semantic_label"] == "chair"
    final_entities = (
        manifest_path.parent / manifest["checkpoints"][2]["entities"]["path"]
    )
    final_record = json.loads(final_entities.read_text(encoding="utf-8"))
    assert final_record["semantic_label"] == "sofa"


def test_rejects_stable_id_type_conflict(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][2]["entities"]["path"])
    record = json.loads(entities_path.read_text(encoding="utf-8"))
    record["metadata"]["entity_type"] = "region"
    entities_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    payload["checkpoints"][2]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="type conflict"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_duplicate_id_within_checkpoint(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    entities_path = Path(payload["checkpoints"][0]["entities"]["path"])
    records = [
        json.loads(line)
        for line in entities_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["entity_id"] = records[0]["entity_id"]
    entities_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    payload["checkpoints"][0]["entities"] = _record(entities_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="unique|duplicate"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_future_trajectory_sample(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    trajectory_path = Path(payload["trajectories"]["path"])
    rows = trajectory_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[3])
    assert row["frame_index"] == 2
    row["timestamp_ns"] = 301
    rows[3] = json.dumps(row, sort_keys=True)
    trajectory_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload["trajectories"] = _record(trajectory_path)
    capture_path = Path(payload["capture_status"]["path"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["trajectories"] = payload["trajectories"]
    _write_json(capture_path, capture)
    payload["capture_status"] = _record(capture_path)
    _rewrite_index(index_path, payload)

    with pytest.raises(ValueError, match="later than checkpoint"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_rejects_exact_source_hash_mismatch(tmp_path: Path) -> None:
    index_path, payload = _build_fixture(tmp_path / "source")
    snapshot_path = Path(payload["checkpoints"][1]["snapshot"]["path"])
    changed = bytearray(snapshot_path.read_bytes())
    changed[-1] ^= 1
    snapshot_path.write_bytes(changed)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export_temporal_artifact(index_path, tmp_path / "output")


def test_copy_failure_cleans_staging_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"
    original_copyfile = exporter_module.shutil.copyfile
    copy_count = 0

    def fail_second_copy(source: object, target: object) -> object:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("injected copy failure")
        return original_copyfile(source, target)

    monkeypatch.setattr(exporter_module.shutil, "copyfile", fail_second_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        export_temporal_artifact(index_path, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))

    monkeypatch.setattr(exporter_module.shutil, "copyfile", original_copyfile)
    manifest = export_temporal_artifact(index_path, output)
    assert manifest.is_file()


def test_temporal_output_is_atomic_no_clobber(tmp_path: Path) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        export_temporal_artifact(index_path, output)

    assert marker.read_bytes() == b"sentinel"
    assert set(output.iterdir()) == {marker}


def test_temporal_publish_failure_after_reservation_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, _ = _build_fixture(tmp_path / "source")
    output = tmp_path / "output"

    def fail_rename(source: object, target: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(RuntimeError, match="publication-uncertain"):
        export_temporal_artifact(index_path, output)

    assert output.is_dir()
