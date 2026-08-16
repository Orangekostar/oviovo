from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.export_tesse_temporal_artifact import (
    export_temporal_artifact,
)
from scripts.evaluation.recover_oviv2_temporal_staging_proxy import (
    NONFORMAL_STATUS,
    recover_temporal_staging_proxy,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _tree_state(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _build_preserved_staging(root: Path) -> Path:
    root.mkdir()
    scene = "apartment"
    frame_index = 0
    timestamp_ns = 100

    schedule_path = root / "inputs/schedule.json"
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
                    "frame_count": 1,
                    "entries": [
                        {"frame_index": frame_index, "timestamp_ns": timestamp_ns}
                    ],
                }
            },
        },
    )

    trajectories_path = root / "trajectories.jsonl"
    trajectories_path.write_text(
        json.dumps(
            {
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "entity_id": "entity-a",
                "centroid_xyz": [0.0, 0.0, 1.0],
                "observation_count": 1,
                "dynamic_state": "static",
                "motion_confidence": 0.0,
                "geometry_epoch": 0,
                "readout_valid": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    coverage_path = root / "temporal_frame_coverage.jsonl"
    coverage_path.write_text(
        json.dumps(
            {
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "record_count": 1,
                "event_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle_path = root / "lifecycle_transitions.jsonl"
    lifecycle_path.write_text("", encoding="utf-8")

    checkpoint_root = root / "checkpoints/00000000-100"
    status_path = checkpoint_root / "checkpoint_status.json"
    _write_json(
        status_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "checkpoint_frame": frame_index,
            "timestamp_ns": timestamp_ns,
            "event_ids": [],
            "roles": ["official_v2"],
            "consumed_through_frame": frame_index,
            "consumed_through_frame_exclusive": frame_index + 1,
        },
    )
    snapshot = MapSnapshot(
        method="OVIV2",
        scene_id=scene,
        timestamp=float(timestamp_ns),
        entities=[
            EntityPrediction(
                entity_id="entity-a",
                points_xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                semantic_embedding=None,
                semantic_label="chair",
                semantic_score=1.0,
                lifecycle_state="active",
                first_seen=0.0,
                last_seen=0.0,
                metadata={"entity_type": "object"},
            )
        ],
        background_xyz=None,
        scope="current",
    )
    write_map_snapshot(snapshot, checkpoint_root / "neutral_current")

    capture_path = root / "capture_status.json"
    _write_json(
        capture_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "scene": scene,
            "mode": "causal_checkpoints",
            "scheduled_frame_indices": [frame_index],
            "captured_frame_indices": [frame_index],
            "schedule": _record(schedule_path, root=root),
            "trajectories": _record(trajectories_path, root=root),
            "frame_coverage": _record(coverage_path, root=root),
            "lifecycle_transitions": _record(lifecycle_path, root=root),
            "checkpoint_statuses": [_record(status_path, root=root)],
        },
    )
    return root


def test_recovers_exporter_readable_proxy_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = _build_preserved_staging(tmp_path / "preserved")
    before = _tree_state(source)
    output = tmp_path / "proxy"

    result = recover_temporal_staging_proxy(
        source_root=source,
        output_root=output,
    )

    assert result == output / "source_index.json"
    assert _tree_state(source) == before
    index = json.loads(result.read_text(encoding="utf-8"))
    assert set(index) == {
        "schema_version",
        "dataset",
        "mode",
        "method",
        "scene",
        "schedule",
        "capture_status",
        "trajectories",
        "frame_coverage",
        "lifecycle_transitions",
        "checkpoints",
    }
    assert not set(index) & {
        "runtime_diagnostics",
        "run_manifest",
        "run_execution",
        "frozen_run_identity",
        "nonformal_authorization",
    }
    receipt = json.loads((output / "recovery_receipt.json").read_text())
    assert set(receipt) == {
        "schema_version",
        "status",
        "submission_eligible",
        "source_root",
        "source_files",
        "source_index",
        "recovery_tool",
    }
    assert receipt["status"] == NONFORMAL_STATUS
    assert receipt["submission_eligible"] is False

    temporal_manifest = export_temporal_artifact(
        result,
        tmp_path / "temporal",
    )
    assert temporal_manifest.name == "temporal_manifest.json"
    assert temporal_manifest.is_file()
