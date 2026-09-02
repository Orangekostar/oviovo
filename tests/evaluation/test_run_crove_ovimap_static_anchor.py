from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_crove_ovimap_static_anchor import compose_run
from scripts.evaluation.export_tesse_temporal_artifact import export_temporal_artifact
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


_TIMESTAMP_ORIGIN_NS = 4_000_000_000


def _timestamp_ns(frame: int) -> int:
    return _TIMESTAMP_ORIGIN_NS + frame * 1_000_000_000


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path if root is None else path.relative_to(root)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _prediction(entity_id: str, x: float, *, temporal_id: int | None = None) -> EntityPrediction:
    metadata: dict[str, object] = {"authority": "ovimap_anchor"}
    if temporal_id is not None:
        metadata = {
            "temporal_entity_id": temporal_id,
            "semantic_id": 0,
            "geometry_epoch": int(x > 0.5),
            "readout_valid": True,
        }
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(((x - 0.05, 0.0, 0.0), (x + 0.05, 0.0, 0.0)), dtype=np.float32),
        semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        semantic_label="Chair",
        semantic_score=1.0,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=3.0,
        metadata=metadata,
    )


def _write_anchor_package(tmp_path: Path, *, anchor_x: float = 0.0) -> Path:
    root = tmp_path / "anchor"
    config = tmp_path / "anchor_config.json"
    vocabulary = tmp_path / "vocabulary.json"
    schedule = tmp_path / "schedule.json"
    _write_json(
        config,
        {
            "minimum_spatial_iou": 0.01,
            "maximum_centroid_distance_m": 0.75,
            "minimum_semantic_cosine": 0.65,
            "moved_displacement_m": 0.2,
            "background_voxel_size_m": 0.05,
        },
    )
    _write_json(
        vocabulary,
        {"dataset": "TESSE-CD", "scene": "apartment", "classes": ["Chair"]},
    )
    _write_json(
        schedule,
        {
            "schema_version": 2,
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "dataset": "TESSE-CD",
            "method_predictions_used": False,
            "parameters": {"frame_indexing": "zero_based"},
            "scenes": {
                "apartment": {
                    "frame_count": 4,
                    "entries": [
                        {"frame_index": frame, "timestamp_ns": _timestamp_ns(frame)}
                        for frame in (2, 3)
                    ],
                }
            },
        },
    )
    written = write_map_snapshot(
        MapSnapshot(
            method="OVI-MAP causal static anchor",
            scene_id="apartment",
            timestamp=1.0,
            entities=[_prediction("ovimap:1", anchor_x)],
            background_xyz=np.asarray(((4.0, 0.0, 0.0),), dtype=np.float32),
            scope="current",
        ),
        root,
    )
    manifest = root / "anchor_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_v1",
            "status": "PASS",
            "method": "OVI-MAP causal static anchor",
            "scene": "apartment",
            "anchor_id": "a" * 64,
            "causality": {
                "first_source_frame": 0,
                "last_source_frame": 1,
                "maximum_source_frame": 1,
                "strictly_pre_intervention": True,
            },
            "sources": {
                "config": _record(config),
                "vocabulary": _record(vocabulary),
                "schedule": _record(schedule),
            },
            "outputs": {
                "snapshot": _record(written["snapshot"], root=root),
                "entities": _record(written["entities"], root=root),
            },
        },
    )
    return manifest


def _sample(frame: int, x: float) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ns": _timestamp_ns(frame),
        "entity_id": 7,
        "centroid_xyz": [x, 0.0, 0.0],
        "observation_count": frame + 1,
        "dynamic_state": "dynamic" if x > 0.5 else "static",
        "motion_confidence": 0.9 if x > 0.5 else 0.0,
        "geometry_epoch": int(x > 0.5),
        "readout_valid": True,
    }


def _write_source_run(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    trajectories = root / "trajectories.jsonl"
    lifecycle = root / "lifecycle_transitions.jsonl"
    coverage = root / "temporal_frame_coverage.jsonl"
    samples = [_sample(frame, 0.0 if frame < 3 else 1.0) for frame in range(4)]
    _write_jsonl(trajectories, samples)
    _write_jsonl(lifecycle, [])
    _write_jsonl(
        coverage,
        [
            {
                "frame_index": frame,
                "timestamp_ns": _timestamp_ns(frame),
                "record_count": 1,
                "event_count": 0,
            }
            for frame in range(4)
        ],
    )
    source_index = root / "source_index.json"
    _write_json(
        source_index,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": "apartment",
            "trajectories": _record(trajectories, root=root),
            "lifecycle_transitions": _record(lifecycle, root=root),
            "frame_coverage": _record(coverage, root=root),
        },
    )
    checkpoints = []
    for frame, x in ((2, 0.0), (3, 1.0)):
        written = write_map_snapshot(
            MapSnapshot(
                method="OVIV2-temporal",
                scene_id="apartment",
                timestamp=float(_timestamp_ns(frame)),
                entities=[_prediction("temporal:7", x, temporal_id=7)],
                background_xyz=None,
                scope="current",
            ),
            root / "checkpoints" / f"{frame:08d}-{_timestamp_ns(frame)}",
        )
        checkpoints.append(
            {
                "scene": "apartment",
                "frame_index": frame,
                "timestamp_ns": _timestamp_ns(frame),
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "neutral_snapshot": _record(written["snapshot"], root=root),
                "neutral_entities": _record(written["entities"], root=root),
            }
        )
    manifest = root / "run_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": "apartment",
            "mode": "dual_readout_causal_checkpoints",
            "processed_frame_count": 4,
            "covered_frame_count": 4,
            "first_frame_index": 0,
            "last_frame_index": 3,
            "temporal_export_schema_version": 1,
            "captured_frame_indices": [2, 3],
            "source_index": _record(source_index, root=root),
            "checkpoints": checkpoints,
        },
    )
    return manifest


def test_composed_run_publishes_hash_bound_checkpoints(tmp_path: Path) -> None:
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=_write_anchor_package(tmp_path),
        output_root=tmp_path / "composed",
    )

    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["method"] == "CROVE + OVI-MAP static anchor (composed)"
    assert manifest["integration"] == "composed"
    assert manifest["execution_mode"] == "online_after_causal_initialization"
    assert manifest["processed_frame_count"] == 4
    assert manifest["official_state_count"] == 2
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [2, 3]
    assert manifest["checkpoints"][0]["diagnostics"]["unchanged_anchor_ids"] == ["ovimap:1"]
    assert manifest["checkpoints"][1]["diagnostics"]["moved_anchor_ids"] == ["ovimap:1"]
    for record in manifest["inputs"].values():
        assert set(record) == {"path", "sha256", "byte_count"}
    for checkpoint in manifest["checkpoints"]:
        for role in ("snapshot", "entities", "diagnostics_file"):
            path = result.parent / checkpoint[role]["path"]
            assert _record(path, root=result.parent) == checkpoint[role]

    source_index = result.parent / manifest["source_index"]["path"]
    exported = export_temporal_artifact(source_index, tmp_path / "exported")
    temporal = json.loads(exported.read_text(encoding="utf-8"))
    assert temporal["method"] == "OVIV2"
    assert [item["frame_index"] for item in temporal["checkpoints"]] == [2, 3]
    assert {item["entity_id"] for item in temporal["entity_lifecycles"]} == {
        "ovimap:1"
    }


def test_composed_run_never_overwrites_existing_output(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    anchor = _write_anchor_package(tmp_path)
    output = tmp_path / "composed"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=anchor,
            output_root=output,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_unbound_anchor_has_explicit_causal_presence_for_official_export(
    tmp_path: Path,
) -> None:
    result = compose_run(
        source_run_manifest=_write_source_run(tmp_path),
        anchor_manifest=_write_anchor_package(tmp_path, anchor_x=5.0),
        output_root=tmp_path / "composed",
    )
    manifest = json.loads(result.read_text(encoding="utf-8"))
    exported = export_temporal_artifact(
        result.parent / manifest["source_index"]["path"],
        tmp_path / "exported",
    )
    temporal = json.loads(exported.read_text(encoding="utf-8"))

    assert {item["entity_id"] for item in temporal["entity_lifecycles"]} == {
        "ovimap:1"
    }


def test_composed_run_rejects_checkpoint_ahead_of_temporal_export(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["timestamp_ns"] += 1
    _write_json(source, payload)

    with pytest.raises(ValueError, match="checkpoint.*temporal export"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=_write_anchor_package(tmp_path),
            output_root=tmp_path / "composed",
        )


def test_composed_run_rejects_tampered_temporal_sidecar(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    trajectories = source.parent / "trajectories.jsonl"
    trajectories.write_bytes(trajectories.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="trajectories binding mismatch"):
        compose_run(
            source_run_manifest=source,
            anchor_manifest=_write_anchor_package(tmp_path),
            output_root=tmp_path / "composed",
        )

    assert not (tmp_path / "composed").exists()
