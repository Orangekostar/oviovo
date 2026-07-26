from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from scripts.evaluation import finalize_oviv2_tesse_common_v2_release as release_module
from scripts.evaluation.finalize_oviv2_tesse_common_v2_release import (
    finalize_oviv2_common_v2_release,
)
from scripts.evaluation.export_tesse_temporal_artifact import _presence_intervals
from scripts.evaluation.run_oviv2_tesse_cd import (
    TesseCausalCheckpoint,
    _checkpoint_record,
    _compact_checkpoint_record,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICALIZER = (
    ROOT / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py"
)
RELEASE_FINALIZER = (
    ROOT / "scripts/evaluation/finalize_oviv2_tesse_common_v2_release.py"
)
COMMON_FINALIZER = ROOT / "scripts/evaluation/finalize_tesse_common_v2.py"
OFFICIAL_FINALIZER = ROOT / "scripts/evaluation/finalize_tesse_t2.py"
EVALUATOR = ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"
COMMON_FRAMES = tuple(range(100, 551, 50))
OFFICIAL_FRAMES = (90, *COMMON_FRAMES)
EVALUATION_FRAMES = (80, COMMON_FRAMES[0])
SCHEDULED_FRAMES = tuple(sorted(set(OFFICIAL_FRAMES) | set(EVALUATION_FRAMES)))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _relative_record(path: Path, *, root: Path) -> dict[str, object]:
    record = _record(path)
    return {
        "path": Path(os.path.relpath(path, start=root)).as_posix(),
        "sha256": record["sha256"],
        "byte_count": record["byte_count"],
    }


def _compact_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_package(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir()
    arrays = root / "targets.npz"
    arrays.write_bytes(b"deterministic target arrays")
    source = root / "source.json"
    source.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    schedule = root / "schedule.json"
    schedule.write_text('{"method_predictions_used":false}\n', encoding="utf-8")
    ground_truth = root / "ground-truth.bin"
    ground_truth.write_bytes(b"ground truth")
    observability = {
        "apartment_event_01": True,
        "office_event_01": True,
    }
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_targets",
            "dataset": "TESSE-CD",
            "status": "GENERATED",
            "targets_generated": True,
            "prediction_inputs_used": False,
            "metadata": {
                "protocol_complete": True,
                "window_frames": 450,
                "voxel_size_m": 0.05,
                "scenes": ["apartment", "office"],
                "prediction_inputs_used": False,
                "background_observable_by_event": observability,
                "background_observable_event_count": 2,
                "unobservable_revealed_target_event_count": 0,
                "source_manifest": _record(source),
                "schedule": _record(schedule),
                "declared_source_records": {"ground_truth": _record(ground_truth)},
            },
            "sources": [_record(source), _record(schedule), _record(ground_truth)],
            "target_arrays": {
                "path": arrays.name,
                "sha256": _record(arrays)["sha256"],
                "byte_count": arrays.stat().st_size,
                "count": 2,
                "arrays": {
                    "apartment_event_01.revealed_background": {
                        "shape": [1, 3],
                        "dtype": "int64",
                        "element_count": 3,
                    },
                    "office_event_01.revealed_background": {
                        "shape": [1, 3],
                        "dtype": "int64",
                        "element_count": 3,
                    },
                },
            },
        },
    )
    return manifest, arrays, schedule


def _event_metrics(scene: str) -> dict[str, object]:
    event_id = f"{scene}_event_01"
    return {
        "event_count": 1,
        "background_observable_event_count": 1,
        "unobservable_revealed_target_event_count": 0,
        "recovered_event_count": 1,
        "censored_event_count": 0,
        "events": {
            event_id: {
                "intervention_frame_id": 100,
                "frame_ids": list(range(100, 551, 50)),
                "background_observable": True,
                "recovered": True,
                "recovery_frames": 100,
                "right_censored": False,
                "censor_frame": 200,
                "overlapping_intervention": False,
                "censor_reason": None,
            }
        },
    }


def _summary(
    root: Path,
    *,
    scene: str,
    target_manifest: Path,
    target_arrays: Path,
    target_schedule: Path,
    aliases: Path,
    label_space: Path,
    entity_content: str = '{"entity_id":"one"}\n',
) -> Path:
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    sidecars = root / "temporal/sidecars"
    sidecars.mkdir(parents=True)
    temporal = root / "temporal/temporal_manifest.json"
    temporal.write_text(
        json.dumps({"method": "OVIV2", "scene": scene}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schedule = sidecars / "schedule.json"
    schedule.write_bytes(target_schedule.read_bytes())
    checkpoint_sources: dict[str, dict[str, object]] = {}
    for frame in COMMON_FRAMES:
        checkpoint = root / f"temporal/checkpoints/{frame:08d}"
        checkpoint.mkdir(parents=True)
        snapshot = checkpoint / "snapshot.npz"
        snapshot.write_bytes(f"snapshot {frame}\n".encode("ascii"))
        entities = checkpoint / "entities.jsonl"
        entities.write_text(entity_content, encoding="utf-8")
        checkpoint_sources[f"snapshot.{frame:06d}"] = _record(snapshot)
        checkpoint_sources[f"entities.{frame:06d}"] = _record(entities)
    path = evaluation / "summary.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_scene_summary",
            "dataset": "TESSE-CD",
            "protocol": "tesse_cd_common_v2",
            "status": "PASS",
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "scene": scene,
            "metrics": {
                "current_miou": 0.8 if scene == "apartment" else 0.6,
                "ghost_rate": 0.2 if scene == "apartment" else 0.4,
                "background_f5": 0.7 if scene == "apartment" else 0.5,
                "recovery_frames": 100.0 if scene == "apartment" else 300.0,
                "recovery_background_f5": 0.9,
                "recovery_consecutive": 3,
                "checkpoint_step_frames": 50,
                "recovery_horizon_frames": 450,
                **_event_metrics(scene),
            },
            "frames": [],
            "sources": {
                "temporal_index": _record(temporal),
                "target_manifest": _record(target_manifest),
                "target_arrays": _record(target_arrays),
                "schedule": _record(schedule),
                "aliases": _record(aliases),
                "label_space": _record(label_space),
                "evaluator": _record(EVALUATOR),
                **checkpoint_sources,
            },
        },
    )
    return path


def _add_formal_run_artifacts(
    root: Path,
    *,
    scene: str,
    repeat: int,
    frozen_identity: dict[str, object],
) -> None:
    root_status = root.stat()
    execution_base = {
        "schema_version": 1,
        "run_slot": f"{scene}_run{repeat}",
        "output_root": str(root.resolve()),
        "root_device": root_status.st_dev,
        "root_inode": root_status.st_ino,
    }
    run_execution = {
        **execution_base,
        "execution_id": _compact_hash(execution_base),
    }
    formal = {
        "frozen_run_identity": frozen_identity,
        "run_execution": run_execution,
    }
    runner_inputs = root / "inputs"
    runner_inputs.mkdir()
    runner_schedule = runner_inputs / "schedule.json"
    runner_schedule.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    trajectories = root / "trajectories.jsonl"
    trajectories.write_text("", encoding="utf-8")
    runtime_diagnostics = root / "runtime_diagnostics.json"
    _write_json(
        runtime_diagnostics,
        {
            "schema_version": 1,
            "execution_profile": "a2",
            "counters": {"absence": 1},
        },
    )
    source_checkpoints: list[dict[str, object]] = []
    run_checkpoints: list[dict[str, object]] = []
    checkpoint_statuses: list[dict[str, object]] = []
    for frame in SCHEDULED_FRAMES:
        checkpoint_root = root / f"checkpoints/{frame:08d}-{frame}"
        checkpoint_root.mkdir(parents=True)
        checkpoint_status = checkpoint_root / "checkpoint_status.json"
        _write_json(
            checkpoint_status,
            {"schema_version": 1, "frame_index": frame},
        )
        roles = (
            ("common_v2",)
            if frame in COMMON_FRAMES
            else ("official",)
            if frame in OFFICIAL_FRAMES
            else ("occlusion_v1",)
        )
        checkpoint = TesseCausalCheckpoint(
            frame_index=frame,
            timestamp_ns=frame,
            relative_timestamp_ns=frame,
            event_ids=(),
            roles=roles,
        )
        if frame not in OFFICIAL_FRAMES:
            ownership = checkpoint_root / "ownership_checkpoint"
            ownership.mkdir()
            (ownership / "checksums.json").write_text("{}\n", encoding="utf-8")
            run_record = _compact_checkpoint_record(
                checkpoint,
                scene=scene,
                run_root=root,
                checkpoint_root=checkpoint_root,
            )
            run_record["checkpoint_status"] = _relative_record(
                checkpoint_status, root=root
            )
            run_checkpoints.append(run_record)
            continue
        artifact = checkpoint_root / "artifact"
        neutral_snapshot = artifact / f"snapshots/{frame:.6f}_current.npz"
        neutral_snapshot.parent.mkdir(parents=True)
        temporal_snapshot = root / f"temporal/checkpoints/{frame:08d}/snapshot.npz"
        neutral_snapshot.write_bytes(
            temporal_snapshot.read_bytes()
            if temporal_snapshot.exists()
            else f"snapshot {frame}\n".encode("ascii")
        )
        neutral_entities = artifact / f"entities/{frame:.6f}_current.jsonl"
        neutral_entities.parent.mkdir()
        temporal_entities = root / f"temporal/checkpoints/{frame:08d}/entities.jsonl"
        neutral_entities.write_bytes(
            temporal_entities.read_bytes()
            if temporal_entities.exists()
            else b'{"entity_id":"one"}\n'
        )
        voxel_snapshot = checkpoint_root / "voxel_snapshot"
        voxel_snapshot.mkdir()
        (voxel_snapshot / "checksums.json").write_text("{}\n", encoding="utf-8")
        checkpoint_statuses.append(_relative_record(checkpoint_status, root=root))
        source_checkpoints.append(
            {
                "frame_index": frame,
                "timestamp_ns": frame,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "checkpoint_status": _relative_record(checkpoint_status, root=root),
                "snapshot": _relative_record(neutral_snapshot, root=root),
                "entities": _relative_record(neutral_entities, root=root),
            }
        )
        run_record = _checkpoint_record(
            checkpoint,
            scene=scene,
            run_root=root,
            checkpoint_root=checkpoint_root,
        )
        run_record.update(
            {
                "checkpoint_status": _relative_record(checkpoint_status, root=root),
                "neutral_snapshot": _relative_record(neutral_snapshot, root=root),
                "neutral_entities": _relative_record(neutral_entities, root=root),
            }
        )
        run_checkpoints.append(run_record)
    capture_status = root / "capture_status.json"
    _write_json(
        capture_status,
        {
            "schema_version": 1,
            "status": "PASS",
            "scene": scene,
            "mode": "causal_checkpoints",
            "scheduled_frame_indices": list(OFFICIAL_FRAMES),
            "captured_frame_indices": list(OFFICIAL_FRAMES),
            "schedule": _relative_record(runner_schedule, root=root),
            "trajectories": _relative_record(trajectories, root=root),
            "checkpoint_statuses": checkpoint_statuses,
        },
    )
    _write_json(
        root / "source_index.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": scene,
            "schedule": _relative_record(runner_schedule, root=root),
            "capture_status": _relative_record(capture_status, root=root),
            "trajectories": _relative_record(trajectories, root=root),
            "runtime_diagnostics": _relative_record(
                runtime_diagnostics, root=root
            ),
            "checkpoints": source_checkpoints,
            **formal,
        },
    )
    normalized_config = root / "normalized_run_config.json"
    _write_json(
        normalized_config,
        {
            "missing_observation_policy": "signed_depth",
            "algorithm_hash": frozen_identity["algorithm_hash"],
        },
    )
    occlusion_index = root / "occlusion_checkpoint_index.json"
    _write_json(
        occlusion_index,
        {
            "schema_version": 2,
            "manifest_id": "oviv2_tesse_cd_occlusion_checkpoints_v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "algorithm_hash": frozen_identity["algorithm_hash"],
            "run_config": _relative_record(normalized_config, root=root),
            "target_manifest": {"sha256": "1" * 64, "byte_count": 1},
            "evaluation_checkpoint_frames_sha256": "2" * 64,
            "snapshots": [
                {
                    "scene": scene,
                    "frame_index": frame,
                    "timestamp_ns": frame,
                    "relative_timestamp_ns": frame,
                    "consumed_through_frame": frame,
                    "consumed_through_frame_exclusive": frame + 1,
                    "format": (
                        "oviv2_voxel_map_snapshot"
                        if frame in OFFICIAL_FRAMES
                        else "oviv2_compact_ownership_checkpoint"
                    ),
                    "path": (
                        f"checkpoints/{frame:08d}-{frame}/voxel_snapshot"
                        if frame in OFFICIAL_FRAMES
                        else f"checkpoints/{frame:08d}-{frame}/ownership_checkpoint"
                    ),
                    "checksums_sha256": "3" * 64,
                }
                for frame in EVALUATION_FRAMES
            ],
            **formal,
        },
    )
    _write_json(
        root / "run_manifest.json",
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "mode": "causal_checkpoints",
            "scene": scene,
            "missing_observation_policy": "signed_depth",
            "algorithm_hash": frozen_identity["algorithm_hash"],
            "normalized_algorithm_config": {
                "missing_observation_policy": "signed_depth"
            },
            "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
            "maintenance_parameters": {
                "visibility_depth_tolerance_m": None,
                "absence_negative_support": None,
                "ownership_min_net_support": None,
            },
            "processed_frame_count": max(OFFICIAL_FRAMES) + 1,
            "official_schedule_frame_indices": list(OFFICIAL_FRAMES),
            "evaluation_checkpoint_frames": list(EVALUATION_FRAMES),
            "scheduled_frame_indices": list(SCHEDULED_FRAMES),
            "captured_frame_indices": list(SCHEDULED_FRAMES),
            "config": frozen_identity["config"],
            "schedule": {
                "sha256": _record(runner_schedule)["sha256"],
                "byte_count": runner_schedule.stat().st_size,
            },
            "source_bindings": {"fixture": "bound"},
            "occlusion_checkpoint_index": _relative_record(
                occlusion_index, root=root
            ),
            "checkpoints": run_checkpoints,
            **formal,
        },
    )
    temporal_sidecar = root / "temporal/sidecars/source_index.json"
    sidecar_schedule = root / "temporal/sidecars/schedule.json"
    sidecar_capture = root / "temporal/sidecars/capture_status.json"
    _write_json(sidecar_capture, {"schema_version": 1, "status": "PASS"})
    sidecar_trajectories = root / "temporal/trajectories.jsonl"
    sidecar_trajectories.write_text("", encoding="utf-8")
    sidecar_runtime_diagnostics = (
        root / "temporal/sidecars/runtime_diagnostics.json"
    )
    sidecar_runtime_diagnostics.write_bytes(runtime_diagnostics.read_bytes())
    sidecar_checkpoints: list[dict[str, object]] = []
    temporal_checkpoints: list[dict[str, object]] = []
    temporal_statuses: list[dict[str, object]] = []
    for frame in OFFICIAL_FRAMES:
        temporal_checkpoint = root / f"temporal/checkpoints/{frame:08d}"
        temporal_checkpoint.mkdir(parents=True, exist_ok=True)
        temporal_snapshot = temporal_checkpoint / "snapshot.npz"
        temporal_entities = temporal_checkpoint / "entities.jsonl"
        if not temporal_snapshot.exists():
            temporal_snapshot.write_bytes(f"snapshot {frame}\n".encode("ascii"))
        if not temporal_entities.exists():
            temporal_entities.write_text('{"entity_id":"one"}\n', encoding="utf-8")
        temporal_status = (
            root / f"temporal/sidecars/checkpoint_statuses/{frame:08d}.json"
        )
        temporal_status.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            temporal_status,
            {"schema_version": 1, "frame_index": frame},
        )
        temporal_statuses.append(
            _relative_record(temporal_status, root=root / "temporal")
        )
        sidecar_checkpoints.append(
            {
                "frame_index": frame,
                "timestamp_ns": frame,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "checkpoint_status": _relative_record(
                    temporal_status, root=temporal_sidecar.parent
                ),
                "snapshot": _relative_record(
                    temporal_snapshot, root=temporal_sidecar.parent
                ),
                "entities": _relative_record(
                    temporal_entities, root=temporal_sidecar.parent
                ),
            }
        )
        temporal_checkpoints.append(
            {
                "frame_index": frame,
                "timestamp_ns": frame,
                "consumed_through_frame": frame,
                "consumed_through_frame_exclusive": frame + 1,
                "snapshot": _relative_record(
                    temporal_snapshot, root=root / "temporal"
                ),
                "entities": _relative_record(
                    temporal_entities, root=root / "temporal"
                ),
            }
        )
    _write_json(
        temporal_sidecar,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoint_exports",
            "method": "OVIV2",
            "scene": scene,
            "schedule": _relative_record(
                sidecar_schedule, root=temporal_sidecar.parent
            ),
            "capture_status": _relative_record(
                sidecar_capture, root=temporal_sidecar.parent
            ),
            "trajectories": _relative_record(
                sidecar_trajectories, root=temporal_sidecar.parent
            ),
            "runtime_diagnostics": _relative_record(
                sidecar_runtime_diagnostics, root=temporal_sidecar.parent
            ),
            "checkpoints": sidecar_checkpoints,
            **formal,
        },
    )
    temporal = root / "temporal/temporal_manifest.json"
    _write_json(
        temporal,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoints",
            "method": "OVIV2",
            "scene": scene,
            "sources": {
                "source_index": {
                    "path": "sidecars/source_index.json",
                    "sha256": _record(temporal_sidecar)["sha256"],
                    "byte_count": temporal_sidecar.stat().st_size,
                },
                "schedule": _relative_record(sidecar_schedule, root=root / "temporal"),
                "capture_status": _relative_record(
                    sidecar_capture, root=root / "temporal"
                ),
                "trajectories": _relative_record(
                    sidecar_trajectories, root=root / "temporal"
                ),
                "runtime_diagnostics": _relative_record(
                    sidecar_runtime_diagnostics, root=root / "temporal"
                ),
                "checkpoint_statuses": temporal_statuses,
            },
            "checkpoints": temporal_checkpoints,
            "entity_lifecycles": _presence_intervals(
                {
                    "one": {
                        "semantic_label": "fixture",
                        "entity_type": "object",
                        "positions": list(range(len(OFFICIAL_FRAMES))),
                    }
                },
                [
                    {"frame_index": frame, "timestamp_ns": frame}
                    for frame in OFFICIAL_FRAMES
                ],
            ),
            "trajectories": _relative_record(
                sidecar_trajectories, root=root / "temporal"
            ),
            **formal,
        },
    )
    summary = root / "evaluation/summary.json"
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["sources"]["temporal_index"] = _record(temporal)
    _write_json(summary, summary_payload)


def _rewrite_formal_identity(
    root: Path,
    *,
    frozen_identity: dict[str, object] | None = None,
    run_execution: dict[str, object] | None = None,
) -> None:
    artifact_paths = [
        root / "run_manifest.json",
        root / "source_index.json",
        root / "occlusion_checkpoint_index.json",
        root / "temporal/sidecars/source_index.json",
        root / "temporal/temporal_manifest.json",
    ]
    for path in artifact_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if frozen_identity is not None:
            payload["frozen_run_identity"] = frozen_identity
        if run_execution is not None:
            payload["run_execution"] = run_execution
        _write_json(path, payload)
    temporal = root / "temporal/temporal_manifest.json"
    temporal_payload = json.loads(temporal.read_text(encoding="utf-8"))
    sidecar = root / "temporal/sidecars/source_index.json"
    temporal_payload["sources"]["source_index"] = {
        "path": "sidecars/source_index.json",
        "sha256": _record(sidecar)["sha256"],
        "byte_count": sidecar.stat().st_size,
    }
    _write_json(temporal, temporal_payload)
    summary = root / "evaluation/summary.json"
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["sources"]["temporal_index"] = _record(temporal)
    _write_json(summary, summary_payload)


def _fixture(tmp_path: Path, *, repeat_entity_content: str = '{"entity_id":"one"}\n') -> Path:
    target_manifest, target_arrays, schedule = _target_package(tmp_path / "targets")
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("aliases: {}\n", encoding="utf-8")
    labels = {}
    for scene in ("apartment", "office"):
        labels[scene] = tmp_path / f"{scene}-labels.yaml"
        labels[scene].write_text("label_names: {}\n", encoding="utf-8")
        for repeat in (1, 2):
            _summary(
                tmp_path / scene / f"run{repeat}",
                scene=scene,
                target_manifest=target_manifest,
                target_arrays=target_arrays,
                target_schedule=schedule,
                aliases=aliases,
                label_space=labels[scene],
                entity_content=(
                    repeat_entity_content if repeat == 2 else '{"entity_id":"one"}\n'
                ),
            )
    config_payload = {"missing_observation_policy": "signed_depth"}
    algorithm_hash = _compact_hash(config_payload)
    configs: dict[str, Path] = {}
    for scene in ("apartment", "office"):
        configs[scene] = tmp_path / f"{scene}-frozen-config.json"
        _write_json(
            configs[scene],
            {**config_payload, "algorithm_hash": algorithm_hash},
        )
    freeze = tmp_path / "freeze.json"
    freeze_payload = {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "status": "FROZEN",
            "method": "OVIV2",
            "dataset": "TESSE-CD",
            "repository": {
                "commit": "a" * 40,
                "parents": ["c" * 40],
                "tree": "d" * 40,
                "commit_time_utc": "2026-07-22T00:00:00+00:00",
                "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
                "stage3_is_ancestor": True,
                "clean": True,
            },
            "algorithm": {
                "sha256": algorithm_hash,
                "normalized_config": config_payload,
            },
            "shared_bindings": {
                "common_target_manifest": _record(target_manifest),
                "common_target_arrays": _record(target_arrays),
                "schedule": _record(schedule),
                "alias_map": _record(aliases),
                "evaluator": _record(EVALUATOR),
                "finalizers": {
                    "common_v2": _record(COMMON_FINALIZER),
                    "official_t2": _record(OFFICIAL_FINALIZER),
                },
            },
            "release_bindings": {
                "canonical_summary_generator": _record(CANONICALIZER),
                "release_finalizer": _record(RELEASE_FINALIZER),
                "label_spaces": {
                    scene: _record(path) for scene, path in labels.items()
                },
            },
            "scenes": {
                scene: {"frozen_config": _record(configs[scene])}
                for scene in ("apartment", "office")
            },
            "output_roots": {
                f"{scene}_run{repeat}": str(
                    (tmp_path / scene / f"run{repeat}").resolve()
                )
                for scene in ("apartment", "office")
                for repeat in (1, 2)
            },
        }
    _write_json(freeze, freeze_payload)
    freeze_record = _record(freeze)
    for scene in ("apartment", "office"):
        selected = freeze_payload["scenes"][scene]
        frozen_config = selected["frozen_config"]
        frozen_identity = {
            "schema_version": 1,
            "freeze_id": "oviv2-tessecd-v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "freeze_manifest": {
                "sha256": freeze_record["sha256"],
                "byte_count": freeze_record["byte_count"],
            },
            "repository": {"commit": "a" * 40, "tree": "d" * 40},
            "config": {
                "sha256": frozen_config["sha256"],
                "byte_count": frozen_config["byte_count"],
            },
            "algorithm_hash": algorithm_hash,
            "missing_observation_policy": "signed_depth",
            "input_bindings_sha256": _compact_hash(
                {
                    "shared_bindings": freeze_payload["shared_bindings"],
                    "scene": selected,
                }
            ),
        }
        for repeat in (1, 2):
            _add_formal_run_artifacts(
                tmp_path / scene / f"run{repeat}",
                scene=scene,
                repeat=repeat,
                frozen_identity=frozen_identity,
            )
    return freeze


def test_release_finalizer_accepts_canonical_repeats_from_independent_roots(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)

    result = finalize_oviv2_common_v2_release(
        freeze,
        run_id="oviv2-tessecd-v1-test",
        output=tmp_path / "result.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "VERIFIED"
    assert payload["metrics"] == {
        "background_f5": pytest.approx(0.6),
        "current_miou": pytest.approx(0.7),
        "ghost_rate": pytest.approx(0.3),
        "recovery_frames": pytest.approx(200.0),
    }
    assert payload["protocol"]["deterministic_repeat"] == (
        "canonical-role-content-byte-identical"
    )
    assert {item["token"] for item in payload["token_bindings"]} == {
        "T2_OVIV2_CURRENT_MIOU",
        "T2_OVIV2_GHOST_RATE",
        "T2_OVIV2_BG_F5",
        "T2_OVIV2_RECOVERY_FRAMES",
    }
    assert set(payload["formal_run_artifacts"]) == {"apartment", "office"}
    assert set(
        payload["formal_run_artifacts"]["apartment"]["primary"]
    ) == {
        "run_manifest",
        "source_index",
        "runtime_diagnostics",
        "occlusion_checkpoint_index",
        "temporal_manifest",
        "temporal_source_index",
        "temporal_runtime_diagnostics",
    }
    assert payload["frozen_identity"]["scene_run_identities"]["office"][
        "algorithm_hash"
    ] == payload["frozen_identity"]["algorithm_sha256"]


def test_release_finalizer_rejects_real_repeat_content_drift(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path, repeat_entity_content='{"entity_id":"two"}\n')

    with pytest.raises(ValueError, match="canonical summaries must be byte-identical"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="drift",
            output=tmp_path / "result.json",
        )


def test_release_revalidates_runtime_diagnostics_binding(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    diagnostics = tmp_path / "apartment/run1/runtime_diagnostics.json"
    diagnostics.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="runtime diagnostics.*differ"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="runtime-diagnostics-drift",
            output=tmp_path / "result.json",
        )


def test_release_finalizer_rejects_frozen_tool_binding_drift(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["release_bindings"]["canonical_summary_generator"]["sha256"] = "0" * 64
    _write_json(freeze, payload)

    with pytest.raises(ValueError, match="canonical_summary_generator content mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="binding-drift",
            output=tmp_path / "result.json",
        )


def test_release_finalizer_rejects_noncanonical_frozen_binding_path(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["release_bindings"]["canonical_summary_generator"]["path"] = str(
        CANONICALIZER.parent / "../evaluation" / CANONICALIZER.name
    )
    _write_json(freeze, payload)

    with pytest.raises(ValueError, match="canonical_summary_generator path must be canonical"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="path-escape",
            output=tmp_path / "result.json",
        )


def test_release_finalizer_rejects_hardlinked_run_local_sources(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)
    primary = tmp_path / "apartment/run1/temporal/checkpoints/00000100/entities.jsonl"
    repeat = tmp_path / "apartment/run2/temporal/checkpoints/00000100/entities.jsonl"
    repeat.unlink()
    os.link(primary, repeat)

    with pytest.raises(ValueError, match="run-local sources must be independent files"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="hardlink",
            output=tmp_path / "result.json",
        )


def test_release_uses_one_summary_snapshot_for_validation_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _fixture(tmp_path)
    original = release_module.capture_and_canonicalize_summary
    captured_by_scene: dict[str, tuple[float, dict[str, object]]] = {}

    def mutate_after_capture(
        path: Path,
        *,
        artifact_root: Path,
        external_sources: dict[str, Path],
        expected_artifact_root_identity: tuple[int, int] | None = None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, tuple[int, int]],
        dict[str, object],
    ]:
        raw, canonical, raw_record, identities, artifacts = original(
            path,
            artifact_root=artifact_root,
            external_sources=external_sources,
            expected_artifact_root_identity=expected_artifact_root_identity,
        )
        scene = str(raw["scene"])
        captured_by_scene.setdefault(
            scene,
            (float(raw["metrics"]["current_miou"]), raw_record),
        )
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["metrics"]["current_miou"] = 0.1
        _write_json(path, changed)
        return raw, canonical, raw_record, identities, artifacts

    monkeypatch.setattr(
        release_module,
        "capture_and_canonicalize_summary",
        mutate_after_capture,
    )

    result = finalize_oviv2_common_v2_release(
        freeze,
        run_id="single-summary-snapshot",
        output=tmp_path / "result.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    for scene in ("apartment", "office"):
        expected_metric, expected_raw_record = captured_by_scene[scene]
        assert payload["scene_metrics"][scene]["current_miou"] == expected_metric
        assert payload["scene_summaries"][scene]["raw_primary"] == {
            "role": f"{scene}.run1.raw_summary",
            **expected_raw_record,
        }
        on_disk_metric = json.loads(
            (tmp_path / scene / "run1/evaluation/summary.json").read_text(
                encoding="utf-8"
            )
        )["metrics"]["current_miou"]
        assert on_disk_metric == 0.1
        assert payload["scene_metrics"][scene]["current_miou"] != (
            on_disk_metric
        )


def test_release_rejects_captured_run_local_inode_reuse_after_path_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _fixture(tmp_path)
    apartment_source = (
        tmp_path / "apartment/run1/temporal/checkpoints/00000100/entities.jsonl"
    )
    office_source = (
        tmp_path / "office/run1/temporal/checkpoints/00000100/entities.jsonl"
    )
    office_source.unlink()
    os.link(apartment_source, office_source)
    original = release_module.capture_and_canonicalize_summary

    def split_after_capture(
        path: Path,
        *,
        artifact_root: Path,
        external_sources: dict[str, Path],
        expected_artifact_root_identity: tuple[int, int] | None = None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, tuple[int, int]],
        dict[str, object],
    ]:
        captured = original(
            path,
            artifact_root=artifact_root,
            external_sources=external_sources,
            expected_artifact_root_identity=expected_artifact_root_identity,
        )
        if artifact_root == tmp_path / "office/run1":
            content = office_source.read_bytes()
            office_source.unlink()
            office_source.write_bytes(content)
        return captured

    monkeypatch.setattr(
        release_module,
        "capture_and_canonicalize_summary",
        split_after_capture,
    )

    with pytest.raises(ValueError, match="run-local sources must be independent files"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="captured-inode-reuse",
            output=tmp_path / "result.json",
        )


def test_release_does_not_reopen_target_manifest_after_stable_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _fixture(tmp_path)
    target_manifest = tmp_path / "targets/manifest.json"
    original = release_module.pinned_common._validate_target_package

    def mutate_before_pinned_validation(
        manifest_record: dict[str, object],
        arrays_record: dict[str, object],
        schedule_record: dict[str, object],
    ) -> dict[str, object]:
        changed = json.loads(target_manifest.read_text(encoding="utf-8"))
        changed["target_arrays"]["arrays"][
            "apartment_event_01.revealed_background"
        ]["element_count"] = 4
        _write_json(target_manifest, changed)
        return original(manifest_record, arrays_record, schedule_record)

    monkeypatch.setattr(
        release_module.pinned_common,
        "_validate_target_package",
        mutate_before_pinned_validation,
    )

    result = finalize_oviv2_common_v2_release(
        freeze,
        run_id="target-single-snapshot",
        output=tmp_path / "result.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["target_package"]["manifest"]["sha256"] == (
        hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    )


def test_release_rejects_run_identity_algorithm_drift(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    for scene in ("apartment", "office"):
        primary = tmp_path / scene / "run1/run_manifest.json"
        wrong_identity = json.loads(primary.read_text(encoding="utf-8"))[
            "frozen_run_identity"
        ]
        wrong_identity["algorithm_hash"] = "0" * 64
        for repeat in (1, 2):
            _rewrite_formal_identity(
                tmp_path / scene / f"run{repeat}",
                frozen_identity=wrong_identity,
            )

    with pytest.raises(ValueError, match="frozen run identity mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="algorithm-identity-drift",
            output=tmp_path / "result.json",
        )


def test_release_rejects_reused_execution_identity(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    primary = tmp_path / "apartment/run1/run_manifest.json"
    reused_execution = json.loads(primary.read_text(encoding="utf-8"))[
        "run_execution"
    ]
    _rewrite_formal_identity(
        tmp_path / "apartment/run2",
        run_execution=reused_execution,
    )

    with pytest.raises(ValueError, match="execution identity mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="reused-execution",
            output=tmp_path / "result.json",
        )


def test_release_rejects_identity_disagreement_within_one_run(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    run_manifest = tmp_path / "office/run1/run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["frozen_run_identity"]["input_bindings_sha256"] = "0" * 64
    _write_json(run_manifest, payload)

    with pytest.raises(ValueError, match="frozen run identity mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="artifact-identity-disagreement",
            output=tmp_path / "result.json",
        )


@pytest.mark.parametrize(
    ("identity_field", "malformed_value"),
    (
        ("frozen_run_identity", True),
        ("run_execution", 1.0),
    ),
)
def test_release_rejects_type_confused_identity_schema_version(
    tmp_path: Path,
    identity_field: str,
    malformed_value: object,
) -> None:
    freeze = _fixture(tmp_path)
    run_manifest = tmp_path / "apartment/run1/run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload[identity_field]["schema_version"] = malformed_value
    _write_json(run_manifest, payload)

    with pytest.raises(ValueError, match="identity mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="type-confused-identity",
            output=tmp_path / "result.json",
        )


def test_release_rejects_output_root_replacement_before_evidence_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _fixture(tmp_path)
    target_root = tmp_path / "apartment/run1"
    parked_root = tmp_path / "apartment/run1-original"
    original = release_module.capture_and_canonicalize_summary
    replaced = False

    def replace_root_before_capture(
        path: Path,
        *,
        artifact_root: Path,
        external_sources: dict[str, Path],
        expected_artifact_root_identity: tuple[int, int] | None = None,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, tuple[int, int]],
        dict[str, object],
    ]:
        nonlocal replaced
        if artifact_root == target_root and not replaced:
            target_root.rename(parked_root)
            shutil.copytree(parked_root, target_root)
            replaced = True
        return original(
            path,
            artifact_root=artifact_root,
            external_sources=external_sources,
            expected_artifact_root_identity=expected_artifact_root_identity,
        )

    monkeypatch.setattr(
        release_module,
        "capture_and_canonicalize_summary",
        replace_root_before_capture,
    )

    with pytest.raises(ValueError, match="artifact root identity changed"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="root-replacement",
            output=tmp_path / "result.json",
        )


def test_release_rejects_non_runner_formal_artifact_contract(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path)
    run_manifest = tmp_path / "office/run2/run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["method"] = payload.pop("method_id")
    _write_json(run_manifest, payload)

    with pytest.raises(ValueError, match="run_manifest contract mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="non-runner-artifact",
            output=tmp_path / "result.json",
        )


def test_release_rejects_summary_checkpoint_outside_event_and_temporal_contract(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)
    for repeat in (1, 2):
        root = tmp_path / f"apartment/run{repeat}"
        checkpoint = root / "temporal/checkpoints/00000999"
        checkpoint.mkdir()
        snapshot = checkpoint / "snapshot.npz"
        snapshot.write_bytes(b"unbound snapshot\n")
        entities = checkpoint / "entities.jsonl"
        entities.write_text('{"entity_id":"unbound"}\n', encoding="utf-8")
        summary = root / "evaluation/summary.json"
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["sources"]["snapshot.000999"] = _record(snapshot)
        payload["sources"]["entities.000999"] = _record(entities)
        _write_json(summary, payload)

    with pytest.raises(ValueError, match="summary checkpoint coverage mismatch"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="summary-extra-checkpoint",
            output=tmp_path / "result.json",
        )


def test_release_rejects_temporal_checkpoint_content_disagreement(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)
    for repeat in (1, 2):
        root = tmp_path / f"office/run{repeat}"
        temporal = root / "temporal/temporal_manifest.json"
        payload = json.loads(temporal.read_text(encoding="utf-8"))
        payload["checkpoints"][1]["snapshot"]["sha256"] = "9" * 64
        _write_json(temporal, payload)
        summary = root / "evaluation/summary.json"
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
        summary_payload["sources"]["temporal_index"] = _record(temporal)
        _write_json(summary, summary_payload)

    with pytest.raises(ValueError, match="temporal snapshot content records differ"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="temporal-content-disagreement",
            output=tmp_path / "result.json",
        )


def test_release_rejects_root_source_snapshot_disagreement_with_temporal(
    tmp_path: Path,
) -> None:
    freeze = _fixture(tmp_path)
    for repeat in (1, 2):
        root = tmp_path / f"apartment/run{repeat}"
        source_index = root / "source_index.json"
        source_payload = json.loads(source_index.read_text(encoding="utf-8"))
        source_checkpoint = next(
            item
            for item in source_payload["checkpoints"]
            if item["frame_index"] == OFFICIAL_FRAMES[0]
        )
        source_checkpoint["snapshot"]["sha256"] = "8" * 64
        _write_json(source_index, source_payload)

        run_manifest = root / "run_manifest.json"
        run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        run_checkpoint = next(
            item
            for item in run_payload["checkpoints"]
            if item["frame_index"] == OFFICIAL_FRAMES[0]
        )
        run_checkpoint["neutral_snapshot"]["sha256"] = "8" * 64
        _write_json(run_manifest, run_payload)

    with pytest.raises(
        ValueError, match="root-to-temporal snapshot content records differ"
    ):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="root-temporal-content-disagreement",
            output=tmp_path / "result.json",
        )
