from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from scripts.evaluation import evaluate_tesse_cd_common_v2 as evaluator_module
from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    KhronosTextHead,
    _background_unobservable_events,
    classify_khronos_open_embedding,
    evaluate_common_v2,
    evaluate_frozen_common_v2,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SCHEDULE = (
    ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
)
OFFICIAL_SCHEDULE_SHA256 = (
    "fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"
)


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    empty: bool = False,
    ghost: bool = False,
    target_status: str = "GENERATED",
    revealed_observable: bool = True,
) -> tuple[Path, Path]:
    root.mkdir()
    frames = list(range(10, 461, 50))
    schedule_path = root / "schedule.json"
    _write_json(
        schedule_path,
        {
            "schema_version": 2,
            "manifest_id": "tesse_cd_causal_schedule_v2",
            "dataset": "TESSE-CD",
            "method_predictions_used": False,
            "scenes": {
                "apartment": {
                    "frame_count": 500,
                    "events": [
                        {
                            "event_id": "apartment_event_01",
                            "intervention_frame_index": 10,
                            "common_checkpoint_frame_indices": frames,
                        }
                    ],
                    "entries": [
                        {
                            "frame_index": frame,
                            "timestamp_ns": frame * 100,
                            "roles": ["common_v2"],
                            "event_ids": ["apartment_event_01"],
                        }
                        for frame in frames
                    ],
                }
            },
        },
    )
    label_space = root / "labels.yaml"
    label_space.write_text(
        yaml.safe_dump(
            {
                "object_labels": [1],
                "label_names": [
                    {"label": 0, "name": "Unknown"},
                    {"label": 1, "name": "Chair"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    aliases = root / "aliases.yaml"
    aliases.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "protocol": "tesse_cd_common_v2",
                "unknown_name": "Unknown",
                "scenes": {
                    "apartment": {
                        "label_space": {
                            "filename": label_space.name,
                            "sha256": hashlib.sha256(label_space.read_bytes()).hexdigest(),
                        },
                        "aliases": {"Chair": ["chair"]},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    entity_points = np.empty((0, 3), dtype=np.float32)
    entities: list[EntityPrediction] = []
    background = None
    if not empty:
        entity_points = np.asarray(
            [[0.075, 0.025, 1.025] if ghost else [0.025, 0.025, 1.025]],
            dtype=np.float32,
        )
        entities = [
            EntityPrediction(
                entity_id="chair-1",
                points_xyz=entity_points,
                semantic_embedding=None,
                semantic_label="chair",
                semantic_score=1.0,
                lifecycle_state="active",
                first_seen=0.0,
                last_seen=0.1,
                metadata={"entity_type": "object"},
            )
        ]
        background = np.asarray([[0.025, 0.025, 1.125]], dtype=np.float32)
    paths = write_map_snapshot(
        MapSnapshot(
            method="OVI-MAP (frozen)",
            scene_id="apartment",
            timestamp=0.1,
            entities=entities,
            background_xyz=background,
            scope="current",
        ),
        root / "snapshot",
    )
    checkpoints = [
        {
            "frame_index": frame,
            "timestamp_ns": frame * 100,
            "event_ids": ["apartment_event_01"],
            "snapshot_source": "frozen_snapshot",
            "consumed_through_frame_exclusive": 10,
        }
        for frame in frames
    ]
    index = root / "index.json"
    _write_json(
        index,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_frozen_temporal_artifact",
            "dataset": "TESSE-CD",
            "protocol": "tesse_cd_common_v2",
            "status": "PASS",
            "method": "OVIMAP_FROZEN",
            "mode": "frozen",
            "scene": "apartment",
            "updates_after_freeze": 0,
            "freeze_stop_exclusive": 10,
            "prediction_semantics": "pre_intervention_snapshot_reused_without_updates",
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "frozen_snapshot": {
                "snapshot": _record(paths["snapshot"]),
                "entities": _record(paths["entities"]),
                "snapshot_method": "OVI-MAP (frozen)",
                "source_timestamp": 0.1,
                "entity_count": len(entities),
                "unlabeled_entity_count": 0,
                "background_present": background is not None,
            },
            "sources": {
                "schedule": _record(schedule_path),
                "aliases": _record(aliases),
                "label_space": _record(label_space),
            },
        },
    )

    arrays: dict[str, np.ndarray] = {
        "apartment_event_01.region": np.asarray(
            [[0, 0, 20], [1, 0, 20], [0, 0, 22]], dtype=np.int64
        ),
        "apartment_event_01.revealed_background": np.asarray(
            [[0, 0, 22]] if revealed_observable else [], dtype=np.int64
        ).reshape((-1, 3)),
    }
    for frame in frames:
        arrays[f"apartment_event_01.confirmed_free.{frame:06d}"] = np.asarray(
            [[1, 0, 20]], dtype=np.int64
        )
        arrays[f"apartment.current_semantic.{frame:06d}"] = np.asarray(
            [[0, 0, 20, 1]], dtype=np.int64
        )
    targets_path = root / "targets.npz"
    np.savez_compressed(targets_path, **arrays)
    target_manifest = root / "target_manifest.json"
    _write_json(
        target_manifest,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_targets",
            "dataset": "TESSE-CD",
            "status": target_status,
            "targets_generated": True,
            "prediction_inputs_used": False,
            "metadata": {
                "protocol_complete": target_status == "GENERATED",
                "scenes": ["apartment", "office"],
                "window_frames": 450,
                "voxel_size_m": 0.05,
                "schedule": _record(schedule_path),
                "background_observable_by_event": {
                    "apartment_event_01": revealed_observable,
                },
                "background_observable_event_count": int(revealed_observable),
                "unobservable_revealed_target_event_count": int(
                    not revealed_observable
                ),
            },
            "target_arrays": {
                **_record(targets_path),
                "path": targets_path.name,
                "count": len(arrays),
                "arrays": {
                    name: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "element_count": int(array.size),
                    }
                    for name, array in sorted(arrays.items())
                },
            },
        },
    )
    return index, target_manifest


def _causal_fixture(
    root: Path,
    *,
    future: bool = False,
    temporal_method: str = "DualMap",
    snapshot_method: str = "DualMap",
) -> tuple[Path, Path, Path, Path]:
    frozen_index, targets = _fixture(root)
    frozen = json.loads(frozen_index.read_text(encoding="utf-8"))
    schedule = Path(frozen["sources"]["schedule"]["path"])
    aliases = Path(frozen["sources"]["aliases"]["path"])
    label_space = Path(frozen["sources"]["label_space"]["path"])
    frames = list(range(10, 461, 50))
    checkpoints = []
    for position, frame in enumerate(frames):
        entities = []
        background = None
        if position > 0:
            entities = [
                EntityPrediction(
                    entity_id="chair-1",
                    points_xyz=np.asarray([[0.025, 0.025, 1.025]], dtype=np.float32),
                    semantic_embedding=None,
                    semantic_label="chair",
                    semantic_score=1.0,
                    lifecycle_state="active",
                    first_seen=0.0,
                    last_seen=float(frame),
                    metadata={"entity_type": "object"},
                )
            ]
            background = np.asarray([[0.025, 0.025, 1.125]], dtype=np.float32)
        paths = write_map_snapshot(
            MapSnapshot(
                method=snapshot_method,
                scene_id="apartment",
                timestamp=float(frame * 100),
                entities=entities,
                background_xyz=background,
                scope="current",
            ),
            root / "causal" / f"{frame:08d}",
        )
        checkpoints.append(
            {
                "frame_index": frame,
                "timestamp_ns": frame * 100,
                "consumed_through_frame": frame + (1 if future and position == 0 else 0),
                "consumed_through_frame_exclusive": frame + 1,
                "snapshot": _record(paths["snapshot"]),
                "entities": _record(paths["entities"]),
            }
        )
    temporal = root / "causal_temporal_manifest.json"
    _write_json(
        temporal,
        {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "mode": "causal_checkpoints",
            "method": temporal_method,
            "scene": "apartment",
            "sources": {"schedule": _record(schedule)},
            "checkpoints": checkpoints,
            "entity_lifecycles": [],
        },
    )
    return temporal, targets, aliases, label_space


def _bind_target_schedule(target_manifest: Path, schedule: Path) -> None:
    payload = json.loads(target_manifest.read_text(encoding="utf-8"))
    declaration = _record(schedule)
    declaration["path"] = schedule.relative_to(target_manifest.parent).as_posix()
    payload["metadata"]["schedule"] = declaration
    _write_json(target_manifest, payload)


def test_cli_help_runs_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--target-manifest" in completed.stdout
    assert "--aliases" in completed.stdout
    assert "--label-space" in completed.stdout


def test_perfect_frozen_snapshot_produces_verified_scene_metrics(tmp_path: Path) -> None:
    index, targets = _fixture(tmp_path / "source")

    result = evaluate_frozen_common_v2(index, targets, tmp_path / "output")
    repeat = evaluate_frozen_common_v2(index, targets, tmp_path / "repeat")

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["metrics"]["current_miou"] == pytest.approx(1.0)
    assert payload["metrics"]["ghost_rate"] == pytest.approx(0.0)
    assert payload["metrics"]["background_f5"] == pytest.approx(1.0)
    assert payload["metrics"]["recovery_frames"] == pytest.approx(0.0)
    assert payload["metrics"]["recovered_event_count"] == 1
    assert len(payload["frames"]) == 10
    assert result.read_bytes() == repeat.read_bytes()


def test_empty_frozen_snapshot_is_censored_without_inventing_predictions(
    tmp_path: Path,
) -> None:
    index, targets = _fixture(tmp_path / "source", empty=True)

    result = evaluate_frozen_common_v2(index, targets, tmp_path / "output")

    metrics = json.loads(result.read_text(encoding="utf-8"))["metrics"]
    assert metrics["current_miou"] == pytest.approx(0.0)
    assert metrics["ghost_rate"] == pytest.approx(0.0)
    assert metrics["background_f5"] == pytest.approx(0.0)
    assert metrics["recovery_frames"] == pytest.approx(450.0)
    assert metrics["censored_event_count"] == 1


def test_stale_object_point_in_confirmed_free_space_is_a_ghost(tmp_path: Path) -> None:
    index, targets = _fixture(tmp_path / "source", ghost=True)

    result = evaluate_frozen_common_v2(index, targets, tmp_path / "output")

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["metrics"]["ghost_rate"] == pytest.approx(1.0)
    assert all(frame["ghost_count"] == 1 for frame in payload["frames"])


def test_rejects_smoke_target_package(tmp_path: Path) -> None:
    index, targets = _fixture(tmp_path / "source", target_status="SMOKE")

    with pytest.raises(ValueError, match="GENERATED"):
        evaluate_frozen_common_v2(index, targets, tmp_path / "output")


def test_rejects_mutated_target_array_hash(tmp_path: Path) -> None:
    index, targets = _fixture(tmp_path / "source")
    payload = json.loads(targets.read_text(encoding="utf-8"))
    target_arrays = targets.parent / payload["target_arrays"]["path"]
    target_arrays.write_bytes(target_arrays.read_bytes() + b"mutated")

    with pytest.raises(ValueError, match="target arrays SHA256"):
        evaluate_frozen_common_v2(index, targets, tmp_path / "output")


def test_rejects_future_snapshot_binding(tmp_path: Path) -> None:
    index, targets = _fixture(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["consumed_through_frame_exclusive"] = 11
    _write_json(index, payload)

    with pytest.raises(ValueError, match="future|freeze boundary"):
        evaluate_frozen_common_v2(index, targets, tmp_path / "output")


def test_rejects_scene_without_background_observable_event(tmp_path: Path) -> None:
    index, targets = _fixture(
        tmp_path / "source", revealed_observable=False
    )

    with pytest.raises(ValueError, match="at least one background-observable event"):
        evaluate_frozen_common_v2(index, targets, tmp_path / "output")


def test_causal_temporal_evaluator_loads_each_checkpoint_snapshot(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")

    result = evaluate_common_v2(
        temporal,
        targets,
        aliases,
        label_space,
        tmp_path / "output",
    )
    repeat = evaluate_common_v2(
        temporal,
        targets,
        aliases,
        label_space,
        tmp_path / "repeat",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["method"] == "DualMap"
    assert payload["mode"] == "causal_checkpoints"
    assert payload["frames"][0]["current_miou"] == pytest.approx(0.0)
    assert all(
        frame["current_miou"] == pytest.approx(1.0)
        for frame in payload["frames"][1:]
    )
    assert payload["metrics"]["recovery_frames"] == pytest.approx(50.0)
    assert result.read_bytes() == repeat.read_bytes()
    assert {
        "temporal_index",
        "target_manifest",
        "target_arrays",
        "schedule",
        "aliases",
        "label_space",
        "evaluator",
    } <= set(payload["sources"])


def test_load_targets_accepts_checked_schedule_bytes_at_distinct_paths(
    tmp_path: Path,
) -> None:
    _, targets = _fixture(tmp_path / "source")
    target_schedule = targets.parent / "target_schedule.json"
    temporal_schedule = tmp_path / "temporal_schedule.json"
    target_schedule.write_bytes(OFFICIAL_SCHEDULE.read_bytes())
    temporal_schedule.write_bytes(OFFICIAL_SCHEDULE.read_bytes())
    _bind_target_schedule(targets, target_schedule)
    temporal_record = _record(temporal_schedule)

    arrays, _, metadata = evaluator_module._load_targets(
        targets,
        schedule_record=temporal_record,
    )

    assert arrays
    assert metadata["schedule"]["sha256"] == OFFICIAL_SCHEDULE_SHA256
    assert temporal_record["sha256"] == OFFICIAL_SCHEDULE_SHA256


def test_load_targets_resolves_repository_scoped_schedule_after_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, targets = _fixture(tmp_path / "source")
    repository_root = tmp_path / "relocated-repository"
    relative_schedule = Path(
        "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
    )
    relocated_schedule = repository_root / relative_schedule
    relocated_schedule.parent.mkdir(parents=True)
    source_schedule = tmp_path / "source" / "schedule.json"
    relocated_schedule.write_bytes(source_schedule.read_bytes())
    payload = json.loads(targets.read_text(encoding="utf-8"))
    payload["metadata"]["schedule"] = {
        **_record(relocated_schedule),
        "path": relative_schedule.as_posix(),
        "path_base": "repository",
    }
    _write_json(targets, payload)
    monkeypatch.setattr(evaluator_module, "ROOT", repository_root)

    arrays, _, metadata = evaluator_module._load_targets(
        targets,
        schedule_record=_record(relocated_schedule),
    )

    assert arrays
    assert metadata["schedule"]["path_base"] == "repository"


def test_declared_file_accepts_explicit_absolute_external_path(tmp_path: Path) -> None:
    source = tmp_path / "external.json"
    source.write_text("{}\n", encoding="utf-8")
    declaration = {**_record(source), "path_base": "absolute"}

    path, record = evaluator_module._declared_file(
        declaration,
        label="external source",
    )

    assert path == source.resolve()
    assert record == _record(source)


def test_declared_file_rejects_absolute_path_with_repository_scope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    declaration = {**_record(source), "path_base": "repository"}

    with pytest.raises(ValueError, match="repository.*relative"):
        evaluator_module._declared_file(
            declaration,
            label="repository source",
            base=tmp_path,
        )


def test_declared_file_rejects_repository_scope_escape(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    source = tmp_path / "outside.json"
    source.write_text("{}\n", encoding="utf-8")
    declaration = {
        **_record(source),
        "path": "../outside.json",
        "path_base": "repository",
    }

    with pytest.raises(ValueError, match="repository.*escape"):
        evaluator_module._declared_file(
            declaration,
            label="repository source",
            base=repository_root,
            repository_root=repository_root,
        )


def test_declared_file_rejects_unknown_path_base(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    declaration = {**_record(source), "path_base": "working-directory"}

    with pytest.raises(ValueError, match="unknown path_base"):
        evaluator_module._declared_file(
            declaration,
            label="unknown source",
        )


def test_load_targets_rejects_checked_schedule_content_drift(tmp_path: Path) -> None:
    _, targets = _fixture(tmp_path / "source")
    target_schedule = targets.parent / "target_schedule.json"
    temporal_schedule = tmp_path / "temporal_schedule.json"
    target_schedule.write_bytes(OFFICIAL_SCHEDULE.read_bytes() + b" ")
    temporal_schedule.write_bytes(OFFICIAL_SCHEDULE.read_bytes())
    _bind_target_schedule(targets, target_schedule)

    with pytest.raises(ValueError, match="schedule binding mismatch"):
        evaluator_module._load_targets(
            targets,
            schedule_record=_record(temporal_schedule),
        )


def test_load_targets_rejects_duplicate_manifest_json_keys(tmp_path: Path) -> None:
    _, targets = _fixture(tmp_path / "source")
    payload = json.loads(targets.read_text(encoding="utf-8"))
    schedule_record = payload["metadata"]["schedule"]
    payload.pop("schema_version")
    targets.write_text(
        '{"schema_version":1,"schema_version":1,'
        + json.dumps(payload, sort_keys=True)[1:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluator_module._load_targets(
            targets,
            schedule_record=schedule_record,
        )


def test_schedule_loader_requires_integer_schema_version(tmp_path: Path) -> None:
    index, _ = _fixture(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    schedule = Path(payload["sources"]["schedule"]["path"])
    schedule_payload = json.loads(schedule.read_text(encoding="utf-8"))
    schedule_payload["schema_version"] = 2.0
    _write_json(schedule, schedule_payload)

    with pytest.raises(ValueError, match="schedule identity"):
        evaluator_module._load_schedule(schedule, scene="apartment")


def test_causal_evaluator_accepts_equal_schedule_bytes_at_distinct_paths(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")
    temporal_payload = json.loads(temporal.read_text(encoding="utf-8"))
    source_schedule = Path(temporal_payload["sources"]["schedule"]["path"])
    target_schedule = targets.parent / "target_schedule.json"
    target_schedule.write_bytes(source_schedule.read_bytes())
    _bind_target_schedule(targets, target_schedule)

    result = evaluate_common_v2(
        temporal,
        targets,
        aliases,
        label_space,
        tmp_path / "output",
    )

    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "PASS"


def test_causal_evaluator_rejects_target_schedule_content_drift(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")
    temporal_payload = json.loads(temporal.read_text(encoding="utf-8"))
    source_schedule = Path(temporal_payload["sources"]["schedule"]["path"])
    target_schedule = targets.parent / "target_schedule.json"
    target_schedule.write_bytes(source_schedule.read_bytes() + b" ")
    _bind_target_schedule(targets, target_schedule)

    with pytest.raises(ValueError, match="schedule binding mismatch"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_causal_temporal_evaluator_accepts_registered_dualmap_artifact_label(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(
        tmp_path / "source", temporal_method="DUALMAP"
    )

    result = evaluate_common_v2(
        temporal,
        targets,
        aliases,
        label_space,
        tmp_path / "output",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["method"] == "DUALMAP"


def test_causal_temporal_evaluator_accepts_registered_oviv2_artifact_label(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(
        tmp_path / "source", temporal_method="OVIV2", snapshot_method="OVIV2"
    )

    result = evaluate_common_v2(
        temporal,
        targets,
        aliases,
        label_space,
        tmp_path / "output",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["method"] == "OVIV2"


def test_causal_temporal_evaluator_rejects_unregistered_artifact_label(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(
        tmp_path / "source", temporal_method="CONCEPTGRAPHS"
    )

    with pytest.raises(ValueError, match="unsupported causal.*method"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


@pytest.mark.parametrize("method", ["UNKNOWN_CAUSAL", "OVIOVO"])
def test_causal_temporal_evaluator_rejects_unregistered_matching_method(
    tmp_path: Path, method: str
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(
        tmp_path / "source",
        temporal_method=method,
        snapshot_method=method,
    )

    with pytest.raises(ValueError, match="unsupported causal.*method"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_causal_temporal_evaluator_rejects_future_checkpoint(tmp_path: Path) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(
        tmp_path / "source", future=True
    )

    with pytest.raises(ValueError, match="future"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_causal_temporal_evaluator_rejects_wrong_exclusive_boundary(
    tmp_path: Path,
) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["consumed_through_frame_exclusive"] += 1
    _write_json(temporal, payload)

    with pytest.raises(ValueError, match="exclusive|future"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_declared_file_requires_explicit_byte_count(tmp_path: Path) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["snapshot"].pop("byte_count")
    _write_json(temporal, payload)

    with pytest.raises(ValueError, match="byte count.*non-negative integer"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_khronos_open_uses_cosine_text_head_and_not_numeric_ids() -> None:
    head = KhronosTextHead(
        semantic_ids=np.asarray([1, 2], dtype=np.int64),
        names=("Chair", "Table"),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        manifest_record={},
        arrays_record={},
    )

    assert (
        classify_khronos_open_embedding(
            np.asarray([0.1, 0.9], dtype=np.float32), head
        )
        == "Table"
    )
    assert classify_khronos_open_embedding(np.asarray([], dtype=np.float32), head) is None
    with pytest.raises(ValueError, match="dimension|finite|zero"):
        classify_khronos_open_embedding(np.asarray([1.0, 2.0, 3.0]), head)


def test_khronos_open_fails_closed_without_frozen_text_head(tmp_path: Path) -> None:
    temporal, targets, aliases, label_space = _causal_fixture(tmp_path / "source")
    payload = json.loads(temporal.read_text(encoding="utf-8"))
    payload["method"] = "KHRONOS_OPEN"
    _write_json(temporal, payload)

    with pytest.raises(ValueError, match="text head"):
        evaluate_common_v2(
            temporal,
            targets,
            aliases,
            label_space,
            tmp_path / "output",
        )


def test_observability_accepts_global_mapping_but_filters_current_scene() -> None:
    events = [{"event_id": "apartment_event_01"}]
    arrays = {
        "apartment_event_01.revealed_background": np.asarray(
            [[0, 0, 0]], dtype=np.int64
        )
    }
    metadata = {
        "background_observable_by_event": {
            "apartment_event_01": True,
            "office_event_01": False,
        },
        "background_observable_event_count": 1,
        "unobservable_revealed_target_event_count": 1,
    }

    assert _background_unobservable_events(events, arrays, metadata) == set()

    metadata["background_observable_event_count"] = 2
    with pytest.raises(ValueError, match="counts"):
        _background_unobservable_events(events, arrays, metadata)


def test_frozen_semantic_tree_is_built_once_per_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, targets = _fixture(tmp_path / "source")
    original = evaluator_module.cKDTree
    builds = 0

    def counted_tree(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator_module, "cKDTree", counted_tree)

    evaluate_frozen_common_v2(index, targets, tmp_path / "output")

    assert builds == 1
