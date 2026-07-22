from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.evaluation import finalize_oviv2_tesse_common_v2_release as release_module
from scripts.evaluation.finalize_oviv2_tesse_common_v2_release import (
    finalize_oviv2_common_v2_release,
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
    checkpoint = root / "temporal/checkpoints/00000100"
    checkpoint.mkdir(parents=True)
    sidecars = root / "temporal/sidecars"
    sidecars.mkdir()
    temporal = root / "temporal/temporal_manifest.json"
    temporal.write_text(
        json.dumps({"method": "OVIV2", "scene": scene}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schedule = sidecars / "schedule.json"
    schedule.write_bytes(target_schedule.read_bytes())
    snapshot = checkpoint / "snapshot.npz"
    snapshot.write_bytes(b"snapshot")
    entities = checkpoint / "entities.jsonl"
    entities.write_text(entity_content, encoding="utf-8")
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
                "snapshot.000100": _record(snapshot),
                "entities.000100": _record(entities),
            },
        },
    )
    return path


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
    freeze = tmp_path / "freeze.json"
    _write_json(
        freeze,
        {
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
            "algorithm": {"sha256": "b" * 64, "normalized_config": {}},
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
            "output_roots": {
                f"{scene}_run{repeat}": str(
                    (tmp_path / scene / f"run{repeat}").resolve()
                )
                for scene in ("apartment", "office")
                for repeat in (1, 2)
            },
        },
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


def test_release_finalizer_rejects_real_repeat_content_drift(tmp_path: Path) -> None:
    freeze = _fixture(tmp_path, repeat_entity_content='{"entity_id":"two"}\n')

    with pytest.raises(ValueError, match="canonical summaries must be byte-identical"):
        finalize_oviv2_common_v2_release(
            freeze,
            run_id="drift",
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
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, tuple[int, int]],
    ]:
        raw, canonical, raw_record, identities = original(
            path,
            artifact_root=artifact_root,
            external_sources=external_sources,
        )
        scene = str(raw["scene"])
        captured_by_scene.setdefault(
            scene,
            (float(raw["metrics"]["current_miou"]), raw_record),
        )
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["metrics"]["current_miou"] = 0.1
        _write_json(path, changed)
        return raw, canonical, raw_record, identities

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
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, tuple[int, int]],
    ]:
        captured = original(
            path,
            artifact_root=artifact_root,
            external_sources=external_sources,
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
