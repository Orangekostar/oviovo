from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation import finalize_tesse_common_v2 as finalizer_module
from scripts.evaluation.finalize_tesse_common_v2 import finalize_common_v2
from tools.import_benchmark_results import import_results


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


def _target_package(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.is_file():
        return manifest
    arrays = root / "targets.npz"
    arrays.write_bytes(b"deterministic target fixture")
    source_manifest = root / "official_source.json"
    source_manifest.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    schedule = root / "causal_schedule.json"
    schedule.write_text('{"method_predictions_used":false}\n', encoding="utf-8")
    ground_truth = root / "official_ground_truth.bin"
    ground_truth.write_bytes(b"official ground truth")
    observability = {
        "apartment_event_01": True,
        "apartment_event_02": False,
        "office_event_01": True,
    }
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
                "unobservable_revealed_target_event_count": 1,
                "source_manifest": _record(source_manifest),
                "schedule": _record(schedule),
                "declared_source_records": {
                    "official_ground_truth": _record(ground_truth)
                },
            },
            "sources": [
                _record(source_manifest),
                _record(schedule),
                _record(ground_truth),
            ],
            "target_arrays": {
                "path": "targets.npz",
                "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                "byte_count": arrays.stat().st_size,
                "count": 3,
                "arrays": {
                    "apartment_event_01.revealed_background": {
                        "shape": [1, 3],
                        "dtype": "int64",
                        "element_count": 3,
                    },
                    "apartment_event_02.revealed_background": {
                        "shape": [0, 3],
                        "dtype": "int64",
                        "element_count": 0,
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
    return manifest


def _event_metrics(scene: str) -> dict[str, object]:
    observable = {
        "intervention_frame_id": 100,
        "frame_ids": list(range(100, 551, 50)),
        "background_observable": True,
        "recovered": scene == "office",
        "recovery_frames": 100 if scene == "office" else 450,
        "right_censored": scene != "office",
        "censor_frame": 550,
        "overlapping_intervention": False,
        "censor_reason": None if scene == "office" else "administrative_horizon",
    }
    events: dict[str, object] = {f"{scene}_event_01": observable}
    if scene == "apartment":
        events["apartment_event_02"] = {
            "intervention_frame_id": 700,
            "frame_ids": list(range(700, 1151, 50)),
            "background_observable": False,
            "recovered": None,
            "recovery_frames": None,
            "right_censored": False,
            "censor_frame": None,
            "overlapping_intervention": False,
            "censor_reason": "unobservable_revealed_target",
        }
    return {
        "event_count": len(events),
        "background_observable_event_count": 1,
        "unobservable_revealed_target_event_count": len(events) - 1,
        "recovered_event_count": int(scene == "office"),
        "censored_event_count": int(scene != "office"),
        "events": events,
    }


def _summary(
    path: Path,
    *,
    scene: str,
    method: str = "OVIMAP_FROZEN",
    mode: str = "frozen",
    target_manifest: Path | None = None,
) -> Path:
    source = path.parent / f"{scene}-source.txt"
    source.write_text(scene, encoding="utf-8")
    target_manifest = target_manifest or _target_package(path.parent / "targets")
    target = json.loads(target_manifest.read_text(encoding="utf-8"))
    target_arrays = target_manifest.parent / target["target_arrays"]["path"]
    schedule = Path(target["metadata"]["schedule"]["path"])
    aliases = path.parent / "aliases.yaml"
    aliases.write_text("aliases: {}\n", encoding="utf-8")
    label_space = path.parent / f"{scene}_labels.yaml"
    label_space.write_text("label_names: {}\n", encoding="utf-8")
    evaluator = path.parent / "evaluator.py"
    evaluator.write_text("# evaluator fixture\n", encoding="utf-8")
    _write_json(
        path,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_scene_summary",
            "dataset": "TESSE-CD",
            "protocol": "tesse_cd_common_v2",
            "status": "PASS",
            "method": method,
            "mode": mode,
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
            "sources": {
                "fixture": _record(source),
                "target_manifest": _record(target_manifest),
                "target_arrays": _record(target_arrays),
                "schedule": _record(schedule),
                "aliases": _record(aliases),
                "label_space": _record(label_space),
                "evaluator": _record(evaluator),
            },
        },
    )
    return path


def test_macro_finalizer_binds_four_verified_tokens(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    result = finalize_common_v2(
        apartment,
        apartment_repeat,
        office,
        office_repeat,
        method="OVIMAP_FROZEN",
        run_id="20260722-ovimap-frozen-common-v2",
        output=tmp_path / "result.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "VERIFIED"
    assert payload["dataset"] == {"name": "TESSE-CD", "splits": ["macro_test"]}
    assert payload["metrics"] == {
        "background_f5": pytest.approx(0.6),
        "current_miou": pytest.approx(0.7),
        "ghost_rate": pytest.approx(0.3),
        "recovery_frames": pytest.approx(200.0),
    }
    assert {binding["token"] for binding in payload["token_bindings"]} == {
        "T2_OVIMAP_FROZEN_CURRENT_MIOU",
        "T2_OVIMAP_FROZEN_GHOST_RATE",
        "T2_OVIMAP_FROZEN_BG_F5",
        "T2_OVIMAP_FROZEN_RECOVERY_FRAMES",
    }
    assert all(binding["precision"] == 3 for binding in payload["token_bindings"])
    assert payload["scene_sources"]["apartment"]["aliases"]["sha256"]
    assert payload["scene_sources"]["office"]["evaluator"]["byte_count"] > 0
    assert payload["scene_details"]["apartment"]["censored_event_count"] == 1


def test_rejects_nonidentical_scene_repeat(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes() + b" ")
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="byte-identical"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


@pytest.mark.parametrize(
    ("method", "display_label", "mode", "eligible"),
    [
        ("DUALMAP", "DualMap", "native", True),
        ("PANOPTIC_SHARED", "Panoptic Mapping + shared masks", "composed", True),
        ("KHRONOS_OPEN", "Khronos (open-set)", "online", True),
        ("KHRONOS_ORACLE", "Khronos (GT semantics)", "oracle", False),
        ("OVIV2", "OVIV2", "online", True),
    ],
)
def test_causal_methods_finalize_with_declared_mode_and_ranking(
    tmp_path: Path,
    method: str,
    display_label: str,
    mode: str,
    eligible: bool,
) -> None:
    apartment = _summary(
        tmp_path / "apartment.json",
        scene="apartment",
        method=method,
        mode="causal_checkpoints",
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(
        tmp_path / "office.json",
        scene="office",
        method=method,
        mode="causal_checkpoints",
    )
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    result = finalize_common_v2(
        apartment,
        apartment_repeat,
        office,
        office_repeat,
        method=method,
        run_id=f"test-{method.lower()}",
        output=tmp_path / "result.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["method"] == {
        "key": method,
        "display_label": display_label,
        "mode": mode,
        "eligible_for_ranking": eligible,
    }
    assert {binding["token"] for binding in payload["token_bindings"]} == {
        f"T2_{method}_CURRENT_MIOU",
        f"T2_{method}_GHOST_RATE",
        f"T2_{method}_BG_F5",
        f"T2_{method}_RECOVERY_FRAMES",
    }
    if method == "OVIV2":
        assert finalizer_module.METHODS["OVIV2"]["summary_mode"] == (
            "causal_checkpoints"
        )


def test_common_finalizer_rejects_legacy_oviovo_method_key(tmp_path: Path) -> None:
    apartment = _summary(
        tmp_path / "apartment.json",
        scene="apartment",
        method="OVIOVO",
        mode="causal_checkpoints",
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(
        tmp_path / "office.json",
        scene="office",
        method="OVIOVO",
        mode="causal_checkpoints",
    )
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="unsupported .*method"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIOVO",
            run_id="legacy-key",
            output=tmp_path / "result.json",
        )


def test_rejects_scene_or_method_mismatch(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="office")
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="scene summary identity"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="CONCEPTGRAPHS_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_mutated_hashed_summary_source(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())
    source = tmp_path / "apartment-source.txt"
    source.write_text("mutated", encoding="utf-8")

    with pytest.raises(ValueError, match="source hash"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_same_file_as_independent_repeat(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    office = _summary(tmp_path / "office.json", scene="office")

    with pytest.raises(ValueError, match="independent files"):
        finalize_common_v2(
            apartment,
            apartment,
            office,
            office,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_different_target_packages_between_scenes(tmp_path: Path) -> None:
    apartment = _summary(
        tmp_path / "apartment.json",
        scene="apartment",
        target_manifest=_target_package(tmp_path / "targets-a"),
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(
        tmp_path / "office.json",
        scene="office",
        target_manifest=_target_package(tmp_path / "targets-b"),
    )
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="same target package"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_target_package_that_used_prediction_inputs(tmp_path: Path) -> None:
    target = _target_package(tmp_path / "targets")
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["prediction_inputs_used"] = True
    _write_json(target, payload)
    apartment = _summary(
        tmp_path / "apartment.json", scene="apartment", target_manifest=target
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office", target_manifest=target)
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="prediction-independent"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_target_observability_without_explicit_independence(tmp_path: Path) -> None:
    target = _target_package(tmp_path / "targets")
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["metadata"].pop("prediction_inputs_used")
    _write_json(target, payload)
    apartment = _summary(
        tmp_path / "apartment.json", scene="apartment", target_manifest=target
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office", target_manifest=target)
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="prediction-independent"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_non_null_unobservable_recovery(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    payload = json.loads(apartment.read_text(encoding="utf-8"))
    payload["metrics"]["events"]["apartment_event_02"]["recovered"] = False
    _write_json(apartment, payload)
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())

    with pytest.raises(ValueError, match="unobservable.*null"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_rejects_mutated_target_package_internal_source(tmp_path: Path) -> None:
    target = _target_package(tmp_path / "targets")
    target_payload = json.loads(target.read_text(encoding="utf-8"))
    ground_truth = Path(target_payload["sources"][-1]["path"])
    apartment = _summary(
        tmp_path / "apartment.json", scene="apartment", target_manifest=target
    )
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office", target_manifest=target)
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())
    ground_truth.write_bytes(b"mutated after target generation")

    with pytest.raises(ValueError, match="target source hash mismatch"):
        finalize_common_v2(
            apartment,
            apartment_repeat,
            office,
            office_repeat,
            method="OVIMAP_FROZEN",
            run_id="test",
            output=tmp_path / "result.json",
        )


def test_finalized_four_tokens_are_importer_compatible(tmp_path: Path) -> None:
    apartment = _summary(tmp_path / "apartment.json", scene="apartment")
    apartment_repeat = tmp_path / "apartment-repeat.json"
    apartment_repeat.write_bytes(apartment.read_bytes())
    office = _summary(tmp_path / "office.json", scene="office")
    office_repeat = tmp_path / "office-repeat.json"
    office_repeat.write_bytes(office.read_bytes())
    result = finalize_common_v2(
        apartment,
        apartment_repeat,
        office,
        office_repeat,
        method="OVIMAP_FROZEN",
        run_id="import-test",
        output=tmp_path / "result.json",
    )

    registry = tmp_path / "benchmark_tokens.tsv"
    fields = [
        "token",
        "table",
        "method",
        "dataset",
        "split",
        "metric",
        "direction",
        "precision",
        "source_json",
        "json_pointer",
        "status",
        "note",
    ]
    suffixes = ("CURRENT_MIOU", "GHOST_RATE", "BG_F5", "RECOVERY_FRAMES")
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for suffix in suffixes:
            writer.writerow(
                {
                    "token": f"T2_OVIMAP_FROZEN_{suffix}",
                    "table": "T2",
                    "method": "OVIMAP_FROZEN",
                    "dataset": "TESSE-CD",
                    "split": "macro_test",
                    "metric": suffix,
                    "direction": "lower" if suffix in {"GHOST_RATE", "RECOVERY_FRAMES"} else "higher",
                    "precision": "3",
                    "status": "UNFILLED",
                    "note": "Pending benchmark run.",
                }
            )
    markdown = tmp_path / "table.md"
    latex = tmp_path / "table.tex"
    markdown.write_text(
        " ".join(f"{{{{T2_OVIMAP_FROZEN_{suffix}}}}}" for suffix in suffixes)
        + "\n",
        encoding="utf-8",
    )
    latex.write_text(markdown.read_text(encoding="utf-8"), encoding="utf-8")

    outputs = import_results(
        registry,
        [result],
        markdown,
        latex,
        tmp_path / "rendered.md",
        tmp_path / "rendered.tex",
    )

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["status"] for row in rows} == {"VERIFIED"}
    assert outputs["markdown"].read_text(encoding="utf-8") == (
        "0.700 0.300 0.600 200.000\n"
    )
