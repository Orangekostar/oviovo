from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_crove_ovimap_static_anchor import (
    GateEvaluationDependencies,
    decide_apartment_gate,
    evaluate_apartment_candidate,
)


def _metrics() -> dict[str, object]:
    return {
        "obj_f1": 0.372762,
        "dyn_f1": 0.01,
        "chg_f1": 0.060854,
        "current_miou": 0.142897,
        "ghost_rate": 0.646882,
        "processed_frames": 1745,
        "official_state_count": 43,
    }


def test_apartment_gate_accepts_exact_static_floors_and_dynamic_gains() -> None:
    decision = decide_apartment_gate(**_metrics())

    assert decision.status == "PASS_APARTMENT"
    assert decision.office_authorized is True
    assert decision.failed_gates == ()
    assert not hasattr(decision, "token_bindings")


@pytest.mark.parametrize(
    ("name", "value", "failed_gate"),
    (
        ("obj_f1", 0.372761, "object_f1_floor"),
        ("dyn_f1", None, "dynamic_f1_finite"),
        ("dyn_f1", math.nan, "dynamic_f1_finite"),
        ("chg_f1", 0.060853, "change_f1_strict_gain"),
        ("current_miou", 0.142896, "current_miou_floor"),
        ("ghost_rate", 0.646883, "ghost_rate_strict_reduction"),
        ("processed_frames", 1744, "complete_frame_coverage"),
        ("official_state_count", 42, "complete_official_states"),
    ),
)
def test_apartment_gate_rejects_each_failed_hard_metric(
    name: str, value: object, failed_gate: str
) -> None:
    metrics = _metrics()
    metrics[name] = value

    decision = decide_apartment_gate(**metrics)

    assert decision.status == "REJECTED_RETAIN_A6"
    assert decision.office_authorized is False
    assert failed_gate in decision.failed_gates


def test_gate_decision_serialization_contains_no_paper_bindings() -> None:
    payload = decide_apartment_gate(**_metrics()).to_json_record()

    assert payload["status"] == "PASS_APARTMENT"
    assert payload["baseline"] == {
        "object_f1": 0.372762,
        "change_f1": 0.060853,
        "current_miou": 0.142897,
        "ghost_rate": 0.646883,
    }
    assert "token_bindings" not in payload


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _record(path: Path, *, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def test_candidate_evaluation_repeats_metrics_and_publishes_bound_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "composition"
    temporal_index = root / "source_index.json"
    _write(temporal_index, {"fixture": True})
    composition = root / "run_manifest.json"
    _write(
        composition,
        {
            "schema_version": 1,
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "integration": "composed",
            "processed_frame_count": 1745,
            "checkpoints": [{"frame_index": index} for index in range(43)],
            "source_index": _record(temporal_index, root=root),
        },
    )
    official = tmp_path / "official"
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        path = official / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
    inputs = []
    for name in ("targets.json", "aliases.yaml", "labels.yaml"):
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        inputs.append(path)

    calls = {"common": 0, "official": 0}

    def common_evaluator(
        temporal: Path,
        targets: Path,
        aliases: Path,
        labels: Path,
        output: Path,
    ) -> Path:
        calls["common"] += 1
        summary = output / "summary.json"
        _write(
            summary,
            {
                "status": "PASS",
                "dataset": "TESSE-CD",
                "scene": "apartment",
                "metrics": {
                    "current_miou": 0.15,
                    "ghost_rate": 0.60,
                    "background_f5": 0.11,
                },
            },
        )
        return summary

    def official_summarizer(results: Path) -> dict[str, object]:
        calls["official"] += 1
        return {
            "state_count": 43,
            "object_f1": 0.40,
            "dynamic_f1": 0.10,
            "change_f1": 0.08,
            "background_f1_at_0_2": 0.66,
        }

    dependencies = GateEvaluationDependencies(
        common_evaluator=common_evaluator,
        official_summarizer=official_summarizer,
    )
    first = evaluate_apartment_candidate(
        composition_manifest=composition,
        official_results_dir=official,
        target_manifest=inputs[0],
        aliases=inputs[1],
        label_space=inputs[2],
        output_root=tmp_path / "gate-first",
        dependencies=dependencies,
    )
    second = evaluate_apartment_candidate(
        composition_manifest=composition,
        official_results_dir=official,
        target_manifest=inputs[0],
        aliases=inputs[1],
        label_space=inputs[2],
        output_root=tmp_path / "gate-second",
        dependencies=dependencies,
    )

    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "PASS_APARTMENT"
    assert payload["decision"]["office_authorized"] is True
    assert "token_bindings" not in payload
    assert calls == {"common": 4, "official": 4}
    assert first.read_bytes() == second.read_bytes()
