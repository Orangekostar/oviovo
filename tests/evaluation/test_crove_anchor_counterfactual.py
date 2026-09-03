from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_anchor_counterfactual import (
    COUNTERFACTUAL_METRICS,
    build_anchor_counterfactual_features,
    build_counterfactual_metric_rows,
    derive_counterfactual_variants,
)
from scripts.evaluation.run_crove_anchor_counterfactuals import (
    create_counterfactual_plan,
    load_counterfactual_plan_variant,
    load_variant_gate_metrics,
    publish_counterfactual_collection,
)


def _transitions() -> list[dict[str, object]]:
    return [
        {
            "frame_index": 300 + index,
            "timestamp_ns": 1_000 + index,
            "entity_id": f"anchor:ovimap:{index:02d}",
            "before": "active",
            "after": "dormant",
            "evidence": "visible_absent",
            "readout_valid": False,
        }
        for index in range(10)
    ]


def _attribution() -> dict[str, object]:
    return {
        "status": "PASS",
        "scene": "apartment",
        "frames": [
            {
                "official_cross_check": "PASS",
                "entity_attribution": [
                    {
                        "authority": "ovimap_anchor_unbound",
                        "entity_id": f"ovimap:{index:02d}",
                        "predicted_points": 100,
                        "ghost_matches": 100 - index,
                        "ghost_rate": (100 - index) / 100,
                    }
                    for index in range(10)
                ],
            }
        ],
    }


def test_counterfactual_variants_are_complete_deterministic_and_diagnostic() -> None:
    variants = derive_counterfactual_variants(_transitions(), _attribution())

    assert len(variants) == 24
    assert [item.variant_id for item in variants[:2]] == ["CF0", "CF1"]
    assert [item.variant_id for item in variants[2:12]] == [
        f"CF2_{index:02d}" for index in range(10)
    ]
    assert [item.variant_id for item in variants[12:22]] == [
        f"CF3_{index:02d}" for index in range(10)
    ]
    assert [item.variant_id for item in variants[-2:]] == ["G1", "G2"]
    all_ids = tuple(f"ovimap:{index:02d}" for index in range(10))
    assert variants[0].selected_anchor_ids == ()
    assert variants[1].selected_anchor_ids == all_ids
    assert variants[2].selected_anchor_ids == all_ids[1:]
    assert variants[2].focal_anchor_id == "ovimap:00"
    assert variants[12].selected_anchor_ids == ("ovimap:00",)
    assert variants[12].focal_anchor_id == "ovimap:00"
    assert variants[-2].selected_anchor_ids == all_ids[:5]
    assert variants[-1].selected_anchor_ids == all_ids[5:]
    assert all(item.diagnostic_only for item in variants)
    assert variants == derive_counterfactual_variants(
        list(reversed(_transitions())), _attribution()
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda values: values.append(dict(values[0])),
            "unique active-to-dormant",
        ),
        (
            lambda values: values[0].update(after="active"),
            "active-to-dormant",
        ),
        (
            lambda values: values[0].update(entity_id="ovimap:00"),
            "anchor entity",
        ),
    ),
)
def test_counterfactual_variants_reject_invalid_transitions(
    mutate, match: str
) -> None:
    transitions = _transitions()
    mutate(transitions)

    with pytest.raises(ValueError, match=match):
        derive_counterfactual_variants(transitions, _attribution())


def test_counterfactual_variants_reject_untrusted_attribution() -> None:
    attribution = _attribution()
    attribution["status"] = "FAIL"

    with pytest.raises(ValueError, match="attribution"):
        derive_counterfactual_variants(_transitions(), attribution)


def _entity(entity_id: str, *, label: str | None = "Chair") -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(
            ((0.01, 0.01, 0.01), (0.08, 0.02, 0.03), (0.11, 0.06, 0.09)),
            dtype=np.float32,
        ),
        semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        semantic_label=label,
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=262.0,
        metadata={"authority": "ovimap_anchor"},
    )


def _anchor() -> MapSnapshot:
    return MapSnapshot(
        method="OVI-MAP causal static anchor",
        scene_id="apartment",
        timestamp=262.0,
        entities=[_entity("ovimap:00"), _entity("ovimap:01", label=None)],
        background_xyz=None,
        scope="current",
    )


def _diagnostics() -> dict[str, object]:
    return {
        "manifest_id": "crove_ovimap_unbound_visibility_diagnostics_v1",
        "entities": [
            {
                "entity_id": "ovimap:00",
                "sampled_voxel_count": 3,
                "evidence_counts": {
                    "present": 4,
                    "visible_absent": 9,
                    "occluded": 2,
                    "depth_unknown": 5,
                },
                "first_absence_frame": 290,
                "transition": {
                    "frame_index": 300,
                    "absence_observation_count": 6,
                    "distinct_absence_viewpoint_count": 3,
                },
            },
            {
                "entity_id": "ovimap:01",
                "sampled_voxel_count": 2,
                "evidence_counts": {
                    "present": 0,
                    "visible_absent": 6,
                    "occluded": 1,
                    "depth_unknown": 13,
                },
                "first_absence_frame": 291,
                "transition": {
                    "frame_index": 301,
                    "absence_observation_count": 6,
                    "distinct_absence_viewpoint_count": 3,
                },
            },
        ],
    }


def test_anchor_counterfactual_features_bind_geometry_and_transition_support() -> None:
    features = build_anchor_counterfactual_features(
        _anchor(), _diagnostics(), ("ovimap:00", "ovimap:01"), voxel_size_m=0.05
    )

    assert [item.anchor_entity_id for item in features] == [
        "ovimap:00",
        "ovimap:01",
    ]
    first = asdict(features[0])
    assert first == {
        "anchor_entity_id": "ovimap:00",
        "transition_frame": 300,
        "semantic_label": "Chair",
        "anchor_point_count": 3,
        "anchor_voxel_count": 3,
        "sampled_voxel_count": 3,
        "present_evidence_count": 4,
        "absent_evidence_count": 9,
        "occluded_evidence_count": 2,
        "unknown_evidence_count": 5,
        "absence_support_at_transition": 6,
        "distinct_viewpoint_count_at_transition": 3,
        "first_absence_frame": 290,
        "suppression_latency_frames": 10,
        "bbox_extent_x_m": pytest.approx(0.10),
        "bbox_extent_y_m": pytest.approx(0.05),
        "bbox_extent_z_m": pytest.approx(0.08),
    }
    assert features[1].semantic_label is None


def test_anchor_counterfactual_features_reject_missing_transition_support() -> None:
    diagnostics = _diagnostics()
    diagnostics["entities"][0].pop("transition")

    with pytest.raises(ValueError, match="transition"):
        build_anchor_counterfactual_features(
            _anchor(), diagnostics, ("ovimap:00", "ovimap:01"), voxel_size_m=0.05
        )


def _metrics(offset: float = 0.0) -> dict[str, float]:
    return {
        "object_f1": 0.30 + offset,
        "dynamic_f1": 0.10 + offset,
        "change_f1": 0.20 + offset,
        "current_miou": 0.40 + offset,
        "ghost_rate": 0.50 - offset,
    }


def test_metric_rows_report_deltas_against_cf0_and_cf1() -> None:
    variants = derive_counterfactual_variants(_transitions(), _attribution())[:3]
    rows = build_counterfactual_metric_rows(
        variants,
        {
            "CF0": _metrics(0.0),
            "CF1": _metrics(0.01),
            "CF2_00": _metrics(0.02),
        },
    )

    assert COUNTERFACTUAL_METRICS == (
        "object_f1",
        "dynamic_f1",
        "change_f1",
        "current_miou",
        "ghost_rate",
    )
    row = rows[-1]
    assert row["metric_status"] == "COMPLETE"
    for name in COUNTERFACTUAL_METRICS:
        assert row[f"delta_{name}_vs_cf0"] == pytest.approx(
            row[name] - rows[0][name]
        )
        assert row[f"delta_{name}_vs_cf1"] == pytest.approx(
            row[name] - rows[1][name]
        )


def test_metric_rows_preserve_unavailable_metrics_without_estimation() -> None:
    variants = derive_counterfactual_variants(_transitions(), _attribution())[:3]
    partial = _metrics(0.02)
    partial["object_f1"] = None

    rows = build_counterfactual_metric_rows(
        variants,
        {"CF0": _metrics(), "CF1": _metrics(0.01), "CF2_00": partial},
    )

    assert rows[-1]["metric_status"] == "PARTIAL"
    assert rows[-1]["object_f1"] is None
    assert rows[-1]["delta_object_f1_vs_cf0"] is None
    assert rows[-1]["delta_object_f1_vs_cf1"] is None


@pytest.mark.parametrize("bad", (True, -0.1, 1.1, float("nan"), "0.5"))
def test_metric_rows_reject_invalid_available_metric(bad: object) -> None:
    variants = derive_counterfactual_variants(_transitions(), _attribution())[:2]
    metrics = {"CF0": _metrics(), "CF1": _metrics(0.01)}
    metrics["CF1"]["object_f1"] = bad

    with pytest.raises((TypeError, ValueError), match="object_f1"):
        build_counterfactual_metric_rows(variants, metrics)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_counterfactual_plan_is_canonical_hash_bound_and_reloadable(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "runtime_diagnostics.json"
    attribution = tmp_path / "attribution.json"
    _write_json(
        diagnostics,
        {
            "manifest_id": "crove_ovimap_unbound_visibility_diagnostics_v1",
            "transition_count": 10,
            "transitions": _transitions(),
        },
    )
    _write_json(attribution, _attribution())

    plan = create_counterfactual_plan(
        p5_runtime_diagnostics=diagnostics,
        p2_attribution=attribution,
        output=tmp_path / "counterfactual_plan.json",
    )

    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["manifest_id"] == "crove_anchor_counterfactual_plan_v1"
    assert payload["status"] == "PASS"
    assert payload["diagnostic_only"] is True
    assert len(payload["variants"]) == 24
    for name, path in (
        ("p5_runtime_diagnostics", diagnostics),
        ("p2_attribution", attribution),
    ):
        assert payload["sources"][name] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
    assert load_counterfactual_plan_variant(plan, "CF0") == (
        derive_counterfactual_variants(_transitions(), _attribution())[0]
    )
    with pytest.raises(ValueError, match="variant"):
        load_counterfactual_plan_variant(plan, "unknown")
    with pytest.raises(FileExistsError):
        create_counterfactual_plan(
            p5_runtime_diagnostics=diagnostics,
            p2_attribution=attribution,
            output=plan,
        )


def test_counterfactual_cli_help_lists_three_stage_workflow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/run_crove_anchor_counterfactuals.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "plan" in result.stdout
    assert "compose" in result.stdout
    assert "collect" in result.stdout
    compose_help = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/run_crove_anchor_counterfactuals.py",
            "compose",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compose_help.returncode == 0
    assert "--visibility-diagnostics-cache" in compose_help.stdout


def test_counterfactual_collection_publishes_bound_csv_artifacts(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "runtime_diagnostics.json"
    attribution = tmp_path / "attribution.json"
    _write_json(
        diagnostics,
        {
            "manifest_id": "crove_ovimap_unbound_visibility_diagnostics_v1",
            "transition_count": 10,
            "transitions": _transitions(),
        },
    )
    _write_json(attribution, _attribution())
    plan = create_counterfactual_plan(
        p5_runtime_diagnostics=diagnostics,
        p2_attribution=attribution,
        output=tmp_path / "counterfactual_plan.json",
    )
    variants = derive_counterfactual_variants(_transitions(), _attribution())
    metrics = {
        variant.variant_id: _metrics(index / 1_000)
        for index, variant in enumerate(variants)
    }
    receipts = {}
    for variant in variants:
        receipt = tmp_path / "receipts" / f"{variant.variant_id}.json"
        receipt.parent.mkdir(exist_ok=True)
        _write_json(receipt, {"variant_id": variant.variant_id, "status": "PASS"})
        receipts[variant.variant_id] = receipt
    features = build_anchor_counterfactual_features(
        _anchor(), _diagnostics(), ("ovimap:00", "ovimap:01"), voxel_size_m=0.05
    )

    manifest_path = publish_counterfactual_collection(
        plan=plan,
        metrics_by_variant=metrics,
        features=features,
        metric_receipts=receipts,
        output=tmp_path / "collected",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_id"] == "crove_anchor_counterfactual_v1"
    assert manifest["status"] == "COMPLETE"
    assert manifest["variant_count"] == 24
    assert manifest["feature_count"] == 2
    assert set(manifest["outputs"]) == {
        "counterfactual_metrics",
        "counterfactual_anchor_features",
    }
    for record in manifest["outputs"].values():
        path = manifest_path.parent / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["byte_count"]
    assert set(manifest["metric_receipts"]) == {
        variant.variant_id for variant in variants
    }
    metrics_csv = (manifest_path.parent / "counterfactual_metrics.csv").read_text(
        encoding="utf-8"
    )
    features_csv = (
        manifest_path.parent / "counterfactual_anchor_features.csv"
    ).read_text(encoding="utf-8")
    assert "delta_object_f1_vs_cf0" in metrics_csv.splitlines()[0]
    assert len(metrics_csv.splitlines()) == 25
    assert "suppression_latency_frames" in features_csv.splitlines()[0]
    assert len(features_csv.splitlines()) == 3
    with pytest.raises(ValueError, match="metric receipts"):
        publish_counterfactual_collection(
            plan=plan,
            metrics_by_variant=metrics,
            features=features,
            metric_receipts={"CF0": receipts["CF0"]},
            output=tmp_path / "rejected-collection",
        )
    assert not (tmp_path / "rejected-collection").exists()


def test_variant_gate_metrics_require_bound_matching_composition(tmp_path: Path) -> None:
    variant = derive_counterfactual_variants(_transitions(), _attribution())[0]
    composition = tmp_path / "composition" / "run_manifest.json"
    composition.parent.mkdir()
    _write_json(
        composition,
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_composition_v1",
            "status": "PASS",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "readout_contract": {
                "diagnostic_only": True,
                "moved_geometry_mode": "temporal_compact",
                "promotion_eligible": False,
                "readout_role": "counterfactual_diagnostic",
                "unbound_anchor_mode": "causal_visibility_filtered",
            },
            "counterfactual_variant": variant.to_json_record(),
        },
    )
    gate = tmp_path / "gate" / "gate_decision.json"
    gate.parent.mkdir()
    _write_json(
        gate,
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_apartment_gate_v1",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "decision": {
                "metrics": {
                    "object_f1": 0.4,
                    "dynamic_f1": 0.1,
                    "change_f1": 0.2,
                    "current_miou": 0.3,
                    "ghost_rate": 0.5,
                    "processed_frames": 1745,
                    "official_state_count": 43,
                }
            },
            "sources": {"composition_manifest": _file_record(composition)},
        },
    )

    assert load_variant_gate_metrics(
        gate=gate, composition=composition, variant=variant
    ) == {
        "object_f1": 0.4,
        "dynamic_f1": 0.1,
        "change_f1": 0.2,
        "current_miou": 0.3,
        "ghost_rate": 0.5,
    }
    composition.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="composition binding"):
        load_variant_gate_metrics(gate=gate, composition=composition, variant=variant)
