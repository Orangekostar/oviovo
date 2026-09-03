from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_anchor_counterfactual import (
    COUNTERFACTUAL_METRICS,
    build_anchor_counterfactual_features,
    build_counterfactual_metric_rows,
    derive_counterfactual_variants,
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

