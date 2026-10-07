"""Focused checks on production evidence/decision adapters, not package kernels."""

import importlib
from pathlib import Path

import numpy as np
import pytest


def module(name):
    path = Path(__file__).parents[2] / "src/static_ovmap/evidence_exploration" / (name + ".py")
    assert path.is_file(), f"production {name} adapter is missing"
    return importlib.import_module("static_ovmap.evidence_exploration." + name)


def baseline(scores=(.51, .49)):
    return {"available": True, "label": 7, "scores": list(scores), "aggregate_vector_sha256": "original-vector"}


def test_trigger_off_preserves_baseline_instead_of_reclassifying():
    v = module("verifier")
    b = baseline()
    out = v.decide(b, [7, 9], triggered=False, purity=.7, target_scores=[0, 1], controls=[[0, 1]], decision="contrast_margin")
    assert out["raw_cosines"] == b["scores"] and out["decision_scores"] == b["scores"]
    assert out["proposed_class"] == b["label"] and out["accepted"]
    assert out["baseline_vector_sha256"] == b["aggregate_vector_sha256"]


def test_empty_control_is_margin_only_and_availability_is_not_acceptance():
    v = module("verifier")
    out = v.decide(baseline((.505, .5)), [7, 9], triggered=True, purity=.1, decision="contrast_margin")
    assert out["raw_cosines"] == out["decision_scores"] == [.505, .5]
    assert out["feature_available"] and not out["accepted"]
    assert out["defer_reason"] == "LOW_MARGIN"


def test_same_class_neighbor_can_harm_and_ties_use_original_order():
    v = module("verifier")
    harmed = v.decide(baseline(), [7, 9], triggered=True, purity=.1, controls=[[1., -1.]], decision="contrast_margin")
    assert harmed["proposed_class"] == 9 and harmed["accepted"]
    tied = v.decide(baseline((.5, .5)), [7, 9], triggered=True, purity=.1, decision="margin")
    assert tied["proposed_class"] == 7 and not tied["accepted"]


def test_target_gate_checks_all_five_metrics_and_strict_D2_improvement():
    v = module("verifier")
    b1 = {c: dict.fromkeys(v.METRICS, .2) for c in ("replica8", "scannet_cf18")}
    b0 = {c: dict(values) for c, values in b1.items()}
    candidate = {c: dict(values) for c, values in b1.items()}
    assert not v.target_flags(candidate, b1, b0)["target_met"]
    candidate["scannet_cf18"]["apall"] += .001
    assert v.target_flags(candidate, b1, b0)["material_target_met"]
    candidate["replica8"]["macc"] -= 1e-6
    assert not v.target_flags(candidate, b1, b0)["target_met"]


def test_control_selection_uses_distinct_owner_area_and_numeric_ties():
    e = module("evidence")
    target = np.zeros((40, 60), bool); target[10:20, 20:30] = True
    owners = np.zeros(target.shape, np.int64)
    owners[2:17, 12:20] = 90
    owners[2:17, 30:38] = 4
    owners[target] = 7
    controls = e.competing_regions(target, owners, [20, 10, 30, 20])
    assert [x["owner"] for x in controls] == [4, 90]
    assert controls[0]["pixels"] == 120
    assert not (controls[0]["mask"] & target).any()


def test_zero_mass_trigger_is_invariant_failure():
    e = module("evidence")
    with pytest.raises(ValueError, match="mass"):
        e.occupancy_trigger(np.zeros((2, 2)), 0)
    assert e.occupancy_trigger(np.array([[.2, .2]]), 4)[0]
    assert not e.occupancy_trigger(np.ones((2, 2)), 4)[0]


def test_geometry_unknown_depth_is_neutral_and_window_zeros_stay_zero():
    import torch
    a = module("anyup_adapter")
    factor = a.geometry_factor(torch.tensor([1., 0.]), torch.tensor([0., 1.]),
                               torch.tensor([0., 2.]), torch.tensor([0., 0.]), use_depth=True)
    assert torch.equal(factor, torch.ones(2, 2))
    attention = torch.tensor([[1., 0.], [.2, .8]])
    actual = a.reweight_attention(attention, torch.tensor([[.1, .4], [.1, .4]]))
    assert actual[0, 1] == 0
    torch.testing.assert_close(actual.sum(1), torch.ones(2))
    torch.testing.assert_close(a.reweight_attention(attention, torch.ones_like(attention)), attention)
    with pytest.raises(ValueError, match="all-masked"):
        a.reweight_attention(torch.zeros(1, 2), torch.ones(1, 2))


def test_paired_score_diagnosis_preserves_released_duplicate_entry_multiplicity():
    d = module("diagnostic_details")
    row = {"kind": "FP_DUPLICATE", "owner": 2, "gt_id": 1001,
           "class_label": "chair", "distance_index": 0, "ambiguous_tie": False}
    result = d.compare_entries([row], [row, row])
    assert result["removed_definite_FP_score_entries50"] == 1
    assert len(result["removed_score_entries50"]) == 1
    assert result["lost_unique_GT_matches50"] == 0
