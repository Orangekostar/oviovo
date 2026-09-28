import numpy as np
import pytest

from src.static_ovmap.m2_reviewer_study.calibration import fit_shared


def examples(scene, count=6):
    return [{"scene_id": scene, "owner_id": i, "gt_label": 1 + i % 2,
             "scores": [2., 0.] if i % 2 == 0 else [0., 2.]} for i in range(count)]


def test_shared_support_is_required_for_every_source_and_only_cal_can_fit():
    rows = examples("scene0056_00")
    fit = fit_shared({"N0": rows, "Q": rows[:4]}, ["scene0056_00"], [1, 2])
    assert fit["temperature"] == .07
    assert fit["status"] == "UNCALIBRATED_DEFAULT_T"
    with pytest.raises(ValueError):
        fit_shared({"N0": rows}, ["room0"], [1, 2])


def test_training_fold_excludes_held_out_rows_and_fits_same_source_objective():
    rows = examples("scene0056_00")
    bad = [{**r, "scene_id": "scene0534_00", "scores": [np.nan, 0.]} for r in rows]
    fit = fit_shared({"N0": rows + bad, "Q": rows + bad}, ["scene0056_00"], [1, 2])
    assert fit["status"] == "FITTED"
    assert fit["nll_after"] <= fit["nll_before"]
    assert all(x[0] == "scene0056_00" for s in fit["sources"].values() for x in s["example_ids"])
    fallback = fit_shared({"N0": bad}, ["scene0534_00"], [1, 2])
    assert fallback["temperature"] == .07
    assert fallback["reason"] == "NONFINITE_INPUT"
