"""CAL isolation, bounded source fitting and composition-only nomination."""

import numpy as np
import pytest

CAL = ("scene0056_00", "scene0534_00")


def test_temperature_support_and_cross_scene_isolation():
    from src.static_ovmap.composition_study.temperature import fit_temperature

    examples = [
        {
            "scene_id": CAL[0],
            "owner_id": i,
            "scores": [2.0, 0.0],
            "gt_label": 4 if i % 2 else 7,
        }
        for i in range(6)
    ]
    unsupported = fit_temperature(examples[:4], [CAL[0]], [4, 7])
    assert unsupported["temperature"] == 0.07
    assert unsupported["status"] == "UNCALIBRATED_DEFAULT_T"
    fitted = fit_temperature(examples, [CAL[0]], [4, 7])
    assert fitted["status"] == "FITTED"
    assert 0.01 <= fitted["temperature"] <= 2.0
    assert fitted["nll_after"] < fitted["nll_before"]
    opposite = [
        {"scene_id": CAL[1], "owner_id": i, "scores": [0.0, 100.0], "gt_label": 4}
        for i in range(50)
    ]
    assert fit_temperature(examples + opposite, [CAL[0]], [4, 7]) == fitted
    with pytest.raises(ValueError, match="CAL"):
        fit_temperature(examples, ["scene0445_00"], [4, 7])


def test_temperature_scene_weighting_does_not_follow_object_count():
    from src.static_ovmap.composition_study.temperature import fit_temperature

    a = [
        {
            "scene_id": CAL[0],
            "owner_id": i,
            "scores": [1.0, 0.0],
            "gt_label": 4 if i < 5 else 7,
        }
        for i in range(6)
    ]
    b = [
        {
            "scene_id": CAL[1],
            "owner_id": i,
            "scores": [1.0, 0.0],
            "gt_label": 4 if i < 1 else 7,
        }
        for i in range(6)
    ]
    normal = fit_temperature(a + b, CAL, [4, 7])
    expanded = [
        dict(row, owner_id=100 * repeat + row["owner_id"])
        for repeat in range(5)
        for row in a
    ]
    duplicated = fit_temperature(expanded + b, CAL, [4, 7])
    assert normal["temperature"] == pytest.approx(duplicated["temperature"], abs=1e-7)
    assert np.isfinite(normal["nll_after"])


def _rows():
    methods = [
        "N0",
        "Q_COMBINE",
        "Q_GAIN",
        "S_SIGLIP2_AREA",
        "CP_M1_AGREE_KEEP",
        "CP_M2_EQUAL_RAW",
        "CP_M2_EQUAL_CAL",
        "CP_M3_COMBINE_S2",
        "CP_M4_GAIN_S2",
        "CP_M5_MIX50_NATIVE",
    ]
    return [
        {
            "scene_id": scene,
            "role": "compose_cal",
            "method_id": method,
            "metrics": {"uap": 0.4, "miou": 0.5},
            "changed_owners": 1,
            "logical_requests": 200,
        }
        for scene in CAL
        for method in methods
    ]


def test_m6_requires_both_components_and_only_cal_metrics():
    from src.static_ovmap.composition_study.selection import m6_gate

    rows = _rows()
    assert m6_gate(rows)["enabled"] is False
    for row in rows:
        if row["method_id"] == "CP_M3_COMBINE_S2":
            row["metrics"]["uap"] = 0.41
    assert m6_gate(rows)["enabled"] is False
    for row in rows:
        if row["method_id"] == "CP_M5_MIX50_NATIVE":
            row["metrics"]["miou"] = 0.51
    assert m6_gate(rows)["enabled"] is True
    rows[0]["role"] = "regression_only"
    with pytest.raises(ValueError, match="CAL"):
        m6_gate(rows)


def test_nomination_excludes_no_intervention_and_does_not_require_dominance():
    from src.static_ovmap.composition_study.selection import nominate

    rows = _rows()
    for row in rows:
        if row["method_id"] == "CP_M1_AGREE_KEEP":
            row["metrics"]["uap"] = 0.9
            row["changed_owners"] = 0
        elif row["method_id"] == "CP_M2_EQUAL_RAW":
            row["metrics"] = {"uap": 0.45, "miou": 0.4}
    result = nominate(rows, m6_enabled=False)
    assert result["nominee"] == "CP_M2_EQUAL_RAW"
    assert result["best_single_CAL"] == "N0"
    assert result["confirmation_required"] is True
    assert result["calibration_recommendation"]["versus_best_single"] == "TRADEOFF"
    with pytest.raises(ValueError, match="complete"):
        nominate(rows[1:], m6_enabled=False)
