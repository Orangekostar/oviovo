"""Selection uses official fraction-scale bands and additional standalone cost."""


def test_feasibility_gain_and_bands_use_percentage_points_in_fraction_units():
    from static_ovmap.recovery_wave2.selection import banded_rank, feasible, net_gain

    baseline = {"apall": .1, "ap50": .2, "miou": .3}
    assert feasible({"apall": .0995, "ap50": .199, "miou": .299}, baseline)
    assert not feasible({"apall": .0994, "ap50": .2, "miou": .3}, baseline)
    assert net_gain({"apall": .102, "ap50": .199, "miou": .299}, baseline)
    assert not net_gain({"apall": .1019, "ap50": .2, "miou": .3}, baseline)
    rows = [{"id": "base", "metrics": baseline, "standalone_new_encoder_inputs": 0,
             "changed_blocks": 0, "parameter_distance": 0},
            {"id": "costly", "metrics": dict(baseline, apall=.1004), "standalone_new_encoder_inputs": 6,
             "changed_blocks": 1, "parameter_distance": .1}]
    rank = banded_rank(rows)
    assert rank["banded_preference"] == ["base", "costly"]
    assert rank["strict_metric_ranking"] == ["costly", "base"]


def test_scalar_and_recovery_selection_keeps_none_and_does_not_make_a_grid():
    from static_ovmap.recovery_wave2.selection import select_components

    base = {"apall": .1, "ap50": .2, "miou": .3}
    rows = [{"id": method, "metrics": dict(base, apall=.09 if "W0" in method or "FCEQ" in method else .1),
             "standalone_new_encoder_inputs": 0, "changed_blocks": int(method != "RW_B_D2"),
             "parameter_distance": 0} for method in ["RW_B_D2", "RW_B_FCEQ", "RW_W040", "RW_W045",
                "RW_U1_NATIVE_SINGLE", "RW_UQ_PAID_QUERY", "RW_U2_FC_MATCHED_SINGLE", "RW_U3_FC_CAPTURED"]]
    selection = select_components(rows)
    assert selection["gamma"] == .5 and selection["recovery"] is None
    assert selection["combination"] == {"RW_LIGHT_COMBO": {"gamma": .5, "recovery": None}}
