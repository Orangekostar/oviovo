import pytest


def test_cal_official_order_tolerance_cost_registry_and_reject_replica():
    from src.static_ovmap.a7_evidence_upgrade.selection import rank_candidates

    def row(name, apall, miou, cost):
        return {"method": name, "split": "cal", "rank_mode": "OFFICIAL_CURRENT_CLASS", "status": "COMPLETE",
                "metrics": {"apall": apall, "miou": miou, "ap50": .2}, "unique_required_operations": cost}

    rows = [row("a", .1, .3, 100), row("b", .1 + 5e-11, .4, 200), row("c", .1, .4, 150), row("d", .1, .4, 150)]
    assert [r["method"] for r in rank_candidates(rows, ["d", "c", "b", "a"])] == ["d", "c", "b", "a"]
    assert rank_candidates([row("a", .11, .1, 1000), row("b", .1, .9, 0)], ["b", "a"])[0]["method"] == "a"
    with pytest.raises(ValueError, match="CAL"):
        rank_candidates([{**rows[0], "split": "replica"}], ["a"])
