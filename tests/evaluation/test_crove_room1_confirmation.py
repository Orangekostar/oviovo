import pytest

from scripts.evaluation.run_crove_room1_confirmation import (
    validate_scored_invariants,
    validate_selection,
)


def valid_selection():
    return {
        "status": "DEV_SELECTION_FROZEN",
        "selection_policy": {"id": "frozen-policy"},
        "static_confirmation": {
            "scene": "room1",
            "methods": ["native", "new"],
            "retuning_allowed": False,
        },
        "task_selections": [
            {
                "family": f,
                "task": "STATIC_INSTANCE" if f == "M3" else "STATIC_SEMANTIC",
                "metric_winner": {
                    "exact_implementation": "native",
                    "direct_control": None,
                },
                "best_new_candidate": {
                    "exact_implementation": "new",
                    "direct_control": "native",
                },
            }
            for f in ("M1", "M2", "M3", "M4")
        ],
    }


def test_confirmation_requires_all_four_families():
    selection = valid_selection()
    selection["task_selections"].pop()
    with pytest.raises(ValueError):
        validate_selection(selection, {"id": "frozen-policy"})


def test_baseline_winner_does_not_hide_missing_new_representative():
    selection = valid_selection()
    selection["static_confirmation"]["methods"] = ["native"]
    with pytest.raises(ValueError):
        validate_selection(selection, {"id": "frozen-policy"})


def test_changed_policy_or_unfrozen_selection_is_rejected():
    selection = valid_selection()
    with pytest.raises(ValueError):
        validate_selection(selection, {"id": "another-policy"})
    selection["status"] = "PARTIAL"
    with pytest.raises(ValueError):
        validate_selection(selection, {"id": "frozen-policy"})


def test_complete_frozen_selection_returns_exact_methods():
    assert validate_selection(valid_selection(), {"id": "frozen-policy"}) == [
        "native",
        "new",
    ]


@pytest.mark.parametrize(
    "method,field",
    [("MV_QUALITY", "ap50"), ("INST_CONSENSUS_OWNER", "miou"), ("MV_QUALITY", "f5")],
)
def test_confirmation_rejects_broken_scored_invariant(method, field):
    metrics = {"miou": 0.3, "f_miou": 0.4, "ap25": 0.6, "ap50": 0.5, "f5": 0.9}
    changed = dict(metrics)
    changed[field] += 0.01
    with pytest.raises(ValueError):
        validate_scored_invariants(
            {"B_SEM_OVI_NATIVE": {"metrics": metrics}, method: {"metrics": changed}}
        )


def test_semantic_gain_and_owner_only_ap_change_remain_allowed():
    baseline = {"miou": 0.3, "f_miou": 0.4, "ap25": 0.6, "ap50": 0.5, "f5": 0.9}
    semantic = dict(baseline, miou=0.35)
    instance = dict(baseline, ap50=0.2)
    checks = validate_scored_invariants(
        {
            "B_SEM_OVI_NATIVE": {"metrics": baseline},
            "MV_QUALITY": {"metrics": semantic},
            "INST_CONSENSUS_OWNER": {"metrics": instance},
        }
    )
    assert checks["MV_QUALITY"]["ap50"]
    assert checks["INST_CONSENSUS_OWNER"]["miou"]
