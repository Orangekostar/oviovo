"""Many-to-one planning and whole-owner union gates are separate decisions."""

from types import SimpleNamespace

import numpy as np


def fragments(*masks):
    return [SimpleNamespace(mask=m, input_group=i + 1, is_thing=True) for i, m in enumerate(masks)]


def test_partial_followers_share_owner_and_pass_joint_reverse_coverage():
    from static_ovmap.recovery_wave2.association import plan_objects

    prior = np.full((50, 20), 55)
    first, second = np.zeros_like(prior, bool), np.zeros_like(prior, bool)
    first[:6], second[6:12] = True, True
    rows = fragments(first, second)
    assert len(plan_objects(rows, np.ones_like(prior), prior, mode="A1")["accepted_pairs"]) == 1
    for mode in ("A2", "A3"):
        plan = plan_objects(rows, np.ones_like(prior), prior, mode=mode)
        assert plan["accepted_pairs"] == [[1, 55], [2, 55]]
        assert all(a["action"] == "ASSIGN_EXISTING" for a in plan["actions"])
        assert plan["duplicate_owner_followers"] == 1


def test_specificity_uses_all_known_intersections_and_returns_native_fallback():
    from static_ovmap.recovery_wave2.association import plan_objects

    prior = np.full((15, 20), 55)
    prior[9:] = 7
    mask = np.ones_like(prior, bool)
    assert plan_objects(fragments(mask), np.ones_like(prior), prior, mode="A2")["accepted_pairs"] == [[1, 55]]
    plan = plan_objects(fragments(mask), np.ones_like(prior), prior, mode="A3")
    assert not plan["accepted_pairs"]
    assert plan["actions"] == [{"local_group": 1, "action": "USE_NATIVE", "existing_owner": None}]
    assert plan["specificity"]["1"]["dominance"] == .6


def test_exact_tie_uses_physical_prior_support_not_numeric_owner_id():
    from static_ovmap.recovery_wave2.association import plan_objects

    prior = np.full((20, 20), 55)
    prior[:, 10:] = 7
    mask = np.ones_like(prior, bool)
    first = plan_objects(fragments(mask), np.ones_like(prior), prior, mode="A2")
    renamed = np.where(prior == 55, 7, 55)
    second = plan_objects(fragments(mask), np.ones_like(prior), renamed, mode="A2")
    selected = first["actions"][0]["existing_owner"]
    replacement = second["actions"][0]["existing_owner"]
    np.testing.assert_array_equal(prior == selected, renamed == replacement)
