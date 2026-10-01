import pytest

from static_ovmap.recovery_wave2.mechanisms import released_attribution


def trace(events, truth):
    common = dict(distance_index=0, overlap_index=0, overlap_threshold=.5,
                  class_index=1, class_label="chair", scene_key="/annotations/scene/gt.npy")
    return {"parity": {"ap_exact": True, "pr_and_fn_exact": True},
            "events": [{**common, **event} for event in events],
            "states": [{**common, "y_true": truth, "hard_false_negatives": 0}]}


def first(owner, score=.5):
    return dict(event="first_match", gt_id=2001, candidate_file=f"/masks/owner_{owner}.npy",
                confidence=score)


def duplicate(owner, score, previous):
    return dict(event="duplicate", gt_id=2001, candidate_file=f"/masks/owner_{owner}.npy",
                confidence=score, previous_score=previous)


def test_duplicate_and_unvisited_fp_are_separate_score_entries():
    value = trace([first(1, .8), duplicate(2, .3, .8),
                   dict(event="ignore_test", candidate_file="/masks/owner_2.npy", counted_fp=True)], [1, 0, 0])
    result = released_attribution(value, {2})
    row = result["by_overlap"]["0.5"]
    assert row["added_tp_score_entries"] == 0
    assert row["added_fp_score_entries"] == 2
    assert row["total_tp_score_entries"] == 1
    assert row["total_fp_score_entries"] == 2


def test_tie_keeps_both_possible_owners_until_a_strict_winner():
    result = released_attribution(trace([first(1), duplicate(2, .5, .5)], [1, 0]), {2})
    row = result["by_overlap"]["0.5"]
    assert row["ambiguous_added_tp_score_entries"] == 1
    assert row["ambiguous_added_fp_score_entries"] == 1
    result = released_attribution(trace([first(1), duplicate(2, .5, .5), duplicate(3, .9, .5)], [1, 0, 0]), {2, 3})
    assert result["by_overlap"]["0.5"]["added_tp_score_entries"] == 1
    assert result["tp_matches"][0]["possible_owners"] == [3]


def test_unverified_trace_or_inconsistent_terminal_entries_rejected():
    value = trace([first(1)], [1])
    value["parity"]["ap_exact"] = False
    with pytest.raises(ValueError, match="parity"):
        released_attribution(value, set())
    with pytest.raises(ValueError, match="terminal"):
        released_attribution(trace([first(1)], [1, 0]), set())
