from scripts.evaluation.select_crove_multimethod_readouts import rank_candidates


def row(name, score, secondary=0.0, eligible=True):
    return {
        "exact_implementation": name,
        "score": score,
        "secondary": secondary,
        "passes_original_per_case_gates": eligible,
    }


def test_metric_winner_and_eligible_winner_are_separate():
    values = [row("baseline", 0.4), row("new", 0.5, eligible=False)]
    assert (
        rank_candidates(values, "score", ["secondary"])["exact_implementation"] == "new"
    )
    assert (
        rank_candidates(values, "score", ["secondary"], eligible_only=True)[
            "exact_implementation"
        ]
        == "baseline"
    )


def test_primary_ties_are_measured_from_maximum_not_transitive_pairs():
    values = [row("a", 0.5, 100), row("b", 0.5000009, 10), row("c", 0.5000018, 1)]
    assert (
        rank_candidates(values, "score", ["secondary"])["exact_implementation"] == "b"
    )


def test_tied_baseline_is_not_disfavored_and_missing_runtime_is_not_zero():
    values = [row("z_new", 0.5), row("a_baseline", 0.5)]
    values[0]["runtime_stages"] = {"inference_seconds": 0.0}
    assert (
        rank_candidates(values, "score", ["secondary"])["exact_implementation"]
        == "a_baseline"
    )


def test_secondary_metrics_break_primary_tie_in_declared_order():
    values = [row("a", 0.5, 0.6), row("b", 0.5000001, 0.7)]
    assert (
        rank_candidates(values, "score", ["secondary"])["exact_implementation"] == "b"
    )


def test_no_eligible_configuration_stays_empty():
    assert (
        rank_candidates(
            [row("failed", 0.9, eligible=False)], "score", [], eligible_only=True
        )
        is None
    )
