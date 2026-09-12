from scripts.evaluation.summarize_crove_multimethod_readouts import selection_annotations


def test_unfrozen_summary_does_not_invent_a_winner():
    assert selection_annotations("room0", "M", None) == {
        "selection_status": "NOT_FROZEN", "selection_tasks": [],
        "metric_winner_for": [], "best_new_candidate_for": [],
    }


def test_frozen_selection_applies_only_to_its_dataset_and_candidate_domain():
    selected = {"task_selections": [{
        "family": "M1", "task": "DYNAMIC_SEMANTIC", "dev_cases": ["apartment"],
        "candidate_methods": ["baseline", "M"],
        "metric_winner": {"exact_implementation": "baseline"},
        "best_new_candidate": {"exact_implementation": "M"},
    }]}
    result = selection_annotations("apartment", "M", selected)
    assert result["selection_status"] == "DEV_SELECTION_FROZEN"
    assert result["selection_tasks"] == ["M1/DYNAMIC_SEMANTIC"]
    assert result["metric_winner_for"] == []
    assert result["best_new_candidate_for"] == ["M1/DYNAMIC_SEMANTIC"]
    assert selection_annotations("room0", "M", selected)["selection_tasks"] == []
