from scripts.evaluation.summarize_crove_multimethod_readouts import selection_annotations
from scripts.evaluation.summarize_crove_multimethod_readouts import case_coverage
from scripts.evaluation.summarize_crove_multimethod_readouts import method_provenance
from scripts.evaluation.summarize_crove_multimethod_readouts import selection_evidence_role
import pytest


def test_late_score_cannot_be_mistaken_for_frozen_evidence():
    frozen = {"old.json": "hash"}
    assert selection_evidence_role("old.json", "hash", frozen) == "FROZEN_DEV_EVIDENCE"
    assert selection_evidence_role("new.json", "new", frozen) == "POST_FREEZE_SUPPLEMENT"
    with pytest.raises(ValueError, match="frozen"):
        selection_evidence_role("old.json", "changed", frozen)


def test_adapter_pooling_control_and_learned_head_share_checkpoint_source():
    model = {"checkpoint_source": "official", "checkpoint_sha256": "hash"}
    a = method_provenance("ADAPTER_CLIP_MEAN", model)
    b = method_provenance("ADAPTER_LEARNED_GRAPH_GEOM", model)
    assert a["checkpoint"] == b["checkpoint"] == "official"
    assert a["checkpoint_sha256"] == b["checkpoint_sha256"] == "hash"


def test_owner_consensus_and_resem_have_distinct_semantic_sources():
    owner = method_provenance("INST_CONSENSUS_OWNER", {})
    resem = method_provenance("INST_CONSENSUS_RESEM", {})
    assert "unchanged native semantics" in owner["model_and_source"]
    assert "SigLIP" in resem["model_and_source"]


def test_unselected_confirmation_is_not_reported_as_missing():
    rows = [{"exact_implementation": "M", "case": "apartment", "state_variant": "B3"}]
    result = case_coverage("M", rows, [], {})
    assert result["cases_run"] == ["apartment/B3"]
    assert result["cases_missing"] == ["apartment/H2"]
    assert result["static_confirmation_status"] == "NOT_SELECTED"


def test_selected_confirmation_requires_actual_matching_score():
    rows = [{"exact_implementation": "M", "case": "room0", "state_variant": None}]
    assert case_coverage("M", rows, ["M"], {})["cases_missing"] == ["room1"]
    result = case_coverage("M", rows, ["M"], {"M": "FULL_MAP_EVALUATED"})
    assert result["cases_run"] == ["room0", "room1"]
    assert result["cases_missing"] == []
    assert result["static_confirmation_status"] == "COMPLETE"


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
