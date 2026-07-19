from __future__ import annotations

import copy

import pytest

from scripts.evaluation.finalize_oviv2_reevaluation import validate_metric_transition


def _metrics(ap25: float = 0.2, ap50: float = 0.1) -> dict:
    return {
        "miou": 0.3,
        "macc": 0.4,
        "f_miou": 0.5,
        "f5": 0.9,
        "ap25": ap25,
        "ap50": ap50,
        "semantic": {"miou": 0.3},
        "geometry": {"f5": 0.9},
        "projection": {"matched_vertex_count": 5},
        "instance": {
            "class_agnostic": {"ap25": ap25, "ap50": ap50},
            "semantic_class_constrained": {"ap25": 0.1, "ap50": 0.05},
        },
        "protocol": {
            "headline_instance_protocol": "class_agnostic",
            "semantic_instance_protocol": "diagnostic_only",
        },
    }


def test_transition_allows_only_instance_protocol_changes() -> None:
    old = _metrics(0.1, 0.05)
    new = _metrics(0.4, 0.2)

    transition = validate_metric_transition(old, new, scene="room0")

    assert transition == {
        "old": {"ap25": pytest.approx(0.1), "ap50": pytest.approx(0.05)},
        "new": {"ap25": pytest.approx(0.4), "ap50": pytest.approx(0.2)},
        "delta": {"ap25": pytest.approx(0.3), "ap50": pytest.approx(0.15)},
    }


def test_transition_rejects_noninstance_change_and_headline_mismatch() -> None:
    old = _metrics()
    changed_semantic = copy.deepcopy(_metrics(0.4, 0.2))
    changed_semantic["semantic"]["miou"] = 0.31
    with pytest.raises(ValueError, match="semantic"):
        validate_metric_transition(old, changed_semantic, scene="room0")

    mismatched_headline = _metrics(0.4, 0.2)
    mismatched_headline["ap25"] = 0.8
    with pytest.raises(ValueError, match="class_agnostic"):
        validate_metric_transition(old, mismatched_headline, scene="room0")
