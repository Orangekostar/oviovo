"""Supervision uses whole GT objects and never modifies locked predictions."""

import numpy as np

from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.module_validation.study_targets import semantic_events


def test_semantic_events_use_strict_class_independent_whole_object_matching():
    owners = np.repeat([7, 0, 8, 9, 0], [120, 30, 100, 50, 60])
    labels = np.where(owners > 0, 42, 0)
    native = PredictionPayload("N0", "N0", "sceneA", GeometryIdentity("a" * 64, "b" * 64,
        "c" * 64, "projection", len(owners)), owners, labels, ((7, 1.), (8, 1.), (9, 0.)), {})
    native.lock()
    before = native.record_key
    combined = np.repeat([5001, 42002, 5003, 0], [150, 100, 100, 10])
    suggestions = {owner: {"label_id": 5, "technical_fallback": False} for owner in (7, 8, 9)}
    rows = semantic_events(native, suggestions, np.arange(len(owners)), np.ones(len(owners), bool),
                           combined, (5, 42))
    assert [row["event"] for row in rows] == ["GAIN", "HARM", None]
    assert rows[0]["geometry_iou"] == 0.8
    assert rows[2]["correspondence"] == "unmatched"  # Exactly .5 is not accepted.
    assert native.record_key == before
    suggestions[7]["technical_fallback"] = True
    assert semantic_events(native, suggestions, np.arange(len(owners)), np.ones(len(owners), bool),
                           combined, (5, 42))[0]["event"] is None
