"""Direct semantic readouts, paired features, and fixed selector heads."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.static_ovmap.module_validation.semantic_selector import (
    FEATURE_COUNT,
    FeatureStandardizer,
    SemanticFeatureInputs,
    SemanticRequestStats,
    adoption_values,
    apply_adoption_threshold,
    area_readout,
    build_semantic_features,
    event_support_status,
    head_parameter_counts,
    inverse_scene_weights,
    mask_features_for_head,
    select_adoption_threshold,
    select_teacher,
    semantic_event_label,
    train_semantic_head,
    vote_readout,
)


def test_area_and_vote_readouts_use_literal_weight_and_tie_rules() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    text = np.eye(2)
    area = area_readout(features, np.array([3.0, 1.0]), text, valid_ids=(10, 20))
    assert area.label_id == 10
    np.testing.assert_allclose(area.aggregate_feature, [3 / math.sqrt(10), 1 / math.sqrt(10)])

    vote = vote_readout(
        label_ids=(20, 10), request_ranks=(1, 0), valid_ids=(10, 20)
    )
    assert vote.label_id == 10
    tied_rank = vote_readout(
        label_ids=(20, 10), request_ranks=(0, 0), valid_ids=(10, 20)
    )
    assert tied_rank.label_id == 10


def _feature_inputs() -> SemanticFeatureInputs:
    raw = np.zeros((2, 3, 2), dtype=np.float64)
    raw[:, :, 0] = 0.8
    raw[:, :, 1] = 0.2
    foreground = np.array(
        [
            [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]],
            [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
        ]
    )
    background = np.zeros((2, 3, 2), dtype=np.float64)
    background[:, :, 0] = 0.2
    background[:, :, 1] = 0.6
    return SemanticFeatureInputs(
        class_count=2,
        keep_index=0,
        replace_index=1,
        native_aggregate_scores=np.array([0.8, 0.2]),
        native_view_scores=np.array([[0.9, 0.1], [0.2, 0.8]]),
        native_view_features=np.eye(2),
        native_requested_views=2,
        teacher_aggregate_scores=np.array([0.3, 0.7]),
        teacher_view_labels=(1, 1),
        teacher_view_strengths=np.array([[0.3, 0.7], [0.4, 0.6]]),
        teacher_requested_views=2,
        name_mapping_gaps=(0.2, 0.4),
        source_mask_point_count=99,
        request_stats=(
            SemanticRequestStats(9, 18, 0.8, 0.5, True, False, False, True),
            SemanticRequestStats(15, 30, 0.6, 0.7, False, True, False, True),
        ),
        native_raw_scores=raw,
        native_foreground_scores=foreground,
        native_background_scores=background,
        background_usable=(True, True),
    )


def test_hand_computed_32_features_and_availability() -> None:
    result = build_semantic_features(_feature_inputs())
    values, available = result.values, result.available

    expected = np.array(
        [
            0.8,
            0.2,
            -0.6,
            0.6,
            0.0026154667475286222,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.4,
            0.3,
            0.7,
            0.3,
            math.log(100),
            (math.log(10) + math.log(16)) / 2,
            0.5,
            0.6,
            0.6,
            0.5,
            0.5,
            0.0,
            0.15,
            -0.15,
            0.4,
            -0.3,
            math.sqrt(2 / 300),
            math.sqrt(2 / 300),
            0.1,
            1.0,
        ]
    )
    np.testing.assert_allclose(values, expected, atol=1e-8)
    assert available.tolist() == [True] * FEATURE_COUNT


def test_missing_keep_and_one_view_pairs_are_unavailable_not_negative_indexed() -> None:
    source = _feature_inputs()
    result = build_semantic_features(
        SemanticFeatureInputs(
            **{
                **source.__dict__,
                "keep_index": None,
                "native_view_features": np.array([[1.0, 0.0]]),
                "native_view_scores": np.array([[0.2, 0.8]]),
                "native_requested_views": 1,
                "native_raw_scores": source.native_raw_scores[:1],
                "native_foreground_scores": source.native_foreground_scores[:1],
                "native_background_scores": source.native_background_scores[:1],
                "background_usable": (True,),
            }
        )
    )
    assert result.available[0] == 0
    assert result.available[2] == 0
    assert result.available[7] == 0
    assert result.available[24] == 0
    assert result.available[26] == 0
    assert result.values[[0, 2, 7, 24, 26]].tolist() == [0.0] * 5


def test_fit_only_standardization_missing_fill_and_fixed_head_masks() -> None:
    first = build_semantic_features(_feature_inputs())
    missing = first.with_missing((0, 7))
    standardizer = FeatureStandardizer.fit((first, missing))

    transformed = standardizer.transform(missing)
    assert transformed.shape == (64,)
    assert transformed[0] == 0.0
    assert transformed[32] == 0.0
    assert transformed[39] == 0.0
    assert np.max(np.abs(transformed[:32])) <= 8.0

    simple = mask_features_for_head(np.ones(64), "S_SIMPLE")
    active = {0, 1, 2, 3, 8, 9, 12, 19}
    assert {index for index in range(32) if simple[index]} == active
    assert {index - 32 for index in range(32, 64) if simple[index]} == active
    assert head_parameter_counts("S_SIMPLE") == {"full": 2179, "active": 643}
    assert head_parameter_counts("S_NO_CONTEXT") == {"full": 2179, "active": 1667}
    assert head_parameter_counts("S_PAIRED") == {"full": 2179, "active": 2179}


def test_event_support_teacher_and_threshold_selection_are_frozen_and_strict() -> None:
    events = [
        {"event": "GAIN", "scene_id": "a"},
        {"event": "GAIN", "scene_id": "b"},
        {"event": "GAIN", "scene_id": "a"},
        {"event": "GAIN", "scene_id": "b"},
        {"event": "GAIN", "scene_id": "a"},
        {"event": "HARM", "scene_id": "a"},
        {"event": "HARM", "scene_id": "b"},
        {"event": "HARM", "scene_id": "a"},
        {"event": "HARM", "scene_id": "b"},
        {"event": "HARM", "scene_id": "a"},
    ]
    assert event_support_status(events) == "SUPPORTED"
    assert event_support_status(events[:-1]) == "BLOCKED_EVENT_SUPPORT"

    teacher = select_teacher(
        (
            {"method": "S_SIGLIP2_AREA", "mean_uap": 0.4, "mean_miou": 0.5, "median_cost": 2.0},
            {"method": "S_SIGLIP2_VOTE", "mean_uap": 0.4, "mean_miou": 0.5, "median_cost": 1.0},
            {"method": "S_WOW_VOTE", "mean_uap": 0.3, "mean_miou": 0.9, "median_cost": 0.1},
        )
    )
    assert teacher == "S_SIGLIP2_VOTE"

    threshold = select_adoption_threshold(
        (
            {"threshold": 0.1, "mean_uap": 0.5, "mean_miou": 0.4, "harmful": 1, "replacements": 5},
            {"threshold": 0.2, "mean_uap": 0.5, "mean_miou": 0.4, "harmful": 1, "replacements": 4},
            {"threshold": 0.3, "mean_uap": 0.5, "mean_miou": 0.4, "harmful": 1, "replacements": 4},
        )
    )
    assert threshold == 0.3
    assert apply_adoption_threshold(np.array([0.3, 0.30001]), threshold).tolist() == [False, True]
    assert apply_adoption_threshold(np.array([100.0]), "KEEP_ALL").tolist() == [False]


def test_selector_rejects_nonfinite_or_unknown_threshold_rows() -> None:
    with pytest.raises(ValueError):
        select_adoption_threshold(
            ({"threshold": 0.0, "mean_uap": float("nan"), "mean_miou": 0.0, "harmful": 0, "replacements": 0},)
        )
    with pytest.raises(ValueError):
        apply_adoption_threshold(np.array([0.2]), 0.25)


def test_fit_event_labels_and_inverse_scene_weights_are_object_level() -> None:
    assert semantic_event_label(1, 2, 2, technically_available=True) == "GAIN"
    assert semantic_event_label(1, 2, 1, technically_available=True) == "HARM"
    assert semantic_event_label(1, 2, 3, technically_available=True) == "OTHER"
    assert semantic_event_label(1, 1, 1, technically_available=True) is None
    assert semantic_event_label(1, 2, 2, technically_available=False) is None
    np.testing.assert_allclose(
        inverse_scene_weights(("scene-a", "scene-a", "scene-b")),
        [0.5, 0.5, 1.0],
    )


def test_fixed_head_training_checkpoint_and_adoption_values() -> None:
    pytest.importorskip("torch", exc_type=OSError)
    rng = np.random.default_rng(17)
    fit = rng.normal(size=(18, 64)).astype(np.float32)
    cal = rng.normal(size=(6, 64)).astype(np.float32)
    labels = ("GAIN", "HARM", "OTHER") * 6
    cal_labels = ("GAIN", "HARM", "OTHER") * 2
    weights = inverse_scene_weights(("a",) * 9 + ("b",) * 9)

    first = train_semantic_head(
        "S_SIMPLE",
        fit,
        labels,
        weights,
        cal_features=cal,
        cal_labels=cal_labels,
        max_epochs=10,
    )
    second = train_semantic_head(
        "S_SIMPLE",
        fit,
        labels,
        weights,
        cal_features=cal,
        cal_labels=cal_labels,
        max_epochs=10,
    )

    assert first.status == "COMPLETE"
    assert first.checkpoint_epoch in (5, 10)
    assert first.optimizer_state_dict
    assert first.scaler_state_dict == {}
    assert first.parameter_counts == {"full": 2179, "active": 643}
    for name in first.state_dict:
        np.testing.assert_array_equal(
            first.state_dict[name].numpy(), second.state_dict[name].numpy()
        )
    values = adoption_values("S_SIMPLE", first.state_dict, cal)
    assert values.shape == (6,)
    assert np.isfinite(values).all()

    uncalibrated = train_semantic_head(
        "S_PAIRED", fit, labels, weights, max_epochs=2
    )
    assert uncalibrated.status == "UNCALIBRATED"
    assert uncalibrated.checkpoint_epoch == 2
