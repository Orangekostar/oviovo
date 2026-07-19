from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
import math

import numpy as np
import pytest

from src.oviv2.semantic_memory import (
    FeaturePrototype,
    FeaturePrototypeBank,
    InformativeView,
    InformativeViewBank,
    SparseClassPosterior,
)


def test_later_consistent_evidence_corrects_early_label() -> None:
    empty = SparseClassPosterior.empty()
    posterior = empty.update_label(2, confidence=0.9, quality=1.0)
    for _ in range(3):
        posterior = posterior.update_label(3, confidence=0.9, quality=1.0)

    assert empty.log_evidence == ()
    assert posterior.best_semantic_id == 3
    assert posterior.margin > 0.0
    assert posterior.effective_support == pytest.approx(3.6)
    assert sum(probability for _, probability in posterior.probabilities) == pytest.approx(1.0)


def test_zero_mass_posterior_update_is_a_no_op() -> None:
    posterior = SparseClassPosterior.empty()

    assert posterior.update_label(1, confidence=0.0, quality=1.0) is posterior
    assert posterior.update_label(1, confidence=1.0, quality=0.0) is posterior


@pytest.mark.parametrize("semantic_id", [0, -1, True, 1.5, "1"])
def test_posterior_rejects_invalid_semantic_id(semantic_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="semantic_id"):
        SparseClassPosterior.empty().update_label(semantic_id, 1.0, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", np.nan),
        ("confidence", np.inf),
        ("confidence", True),
        ("quality", -0.1),
        ("quality", 1.1),
        ("quality", np.nan),
        ("quality", np.inf),
        ("quality", False),
    ],
)
def test_posterior_rejects_invalid_update_weight(field_name: str, value: object) -> None:
    arguments = {"semantic_id": 1, "confidence": 1.0, "quality": 1.0}
    arguments[field_name] = value

    with pytest.raises((TypeError, ValueError), match=field_name):
        SparseClassPosterior.empty().update_label(**arguments)  # type: ignore[arg-type]


def test_posterior_statistics_are_stable_for_empty_single_and_tied_classes() -> None:
    empty = SparseClassPosterior.empty()
    single = empty.update_label(8, 0.5, 1.0)
    tied = SparseClassPosterior([[9, 1_000.0], [3, 1_000.0]], 2.0)  # type: ignore[arg-type]

    assert empty.best_semantic_id == 0
    assert empty.entropy == 0.0
    assert empty.margin == 0.0
    assert single.best_semantic_id == 8
    assert single.entropy == 0.0
    assert single.margin == pytest.approx(1.0)
    assert tied.best_semantic_id == 3
    assert tied.probabilities == ((3, 0.5), (9, 0.5))
    assert tied.entropy == pytest.approx(math.log(2.0))
    assert tied.margin == 0.0


def test_posterior_entropy_is_finite_when_a_probability_underflows() -> None:
    posterior = SparseClassPosterior(((1, -1_000.0), (2, 1_000.0)), 2.0)

    assert math.isfinite(posterior.entropy)
    assert posterior.entropy == pytest.approx(0.0)


def test_posterior_direct_construction_canonicalizes_json_payload() -> None:
    posterior = SparseClassPosterior([[7, math.log(0.25)], [2, math.log(0.75)]], np.float64(1.0))  # type: ignore[arg-type]

    assert posterior.log_evidence == (
        (2, pytest.approx(math.log(0.75))),
        (7, pytest.approx(math.log(0.25))),
    )
    assert type(posterior.effective_support) is float
    json.dumps(asdict(posterior), allow_nan=False)


@pytest.mark.parametrize(
    ("log_evidence", "effective_support", "message"),
    [
        ((((True, 0.0),)), 1.0, "semantic_id"),
        ((((0, 0.0),)), 1.0, "semantic_id"),
        ((((1, np.nan),)), 1.0, "log_evidence"),
        ((((1, 0.0), (1, 1.0))), 1.0, "unique"),
        ((), -1.0, "effective_support"),
        ((), np.inf, "effective_support"),
        ((), True, "effective_support"),
    ],
)
def test_posterior_direct_construction_rejects_invalid_payload(
    log_evidence: object,
    effective_support: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SparseClassPosterior(log_evidence, effective_support)  # type: ignore[arg-type]


def test_posterior_is_frozen() -> None:
    posterior = SparseClassPosterior.empty()

    with pytest.raises(FrozenInstanceError):
        posterior.effective_support = 1.0  # type: ignore[misc]


def test_prototype_bank_keeps_distinct_hypotheses_and_merges_similar_views() -> None:
    bank = FeaturePrototypeBank(max_prototypes=3, merge_cosine=0.90)
    bank = bank.update((1.0, 0.0), "clip:a", quality=1.0)
    bank = bank.update((0.99, 0.01), "clip:a", quality=2.0)
    bank = bank.update((0.0, 1.0), "clip:a", quality=1.0)

    assert len(bank.prototypes) == 2
    assert bank.prototypes[0].support == pytest.approx(3.0)
    assert bank.prototypes[0].observation_count == 2
    assert np.linalg.norm(bank.prototypes[0].vector) == pytest.approx(1.0)


@pytest.mark.parametrize("residual", [0.0, 1e-16], ids=("exact", "near-zero"))
def test_cancelling_prototype_merge_keeps_independent_hypotheses(residual: float) -> None:
    bank = FeaturePrototypeBank(max_prototypes=2, merge_cosine=-1.0)
    bank = bank.update((1.0, 0.0), "clip:a", quality=1.0)

    updated = bank.update((-1.0, residual), "clip:a", quality=1.0)

    assert len(updated.prototypes) == 2
    assert [item.support for item in updated.prototypes] == pytest.approx([1.0, 1.0])


def test_cancelling_prototype_merge_respects_single_slot_replacement_rule() -> None:
    bank = FeaturePrototypeBank(max_prototypes=1, merge_cosine=-1.0)
    bank = bank.update((1.0, 0.0), "clip:a", quality=1.0)

    updated = bank.update((-1.0, 0.0), "clip:a", quality=1.0)

    assert updated is bank
    assert len(updated.prototypes) == 1
    assert updated.prototypes[0].vector == pytest.approx((1.0, 0.0))


def test_prototype_bank_keeps_model_and_dimension_compatibility_domains_separate() -> None:
    bank = FeaturePrototypeBank(max_prototypes=3)
    bank = bank.update((1.0, 0.0), "clip:a", 1.0)
    bank = bank.update((1.0, 0.0), "clip:b", 1.0)
    bank = bank.update((1.0, 0.0, 0.0), "clip:a", 1.0)

    assert len(bank.prototypes) == 3
    assert bank.maximum_cosine((1.0, 0.0), "clip:a") == pytest.approx(1.0)
    assert bank.maximum_cosine((0.0, 1.0, 0.0), "clip:a") == pytest.approx(0.0)
    assert bank.maximum_cosine((1.0, 0.0, 0.0, 0.0), "clip:a") is None
    assert bank.maximum_cosine((1.0, 0.0), "clip:missing") is None


def test_full_prototype_bank_replaces_only_strictly_weaker_prototype() -> None:
    bank = FeaturePrototypeBank(max_prototypes=2)
    bank = bank.update((1.0, 0.0), "clip:a", 1.0)
    bank = bank.update((0.0, 1.0), "clip:a", 2.0)

    unchanged = bank.update((-1.0, 0.0), "clip:a", 1.0)
    replaced = bank.update((-1.0, 0.0), "clip:a", 1.1)

    assert unchanged == bank
    assert [item.support for item in replaced.prototypes] == pytest.approx([2.0, 1.1])
    assert replaced.maximum_cosine((-1.0, 0.0), "clip:a") == pytest.approx(1.0)


def test_prototype_replacement_tie_is_deterministic() -> None:
    bank = FeaturePrototypeBank(
        max_prototypes=2,
        prototypes=(
            FeaturePrototype((1.0, 0.0), "model:b", 1.0, 1),
            FeaturePrototype((1.0, 0.0), "model:a", 1.0, 1),
        ),
    )

    replaced = bank.update((1.0, 0.0), "model:c", 2.0)

    assert [item.model_id for item in bank.prototypes] == ["model:a", "model:b"]
    assert [item.model_id for item in replaced.prototypes] == ["model:c", "model:b"]


@pytest.mark.parametrize("magnitude", [1e-300, 1e300], ids=("tiny", "large"))
def test_prototype_normalizes_extreme_finite_features(magnitude: float) -> None:
    prototype = FeaturePrototype((magnitude, 0.0), " clip:a ", 1.0, 1)

    assert prototype.vector == pytest.approx((1.0, 0.0))
    assert prototype.model_id == "clip:a"
    assert all(type(value) is float for value in prototype.vector)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"vector": ()}, ValueError, "non-empty"),
        ({"vector": ((1.0, 0.0),)}, ValueError, "one dimensional"),
        ({"vector": (0.0, 0.0)}, ValueError, "nonzero"),
        ({"vector": (np.nan, 0.0)}, ValueError, "finite"),
        ({"model_id": " "}, ValueError, "model_id"),
        ({"model_id": 4}, TypeError, "model_id"),
        ({"support": 0.0}, ValueError, "support"),
        ({"support": np.inf}, ValueError, "support"),
        ({"support": True}, TypeError, "support"),
        ({"observation_count": 0}, ValueError, "observation_count"),
        ({"observation_count": True}, TypeError, "observation_count"),
    ],
)
def test_prototype_direct_construction_rejects_invalid_payload(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values = {
        "vector": (1.0, 0.0),
        "model_id": "clip:a",
        "support": 1.0,
        "observation_count": 1,
    }
    values.update(changes)

    with pytest.raises(error, match=message):
        FeaturePrototype(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("feature", "model_id", "quality", "message"),
    [
        ((), "clip:a", 1.0, "non-empty"),
        (((1.0, 0.0),), "clip:a", 1.0, "one dimensional"),
        ((0.0, 0.0), "clip:a", 1.0, "nonzero"),
        ((np.inf, 0.0), "clip:a", 1.0, "finite"),
        ((1.0, 0.0), " ", 1.0, "model_id"),
        ((1.0, 0.0), 1, 1.0, "model_id"),
        ((1.0, 0.0), "clip:a", 0.0, "quality"),
        ((1.0, 0.0), "clip:a", np.nan, "quality"),
        ((1.0, 0.0), "clip:a", True, "quality"),
    ],
)
def test_prototype_bank_update_rejects_invalid_input(
    feature: object,
    model_id: object,
    quality: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FeaturePrototypeBank().update(feature, model_id, quality)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_prototypes": 0}, "max_prototypes"),
        ({"max_prototypes": True}, "max_prototypes"),
        ({"merge_cosine": -1.1}, "merge_cosine"),
        ({"merge_cosine": 1.1}, "merge_cosine"),
        ({"merge_cosine": np.nan}, "merge_cosine"),
        ({"prototypes": (object(),)}, "prototypes"),
        (
            {
                "max_prototypes": 1,
                "prototypes": (
                    FeaturePrototype((1.0, 0.0), "a", 1.0, 1),
                    FeaturePrototype((0.0, 1.0), "a", 1.0, 1),
                ),
            },
            "max_prototypes",
        ),
    ],
)
def test_prototype_bank_direct_construction_rejects_invalid_payload(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FeaturePrototypeBank(**changes)  # type: ignore[arg-type]


def test_prototype_values_are_frozen_and_json_friendly() -> None:
    prototype = FeaturePrototype((3.0, 4.0), "clip:a", np.float64(2.0), np.int64(3))
    bank = FeaturePrototypeBank(prototypes=[prototype])  # type: ignore[arg-type]

    assert type(prototype.support) is float
    assert type(prototype.observation_count) is int
    assert bank.prototypes == (prototype,)
    with pytest.raises(FrozenInstanceError):
        prototype.support = 3.0  # type: ignore[misc]
    json.dumps(asdict(bank), allow_nan=False)


def test_view_bank_rejects_redundant_low_quality_view() -> None:
    bank = InformativeViewBank(max_views=2, minimum_novelty_cosine=0.05)
    bank = bank.update(InformativeView(1, 1, 100, 0.8, (1.0, 0.0, 0.0)))
    bank = bank.update(InformativeView(2, 2, 80, 0.4, (1.0, 0.0, 0.0)))

    assert [view.observation_id for view in bank.views] == [1]


def test_view_bank_duplicate_observation_id_is_a_no_op() -> None:
    bank = InformativeViewBank().update(
        InformativeView(1, 1, 100, 0.4, (1.0, 0.0, 0.0))
    )

    duplicate = bank.update(InformativeView(1, 2, 200, 1.0, (0.0, 1.0, 0.0)))

    assert duplicate is bank


def test_similar_higher_quality_view_replaces_existing_view() -> None:
    bank = InformativeViewBank(minimum_novelty_cosine=0.05)
    bank = bank.update(InformativeView(1, 1, 100, 0.4, (1.0, 0.0, 0.0)))
    bank = bank.update(InformativeView(2, 2, 80, 0.8, (0.999, 0.001, 0.0)))

    assert [view.observation_id for view in bank.views] == [2]


def test_full_view_bank_replaces_only_strictly_lower_quality_distinct_view() -> None:
    bank = InformativeViewBank(max_views=2)
    bank = bank.update(InformativeView(1, 1, 100, 0.2, (1.0, 0.0, 0.0)))
    bank = bank.update(InformativeView(2, 2, 100, 0.8, (0.0, 1.0, 0.0)))

    unchanged = bank.update(InformativeView(3, 3, 100, 0.2, (0.0, 0.0, 1.0)))
    replaced = bank.update(InformativeView(3, 3, 100, 0.3, (0.0, 0.0, 1.0)))

    assert unchanged == bank
    assert [view.observation_id for view in replaced.views] == [2, 3]


def test_view_replacement_tie_uses_lowest_observation_id_deterministically() -> None:
    bank = InformativeViewBank(
        max_views=2,
        views=(
            InformativeView(2, 2, 10, 0.5, (0.0, 1.0, 0.0)),
            InformativeView(1, 1, 10, 0.5, (1.0, 0.0, 0.0)),
        ),
    )

    replaced = bank.update(InformativeView(3, 3, 10, 0.9, (0.0, 0.0, 1.0)))

    assert [view.observation_id for view in bank.views] == [1, 2]
    assert [view.observation_id for view in replaced.views] == [3, 2]


@pytest.mark.parametrize("magnitude", [1e-300, 1e300], ids=("tiny", "large"))
def test_informative_view_normalizes_extreme_direction(magnitude: float) -> None:
    view = InformativeView(1, 2, 3, np.float64(0.5), (magnitude, 0.0, 0.0))

    assert view.view_direction_xyz == pytest.approx((1.0, 0.0, 0.0))
    assert type(view.quality) is float
    assert all(type(value) is float for value in view.view_direction_xyz)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"observation_id": -1}, ValueError, "observation_id"),
        ({"observation_id": True}, TypeError, "observation_id"),
        ({"frame_id": -1}, ValueError, "frame_id"),
        ({"frame_id": False}, TypeError, "frame_id"),
        ({"visible_pixel_count": -1}, ValueError, "visible_pixel_count"),
        ({"visible_pixel_count": True}, TypeError, "visible_pixel_count"),
        ({"quality": -0.1}, ValueError, "quality"),
        ({"quality": 1.1}, ValueError, "quality"),
        ({"quality": np.nan}, ValueError, "quality"),
        ({"quality": True}, TypeError, "quality"),
        ({"view_direction_xyz": (1.0, 0.0)}, ValueError, "three dimensional"),
        ({"view_direction_xyz": (np.inf, 0.0, 0.0)}, ValueError, "finite"),
        ({"view_direction_xyz": (0.0, 0.0, 0.0)}, ValueError, "nonzero"),
    ],
)
def test_informative_view_direct_construction_rejects_invalid_payload(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values = {
        "observation_id": 1,
        "frame_id": 2,
        "visible_pixel_count": 3,
        "quality": 0.5,
        "view_direction_xyz": (1.0, 0.0, 0.0),
    }
    values.update(changes)

    with pytest.raises(error, match=message):
        InformativeView(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_views": 0}, "max_views"),
        ({"max_views": True}, "max_views"),
        ({"minimum_novelty_cosine": -0.1}, "minimum_novelty_cosine"),
        ({"minimum_novelty_cosine": 2.1}, "minimum_novelty_cosine"),
        ({"minimum_novelty_cosine": np.nan}, "minimum_novelty_cosine"),
        ({"views": (object(),)}, "views"),
        (
            {
                "views": (
                    InformativeView(1, 1, 1, 0.5, (1.0, 0.0, 0.0)),
                    InformativeView(1, 2, 1, 0.6, (0.0, 1.0, 0.0)),
                ),
            },
            "unique",
        ),
        (
            {
                "max_views": 1,
                "views": (
                    InformativeView(1, 1, 1, 0.5, (1.0, 0.0, 0.0)),
                    InformativeView(2, 2, 1, 0.6, (0.0, 1.0, 0.0)),
                ),
            },
            "max_views",
        ),
    ],
)
def test_view_bank_direct_construction_rejects_invalid_payload(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        InformativeViewBank(**changes)  # type: ignore[arg-type]


def test_view_bank_direct_payload_is_canonical_frozen_and_json_friendly() -> None:
    low = InformativeView(np.int64(2), np.int64(2), np.int64(5), 0.2, (0.0, 2.0, 0.0))
    high = InformativeView(1, 1, 10, 0.8, (2.0, 0.0, 0.0))
    bank = InformativeViewBank(views=[low, high])  # type: ignore[arg-type]

    assert [view.observation_id for view in bank.views] == [1, 2]
    assert type(low.observation_id) is int
    with pytest.raises(FrozenInstanceError):
        bank.views = ()  # type: ignore[misc]
    json.dumps(asdict(bank), allow_nan=False)
