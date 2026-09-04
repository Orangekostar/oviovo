"""Pareto metrics for two-visit current-state maps."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

_IDENTITY_STATES = frozenset(
    {
        "persistent_static",
        "persistent_moved",
        "appeared",
        "removed_candidate",
        "split",
        "merge",
        "ambiguous",
    }
)


def _points(value: object, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


def _mask(value: object, *, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != (count,):
        raise ValueError(f"{name} must be a boolean vector with one value per point")
    result = np.array(raw, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError("distance_threshold_m must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive")
    return result


def _match_count(query: np.ndarray, reference: np.ndarray, threshold: float) -> int:
    if not len(query) or not len(reference):
        return 0
    distances, _ = cKDTree(reference).query(query, k=1, workers=-1)
    return int(np.count_nonzero(np.asarray(distances) < threshold))


def _precision(matches: int, predicted_count: int, target_count: int) -> float:
    if predicted_count:
        return matches / predicted_count
    return 1.0 if target_count == 0 else 0.0


def _recall(matches: int, target_count: int) -> float:
    return matches / target_count if target_count else 1.0


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


@dataclass(frozen=True, slots=True)
class RegionEvaluationInput:
    """Evaluator-only point sets and masks for completeness diagnostics."""

    predicted_current_xyz: np.ndarray
    ground_truth_current_xyz: np.ndarray
    ground_truth_t1_unobserved_mask: np.ndarray
    retained_t0_xyz: np.ndarray
    retained_t0_t1_observed_mask: np.ndarray
    confirmed_free_xyz: np.ndarray
    predicted_background_xyz: np.ndarray
    ground_truth_background_xyz: np.ndarray
    distance_threshold_m: float = 0.05

    def __post_init__(self) -> None:
        point_names = (
            "predicted_current_xyz",
            "ground_truth_current_xyz",
            "retained_t0_xyz",
            "confirmed_free_xyz",
            "predicted_background_xyz",
            "ground_truth_background_xyz",
        )
        for name in point_names:
            object.__setattr__(self, name, _points(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "ground_truth_t1_unobserved_mask",
            _mask(
                self.ground_truth_t1_unobserved_mask,
                count=len(self.ground_truth_current_xyz),
                name="ground-truth unobserved mask",
            ),
        )
        object.__setattr__(
            self,
            "retained_t0_t1_observed_mask",
            _mask(
                self.retained_t0_t1_observed_mask,
                count=len(self.retained_t0_xyz),
                name="retained-t0 observed mask",
            ),
        )
        object.__setattr__(
            self, "distance_threshold_m", _threshold(self.distance_threshold_m)
        )


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    distance_threshold_m: float
    surface_precision: float
    surface_recall: float
    surface_f1: float
    total_current_surface_coverage: float
    unobserved_region_recall: float
    observed_region_stale_precision: float
    confirmed_free_ghost_rate: float
    background_precision: float
    background_recall: float
    background_f1: float
    predicted_current_count: int
    ground_truth_current_count: int
    unobserved_target_count: int
    observed_historical_count: int
    confirmed_free_ghost_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_regions(value: RegionEvaluationInput) -> RegionMetrics:
    """Measure quality and coverage without collapsing them into one score."""

    if not isinstance(value, RegionEvaluationInput):
        raise TypeError("value must be RegionEvaluationInput")
    threshold = value.distance_threshold_m
    prediction = value.predicted_current_xyz
    target = value.ground_truth_current_xyz
    precision_matches = _match_count(prediction, target, threshold)
    recall_matches = _match_count(target, prediction, threshold)
    surface_precision = _precision(precision_matches, len(prediction), len(target))
    surface_recall = _recall(recall_matches, len(target))

    unobserved = target[value.ground_truth_t1_unobserved_mask]
    unobserved_matches = _match_count(unobserved, prediction, threshold)
    unobserved_recall = _recall(unobserved_matches, len(unobserved))

    observed_historical = value.retained_t0_xyz[
        value.retained_t0_t1_observed_mask
    ]
    stale_count = _match_count(observed_historical, value.confirmed_free_xyz, threshold)
    stale_precision = (
        1.0 - stale_count / len(observed_historical)
        if len(observed_historical)
        else 1.0
    )
    ghost_count = _match_count(prediction, value.confirmed_free_xyz, threshold)
    ghost_rate = ghost_count / len(prediction) if len(prediction) else 0.0

    background_precision_matches = _match_count(
        value.predicted_background_xyz,
        value.ground_truth_background_xyz,
        threshold,
    )
    background_recall_matches = _match_count(
        value.ground_truth_background_xyz,
        value.predicted_background_xyz,
        threshold,
    )
    background_precision = _precision(
        background_precision_matches,
        len(value.predicted_background_xyz),
        len(value.ground_truth_background_xyz),
    )
    background_recall = _recall(
        background_recall_matches, len(value.ground_truth_background_xyz)
    )
    return RegionMetrics(
        distance_threshold_m=threshold,
        surface_precision=surface_precision,
        surface_recall=surface_recall,
        surface_f1=_f1(surface_precision, surface_recall),
        total_current_surface_coverage=surface_recall,
        unobserved_region_recall=unobserved_recall,
        observed_region_stale_precision=stale_precision,
        confirmed_free_ghost_rate=ghost_rate,
        background_precision=background_precision,
        background_recall=background_recall,
        background_f1=_f1(background_precision, background_recall),
        predicted_current_count=len(prediction),
        ground_truth_current_count=len(target),
        unobserved_target_count=len(unobserved),
        observed_historical_count=len(observed_historical),
        confirmed_free_ghost_count=ghost_count,
    )


@dataclass(frozen=True, slots=True)
class IdentityRelation:
    t0_entity_ids: tuple[str, ...]
    t1_entity_ids: tuple[str, ...]
    state: str

    def __post_init__(self) -> None:
        t0_ids = tuple(str(item).strip() for item in self.t0_entity_ids)
        t1_ids = tuple(str(item).strip() for item in self.t1_entity_ids)
        if (
            any(not item for item in (*t0_ids, *t1_ids))
            or t0_ids != tuple(sorted(set(t0_ids)))
            or t1_ids != tuple(sorted(set(t1_ids)))
        ):
            raise ValueError("identity entity IDs must be sorted, unique, and non-empty")
        if self.state not in _IDENTITY_STATES:
            raise ValueError("identity state is invalid")
        expected_shape = {
            "persistent_static": bool(t0_ids) and bool(t1_ids),
            "persistent_moved": bool(t0_ids) and bool(t1_ids),
            "appeared": not t0_ids and bool(t1_ids),
            "removed_candidate": bool(t0_ids) and not t1_ids,
            "split": len(t0_ids) == 1 and len(t1_ids) >= 2,
            "merge": len(t0_ids) >= 2 and len(t1_ids) == 1,
            "ambiguous": bool(t0_ids or t1_ids),
        }[self.state]
        if not expected_shape:
            raise ValueError(f"identity relation shape is invalid for {self.state}")
        object.__setattr__(self, "t0_entity_ids", t0_ids)
        object.__setattr__(self, "t1_entity_ids", t1_ids)


@dataclass(frozen=True, slots=True)
class IdentityMetrics:
    available: bool
    unavailable_reason: str | None
    persistent_precision: float | None
    persistent_recall: float | None
    moved_accuracy: float | None
    appeared_precision: float | None
    appeared_recall: float | None
    removed_precision: float | None
    removed_recall: float | None
    split_accuracy: float | None
    merge_accuracy: float | None
    predicted_relation_count: int
    ground_truth_relation_count: int | None

    def as_dict(self) -> dict[str, bool | float | int | str | None]:
        return asdict(self)


def _relation_set(
    values: tuple[IdentityRelation, ...], states: frozenset[str]
) -> set[IdentityRelation]:
    return {item for item in values if item.state in states}


def _precision_recall(
    predicted: set[IdentityRelation], expected: set[IdentityRelation]
) -> tuple[float | None, float | None]:
    matches = len(predicted & expected)
    precision = matches / len(predicted) if predicted else None
    recall = matches / len(expected) if expected else None
    return precision, recall


def _class_accuracy(
    predicted: tuple[IdentityRelation, ...],
    expected: tuple[IdentityRelation, ...],
    state: str,
) -> float | None:
    target = _relation_set(expected, frozenset({state}))
    if not target:
        return None
    observed = _relation_set(predicted, frozenset({state}))
    return len(observed & target) / len(target)


def evaluate_identity(
    predicted: tuple[IdentityRelation, ...],
    ground_truth: tuple[IdentityRelation, ...] | None,
) -> IdentityMetrics:
    """Evaluate exact entity relations while retaining unavailable values as N/A."""

    if not isinstance(predicted, tuple) or any(
        not isinstance(item, IdentityRelation) for item in predicted
    ):
        raise TypeError("predicted must contain IdentityRelation values")
    if len(predicted) != len(set(predicted)):
        raise ValueError("predicted identity relations must be unique")
    if ground_truth is None:
        return IdentityMetrics(
            available=False,
            unavailable_reason="identity_ground_truth_unavailable",
            persistent_precision=None,
            persistent_recall=None,
            moved_accuracy=None,
            appeared_precision=None,
            appeared_recall=None,
            removed_precision=None,
            removed_recall=None,
            split_accuracy=None,
            merge_accuracy=None,
            predicted_relation_count=len(predicted),
            ground_truth_relation_count=None,
        )
    if not isinstance(ground_truth, tuple) or any(
        not isinstance(item, IdentityRelation) for item in ground_truth
    ):
        raise TypeError("ground_truth must contain IdentityRelation values")
    if len(ground_truth) != len(set(ground_truth)):
        raise ValueError("ground-truth identity relations must be unique")
    persistent_states = frozenset({"persistent_static", "persistent_moved"})
    persistent_precision, persistent_recall = _precision_recall(
        _relation_set(predicted, persistent_states),
        _relation_set(ground_truth, persistent_states),
    )
    appeared_precision, appeared_recall = _precision_recall(
        _relation_set(predicted, frozenset({"appeared"})),
        _relation_set(ground_truth, frozenset({"appeared"})),
    )
    removed_precision, removed_recall = _precision_recall(
        _relation_set(predicted, frozenset({"removed_candidate"})),
        _relation_set(ground_truth, frozenset({"removed_candidate"})),
    )
    return IdentityMetrics(
        available=True,
        unavailable_reason=None,
        persistent_precision=persistent_precision,
        persistent_recall=persistent_recall,
        moved_accuracy=_class_accuracy(predicted, ground_truth, "persistent_moved"),
        appeared_precision=appeared_precision,
        appeared_recall=appeared_recall,
        removed_precision=removed_precision,
        removed_recall=removed_recall,
        split_accuracy=_class_accuracy(predicted, ground_truth, "split"),
        merge_accuracy=_class_accuracy(predicted, ground_truth, "merge"),
        predicted_relation_count=len(predicted),
        ground_truth_relation_count=len(ground_truth),
    )


__all__ = [
    "IdentityMetrics",
    "IdentityRelation",
    "RegionEvaluationInput",
    "RegionMetrics",
    "evaluate_identity",
    "evaluate_regions",
]
