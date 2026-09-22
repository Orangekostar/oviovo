"""Frozen CAL/SELECT hierarchy for semantic, geometry, and query modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .metric_order import TOLERANCE, metric_max


@dataclass(frozen=True)
class SceneMethodMetrics:
    method_id: str
    scene_id: str
    uap: float | None
    miou: float | None
    canonical_ap50: float | None = None
    canonical_ap75: float | None = None
    logical_attempts: int = 0
    added_cost: float = 0.0
    effective_changes: int = 0

    def __post_init__(self) -> None:
        if not self.method_id or not self.scene_id:
            raise ValueError("scene method metric identity is required")
        for name in ("uap", "miou", "canonical_ap50", "canonical_ap75"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be null or lie in [0, 1]")
        if self.logical_attempts < 0 or self.added_cost < 0.0 or self.effective_changes < 0:
            raise ValueError("selection costs and change counts must be nonnegative")


def _method_rows(
    rows: Mapping[str, Sequence[SceneMethodMetrics]], method_id: str
) -> tuple[SceneMethodMetrics, ...]:
    if method_id not in rows:
        raise ValueError(f"required selection method is missing: {method_id}")
    result = tuple(rows[method_id])
    if not result or any(row.method_id != method_id for row in result):
        raise ValueError(f"selection rows do not match method {method_id}")
    scenes = [row.scene_id for row in result]
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"selection method {method_id} repeats a scene")
    return result


def _aligned(
    rows: Mapping[str, Sequence[SceneMethodMetrics]], methods: Sequence[str]
) -> dict[str, tuple[SceneMethodMetrics, ...]]:
    normalized = {method: _method_rows(rows, method) for method in methods}
    reference = {row.scene_id for row in normalized[methods[0]]}
    if any({row.scene_id for row in values} != reference for values in normalized.values()):
        raise ValueError("selection methods must cover the same physical scenes")
    return {
        method: tuple(sorted(values, key=lambda row: row.scene_id))
        for method, values in normalized.items()
    }


def _mean(rows: Sequence[SceneMethodMetrics], name: str) -> float | None:
    values = [getattr(row, name) for row in rows]
    if any(value is None for value in values):
        return None
    return float(np.mean(values))


def _all_delta(
    candidate: Sequence[SceneMethodMetrics],
    baseline: Sequence[SceneMethodMetrics],
    name: str,
    *,
    strict: bool,
) -> bool:
    values = [
        (getattr(left, name), getattr(right, name))
        for left, right in zip(candidate, baseline, strict=True)
    ]
    if any(left is None or right is None for left, right in values):
        return False
    if strict:
        return all(float(left) > float(right) + TOLERANCE for left, right in values)
    return all(float(left) >= float(right) - TOLERANCE for left, right in values)


def _not_below(candidate: float | None, baseline: float | None) -> bool:
    return candidate is not None and baseline is not None and candidate >= baseline - TOLERANCE


def _strictly_above(candidate: float | None, baseline: float | None) -> bool:
    return candidate is not None and baseline is not None and candidate > baseline + TOLERANCE


@dataclass(frozen=True)
class SemanticSelection:
    selected_method: str
    frozen_teacher_id: str
    paired_eligible: bool
    mechanism_status: str
    status: str


def select_semantic_module(
    rows: Mapping[str, Sequence[SceneMethodMetrics]],
    *,
    frozen_teacher_id: str,
) -> SemanticSelection:
    methods = (
        "N0",
        frozen_teacher_id,
        "S_SIMPLE",
        "S_NO_CONTEXT",
        "S_PAIRED",
    )
    aligned = _aligned(rows, methods)
    n0 = aligned["N0"]
    simple = aligned["S_SIMPLE"]
    no_context = aligned["S_NO_CONTEXT"]
    paired = aligned["S_PAIRED"]
    paired_eligible = (
        sum(row.effective_changes for row in paired) >= 1
        and _strictly_above(_mean(paired, "uap"), _mean(n0, "uap"))
        and _strictly_above(_mean(paired, "uap"), _mean(simple, "uap"))
        and _not_below(_mean(paired, "miou"), _mean(n0, "miou"))
        and _not_below(_mean(paired, "miou"), _mean(simple, "miou"))
        and _all_delta(paired, n0, "uap", strict=True)
    )
    no_context_matches = (
        paired_eligible
        and _not_below(_mean(no_context, "uap"), _mean(paired, "uap"))
        and _not_below(_mean(no_context, "miou"), _mean(paired, "miou"))
        and _all_delta(no_context, n0, "uap", strict=False)
    )
    if paired_eligible and no_context_matches:
        mechanism = "UNSUPPORTED_CONTEXT_MATCHED_BY_NO_CONTEXT"
    elif paired_eligible:
        mechanism = "SUPPORTED_PAIRED_MECHANISM"
    else:
        mechanism = "NOT_ELIGIBLE"

    complexity = {
        "N0": 0,
        frozen_teacher_id: 1,
        "S_SIMPLE": 2,
        "S_NO_CONTEXT": 3,
        "S_PAIRED": 4,
    }
    if any(
        _mean(aligned[method], metric) is None
        for method in methods
        for metric in ("uap", "miou")
    ):
        return SemanticSelection(
            "N0",
            frozen_teacher_id,
            paired_eligible,
            mechanism,
            "INCONCLUSIVE_UNDEFINED_METRIC",
        )
    best = metric_max(
        methods,
        key=lambda method: (
            _mean(aligned[method], "uap"),
            _mean(aligned[method], "miou"),
            -float(np.mean([row.added_cost for row in aligned[method]])),
            -complexity[method],
        ),
    )
    usable = (
        best != "N0"
        and _strictly_above(_mean(aligned[best], "uap"), _mean(n0, "uap"))
        and _not_below(_mean(aligned[best], "miou"), _mean(n0, "miou"))
        and _all_delta(aligned[best], n0, "uap", strict=False)
    )
    return SemanticSelection(
        best if usable else "N0",
        frozen_teacher_id,
        paired_eligible,
        mechanism,
        "RETAINED" if usable else "N0_FALLBACK",
    )


@dataclass(frozen=True)
class GeometrySelection:
    selected_method: str
    quality_eligible: bool
    status: str


def _geometry_retention_gate(
    candidate: Sequence[SceneMethodMetrics],
    original: Sequence[SceneMethodMetrics],
) -> bool:
    return (
        sum(row.effective_changes for row in candidate) >= 1
        and _strictly_above(
            _mean(candidate, "canonical_ap50"), _mean(original, "canonical_ap50")
        )
        and _not_below(
            _mean(candidate, "canonical_ap75"), _mean(original, "canonical_ap75")
        )
        and _not_below(_mean(candidate, "uap"), _mean(original, "uap"))
        and _not_below(_mean(candidate, "miou"), _mean(original, "miou"))
        and _all_delta(candidate, original, "canonical_ap50", strict=True)
    )


def select_geometry_module(
    rows: Mapping[str, Sequence[SceneMethodMetrics]],
) -> GeometrySelection:
    methods = ("G_ORIGINAL", "G_AGREEMENT", "G_QUALITY")
    aligned = _aligned(rows, methods)
    original = aligned["G_ORIGINAL"]
    quality = aligned["G_QUALITY"]
    quality_eligible = (
        _geometry_retention_gate(quality, original)
        and _strictly_above(
            _mean(quality, "canonical_ap50"),
            _mean(aligned["G_AGREEMENT"], "canonical_ap50"),
        )
    )
    required = ("canonical_ap50", "canonical_ap75", "uap", "miou")
    if any(
        _mean(aligned[method], metric) is None
        for method in methods
        for metric in required
    ):
        return GeometrySelection(
            "G_ORIGINAL", quality_eligible, "INCONCLUSIVE_UNDEFINED_METRIC"
        )
    originality = {"G_ORIGINAL": 2, "G_AGREEMENT": 1, "G_QUALITY": 0}
    best = metric_max(
        methods,
        metrics=3,
        key=lambda method: (
            _mean(aligned[method], "canonical_ap50"),
            _mean(aligned[method], "canonical_ap75"),
            _mean(aligned[method], "uap"),
            -sum(row.effective_changes for row in aligned[method]),
            originality[method],
        ),
    )
    retained = best != "G_ORIGINAL" and _geometry_retention_gate(
        aligned[best], original
    )
    return GeometrySelection(
        best if retained else "G_ORIGINAL",
        quality_eligible,
        "RETAINED" if retained else "ORIGINAL_FALLBACK",
    )


@dataclass(frozen=True)
class QuerySelection:
    selected_method: str
    locked_comparator: str
    gain_status: str
    status: str


def select_query_module(
    cal_rows: Mapping[str, Sequence[SceneMethodMetrics]],
    select_rows: Mapping[str, Sequence[SceneMethodMetrics]],
) -> QuerySelection:
    comparator_order = ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY")
    cal = _aligned(cal_rows, comparator_order)
    if any(
        _mean(cal[method], metric) is None
        for method in comparator_order
        for metric in ("uap", "miou")
    ):
        return QuerySelection(
            "Q_COMBINE",
            "Q_COMBINE",
            "INCONCLUSIVE_UNDEFINED_METRIC",
            "INCONCLUSIVE_UNDEFINED_METRIC",
        )
    locked = metric_max(
        comparator_order,
        key=lambda method: (
            _mean(cal[method], "uap"),
            _mean(cal[method], "miou"),
            -float(np.mean([row.logical_attempts for row in cal[method]])),
            -comparator_order.index(method),
        ),
    )
    select = _aligned(select_rows, (locked, "Q_GAIN"))
    baseline = select[locked]
    gain = select["Q_GAIN"]
    if any(_mean(select[method], metric) is None for method in select for metric in ("uap", "miou")):
        return QuerySelection(
            locked, locked, "INCONCLUSIVE_UNDEFINED_METRIC", "COMPARATOR_RETAINED"
        )
    nonnegative = _all_delta(gain, baseline, "uap", strict=False)
    at_least_one_positive = any(
        float(left.uap) > float(right.uap) + TOLERANCE
        for left, right in zip(gain, baseline, strict=True)
    )
    eligible = (
        _strictly_above(_mean(gain, "uap"), _mean(baseline, "uap"))
        and _not_below(_mean(gain, "miou"), _mean(baseline, "miou"))
        and nonnegative
        and at_least_one_positive
    )
    cheaper = np.mean([row.logical_attempts for row in gain]) < np.mean(
        [row.logical_attempts for row in baseline]
    )
    efficiency = (
        not eligible
        and _not_below(_mean(gain, "uap"), _mean(baseline, "uap"))
        and _not_below(_mean(gain, "miou"), _mean(baseline, "miou"))
        and nonnegative
        and cheaper
    )
    if eligible:
        return QuerySelection("Q_GAIN", locked, "ELIGIBLE_ACCURACY", "RETAINED")
    if efficiency:
        return QuerySelection("Q_GAIN", locked, "EFFICIENCY_ONLY", "RETAINED_ENGINEERING")
    return QuerySelection(locked, locked, "NOT_ELIGIBLE", "COMPARATOR_RETAINED")


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    lower: float
    upper: float
    seed: int
    resamples: int


def bootstrap_mean_delta(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 17,
    resamples: int = 2000,
) -> BootstrapInterval:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("bootstrap inputs must be aligned nonempty scene vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("bootstrap scene metrics must be finite")
    if seed != 17 or resamples != 2000:
        raise ValueError("descriptive bootstrap is frozen to seed17/2000 resamples")
    delta = left - right
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(delta), size=(resamples, len(delta)))
    samples = np.mean(delta[indices], axis=1)
    return BootstrapInterval(
        float(np.mean(delta)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        seed,
        resamples,
    )


@dataclass(frozen=True)
class ConfirmationPlan:
    status: str
    rows: tuple[str, ...]


def plan_confirmation(
    final_candidate: str,
    *,
    nearest_comparison: str | None = None,
) -> ConfirmationPlan:
    if final_candidate == "N0":
        return ConfirmationPlan("NOT_REQUIRED_NO_RETAINED_CANDIDATE", ())
    rows = tuple(
        dict.fromkeys(
            value
            for value in ("N0", final_candidate, nearest_comparison)
            if value is not None
        )
    )
    if len(rows) > 4:
        raise ValueError("confirmation may contain at most four rows per scene")
    return ConfirmationPlan("REQUIRED_FROZEN", rows)


@dataclass(frozen=True)
class CombinationPlan:
    required: tuple[str, ...]
    blocked: Mapping[str, str]
    not_required: Mapping[str, str]


def plan_combinations(
    *,
    semantic_method: str,
    geometry_method: str,
    query_method: str,
    fresh_mask_supported: bool,
    query_provenance_supported: bool,
) -> CombinationPlan:
    static_retained = semantic_method != "N0" or geometry_method != "G_ORIGINAL"
    both_static = semantic_method != "N0" and geometry_method != "G_ORIGINAL"
    required: list[str] = []
    blocked: dict[str, str] = {}
    not_required: dict[str, str] = {}
    if both_static:
        if fresh_mask_supported:
            required.append("COMBO_GS")
        else:
            blocked["COMBO_GS"] = "BLOCKED_FRESH_MASK_PROVENANCE"
    else:
        not_required["COMBO_GS"] = "REQUIRES_RETAINED_S_AND_CHANGED_G"
    if static_retained and query_method == "Q_GAIN":
        if fresh_mask_supported and query_provenance_supported:
            required.append("COMBO_Q_REFINEMENT")
        else:
            blocked["COMBO_Q_REFINEMENT"] = "BLOCKED_FRESH_MASK_PROVENANCE"
    else:
        not_required["COMBO_Q_REFINEMENT"] = "REQUIRES_STATIC_REFINEMENT_AND_Q_GAIN"
    if len(required) > 2:
        raise ValueError("at most two combination variants are permitted")
    return CombinationPlan(tuple(required), blocked, not_required)


@dataclass(frozen=True)
class FinalSelection:
    selected_method: str
    science_status: str
    eligible_methods: tuple[str, ...]


def select_final_candidate(
    candidates: Mapping[str, Sequence[SceneMethodMetrics]],
    *,
    matched_baselines: Mapping[str, Sequence[SceneMethodMetrics]],
    simplicity_order: Sequence[str],
) -> FinalSelection:
    methods = tuple(simplicity_order)
    if set(methods) != set(candidates) or set(methods) != set(matched_baselines):
        raise ValueError("final candidates, matched baselines, and simplicity order must align")
    eligible: list[str] = []
    undefined = False
    normalized: dict[str, tuple[SceneMethodMetrics, ...]] = {}
    for method in methods:
        candidate = tuple(sorted(_method_rows(candidates, method), key=lambda row: row.scene_id))
        baseline = tuple(sorted(matched_baselines[method], key=lambda row: row.scene_id))
        if not baseline or len({row.scene_id for row in baseline}) != len(baseline):
            raise ValueError(f"matched baseline for {method} is empty or repeats a scene")
        if {row.scene_id for row in candidate} != {row.scene_id for row in baseline}:
            raise ValueError(f"matched baseline scenes differ for {method}")
        normalized[method] = candidate
        if any(_mean(values, metric) is None for values in (candidate, baseline)
               for metric in ("uap", "miou")):
            undefined = True
        if (
            _strictly_above(_mean(candidate, "uap"), _mean(baseline, "uap"))
            and _not_below(_mean(candidate, "miou"), _mean(baseline, "miou"))
            and _all_delta(candidate, baseline, "uap", strict=False)
        ):
            eligible.append(method)
    if not eligible:
        return FinalSelection(
            "N0", "INCONCLUSIVE_UNDEFINED_METRIC" if undefined else "NO_NET_GAIN", ()
        )
    selected = metric_max(
        eligible,
        key=lambda method: (
            _mean(normalized[method], "uap"),
            _mean(normalized[method], "miou"),
            -float(np.mean([row.added_cost for row in normalized[method]])),
            -methods.index(method),
        ),
    )
    return FinalSelection(selected, "RETAINED_NET_GAIN", tuple(eligible))


__all__ = [
    "TOLERANCE",
    "BootstrapInterval",
    "CombinationPlan",
    "ConfirmationPlan",
    "FinalSelection",
    "GeometrySelection",
    "QuerySelection",
    "SceneMethodMetrics",
    "SemanticSelection",
    "bootstrap_mean_delta",
    "plan_combinations",
    "plan_confirmation",
    "select_final_candidate",
    "select_geometry_module",
    "select_query_module",
    "select_semantic_module",
]
