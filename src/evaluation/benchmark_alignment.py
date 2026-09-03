"""Pre-registered benchmark-alignment classifications."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HEADLINE_METRICS = ("object_f1", "dynamic_f1", "change_f1")


@dataclass(frozen=True, slots=True)
class GapMetric:
    gap_open: float
    oracle_gain: float
    closure: float


@dataclass(frozen=True, slots=True)
class FrontendGapResult:
    status: str
    metrics: Mapping[str, GapMetric]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "metrics": {
                name: {
                    "gap_open": value.gap_open,
                    "oracle_gain": value.oracle_gain,
                    "closure": value.closure,
                }
                for name, value in self.metrics.items()
            },
        }


def _finite_metric(
    values: Mapping[str, float | None] | None, name: str
) -> float | None:
    if values is None:
        return None
    raw = values.get(name)
    if raw is None or isinstance(raw, bool):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def classify_frontend_gap(
    khronos_gt: Mapping[str, float | None] | None,
    crove_open: Mapping[str, float | None] | None,
    crove_gt: Mapping[str, float | None] | None,
) -> FrontendGapResult:
    """Classify the GT-semantics oracle using the frozen 50% majority rule."""

    metrics: dict[str, GapMetric] = {}
    for name in HEADLINE_METRICS:
        k0 = _finite_metric(khronos_gt, name)
        c0 = _finite_metric(crove_open, name)
        c1 = _finite_metric(crove_gt, name)
        if k0 is None or c0 is None or c1 is None:
            continue
        gap = k0 - c0
        if gap <= 0.0:
            continue
        gain = c1 - c0
        metrics[name] = GapMetric(gap, gain, gain / gap)
    if len(metrics) < 2:
        return FrontendGapResult("INCONCLUSIVE_MISSING_CONDITION", metrics)
    closed = sum(metric.closure >= 0.5 for metric in metrics.values())
    status = (
        "FRONTEND_DOMINATED"
        if closed > len(metrics) / 2
        else "MAPPER_OR_PROTOCOL_DOMINATED"
    )
    return FrontendGapResult(status, metrics)


@dataclass(frozen=True, slots=True)
class RankMismatchResult:
    material: bool
    systematic_direction_reversal: bool
    multiple_current_gains_with_official_nonresponse: bool
    retrospective_dominance: bool
    comparable_count: int
    direction_reversal_count: int
    official_ranks: Mapping[str, tuple[str, ...]]
    current_ranks: Mapping[str, tuple[str, ...]]
    pairwise_deltas: Mapping[str, Mapping[str, Mapping[str, object]]]

    def to_dict(self) -> dict[str, object]:
        return {
            "material": self.material,
            "signals": {
                "systematic_direction_reversal": self.systematic_direction_reversal,
                "multiple_current_gains_with_official_nonresponse": (
                    self.multiple_current_gains_with_official_nonresponse
                ),
                "retrospective_dominance": self.retrospective_dominance,
            },
            "comparable_count": self.comparable_count,
            "direction_reversal_count": self.direction_reversal_count,
            "method_rank_official": {
                metric: list(methods) for metric, methods in self.official_ranks.items()
            },
            "method_rank_current": {
                metric: list(methods) for metric, methods in self.current_ranks.items()
            },
            "pairwise_deltas": {
                method: {metric: dict(values) for metric, values in metrics.items()}
                for method, metrics in self.pairwise_deltas.items()
            },
        }


def _direction(delta: float) -> int:
    return (delta > 0.0) - (delta < 0.0)


def _ranks(
    values: Mapping[str, Mapping[str, float | None]],
    higher_is_better: Mapping[str, bool],
) -> dict[str, tuple[str, ...]]:
    metrics = sorted({metric for method in values.values() for metric in method})
    result: dict[str, tuple[str, ...]] = {}
    for metric in metrics:
        orientation = 1.0 if higher_is_better.get(metric, True) else -1.0
        finite = [
            (method, value)
            for method, method_values in values.items()
            if (value := _finite_metric(method_values, metric)) is not None
        ]
        result[metric] = tuple(
            method
            for method, _ in sorted(
                finite,
                key=lambda item: (-orientation * item[1], item[0]),
            )
        )
    return result


def classify_rank_mismatch(
    official: Mapping[str, Mapping[str, float | None]],
    current: Mapping[str, Mapping[str, float | None]],
    *,
    reference: str = "A6",
    candidates: Sequence[str] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    retrospective_mass_fraction: float | None = None,
    bounded_current_claim: bool = True,
) -> RankMismatchResult:
    """Evaluate the three frozen mismatch signals without a weighted score."""

    if retrospective_mass_fraction is not None and (
        not math.isfinite(retrospective_mass_fraction)
        or not 0.0 <= retrospective_mass_fraction <= 1.0
    ):
        raise ValueError("retrospective mass fraction must be finite and in [0, 1]")
    directions = higher_is_better or {}
    official_ranks = _ranks(official, directions)
    current_ranks = _ranks(current, directions)
    if reference not in official or reference not in current:
        return RankMismatchResult(
            False, False, False, False, 0, 0, official_ranks, current_ranks, {}
        )
    selected = tuple(candidates or sorted((set(official) & set(current)) - {reference}))
    comparable = 0
    reversals = 0
    multi_gain_nonresponse = False
    pairwise: dict[str, dict[str, dict[str, object]]] = {}
    for candidate in selected:
        if candidate not in official or candidate not in current:
            continue
        pairwise[candidate] = {}
        metric_names = (
            set(official[reference])
            & set(official[candidate])
            & set(current[reference])
            & set(current[candidate])
        )
        nonresponsive_current_gains = 0
        for metric in metric_names:
            values = (
                _finite_metric(official[reference], metric),
                _finite_metric(official[candidate], metric),
                _finite_metric(current[reference], metric),
                _finite_metric(current[candidate], metric),
            )
            if any(value is None for value in values):
                continue
            official_reference, official_candidate, current_reference, current_candidate = values
            assert official_reference is not None
            assert official_candidate is not None
            assert current_reference is not None
            assert current_candidate is not None
            orientation = 1.0 if directions.get(metric, True) else -1.0
            official_direction = _direction(
                orientation * (official_candidate - official_reference)
            )
            current_direction = _direction(
                orientation * (current_candidate - current_reference)
            )
            pairwise[candidate][metric] = {
                "official_delta": official_candidate - official_reference,
                "current_delta": current_candidate - current_reference,
                "reversal": bool(
                    official_direction
                    and current_direction
                    and official_direction != current_direction
                ),
            }
            if official_direction and current_direction:
                comparable += 1
                reversals += official_direction != current_direction
            if current_direction > 0 and official_direction <= 0:
                nonresponsive_current_gains += 1
        multi_gain_nonresponse = (
            multi_gain_nonresponse or nonresponsive_current_gains >= 2
        )
    systematic = reversals >= 2 and reversals > comparable / 2
    retrospective = (
        bounded_current_claim
        and retrospective_mass_fraction is not None
        and math.isfinite(retrospective_mass_fraction)
        and retrospective_mass_fraction > 0.5
    )
    return RankMismatchResult(
        systematic or multi_gain_nonresponse or retrospective,
        systematic,
        multi_gain_nonresponse,
        retrospective,
        comparable,
        reversals,
        official_ranks,
        current_ranks,
        pairwise,
    )
