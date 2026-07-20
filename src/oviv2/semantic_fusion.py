from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Integral, Real


_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class SemanticFusionConfig:
    entity_weight_scale: float = 0.5

    def __post_init__(self) -> None:
        value = self.entity_weight_scale
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("entity_weight_scale must be a real number")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("entity_weight_scale must lie in [0, 1]")
        object.__setattr__(self, "entity_weight_scale", normalized)


@dataclass(frozen=True)
class FusedSemanticLabel:
    semantic_id: int
    confidence: float
    entity_weight: float


def _distribution(
    values: Iterable[tuple[int, float]],
    *,
    value_name: str,
    normalize: bool,
) -> dict[int, float]:
    try:
        entries = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{value_name} values must be an iterable of pairs") from exc
    result: dict[int, float] = {}
    for entry in entries:
        try:
            semantic_id, raw_value = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value_name} entries must be pairs") from exc
        if isinstance(semantic_id, bool) or not isinstance(semantic_id, Integral):
            raise TypeError("semantic ID must be a positive integer")
        normalized_id = int(semantic_id)
        if normalized_id <= 0:
            raise ValueError("semantic ID must be a positive integer")
        if normalized_id > _INT64_MAX:
            raise ValueError("semantic ID must fit signed int64")
        if normalized_id in result:
            raise ValueError("semantic IDs must be unique")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"{value_name} must be a real number")
        normalized_value = float(raw_value)
        if not math.isfinite(normalized_value) or normalized_value <= 0.0:
            raise ValueError(f"{value_name} must be finite and positive")
        result[normalized_id] = normalized_value
    if not result:
        return result
    total = sum(result.values())
    if not normalize and not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("entity probabilities must sum to one")
    return {semantic_id: value / total for semantic_id, value in result.items()}


def fuse_semantics(
    dense_support: Iterable[tuple[int, float]],
    entity_probabilities: Iterable[tuple[int, float]],
    ownership_confidence: float,
    config: SemanticFusionConfig = SemanticFusionConfig(),
) -> FusedSemanticLabel:
    if not isinstance(config, SemanticFusionConfig):
        raise TypeError("config must be SemanticFusionConfig")
    if isinstance(ownership_confidence, bool) or not isinstance(
        ownership_confidence,
        Real,
    ):
        raise TypeError("ownership_confidence must be a real number")
    ownership = float(ownership_confidence)
    if not math.isfinite(ownership) or not 0.0 <= ownership <= 1.0:
        raise ValueError("ownership_confidence must lie in [0, 1]")

    dense = _distribution(dense_support, value_name="dense support", normalize=True)
    entity = _distribution(
        entity_probabilities,
        value_name="entity probability",
        normalize=False,
    )
    dense_confidence = max(dense.values(), default=0.0)
    entity_confidence = max(entity.values(), default=0.0)
    entity_weight = (
        config.entity_weight_scale
        * ownership
        * entity_confidence
        * (1.0 - dense_confidence)
    )
    scores = {
        semantic_id: (1.0 - entity_weight) * probability
        for semantic_id, probability in dense.items()
    }
    for semantic_id, probability in entity.items():
        scores[semantic_id] = scores.get(semantic_id, 0.0) + entity_weight * probability
    total = sum(scores.values())
    if total <= 0.0:
        return FusedSemanticLabel(0, 0.0, 0.0)
    semantic_id, score = min(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return FusedSemanticLabel(
        semantic_id=semantic_id,
        confidence=score / total,
        entity_weight=entity_weight,
    )
