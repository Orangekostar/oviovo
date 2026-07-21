from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.evaluation.oviv2_instance_head import InstanceHypothesis
from src.oviv2.meshing import LabeledMesh


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True)
class AuxiliaryInstanceConfig:
    score_weight: float = 1.0
    support_exponent: float = 1.0
    score_bias: float = 0.0

    def __post_init__(self) -> None:
        weight = _finite_real(self.score_weight, "score_weight")
        if weight <= 0.0:
            raise ValueError("score_weight must be positive")
        exponent = _finite_real(self.support_exponent, "support_exponent")
        if exponent < 0.0:
            raise ValueError("support_exponent must be non-negative")
        bias = _finite_real(self.score_bias, "score_bias")
        if not 0.0 <= bias <= 1.0:
            raise ValueError("score_bias must lie in [0, 1]")
        object.__setattr__(self, "score_weight", weight)
        object.__setattr__(self, "support_exponent", exponent)
        object.__setattr__(self, "score_bias", bias)


def build_auxiliary_entity_hypotheses(
    mesh: LabeledMesh,
    entity_semantic_ids: Mapping[int, int],
    config: AuxiliaryInstanceConfig = AuxiliaryInstanceConfig(),
) -> tuple[InstanceHypothesis, ...]:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    if not isinstance(entity_semantic_ids, Mapping):
        raise TypeError("entity_semantic_ids must be a mapping")
    if not isinstance(config, AuxiliaryInstanceConfig):
        raise TypeError("config must be an AuxiliaryInstanceConfig")

    normalized_semantics: dict[int, int] = {}
    for entity_id, semantic_id in entity_semantic_ids.items():
        normalized_id = _positive_integer(entity_id, "entity_id")
        normalized_semantics[normalized_id] = _positive_integer(
            semantic_id,
            "semantic_id",
        )

    entity_ids = sorted(
        int(value) for value in np.unique(mesh.entity_ids) if int(value) > 0
    )
    missing = [entity_id for entity_id in entity_ids if entity_id not in normalized_semantics]
    if missing:
        raise ValueError(f"missing semantic IDs for auxiliary entities {missing}")
    if not entity_ids:
        return ()

    indices_by_entity = {
        entity_id: np.flatnonzero(mesh.entity_ids == entity_id)
        for entity_id in entity_ids
    }
    maximum_support = max(len(indices) for indices in indices_by_entity.values())
    hypotheses: list[InstanceHypothesis] = []
    for entity_id in entity_ids:
        indices = indices_by_entity[entity_id]
        support = len(indices) / maximum_support
        score = float(
            np.clip(
                config.score_weight * support**config.support_exponent
                + config.score_bias,
                0.0,
                1.0,
            )
        )
        hypotheses.append(
            InstanceHypothesis(
                hypothesis_id=f"aux:e{entity_id}:parent",
                entity_id=entity_id,
                semantic_id=normalized_semantics[entity_id],
                kind="parent",
                vertex_indices=indices,
                score=score,
            )
        )
    return tuple(hypotheses)
