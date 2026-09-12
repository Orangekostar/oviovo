"""CROVE V2 readout roles, independent of instance ownership.

Integer codes are persisted in compact prediction sidecars. Missing observations
use KEEP_SOURCE; PREDICTED_UNKNOWN denotes an explicit model rejection only.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from src.evaluation.baselines.tesse_semantics import TesseSemanticCrosswalk

BRIDGE_VERSION = "CROVE_POINTWISE_READOUT_V1_REVISED_V2"


class LegacySource(IntEnum):
    EXPLICIT_BACKGROUND = 0
    ENTITY_STUFF = 1
    ENTITY_THING = 2
    ENTITY_UNKNOWN = 3


class EvalRole(IntEnum):
    BACKGROUND = 0
    OBJECT = 1
    UNKNOWN_ENTITY = 2


class SemanticUpdateKind(IntEnum):
    KEEP_SOURCE = 0
    PREDICTED_KNOWN = 1
    PREDICTED_UNKNOWN = 2


def _known_ids(crosswalk: TesseSemanticCrosswalk) -> list[int]:
    return sorted({v.semantic_id for v in crosswalk.aliases.values() if v.matched})


def _integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return value


def validate_semantic_roles(
    semantic_ids: np.ndarray, roles: np.ndarray, crosswalk: TesseSemanticCrosswalk
) -> None:
    ids = _integer_vector(semantic_ids, "semantic_ids")
    roles = _integer_vector(roles, "eval_role")
    if roles.shape != ids.shape or not np.isin(roles, [0, 1, 2]).all():
        raise ValueError("eval_role must contain one valid role per point")
    thing = np.isin(ids, sorted(crosswalk.valid_semantic_ids))
    unknown = ids == crosswalk.unknown.semantic_id
    known = np.isin(ids, _known_ids(crosswalk))
    if np.any((roles == EvalRole.OBJECT) & ~thing):
        raise ValueError("OBJECT requires a known thing semantic ID")
    if np.any((roles == EvalRole.UNKNOWN_ENTITY) & ~unknown):
        raise ValueError("UNKNOWN_ENTITY requires the unknown semantic ID")
    if np.any((roles == EvalRole.BACKGROUND) & (~(known | unknown) | thing)):
        raise ValueError("BACKGROUND requires a stuff or explicit-background ID")


def legacy_roles(
    semantic_ids: np.ndarray,
    explicit_background: np.ndarray,
    crosswalk: TesseSemanticCrosswalk,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source provenance from old snapshot membership and crosswalk IDs."""
    ids = _integer_vector(semantic_ids, "semantic_ids")
    explicit = np.asarray(explicit_background)
    if explicit.dtype != np.bool_ or explicit.shape != ids.shape:
        raise ValueError("explicit_background must have one boolean per row")
    known = np.isin(ids, _known_ids(crosswalk))
    thing = np.isin(ids, sorted(crosswalk.valid_semantic_ids))
    sources = np.full(len(ids), LegacySource.ENTITY_UNKNOWN, dtype=np.uint8)
    roles = np.full(len(ids), EvalRole.UNKNOWN_ENTITY, dtype=np.uint8)
    sources[known & ~thing] = LegacySource.ENTITY_STUFF
    roles[known & ~thing] = EvalRole.BACKGROUND
    sources[thing] = LegacySource.ENTITY_THING
    roles[thing] = EvalRole.OBJECT
    sources[explicit] = LegacySource.EXPLICIT_BACKGROUND
    roles[explicit] = EvalRole.BACKGROUND
    validate_semantic_roles(ids, roles, crosswalk)
    return sources, roles


def resolve_semantic_update(
    source_semantic_ids: np.ndarray,
    source_eval_role: np.ndarray,
    proposed_semantic_ids: np.ndarray,
    update_kind: np.ndarray,
    crosswalk: TesseSemanticCrosswalk,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply supported semantics without deriving role from an owner ID."""
    ids = _integer_vector(source_semantic_ids, "source_semantic_ids")
    validate_semantic_roles(ids, source_eval_role, crosswalk)
    proposed = _integer_vector(proposed_semantic_ids, "proposed_semantic_ids")
    kind = _integer_vector(update_kind, "update_kind")
    if proposed.shape != ids.shape or kind.shape != ids.shape:
        raise ValueError("semantic updates must have one value per source row")
    if not np.isin(kind, [0, 1, 2]).all():
        raise ValueError("unknown semantic update kind")
    known = kind == SemanticUpdateKind.PREDICTED_KNOWN
    if not np.isin(proposed[known], _known_ids(crosswalk)).all():
        raise ValueError("PREDICTED_KNOWN requires a known crosswalk class")
    result = ids.copy()
    roles = np.array(source_eval_role, dtype=np.uint8, copy=True)
    result[known] = proposed[known]
    roles[known] = np.where(
        np.isin(proposed[known], sorted(crosswalk.valid_semantic_ids)),
        EvalRole.OBJECT,
        EvalRole.BACKGROUND,
    )
    rejected = kind == SemanticUpdateKind.PREDICTED_UNKNOWN
    result[rejected] = crosswalk.unknown.semantic_id
    roles[rejected] = EvalRole.UNKNOWN_ENTITY
    return result, roles
