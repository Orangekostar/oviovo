"""Observation identity rules for frontend-first object tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.data_structures import ObjectMap, Patch3D
from src.modules.semantic_memory import object_association_identity_semantic_label, object_semantic_commit_state


@dataclass(frozen=True)
class ObservationIdentityDecision:
    """Decision for whether a patch may update an existing object."""

    can_update: bool
    relation: str
    patch_label: str = ""
    object_label: str = ""
    patch_confidence: float = 0.0
    reason: str = ""


def normalize_label(label: object) -> str:
    """Normalize frontend/object labels for exact identity comparison."""
    return str(label or "").strip().lower()


def patch_anchor_label(patch: Patch3D) -> str:
    """Return the frontend semantic identity assigned to a patch."""
    return normalize_label(patch.metadata.get("anchor_class_name", ""))


def patch_anchor_confidence(patch: Patch3D) -> float:
    """Return patch anchor confidence with malformed metadata treated as zero."""
    try:
        confidence = float(patch.metadata.get("anchor_confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return confidence if math.isfinite(confidence) else 0.0


def object_identity_label(obj: ObjectMap) -> str:
    """Return association-only semantic identity used for association blocking."""
    return normalize_label(object_association_identity_semantic_label(obj))


def patch_is_contained_residual(patch: Patch3D) -> bool:
    """Return True when the frontend intentionally kept a proposal as unknown residual."""
    metadata = patch.metadata
    label_strength = normalize_label(metadata.get("anchor_label_strength", ""))
    residual_policy = normalize_label(metadata.get("residual_semantic_policy", ""))
    relation = normalize_label(metadata.get("mask_anchor_relation", ""))
    return bool(
        metadata.get("semantic_commit_allowed") is False
        and residual_policy == "unknown"
        and (relation == "contained_residual" or label_strength == "none")
    )


def classify_observation_identity(
    patch: Patch3D,
    obj: ObjectMap,
    *,
    min_patch_confidence: float,
) -> ObservationIdentityDecision:
    """Decide if a patch is allowed to update an object identity.

    Confident cross-label observations are not association updates; they are
    contested residual observations that need their own identity track.
    """
    patch_label = patch_anchor_label(patch)
    patch_confidence = patch_anchor_confidence(patch)
    commit_state = object_semantic_commit_state(obj)
    object_label = object_identity_label(obj)

    if patch_is_contained_residual(patch):
        return ObservationIdentityDecision(
            can_update=False,
            relation="contained_residual",
            patch_label=patch_label,
            object_label=object_label,
            patch_confidence=patch_confidence,
            reason="contained_residual_identity_guard",
        )
    if not patch_label:
        return ObservationIdentityDecision(
            can_update=True,
            relation="unlabeled_patch",
            patch_label="",
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    if patch_confidence < float(min_patch_confidence):
        return ObservationIdentityDecision(
            can_update=True,
            relation="low_confidence_patch",
            patch_label=patch_label,
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    if not object_label:
        relation = "uncommitted_object" if str(commit_state.get("posterior_label", "")).strip() else "unlabeled_object"
        return ObservationIdentityDecision(
            can_update=True,
            relation=relation,
            patch_label=patch_label,
            object_label="",
            patch_confidence=patch_confidence,
        )
    if patch_label == object_label:
        return ObservationIdentityDecision(
            can_update=True,
            relation="same_label",
            patch_label=patch_label,
            object_label=object_label,
            patch_confidence=patch_confidence,
        )
    return ObservationIdentityDecision(
        can_update=False,
        relation="cross_label_contested",
        patch_label=patch_label,
        object_label=object_label,
        patch_confidence=patch_confidence,
        reason="cross_label_observation_identity",
    )
