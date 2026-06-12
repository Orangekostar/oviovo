"""Module 8: Semantic Memory (Layer 7).

Responsibility: Maintain object-level semantic memory.
Semantics start ONLY after instance stabilization.
No dense per-point language features — only object-level embeddings.
Semantics must not drive low-level patch association (Rule C).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import ObjectMap, ObjectState, Patch3D, SystemState
from src.models.semantic_backend import PlaceholderSemanticBackend, SemanticBackend

logger = logging.getLogger("oviovo.modules.semantic_memory")

ANCHOR_HIGH_CONFIDENCE_THRESHOLD = 0.75
ANCHOR_HIGH_VIEW_QUALITY_THRESHOLD = 0.25
ANCHOR_COMMIT_MIN_FRAME_HITS = 2
ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS = 1
ANCHOR_COMMIT_MIN_WEIGHTED_SCORE = 1.0
ANCHOR_COMMIT_MIN_SCORE_MARGIN = 0.15
ANCHOR_PROVISIONAL_EXPORT_FALLBACK = False
ANCHOR_EXPORT_ENABLED = True
ANCHOR_EXPORT_MIN_FRAME_HITS = 1
ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS = 1
ANCHOR_EXPORT_MIN_WEIGHTED_SCORE = 0.75
ANCHOR_EXPORT_MIN_SCORE_MARGIN = 0.05
ANCHOR_EXPORT_MIN_CONFIDENCE = 0.75
ANCHOR_EXPORT_MIN_VIEW_QUALITY = 0.25
ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE = True
ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE = 0.90
ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY = 0.50
ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED = False
ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS = 3
ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE = 1.0
ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN = 0.25
ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY = 0.50
ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE = 0.25


@dataclass(frozen=True)
class AnchorCommitPolicy:
    min_frame_hits: int = ANCHOR_COMMIT_MIN_FRAME_HITS
    min_high_quality_hits: int = ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS
    min_weighted_score: float = ANCHOR_COMMIT_MIN_WEIGHTED_SCORE
    min_score_margin: float = ANCHOR_COMMIT_MIN_SCORE_MARGIN
    provisional_export_fallback: bool = ANCHOR_PROVISIONAL_EXPORT_FALLBACK


@dataclass(frozen=True)
class AnchorExportPolicy:
    enabled: bool = ANCHOR_EXPORT_ENABLED
    min_frame_hits: int = ANCHOR_EXPORT_MIN_FRAME_HITS
    min_high_quality_hits: int = ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS
    min_weighted_score: float = ANCHOR_EXPORT_MIN_WEIGHTED_SCORE
    min_score_margin: float = ANCHOR_EXPORT_MIN_SCORE_MARGIN
    min_confidence: float = ANCHOR_EXPORT_MIN_CONFIDENCE
    min_view_quality: float = ANCHOR_EXPORT_MIN_VIEW_QUALITY
    allow_single_frame_high_confidence: bool = ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE
    single_frame_min_confidence: float = ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE
    single_frame_min_view_quality: float = ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY
    repeated_evidence_enabled: bool = ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED
    repeated_min_frame_hits: int = ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS
    repeated_min_weighted_score: float = ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE
    repeated_min_score_margin: float = ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN
    repeated_min_view_quality: float = ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY
    repeated_min_confidence: float = ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE


_DEFAULT_ANCHOR_COMMIT_POLICY = AnchorCommitPolicy()
_CURRENT_ANCHOR_COMMIT_POLICY = _DEFAULT_ANCHOR_COMMIT_POLICY
_DEFAULT_ANCHOR_EXPORT_POLICY = AnchorExportPolicy()
_CURRENT_ANCHOR_EXPORT_POLICY = _DEFAULT_ANCHOR_EXPORT_POLICY


def current_anchor_commit_policy() -> AnchorCommitPolicy:
    return _CURRENT_ANCHOR_COMMIT_POLICY


def set_anchor_commit_policy(policy: AnchorCommitPolicy | dict[str, Any] | None) -> None:
    global _CURRENT_ANCHOR_COMMIT_POLICY
    if policy is None:
        _CURRENT_ANCHOR_COMMIT_POLICY = _DEFAULT_ANCHOR_COMMIT_POLICY
        return
    if isinstance(policy, AnchorCommitPolicy):
        _CURRENT_ANCHOR_COMMIT_POLICY = policy
        return
    _CURRENT_ANCHOR_COMMIT_POLICY = AnchorCommitPolicy(
        min_frame_hits=int(policy.get("min_frame_hits", ANCHOR_COMMIT_MIN_FRAME_HITS)),
        min_high_quality_hits=int(policy.get("min_high_quality_hits", ANCHOR_COMMIT_MIN_HIGH_QUALITY_HITS)),
        min_weighted_score=float(policy.get("min_weighted_score", ANCHOR_COMMIT_MIN_WEIGHTED_SCORE)),
        min_score_margin=float(policy.get("min_score_margin", ANCHOR_COMMIT_MIN_SCORE_MARGIN)),
        provisional_export_fallback=bool(policy.get("provisional_export_fallback", ANCHOR_PROVISIONAL_EXPORT_FALLBACK)),
    )


def current_anchor_export_policy() -> AnchorExportPolicy:
    return _CURRENT_ANCHOR_EXPORT_POLICY


def set_anchor_export_policy(policy: AnchorExportPolicy | dict[str, Any] | None) -> None:
    global _CURRENT_ANCHOR_EXPORT_POLICY
    if policy is None:
        _CURRENT_ANCHOR_EXPORT_POLICY = _DEFAULT_ANCHOR_EXPORT_POLICY
        return
    if isinstance(policy, AnchorExportPolicy):
        _CURRENT_ANCHOR_EXPORT_POLICY = policy
        return
    _CURRENT_ANCHOR_EXPORT_POLICY = AnchorExportPolicy(
        enabled=bool(policy.get("enabled", ANCHOR_EXPORT_ENABLED)),
        min_frame_hits=int(policy.get("min_frame_hits", ANCHOR_EXPORT_MIN_FRAME_HITS)),
        min_high_quality_hits=int(policy.get("min_high_quality_hits", ANCHOR_EXPORT_MIN_HIGH_QUALITY_HITS)),
        min_weighted_score=float(policy.get("min_weighted_score", ANCHOR_EXPORT_MIN_WEIGHTED_SCORE)),
        min_score_margin=float(policy.get("min_score_margin", ANCHOR_EXPORT_MIN_SCORE_MARGIN)),
        min_confidence=float(policy.get("min_confidence", ANCHOR_EXPORT_MIN_CONFIDENCE)),
        min_view_quality=float(policy.get("min_view_quality", ANCHOR_EXPORT_MIN_VIEW_QUALITY)),
        allow_single_frame_high_confidence=bool(
            policy.get("allow_single_frame_high_confidence", ANCHOR_EXPORT_ALLOW_SINGLE_FRAME_HIGH_CONFIDENCE)
        ),
        single_frame_min_confidence=float(
            policy.get("single_frame_min_confidence", ANCHOR_EXPORT_SINGLE_FRAME_MIN_CONFIDENCE)
        ),
        single_frame_min_view_quality=float(
            policy.get("single_frame_min_view_quality", ANCHOR_EXPORT_SINGLE_FRAME_MIN_VIEW_QUALITY)
        ),
        repeated_evidence_enabled=bool(
            policy.get("repeated_evidence_enabled", ANCHOR_EXPORT_REPEATED_EVIDENCE_ENABLED)
        ),
        repeated_min_frame_hits=int(policy.get("repeated_min_frame_hits", ANCHOR_EXPORT_REPEATED_MIN_FRAME_HITS)),
        repeated_min_weighted_score=float(
            policy.get("repeated_min_weighted_score", ANCHOR_EXPORT_REPEATED_MIN_WEIGHTED_SCORE)
        ),
        repeated_min_score_margin=float(
            policy.get("repeated_min_score_margin", ANCHOR_EXPORT_REPEATED_MIN_SCORE_MARGIN)
        ),
        repeated_min_view_quality=float(
            policy.get("repeated_min_view_quality", ANCHOR_EXPORT_REPEATED_MIN_VIEW_QUALITY)
        ),
        repeated_min_confidence=float(
            policy.get("repeated_min_confidence", ANCHOR_EXPORT_REPEATED_MIN_CONFIDENCE)
        ),
    )


def preferred_object_semantic_label(obj: ObjectMap) -> str:
    """Return the compatibility identity label for an object."""
    return object_identity_semantic_label(obj)


def object_identity_semantic_label(obj: ObjectMap) -> str:
    """Return the committed identity semantic label, never a provisional anchor."""
    anchor_state = obj.debug.get("anchor_semantics")
    if isinstance(anchor_state, dict):
        commit_state = object_semantic_commit_state(obj)
        committed_label = str(commit_state.get("committed_label", "")).strip()
        if str(commit_state.get("semantic_state", "")).strip() == "committed" and committed_label:
            return committed_label
        return ""

    hypotheses = getattr(obj.semantic_memory, "label_hypotheses", [])
    if not hypotheses:
        return ""
    return str(hypotheses[0][0]).strip()


def object_association_identity_semantic_label(obj: ObjectMap) -> str:
    """Return the semantic identity used only to block cross-class association."""
    committed = object_identity_semantic_label(obj).strip()
    if committed:
        return committed

    anchor_state = obj.debug.get("anchor_semantics")
    if not isinstance(anchor_state, dict):
        return ""

    candidate = str(anchor_state.get("canonical_label", "")).strip()
    if not candidate or not _has_direct_anchor_evidence(anchor_state, candidate):
        return ""

    policy = current_anchor_export_policy()
    if not policy.repeated_evidence_enabled:
        return ""

    stats = _anchor_label_stats(anchor_state, candidate)
    if (
        int(stats["frame_hits"]) >= policy.repeated_min_frame_hits
        and float(stats["weighted_score"]) >= policy.repeated_min_weighted_score
        and float(stats["margin"]) >= policy.repeated_min_score_margin
        and float(stats["best_view_quality"]) >= policy.repeated_min_view_quality
        and float(stats["max_confidence"]) >= policy.repeated_min_confidence
    ):
        return candidate
    return ""


def object_export_semantic_label(obj: ObjectMap) -> str:
    """Return the export-safe semantic label for an object."""
    return str(object_export_semantic_state(obj).get("export_label", "")).strip()


def object_export_semantic_state(obj: ObjectMap) -> dict[str, Any]:
    """Return semantic commit state plus export-safe label metadata."""
    commit_state = object_semantic_commit_state(obj)
    anchor_state = obj.debug.get("anchor_semantics")
    export_state = dict(commit_state)

    if isinstance(anchor_state, dict):
        committed_label = str(commit_state.get("committed_label", "")).strip()
        if str(commit_state.get("semantic_state", "")).strip() == "committed" and committed_label:
            export_state.update(
                {
                    "export_label": committed_label,
                    "export_state": "committed",
                    "export_source": "anchor_committed",
                    "export_reason": "committed_anchor_semantic",
                }
            )
            return export_state

        safe_label, reason = _safe_provisional_export_label(anchor_state)
        if safe_label:
            export_state.update(
                {
                    "export_label": safe_label.strip(),
                    "export_state": "safe_provisional",
                    "export_source": "anchor_safe_provisional",
                    "export_reason": reason,
                }
            )
            return export_state
        export_state.update(
            {
                "export_label": "",
                "export_state": "unlabeled",
                "export_source": "none",
                "export_reason": reason,
            }
        )
        return export_state

    hypotheses = getattr(obj.semantic_memory, "label_hypotheses", [])
    if hypotheses:
        export_state.update(
            {
                "export_label": str(hypotheses[0][0]).strip(),
                "export_state": "semantic_memory",
                "export_source": "semantic_memory_no_anchor",
                "export_reason": "no_anchor_semantics_fallback",
            }
        )
        return export_state

    export_state.update(
        {
            "export_label": "",
            "export_state": "unlabeled",
            "export_source": "none",
            "export_reason": "no_export_semantic_evidence",
        }
    )
    return export_state


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _anchor_label_stats(anchor_state: dict[str, Any], candidate: str) -> dict[str, float | int]:
    weighted_scores = anchor_state.get("label_weighted_score", {})
    frame_hits_by_label = anchor_state.get("label_frame_hits", {})
    high_quality_hits_by_label = anchor_state.get("label_high_quality_hits", {})
    max_confidence_by_label = anchor_state.get("label_max_confidence", {})
    best_view_quality_by_label = anchor_state.get("label_best_view_quality", {})

    weighted_score = _safe_float(weighted_scores.get(candidate, 0.0)) if isinstance(weighted_scores, dict) else 0.0
    frame_hits = _safe_int(frame_hits_by_label.get(candidate, 0)) if isinstance(frame_hits_by_label, dict) else 0
    high_quality_hits = (
        _safe_int(high_quality_hits_by_label.get(candidate, 0)) if isinstance(high_quality_hits_by_label, dict) else 0
    )
    max_confidence = (
        _safe_float(max_confidence_by_label.get(candidate, 0.0)) if isinstance(max_confidence_by_label, dict) else 0.0
    )
    best_view_quality = (
        _safe_float(best_view_quality_by_label.get(candidate, 0.0))
        if isinstance(best_view_quality_by_label, dict)
        else 0.0
    )
    runner_up_score = (
        max((_safe_float(score) for label, score in weighted_scores.items() if str(label) != candidate), default=0.0)
        if isinstance(weighted_scores, dict)
        else 0.0
    )
    margin = _safe_float(weighted_score - runner_up_score)
    return {
        "weighted_score": weighted_score,
        "frame_hits": frame_hits,
        "high_quality_hits": high_quality_hits,
        "max_confidence": max_confidence,
        "best_view_quality": best_view_quality,
        "margin": margin,
    }


def _has_direct_anchor_evidence(anchor_state: dict[str, Any], candidate: str) -> bool:
    evidence = anchor_state.get("evidence", [])
    evidence_items = evidence if isinstance(evidence, list) else []
    return any(
        isinstance(item, dict) and str(item.get("label", "")).strip() == candidate
        for item in evidence_items
    )


def _safe_provisional_export_label(anchor_state: dict[str, Any]) -> tuple[str, str]:
    policy = current_anchor_export_policy()
    if not policy.enabled:
        return "", "export_provisional_disabled"

    candidate = str(anchor_state.get("canonical_label", "")).strip()
    evidence = anchor_state.get("evidence", [])
    evidence_items = evidence if isinstance(evidence, list) else []
    if not candidate:
        if not evidence_items:
            return "", "no_direct_strong_anchor_evidence"
        return "", "no_posterior_label"

    if not _has_direct_anchor_evidence(anchor_state, candidate):
        return "", "no_direct_strong_anchor_evidence"

    stats = _anchor_label_stats(anchor_state, candidate)
    weighted_score = float(stats["weighted_score"])
    frame_hits = int(stats["frame_hits"])
    high_quality_hits = int(stats["high_quality_hits"])
    max_confidence = float(stats["max_confidence"])
    best_view_quality = float(stats["best_view_quality"])
    margin = float(stats["margin"])

    enough_frames = frame_hits >= policy.min_frame_hits
    enough_quality = high_quality_hits >= policy.min_high_quality_hits
    enough_score = weighted_score >= policy.min_weighted_score
    enough_margin = margin >= policy.min_score_margin
    enough_confidence = max_confidence >= policy.min_confidence
    enough_view = best_view_quality >= policy.min_view_quality
    single_frame_override = (
        policy.allow_single_frame_high_confidence
        and frame_hits == 1
        and max_confidence >= policy.single_frame_min_confidence
        and best_view_quality >= policy.single_frame_min_view_quality
        and enough_score
        and enough_margin
    )
    repeated_evidence_override = (
        policy.repeated_evidence_enabled
        and frame_hits >= policy.repeated_min_frame_hits
        and weighted_score >= policy.repeated_min_weighted_score
        and margin >= policy.repeated_min_score_margin
        and best_view_quality >= policy.repeated_min_view_quality
        and max_confidence >= policy.repeated_min_confidence
    )
    reason_fields = (
        f"frames={frame_hits},"
        f"high_quality={high_quality_hits},"
        f"score={weighted_score:.6f},"
        f"margin={margin:.6f},"
        f"confidence={max_confidence:.6f},"
        f"view_quality={best_view_quality:.6f}"
    )

    if (
        (
            (enough_frames or single_frame_override)
            and enough_quality
            and enough_score
            and enough_margin
            and enough_confidence
            and enough_view
        )
        or repeated_evidence_override
    ):
        return candidate, f"safe_provisional_anchor_evidence:{reason_fields}"
    return "", f"waiting_for_export:{reason_fields}"


def object_semantic_commit_state(obj: ObjectMap) -> dict[str, Any]:
    """Return export-facing semantic state for an object."""
    anchor_state = obj.debug.get("anchor_semantics")
    if not isinstance(anchor_state, dict):
        return {
            "semantic_state": "unlabeled",
            "posterior_label": "",
            "committed_label": "",
            "posterior_score": 0.0,
            "commit_reason": "no_anchor_semantics",
        }
    posterior_label = str(anchor_state.get("canonical_label", "")).strip()
    committed_label = str(anchor_state.get("committed_label", "")).strip()
    semantic_state = str(anchor_state.get("semantic_state", "")).strip()
    if not semantic_state:
        semantic_state = "committed" if committed_label else ("provisional" if posterior_label else "unlabeled")
    return {
        "semantic_state": semantic_state,
        "posterior_label": posterior_label,
        "committed_label": committed_label,
        "posterior_score": _safe_float(anchor_state.get("canonical_score", 0.0)),
        "commit_reason": str(anchor_state.get("commit_reason", "")),
    }


def accumulate_anchor_semantic_vote(obj: ObjectMap, patch: Patch3D) -> None:
    """Accumulate an anchor-class vote into the object's semantic state."""
    label = str(patch.metadata.get("anchor_class_name", "")).strip()
    if not label:
        _record_ignored_anchor_observation(obj, "missing_anchor_label")
        return
    confidence = float(patch.metadata.get("anchor_confidence", 0.0))
    frame_id = int(patch.source_frame_id)
    strength = str(patch.metadata.get("anchor_label_strength", "strong")).strip()
    if patch.metadata.get("semantic_commit_allowed", True) is False:
        _record_delayed_anchor_observation(
            obj,
            patch=patch,
            label=label,
            confidence=confidence,
            frame_id=frame_id,
            strength=strength,
        )
        _record_ignored_anchor_observation(obj, "semantic_commit_blocked")
        return
    if strength != "strong":
        _record_contextual_anchor_observation(
            obj,
            label=label,
            confidence=confidence,
            frame_id=frame_id,
            strength=strength,
        )
        _record_ignored_anchor_observation(obj, "non_strong_anchor_label")
        return
    _accumulate_anchor_vote(
        obj,
        label=label,
        confidence=confidence,
        frame_id=frame_id,
        view_quality=_patch_anchor_view_quality(patch),
    )


def rebuild_anchor_semantic_votes_from_observations(obj: ObjectMap) -> None:
    """Rebuild anchor-vote semantics from stored object observations."""
    obj.debug.pop("anchor_semantics", None)
    for observation in obj.observations:
        accumulate_anchor_semantic_vote(obj, observation.patch)


def _record_ignored_anchor_observation(obj: ObjectMap, reason: str) -> None:
    state = _anchor_semantic_state(obj)
    state["ignored_observation_count"] = int(state.get("ignored_observation_count", 0)) + 1
    reasons = state["ignored_observation_reasons"]
    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1


def _record_contextual_anchor_observation(
    obj: ObjectMap,
    *,
    label: str,
    confidence: float,
    frame_id: int,
    strength: str,
) -> None:
    state = _anchor_semantic_state(obj)
    contextual_score = state["contextual_label_score_sum"]
    contextual_score[str(label)] = float(contextual_score.get(str(label), 0.0) + float(confidence))
    state["contextual_evidence"].append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "anchor_label_strength": str(strength),
            "commit_eligible": False,
        }
    )


def _record_delayed_anchor_observation(
    obj: ObjectMap,
    *,
    patch: Patch3D,
    label: str,
    confidence: float,
    frame_id: int,
    strength: str,
) -> None:
    state = _anchor_semantic_state(obj)
    state["delayed_evidence"].append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "view_quality": float(_patch_anchor_view_quality(patch)),
            "anchor_label_strength": str(strength),
            "semantic_commit_allowed": False,
            "residual_semantic_policy": str(patch.metadata.get("residual_semantic_policy", "")),
            "mask_anchor_relation": str(patch.metadata.get("mask_anchor_relation", "")),
            "reason": str(
                patch.metadata.get(
                    "semantic_commit_blocked_reason",
                    patch.metadata.get("contested_reason", ""),
                )
            ),
            "commit_eligible": False,
        }
    )


def _accumulate_anchor_vote(
    obj: ObjectMap,
    *,
    label: str,
    confidence: float,
    frame_id: int,
    view_quality: float,
) -> None:
    state = _anchor_semantic_state(obj)
    previous_label = str(state.get("canonical_label", ""))

    vote_weight = float(confidence) if confidence > 0.0 else 1.0
    view_weight = float(np.clip(view_quality, 0.0, 1.0))
    weighted_vote = vote_weight * (0.25 + 0.75 * view_weight)
    state["evidence"].append(
        {
            "label": str(label),
            "confidence": float(confidence),
            "frame_id": int(frame_id),
            "view_quality": float(view_weight),
            "vote_weight": float(vote_weight),
            "weighted_vote": float(weighted_vote),
        }
    )
    _recompute_anchor_semantic_state(state)
    best_label = str(state.get("canonical_label", ""))
    if previous_label and best_label and previous_label != best_label:
        state.setdefault("posterior_relabel_events", []).append(
            {
                "frame_id": int(frame_id),
                "old_label": previous_label,
                "new_label": best_label,
                "new_confidence": float(confidence),
                "new_view_quality": float(view_weight),
                "reason": "stronger_anchor_evidence_ledger",
            }
        )


def _recompute_anchor_semantic_state(state: dict[str, Any]) -> None:
    previous_committed_label = str(state.get("committed_label", "")).strip()
    score_sum: dict[str, float] = {}
    weighted_score: dict[str, float] = {}
    max_confidence: dict[str, float] = {}
    best_view_quality: dict[str, float] = {}
    seen_frame_sets: dict[str, set[int]] = {}
    high_quality_hits: dict[str, int] = {}

    for item in state.get("evidence", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        confidence = float(item.get("confidence", 0.0))
        frame_id = int(item.get("frame_id", 0))
        view_weight = float(np.clip(float(item.get("view_quality", 0.0)), 0.0, 1.0))
        vote_weight = float(item.get("vote_weight", confidence if confidence > 0.0 else 1.0))
        weighted_vote = float(
            item.get("weighted_vote", vote_weight * (0.25 + 0.75 * view_weight))
        )

        score_sum[label] = float(score_sum.get(label, 0.0) + vote_weight)
        weighted_score[label] = float(weighted_score.get(label, 0.0) + weighted_vote)
        max_confidence[label] = max(float(max_confidence.get(label, 0.0)), confidence)
        best_view_quality[label] = max(float(best_view_quality.get(label, 0.0)), view_weight)
        seen_frame_sets.setdefault(label, set()).add(frame_id)
        if (
            confidence >= ANCHOR_HIGH_CONFIDENCE_THRESHOLD
            and view_weight >= ANCHOR_HIGH_VIEW_QUALITY_THRESHOLD
        ):
            high_quality_hits[label] = int(high_quality_hits.get(label, 0) + 1)

    seen_frames = {
        label: sorted(int(frame_id) for frame_id in frames)
        for label, frames in seen_frame_sets.items()
    }
    frame_hits = {label: int(len(frames)) for label, frames in seen_frames.items()}

    best_label = ""
    best_key: tuple[int, int, float, float, float, str] | None = None
    for candidate_label in sorted(score_sum):
        candidate_key = (
            int(high_quality_hits.get(candidate_label, 0)),
            int(frame_hits.get(candidate_label, 0)),
            float(weighted_score.get(candidate_label, 0.0)),
            float(max_confidence.get(candidate_label, 0.0)),
            float(best_view_quality.get(candidate_label, 0.0)),
            candidate_label,
        )
        if best_key is None or candidate_key > best_key:
            best_label = candidate_label
            best_key = candidate_key

    state["label_score_sum"] = score_sum
    state["label_weighted_score"] = weighted_score
    state["label_recent_score"] = dict(weighted_score)
    state["label_max_confidence"] = max_confidence
    state["label_best_view_quality"] = best_view_quality
    state["label_seen_frames"] = seen_frames
    state["label_frame_hits"] = frame_hits
    state["label_high_quality_hits"] = high_quality_hits
    state["canonical_label"] = best_label
    state["canonical_score"] = float(weighted_score.get(best_label, 0.0)) if best_label else 0.0
    state["canonical_frame_hits"] = int(frame_hits.get(best_label, 0)) if best_label else 0
    state["canonical_confidence"] = float(max_confidence.get(best_label, 0.0)) if best_label else 0.0
    state["canonical_best_view_quality"] = float(best_view_quality.get(best_label, 0.0)) if best_label else 0.0
    state["source"] = "anchor_vote" if best_label else ""

    if not best_label:
        state["semantic_state"] = "committed" if previous_committed_label else "unlabeled"
        state["committed_label"] = previous_committed_label
        state["commit_reason"] = "no_valid_evidence"
        return

    runner_up_score = max(
        (float(score) for label, score in weighted_score.items() if label != best_label),
        default=0.0,
    )
    best_weighted_score = float(weighted_score.get(best_label, 0.0))
    score_margin = float(best_weighted_score - runner_up_score)
    best_frame_hits = int(frame_hits.get(best_label, 0))
    best_high_quality_hits = int(high_quality_hits.get(best_label, 0))
    policy = current_anchor_commit_policy()

    enough_frames = best_frame_hits >= policy.min_frame_hits
    enough_quality = best_high_quality_hits >= policy.min_high_quality_hits
    enough_score = best_weighted_score >= policy.min_weighted_score
    enough_margin = score_margin >= policy.min_score_margin

    if enough_frames and enough_quality and enough_score and enough_margin:
        state["semantic_state"] = "committed"
        state["committed_label"] = best_label
        state["commit_reason"] = "multiframe_strong_anchor_evidence"
        if previous_committed_label and previous_committed_label != best_label:
            state.setdefault("relabel_events", []).append(
                {
                    "frame_id": int(max(seen_frames.get(best_label, [0]))),
                    "old_label": previous_committed_label,
                    "new_label": best_label,
                    "reason": "committed_label_changed_by_stronger_posterior",
                }
            )
        return

    waiting_reason = (
        "waiting_for_commit:"
        f"frames={best_frame_hits},"
        f"high_quality={best_high_quality_hits},"
        f"score={best_weighted_score:.6f},"
        f"margin={score_margin:.6f}"
    )
    if previous_committed_label and previous_committed_label == best_label and enough_margin:
        state["semantic_state"] = "committed"
        state["committed_label"] = previous_committed_label
        state["commit_reason"] = f"pending_relabel_{waiting_reason}"
        return

    state["semantic_state"] = "provisional"
    state["committed_label"] = ""
    state["commit_reason"] = waiting_reason


def _anchor_semantic_state(obj: ObjectMap) -> dict[str, Any]:
    state = obj.debug.get("anchor_semantics")
    if not isinstance(state, dict):
        state = {}
        obj.debug["anchor_semantics"] = state

    if not isinstance(state.get("label_score_sum"), dict):
        state["label_score_sum"] = {}
    if not isinstance(state.get("label_weighted_score"), dict):
        state["label_weighted_score"] = {}
    if not isinstance(state.get("label_recent_score"), dict):
        state["label_recent_score"] = {}
    if not isinstance(state.get("label_max_confidence"), dict):
        state["label_max_confidence"] = {}
    if not isinstance(state.get("label_best_view_quality"), dict):
        state["label_best_view_quality"] = {}
    if not isinstance(state.get("label_seen_frames"), dict):
        state["label_seen_frames"] = {}
    if not isinstance(state.get("label_frame_hits"), dict):
        state["label_frame_hits"] = {}
    if not isinstance(state.get("label_high_quality_hits"), dict):
        state["label_high_quality_hits"] = {}
    if not isinstance(state.get("evidence"), list):
        state["evidence"] = []
    if not isinstance(state.get("contextual_label_score_sum"), dict):
        state["contextual_label_score_sum"] = {}
    if not isinstance(state.get("contextual_evidence"), list):
        state["contextual_evidence"] = []
    if not isinstance(state.get("delayed_evidence"), list):
        state["delayed_evidence"] = []
    if not isinstance(state.get("relabel_events"), list):
        state["relabel_events"] = []
    if not isinstance(state.get("posterior_relabel_events"), list):
        state["posterior_relabel_events"] = []
    if not isinstance(state.get("ignored_observation_reasons"), dict):
        state["ignored_observation_reasons"] = {}
    state.setdefault("ignored_observation_count", 0)
    state.setdefault("canonical_label", "")
    state.setdefault("canonical_score", 0.0)
    state.setdefault("canonical_frame_hits", 0)
    state.setdefault("canonical_confidence", 0.0)
    state.setdefault("canonical_best_view_quality", 0.0)
    state.setdefault("source", "")
    return state


def _patch_anchor_view_quality(patch: Patch3D) -> float:
    if "anchor_view_quality" in patch.metadata:
        return float(np.clip(float(patch.metadata.get("anchor_view_quality", 0.0)), 0.0, 1.0))
    lifted_points = int(patch.metadata.get("lifted_point_count", len(patch.points)))
    point_score = min(1.0, lifted_points / 512.0)
    bbox = patch.metadata.get("source_bbox_xyxy")
    bbox_score = 0.0
    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
        area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        bbox_score = min(1.0, area / (160.0 * 160.0))
    protected_bonus = 0.15 if bool(patch.metadata.get("protected_small_anchor", False)) else 0.0
    return float(np.clip(0.65 * point_score + 0.35 * bbox_score + protected_bonus, 0.0, 1.0))


class SemanticMemoryModule:
    """Object-level semantic memory with swappable VLM backend.

    Semantic update happens ONLY after instance stabilization:
      - obj.update_count >= min_stable_observations
      - obj.state == ACTIVE

    Uses instance-level feature aggregation and text matching.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.backend: SemanticBackend = self._create_backend(config)
        self.backend.initialize(config)
        self.top_k = config.get("top_k_labels", 5)
        self.confidence_decay = config.get("confidence_decay", 0.95)
        self.min_obs = config.get("min_observations_for_label", 3)
        self.min_stable_observations = config.get("min_stable_observations", 3)
        self.min_stability_score = config.get("min_stability_score", 0.2)
        anchor_commit_cfg = config.get("anchor_commit", {})
        set_anchor_commit_policy(anchor_commit_cfg)
        anchor_commit_policy = current_anchor_commit_policy()
        self.anchor_commit_min_frame_hits = anchor_commit_policy.min_frame_hits
        self.anchor_commit_min_high_quality_hits = anchor_commit_policy.min_high_quality_hits
        self.anchor_commit_min_weighted_score = anchor_commit_policy.min_weighted_score
        self.anchor_commit_min_score_margin = anchor_commit_policy.min_score_margin
        self.anchor_provisional_export_fallback = anchor_commit_policy.provisional_export_fallback
        anchor_export_cfg = config.get("anchor_export", {})
        set_anchor_export_policy(anchor_export_cfg)
        anchor_export_policy = current_anchor_export_policy()
        self.anchor_export_enabled = anchor_export_policy.enabled
        self.anchor_export_min_frame_hits = anchor_export_policy.min_frame_hits
        self.anchor_export_min_high_quality_hits = anchor_export_policy.min_high_quality_hits
        self.anchor_export_min_weighted_score = anchor_export_policy.min_weighted_score
        self.anchor_export_min_score_margin = anchor_export_policy.min_score_margin
        self.anchor_export_min_confidence = anchor_export_policy.min_confidence
        self.anchor_export_min_view_quality = anchor_export_policy.min_view_quality
        self.anchor_export_allow_single_frame_high_confidence = anchor_export_policy.allow_single_frame_high_confidence
        self.anchor_export_single_frame_min_confidence = anchor_export_policy.single_frame_min_confidence
        self.anchor_export_single_frame_min_view_quality = anchor_export_policy.single_frame_min_view_quality
        self.label_candidates = list(config.get(
            "label_candidates",
            ["chair", "table", "cabinet", "sofa", "bed", "plant", "monitor", "lamp"],
        ))
        self.text_features = self._build_text_feature_bank(self.label_candidates)
        self.last_updated_object_ids: List[int] = []
        logger.info(f"SemanticMemoryModule initialized with backend: {config.get('backend', 'placeholder')}")

    def _create_backend(self, config: Dict[str, Any]) -> SemanticBackend:
        """Factory for semantic backends."""
        backend_name = config.get("backend", "placeholder")
        if backend_name == "placeholder":
            return PlaceholderSemanticBackend()
        logger.warning(f"Unknown backend '{backend_name}', falling back to placeholder.")
        return PlaceholderSemanticBackend()

    def _build_text_feature_bank(self, labels: List[str]) -> np.ndarray:
        if not labels:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self.backend.encode_text(label) for label in labels], axis=0)

    def process(
        self,
        state: SystemState,
        rgb: np.ndarray,
        updated_object_ids: List[int],
    ) -> SystemState:
        """Update semantic memory for recently associated objects.

        Gated on instance stability: only update objects that are
        sufficiently stable (enough observations, ACTIVE state).

        Args:
            state: Current system state.
            rgb: (H, W, 3) current frame RGB.
            updated_object_ids: IDs of objects that were updated this frame.

        Returns:
            Updated system state.
        """
        self.last_updated_object_ids = []
        for obj_id in updated_object_ids:
            obj = state.objects.get(obj_id)
            if obj is None:
                continue

            stability_score = float(
                obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
            )

            # Gate: semantics only after instance stabilization
            if obj.state != ObjectState.ACTIVE:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "object_not_active",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                continue
            if obj.update_count < self.min_stable_observations:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "insufficient_observations",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                logger.debug(
                    f"Object {obj_id}: skipping semantic update "
                    f"(update_count={obj.update_count} < {self.min_stable_observations})"
                )
                continue
            if stability_score < self.min_stability_score:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "insufficient_tsdf_stability",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                continue

            self._update_semantic(obj, rgb)
            self.last_updated_object_ids.append(obj_id)

        return state

    def _update_semantic(self, obj: ObjectMap, rgb: np.ndarray) -> None:
        """Update semantic memory for a single stabilized object.

        Uses object-centric view selection and instance-level feature aggregation.
        """
        mem = obj.semantic_memory

        crop, bbox_xyxy, view_selection_score = self._select_object_view(obj, rgb)
        feature = self.backend.encode_image(crop)

        # Add to feature bank
        mem.feature_bank.append(feature)
        mem.observation_count += 1

        feature_weights = list(mem.debug.get("feature_weights", []))
        feature_weights.append(max(0.05, float(view_selection_score)))
        if len(feature_weights) > len(mem.feature_bank):
            feature_weights = feature_weights[-len(mem.feature_bank):]

        # Instance-level feature aggregation (view-quality + recency weighted mean).
        feature_stack = np.asarray(mem.feature_bank, dtype=np.float32)
        weights = np.asarray(feature_weights, dtype=np.float32)
        recency = np.asarray(
            [self.confidence_decay ** age for age in range(len(feature_stack) - 1, -1, -1)],
            dtype=np.float32,
        )
        effective_weights = np.maximum(weights * recency, 1e-6)
        mem.aggregated_feature = np.average(feature_stack, axis=0, weights=effective_weights)
        norm = np.linalg.norm(mem.aggregated_feature)
        if norm > 1e-8:
            mem.aggregated_feature /= norm

        similarities = self._match_candidate_labels(mem.aggregated_feature)
        ranked_indices = np.argsort(similarities)[::-1]
        top_indices = ranked_indices[: self.top_k]
        if mem.observation_count >= self.min_obs:
            mem.label_hypotheses = [
                (self.label_candidates[idx], float(similarities[idx]))
                for idx in top_indices
            ]

        mem.confidence_history.append(float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0)
        mem.debug = {
            "stability_gate": {
                "passed": True,
                "reason": "stable_instance",
                "update_count": int(obj.update_count),
                "stability_score": float(
                    obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
                ),
            },
            "selected_bbox_xyxy": bbox_xyxy,
            "view_selection_score": float(view_selection_score),
            "feature_weights": [float(value) for value in feature_weights],
            "effective_feature_weights": [float(value) for value in effective_weights.tolist()],
            "candidate_labels": list(self.label_candidates),
            "similarity_scores": [float(score) for score in similarities.tolist()],
        }

        logger.debug(
            f"Object {obj.object_id}: semantic updated, "
            f"{mem.observation_count} observations, "
            f"{len(mem.feature_bank)} features in bank."
        )

    def _select_object_view(self, obj: ObjectMap, rgb: np.ndarray) -> tuple[np.ndarray, List[int], float]:
        """Choose an object-centric crop from recent observations."""
        best_bbox: np.ndarray | None = None
        best_score = -1.0
        h, w = rgb.shape[:2]

        for observation in obj.observations[-5:]:
            bbox = observation.crop_bbox
            if bbox is None:
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            x1 = int(np.clip(np.floor(bbox[0]), 0, max(w - 1, 0)))
            y1 = int(np.clip(np.floor(bbox[1]), 0, max(h - 1, 0)))
            x2 = int(np.clip(np.ceil(bbox[2]), x1 + 1, w))
            y2 = int(np.clip(np.ceil(bbox[3]), y1 + 1, h))
            area = max((x2 - x1) * (y2 - y1), 1)
            area_score = min(1.0, area / max(h * w * 0.25, 1))
            geometry_score = min(1.0, len(observation.patch.points) / 128.0)
            score = 0.6 * area_score + 0.4 * geometry_score
            if score > best_score:
                best_score = score
                best_bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

        if best_bbox is None:
            side = min(h, w, 32)
            x1 = max((w - side) // 2, 0)
            y1 = max((h - side) // 2, 0)
            best_bbox = np.array([x1, y1, x1 + side, y1 + side], dtype=np.int32)
            best_score = 0.0

        x1, y1, x2, y2 = best_bbox.tolist()
        crop = rgb[y1:y2, x1:x2]
        return crop, [int(x1), int(y1), int(x2), int(y2)], float(best_score)

    def _match_candidate_labels(self, feature: np.ndarray) -> np.ndarray:
        """Explicit instance-level text matching."""
        if self.text_features.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.backend.compute_similarity(feature, self.text_features).astype(np.float32)
