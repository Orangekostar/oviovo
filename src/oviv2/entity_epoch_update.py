"""Entity identity and location-episode updates over source OVI surface rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.oviv2.current_surface import CurrentEvidenceState
from src.oviv2.fine_dynamic_policy import (
    PriorSurfaceState,
    SurfaceRetirementReason,
)
from src.oviv2.fine_surface_validity import FineSurfaceEvidence
from src.oviv2.two_visit_registration import validate_rigid_transform


def _immutable(values: object, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EntityEpochUpdateConfig:
    minimum_absent_observations: int = 2
    minimum_distinct_absent_viewpoints: int = 2
    minimum_positive_observations: int = 1
    evidence_is_new_measurement: bool = False

    def __post_init__(self) -> None:
        for name in (
            "minimum_absent_observations",
            "minimum_distinct_absent_viewpoints",
            "minimum_positive_observations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.evidence_is_new_measurement) is not bool:
            raise TypeError("evidence_is_new_measurement must be boolean")


class SurfaceActionReason(str, Enum):
    T1_REPLACED = "t1_replaced"
    RELIABLE_VISIBLE_FREE = "reliable_visible_free"
    LOCAL_POSITIVE_CORRECTION = "local_positive_correction"
    MOVED_LOCATION_RETIRED = "moved_location_retired"


class RelationInferenceState(str, Enum):
    STATIC = "static"
    MOVED = "moved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class RelationSupport:
    relation_id: str
    relation_source: str
    stable_entity_id: str
    t0_owner_entity_id: int
    t1_owner_entity_id: int
    t0_source_surface_id: str
    t1_source_surface_id: str
    t0_source_vertex_indices: np.ndarray
    t1_source_vertex_indices: np.ndarray
    confidence: float
    inference_state: RelationInferenceState
    accepted: bool
    motion_verified: bool = False
    transform_world_from_t0: np.ndarray | None = None
    t0_evidence_frame_ids: tuple[int, ...] = ()
    t1_evidence_frame_ids: tuple[int, ...] = ()
    t0_entity_id: str | None = None
    t1_entity_id: str | None = None
    assignment_is_null: bool = False
    assignment_margin: float | None = None
    centered_shape_score: float | None = None
    size_score: float | None = None
    extent_score: float | None = None
    registration_score: float | None = None
    visible_support_score: float | None = None
    appearance_cosine: float | None = None
    appearance_feature_space: str | None = None
    semantic_compatibility: float | None = None
    original_location_iou: float | None = None
    t0_mask_coverage: float | None = None
    t1_mask_coverage: float | None = None
    t0_mask_purity: float | None = None
    t1_mask_purity: float | None = None
    competition_margin: float | None = None
    spatial_support_patch_count: int | None = None
    residual_improvement_m: float | None = None
    motion_rejection_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "relation_id",
            "relation_source",
            "stable_entity_id",
            "t0_source_surface_id",
            "t1_source_surface_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("t0_owner_entity_id", "t1_owner_entity_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "t0_source_vertex_indices",
            "t1_source_vertex_indices",
        ):
            values = _immutable(getattr(self, name), np.int64)
            if (
                values.ndim != 1
                or np.any(values < 0)
                or (len(values) > 1 and np.any(values[1:] <= values[:-1]))
            ):
                raise ValueError(f"{name} must be ordered and unique")
            object.__setattr__(self, name, values)
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.inference_state, RelationInferenceState):
            raise TypeError("inference_state must be RelationInferenceState")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be boolean")
        if type(self.motion_verified) is not bool:
            raise TypeError("motion_verified must be boolean")
        if type(self.assignment_is_null) is not bool:
            raise TypeError("assignment_is_null must be boolean")
        if self.accepted and self.assignment_is_null:
            raise ValueError("accepted relation cannot be a null assignment")
        if self.accepted and (
            not len(self.t0_source_vertex_indices)
            or not len(self.t1_source_vertex_indices)
        ):
            raise ValueError("accepted relation requires local support at both visits")
        transform = self.transform_world_from_t0
        if transform is not None:
            transform = validate_rigid_transform(transform)
            object.__setattr__(self, "transform_world_from_t0", transform)
        if (
            self.accepted
            and self.inference_state is RelationInferenceState.STATIC
            and transform is None
        ):
            raise ValueError("accepted static relation requires a rigid transform")
        if self.motion_verified and (
            not self.accepted
            or self.inference_state is not RelationInferenceState.MOVED
            or transform is None
        ):
            raise ValueError("verified motion requires an accepted moved transform")
        for name in ("t0_evidence_frame_ids", "t1_evidence_frame_ids"):
            values = tuple(getattr(self, name))
            if (
                any(type(value) is not int or value < 0 for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(f"{name} must be ordered unique nonnegative integers")
            object.__setattr__(self, name, values)
        for name in ("t0_entity_id", "t1_entity_id", "appearance_feature_space"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when present")
            if value is not None:
                object.__setattr__(self, name, value.strip())
        for name in (
            "centered_shape_score",
            "size_score",
            "extent_score",
            "registration_score",
            "visible_support_score",
            "semantic_compatibility",
            "original_location_iou",
            "t0_mask_coverage",
            "t1_mask_coverage",
            "t0_mask_purity",
            "t1_mask_purity",
        ):
            value = getattr(self, name)
            if value is not None:
                number = float(value)
                if not np.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise ValueError(f"{name} must be finite and in [0, 1]")
                object.__setattr__(self, name, number)
        if self.appearance_cosine is not None:
            cosine = float(self.appearance_cosine)
            if not np.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
                raise ValueError("appearance_cosine must be finite and in [-1, 1]")
            object.__setattr__(self, "appearance_cosine", cosine)
        for name in (
            "assignment_margin",
            "competition_margin",
            "residual_improvement_m",
        ):
            value = getattr(self, name)
            if value is not None:
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(f"{name} must be finite when present")
                object.__setattr__(self, name, number)
        if self.spatial_support_patch_count is not None and (
            type(self.spatial_support_patch_count) is not int
            or self.spatial_support_patch_count < 0
        ):
            raise ValueError("spatial_support_patch_count must be nonnegative")
        for name in ("motion_rejection_reasons", "rejection_reasons"):
            reasons = tuple(getattr(self, name))
            if any(
                not isinstance(reason, str) or not reason.strip() for reason in reasons
            ):
                raise ValueError(f"{name} must contain non-empty strings")
            if tuple(sorted(set(reasons))) != reasons:
                raise ValueError(f"{name} must be ordered and unique")
            object.__setattr__(self, name, reasons)
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted relation cannot contain rejection reasons")


@dataclass(frozen=True, slots=True)
class IdentityAliasRecord:
    visit_id: int
    owner_entity_id: int
    stable_entity_id: str
    relation_id: str | None


class EpisodeLifecycle(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    LOCATION_RETIRED = "location_retired"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class EntityEpisodeRecord:
    stable_entity_id: str
    episode_id: str
    source_visit: int
    source_surface_id: str
    owner_entity_id: int
    source_vertex_indices: np.ndarray
    first_observation_frame: int
    last_observation_frame: int
    lifecycle_state: EpisodeLifecycle
    relation_id: str | None
    relation_confidence: float | None
    transform_world_from_t0: np.ndarray | None
    retirement_reasons: tuple[SurfaceRetirementReason, ...]
    revival_reasons: tuple[SurfaceActionReason, ...]

    def __post_init__(self) -> None:
        vertices = _immutable(self.source_vertex_indices, np.int64)
        if (
            vertices.ndim != 1
            or np.any(vertices < 0)
            or (len(vertices) > 1 and np.any(vertices[1:] <= vertices[:-1]))
        ):
            raise ValueError("episode source vertex indices must be ordered and unique")
        object.__setattr__(self, "source_vertex_indices", vertices)
        transform = self.transform_world_from_t0
        if transform is not None:
            transform = _immutable(transform, np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("episode transform must be a finite 4x4 matrix")
            object.__setattr__(self, "transform_world_from_t0", transform)


@dataclass(frozen=True, slots=True)
class SurfaceStateDelta:
    source_surface_id: str
    source_vertex_indices: np.ndarray
    owner_entity_id: int
    stable_entity_id_before: str | None
    stable_entity_id_after: str | None
    episode_id_before: str | None
    episode_id_after: str | None
    current_valid_before: bool
    current_valid_after: bool
    retirement_reason_before: SurfaceRetirementReason
    retirement_reason_after: SurfaceRetirementReason
    action_reason: SurfaceActionReason
    evidence_frame_ids: tuple[int, ...]
    used_new_measurement: bool
    relation_id: str | None = None

    def __post_init__(self) -> None:
        indices = _immutable(self.source_vertex_indices, np.int64)
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or (len(indices) > 1 and np.any(indices[1:] <= indices[:-1]))
        ):
            raise ValueError("delta source vertex indices must be ordered and unique")
        object.__setattr__(self, "source_vertex_indices", indices)


@dataclass(frozen=True, slots=True)
class EntityEpochUpdateResult:
    current_valid: np.ndarray
    retirement_reason_codes: np.ndarray
    evidence_state_codes: np.ndarray
    surface_deltas: tuple[SurfaceStateDelta, ...] = ()
    identity_aliases: tuple[IdentityAliasRecord, ...] = ()
    episode_records: tuple[EntityEpisodeRecord, ...] = ()

    def __post_init__(self) -> None:
        current = _immutable(self.current_valid, np.bool_)
        reasons = _immutable(self.retirement_reason_codes, np.uint8)
        states = _immutable(self.evidence_state_codes, np.uint8)
        if (
            current.ndim != 1
            or reasons.shape != current.shape
            or states.shape != current.shape
        ):
            raise ValueError(
                "update result arrays must have equal one-dimensional shape"
            )
        allowed_reasons = {int(value) for value in SurfaceRetirementReason}
        allowed_states = {int(value) for value in CurrentEvidenceState}
        if not {int(value) for value in np.unique(reasons)} <= allowed_reasons:
            raise ValueError("update result contains an unknown retirement reason")
        if not {int(value) for value in np.unique(states)} <= allowed_states:
            raise ValueError("update result contains an unknown evidence state")
        if np.any(current != (reasons == int(SurfaceRetirementReason.NONE))):
            raise ValueError("update validity and retirement reasons disagree")
        object.__setattr__(self, "current_valid", current)
        object.__setattr__(
            self,
            "retirement_reason_codes",
            reasons,
        )
        object.__setattr__(
            self,
            "evidence_state_codes",
            states,
        )


def _evidence_frames(
    evidence: FineSurfaceEvidence, rows: np.ndarray
) -> tuple[int, ...]:
    values = np.concatenate(
        (
            evidence.last_supported_frames[rows],
            evidence.last_absent_frames[rows],
            evidence.last_occluded_frames[rows],
        )
    )
    return tuple(int(value) for value in np.unique(values[values >= 0]))


def _surface_deltas(
    *,
    source_surface_id: str,
    source_vertex_indices: np.ndarray,
    owner_ids: np.ndarray,
    prior: PriorSurfaceState,
    evidence: FineSurfaceEvidence,
    current: np.ndarray,
    reasons: np.ndarray,
    action_rows: dict[SurfaceActionReason, np.ndarray],
    relation_ids_by_row: dict[int, str],
    aliases: dict[tuple[int, int], IdentityAliasRecord],
    used_new_measurement: bool,
) -> tuple[SurfaceStateDelta, ...]:
    deltas: list[SurfaceStateDelta] = []
    for action in sorted(action_rows, key=lambda value: value.value):
        rows = action_rows[action]
        group_keys = sorted(
            {
                (
                    int(owner_ids[row]),
                    bool(prior.current_valid[row]),
                    int(prior.retirement_reason_codes[row]),
                    relation_ids_by_row.get(int(row)),
                )
                for row in rows
            },
            key=lambda value: (*value[:3], value[3] or ""),
        )
        for owner_id, before_valid, before_reason, relation_id in group_keys:
            stable_id_before = None if owner_id == 0 else f"ovi-t0:{owner_id}"
            alias = aliases.get((0, owner_id))
            stable_id_after = (
                stable_id_before if alias is None else alias.stable_entity_id
            )
            episode_id_before = (
                None
                if stable_id_before is None
                else f"{stable_id_before}/location:t0:{owner_id}"
            )
            episode_id_after = (
                None
                if stable_id_after is None
                else f"{stable_id_after}/location:t0:{owner_id}"
            )
            selected = np.asarray(
                [
                    row
                    for row in rows
                    if int(owner_ids[row]) == owner_id
                    and bool(prior.current_valid[row]) == before_valid
                    and int(prior.retirement_reason_codes[row]) == before_reason
                    and relation_ids_by_row.get(int(row)) == relation_id
                ],
                dtype=np.int64,
            )
            deltas.append(
                SurfaceStateDelta(
                    source_surface_id=source_surface_id,
                    source_vertex_indices=source_vertex_indices[selected],
                    owner_entity_id=owner_id,
                    stable_entity_id_before=stable_id_before,
                    stable_entity_id_after=stable_id_after,
                    episode_id_before=episode_id_before,
                    episode_id_after=episode_id_after,
                    current_valid_before=before_valid,
                    current_valid_after=bool(current[selected[0]]),
                    retirement_reason_before=SurfaceRetirementReason(before_reason),
                    retirement_reason_after=SurfaceRetirementReason(
                        int(reasons[selected[0]])
                    ),
                    action_reason=action,
                    evidence_frame_ids=_evidence_frames(evidence, selected),
                    used_new_measurement=used_new_measurement,
                    relation_id=relation_id,
                )
            )
    return tuple(deltas)


def _observation_range(
    values: tuple[int, ...], fallback: np.ndarray
) -> tuple[int, int]:
    observed = values + tuple(int(value) for value in fallback if value >= 0)
    return (min(observed), max(observed)) if observed else (-1, -1)


def _episode_records(
    *,
    source_surface_id: str,
    source_vertex_indices: np.ndarray,
    owner_ids: np.ndarray,
    current: np.ndarray,
    reasons: np.ndarray,
    evidence: FineSurfaceEvidence,
    aliases: dict[tuple[int, int], IdentityAliasRecord],
    relations: tuple[RelationSupport, ...],
    action_rows: dict[SurfaceActionReason, np.ndarray],
) -> tuple[EntityEpisodeRecord, ...]:
    accepted_by_t0 = {
        relation.t0_owner_entity_id: relation
        for relation in relations
        if relation.accepted
    }
    records: list[EntityEpisodeRecord] = []
    t0_records: dict[int, EntityEpisodeRecord] = {}
    recovered = action_rows.get(
        SurfaceActionReason.LOCAL_POSITIVE_CORRECTION,
        np.empty(0, dtype=np.int64),
    )
    for owner_id in sorted(int(value) for value in np.unique(owner_ids) if value > 0):
        rows = np.flatnonzero(owner_ids == owner_id)
        relation = accepted_by_t0.get(owner_id)
        stable_id = aliases[(0, owner_id)].stable_entity_id
        if not np.any(current[rows]):
            lifecycle = EpisodeLifecycle.LOCATION_RETIRED
        elif (
            relation is not None
            and relation.inference_state is RelationInferenceState.MOVED
        ):
            lifecycle = EpisodeLifecycle.UNCERTAIN
        elif (
            relation is not None
            and relation.inference_state is RelationInferenceState.STATIC
        ):
            lifecycle = EpisodeLifecycle.ACTIVE
        else:
            lifecycle = EpisodeLifecycle.DORMANT
        relation_frames = () if relation is None else relation.t0_evidence_frame_ids
        first_frame, last_frame = _observation_range(
            relation_frames, evidence.last_supported_frames[rows]
        )
        episode_id = f"{stable_id}/location:t0:{owner_id}"
        record = EntityEpisodeRecord(
            stable_entity_id=stable_id,
            episode_id=episode_id,
            source_visit=0,
            source_surface_id=source_surface_id,
            owner_entity_id=owner_id,
            source_vertex_indices=source_vertex_indices[rows],
            first_observation_frame=first_frame,
            last_observation_frame=last_frame,
            lifecycle_state=lifecycle,
            relation_id=None if relation is None else relation.relation_id,
            relation_confidence=None if relation is None else relation.confidence,
            transform_world_from_t0=None,
            retirement_reasons=tuple(
                SurfaceRetirementReason(int(value))
                for value in sorted(
                    {int(value) for value in reasons[rows]}
                    - {int(SurfaceRetirementReason.NONE)}
                )
            ),
            revival_reasons=(
                (SurfaceActionReason.LOCAL_POSITIVE_CORRECTION,)
                if np.intersect1d(rows, recovered, assume_unique=True).size
                else ()
            ),
        )
        records.append(record)
        t0_records[owner_id] = record

    for relation in sorted(
        (item for item in relations if item.accepted),
        key=lambda item: item.t1_owner_entity_id,
    ):
        t0_record = t0_records[relation.t0_owner_entity_id]
        share_episode = (
            relation.inference_state is RelationInferenceState.STATIC
            and t0_record.lifecycle_state is not EpisodeLifecycle.LOCATION_RETIRED
        )
        first_frame, last_frame = _observation_range(
            relation.t1_evidence_frame_ids,
            np.empty(0, dtype=np.int32),
        )
        records.append(
            EntityEpisodeRecord(
                stable_entity_id=relation.stable_entity_id,
                episode_id=(
                    t0_record.episode_id
                    if share_episode
                    else f"{relation.stable_entity_id}/location:t1:{relation.t1_owner_entity_id}"
                ),
                source_visit=1,
                source_surface_id=relation.t1_source_surface_id,
                owner_entity_id=relation.t1_owner_entity_id,
                source_vertex_indices=relation.t1_source_vertex_indices,
                first_observation_frame=first_frame,
                last_observation_frame=last_frame,
                lifecycle_state=EpisodeLifecycle.ACTIVE,
                relation_id=relation.relation_id,
                relation_confidence=relation.confidence,
                transform_world_from_t0=relation.transform_world_from_t0,
                retirement_reasons=(),
                revival_reasons=(),
            )
        )
    return tuple(
        sorted(records, key=lambda item: (item.source_visit, item.owner_entity_id))
    )


def resolve_entity_epoch_update(
    *,
    prior: PriorSurfaceState,
    source_surface_id: str,
    source_vertex_indices: np.ndarray,
    t0_owner_entity_ids: np.ndarray,
    evidence: FineSurfaceEvidence,
    t1_replacement_mask: np.ndarray,
    relation_support: tuple[RelationSupport, ...],
    config: EntityEpochUpdateConfig,
) -> EntityEpochUpdateResult:
    """Start every update from the prior B3 state, including tombstones."""

    if not isinstance(prior, PriorSurfaceState):
        raise TypeError("prior must be PriorSurfaceState")
    count = len(prior.current_valid)
    for values in (
        source_vertex_indices,
        t0_owner_entity_ids,
        t1_replacement_mask,
    ):
        if np.asarray(values).shape != (count,):
            raise ValueError("surface update inputs must align with the prior state")
    if (
        not isinstance(evidence, FineSurfaceEvidence)
        or len(evidence.present_observations) != count
    ):
        raise ValueError("surface evidence must align with the prior state")
    if not isinstance(relation_support, tuple) or any(
        not isinstance(item, RelationSupport) for item in relation_support
    ):
        raise TypeError("relation_support must contain RelationSupport values")
    if not isinstance(source_surface_id, str) or not source_surface_id:
        raise ValueError("source_surface_id must be non-empty")
    if not isinstance(config, EntityEpochUpdateConfig):
        raise TypeError("config must be EntityEpochUpdateConfig")
    source_rows = np.asarray(source_vertex_indices)
    owner_ids = np.asarray(t0_owner_entity_ids)
    replacement = np.asarray(t1_replacement_mask)
    if (
        source_rows.dtype.kind not in "iu"
        or np.any(source_rows < 0)
        or (len(source_rows) > 1 and np.any(source_rows[1:] <= source_rows[:-1]))
    ):
        raise ValueError("source vertex indices must be ordered nonnegative integers")
    if owner_ids.dtype.kind not in "iu" or np.any(owner_ids < 0):
        raise ValueError("owner entity IDs must be nonnegative integers")
    if replacement.dtype != np.bool_:
        raise ValueError("t1 replacement mask must be boolean")

    aliases: dict[tuple[int, int], IdentityAliasRecord] = {
        (0, int(owner_id)): IdentityAliasRecord(
            visit_id=0,
            owner_entity_id=int(owner_id),
            stable_entity_id=f"ovi-t0:{int(owner_id)}",
            relation_id=None,
        )
        for owner_id in np.unique(owner_ids)
        if owner_id > 0
    }
    related_t0: set[int] = set()
    related_t1: set[int] = set()
    relation_ids: set[str] = set()
    relation_positions: dict[str, np.ndarray] = {}
    for relation in relation_support:
        if relation.relation_id in relation_ids:
            raise ValueError("relation IDs must be unique")
        relation_ids.add(relation.relation_id)
        if not relation.accepted:
            continue
        if relation.t0_source_surface_id != source_surface_id:
            raise ValueError("accepted relation references another t0 source surface")
        if (
            relation.t0_owner_entity_id in related_t0
            or relation.t1_owner_entity_id in related_t1
        ):
            raise ValueError("accepted relations must be one-to-one")
        positions = np.searchsorted(source_rows, relation.t0_source_vertex_indices)
        if (
            not len(positions)
            or np.any(positions >= count)
            or not np.array_equal(
                source_rows[positions], relation.t0_source_vertex_indices
            )
            or np.any(owner_ids[positions] != relation.t0_owner_entity_id)
        ):
            raise ValueError("accepted relation t0 support does not match its owner")
        related_t0.add(relation.t0_owner_entity_id)
        related_t1.add(relation.t1_owner_entity_id)
        relation_positions[relation.relation_id] = positions
        aliases[(0, relation.t0_owner_entity_id)] = IdentityAliasRecord(
            visit_id=0,
            owner_entity_id=relation.t0_owner_entity_id,
            stable_entity_id=relation.stable_entity_id,
            relation_id=relation.relation_id,
        )
        aliases[(1, relation.t1_owner_entity_id)] = IdentityAliasRecord(
            visit_id=1,
            owner_entity_id=relation.t1_owner_entity_id,
            stable_entity_id=relation.stable_entity_id,
            relation_id=relation.relation_id,
        )

    current = np.array(prior.current_valid, dtype=np.bool_, copy=True)
    reasons = np.array(prior.retirement_reason_codes, dtype=np.uint8, copy=True)
    states = np.array(prior.evidence_state_codes, dtype=np.uint8, copy=True)
    action_rows: dict[SurfaceActionReason, np.ndarray] = {}
    relation_ids_by_row: dict[int, str] = {}

    changed_replacement = replacement & (
        current
        | (reasons != int(SurfaceRetirementReason.REPLACED))
        | (states != int(CurrentEvidenceState.REPLACED_BY_CURRENT))
    )
    current[replacement] = False
    reasons[replacement] = int(SurfaceRetirementReason.REPLACED)
    states[replacement] = int(CurrentEvidenceState.REPLACED_BY_CURRENT)
    if np.any(changed_replacement):
        action_rows[SurfaceActionReason.T1_REPLACED] = np.flatnonzero(
            changed_replacement
        )

    negative_is_latest = (evidence.last_absent_frames >= 0) & (
        evidence.last_absent_frames > evidence.last_supported_frames
    )
    reliable_negative = (
        ~replacement
        & current
        & (evidence.visible_absent_observations >= config.minimum_absent_observations)
        & (
            evidence.distinct_absent_viewpoints
            >= config.minimum_distinct_absent_viewpoints
        )
        & negative_is_latest
    )
    current[reliable_negative] = False
    reasons[reliable_negative] = int(SurfaceRetirementReason.DIRECT_FREE)
    states[reliable_negative] = int(CurrentEvidenceState.REVOKED_VISIBLE_FREE)
    moved_retirement = np.zeros(count, dtype=np.bool_)
    for relation in relation_support:
        if relation.accepted and relation.motion_verified:
            positions = relation_positions[relation.relation_id]
            moved_retirement[positions] = reliable_negative[positions]
            relation_ids_by_row.update(
                {
                    int(row): relation.relation_id
                    for row in positions[reliable_negative[positions]]
                }
            )
    ordinary_negative = reliable_negative & ~moved_retirement
    if np.any(ordinary_negative):
        action_rows[SurfaceActionReason.RELIABLE_VISIBLE_FREE] = np.flatnonzero(
            ordinary_negative
        )
    if np.any(moved_retirement):
        action_rows[SurfaceActionReason.MOVED_LOCATION_RETIRED] = np.flatnonzero(
            moved_retirement
        )

    recoverable_reason = np.isin(
        prior.retirement_reason_codes,
        [
            int(SurfaceRetirementReason.DIRECT_FREE),
            int(SurfaceRetirementReason.COARSE_NEIGHBORHOOD),
            int(SurfaceRetirementReason.ENTITY_LIFT),
        ],
    )
    positive_is_latest = (evidence.last_supported_frames >= 0) & (
        evidence.last_supported_frames > evidence.last_absent_frames
    )
    recover = (
        ~replacement
        & ~current
        & recoverable_reason
        & (evidence.present_observations >= config.minimum_positive_observations)
        & positive_is_latest
    )
    current[recover] = True
    reasons[recover] = int(SurfaceRetirementReason.NONE)
    states[recover] = int(CurrentEvidenceState.CURRENT_OBSERVED)
    if np.any(recover):
        action_rows[SurfaceActionReason.LOCAL_POSITIVE_CORRECTION] = np.flatnonzero(
            recover
        )

    return EntityEpochUpdateResult(
        current_valid=current,
        retirement_reason_codes=reasons,
        evidence_state_codes=states,
        surface_deltas=_surface_deltas(
            source_surface_id=source_surface_id,
            source_vertex_indices=source_rows,
            owner_ids=owner_ids,
            prior=prior,
            evidence=evidence,
            current=current,
            reasons=reasons,
            action_rows=action_rows,
            relation_ids_by_row=relation_ids_by_row,
            aliases=aliases,
            used_new_measurement=config.evidence_is_new_measurement,
        ),
        identity_aliases=tuple(aliases[key] for key in sorted(aliases)),
        episode_records=_episode_records(
            source_surface_id=source_surface_id,
            source_vertex_indices=source_rows,
            owner_ids=owner_ids,
            current=current,
            reasons=reasons,
            evidence=evidence,
            aliases=aliases,
            relations=relation_support,
            action_rows=action_rows,
        ),
    )


__all__ = [
    "EntityEpisodeRecord",
    "EntityEpochUpdateConfig",
    "EntityEpochUpdateResult",
    "EpisodeLifecycle",
    "IdentityAliasRecord",
    "RelationInferenceState",
    "RelationSupport",
    "SurfaceActionReason",
    "SurfaceStateDelta",
    "resolve_entity_epoch_update",
]
