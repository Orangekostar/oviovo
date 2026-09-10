"""Compose source-preserving OVI fine surfaces with CROVE currentness."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.oviv2.current_surface import (
    CurrentEvidenceState,
    CurrentSurfaceView,
    SemanticSource,
)
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateResult,
    SurfaceActionReason,
)
from src.oviv2.fine_dynamic_policy import SurfaceRetirementReason
from src.oviv2.fine_surface_validity import (
    FineSurfaceEvidence,
    FineValidityConfig,
    resolve_fine_current_validity,
)

_ARRAY_DTYPES: dict[str, object] = {
    "vertices_xyz": np.float32,
    "normals_xyz": np.float32,
    "triangles": np.int64,
    "source_vertex_indices": np.int64,
    "observed_rgb_uint8": np.uint8,
    "rgb_valid": np.bool_,
    "last_supported_frames": np.int32,
    "owner_entity_ids": np.int64,
    "owner_confidences": np.float32,
    "semantic_ids": np.int32,
    "semantic_confidences": np.float32,
    "semantic_support_reliabilities": np.float32,
    "semantic_source_codes": np.uint8,
}


def _immutable(value: object, dtype: object) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.flags.writeable or result.base is not None:
        result = result.copy()
    result.setflags(write=False)
    return result


def _validate_probability(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class FineVisitSurface:
    """One native fine mesh and its prediction attributes in source row order."""

    visit_id: int
    source_surface_index: int
    geometry_epoch: int
    vertices_xyz: np.ndarray
    normals_xyz: np.ndarray
    triangles: np.ndarray
    source_vertex_indices: np.ndarray
    observed_rgb_uint8: np.ndarray
    rgb_valid: np.ndarray
    last_supported_frames: np.ndarray
    owner_entity_ids: np.ndarray
    owner_confidences: np.ndarray
    semantic_ids: np.ndarray
    semantic_confidences: np.ndarray
    semantic_support_reliabilities: np.ndarray
    semantic_source_codes: np.ndarray

    def __post_init__(self) -> None:
        if type(self.visit_id) is not int or not 0 <= self.visit_id <= np.iinfo(np.int16).max:
            raise ValueError("visit_id must fit a nonnegative int16")
        if (
            type(self.source_surface_index) is not int
            or not 0 <= self.source_surface_index <= np.iinfo(np.uint16).max
        ):
            raise ValueError("source_surface_index must fit uint16")
        if (
            type(self.geometry_epoch) is not int
            or not 0 <= self.geometry_epoch <= np.iinfo(np.int32).max
        ):
            raise ValueError("geometry_epoch must fit a nonnegative int32")
        for field in fields(self):
            if field.name in {"visit_id", "source_surface_index", "geometry_epoch"}:
                continue
            object.__setattr__(
                self,
                field.name,
                _immutable(getattr(self, field.name), _ARRAY_DTYPES[field.name]),
            )

        if self.vertices_xyz.ndim != 2 or self.vertices_xyz.shape[1:] != (3,):
            raise ValueError("vertices_xyz must have shape (N, 3)")
        count = len(self.vertices_xyz)
        if self.normals_xyz.shape != (count, 3):
            raise ValueError("normals_xyz must have shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,):
            raise ValueError("triangles must have shape (M, 3)")
        if self.observed_rgb_uint8.shape != (count, 3):
            raise ValueError("observed_rgb_uint8 must have shape (N, 3)")
        for name in _ARRAY_DTYPES:
            if name in {"vertices_xyz", "normals_xyz", "triangles", "observed_rgb_uint8"}:
                continue
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        if not np.isfinite(self.vertices_xyz).all() or not np.isfinite(
            self.normals_xyz
        ).all():
            raise ValueError("surface geometry must be finite")
        if self.triangles.size and (
            int(self.triangles.min()) < 0 or int(self.triangles.max()) >= count
        ):
            raise ValueError("triangle indices lie outside the vertex array")
        if np.any(self.source_vertex_indices < 0) or (
            count > 1 and np.any(np.diff(self.source_vertex_indices) <= 0)
        ):
            raise ValueError("source vertex indices must be nonnegative and strictly increasing")
        if np.any(self.last_supported_frames < -1):
            raise ValueError("last supported frame cannot be below -1")
        if np.any(self.owner_entity_ids < 0) or np.any(self.semantic_ids < 0):
            raise ValueError("owner and semantic identifiers must be nonnegative")
        _validate_probability("owner_confidences", self.owner_confidences)
        _validate_probability("semantic_confidences", self.semantic_confidences)
        _validate_probability(
            "semantic_support_reliabilities", self.semantic_support_reliabilities
        )
        source_values = {int(value) for value in SemanticSource}
        if not set(int(value) for value in np.unique(self.semantic_source_codes)) <= source_values:
            raise ValueError("semantic_source_codes contain an unknown source")


def _immutable_text(value: object) -> np.ndarray:
    result = np.array(value, dtype=np.str_, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError("text sidecar arrays must be one dimensional")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EntityEpochStateSidecar:
    """Per-canonical-row entity state and evidence provenance."""

    surface_id: str
    source_surface_indices: np.ndarray
    source_vertex_indices: np.ndarray
    source_visit_ids: np.ndarray
    source_existed_before: np.ndarray
    stable_entity_ids_before: np.ndarray
    stable_entity_ids_after: np.ndarray
    episode_ids_before: np.ndarray
    episode_ids_after: np.ndarray
    current_valid_before: np.ndarray
    current_valid_after: np.ndarray
    retirement_reason_before_codes: np.ndarray
    retirement_reason_after_codes: np.ndarray
    action_reasons: np.ndarray
    used_new_measurement: np.ndarray
    relation_ids: np.ndarray
    evidence_frame_offsets: np.ndarray
    evidence_frame_ids: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        typed = {
            "source_surface_indices": np.uint16,
            "source_vertex_indices": np.int64,
            "source_visit_ids": np.int16,
            "source_existed_before": np.bool_,
            "current_valid_before": np.bool_,
            "current_valid_after": np.bool_,
            "retirement_reason_before_codes": np.uint8,
            "retirement_reason_after_codes": np.uint8,
            "used_new_measurement": np.bool_,
            "evidence_frame_offsets": np.int64,
            "evidence_frame_ids": np.int32,
        }
        for name, dtype in typed.items():
            object.__setattr__(self, name, _immutable(getattr(self, name), dtype))
        for name in (
            "stable_entity_ids_before",
            "stable_entity_ids_after",
            "episode_ids_before",
            "episode_ids_after",
            "action_reasons",
            "relation_ids",
        ):
            object.__setattr__(self, name, _immutable_text(getattr(self, name)))
        count = len(self.source_vertex_indices)
        row_fields = (
            "source_surface_indices",
            "source_visit_ids",
            "source_existed_before",
            "stable_entity_ids_before",
            "stable_entity_ids_after",
            "episode_ids_before",
            "episode_ids_after",
            "current_valid_before",
            "current_valid_after",
            "retirement_reason_before_codes",
            "retirement_reason_after_codes",
            "action_reasons",
            "used_new_measurement",
            "relation_ids",
        )
        if self.source_vertex_indices.ndim != 1 or any(
            getattr(self, name).shape != (count,) for name in row_fields
        ):
            raise ValueError("entity epoch row arrays must have equal lengths")
        if np.any(self.source_vertex_indices < 0) or np.any(self.source_visit_ids < 0):
            raise ValueError("entity epoch source keys must be nonnegative")
        if count > 1:
            previous_surface = self.source_surface_indices[:-1]
            current_surface = self.source_surface_indices[1:]
            previous_vertex = self.source_vertex_indices[:-1]
            current_vertex = self.source_vertex_indices[1:]
            if np.any(
                (current_surface < previous_surface)
                | (
                    (current_surface == previous_surface)
                    & (current_vertex <= previous_vertex)
                )
            ):
                raise ValueError("entity epoch source keys must be ordered and unique")
        reason_values = {int(value) for value in SurfaceRetirementReason}
        if not {
            int(value)
            for value in np.unique(
                np.concatenate(
                    (
                        self.retirement_reason_before_codes,
                        self.retirement_reason_after_codes,
                    )
                )
            )
        } <= reason_values:
            raise ValueError("entity epoch sidecar contains an unknown retirement reason")
        reason_is_none_before = self.retirement_reason_before_codes == int(
            SurfaceRetirementReason.NONE
        )
        if np.any(
            self.source_existed_before
            & (self.current_valid_before != reason_is_none_before)
        ):
            raise ValueError("entity epoch before-state validity and reason disagree")
        if np.any(
            self.current_valid_after
            != (
                self.retirement_reason_after_codes
                == int(SurfaceRetirementReason.NONE)
            )
        ):
            raise ValueError("entity epoch after-state validity and reason disagree")
        action_values = {
            "",
            "t1_current_observed",
            *(value.value for value in SurfaceActionReason),
        }
        if not set(self.action_reasons.tolist()) <= action_values:
            raise ValueError("entity epoch sidecar contains an unknown action reason")
        if self.evidence_frame_offsets.shape != (count + 1,):
            raise ValueError("evidence_frame_offsets must have N + 1 values")
        if (
            self.evidence_frame_offsets[0] != 0
            or self.evidence_frame_offsets[-1] != len(self.evidence_frame_ids)
            or np.any(np.diff(self.evidence_frame_offsets) < 0)
            or np.any(self.evidence_frame_ids < 0)
        ):
            raise ValueError("evidence frame CSR is invalid")

    @property
    def stable_entity_ids(self) -> np.ndarray:
        return self.stable_entity_ids_after

    @property
    def episode_ids(self) -> np.ndarray:
        return self.episode_ids_after

    def evidence_frames_for_row(self, row: int) -> tuple[int, ...]:
        if type(row) is not int or not 0 <= row < len(self.source_vertex_indices):
            raise IndexError("entity epoch row is outside the sidecar")
        start = int(self.evidence_frame_offsets[row])
        end = int(self.evidence_frame_offsets[row + 1])
        return tuple(int(value) for value in self.evidence_frame_ids[start:end])


@dataclass(frozen=True, slots=True)
class EntityEpochComposition:
    surface: CurrentSurfaceView
    state_sidecar: EntityEpochStateSidecar

    def __post_init__(self) -> None:
        if not isinstance(self.surface, CurrentSurfaceView):
            raise TypeError("surface must be CurrentSurfaceView")
        if not isinstance(self.state_sidecar, EntityEpochStateSidecar):
            raise TypeError("state_sidecar must be EntityEpochStateSidecar")
        sidecar = self.state_sidecar
        if sidecar.surface_id != self.surface.surface_id:
            raise ValueError("surface and entity epoch sidecar IDs differ")
        for name in (
            "source_surface_indices",
            "source_vertex_indices",
            "source_visit_ids",
        ):
            if not np.array_equal(getattr(sidecar, name), getattr(self.surface, name)):
                raise ValueError(f"surface and entity epoch {name} differ")
        if not np.array_equal(sidecar.current_valid_after, self.surface.current_valid):
            raise ValueError("surface and entity epoch current validity differ")


def _concatenate(t0: FineVisitSurface, t1: FineVisitSurface, name: str) -> np.ndarray:
    return np.concatenate((getattr(t0, name), getattr(t1, name)), axis=0)


def current_surface_from_visit(
    surface: FineVisitSurface,
    *,
    surface_id: str,
) -> CurrentSurfaceView:
    """Expose one native visit as an entirely current canonical surface."""

    if not isinstance(surface, FineVisitSurface):
        raise TypeError("surface must be a FineVisitSurface")
    count = len(surface.vertices_xyz)
    return CurrentSurfaceView(
        surface_id=surface_id,
        vertices_xyz=surface.vertices_xyz,
        normals_xyz=surface.normals_xyz,
        triangles=surface.triangles,
        source_surface_indices=np.full(
            count, surface.source_surface_index, dtype=np.uint16
        ),
        source_vertex_indices=surface.source_vertex_indices,
        source_visit_ids=np.full(count, surface.visit_id, dtype=np.int16),
        geometry_epochs=np.full(count, surface.geometry_epoch, dtype=np.int32),
        observed_rgb_uint8=surface.observed_rgb_uint8,
        rgb_valid=surface.rgb_valid,
        current_valid=np.ones(count, dtype=np.bool_),
        evidence_state_codes=np.full(
            count, int(CurrentEvidenceState.CURRENT_OBSERVED), dtype=np.uint8
        ),
        last_supported_frames=surface.last_supported_frames,
        owner_entity_ids=surface.owner_entity_ids,
        owner_confidences=surface.owner_confidences,
        semantic_ids=surface.semantic_ids,
        semantic_confidences=surface.semantic_confidences,
        semantic_support_reliabilities=surface.semantic_support_reliabilities,
        semantic_source_codes=surface.semantic_source_codes,
    )


def compose_two_visit_fine_surface(
    *,
    t0: FineVisitSurface,
    t1: FineVisitSurface,
    t0_evidence: FineSurfaceEvidence,
    coarse_visible_free_candidates: np.ndarray,
    validity_config: FineValidityConfig,
    surface_id: str,
) -> CurrentSurfaceView:
    """Keep native source rows while resolving current validity at fine scale."""

    if not isinstance(t0, FineVisitSurface) or not isinstance(t1, FineVisitSurface):
        raise TypeError("t0 and t1 must be FineVisitSurface values")
    if (t0.visit_id, t1.visit_id) != (0, 1):
        raise ValueError("two-visit composition requires visit IDs zero and one")
    if t0.source_surface_index == t1.source_surface_index:
        raise ValueError("source surfaces must have distinct indices")
    if not isinstance(t0_evidence, FineSurfaceEvidence):
        raise TypeError("t0_evidence must be FineSurfaceEvidence")
    if not isinstance(validity_config, FineValidityConfig):
        raise TypeError("validity_config must be FineValidityConfig")

    t0_validity = resolve_fine_current_validity(
        source_visit_ids=np.full(len(t0.vertices_xyz), t0.visit_id, dtype=np.int16),
        latest_visit_id=t1.visit_id,
        coarse_visible_free_candidates=coarse_visible_free_candidates,
        evidence=t0_evidence,
        config=validity_config,
    )
    t1_count = len(t1.vertices_xyz)
    offset_triangles = t1.triangles + len(t0.vertices_xyz)
    triangles = np.concatenate((t0.triangles, offset_triangles), axis=0)
    current_valid = np.concatenate(
        (t0_validity.current_valid, np.ones(t1_count, dtype=np.bool_))
    )
    evidence_states = np.concatenate(
        (
            t0_validity.evidence_state_codes,
            np.full(
                t1_count,
                int(CurrentEvidenceState.CURRENT_OBSERVED),
                dtype=np.uint8,
            ),
        )
    )
    return CurrentSurfaceView(
        surface_id=surface_id,
        vertices_xyz=_concatenate(t0, t1, "vertices_xyz"),
        normals_xyz=_concatenate(t0, t1, "normals_xyz"),
        triangles=triangles,
        source_surface_indices=np.concatenate(
            (
                np.full(
                    len(t0.vertices_xyz), t0.source_surface_index, dtype=np.uint16
                ),
                np.full(t1_count, t1.source_surface_index, dtype=np.uint16),
            )
        ),
        source_vertex_indices=_concatenate(t0, t1, "source_vertex_indices"),
        source_visit_ids=np.concatenate(
            (
                np.full(len(t0.vertices_xyz), t0.visit_id, dtype=np.int16),
                np.full(t1_count, t1.visit_id, dtype=np.int16),
            )
        ),
        geometry_epochs=np.concatenate(
            (
                np.full(len(t0.vertices_xyz), t0.geometry_epoch, dtype=np.int32),
                np.full(t1_count, t1.geometry_epoch, dtype=np.int32),
            )
        ),
        observed_rgb_uint8=_concatenate(t0, t1, "observed_rgb_uint8"),
        rgb_valid=_concatenate(t0, t1, "rgb_valid"),
        current_valid=current_valid,
        evidence_state_codes=evidence_states,
        last_supported_frames=np.concatenate(
            (t0_validity.last_supported_frames, t1.last_supported_frames)
        ),
        owner_entity_ids=_concatenate(t0, t1, "owner_entity_ids"),
        owner_confidences=_concatenate(t0, t1, "owner_confidences"),
        semantic_ids=_concatenate(t0, t1, "semantic_ids"),
        semantic_confidences=_concatenate(t0, t1, "semantic_confidences"),
        semantic_support_reliabilities=_concatenate(
            t0, t1, "semantic_support_reliabilities"
        ),
        semantic_source_codes=_concatenate(t0, t1, "semantic_source_codes"),
    )


def _entity_epoch_evidence_states(update: EntityEpochUpdateResult) -> np.ndarray:
    states = np.array(update.evidence_state_codes, dtype=np.uint8, copy=True)
    invalid = ~update.current_valid
    replaced = update.retirement_reason_codes == int(SurfaceRetirementReason.REPLACED)
    states[invalid & replaced] = int(CurrentEvidenceState.REPLACED_BY_CURRENT)
    states[invalid & ~replaced] = int(CurrentEvidenceState.REVOKED_VISIBLE_FREE)
    terminal = np.isin(
        states,
        [
            int(CurrentEvidenceState.REPLACED_BY_CURRENT),
            int(CurrentEvidenceState.REVOKED_VISIBLE_FREE),
        ],
    )
    if np.any(update.current_valid & terminal):
        raise ValueError("valid entity epoch rows cannot carry a terminal state")
    return states


def _row_positions(
    surface: FineVisitSurface,
    source_rows: np.ndarray,
    *,
    owner_entity_id: int,
) -> np.ndarray:
    positions = np.searchsorted(surface.source_vertex_indices, source_rows)
    if (
        not len(positions)
        or np.any(positions >= len(surface.source_vertex_indices))
        or not np.array_equal(surface.source_vertex_indices[positions], source_rows)
        or np.any(surface.owner_entity_ids[positions] != owner_entity_id)
    ):
        raise ValueError("entity epoch provenance rows do not match their OVI owner")
    return positions


def _default_identity(
    *,
    visit_id: int,
    owner_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stable = np.asarray(
        [f"ovi-t{visit_id}:{int(owner)}" if owner > 0 else "" for owner in owner_ids],
        dtype=object,
    )
    episodes = np.asarray(
        [
            f"{stable_id}/location:t{visit_id}:{int(owner)}" if owner > 0 else ""
            for stable_id, owner in zip(stable, owner_ids, strict=True)
        ],
        dtype=object,
    )
    return stable, episodes


def _entity_epoch_sidecar(
    *,
    t0: FineVisitSurface,
    t1: FineVisitSurface,
    update: EntityEpochUpdateResult,
    surface: CurrentSurfaceView,
) -> EntityEpochStateSidecar:
    t0_count = len(t0.vertices_xyz)
    total = len(surface.vertices_xyz)
    existed = np.concatenate(
        (np.ones(t0_count, dtype=np.bool_), np.zeros(len(t1.vertices_xyz), dtype=np.bool_))
    )
    before_valid = np.concatenate(
        (np.array(update.current_valid, copy=True), np.zeros(len(t1.vertices_xyz), dtype=np.bool_))
    )
    before_reasons = np.concatenate(
        (
            np.array(update.retirement_reason_codes, copy=True),
            np.full(
                len(t1.vertices_xyz),
                int(SurfaceRetirementReason.NONE),
                dtype=np.uint8,
            ),
        )
    )
    after_reasons = np.concatenate(
        (
            update.retirement_reason_codes,
            np.full(
                len(t1.vertices_xyz),
                int(SurfaceRetirementReason.NONE),
                dtype=np.uint8,
            ),
        )
    )
    action_reasons = np.asarray(
        [""] * t0_count + ["t1_current_observed"] * len(t1.vertices_xyz),
        dtype=object,
    )
    used_new_measurement = np.concatenate(
        (np.zeros(t0_count, dtype=np.bool_), np.ones(len(t1.vertices_xyz), dtype=np.bool_))
    )
    t0_stable_before, t0_episode_before = _default_identity(
        visit_id=0, owner_ids=t0.owner_entity_ids
    )
    t1_stable_after, t1_episode_after = _default_identity(
        visit_id=1, owner_ids=t1.owner_entity_ids
    )
    stable_before = np.concatenate(
        (t0_stable_before, np.asarray([""] * len(t1.vertices_xyz), dtype=object))
    )
    episode_before = np.concatenate(
        (t0_episode_before, np.asarray([""] * len(t1.vertices_xyz), dtype=object))
    )
    stable_after = np.concatenate((t0_stable_before.copy(), t1_stable_after))
    episode_after = np.concatenate((t0_episode_before.copy(), t1_episode_after))
    relation_ids = np.asarray([""] * total, dtype=object)
    frame_sets = [
        ({int(frame)} if frame >= 0 else set())
        for frame in np.concatenate(
            (t0.last_supported_frames, t1.last_supported_frames)
        )
    ]
    surfaces = {0: t0, 1: t1}
    offsets = {0: 0, 1: t0_count}

    aliases: set[tuple[int, int]] = set()
    for alias in update.identity_aliases:
        key = (alias.visit_id, alias.owner_entity_id)
        if key in aliases or alias.visit_id not in surfaces:
            raise ValueError("entity epoch identity aliases are invalid")
        aliases.add(key)
        local_rows = np.flatnonzero(
            surfaces[alias.visit_id].owner_entity_ids == alias.owner_entity_id
        )
        if not len(local_rows):
            raise ValueError("entity epoch identity alias references an absent owner")
        rows = local_rows + offsets[alias.visit_id]
        stable_after[rows] = alias.stable_entity_id
        episode_after[rows] = np.asarray(
            [
                f"{alias.stable_entity_id}/location:t{alias.visit_id}:{alias.owner_entity_id}"
            ]
            * len(rows),
            dtype=object,
        )

    episodes: set[tuple[int, int]] = set()
    for episode in update.episode_records:
        key = (episode.source_visit, episode.owner_entity_id)
        if key in episodes or episode.source_visit not in surfaces:
            raise ValueError("entity epoch episode records are invalid")
        episodes.add(key)
        source = surfaces[episode.source_visit]
        local_owner_rows = np.flatnonzero(
            source.owner_entity_ids == episode.owner_entity_id
        )
        if not len(local_owner_rows):
            raise ValueError("entity epoch episode references an absent owner")
        owner_rows = local_owner_rows + offsets[episode.source_visit]
        stable_after[owner_rows] = episode.stable_entity_id
        episode_after[owner_rows] = episode.episode_id
        local_support = _row_positions(
            source,
            episode.source_vertex_indices,
            owner_entity_id=episode.owner_entity_id,
        )
        support = local_support + offsets[episode.source_visit]
        if episode.relation_id is not None:
            relation_ids[support] = episode.relation_id
        episode_frames = {
            frame
            for frame in (
                episode.first_observation_frame,
                episode.last_observation_frame,
            )
            if frame >= 0
        }
        for row in support:
            frame_sets[int(row)].update(episode_frames)

    changed_rows: set[int] = set()
    for delta in update.surface_deltas:
        positions = _row_positions(
            t0,
            delta.source_vertex_indices,
            owner_entity_id=delta.owner_entity_id,
        )
        if any(int(position) in changed_rows for position in positions):
            raise ValueError("entity epoch surface deltas overlap")
        changed_rows.update(int(position) for position in positions)
        if (
            np.any(update.current_valid[positions] != delta.current_valid_after)
            or np.any(
                update.retirement_reason_codes[positions]
                != int(delta.retirement_reason_after)
            )
        ):
            raise ValueError("surface delta after-state differs from the update")
        before_valid[positions] = delta.current_valid_before
        before_reasons[positions] = int(delta.retirement_reason_before)
        action_reasons[positions] = delta.action_reason.value
        used_new_measurement[positions] = delta.used_new_measurement
        stable_before[positions] = delta.stable_entity_id_before or ""
        stable_after[positions] = delta.stable_entity_id_after or ""
        episode_before[positions] = delta.episode_id_before or ""
        episode_after[positions] = delta.episode_id_after or ""
        relation_ids[positions] = delta.relation_id or ""
        for position in positions:
            frame_sets[int(position)] = set(delta.evidence_frame_ids)

    flattened_frames = np.asarray(
        [frame for values in frame_sets for frame in sorted(values)], dtype=np.int32
    )
    frame_offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum([len(values) for values in frame_sets], dtype=np.int64),
        )
    )
    return EntityEpochStateSidecar(
        surface_id=surface.surface_id,
        source_surface_indices=surface.source_surface_indices,
        source_vertex_indices=surface.source_vertex_indices,
        source_visit_ids=surface.source_visit_ids,
        source_existed_before=existed,
        stable_entity_ids_before=stable_before,
        stable_entity_ids_after=stable_after,
        episode_ids_before=episode_before,
        episode_ids_after=episode_after,
        current_valid_before=before_valid,
        current_valid_after=surface.current_valid,
        retirement_reason_before_codes=before_reasons,
        retirement_reason_after_codes=after_reasons,
        action_reasons=action_reasons,
        used_new_measurement=used_new_measurement,
        relation_ids=relation_ids,
        evidence_frame_offsets=frame_offsets,
        evidence_frame_ids=flattened_frames,
    )


def compose_entity_epoch_fine_surface(
    *,
    t0: FineVisitSurface,
    t1: FineVisitSurface,
    update: EntityEpochUpdateResult,
    surface_id: str,
) -> EntityEpochComposition:
    """Compose one canonical source surface plus aligned entity-epoch state."""

    if not isinstance(t0, FineVisitSurface) or not isinstance(t1, FineVisitSurface):
        raise TypeError("t0 and t1 must be FineVisitSurface values")
    if (t0.visit_id, t1.visit_id) != (0, 1):
        raise ValueError("entity epoch composition requires visit IDs zero and one")
    if t0.source_surface_index >= t1.source_surface_index:
        raise ValueError("source surface indices must follow visit order")
    if not isinstance(update, EntityEpochUpdateResult):
        raise TypeError("update must be EntityEpochUpdateResult")
    if len(update.current_valid) != len(t0.vertices_xyz):
        raise ValueError("entity epoch update must align with the t0 surface")
    t1_count = len(t1.vertices_xyz)
    triangles = np.concatenate(
        (t0.triangles, t1.triangles + len(t0.vertices_xyz)), axis=0
    )
    surface = CurrentSurfaceView(
        surface_id=surface_id,
        vertices_xyz=_concatenate(t0, t1, "vertices_xyz"),
        normals_xyz=_concatenate(t0, t1, "normals_xyz"),
        triangles=triangles,
        source_surface_indices=np.concatenate(
            (
                np.full(
                    len(t0.vertices_xyz), t0.source_surface_index, dtype=np.uint16
                ),
                np.full(t1_count, t1.source_surface_index, dtype=np.uint16),
            )
        ),
        source_vertex_indices=_concatenate(t0, t1, "source_vertex_indices"),
        source_visit_ids=np.concatenate(
            (
                np.full(len(t0.vertices_xyz), t0.visit_id, dtype=np.int16),
                np.full(t1_count, t1.visit_id, dtype=np.int16),
            )
        ),
        geometry_epochs=np.concatenate(
            (
                np.full(len(t0.vertices_xyz), t0.geometry_epoch, dtype=np.int32),
                np.full(t1_count, t1.geometry_epoch, dtype=np.int32),
            )
        ),
        observed_rgb_uint8=_concatenate(t0, t1, "observed_rgb_uint8"),
        rgb_valid=_concatenate(t0, t1, "rgb_valid"),
        current_valid=np.concatenate(
            (update.current_valid, np.ones(t1_count, dtype=np.bool_))
        ),
        evidence_state_codes=np.concatenate(
            (
                _entity_epoch_evidence_states(update),
                np.full(
                    t1_count,
                    int(CurrentEvidenceState.CURRENT_OBSERVED),
                    dtype=np.uint8,
                ),
            )
        ),
        last_supported_frames=_concatenate(t0, t1, "last_supported_frames"),
        owner_entity_ids=_concatenate(t0, t1, "owner_entity_ids"),
        owner_confidences=_concatenate(t0, t1, "owner_confidences"),
        semantic_ids=_concatenate(t0, t1, "semantic_ids"),
        semantic_confidences=_concatenate(t0, t1, "semantic_confidences"),
        semantic_support_reliabilities=_concatenate(
            t0, t1, "semantic_support_reliabilities"
        ),
        semantic_source_codes=_concatenate(t0, t1, "semantic_source_codes"),
    )
    return EntityEpochComposition(
        surface=surface,
        state_sidecar=_entity_epoch_sidecar(
            t0=t0,
            t1=t1,
            update=update,
            surface=surface,
        ),
    )


__all__ = [
    "EntityEpochComposition",
    "EntityEpochStateSidecar",
    "FineVisitSurface",
    "compose_entity_epoch_fine_surface",
    "compose_two_visit_fine_surface",
    "current_surface_from_visit",
]
