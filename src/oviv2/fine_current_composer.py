"""Compose source-preserving OVI fine surfaces with CROVE currentness."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.oviv2.current_surface import (
    CurrentEvidenceState,
    CurrentSurfaceView,
    SemanticSource,
)
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


__all__ = [
    "FineVisitSurface",
    "compose_two_visit_fine_surface",
    "current_surface_from_visit",
]
