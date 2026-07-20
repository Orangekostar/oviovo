from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement

from src.oviv2.addressing import point_to_voxel
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.semantic_fusion import SemanticFusionConfig, fuse_semantics


_INT64_MAX = int(np.iinfo(np.int64).max)


def _int64_identifier(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if normalized > _INT64_MAX:
        raise ValueError(f"{name} must fit signed int64")
    return normalized


def _read_only_array(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LabeledMesh:
    vertices_xyz: np.ndarray
    triangles: np.ndarray
    colors_rgb: np.ndarray
    semantic_ids: np.ndarray
    entity_ids: np.ndarray
    semantic_confidence: np.ndarray
    ownership_confidence: np.ndarray

    def __post_init__(self) -> None:
        dtypes = {
            "vertices_xyz": np.float32,
            "triangles": np.int64,
            "colors_rgb": np.float32,
            "semantic_ids": np.int64,
            "entity_ids": np.int64,
            "semantic_confidence": np.float32,
            "ownership_confidence": np.float32,
        }
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                _read_only_array(getattr(self, field.name), dtype=dtypes[field.name]),
            )

        vertex_count = self.vertices_xyz.shape[0]
        if self.vertices_xyz.shape != (vertex_count, 3):
            raise ValueError("vertices_xyz must have shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")
        if self.colors_rgb.shape != (vertex_count, 3):
            raise ValueError("colors_rgb must have shape (N, 3)")
        for name in (
            "semantic_ids",
            "entity_ids",
            "semantic_confidence",
            "ownership_confidence",
        ):
            if getattr(self, name).shape != (vertex_count,):
                raise ValueError(f"{name} must have shape (N,)")
        if not np.isfinite(self.vertices_xyz).all() or not np.isfinite(self.colors_rgb).all():
            raise ValueError("mesh positions and colors must be finite")
        if np.any(self.semantic_ids < 0) or np.any(self.entity_ids < 0):
            raise ValueError("mesh semantic and entity IDs must be non-negative")
        for name in ("semantic_confidence", "ownership_confidence"):
            confidence = getattr(self, name)
            if not np.isfinite(confidence).all() or np.any(confidence < 0.0) or np.any(confidence > 1.0):
                raise ValueError(f"{name} values must be finite and lie in [0, 1]")
        if self.triangles.size:
            if self.triangles.min() < 0 or self.triangles.max() >= vertex_count:
                raise ValueError("triangle indices lie outside the vertex array")


def canonicalize_labeled_mesh(mesh: LabeledMesh) -> LabeledMesh:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    vertex_count = mesh.vertices_xyz.shape[0]
    if vertex_count == 0:
        return mesh
    order = np.lexsort(
        (
            mesh.ownership_confidence,
            mesh.semantic_confidence,
            mesh.entity_ids,
            mesh.semantic_ids,
            mesh.colors_rgb[:, 2],
            mesh.colors_rgb[:, 1],
            mesh.colors_rgb[:, 0],
            mesh.vertices_xyz[:, 2],
            mesh.vertices_xyz[:, 1],
            mesh.vertices_xyz[:, 0],
        )
    )
    inverse = np.empty(vertex_count, dtype=np.int64)
    inverse[order] = np.arange(vertex_count, dtype=np.int64)
    triangles = inverse[mesh.triangles]
    if len(triangles):
        minimum_slots = triangles.argmin(axis=1)
        canonical_triangles = triangles.copy()
        canonical_triangles[minimum_slots == 1] = triangles[minimum_slots == 1][:, [1, 2, 0]]
        canonical_triangles[minimum_slots == 2] = triangles[minimum_slots == 2][:, [2, 0, 1]]
        triangle_order = np.lexsort(
            (
                canonical_triangles[:, 2],
                canonical_triangles[:, 1],
                canonical_triangles[:, 0],
            )
        )
        canonical_triangles = canonical_triangles[triangle_order]
    else:
        canonical_triangles = triangles
    return LabeledMesh(
        vertices_xyz=mesh.vertices_xyz[order],
        triangles=canonical_triangles,
        colors_rgb=mesh.colors_rgb[order],
        semantic_ids=mesh.semantic_ids[order],
        entity_ids=mesh.entity_ids[order],
        semantic_confidence=mesh.semantic_confidence[order],
        ownership_confidence=mesh.ownership_confidence[order],
    )


def _validated_entity_semantics(
    value: Mapping[int, tuple[int, float]] | None,
) -> dict[int, tuple[int, float]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("entity_semantics must be a mapping or None")
    normalized: dict[int, tuple[int, float]] = {}
    for entity_id, semantics in value.items():
        if (
            isinstance(entity_id, (bool, np.bool_))
            or not isinstance(entity_id, Integral)
            or int(entity_id) <= 0
        ):
            raise ValueError("entity_semantics keys must be positive integers")
        normalized_entity_id = _int64_identifier(
            entity_id,
            "entity_semantics entity ID",
            minimum=1,
        )
        if not isinstance(semantics, tuple):
            raise TypeError("entity_semantics values must be 2-tuples")
        if len(semantics) != 2:
            raise ValueError("entity_semantics values must be 2-tuples")
        semantic_id, confidence = semantics
        if (
            isinstance(semantic_id, (bool, np.bool_))
            or not isinstance(semantic_id, Integral)
            or int(semantic_id) < 0
        ):
            raise ValueError("entity semantic IDs must be non-negative integers")
        normalized_semantic_id = _int64_identifier(
            semantic_id,
            "entity semantic ID",
            minimum=0,
        )
        if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real):
            raise ValueError("entity semantic confidence must be finite and lie in [0, 1]")
        normalized_confidence = float(confidence)
        if not np.isfinite(normalized_confidence) or not 0.0 <= normalized_confidence <= 1.0:
            raise ValueError("entity semantic confidence must be finite and lie in [0, 1]")
        normalized[normalized_entity_id] = (normalized_semantic_id, normalized_confidence)
    return normalized


def _validated_entity_posteriors(
    value: Mapping[int, tuple[tuple[int, float], ...]] | None,
    valid_semantic_ids: frozenset[int] | None,
) -> dict[int, tuple[tuple[int, float], ...]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("entity_posteriors must be a mapping or None")
    normalized: dict[int, tuple[tuple[int, float], ...]] = {}
    for entity_id, probabilities in value.items():
        normalized_entity_id = _int64_identifier(
            entity_id,
            "entity_posteriors entity ID",
            minimum=1,
        )
        if not isinstance(probabilities, tuple):
            raise TypeError("entity_posteriors values must be tuples")
        normalized_probabilities = tuple(probabilities)
        fuse_semantics((), normalized_probabilities, 1.0)
        if valid_semantic_ids is not None and any(
            semantic_id not in valid_semantic_ids
            for semantic_id, _ in normalized_probabilities
        ):
            raise ValueError("entity posterior class is outside frozen vocabulary")
        normalized[normalized_entity_id] = normalized_probabilities
    return normalized


def _validated_semantic_id_set(value: object) -> frozenset[int] | None:
    if value is None:
        return None
    if not isinstance(value, (set, frozenset)):
        raise TypeError("valid_semantic_ids must be a set or frozenset")
    normalized = frozenset(
        _int64_identifier(item, "valid semantic ID", minimum=1)
        for item in value
    )
    if not normalized:
        raise ValueError("valid_semantic_ids must be non-empty")
    return normalized


def derive_labeled_mesh(
    geometry: SparseTsdfVolume,
    evidence: SparseEvidenceStore,
    ownership: ReversibleOwnershipStore,
    *,
    weight_threshold: float = 1.0,
    entity_semantics: Mapping[int, tuple[int, float]] | None = None,
    entity_posteriors: Mapping[int, tuple[tuple[int, float], ...]] | None = None,
    semantic_fusion: SemanticFusionConfig | None = None,
    valid_semantic_ids: set[int] | frozenset[int] | None = None,
) -> LabeledMesh:
    if evidence.config.block_resolution != geometry.config.block_resolution:
        raise ValueError("evidence block_resolution does not match geometry")
    if ownership.block_resolution != geometry.config.block_resolution:
        raise ValueError("ownership block_resolution does not match geometry")
    if (entity_posteriors is None) != (semantic_fusion is None):
        raise ValueError("entity_posteriors and semantic_fusion must be supplied together")
    if entity_semantics is not None and semantic_fusion is not None:
        raise ValueError("entity_semantics and semantic_fusion are mutually exclusive")
    if semantic_fusion is not None and not isinstance(
        semantic_fusion,
        SemanticFusionConfig,
    ):
        raise TypeError("semantic_fusion must be SemanticFusionConfig or None")
    normalized_valid_semantic_ids = _validated_semantic_id_set(valid_semantic_ids)
    normalized_entity_semantics = _validated_entity_semantics(entity_semantics)
    normalized_entity_posteriors = _validated_entity_posteriors(
        entity_posteriors,
        normalized_valid_semantic_ids,
    )
    raw_mesh = geometry.extract_mesh(weight_threshold=weight_threshold)
    vertices = raw_mesh.vertex.positions.numpy()
    triangles = raw_mesh.triangle.indices.numpy()
    if "colors" in raw_mesh.vertex:
        colors = raw_mesh.vertex.colors.numpy()
    else:
        colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)

    vertex_count = vertices.shape[0]
    semantic_ids = np.zeros(vertex_count, dtype=np.int64)
    entity_ids = np.zeros(vertex_count, dtype=np.int64)
    semantic_confidence = np.zeros(vertex_count, dtype=np.float32)
    ownership_confidence = np.zeros(vertex_count, dtype=np.float32)
    for index, vertex in enumerate(vertices):
        voxel_key = point_to_voxel(vertex, geometry.config.voxel_size_m)
        semantic_candidates = evidence.semantic_candidates(voxel_key)
        if normalized_valid_semantic_ids is not None and any(
            candidate.label_id not in normalized_valid_semantic_ids
            for candidate in semantic_candidates
        ):
            raise ValueError("semantic evidence class is outside frozen vocabulary")
        if semantic_candidates:
            strongest = max(
                semantic_candidates,
                key=lambda candidate: (candidate.support, -candidate.label_id),
            )
            total_support = sum(candidate.support for candidate in semantic_candidates)
            semantic_ids[index] = _int64_identifier(
                strongest.label_id,
                "semantic evidence label ID",
                minimum=0,
            )
            if total_support > 0.0:
                semantic_confidence[index] = strongest.support / total_support
        owner = ownership.owner_of(voxel_key)
        if owner is not None:
            owner_entity_id = _int64_identifier(
                owner.entity_id,
                "owner entity ID",
                minimum=1,
            )
            entity_ids[index] = owner_entity_id
            ownership_confidence[index] = owner.confidence
            if normalized_entity_semantics is not None:
                current_semantics = normalized_entity_semantics.get(owner_entity_id)
                if current_semantics is not None and current_semantics[0] > 0:
                    semantic_ids[index] = current_semantics[0]
                    semantic_confidence[index] = current_semantics[1]
            elif normalized_entity_posteriors is not None:
                current_posterior = normalized_entity_posteriors.get(owner_entity_id)
                if current_posterior:
                    assert semantic_fusion is not None
                    fused = fuse_semantics(
                        tuple(
                            (candidate.label_id, candidate.support)
                            for candidate in semantic_candidates
                        ),
                        current_posterior,
                        owner.confidence,
                        semantic_fusion,
                    )
                    semantic_ids[index] = _int64_identifier(
                        fused.semantic_id,
                        "fused semantic ID",
                        minimum=0,
                    )
                    semantic_confidence[index] = fused.confidence

    return canonicalize_labeled_mesh(
        LabeledMesh(
            vertices_xyz=vertices,
            triangles=triangles,
            colors_rgb=colors,
            semantic_ids=semantic_ids,
            entity_ids=entity_ids,
            semantic_confidence=semantic_confidence,
            ownership_confidence=ownership_confidence,
        )
    )


def write_labeled_mesh(path: str | Path, mesh: LabeledMesh) -> None:
    if not isinstance(mesh, LabeledMesh):
        raise TypeError("mesh must be a LabeledMesh")
    destination = Path(path)
    if destination.suffix.lower() != ".ply":
        raise ValueError("labeled mesh path must end in .ply")
    destination.parent.mkdir(parents=True, exist_ok=True)
    vertex_data = np.empty(
        mesh.vertices_xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_id", "i4"),
            ("entity_id", "i4"),
            ("semantic_confidence", "f4"),
            ("ownership_confidence", "f4"),
        ],
    )
    vertex_data["x"] = mesh.vertices_xyz[:, 0]
    vertex_data["y"] = mesh.vertices_xyz[:, 1]
    vertex_data["z"] = mesh.vertices_xyz[:, 2]
    colors = np.rint(np.clip(mesh.colors_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["semantic_id"] = mesh.semantic_ids.astype(np.int32)
    vertex_data["entity_id"] = mesh.entity_ids.astype(np.int32)
    vertex_data["semantic_confidence"] = mesh.semantic_confidence
    vertex_data["ownership_confidence"] = mesh.ownership_confidence
    face_data = np.empty(
        mesh.triangles.shape[0],
        dtype=[("vertex_indices", "i4", (3,))],
    )
    face_data["vertex_indices"] = mesh.triangles.astype(np.int32)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".ply",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        PlyData(
            [
                PlyElement.describe(vertex_data, "vertex"),
                PlyElement.describe(face_data, "face"),
            ],
            text=False,
        ).write(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
