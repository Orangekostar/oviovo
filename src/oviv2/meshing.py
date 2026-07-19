from __future__ import annotations

from dataclasses import dataclass, fields
import os
from pathlib import Path
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement

from src.oviv2.addressing import point_to_voxel
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore


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


def derive_labeled_mesh(
    geometry: SparseTsdfVolume,
    evidence: SparseEvidenceStore,
    ownership: ReversibleOwnershipStore,
    *,
    weight_threshold: float = 1.0,
) -> LabeledMesh:
    if evidence.config.block_resolution != geometry.config.block_resolution:
        raise ValueError("evidence block_resolution does not match geometry")
    if ownership.block_resolution != geometry.config.block_resolution:
        raise ValueError("ownership block_resolution does not match geometry")
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
        if semantic_candidates:
            strongest = max(
                semantic_candidates,
                key=lambda candidate: (candidate.support, -candidate.label_id),
            )
            total_support = sum(candidate.support for candidate in semantic_candidates)
            semantic_ids[index] = strongest.label_id
            if total_support > 0.0:
                semantic_confidence[index] = strongest.support / total_support
        owner = ownership.owner_of(voxel_key)
        if owner is not None:
            entity_ids[index] = owner.entity_id
            ownership_confidence[index] = owner.confidence

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
