from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

from src.oviv2.meshing import LabeledMesh


def _finite_distance(value: Real, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real distance")
    normalized = float(value)
    if not np.isfinite(normalized) or (normalized <= 0.0 if positive else normalized < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


@dataclass(frozen=True)
class GeometrySemanticStabilizationConfig:
    maximum_transfer_distance_m: float
    novelty_tolerance_m: float = 1e-7

    def __post_init__(self) -> None:
        maximum = _finite_distance(
            self.maximum_transfer_distance_m,
            "maximum_transfer_distance_m",
            positive=True,
        )
        tolerance = _finite_distance(
            self.novelty_tolerance_m,
            "novelty_tolerance_m",
            positive=False,
        )
        if tolerance >= maximum:
            raise ValueError("novelty_tolerance_m must be below the transfer distance")
        object.__setattr__(self, "maximum_transfer_distance_m", maximum)
        object.__setattr__(self, "novelty_tolerance_m", tolerance)


@dataclass(frozen=True)
class GeometrySemanticStabilizationResult:
    mesh: LabeledMesh
    transferred_vertex_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, LabeledMesh):
            raise TypeError("mesh must be a LabeledMesh")
        if (
            not isinstance(self.transferred_vertex_count, int)
            or isinstance(self.transferred_vertex_count, bool)
            or self.transferred_vertex_count < 0
        ):
            raise ValueError("transferred_vertex_count must be a non-negative integer")


def stabilize_low_support_semantics(
    mesh: LabeledMesh,
    reference_mesh: LabeledMesh,
    config: GeometrySemanticStabilizationConfig,
) -> GeometrySemanticStabilizationResult:
    if not isinstance(mesh, LabeledMesh) or not isinstance(reference_mesh, LabeledMesh):
        raise TypeError("mesh values must be LabeledMesh instances")
    if not isinstance(config, GeometrySemanticStabilizationConfig):
        raise TypeError("config must be GeometrySemanticStabilizationConfig")
    if not len(mesh.vertices_xyz) or not len(reference_mesh.vertices_xyz):
        return GeometrySemanticStabilizationResult(mesh, 0)

    distances, nearest = cKDTree(reference_mesh.vertices_xyz).query(
        mesh.vertices_xyz,
        k=1,
        workers=-1,
    )
    distances = np.asarray(distances, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    transfer = (
        (distances > config.novelty_tolerance_m)
        & (distances <= config.maximum_transfer_distance_m)
    )
    if not np.any(transfer):
        return GeometrySemanticStabilizationResult(mesh, 0)

    semantic_ids = mesh.semantic_ids.copy()
    semantic_confidence = mesh.semantic_confidence.copy()
    semantic_ids[transfer] = reference_mesh.semantic_ids[nearest[transfer]]
    semantic_confidence[transfer] = reference_mesh.semantic_confidence[nearest[transfer]]
    stabilized = replace(
        mesh,
        semantic_ids=semantic_ids,
        semantic_confidence=semantic_confidence,
    )
    return GeometrySemanticStabilizationResult(stabilized, int(np.sum(transfer)))
