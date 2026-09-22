"""Complete native-leaf partition hypotheses on immutable surface rows."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .assets import sha256_file
from .contracts import atomic_write_json


def _readonly(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SurfaceGraph:
    edges: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    source: str

    def __post_init__(self) -> None:
        edges = _readonly(self.edges, np.int64)
        normals = _readonly(self.normals, np.float64)
        valid = _readonly(self.normal_valid, bool)
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError("surface edges must be Ex2")
        if normals.ndim != 2 or normals.shape[1:] != (3,) or valid.shape != (len(normals),):
            raise ValueError("surface normals and validity must align")
        if edges.size and (
            np.any(edges < 0)
            or np.any(edges >= len(normals))
            or np.any(edges[:, 0] >= edges[:, 1])
        ):
            raise ValueError("surface edges must be ordered valid row pairs")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "normal_valid", valid)


def _face_edges(faces: np.ndarray, row_count: int) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("surface faces must be Fx3")
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64)
    if np.any(triangles < 0) or np.any(triangles >= row_count):
        raise ValueError("surface faces contain invalid row indices")
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def _mutual_knn_edges(
    xyz: np.ndarray,
    *,
    neighbors: int = 8,
    maximum_distance: float = 0.03,
    chunk_size: int = 65536,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    row_count = len(xyz)
    if row_count < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(xyz)
    directed_chunks: list[np.ndarray] = []
    query_k = min(neighbors + 1, row_count)
    for start in range(0, row_count, chunk_size):
        stop = min(row_count, start + chunk_size)
        distances, indices = tree.query(
            xyz[start:stop],
            k=query_k,
            distance_upper_bound=maximum_distance,
            workers=1,
        )
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        codes: list[int] = []
        for offset, (row_distances, row_indices) in enumerate(
            zip(distances, indices, strict=True)
        ):
            source = start + offset
            candidates = sorted(
                (
                    (float(distance), int(target))
                    for distance, target in zip(
                        np.atleast_1d(row_distances),
                        np.atleast_1d(row_indices),
                        strict=True,
                    )
                    if target < row_count
                    and target != source
                    and distance <= maximum_distance
                ),
                key=lambda item: (item[0], item[1]),
            )[:neighbors]
            codes.extend(source * row_count + target for _, target in candidates)
        if codes:
            directed_chunks.append(np.asarray(codes, dtype=np.int64))
    if not directed_chunks:
        return np.empty((0, 2), dtype=np.int64)
    directed = np.unique(np.concatenate(directed_chunks))
    sources = directed // row_count
    targets = directed % row_count
    reversed_codes = targets * row_count + sources
    positions = np.searchsorted(directed, reversed_codes)
    positions = np.minimum(positions, len(directed) - 1)
    mutual = directed[positions] == reversed_codes
    edges = np.column_stack((sources[mutual], targets[mutual]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _complete_normals(
    xyz: np.ndarray, native_normals: np.ndarray, native_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    normals = np.asarray(native_normals, dtype=np.float64).copy()
    valid = np.asarray(native_valid, dtype=bool).copy()
    if normals.shape != xyz.shape or valid.shape != (len(xyz),):
        raise ValueError("native normals and validity must align with surface rows")
    norms = np.linalg.norm(normals, axis=1)
    valid &= np.isfinite(normals).all(axis=1) & (norms > 0.0)
    normals[valid] /= norms[valid, None]
    missing = np.flatnonzero(~valid)
    if len(missing) and len(xyz) >= 3:
        tree = cKDTree(xyz)
        count = min(16, len(xyz))
        _distances, indices = tree.query(xyz[missing], k=count, workers=1)
        if count == 1:
            indices = indices[:, None]
        for row, neighbor_rows in zip(missing, indices, strict=True):
            points = xyz[np.asarray(neighbor_rows, dtype=np.int64)]
            centered = points - points.mean(axis=0)
            covariance = centered.T @ centered / max(len(points), 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if np.isfinite(eigenvalues).all() and eigenvalues[-2] > 1e-12:
                normal = eigenvectors[:, 0]
                pivot = int(np.argmax(np.abs(normal)))
                if normal[pivot] < 0.0:
                    normal = -normal
                normals[row] = normal / np.linalg.norm(normal)
                valid[row] = True
    normals[~valid] = 0.0
    return normals, valid


def build_surface_graph(
    surface_xyz: Any,
    surface_faces: Any,
    surface_normals: Any,
    normal_valid: Any,
) -> SurfaceGraph:
    xyz = np.asarray(surface_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError("surface_xyz must be a finite Nx3 matrix")
    edges = _face_edges(np.asarray(surface_faces), len(xyz))
    source = "MESH_FACES"
    if not len(edges):
        edges = _mutual_knn_edges(xyz)
        source = "MUTUAL_8NN_0.03M"
    normals, valid = _complete_normals(
        xyz, np.asarray(surface_normals), np.asarray(normal_valid)
    )
    return SurfaceGraph(edges, normals, valid, source)


@dataclass(frozen=True)
class NativeLeaf:
    leaf_id: int
    owner_id: int
    segment_label: int
    source_rows: np.ndarray
    point_count: int
    min_source_row: int
    centroid: np.ndarray

    def __post_init__(self) -> None:
        rows = _readonly(self.source_rows, np.int64)
        centroid = _readonly(self.centroid, np.float64)
        if rows.ndim != 1 or not len(rows) or np.any(rows < 0):
            raise ValueError("leaf source rows must be nonempty and nonnegative")
        if self.point_count != len(rows) or self.min_source_row != int(rows.min()):
            raise ValueError("leaf size and source-row identity do not align")
        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            raise ValueError("leaf centroid must be a finite 3-vector")
        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(self, "centroid", centroid)


@dataclass(frozen=True)
class LeafContact:
    left_leaf: int
    right_leaf: int
    edge_count: int
    mean_distance: float
    mean_abs_normal_dot: float
    normal_available: bool


@dataclass(frozen=True)
class NativeLeaves:
    row_leaf_ids: np.ndarray
    resolved_segment_labels: np.ndarray
    leaves: tuple[NativeLeaf, ...]
    contacts: tuple[LeafContact, ...]

    def __post_init__(self) -> None:
        row_ids = _readonly(self.row_leaf_ids, np.int64)
        segments = _readonly(self.resolved_segment_labels, np.int64)
        if row_ids.shape != segments.shape or row_ids.ndim != 1:
            raise ValueError("leaf and segment rows must align")
        if len(row_ids) and set(np.unique(row_ids)) != set(range(len(self.leaves))):
            raise ValueError("leaf IDs must be contiguous")
        object.__setattr__(self, "row_leaf_ids", row_ids)
        object.__setattr__(self, "resolved_segment_labels", segments)


def _resolve_aliases(labels: np.ndarray, alias_table: Sequence[tuple[int, int]]) -> np.ndarray:
    aliases = {int(old): int(new) for old, new in alias_table}
    if len(aliases) != len(alias_table):
        raise ValueError("segment aliases must have unique sources")
    resolved = labels.copy()
    cache: dict[int, int] = {}
    for label in np.unique(labels):
        current = int(label)
        trail: set[int] = set()
        while current in aliases:
            if current in trail:
                raise ValueError("segment alias table contains a cycle")
            trail.add(current)
            current = aliases[current]
        cache[int(label)] = current
    for source, target in cache.items():
        resolved[labels == source] = target
    return resolved


def build_native_leaves(
    surface_xyz: Any,
    original_owner: Any,
    segment_labels: Any,
    *,
    alias_table: Sequence[tuple[int, int]],
    surface_graph: SurfaceGraph,
) -> NativeLeaves:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    xyz = np.asarray(surface_xyz, dtype=np.float64)
    owners = np.asarray(original_owner, dtype=np.int64)
    segments = np.asarray(segment_labels, dtype=np.int64)
    row_count = len(xyz)
    if (
        xyz.shape != (row_count, 3)
        or owners.shape != (row_count,)
        or segments.shape != (row_count,)
        or len(surface_graph.normals) != row_count
    ):
        raise ValueError("native surface inputs must align")
    resolved = _resolve_aliases(segments, alias_table)
    edges = surface_graph.edges
    same_pair = (
        (owners[edges[:, 0]] == owners[edges[:, 1]])
        & (resolved[edges[:, 0]] == resolved[edges[:, 1]])
    ) if len(edges) else np.empty(0, dtype=bool)
    joined = edges[same_pair]
    rows = np.concatenate((joined[:, 0], joined[:, 1])) if len(joined) else np.empty(0, dtype=np.int64)
    columns = np.concatenate((joined[:, 1], joined[:, 0])) if len(joined) else np.empty(0, dtype=np.int64)
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(row_count, row_count),
    ).tocsr()
    _count, component = connected_components(adjacency, directed=False)
    minimum_rows = np.full(int(component.max()) + 1, row_count, dtype=np.int64)
    np.minimum.at(minimum_rows, component, np.arange(row_count, dtype=np.int64))
    component_order = np.argsort(minimum_rows, kind="stable")
    remap = np.empty(len(component_order), dtype=np.int64)
    remap[component_order] = np.arange(len(component_order))
    row_leaf_ids = remap[component]
    order = np.argsort(row_leaf_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(row_leaf_ids[order])) + 1
    row_groups = np.split(order, boundaries)
    leaves = tuple(
        NativeLeaf(
            leaf_id=leaf_id,
            owner_id=int(owners[group[0]]),
            segment_label=int(resolved[group[0]]),
            source_rows=group,
            point_count=len(group),
            min_source_row=int(group.min()),
            centroid=xyz[group].mean(axis=0),
        )
        for leaf_id, group in enumerate(row_groups)
    )

    if len(edges):
        left = row_leaf_ids[edges[:, 0]]
        right = row_leaf_ids[edges[:, 1]]
        cross = left != right
        cross_edges = edges[cross]
        left = left[cross]
        right = right[cross]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        codes = low * len(leaves) + high
        unique_codes, inverse, counts = np.unique(
            codes, return_inverse=True, return_counts=True
        )
        distances = np.linalg.norm(
            xyz[cross_edges[:, 0]] - xyz[cross_edges[:, 1]], axis=1
        )
        distance_sums = np.bincount(inverse, weights=distances)
        normal_mask = (
            surface_graph.normal_valid[cross_edges[:, 0]]
            & surface_graph.normal_valid[cross_edges[:, 1]]
        )
        dots = np.abs(
            np.sum(
                surface_graph.normals[cross_edges[:, 0]]
                * surface_graph.normals[cross_edges[:, 1]],
                axis=1,
            )
        )
        normal_counts = np.bincount(inverse, weights=normal_mask.astype(float))
        normal_sums = np.bincount(inverse, weights=dots * normal_mask)
        contacts = tuple(
            LeafContact(
                left_leaf=int(code // len(leaves)),
                right_leaf=int(code % len(leaves)),
                edge_count=int(counts[index]),
                mean_distance=float(distance_sums[index] / counts[index]),
                mean_abs_normal_dot=(
                    float(normal_sums[index] / normal_counts[index])
                    if normal_counts[index]
                    else 0.0
                ),
                normal_available=bool(normal_counts[index]),
            )
            for index, code in enumerate(unique_codes)
        )
    else:
        contacts = ()
    return NativeLeaves(row_leaf_ids, resolved, leaves, contacts)


@dataclass(frozen=True)
class FrameLeafEvidence:
    frame_id: int
    pixel_leaf_ids: np.ndarray
    pixel_entity_ids: np.ndarray
    observed_pixels: np.ndarray
    dominant_pixels: np.ndarray
    entity_by_leaf: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            _readonly(self.pixel_leaf_ids, np.int64),
            _readonly(self.pixel_entity_ids, np.int64),
            _readonly(self.observed_pixels, np.int64),
            _readonly(self.dominant_pixels, np.int64),
            _readonly(self.entity_by_leaf, np.int64),
        )
        if arrays[0].shape != arrays[1].shape or arrays[0].ndim != 1:
            raise ValueError("frame pixel leaf/entity rows must align")
        if not (arrays[2].shape == arrays[3].shape == arrays[4].shape):
            raise ValueError("frame leaf summaries must align")
        for name, value in zip(
            (
                "pixel_leaf_ids",
                "pixel_entity_ids",
                "observed_pixels",
                "dominant_pixels",
                "entity_by_leaf",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)


def select_geometry_frames(frames: Sequence[Any], *, limit: int = 32) -> tuple[Any, ...]:
    """Select ascending, exact-pose-distinct frames with fixed even subsampling."""

    if limit <= 0:
        raise ValueError("geometry frame limit must be positive")
    ordered = sorted(frames, key=lambda frame: int(frame.frame_id))
    unique = []
    seen: set[bytes] = set()
    for frame in ordered:
        pose = np.asarray(frame.pose_c2w, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("geometry frame pose must be a finite 4x4 matrix")
        identity = pose.tobytes(order="C")
        if identity not in seen:
            seen.add(identity)
            unique.append(frame)
    if len(unique) <= limit:
        return tuple(unique)
    indices = np.rint(np.linspace(0, len(unique) - 1, limit)).astype(np.int64)
    return tuple(unique[int(index)] for index in indices)


def build_frame_leaf_evidence(
    frame_id: int,
    pixel_leaf_ids: Any,
    local_entity_ids: Any,
    *,
    leaf_count: int,
    minimum_pixels: int = 16,
    dominance: float = 0.6,
) -> FrameLeafEvidence:
    leaves = np.asarray(pixel_leaf_ids, dtype=np.int64)
    entities = np.asarray(local_entity_ids, dtype=np.int64)
    if leaves.ndim != 1 or entities.shape != leaves.shape:
        raise ValueError("frame leaf and entity pixels must align")
    if leaf_count <= 0 or np.any(leaves < 0) or np.any(leaves >= leaf_count):
        raise ValueError("frame pixel leaf IDs are invalid")
    if np.any(entities < 0) or minimum_pixels <= 0 or not 0.0 <= dominance <= 1.0:
        raise ValueError("frame entity settings are invalid")
    observed = np.bincount(leaves, minlength=leaf_count).astype(np.int64)
    dominant_pixels = np.zeros(leaf_count, dtype=np.int64)
    entity_by_leaf = np.zeros(leaf_count, dtype=np.int64)
    positive = entities > 0
    if np.any(positive):
        pairs, counts = np.unique(np.column_stack((leaves[positive], entities[positive])),
                                  axis=0, return_counts=True)
        np.maximum.at(dominant_pixels, pairs[:, 0], counts)
        winners = pairs[counts == dominant_pixels[pairs[:, 0]]]
        # np.unique orders (leaf, entity); the first tied entity is the smallest.
        winners = winners[np.r_[True, winners[1:, 0] != winners[:-1, 0]]]
        ids = winners[:, 0]
        usable = (observed[ids] >= minimum_pixels) & (dominant_pixels[ids] >= dominance * observed[ids])
        entity_by_leaf[ids[usable]] = winners[usable, 1]
    return FrameLeafEvidence(
        int(frame_id), leaves, entities, observed, dominant_pixels, entity_by_leaf
    )


def project_frame_leaf_evidence(
    frame_id: int,
    surface_xyz: Any,
    row_leaf_ids: Any,
    pose_c2w: Any,
    intrinsics: Any,
    depth_m: Any,
    local_entity_raster: Any,
    *,
    depth_tolerance: float = 0.05,
) -> FrameLeafEvidence:
    from .static_regions import project_visible_rows

    xyz = np.asarray(surface_xyz, dtype=np.float64)
    leaf_ids = np.asarray(row_leaf_ids, dtype=np.int64)
    pose = np.asarray(pose_c2w, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    entities = np.asarray(local_entity_raster, dtype=np.int64)
    if depth.ndim != 2 or entities.shape != depth.shape:
        raise ValueError("depth and local entity rasters must align")
    if xyz.shape != (len(leaf_ids), 3) or pose.shape != (4, 4) or camera_matrix.shape != (3, 3):
        raise ValueError("projection inputs have invalid dimensions")
    winners, pixels = project_visible_rows(xyz, pose, camera_matrix, depth, depth_tolerance=depth_tolerance)
    return build_frame_leaf_evidence(
        frame_id,
        leaf_ids[winners],
        entities.reshape(-1)[pixels],
        leaf_count=int(leaf_ids.max()) + 1,
    )


@dataclass(frozen=True)
class ConflictGroup:
    group_id: str
    owner_ids: tuple[int, ...]
    leaf_ids: tuple[int, ...]
    conflict_strength: float


def _known_pair_counts(
    left: int, right: int, frames: Sequence[FrameLeafEvidence]
) -> tuple[int, int]:
    known = same = 0
    for frame in frames:
        left_entity = int(frame.entity_by_leaf[left])
        right_entity = int(frame.entity_by_leaf[right])
        if left_entity > 0 and right_entity > 0:
            known += 1
            same += int(left_entity == right_entity)
    return known, same


def build_conflict_groups(
    leaves: NativeLeaves,
    frames: Sequence[FrameLeafEvidence],
    *,
    maximum_groups: int = 64,
) -> tuple[tuple[ConflictGroup, ...], tuple[int, ...]]:
    owners = sorted({leaf.owner_id for leaf in leaves.leaves if leaf.owner_id > 0})
    adjacent_pairs: set[tuple[int, int]] = set()
    for contact in leaves.contacts:
        left_owner = leaves.leaves[contact.left_leaf].owner_id
        right_owner = leaves.leaves[contact.right_leaf].owner_id
        if left_owner > 0 and right_owner > 0 and left_owner != right_owner:
            adjacent_pairs.add(tuple(sorted((left_owner, right_owner))))
    candidates: list[tuple[float, tuple[int, int]]] = []
    for owner_pair in adjacent_pairs:
        cross_known = cross_same = within_known = within_different = 0
        for contact in leaves.contacts:
            leaf_pair = (contact.left_leaf, contact.right_leaf)
            contact_owners = (
                leaves.leaves[contact.left_leaf].owner_id,
                leaves.leaves[contact.right_leaf].owner_id,
            )
            known, same = _known_pair_counts(*leaf_pair, frames)
            if set(contact_owners) == set(owner_pair) and contact_owners[0] != contact_owners[1]:
                cross_known += known
                cross_same += same
            elif contact_owners[0] == contact_owners[1] and contact_owners[0] in owner_pair:
                within_known += known
                within_different += known - same
        strength = (
            (cross_same / cross_known if cross_known else 0.0)
            + (within_different / within_known if within_known else 0.0)
        )
        candidates.append((strength, owner_pair))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    assigned: set[int] = set()
    accepted: list[tuple[float, tuple[int, ...]]] = []
    for strength, owner_pair in candidates:
        if assigned.isdisjoint(owner_pair):
            assigned.update(owner_pair)
            accepted.append((strength, owner_pair))
    accepted.extend((0.0, (owner,)) for owner in owners if owner not in assigned)
    accepted.sort(key=lambda row: (-row[0], row[1]))
    selected = accepted[:maximum_groups]
    groups = tuple(
        ConflictGroup(
            group_id="owners:" + "+".join(map(str, owner_ids)),
            owner_ids=owner_ids,
            leaf_ids=tuple(
                leaf.leaf_id
                for leaf in leaves.leaves
                if leaf.owner_id in set(owner_ids)
            ),
            conflict_strength=float(strength),
        )
        for strength, owner_ids in selected
    )
    covered = {owner for group in groups for owner in group.owner_ids}
    return groups, tuple(owner for owner in owners if owner not in covered)


@dataclass(frozen=True)
class PartitionHypothesis:
    hypothesis_id: str
    scene_id: str
    group_id: str
    owner_ids: tuple[int, ...]
    leaf_ids: tuple[int, ...]
    components: tuple[int, ...]
    kind: str
    frame_id: int | None
    known_pixel_support: int
    uncertain_fill_fraction: float

    @property
    def component_count(self) -> int:
        return len(set(self.components))

    @classmethod
    def create(
        cls,
        scene_id: str,
        group: ConflictGroup,
        leaves: NativeLeaves,
        kind: str,
        components: Sequence[int],
        *,
        frame_id: int | None = None,
        known_pixel_support: int = 0,
        uncertain_fill_fraction: float = 0.0,
    ) -> PartitionHypothesis:
        leaf_ids = tuple(group.leaf_ids)
        labels = tuple(int(value) for value in components)
        if len(labels) != len(leaf_ids) or any(value < 0 for value in labels):
            raise ValueError("partition must assign every group leaf")
        old_components = sorted(
            set(labels),
            key=lambda component: min(
                leaves.leaves[leaf_id].min_source_row
                for leaf_id, label in zip(leaf_ids, labels, strict=True)
                if label == component
            ),
        )
        remap = {old: new for new, old in enumerate(old_components)}
        canonical = tuple(remap[value] for value in labels)
        payload = f"{scene_id}|{group.group_id}|{canonical}"
        identity = hashlib.sha256(payload.encode()).hexdigest()
        return cls(
            identity,
            scene_id,
            group.group_id,
            tuple(group.owner_ids),
            leaf_ids,
            canonical,
            kind,
            frame_id,
            int(known_pixel_support),
            float(uncertain_fill_fraction),
        )


def _graph_partition(
    group: ConflictGroup,
    leaves: NativeLeaves,
    frames: Sequence[FrameLeafEvidence],
    clusters: int,
) -> tuple[int, ...] | None:
    from sklearn.cluster import KMeans

    leaf_ids = tuple(group.leaf_ids)
    count = len(leaf_ids)
    if count < clusters or count > 1024:
        return None
    local = {leaf_id: index for index, leaf_id in enumerate(leaf_ids)}
    affinity = np.zeros((count, count), dtype=np.float64)
    for contact in leaves.contacts:
        if contact.left_leaf not in local or contact.right_leaf not in local:
            continue
        known, same = _known_pair_counts(
            contact.left_leaf, contact.right_leaf, frames
        )
        same_fraction = same / known if known else 0.5
        normal = contact.mean_abs_normal_dot if contact.normal_available else 1.0
        weight = (
            math.exp(-contact.mean_distance / 0.03)
            * (0.1 + 0.9 * normal)
            * (0.25 + 0.75 * same_fraction)
        )
        left = local[contact.left_leaf]
        right = local[contact.right_leaf]
        affinity[left, right] = affinity[right, left] = weight
    degree = affinity.sum(axis=1)
    inverse = np.zeros_like(degree)
    inverse[degree > 0.0] = 1.0 / np.sqrt(degree[degree > 0.0])
    laplacian = np.diag((degree > 0.0).astype(float)) - (
        inverse[:, None] * affinity * inverse[None, :]
    )
    _values, vectors = np.linalg.eigh(laplacian)
    embedding = vectors[:, :clusters]
    for column in range(embedding.shape[1]):
        pivot = int(np.argmax(np.abs(embedding[:, column])))
        if embedding[pivot, column] < 0.0:
            embedding[:, column] *= -1.0
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = np.divide(embedding, norms, out=np.zeros_like(embedding), where=norms > 0.0)
    labels = KMeans(n_clusters=clusters, random_state=17, n_init=1).fit_predict(embedding)
    if len(set(map(int, labels))) != clusters:
        return None
    return tuple(map(int, labels))


def generate_complete_hypotheses(
    scene_id: str,
    group: ConflictGroup,
    leaves: NativeLeaves,
    frames: Sequence[FrameLeafEvidence],
    *,
    maximum_hypotheses: int = 8,
) -> tuple[PartitionHypothesis, ...]:
    owner_component = {
        owner: index
        for index, owner in enumerate(
            sorted(
                group.owner_ids,
                key=lambda owner: min(
                    leaf.min_source_row
                    for leaf in leaves.leaves
                    if leaf.owner_id == owner
                ),
            )
        )
    }
    original = tuple(
        owner_component[leaves.leaves[leaf_id].owner_id] for leaf_id in group.leaf_ids
    )
    ordered: list[PartitionHypothesis] = [
        PartitionHypothesis.create(scene_id, group, leaves, "ORIGINAL", original)
    ]
    if len(group.owner_ids) == 2:
        ordered.append(
            PartitionHypothesis.create(
                scene_id, group, leaves, "MERGE", (0,) * len(group.leaf_ids)
            )
        )
    for clusters in (2, 3):
        partition = _graph_partition(group, leaves, frames, clusters)
        if partition is not None:
            ordered.append(
                PartitionHypothesis.create(
                    scene_id,
                    group,
                    leaves,
                    f"GRAPH_SPLIT{clusters}",
                    partition,
                )
            )
    frame_rows: list[PartitionHypothesis] = []
    for frame in frames:
        known = {
            leaf_id: int(frame.entity_by_leaf[leaf_id])
            for leaf_id in group.leaf_ids
            if frame.entity_by_leaf[leaf_id] > 0
        }
        entity_ids = sorted(set(known.values()))
        if len(entity_ids) not in (2, 3):
            continue
        entity_centroids: dict[int, np.ndarray] = {}
        entity_min_rows: dict[int, int] = {}
        for entity in entity_ids:
            entity_leaves = [leaf_id for leaf_id, value in known.items() if value == entity]
            weights = np.asarray(
                [leaves.leaves[leaf_id].point_count for leaf_id in entity_leaves],
                dtype=np.float64,
            )
            points = np.stack([leaves.leaves[leaf_id].centroid for leaf_id in entity_leaves])
            entity_centroids[entity] = np.average(points, axis=0, weights=weights)
            entity_min_rows[entity] = min(
                leaves.leaves[leaf_id].min_source_row for leaf_id in entity_leaves
            )
        assignments: list[int] = []
        unknown_count = 0
        for leaf_id in group.leaf_ids:
            if leaf_id in known:
                assignments.append(known[leaf_id])
                continue
            unknown_count += 1
            centroid = leaves.leaves[leaf_id].centroid
            assignments.append(
                min(
                    entity_ids,
                    key=lambda entity: (
                        float(np.linalg.norm(centroid - entity_centroids[entity])),
                        entity_min_rows[entity],
                    ),
                )
            )
        support = sum(frame.dominant_pixels[leaf_id] for leaf_id in known)
        frame_rows.append(
            PartitionHypothesis.create(
                scene_id,
                group,
                leaves,
                "FRAME_ENTITY",
                assignments,
                frame_id=frame.frame_id,
                known_pixel_support=int(support),
                uncertain_fill_fraction=unknown_count / len(group.leaf_ids),
            )
        )
    frame_rows.sort(key=lambda row: (-row.known_pixel_support, row.frame_id))
    ordered.extend(frame_rows)
    unique: list[PartitionHypothesis] = []
    seen: set[tuple[int, ...]] = set()
    for hypothesis in ordered:
        if hypothesis.components in seen:
            continue
        seen.add(hypothesis.components)
        unique.append(hypothesis)
        if len(unique) == maximum_hypotheses:
            break
    return tuple(unique)


@dataclass(frozen=True)
class FinalOwnership:
    owner_ids: np.ndarray
    ancestry: Mapping[int, tuple[int, ...]]
    selected_hypotheses: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_ids", _readonly(self.owner_ids, np.int64))
        object.__setattr__(self, "ancestry", MappingProxyType(dict(self.ancestry)))
        object.__setattr__(
            self,
            "selected_hypotheses",
            MappingProxyType(dict(self.selected_hypotheses)),
        )


def apply_selected_partitions(
    scene_id: str,
    original_owner: Any,
    leaves: NativeLeaves,
    selected: Mapping[str, PartitionHypothesis],
) -> FinalOwnership:
    original = np.asarray(original_owner, dtype=np.int64)
    if original.shape != leaves.row_leaf_ids.shape:
        raise ValueError("final ownership must align with native surface rows")
    result = original.copy()
    used = set(map(int, np.unique(original)))
    touched_leaves: set[int] = set()
    ancestry: dict[int, tuple[int, ...]] = {}
    selected_ids: dict[str, str] = {}
    for group_id in sorted(selected):
        hypothesis = selected[group_id]
        if hypothesis.scene_id != scene_id or hypothesis.group_id != group_id:
            raise ValueError("selected hypothesis scene/group identity mismatch")
        if not touched_leaves.isdisjoint(hypothesis.leaf_ids):
            raise ValueError("selected geometry groups must be disjoint")
        touched_leaves.update(hypothesis.leaf_ids)
        selected_ids[group_id] = hypothesis.hypothesis_id
        for component in range(hypothesis.component_count):
            component_leaves = [
                leaf_id
                for leaf_id, label in zip(
                    hypothesis.leaf_ids, hypothesis.components, strict=True
                )
                if label == component
            ]
            rows = np.sort(
                np.concatenate([leaves.leaves[leaf_id].source_rows for leaf_id in component_leaves])
            )
            parent_ids = tuple(sorted({leaves.leaves[leaf_id].owner_id for leaf_id in component_leaves}))
            retained = None
            if len(parent_ids) == 1:
                parent_rows = np.flatnonzero(original == parent_ids[0])
                if np.array_equal(rows, parent_rows):
                    retained = parent_ids[0]
            if retained is None:
                payload = f"{scene_id}|{group_id}|{hypothesis.hypothesis_id}|{component}"
                candidate = int.from_bytes(
                    hashlib.sha256(payload.encode()).digest()[:4], "big"
                ) & 0x7FFFFFFF
                candidate = max(candidate, 1)
                while candidate in used:
                    candidate = 1 if candidate == 0x7FFFFFFF else candidate + 1
                owner_id = candidate
            else:
                owner_id = retained
            used.add(owner_id)
            result[rows] = owner_id
            ancestry[owner_id] = parent_ids
    if np.any(result[original == 0] != 0):
        raise RuntimeError("owner0 geometry changed during partition application")
    return FinalOwnership(result, ancestry, selected_ids)


@dataclass(frozen=True)
class FinalGeometryMap:
    scene_id: str
    surface_xyz: np.ndarray
    surface_faces: np.ndarray
    owner_ids: np.ndarray
    tsdf_sha256: str
    projection_identity: str
    ancestry: Mapping[int, tuple[int, ...]]
    selected_hypotheses: Mapping[str, str]
    rank_by_owner: Mapping[int, int]

    def __post_init__(self) -> None:
        xyz = _readonly(self.surface_xyz, np.float64)
        faces = _readonly(self.surface_faces, np.int64)
        owners = _readonly(self.owner_ids, np.int64)
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or owners.shape != (len(xyz),):
            raise ValueError("final geometry rows must align")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or (
            faces.size and (np.any(faces < 0) or np.any(faces >= len(xyz)))
        ):
            raise ValueError("final geometry faces must contain valid row indices")
        if np.any(owners < 0) or not np.isfinite(xyz).all():
            raise ValueError("final geometry values must be finite and nonnegative")
        if re.fullmatch(r"[0-9a-f]{64}", self.tsdf_sha256) is None:
            raise ValueError("final geometry TSDF identity must be SHA-256")
        if not self.scene_id or not self.projection_identity:
            raise ValueError("final geometry scene/projection identity is required")
        expected_ranks = {
            int(owner): int(np.count_nonzero(owners == owner))
            for owner in np.unique(owners)
        }
        ranks = {int(owner): int(rank) for owner, rank in self.rank_by_owner.items()}
        if ranks != expected_ranks:
            raise ValueError("final geometry ranks must equal component point counts")
        object.__setattr__(self, "surface_xyz", xyz)
        object.__setattr__(self, "surface_faces", faces)
        object.__setattr__(self, "owner_ids", owners)
        object.__setattr__(self, "ancestry", MappingProxyType(dict(self.ancestry)))
        object.__setattr__(
            self,
            "selected_hypotheses",
            MappingProxyType(dict(self.selected_hypotheses)),
        )
        object.__setattr__(self, "rank_by_owner", MappingProxyType(ranks))


def finalize_geometry_map(
    scene_id: str,
    surface_xyz: Any,
    surface_faces: Any,
    original_owner: Any,
    leaves: NativeLeaves,
    selected: Mapping[str, PartitionHypothesis],
    *,
    tsdf_sha256: str,
    projection_identity: str,
) -> FinalGeometryMap:
    xyz = np.asarray(surface_xyz)
    faces = np.asarray(surface_faces)
    ownership = apply_selected_partitions(
        scene_id, original_owner, leaves, selected
    )
    ranks = {
        int(owner): int(np.count_nonzero(ownership.owner_ids == owner))
        for owner in np.unique(ownership.owner_ids)
    }
    return FinalGeometryMap(
        scene_id=scene_id,
        surface_xyz=xyz,
        surface_faces=faces,
        owner_ids=ownership.owner_ids,
        tsdf_sha256=tsdf_sha256,
        projection_identity=projection_identity,
        ancestry=ownership.ancestry,
        selected_hypotheses=ownership.selected_hypotheses,
        rank_by_owner=ranks,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_final_geometry_map(
    final: FinalGeometryMap, output_directory: Path | str
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        surface_xyz=final.surface_xyz,
        surface_faces=final.surface_faces,
        owner_ids=final.owner_ids,
    )
    arrays_path = output / "arrays.npz"
    _atomic_write_bytes(arrays_path, buffer.getvalue())
    manifest = {
        "schema_version": 1,
        "artifact_type": "OVIMAP_FINAL_GEOMETRY_MAP",
        "scene_id": final.scene_id,
        "row_count": len(final.surface_xyz),
        "face_count": len(final.surface_faces),
        "arrays": {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
        "tsdf_sha256": final.tsdf_sha256,
        "projection_identity": final.projection_identity,
        "ancestry": {
            str(owner): list(parents)
            for owner, parents in sorted(final.ancestry.items())
        },
        "selected_hypotheses": dict(sorted(final.selected_hypotheses.items())),
        "rank_by_owner": {
            str(owner): rank for owner, rank in sorted(final.rank_by_owner.items())
        },
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest
