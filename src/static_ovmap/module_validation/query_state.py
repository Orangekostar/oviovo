"""Causal query candidates, policy-local state, and acquisition accounting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _readonly(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _unit_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("acquired feature must be a finite nonempty vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("acquired feature must have nonzero norm")
    return _readonly(vector / norm, np.float64)


@dataclass(frozen=True)
class QueryCandidate:
    scene_id: str
    frame_index: int
    frame_id: int
    request_id: str
    owner_id: int
    local_entity_id: int
    majority_entity_zero: bool
    global_mask: np.ndarray
    local_mask: np.ndarray
    union_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    global_pixels: int
    local_pixels: int
    overlap_pixels: int
    valid_depth_fraction: float
    median_depth: float
    depth_iqr: float
    spherical_cells: frozenset[int]
    camera_pose: np.ndarray
    world_points: np.ndarray | None = None
    spherical_grid: tuple[int, int] = (8, 16)

    def __post_init__(self) -> None:
        global_mask = _readonly(self.global_mask, bool)
        local_mask = _readonly(self.local_mask, bool)
        union_mask = _readonly(self.union_mask, bool)
        pose = _readonly(self.camera_pose, np.float64)
        if not self.scene_id or not self.request_id:
            raise ValueError("scene and request IDs must be nonempty")
        if self.frame_index < 0 or self.owner_id <= 0 or self.local_entity_id < 0:
            raise ValueError("candidate indices and IDs are invalid")
        if global_mask.ndim != 2 or local_mask.shape != global_mask.shape:
            raise ValueError("candidate masks must be aligned 2D arrays")
        if union_mask.shape != global_mask.shape or not np.array_equal(
            union_mask, global_mask | local_mask
        ):
            raise ValueError("candidate union mask must equal local OR global")
        x1, y1, x2, y2 = (int(value) for value in self.bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("candidate native crop must be nondegenerate")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("candidate camera pose must be finite 4x4")
        counts = (self.global_pixels, self.local_pixels, self.overlap_pixels)
        if min(counts) <= 0:
            raise ValueError("candidate pixel counts must be positive")
        if self.global_pixels != int(np.count_nonzero(global_mask)):
            raise ValueError("global candidate area does not match its mask")
        if self.local_pixels != int(np.count_nonzero(local_mask)):
            raise ValueError("local candidate area does not match its mask")
        if self.overlap_pixels != int(np.count_nonzero(global_mask & local_mask)):
            raise ValueError("candidate overlap area does not match its masks")
        numeric = (self.valid_depth_fraction, self.median_depth, self.depth_iqr)
        if not np.isfinite(numeric).all() or not 0.0 <= self.valid_depth_fraction <= 1.0:
            raise ValueError("candidate depth statistics are invalid")
        cells = frozenset(int(value) for value in self.spherical_cells)
        if any(value < 0 for value in cells):
            raise ValueError("spherical cell IDs must be nonnegative")
        grid = tuple(int(value) for value in self.spherical_grid)
        if len(grid) != 2 or min(grid) <= 0:
            raise ValueError("spherical grid must contain two positive dimensions")
        world_points = None
        if self.world_points is not None:
            world_points = _readonly(self.world_points, np.float32)
            if (
                world_points.ndim != 2
                or world_points.shape[1:] != (3,)
                or not len(world_points)
                or not np.isfinite(world_points).all()
            ):
                raise ValueError("candidate world points must be a finite nonempty Nx3 matrix")
        object.__setattr__(self, "global_mask", global_mask)
        object.__setattr__(self, "local_mask", local_mask)
        object.__setattr__(self, "union_mask", union_mask)
        object.__setattr__(self, "bbox_xyxy", (x1, y1, x2, y2))
        object.__setattr__(self, "spherical_cells", cells)
        object.__setattr__(self, "camera_pose", pose)
        object.__setattr__(self, "world_points", world_points)
        object.__setattr__(self, "spherical_grid", grid)


def _request_id(
    scene_id: str,
    frame_index: int,
    frame_id: int,
    owner_id: int,
    local_entity_id: int,
) -> str:
    identity = f"{scene_id}\0{frame_index}\0{frame_id}\0{owner_id}\0{local_entity_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{scene_id}:f{frame_index}:n{frame_id}:o{owner_id}:l{local_entity_id}:{digest}"


def _spherical_cells(
    points: np.ndarray,
    grid: tuple[int, int],
    *,
    center: np.ndarray | None = None,
) -> frozenset[int]:
    if len(points) == 0:
        return frozenset()
    center = np.mean(points, axis=0) if center is None else np.asarray(center)
    rays = points - center
    norms = np.linalg.norm(rays, axis=1)
    rays = rays[norms > 1e-12] / norms[norms > 1e-12, None]
    if len(rays) == 0:
        return frozenset()
    height, width = grid
    theta = np.floor(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)) / math.pi * height)
    phi = np.floor((np.arctan2(rays[:, 1], rays[:, 0]) + math.pi) / (2 * math.pi) * width)
    theta = np.clip(theta.astype(np.int64), 0, height - 1)
    phi = np.clip(phi.astype(np.int64), 0, width - 1)
    return frozenset(int(value) for value in np.unique(theta * width + phi))


def build_query_candidates(
    scene_id: str,
    frame_index: int,
    frame_id: int,
    global_owner_map: Any,
    local_entity_map: Any,
    depth: Any,
    valid_depth_mask: Any,
    camera_pose: Any,
    points_map: Any,
    *,
    visibility_area_threshold: int = 1000,
    spherical_grid: tuple[int, int] | None = None,
    request_ids_by_owner: Mapping[int, str] | None = None,
) -> tuple[QueryCandidate, ...]:
    """Enumerate native-compatible requests without mutating policy state."""

    owners = np.asarray(global_owner_map)
    local = np.asarray(local_entity_map)
    depths = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid_depth_mask, dtype=bool)
    points = np.asarray(points_map, dtype=np.float64)
    pose = np.asarray(camera_pose, dtype=np.float64)
    if owners.ndim != 2 or local.shape != owners.shape or depths.shape != owners.shape:
        raise ValueError("owner, local-entity, and depth maps must align")
    if valid.shape != owners.shape or points.shape != (*owners.shape, 3):
        raise ValueError("valid-depth and point maps must align with owner pixels")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera pose must be finite 4x4")
    grid = (
        (max(1, owners.shape[0] // 8), max(1, owners.shape[1] // 8))
        if spherical_grid is None
        else tuple(int(value) for value in spherical_grid)
    )
    if visibility_area_threshold <= 0 or len(grid) != 2 or min(grid) <= 0:
        raise ValueError("candidate thresholds and spherical grid must be positive")

    candidates: list[QueryCandidate] = []
    for owner_id_value in np.unique(owners):
        owner_id = int(owner_id_value)
        if owner_id == 0:
            continue
        global_mask = owners == owner_id_value
        global_pixels = int(np.count_nonzero(global_mask))
        if global_pixels < 0.5 * visibility_area_threshold:
            continue
        entity_values, entity_counts = np.unique(local[global_mask], return_counts=True)
        if len(entity_values) == 0:
            continue
        majority_index = int(np.argmax(entity_counts))
        local_entity_id = int(entity_values[majority_index])
        local_mask = local == entity_values[majority_index]
        local_pixels = int(np.count_nonzero(local_mask))
        if local_pixels < visibility_area_threshold:
            continue
        overlap_pixels = int(entity_counts[majority_index])
        usable = global_mask & valid & np.isfinite(depths) & (depths > 0.0)
        valid_depth_pixels = int(np.count_nonzero(usable))
        if valid_depth_pixels < 0.1 * visibility_area_threshold:
            continue
        y_coords, x_coords = np.nonzero(global_mask)
        x1, x2 = int(x_coords.min()), int(x_coords.max())
        y1, y2 = int(y_coords.min()), int(y_coords.max())
        if x2 <= x1 or y2 <= y1:
            continue
        visible_depth = depths[usable]
        world_points = points[usable].astype(np.float32) @ pose[:3, :3].T + pose[:3, 3]
        world_points = world_points.astype(np.float32)
        finite_points = np.isfinite(world_points).all(axis=1)
        world_points = world_points[finite_points]
        if len(world_points) == 0:
            continue
        union_mask = global_mask | local_mask
        request_id = (
            _request_id(str(scene_id), frame_index, frame_id, owner_id, local_entity_id)
            if request_ids_by_owner is None
            else str(request_ids_by_owner.get(owner_id, ""))
        )
        if not request_id:
            raise ValueError(f"captured request ID is missing for owner {owner_id}")
        candidates.append(
            QueryCandidate(
                scene_id=str(scene_id),
                frame_index=int(frame_index),
                frame_id=int(frame_id),
                request_id=request_id,
                owner_id=owner_id,
                local_entity_id=local_entity_id,
                majority_entity_zero=local_entity_id == 0,
                global_mask=global_mask,
                local_mask=local_mask,
                union_mask=union_mask,
                bbox_xyxy=(x1, y1, x2, y2),
                global_pixels=global_pixels,
                local_pixels=local_pixels,
                overlap_pixels=overlap_pixels,
                valid_depth_fraction=valid_depth_pixels / float(global_pixels),
                median_depth=float(np.median(visible_depth)),
                depth_iqr=float(
                    np.percentile(visible_depth, 75) - np.percentile(visible_depth, 25)
                ),
                spherical_cells=_spherical_cells(world_points, grid),
                camera_pose=pose,
                world_points=world_points,
                spherical_grid=grid,
            )
        )
    return tuple(candidates)


@dataclass
class NativeCombineState:
    """State for the literal native combine control, including its side effects."""

    coverage_by_owner: dict[int, set[int]] = field(default_factory=dict)
    center_by_owner: dict[int, np.ndarray] = field(default_factory=dict)
    successful_overlaps_by_owner: dict[int, list[int]] = field(default_factory=dict)


def _native_bbox_center(points: np.ndarray) -> np.ndarray:
    """Match the native voxel/outlier/OBB center calculation."""

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud = cloud.voxel_down_sample(0.01)
    cloud, _ = cloud.remove_statistical_outlier(10, 2.0)
    try:
        center = np.asarray(cloud.get_oriented_bounding_box().center)
    except RuntimeError:
        box = cloud.get_axis_aligned_bounding_box()
        center = (np.asarray(box.max_bound) + np.asarray(box.min_bound)) / 2.0
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("native bounding-box center is unavailable")
    return _readonly(center, np.float64)


def native_combine_candidates(
    candidates: Sequence[QueryCandidate],
    state: NativeCombineState,
    *,
    visibility_area_threshold: int = 1000,
    maximum_retained: int = 10,
    overlap_ratio_threshold: float = 0.9,
) -> tuple[QueryCandidate, ...]:
    """Apply the historical combine gates and preserve coverage-before-area mutation."""

    if visibility_area_threshold <= 0 or maximum_retained <= 0:
        raise ValueError("native combine limits must be positive")
    if not 0.0 <= overlap_ratio_threshold <= 1.0:
        raise ValueError("native combine overlap threshold must lie in [0, 1]")
    selected: list[QueryCandidate] = []
    for candidate in sorted(candidates, key=lambda row: (row.owner_id, row.request_id)):
        owner = candidate.owner_id
        valid_depth_pixels = round(candidate.valid_depth_fraction * candidate.global_pixels)
        if owner not in state.coverage_by_owner:
            if valid_depth_pixels < 0.5 * visibility_area_threshold:
                continue
            state.coverage_by_owner[owner] = set()
            if candidate.world_points is not None:
                state.center_by_owner[owner] = _native_bbox_center(candidate.world_points)
        coverage = state.coverage_by_owner[owner]
        cells = set(candidate.spherical_cells)
        if candidate.world_points is not None and owner in state.center_by_owner:
            cells = set(
                _spherical_cells(
                    candidate.world_points,
                    candidate.spherical_grid,
                    center=state.center_by_owner[owner],
                )
            )
        if not cells:
            continue
        overlap_ratio = len(coverage & cells) / len(cells)
        if overlap_ratio > overlap_ratio_threshold:
            continue
        coverage.update(cells)
        history = state.successful_overlaps_by_owner.setdefault(owner, [])
        if len(history) >= maximum_retained:
            threshold = sorted(history)[-maximum_retained]
            if candidate.overlap_pixels <= threshold:
                continue
        history.append(candidate.overlap_pixels)
        selected.append(candidate)
    return tuple(selected)


@dataclass(frozen=True)
class AcquiredFeature:
    request_id: str
    owner_id: int
    frame_index: int
    overlap_pixels: int
    feature: np.ndarray
    spherical_cells: frozenset[int]

    def __post_init__(self) -> None:
        if not self.request_id or self.owner_id <= 0 or self.frame_index < 0:
            raise ValueError("acquired feature identity is invalid")
        if self.overlap_pixels <= 0:
            raise ValueError("acquired feature overlap must be positive")
        object.__setattr__(self, "feature", _unit_vector(self.feature))
        object.__setattr__(
            self, "spherical_cells", frozenset(int(value) for value in self.spherical_cells)
        )


@dataclass
class ObjectQueryState:
    attempts: int = 0
    successes: int = 0
    attempted_request_ids: set[str] = field(default_factory=set)
    features: list[AcquiredFeature] = field(default_factory=list)
    geometric_cells: set[int] = field(default_factory=set)
    last_success_frame: int | None = None
    best_paid_overlap: int = 0
    last_paid_pose: np.ndarray | None = None
    last_paid_frame: int | None = None
    cached_scores: np.ndarray | None = None
    lineage_available: bool = True

    @property
    def acquired_evidence_cells(self) -> set[int]:
        return set().union(*(row.spherical_cells for row in self.features)) if self.features else set()

    def retain_feature(self, feature: AcquiredFeature) -> None:
        self.features.append(feature)
        self.features.sort(
            key=lambda row: (-row.overlap_pixels, row.frame_index, row.request_id)
        )
        del self.features[10:]


@dataclass
class LogicalAcquisitionLedger:
    attempts: int = 0
    successes: int = 0
    crop_inputs: int = 0
    failures: int = 0


@dataclass
class PhysicalExecutionLedger:
    model_loads: int = 0
    model_forwards: int = 0
    cache_hits: int = 0
    crop_inputs: int = 0
    tiles: int = 0
    inference_seconds: float = 0.0


@dataclass
class QueryPolicyState:
    policy_id: str
    class_count: int
    objects: dict[int, ObjectQueryState] = field(default_factory=dict)
    aliases: dict[int, int] = field(default_factory=dict)
    logical_ledger: LogicalAcquisitionLedger = field(
        default_factory=LogicalAcquisitionLedger
    )
    blocked_reasons: set[str] = field(default_factory=set)
    _token_serial: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.policy_id or self.class_count <= 0:
            raise ValueError("policy ID and class count must be valid")

    def canonical_owner(self, owner_id: int) -> int:
        current = int(owner_id)
        trail: set[int] = set()
        while current in self.aliases:
            if current in trail:
                raise ValueError("query-state aliases contain a cycle")
            trail.add(current)
            current = self.aliases[current]
        return current

    def object_state(self, owner_id: int) -> ObjectQueryState:
        canonical = self.canonical_owner(owner_id)
        return self.objects.setdefault(canonical, ObjectQueryState())

    def successful_features(self, owner_id: int) -> tuple[AcquiredFeature, ...]:
        return tuple(self.object_state(owner_id).features)

    def _mint_token(self, request_id: str) -> _BudgetToken:
        self._token_serial += 1
        return _BudgetToken(id(self), self._token_serial, request_id)


@dataclass(frozen=True)
class AcquisitionPayload:
    feature: np.ndarray | None
    attempted_crop_inputs: int
    inference_seconds: float
    failure_reason: str | None = None
    physical_tiles: int = 0

    def __post_init__(self) -> None:
        if self.attempted_crop_inputs < 0 or self.attempted_crop_inputs > 6:
            raise ValueError("a native request attempts zero to six crop inputs")
        if not np.isfinite(self.inference_seconds) or self.inference_seconds < 0.0:
            raise ValueError("inference duration must be finite and nonnegative")
        if self.physical_tiles < 0:
            raise ValueError("physical tile count must be nonnegative")
        if self.feature is not None:
            object.__setattr__(self, "feature", _unit_vector(self.feature))


@dataclass
class _BudgetToken:
    policy_identity: int
    serial: int
    request_id: str
    used: bool = False


class FeatureStore:
    """Shared physical cache whose contents are accessible only through paid tokens."""

    def __init__(
        self,
        loader: Callable[[QueryCandidate], AcquisitionPayload],
        *,
        initial_cache: Mapping[str, AcquisitionPayload] | None = None,
        model_loads: int = 0,
    ) -> None:
        if model_loads < 0:
            raise ValueError("physical model-load count must be nonnegative")
        self._loader = loader
        self._cache = dict(initial_cache or {})
        if any(not key or not isinstance(value, AcquisitionPayload) for key, value in self._cache.items()):
            raise ValueError("initial feature cache must map request IDs to payloads")
        self.physical_ledger = PhysicalExecutionLedger(model_loads=model_loads)

    def acquire(
        self, candidate: QueryCandidate, budget_token: _BudgetToken
    ) -> AcquisitionPayload:
        if budget_token.used or budget_token.request_id != candidate.request_id:
            raise PermissionError("feature acquisition requires a fresh matching budget token")
        budget_token.used = True
        cached = self._cache.get(candidate.request_id)
        if cached is not None:
            self.physical_ledger.cache_hits += 1
            return cached
        payload = self._loader(candidate)
        if not isinstance(payload, AcquisitionPayload):
            raise TypeError("feature loader must return AcquisitionPayload")
        self.physical_ledger.model_forwards += 1
        self.physical_ledger.crop_inputs += payload.attempted_crop_inputs
        self.physical_ledger.tiles += payload.physical_tiles
        self.physical_ledger.inference_seconds += payload.inference_seconds
        if payload.feature is not None:
            self._cache[candidate.request_id] = payload
        return payload


@dataclass(frozen=True)
class AcquisitionResult:
    request_id: str
    owner_id: int
    success: bool
    attempted_crop_inputs: int
    physical_cache_hit: bool
    failure_reason: str | None


def cumulative_quota(
    budget: int, *, frame_index: int, frame_count: int, spent: int
) -> int:
    if budget < 0 or frame_count <= 0 or not 0 <= frame_index < frame_count or spent < 0:
        raise ValueError("cumulative allowance inputs are invalid")
    allowance = math.floor(budget * (frame_index + 1) / frame_count)
    return max(0, allowance - spent)


def observe_geometric_candidates(
    state: QueryPolicyState, candidates: Sequence[QueryCandidate]
) -> None:
    """Admit free current geometry without treating it as acquired visual evidence."""

    for candidate in candidates:
        state.object_state(candidate.owner_id).geometric_cells.update(
            candidate.spherical_cells
        )


def dispatch_frame_batch(
    state: QueryPolicyState,
    feature_store: FeatureStore,
    ranked_candidates: Sequence[QueryCandidate],
    *,
    quota: int,
) -> tuple[AcquisitionResult, ...]:
    """Debit a full frame batch, retrieve it, then admit successes at one barrier."""

    if quota < 0:
        raise ValueError("frame quota must be nonnegative")
    batch: list[QueryCandidate] = []
    seen: set[str] = set()
    for candidate in ranked_candidates:
        object_state = state.object_state(candidate.owner_id)
        if (
            candidate.request_id in seen
            or candidate.request_id in object_state.attempted_request_ids
        ):
            continue
        seen.add(candidate.request_id)
        batch.append(candidate)
        if len(batch) == quota:
            break
    if not batch:
        return ()

    tokens: list[_BudgetToken] = []
    for candidate in batch:
        object_state = state.object_state(candidate.owner_id)
        object_state.attempts += 1
        object_state.attempted_request_ids.add(candidate.request_id)
        object_state.best_paid_overlap = max(
            object_state.best_paid_overlap, candidate.overlap_pixels
        )
        object_state.last_paid_pose = _readonly(candidate.camera_pose, np.float64)
        object_state.last_paid_frame = candidate.frame_index
        state.logical_ledger.attempts += 1
        tokens.append(state._mint_token(candidate.request_id))

    retrieved: list[tuple[QueryCandidate, AcquisitionPayload, bool]] = []
    for candidate, token in zip(batch, tokens, strict=True):
        cache_hits_before = feature_store.physical_ledger.cache_hits
        payload = feature_store.acquire(candidate, token)
        retrieved.append(
            (
                candidate,
                payload,
                feature_store.physical_ledger.cache_hits > cache_hits_before,
            )
        )

    results: list[AcquisitionResult] = []
    for candidate, payload, cache_hit in retrieved:
        object_state = state.object_state(candidate.owner_id)
        state.logical_ledger.crop_inputs += payload.attempted_crop_inputs
        success = payload.feature is not None
        if success:
            acquired = AcquiredFeature(
                candidate.request_id,
                object_state_owner(state, candidate.owner_id),
                candidate.frame_index,
                candidate.overlap_pixels,
                payload.feature,
                candidate.spherical_cells,
            )
            object_state.retain_feature(acquired)
            object_state.successes += 1
            object_state.last_success_frame = candidate.frame_index
            object_state.cached_scores = None
            state.logical_ledger.successes += 1
        else:
            state.logical_ledger.failures += 1
        results.append(
            AcquisitionResult(
                request_id=candidate.request_id,
                owner_id=candidate.owner_id,
                success=success,
                attempted_crop_inputs=payload.attempted_crop_inputs,
                physical_cache_hit=cache_hit,
                failure_reason=payload.failure_reason,
            )
        )
    return tuple(results)


def object_state_owner(state: QueryPolicyState, owner_id: int) -> int:
    return state.canonical_owner(owner_id)


def update_cached_class_scores(
    state: QueryPolicyState, owner_id: int, text_features: Any
) -> np.ndarray | None:
    object_state = state.object_state(owner_id)
    if not object_state.features:
        object_state.cached_scores = None
        return None
    text = np.asarray(text_features, dtype=np.float64)
    if text.ndim != 2 or text.shape[0] != state.class_count:
        raise ValueError("text features must contain one row per policy class")
    text_norms = np.linalg.norm(text, axis=1, keepdims=True)
    if not np.isfinite(text).all() or np.any(text_norms <= 0.0):
        raise ValueError("text features must be finite and nonzero")
    text = text / text_norms
    feature_rows = np.stack([row.feature for row in object_state.features])
    if feature_rows.shape[1] != text.shape[1]:
        raise ValueError("image and text feature dimensions do not align")
    weights = np.asarray(
        [row.overlap_pixels for row in object_state.features], dtype=np.float64
    )
    aggregate = np.average(feature_rows, axis=0, weights=weights)
    norm = float(np.linalg.norm(aggregate))
    if norm <= 0.0:
        raise ValueError("area-weighted visual feature has zero norm")
    scores = _readonly(text @ (aggregate / norm), np.float64)
    object_state.cached_scores = scores
    return scores


def export_query_labels(
    state: QueryPolicyState,
    owner_ids: Sequence[int],
    text_features: Any,
    *,
    valid_class_ids: Sequence[int],
) -> dict[int, int]:
    """Export common readout labels, assigning class 0 to owners without evidence."""

    class_ids = tuple(int(value) for value in valid_class_ids)
    if len(class_ids) != state.class_count or len(set(class_ids)) != len(class_ids):
        raise ValueError("valid class IDs must uniquely match the policy vocabulary")
    labels: dict[int, int] = {}
    for owner_id in owner_ids:
        owner = int(owner_id)
        scores = update_cached_class_scores(state, owner, text_features)
        labels[owner] = 0 if scores is None else class_ids[int(np.argmax(scores))]
    return labels


def apply_alias_merge(
    state: QueryPolicyState, *, old_owner: int, new_owner: int
) -> None:
    old_canonical = state.canonical_owner(old_owner)
    new_canonical = state.canonical_owner(new_owner)
    if old_canonical == new_canonical:
        state.aliases[int(old_owner)] = new_canonical
        return
    target = state.object_state(new_canonical)
    source = state.objects.pop(old_canonical, ObjectQueryState())
    target.attempts += source.attempts
    target.successes += source.successes
    target.attempted_request_ids.update(source.attempted_request_ids)
    target.geometric_cells.update(source.geometric_cells)
    for acquired in source.features:
        target.retain_feature(acquired)
    if source.last_success_frame is not None and (
        target.last_success_frame is None
        or source.last_success_frame > target.last_success_frame
    ):
        target.last_success_frame = source.last_success_frame
    if source.last_paid_frame is not None and (
        target.last_paid_frame is None or source.last_paid_frame > target.last_paid_frame
    ):
        target.last_paid_pose = _readonly(source.last_paid_pose, np.float64)
        target.last_paid_frame = source.last_paid_frame
    target.best_paid_overlap = max(target.best_paid_overlap, source.best_paid_overlap)
    target.cached_scores = None
    target.lineage_available &= source.lineage_available
    for alias, destination in tuple(state.aliases.items()):
        if destination == old_canonical:
            state.aliases[alias] = new_canonical
    state.aliases[old_canonical] = new_canonical
    state.aliases[int(old_owner)] = new_canonical


def apply_alias_split(
    state: QueryPolicyState,
    *,
    parent_owner: int,
    child_owners: Sequence[int],
    request_to_child: Mapping[str, int] | None = None,
) -> str:
    """Assign known feature provenance or explicitly block unresolved causal lineage."""

    parent = state.object_state(parent_owner)
    children = tuple(dict.fromkeys(int(value) for value in child_owners))
    if not children or any(value <= 0 for value in children):
        raise ValueError("split child owners must be unique positive IDs")
    mappings = {} if request_to_child is None else {
        str(request): int(child) for request, child in request_to_child.items()
    }
    unresolved = parent.attempted_request_ids - mappings.keys()
    invalid = set(mappings.values()) - set(children)
    if unresolved or invalid:
        for child in children:
            state.object_state(child).lineage_available = False
        state.blocked_reasons.add("BLOCKED_CAUSAL_LINEAGE")
        return "BLOCKED_CAUSAL_LINEAGE"
    for child in children:
        child_state = state.object_state(child)
        assigned_ids = {request for request, owner in mappings.items() if owner == child}
        child_state.attempted_request_ids.update(assigned_ids)
        child_state.attempts += len(assigned_ids)
        for acquired in parent.features:
            if acquired.request_id in assigned_ids:
                child_state.retain_feature(acquired)
                child_state.successes += 1
        child_state.lineage_available = True
    return "COMPLETE"


__all__ = [
    "AcquiredFeature",
    "AcquisitionPayload",
    "AcquisitionResult",
    "FeatureStore",
    "LogicalAcquisitionLedger",
    "NativeCombineState",
    "ObjectQueryState",
    "PhysicalExecutionLedger",
    "QueryCandidate",
    "QueryPolicyState",
    "apply_alias_merge",
    "apply_alias_split",
    "build_query_candidates",
    "cumulative_quota",
    "dispatch_frame_batch",
    "export_query_labels",
    "native_combine_candidates",
    "observe_geometric_candidates",
    "update_cached_class_scores",
]
