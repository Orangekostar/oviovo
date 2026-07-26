from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.oviv2.temporal_config import TemporalProposalConfig


_INT64_MAX = 2**63 - 1


def _exact_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if value > _INT64_MAX:
        raise ValueError(f"{name} must fit signed int64")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return normalized


def _model_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("appearance_model_id must be a non-empty string")
    return value.strip()


def _array(value: object, name: str, *, ndim: int, dtype_kind: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an ndarray")
    if value.ndim != ndim or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} has invalid shape")
    if dtype_kind == "bool" and value.dtype != np.bool_:
        raise TypeError(f"{name} must have bool dtype")
    if dtype_kind == "float" and value.dtype.kind != "f":
        raise TypeError(f"{name} must have floating dtype")
    if dtype_kind == "float" and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True, eq=False)
class ProjectedIdentitySearchRegion:
    identity_id: int
    source_frame_id: int
    mask: np.ndarray
    expected_depth_m: np.ndarray
    projection_provenance_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _exact_int(self.identity_id, "identity_id", 1))
        object.__setattr__(self, "source_frame_id", _exact_int(self.source_frame_id, "source_frame_id"))
        mask = _array(self.mask, "mask", ndim=2, dtype_kind="bool")
        depth = _array(self.expected_depth_m, "expected_depth_m", ndim=2, dtype_kind="float", shape=mask.shape)
        if np.any(depth <= 0.0):
            raise ValueError("expected_depth_m must be positive")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "expected_depth_m", depth)
        object.__setattr__(self, "projection_provenance_hash", _hash(self.projection_provenance_hash, "projection_provenance_hash"))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProjectedIdentitySearchRegion) and (
            self.identity_id, self.source_frame_id, self.projection_provenance_hash
        ) == (other.identity_id, other.source_frame_id, other.projection_provenance_hash) and np.array_equal(self.mask, other.mask) and np.array_equal(self.expected_depth_m, other.expected_depth_m)


@dataclass(frozen=True, eq=False)
class ProposalRecoveryInput:
    frame_id: int
    timestamp: float
    depth_m: np.ndarray
    current_xyz: np.ndarray
    segmentation_occupied: np.ndarray
    semantic_support: np.ndarray
    semantic_source_frame_id: int
    semantic_provenance_hash: str
    appearance_support: np.ndarray | None
    appearance_source_frame_id: int | None
    appearance_model_id: str | None
    appearance_provenance_hash: str | None
    search_regions: tuple[ProjectedIdentitySearchRegion, ...]

    def __post_init__(self) -> None:
        frame_id = _exact_int(self.frame_id, "frame_id")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        depth = _array(self.depth_m, "depth_m", ndim=2, dtype_kind="float")
        if np.any(depth <= 0.0):
            raise ValueError("depth_m must be positive")
        shape = depth.shape
        xyz = _array(self.current_xyz, "current_xyz", ndim=3, dtype_kind="float", shape=shape + (3,))
        occupied = _array(self.segmentation_occupied, "segmentation_occupied", ndim=2, dtype_kind="bool", shape=shape)
        semantic = _array(self.semantic_support, "semantic_support", ndim=2, dtype_kind="bool", shape=shape)
        source = _exact_int(self.semantic_source_frame_id, "semantic_source_frame_id")
        if source != frame_id:
            raise ValueError("semantic support must come from the current frame, not a historical or future frame")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "current_xyz", xyz)
        object.__setattr__(self, "segmentation_occupied", occupied)
        object.__setattr__(self, "semantic_support", semantic)
        object.__setattr__(self, "semantic_source_frame_id", source)
        object.__setattr__(self, "semantic_provenance_hash", _hash(self.semantic_provenance_hash, "semantic_provenance_hash"))
        if self.appearance_support is None:
            if any(value is not None for value in (self.appearance_source_frame_id, self.appearance_model_id, self.appearance_provenance_hash)):
                raise ValueError("unavailable appearance support cannot have provenance")
        else:
            appearance = _array(self.appearance_support, "appearance_support", ndim=2, dtype_kind="bool", shape=shape)
            appearance_source = _exact_int(self.appearance_source_frame_id, "appearance_source_frame_id")
            if appearance_source != frame_id:
                raise ValueError("appearance support must come from the current frame, not a historical or future frame")
            object.__setattr__(self, "appearance_support", appearance)
            object.__setattr__(self, "appearance_source_frame_id", appearance_source)
            object.__setattr__(self, "appearance_model_id", _model_id(self.appearance_model_id))
            object.__setattr__(self, "appearance_provenance_hash", _hash(self.appearance_provenance_hash, "appearance_provenance_hash"))
        if type(self.search_regions) is not tuple:
            raise TypeError("search_regions must be an exact tuple")
        if any(not isinstance(item, ProjectedIdentitySearchRegion) for item in self.search_regions):
            raise TypeError("search_regions contain invalid values")
        keys = [(item.identity_id, item.source_frame_id, item.projection_provenance_hash) for item in self.search_regions]
        if len(keys) != len(set(keys)):
            raise ValueError("search_regions must be unique")
        for item in self.search_regions:
            if item.mask.shape != shape:
                raise ValueError("search region shape mismatch")
            if item.source_frame_id >= frame_id:
                raise ValueError("search region must come from a strictly past frame, not the same or a future frame")


@dataclass(frozen=True, eq=False)
class RecoveredTemporalProposal:
    proposal_id: int
    frame_id: int
    timestamp: float
    identity_hint: int
    mask: np.ndarray
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    mean_depth_m: float
    projection_source_frame_id: int
    projection_provenance_hash: str
    semantic_source_frame_id: int
    semantic_provenance_hash: str
    appearance_available: bool
    appearance_source_frame_id: int | None
    appearance_model_id: str | None
    appearance_provenance_hash: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _exact_int(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "frame_id", _exact_int(self.frame_id, "frame_id"))
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        object.__setattr__(self, "identity_hint", _exact_int(self.identity_hint, "identity_hint", 1))
        mask = _array(self.mask, "mask", ndim=2, dtype_kind="bool")
        area = _exact_int(self.area_px, "area_px", 1)
        if area != int(mask.sum()):
            raise ValueError("area_px must equal mask area")
        if type(self.bbox_xyxy) is not tuple or len(self.bbox_xyxy) != 4:
            raise TypeError("bbox_xyxy must be an exact integer tuple")
        bbox = tuple(_exact_int(value, "bbox coordinate") for value in self.bbox_xyxy)
        rows, columns = np.nonzero(mask)
        expected_bbox = (
            int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)
        )
        if bbox != expected_bbox:
            raise ValueError("bbox_xyxy must tightly bound mask")
        centroid = np.asarray(self.centroid_xyz, dtype=np.float64)
        bounds_min = np.asarray(self.bounds_min_xyz, dtype=np.float64)
        bounds_max = np.asarray(self.bounds_max_xyz, dtype=np.float64)
        if any(item.shape != (3,) or not np.isfinite(item).all() for item in (centroid, bounds_min, bounds_max)):
            raise ValueError("proposal geometry must contain finite 3D points")
        if np.any(bounds_min > bounds_max) or np.any(centroid < bounds_min) or np.any(centroid > bounds_max):
            raise ValueError("proposal centroid must lie within ordered bounds")
        mean_depth = _finite(self.mean_depth_m, "mean_depth_m")
        if mean_depth <= 0.0:
            raise ValueError("mean_depth_m must be positive")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "area_px", area)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "centroid_xyz", tuple(float(x) for x in centroid))
        object.__setattr__(self, "bounds_min_xyz", tuple(float(x) for x in bounds_min))
        object.__setattr__(self, "bounds_max_xyz", tuple(float(x) for x in bounds_max))
        object.__setattr__(self, "mean_depth_m", mean_depth)
        object.__setattr__(self, "projection_source_frame_id", _exact_int(self.projection_source_frame_id, "projection_source_frame_id"))
        object.__setattr__(self, "semantic_source_frame_id", _exact_int(self.semantic_source_frame_id, "semantic_source_frame_id"))
        if self.projection_source_frame_id >= self.frame_id:
            raise ValueError("projection provenance must come from a strictly past frame")
        if self.semantic_source_frame_id != self.frame_id:
            raise ValueError("semantic provenance must come from the current frame")
        object.__setattr__(self, "projection_provenance_hash", _hash(self.projection_provenance_hash, "projection_provenance_hash"))
        object.__setattr__(self, "semantic_provenance_hash", _hash(self.semantic_provenance_hash, "semantic_provenance_hash"))
        if type(self.appearance_available) is not bool:
            raise TypeError("appearance_available must be an exact bool")
        appearance_values = (
            self.appearance_source_frame_id,
            self.appearance_model_id,
            self.appearance_provenance_hash,
        )
        if not self.appearance_available:
            if any(item is not None for item in appearance_values):
                raise ValueError("unavailable appearance cannot have provenance")
        else:
            source = _exact_int(self.appearance_source_frame_id, "appearance_source_frame_id")
            if source != self.frame_id:
                raise ValueError("appearance provenance must come from the current frame")
            object.__setattr__(self, "appearance_source_frame_id", source)
            object.__setattr__(self, "appearance_model_id", _model_id(self.appearance_model_id))
            object.__setattr__(self, "appearance_provenance_hash", _hash(self.appearance_provenance_hash, "appearance_provenance_hash"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecoveredTemporalProposal):
            return NotImplemented
        left = tuple(value for key, value in self.__dict__.items() if key != "mask")
        right = tuple(value for key, value in other.__dict__.items() if key != "mask")
        return left == right and np.array_equal(self.mask, other.mask)


@dataclass(frozen=True)
class ProposalRecoveryResult:
    proposals: tuple[RecoveredTemporalProposal, ...]
    opportunity_count: int
    trigger_count: int

    def __post_init__(self) -> None:
        if type(self.proposals) is not tuple or any(
            not isinstance(item, RecoveredTemporalProposal) for item in self.proposals
        ):
            raise TypeError("proposals must be an exact tuple of recovered proposals")
        opportunity = _exact_int(self.opportunity_count, "opportunity_count")
        trigger = _exact_int(self.trigger_count, "trigger_count")
        if trigger != len(self.proposals):
            raise ValueError("trigger_count must equal proposal count")
        if opportunity < trigger:
            raise ValueError("opportunity_count cannot be below trigger_count")
        if tuple(item.proposal_id for item in self.proposals) != tuple(range(trigger)):
            raise ValueError("proposal IDs must be canonical and contiguous")
        object.__setattr__(self, "opportunity_count", opportunity)
        object.__setattr__(self, "trigger_count", trigger)


def _components(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    seen = np.zeros(mask.shape, dtype=bool)
    output: list[np.ndarray] = []
    height, width = mask.shape
    for row, column in zip(*np.nonzero(mask)):
        if seen[row, column]:
            continue
        stack = [(int(row), int(column))]
        seen[row, column] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            pixels.append(current)
            for rr, cc in ((current[0] - 1, current[1]), (current[0], current[1] - 1), (current[0], current[1] + 1), (current[0] + 1, current[1])):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        component = np.zeros(mask.shape, dtype=bool)
        rows, columns = zip(*pixels)
        component[rows, columns] = True
        output.append(component)
    return tuple(output)


def _expand_metric_region(
    mask: np.ndarray,
    current_xyz: np.ndarray,
    expansion_m: float,
) -> np.ndarray:
    expanded = mask.copy()
    if expansion_m == 0.0 or not mask.any():
        return expanded
    flat_xyz = current_xyz.reshape(-1, 3)
    within = np.zeros(flat_xyz.shape[0], dtype=bool)
    squared_limit = expansion_m * expansion_m
    for source_xyz in current_xyz[mask]:
        delta = flat_xyz - source_xyz
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        within |= squared_distance <= squared_limit
    return within.reshape(mask.shape)


def recover_temporal_proposals(value: ProposalRecoveryInput, config: TemporalProposalConfig) -> ProposalRecoveryResult:
    if not isinstance(value, ProposalRecoveryInput):
        raise TypeError("value must be a ProposalRecoveryInput")
    if not isinstance(config, TemporalProposalConfig):
        raise TypeError("config must be a TemporalProposalConfig")
    minimum_area = _exact_int(config.minimum_residual_area_px, "config.minimum_residual_area_px", 1)
    capacity = _exact_int(config.maximum_recovered_proposals, "config.maximum_recovered_proposals", 1)
    _finite(config.search_region_expansion_m, "config.search_region_expansion_m")
    residual_threshold = _finite(config.minimum_depth_residual_m, "config.minimum_depth_residual_m")
    if config.search_region_expansion_m < 0.0 or residual_threshold <= 0.0:
        raise ValueError("proposal distances must be positive/non-negative")

    ownership = np.full(value.depth_m.shape, np.iinfo(np.int64).max, dtype=np.int64)
    raw: list[tuple[ProjectedIdentitySearchRegion, np.ndarray]] = []
    for item in sorted(value.search_regions, key=lambda x: (x.identity_id, x.source_frame_id, x.projection_provenance_hash)):
        search_mask = _expand_metric_region(
            item.mask, value.current_xyz, config.search_region_expansion_m
        )
        residual = item.expected_depth_m - value.depth_m >= residual_threshold
        eligible = (
            search_mask
            & value.semantic_support
            & ~value.segmentation_occupied
            & residual
        )
        ownership[eligible] = np.minimum(ownership[eligible], item.identity_id)
        raw.append((item, eligible))

    candidates: list[tuple[tuple[object, ...], ProjectedIdentitySearchRegion, np.ndarray]] = []
    for item, eligible in raw:
        arbitrated = eligible & (ownership == item.identity_id)
        for component in _components(arbitrated):
            area = int(component.sum())
            if area < minimum_area:
                continue
            row, column = np.argwhere(component)[0]
            candidates.append(((-area, item.identity_id, int(row), int(column), component.tobytes()), item, component))
    candidates.sort(key=lambda item: item[0])
    opportunity_count = len(candidates)
    proposals: list[RecoveredTemporalProposal] = []
    for proposal_id, (_, item, mask) in enumerate(candidates[:capacity]):
        rows, columns = np.nonzero(mask)
        xyz = value.current_xyz[mask]
        frozen_mask = np.frombuffer(np.ascontiguousarray(mask).tobytes(), dtype=np.bool_).reshape(mask.shape)
        appearance_available = bool(
            value.appearance_support is not None
            and np.any(mask & value.appearance_support)
        )
        proposals.append(RecoveredTemporalProposal(
            proposal_id, value.frame_id, value.timestamp, item.identity_id, frozen_mask,
            int(mask.sum()), (int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)),
            tuple(float(x) for x in xyz.mean(axis=0)), tuple(float(x) for x in xyz.min(axis=0)),
            tuple(float(x) for x in xyz.max(axis=0)), float(value.depth_m[mask].mean()),
            item.source_frame_id, item.projection_provenance_hash,
            value.semantic_source_frame_id, value.semantic_provenance_hash,
            appearance_available,
            value.appearance_source_frame_id if appearance_available else None,
            value.appearance_model_id if appearance_available else None,
            value.appearance_provenance_hash if appearance_available else None,
        ))
    return ProposalRecoveryResult(tuple(proposals), opportunity_count, len(proposals))
