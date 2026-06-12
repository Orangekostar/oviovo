"""Core data structures for the OVIOVO mapping system.

Defines all shared types used across modules: frames, proposals, patches,
object maps, background maps, association results, and system state.

Layer distinction (Rule D):
  raw proposal -> refined proposal -> 3D patch -> instance -> local geometry memory -> semantic memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================
# Camera & Frame
# ============================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass
class Frame:
    """A single RGB-D frame with camera pose.

    Attributes:
        frame_id: Sequential processed-frame index used internally by the pipeline.
        rgb: (H, W, 3) uint8 color image.
        depth: (H, W) float32 depth in meters.
        pose: (4, 4) camera-to-world transform.
        intrinsics: Camera intrinsic parameters.
        timestamp: Optional timestamp in seconds.
        source_frame_id: Optional source/dataset frame index used for external cache lookups.
    """
    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp: float = 0.0
    source_frame_id: int | None = None


# ============================================================
# Proposals & Patches
# ============================================================

@dataclass
class Proposal2D:
    """A class-agnostic 2D object proposal (mask) — raw SAM2 output.

    Attributes:
        proposal_id: Unique id within the frame.
        mask: (H, W) bool mask.
        bbox_xyxy: [x1, y1, x2, y2] bounding box.
        area: Number of True pixels.
        confidence: Proposal confidence score.
        backend_name: Proposal backend provenance, e.g. "sam2" or "cropformer".
        metadata: Optional backend-specific debug metadata.
    """
    proposal_id: int
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    area: int
    confidence: float = 1.0
    backend_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anchor2D:
    """Object-level 2D anchor box from a detector prior."""
    anchor_id: int
    bbox_xyxy: np.ndarray
    class_name: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorAssignment:
    """Assignment from a raw proposal to a detector anchor."""
    proposal_id: int
    anchor_id: int = -1
    class_name: str = ""
    confidence: float = 0.0
    bbox_iou: float = 0.0
    center_inside: bool = False
    keepalive: bool = False


@dataclass
class ProposalGeometricFeatures:
    """Geometric features computed for a refined proposal.

    These are explicit, inspectable quantities — not hidden heuristics.
    """
    depth_valid_ratio: float = 0.0
    depth_variance: float = 0.0
    border_touch_ratio: float = 0.0
    planar_fit_residual: float = float("inf")


@dataclass
class ProposalSoftScores:
    """Soft scores for a refined proposal.

    Explicit weighted functions of geometric features.
    No hard labels — only continuous evidence.
    """
    objectness_score: float = 0.0
    backgroundness_score: float = 0.0
    attachedness_score: float = 0.0


@dataclass
class RefinedProposal2D:
    """A depth-refined 2D proposal with explicit geometric features and soft scores.

    This is the output of depth-aware refinement (Layer 2).
    Raw SAM2 masks are refined using depth discontinuity and connectivity.
    """
    proposal_id: int
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    area: int
    confidence: float = 1.0
    backend_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    geometric_features: ProposalGeometricFeatures = field(default_factory=ProposalGeometricFeatures)
    soft_scores: ProposalSoftScores = field(default_factory=ProposalSoftScores)


@dataclass
class Patch3D:
    """A 3D object patch lifted from a 2D proposal (Layer 3).

    Each patch is only a partial observation — not a complete object.

    Attributes:
        patch_id: Matches the source proposal_id.
        points: (N, 3) world-coordinate points.
        centroid: (3,) centroid of the patch.
        bbox_min: (3,) axis-aligned bounding box minimum.
        bbox_max: (3,) axis-aligned bounding box maximum.
        normals: Optional (N, 3) surface normals.
        timestamp: Frame timestamp.
        source_frame_id: ID of the source frame.
        soft_scores: Inherited soft scores from the refined proposal.
    """
    patch_id: int
    points: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    normals: Optional[np.ndarray] = None
    timestamp: float = 0.0
    source_frame_id: int = 0
    soft_scores: ProposalSoftScores = field(default_factory=ProposalSoftScores)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# TSDF Instance Substrate (Layer 4)
# ============================================================

@dataclass
class VoxelOwnerSupport:
    """Per-voxel owner support tracking for label stabilization.

    Maps instance_id -> accumulated support. Owner = argmax support.
    No hard overwrites — support is accumulated and decayed.
    """
    support: Dict[int, float] = field(default_factory=dict)

    @property
    def owner_id(self) -> int:
        """Current owner is argmax of support. -1 if unowned."""
        if not self.support:
            return -1
        return max(self.support, key=self.support.get)

    @property
    def owner_support_value(self) -> float:
        if not self.support:
            return 0.0
        return max(self.support.values())


@dataclass
class TSDFInstanceVolume:
    """Global TSDF-based instance map (Layer 4 backbone).

    Each voxel stores TSDF value + weight + owner instance id + owner support.
    This is the true low-level stable backbone of the system.

    Uses sparse hash-map representation (dict keyed by voxel index tuple).
    """
    voxel_size: float = 0.05
    truncation_distance: float = 0.15
    tsdf: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    weight: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    owner_support: Dict[Tuple[int, int, int], VoxelOwnerSupport] = field(default_factory=dict)
    support_increment: float = 1.0
    support_decay: float = 0.95


# ============================================================
# Voxel Voting & Association (Layer 4 output)
# ============================================================

@dataclass
class VoxelVoteResult:
    """Exposed voting metadata for one patch-to-instance association attempt.

    All quantities are explicit and inspectable.
    """
    touched_voxel_count: int = 0
    supported_voxel_count: int = 0
    owner_votes: Dict[int, int] = field(default_factory=dict)
    normalized_vote_score: float = 0.0
    best_instance_id: int = -1
    geometry_consistency_score: float = 0.0
    final_association_score: float = 0.0
    new_instance_created: bool = False


# ============================================================
# Active Set (Layer 6)
# ============================================================

@dataclass
class ActiveSet:
    """Restricted local candidate set derived from global instance map.

    Only for local reasoning, stability check, split/merge diagnostics.
    Must NOT replace global memory.
    """
    visible_ids: Set[int] = field(default_factory=set)
    nearby_ids: Set[int] = field(default_factory=set)
    whole_prior_ids: Set[int] = field(default_factory=set)
    new_object_candidate_ids: Set[int] = field(default_factory=set)

    @property
    def all_candidate_ids(self) -> Set[int]:
        return self.visible_ids | self.nearby_ids | self.whole_prior_ids | self.new_object_candidate_ids


# ============================================================
# Whole-Object Evidence (soft only — Rule B)
# ============================================================

@dataclass
class WholeEvidenceScores:
    """Soft whole-object evidence. No hard is_whole_object labels.

    Rule B: use only soft evidence, never hard whole-object labels.
    """
    whole_evidence_score: float = 0.0
    part_evidence_score: float = 0.0
    assignment_score: float = 0.0


# ============================================================
# Object Representation
# ============================================================

class ObjectState(Enum):
    """Lifecycle states for tracked objects."""
    ACTIVE = "active"               # Currently being observed and updated
    INACTIVE = "inactive"           # Not observed recently, pending review
    GHOST = "ghost"                 # Suspected to no longer exist
    REMOVED = "removed"             # Marked for deletion
    DORMANT = "dormant"             # Stable but not recently seen, awaiting re-id


class SurfaceTier(Enum):
    """Residency tier for dense semantic surface storage."""
    ACTIVE = "active"
    WARM = "warm"
    COLD = "cold"


@dataclass
class SemanticMemory:
    """Object-level semantic memory (Layer 7).

    Stores aggregated visual-language features and label hypotheses.
    No dense per-point features — only object-level embeddings.
    Semantics start only after instance stabilization.

    Attributes:
        feature_bank: List of embedding vectors from different views.
        aggregated_feature: Fused embedding (e.g. mean / medoid).
        label_hypotheses: Top-k (label, confidence) pairs.
        observation_count: Number of semantic observations integrated.
        confidence_history: Per-update confidence values.
    """
    feature_bank: List[np.ndarray] = field(default_factory=list)
    aggregated_feature: Optional[np.ndarray] = None
    label_hypotheses: List[tuple] = field(default_factory=list)  # [(label, score), ...]
    observation_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationRecord:
    """Record of a single observation of an object.

    Attributes:
        frame_id: Frame in which the observation was made.
        patch: The 3D patch observed.
        crop_bbox: 2D bounding box of the crop used for semantic encoding.
        timestamp: Observation time.
        source_frame_id: Source frame that produced the underlying proposal.
        source_proposal_id: Source proposal identifier from the proposal backend.
        anchor_id: Anchor identifier used to connect coarse/fine observations.
        observation_layer: Observation layer name, such as coarse or refined.
        refinement_key: Stable key for matching refinement observations.
        replaced_by_refinement: Whether a later refinement superseded this observation.
    """
    frame_id: int
    patch: Patch3D
    crop_bbox: Optional[np.ndarray] = None
    timestamp: float = 0.0
    source_frame_id: int = 0
    source_proposal_id: int = -1
    anchor_id: int = -1
    observation_layer: str = ""
    refinement_key: str = ""
    replaced_by_refinement: bool = False


@dataclass
class ObjectMap:
    """A single object instance in the map.

    Attributes:
        object_id: Unique object identifier.
        state: Current lifecycle state.
        local_pcd: (N, 3) v1 object pool geometry (point cloud).
            This is the object-level export source used to assemble the
            pool-based semantic instance map in v1.
            It must NOT replace the global TSDF instance substrate for
            owner decisions or support/stability bookkeeping.
        association_pcd: (M, 3) bounded representative geometry used only
            for association nearest-neighbor matching.
        centroid: (3,) current centroid.
        bbox_min: (3,) AABB min.
        bbox_max: (3,) AABB max.
        whole_evidence: Soft whole-object evidence scores (Rule B).
        semantic_memory: Object-level semantic information (Layer 7).
        observations: History of observation records.
        confidence: Overall confidence in this object's existence.
        last_seen_frame: Frame ID of most recent observation.
        creation_frame: Frame ID when object was first created.
        update_count: Number of geometry updates.
    """
    object_id: int
    state: ObjectState = ObjectState.ACTIVE
    local_pcd: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    association_pcd: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    whole_evidence: WholeEvidenceScores = field(default_factory=WholeEvidenceScores)
    semantic_memory: SemanticMemory = field(default_factory=SemanticMemory)
    observations: List[ObservationRecord] = field(default_factory=list)
    confidence: float = 1.0
    last_seen_frame: int = 0
    creation_frame: int = 0
    update_count: int = 0
    surface_tier: SurfaceTier = SurfaceTier.ACTIVE
    last_dense_refresh_frame: int = 0
    dense_surface_resident: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        """Backward-compatible alias for local_pcd."""
        return self.local_pcd

    @points.setter
    def points(self, value: np.ndarray) -> None:
        self.local_pcd = value


# ============================================================
# Background Map
# ============================================================

@dataclass
class BackgroundMap:
    """Global background map (TSDF-like placeholder).

    Attributes:
        voxel_size: Resolution of the voxel grid.
        origin: (3,) world-coordinate origin of the volume.
        tsdf_volume: Placeholder for TSDF data.
        weight_volume: Placeholder for integration weights.
        point_cloud: Accumulated background points (simple fallback).
        frame_count: Number of frames integrated.
    """
    voxel_size: float = 0.05
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tsdf_volume: Optional[np.ndarray] = None
    weight_volume: Optional[np.ndarray] = None
    point_cloud: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    frame_count: int = 0


# ============================================================
# Dense Surface Map
# ============================================================

@dataclass
class DenseSurfaceEntry:
    """Per-object resident dense semantic surface."""
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    object_id: int = -1
    semantic_label: str = ""
    last_refresh_frame: int = 0
    resident: bool = True


@dataclass
class DenseSurfaceMap:
    """Online dense semantic surface storage for active and warm objects."""
    entries: Dict[int, DenseSurfaceEntry] = field(default_factory=dict)


@dataclass
class StructuralOverlayVoxel:
    """Voxel-level semantic votes for stuff-like structural classes."""
    label_votes: Dict[str, float] = field(default_factory=dict)
    observation_count: int = 0
    last_seen_frame: int = -1

    def add_vote(self, label: str, weight: float, frame_id: int) -> None:
        normalized = str(label).strip().lower()
        if not normalized:
            return
        self.label_votes[normalized] = float(self.label_votes.get(normalized, 0.0) + float(weight))
        self.observation_count += 1
        self.last_seen_frame = max(int(self.last_seen_frame), int(frame_id))

    @property
    def top_label(self) -> str:
        if not self.label_votes:
            return ""
        return max(self.label_votes, key=self.label_votes.get)

    @property
    def top_support(self) -> float:
        if not self.label_votes:
            return 0.0
        return float(max(self.label_votes.values()))


@dataclass
class StructuralOverlayMap:
    """Sparse dense overlay for planar or stuff-like structure labels."""
    voxel_size: float = 0.05
    voxels: Dict[Tuple[int, int, int], StructuralOverlayVoxel] = field(default_factory=dict)
    update_count: int = 0


# ============================================================
# Provisional Local Object Pool
# ============================================================

@dataclass
class ProvisionalObject:
    """Local-only provisional object accumulated before global promotion."""
    provisional_id: int
    anchor_class_name: str = ""
    anchor_confidence: float = 0.0
    local_pcd: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    observations: List[ObservationRecord] = field(default_factory=list)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    hit_count: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Association
# ============================================================

@dataclass
class AssociationScore:
    """Scoring breakdown for a candidate object-patch association.

    Now backed by TSDF spatial voting (Layer 4).

    Attributes:
        centroid_distance: Centroid proximity score.
        bbox_overlap: IoU of 3D bounding boxes.
        geometry_overlap: Local geometry consistency score.
        voxel_vote_score: Normalized voxel owner vote score.
        total_score: Weighted combination of all scores.
        vote_result: Full voxel voting metadata (exposed for debugging).
    """
    centroid_distance: float = 0.0
    bbox_overlap: float = 0.0
    geometry_overlap: float = 0.0
    voxel_vote_score: float = 0.0
    total_score: float = 0.0
    vote_result: Optional[VoxelVoteResult] = None


@dataclass
class ContestedAssociation:
    """A confident frontend observation blocked from updating a cross-label object.

    The spatial candidate is useful context, but it is not an update target.
    ObjectUpdateModule should route the patch into a residual/provisional identity path.
    """
    patch_id: int
    blocked_object_id: int
    patch_label: str = ""
    object_label: str = ""
    reason: str = "cross_label_observation_identity"
    score: Optional[AssociationScore] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssociationResult:
    """Result of associating object patches to existing objects.

    Attributes:
        matched: List of (patch_id, object_id, score) for successful same-identity matches.
        new_object_patches: Patch IDs that should create normal new objects.
        contested_object_patches: Patch IDs that were observed as a different confident label
            from their best spatial candidate and must enter residual/provisional tracking.
        contested_matches: Detailed records for blocked cross-label spatial candidates.
        scores: Full scoring details per candidate pair.
    """
    matched: List[tuple] = field(default_factory=list)
    new_object_patches: List[int] = field(default_factory=list)
    contested_object_patches: List[int] = field(default_factory=list)
    contested_matches: List[ContestedAssociation] = field(default_factory=list)
    scores: Dict[tuple, AssociationScore] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# System State
# ============================================================

@dataclass
class SystemState:
    """Top-level container for the entire mapping system state.

    Attributes:
        objects: Dict mapping object_id to ObjectMap.
        background: The global background map.
        tsdf_volume: Global TSDF instance substrate (Layer 4 backbone).
        dense_surface_map: Resident dense semantic surface storage.
        provisional_objects: Local-only provisional object pool.
        active_set: Current-frame restricted active set (Layer 6, local only).
        next_object_id: Counter for new object IDs.
        next_provisional_id: Counter for provisional object IDs.
        frame_count: Total frames processed.
    """
    objects: Dict[int, ObjectMap] = field(default_factory=dict)
    background: BackgroundMap = field(default_factory=BackgroundMap)
    tsdf_volume: TSDFInstanceVolume = field(default_factory=TSDFInstanceVolume)
    dense_surface_map: DenseSurfaceMap = field(default_factory=DenseSurfaceMap)
    structural_overlay_map: StructuralOverlayMap = field(default_factory=StructuralOverlayMap)
    provisional_objects: Dict[int, ProvisionalObject] = field(default_factory=dict)
    active_set: ActiveSet = field(default_factory=ActiveSet)
    next_object_id: int = 0
    next_provisional_id: int = 0
    frame_count: int = 0
