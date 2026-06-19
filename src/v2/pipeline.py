"""Simplified parallel V2 semantic mapping pipeline."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml

from src.core.data_structures import (
    BackgroundMap,
    CameraIntrinsics,
    Frame,
    ObjectMap,
    ObjectState,
    SystemState,
    WholeEvidenceScores,
)
from src.modules.bg_obj_split import BgObjSplitModule
from src.modules.depth_refinement import DepthRefinementModule
from src.modules.frame_input import FrameInputModule
from src.modules.object_anchor import ObjectAnchorModule
from src.modules.patch_lifting import PatchLiftingModule
from src.modules.proposal import ProposalModule
from src.modules.runtime_vis import RuntimeVisModule
from src.utils.geometry import bbox_iou_3d, compute_bbox
from src.utils.logging import setup_logging
from src.v2.types import (
    CoarseVoxelOwner,
    ObjectPool,
    ProvisionalSpatialBucket,
    SemanticMapV2State,
    V2ObjectState,
)

logger = logging.getLogger("oviovo.v2.pipeline")


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class SemanticMapV2Pipeline:
    """Parallel V2 semantic mapping pipeline focused on task-oriented semantic maps."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = self._load_config(config_path)

        log_cfg = self.config.get("logging", {})
        setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file", ""))

        self.frame_input = FrameInputModule(self.config.get("frame_input", {}))
        self.proposal = ProposalModule(self.config.get("proposal", {}))
        self.object_anchor = ObjectAnchorModule(self.config.get("anchor_frontend", {}))
        self.runtime_vis = RuntimeVisModule(self.config.get("runtime_vis", {}))
        self.depth_refinement = DepthRefinementModule(self.config.get("depth_refinement", {}))
        self.patch_lifting = PatchLiftingModule(self.config.get("patch_lifting", {}))
        self.bg_obj_split = BgObjSplitModule(self.config.get("bg_obj_split", {}))

        v2_cfg = self.config.get("semantic_map_v2", {})
        self.geometry_sample_stride = int(v2_cfg.get("geometry_sample_stride", 8))
        self.geometry_voxel_size = float(v2_cfg.get("geometry_voxel_size", 0.03))
        self.coarse_voxel_size = float(v2_cfg.get("coarse_voxel_size", 0.05))
        self.pool_voxel_size = float(v2_cfg.get("pool_voxel_size", 0.02))
        self.max_points_per_object = int(v2_cfg.get("max_points_per_object", 20000))
        self.match_threshold = float(v2_cfg.get("match_threshold", 0.45))
        self.match_distance_scale = float(v2_cfg.get("match_distance_scale", 2.0))
        self.nearby_match_radius = float(v2_cfg.get("nearby_match_radius", 1.2))
        self.coarse_weight = float(v2_cfg.get("coarse_weight", 0.5))
        self.centroid_weight = float(v2_cfg.get("centroid_weight", 0.25))
        self.bbox_weight = float(v2_cfg.get("bbox_weight", 0.15))
        self.overlap_weight = float(v2_cfg.get("overlap_weight", 0.10))
        self.provisional_match_distance = float(v2_cfg.get("provisional_match_distance", 0.45))
        self.provisional_promotion_hits = int(v2_cfg.get("provisional_promotion_hits", 2))
        self.provisional_max_idle_frames = int(v2_cfg.get("provisional_max_idle_frames", 60))
        self.freeze_confidence_threshold = float(v2_cfg.get("freeze_confidence_threshold", 0.6))
        self.freeze_hits = int(v2_cfg.get("freeze_hits", 2))
        self.coarse_support_increment = float(v2_cfg.get("coarse_support_increment", 1.0))
        self.coarse_support_decay = float(v2_cfg.get("coarse_support_decay", 0.95))
        self.archive_after_frames = int(v2_cfg.get("archive_after_frames", 60))
        self.projection_neighbor_radius = int(v2_cfg.get("projection_neighbor_radius", 1))
        self.export_pool_downsample_voxel = float(v2_cfg.get("export_pool_downsample_voxel", 0.0))
        self.verbose = bool(self.config.get("pipeline", {}).get("verbose", True))

        self.state = SemanticMapV2State()
        self.last_raw_proposals = []
        self.last_merged_proposals = []
        self.last_refined_proposals = []
        self.last_patches = []
        self.last_candidate_patches = []
        self.last_background_patches = []
        self.last_runtime_vis_output = None
        self.last_frame_debug: Dict[str, Any] = {}
        logger.info("SemanticMapV2Pipeline initialized.")

    def _load_config(self, config_path: str | Path | None) -> Dict[str, Any]:
        default_path = self.project_root / "configs" / "default.yaml"
        base = yaml.safe_load(default_path.read_text(encoding="utf-8")) if default_path.exists() else {}
        if config_path is None:
            return base
        override_path = Path(config_path)
        if not override_path.exists():
            logger.warning("V2 config path not found at %s; using default config.", override_path)
            return base
        override = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        return _deep_merge_dict(base, override)

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        timestamp: float = 0.0,
        source_frame_id: int | None = None,
    ) -> SemanticMapV2State:
        frame = self.frame_input.process(
            rgb,
            depth,
            pose,
            intrinsics,
            timestamp,
            source_frame_id=source_frame_id,
        )
        if self.verbose:
            logger.info("=== V2 Frame %d ===", frame.frame_id)

        self._accumulate_geometry(frame)

        proposal_source = "proposal_backend"
        try:
            if self.object_anchor.enabled and self.object_anchor.use_sam_intersection_proposals:
                sam_proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                anchors, proposals, anchor_assignments = self.object_anchor.generate_proposals(frame.rgb, sam_proposals)
                proposal_source = "anchor_sam_union"
            else:
                proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                anchors, anchor_assignments = self.object_anchor.process(frame.rgb, proposals)
            anchor_runtime_error = ""
        except Exception as exc:
            logger.warning("V2 anchor inference failed; continuing without anchors: %s", exc)
            anchors = []
            anchor_assignments = []
            anchor_runtime_error = str(exc)
            if proposal_source == "anchor_primary":
                proposals = []
            else:
                proposals = self.proposal.process(frame.rgb, frame.depth, frame=frame)
                for proposal in proposals:
                    proposal.metadata = dict(proposal.metadata)
                    proposal.metadata.setdefault("anchor_id", -1)
                    proposal.metadata.setdefault("anchor_class_name", "")
                    proposal.metadata.setdefault("anchor_confidence", 0.0)
                    proposal.metadata.setdefault("anchor_bbox_iou", 0.0)
                    proposal.metadata.setdefault("anchor_center_inside", False)
                    proposal.metadata.setdefault("anchor_keepalive", False)
        self.last_raw_proposals = proposals

        runtime_state = self._build_runtime_state()
        runtime_output = self.runtime_vis.process(frame, proposals, runtime_state)
        self.last_runtime_vis_output = runtime_output
        merged_proposals = self._attach_anchor_summary(runtime_output.raw_proposals, runtime_output.merged_proposals)
        self.last_merged_proposals = merged_proposals

        refined = self.depth_refinement.process(frame.depth, merged_proposals)
        self.last_refined_proposals = refined

        patches = self.patch_lifting.process(
            refined,
            frame.depth,
            frame.pose,
            frame.intrinsics,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )
        self.last_patches = patches

        bg_patches, obj_patches, amb_patches = self.bg_obj_split.process(patches, BackgroundMap(), runtime_state.objects)
        candidate_patches = [*obj_patches, *amb_patches]
        self.last_background_patches = bg_patches
        self.last_candidate_patches = candidate_patches

        matched_count = 0
        promoted_count = 0
        frozen_count = 0
        for patch in candidate_patches:
            object_id = self._match_patch_to_object_id(patch)
            if object_id is not None:
                pool = self.state.object_pools[object_id]
                self._merge_patch_into_pool(pool, patch, frame.frame_id)
                if self._update_pool_semantics(pool, patch, frame.frame_id):
                    frozen_count += 1
                self._update_coarse_ownership(pool.object_id, patch.points, frame.frame_id)
                matched_count += 1
            else:
                promoted_pool = self._update_provisional_bucket(patch, frame.frame_id)
                if promoted_pool is not None:
                    self._update_coarse_ownership(promoted_pool.object_id, promoted_pool.points, frame.frame_id)
                    promoted_count += 1
                    if promoted_pool.label_frozen:
                        frozen_count += 1

        self._prune_provisional_buckets(frame.frame_id)
        self._update_archive_states(frame.frame_id)
        self.state.frame_count = int(frame.frame_id + 1)

        object_state_counts = Counter(pool.state.value for pool in self.state.object_pools.values())
        self.last_frame_debug = {
            "frame_id": int(frame.frame_id),
            "proposal_source": proposal_source,
            "raw_proposal_count": int(len(proposals)),
            "anchor_count": int(len(anchors)),
            "anchor_runtime_error": anchor_runtime_error,
            "merged_group_count": int(len(runtime_output.groups)),
            "refined_proposal_count": int(len(refined)),
            "patch_count": int(len(patches)),
            "background_patch_count": int(len(bg_patches)),
            "candidate_patch_count": int(len(candidate_patches)),
            "matched_patch_count": int(matched_count),
            "new_object_count": int(promoted_count),
            "frozen_label_event_count": int(frozen_count),
            "active_object_count": int(object_state_counts.get(V2ObjectState.ACTIVE.value, 0)),
            "archived_object_count": int(object_state_counts.get(V2ObjectState.ARCHIVED.value, 0)),
            "provisional_bucket_count": int(len(self.state.provisional_buckets)),
            "coarse_owner_count": int(len(self.state.coarse_owners)),
            "object_pool_point_count_total": int(sum(pool.point_count for pool in self.state.object_pools.values())),
        }
        self.state.last_frame_debug = dict(self.last_frame_debug)
        return self.state

    def _accumulate_geometry(self, frame: Frame) -> None:
        rgb = frame.rgb[:: self.geometry_sample_stride, :: self.geometry_sample_stride]
        depth = frame.depth[:: self.geometry_sample_stride, :: self.geometry_sample_stride]
        u, v = np.meshgrid(
            np.arange(0, frame.rgb.shape[1], self.geometry_sample_stride),
            np.arange(0, frame.rgb.shape[0], self.geometry_sample_stride),
        )
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return

        z = depth[valid].astype(np.float32)
        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)
        x = (u - frame.intrinsics.cx) * z / frame.intrinsics.fx
        y = (v - frame.intrinsics.cy) * z / frame.intrinsics.fy
        points_cam = np.stack([x, y, z], axis=1)
        R = frame.pose[:3, :3].astype(np.float32)
        t = frame.pose[:3, 3].astype(np.float32)
        points_world = (R @ points_cam.T).T + t
        colors = rgb[valid].astype(np.float32)

        voxel_indices = np.floor(points_world / self.geometry_voxel_size).astype(np.int32)
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        counts_per_voxel = np.bincount(inverse)

        sum_points = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        sum_colors = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        for axis in range(3):
            sum_points[:, axis] = np.bincount(inverse, weights=points_world[:, axis], minlength=len(unique_voxels))
            sum_colors[:, axis] = np.bincount(inverse, weights=colors[:, axis], minlength=len(unique_voxels))

        mean_points = (sum_points / counts_per_voxel[:, None]).astype(np.float32)
        mean_colors = (sum_colors / counts_per_voxel[:, None]).astype(np.float32)

        for voxel, point, color, local_count in zip(unique_voxels, mean_points, mean_colors, counts_per_voxel):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            if key not in self.state.geometry_accum:
                self.state.geometry_accum[key] = [point.astype(np.float64), color.astype(np.float64), int(local_count)]
            else:
                prev_point, prev_color, prev_count = self.state.geometry_accum[key]
                total = prev_count + int(local_count)
                self.state.geometry_accum[key][0] = (prev_point * prev_count + point * local_count) / total
                self.state.geometry_accum[key][1] = (prev_color * prev_count + color * local_count) / total
                self.state.geometry_accum[key][2] = total

    def _build_runtime_state(self) -> SystemState:
        objects: Dict[int, ObjectMap] = {}
        for object_id, pool in self.state.object_pools.items():
            if pool.point_count == 0:
                continue
            state = ObjectState.ACTIVE if pool.state == V2ObjectState.ACTIVE else ObjectState.DORMANT
            whole_evidence = WholeEvidenceScores(
                whole_evidence_score=float(min(1.0, max(pool.observation_count, 1) / 3.0)),
                part_evidence_score=0.0,
                assignment_score=float(min(1.0, max(pool.observation_count, 1) / 5.0)),
            )
            obj = ObjectMap(
                object_id=int(object_id),
                state=state,
                local_pcd=np.asarray(pool.points, dtype=np.float32).copy(),
                centroid=np.asarray(pool.centroid, dtype=np.float32).copy(),
                bbox_min=np.asarray(pool.bbox_min, dtype=np.float32).copy(),
                bbox_max=np.asarray(pool.bbox_max, dtype=np.float32).copy(),
                whole_evidence=whole_evidence,
                last_seen_frame=int(pool.last_seen_frame),
                creation_frame=max(0, int(pool.last_seen_frame) - max(pool.observation_count - 1, 0)),
                update_count=int(max(pool.observation_count, 1)),
            )
            objects[int(object_id)] = obj
        return SystemState(objects=objects, next_object_id=self.state.next_object_id, frame_count=self.state.frame_count)

    def _attach_anchor_summary(self, raw_proposals: list[Any], merged_proposals: list[Any]) -> list[Any]:
        raw_lookup = {int(proposal.proposal_id): proposal for proposal in raw_proposals}
        merged = []
        for proposal in merged_proposals:
            member_ids = [int(value) for value in proposal.metadata.get("member_mask_ids", [proposal.proposal_id])]
            best_label = ""
            best_conf = 0.0
            label_votes: Dict[str, float] = defaultdict(float)
            anchor_hits = 0
            for raw_id in member_ids:
                raw = raw_lookup.get(int(raw_id))
                if raw is None:
                    continue
                label = str(raw.metadata.get("anchor_class_name", "")).strip()
                conf = float(raw.metadata.get("anchor_confidence", 0.0))
                if not label:
                    continue
                anchor_hits += 1
                label_votes[label] += conf
                if conf > best_conf:
                    best_conf = conf
                    best_label = label
            proposal.metadata = dict(
                proposal.metadata,
                anchor_class_name=best_label,
                anchor_confidence=float(best_conf),
                anchor_label_votes=dict(sorted(label_votes.items())),
                anchor_hit_count=int(anchor_hits),
            )
            merged.append(proposal)
        return merged

    def _match_patch_to_object_id(self, patch: Any) -> int | None:
        coarse_votes = self._coarse_votes_for_patch(patch.points)
        candidate_ids: set[int] = set(coarse_votes)
        for object_id, pool in self.state.object_pools.items():
            if pool.point_count == 0:
                continue
            dist = float(np.linalg.norm(np.asarray(patch.centroid, dtype=np.float32) - pool.centroid))
            if dist <= self.nearby_match_radius:
                candidate_ids.add(int(object_id))
        if not candidate_ids:
            return None

        patch_voxels = self._point_voxel_keys(np.asarray(patch.points, dtype=np.float32), self.pool_voxel_size)
        best_id = None
        best_score = -1.0
        for object_id in candidate_ids:
            pool = self.state.object_pools.get(int(object_id))
            if pool is None or pool.point_count == 0:
                continue
            total = self._match_score(pool, patch, patch_voxels, coarse_votes)
            if total > best_score:
                best_score = total
                best_id = int(object_id)
        if best_id is None or best_score < self.match_threshold:
            return None
        return int(best_id)

    def _match_score(
        self,
        pool: ObjectPool,
        patch: Any,
        patch_voxels: set[tuple[int, int, int]],
        coarse_votes: Dict[int, float],
    ) -> float:
        total_vote = max(sum(coarse_votes.values()), 1e-6)
        coarse_score = float(coarse_votes.get(int(pool.object_id), 0.0) / total_vote)
        dist = float(np.linalg.norm(np.asarray(patch.centroid, dtype=np.float32) - pool.centroid))
        centroid_score = max(0.0, 1.0 - dist / max(self.match_distance_scale, 1e-6))
        bbox_score = bbox_iou_3d(
            np.asarray(patch.bbox_min, dtype=np.float32),
            np.asarray(patch.bbox_max, dtype=np.float32),
            np.asarray(pool.bbox_min, dtype=np.float32),
            np.asarray(pool.bbox_max, dtype=np.float32),
        )
        overlap_den = max(len(patch_voxels), 1)
        overlap_score = float(len(patch_voxels & pool.spatial_index_keys) / overlap_den) if pool.spatial_index_keys else 0.0
        return float(
            self.coarse_weight * coarse_score
            + self.centroid_weight * centroid_score
            + self.bbox_weight * bbox_score
            + self.overlap_weight * overlap_score
        )

    def _coarse_votes_for_patch(self, points: np.ndarray) -> Dict[int, float]:
        votes: Dict[int, float] = defaultdict(float)
        for voxel_key in self._point_voxel_keys(points, self.coarse_voxel_size):
            owner = self.state.coarse_owners.get(voxel_key)
            if owner is None or owner.owner_object_id < 0:
                continue
            votes[int(owner.owner_object_id)] += float(owner.owner_confidence)
        return votes

    def _update_provisional_bucket(self, patch: Any, frame_id: int) -> ObjectPool | None:
        bucket = self._match_provisional_bucket(patch)
        if bucket is None:
            bucket = ProvisionalSpatialBucket(bucket_id=int(self.state.next_bucket_id))
            self.state.provisional_buckets[bucket.bucket_id] = bucket
            self.state.next_bucket_id += 1
        patch_confidence = self._patch_confidence(patch)
        self._merge_points_into_bucket(bucket, np.asarray(patch.points, dtype=np.float32), patch_confidence, frame_id)
        bucket.hit_count += 1
        bucket.last_seen_frame = int(frame_id)
        self._update_bucket_label_history(bucket, patch, frame_id)
        if bucket.hit_count < self.provisional_promotion_hits:
            return None
        promoted = self._promote_bucket(bucket, frame_id)
        del self.state.provisional_buckets[bucket.bucket_id]
        return promoted

    def _match_provisional_bucket(self, patch: Any) -> ProvisionalSpatialBucket | None:
        best = None
        best_dist = float("inf")
        patch_centroid = np.asarray(patch.centroid, dtype=np.float32)
        for bucket in self.state.provisional_buckets.values():
            dist = float(np.linalg.norm(bucket.centroid - patch_centroid))
            if dist > self.provisional_match_distance:
                continue
            if dist < best_dist:
                best = bucket
                best_dist = dist
        return best

    def _promote_bucket(self, bucket: ProvisionalSpatialBucket, frame_id: int) -> ObjectPool:
        object_id = int(self.state.next_object_id)
        self.state.next_object_id += 1
        label, confidence, frozen = self._derive_semantic_state(
            bucket.label_candidate_history,
            fallback_votes=bucket.candidate_label_votes,
        )
        pool = ObjectPool(
            object_id=object_id,
            canonical_label=label,
            label_confidence=float(confidence),
            label_frozen=bool(frozen),
            label_candidate_history=deepcopy(bucket.label_candidate_history),
            points=bucket.points.copy(),
            point_observation_count=bucket.point_observation_count.copy(),
            point_last_seen_frame=bucket.point_last_seen_frame.copy(),
            point_confidence=bucket.point_confidence.copy(),
            centroid=bucket.centroid.copy(),
            bbox_min=bucket.bbox_min.copy(),
            bbox_max=bucket.bbox_max.copy(),
            spatial_index_keys=set(bucket.spatial_index_keys),
            last_seen_frame=int(frame_id),
            observation_count=int(bucket.hit_count),
            state=V2ObjectState.ACTIVE,
            debug={"promoted_from_bucket": int(bucket.bucket_id)},
        )
        self.state.object_pools[object_id] = pool
        return pool

    def _update_pool_semantics(self, pool: ObjectPool, patch: Any, frame_id: int) -> bool:
        label = str(patch.metadata.get("anchor_class_name", "")).strip()
        confidence = float(patch.metadata.get("anchor_confidence", 0.0))
        if not label:
            return False
        record = pool.label_candidate_history.setdefault(
            label,
            {"score_sum": 0.0, "max_confidence": 0.0, "high_conf_frames": set()},
        )
        record["score_sum"] = float(record["score_sum"]) + confidence
        record["max_confidence"] = max(float(record["max_confidence"]), confidence)
        if confidence >= self.freeze_confidence_threshold:
            record["high_conf_frames"].add(int(frame_id))
        canonical_label, label_confidence, frozen = self._derive_semantic_state(
            pool.label_candidate_history,
            fallback_votes=None,
        )
        previously_frozen = bool(pool.label_frozen)
        if not pool.label_frozen or frozen:
            pool.canonical_label = canonical_label
            pool.label_confidence = float(label_confidence)
            pool.label_frozen = bool(frozen)
        return bool(pool.label_frozen and not previously_frozen)

    def _update_bucket_label_history(self, bucket: ProvisionalSpatialBucket, patch: Any, frame_id: int) -> None:
        label = str(patch.metadata.get("anchor_class_name", "")).strip()
        confidence = float(patch.metadata.get("anchor_confidence", 0.0))
        if label:
            bucket.candidate_label_votes[label] = float(bucket.candidate_label_votes.get(label, 0.0) + confidence)
            record = bucket.label_candidate_history.setdefault(
                label,
                {"score_sum": 0.0, "max_confidence": 0.0, "high_conf_frames": set()},
            )
            record["score_sum"] = float(record["score_sum"]) + confidence
            record["max_confidence"] = max(float(record["max_confidence"]), confidence)
            if confidence >= self.freeze_confidence_threshold:
                record["high_conf_frames"].add(int(frame_id))

    def _derive_semantic_state(
        self,
        label_history: Dict[str, Dict[str, Any]],
        *,
        fallback_votes: Dict[str, float] | None,
    ) -> tuple[str, float, bool]:
        best_label = ""
        best_confidence = 0.0
        best_tuple = (-1, -1.0, -1.0, "")
        for label, record in label_history.items():
            high_conf_hits = len(record.get("high_conf_frames", set()))
            max_confidence = float(record.get("max_confidence", 0.0))
            score_sum = float(record.get("score_sum", 0.0))
            candidate_tuple = (high_conf_hits, max_confidence, score_sum, label)
            if candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_label = str(label)
                best_confidence = max_confidence
        if not best_label and fallback_votes:
            best_label, best_confidence = max(
                ((label, float(score)) for label, score in fallback_votes.items()),
                key=lambda item: (item[1], item[0]),
                default=("", 0.0),
            )
        frozen = bool(best_label and best_tuple[0] >= self.freeze_hits)
        return best_label, best_confidence, frozen

    def _merge_patch_into_pool(self, pool: ObjectPool, patch: Any, frame_id: int) -> None:
        patch_confidence = self._patch_confidence(patch)
        self._merge_points_into_pool(pool, np.asarray(patch.points, dtype=np.float32), patch_confidence, frame_id)
        pool.last_seen_frame = int(frame_id)
        pool.observation_count += 1
        pool.state = V2ObjectState.ACTIVE
        pool.debug["last_patch_point_count"] = int(len(patch.points))
        pool.debug["label_source"] = str(patch.metadata.get("anchor_class_name", ""))

    def _merge_points_into_pool(self, pool: ObjectPool, points: np.ndarray, confidence: float, frame_id: int) -> None:
        patch_points, patch_confidences = self._downsample_points(points, confidence, self.pool_voxel_size)
        if patch_points.size == 0:
            return
        if pool.point_count == 0:
            pool.points = patch_points
            pool.point_observation_count = np.ones(len(patch_points), dtype=np.int32)
            pool.point_last_seen_frame = np.full(len(patch_points), int(frame_id), dtype=np.int32)
            pool.point_confidence = patch_confidences
        else:
            self._merge_points_into_arrays(
                pool_points=pool.points,
                pool_counts=pool.point_observation_count,
                pool_frames=pool.point_last_seen_frame,
                pool_confidences=pool.point_confidence,
                new_points=patch_points,
                new_confidences=patch_confidences,
                frame_id=frame_id,
                voxel_size=self.pool_voxel_size,
            )
        self._cap_object_pool(pool)
        self._refresh_pool_geometry(pool)

    def _merge_points_into_bucket(self, bucket: ProvisionalSpatialBucket, points: np.ndarray, confidence: float, frame_id: int) -> None:
        patch_points, patch_confidences = self._downsample_points(points, confidence, self.pool_voxel_size)
        if patch_points.size == 0:
            return
        if len(bucket.points) == 0:
            bucket.points = patch_points
            bucket.point_observation_count = np.ones(len(patch_points), dtype=np.int32)
            bucket.point_last_seen_frame = np.full(len(patch_points), int(frame_id), dtype=np.int32)
            bucket.point_confidence = patch_confidences
        else:
            self._merge_points_into_arrays(
                pool_points=bucket.points,
                pool_counts=bucket.point_observation_count,
                pool_frames=bucket.point_last_seen_frame,
                pool_confidences=bucket.point_confidence,
                new_points=patch_points,
                new_confidences=patch_confidences,
                frame_id=frame_id,
                voxel_size=self.pool_voxel_size,
            )
        self._refresh_bucket_geometry(bucket)

    @staticmethod
    def _merge_points_into_arrays(
        *,
        pool_points: np.ndarray,
        pool_counts: np.ndarray,
        pool_frames: np.ndarray,
        pool_confidences: np.ndarray,
        new_points: np.ndarray,
        new_confidences: np.ndarray,
        frame_id: int,
        voxel_size: float,
    ) -> None:
        existing_voxels = np.floor(pool_points / voxel_size).astype(np.int32)
        existing_lookup = {tuple(map(int, voxel)): idx for idx, voxel in enumerate(existing_voxels.tolist())}
        append_points = []
        append_counts = []
        append_frames = []
        append_confidences = []

        for point, confidence in zip(new_points, new_confidences):
            key = tuple(int(value) for value in np.floor(point / voxel_size).astype(np.int32).tolist())
            existing_idx = existing_lookup.get(key)
            if existing_idx is None:
                existing_lookup[key] = len(pool_points) + len(append_points)
                append_points.append(point.astype(np.float32, copy=False))
                append_counts.append(1)
                append_frames.append(int(frame_id))
                append_confidences.append(float(confidence))
                continue
            count = int(pool_counts[existing_idx])
            pool_points[existing_idx] = (
                (pool_points[existing_idx] * count + point.astype(np.float32)) / float(count + 1)
            )
            pool_counts[existing_idx] = count + 1
            pool_frames[existing_idx] = int(frame_id)
            pool_confidences[existing_idx] = max(float(pool_confidences[existing_idx]), float(confidence))

        if append_points:
            pool_points.resize((len(pool_points) + len(append_points), 3), refcheck=False)
            pool_points[-len(append_points) :] = np.asarray(append_points, dtype=np.float32)
            pool_counts.resize((len(pool_counts) + len(append_counts),), refcheck=False)
            pool_counts[-len(append_counts) :] = np.asarray(append_counts, dtype=np.int32)
            pool_frames.resize((len(pool_frames) + len(append_frames),), refcheck=False)
            pool_frames[-len(append_frames) :] = np.asarray(append_frames, dtype=np.int32)
            pool_confidences.resize((len(pool_confidences) + len(append_confidences),), refcheck=False)
            pool_confidences[-len(append_confidences) :] = np.asarray(append_confidences, dtype=np.float32)

    def _cap_object_pool(self, pool: ObjectPool) -> None:
        if pool.point_count <= self.max_points_per_object:
            return
        indices = self._deterministic_spatial_indices(pool.points, self.max_points_per_object)
        pool.points = pool.points[indices]
        pool.point_observation_count = pool.point_observation_count[indices]
        pool.point_last_seen_frame = pool.point_last_seen_frame[indices]
        pool.point_confidence = pool.point_confidence[indices]

    def _refresh_pool_geometry(self, pool: ObjectPool) -> None:
        if pool.point_count == 0:
            pool.centroid = np.zeros(3, dtype=np.float32)
            pool.bbox_min = np.zeros(3, dtype=np.float32)
            pool.bbox_max = np.zeros(3, dtype=np.float32)
            pool.spatial_index_keys = set()
            return
        pool.centroid = pool.points.mean(axis=0).astype(np.float32)
        pool.bbox_min, pool.bbox_max = compute_bbox(pool.points)
        pool.spatial_index_keys = self._point_voxel_keys(pool.points, self.pool_voxel_size)

    def _refresh_bucket_geometry(self, bucket: ProvisionalSpatialBucket) -> None:
        if len(bucket.points) == 0:
            bucket.centroid = np.zeros(3, dtype=np.float32)
            bucket.bbox_min = np.zeros(3, dtype=np.float32)
            bucket.bbox_max = np.zeros(3, dtype=np.float32)
            bucket.spatial_index_keys = set()
            return
        bucket.centroid = bucket.points.mean(axis=0).astype(np.float32)
        bucket.bbox_min, bucket.bbox_max = compute_bbox(bucket.points)
        bucket.spatial_index_keys = self._point_voxel_keys(bucket.points, self.pool_voxel_size)

    def _update_coarse_ownership(self, object_id: int, points: np.ndarray, frame_id: int) -> None:
        object_pool = self.state.object_pools.get(int(object_id))
        label = object_pool.canonical_label if object_pool is not None else ""
        for voxel_key in self._point_voxel_keys(points, self.coarse_voxel_size):
            support = self.state.coarse_support.setdefault(voxel_key, {})
            support[int(object_id)] = float(support.get(int(object_id), 0.0) + self.coarse_support_increment)
            for other_id in list(support.keys()):
                if int(other_id) == int(object_id):
                    continue
                support[other_id] *= self.coarse_support_decay
                if support[other_id] < 0.01:
                    del support[other_id]
            owner_object_id = max(support, key=support.get)
            total = max(sum(support.values()), 1e-6)
            owner_label = self.state.object_pools.get(int(owner_object_id)).canonical_label if int(owner_object_id) in self.state.object_pools else label
            self.state.coarse_owners[voxel_key] = CoarseVoxelOwner(
                voxel_key=voxel_key,
                owner_object_id=int(owner_object_id),
                owner_confidence=float(support[owner_object_id] / total),
                canonical_label=str(owner_label),
                last_seen_frame=int(frame_id),
            )

    def _prune_provisional_buckets(self, frame_id: int) -> None:
        stale = [
            bucket_id
            for bucket_id, bucket in self.state.provisional_buckets.items()
            if int(frame_id) - int(bucket.last_seen_frame) > self.provisional_max_idle_frames
        ]
        for bucket_id in stale:
            del self.state.provisional_buckets[bucket_id]

    def _update_archive_states(self, frame_id: int) -> None:
        for pool in self.state.object_pools.values():
            if int(frame_id) - int(pool.last_seen_frame) > self.archive_after_frames:
                pool.state = V2ObjectState.ARCHIVED

    @staticmethod
    def _patch_confidence(patch: Any) -> float:
        return float(
            max(
                float(patch.metadata.get("anchor_confidence", 0.0)),
                float(getattr(patch.soft_scores, "objectness_score", 0.0)),
            )
        )

    @staticmethod
    def _point_voxel_keys(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
        if points.size == 0:
            return set()
        voxel_indices = np.floor(np.asarray(points, dtype=np.float32) / voxel_size).astype(np.int32)
        return {tuple(int(value) for value in voxel.tolist()) for voxel in voxel_indices}

    @staticmethod
    def _downsample_points(points: np.ndarray, confidence: float, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        downsampled = np.zeros((len(unique_voxels), 3), dtype=np.float32)
        for axis in range(3):
            downsampled[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts
        confidences = np.full(len(unique_voxels), float(confidence), dtype=np.float32)
        return downsampled, confidences

    @staticmethod
    def _deterministic_spatial_indices(points: np.ndarray, max_points: int) -> np.ndarray:
        if max_points <= 0 or len(points) <= max_points:
            return np.arange(len(points), dtype=np.int32)
        bbox_min = points.min(axis=0)
        bbox_span = np.maximum(points.max(axis=0) - bbox_min, 1e-6)
        grid_resolution = max(1, int(np.ceil(max_points ** (1.0 / 3.0))))
        normalized = np.clip((points - bbox_min) / bbox_span, 0.0, 1.0 - 1e-6)
        voxel_indices = np.floor(normalized * grid_resolution).astype(np.int32)
        order = np.lexsort(
            (
                points[:, 2],
                points[:, 1],
                points[:, 0],
                voxel_indices[:, 2],
                voxel_indices[:, 1],
                voxel_indices[:, 0],
            )
        )
        sorted_voxels = voxel_indices[order]
        _, start_indices, counts = np.unique(
            sorted_voxels,
            axis=0,
            return_index=True,
            return_counts=True,
        )
        if len(start_indices) >= max_points:
            selected = np.linspace(0, len(start_indices) - 1, num=max_points, dtype=np.int32)
            return order[start_indices[selected]]

        selected_rows = [int(start_index) for start_index in start_indices.tolist()]
        layer = 1
        max_count = int(counts.max()) if len(counts) > 0 else 0
        while len(selected_rows) < max_points and layer < max_count:
            available = [idx for idx, count in enumerate(counts.tolist()) if count > layer]
            if not available:
                break
            remaining = max_points - len(selected_rows)
            if len(available) > remaining:
                sample_positions = np.linspace(0, len(available) - 1, num=remaining, dtype=np.int32)
                available = [available[int(position)] for position in sample_positions.tolist()]
            for voxel_index in available:
                selected_rows.append(int(start_indices[voxel_index] + layer))
                if len(selected_rows) >= max_points:
                    break
            layer += 1
        selected_rows = np.asarray(selected_rows[:max_points], dtype=np.int32)
        return order[selected_rows]
