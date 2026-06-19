"""Module 6: Object Association (Layer 4 — spatial voting).

Responsibility: Associate object patches to existing object instances
using TSDF-based spatial voting. Association is geometry-first /
spatial-first — semantics must not drive low-level association (Rule C).

Exposes explicit voting scores:
  - touched voxels, owner votes, normalized vote score
  - optional local geometry consistency
  - final association score
  - whether a new instance is created
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
import time
from typing import Any, Dict, List

import numpy as np

try:  # Optional acceleration; keep numpy fallback when scipy is unavailable.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised by monkeypatch/fallback tests.
    cKDTree = None

from src.core.data_structures import (
    ActiveSet,
    AssociationResult,
    AssociationScore,
    ContestedAssociation,
    ObjectMap,
    Patch3D,
    TSDFInstanceVolume,
    VoxelVoteResult,
)
from src.modules.observation_identity import classify_observation_identity
from src.modules.semantic_memory import object_association_identity_semantic_label
from src.modules.tsdf_instance_map import TSDFInstanceMapModule
from src.utils.geometry import bbox_iou_3d

logger = logging.getLogger("oviovo.modules.association")


class AssociationModule:
    """Geometry-first association of patches to existing objects.

    Uses TSDF spatial voting as primary signal, with optional
    centroid/bbox/geometry fallback for objects not yet in TSDF.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.w_voxel_vote = config.get("voxel_vote_weight", 0.5)
        self.w_centroid = config.get("centroid_distance_weight", 0.2)
        self.w_bbox = config.get("bbox_overlap_weight", 0.2)
        self.w_geometry = config.get("geometry_overlap_weight", 0.1)
        self.match_threshold = config.get("match_threshold", 0.3)
        self.two_stage_enabled = bool(config.get("two_stage_enabled", True))
        self.max_scored_candidates = int(config.get("max_scored_candidates", 0))
        self.max_geometry_candidates = int(config.get("max_geometry_candidates", 0))
        self.top_k_geometry = int(config.get("top_k_geometry", self.max_geometry_candidates))
        self.top_k_final = int(config.get("top_k_final", self.max_scored_candidates))
        self.geometry_candidate_min_cheap_score = float(config.get("geometry_candidate_min_cheap_score", 0.0))
        self.geometry_vote_owner_priority = bool(config.get("geometry_vote_owner_priority", True))
        self.geometry_nn_patch_sample = int(config.get("geometry_nn_patch_sample", 200))
        self.geometry_nn_object_sample = int(config.get("geometry_nn_object_sample", 200))
        self.geometry_nn_backend = str(config.get("geometry_nn_backend", "auto")).strip().lower()
        if self.geometry_nn_backend not in {"auto", "vectorized", "kdtree"}:
            logger.warning(
                "Unknown association.geometry_nn_backend=%r; falling back to auto.",
                self.geometry_nn_backend,
            )
            self.geometry_nn_backend = "auto"
        self.geometry_vectorized_max_points = int(config.get("geometry_vectorized_max_points", 160))
        self.indexed_retrieval_enabled = bool(config.get("indexed_retrieval_enabled", False))
        self.centroid_kdtree_enabled = bool(config.get("centroid_kdtree_enabled", False))
        self.max_spatial_candidates = int(config.get("max_spatial_candidates", 0))
        self.spatial_candidate_radius = float(config.get("spatial_candidate_radius", 0.0))
        self.kdtree_cache_enabled = bool(
            config.get("geometry_kdtree_cache_enabled", config.get("kdtree_cache_enabled", False))
        )
        self.kdtree_cache_min_object_points = int(
            config.get("geometry_kdtree_cache_min_points", config.get("kdtree_cache_min_object_points", 256))
        )
        self.semantic_label_candidate_boost = float(config.get("semantic_label_candidate_boost", 0.0))
        self.semantic_conflict_gate_enabled = bool(config.get("semantic_conflict_gate_enabled", False))
        self.semantic_conflict_min_patch_confidence = float(
            config.get("semantic_conflict_min_patch_confidence", 0.35)
        )
        self.observation_identity_gate_enabled = bool(
            config.get("observation_identity_gate_enabled", config.get("semantic_conflict_gate_enabled", False))
        )
        self.observation_identity_min_patch_confidence = float(
            config.get(
                "observation_identity_min_patch_confidence",
                config.get("semantic_conflict_min_patch_confidence", 0.35),
            )
        )
        self.score_parallel_enabled = bool(config.get("score_parallel_enabled", False))
        self.score_parallel_workers = int(config.get("score_parallel_workers", 0))
        self.score_parallel_min_candidates = int(config.get("score_parallel_min_candidates", 16))
        self.semantic_conflict_min_object_confidence = float(
            config.get("semantic_conflict_min_object_confidence", 0.35)
        )
        self.semantic_conflict_subregion_point_ratio = float(
            config.get("semantic_conflict_subregion_point_ratio", 0.35)
        )
        self.semantic_conflict_subregion_volume_ratio = float(
            config.get("semantic_conflict_subregion_volume_ratio", 0.35)
        )
        self._association_kdtree_cache: dict[tuple[Any, ...], Any] = {}
        self._association_kdtree_cache_lock = threading.Lock()
        self.association_kdtree_cache_hit_count = 0
        self.association_kdtree_cache_miss_count = 0
        self.association_geometry_score_sec = 0.0
        self.tsdf_module = TSDFInstanceMapModule(config.get("tsdf", {}))
        logger.info("AssociationModule initialized.")

    def process(
        self,
        object_patches: List[Patch3D],
        objects: Dict[int, ObjectMap],
        tsdf_volume: TSDFInstanceVolume,
        active_set: ActiveSet | None = None,
    ) -> AssociationResult:
        """Associate patches to existing objects via spatial voting.

        Args:
            object_patches: 3D patches classified as objects.
            objects: Current object map.
            tsdf_volume: Global TSDF instance volume for spatial voting.

        Returns:
            AssociationResult with matches and new-object requests.
        """
        result = AssociationResult()
        kdtree_hit_start = int(self.association_kdtree_cache_hit_count)
        kdtree_miss_start = int(self.association_kdtree_cache_miss_count)
        geometry_score_sec_start = float(self.association_geometry_score_sec)
        all_candidate_ids = [
            obj_id for obj_id, obj in objects.items()
            if obj.state.value != "removed"
        ]
        active_candidate_ids = self._resolve_candidate_ids(objects, active_set)
        use_active_set = active_set is not None
        centroid_index = self._build_centroid_candidate_index(
            objects,
            active_candidate_ids if use_active_set else all_candidate_ids,
        )
        result.debug = {
            "used_active_set": use_active_set,
            "active_set_candidate_ids": active_candidate_ids,
            "fallback_to_all_objects": False,
            "candidate_source": "active_set" if use_active_set else "global_object_map",
            "per_patch": {},
        }

        for patch in object_patches:
            # Step 1: TSDF spatial voting
            vote = self.tsdf_module.vote_patch_to_instance(tsdf_volume, patch)
            candidate_ids = active_candidate_ids if use_active_set else all_candidate_ids
            candidate_count_before_indexed = int(len(candidate_ids))
            candidate_ids, indexed_debug = self._select_indexed_candidate_ids(
                patch,
                objects,
                list(candidate_ids),
                vote,
                centroid_index,
            )
            patch_debug = {
                "candidate_object_ids": list(candidate_ids),
                "candidate_count_before_indexed_retrieval": candidate_count_before_indexed,
                "candidate_count_after_indexed_retrieval": int(len(candidate_ids)),
                "indexed_retrieval": indexed_debug,
                "vote": {
                    "touched_voxel_count": vote.touched_voxel_count,
                    "supported_voxel_count": vote.supported_voxel_count,
                    "owner_votes": dict(vote.owner_votes),
                    "best_instance_id": vote.best_instance_id,
                    "normalized_vote_score": vote.normalized_vote_score,
                },
            }
            geometry_candidate_ids, cheap_by_id = self._select_geometry_candidate_ids(
                patch,
                objects,
                list(candidate_ids),
                vote,
            )
            patch_debug["candidate_count_before_geometry"] = int(len(candidate_ids))
            patch_debug["geometry_candidate_object_ids"] = [
                int(obj_id) for obj_id in candidate_ids if int(obj_id) in geometry_candidate_ids
            ]
            patch_debug["geometry_candidate_count"] = int(len(geometry_candidate_ids))
            scored_candidate_ids = self._select_scored_candidate_ids(list(candidate_ids), cheap_by_id, vote)
            patch_debug["candidate_count_before_scored_cap"] = int(len(candidate_ids))
            patch_debug["candidate_count_after_scored_cap"] = int(len(scored_candidate_ids))
            patch_debug["candidate_object_ids"] = list(scored_candidate_ids)
            geometry_scored_ids = [
                int(obj_id) for obj_id in scored_candidate_ids if int(obj_id) in geometry_candidate_ids
            ]
            patch_debug["geometry_scored_object_ids"] = list(geometry_scored_ids)
            patch_debug["geometry_scored_count"] = int(len(geometry_scored_ids))

            best_score = None
            best_obj_id = None
            blocked_candidates: list[dict[str, Any]] = []

            scored_candidates, used_parallel = self._score_candidate_objects(
                patch,
                objects,
                scored_candidate_ids,
                vote,
                cheap_by_id,
                set(geometry_scored_ids),
            )
            patch_debug["score_parallel_used"] = bool(used_parallel)
            patch_debug["score_parallel_candidate_count"] = int(len(scored_candidates)) if used_parallel else 0

            for obj_id, score in scored_candidates:
                result.scores[(patch.patch_id, obj_id)] = score

                if score.total_score > self.match_threshold:
                    obj = objects[obj_id]
                    identity_decision = classify_observation_identity(
                        patch,
                        obj,
                        min_patch_confidence=self.observation_identity_min_patch_confidence,
                    )
                    if self.observation_identity_gate_enabled and not identity_decision.can_update:
                        blocked_candidates.append(
                            {
                                "object_id": int(obj_id),
                                "score": float(score.total_score),
                                "reason": str(identity_decision.reason),
                                "patch_label": str(identity_decision.patch_label),
                                "object_label": str(identity_decision.object_label),
                                "relation": str(identity_decision.relation),
                                "patch_confidence": float(identity_decision.patch_confidence),
                            }
                        )
                        continue
                    if best_score is None or score.total_score > best_score.total_score:
                        best_score = score
                        best_obj_id = obj_id

            if best_obj_id is not None:
                result.matched.append((patch.patch_id, best_obj_id, best_score))
                patch_debug["selected_object_id"] = best_obj_id
                patch_debug["final_association_score"] = float(best_score.total_score)
                patch_debug["new_instance_created"] = False
                patch_debug["final_outcome"] = "matched"
            else:
                if blocked_candidates:
                    best_blocked = max(blocked_candidates, key=lambda item: float(item["score"]))
                    blocked_obj_id = int(best_blocked["object_id"])
                    result.contested_object_patches.append(patch.patch_id)
                    result.contested_matches.append(
                        ContestedAssociation(
                            patch_id=patch.patch_id,
                            blocked_object_id=blocked_obj_id,
                            patch_label=str(best_blocked["patch_label"]),
                            object_label=str(best_blocked["object_label"]),
                            reason=str(best_blocked["reason"]),
                            score=result.scores[(patch.patch_id, blocked_obj_id)],
                            debug={
                                "relation": str(best_blocked["relation"]),
                                "patch_confidence": float(best_blocked["patch_confidence"]),
                            },
                        )
                    )
                    patch.metadata["contested_parent_object_id"] = blocked_obj_id
                    patch.metadata["contested_parent_label"] = str(best_blocked["object_label"])
                    patch.metadata["contested_patch_label"] = str(best_blocked["patch_label"])
                    patch.metadata["contested_reason"] = str(best_blocked["reason"])
                    patch.metadata["semantic_split_candidate_from_object_id"] = blocked_obj_id
                    patch.metadata["semantic_split_candidate_parent_label"] = str(best_blocked["object_label"])
                    patch.metadata["semantic_split_candidate_new_label"] = str(best_blocked["patch_label"])
                    patch.metadata["semantic_split_candidate_reason"] = str(best_blocked["reason"])
                    patch_debug["selected_object_id"] = None
                    patch_debug["blocked_object_id"] = blocked_obj_id
                    patch_debug["final_association_score"] = float(best_blocked["score"])
                    patch_debug["new_instance_created"] = False
                    patch_debug["final_outcome"] = "contested_residual"
                else:
                    result.new_object_patches.append(patch.patch_id)
                    vote.new_instance_created = True
                    patch_debug["selected_object_id"] = None
                    patch_debug["final_association_score"] = 0.0
                    patch_debug["new_instance_created"] = True
                    patch_debug["final_outcome"] = "new_object"

            patch_debug["semantic_conflict_blocked_candidates"] = blocked_candidates
            patch_debug["observation_identity_blocked_candidates"] = blocked_candidates
            if blocked_candidates:
                patch_debug["blocked_candidate_source"] = "observation_identity"
            result.debug["per_patch"][patch.patch_id] = patch_debug

        per_patch_debug = result.debug.get("per_patch", {}) or {}
        candidate_score_count = int(
            sum(len(item.get("candidate_object_ids", []) or []) for item in per_patch_debug.values())
        )
        geometry_score_count = int(
            sum(int(item.get("geometry_scored_count", 0)) for item in per_patch_debug.values())
        )
        score_parallel_used_count = int(
            sum(1 for item in per_patch_debug.values() if bool(item.get("score_parallel_used", False)))
        )
        score_parallel_candidate_count_total = int(
            sum(int(item.get("score_parallel_candidate_count", 0)) for item in per_patch_debug.values())
        )
        indexed_debug_items = [
            dict(item.get("indexed_retrieval", {}) or {}) for item in per_patch_debug.values()
        ]
        indexed_pruned_candidate_count = int(
            sum(int(item.get("pruned_candidate_count", 0)) for item in indexed_debug_items)
        )
        indexed_query_count = int(sum(1 for item in indexed_debug_items if bool(item.get("query_used", False))))
        indexed_fallback_count = int(sum(1 for item in indexed_debug_items if bool(item.get("fallback_used", False))))
        indexed_query_sec = float(sum(float(item.get("query_sec", 0.0)) for item in indexed_debug_items))
        result.debug["summary"] = {
            "patch_count": int(len(object_patches)),
            "candidate_score_count": candidate_score_count,
            "geometry_score_count": geometry_score_count,
            "geometry_pruned_candidate_count": int(max(0, candidate_score_count - geometry_score_count)),
            "two_stage_enabled": bool(self.two_stage_enabled),
            "top_k_final": int(self.top_k_final),
            "top_k_geometry": int(self.top_k_geometry),
            "max_scored_candidates": int(self.max_scored_candidates),
            "max_geometry_candidates": int(self.max_geometry_candidates),
            "score_parallel_enabled": bool(self.score_parallel_enabled),
            "score_parallel_used_count": score_parallel_used_count,
            "score_parallel_candidate_count_total": score_parallel_candidate_count_total,
            "indexed_retrieval_enabled": bool(self.indexed_retrieval_enabled),
            "centroid_kdtree_enabled": bool(self.centroid_kdtree_enabled),
            "max_spatial_candidates": int(self.max_spatial_candidates),
            "spatial_candidate_radius": float(self.spatial_candidate_radius),
            "indexed_candidate_query_count": indexed_query_count,
            "indexed_candidate_fallback_count": indexed_fallback_count,
            "indexed_pruned_candidate_count": indexed_pruned_candidate_count,
            "indexed_candidate_query_sec": indexed_query_sec,
            "centroid_index_build_count": int(centroid_index.get("build_count", 0)),
            "centroid_index_build_sec": float(centroid_index.get("build_sec", 0.0)),
            "centroid_index_candidate_count": int(len(centroid_index.get("candidate_ids", []) or [])),
            "centroid_index_backend": str(centroid_index.get("backend", "disabled")),
            "geometry_kdtree_cache_enabled": bool(self.kdtree_cache_enabled),
            "geometry_kdtree_cache_min_points": int(self.kdtree_cache_min_object_points),
            "association_kdtree_cache_hit_count": int(
                self.association_kdtree_cache_hit_count - kdtree_hit_start
            ),
            "association_kdtree_cache_miss_count": int(
                self.association_kdtree_cache_miss_count - kdtree_miss_start
            ),
            "association_geometry_score_sec": float(
                self.association_geometry_score_sec - geometry_score_sec_start
            ),
        }

        logger.debug(
            f"Association: {len(result.matched)} matched, "
            f"{len(result.new_object_patches)} new objects requested."
        )
        return result

    def _resolve_candidate_ids(
        self,
        objects: Dict[int, ObjectMap],
        active_set: ActiveSet | None,
    ) -> List[int]:
        """Resolve the local candidate set from the active set."""
        if active_set is None or not active_set.all_candidate_ids:
            return []

        return sorted(
            obj_id
            for obj_id in active_set.all_candidate_ids
            if obj_id in objects and objects[obj_id].state.value != "removed"
        )

    def _build_centroid_candidate_index(
        self,
        objects: Dict[int, ObjectMap],
        candidate_ids: List[int],
    ) -> dict[str, Any]:
        if (
            not self.indexed_retrieval_enabled
            or not self.centroid_kdtree_enabled
            or self.max_spatial_candidates <= 0
        ):
            return {"backend": "disabled", "candidate_ids": list(candidate_ids)}

        start = time.perf_counter()
        valid_ids: list[int] = []
        centers: list[np.ndarray] = []
        for obj_id in candidate_ids:
            obj = objects.get(int(obj_id))
            if obj is None or obj.state.value == "removed":
                continue
            center = np.asarray(getattr(obj, "centroid", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                continue
            valid_ids.append(int(obj_id))
            centers.append(center)

        if not valid_ids:
            return {
                "backend": "empty",
                "candidate_ids": [],
                "build_count": 0,
                "build_sec": float(time.perf_counter() - start),
            }

        points = np.asarray(centers, dtype=np.float32)
        backend = "vectorized"
        tree = None
        if cKDTree is not None:
            try:
                tree = cKDTree(points)
                backend = "kdtree"
            except Exception:
                tree = None
                backend = "vectorized"

        return {
            "backend": backend,
            "candidate_ids": valid_ids,
            "points": points,
            "tree": tree,
            "build_count": 1,
            "build_sec": float(time.perf_counter() - start),
        }

    def _select_indexed_candidate_ids(
        self,
        patch: Patch3D,
        objects: Dict[int, ObjectMap],
        candidate_ids: List[int],
        vote: VoxelVoteResult,
        centroid_index: dict[str, Any],
    ) -> tuple[List[int], dict[str, Any]]:
        debug = {
            "enabled": bool(self.indexed_retrieval_enabled),
            "centroid_kdtree_enabled": bool(self.centroid_kdtree_enabled),
            "query_used": False,
            "fallback_used": False,
            "candidate_count_before": int(len(candidate_ids)),
            "candidate_count_after": int(len(candidate_ids)),
            "pruned_candidate_count": 0,
            "owner_vote_candidate_count": int(len(vote.owner_votes)),
            "backend": str(centroid_index.get("backend", "disabled")),
            "query_sec": 0.0,
        }
        if (
            not self.indexed_retrieval_enabled
            or not self.centroid_kdtree_enabled
            or self.max_spatial_candidates <= 0
            or len(candidate_ids) <= self.max_spatial_candidates
        ):
            return candidate_ids, debug

        start = time.perf_counter()
        candidate_set = {int(obj_id) for obj_id in candidate_ids}
        indexed_ids = [int(obj_id) for obj_id in centroid_index.get("candidate_ids", []) if int(obj_id) in candidate_set]
        points = np.asarray(centroid_index.get("points", np.empty((0, 3), dtype=np.float32)), dtype=np.float32)
        if not indexed_ids or len(points) != len(centroid_index.get("candidate_ids", []) or []):
            debug["fallback_used"] = True
            debug["query_sec"] = float(time.perf_counter() - start)
            return candidate_ids, debug

        patch_centroid = np.asarray(patch.centroid, dtype=np.float32)
        index_positions = [
            pos
            for pos, obj_id in enumerate(centroid_index.get("candidate_ids", []))
            if int(obj_id) in candidate_set
        ]
        if not index_positions:
            debug["fallback_used"] = True
            debug["query_sec"] = float(time.perf_counter() - start)
            return candidate_ids, debug

        limit = min(int(self.max_spatial_candidates), len(index_positions))
        selected_ids: set[int] = set()
        tree = centroid_index.get("tree")
        backend = str(centroid_index.get("backend", "vectorized"))
        try:
            if backend == "kdtree" and tree is not None and len(index_positions) == len(indexed_ids):
                query_k = len(indexed_ids) if self.spatial_candidate_radius > 0.0 else limit
                distances, indices = tree.query(patch_centroid, k=max(1, int(query_k)))
                distances_arr = np.atleast_1d(np.asarray(distances, dtype=np.float32))
                indices_arr = np.atleast_1d(np.asarray(indices, dtype=np.int64))
                for distance, index in zip(distances_arr, indices_arr):
                    if int(index) < 0 or int(index) >= len(indexed_ids):
                        continue
                    if self.spatial_candidate_radius > 0.0 and float(distance) > self.spatial_candidate_radius:
                        continue
                    selected_ids.add(indexed_ids[int(index)])
                    if len(selected_ids) >= limit:
                        break
            else:
                indexed_points = points[np.asarray(index_positions, dtype=np.int64)]
                distances = np.linalg.norm(indexed_points - patch_centroid[None, :], axis=1)
                if self.spatial_candidate_radius > 0.0:
                    order = np.argsort(distances, kind="stable")
                    for order_idx in order:
                        if float(distances[int(order_idx)]) > self.spatial_candidate_radius:
                            continue
                        selected_ids.add(int(centroid_index["candidate_ids"][index_positions[int(order_idx)]]))
                        if len(selected_ids) >= limit:
                            break
                else:
                    nearest = np.argsort(distances, kind="stable")[:limit]
                    selected_ids.update(
                        int(centroid_index["candidate_ids"][index_positions[int(order_idx)]])
                        for order_idx in nearest
                    )
        except Exception:
            debug["fallback_used"] = True
            debug["query_sec"] = float(time.perf_counter() - start)
            return candidate_ids, debug

        owner_vote_ids = {int(obj_id) for obj_id in vote.owner_votes if int(obj_id) in candidate_set}
        selected_ids.update(owner_vote_ids)
        if not selected_ids:
            debug["fallback_used"] = True
            debug["query_sec"] = float(time.perf_counter() - start)
            return candidate_ids, debug

        selected_candidate_ids = [int(obj_id) for obj_id in candidate_ids if int(obj_id) in selected_ids]
        debug.update(
            {
                "query_used": True,
                "candidate_count_after": int(len(selected_candidate_ids)),
                "pruned_candidate_count": int(max(0, len(candidate_ids) - len(selected_candidate_ids))),
                "selected_owner_vote_candidate_count": int(len(owner_vote_ids)),
                "query_sec": float(time.perf_counter() - start),
            }
        )
        return selected_candidate_ids, debug

    def _voxel_score_for_object(self, obj_id: int, vote: VoxelVoteResult) -> float:
        vote_denominator = max(vote.supported_voxel_count, 1)
        if vote.owner_votes and int(obj_id) in vote.owner_votes:
            return float(vote.owner_votes[int(obj_id)] / vote_denominator)
        return 0.0

    def _cheap_score_components(
        self,
        patch: Patch3D,
        obj: ObjectMap,
        vote: VoxelVoteResult,
    ) -> tuple[float, float, float, float, float]:
        voxel_score = self._voxel_score_for_object(int(obj.object_id), vote)
        dist = float(np.linalg.norm(patch.centroid - obj.centroid))
        centroid_score = max(0.0, 1.0 - dist / 2.0)
        bbox_score = bbox_iou_3d(patch.bbox_min, patch.bbox_max, obj.bbox_min, obj.bbox_max)
        cheap_total = (
            self.w_voxel_vote * voxel_score
            + self.w_centroid * centroid_score
            + self.w_bbox * bbox_score
        )
        ranking_total = cheap_total + self._semantic_label_ranking_boost(patch, obj)
        return voxel_score, centroid_score, bbox_score, cheap_total, ranking_total

    def _semantic_label_ranking_boost(self, patch: Patch3D, obj: ObjectMap) -> float:
        """Optional candidate-pruning boost only; not part of AssociationScore."""
        if self.semantic_label_candidate_boost <= 0.0:
            return 0.0
        patch_label = str(patch.metadata.get("anchor_class_name", "")).strip()
        if not patch_label:
            return 0.0
        object_label = object_association_identity_semantic_label(obj).strip()
        if not object_label or object_label != patch_label:
            return 0.0
        return float(self.semantic_label_candidate_boost)

    def _select_geometry_candidate_ids(
        self,
        patch: Patch3D,
        objects: Dict[int, ObjectMap],
        candidate_ids: List[int],
        vote: VoxelVoteResult,
    ) -> tuple[set[int], dict[int, tuple[float, ...]]]:
        cheap_by_id: dict[int, tuple[float, ...]] = {}
        scored: list[tuple[tuple[int, float, float, int], int]] = []
        for obj_id in candidate_ids:
            obj = objects.get(obj_id)
            if obj is None or obj.state.value == "removed":
                continue
            components = self._cheap_score_components(patch, obj, vote)
            cheap_by_id[int(obj_id)] = components
            voxel_score, _centroid_score, _bbox_score, cheap_total = components[:4]
            if cheap_total < self.geometry_candidate_min_cheap_score and voxel_score <= 0.0:
                continue
            vote_owner_rank = 1 if self.geometry_vote_owner_priority and voxel_score > 0.0 else 0
            scored.append(((vote_owner_rank, cheap_total, voxel_score, -int(obj_id)), int(obj_id)))

        geometry_limit = self._candidate_limit(self.top_k_geometry, self.max_geometry_candidates)
        if not self.two_stage_enabled or geometry_limit <= 0:
            return set(cheap_by_id), cheap_by_id

        scored.sort(reverse=True)
        selected = [obj_id for _key, obj_id in scored[:geometry_limit]]
        return set(selected), cheap_by_id

    def _select_scored_candidate_ids(
        self,
        candidate_ids: List[int],
        cheap_by_id: dict[int, tuple[float, ...]],
        vote: VoxelVoteResult,
    ) -> List[int]:
        valid_ids = [int(obj_id) for obj_id in candidate_ids if int(obj_id) in cheap_by_id]
        final_limit = self._candidate_limit(self.top_k_final, self.max_scored_candidates)
        if not self.two_stage_enabled or final_limit <= 0 or len(valid_ids) <= final_limit:
            return valid_ids
        vote_owner_ids = {int(obj_id) for obj_id in vote.owner_votes}
        ranked = sorted(
            valid_ids,
            key=lambda obj_id: (
                1 if obj_id in vote_owner_ids else 0,
                cheap_by_id[obj_id][4] if len(cheap_by_id[obj_id]) > 4 else cheap_by_id[obj_id][3],
                cheap_by_id[obj_id][0],
                -obj_id,
            ),
            reverse=True,
        )
        return ranked[:final_limit]

    @staticmethod
    def _candidate_limit(primary_limit: int, compatibility_limit: int) -> int:
        limits = [int(limit) for limit in (primary_limit, compatibility_limit) if int(limit) > 0]
        return min(limits) if limits else 0

    @staticmethod
    def _resolve_worker_count(configured_workers: int, task_count: int) -> int:
        if task_count <= 0:
            return 1
        if configured_workers > 0:
            return max(1, min(int(configured_workers), int(task_count)))
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, int(cpu_count), int(task_count)))

    def _score_candidate_objects(
        self,
        patch: Patch3D,
        objects: Dict[int, ObjectMap],
        candidate_ids: List[int],
        vote: VoxelVoteResult,
        cheap_by_id: dict[int, tuple[float, ...]],
        geometry_candidate_ids: set[int],
    ) -> tuple[list[tuple[int, AssociationScore]], bool]:
        valid_ids = [
            int(obj_id)
            for obj_id in candidate_ids
            if obj_id in objects and objects[obj_id].state.value != "removed"
        ]
        worker_count = self._resolve_worker_count(self.score_parallel_workers, len(valid_ids))
        use_parallel = (
            self.score_parallel_enabled
            and len(valid_ids) >= self.score_parallel_min_candidates
            and worker_count > 1
        )

        def score_one(obj_id: int) -> tuple[int, AssociationScore]:
            return (
                obj_id,
                self._compute_score(
                    patch,
                    objects[obj_id],
                    vote,
                    cheap_components=cheap_by_id.get(int(obj_id)),
                    compute_geometry=int(obj_id) in geometry_candidate_ids,
                ),
            )

        if not use_parallel:
            return [score_one(obj_id) for obj_id in valid_ids], False

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(score_one, valid_ids)), True

    def _compute_score(
        self,
        patch: Patch3D,
        obj: ObjectMap,
        vote: VoxelVoteResult,
        *,
        cheap_components: tuple[float, ...] | None = None,
        compute_geometry: bool = True,
    ) -> AssociationScore:
        """Compute association score combining voxel voting + geometric fallback.

        Scoring is spatial-first. Semantics are NOT used here (Rule C).
        """
        if cheap_components is None:
            voxel_score, centroid_score, bbox_score, _cheap_total = self._cheap_score_components(patch, obj, vote)[:4]
        else:
            voxel_score, centroid_score, bbox_score, _cheap_total = cheap_components[:4]

        geometry_score = self._geometry_consistency(patch, obj) if compute_geometry else 0.0

        total = (
            self.w_voxel_vote * voxel_score
            + self.w_centroid * centroid_score
            + self.w_bbox * bbox_score
            + self.w_geometry * geometry_score
        )

        return AssociationScore(
            centroid_distance=centroid_score,
            bbox_overlap=bbox_score,
            geometry_overlap=geometry_score,
            voxel_vote_score=voxel_score,
            total_score=total,
            vote_result=VoxelVoteResult(
                touched_voxel_count=vote.touched_voxel_count,
                supported_voxel_count=vote.supported_voxel_count,
                owner_votes=dict(vote.owner_votes),
                normalized_vote_score=voxel_score,
                best_instance_id=vote.best_instance_id,
                geometry_consistency_score=geometry_score,
                final_association_score=total,
                new_instance_created=False,
            ),
        )

    def _semantic_conflict_reason(self, patch: Patch3D, obj: ObjectMap) -> str:
        """Legacy semantic-conflict helper retained for compatibility/debug history.

        The main association path now uses observation identity classification
        and no longer calls this helper.
        """
        if not self.semantic_conflict_gate_enabled:
            return ""
        patch_label = str(patch.metadata.get("anchor_class_name", "")).strip()
        if not patch_label:
            return ""
        patch_confidence = float(patch.metadata.get("anchor_confidence", 0.0))
        if patch_confidence < self.semantic_conflict_min_patch_confidence:
            return ""

        object_label = object_association_identity_semantic_label(obj).strip()
        if not object_label or object_label == patch_label:
            return ""
        object_confidence = self._object_label_confidence(obj, object_label)
        if object_confidence < self.semantic_conflict_min_object_confidence:
            return ""
        if not self._patch_is_subregion_of_object(patch, obj):
            return ""
        return "semantic_conflict_subregion"

    def _object_label_confidence(self, obj: ObjectMap, label: str) -> float:
        anchor_state = obj.debug.get("anchor_semantics", {})
        if isinstance(anchor_state, dict):
            max_conf = anchor_state.get("label_max_confidence", {}) or {}
            if label in max_conf:
                return float(max_conf.get(label, 0.0))
            if str(anchor_state.get("canonical_label", "")) == label:
                return min(1.0, float(anchor_state.get("canonical_score", 0.0)))
        hypotheses = getattr(obj.semantic_memory, "label_hypotheses", [])
        for candidate_label, confidence in hypotheses:
            if str(candidate_label) == label:
                return float(confidence)
        return 0.0

    def _patch_is_subregion_of_object(self, patch: Patch3D, obj: ObjectMap) -> bool:
        if len(patch.points) == 0 or len(obj.local_pcd) == 0:
            return False
        point_ratio = float(len(patch.points) / max(len(obj.local_pcd), 1))
        patch_volume = self._bbox_volume(patch.bbox_min, patch.bbox_max)
        object_volume = self._bbox_volume(obj.bbox_min, obj.bbox_max)
        volume_ratio = float(patch_volume / max(object_volume, 1e-9))
        centroid_inside = self._point_inside_bbox(patch.centroid, obj.bbox_min, obj.bbox_max, margin=0.05)
        return bool(
            centroid_inside
            and (
                point_ratio <= self.semantic_conflict_subregion_point_ratio
                or volume_ratio <= self.semantic_conflict_subregion_volume_ratio
            )
        )

    @staticmethod
    def _bbox_volume(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
        extent = np.maximum(np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32), 1e-3)
        return float(np.prod(extent))

    @staticmethod
    def _point_inside_bbox(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float) -> bool:
        point = np.asarray(point, dtype=np.float32)
        bbox_min = np.asarray(bbox_min, dtype=np.float32) - float(margin)
        bbox_max = np.asarray(bbox_max, dtype=np.float32) + float(margin)
        return bool(np.all(point >= bbox_min) and np.all(point <= bbox_max))

    def _object_geometry_points(self, obj: ObjectMap) -> np.ndarray:
        """Return bounded association geometry when available, else local pool geometry."""
        association_pcd = getattr(obj, "association_pcd", None)
        if association_pcd is not None and len(association_pcd) > 0:
            return np.asarray(association_pcd, dtype=np.float32)
        return np.asarray(obj.local_pcd, dtype=np.float32)

    def _geometry_consistency(self, patch: Patch3D, obj: ObjectMap) -> float:
        """Compute local geometry consistency between patch and object geometry memory.

        Uses mean nearest-neighbor distance as a proxy.
        """
        score_start = time.perf_counter()
        try:
            return self._geometry_consistency_impl(patch, obj)
        finally:
            elapsed = float(time.perf_counter() - score_start)
            self.association_geometry_score_sec += elapsed

    def _geometry_consistency_impl(self, patch: Patch3D, obj: ObjectMap) -> float:
        object_points = self._object_geometry_points(obj)
        if len(object_points) == 0 or len(patch.points) == 0:
            return 0.0

        # Deterministically subsample for stable association scores.
        p_pts = patch.points
        o_pts = object_points
        if len(p_pts) > self.geometry_nn_patch_sample:
            idx = np.linspace(0, len(p_pts) - 1, num=self.geometry_nn_patch_sample, dtype=np.int64)
            p_pts = p_pts[idx]
        if len(o_pts) > self.geometry_nn_object_sample:
            idx = np.linspace(0, len(o_pts) - 1, num=self.geometry_nn_object_sample, dtype=np.int64)
            o_pts = o_pts[idx]

        nn_backend = self._select_geometry_nn_backend(len(p_pts), len(o_pts))
        use_kdtree_cache = (
            nn_backend == "kdtree"
            and self.kdtree_cache_enabled
            and len(object_points) >= self.kdtree_cache_min_object_points
        )
        object_signature = self._object_geometry_signature(obj, object_points) if use_kdtree_cache else None
        min_dists = self._patch_to_object_nn_distances(
            p_pts,
            o_pts,
            backend=nn_backend,
            object_id=int(obj.object_id) if use_kdtree_cache else None,
            object_signature=object_signature,
        )
        mean_nn_dist = float(min_dists.mean())

        # Convert to score: close = high score
        return float(max(0.0, 1.0 - mean_nn_dist / 0.5))

    def _object_geometry_signature(self, obj: ObjectMap, object_points: np.ndarray) -> tuple[Any, ...]:
        points = np.asarray(object_points)
        if len(points) == 0:
            return ("empty",)

        # Content-addressable hash: stable across frames when geometry is unchanged
        geometry_debug = obj.debug.get("association_geometry", {}) if isinstance(obj.debug, dict) else {}
        debug_revision = geometry_debug.get("revision", None)
        if debug_revision is None:
            debug_revision = geometry_debug.get("association_geometry_revision", None)

        content_hash = hash(points.astype(np.float32).tobytes())
        return (
            int(getattr(obj, "object_id", -1)),
            debug_revision,
            content_hash,
        )

    def _select_geometry_nn_backend(self, patch_point_count: int, object_point_count: int) -> str:
        if self.geometry_nn_backend == "vectorized":
            return "vectorized"
        if self.geometry_nn_backend == "kdtree":
            return "kdtree"
        max_vectorized_points = max(0, int(self.geometry_vectorized_max_points))
        if (
            int(patch_point_count) <= max_vectorized_points
            and int(object_point_count) <= max_vectorized_points
        ):
            return "vectorized"
        return "kdtree"

    def _patch_to_object_nn_distances(
        self,
        patch_points: np.ndarray,
        object_points: np.ndarray,
        *,
        backend: str = "auto",
        object_id: int | None = None,
        object_signature: tuple[Any, ...] | None = None,
    ) -> np.ndarray:
        if backend == "vectorized" or cKDTree is None:
            return self._patch_to_object_nn_distances_vectorized(patch_points, object_points)

        if backend == "kdtree":
            cache_key = None
            if object_id is not None and object_signature is not None:
                cache_key = (int(object_id), object_signature)
                with self._association_kdtree_cache_lock:
                    tree = self._association_kdtree_cache.get(cache_key)
                    if tree is not None:
                        self.association_kdtree_cache_hit_count += 1
                if tree is not None:
                    try:
                        distances, _indices = tree.query(patch_points, k=1)
                        return np.asarray(distances, dtype=np.float32)
                    except Exception:
                        with self._association_kdtree_cache_lock:
                            self._association_kdtree_cache.pop(cache_key, None)

            try:
                tree = cKDTree(object_points)
                if cache_key is not None:
                    with self._association_kdtree_cache_lock:
                        self._association_kdtree_cache[cache_key] = tree
                        self.association_kdtree_cache_miss_count += 1
                distances, _indices = tree.query(patch_points, k=1)
                return np.asarray(distances, dtype=np.float32)
            except Exception:
                pass

        return self._patch_to_object_nn_distances_vectorized(patch_points, object_points)

    @staticmethod
    def _patch_to_object_nn_distances_vectorized(
        patch_points: np.ndarray,
        object_points: np.ndarray,
    ) -> np.ndarray:
        diffs = patch_points[:, None, :] - object_points[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return dists.min(axis=1)
