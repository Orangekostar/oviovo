from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import ObjectMap, Patch3D, SemanticMemory, SystemState, VoxelVoteResult
from src.modules.association import AssociationModule


def _patch(points: np.ndarray, patch_id: int = 1, label: str = "") -> Patch3D:
    metadata = {"anchor_class_name": label} if label else {}
    return Patch3D(
        patch_id=patch_id,
        points=points.astype(np.float32),
        centroid=points.mean(axis=0).astype(np.float32),
        bbox_min=points.min(axis=0).astype(np.float32),
        bbox_max=points.max(axis=0).astype(np.float32),
        metadata=metadata,
    )


def _object(object_id: int, points: np.ndarray, label: str = "") -> ObjectMap:
    obj = ObjectMap(
        object_id=object_id,
        local_pcd=points.astype(np.float32),
        association_pcd=points.astype(np.float32),
        centroid=points.mean(axis=0).astype(np.float32),
        bbox_min=points.min(axis=0).astype(np.float32),
        bbox_max=points.max(axis=0).astype(np.float32),
    )
    if label:
        obj.semantic_memory = SemanticMemory(label_hypotheses=[(label, 0.99)])
    return obj


def _line_points(count: int, offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num=count, dtype=np.float32)
    return np.stack(
        [x + np.float32(offset), np.zeros_like(x), np.ones_like(x)],
        axis=1,
    )


def test_two_stage_top_k_limits_geometry_calls_and_keeps_metrics():
    module = AssociationModule(
        {
            "match_threshold": 0.1,
            "voxel_vote_weight": 0.0,
            "centroid_distance_weight": 0.7,
            "bbox_overlap_weight": 0.1,
            "geometry_overlap_weight": 0.2,
            "two_stage_enabled": True,
            "top_k_final": 4,
            "top_k_geometry": 2,
        }
    )
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=10)
    objects = {
        obj_id: _object(obj_id, patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32))
        for obj_id, offset in [(1, 0.01), (2, 0.20), (3, 0.40), (4, 0.60), (5, 0.80)]
    }
    called_object_ids: list[int] = []

    def fake_geometry(_patch, obj):
        called_object_ids.append(int(obj.object_id))
        return 1.0

    module._geometry_consistency = fake_geometry

    result = module.process([patch], objects, SystemState().tsdf_volume)
    patch_debug = result.debug["per_patch"][10]
    summary = result.debug["summary"]

    assert called_object_ids == [1, 2]
    assert patch_debug["candidate_count_after_scored_cap"] == 4
    assert patch_debug["geometry_scored_count"] == 2
    assert summary["candidate_score_count"] == 4
    assert summary["geometry_score_count"] == 2
    assert summary["score_parallel_used_count"] == 0
    assert summary["two_stage_enabled"] is True
    assert summary["top_k_final"] == 4
    assert summary["top_k_geometry"] == 2


def test_two_stage_disabled_keeps_uncapped_legacy_scoring():
    module = AssociationModule({"two_stage_enabled": False, "top_k_final": 1, "top_k_geometry": 1})
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=11)
    objects = {
        obj_id: _object(obj_id, patch_points + np.array([float(obj_id), 0.0, 0.0], dtype=np.float32))
        for obj_id in range(3)
    }
    called_object_ids: list[int] = []

    def fake_geometry(_patch, obj):
        called_object_ids.append(int(obj.object_id))
        return 0.0

    module._geometry_consistency = fake_geometry

    result = module.process([patch], objects, SystemState().tsdf_volume)

    assert called_object_ids == [0, 1, 2]
    assert len(result.scores) == 3
    assert result.debug["summary"]["candidate_score_count"] == 3
    assert result.debug["summary"]["geometry_score_count"] == 3


def test_indexed_retrieval_disabled_preserves_candidate_set():
    module = AssociationModule({"two_stage_enabled": False, "indexed_retrieval_enabled": False, "max_spatial_candidates": 1})
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=15)
    objects = {
        obj_id: _object(obj_id, patch_points + np.array([float(obj_id), 0.0, 0.0], dtype=np.float32))
        for obj_id in range(1, 5)
    }

    result = module.process([patch], objects, SystemState().tsdf_volume)
    debug = result.debug["per_patch"][15]
    summary = result.debug["summary"]

    assert debug["candidate_count_before_indexed_retrieval"] == 4
    assert debug["candidate_count_after_indexed_retrieval"] == 4
    assert summary["indexed_candidate_query_count"] == 0
    assert summary["indexed_pruned_candidate_count"] == 0
    assert len(result.scores) == 4


def test_centroid_indexed_retrieval_prunes_candidates_and_preserves_owner_votes():
    module = AssociationModule(
        {
            "two_stage_enabled": False,
            "indexed_retrieval_enabled": True,
            "centroid_kdtree_enabled": True,
            "max_spatial_candidates": 2,
        }
    )
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=16)
    objects = {
        obj_id: _object(obj_id, patch_points + np.array([offset, 0.0, 0.0], dtype=np.float32))
        for obj_id, offset in [(1, 0.01), (2, 0.10), (3, 1.0), (4, 2.0), (5, 10.0)]
    }

    def fake_vote(_volume, _patch):
        return VoxelVoteResult(touched_voxel_count=1, supported_voxel_count=1, owner_votes={5: 1.0})

    module.tsdf_module.vote_patch_to_instance = fake_vote

    result = module.process([patch], objects, SystemState().tsdf_volume)
    debug = result.debug["per_patch"][16]
    summary = result.debug["summary"]

    assert debug["candidate_count_before_indexed_retrieval"] == 5
    assert debug["candidate_count_after_indexed_retrieval"] == 3
    assert set(debug["candidate_object_ids"]) == {1, 2, 5}
    assert debug["indexed_retrieval"]["query_used"] is True
    assert debug["indexed_retrieval"]["selected_owner_vote_candidate_count"] == 1
    assert summary["indexed_candidate_query_count"] == 1
    assert summary["indexed_pruned_candidate_count"] == 2
    assert summary["centroid_index_build_count"] == 1
    assert summary["centroid_index_candidate_count"] == 5


def test_geometry_consistency_falls_back_when_ckdtree_unavailable(monkeypatch):
    import src.modules.association as association_module

    monkeypatch.setattr(association_module, "cKDTree", None)
    module = AssociationModule({"geometry_nn_patch_sample": 8, "geometry_nn_object_sample": 8})
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)

    assert module._geometry_consistency(_patch(points), _object(1, points.copy())) == pytest.approx(1.0)


def test_geometry_consistency_falls_back_when_ckdtree_query_raises(monkeypatch):
    import src.modules.association as association_module

    class RaisingTree:
        def __init__(self, _points):
            pass

        def query(self, _points, k=1):
            raise RuntimeError("forced fallback")

    monkeypatch.setattr(association_module, "cKDTree", RaisingTree)
    module = AssociationModule(
        {
            "geometry_nn_backend": "kdtree",
            "geometry_nn_patch_sample": 8,
            "geometry_nn_object_sample": 8,
        }
    )
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)

    assert module._geometry_consistency(_patch(points), _object(1, points.copy())) == pytest.approx(1.0)


def test_geometry_consistency_uses_vectorized_backend_for_small_samples_by_default(monkeypatch):
    import src.modules.association as association_module

    class CountingTree:
        build_count = 0

        def __init__(self, points):
            CountingTree.build_count += 1
            self.points = np.asarray(points, dtype=np.float32).copy()

        def query(self, patch_points, k=1):
            diffs = np.asarray(patch_points, dtype=np.float32)[:, None, :] - self.points[None, :, :]
            return np.linalg.norm(diffs, axis=2).min(axis=1), None

    monkeypatch.setattr(association_module, "cKDTree", CountingTree)
    module = AssociationModule({"geometry_nn_patch_sample": 8, "geometry_nn_object_sample": 8})
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)
    obj = _object(1, points.copy())
    patch = _patch(points.copy())

    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)
    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)

    assert CountingTree.build_count == 0
    assert module._association_kdtree_cache == {}
    assert module.association_kdtree_cache_miss_count == 0
    assert module.association_kdtree_cache_hit_count == 0


def test_geometry_consistency_kdtree_backend_does_not_cache_by_default(monkeypatch):
    import src.modules.association as association_module

    class CountingTree:
        build_count = 0

        def __init__(self, points):
            CountingTree.build_count += 1
            self.points = np.asarray(points, dtype=np.float32).copy()

        def query(self, patch_points, k=1):
            diffs = np.asarray(patch_points, dtype=np.float32)[:, None, :] - self.points[None, :, :]
            return np.linalg.norm(diffs, axis=2).min(axis=1), None

    monkeypatch.setattr(association_module, "cKDTree", CountingTree)
    module = AssociationModule(
        {
            "geometry_nn_backend": "kdtree",
            "geometry_nn_patch_sample": 8,
            "geometry_nn_object_sample": 8,
        }
    )
    points = _line_points(300)
    obj = _object(1, points.copy())
    patch = _patch(points.copy())

    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)
    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)

    assert CountingTree.build_count == 2
    assert module._association_kdtree_cache == {}
    assert module.association_kdtree_cache_miss_count == 0
    assert module.association_kdtree_cache_hit_count == 0


def test_geometry_consistency_reuses_cached_kdtree_when_enabled(monkeypatch):
    import src.modules.association as association_module

    class CountingTree:
        build_count = 0

        def __init__(self, points):
            CountingTree.build_count += 1
            self.points = np.asarray(points, dtype=np.float32).copy()

        def query(self, patch_points, k=1):
            diffs = np.asarray(patch_points, dtype=np.float32)[:, None, :] - self.points[None, :, :]
            return np.linalg.norm(diffs, axis=2).min(axis=1), None

    monkeypatch.setattr(association_module, "cKDTree", CountingTree)
    module = AssociationModule(
        {
            "geometry_nn_backend": "kdtree",
            "geometry_nn_patch_sample": 8,
            "geometry_nn_object_sample": 8,
            "kdtree_cache_enabled": True,
        }
    )
    points = _line_points(300)
    obj = _object(1, points.copy())
    patch = _patch(points.copy())

    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)
    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)

    assert CountingTree.build_count == 1
    assert module.association_kdtree_cache_miss_count == 1
    assert module.association_kdtree_cache_hit_count == 1


def test_geometry_consistency_refreshes_kdtree_when_object_geometry_changes(monkeypatch):
    import src.modules.association as association_module

    class SnapshotTree:
        build_count = 0

        def __init__(self, points):
            SnapshotTree.build_count += 1
            self.points = np.asarray(points, dtype=np.float32).copy()

        def query(self, patch_points, k=1):
            diffs = np.asarray(patch_points, dtype=np.float32)[:, None, :] - self.points[None, :, :]
            return np.linalg.norm(diffs, axis=2).min(axis=1), None

    monkeypatch.setattr(association_module, "cKDTree", SnapshotTree)
    module = AssociationModule(
        {
            "geometry_nn_patch_sample": 8,
            "geometry_nn_object_sample": 8,
            "geometry_nn_backend": "kdtree",
            "kdtree_cache_enabled": True,
            "kdtree_cache_min_object_points": 1,
        }
    )
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)
    obj = _object(1, points.copy())
    patch = _patch(points.copy())

    assert module._geometry_consistency(patch, obj) == pytest.approx(1.0)
    obj.association_pcd += np.array([1.0, 0.0, 0.0], dtype=np.float32)
    obj.local_pcd = obj.association_pcd.copy()
    changed_score = module._geometry_consistency(patch, obj)

    assert changed_score < 1.0
    assert SnapshotTree.build_count == 2
    assert module.association_kdtree_cache_miss_count == 2
    assert module.association_kdtree_cache_hit_count == 0


def test_association_summary_includes_kdtree_cache_and_geometry_timing(monkeypatch):
    import src.modules.association as association_module

    class CountingTree:
        def __init__(self, points):
            self.points = np.asarray(points, dtype=np.float32).copy()

        def query(self, patch_points, k=1):
            diffs = np.asarray(patch_points, dtype=np.float32)[:, None, :] - self.points[None, :, :]
            return np.linalg.norm(diffs, axis=2).min(axis=1), None

    monkeypatch.setattr(association_module, "cKDTree", CountingTree)
    module = AssociationModule(
        {
            "two_stage_enabled": False,
            "match_threshold": 0.1,
            "voxel_vote_weight": 0.0,
            "centroid_distance_weight": 0.0,
            "bbox_overlap_weight": 0.0,
            "geometry_overlap_weight": 1.0,
            "geometry_nn_backend": "kdtree",
            "kdtree_cache_enabled": True,
            "kdtree_cache_min_object_points": 1,
        }
    )
    points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)
    objects = {1: _object(1, points.copy())}

    result = module.process(
        [_patch(points.copy(), patch_id=21), _patch(points.copy(), patch_id=22)],
        objects,
        SystemState().tsdf_volume,
    )
    summary = result.debug["summary"]

    assert summary["association_kdtree_cache_miss_count"] == 1
    assert summary["association_kdtree_cache_hit_count"] == 1
    assert summary["association_geometry_score_sec"] >= 0.0


def test_legacy_max_candidates_remain_hard_caps_with_top_k_overrides():
    module = AssociationModule(
        {
            "two_stage_enabled": True,
            "top_k_final": 5,
            "max_scored_candidates": 3,
            "top_k_geometry": 4,
            "max_geometry_candidates": 2,
        }
    )
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=14)
    objects = {
        obj_id: _object(obj_id, patch_points + np.array([obj_id * 0.05, 0.0, 0.0], dtype=np.float32))
        for obj_id in range(1, 7)
    }

    result = module.process([patch], objects, SystemState().tsdf_volume)
    debug = result.debug["per_patch"][14]

    assert debug["candidate_count_after_scored_cap"] == 3
    assert debug["geometry_scored_count"] == 2


def test_small_deterministic_fixture_matches_with_and_without_two_stage():
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.05, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=12)
    objects = {
        1: _object(1, patch_points + np.array([0.02, 0.0, 0.0], dtype=np.float32)),
        2: _object(2, patch_points + np.array([0.80, 0.0, 0.0], dtype=np.float32)),
        3: _object(3, patch_points + np.array([1.20, 0.0, 0.0], dtype=np.float32)),
    }
    config = {
        "match_threshold": 0.1,
        "voxel_vote_weight": 0.0,
        "centroid_distance_weight": 0.7,
        "bbox_overlap_weight": 0.1,
        "geometry_overlap_weight": 0.2,
    }

    baseline = AssociationModule({**config, "two_stage_enabled": False})
    accelerated = AssociationModule({**config, "two_stage_enabled": True, "top_k_final": 3, "top_k_geometry": 3})

    baseline_result = baseline.process([patch], objects, SystemState().tsdf_volume)
    accelerated_result = accelerated.process([patch], objects, SystemState().tsdf_volume)

    assert accelerated_result.matched[0][0:2] == baseline_result.matched[0][0:2]
    for key in sorted(baseline_result.scores):
        assert accelerated_result.scores[key].total_score == pytest.approx(baseline_result.scores[key].total_score)


def test_semantic_label_boost_only_changes_candidate_pruning_not_score():
    patch_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(patch_points, patch_id=13, label="chair")
    near_unlabeled = patch_points + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    farther_labeled = patch_points + np.array([0.20, 0.0, 0.0], dtype=np.float32)
    objects = {1: _object(1, near_unlabeled), 2: _object(2, farther_labeled, label="chair")}
    config = {
        "two_stage_enabled": True,
        "top_k_final": 1,
        "top_k_geometry": 1,
        "semantic_label_candidate_boost": 2.0,
    }

    boosted = AssociationModule(config).process([patch], objects, SystemState().tsdf_volume)
    unboosted = AssociationModule({**config, "top_k_final": 2, "semantic_label_candidate_boost": 0.0}).process(
        [patch], objects, SystemState().tsdf_volume
    )

    assert boosted.debug["per_patch"][13]["candidate_object_ids"] == [2]
    assert boosted.scores[(13, 2)].total_score == pytest.approx(unboosted.scores[(13, 2)].total_score)
