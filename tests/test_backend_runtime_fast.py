from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    AssociationResult,
    CameraIntrinsics,
    DenseSurfaceEntry,
    ObjectMap,
    ObservationRecord,
    Patch3D,
    RefinedProposal2D,
    SemanticMemory,
    SystemState,
    TSDFInstanceVolume,
    VoxelOwnerSupport,
    WholeEvidenceScores,
)
from src.modules.active_set import ActiveSetModule
from src.modules.dense_surface import DenseSurfaceModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.patch_lifting import PatchLiftingModule
from src.modules.tsdf_instance_map import (
    TSDFInstanceMapModule,
    filter_patch_voxel_view,
    get_patch_voxel_view,
    get_voxel_owner_id,
    patch_cached_voxels,
)


def _patch(points: np.ndarray, *, patch_id: int = 1, frame_id: int = 0, metadata: dict | None = None) -> Patch3D:
    points = np.asarray(points, dtype=np.float32)
    return Patch3D(
        patch_id=patch_id,
        points=points,
        centroid=points.mean(axis=0).astype(np.float32),
        bbox_min=points.min(axis=0).astype(np.float32),
        bbox_max=points.max(axis=0).astype(np.float32),
        source_frame_id=frame_id,
        metadata=dict(metadata or {}),
    )


def _lift_single_test_patch(config: dict | None = None) -> Patch3D:
    proposal = RefinedProposal2D(
        proposal_id=7,
        mask=np.array([[True, True], [False, True]], dtype=bool),
        bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
        area=3,
    )
    depth = np.ones((2, 2), dtype=np.float32)
    module = PatchLiftingModule({"min_points": 1, "voxel_size": 0.5, **dict(config or {})})

    patches = module.process(
        [proposal],
        depth,
        np.eye(4, dtype=np.float32),
        CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2),
    )

    assert len(patches) == 1
    return patches[0]


def test_patch_lifting_cache_disabled_skips_voxel_metadata() -> None:
    patch = _lift_single_test_patch({"voxel_cache_enabled": False})

    assert patch.points.shape == (3, 3)
    assert "voxel_size" not in patch.metadata
    assert "voxel_indices" not in patch.metadata
    assert "unique_voxel_indices" not in patch.metadata
    assert "representative_point_indices" not in patch.metadata
    assert "voxel_cache_point_count" not in patch.metadata


def test_patch_lifting_cache_enabled_writes_voxel_cache_without_changing_points() -> None:
    uncached = _lift_single_test_patch({"voxel_cache_enabled": False})
    patch = _lift_single_test_patch({"voxel_cache_enabled": True})

    assert patch.points.shape == (3, 3)
    np.testing.assert_array_equal(patch.points, uncached.points)
    assert patch.metadata["voxel_size"] == 0.5
    assert patch.metadata["voxel_indices"].shape == (3, 3)
    assert patch.metadata["unique_voxel_indices"].shape[1] == 3
    reps = patch.metadata["representative_point_indices"]
    assert reps.ndim == 1
    assert np.all((reps >= 0) & (reps < len(patch.points)))


def test_tsdf_vote_integrate_reuse_cache_and_match_recomputed_result() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    base = _patch(points)
    voxels = np.floor(points / 1.0).astype(np.int64)
    unique, reps = np.unique(voxels, axis=0, return_index=True)
    cached = _patch(
        points,
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": unique,
            "representative_point_indices": reps.astype(np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )
    assert patch_cached_voxels(cached, 1.0)[3] is True
    assert patch_cached_voxels(cached, 0.5)[3] is False

    module = TSDFInstanceMapModule({})
    cached_volume = TSDFInstanceVolume(voxel_size=1.0)
    base_volume = TSDFInstanceVolume(voxel_size=1.0)
    module.integrate_patch(cached_volume, cached, 4)
    module.integrate_patch(base_volume, base, 4)

    assert cached_volume.owner_support.keys() == base_volume.owner_support.keys()
    assert module.vote_patch_to_instance(cached_volume, cached) == module.vote_patch_to_instance(base_volume, base)
    module.remove_patch_support(cached_volume, cached, 4)
    module.remove_patch_support(base_volume, base, 4)
    assert cached_volume.owner_support == base_volume.owner_support


def test_tsdf_cached_uncached_and_explicit_lazy_view_results_match() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1], [1.2, 0.1, 0.1]],
        dtype=np.float32,
    )
    base = _patch(points)
    voxels = np.floor(points / 1.0).astype(np.int64)
    unique, reps = np.unique(voxels, axis=0, return_index=True)
    cached = _patch(
        points,
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": unique,
            "representative_point_indices": reps.astype(np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )
    module = TSDFInstanceMapModule({})
    cached_volume = TSDFInstanceVolume(voxel_size=1.0)
    uncached_volume = TSDFInstanceVolume(voxel_size=1.0)
    lazy_volume = TSDFInstanceVolume(voxel_size=1.0)

    cached_view = get_patch_voxel_view(cached, 1.0)
    lazy_view = get_patch_voxel_view(base, 1.0)
    assert cached_view.cache_used is True
    assert lazy_view.cache_used is False

    module.integrate_patch(cached_volume, cached, 4, voxel_view=cached_view)
    module.integrate_patch(uncached_volume, base, 4)
    module.integrate_patch(lazy_volume, base, 4, voxel_view=lazy_view)

    assert cached_volume.owner_support.keys() == uncached_volume.owner_support.keys()
    assert lazy_volume.owner_support.keys() == uncached_volume.owner_support.keys()
    assert module.vote_patch_to_instance(cached_volume, cached, cached_view) == module.vote_patch_to_instance(
        uncached_volume,
        base,
    )
    assert module.vote_patch_to_instance(lazy_volume, base, lazy_view) == module.vote_patch_to_instance(
        uncached_volume,
        base,
    )

    keep_mask = np.array([True, False, True, False], dtype=bool)
    filtered_view = filter_patch_voxel_view(lazy_view, keep_mask)
    filtered_patch = _patch(points[keep_mask])
    assert filtered_view is not None
    filtered_volume = TSDFInstanceVolume(voxel_size=1.0)
    recomputed_filtered_volume = TSDFInstanceVolume(voxel_size=1.0)
    module.integrate_patch(filtered_volume, filtered_patch, 8, voxel_view=filtered_view)
    module.integrate_patch(recomputed_filtered_volume, filtered_patch, 8)
    assert filtered_volume.owner_support.keys() == recomputed_filtered_volume.owner_support.keys()


def test_tsdf_instance_support_index_matches_owner_support_after_integrate_and_remove() -> None:
    module = TSDFInstanceMapModule({})
    volume = TSDFInstanceVolume(voxel_size=1.0)
    patch_a = _patch(np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32))
    patch_b = _patch(np.array([[1.1, 0.1, 0.1], [2.1, 0.1, 0.1]], dtype=np.float32))

    module.integrate_patch(volume, patch_a, 4)
    module.integrate_patch(volume, patch_b, 5)
    module.remove_patch_support(volume, patch_a, 4)

    assert volume.voxel_owner_id == {
        voxel_key: support.owner_id
        for voxel_key, support in volume.owner_support.items()
        if support.owner_id >= 0
    }

    def scan(instance_id: int) -> dict[str, float]:
        owner_values = []
        competing_values = []
        for support in volume.owner_support.values():
            if instance_id not in support.support:
                continue
            if support.owner_id == instance_id:
                owner_values.append(float(support.support[instance_id]))
            else:
                competing_values.append(float(support.support[instance_id]))
        return {
            "owned_voxel_count": float(len(owner_values)),
            "support_mass": float(sum(owner_values)),
            "competing_support_mass": float(sum(competing_values)),
        }

    for instance_id in (4, 5):
        indexed = module.summarize_instance_support(volume, instance_id)
        scanned = scan(instance_id)
        assert indexed["owned_voxel_count"] == scanned["owned_voxel_count"]
        assert indexed["support_mass"] == pytest.approx(scanned["support_mass"])
        assert indexed["competing_support_mass"] == pytest.approx(scanned["competing_support_mass"])


def test_tsdf_voxel_owner_helper_lazily_rebuilds_manual_owner_support() -> None:
    volume = TSDFInstanceVolume(voxel_size=1.0)
    volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0, 4: 0.5})
    volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({4: 2.0})
    volume.support_index_valid = True
    revision_before = volume.owner_index_revision

    assert get_voxel_owner_id(volume, (1, 0, 0)) == 9
    assert get_voxel_owner_id(volume, (2, 0, 0)) == 4
    assert get_voxel_owner_id(volume, (3, 0, 0)) == -1
    assert volume.voxel_owner_id == {(1, 0, 0): 9, (2, 0, 0): 4}
    assert volume.owner_index_revision == revision_before + 1


def test_tsdf_owner_index_rebuilds_when_manual_owner_index_is_partial() -> None:
    volume = TSDFInstanceVolume(voxel_size=1.0)
    volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})
    volume.owner_support[(2, 0, 0)] = VoxelOwnerSupport({4: 2.0})
    volume.voxel_owner_id = {(1, 0, 0): 9}
    volume.support_index_valid = True
    module = TSDFInstanceMapModule({})

    assert get_voxel_owner_id(volume, (2, 0, 0)) == 4
    assert volume.voxel_owner_id == {(1, 0, 0): 9, (2, 0, 0): 4}
    assert module.get_instance_voxel_count(volume, 4) == 1


def test_tsdf_owner_index_rebuilds_when_manual_owner_index_has_stale_extra_key() -> None:
    volume = TSDFInstanceVolume(voxel_size=1.0)
    volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})
    volume.voxel_owner_id = {(1, 0, 0): 9, (9, 0, 0): 99}
    volume.support_index_valid = True
    module = TSDFInstanceMapModule({})

    assert get_voxel_owner_id(volume, (9, 0, 0)) == -1
    assert volume.voxel_owner_id == {(1, 0, 0): 9}

    stale_patch = _patch(np.array([[9.1, 0.1, 0.1]], dtype=np.float32))
    vote = module.vote_patch_to_instance(volume, stale_patch)
    assert vote.owner_votes == {}
    assert vote.best_instance_id == -1


def test_tsdf_owner_index_revision_tracks_integrate_and_remove_owner_changes() -> None:
    module = TSDFInstanceMapModule({})
    volume = TSDFInstanceVolume(voxel_size=1.0)
    patch = _patch(np.array([[0.1, 0.1, 0.1]], dtype=np.float32))

    assert volume.owner_index_revision == 0
    module.integrate_patch(volume, patch, 4)
    assert volume.voxel_owner_id[(0, 0, 0)] == 4
    assert volume.owner_index_revision == 1

    module.integrate_patch(volume, patch, 4)
    assert volume.voxel_owner_id[(0, 0, 0)] == 4
    assert volume.owner_index_revision == 1

    module.integrate_patch(volume, patch, 5)
    assert volume.voxel_owner_id[(0, 0, 0)] == 4
    assert volume.owner_index_revision == 1

    module.integrate_patch(volume, patch, 5)
    assert volume.voxel_owner_id[(0, 0, 0)] == 5
    assert volume.owner_index_revision == 2

    module.remove_patch_support(volume, patch, 5)
    assert volume.voxel_owner_id[(0, 0, 0)] == 4
    assert volume.owner_index_revision == 3

    module.remove_patch_support(volume, patch, 4)
    assert volume.voxel_owner_id[(0, 0, 0)] == 5
    assert volume.owner_index_revision == 4

    module.remove_patch_support(volume, patch, 5)
    assert volume.voxel_owner_id[(0, 0, 0)] == 4
    assert volume.owner_index_revision == 5

    module.remove_patch_support(volume, patch, 4)
    assert (0, 0, 0) not in volume.voxel_owner_id
    assert volume.owner_index_revision == 6


def test_patch_cached_voxels_rejects_stale_point_fingerprint() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    voxels = np.floor(points / 1.0).astype(np.int64)
    unique, reps = np.unique(voxels, axis=0, return_index=True)
    patch = _patch(
        points.copy(),
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": unique,
            "representative_point_indices": reps.astype(np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )

    patch.points[2] = np.array([9.1, 0.1, 0.1], dtype=np.float32)
    _voxels, unique_voxels, _reps, cache_used = patch_cached_voxels(patch, 1.0)

    assert cache_used is False
    assert {tuple(row.tolist()) for row in unique_voxels} == {(0, 0, 0), (9, 0, 0)}


def test_patch_cached_voxels_rejects_stale_voxels_when_bbox_is_unchanged() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [1.1, 0.1, 0.1], [1.9, 0.1, 0.1]],
        dtype=np.float32,
    )
    voxels = np.floor(points / 1.0).astype(np.int64)
    unique, reps = np.unique(voxels, axis=0, return_index=True)
    patch = _patch(
        points.copy(),
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": unique,
            "representative_point_indices": reps.astype(np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )

    patch.points[1] = np.array([1.05, 0.1, 0.1], dtype=np.float32)
    patch.points[2] = np.array([0.95, 0.1, 0.1], dtype=np.float32)
    _voxels, unique_voxels, _reps, cache_used = patch_cached_voxels(patch, 1.0)

    assert cache_used is False
    assert {tuple(row.tolist()) for row in unique_voxels} == {(0, 0, 0), (1, 0, 0)}


def test_patch_cached_voxels_falls_back_when_unique_metadata_is_malformed() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    voxels = np.floor(points / 1.0).astype(np.int64)
    patch = _patch(
        points,
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": np.array([[0, 0, 0]], dtype=np.int64),
            "representative_point_indices": np.array([0], dtype=np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )

    view = get_patch_voxel_view(patch, 1.0, need_inverse=True)

    assert view.cache_used is False
    assert {tuple(row.tolist()) for row in view.unique_voxels} == {(0, 0, 0), (1, 0, 0)}
    assert view.inverse is not None


def test_surface_gate_representative_mode_uses_cache_and_filtered_patch_drops_stale_cache() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    voxels = np.floor(points / 1.0).astype(np.int64)
    unique, reps = np.unique(voxels, axis=0, return_index=True)
    patch = _patch(
        points,
        metadata={
            "voxel_indices": voxels,
            "unique_voxel_indices": unique,
            "representative_point_indices": reps.astype(np.int64),
            "voxel_size": 1.0,
            "voxel_cache_point_count": int(len(points)),
            "voxel_cache_bbox_min": points.min(axis=0).astype(np.float32),
            "voxel_cache_bbox_max": points.max(axis=0).astype(np.float32),
        },
    )
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert structural is None
    assert filtered is not None
    assert debug["cached_voxel_indices_used"] is True
    assert debug["decision_voxel_count"] == 2
    assert len(filtered.points) == 2
    assert "voxel_indices" not in filtered.metadata
    assert "unique_voxel_indices" not in filtered.metadata
    assert "representative_point_indices" not in filtered.metadata
    assert "voxel_size" not in filtered.metadata
    assert "voxel_cache_point_count" not in filtered.metadata
    assert "voxel_cache_bbox_min" not in filtered.metadata
    assert "voxel_cache_bbox_max" not in filtered.metadata


def test_surface_gate_partial_accept_matches_tuple_membership_path_and_counters() -> None:
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [2.1, 0.1, 0.1],
            [2.2, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    patch = _patch(points)
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    rejected_voxel_keys = {(1, 0, 0)}
    expected_mask = np.array(
        [tuple(np.floor(point / 1.0).astype(np.int64).tolist()) not in rejected_voxel_keys for point in points],
        dtype=bool,
    )
    assert structural is None
    assert filtered is not None
    np.testing.assert_array_equal(filtered.points, points[expected_mask])
    assert voxel_view is not None
    np.testing.assert_array_equal(voxel_view.unique_voxels, np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int64))
    assert debug["accepted_point_count"] == int(expected_mask.sum())
    assert debug["rejected_point_count"] == int((~expected_mask).sum())
    assert debug["surface_gate_filtered_point_count"] == int(expected_mask.sum())
    assert debug["surface_gate_rejected_point_count"] == int((~expected_mask).sum())
    assert debug["surface_gate_partial_filter_patch_count"] == 1
    assert debug["surface_gate_decision_lookup_count"] == 3
    assert debug["surface_gate_unique_decision_voxel_count"] == 3
    assert debug["surface_gate_inverse_requested_count"] == 1
    assert debug["surface_gate_inverse_build_sec"] >= 0.0
    assert debug["surface_gate_clean_inverse_skipped_count"] == 0
    assert debug["surface_gate_patch_copy_sec"] >= 0.0


def test_surface_gate_full_reject_without_structural_copy_does_not_materialize_point_mask() -> None:
    patch = _patch(
        np.array(
            [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [2.1, 0.1, 0.1]],
            dtype=np.float32,
        )
    )
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 0.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    for key in ((0, 0, 0), (1, 0, 0), (2, 0, 0)):
        state.tsdf_volume.owner_support[key] = VoxelOwnerSupport({9: 1.0})

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is None
    assert structural is None
    assert voxel_view is None
    assert debug["passed"] is False
    assert debug["point_mask_materialized"] is False
    assert debug["patch_copied"] is False
    assert debug["surface_gate_reject_without_copy_patch_count"] == 1
    assert debug["surface_gate_patch_copy_sec"] == 0.0
    assert debug["surface_gate_structural_copy_sec"] == 0.0


def test_surface_gate_representative_mode_runs_without_eager_cache() -> None:
    patch = _patch(
        np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
            dtype=np.float32,
        )
    )
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert structural is None
    assert filtered is not None
    assert debug["cached_voxel_indices_used"] is False
    assert debug["decision_voxel_count"] == 2
    assert len(filtered.points) == 2


def test_surface_gate_clean_representative_pass_through_reuses_patch_and_view() -> None:
    patch = _patch(
        np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
            dtype=np.float32,
        )
    )
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is not patch
    assert filtered.points is patch.points
    assert structural is None
    assert voxel_view is not None
    assert debug["fast_path"] is True
    assert debug["point_mask_materialized"] is False
    assert debug["patch_copied"] is False
    assert debug["surface_gate_inverse_requested_count"] == 0
    assert debug["surface_gate_inverse_build_sec"] == 0.0
    assert debug["surface_gate_clean_inverse_skipped_count"] == 1
    assert voxel_view is not None
    assert voxel_view.inverse is None
    assert "surface_owner_gate" not in patch.metadata
    assert filtered.metadata["surface_owner_gate"]["fast_path"] is True
    assert filtered.metadata["surface_owner_gate_original_lifted_point_count"] == len(patch.points)

    module._record_surface_gate_debug(debug)
    module.last_surface_gate_stats = module._summarize_surface_gate_records()
    assert module.last_surface_gate_stats["surface_gate_fast_path_patch_count"] == 1
    assert module.last_surface_gate_stats["surface_gate_point_mask_materialized_count"] == 0
    assert module.last_surface_gate_stats["surface_gate_patch_copy_count"] == 0
    assert module.last_surface_gate_stats["surface_gate_decision_lookup_count"] == 2
    assert module.last_surface_gate_stats["surface_gate_unique_decision_voxel_count"] == 2
    assert module.last_surface_gate_stats["surface_gate_filtered_point_count"] == len(patch.points)
    assert module.last_surface_gate_stats["surface_gate_rejected_point_count"] == 0
    assert module.last_surface_gate_stats["surface_gate_owner_lookup_sec"] >= 0.0
    assert module.last_surface_gate_stats["surface_gate_mask_expand_sec"] == 0.0
    assert module.last_surface_gate_stats["surface_gate_inverse_requested_count"] == 0
    assert module.last_surface_gate_stats["surface_gate_inverse_build_sec"] == 0.0
    assert module.last_surface_gate_stats["surface_gate_clean_inverse_skipped_count"] == 1


def test_surface_gate_eager_inverse_policy_builds_inverse_once_on_clean_patch() -> None:
    patch = _patch(
        np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
            dtype=np.float32,
        )
    )
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "inverse_policy": "eager",
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            }
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))

    filtered, structural, debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert filtered is not None
    assert structural is None
    assert voxel_view is not None
    assert voxel_view.inverse is not None
    assert debug["fast_path"] is True
    assert debug["surface_gate_inverse_policy"] == "eager"
    assert debug["surface_gate_inverse_requested_count"] == 1
    assert debug["surface_gate_clean_inverse_skipped_count"] == 0
    assert debug["surface_gate_unique_sec"] >= 0.0


def test_object_update_process_populates_stage_timings_and_reuses_surface_view() -> None:
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32)
    patch = _patch(points, patch_id=12, frame_id=3)
    obj = ObjectMap(
        object_id=4,
        local_pcd=points[:1].copy(),
        centroid=points[0].copy(),
        bbox_min=points[0].copy(),
        bbox_max=points[0].copy(),
        last_seen_frame=2,
        creation_frame=2,
    )
    state = SystemState(objects={4: obj}, tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    module = ObjectUpdateModule(
        {
            "refresh_object_debug_interval": 10,
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )

    module.process(AssociationResult(matched=[(12, 4, 1.0)]), [patch], state)

    expected = {
        "object_update_current_frame_visibility_gate",
        "object_update_surface_owner_gate",
        "object_update_tsdf_integrate",
        "object_update_local_pcd_update",
        "object_update_association_geometry_update",
        "object_update_refresh_object_debug",
        "object_update_semantic_vote",
        "object_update_provisional_pool",
        "object_update_surface_gate_summary",
    }
    assert expected.issubset(module.last_stage_timings)
    assert all(module.last_stage_timings[key] >= 0.0 for key in expected)
    assert module.last_surface_gate_stats["surface_gate_fast_path_patch_count"] == 1
    assert obj.debug["global_instance_substrate"]["owned_voxel_count"] == 2


def test_filtered_surface_view_integrates_only_accepted_voxels() -> None:
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    patch = _patch(points)
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "representative_voxel_mode": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.1,
                "max_foreign_owner_ratio": 1.0,
                "max_background_owner_ratio": 1.0,
            },
            "tsdf": {"voxel_size": 1.0},
        }
    )
    state = SystemState(tsdf_volume=TSDFInstanceVolume(voxel_size=1.0))
    state.tsdf_volume.owner_support[(1, 0, 0)] = VoxelOwnerSupport({9: 1.0})

    filtered, structural, _debug, voxel_view = module._filter_patch_by_surface_owner(
        patch,
        target_object_id=7,
        state=state,
        background_support=set(),
        new_object=False,
    )

    assert structural is None
    assert filtered is not None
    assert voxel_view is not None
    np.testing.assert_array_equal(voxel_view.unique_voxels, np.array([[0, 0, 0]], dtype=np.int64))
    assert _debug["surface_gate_partial_filter_patch_count"] == 1

    volume = TSDFInstanceVolume(voxel_size=1.0)
    module.tsdf_module.integrate_patch(volume, filtered, 7, voxel_view=voxel_view)
    assert set(volume.owner_support) == {(0, 0, 0)}


def test_refresh_object_debug_interval_keeps_runtime_fields_current_between_periods(monkeypatch) -> None:
    patch = _patch(np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float32), frame_id=0)
    obj = ObjectMap(
        object_id=2,
        local_pcd=patch.points.copy(),
        centroid=patch.centroid.copy(),
        bbox_min=patch.bbox_min.copy(),
        bbox_max=patch.bbox_max.copy(),
        last_seen_frame=0,
        creation_frame=0,
    )
    volume = TSDFInstanceVolume(voxel_size=1.0)
    module = ObjectUpdateModule({"refresh_object_debug_interval": 10})
    module.tsdf_module.integrate_patch(volume, patch, obj.object_id)

    calls = {"count": 0}
    original = module.tsdf_module.summarize_instance_support

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.tsdf_module, "summarize_instance_support", counted)
    module._refresh_object_debug(obj, volume, current_frame=0, created=True)
    first_owned_count = obj.debug["global_instance_substrate"]["owned_voxel_count"]
    extra_patch = _patch(np.array([[2.1, 0.1, 0.1]], dtype=np.float32), frame_id=1)
    module.tsdf_module.integrate_patch(volume, extra_patch, obj.object_id)
    module._refresh_object_debug(obj, volume, current_frame=1, created=False)

    assert calls["count"] == 2
    assert "global_instance_substrate" in obj.debug
    assert "local_geometry_memory" in obj.debug
    assert obj.debug["global_instance_substrate"]["owned_voxel_count"] == first_owned_count + 1
    assert obj.debug["global_instance_substrate"]["last_incremental_refresh_frame"] == 1


def test_dense_surface_lazy_mode_defers_points_and_refresh_all_for_export_builds_surface() -> None:
    points = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]], dtype=np.float32)
    patch = _patch(points, frame_id=3)
    obj = ObjectMap(
        object_id=2,
        local_pcd=points.copy(),
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        observations=[ObservationRecord(frame_id=3, patch=patch)],
        last_seen_frame=3,
        creation_frame=3,
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    state = SystemState(objects={2: obj})
    state.dense_surface_map.entries[2] = DenseSurfaceEntry(
        points=np.empty((0, 3), dtype=np.float32),
        object_id=2,
        semantic_label="old",
        last_refresh_frame=-1,
        resident=True,
    )
    module = DenseSurfaceModule(
        {
            "dense_surface_voxel": 0.001,
            "dense_surface_cap_per_object": 8,
            "dense_surface_lazy_export": True,
        }
    )

    state = module.process(state, frame_id=3)
    assert len(state.dense_surface_map.entries[2].points) == 0
    assert state.dense_surface_map.entries[2].semantic_label == "chair"

    state = module.refresh_all_for_export(state, frame_id=3)
    assert len(state.dense_surface_map.entries[2].points) > 0
    assert state.dense_surface_map.entries[2].semantic_label == "chair"


def _active_set_state() -> SystemState:
    state = SystemState()
    state.objects = {
        1: ObjectMap(
            object_id=1,
            centroid=np.array([0.2, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([0.0, -0.1, -0.1], dtype=np.float32),
            bbox_max=np.array([0.4, 0.1, 0.1], dtype=np.float32),
            whole_evidence=WholeEvidenceScores(whole_evidence_score=0.1),
        ),
        2: ObjectMap(
            object_id=2,
            centroid=np.array([4.0, 0.0, 0.0], dtype=np.float32),
            bbox_min=np.array([3.9, -0.1, -0.1], dtype=np.float32),
            bbox_max=np.array([4.1, 0.1, 0.1], dtype=np.float32),
            whole_evidence=WholeEvidenceScores(whole_evidence_score=0.7),
        ),
    }
    return state


def test_active_set_cache_hit_reuses_equal_results() -> None:
    state = _active_set_state()
    module = ActiveSetModule({"cache_enabled": True, "nearby_radius": 1.0, "whole_prior_threshold": 0.5})
    calls = {"visible": 0}

    def visible(_volume, _pose, _intrinsics):
        calls["visible"] += 1
        return {2}

    module.tsdf_module.query_visible_instances = visible
    pose = np.eye(4, dtype=np.float32)
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=4, height=4)

    first = module.process(state, pose, intr)
    second = module.process(state, pose.copy(), intr)

    assert calls["visible"] == 1
    assert second.visible_ids == first.visible_ids == {2}
    assert second.nearby_ids == first.nearby_ids == {1}
    assert second.whole_prior_ids == first.whole_prior_ids == {2}
    assert module.last_debug["visible_cache_hit"] == 1
    assert module.last_debug["nearby_cache_hit"] == 1
    assert module.last_debug["whole_prior_cache_hit"] == 1


def test_active_set_cache_misses_after_tsdf_and_object_signature_changes() -> None:
    state = _active_set_state()
    module = ActiveSetModule({"cache_enabled": True, "nearby_radius": 1.0, "whole_prior_threshold": 0.5})
    calls = {"visible": 0}

    def visible(_volume, _pose, _intrinsics):
        calls["visible"] += 1
        return {calls["visible"]}

    module.tsdf_module.query_visible_instances = visible
    pose = np.eye(4, dtype=np.float32)
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=4, height=4)

    module.process(state, pose, intr)
    state.tsdf_volume.owner_index_revision += 1
    state.objects[1].centroid = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    state.objects[1].update_count += 1
    state.objects[2].whole_evidence.whole_evidence_score = 0.2
    second = module.process(state, pose, intr)

    assert calls["visible"] == 2
    assert second.visible_ids == {2}
    assert second.nearby_ids == set()
    assert second.whole_prior_ids == set()
    assert module.last_debug["visible_cache_miss"] == 1
    assert module.last_debug["nearby_cache_miss"] == 1
    assert module.last_debug["whole_prior_cache_miss"] == 1


def test_active_set_cache_disabled_preserves_uncached_visible_calls() -> None:
    state = _active_set_state()
    module = ActiveSetModule({"cache_enabled": False, "nearby_radius": 1.0, "whole_prior_threshold": 0.5})
    calls = {"visible": 0}

    def visible(_volume, _pose, _intrinsics):
        calls["visible"] += 1
        return {2}

    module.tsdf_module.query_visible_instances = visible
    pose = np.eye(4, dtype=np.float32)
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=4, height=4)

    first = module.process(state, pose, intr)
    second = module.process(state, pose, intr)

    assert calls["visible"] == 2
    assert second.all_candidate_ids == first.all_candidate_ids
    assert module.last_debug["active_set_cache_enabled"] is False
    assert module.last_debug["visible_cache_hit"] == 0
    assert module.last_debug["nearby_cache_hit"] == 0
    assert module.last_debug["whole_prior_cache_hit"] == 0


def test_backend_fast_config_contains_runtime_knobs() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "replica_yoloworld_anchor_first_sam_backend_fast_4090.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["association"]["max_scored_candidates"] == 16
    assert config["association"]["max_geometry_candidates"] == 4
    assert config["association"]["two_stage_enabled"] is True
    assert config["association"]["top_k_geometry"] == 3
    assert config["association"]["top_k_final"] == 16
    assert config["active_set"]["cache_enabled"] is False
    assert config["association"]["geometry_nn_backend"] == "auto"
    assert config["association"]["geometry_vectorized_max_points"] == 160
    assert config["association"]["indexed_retrieval_enabled"] is True
    assert config["association"]["centroid_kdtree_enabled"] is True
    assert config["association"]["max_spatial_candidates"] == 12
    assert config["association"]["spatial_candidate_radius"] == 0.0
    assert config["association"]["geometry_kdtree_cache_enabled"] is True
    assert config["association"]["geometry_kdtree_cache_min_points"] == 64
    assert config["association"]["kdtree_cache_enabled"] is True
    assert config["association"]["kdtree_cache_min_object_points"] == 64
    assert config["patch_lifting"]["voxel_cache_enabled"] is False
    assert config["object_update"]["max_points_per_object"] == 20000
    assert config["object_update"]["local_pcd_incremental_bounds_enabled"] is True
    assert config["object_update"]["local_pcd_chunk_pool"]["enabled"] is True
    assert config["object_update"]["local_pcd_chunk_pool"]["bounded_compaction_enabled"] is True
    assert config["object_update"]["local_pcd_chunk_pool"]["compact_to_max_points"] is True
    assert config["object_update"]["local_pcd_chunk_pool"]["materialize_interval"] == 20
    assert config["object_update"]["local_pcd_chunk_pool"]["max_pending_points"] == 50000
    assert config["object_update"]["local_pcd_voxel_pool"]["enabled"] is True
    assert config["object_update"]["local_pcd_voxel_pool"]["materialize_on_interval"] is False
    assert config["object_update"]["local_pcd_voxel_pool"]["voxel_size"] == 0.01
    assert config["object_update"]["local_pcd_voxel_pool"]["max_keys_per_object"] == 20000
    assert config["object_update"]["association_geometry"]["sketch_enabled"] is True
    assert config["object_update"]["association_geometry"]["materialize_interval"] == 20
    assert config["object_update"]["association_geometry"]["patch_budget_points"] == 256
    assert config["object_update"]["association_geometry"]["update_budget_new_keys"] == 128
    assert config["object_update"]["refresh_object_debug_interval"] == 10
    assert config["object_update"]["refresh_object_debug_on_create"] is True
    assert config["object_update"]["surface_owner_gate"]["representative_voxel_mode"] is True
    assert config["object_update"]["surface_owner_gate"]["inverse_policy"] == "eager"
    assert config["dual_map"]["dense_surface_update_interval"] == 5
    assert config["dual_map"]["dense_surface_cap_per_object"] == 2000
    assert config["dual_map"]["dense_surface_lazy_export"] is True
    assert config["runtime_vis"]["benchmark_profile"] == "fast"
    assert config["runtime_vis"]["debug_images_enabled"] is False
    assert config["runtime_vis"]["debug_every"] == 0
