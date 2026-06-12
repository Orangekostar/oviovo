"""Tests for the provisional local object pool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data_structures import (
    AssociationResult,
    CameraIntrinsics,
    ObjectMap,
    Patch3D,
    ProposalSoftScores,
    SystemState,
    VoxelOwnerSupport,
)
from src.modules.dense_surface import DenseSurfaceModule
from src.modules.object_update import ObjectUpdateModule
from src.modules.semantic_memory import object_export_semantic_label


def _make_patch(patch_id: int, frame_id: int, offset: float = 0.0) -> Patch3D:
    points = np.array(
        [
            [0.0 + offset, 0.0, 1.0],
            [0.1 + offset, 0.0, 1.0],
            [0.0 + offset, 0.1, 1.0],
            [0.1 + offset, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    return Patch3D(
        patch_id=patch_id,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=frame_id,
    )


def _make_dense_patch(
    patch_id: int,
    frame_id: int,
    *,
    split_origin: str,
    objectness_score: float = 0.8,
    backgroundness_score: float = 0.2,
) -> Patch3D:
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16))
    zs = np.full_like(xs, 1.0 + 0.01 * patch_id)
    points = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    return Patch3D(
        patch_id=patch_id,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=frame_id,
        soft_scores=ProposalSoftScores(
            objectness_score=objectness_score,
            backgroundness_score=backgroundness_score,
            attachedness_score=0.6,
        ),
        metadata={"split_origin": split_origin},
    )


def test_provisional_pool_buffers_then_promotes_new_objects() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_voxel_size": 0.02,
            "downsample_interval": 5,
            "max_points_per_object": 10000,
            "provisional_pool": {
                "enabled": True,
                "match_distance": 0.3,
                "promotion_hits": 3,
                "max_idle_frames": 30,
                "downsample_voxel_size": 0.02,
                "max_points_per_object": 6000,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    state = SystemState()

    for frame_id in range(3):
        patch = _make_patch(patch_id=frame_id, frame_id=frame_id, offset=0.01 * frame_id)
        patch.metadata["anchor_class_name"] = "rug"
        patch.metadata["anchor_confidence"] = 0.9
        patch.metadata["anchor_view_quality"] = 0.8
        association = AssociationResult(new_object_patches=[patch.patch_id])
        state = module.process(association, [patch], state)
        if frame_id < 2:
            assert len(state.objects) == 0
            assert len(state.provisional_objects) == 1

    assert len(state.provisional_objects) == 0
    assert len(state.objects) == 1
    obj = next(iter(state.objects.values()))
    assert obj.creation_frame == 2
    assert obj.update_count >= 3
    assert len(state.tsdf_volume.owner_support) > 0
    assert object_export_semantic_label(obj) == "rug"
    assert obj.debug["anchor_semantics"]["canonical_label"] == "rug"


def test_provisional_pool_prunes_stale_entries() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "match_distance": 0.3,
                "promotion_hits": 5,
                "max_idle_frames": 1,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    state = SystemState()

    patch0 = _make_patch(patch_id=0, frame_id=0)
    state = module.process(AssociationResult(new_object_patches=[0]), [patch0], state)
    assert len(state.provisional_objects) == 1

    patch2 = _make_patch(patch_id=2, frame_id=2, offset=1.0)
    state = module.process(AssociationResult(new_object_patches=[2]), [patch2], state)
    assert len(state.provisional_objects) == 1
    provisional = next(iter(state.provisional_objects.values()))
    assert provisional.first_seen_frame == 2


def test_provisional_pool_records_semantic_split_candidates() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 3,
                "max_idle_frames": 30,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=9, frame_id=4)
    patch.metadata.update(
        {
            "anchor_class_name": "cushion",
            "anchor_confidence": 0.8,
            "semantic_split_candidate_from_object_id": 2,
            "semantic_split_candidate_parent_label": "sofa",
            "semantic_split_candidate_new_label": "cushion",
            "semantic_split_candidate_reason": "semantic_conflict_subregion",
        }
    )

    state = module.process(AssociationResult(new_object_patches=[9]), [patch], SystemState())

    provisional = next(iter(state.provisional_objects.values()))
    assert provisional.debug["semantic_split_candidate_from_object_id"] == 2
    assert provisional.debug["semantic_split_candidate_parent_label"] == "sofa"
    assert provisional.debug["semantic_split_candidate_new_label"] == "cushion"


def test_contested_patch_does_not_update_parent_object_memory_or_votes() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {"enabled": True, "promotion_hits": 3, "max_idle_frames": 30},
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    sofa_points = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]], dtype=np.float32)
    sofa = ObjectMap(
        object_id=31,
        local_pcd=sofa_points.copy(),
        centroid=sofa_points.mean(axis=0),
        bbox_min=sofa_points.min(axis=0),
        bbox_max=sofa_points.max(axis=0),
    )
    sofa.debug["anchor_semantics"] = {
        "canonical_label": "sofa",
        "canonical_score": 1.0,
        "label_votes": {"sofa": 1.0},
        "label_max_confidence": {"sofa": 0.95},
    }
    blanket_patch = _make_patch(patch_id=22, frame_id=8)
    blanket_patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.88,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = SystemState(objects={31: sofa}, next_object_id=32)
    association = AssociationResult(contested_object_patches=[22])

    updated = module.process(association, [blanket_patch], state)

    assert np.array_equal(updated.objects[31].local_pcd, sofa_points)
    assert updated.objects[31].update_count == 0
    assert updated.objects[31].observations == []
    assert updated.objects[31].debug["anchor_semantics"]["label_votes"] == {"sofa": 1.0}
    assert len(updated.provisional_objects) == 1
    residual = next(iter(updated.provisional_objects.values()))
    assert residual.anchor_class_name == "blanket"
    assert residual.debug["contested_parent_object_id"] == 31
    assert residual.debug["contested_parent_label"] == "sofa"
    assert residual.debug["contested_patch_label"] == "blanket"


def test_contested_patch_survives_parent_owned_surface_gate() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 1,
                "min_new_object_accept_ratio": 0.55,
                "max_foreign_owner_ratio": 0.25,
            },
            "provisional_pool": {"enabled": True, "promotion_hits": 3, "max_idle_frames": 30},
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=122, frame_id=12)
    patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.88,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    sofa = ObjectMap(object_id=31)
    state = SystemState(objects={31: sofa}, next_object_id=32)
    for voxel in module._points_to_voxels(patch.points, state.tsdf_volume.voxel_size):
        state.tsdf_volume.owner_support[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = VoxelOwnerSupport(
            support={31: 1.0}
        )

    updated = module.process(AssociationResult(contested_object_patches=[122]), [patch], state)

    assert len(updated.provisional_objects) == 1
    residual = next(iter(updated.provisional_objects.values()))
    assert residual.anchor_class_name == "blanket"
    assert module.last_contested_residual_patch_ids == [122]
    gate_record = module.last_surface_gate_records[-1]
    assert gate_record["mode"] == "contested_residual"
    assert gate_record["same_owner_point_count"] == len(patch.points)
    assert gate_record["foreign_owner_point_count"] == 0
    assert gate_record["passed"] is True


def test_contested_patch_still_rejects_third_party_owned_surface() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 1,
                "min_new_object_accept_ratio": 0.55,
                "max_foreign_owner_ratio": 0.25,
            },
            "provisional_pool": {"enabled": True, "promotion_hits": 3, "max_idle_frames": 30},
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=123, frame_id=12)
    patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.88,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = SystemState(objects={31: ObjectMap(object_id=31), 44: ObjectMap(object_id=44)}, next_object_id=45)
    for voxel in module._points_to_voxels(patch.points, state.tsdf_volume.voxel_size):
        state.tsdf_volume.owner_support[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = VoxelOwnerSupport(
            support={44: 1.0}
        )

    updated = module.process(AssociationResult(contested_object_patches=[123]), [patch], state)

    assert len(updated.provisional_objects) == 0
    assert module.last_contested_residual_patch_ids == []
    gate_record = module.last_surface_gate_records[-1]
    assert gate_record["mode"] == "contested_residual"
    assert gate_record["foreign_owner_point_count"] == len(patch.points)
    assert "foreign_owner_ratio" in gate_record["rejection_reasons"]


def test_contested_patch_promotes_when_normal_provisional_pool_disabled() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {"enabled": False, "promotion_hits": 1},
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    initial_object_id = 41
    blanket_patch = _make_patch(patch_id=23, frame_id=9)
    blanket_patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.91,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = SystemState(next_object_id=initial_object_id)

    updated = module.process(
        AssociationResult(contested_object_patches=[23]),
        [blanket_patch],
        state,
    )

    assert updated.next_object_id == initial_object_id + 1
    assert len(updated.provisional_objects) == 0
    assert list(updated.objects) == [initial_object_id]
    promoted = updated.objects[initial_object_id]
    assert promoted.debug["promoted_contested_residual"] == {
        "parent_object_id": 31,
        "parent_label": "sofa",
        "patch_label": "blanket",
        "reason": "cross_label_observation_identity",
    }
    assert module.last_contested_residual_promoted_object_ids == [initial_object_id]


def test_contested_residual_promotes_after_repeated_observations() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 2,
                "match_distance": 0.40,
                "max_idle_frames": 30,
            },
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    state = SystemState(
        objects={
            31: ObjectMap(
                object_id=31,
                local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                bbox_min=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                bbox_max=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
        },
        next_object_id=32,
    )

    first_patch = _make_patch(patch_id=101, frame_id=10)
    first_patch.metadata.update(
        {
            "anchor_class_name": "cushion",
            "anchor_confidence": 0.86,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "cushion",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = module.process(AssociationResult(contested_object_patches=[101]), [first_patch], state)

    assert 32 not in state.objects
    assert len(state.provisional_objects) == 1
    assert module.last_contested_residual_promoted_object_ids == []

    second_patch = _make_patch(patch_id=102, frame_id=11)
    second_patch.metadata.update(
        {
            "anchor_class_name": "cushion",
            "anchor_confidence": 0.86,
            "contested_parent_object_id": 31,
            "contested_parent_label": "sofa",
            "contested_patch_label": "cushion",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = module.process(AssociationResult(contested_object_patches=[102]), [second_patch], state)

    assert 32 in state.objects
    promoted = state.objects[32]
    assert promoted.debug["anchor_semantics"]["canonical_label"] == "cushion"
    assert promoted.debug["promoted_contested_residual"]["parent_object_id"] == 31
    assert promoted.debug["promoted_contested_residual"]["parent_label"] == "sofa"
    assert promoted.debug["promoted_contested_residual"]["patch_label"] == "cushion"
    assert len(state.provisional_objects) == 0
    assert module.last_contested_residual_promoted_object_ids == [32]
    assert len(state.tsdf_volume.owner_support) > 0


def test_blocked_residual_object_exports_later_direct_anchor_label() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 2,
                "match_distance": 0.40,
                "max_idle_frames": 30,
            },
            "contested_residual_pool": {"enabled": True},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    state = SystemState(next_object_id=10)

    blocked_patch = _make_patch(patch_id=201, frame_id=20)
    blocked_patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.94,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
            "semantic_commit_allowed": False,
            "residual_semantic_policy": "unknown",
            "mask_anchor_relation": "contained_residual",
            "contested_parent_object_id": 3,
            "contested_parent_label": "sofa",
            "contested_patch_label": "blanket",
            "contested_reason": "cross_label_observation_identity",
        }
    )
    state = module.process(AssociationResult(contested_object_patches=[201]), [blocked_patch], state)

    direct_patch = _make_patch(patch_id=202, frame_id=21, offset=0.01)
    direct_patch.metadata.update(
        {
            "anchor_class_name": "blanket",
            "anchor_confidence": 0.95,
            "anchor_view_quality": 0.90,
            "anchor_label_strength": "strong",
        }
    )
    state = module.process(AssociationResult(new_object_patches=[202]), [direct_patch], state)

    assert list(state.objects) == [10]
    obj = state.objects[10]
    anchor_state = obj.debug["anchor_semantics"]
    assert [item["frame_id"] for item in anchor_state["delayed_evidence"]] == [20]
    assert [item["reason"] for item in anchor_state["delayed_evidence"]] == [
        "cross_label_observation_identity"
    ]
    assert [item["frame_id"] for item in anchor_state["evidence"]] == [21]
    assert anchor_state["ignored_observation_reasons"] == {"semantic_commit_blocked": 1}
    assert object_export_semantic_label(obj) == "blanket"


def test_ambiguous_patch_matches_existing_object_and_updates_local_memory() -> None:
    module = ObjectUpdateModule({"tsdf": {"voxel_size": 0.05}})
    patch = _make_dense_patch(patch_id=10, frame_id=5, split_origin="ambiguous")
    existing = ObjectMap(
        object_id=3,
        local_pcd=patch.points[:8].copy(),
        centroid=patch.points[:8].mean(axis=0),
        bbox_min=patch.points[:8].min(axis=0),
        bbox_max=patch.points[:8].max(axis=0),
    )
    state = SystemState(objects={3: existing}, next_object_id=4)
    association = AssociationResult(matched=[(10, 3, None)])

    updated = module.process(association, [patch], state)

    assert len(updated.objects[3].local_pcd) > len(patch.points[:8])
    assert updated.objects[3].last_seen_frame == 5
    assert len(updated.objects[3].observations) == 1
    assert len(updated.tsdf_volume.owner_support) > 0


def test_ambiguous_patch_only_enters_provisional_pool_when_unmatched() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 3,
                "max_idle_frames": 30,
                "unanchored_min_points": 32,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_dense_patch(
        patch_id=11,
        frame_id=7,
        split_origin="ambiguous",
        objectness_score=0.7,
        backgroundness_score=0.3,
    )
    state = module.process(AssociationResult(new_object_patches=[11]), [patch], SystemState())

    assert len(state.objects) == 0
    assert len(state.provisional_objects) == 1


def test_ambiguous_patch_does_not_create_direct_object_when_provisional_disabled() -> None:
    module = ObjectUpdateModule({"provisional_pool": {"enabled": False}, "tsdf": {"voxel_size": 0.05}})
    patch = _make_dense_patch(
        patch_id=12,
        frame_id=3,
        split_origin="ambiguous",
        objectness_score=0.8,
        backgroundness_score=0.2,
    )
    state = module.process(AssociationResult(new_object_patches=[12]), [patch], SystemState())

    assert len(state.objects) == 0
    assert len(state.provisional_objects) == 0


def test_background_patch_stays_out_of_object_updates() -> None:
    module = ObjectUpdateModule(
        {
            "provisional_pool": {
                "enabled": True,
                "promotion_hits": 3,
                "max_idle_frames": 30,
                "unanchored_min_points": 32,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_dense_patch(
        patch_id=13,
        frame_id=4,
        split_origin="background",
        objectness_score=0.1,
        backgroundness_score=0.9,
    )
    state = module.process(AssociationResult(new_object_patches=[13]), [patch], SystemState())

    assert len(state.objects) == 0
    assert len(state.provisional_objects) == 0


def test_surface_owner_gate_rejects_foreign_owned_voxels_before_object_update() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.25,
                "max_foreign_owner_ratio": 0.25,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=21, frame_id=2)
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=1,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={1: existing}, next_object_id=2)
    for voxel in module._points_to_voxels(patch.points, state.tsdf_volume.voxel_size):
        state.tsdf_volume.owner_support[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] = VoxelOwnerSupport(
            support={2: 1.0}
        )

    updated = module.process(AssociationResult(matched=[(21, 1, None)]), [patch], state)

    assert np.array_equal(updated.objects[1].local_pcd, existing_points)
    assert len(updated.objects[1].observations) == 0
    assert module.last_surface_gate_stats["rejected_patch_count"] == 1
    assert "foreign_owner_ratio" in updated.objects[1].debug["last_surface_owner_gate"]["rejection_reasons"]


def test_surface_owner_gate_rejects_background_owned_new_object_patch() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 1,
                "min_new_object_accept_ratio": 0.55,
                "max_background_owner_ratio": 0.35,
            },
            "provisional_pool": {"enabled": False},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=22, frame_id=3)
    patch.metadata["anchor_class_name"] = "stool"

    updated = module.process(
        AssociationResult(new_object_patches=[22]),
        [patch],
        SystemState(),
        background_patches=[patch],
    )

    assert len(updated.objects) == 0
    assert len(updated.provisional_objects) == 0
    assert len(module.last_structural_reject_patches) == 1
    assert module.last_structural_reject_patches[0].metadata["split_origin"] == "background"
    assert module.last_surface_gate_stats["background_owner_point_count"] == len(patch.points)


def test_surface_owner_gate_allows_small_attached_surface_class_on_background() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 20,
                "min_new_object_accept_ratio": 0.55,
                "max_background_owner_ratio": 0.35,
                "attached_surface_classes": ["switch"],
                "attached_max_voxels": 40,
            },
            "provisional_pool": {"enabled": False},
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=24, frame_id=3)
    patch.metadata["anchor_class_name"] = "switch"

    updated = module.process(
        AssociationResult(new_object_patches=[24]),
        [patch],
        SystemState(),
        background_patches=[patch],
    )

    assert len(updated.objects) == 1
    obj = next(iter(updated.objects.values()))
    assert obj.debug["last_surface_owner_gate"]["attached_surface_override"] is True
    assert obj.debug["last_surface_owner_gate"]["min_accept_points"] == 1


def test_surface_owner_gate_writes_only_accepted_patch_points_to_memory_and_dense_surface() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {
                "enabled": True,
                "min_accept_points": 1,
                "min_update_accept_ratio": 0.25,
                "max_background_owner_ratio": 0.75,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=23, frame_id=6)
    background_points = patch.points[:2].copy()
    accepted_points = patch.points[2:].copy()
    background_patch = Patch3D(
        patch_id=90,
        points=background_points,
        centroid=background_points.mean(axis=0),
        bbox_min=background_points.min(axis=0),
        bbox_max=background_points.max(axis=0),
        source_frame_id=6,
    )
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)

    updated = module.process(
        AssociationResult(matched=[(23, 4, None)]),
        [patch],
        state,
        background_patches=[background_patch],
    )

    obj = updated.objects[4]
    assert len(obj.observations) == 1
    assert len(obj.observations[0].patch.points) == len(accepted_points)
    assert np.allclose(obj.observations[0].patch.points, accepted_points)
    for rejected_point in background_points:
        assert not any(np.allclose(point, rejected_point) for point in obj.local_pcd)
    assert module.last_surface_gate_stats["accepted_point_count"] == len(accepted_points)
    assert len(module.last_structural_reject_patches) == 1

    updated = DenseSurfaceModule({"dense_surface_voxel": 0.001, "dense_surface_cap_per_object": 16}).process(
        updated,
        frame_id=6,
    )
    dense_points = updated.dense_surface_map.entries[4].points
    for rejected_point in background_points:
        assert not any(np.allclose(point, rejected_point) for point in dense_points)
    assert any(np.allclose(point, accepted_points[0]) for point in dense_points)


def test_current_frame_visibility_gate_writes_only_depth_consistent_points() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {"enabled": False},
            "current_frame_visibility_gate": {
                "enabled": True,
                "distance_threshold": 0.05,
                "min_accept_points": 1,
                "min_accept_ratio": 0.25,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 2.0],
            [0.0, -0.1, 1.0],
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=33,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=3,
    )
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[2, 3] = 0.5
    depth[3, 3] = 1.0

    updated = module.process(
        AssociationResult(matched=[(33, 4, None)]),
        [patch],
        state,
        current_depth=depth,
        current_pose=np.eye(4, dtype=np.float64),
        current_intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
    )

    obj = updated.objects[4]
    assert len(obj.local_pcd) == 3
    np.testing.assert_allclose(obj.local_pcd[1:], np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], dtype=np.float32))
    gate = obj.debug["last_current_frame_visibility_gate"]
    assert gate["accepted_point_count"] == 2
    assert gate["depth_rejected_point_count"] == 1
    assert gate["invalid_depth_point_count"] == 1


def test_current_frame_visibility_gate_patch_metadata_is_not_surface_owner_metadata() -> None:
    module = ObjectUpdateModule(
        {
            "downsample_interval": 99,
            "surface_owner_gate": {"enabled": False},
            "current_frame_visibility_gate": {
                "enabled": True,
                "distance_threshold": 0.05,
                "min_accept_points": 1,
                "min_accept_ratio": 0.25,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 2.0],
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    patch = Patch3D(
        patch_id=35,
        points=points,
        centroid=points.mean(axis=0),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        source_frame_id=3,
        metadata={"lifted_point_count": 7},
    )
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0
    depth[3, 3] = 1.0

    updated = module.process(
        AssociationResult(matched=[(35, 4, None)]),
        [patch],
        state,
        current_depth=depth,
        current_pose=np.eye(4, dtype=np.float64),
        current_intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
    )

    metadata = updated.objects[4].observations[0].patch.metadata
    assert "current_frame_visibility_gate" in metadata
    assert metadata["current_frame_visibility_gate"]["accepted_point_count"] == 2
    assert metadata["current_frame_visibility_gate_original_lifted_point_count"] == 7
    assert "surface_owner_gate_original_lifted_point_count" not in metadata


def test_current_frame_visibility_gate_rejects_patch_when_too_few_points_remain() -> None:
    module = ObjectUpdateModule(
        {
            "surface_owner_gate": {"enabled": False},
            "current_frame_visibility_gate": {
                "enabled": True,
                "distance_threshold": 0.05,
                "min_accept_points": 2,
                "min_accept_ratio": 0.75,
            },
            "tsdf": {"voxel_size": 0.05},
        }
    )
    patch = _make_patch(patch_id=34, frame_id=3)
    existing_points = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    existing = ObjectMap(
        object_id=4,
        local_pcd=existing_points.copy(),
        centroid=existing_points[0],
        bbox_min=existing_points[0],
        bbox_max=existing_points[0],
    )
    state = SystemState(objects={4: existing}, next_object_id=5)
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 1.0

    updated = module.process(
        AssociationResult(matched=[(34, 4, None)]),
        [patch],
        state,
        current_depth=depth,
        current_pose=np.eye(4, dtype=np.float64),
        current_intrinsics=CameraIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=5, height=5),
    )

    assert np.array_equal(updated.objects[4].local_pcd, existing_points)
    assert module.last_current_frame_visibility_gate_stats["rejected_patch_count"] == 1
    assert "low_accept_ratio" in updated.objects[4].debug["last_current_frame_visibility_gate"]["rejection_reasons"]


def test_deterministic_spatial_cap_preserves_quadrant_coverage() -> None:
    quadrants = []
    for base_x, base_y in ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)):
        xs, ys = np.meshgrid(np.linspace(base_x, base_x + 0.9, 5), np.linspace(base_y, base_y + 0.9, 5))
        zs = np.full_like(xs, 1.0)
        quadrants.append(np.stack([xs, ys, zs], axis=-1).reshape(-1, 3))
    points = np.concatenate(quadrants, axis=0).astype(np.float32)

    capped_a = ObjectUpdateModule._deterministic_spatial_cap(points, 8)
    capped_b = ObjectUpdateModule._deterministic_spatial_cap(points, 8)

    assert np.array_equal(capped_a, capped_b)
    quadrant_hits = {
        (int(point[0] >= 1.5), int(point[1] >= 1.5))
        for point in capped_a
    }
    assert quadrant_hits == {(0, 0), (1, 0), (0, 1), (1, 1)}
