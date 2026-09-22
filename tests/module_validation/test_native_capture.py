"""Immutable native capture records and leakage boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.static_ovmap.module_validation.native_capture import (
    FrameObservation,
    NativeCaptureSession,
    NativeScenePack,
    RegionRequest,
    load_native_scene_pack,
    write_native_scene_pack,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OVI_PATCH = (
    REPOSITORY_ROOT
    / "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch"
)


def _request() -> RegionRequest:
    return RegionRequest(
        scene_id="scene0000_00",
        frame_id=10,
        target_id="owner:7",
        lineage=("segment:2", "owner:7"),
        source_map_version="map:10",
        target_mask_sha256=SHA_A,
        bbox_xyxy=(1, 2, 8, 9),
        native_union_mask_sha256=SHA_B,
        visible_target_pixels=24,
        crop_convention="native_global_bbox_union_v1",
        requested_view_rank=0,
        image_sha256=SHA_C,
    )


def _frame(request: RegionRequest | None = None) -> FrameObservation:
    request = request or _request()
    return FrameObservation(
        scene_id="scene0000_00",
        frame_id=10,
        pose_c2w=np.eye(4, dtype=np.float32),
        image_size_hw=(12, 16),
        intrinsics=np.array(
            [[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        rgb_path="frames/000010.rgb.png",
        rgb_sha256=SHA_A,
        depth_path="frames/000010.depth.npz",
        depth_sha256=SHA_B,
        panoptic_path="frames/000010.panoptic.png",
        panoptic_sha256=SHA_C,
        global_owner_path="frames/000010.owner.png",
        global_owner_sha256="d" * 64,
        map_state_id="map:10",
        refined_segment_ids=np.array([1, 2], dtype=np.int64),
        registered_labels=np.array([4, 7], dtype=np.int64),
        requests=(request,),
        native_selected_request_ids=(request.request_id,),
        request_completion_boundary=3,
    )


def _pack() -> NativeScenePack:
    return NativeScenePack(
        scene_id="scene0000_00",
        family_id="scene0000",
        split="FIT",
        domain="N",
        source_config={"mapper": "native"},
        source_revisions={"ovi": "f" * 40},
        surface_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        surface_normals=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normal_valid=np.array([True, False]),
        original_owner=np.array([7, 0], dtype=np.int64),
        segment_labels=np.array([4, 0], dtype=np.int64),
        alias_table=((9, 4),),
        instance_registry={7: {"class_id": 2, "source": "native"}},
        tsdf_sha256=SHA_A,
        projection_identity="native-source-projection-v1",
        frames=(_frame(),),
        class_vocabulary=("wall", "chair"),
        text_space_id="native-siglip-l16-384:replica51",
    )


def test_native_scene_pack_requires_aligned_surface_rows() -> None:
    pack = _pack()

    with pytest.raises(ValueError, match="surface rows"):
        replace(pack, original_owner=np.array([7], dtype=np.int64))


def test_native_scene_pack_copies_and_freezes_arrays() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    pack = replace(_pack(), surface_xyz=xyz)
    xyz[0, 0] = 99.0

    assert pack.surface_xyz[0, 0] == 0.0
    assert pack.surface_xyz.flags.writeable is False
    with pytest.raises(ValueError):
        pack.original_owner[0] = 2


def test_region_request_identity_covers_lineage_and_rejects_wrong_frame_state() -> None:
    request = _request()
    changed = replace(request, lineage=("segment:3", "owner:7"))

    assert request.request_id != changed.request_id
    with pytest.raises(ValueError, match="source map version"):
        _frame(replace(request, source_map_version="map:9"))
    with pytest.raises(ValueError, match="lineage"):
        replace(request, lineage=("segment:2",))


def test_native_scene_pack_rejects_prediction_leakage_keys() -> None:
    with pytest.raises(ValueError, match="forbidden prediction metadata"):
        replace(_pack(), source_config={"gt_path": "/secret/labels.ply"})


def test_native_scene_pack_roundtrip_verifies_array_hash(tmp_path: Path) -> None:
    pack = _pack()
    output = tmp_path / "pack"

    manifest = write_native_scene_pack(pack, output)
    loaded = load_native_scene_pack(output / "manifest.json")

    assert loaded.identity == pack.identity
    assert loaded.original_owner.tolist() == [7, 0]
    assert manifest["arrays"]["sha256"] == loaded.array_sha256

    arrays_path = output / manifest["arrays"]["path"]
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="array hash"):
        load_native_scene_pack(output / "manifest.json")


def test_ovi_patch_binds_locked_numerical_native_accessors() -> None:
    patch = OVI_PATCH.read_text(encoding="utf-8")

    assert "pybind11::dict exportStudyFrameState();" in patch
    assert "exportStudySurfaceLabels(" in patch
    assert "std::lock_guard<std::mutex> mesh_layer_lock" in patch
    assert "std::lock_guard<std::mutex> label_tsdf_layers_lock" in patch
    assert "mesh_instance_layer_->getAllAllocatedMeshes" in patch
    assert "mesh_instance_layer_->getMeshPtrByIndex" in patch
    assert "label_layer_->getBlockPtrByIndex" in patch
    assert "computeVoxelIndexFromCoordinates" in patch
    assert "native_xyz rows do not match the generated instance mesh" in patch
    assert "getInstanceLabel(segment_label, 0.1f)" in patch
    assert '.def("exportStudyFrameState"' in patch
    assert '.def("exportStudySurfaceLabels"' in patch


def test_native_frame_snapshot_includes_unobserved_registered_membership() -> None:
    patch = OVI_PATCH.read_text(encoding="utf-8")
    assert "for (const auto &label : fusion->label_frames_count_)" in patch
    assert 'state["label_instances_scope"] = "all_known_labels";' in patch


def test_ovi_patch_places_three_capture_hooks_on_active_call_chain() -> None:
    patch = OVI_PATCH.read_text(encoding="utf-8")

    anchors = (
        "study_capture.before_insertion(",
        "gsm_node.insertSegmentsOpen(",
        "gsm_node.integrateFrame()",
        "study_capture.after_integration(",
        "gsm_node.raycastInstancePredictions(",
        "enumerate_admissible_views(",
        "select_views_for_frame(",
        "study_capture.after_raycast(",
        "feature_extractor['worker_stdin'].write((request + \"\\n\").encode())",
        "gsm_node.clearTemporaryMemory()",
    )
    positions = [patch.index(anchor) for anchor in anchors]
    assert positions == sorted(positions)


@pytest.mark.parametrize("wrong_color", [False, True])
def test_native_capture_session_records_all_boundaries_and_aligned_surface(
    tmp_path: Path, wrong_color: bool,
) -> None:
    session = NativeCaptureSession(
        capture_root=tmp_path / "capture",
        scene_id="scene0000_00",
        dataset="scannet_nyu",
        image_size_hw=(3, 4),
        intrinsics=np.eye(3, dtype=np.float32),
        scheduled_frame_ids=range(10, 12),
        source_config={"instance_association": 6},
    )
    segments = [
        SimpleNamespace(
            index=0,
            points=np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            instance_label=np.float32(3),
            class_label=1,
            is_thing=True,
        )
    ]
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.ones((3, 4), dtype=np.float32)
    panoptic = np.zeros((3, 4), dtype=np.int32)
    panoptic[0, 0] = 3
    panoptic[1, 1] = 3
    session.before_insertion(
        frame_id=10,
        rgb_image=rgb,
        depth_m=depth,
        pose_c2w=np.eye(4, dtype=np.float32),
        panoptic_raster=panoptic,
        segments=segments,
        rgb_source_path="source/rgb.png",
        depth_source_path="source/depth.png",
        panoptic_source_path="source/panoptic.png",
    )
    session.after_integration(
        frame_id=10,
        native_state={
            "integrated_frame_count": 1,
            "segments": [
                {
                    "local_index": 0,
                    "input_instance_label": 3,
                    "semantic_label": 1,
                    "registered_label": 7,
                    "point_count": 2,
                }
            ],
            "aliases": [],
            "label_instances": [
                {"segment_label": 7, "instance_label": 11, "semantic_label": 1}
            ],
            "association": {"instance_association": 6},
        },
    )
    owner_mask = np.zeros((3, 4), dtype=bool)
    owner_mask[0, 0] = True
    owner_mask[1, 1] = True
    candidate = {
        "glo_inst_id": 11,
        "glo_inst_mask": owner_mask,
        "glo_inst_area": 2,
        "pano_id": 3,
        "pano_mask": owner_mask.copy(),
        "pano_id_area": 2,
        "overlap_area": 2,
        "overlap_mask": owner_mask.copy(),
        "union_mask": owner_mask.copy(),
        "bbox_xyxy_native": (0, 0, 1, 1),
        "valid_depth_pixels": 2,
        "depth_valid_fraction": 1.0,
        "world_centroid": np.array([0.5, 0.5, 1.0], dtype=np.float32),
    }
    session.after_raycast(
        frame_id=10,
        global_owner_raster=np.where(owner_mask, 11, 0).astype(np.uint16),
        admissible_views=[candidate],
        selected_views=[candidate],
        request_completion_boundary=0,
    )

    mesh_path = tmp_path / "instance_mesh_2.ply"
    mesh_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float normal_x\nproperty float normal_y\nproperty float normal_z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0 0 0 1 10 20 30\n1 0 0 0 0 1 10 20 30\n0 1 0 0 0 1 0 0 0\n"
        "3 0 1 2\n",
        encoding="ascii",
    )

    class FakeGsm:
        def getInstanceColor(self, owner):
            if wrong_color:
                return np.array([255, 255, 255], dtype=np.uint8)
            return np.array([10, 20, 30] if owner == 11 else [0, 0, 0], dtype=np.uint8)

        def exportStudyTsdfState(self):
            return {
                "block_indices": np.array([[1, -2, 3]], dtype=np.int32),
                "distance": np.arange(8, dtype=np.float32) * 0.01,
                "weight": np.arange(8, dtype=np.float32),
                "color": np.tile(np.array([[10, 20, 30, 255]], dtype=np.uint8), (8, 1)),
                "voxel_size": 0.01,
                "voxels_per_side": 2,
            }

        def exportStudySurfaceLabels(self, xyz):
            assert xyz.dtype == np.float32
            assert xyz.shape == (3, 3)
            return {
                "segment_labels": np.array([7, 7, 0], dtype=np.uint16),
                "instance_labels": np.array([11, 11, 0], dtype=np.uint16),
                "semantic_labels": np.array([1, 1, 0], dtype=np.uint16),
                "valid": np.array([1, 1, 0], dtype=np.uint8),
                "label_mapping_count_threshold_factor": np.float32(0.1),
            }

    if wrong_color:
        with pytest.raises(ValueError, match="owner export disagrees"):
            session.finalize(gsm_node=FakeGsm(), instance_mesh_path=mesh_path)
        assert not (session.scene_root / "manifest.json").exists()
        return
    manifest = session.finalize(gsm_node=FakeGsm(), instance_mesh_path=mesh_path)

    assert manifest["completed_frame_ids"] == [10]
    assert manifest["tsdf"]["voxel_count"] == 8
    assert manifest["tsdf"]["block_count"] == 1
    assert manifest["native_owner_mesh_parity"]["exact"] is True
    assert manifest["native_owner_mesh_parity"]["checked_rows"] == 3
    with np.load(tmp_path / "capture/scene0000_00" / manifest["tsdf"]["path"]) as tsdf:
        np.testing.assert_array_equal(tsdf["block_indices"], [[1, -2, 3]])
        np.testing.assert_array_equal(tsdf["weight"], np.arange(8, dtype=np.float32))
        assert tsdf["color"][0].tolist() == [10, 20, 30, 255]
    assert manifest["surface"]["row_count"] == 3
    frame_manifest = manifest["frames"][0]
    assert frame_manifest["refined_segment_ids_array"] == (
        "frame_0000_refined_segment_ids"
    )
    assert frame_manifest["registered_labels_array"] == "frame_0000_registered_labels"
    assert len(frame_manifest["requests"]) == 1
    assert frame_manifest["native_selected_request_ids"] == [
        frame_manifest["requests"][0]["request_id"]
    ]
    with np.load(tmp_path / "capture" / "scene0000_00" / "surface.npz") as arrays:
        assert arrays["segment_labels"].tolist() == [7, 7, 0]
        assert arrays["original_owner"].tolist() == [11, 11, 0]
        assert arrays["surface_faces"].tolist() == [[0, 1, 2]]
        assert arrays["normal_valid"].tolist() == [True, True, True]
        assert arrays[frame_manifest["refined_segment_ids_array"]].tolist() == [1]
        assert arrays[frame_manifest["registered_labels_array"]].tolist() == [7]

    from src.static_ovmap.module_validation.boundary_jobs import verify_capture

    verify_capture(session.scene_root / "manifest.json", allow_skipped=True)
    tsdf_path = session.scene_root / manifest["tsdf"]["path"]
    tsdf_path.write_bytes(tsdf_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_capture(session.scene_root / "manifest.json", allow_skipped=True)
