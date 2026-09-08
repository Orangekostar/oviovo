from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from scripts.evaluation.prepare_ovi_observations import (
    RealFrameAssets,
    build_observation_bank_from_frames,
    build_observation_source_manifest,
    encode_parent_region_features,
    find_pair_record,
    load_model_observation_bundle,
    load_real_frame_assets,
    mapping_receipt_pair_id,
    pair_alignment_row,
    project_frame_region_support,
    publish_prepared_observation_bank,
    resolve_runtime_pair,
    resolve_siglip_weight_binding,
    save_model_observation_bundle,
    select_region_budget,
    write_region_pixel_sidecar,
)
from src.oviv2.observation_query.observations import DepthRelation, extract_raw_regions
from src.oviv2.rescene_input_bridge import ReSceneModelInput


def _regions(
    *, scan_uuid: str, visit_id: int, frame_id: int, parent_id: int
):
    frontend = np.zeros((4, 6), dtype=np.uint8)
    frontend[1:3, 1:5] = parent_id
    geometric = np.zeros((4, 6), dtype=np.uint8)
    geometric[1:3, 1:3] = 5
    geometric[1:3, 3:5] = 6
    return extract_raw_regions(
        scan_uuid=scan_uuid,
        visit_id=visit_id,
        source_frame_id=frame_id,
        frontend_labels_color=frontend,
        geometric_labels_depth=geometric,
        depth_m=np.ones((4, 6), dtype=np.float32),
        color_to_depth_homography=np.eye(3),
        minimum_valid_depth_pixels=2,
    )


def _model_input() -> ReSceneModelInput:
    centered = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    return ReSceneModelInput(
        coordinates_bxyzt=np.column_stack(
            (np.zeros(2, dtype=np.float32), centered, np.asarray([0.0, 1.0]))
        ),
        grid_coordinates_xyz=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        features=np.column_stack(
            (
                centered,
                np.full((2, 3), 0.5, dtype=np.float32),
                np.tile([0.0, 0.0, 1.0], (2, 1)),
            )
        ),
        sparse_batch_offsets=np.asarray([1, 2], dtype=np.int64),
        point2segment=np.asarray([0, 1], dtype=np.int64),
        adapter_to_model=np.asarray([0, 0, 1], dtype=np.int64),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        representative_source_point_indices=np.asarray([0, 2], dtype=np.int64),
        local_frame_indices=np.asarray([0, 0], dtype=np.int64),
        global_frame_indices=np.asarray([0, 0], dtype=np.int64),
        rows=np.asarray([1, 1], dtype=np.int64),
        columns=np.asarray([1, 1], dtype=np.int64),
        camera_depth_m=np.ones(2, dtype=np.float32),
        observed_depth_m=np.ones(2, dtype=np.float32),
        depth_residual_m=np.zeros(2, dtype=np.float32),
        neural_voxel_size_m=0.02,
        adapter_geometry_sha256="a" * 64,
        native_sampling_sha256="b" * 64,
        surface_attributes_sha256="c" * 64,
        model_candidates_sha256="d" * 64,
        sampler_source_sha256="e" * 64,
    )


def test_region_budget_covers_parents_before_largest_fragments() -> None:
    regions = _regions(scan_uuid="scan-a", visit_id=0, frame_id=0, parent_id=1) + _regions(
        scan_uuid="scan-a", visit_id=0, frame_id=10, parent_id=2
    )

    selected = select_region_budget(
        regions,
        maximum_regions=4,
        maximum_fragments_per_parent=3,
    )

    assert len(selected) == 4
    assert {(item.source_frame_id, item.fragment_id) for item in selected[:2]} == {
        (0, -1),
        (10, -1),
    }
    assert all(item.parent_region_key in {candidate.key for candidate in selected} for item in selected)
    assert sum(item.fragment_id >= 0 for item in selected) == 2


def test_parent_feature_encoding_is_once_per_crop_and_shared_with_fragments() -> None:
    regions = _regions(scan_uuid="scan-a", visit_id=0, frame_id=0, parent_id=1)
    calls: list[tuple[tuple[int, int, int, int], int]] = []

    def load_rgb(_visit_id: int, _frame_id: int) -> np.ndarray:
        return np.full((4, 6, 3), 127, dtype=np.uint8)

    def load_frontend(_visit_id: int, _frame_id: int) -> np.ndarray:
        labels = np.zeros((4, 6), dtype=np.uint8)
        labels[1:3, 1:5] = 1
        return labels

    def encode(_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        calls.append((bbox, int(np.count_nonzero(mask))))
        return np.asarray([3.0, 4.0], dtype=np.float64)

    features = encode_parent_region_features(
        regions,
        load_rgb=load_rgb,
        load_frontend_labels=load_frontend,
        encode_six_crop=encode,
    )

    assert calls == [((1, 1, 5, 3), 8)]
    assert features.shape == (3, 2)
    np.testing.assert_allclose(features, np.tile([0.6, 0.8], (3, 1)))
    assert features.dtype == np.float32
    assert features.flags.writeable is False


def test_parent_feature_encoding_rejects_zero_or_inconsistent_dimensions() -> None:
    first = _regions(scan_uuid="scan-a", visit_id=0, frame_id=0, parent_id=1)[0]
    second = _regions(scan_uuid="scan-a", visit_id=0, frame_id=10, parent_id=2)[0]
    count = 0

    def encode(_rgb: np.ndarray, _mask: np.ndarray, _bbox: tuple[int, int, int, int]) -> np.ndarray:
        nonlocal count
        count += 1
        return np.zeros(2) if count == 1 else np.ones(3)

    with pytest.raises(ValueError, match="nonzero"):
        encode_parent_region_features(
            (first, second),
            load_rgb=lambda _visit, _frame: np.zeros((4, 6, 3), dtype=np.uint8),
            load_frontend_labels=lambda _visit, _frame: np.ones((4, 6), dtype=np.uint8),
            encode_six_crop=encode,
        )


def test_frame_projection_links_only_same_visit_supported_fragment() -> None:
    depth = np.ones((5, 5), dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[1:4, 1:4] = 4
    geometric = np.zeros((5, 5), dtype=np.uint8)
    geometric[1:4, 1:4] = 9
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = project_frame_region_support(
        model_points_reference_xyz=points,
        model_visit_ids=np.asarray([0, 0, 1], dtype=np.int8),
        frame_visit_id=0,
        source_frame_id=12,
        camera_to_reference=np.eye(4),
        depth_intrinsic=np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
        ),
        depth_m=depth,
        registered_frontend_labels=labels,
        geometric_labels_depth=geometric,
        selected_region_indices={(4, -1): 0, (4, 9): 1},
        eligible_fragment_identities={(4, 9)},
        tolerance_m=0.03,
        minimum_valid_neighbours=5,
    )

    assert [(item.region_index, item.model_index, item.is_fragment) for item in result.candidates] == [
        (1, 0, True)
    ]
    assert result.relation_counts == {
        DepthRelation.UNKNOWN.name.lower(): 0,
        DepthRelation.SUPPORTED.name.lower(): 1,
        DepthRelation.OCCLUDED.name.lower(): 1,
        DepthRelation.VISIBLE_FREE.name.lower(): 0,
    }
    np.testing.assert_allclose(result.signed_residuals_by_region[1], [0.0])


def test_model_observation_bundle_round_trips_reference_points(tmp_path) -> None:
    model_input = _model_input()
    center = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)

    paths = save_model_observation_bundle(
        model_input=model_input,
        shared_center_xyz=center,
        pair_content_sha256="f" * 64,
        output_root=tmp_path / "model",
    )
    loaded, points, loaded_center, manifest = load_model_observation_bundle(paths.root)

    assert loaded.content_sha256() == model_input.content_sha256()
    np.testing.assert_allclose(points, model_input.features[:, :3] + center)
    np.testing.assert_array_equal(loaded_center, center)
    assert manifest["pair_content_sha256"] == "f" * 64
    assert points.flags.writeable is False


def test_pair_record_selection_requires_one_exact_pair() -> None:
    selected = {"pair_id": "pair-a", "sessions": []}
    assert find_pair_record({"pairs": [selected]}, "pair-a") is selected

    with pytest.raises(ValueError, match="unique"):
        find_pair_record({"pairs": [selected, dict(selected)]}, "pair-a")
    with pytest.raises(ValueError, match="unique"):
        find_pair_record({"pairs": [selected]}, "pair-b")


def test_d2_mapping_receipt_reads_pair_identity_from_selection() -> None:
    receipt = {
        "status": "REAL_D2_PAIR_PASS",
        "selection": {"pair_id": "pair-a", "role": "D2_DEV"},
        "visits": [{}, {}],
    }

    assert mapping_receipt_pair_id(receipt) == "pair-a"


def test_runtime_pair_resolution_uses_explicit_role_and_pair_identity() -> None:
    runtime = {
        "pairs": {
            "train": {
                "pair_id": "pair-train",
                "role": "TRAIN",
                "status": "READY",
                "scan_ids": ["scan-a", "scan-b"],
            },
            "dev": {
                "pair_id": "pair-dev",
                "role": "DEV",
                "status": "READY",
                "scan_ids": ["scan-c", "scan-d"],
            },
        }
    }

    selected = resolve_runtime_pair(
        runtime,
        pair_id="pair-train",
        role="TRAIN",
    )

    assert selected["scan_ids"] == ["scan-a", "scan-b"]
    with pytest.raises(ValueError, match="role and pair"):
        resolve_runtime_pair(runtime, pair_id="pair-dev", role="TRAIN")


def test_siglip_weight_binding_is_owned_by_observation_preparation(tmp_path) -> None:
    model_root = tmp_path / "siglip"
    model_root.mkdir()
    weights = model_root / "model.safetensors"
    weights.write_bytes(b"real-siglip-weights")
    runtime = {"assets": {"siglip_model": str(model_root)}}

    binding = resolve_siglip_weight_binding(
        runtime_config=runtime,
        mapping_receipt={},
    )

    assert binding["path"] == str(weights)
    assert binding["byte_count"] == len(b"real-siglip-weights")
    assert binding["sha256"] == (
        "9b3fed4bf437a85aae4d5f16cab2eba12535b836f642c8c420d8ee5d96804ab8"
    )


def test_pair_alignment_parser_requires_official_row_vector_convention() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, :3] = [1.0, 2.0, 3.0]
    pair = {
        "common_method_inputs": {
            "global_alignment": {
                "application": "homogeneous_row_vector_right_multiply",
                "direction": "rescan_row_vector_to_reference",
                "storage": "row_major_flat_4x4",
                "matrix": matrix.ravel().tolist(),
            }
        }
    }

    parsed = pair_alignment_row(pair)

    np.testing.assert_array_equal(parsed, matrix)
    assert parsed.flags.writeable is False
    pair["common_method_inputs"]["global_alignment"]["direction"] = "reference_to_rescan"
    with pytest.raises(ValueError, match="alignment convention"):
        pair_alignment_row(pair)


def test_real_frame_loader_uses_source_mapping_depth_scale_and_calibration(tmp_path) -> None:
    materialized = tmp_path / "materialized"
    attempt = tmp_path / "attempt"
    for path in (
        materialized / "color",
        materialized / "depth",
        materialized / "pose",
        attempt / "frontend",
        attempt / "geometric_segments",
    ):
        path.mkdir(parents=True, exist_ok=True)
    rgb = np.full((4, 6, 3), [11, 22, 33], dtype=np.uint8)
    depth = np.full((2, 3), 1250, dtype=np.uint16)
    frontend = np.zeros((4, 6), dtype=np.uint16)
    frontend[:, 2:6] = 300
    geometric = np.zeros((2, 3), dtype=np.uint16)
    geometric[:, 1:3] = 500
    Image.fromarray(rgb).save(materialized / "color/0.jpg")
    Image.fromarray(depth).save(materialized / "depth/0.png")
    Image.fromarray(frontend).save(attempt / "frontend/0.png")
    Image.fromarray(geometric).save(attempt / "geometric_segments/00000_mask.png")
    np.savetxt(materialized / "pose/0.txt", np.eye(4))
    color_intrinsic = np.eye(4)
    color_intrinsic[0, 0] = color_intrinsic[1, 1] = 2.0
    depth_intrinsic = np.eye(4)
    depth_intrinsic[0, 0] = depth_intrinsic[1, 1] = 1.0
    manifest = {
        "status": "MATERIALIZED_INPUT_PASS",
        "scan_id": "scan-b",
        "frame_count": 1,
        "frame_map": [{"target_frame_id": 0, "source_frame_id": 7}],
        "depth_shift": 1000.0,
        "dimensions": {
            "color": {"height": 4, "width": 6},
            "depth": {"height": 2, "width": 3},
        },
        "calibration": {
            "color_intrinsic": color_intrinsic.tolist(),
            "depth_intrinsic": depth_intrinsic.tolist(),
            "color_extrinsic": np.eye(4).tolist(),
            "depth_extrinsic": np.eye(4).tolist(),
        },
    }
    alignment = np.eye(4)
    alignment[3, 0] = 4.0

    frame = load_real_frame_assets(
        visit_id=1,
        scan_uuid="scan-b",
        target_frame_id=0,
        materialized_root=materialized,
        native_attempt_root=attempt,
        materialized_manifest=manifest,
        rescan_to_reference_row=alignment,
        minimum_valid_depth_pixels=2,
    )

    assert frame.source_frame_id == 7
    np.testing.assert_allclose(frame.depth_m, 1.25)
    np.testing.assert_allclose(frame.color_to_depth_homography, np.diag([0.5, 0.5, 1.0]))
    np.testing.assert_allclose(frame.camera_to_reference[:3, 3], [4.0, 0.0, 0.0])
    assert set(np.unique(frame.registered_frontend_labels)) == {0, 300}
    assert {(region.frontend_instance_id, region.fragment_id) for region in frame.raw_regions} == {
        (300, -1),
        (300, 500),
    }


def test_frame_set_builds_two_visit_bank_without_cross_visit_edges(tmp_path) -> None:
    frames = []
    for visit_id in (0, 1):
        frontend = np.zeros((4, 6), dtype=np.uint8)
        frontend[1:3, 1:5] = visit_id + 1
        geometric = np.zeros((4, 6), dtype=np.uint8)
        geometric[1:3, 1:3] = 5
        geometric[1:3, 3:5] = 6
        regions = _regions(
            scan_uuid=f"scan-{visit_id}",
            visit_id=visit_id,
            frame_id=visit_id + 7,
            parent_id=visit_id + 1,
        )
        frames.append(
            RealFrameAssets(
                visit_id=visit_id,
                scan_uuid=f"scan-{visit_id}",
                target_frame_id=0,
                source_frame_id=visit_id + 7,
                rgb=np.full((4, 6, 3), 127, dtype=np.uint8),
                frontend_labels_color=frontend,
                registered_frontend_labels=frontend,
                geometric_labels_depth=geometric,
                depth_m=np.ones((4, 6), dtype=np.float32),
                depth_intrinsic=np.asarray(
                    [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
                ),
                color_to_depth_homography=np.eye(3),
                camera_to_reference=np.eye(4),
                paths={"rgb": tmp_path / f"rgb-{visit_id}.png"},
                raw_regions=regions,
            )
        )
    points = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    prepared = build_observation_bank_from_frames(
        pair_id="pair-a",
        model_input_sha256="a" * 64,
        model_points_reference_xyz=points,
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        frames=frames,
        encode_six_crop=lambda _rgb, _mask, _bbox: np.asarray([3.0, 4.0]),
        source_manifest={"source": "synthetic-no-gt"},
        maximum_regions_per_visit=3,
        maximum_fragments_per_parent=2,
        maximum_distinct_frames_per_model_point=8,
        tolerance_m=0.03,
    )

    assert len(prepared.regions) == 6
    assert prepared.bank.region_features.shape == (6, 2)
    assert len(prepared.bank.csr_model_indices) == 2
    np.testing.assert_array_equal(
        prepared.bank.model_visit_ids[prepared.bank.csr_model_indices],
        prepared.bank.region_visit_ids[
            np.repeat(np.arange(6), np.diff(prepared.bank.csr_indptr))
        ],
    )
    assert prepared.relation_counts == {
        "unknown": 0,
        "supported": 2,
        "occluded": 0,
        "visible_free": 0,
    }

    paths = publish_prepared_observation_bank(
        prepared=prepared,
        frames=frames,
        output_root=tmp_path / "published",
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    diagnostics = json.loads(paths.diagnostics.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["observation_bank"]["content_sha256"] == prepared.bank.content_sha256()
    assert diagnostics["region_count"] == 6
    assert diagnostics["edge_count"] == 2
    assert diagnostics["model_points_with_support"] == 2
    assert len(paths.overlays) == 1
    with Image.open(paths.overlays[0]) as overlay:
        assert overlay.size == (18, 4)


def test_observation_source_manifest_excludes_evaluator_only_fields(tmp_path) -> None:
    bound_paths = []
    for name in (
        "selection.json",
        "mapping.json",
        "model.json",
        "materialized-0.json",
        "materialized-1.json",
        "native-0.json",
        "native-1.json",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        bound_paths.append(path)
    frames = []
    for visit_id in (0, 1):
        paths = {}
        for name in ("rgb", "depth", "pose", "frontend", "geometric"):
            path = tmp_path / f"{visit_id}-{name}.bin"
            path.write_bytes(f"{visit_id}-{name}".encode())
            paths[name] = path
        frames.append(
            RealFrameAssets(
                visit_id=visit_id,
                scan_uuid=f"scan-{visit_id}",
                target_frame_id=visit_id,
                source_frame_id=visit_id + 10,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                frontend_labels_color=np.zeros((2, 2), dtype=np.uint8),
                registered_frontend_labels=np.zeros((2, 2), dtype=np.uint8),
                geometric_labels_depth=np.zeros((2, 2), dtype=np.uint8),
                depth_m=np.ones((2, 2), dtype=np.float32),
                depth_intrinsic=np.eye(3),
                color_to_depth_homography=np.eye(3),
                camera_to_reference=np.eye(4),
                paths=paths,
                raw_regions=(),
            )
        )
    pair = {
        "common_method_inputs": {
            "global_alignment": {
                "application": "homogeneous_row_vector_right_multiply",
                "direction": "rescan_row_vector_to_reference",
                "storage": "row_major_flat_4x4",
                "matrix": np.eye(4).ravel().tolist(),
            }
        },
        "evaluator_only": {"secret_gt_identity": [1, 2, 3]},
    }

    manifest = build_observation_source_manifest(
        pair_id="pair-a",
        pair_record=pair,
        selection_manifest_path=bound_paths[0],
        mapping_receipt_path=bound_paths[1],
        model_bundle_manifest_path=bound_paths[2],
        materialized_manifest_paths=bound_paths[3:5],
        native_mapping_manifest_paths=bound_paths[5:7],
        frames=frames,
        feature_encoder_manifest={"name": "fake-six-crop"},
        observation_config={"max_frames_per_visit": 16},
    )

    encoded = json.dumps(manifest, sort_keys=True)
    assert "evaluator_only" not in encoded
    assert "secret_gt_identity" not in encoded
    assert manifest["visits"][1]["selected_frames"] == [
        {"target_frame_id": 1, "source_frame_id": 11}
    ]
    assert manifest["region_construction"] == "raw_region_plus_depth_fragments"


def test_region_pixel_sidecar_preserves_variable_length_masks(tmp_path) -> None:
    regions = _regions(scan_uuid="scan-a", visit_id=0, frame_id=0, parent_id=1)

    path = write_region_pixel_sidecar(regions, tmp_path / "pixels.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "mask_indptr",
            "mask_pixel_indices",
            "support_indptr",
            "support_pixel_indices",
            "parent_region_indices",
            "fragment_ids",
            "bbox_xyxy",
            "parent_color_bbox_xyxy",
            "depth_shapes",
        }
        np.testing.assert_array_equal(archive["mask_indptr"], [0, 8, 12, 16])
        np.testing.assert_array_equal(archive["support_indptr"], [0, 8, 12, 16])
        np.testing.assert_array_equal(archive["parent_region_indices"], [0, 0, 0])
        np.testing.assert_array_equal(archive["fragment_ids"], [-1, 5, 6])
