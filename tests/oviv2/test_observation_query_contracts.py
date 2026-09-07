from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.oviv2.observation_query.contracts import (
    OBSERVATION_METADATA_COLUMNS,
    ObservationBank,
    ObservationBankError,
    load_observation_bank,
    save_observation_bank,
)
from src.oviv2.observation_query.observations import (
    DepthRelation,
    RegionSupportCandidate,
    build_region_metadata,
    build_region_model_incidence,
    camera_to_reference,
    classify_depth_relations,
    extract_raw_regions,
    neighborhood_depth_reliability,
    prepare_observation_bank,
    register_labels_nearest,
    select_observation_frames,
    stable_region_key,
)


def _metadata(region_count: int) -> np.ndarray:
    return np.zeros((region_count, len(OBSERVATION_METADATA_COLUMNS)), dtype=np.float32)


def _bank(**changes: object) -> ObservationBank:
    values: dict[str, object] = {
        "pair_id": "scene0000_00-scene0000_01",
        "model_input_sha256": "a" * 64,
        "region_keys": (
            stable_region_key("scan-a", 0, 10, 3, -1),
            stable_region_key("scan-b", 1, 20, 7, 2),
        ),
        "region_visit_ids": np.asarray([0, 1], dtype=np.int8),
        "region_frame_ids": np.asarray([10, 20], dtype=np.int64),
        "region_features": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "region_metadata": _metadata(2),
        "csr_indptr": np.asarray([0, 1, 2], dtype=np.int64),
        "csr_model_indices": np.asarray([0, 2], dtype=np.int64),
        "csr_weights": np.asarray([2.0, 1.0], dtype=np.float32),
        "region_reliability": np.asarray([0.8, 0.6], dtype=np.float32),
        "model_visit_ids": np.asarray([0, 0, 1], dtype=np.int8),
        "source_manifest": {"coordinate_frame": "shared_reference"},
    }
    values.update(changes)
    return ObservationBank(**values)


def test_observation_bank_validates_nonidentity_domains_and_round_trips(tmp_path: Path) -> None:
    bank = _bank()

    paths = save_observation_bank(bank, tmp_path / "bank")
    loaded = load_observation_bank(paths.root)

    assert loaded.content_sha256() == bank.content_sha256()
    assert loaded.region_keys == bank.region_keys
    assert loaded.source_manifest == bank.source_manifest
    for name in (
        "region_visit_ids",
        "region_frame_ids",
        "region_features",
        "region_metadata",
        "csr_indptr",
        "csr_model_indices",
        "csr_weights",
        "region_reliability",
        "model_visit_ids",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(bank, name))
        assert getattr(loaded, name).flags.writeable is False


def test_observation_bank_rejects_cross_visit_csr_edges() -> None:
    with pytest.raises(ObservationBankError, match="crosses visits"):
        _bank(csr_model_indices=np.asarray([2, 2], dtype=np.int64))


def test_empty_observation_bank_is_a_valid_pure_3d_fallback() -> None:
    bank = prepare_observation_bank(
        pair_id="pair",
        model_input_sha256="b" * 64,
        region_keys=(),
        region_visit_ids=np.empty(0, dtype=np.int8),
        region_frame_ids=np.empty(0, dtype=np.int64),
        region_features=np.empty((0, 8), dtype=np.float32),
        region_metadata=_metadata(0),
        region_reliability=np.empty(0, dtype=np.float32),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        support_candidates=(),
        source_manifest={"filtered_regions": 4},
    )

    assert bank.region_features.shape == (0, 8)
    np.testing.assert_array_equal(bank.csr_indptr, [0])
    assert bank.csr_model_indices.size == bank.csr_weights.size == 0
    assert np.all(np.isfinite(bank.region_features))


def test_depth_relations_follow_signed_residual_and_reliability() -> None:
    observed = np.asarray([1.00, 0.94, 1.06, 1.03, 0.0, 1.0], dtype=np.float64)
    projected = np.ones(6, dtype=np.float64)
    reliable = np.asarray([True, True, True, True, True, False])

    relations = classify_depth_relations(
        observed_depth_m=observed,
        projected_depth_m=projected,
        neighborhood_reliable=reliable,
        tolerance_m=0.03,
    )

    np.testing.assert_array_equal(
        relations,
        np.asarray(
            [
                DepthRelation.SUPPORTED,
                DepthRelation.OCCLUDED,
                DepthRelation.VISIBLE_FREE,
                DepthRelation.SUPPORTED,
                DepthRelation.UNKNOWN,
                DepthRelation.UNKNOWN,
            ],
            dtype=np.int8,
        ),
    )


def test_region_model_incidence_deduplicates_pixels_and_prefers_fragments() -> None:
    candidates = (
        RegionSupportCandidate(0, 0, 0, 10, 55, 1.0, False),
        RegionSupportCandidate(0, 0, 0, 10, 55, 1.0, False),
        RegionSupportCandidate(1, 0, 0, 10, 55, 1.0, True),
        RegionSupportCandidate(1, 0, 0, 10, 56, 2.0, True),
        RegionSupportCandidate(2, 1, 1, 20, 88, 3.0, True),
    )

    indptr, indices, weights = build_region_model_incidence(
        region_visit_ids=np.asarray([0, 0, 1], dtype=np.int8),
        model_visit_ids=np.asarray([0, 1], dtype=np.int8),
        support_candidates=candidates,
        maximum_distinct_frames_per_model_point=8,
    )

    np.testing.assert_array_equal(indptr, [0, 0, 1, 2])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(weights, [3.0, 3.0])


def test_region_boundary_conflict_splits_one_physical_pixel_without_double_vote() -> None:
    candidates = (
        RegionSupportCandidate(0, 0, 0, 10, 55, 1.0, True),
        RegionSupportCandidate(1, 0, 0, 10, 55, 1.0, True),
    )

    indptr, indices, weights = build_region_model_incidence(
        region_visit_ids=np.asarray([0, 0], dtype=np.int8),
        model_visit_ids=np.asarray([0], dtype=np.int8),
        support_candidates=candidates,
    )

    np.testing.assert_array_equal(indptr, [0, 1, 2])
    np.testing.assert_array_equal(indices, [0, 0])
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert weights.sum() == pytest.approx(1.0)


def test_region_model_incidence_caps_distinct_frames_per_model_point() -> None:
    candidates = tuple(
        RegionSupportCandidate(frame, 0, 0, frame, frame, float(frame + 1), True)
        for frame in range(3)
    )

    indptr, indices, weights = build_region_model_incidence(
        region_visit_ids=np.zeros(3, dtype=np.int8),
        model_visit_ids=np.asarray([0], dtype=np.int8),
        support_candidates=candidates,
        maximum_distinct_frames_per_model_point=2,
    )

    np.testing.assert_array_equal(indptr, [0, 0, 1, 2])
    np.testing.assert_array_equal(indices, [0, 0])
    np.testing.assert_allclose(weights, [2.0, 3.0])


def test_nearest_registration_preserves_label_ids() -> None:
    labels = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint16)
    color_to_depth = np.asarray(
        [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    registered = register_labels_nearest(labels, color_to_depth, (2, 2))

    assert registered.dtype == labels.dtype
    np.testing.assert_array_equal(registered, [[1, 3], [5, 7]])
    assert set(np.unique(registered)).issubset(set(np.unique(labels)))


def test_raw_regions_keep_parent_and_depth_fragments_as_distinct_tokens() -> None:
    frontend = np.asarray(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 2, 2],
        ],
        dtype=np.uint16,
    )
    geometric = np.asarray(
        [
            [0, 5, 5, 0],
            [0, 6, 6, 0],
            [7, 7, 8, 8],
        ],
        dtype=np.uint16,
    )
    depth = np.ones((3, 4), dtype=np.float32)
    depth[2, 3] = 0.0

    regions = extract_raw_regions(
        scan_uuid="scan-a",
        visit_id=0,
        source_frame_id=12,
        frontend_labels_color=frontend,
        geometric_labels_depth=geometric,
        depth_m=depth,
        color_to_depth_homography=np.eye(3),
        minimum_valid_depth_pixels=2,
    )

    assert [
        (item.frontend_instance_id, item.fragment_id) for item in regions
    ] == [(1, -1), (1, 5), (1, 6), (2, -1), (2, 7)]
    parent = regions[0]
    assert parent.key == parent.parent_region_key
    assert regions[1].parent_region_key == parent.key
    assert parent.bbox_xyxy == (1, 0, 3, 2)
    np.testing.assert_array_equal(parent.mask_pixel_indices, [1, 2, 5, 6])
    np.testing.assert_array_equal(parent.support_pixel_indices, [1, 2, 5, 6])
    assert parent.mask_pixel_indices.flags.writeable is False
    assert regions[-1].valid_depth_fraction == pytest.approx(1.0)


def test_frame_selection_is_bounded_uniform_and_keeps_endpoints() -> None:
    selected = select_observation_frames(frame_count=63, maximum_frames=16)

    assert len(selected) == 16
    assert selected[0] == 0
    assert selected[-1] == 62
    assert tuple(sorted(set(selected))) == selected
    assert max(np.diff(selected)) - min(np.diff(selected)) <= 1

    assert select_observation_frames(frame_count=3, maximum_frames=16) == (0, 1, 2)


def test_rescan_camera_pose_uses_transposed_row_alignment_once() -> None:
    camera_to_local = np.eye(4, dtype=np.float64)
    camera_to_local[:3, 3] = [1.0, 2.0, 3.0]
    row_alignment = np.eye(4, dtype=np.float64)
    row_alignment[3, :3] = [10.0, 20.0, 30.0]

    reference = camera_to_reference(camera_to_local, 0, row_alignment)
    rescan = camera_to_reference(camera_to_local, 1, row_alignment)

    np.testing.assert_array_equal(reference, camera_to_local)
    np.testing.assert_allclose(rescan[:3, 3], [11.0, 22.0, 33.0])
    point_local_row = np.asarray([1.0, 2.0, 3.0, 1.0])
    point_camera_column = camera_to_local @ np.asarray([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        point_local_row @ row_alignment,
        (row_alignment.T @ point_camera_column),
    )


def test_depth_neighborhood_rejects_boundaries_sparse_and_depth_edges() -> None:
    depth = np.ones((5, 5), dtype=np.float32)
    depth[2, 3] = 1.2
    rows = np.asarray([2, 0, 4, 2], dtype=np.int64)
    columns = np.asarray([2, 2, 2, 4], dtype=np.int64)

    reliable = neighborhood_depth_reliability(
        depth, rows, columns, tolerance_m=0.03, minimum_valid_neighbours=5
    )

    np.testing.assert_array_equal(reliable, [False, False, False, False])

    depth[2, 3] = 1.0
    assert neighborhood_depth_reliability(
        depth,
        np.asarray([2]),
        np.asarray([2]),
        tolerance_m=0.03,
        minimum_valid_neighbours=5,
    ).tolist() == [True]


def test_region_metadata_uses_reference_geometry_without_gt_fields() -> None:
    frontend = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    depth = np.asarray([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    (region,) = extract_raw_regions(
        scan_uuid="scan-a",
        visit_id=0,
        source_frame_id=0,
        frontend_labels_color=frontend,
        geometric_labels_depth=np.zeros((2, 2), dtype=np.uint8),
        depth_m=depth,
        color_to_depth_homography=np.eye(3),
        minimum_valid_depth_pixels=2,
    )
    intrinsic = np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 0.0, 0.0]

    metadata, reliability = build_region_metadata(
        region,
        depth_m=depth,
        depth_intrinsic=intrinsic,
        camera_to_reference=pose,
        signed_depth_residuals_m=np.asarray([0.01, -0.01]),
    )

    assert metadata.shape == (len(OBSERVATION_METADATA_COLUMNS),)
    np.testing.assert_allclose(metadata[:3], [10.5, 0.0, 2.0])
    np.testing.assert_allclose(metadata[3:6], [1.0, 0.0, 0.0])
    assert metadata[6] == pytest.approx(1.0)
    assert metadata[12] == pytest.approx(0.0)
    assert metadata[14] == pytest.approx(2.0)
    assert 0.0 <= reliability <= 1.0
    assert np.all(np.isfinite(metadata))
