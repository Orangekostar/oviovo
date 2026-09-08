from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.evaluation.diagnose_ovi_observation_support import (
    build_camera_visible_support_mask,
    filter_native_sample_by_visibility,
    resolve_diagnostic_pair_runtime,
    strip_native_ground_truth,
)
from scripts.evaluation.prepare_ovi_observations import RealFrameAssets


def _frame(visit_id: int) -> RealFrameAssets:
    depth = np.ones((5, 5), dtype=np.float32)
    return RealFrameAssets(
        visit_id=visit_id,
        scan_uuid=f"scan-{visit_id}",
        target_frame_id=0,
        source_frame_id=7 + visit_id,
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        frontend_labels_color=np.zeros((5, 5), dtype=np.uint8),
        registered_frontend_labels=np.zeros((5, 5), dtype=np.uint8),
        geometric_labels_depth=np.zeros((5, 5), dtype=np.uint8),
        depth_m=depth,
        depth_intrinsic=np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
        ),
        color_to_depth_homography=np.eye(3),
        camera_to_reference=np.eye(4),
        paths={"rgb": Path("unused")},
        raw_regions=(),
    )


def test_diagnostic_resolves_one_explicit_pair_from_multiple_dev_pairs() -> None:
    selected = resolve_diagnostic_pair_runtime(
        {
            "pairs": {
                "dev_a": {
                    "pair_id": "scene0001_00-scene0001_01",
                    "role": "DEV",
                    "status": "READY",
                },
                "dev_b": {
                    "pair_id": "scene0002_00-scene0002_01",
                    "role": "DEV",
                    "status": "READY",
                },
            }
        },
        "scene0002_00-scene0002_01",
    )

    assert selected["pair_id"] == "scene0002_00-scene0002_01"


def test_camera_visible_support_uses_supported_depth_not_occlusion() -> None:
    result = build_camera_visible_support_mask(
        points_reference_xyz=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2], [10.0, 0.0, 1.0]]
        ),
        frames=(_frame(0),),
        visit_id=0,
        tolerance_m=0.03,
        minimum_valid_neighbours=5,
    )

    np.testing.assert_array_equal(result.support_mask, [True, False, False])
    np.testing.assert_array_equal(result.support_frame_counts, [1, 0, 0])
    assert result.relation_counts == {
        "unknown": 1,
        "supported": 1,
        "occluded": 1,
        "visible_free": 0,
    }


def test_native_sample_filter_preserves_all_aligned_arrays_and_visit_order() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 1.0]]
    )
    sample = (
        coordinates.copy(),
        np.arange(18).reshape(3, 6),
        np.arange(12).reshape(3, 4),
        "pair-a",
        np.arange(9).reshape(3, 3),
        np.arange(9, 18).reshape(3, 3),
        coordinates.copy(),
        4,
        None,
    )

    filtered = filter_native_sample_by_visibility(
        sample,
        support_masks=(
            np.asarray([False, True]),
            np.asarray([True]),
        ),
    )

    for index in (0, 1, 2, 4, 5, 6):
        np.testing.assert_array_equal(filtered[index], sample[index][[1, 2]])
    assert filtered[3] == "pair-a"
    assert filtered[7:] == (4, None)


def test_native_forward_sample_removes_gt_but_preserves_mesh_segments() -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
    labels = np.asarray([[3, 17, 1, 41], [4, 22, 0, 77]], dtype=np.int32)
    sample = (
        coordinates.copy(),
        np.zeros((2, 6)),
        labels,
        "pair-a",
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        coordinates.copy(),
        4,
        None,
    )

    stripped = strip_native_ground_truth(sample)

    np.testing.assert_array_equal(
        stripped[2],
        np.asarray([[255, 41, 0, 41], [255, 77, 0, 77]], dtype=np.int32),
    )
    np.testing.assert_array_equal(sample[2], labels)
