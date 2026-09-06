from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.rscan_method_views import (
    RScanMethodViewError,
    build_execution_plans,
    build_method_pair_view,
    build_method_pair_view_from_manifest,
    domain_coverage,
)


def _processed(*, offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [0.00 + offset, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 4, 9, 101],
            [0.01 + offset, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 4, 9, 101],
            [1.00 + offset, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 8, 7, 202],
        ],
        dtype=np.float32,
    )


def _pair(**kwargs: object):
    domain_id = kwargs.pop("domain_id", "D0_NATIVE_PROCESSED")
    return build_method_pair_view(
        pair_id="scene0001_00-scene0001_01",
        scan_ids=("scan-a", "scan-b"),
        processed_visits=(_processed(), _processed(offset=0.1)),
        source_manifest_sha256="a" * 64,
        domain_id=domain_id,
        **kwargs,
    )


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_method_tensors_exclude_semantic_and_instance_ground_truth() -> None:
    original = _pair()
    first = _processed()
    second = _processed(offset=0.1)
    first[:, 10:] = [[40, 999], [1, -1], [2, 555]]
    second[:, 10:] = [[3, 4], [5, 6], [7, 8]]

    changed_gt = build_method_pair_view(
        pair_id=original.pair_id,
        scan_ids=("scan-a", "scan-b"),
        processed_visits=(first, second),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    )

    assert changed_gt.method_tensor_sha256() == original.method_tensor_sha256()
    assert all(visit.feature_schema == "xyz_rgb_normal_mesh_segment" for visit in original.visits)
    assert not hasattr(original.visits[0], "semantic_labels")
    assert not hasattr(original.visits[0], "instance_ids")


def test_processed_coordinates_are_declared_pre_aligned_and_not_transformed_again() -> None:
    pair = _pair()

    assert pair.coordinate_frame_id == "3rscan_reference_pre_aligned"
    assert pair.global_alignment_application == "already_materialized_by_preprocessing"
    np.testing.assert_array_equal(pair.visits[1].points_xyz, _processed(offset=0.1)[:, :3])


def test_d1_is_an_explicit_nested_support_view() -> None:
    masks = (
        np.asarray([True, False, True]),
        np.asarray([False, True, True]),
    )
    full = _pair()
    supported = _pair(support_masks=masks, domain_id="D1_NATIVE_SENSOR_SUPPORT")

    assert [visit.point_count for visit in supported.visits] == [2, 2]
    assert supported.parent_method_tensor_sha256 == full.method_tensor_sha256()
    for visit, full_visit in zip(supported.visits, full.visits, strict=True):
        assert set(visit.source_point_indices).issubset(full_visit.source_point_indices)
    assert supported.method_tensor_sha256() != full.method_tensor_sha256()


@pytest.mark.parametrize(
    "masks",
    [
        None,
        (np.asarray([True, True]), np.asarray([True, True, True])),
        (np.asarray([False, False, False]), np.asarray([True, True, True])),
    ],
)
def test_d1_rejects_missing_misaligned_or_empty_support(masks: object) -> None:
    with pytest.raises(RScanMethodViewError, match="support"):
        _pair(domain_id="D1_NATIVE_SENSOR_SUPPORT", support_masks=masks)


def test_candidates_are_gt_free_mesh_segments_shared_by_all_methods() -> None:
    pair = _pair()
    plans = build_execution_plans(pair)

    assert pair.visits[0].candidate_ids == ("segment:000004", "segment:000008")
    assert {plan.method_input_sha256 for plan in plans} == {
        pair.method_tensor_sha256()
    }
    assert {plan.candidate_schema for plan in plans} == {
        "per_visit_mesh_segments"
    }


def test_f_is_independent_and_r_is_joint_with_one_reused_raw_forward() -> None:
    plans = {plan.method_id: plan for plan in build_execution_plans(_pair())}

    assert plans["F"].forward_mode == "independent_single_visit"
    assert len(plans["F"].forward_input_sha256) == 2
    assert plans["F"].forward_input_sha256[0] != plans["F"].forward_input_sha256[1]
    assert plans["R_legacy"].forward_mode == "joint_two_visit"
    assert plans["R_supported"].forward_mode == "joint_two_visit"
    assert plans["R_legacy"].raw_forward_cache_key == plans["R_supported"].raw_forward_cache_key
    assert plans["R_legacy"].raw_forward_cache_key is not None


def test_geometric_sample_preserves_all_candidate_provenance() -> None:
    pair = _pair()

    sample = pair.geometric_sample(neural_voxel_size_m=0.02)

    assert set(zip(sample.visit_ids.tolist(), sample.token_entity_ids, strict=True)) == {
        (0, "segment:000004"),
        (0, "segment:000008"),
        (1, "segment:000004"),
        (1, "segment:000008"),
    }
    assert sample.source_point_count == 6
    assert all(
        semantic.semantic_label is None
        and semantic.semantic_embedding is None
        and semantic.semantic_score == 0.0
        for semantic in sample.entity_semantics
    )


def test_manifest_loader_rechecks_processed_bindings(tmp_path: Path) -> None:
    paths = (tmp_path / "t0.npy", tmp_path / "t1.npy")
    np.save(paths[0], _processed())
    np.save(paths[1], _processed(offset=0.1))
    pair_record = {
        "pair_id": "scene0001_00-scene0001_01",
        "sessions": [
            {
                "visit_index": visit,
                "scan_id": f"scan-{visit}",
                "processed_assets": {"points": _file_record(path)},
            }
            for visit, path in enumerate(paths)
        ],
    }

    observed = build_method_pair_view_from_manifest(
        pair_record=pair_record,
        source_manifest_sha256="b" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    )
    assert [visit.point_count for visit in observed.visits] == [3, 3]

    paths[1].write_bytes(paths[1].read_bytes() + b"tamper")
    with pytest.raises(RScanMethodViewError, match="binding"):
        build_method_pair_view_from_manifest(
            pair_record=pair_record,
            source_manifest_sha256="b" * 64,
            domain_id="D0_NATIVE_PROCESSED",
        )


def test_domain_coverage_marks_missing_d2_instead_of_zero() -> None:
    rows = domain_coverage(
        d0=_pair(),
        d1=None,
        d2=None,
    )

    assert [(row.domain_id, row.status) for row in rows] == [
        ("D0_NATIVE_PROCESSED", "PASS"),
        ("D1_NATIVE_SENSOR_SUPPORT", "MISSING_ASSET"),
        ("D2_OVI_RECONSTRUCTION", "MISSING_ASSET"),
    ]
    assert rows[2].method_tensor_sha256 is None
    assert rows[2].metrics is None
