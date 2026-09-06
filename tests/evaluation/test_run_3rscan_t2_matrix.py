from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_3rscan_t2_matrix import (
    MatrixRunError,
    NativePairArtifact,
    aggregate_custom_method_rows,
    classify_official_metric_keys,
    compute_method_rows,
    load_native_pair_artifact,
    publish_native_pair_artifact,
)
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    IdentityRules,
    voxelize_points,
)
from src.evaluation.rscan_method_views import build_method_pair_view
from src.oviv2.geometric_pair_reasoner import GeometricReasonerConfig
from src.oviv2.query_instance_projection import ProjectionConfig


def _artifact() -> NativePairArtifact:
    return NativePairArtifact(
        pair_id="scene0001_00-scene0001_01",
        method_input_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        source_commit="c" * 40,
        arrays={
            "pred_masks_nk": np.asarray(
                [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8
            ),
            "pred_scores_k": np.asarray([0.9, 0.8], dtype=np.float32),
            "pred_classes_k": np.asarray([3, 5], dtype=np.int64),
            "target_masks_in": np.asarray(
                [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8
            ),
            "target_labels_i": np.asarray([3, 5], dtype=np.int64),
            "target_changes_i": np.asarray([0, 1], dtype=np.int64),
            "target_ids_i": np.asarray([10, 20], dtype=np.int64),
            "target_temporal_stages_n": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "raw_masks_sq": np.asarray(
                [[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32
            ),
            "raw_logits_qc": np.asarray(
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32
            ),
            "inverse_n": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "lowres_point2segment_m": np.asarray([0, 1], dtype=np.int64),
            "lowres_visit_ids_m": np.asarray([0, 1], dtype=np.int8),
            "full_visit_ids_n": np.asarray([0, 0, 1, 1], dtype=np.int8),
            "full_original_segment_ids_n": np.asarray([4, 4, 8, 8], dtype=np.int64),
            "backbone_features_mf": np.asarray(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
        },
        official_metrics={
            "dev_mean_t-AP": 0.25,
            "dev_mean_t-REC": 0.50,
            "dev_mean_AP": 0.30,
            "dev_mean_stage1-AP": 0.20,
            "dev_mean_stage2-AP": 0.40,
        },
        target_ambiguities=((10, 11),),
        runtime_s=1.25,
        peak_gpu_bytes=4096,
        peak_reserved_gpu_bytes=8192,
        peak_rss_bytes=16384,
        device_name="stub-gpu",
    )


def test_native_artifact_roundtrip_and_raw_forward_arrays_are_single_source(
    tmp_path: Path,
) -> None:
    source = _artifact()

    publish_native_pair_artifact(source, tmp_path / "pair")
    loaded = load_native_pair_artifact(tmp_path / "pair")

    assert loaded.pair_id == source.pair_id
    assert loaded.official_metrics == source.official_metrics
    assert loaded.target_ambiguities == ((10, 11),)
    for name, expected in source.arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], expected)
    assert set(loaded.arrays) == set(source.arrays)


def test_native_artifact_is_no_clobber_and_detects_array_tampering(tmp_path: Path) -> None:
    root = tmp_path / "pair"
    publish_native_pair_artifact(_artifact(), root)

    with pytest.raises(MatrixRunError, match="already exists"):
        publish_native_pair_artifact(_artifact(), root)

    arrays_path = root / "arrays.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")
    with pytest.raises(MatrixRunError, match="binding mismatch"):
        load_native_pair_artifact(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_masks_sq", np.zeros((3, 2), dtype=np.float32), "raw segment"),
        ("inverse_n", np.asarray([0, 0, 2, 1]), "inverse"),
        ("full_visit_ids_n", np.asarray([0, 0, 0, 0]), "both visits"),
        ("target_masks_in", np.zeros((3, 4), dtype=np.uint8), "target instance"),
    ],
)
def test_native_artifact_rejects_cross_domain_shape_mismatch(
    field: str, value: np.ndarray, message: str
) -> None:
    source = _artifact()
    arrays = dict(source.arrays)
    arrays[field] = value

    with pytest.raises(MatrixRunError, match=message):
        NativePairArtifact(
            pair_id=source.pair_id,
            method_input_sha256=source.method_input_sha256,
            checkpoint_sha256=source.checkpoint_sha256,
            source_commit=source.source_commit,
            arrays=arrays,
            official_metrics=source.official_metrics,
            target_ambiguities=source.target_ambiguities,
            runtime_s=source.runtime_s,
            peak_gpu_bytes=source.peak_gpu_bytes,
            peak_reserved_gpu_bytes=source.peak_reserved_gpu_bytes,
            peak_rss_bytes=source.peak_rss_bytes,
            device_name=source.device_name,
        )


def test_official_metric_namespaces_remain_separate_from_custom_metrics() -> None:
    groups = classify_official_metric_keys(
        {
            "dev_mean_t-AP": 0.25,
            "dev_mean_t-REC": 0.50,
            "dev_mean_AP": 0.30,
            "dev_mean_stage1-AP": 0.20,
            "dev_mean_stage2-AP": 0.40,
            "dev_chair_t-AP_50": 0.10,
        }
    )

    assert groups["temporal_ap"] == ("dev_chair_t-AP_50", "dev_mean_t-AP")
    assert groups["temporal_recall"] == ("dev_mean_t-REC",)
    assert groups["legacy_ap"] == ("dev_mean_AP",)
    assert groups["stage_ap"] == ("dev_mean_stage1-AP", "dev_mean_stage2-AP")

    with pytest.raises(MatrixRunError, match="custom"):
        classify_official_metric_keys({"association_precision": 0.5})


def test_manifest_contains_only_relative_artifact_members(tmp_path: Path) -> None:
    root = tmp_path / "pair"
    publish_native_pair_artifact(_artifact(), root)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arrays"]["path"] == "arrays.npz"
    assert Path(manifest["arrays"]["path"]).is_absolute() is False
    assert manifest["raw_forward_reuse"] == ["F", "R_legacy", "R_supported"]


def _method_pair_and_gt():
    def processed(visit_id: int) -> np.ndarray:
        segment_id = 4 if visit_id == 0 else 8
        return np.asarray(
            [
                [0.00, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, segment_id, 1, 10],
                [0.01, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0, 0.0, 0.0, segment_id, 1, 10],
            ],
            dtype=np.float32,
        )

    pair = build_method_pair_view(
        pair_id="scene0001_00-scene0001_01",
        scan_ids=("scan-a", "scan-b"),
        processed_visits=(processed(0), processed(1)),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    )
    visits = tuple(
        (
            GroundTruthInstance(
                10,
                "chair",
                voxelize_points(visit.points_xyz, voxel_size_m=0.05),
            ),
        )
        for visit in pair.visits
    )
    gt = GroundTruthPair(
        pair_id=pair.pair_id,
        voxel_size_m=0.05,
        visits=visits,
        identity_rules=IdentityRules.from_official_records(
            changes={"rigid": [], "nonrigid": [], "removed": []},
            ambiguity=[],
        ),
    )
    return pair, gt


def test_cpu_method_rows_reuse_raw_forward_and_keep_metric_namespaces_separate() -> None:
    pair, gt = _method_pair_and_gt()
    source = _artifact()
    arrays = dict(source.arrays)
    arrays["raw_masks_sq"] = np.asarray(
        [[2.0, -2.0], [2.0, -2.0]], dtype=np.float32
    )
    arrays["full_original_segment_ids_n"] = np.asarray(
        [4, 4, 13, 13], dtype=np.int64
    )
    artifact = NativePairArtifact(
        pair_id=pair.pair_id,
        method_input_sha256=pair.method_tensor_sha256(),
        checkpoint_sha256=source.checkpoint_sha256,
        source_commit=source.source_commit,
        arrays=arrays,
        official_metrics=source.official_metrics,
        target_ambiguities=source.target_ambiguities,
        runtime_s=source.runtime_s,
        peak_gpu_bytes=source.peak_gpu_bytes,
        peak_reserved_gpu_bytes=source.peak_reserved_gpu_bytes,
        peak_rss_bytes=source.peak_rss_bytes,
        device_name=source.device_name,
    )

    result = compute_method_rows(
        pair=pair,
        artifact=artifact,
        ground_truth=gt,
        geometric_config=GeometricReasonerConfig(),
        projection_config=ProjectionConfig(),
        resolver_minimum_candidate_score=0.5,
        minimum_gt_observed_fraction=0.05,
    )

    assert set(result["methods"]) == {
        "G_full",
        "G_supported",
        "F",
        "R_legacy",
        "R_supported",
    }
    assert result["methods"]["G_supported"]["status"] == "MISSING_ASSET"
    assert result["methods"]["R_legacy"]["raw_forward_cache_key"] == result[
        "methods"
    ]["R_supported"]["raw_forward_cache_key"]
    assert result["official_stmetrics"]["scope"] == "native_raw_shared"
    assert "official_stmetrics" not in result["methods"]["G_full"]
    assert result["methods"]["R_supported"]["custom_association"]["status"] == "PASS"

    aggregate = aggregate_custom_method_rows((result,))
    supported = aggregate["methods"]["R_supported"]
    custom = result["methods"]["R_supported"]["custom_association"]
    assert supported["pair_count"] == 1
    assert supported["pooled"]["iou_0_50"]["paired_precision"] == custom[
        "thresholds"
    ]["iou_0_50"]["paired_precision"]
    assert aggregate["methods"]["G_supported"]["status"] == "MISSING_ASSET"
