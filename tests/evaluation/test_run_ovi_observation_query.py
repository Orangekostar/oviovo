from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.run_ovi_observation_query import (
    EVALUATION_DOMAINS,
    LONG_TABLE_FIELDS,
    REPO_ROOT,
    _checkpoint_directory,
    _geometry_metrics,
    _publish,
    _raw_metrics,
    _resolve_inference_intervention,
    _runtime_pair,
    _split_pair,
    _validate_checkpoint_evaluation_contract,
    _voxel_membership,
    unavailable_result_rows,
)
from src.evaluation.rscan_gt_instances import GroundTruthInstance


def test_full_checkpoint_inference_interventions_preserve_loaded_model_identity() -> None:
    assert _resolve_inference_intervention("OBS_FULL", "full", None) == (
        "I_FULL",
        "full",
    )
    assert _resolve_inference_intervention("OBS_FULL", "full", "I_FULL") == (
        "I_FULL",
        "full",
    )
    assert _resolve_inference_intervention("OBS_FULL", "full", "I_BETA0") == (
        "I_BETA0",
        "no_feedback",
    )
    assert _resolve_inference_intervention(
        "OBS_FULL", "full", "I_ALPHA0_BETA0"
    ) == ("I_ALPHA0_BETA0", "base_tuned")


def test_inference_intervention_rejects_non_full_or_unknown_modes() -> None:
    with pytest.raises(ValueError, match="OBS_FULL"):
        _resolve_inference_intervention("OBS_FUSE", "fuse", "I_BETA0")
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_inference_intervention("OBS_FULL", "full", "I_UNKNOWN")


def test_unavailable_rows_keep_exact_long_schema_and_do_not_fabricate_zeroes() -> None:
    rows = unavailable_result_rows(
        run_id="run",
        method_id="OBS_FULL",
        pair_id="pair",
        environment_uuid="environment",
        split_role="DEV",
        previously_inspected=True,
        checkpoint_id=None,
        seed=45,
        reason="missing trained checkpoint because TRAIN RGB-D assets are unavailable",
    )

    assert len(rows) == 10
    assert all(tuple(row) == LONG_TABLE_FIELDS for row in rows)
    assert {row["visit_id"] for row in rows} == {0, 1}
    assert {row["evaluation_domain"] for row in rows} == set(EVALUATION_DOMAINS)
    assert EVALUATION_DOMAINS == (
        "FULL_GT_V2",
        "COMMON_INPUT_SUPPORT_V2",
        "CAMERA_VISIBLE_INTERSECT_D_V1",
        "RAW_FULL_GT_V2",
        "RAW_COMMON_INPUT_SUPPORT_V2",
    )
    assert all(row["status"] == "NOT_RUN_MISSING_TRAIN_ASSETS" for row in rows)
    assert all(row["tp50"] is None for row in rows)
    assert all(row["raw_query_count"] is None for row in rows)
    assert all(row["unavailable_reason"] for row in rows)


def test_camera_visible_d0_points_map_to_dense_d2_by_fixed_voxel_not_row_id() -> None:
    dense = np.asarray([[0.001, 0.0, 0.0], [0.049, 0.0, 0.0], [0.051, 0.0, 0.0]])
    visible_d0 = np.asarray([[0.025, 0.0, 0.0]])

    mapped = _voxel_membership(dense, visible_d0, 0.05)

    assert mapped.tolist() == [True, True, False]


def test_published_artifact_paths_are_relative_to_the_final_run(tmp_path) -> None:
    row = {field: None for field in LONG_TABLE_FIELDS}
    output = tmp_path / "run"

    _publish(
        output,
        rows=[row],
        summary={"source_bindings": {}},
        predictions={"value": np.asarray([1], dtype=np.int64)},
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["artifacts"]["results"]["path"] == "results.csv"
    assert summary["artifacts"]["predictions"]["path"] == "predictions.npz"
    assert set(summary["source_bindings"]) == {
        "dense_adapter",
        "dense_metrics",
        "evaluation_runner",
    }


def test_checked_frozen_result_reports_gpu_forward_time() -> None:
    source = (
        REPO_ROOT
        / "configs/evaluation/results/observation_query/runs"
        / "20260907_scene0109_obs_base_frozen_v2/results.csv"
    )
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))

    assert rows
    assert all(float(row["gpu_seconds"]) > 0.0 for row in rows)
    assert all(row["gpu_seconds"] == row["inference_seconds"] for row in rows)


def test_checked_frozen_result_uses_v2_domains_and_reports_gt_coverage() -> None:
    source = (
        REPO_ROOT
        / "configs/evaluation/results/observation_query_training_v2"
        / "baseline_current_v2"
        / "results.csv"
    )
    assert source.is_file(), "corrected frozen baseline has not been published"
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))

    assert len(rows) == 10
    assert {row["evaluation_domain"] for row in rows} == set(EVALUATION_DOMAINS)
    assert all(int(row["full_gt_target_count"]) > 0 for row in rows)
    assert all(row["domain_target_count"] for row in rows)
    assert all(row["zero_support_gt_count"] for row in rows)
    assert all(row["mean_gt_support_fraction"] for row in rows)
    assert all(json.loads(row["gt_support_fractions_json"]) for row in rows)


def test_checkpoint_directory_accepts_checkpoint_or_training_run_root(tmp_path) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "manifest.json").write_text("{}", encoding="utf-8")
    (direct / "trainable.safetensors").write_bytes(b"weights")
    nested = tmp_path / "run" / "checkpoint"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("{}", encoding="utf-8")
    (nested / "trainable.safetensors").write_bytes(b"weights")

    assert _checkpoint_directory(direct) == direct
    assert _checkpoint_directory(nested.parent) == nested


def test_checkpoint_directory_rejects_incomplete_checkpoint(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")

    try:
        _checkpoint_directory(root)
    except ValueError as error:
        assert "manifest.json and trainable.safetensors" in str(error)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def _record(path):
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _checkpoint_metadata(config, config_path, split_path):
    sources = {
        "config": config_path,
        "split": split_path,
        "model": REPO_ROOT / "src/oviv2/observation_query/model.py",
        "losses": REPO_ROOT / "src/oviv2/observation_query/losses.py",
        "training_state": REPO_ROOT / "src/oviv2/observation_query/training.py",
    }
    return SimpleNamespace(
        model_variant="OBS_FULL",
        resolved_config={
            "method": "OBS_FULL",
            "stage": "smoke",
            "model": config["model"],
            "loss": config["loss"],
            "training": config["training"],
            "method_config": config["methods"]["OBS_FULL"],
            "source_bindings": {name: _record(path) for name, path in sources.items()},
        },
        split_id="SPLIT_V1",
        observation_sha256="a" * 64,
        seed=45,
        model_architecture_version="OVI_OBSERVATION_QUERY_V1",
        input_feature_schema={
            "observation_feature_dim": 1024,
            "observation_metadata_dim": 11,
            "model_input_feature_dim": 9,
        },
    )


def test_checkpoint_evaluation_contract_allows_distinct_eval_inputs(
    tmp_path,
) -> None:
    config = {
        "model": {"hidden_dim": 128},
        "loss": {"native_weight": 1.0},
        "training": {"seed": 45},
        "methods": {"OBS_FULL": {"trained": True}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text('{"artifact_id":"SPLIT_V1"}', encoding="utf-8")
    metadata = _checkpoint_metadata(config, config_path, split_path)

    _validate_checkpoint_evaluation_contract(
        metadata=metadata,
        config=config,
        method="OBS_FULL",
        observation_feature_dim=1024,
        observation_metadata_dim=11,
        model_input_feature_dim=9,
    )

    config_path.write_text("{}", encoding="utf-8")
    config["loss"] = {"native_weight": 0.5}
    config["training"] = {"seed": 999}
    _validate_checkpoint_evaluation_contract(
        metadata=metadata,
        config=config,
        method="OBS_FULL",
        observation_feature_dim=1024,
        observation_metadata_dim=11,
        model_input_feature_dim=9,
    )

    config["model"] = {"hidden_dim": 256}
    with pytest.raises(ValueError, match="model architecture"):
        _validate_checkpoint_evaluation_contract(
            metadata=metadata,
            config=config,
            method="OBS_FULL",
            observation_feature_dim=1024,
            observation_metadata_dim=11,
            model_input_feature_dim=9,
        )


def test_checkpoint_evaluation_contract_rejects_feature_schema_change(tmp_path) -> None:
    config = {
        "model": {"hidden_dim": 128},
        "loss": {"native_weight": 1.0},
        "training": {"seed": 45},
        "methods": {"OBS_FULL": {"trained": True}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text('{"artifact_id":"SPLIT_V1"}', encoding="utf-8")
    metadata = _checkpoint_metadata(config, config_path, split_path)
    with pytest.raises(ValueError, match="input feature schema"):
        _validate_checkpoint_evaluation_contract(
            metadata=metadata,
            config=config,
            method="OBS_FULL",
            observation_feature_dim=768,
            observation_metadata_dim=11,
            model_input_feature_dim=9,
        )


def test_pair_resolution_uses_explicit_role_and_pair_id() -> None:
    split = {
        "environments": [
            {"role": "DEV", "pair_id": "pair-a"},
            {"role": "DEV", "pair_id": "pair-b"},
        ]
    }
    runtime = {
        "pairs": {
            "dev_a": {"role": "DEV", "pair_id": "pair-a"},
            "dev_b": {"role": "DEV", "pair_id": "pair-b"},
        }
    }

    assert _split_pair(split, "DEV", "pair-b")["pair_id"] == "pair-b"
    assert _runtime_pair(runtime, "DEV", "pair-a")["pair_id"] == "pair-a"
    with pytest.raises(ValueError, match="pair is absent or ambiguous"):
        _split_pair(split, "DEV", None)


def test_full_gt_keeps_unreconstructed_voxels_while_common_reports_coverage() -> None:
    pair = SimpleNamespace(
        visits=(
            SimpleNamespace(points_xyz=np.asarray([[0.1, 0.1, 0.1]])),
            SimpleNamespace(points_xyz=np.asarray([[0.1, 0.1, 0.1]])),
        )
    )
    candidate = SimpleNamespace(
        candidate_id="prediction",
        point_indices=np.asarray([0], dtype=np.int64),
    )
    view = SimpleNamespace(candidates=((candidate,), (candidate,)))
    target = GroundTruthInstance(
        instance_id=1,
        semantic_label="chair",
        voxels=frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
    )
    ground_truth = SimpleNamespace(
        voxel_size_m=1.0,
        visits=((target,), (target,)),
    )
    support = (
        np.asarray([True], dtype=np.bool_),
        np.asarray([True], dtype=np.bool_),
    )

    full = _geometry_metrics(
        pair, view, ground_truth, support, target_domain="full"
    )[0]
    common = _geometry_metrics(
        pair, view, ground_truth, support, target_domain="support"
    )[0]

    assert full["primary"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "null_reasons": (),
    }
    assert common["primary"]["tp"] == 1
    assert common["primary"]["f1"] == 1.0
    assert common["coverage"] == {
        "full_gt_target_count": 1,
        "domain_target_count": 1,
        "zero_support_gt_count": 0,
        "mean_gt_support_fraction": 1.0 / 3.0,
        "gt_support_fractions": ({"instance_id": 1, "fraction": 1.0 / 3.0},),
    }


def test_raw_full_and_common_use_distinct_fixed_target_domains() -> None:
    pair = SimpleNamespace(
        visits=(
            SimpleNamespace(points_xyz=np.asarray([[0.1, 0.1, 0.1]])),
            SimpleNamespace(points_xyz=np.asarray([[0.1, 0.1, 0.1]])),
        )
    )
    proposal = SimpleNamespace(point_indices=np.asarray([0], dtype=np.int64))
    readout = SimpleNamespace(raw_proposals=((proposal,), (proposal,)))
    target = GroundTruthInstance(
        instance_id=1,
        semantic_label="chair",
        voxels=frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
    )
    ground_truth = SimpleNamespace(
        voxel_size_m=1.0,
        visits=((target,), (target,)),
    )
    support = (
        np.asarray([True], dtype=np.bool_),
        np.asarray([True], dtype=np.bool_),
    )

    full = _raw_metrics(
        pair, readout, ground_truth, support, target_domain="full"
    )[0]
    common = _raw_metrics(
        pair, readout, ground_truth, support, target_domain="support"
    )[0]

    assert full["raw_best_iou_mean"] == 1.0 / 3.0
    assert full["raw_ar50"] == 0.0
    assert common["raw_best_iou_mean"] == 1.0
    assert common["raw_ar50"] == 1.0
    assert common["coverage"]["gt_support_fractions"] == (
        {"instance_id": 1, "fraction": 1.0 / 3.0},
    )
