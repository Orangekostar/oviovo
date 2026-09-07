from __future__ import annotations

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.run_ovi_observation_query import (
    LONG_TABLE_FIELDS,
    REPO_ROOT,
    _checkpoint_directory,
    _publish,
    _validate_checkpoint_evaluation_contract,
    _voxel_membership,
    unavailable_result_rows,
)


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

    assert len(rows) == 8
    assert all(tuple(row) == LONG_TABLE_FIELDS for row in rows)
    assert {row["visit_id"] for row in rows} == {0, 1}
    assert {row["evaluation_domain"] for row in rows} == {
        "FULL_GT_LEGACY",
        "COMMON_M_SUPPORTED",
        "CAMERA_VISIBLE_DIAGNOSTIC",
        "RAW_QUERY_DIAGNOSTIC",
    }
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
    )


def test_checkpoint_evaluation_contract_accepts_only_matching_training_inputs(
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
        config_path=config_path,
        split_path=split_path,
        split_id="SPLIT_V1",
        observation_sha256="a" * 64,
    )

    config["model"] = {"hidden_dim": 256}
    with pytest.raises(ValueError, match="resolved training config"):
        _validate_checkpoint_evaluation_contract(
            metadata=metadata,
            config=config,
            method="OBS_FULL",
            config_path=config_path,
            split_path=split_path,
            split_id="SPLIT_V1",
            observation_sha256="a" * 64,
        )


def test_checkpoint_evaluation_contract_rejects_tampered_bound_source(tmp_path) -> None:
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
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="source binding differs"):
        _validate_checkpoint_evaluation_contract(
            metadata=metadata,
            config=config,
            method="OBS_FULL",
            config_path=config_path,
            split_path=split_path,
            split_id="SPLIT_V1",
            observation_sha256="a" * 64,
        )
