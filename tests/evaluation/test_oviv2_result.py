from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.evaluation.oviv2_result import (
    REPLICA7_SCENES,
    REPLICA8_SCENES,
    aggregate_oviv2_replica,
    verify_byte_identical_evaluation_dirs,
    validate_scene_run_manifests,
)


def _metrics(value: float) -> dict:
    return {
        "miou": value,
        "macc": value + 0.1,
        "f_miou": value + 0.2,
        "ap25": value + 0.3,
        "ap50": value + 0.4,
        "f5": value + 0.5,
    }


def _run(scene: str, algorithm_hash: str = "mapping", frontend_hash: str = "frontend") -> dict:
    return {
        "method": "OVIV2",
        "scene": scene,
        "final_revision": 200,
        "algorithm_hash": algorithm_hash,
        "frontend_algorithm_hash": frontend_hash,
        "vocabulary_hash": "vocabulary",
        "benchmark_manifest_hash": "benchmark",
        "frame_selection": {"sampled_frame_count": 200},
    }


def test_aggregate_accepts_exact_eight_direct_oviv2_metrics() -> None:
    scene_metrics = {
        scene: _metrics(index / 10.0) for index, scene in enumerate(REPLICA8_SCENES)
    }

    result = aggregate_oviv2_replica(scene_metrics)

    assert result["replica_8_compat"]["scene_ids"] == list(REPLICA8_SCENES)
    assert result["replica_7_heldout"]["scene_ids"] == list(REPLICA7_SCENES)
    assert result["replica_8_compat"]["semantic"]["miou"] == pytest.approx(0.35)
    assert result["replica_7_heldout"]["instance"]["ap50"] == pytest.approx(0.8)


def test_aggregate_rejects_missing_scene_and_nonfinite_metric() -> None:
    scene_metrics = {scene: _metrics(0.1) for scene in REPLICA8_SCENES}
    scene_metrics.pop("office4")
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_oviv2_replica(scene_metrics)

    scene_metrics["office4"] = _metrics(0.1)
    scene_metrics["office4"]["miou"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        aggregate_oviv2_replica(scene_metrics)


def test_run_validation_rejects_mixed_algorithm_hashes() -> None:
    runs = {scene: _run(scene) for scene in REPLICA8_SCENES}
    runs["office4"] = _run("office4", algorithm_hash="changed")

    with pytest.raises(ValueError, match="algorithm hash"):
        validate_scene_run_manifests(runs)


def test_run_validation_requires_200_frame_verified_oviv2_contract() -> None:
    runs = {scene: _run(scene) for scene in REPLICA8_SCENES}
    contract = validate_scene_run_manifests(runs)
    assert contract == {
        "algorithm_hash": "mapping",
        "frontend_algorithm_hash": "frontend",
        "vocabulary_hash": "vocabulary",
        "benchmark_manifest_hash": "benchmark",
    }

    runs["room0"]["final_revision"] = 199
    with pytest.raises(ValueError, match="revision 200"):
        validate_scene_run_manifests(runs)


def test_repeat_evaluation_requires_all_files_to_be_byte_identical(tmp_path: Path) -> None:
    original = tmp_path / "original"
    repeated = tmp_path / "repeated"
    original.mkdir()
    repeated.mkdir()
    names = (
        "metrics.json",
        "per_class_semantic.json",
        "per_class_instance_ap.json",
        "gt_aligned_semantic_ids.npy",
        "gt_aligned_instance_ids.npy",
        "oviv2_instance_mesh.ply",
        "semantic_map_gt.ply",
        "instance_map_gt.ply",
    )
    for name in names:
        (original / name).write_bytes(name.encode())
        (repeated / name).write_bytes(name.encode())

    hashes = verify_byte_identical_evaluation_dirs(original, repeated)
    assert set(hashes) == set(names)

    (repeated / "metrics.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_byte_identical_evaluation_dirs(original, repeated)
