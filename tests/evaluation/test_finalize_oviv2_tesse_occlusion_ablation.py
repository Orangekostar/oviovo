from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.evaluation.build_oviv2_tesse_occlusion_ablation import (
    build_ablation,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.finalize_oviv2_tesse_occlusion_ablation import (
    finalize_occlusion_ablation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _compact_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _result_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: object, *, result: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_result_bytes(value) if result else _compact_bytes(value))


def _config(scene: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": scene,
        "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
        "missing_observation_policy": "signed_depth",
        "visibility_depth_tolerance_m": 0.1,
        "absence_negative_support": 1.0,
        "ownership_min_net_support": 0.5,
        "dataset_root": f"/data/{scene}",
        "frontend_cache_dir": f"/cache/{scene}/frontend",
        "frontend_manifest": f"/cache/{scene}/frontend/manifest.json",
        "dense_cache_dir": f"/cache/{scene}/dense",
        "dense_manifest": f"/cache/{scene}/dense/manifest.json",
        "occlusion_target_manifest": "/targets/manifest.json",
        "occlusion_target_manifest_sha256": "1" * 64,
        "evaluation_checkpoint_frames": [0, 4] if scene == "apartment" else [0, 7],
        "evaluation_checkpoint_frames_sha256": "2" * 64,
        "frame_count": 10 if scene == "apartment" else 12,
    }
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    return config


def _parent_freeze(root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    configs = {scene: _config(scene) for scene in ("apartment", "office")}
    scenes: dict[str, Any] = {}
    for scene, config in configs.items():
        path = root / f"{scene}.json"
        _write(path, config)
        raw = path.read_bytes()
        scenes[scene] = {
            "frozen_config": {
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "byte_count": len(raw),
            }
        }
    manifest = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "status": "FROZEN",
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "repository": {
            "commit": "a" * 40,
            "stage3_is_ancestor": True,
            "clean": True,
        },
        "algorithm": {
            "sha256": configs["apartment"]["algorithm_hash"],
            "normalized_config": canonical_algorithm_config(configs["apartment"]),
        },
        "selection": {"selected_parameters": {"absence_negative_support": 1.0}},
        "shared_bindings": {"schedule": {"sha256": "3" * 64}},
        "models": {"frontend": {}, "dense": {}},
        "scenes": scenes,
    }
    path = root / "freeze.json"
    _write(path, manifest)
    return path, configs


def _bundle(
    root: Path,
    configs: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    paths: list[Path] = []
    for scene in ("apartment", "office"):
        scene_root = root / scene
        config_path = scene_root / "normalized_run_config.json"
        _write(config_path, configs[scene])
        config_raw = config_path.read_bytes()
        payload = {
            "schema_version": 2,
            "manifest_id": "oviv2_tesse_cd_occlusion_checkpoints_v1",
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "algorithm_hash": configs[scene]["algorithm_hash"],
            "run_config": {
                "path": config_path.name,
                "sha256": _sha256_bytes(config_raw),
                "byte_count": len(config_raw),
            },
            "target_manifest": {"sha256": "1" * 64, "byte_count": 123},
            "evaluation_checkpoint_frames_sha256": "2" * 64,
            "snapshots": [],
        }
        index_path = scene_root / "occlusion_checkpoint_index.json"
        _write(index_path, payload)
        paths.append(index_path)
    return paths[0], paths[1]


def _headline(
    policy: str,
    *,
    passed: bool,
    false_releases: int,
    recall: float,
) -> dict[str, Any]:
    return {
        "stress_layer": "0.90",
        "missing_observation_policy": policy,
        "episode_count": 2,
        "anchor_mapped_episode_count": 2,
        "anchor_owned_target_voxels": 4,
        "false_release_count": false_releases,
        "false_reassignment_count": 0,
        "retained_ownership_recall": recall,
        "scene_coverage": {
            scene: {"episode_count": 1, "anchor_mapped_episode_count": 1}
            for scene in ("apartment", "office")
        },
        "passed": passed,
    }


def _result(
    indexes: tuple[Path, Path],
    *,
    policy: str,
    passed: bool,
    false_releases: int,
    recall: float,
) -> dict[str, Any]:
    index_records = []
    config_records = []
    for path in indexes:
        index_raw = path.read_bytes()
        index_records.append(
            {"sha256": _sha256_bytes(index_raw), "byte_count": len(index_raw)}
        )
        index = json.loads(index_raw)
        config_path = path.parent / index["run_config"]["path"]
        config_raw = config_path.read_bytes()
        config_records.append(
            {"sha256": _sha256_bytes(config_raw), "byte_count": len(config_raw)}
        )
    headline = _headline(
        policy,
        passed=passed,
        false_releases=false_releases,
        recall=recall,
    )
    layer = {
        "episode_count": 2,
        "anchor_mapped_episode_count": 2,
        "anchor_owned_target_voxels": 4,
        "false_release_count": false_releases,
        "false_reassignment_count": 0,
        "retained_ownership_recall": recall,
    }
    return {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_evaluation_v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "missing_observation_policy": policy,
        "target_manifest": {"sha256": "1" * 64, "byte_count": 123},
        "checkpoint_count": 4,
        "checkpoint_index": index_records,
        "run_config": config_records,
        "evaluation_checkpoint_frames_sha256": "2" * 64,
        "fixed_anchor_mapping_rule": "majority_owner_then_lowest_entity_id",
        "fixed_anchor_mappings": [],
        "stress_layers": {"0.90": layer},
        "headline_gate": headline,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    parent_path, signed_configs = _parent_freeze(tmp_path / "parent")
    ablation_root = tmp_path / "ablation-config"
    build_ablation(parent_freeze=parent_path, output_dir=ablation_root)
    ablation_configs = {
        scene: json.loads((ablation_root / f"{scene}.json").read_text())
        for scene in ("apartment", "office")
    }
    signed_indexes = _bundle(tmp_path / "signed", signed_configs)
    ablation_indexes = _bundle(tmp_path / "ablation", ablation_configs)
    signed_result = _result(
        signed_indexes,
        policy="signed_depth",
        passed=True,
        false_releases=0,
        recall=1.0,
    )
    ablation_result = _result(
        ablation_indexes,
        policy="missing_as_absence",
        passed=False,
        false_releases=2,
        recall=0.5,
    )
    signed_result_path = tmp_path / "signed-result.json"
    ablation_result_path = tmp_path / "ablation-result.json"
    _write(signed_result_path, signed_result, result=True)
    _write(ablation_result_path, ablation_result, result=True)
    return {
        "parent": parent_path,
        "ablation_manifest": ablation_root / "manifest.json",
        "signed_indexes": signed_indexes,
        "ablation_indexes": ablation_indexes,
        "signed_result": signed_result,
        "ablation_result": ablation_result,
        "signed_result_path": signed_result_path,
        "ablation_result_path": ablation_result_path,
        "targets": tmp_path / "targets",
        "dataset_root": tmp_path / "dataset",
    }


def _reevaluator(fixture: dict[str, Any]):
    def evaluate(**kwargs: Any) -> dict[str, Any]:
        indexes = tuple(Path(path) for path in kwargs["checkpoint_index"])
        if indexes == fixture["signed_indexes"]:
            return fixture["signed_result"]
        if indexes == fixture["ablation_indexes"]:
            return fixture["ablation_result"]
        raise AssertionError(f"unexpected checkpoint bundle: {indexes}")

    return evaluate


def _finalize(
    fixture: dict[str, Any],
    tmp_path: Path,
    *,
    reevaluate: Any | None = None,
) -> dict[str, Any]:
    return finalize_occlusion_ablation(
        parent_freeze=fixture["parent"],
        ablation_manifest=fixture["ablation_manifest"],
        targets=fixture["targets"],
        dataset_root=fixture["dataset_root"],
        signed_result=fixture["signed_result_path"],
        signed_checkpoints=fixture["signed_indexes"],
        ablation_result=fixture["ablation_result_path"],
        ablation_checkpoints=fixture["ablation_indexes"],
        output_path=tmp_path / "final.json",
        reevaluate=reevaluate or _reevaluator(fixture),
    )


def test_finalizes_independent_bundles_and_emits_bounded_eligible_claim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    final = _finalize(fixture, tmp_path)

    assert final["status"] == "PASS"
    assert final["manifest_id"] == "oviv2_tesse_cd_occlusion_ablation_final_v1"
    assert final["target_manifest"] == {"sha256": "1" * 64, "byte_count": 123}
    assert final["evaluation_checkpoint_frames_sha256"] == "2" * 64
    assert final["headline_comparison"]["signed_depth"]["false_release_count"] == 0
    assert final["headline_comparison"]["missing_as_absence"][
        "false_release_count"
    ] == 2
    assert final["claim_eligibility"] == {
        "eligible": True,
        "criteria": {
            "signed_headline_gate_passed": True,
            "ablation_false_release_count_greater": True,
            "ablation_retained_ownership_recall_lower": True,
        },
        "allowed_claims": [
            {
                "claim_id": "tesse_cd_occlusion_retention_vs_missing_as_absence_v1",
                "scope": {
                    "dataset": "TESSE-CD",
                    "scenes": ["apartment", "office"],
                    "stress_layer": "0.90",
                    "headline_policy": "signed_depth",
                    "comparator_policy": "missing_as_absence",
                },
                "supported_outcomes": [
                    "false_release_count",
                    "retained_ownership_recall",
                ],
            }
        ],
    }
    assert set(final["bundles"]) == {"signed_depth", "missing_as_absence"}
    signed_hashes = {
        item["sha256"]
        for item in final["bundles"]["signed_depth"]["checkpoint_indexes"]
    }
    ablation_hashes = {
        item["sha256"]
        for item in final["bundles"]["missing_as_absence"]["checkpoint_indexes"]
    }
    assert signed_hashes.isdisjoint(ablation_hashes)


@pytest.mark.parametrize("case", ["signed_gate", "no_degradation"])
def test_withholds_claim_when_gate_or_directional_comparison_fails(
    tmp_path: Path,
    case: str,
) -> None:
    fixture = _fixture(tmp_path)
    if case == "signed_gate":
        fixture["signed_result"] = _result(
            fixture["signed_indexes"],
            policy="signed_depth",
            passed=False,
            false_releases=1,
            recall=0.75,
        )
        _write(
            fixture["signed_result_path"],
            fixture["signed_result"],
            result=True,
        )
    else:
        fixture["ablation_result"] = _result(
            fixture["ablation_indexes"],
            policy="missing_as_absence",
            passed=False,
            false_releases=0,
            recall=1.0,
        )
        _write(
            fixture["ablation_result_path"],
            fixture["ablation_result"],
            result=True,
        )

    final = _finalize(fixture, tmp_path)

    assert final["claim_eligibility"]["eligible"] is False
    assert final["claim_eligibility"]["allowed_claims"] == []


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("target_manifest", {"sha256": "9" * 64, "byte_count": 123}, "target"),
        ("evaluation_checkpoint_frames_sha256", "9" * 64, "checkpoint plan"),
    ],
)
def test_rejects_target_or_checkpoint_plan_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["ablation_result"][field] = value
    _write(
        fixture["ablation_result_path"],
        fixture["ablation_result"],
        result=True,
    )

    with pytest.raises(ValueError, match=message):
        _finalize(fixture, tmp_path)


def test_rejects_missing_scene_coverage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    del fixture["ablation_result"]["headline_gate"]["scene_coverage"]["office"]
    _write(
        fixture["ablation_result_path"],
        fixture["ablation_result"],
        result=True,
    )

    with pytest.raises(ValueError, match="scene coverage"):
        _finalize(fixture, tmp_path)


def test_rejects_ablation_manifest_not_bound_to_same_parent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["ablation_manifest"].read_text())
    manifest["parent_freeze"]["sha256"] = "9" * 64
    _write(fixture["ablation_manifest"], manifest)

    with pytest.raises(ValueError, match="parent freeze"):
        _finalize(fixture, tmp_path)


def test_direct_cli_help_loads_repository_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/evaluation/finalize_oviv2_tesse_occlusion_ablation.py"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
