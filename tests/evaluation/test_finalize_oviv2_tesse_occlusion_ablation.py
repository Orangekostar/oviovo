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
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    evaluate_occlusion_package,
)
from tests.evaluation.test_evaluate_oviv2_tesse_occlusion import (
    _split_schema2_indexes_by_scene as _split_real_indexes,
    _write_checkpoint_index as _write_real_checkpoint_index,
    _write_targets as _write_real_targets,
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


def _parent_freeze(
    root: Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    configs = configs or {
        scene: _config(scene) for scene in ("apartment", "office")
    }
    scenes: dict[str, Any] = {}
    for scene, config in configs.items():
        path = root / f"{scene}.json"
        _write(path, config)
        raw = path.read_bytes()
        scenes[scene] = {
            "rgbd": {"root": f"/data/{scene}", "frame_count": config["frame_count"]},
            "vocabulary": {"json": {"sha256": "4" * 64}},
            "cache": {
                "frontend_manifest": {"sha256": "5" * 64},
                "dense_manifest": {"sha256": "6" * 64},
            },
            "source_config": {
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "byte_count": len(raw),
            },
            "frozen_config": {
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "byte_count": len(raw),
            }
        }
    repository = {
        "commit": "a" * 40,
        "parents": ["b" * 40],
        "tree": "c" * 40,
        "commit_time_utc": "2026-07-22T00:00:00+00:00",
        "clean": True,
        "stage3_lineage_commit": "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5",
        "stage3_is_ancestor": True,
    }
    binding = lambda digit: {
        "path": f"/evidence/{digit}.json",
        "sha256": digit * 64,
        "byte_count": 1,
    }
    manifest = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "status": "FROZEN",
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "repository": repository,
        "algorithm": {
            "sha256": configs["apartment"]["algorithm_hash"],
            "normalized_config": canonical_algorithm_config(configs["apartment"]),
        },
        "selection": {
            **binding("7"),
            "candidate_count": 18,
            "selected_config_sha256": "8" * 64,
            "selected_parameters": {"absence_negative_support": 1.0},
            "selection_rule": ["maximize_current_miou"],
            "candidates": [{} for _ in range(18)],
        },
        "shared_bindings": {
            "input_manifest": binding("1"),
            "source_manifest": binding("2"),
            "schedule": binding("3"),
            "camera": binding("4"),
            "common_target_manifest": binding("5"),
            "common_target_arrays": binding("6"),
            "alias_map": binding("7"),
            "evaluator": binding("8"),
            "finalizers": {
                "common_v2": binding("9"),
                "official_t2": binding("a"),
            },
        },
        "models": {
            "frontend": {scene: {"model_sha256": "b" * 64} for scene in configs},
            "dense": {scene: {"model_sha256": "c" * 64} for scene in configs},
        },
        "scenes": scenes,
        "environment": {"python": "3.12"},
        "commands": {"mapping": []},
        "output_roots": {"apartment": "/runs/apartment", "office": "/runs/office"},
        "office_pre_freeze_audit": {
            "metric_sources_found": [],
            "output_root_was_empty": False,
            "output_root_had_only_preparation": True,
            "scope": {"selection_scene": "apartment"},
        },
        "preparation": {"manifest": binding("d"), "repository": repository},
    }
    path = root / "freeze.json"
    _write(path, manifest)
    return path, configs


def _formal_identity_fields(
    scene_root: Path,
    *,
    scene: str,
    config: dict[str, Any],
    freeze: dict[str, Any],
    freeze_raw: bytes,
) -> dict[str, Any]:
    frozen_identity = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": scene,
        "freeze_manifest": {
            "sha256": _sha256_bytes(freeze_raw),
            "byte_count": len(freeze_raw),
        },
        "repository": {
            "commit": freeze["repository"]["commit"],
            "tree": freeze["repository"]["tree"],
        },
        "config": {
            "sha256": freeze["scenes"][scene]["frozen_config"]["sha256"],
            "byte_count": freeze["scenes"][scene]["frozen_config"]["byte_count"],
        },
        "algorithm_hash": config["algorithm_hash"],
        "missing_observation_policy": "signed_depth",
        "input_bindings_sha256": hashlib.sha256(
            json.dumps(
                {
                    "shared_bindings": freeze["shared_bindings"],
                    "scene": freeze["scenes"][scene],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    status = scene_root.stat()
    execution = {
        "schema_version": 1,
        "run_slot": f"{scene}_run1",
        "output_root": str(scene_root.resolve()),
        "root_device": status.st_dev,
        "root_inode": status.st_ino,
    }
    execution["execution_id"] = hashlib.sha256(
        json.dumps(
            execution,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "frozen_run_identity": frozen_identity,
        "run_execution": execution,
    }


def _bundle(
    root: Path,
    configs: dict[str, dict[str, Any]],
    *,
    formal_freeze: Path | None = None,
) -> tuple[Path, Path]:
    freeze = (
        json.loads(formal_freeze.read_text(encoding="utf-8"))
        if formal_freeze is not None
        else None
    )
    freeze_raw = formal_freeze.read_bytes() if formal_freeze is not None else None
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
        if freeze is not None and freeze_raw is not None:
            payload.update(
                _formal_identity_fields(
                    scene_root,
                    scene=scene,
                    config=configs[scene],
                    freeze=freeze,
                    freeze_raw=freeze_raw,
                )
            )
        index_path = scene_root / "occlusion_checkpoint_index.json"
        _write(index_path, payload)
        paths.append(index_path)
    return paths[0], paths[1]


def _bind_real_indexes(
    indexes: tuple[Path, Path],
    configs: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    for path in indexes:
        index = json.loads(path.read_text(encoding="utf-8"))
        scene = index["scene"]
        config_path = path.parent / index["run_config"]["path"]
        _write(config_path, configs[scene])
        config_raw = config_path.read_bytes()
        index["algorithm_hash"] = configs[scene]["algorithm_hash"]
        index["run_config"] = {
            "path": config_path.name,
            "sha256": _sha256_bytes(config_raw),
            "byte_count": len(config_raw),
        }
        _write(path, index)
    return indexes


def _bind_formal_indexes(
    indexes: tuple[Path, Path],
    *,
    freeze_path: Path,
    configs: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    for path in indexes:
        index = json.loads(path.read_text(encoding="utf-8"))
        scene = index["scene"]
        index.update(
            _formal_identity_fields(
                path.parent,
                scene=scene,
                config=configs[scene],
                freeze=freeze,
                freeze_raw=freeze_raw,
            )
        )
        _write(path, index)
    return indexes


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
    signed_indexes = _bundle(
        tmp_path / "signed",
        signed_configs,
        formal_freeze=parent_path,
    )
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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("frozen_run_identity", "frozen run identity mismatch"),
        ("run_execution", "run execution identity mismatch"),
    ],
)
def test_rejects_signed_bundle_formal_identity_drift(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    index_path = fixture["signed_indexes"][0]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if field == "frozen_run_identity":
        index[field]["algorithm_hash"] = "0" * 64
    else:
        index[field]["root_inode"] += 1
    _write(index_path, index)

    with pytest.raises(ValueError, match=message):
        _finalize(fixture, tmp_path)


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
    "false_releases, recall, supported_outcome",
    [
        (2, 1.0, "false_release_count"),
        (0, 0.5, "retained_ownership_recall"),
    ],
)
def test_each_directional_difference_independently_enables_bounded_claim(
    tmp_path: Path,
    false_releases: int,
    recall: float,
    supported_outcome: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["ablation_result"] = _result(
        fixture["ablation_indexes"],
        policy="missing_as_absence",
        passed=False,
        false_releases=false_releases,
        recall=recall,
    )
    _write(
        fixture["ablation_result_path"],
        fixture["ablation_result"],
        result=True,
    )

    final = _finalize(fixture, tmp_path)

    eligibility = final["claim_eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["allowed_claims"][0]["supported_outcomes"] == [
        supported_outcome
    ]


def test_rejects_reuse_of_signed_bundle_as_ablation_bundle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["ablation_indexes"] = fixture["signed_indexes"]

    with pytest.raises(ValueError, match="does not match its frozen config"):
        _finalize(fixture, tmp_path)


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


def test_rejects_result_numeric_type_drift_from_fresh_evaluator(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fresh_ablation = json.loads(json.dumps(fixture["ablation_result"]))
    fixture["ablation_result"]["stress_layers"]["0.90"][
        "false_release_count"
    ] = 2.0
    _write(
        fixture["ablation_result_path"],
        fixture["ablation_result"],
        result=True,
    )

    def reevaluate(**kwargs: Any) -> dict[str, Any]:
        indexes = tuple(Path(path) for path in kwargs["checkpoint_index"])
        return (
            fixture["signed_result"]
            if indexes == fixture["signed_indexes"]
            else fresh_ablation
        )

    with pytest.raises(ValueError, match="fresh evaluator output"):
        _finalize(fixture, tmp_path, reevaluate=reevaluate)


@pytest.mark.parametrize("binding", ["ablation_config", "bundle_run_config"])
def test_rejects_intermediate_symlink_in_relative_binding(
    tmp_path: Path,
    binding: str,
) -> None:
    fixture = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if binding == "ablation_config":
        source = fixture["ablation_manifest"].parent / "apartment.json"
        (outside / source.name).write_bytes(source.read_bytes())
        (source.parent / "linked").symlink_to(outside, target_is_directory=True)
        manifest = json.loads(fixture["ablation_manifest"].read_text())
        manifest["scenes"]["apartment"]["ablation_config"]["path"] = (
            "linked/apartment.json"
        )
        _write(fixture["ablation_manifest"], manifest)
    else:
        index_path = fixture["signed_indexes"][0]
        index = json.loads(index_path.read_text())
        config_path = index_path.parent / index["run_config"]["path"]
        (outside / config_path.name).write_bytes(config_path.read_bytes())
        (index_path.parent / "linked").symlink_to(outside, target_is_directory=True)
        index["run_config"]["path"] = f"linked/{config_path.name}"
        _write(index_path, index)
        fixture["signed_result"]["checkpoint_index"][0] = {
            "sha256": _sha256_bytes(index_path.read_bytes()),
            "byte_count": index_path.stat().st_size,
        }
        _write(
            fixture["signed_result_path"],
            fixture["signed_result"],
            result=True,
        )

    with pytest.raises(ValueError, match="symlink component"):
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


def test_default_finalizer_consumes_real_evaluator_artifacts(tmp_path: Path) -> None:
    targets, _ = _write_real_targets(tmp_path / "real-target")
    signed_indexes = _split_real_indexes(
        _write_real_checkpoint_index(tmp_path / "real-signed", targets)
    )
    index = json.loads(signed_indexes[0].read_text(encoding="utf-8"))
    target_hash = index["target_manifest"]["sha256"]
    plan_hash = index["evaluation_checkpoint_frames_sha256"]
    signed_configs: dict[str, dict[str, Any]] = {}
    for scene in ("apartment", "office"):
        config = _config(scene)
        config.update(
            frame_count=2,
            occlusion_target_manifest=str(targets / "manifest.json"),
            occlusion_target_manifest_sha256=target_hash,
            evaluation_checkpoint_frames=[0, 1],
            evaluation_checkpoint_frames_sha256=plan_hash,
        )
        config["algorithm_hash"] = canonical_algorithm_hash(config)
        signed_configs[scene] = config
    _bind_real_indexes(signed_indexes, signed_configs)
    parent, _ = _parent_freeze(tmp_path / "real-parent", signed_configs)
    _bind_formal_indexes(
        signed_indexes,
        freeze_path=parent,
        configs=signed_configs,
    )
    ablation_root = tmp_path / "real-ablation-config"
    build_ablation(parent_freeze=parent, output_dir=ablation_root)
    ablation_configs = {
        scene: json.loads((ablation_root / f"{scene}.json").read_text())
        for scene in ("apartment", "office")
    }
    ablation_indexes = _split_real_indexes(
        _write_real_checkpoint_index(tmp_path / "real-ablation", targets)
    )
    _bind_real_indexes(ablation_indexes, ablation_configs)
    signed_result = tmp_path / "real-signed-result.json"
    ablation_result = tmp_path / "real-ablation-result.json"
    evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=signed_indexes,
        dataset_root=targets.parent / "sources",
        output_path=signed_result,
    )
    evaluate_occlusion_package(
        target_dir=targets,
        checkpoint_index=ablation_indexes,
        dataset_root=targets.parent / "sources",
        output_path=ablation_result,
    )

    final = finalize_occlusion_ablation(
        parent_freeze=parent,
        ablation_manifest=ablation_root / "manifest.json",
        targets=targets,
        dataset_root=targets.parent / "sources",
        signed_result=signed_result,
        signed_checkpoints=signed_indexes,
        ablation_result=ablation_result,
        ablation_checkpoints=ablation_indexes,
        output_path=tmp_path / "real-final.json",
    )

    assert final["status"] == "PASS"
    assert final["claim_eligibility"]["eligible"] is False
    assert final["claim_eligibility"]["allowed_claims"] == []
