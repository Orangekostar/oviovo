from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.build_oviv2_tesse_occlusion_ablation import (
    build_ablation,
    main,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (
    canonical_algorithm_config,
    canonical_algorithm_hash,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _parent_config(scene: str) -> dict[str, Any]:
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
        "input_manifest": "/data/input.json",
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


def _write_parent_freeze(root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    configs = {scene: _parent_config(scene) for scene in ("apartment", "office")}
    scene_records: dict[str, Any] = {}
    for scene, config in configs.items():
        path = root / f"{scene}.json"
        _write_json(path, config)
        raw = path.read_bytes()
        scene_records[scene] = {
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
        "scenes": scene_records,
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
    _write_json(path, manifest)
    return path, configs


def test_builds_parent_bound_ablation_and_changes_only_policy_and_hash(
    tmp_path: Path,
) -> None:
    parent_path, parents = _write_parent_freeze(tmp_path / "parent")
    output = tmp_path / "ablation"

    manifest = build_ablation(parent_freeze=parent_path, output_dir=output)

    assert manifest["status"] == "FROZEN"
    assert manifest["manifest_id"] == "oviv2_tesse_cd_occlusion_ablation_v1"
    assert manifest["parent_freeze"] == {
        "sha256": _sha256_bytes(parent_path.read_bytes()),
        "byte_count": parent_path.stat().st_size,
        "freeze_id": "oviv2-tessecd-v1",
        "repository_commit": "a" * 40,
        "algorithm_hash": parents["apartment"]["algorithm_hash"],
    }
    assert manifest["mutation"] == {
        "only_changed_fields": ["algorithm_hash", "missing_observation_policy"],
        "missing_observation_policy": {
            "from": "signed_depth",
            "to": "missing_as_absence",
        },
    }

    hashes = set()
    for scene, parent in parents.items():
        ablation_path = output / f"{scene}.json"
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        changed = {
            key
            for key in parent
            if parent.get(key) != ablation.get(key)
        }
        assert set(ablation) == set(parent)
        assert changed == {"algorithm_hash", "missing_observation_policy"}
        assert ablation["missing_observation_policy"] == "missing_as_absence"
        assert ablation["algorithm_hash"] == canonical_algorithm_hash(ablation)
        hashes.add(ablation["algorithm_hash"])
        record = manifest["scenes"][scene]["ablation_config"]
        assert record == {
            "path": f"{scene}.json",
            "path_base": "manifest",
            "sha256": _sha256_bytes(ablation_path.read_bytes()),
            "byte_count": ablation_path.stat().st_size,
        }
    assert len(hashes) == 1
    assert manifest["algorithm"]["parent_sha256"] == parents["apartment"][
        "algorithm_hash"
    ]
    assert manifest["algorithm"]["ablation_sha256"] == next(iter(hashes))
    assert (output / "manifest.json").read_bytes() == _canonical_bytes(manifest)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda manifest, configs: manifest.__setitem__("status", "PREPARED"), "FROZEN"),
        (
            lambda manifest, configs: configs["apartment"].__setitem__(
                "missing_observation_policy", "missing_as_absence"
            ),
            "signed_depth",
        ),
        (
            lambda manifest, configs: configs["office"].__setitem__(
                "algorithm_hash", "0" * 64
            ),
            "algorithm_hash",
        ),
        (
            lambda manifest, configs: manifest["repository"].__setitem__(
                "clean", False
            ),
            "repository",
        ),
        (
            lambda manifest, configs: manifest["repository"].__setitem__(
                "commit", "g" * 40
            ),
            "repository",
        ),
        (
            lambda manifest, configs: manifest.pop("selection"),
            "parent freeze fields",
        ),
        (
            lambda manifest, configs: manifest.pop("shared_bindings"),
            "parent freeze fields",
        ),
        (
            lambda manifest, configs: manifest.pop("models"),
            "parent freeze fields",
        ),
        (
            lambda manifest, configs: manifest.pop("preparation"),
            "parent freeze fields",
        ),
        (
            lambda manifest, configs: manifest["scenes"]["office"].pop("cache"),
            "scene record",
        ),
    ],
)
def test_rejects_non_frozen_or_unbound_parent_inputs(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    parent_path, configs = _write_parent_freeze(tmp_path / "parent")
    manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    mutator(manifest, configs)
    for scene, config in configs.items():
        path = Path(manifest["scenes"][scene]["frozen_config"]["path"])
        _write_json(path, config)
        manifest["scenes"][scene]["frozen_config"].update(
            sha256=_sha256_bytes(path.read_bytes()),
            byte_count=path.stat().st_size,
        )
    _write_json(parent_path, manifest)

    with pytest.raises(ValueError, match=message):
        build_ablation(
            parent_freeze=parent_path,
            output_dir=tmp_path / "ablation",
        )


def test_cli_refuses_existing_output_directory(tmp_path: Path) -> None:
    parent_path, _ = _write_parent_freeze(tmp_path / "parent")
    output = tmp_path / "ablation"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        main(
            [
                "--parent-freeze",
                str(parent_path),
                "--output-dir",
                str(output),
            ]
        )
