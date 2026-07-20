from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_oviv2_replica8 import (
    REPLICA8_SCENES,
    build_scene_specs,
    materialize_scene_configs,
    run_replica8,
)


def test_replica8_cli_direct_execution_resolves_repository_imports() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_oviv2_replica8.py", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, scenes: tuple[str, ...] = REPLICA8_SCENES) -> Path:
    manifest = tmp_path / "replica8.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "dataset": "Replica",
            "frame_selection": {"sampled_frames_per_scene": 200},
            "scenes": [
                {"scene": scene, "ground_truth_scene": scene.replace("room", "room_").replace("office", "office_")}
                for scene in scenes
            ],
        },
    )
    base = tmp_path / "base.json"
    _write_json(
        base,
        {
            "scene": "room0",
            "dataset_root": "/old/room0",
            "frontend_cache_dir": "/old/cache",
            "manifest": str(manifest),
            "gt_mesh": "/old/mesh.ply",
            "gt_info": "/old/info.json",
            "num_frames": 200,
            "voxel_size_m": 0.05,
            "structure_enabled": True,
        },
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "manifest": str(manifest),
            "view_root": str(tmp_path / "views"),
            "view_suffix": "_s10_200f",
            "ground_truth_root": str(tmp_path / "gt"),
            "base_runner_config": str(base),
            "frontend": {"gsa_variant_template": "yolo_{scene}_s10_200f"},
        },
    )
    return config


def test_replica8_specs_require_exact_frozen_scene_set(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path, REPLICA8_SCENES[:-1])

    with pytest.raises(ValueError, match="frozen eight scenes"):
        build_scene_specs(config_path)


def test_replica8_specs_vary_only_scene_paths_and_share_algorithm_hash(tmp_path: Path) -> None:
    specs = build_scene_specs(_fixture(tmp_path))

    assert tuple(spec.scene for spec in specs) == REPLICA8_SCENES
    assert len({spec.algorithm_hash for spec in specs}) == 1
    assert specs[0].config["dataset_root"].endswith("room0_s10_200f")
    assert specs[-1].config["frontend_cache_dir"].endswith(
        "office4_s10_200f/gsa_detections_yolo_office4_s10_200f"
    )
    assert specs[-1].config["gt_mesh"].endswith("office_4/habitat/mesh_semantic.ply")


def test_replica8_stage3_specs_expand_scene_specific_dense_cache_paths(
    tmp_path: Path,
) -> None:
    config_path = _fixture(tmp_path)
    batch_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = Path(batch_config["base_runner_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base.update(
        {
            "dense_semantic_mode": "cached_probabilities",
            "dense_cache_dir": "/old/dense-cache",
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.49,
        }
    )
    _write_json(base_path, base)
    batch_config.update(
        {
            "dense_cache_root": str(tmp_path / "dense"),
            "dense_cache_template": "{scene}_radseg_b_sam_s4_k4_200f",
            "semantic_head": "fused_uncertainty",
            "fusion_entity_weight_scale": 0.49,
        }
    )
    _write_json(config_path, batch_config)

    specs = build_scene_specs(config_path)

    assert len({spec.algorithm_hash for spec in specs}) == 1
    assert specs[0].config["dense_cache_dir"].endswith(
        "room0_radseg_b_sam_s4_k4_200f"
    )
    assert specs[-1].config["dense_cache_dir"].endswith(
        "office4_radseg_b_sam_s4_k4_200f"
    )


def test_replica8_runner_records_all_scene_manifests(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    output = tmp_path / "output"
    calls: list[dict] = []

    def fake_runner(args):
        scene_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        calls.append(scene_config)
        args.output.mkdir(parents=True)
        payload = {
            "method": "OVIV2",
            "scene": scene_config["scene"],
            "final_revision": 200,
            "algorithm_hash": scene_config["algorithm_hash"],
            "config_hash": hashlib.sha256(
                json.dumps(
                    scene_config,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "frame_selection": {"sampled_frame_count": 200},
        }
        _write_json(args.output / "run_manifest.json", payload)
        return payload

    batch = run_replica8(config_path, output, scene_runner=fake_runner)

    assert [call["scene"] for call in calls] == list(REPLICA8_SCENES)
    assert len({call["algorithm_hash"] for call in calls}) == 1
    assert [item["scene"] for item in batch["scenes"]] == list(REPLICA8_SCENES)
    assert batch["semantic_head"] == "owner_authoritative"
    assert batch["base_runner_config_sha256"]
    assert all(item["scene_config_sha256"] for item in batch["scenes"])
    assert all(item["config_hash"] for item in batch["scenes"])
    assert (output / "batch_manifest.json").is_file()


def test_materialize_only_writes_exact_scene_configs_without_running_maps(
    tmp_path: Path,
) -> None:
    config_path = _fixture(tmp_path)
    output = tmp_path / "output"

    specs = materialize_scene_configs(config_path, output)

    assert tuple(spec.scene for spec in specs) == REPLICA8_SCENES
    assert sorted(path.stem for path in (output / "scene_configs").glob("*.json")) == sorted(
        REPLICA8_SCENES
    )
    assert not (output / "batch_manifest.json").exists()


@pytest.mark.parametrize("mutated_file", ["batch", "base"])
def test_runner_rejects_frozen_config_changes_during_batch(
    tmp_path: Path,
    mutated_file: str,
) -> None:
    config_path = _fixture(tmp_path)
    batch_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = Path(batch_config["base_runner_config"])
    output = tmp_path / "output"
    changed = False

    def fake_runner(args):
        nonlocal changed
        scene_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        args.output.mkdir(parents=True)
        payload = {
            "method": "OVIV2",
            "scene": scene_config["scene"],
            "final_revision": 200,
            "algorithm_hash": scene_config["algorithm_hash"],
            "config_hash": hashlib.sha256(
                json.dumps(
                    scene_config,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "frame_selection": {"sampled_frame_count": 200},
        }
        _write_json(args.output / "run_manifest.json", payload)
        if not changed:
            target = config_path if mutated_file == "batch" else base_path
            content = json.loads(target.read_text(encoding="utf-8"))
            content["changed_during_run"] = True
            _write_json(target, content)
            changed = True
        return payload

    with pytest.raises(ValueError, match="config changed during batch"):
        run_replica8(config_path, output, scene_runner=fake_runner)

    assert not (output / "batch_manifest.json").exists()
