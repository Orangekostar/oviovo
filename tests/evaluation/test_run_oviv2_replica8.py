from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_oviv2_replica8 import REPLICA8_SCENES, build_scene_specs, run_replica8


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
            "frame_selection": {"sampled_frame_count": 200},
        }
        _write_json(args.output / "run_manifest.json", payload)
        return payload

    batch = run_replica8(config_path, output, scene_runner=fake_runner)

    assert [call["scene"] for call in calls] == list(REPLICA8_SCENES)
    assert len({call["algorithm_hash"] for call in calls}) == 1
    assert [item["scene"] for item in batch["scenes"]] == list(REPLICA8_SCENES)
    assert (output / "batch_manifest.json").is_file()
