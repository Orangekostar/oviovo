from __future__ import annotations

import json
import hashlib
from pathlib import Path

from scripts.run_oviv2_scannet200_5 import (
    SCANNET200_5_SCENES,
    build_scene_specs,
    run_scannet200_5,
)


COUNTS = (238, 465, 444, 190, 147)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    scenes = []
    for scene, count in zip(SCANNET200_5_SCENES, COUNTS, strict=True):
        scenes.append(
            {
                "scene": scene,
                "dataset_root": str(tmp_path / "exported" / scene),
                "frame_count": count,
                "source_frame_ids": list(range(0, count * 10, 10)),
                "official_gt_path": str(tmp_path / "gt" / f"{scene}.ply"),
                "metadata_path": str(tmp_path / "raw" / scene / f"{scene}.txt"),
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "dataset": "ScanNet200",
            "scenes": scenes,
        },
    )
    base = tmp_path / "base.json"
    _write_json(
        base,
        {
            "scene": SCANNET200_5_SCENES[0],
            "dataset_root": "/old/dataset",
            "frontend_cache_dir": "/old/frontend",
            "manifest": str(manifest),
            "gt_mesh": "/old/gt.ply",
            "gt_info": "/old/metadata.txt",
            "num_frames": COUNTS[0],
            "source_stride": 10,
            "voxel_size_m": 0.05,
            "dense_semantic_mode": "cached_probabilities",
            "dense_cache_dir": "/old/dense",
            "fusion_semantic_mode": "uncertainty_linear",
            "fusion_entity_weight_scale": 0.49,
        },
    )
    config = tmp_path / "batch.json"
    _write_json(
        config,
        {
            "manifest": str(manifest),
            "view_root": str(tmp_path / "views"),
            "base_runner_config": str(base),
            "dense_cache_root": str(tmp_path / "dense"),
            "semantic_head": "fused_uncertainty",
            "fusion_entity_weight_scale": 0.49,
            "frontend": {
                "gsa_variant_template": "oviv2_scannet200_{scene}_s10"
            },
        },
    )
    return config


def test_scannet5_specs_bind_scene_paths_counts_and_shared_algorithm(
    tmp_path: Path,
) -> None:
    specs = build_scene_specs(_fixture(tmp_path))

    assert tuple(spec.scene for spec in specs) == SCANNET200_5_SCENES
    assert [spec.config["num_frames"] for spec in specs] == list(COUNTS)
    assert len({spec.algorithm_hash for spec in specs}) == 1
    assert specs[0].config["dataset_root"].endswith("exported/scene0011_00")
    assert specs[-1].config["frontend_cache_dir"].endswith(
        "scene0518_00/gsa_detections_oviv2_scannet200_scene0518_00_s10"
    )
    assert specs[-1].config["dense_cache_dir"].endswith(
        "dense/scene0518_00"
    )
    assert specs[-1].config["gt_mesh"].endswith("gt/scene0518_00.ply")
    assert specs[-1].config["gt_info"].endswith(
        "raw/scene0518_00/scene0518_00.txt"
    )


def test_scannet5_runner_records_variable_complete_scene_runs(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    output = tmp_path / "output"

    def fake_runner(args):
        scene_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        count = scene_config["num_frames"]
        args.output.mkdir(parents=True)
        payload = {
            "method": "OVIV2",
            "dataset_name": "ScanNet200",
            "scene": scene_config["scene"],
            "final_revision": count,
            "algorithm_hash": scene_config["algorithm_hash"],
            "config_hash": hashlib.sha256(
                json.dumps(
                    scene_config,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "frame_selection": {
                "sampled_frame_count": count,
                "source_frame_ids": list(range(0, count * 10, 10)),
            },
            "frontend_algorithm_hash": "frontend-hash",
        }
        _write_json(args.output / "run_manifest.json", payload)
        return payload

    batch = run_scannet200_5(config_path, output, scene_runner=fake_runner)

    assert batch["dataset"] == "ScanNet200"
    assert batch["scene_ids"] == list(SCANNET200_5_SCENES)
    assert batch["frame_counts"] == dict(zip(SCANNET200_5_SCENES, COUNTS, strict=True))
    assert len({record["algorithm_hash"] for record in batch["scenes"]}) == 1
    assert all(record["scene_config_sha256"] for record in batch["scenes"])
    assert all(record["run_manifest_sha256"] for record in batch["scenes"])
    assert (output / "batch_manifest.json").is_file()
