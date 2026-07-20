from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from scripts.run_oviv2_replica import _preflight, algorithm_hash, parse_args, run
from src.datasets.scannet200 import ScanNet200Dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    dataset = tmp_path / "scene0011_00"
    for name in ("color", "depth", "pose", "intrinsic"):
        (dataset / name).mkdir(parents=True)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = 30.0
    intrinsic[1, 1] = 30.0
    intrinsic[0, 2] = 15.5
    intrinsic[1, 2] = 15.5
    np.savetxt(dataset / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    frame_inputs: dict[str, dict[str, str]] = {}
    for cache_index, source_id in enumerate((0, 20)):
        color = dataset / "color" / f"{source_id}.jpg"
        depth = dataset / "depth" / f"{source_id}.png"
        pose = dataset / "pose" / f"{source_id}.txt"
        Image.fromarray(np.full((32, 32, 3), 100 + cache_index, dtype=np.uint8)).save(
            color
        )
        Image.fromarray(np.full((32, 32), 1000, dtype=np.uint16)).save(depth)
        matrix = np.eye(4)
        matrix[0, 3] = cache_index * 0.02
        np.savetxt(pose, matrix)
        frame_inputs[str(source_id)] = {
            "color_sha256": _sha256(color),
            "depth_sha256": _sha256(depth),
            "pose_sha256": _sha256(pose),
        }

    cache = tmp_path / "frontend"
    cache.mkdir()
    cache_hashes: dict[str, str] = {}
    for cache_index in range(2):
        mask = np.zeros((1, 32, 32), dtype=bool)
        mask[:, 8:24, 8:24] = True
        payload = {
            "mask": mask,
            "xyxy": np.asarray([[8, 8, 24, 24]], dtype=np.float32),
            "confidence": np.asarray([0.9], dtype=np.float32),
            "class_id": np.asarray([0], dtype=np.int64),
            "classes": ["chair"],
        }
        path = cache / f"frame{cache_index:06d}.pkl.gz"
        with gzip.open(path, "wb") as stream:
            pickle.dump(payload, stream)
        cache_hashes[path.name] = _sha256(path)
    (cache / "frontend_manifest.json").write_text(
        json.dumps(
            {
                "method": "OVIV2",
                "dataset": "ScanNet200",
                "scene": "scene0011_00",
                "frame_count": 2,
                "source_frame_ids": [0, 20],
                "algorithm_hash": "fixture-frontend",
                "cache_files_sha256": cache_hashes,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "manifest_id": "oviv2_scannet_fixture",
        "dataset": "ScanNet200",
        "frame_selection": {"start": 0, "stride": 10},
        "vocabulary": {"classes": ["wall", "floor", "ceiling", "chair"]},
        "aliases": {},
        "scenes": [
            {
                "scene": "scene0011_00",
                "dataset_root": str(dataset),
                "frame_count": 2,
                "source_frame_ids": [0, 20],
                "frame_inputs": frame_inputs,
                "image_shape": [32, 32],
                "depth_scale": 1000.0,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "scene": "scene0011_00",
        "dataset_root": str(dataset),
        "frontend_cache_dir": str(cache),
        "manifest": str(manifest_path),
        "num_frames": 2,
        "gt_mesh": str(tmp_path / "hostile-gt.ply"),
        "gt_info": str(tmp_path / "hostile-metadata.txt"),
        "source_stride": 10,
        "voxel_size_m": 0.05,
        "block_resolution": 8,
        "pixel_stride": 2,
        "min_valid_points": 1,
        "checkpoint_interval": 1,
        "structure_enabled": True,
        "structure_min_component_pixels": 4,
        "structure_min_component_fraction": 0.0,
    }
    return manifest_path, config


def test_scannet_preflight_uses_explicit_source_frame_ids(tmp_path: Path) -> None:
    _, config = _fixture(tmp_path)

    dataset, benchmark, source_ids, *_ = _preflight(
        config,
        num_frames=2,
        skip_evaluation=True,
    )

    assert isinstance(dataset, ScanNet200Dataset)
    assert benchmark["dataset"] == "ScanNet200"
    assert source_ids == [0, 20]
    assert dataset.frame_indices == (0, 20)
    assert dataset[1].frame_id == 20


def test_scannet_runner_maps_without_reading_ground_truth(tmp_path: Path) -> None:
    _, config = _fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "output"

    result = run(
        parse_args(
            [
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--num-frames",
                "2",
                "--skip-evaluation",
            ]
        )
    )

    assert result["dataset_name"] == "ScanNet200"
    assert result["frame_selection"]["source_frame_ids"] == [0, 20]
    expected_source_hash = hashlib.sha256(
        json.dumps([0, 20], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert result["source_frame_ids_hash"] == expected_source_hash
    assert result["final_revision"] == 2
    assert (output / "final" / "oviv2_voxel_snapshot.npz").is_dir()
    assert (output / "final" / "oviv2_entities.jsonl").is_file()
    assert (output / "final" / "oviv2_instance_mesh.ply").is_file()
    assert not Path(config["gt_mesh"]).exists()
    assert not Path(config["gt_info"]).exists()


def test_stage4_base_preserves_selected_stage3_mapping_parameters() -> None:
    replica = json.loads(
        Path("configs/oviv2_replica_room0_precision_stage3_fused.json").read_text(
            encoding="utf-8"
        )
    )
    scannet = json.loads(
        Path("configs/oviv2_scannet200_stage4_base.json").read_text(encoding="utf-8")
    )
    dataset_fields = {
        "scene",
        "dataset_root",
        "frontend_cache_dir",
        "manifest",
        "gt_mesh",
        "gt_info",
        "dense_cache_dir",
        "num_frames",
        "source_start",
    }

    assert {
        key: value for key, value in replica.items() if key not in dataset_fields
    } == {key: value for key, value in scannet.items() if key not in dataset_fields}
    changed_count = dict(scannet, num_frames=465)
    assert algorithm_hash(scannet) == algorithm_hash(changed_count)
