from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.run_oviv2_replica import parse_args, run
from src.oviv2.snapshot import VoxelMapSnapshot


def _write_fixture(tmp_path: Path, *, cache_frames: int = 2) -> Path:
    dataset = tmp_path / "dataset"
    results = dataset / "results"
    cache = tmp_path / "cache"
    results.mkdir(parents=True)
    cache.mkdir()
    poses: list[str] = []
    for frame_id in range(2):
        Image.fromarray(np.full((32, 32, 3), 100 + frame_id, dtype=np.uint8)).save(
            results / f"frame{frame_id:06d}.jpg"
        )
        Image.fromarray(np.full((32, 32), 6554, dtype=np.uint16)).save(
            results / f"depth{frame_id:06d}.png"
        )
        pose = np.eye(4)
        pose[0, 3] = frame_id * 0.02
        poses.append(" ".join(str(value) for value in pose.reshape(-1)))
    (dataset / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
    for cache_id in range(cache_frames):
        mask = np.zeros((1, 32, 32), dtype=bool)
        mask[:, 8:24, 8:24] = True
        payload = {
            "mask": mask,
            "xyxy": np.asarray([[8, 8, 24, 24]], dtype=np.float32),
            "confidence": np.asarray([0.9], dtype=np.float32),
            "class_id": np.asarray([0], dtype=np.int64),
            "classes": ["chair"],
        }
        with gzip.open(cache / f"frame{cache_id:06d}.pkl.gz", "wb") as stream:
            pickle.dump(payload, stream)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "fixture",
                "dataset": "Replica",
                "frame_selection": {"start": 0, "stop_exclusive": 20, "stride": 10},
                "vocabulary": {"classes": ["wall", "floor", "ceiling", "chair"]},
                "aliases": {},
                "scenes": [{"scene": "room0"}],
                "protocol": {"geometry_primary_threshold_m": 0.05},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "scene": "room0",
                "dataset_root": str(dataset),
                "frontend_cache_dir": str(cache),
                "manifest": str(manifest),
                "gt_mesh": str(tmp_path / "unused.ply"),
                "gt_info": str(tmp_path / "unused.json"),
                "source_start": 0,
                "source_stride": 10,
                "voxel_size_m": 0.05,
                "block_resolution": 8,
                "pixel_stride": 2,
                "min_valid_points": 1,
                "confirm_hits": 2,
                "checkpoint_interval": 1,
            }
        ),
        encoding="utf-8",
    )
    return config


def _args(config: Path, output: Path, *extra: str):
    return parse_args(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--num-frames",
            "2",
            "--skip-evaluation",
            *extra,
        ]
    )


def test_preflight_fails_before_creating_output_for_missing_cache(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, cache_frames=1)
    output = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="frame000001"):
        run(_args(config, output))

    assert not output.exists()


def test_runner_writes_restoreable_voxel_contract_and_exact_frame_selection(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "run"

    manifest = run(_args(config, output))

    final_snapshot = output / "final" / "oviv2_voxel_snapshot.npz"
    checkpoint = output / "checkpoints" / "latest_voxel_snapshot.npz"
    assert VoxelMapSnapshot.load(final_snapshot).metadata.frame_id == 10
    assert VoxelMapSnapshot.load(checkpoint).metadata.revision == 2
    assert manifest["frame_selection"]["source_frame_ids"] == [0, 10]
    assert (output / "final" / "oviv2_entities.jsonl").is_file()
    assert (output / "final" / "oviv2_instance_mesh.ply").is_file()
    assert (output / "timing.json").is_file()
    assert manifest["authoritative_state"] == "sparse_voxel_layers"
    assert manifest["dense_point_cloud_state"] is False


def test_manifest_artifact_checksums_recompute_and_resume_is_idempotent(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    output = tmp_path / "run"
    original = run(_args(config, output))
    manifest_bytes = (output / "run_manifest.json").read_bytes()

    resumed = run(_args(config, output, "--resume"))

    assert resumed == original
    assert (output / "run_manifest.json").read_bytes() == manifest_bytes
    for relative, expected in original["artifact_checksums"].items():
        path = output / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
