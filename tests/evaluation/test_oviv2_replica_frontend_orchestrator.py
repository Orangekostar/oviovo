from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_replica_stride_view import materialize
from scripts.precompute_oviv2_replica_frontend import build_commands


SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")


def _source_scene(root: Path, scene: str = "room0", frame_count: int = 30) -> Path:
    source = root / scene
    results = source / "results"
    results.mkdir(parents=True)
    poses: list[str] = []
    for frame_id in range(frame_count):
        (results / f"frame{frame_id:06d}.jpg").write_bytes(f"rgb-{frame_id}".encode())
        (results / f"depth{frame_id:06d}.png").write_bytes(f"depth-{frame_id}".encode())
        poses.append(" ".join([str(frame_id)] * 16))
    (source / "traj.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
    return source


def test_stride_view_links_exact_source_frames_and_sampled_poses(tmp_path: Path) -> None:
    source = _source_scene(tmp_path / "source")
    target = tmp_path / "views" / "room0_s10_3f"

    summary = materialize(source, target, start=0, stop=30, stride=10)

    assert summary["source_frame_ids"] == [0, 10, 20]
    assert (target / "results" / "frame000001.jpg").is_symlink()
    assert (target / "results" / "frame000001.jpg").resolve() == (
        source / "results" / "frame000010.jpg"
    ).resolve()
    assert (target / "results" / "depth000002.png").resolve() == (
        source / "results" / "depth000020.png"
    ).resolve()
    assert (target / "traj.txt").read_text(encoding="utf-8").splitlines() == [
        " ".join([str(value)] * 16) for value in (0, 10, 20)
    ]
    assert json.loads((target / "frame_manifest.json").read_text())["frame_count"] == 3


def test_stride_view_revalidates_existing_view_without_replacing_it(tmp_path: Path) -> None:
    source = _source_scene(tmp_path / "source")
    target = tmp_path / "views" / "room0_s10_3f"
    first = materialize(source, target, start=0, stop=30, stride=10)
    marker = target / "keep-existing-output.txt"
    marker.write_text("keep", encoding="utf-8")

    second = materialize(source, target, start=0, stop=30, stride=10)

    assert second == first
    assert marker.read_text(encoding="utf-8") == "keep"


def test_frontend_commands_share_one_frozen_algorithm_hash(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "dataset": "Replica",
        "frame_selection": {"sampled_frames_per_scene": 200},
        "scenes": [{"scene": scene} for scene in SCENES],
    }
    frontend = {
        "python": str(tmp_path / "python"),
        "script": str(tmp_path / "streamlined_detections.py"),
        "hydra_config_name": "frozen",
        "hydra_config_path": str(tmp_path / "frozen.yaml"),
        "dataset_config": str(tmp_path / "replica.yaml"),
        "classes_file": str(tmp_path / "classes.txt"),
        "yolo_model_path": str(tmp_path / "yolo.pt"),
        "mobile_sam_model_path": str(tmp_path / "sam.pt"),
        "clip_model_card": "ViT-H-14",
        "clip_pretrained_path": str(tmp_path / "clip.bin"),
        "device": "cuda",
        "desired_height": 480,
        "desired_width": 640,
        "gsa_variant_template": "yolo_{scene}_s10_200f",
        "exp_suffix_template": "benchmark_{scene}_s10_200f",
    }
    config = {
        "view_root": str(tmp_path / "views"),
        "view_suffix": "_s10_200f",
        "frontend_logs_root": str(tmp_path / "logs"),
        "frontend": frontend,
    }

    commands = build_commands(config, manifest, gpu_ids=(0, 1))

    assert len(commands) == 8
    assert {command.algorithm_hash for command in commands} == {commands[0].algorithm_hash}
    assert [command.gpu_id for command in commands] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [command.scene for command in commands] == list(SCENES)
    assert all(f"scene_id={command.scene}_s10_200f" in command.argv for command in commands)
