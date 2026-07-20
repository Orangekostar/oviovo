from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.precompute_oviv2_scannet200_frontend import (
    build_commands,
    main,
    validate_frontend_cache,
)


SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def _config(tmp_path: Path) -> dict:
    return {
        "view_root": str(tmp_path / "views"),
        "frontend_logs_root": str(tmp_path / "logs"),
        "frontend": {
            "python": str(tmp_path / "python"),
            "script": str(tmp_path / "streamlined_detections.py"),
            "hydra_config_name": "frozen",
            "hydra_config_path": str(tmp_path / "frozen.yaml"),
            "dataset_config": str(tmp_path / "replica.yaml"),
            "classes_file": str(tmp_path / "scannet200_classes.txt"),
            "yolo_model_path": str(tmp_path / "yolo.pt"),
            "yolo_clip_model_path": str(tmp_path / "yolo-clip.pt"),
            "yolo_clip_model_sha256": "1" * 64,
            "mobile_sam_model_path": str(tmp_path / "sam.pt"),
            "clip_model_card": "ViT-H-14",
            "clip_pretrained_path": str(tmp_path / "clip.bin"),
            "device": "cuda",
            "desired_height": 480,
            "desired_width": 640,
            "gsa_variant_template": "oviv2_scannet200_{scene}",
            "exp_suffix_template": "oviv2_scannet200_{scene}",
        },
    }


def _manifest() -> dict:
    counts = (238, 465, 444, 190, 147)
    return {
        "schema_version": 1,
        "dataset": "ScanNet200",
        "scenes": [
            {
                "scene": scene,
                "frame_count": count,
                "source_frame_ids": list(range(0, count * 10, 10)),
            }
            for scene, count in zip(SCENES, counts, strict=True)
        ],
    }


def test_frontend_commands_use_stage4_views_and_scene_frame_counts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    commands = build_commands(config, _manifest(), gpu_ids=(0, 1, 2))

    assert [command.scene for command in commands] == list(SCENES)
    assert [command.gpu_id for command in commands] == [0, 1, 2, 0, 1]
    assert len({command.algorithm_hash for command in commands}) == 1
    assert "end=238" in commands[0].argv
    assert "end=465" in commands[1].argv
    assert f"classes_file={config['frontend']['classes_file']}" in commands[0].argv
    assert commands[0].cache_dir == (
        tmp_path
        / "views"
        / "scene0011_00"
        / "gsa_detections_oviv2_scannet200_scene0011_00"
    )
    assert all(
        not command.cache_dir.is_relative_to(Path("/home/ww/oviovo_baseline_runs"))
        for command in commands
    )


def test_frontend_validator_binds_source_ids_models_and_cache_files(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    frontend = config["frontend"]
    classes = ["wall", "floor", "ceiling"] + [
        f"object-{index:03d}" for index in range(197)
    ]
    for key in (
        "script",
        "hydra_config_path",
        "dataset_config",
        "yolo_model_path",
        "yolo_clip_model_path",
        "mobile_sam_model_path",
        "clip_pretrained_path",
    ):
        path = Path(frontend[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode("ascii"))
    classes_path = Path(frontend["classes_file"])
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    frontend["yolo_clip_model_sha256"] = hashlib.sha256(
        Path(frontend["yolo_clip_model_path"]).read_bytes()
    ).hexdigest()
    manifest = _manifest()
    manifest["scenes"][0]["frame_count"] = 1
    manifest["scenes"][0]["source_frame_ids"] = [0]
    command = build_commands(
        config,
        manifest,
        gpu_ids=(0,),
        scenes=("scene0011_00",),
    )[0]
    results = command.cache_dir.parent / "results"
    results.mkdir(parents=True)
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
        results / "frame000000.jpg"
    )
    (command.cache_dir.parent / "view_manifest.json").write_text(
        json.dumps(
            {
                "scene": "scene0011_00",
                "frame_count": 1,
                "source_frame_ids": [0],
                "source_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    command.cache_dir.mkdir()
    object_classes = classes[3:]
    payload = {
        "mask": np.ones((1, 480, 640), dtype=bool),
        "xyxy": np.array([[0.0, 0.0, 639.0, 479.0]], dtype=np.float32),
        "confidence": np.array([0.75], dtype=np.float32),
        "class_id": np.array([0], dtype=np.int64),
        "classes": object_classes,
        "image_feats": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "text_feats": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "image_crops": [],
    }
    cache_path = command.cache_dir / "frame000000.pkl.gz"
    with gzip.open(cache_path, "wb") as stream:
        pickle.dump(payload, stream)

    result = validate_frontend_cache(
        command,
        config,
        input_manifest_sha256="a" * 64,
    )

    assert result["scene"] == "scene0011_00"
    assert result["frame_count"] == 1
    assert result["source_frame_ids"] == [0]
    assert result["class_count"] == 197
    assert result["classes"] == object_classes
    assert result["feature_model_id"].startswith("clip-sha256:")
    assert result["input_manifest_sha256"] == "a" * 64
    assert result["cache_files_sha256"] == {
        "frame000000.pkl.gz": hashlib.sha256(cache_path.read_bytes()).hexdigest()
    }
    assert json.loads(
        (command.cache_dir / "frontend_manifest.json").read_text(encoding="utf-8")
    ) == result


def test_frontend_validator_rejects_extra_cache_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frontend = config["frontend"]
    classes = ["wall", "floor", "ceiling"] + [
        f"object-{index:03d}" for index in range(197)
    ]
    for key in (
        "script",
        "hydra_config_path",
        "dataset_config",
        "yolo_model_path",
        "yolo_clip_model_path",
        "mobile_sam_model_path",
        "clip_pretrained_path",
    ):
        path = Path(frontend[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode("ascii"))
    Path(frontend["classes_file"]).write_text(
        "\n".join(classes) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest()
    manifest["scenes"][0]["frame_count"] = 1
    manifest["scenes"][0]["source_frame_ids"] = [0]
    command = build_commands(config, manifest, gpu_ids=(0,), scenes=(SCENES[0],))[0]
    results = command.cache_dir.parent / "results"
    results.mkdir(parents=True)
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
        results / "frame000000.jpg"
    )
    (command.cache_dir.parent / "view_manifest.json").write_text(
        json.dumps(
            {
                "scene": SCENES[0],
                "frame_count": 1,
                "source_frame_ids": [0],
                "source_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    command.cache_dir.mkdir()
    payload = {
        "mask": np.empty((0, 480, 640), dtype=bool),
        "xyxy": np.empty((0, 4), dtype=np.float32),
        "confidence": np.empty(0, dtype=np.float32),
        "class_id": np.empty(0, dtype=np.int64),
        "classes": classes[3:],
        "image_feats": np.empty((0, 4), dtype=np.float32),
        "text_feats": np.empty((0, 4), dtype=np.float32),
    }
    for name in ("frame000000.pkl.gz", "frame999999.pkl.gz"):
        with gzip.open(command.cache_dir / name, "wb") as stream:
            pickle.dump(payload, stream)

    with pytest.raises(ValueError, match="unexpected cache file"):
        validate_frontend_cache(command, config, input_manifest_sha256="a" * 64)


def test_frontend_cli_dry_run_is_non_mutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    config["manifest"] = str(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--gpu", "0", "--dry-run"])

    assert exit_code == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["scene"] for record in records] == list(SCENES)
    assert not (tmp_path / "views").exists()
    assert not (tmp_path / "logs").exists()


def test_frontend_script_runs_directly_from_repo_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    config["manifest"] = str(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/precompute_oviv2_scannet200_frontend.py",
            "--config",
            str(config_path),
            "--gpu",
            "0",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 5
