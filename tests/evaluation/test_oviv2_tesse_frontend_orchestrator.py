from __future__ import annotations

import copy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from scripts.precompute_oviv2_tesse_frontend import (
    build_commands,
    build_environment,
    main,
    validate_frontend_cache,
    warm_shared_clip_cache,
)


SCENES = ("apartment", "office")
FRAME_COUNTS = {"apartment": 1745, "office": 4346}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_provenance_files(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in (
        "script",
        "hydra_config",
        "dataset_config",
        "yolo_model",
        "yolo_clip_model",
        "mobile_sam_model",
        "clip_model",
    ):
        path = tmp_path / "provenance" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen-{name}".encode("ascii"))
        paths[name] = str(path)
    return paths


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    paths = _write_provenance_files(tmp_path)
    vocabularies = {
        "apartment": ["Fridge", "Books", "Chair"],
        "office": ["Small office objects", "Chairs", "Signs"],
    }
    scenes: dict[str, dict] = {}
    for scene in SCENES:
        root = tmp_path / "rgbd" / scene
        root.mkdir(parents=True)
        vocabulary = tmp_path / f"{scene}.txt"
        vocabulary.write_text("\n".join(vocabularies[scene]) + "\n", encoding="utf-8")
        count = FRAME_COUNTS[scene]
        scenes[scene] = {
            "root": str(root),
            "frame_count": count,
            "image_shape": [480, 720],
            "source_frame_ids": {
                "start": 0,
                "stop_exclusive": count,
                "stride": 1,
            },
            "vocabulary": {
                "txt_path": str(vocabulary),
                "txt_sha256": _sha256(vocabulary),
            },
        }
    manifest = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "camera": {"height": 480, "width": 720},
        "scenes": scenes,
    }
    config = {
        "manifest": str(tmp_path / "manifest.json"),
        "manifest_sha256": "0" * 64,
        "frontend_logs_root": str(tmp_path / "logs"),
        "frontend": {
            "python": str(tmp_path / "python"),
            "script": paths["script"],
            "hydra_config_name": "frozen",
            "hydra_config_path": paths["hydra_config"],
            "dataset_config": paths["dataset_config"],
            "yolo_model_path": paths["yolo_model"],
            "yolo_clip_model_path": paths["yolo_clip_model"],
            "mobile_sam_model_path": paths["mobile_sam_model"],
            "clip_model_card": "ViT-H-14",
            "clip_pretrained_path": paths["clip_model"],
            "device": "cuda",
            "desired_height": 480,
            "desired_width": 720,
            "gsa_variant_template": "oviv2_tesse_{scene}_stage3_v1",
            "exp_suffix_template": "oviv2_tesse_{scene}_stage3_v1",
            "provenance_sha256": {
                name: _sha256(Path(path)) for name, path in paths.items()
            },
        },
    }
    return config, manifest


def test_commands_freeze_two_scenes_gpus_frames_and_vocabularies(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)

    commands = build_commands(config, manifest)

    assert [command.scene for command in commands] == ["apartment", "office"]
    assert [command.gpu_id for command in commands] == [0, 1]
    assert [command.frame_count for command in commands] == [1745, 4346]
    assert commands[0].source_frame_ids == tuple(range(1745))
    assert commands[1].source_frame_ids == tuple(range(4346))
    assert {command.image_shape for command in commands} == {(480, 720)}
    assert len({command.algorithm_hash for command in commands}) == 2
    for command in commands:
        scene_record = manifest["scenes"][command.scene]
        vocabulary = scene_record["vocabulary"]
        assert command.classes_sha256 == vocabulary["txt_sha256"]
        assert f"classes_file={Path(vocabulary['txt_path']).resolve()}" in command.argv
        assert f"scene_id={command.scene}" in command.argv
        assert f"end={FRAME_COUNTS[command.scene]}" in command.argv
        assert "desired_height=480" in command.argv
        assert "desired_width=720" in command.argv
        assert command.cache_dir == (
            Path(scene_record["root"])
            / f"gsa_detections_oviv2_tesse_{command.scene}_stage3_v1"
        )


def test_commands_reject_unknown_scene(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown TESSE-CD scene"):
        build_commands(config, manifest, scenes=("warehouse",))


def test_commands_reject_missing_source_root_and_symlinked_vocabulary(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    Path(manifest["scenes"]["office"]["root"]).rmdir()

    with pytest.raises(ValueError, match="root path"):
        build_commands(config, manifest)

    Path(manifest["scenes"]["office"]["root"]).mkdir()
    vocabulary = Path(manifest["scenes"]["apartment"]["vocabulary"]["txt_path"])
    real_vocabulary = vocabulary.with_suffix(".real.txt")
    vocabulary.rename(real_vocabulary)
    vocabulary.symlink_to(real_vocabulary)
    with pytest.raises(ValueError, match="vocabulary.*regular non-symlink"):
        build_commands(config, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("dataset", "Replica"), "dataset"),
        (
            lambda value: value["camera"].__setitem__("width", 640),
            "camera",
        ),
        (
            lambda value: value["scenes"]["office"].__setitem__("frame_count", 4345),
            "frame_count",
        ),
        (
            lambda value: value["scenes"]["apartment"]["source_frame_ids"].__setitem__(
                "start", 1
            ),
            "source frame IDs",
        ),
        (
            lambda value: value["scenes"]["office"].__setitem__(
                "image_shape", [720, 480]
            ),
            "image shape",
        ),
        (
            lambda value: value["scenes"]["apartment"]["vocabulary"].__setitem__(
                "txt_sha256", "f" * 64
            ),
            "vocabulary hash",
        ),
    ],
)
def test_commands_reject_non_frozen_manifest_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    config, manifest = _fixture(tmp_path)
    changed = copy.deepcopy(manifest)
    mutation(changed)

    with pytest.raises(ValueError, match=message):
        build_commands(config, changed)


def test_fixture_manifest_can_be_serialized_canonically(tmp_path: Path) -> None:
    _, manifest = _fixture(tmp_path)

    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest


def test_frontend_environment_exposes_only_selected_gpu_and_package_root(
    tmp_path: Path,
) -> None:
    conceptgraphs_root = tmp_path / "concept-graphs"

    environment = build_environment(
        1,
        conceptgraphs_root,
        base={"PYTHONPATH": "/existing", "UNCHANGED": "yes"},
    )

    assert environment == {
        "CUDA_VISIBLE_DEVICES": "1",
        "PYTHONPATH": f"{conceptgraphs_root}:/existing",
        "UNCHANGED": "yes",
    }


def test_shared_clip_warmup_accepts_only_the_frozen_existing_weight(
    tmp_path: Path,
) -> None:
    config, _ = _fixture(tmp_path)
    frontend = config["frontend"]

    assert warm_shared_clip_cache(frontend) == frontend["provenance_sha256"][
        "yolo_clip_model"
    ]

    Path(frontend["yolo_clip_model_path"]).write_bytes(b"drifted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        warm_shared_clip_cache(frontend)


def _small_command(tmp_path: Path) -> tuple[dict, object]:
    config, manifest = _fixture(tmp_path)
    config["manifest_sha256"] = "a" * 64
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    return config, replace(
        command,
        frame_count=2,
        source_frame_ids=(0, 1),
    )


def _cache_payload(classes: list[str], *, count: int = 1) -> dict:
    return {
        "mask": np.ones((count, 480, 720), dtype=bool),
        "xyxy": np.tile(
            np.array([[0.0, 0.0, 719.0, 479.0]], dtype=np.float32),
            (count, 1),
        ),
        "confidence": np.full(count, 0.75, dtype=np.float32),
        "class_id": np.zeros(count, dtype=np.int64),
        "classes": classes,
        "image_feats": np.ones((count, 4), dtype=np.float32),
        "text_feats": np.ones((count, 4), dtype=np.float32),
    }


def _write_cache(command, payloads: list[dict]) -> None:
    command.cache_dir.mkdir(parents=True)
    for frame_id, payload in enumerate(payloads):
        with gzip.open(command.cache_dir / f"frame{frame_id:06d}.pkl.gz", "wb") as stream:
            pickle.dump(payload, stream)


def _classes(command) -> list[str]:
    return [
        value.strip()
        for value in command.classes_file.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def test_validator_binds_complete_cache_provenance_and_prefix_digest(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    result = validate_frontend_cache(
        command,
        config,
        input_manifest_sha256="a" * 64,
    )

    assert result["method"] == "OVIV2"
    assert result["dataset"] == "TESSE-CD"
    assert result["scene"] == "apartment"
    assert result["frame_count"] == 2
    assert result["source_frame_ids"] == [0, 1]
    assert result["image_shape"] == [480, 720]
    assert result["classes"] == classes
    assert result["vocabulary_sha256"] == command.classes_sha256
    assert result["feature_model_id"] == (
        f"clip-sha256:{config['frontend']['provenance_sha256']['clip_model']}"
    )
    assert result["input_manifest_sha256"] == "a" * 64
    assert list(result["cache_files_sha256"]) == [
        "frame000000.pkl.gz",
        "frame000001.pkl.gz",
    ]
    prefix = hashlib.sha256()
    for frame_id, digest in enumerate(result["cache_files_sha256"].values()):
        prefix.update(frame_id.to_bytes(8, "little", signed=False))
        prefix.update(bytes.fromhex(digest))
    assert result["cache_prefix_sha256"] == prefix.hexdigest()
    manifest_path = command.cache_dir / "frontend_manifest.json"
    before = manifest_path.read_bytes()
    assert json.loads(before) == result

    assert validate_frontend_cache(
        command,
        config,
        input_manifest_sha256="a" * 64,
    ) == result
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("image_feats"), "required fields"),
        (
            lambda payload: payload.__setitem__(
                "mask", np.ones((1, 480, 640), dtype=bool)
            ),
            "mask/box shape",
        ),
        (
            lambda payload: payload.__setitem__("classes", ["wrong"]),
            "class order",
        ),
        (
            lambda payload: payload.__setitem__(
                "text_feats", np.zeros((1, 4), dtype=np.float32)
            ),
            "zero row",
        ),
    ],
)
def test_validator_rejects_invalid_cache_payload_without_manifest(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    config, command = _small_command(tmp_path)
    payload = _cache_payload(_classes(command))
    mutate(payload)
    _write_cache(command, [payload, _cache_payload(_classes(command), count=0)])

    with pytest.raises(ValueError, match=message):
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256="a" * 64,
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


@pytest.mark.parametrize("extra", [False, True])
def test_validator_rejects_missing_or_unexpected_frame_without_manifest(
    tmp_path: Path,
    extra: bool,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    payloads = [_cache_payload(classes), _cache_payload(classes, count=0)]
    if extra:
        payloads.append(_cache_payload(classes, count=0))
    else:
        payloads.pop()
    _write_cache(command, payloads)

    with pytest.raises(ValueError, match="missing or unexpected cache file"):
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256="a" * 64,
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_validator_rejects_noncontiguous_source_ids_and_provenance_drift(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    with pytest.raises(ValueError, match="source frame IDs"):
        validate_frontend_cache(
            replace(command, source_frame_ids=(0, 2)),
            config,
            input_manifest_sha256="a" * 64,
        )

    Path(config["frontend"]["script"]).write_bytes(b"drifted-script")
    with pytest.raises(ValueError, match="script hash mismatch"):
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256="a" * 64,
        )
    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_validator_never_clobbers_an_existing_manifest(tmp_path: Path) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    manifest_path = command.cache_dir / "frontend_manifest.json"
    sentinel = b'{"sentinel":true}\n'
    manifest_path.write_bytes(sentinel)

    with pytest.raises(ValueError, match="existing frontend manifest"):
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256="a" * 64,
        )

    assert manifest_path.read_bytes() == sentinel


def test_cli_dry_run_emits_frozen_commands_without_creating_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["--config", str(config_path), "--dry-run"]) == 0

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(row["scene"], row["gpu"], row["frame_count"]) for row in rows] == [
        ("apartment", 0, 1745),
        ("office", 1, 4346),
    ]
    assert not Path(config["frontend_logs_root"]).exists()
    assert all(
        not (
            tmp_path
            / "rgbd"
            / scene
            / f"gsa_detections_oviv2_tesse_{scene}_stage3_v1"
        ).exists()
        for scene in SCENES
    )


def test_script_runs_directly_and_selects_office_without_mutation(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/precompute_oviv2_tesse_frontend.py",
            "--config",
            str(config_path),
            "--scene",
            "office",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "scene": "office",
        "gpu": 1,
        "frame_count": 4346,
        "argv": list(build_commands(config, manifest, scenes=("office",))[0].argv),
    }
    assert not Path(config["frontend_logs_root"]).exists()


def test_formal_config_binds_real_inputs_and_contains_no_evaluation_paths() -> None:
    config_path = Path("configs/oviv2_tesse_cd_frontend_stage3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert config["manifest_sha256"] == _sha256(manifest_path)
    assert config["frontend"]["yolo_model_path"] == (
        "/home/ww/vv/paper2/DualMap/model/yolov8l-world.pt"
    )
    assert config["frontend"]["mobile_sam_model_path"] == (
        "/home/ww/vv/paper2/DualMap/model/mobile_sam.pt"
    )
    path_keys = {
        "script": "script",
        "hydra_config": "hydra_config_path",
        "dataset_config": "dataset_config",
        "yolo_model": "yolo_model_path",
        "yolo_clip_model": "yolo_clip_model_path",
        "mobile_sam_model": "mobile_sam_model_path",
        "clip_model": "clip_pretrained_path",
    }
    for name, path_key in path_keys.items():
        assert config["frontend"]["provenance_sha256"][name] == _sha256(
            Path(config["frontend"][path_key])
        )
    serialized = json.dumps(config, sort_keys=True).lower()
    for forbidden in ("ground_truth", "target", "prediction"):
        assert forbidden not in serialized

    commands = build_commands(config, manifest)
    assert [(value.scene, value.gpu_id) for value in commands] == [
        ("apartment", 0),
        ("office", 1),
    ]
