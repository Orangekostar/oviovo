from __future__ import annotations

import copy
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import scripts.precompute_oviv2_tesse_frontend as frontend_module
from scripts.precompute_oviv2_tesse_frontend import (
    FrontendManifestPublicationUncertainError,
    build_commands,
    build_environment,
    main,
    validate_frontend_cache,
    warm_shared_clip_cache,
)


SCENES = ("apartment", "office")
FRAME_COUNTS = {"apartment": 1745, "office": 4346}


class _StubTesseCdRgbdDataset:
    """Small structural stand-in explicitly injected by unit tests only."""

    _fixture_only = True
    generation = 0

    def __init__(self, root, scene, export_manifest, schedule_manifest) -> None:
        self.root = Path(root)
        self.scene = scene
        self.export_manifest_path = Path(export_manifest)
        self.schedule_manifest_path = Path(schedule_manifest)
        self.camera_path = self.root.parent / "cam_params.json"
        self.timestamps_path = self.root / "timestamps.csv"
        self.trajectory_path = self.root / "traj.txt"
        self._length = FRAME_COUNTS[scene]
        self.intrinsics = SimpleNamespace(
            fx=415.0,
            fy=415.0,
            cx=360.0,
            cy=240.0,
            width=720,
            height=480,
        )
        generation = type(self).generation
        self.records = (
            SimpleNamespace(
                frame_index=0,
                timestamp_ns=generation,
                relative_timestamp_ns=0,
                rgb_path=self.root / "results/frame000000.jpg",
                depth_path=self.root / "results/depth000000.png",
                camera_to_world=np.eye(4, dtype=np.float64),
            ),
            SimpleNamespace(
                frame_index=self._length - 1,
                timestamp_ns=self._length - 1,
                relative_timestamp_ns=self._length - 1,
                rgb_path=self.root / f"results/frame{self._length - 1:06d}.jpg",
                depth_path=self.root / f"results/depth{self._length - 1:06d}.png",
                camera_to_world=np.eye(4, dtype=np.float64),
            ),
        )

    def __len__(self) -> int:
        return self._length


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
    rgbd_root = tmp_path / "rgbd"
    camera = rgbd_root / "cam_params.json"
    camera.parent.mkdir(parents=True)
    camera.write_text('{"width":720,"height":480}\n', encoding="utf-8")
    schedule = tmp_path / "schedule.json"
    schedule.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    scenes: dict[str, dict] = {}
    for scene in SCENES:
        root = rgbd_root / scene
        root.mkdir(parents=True)
        export = root / "export_manifest.json"
        export.write_text(
            json.dumps(
                {
                    "dataset": "TESSE-CD",
                    "scene": scene,
                    "frame_count": FRAME_COUNTS[scene],
                    "combined_output_sha256": "a" * 64,
                    "file_hash_count": FRAME_COUNTS[scene] * 2 + 3,
                }
            ),
            encoding="utf-8",
        )
        timestamps = root / "timestamps.csv"
        timestamps.write_text("frame_index,timestamp_ns\n", encoding="utf-8")
        trajectory = root / "traj.txt"
        trajectory.write_text("pose\n", encoding="utf-8")
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
            "export_manifest": {
                "path": str(export),
                "sha256": _sha256(export),
                "combined_output_sha256": "a" * 64,
                "file_hash_count": FRAME_COUNTS[scene] * 2 + 3,
            },
            "timestamps": {
                "path": str(timestamps),
                "sha256": _sha256(timestamps),
            },
            "trajectory": {
                "path": str(trajectory),
                "sha256": _sha256(trajectory),
            },
            "vocabulary": {
                "txt_path": str(vocabulary),
                "txt_sha256": _sha256(vocabulary),
            },
        }
    manifest = {
        "schema_version": 1,
        "dataset": "TESSE-CD",
        "camera": {
            "height": 480,
            "width": 720,
            "path": str(camera),
            "sha256": _sha256(camera),
        },
        "schedule_manifest": {
            "path": str(schedule),
            "sha256": _sha256(schedule),
        },
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _sha256(source_manifest),
        },
        "scenes": scenes,
    }
    config = {
        "manifest": str(tmp_path / "manifest.json"),
        "manifest_sha256": "0" * 64,
        "frontend_cache_root": str(tmp_path / "frontend-cache"),
        "required_free_bytes": 300 * 1024**3,
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
    assert len({command.algorithm_hash for command in commands}) == 1
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
            Path(config["frontend_cache_root"])
            / command.scene
            / f"gsa_detections_oviv2_tesse_{command.scene}_stage3_v1"
        )
        assert f"dataset_root={Path(config['frontend_cache_root'])}" in command.argv
        assert not command.cache_dir.is_relative_to(Path(scene_record["root"]))


def test_commands_apply_gpu_overrides_in_selected_scene_order(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)

    commands = build_commands(config, manifest, gpu_ids=(2, 0))
    office = build_commands(
        config,
        manifest,
        scenes=("office",),
        gpu_ids=(1,),
    )

    assert [command.gpu_id for command in commands] == [2, 0]
    assert [(command.scene, command.gpu_id) for command in office] == [("office", 1)]
    with pytest.raises(ValueError, match="exactly one GPU ID per requested scene"):
        build_commands(config, manifest, gpu_ids=(0,))


def test_commands_reject_cache_root_inside_derived_rgbd_tree(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    config["frontend_cache_root"] = manifest["scenes"]["apartment"]["root"]

    with pytest.raises(ValueError, match="independent of source RGB-D"):
        build_commands(config, manifest)


def test_commands_reject_unknown_scene(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown TESSE-CD scene"):
        build_commands(config, manifest, scenes=("warehouse",))


def test_commands_reject_missing_source_root_and_symlinked_vocabulary(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    office_root = Path(manifest["scenes"]["office"]["root"])
    hidden_root = office_root.with_name("office-hidden")
    office_root.rename(hidden_root)

    with pytest.raises(ValueError, match="root path"):
        build_commands(config, manifest)

    hidden_root.rename(office_root)
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
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
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
        "image_crops": [],
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


def _validate_frontend_cache(*args, **kwargs):
    kwargs.setdefault("dataset_factory", _StubTesseCdRgbdDataset)
    return validate_frontend_cache(*args, **kwargs)


def test_validator_binds_complete_cache_provenance_and_prefix_digest(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    result = _validate_frontend_cache(
        command,
        config,
        input_manifest_sha256=config["manifest_sha256"],
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
    assert result["input_manifest_sha256"] == config["manifest_sha256"]
    assert result["input_witness"]["scene"] == "apartment"
    assert result["input_witness"]["validated_frame_count"] == 1745
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

    assert _validate_frontend_cache(
        command,
        config,
        input_manifest_sha256=config["manifest_sha256"],
    ) == result
    assert manifest_path.read_bytes() == before


class _WriteMarkerOnUnpickle:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def test_validator_rejects_executable_pickle_without_running_it(tmp_path: Path) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    marker = tmp_path / "pickle-executed"
    malicious = _cache_payload(classes)
    malicious["image_feats"] = _WriteMarkerOnUnpickle(marker)
    _write_cache(command, [malicious, _cache_payload(classes, count=0)])

    with pytest.raises(ValueError, match="unsafe pickle"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    assert not marker.exists()
    assert not (command.cache_dir / "frontend_manifest.json").exists()
    source = Path(frontend_module.__file__).read_text(encoding="utf-8")
    assert "pickle.load(" not in source


def test_validator_safely_discards_streamlined_pil_image_crops(tmp_path: Path) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    payload = _cache_payload(classes)
    payload["image_crops"] = [Image.new("RGB", (3, 2), color=(1, 2, 3))]
    _write_cache(command, [payload, _cache_payload(classes, count=0)])

    result = _validate_frontend_cache(
        command,
        config,
        input_manifest_sha256=config["manifest_sha256"],
    )

    assert result["frame_count"] == 2


def test_validator_hashes_and_parses_each_cache_from_one_stable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    cache_paths = {
        command.cache_dir / "frame000000.pkl.gz",
        command.cache_dir / "frame000001.pkl.gz",
    }
    original_open = frontend_module.hybrid_cache._open_regular_input
    cache_open_count = {path: 0 for path in cache_paths}

    def count_cache_open(path: Path, *, field_name: str):
        if path in cache_open_count:
            cache_open_count[path] += 1
        return original_open(path, field_name=field_name)

    monkeypatch.setattr(
        frontend_module.hybrid_cache,
        "_open_regular_input",
        count_cache_open,
    )

    _validate_frontend_cache(
        command,
        config,
        input_manifest_sha256=config["manifest_sha256"],
    )

    assert cache_open_count == {path: 1 for path in cache_paths}


def test_validator_rejects_cache_path_replacement_after_safe_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    trusted = command.cache_dir / "frame000000.pkl.gz"
    replacement = tmp_path / "replacement.pkl.gz"
    with gzip.open(replacement, "wb") as stream:
        pickle.dump(_cache_payload(["replacement"]), stream)
    original_open = frontend_module.hybrid_cache._open_regular_input
    cache_open_count = 0

    def replace_after_open(path: Path, *, field_name: str):
        nonlocal cache_open_count
        source = original_open(path, field_name=field_name)
        if path == trusted:
            cache_open_count += 1
            replacement.replace(trusted)
        return source

    monkeypatch.setattr(
        frontend_module.hybrid_cache,
        "_open_regular_input",
        replace_after_open,
    )

    with pytest.raises(ValueError, match="changed.*snapshot"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    assert cache_open_count == 1
    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_validator_rechecks_all_snapshots_immediately_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    changed = command.cache_dir / "frame000001.pkl.gz"
    original_check = frontend_module._assert_bindings_unchanged
    invoked = False

    def replace_before_final_check(bindings) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            replacement = tmp_path / "changed.pkl.gz"
            with gzip.open(replacement, "wb") as stream:
                pickle.dump(_cache_payload(classes, count=0), stream)
            replacement.replace(changed)
        original_check(bindings)

    monkeypatch.setattr(
        frontend_module,
        "_assert_bindings_unchanged",
        replace_before_final_check,
    )

    with pytest.raises(ValueError, match="changed.*snapshot"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    assert invoked
    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_preflight_hashes_each_unique_upstream_file_once_per_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        frontend_module,
        "_gpu_inventory",
        lambda: {0: "NVIDIA A40", 1: "NVIDIA A40", 2: "NVIDIA A40"},
    )
    provenance_paths = {
        Path(config["frontend"][path_key])
        for path_key in (
            "script",
            "hydra_config_path",
            "dataset_config",
            "yolo_model_path",
            "yolo_clip_model_path",
            "mobile_sam_model_path",
            "clip_pretrained_path",
        )
    }
    original_open = frontend_module.hybrid_cache._open_regular_input
    open_count = {path: 0 for path in provenance_paths}

    def count_provenance_open(path: Path, *, field_name: str):
        if path in open_count:
            open_count[path] += 1
        return original_open(path, field_name=field_name)

    monkeypatch.setattr(
        frontend_module.hybrid_cache,
        "_open_regular_input",
        count_provenance_open,
    )

    assert main(
        [
            "--config",
            str(config_path),
            "--preflight-only",
            "--gpu",
            "0",
            "--gpu",
            "1",
        ],
        dataset_factory=_StubTesseCdRgbdDataset,
    ) == 0

    assert open_count == {path: 1 for path in provenance_paths}


def test_manifest_parent_fsync_failure_is_explicitly_uncertain_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    def fail_parent_fsync(directory: Path) -> None:
        raise OSError("parent fsync failed")

    monkeypatch.setattr(frontend_module, "_fsync_directory", fail_parent_fsync)

    with pytest.raises(FrontendManifestPublicationUncertainError) as error:
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    manifest_path = command.cache_dir / "frontend_manifest.json"
    assert error.value.path == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["scene"] == "apartment"
    assert not list(command.cache_dir.glob(".frontend_manifest.json.*.tmp"))


def test_manifest_prelink_failure_leaves_no_target_and_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    original_link = frontend_module.os.link
    calls = 0

    def fail_first_link(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("manifest link failed")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(frontend_module.os, "link", fail_first_link)
    with pytest.raises(OSError, match="manifest link failed"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )
    manifest_path = command.cache_dir / "frontend_manifest.json"
    assert not manifest_path.exists()
    assert not list(command.cache_dir.glob(".frontend_manifest.json.*.tmp"))

    result = _validate_frontend_cache(
        command,
        config,
        input_manifest_sha256=config["manifest_sha256"],
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result


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
        (
            lambda payload: payload.__setitem__(
                "image_crops", [np.asarray(["unsafe"], dtype=object)]
            ),
            "unsafe dtype",
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
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
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
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_validator_rejects_noncontiguous_source_ids_and_provenance_drift(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    with pytest.raises(ValueError, match="source frame IDs"):
        _validate_frontend_cache(
            replace(command, source_frame_ids=(0, 2)),
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    Path(config["frontend"]["script"]).write_bytes(b"drifted-script")
    with pytest.raises(ValueError, match="script hash mismatch"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
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
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
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

    assert main(
        [
            "--config",
            str(config_path),
            "--gpu",
            "2",
            "--gpu",
            "0",
            "--dry-run",
        ]
    ) == 0

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(row["scene"], row["gpu"], row["frame_count"]) for row in rows] == [
        ("apartment", 2, 1745),
        ("office", 0, 4346),
    ]
    assert not Path(config["frontend_logs_root"]).exists()
    assert all(
        not (
            Path(config["frontend_cache_root"])
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
            "--gpu",
            "1",
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
        "argv": list(
            build_commands(
                config,
                manifest,
                scenes=("office",),
                gpu_ids=(1,),
            )[0].argv
        ),
    }
    assert not Path(config["frontend_logs_root"]).exists()


def test_preflight_only_validates_selected_gpus_without_creating_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        frontend_module,
        "_gpu_inventory",
        lambda: {0: "NVIDIA A40", 1: "NVIDIA A40", 2: "NVIDIA A40"},
    )

    assert main(
        [
            "--config",
            str(config_path),
            "--preflight-only",
            "--gpu",
            "0",
            "--gpu",
            "1",
        ],
        dataset_factory=_StubTesseCdRgbdDataset,
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready"
    assert result["gpu_ids"] == [0, 1]
    assert result["gpu_inventory"] == {
        "0": "NVIDIA A40",
        "1": "NVIDIA A40",
        "2": "NVIDIA A40",
    }
    assert result["output_root"] == str(Path(config["frontend_cache_root"]))
    assert not Path(config["frontend_cache_root"]).exists()
    assert not Path(config["frontend_logs_root"]).exists()


def test_runtime_stages_private_read_only_input_copies(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    source = Path(manifest["scenes"]["apartment"]["root"])
    (source / "results").mkdir()
    source_rgb = source / "results/frame000000.jpg"
    source_rgb.write_bytes(b"rgb")
    command = build_commands(config, manifest, scenes=("apartment",))[0]

    layout = frontend_module._prepare_output_layout((command,))
    frontend_module._verify_output_layout(command, layout)

    output_scene = Path(config["frontend_cache_root"]) / "apartment"
    staged_rgb = output_scene / "results/frame000000.jpg"
    staged_trajectory = output_scene / "traj.txt"
    assert staged_rgb.is_file() and not staged_rgb.is_symlink()
    assert staged_rgb.read_bytes() == source_rgb.read_bytes()
    assert os.lstat(staged_rgb).st_ino != os.lstat(source_rgb).st_ino
    assert os.lstat(staged_rgb).st_nlink == 1
    assert os.lstat(staged_rgb).st_mode & 0o222 == 0
    assert staged_trajectory.is_file() and not staged_trajectory.is_symlink()
    assert staged_trajectory.read_bytes() == (source / "traj.txt").read_bytes()
    assert os.lstat(staged_trajectory).st_mode & 0o222 == 0
    assert not any(source.glob("gsa_detections_oviv2_*"))


def test_inference_aba_on_official_input_is_detected_and_staging_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _fixture(tmp_path)
    source = Path(manifest["scenes"]["apartment"]["root"])
    results = source / "results"
    results.mkdir()
    official = results / "frame000000.jpg"
    official.write_bytes(b"trusted")
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    _, input_binding = frontend_module._load_frozen_input_manifest(config)
    snapshot = frontend_module._capture_run_snapshot(
        config,
        (command,),
        input_binding,
        manifest,
        dataset_factory=_StubTesseCdRgbdDataset,
    )
    layout = frontend_module._prepare_output_layout((command,), snapshot)

    def aba_during_inference(*_args, **_kwargs):
        trusted = results / "trusted.backup"
        official.rename(trusted)
        official.write_bytes(b"attacker")
        official.unlink()
        trusted.rename(official)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(frontend_module.subprocess, "run", aba_during_inference)

    with pytest.raises(ValueError, match="changed after snapshot"):
        frontend_module._run_queue(
            [command],
            config,
            config["manifest_sha256"],
            snapshot,
            manifest,
            _StubTesseCdRgbdDataset,
            layout,
        )

    staged = command.cache_dir.parent / "results/frame000000.jpg"
    assert staged.read_bytes() == b"trusted"
    assert official.read_bytes() == b"trusted"
    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_inference_aba_on_staged_results_rejects_valid_cache_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    source_results = command.source_root / "results"
    source_results.mkdir()
    (source_results / "frame000000.jpg").write_bytes(b"trusted")
    manifest, input_binding = frontend_module._load_frozen_input_manifest(config)
    snapshot = frontend_module._capture_run_snapshot(
        config,
        (command,),
        input_binding,
        manifest,
        dataset_factory=_StubTesseCdRgbdDataset,
    )
    layout = frontend_module._prepare_output_layout((command,), snapshot)
    staged_results = command.cache_dir.parent / "results"
    classes = _classes(command)

    def replace_stage_and_emit_valid_cache(*_args, **_kwargs):
        trusted_results = staged_results.with_name("results.trusted")
        staged_results.rename(trusted_results)
        staged_results.mkdir()
        (staged_results / "frame000000.jpg").write_bytes(b"attacker")
        assert (staged_results / "frame000000.jpg").read_bytes() == b"attacker"
        _write_cache(
            command,
            [_cache_payload(classes), _cache_payload(classes, count=0)],
        )
        (staged_results / "frame000000.jpg").unlink()
        staged_results.rmdir()
        trusted_results.rename(staged_results)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        frontend_module.subprocess,
        "run",
        replace_stage_and_emit_valid_cache,
    )

    with pytest.raises(ValueError, match="staged RGB-D results changed after snapshot"):
        frontend_module._run_queue(
            [command],
            config,
            config["manifest_sha256"],
            snapshot,
            manifest,
            _StubTesseCdRgbdDataset,
            layout,
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_real_dataset_preflight_rejects_empty_scene_directories(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="results directory"):
        main(["--config", str(config_path), "--preflight-only"])


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        ({0: "NVIDIA A40", 1: "NVIDIA A40"}, "at least three"),
        (
            {0: "NVIDIA A40", 1: "NVIDIA RTX 4090", 2: "NVIDIA A40"},
            "0, 1, and 2 must all be NVIDIA A40",
        ),
    ],
)
def test_preflight_rejects_wrong_gpu_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: dict[int, str],
    message: str,
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(frontend_module, "_gpu_inventory", lambda: inventory)

    with pytest.raises(ValueError, match=message):
        main(
            ["--config", str(config_path), "--preflight-only"],
            dataset_factory=_StubTesseCdRgbdDataset,
        )


def test_preflight_rejects_less_than_required_disk_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _fixture(tmp_path)
    config["required_free_bytes"] = 300 * 1024**3
    monkeypatch.setattr(
        frontend_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=config["required_free_bytes"] - 1),
    )

    with pytest.raises(ValueError, match="insufficient free space"):
        frontend_module._validate_output_disk(config)


def test_validator_rejects_dataset_witness_drift_before_manifest(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    manifest, input_binding = frontend_module._load_frozen_input_manifest(config)

    class DriftingDataset(_StubTesseCdRgbdDataset):
        calls = 0

        def __init__(self, *args, **kwargs) -> None:
            type(self).calls += 1
            type(self).generation = type(self).calls
            super().__init__(*args, **kwargs)

    snapshot = frontend_module._capture_run_snapshot(
        config,
        (command,),
        input_binding,
        manifest,
        dataset_factory=DriftingDataset,
    )
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    with pytest.raises(ValueError, match="dataset input witness changed"):
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
            run_snapshot=snapshot,
            manifest=manifest,
            dataset_factory=DriftingDataset,
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_validator_rechecks_dataset_bindings_inside_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])
    timestamps = command.source_root / "timestamps.csv"
    original_publish = frontend_module._publish_json_new

    def drift_before_publish(path, payload, **kwargs) -> None:
        timestamps.write_text("drifted\n", encoding="utf-8")
        original_publish(path, payload, **kwargs)

    monkeypatch.setattr(frontend_module, "_publish_json_new", drift_before_publish)

    with pytest.raises(ValueError, match="changed after snapshot"):
        _validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    assert not (command.cache_dir / "frontend_manifest.json").exists()


def test_preexisting_output_root_is_rejected_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _fixture(tmp_path)
    for scene in SCENES:
        (Path(manifest["scenes"][scene]["root"]) / "results").mkdir()
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    Path(config["frontend_cache_root"]).symlink_to(attacker, target_is_directory=True)
    monkeypatch.setattr(
        frontend_module,
        "_gpu_inventory",
        lambda: {0: "NVIDIA A40", 1: "NVIDIA A40", 2: "NVIDIA A40"},
    )
    called = False

    def forbidden_subprocess(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("external inference must not run")

    monkeypatch.setattr(frontend_module.subprocess, "run", forbidden_subprocess)

    with pytest.raises(ValueError, match="frontend cache root"):
        main(
            ["--config", str(config_path)],
            dataset_factory=_StubTesseCdRgbdDataset,
        )
    assert not called


@pytest.mark.parametrize("unexpected", ["cache", "experiment", "visualization"])
def test_output_layout_rejects_unexpected_paths_before_inference(
    tmp_path: Path,
    unexpected: str,
) -> None:
    config, manifest = _fixture(tmp_path)
    source = Path(manifest["scenes"]["apartment"]["root"])
    (source / "results").mkdir()
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    layout = frontend_module._prepare_output_layout((command,))
    output_scene = command.cache_dir.parent
    paths = {
        "cache": command.cache_dir,
        "experiment": output_scene / "exp_oviv2_tesse_apartment_stage3_v1",
        "visualization": output_scene / "gsa_vis_oviv2_tesse_apartment_stage3_v1",
    }
    paths[unexpected].mkdir()

    with pytest.raises(ValueError, match="must not preexist"):
        frontend_module._verify_output_layout(command, layout)


def test_output_layout_rejects_unexpected_cache_root_member(tmp_path: Path) -> None:
    config, manifest = _fixture(tmp_path)
    source = Path(manifest["scenes"]["apartment"]["root"])
    (source / "results").mkdir()
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    layout = frontend_module._prepare_output_layout((command,))
    (layout.root.path / "unexpected").mkdir()

    with pytest.raises(ValueError, match="cache root contains an unexpected path"):
        frontend_module._verify_output_layout(command, layout)


def test_output_layout_detects_parent_replacement_during_atomic_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, manifest = _fixture(tmp_path)
    config["frontend_cache_root"] = str(tmp_path / "output-parent/frontend-cache")
    source = Path(manifest["scenes"]["apartment"]["root"])
    (source / "results").mkdir()
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    output_root = Path(config["frontend_cache_root"])
    parent = output_root.parent
    parent.mkdir(parents=True)
    moved_parent = parent.with_name("frontend-parent-moved")
    attacker = tmp_path / "attacker-parent"
    attacker.mkdir()
    original_publish = frontend_module._rename_directory_no_replace_at
    replaced = False

    def replace_parent_after_root(directory_fd, source_name, target_name):
        nonlocal replaced
        original_publish(directory_fd, source_name, target_name)
        if target_name == output_root.name and not replaced:
            replaced = True
            parent.rename(moved_parent)
            parent.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(
        frontend_module,
        "_rename_directory_no_replace_at",
        replace_parent_after_root,
    )

    with pytest.raises(ValueError, match="cache parent changed"):
        frontend_module._prepare_output_layout((command,))
    assert replaced
    assert not (attacker / output_root.name).exists()


@pytest.mark.parametrize("replaced_level", ["root", "scene"])
def test_output_layout_fd_chain_never_copies_into_replacement_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_level: str,
) -> None:
    config, manifest = _fixture(tmp_path)
    config["frontend_cache_root"] = str(tmp_path / "output-parent/frontend-cache")
    source = Path(manifest["scenes"]["apartment"]["root"])
    results = source / "results"
    results.mkdir()
    (results / "frame000000.jpg").write_bytes(b"trusted")
    command = build_commands(config, manifest, scenes=("apartment",))[0]
    output_root = Path(config["frontend_cache_root"])
    scene_root = output_root / "apartment"
    moved = (
        output_root.parent / "moved-root"
        if replaced_level == "root"
        else output_root / "moved-scene"
    )
    original_publish = frontend_module._rename_directory_no_replace_at
    replaced = False

    def replace_after_publication(directory_fd, source_name, target_name):
        nonlocal replaced
        original_publish(directory_fd, source_name, target_name)
        if replaced:
            return
        if replaced_level == "root" and target_name == output_root.name:
            replaced = True
            os.rename(
                target_name,
                moved.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(target_name, dir_fd=directory_fd)
        elif replaced_level == "scene" and target_name == command.scene:
            replaced = True
            os.rename(
                target_name,
                moved.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(target_name, dir_fd=directory_fd)

    monkeypatch.setattr(
        frontend_module,
        "_rename_directory_no_replace_at",
        replace_after_publication,
    )

    with pytest.raises(ValueError, match=f"frontend .*{replaced_level}.*changed"):
        frontend_module._prepare_output_layout((command,))

    assert replaced
    assert not (scene_root / "results/frame000000.jpg").exists()
    if replaced_level == "root":
        assert (moved / "apartment/results/frame000000.jpg").read_bytes() == b"trusted"
    else:
        assert (moved / "results/frame000000.jpg").read_bytes() == b"trusted"


def test_formal_modes_cannot_skip_scene_or_change_execution_gpus(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["manifest_sha256"] = _sha256(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must include both"):
        main(
            ["--config", str(config_path), "--preflight-only", "--scene", "office"],
            dataset_factory=_StubTesseCdRgbdDataset,
        )
    with pytest.raises(ValueError, match="must use GPUs 0 and 1"):
        main(
            [
                "--config",
                str(config_path),
                "--preflight-only",
                "--gpu",
                "0",
                "--gpu",
                "2",
            ],
            dataset_factory=_StubTesseCdRgbdDataset,
        )


def test_disk_requirement_cannot_be_lowered_by_custom_config(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    config["required_free_bytes"] = 1

    with pytest.raises(ValueError, match="exactly 300 GiB"):
        frontend_module._validate_output_disk(config)


def test_formal_config_binds_real_inputs_and_contains_no_evaluation_paths() -> None:
    config_path = Path("configs/oviv2_tesse_cd_frontend_stage3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert config["manifest_sha256"] == _sha256(manifest_path)
    assert config["frontend_cache_root"] == (
        "/home/ww/oviovo_frontend_cache/tesse_cd_native_v1"
    )
    assert config["required_free_bytes"] == 300 * 1024**3
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
