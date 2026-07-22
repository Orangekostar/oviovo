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
        "frontend_cache_root": str(tmp_path / "frontend-cache"),
        "minimum_free_bytes": 1,
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


def test_validator_binds_complete_cache_provenance_and_prefix_digest(
    tmp_path: Path,
) -> None:
    config, command = _small_command(tmp_path)
    classes = _classes(command)
    _write_cache(command, [_cache_payload(classes), _cache_payload(classes, count=0)])

    result = validate_frontend_cache(
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
        validate_frontend_cache(
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

    result = validate_frontend_cache(
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

    validate_frontend_cache(
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
        validate_frontend_cache(
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
        validate_frontend_cache(
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
    monkeypatch.setattr(frontend_module, "_visible_gpu_ids", lambda: {0, 1})
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
        ]
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
        validate_frontend_cache(
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
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )
    manifest_path = command.cache_dir / "frontend_manifest.json"
    assert not manifest_path.exists()
    assert not list(command.cache_dir.glob(".frontend_manifest.json.*.tmp"))

    result = validate_frontend_cache(
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
        validate_frontend_cache(
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
        validate_frontend_cache(
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
        validate_frontend_cache(
            replace(command, source_frame_ids=(0, 2)),
            config,
            input_manifest_sha256=config["manifest_sha256"],
        )

    Path(config["frontend"]["script"]).write_bytes(b"drifted-script")
    with pytest.raises(ValueError, match="script hash mismatch"):
        validate_frontend_cache(
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
        validate_frontend_cache(
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
    monkeypatch.setattr(frontend_module, "_visible_gpu_ids", lambda: {0, 1, 2})

    assert main(
        [
            "--config",
            str(config_path),
            "--preflight-only",
            "--gpu",
            "0",
            "--gpu",
            "1",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready"
    assert result["gpu_ids"] == [0, 1]
    assert result["output_root"] == str(Path(config["frontend_cache_root"]))
    assert not Path(config["frontend_cache_root"]).exists()
    assert not Path(config["frontend_logs_root"]).exists()


def test_runtime_view_links_inputs_only_inside_independent_cache_root(
    tmp_path: Path,
) -> None:
    config, manifest = _fixture(tmp_path)
    source = Path(manifest["scenes"]["apartment"]["root"])
    (source / "results").mkdir()
    (source / "traj.txt").write_text("pose\n", encoding="utf-8")
    command = build_commands(config, manifest, scenes=("apartment",))[0]

    frontend_module._prepare_input_view(command)

    output_scene = Path(config["frontend_cache_root"]) / "apartment"
    assert (output_scene / "results").is_symlink()
    assert (output_scene / "results").resolve() == (source / "results").resolve()
    assert (output_scene / "traj.txt").is_symlink()
    assert (output_scene / "traj.txt").resolve() == (source / "traj.txt").resolve()
    assert not any(source.glob("gsa_detections_oviv2_*"))


def test_formal_config_binds_real_inputs_and_contains_no_evaluation_paths() -> None:
    config_path = Path("configs/oviv2_tesse_cd_frontend_stage3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert config["manifest_sha256"] == _sha256(manifest_path)
    assert config["frontend_cache_root"] == (
        "/home/ww/oviovo_frontend_cache/tesse_cd_native_v1"
    )
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
