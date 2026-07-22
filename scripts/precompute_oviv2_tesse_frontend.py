#!/usr/bin/env python3
"""Generate and verify frozen OVIV2 object-frontend caches for TESSE-CD."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.oviv2 import hybrid_cache


SCENES = ("apartment", "office")
SCENE_GPUS = {"apartment": 0, "office": 1}
SCENE_FRAME_COUNTS = {"apartment": 1745, "office": 4346}
IMAGE_SHAPE = (480, 720)
REQUIRED_FREE_BYTES = 300 * 1024**3
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_CACHE_FIELDS = {
    "mask",
    "xyxy",
    "confidence",
    "class_id",
    "classes",
    "image_feats",
    "text_feats",
}
ALLOWED_CACHE_FIELDS = REQUIRED_CACHE_FIELDS | {"image_crops"}
PROVENANCE_PATH_KEYS = {
    "script": "script",
    "hydra_config": "hydra_config_path",
    "dataset_config": "dataset_config",
    "yolo_model": "yolo_model_path",
    "yolo_clip_model": "yolo_clip_model_path",
    "mobile_sam_model": "mobile_sam_model_path",
    "clip_model": "clip_pretrained_path",
}


@dataclass(frozen=True)
class FrontendCommand:
    scene: str
    gpu_id: int
    argv: tuple[str, ...]
    cache_dir: Path
    log_dir: Path
    algorithm_hash: str
    frame_count: int
    source_frame_ids: tuple[int, ...]
    image_shape: tuple[int, int]
    classes_file: Path
    classes_sha256: str
    source_root: Path


class FrontendManifestPublicationUncertainError(RuntimeError):
    """The complete manifest exists, but parent-directory durability is unknown."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"frontend manifest publication durability is uncertain: {path}")


class FrontendLayoutCleanupUncertainError(RuntimeError):
    """A bound layout could not be durably quarantined and removed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"frontend layout cleanup is uncertain: {path}")


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _FileBinding:
    path: Path
    sha256: str
    identity: _FileIdentity


@dataclass(frozen=True)
class _DatasetCapture:
    witness: Mapping[str, Any]
    bindings: tuple[_FileBinding, ...]
    directories: tuple[_DirectoryContentBinding, ...]
    fixture_only: bool


@dataclass(frozen=True)
class _RunSnapshot:
    input_manifest: _FileBinding
    provenance: Mapping[str, _FileBinding]
    vocabularies: Mapping[str, _FileBinding]
    classes: Mapping[str, tuple[str, ...]]
    datasets: Mapping[str, _DatasetCapture]

    @property
    def bindings(self) -> tuple[_FileBinding, ...]:
        unique: dict[Path, _FileBinding] = {self.input_manifest.path: self.input_manifest}
        for binding in (*self.provenance.values(), *self.vocabularies.values()):
            unique.setdefault(binding.path, binding)
        for capture in self.datasets.values():
            for binding in capture.bindings:
                unique.setdefault(binding.path, binding)
        return tuple(unique.values())


@dataclass(frozen=True)
class _DirectoryBinding:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _DirectoryContentBinding:
    path: Path
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _OutputLayout:
    parent: _DirectoryBinding
    root: _DirectoryBinding
    scenes: Mapping[str, _DirectoryBinding]
    results: Mapping[str, _DirectoryContentBinding]
    input_copies: Mapping[Path, _FileBinding]


DatasetFactory = Callable[[Path, str, Path, Path], Any]


class _IgnoredImageCrop:
    """Inert target for unused PIL image crops embedded by the upstream script."""

    def __new__(cls, *_args: object, **_kwargs: object) -> "_IgnoredImageCrop":
        return super().__new__(cls)

    def __setstate__(self, _state: object) -> None:
        return None


class _TesseFrontendRestrictedUnpickler(hybrid_cache._RestrictedUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("PIL.Image", "Image"):
            return _IgnoredImageCrop
        return super().find_class(module, name)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hybrid_cache.sha256(path)


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Path(os.path.abspath(path))


def _file_identity(metadata: os.stat_result) -> _FileIdentity:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("snapshot input must be a regular file")
    return _FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _assert_binding_unchanged(binding: _FileBinding) -> None:
    try:
        hybrid_cache._assert_no_symlink(binding.path, field_name="snapshot input")
        current = _file_identity(os.lstat(binding.path))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"file changed after snapshot: {binding.path}") from exc
    if current != binding.identity:
        raise ValueError(f"file changed after snapshot: {binding.path}")


def _assert_bindings_unchanged(bindings: Iterable[_FileBinding]) -> None:
    seen: set[Path] = set()
    for binding in bindings:
        if binding.path in seen:
            continue
        seen.add(binding.path)
        _assert_binding_unchanged(binding)


def _snapshot_regular_file(
    path: Path,
    *,
    field_name: str,
    retain_bytes: bool = False,
) -> tuple[_FileBinding, bytes | None]:
    source = hybrid_cache._open_regular_input(path, field_name=field_name)
    retained = bytearray() if retain_bytes else None
    digest = hashlib.sha256()
    with source:
        before = _file_identity(os.fstat(source.fileno()))
        while block := source.read(1024 * 1024):
            digest.update(block)
            if retained is not None:
                retained.extend(block)
        after = _file_identity(os.fstat(source.fileno()))
    if before != after:
        raise ValueError(f"file changed during snapshot: {path}")
    binding = _FileBinding(path=path, sha256=digest.hexdigest(), identity=after)
    _assert_binding_unchanged(binding)
    return binding, None if retained is None else bytes(retained)


def _load_frozen_input_manifest(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], _FileBinding]:
    manifest_path = _resolve_path(config.get("manifest"))
    binding, data = _snapshot_regular_file(
        manifest_path,
        field_name="TESSE-CD input manifest",
        retain_bytes=True,
    )
    expected = config.get("manifest_sha256")
    if (
        not isinstance(expected, str)
        or SHA256_PATTERN.fullmatch(expected) is None
        or binding.sha256 != expected
    ):
        raise ValueError("input manifest hash mismatch")
    assert data is not None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("TESSE-CD input manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("TESSE-CD input manifest root must be an object")
    return payload, binding


def _manifest_file_snapshot(
    value: object,
    *,
    role: str,
    retain_bytes: bool = False,
) -> tuple[_FileBinding, bytes | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} binding must be an object")
    path = _resolve_path(value.get("path"))
    expected = value.get("sha256")
    binding, data = _snapshot_regular_file(
        path,
        field_name=role,
        retain_bytes=retain_bytes,
    )
    if (
        not isinstance(expected, str)
        or SHA256_PATTERN.fullmatch(expected) is None
        or binding.sha256 != expected
    ):
        raise ValueError(f"{role} hash mismatch")
    return binding, data


def _record_witness(record: object, *, role: str) -> dict[str, Any]:
    try:
        pose = np.asarray(record.camera_to_world, dtype=np.float64)
        values = {
            "frame_index": int(record.frame_index),
            "timestamp_ns": int(record.timestamp_ns),
            "relative_timestamp_ns": int(record.relative_timestamp_ns),
            "rgb_path": str(Path(record.rgb_path).absolute()),
            "depth_path": str(Path(record.depth_path).absolute()),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} is invalid") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{role} pose is invalid")
    values["camera_to_world_sha256"] = _json_hash(pose.tolist())
    return values


def _capture_dataset_witness(
    manifest: Mapping[str, Any],
    command: FrontendCommand,
    *,
    dataset_factory: DatasetFactory,
) -> _DatasetCapture:
    records = _scene_records(manifest)
    record = records.get(command.scene)
    if not isinstance(record, Mapping):
        raise ValueError(f"scene {command.scene} record must be an object")
    schedule_record = manifest.get("schedule_manifest")
    source_record = manifest.get("source_manifest")
    camera_record = manifest.get("camera")
    export_record = record.get("export_manifest")
    timestamps_record = record.get("timestamps")
    trajectory_record = record.get("trajectory")
    for value, role in (
        (schedule_record, "schedule manifest"),
        (source_record, "source manifest"),
        (camera_record, "camera manifest"),
        (export_record, f"{command.scene} export manifest"),
        (timestamps_record, f"{command.scene} timestamps"),
        (trajectory_record, f"{command.scene} trajectory"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{role} binding must be an object")

    export_path = _resolve_path(export_record.get("path"))
    schedule_path = _resolve_path(schedule_record.get("path"))
    dataset = dataset_factory(
        command.source_root,
        command.scene,
        export_path,
        schedule_path,
    )
    expected_count = record.get("frame_count")
    if type(expected_count) is not int or len(dataset) != expected_count:
        raise ValueError(f"scene {command.scene} validated dataset frame count mismatch")
    if expected_count <= 0:
        raise ValueError(f"scene {command.scene} validated dataset is empty")

    expected_paths = {
        "root": command.source_root,
        "export_manifest_path": export_path,
        "schedule_manifest_path": schedule_path,
        "camera_path": _resolve_path(camera_record.get("path")),
        "timestamps_path": _resolve_path(timestamps_record.get("path")),
        "trajectory_path": _resolve_path(trajectory_record.get("path")),
    }
    for attribute, expected_path in expected_paths.items():
        actual_path = Path(os.path.abspath(Path(getattr(dataset, attribute))))
        if actual_path != expected_path:
            raise ValueError(
                f"scene {command.scene} dataset {attribute} does not match manifest"
            )

    schedule_binding, _ = _manifest_file_snapshot(
        schedule_record,
        role="schedule manifest",
    )
    source_binding, _ = _manifest_file_snapshot(
        source_record,
        role="source manifest",
    )
    camera_binding, _ = _manifest_file_snapshot(
        camera_record,
        role="camera manifest",
    )
    export_binding, export_bytes = _manifest_file_snapshot(
        export_record,
        role=f"{command.scene} export manifest",
        retain_bytes=True,
    )
    timestamps_binding, _ = _manifest_file_snapshot(
        timestamps_record,
        role=f"{command.scene} timestamps",
    )
    trajectory_binding, _ = _manifest_file_snapshot(
        trajectory_record,
        role=f"{command.scene} trajectory",
    )
    assert export_bytes is not None
    try:
        export_payload = json.loads(export_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{command.scene} export manifest is invalid") from exc
    if not isinstance(export_payload, Mapping):
        raise ValueError(f"{command.scene} export manifest must be an object")
    combined = export_record.get("combined_output_sha256")
    file_hash_count = export_record.get("file_hash_count")
    if (
        not isinstance(combined, str)
        or SHA256_PATTERN.fullmatch(combined) is None
        or export_payload.get("combined_output_sha256") != combined
        or type(file_hash_count) is not int
        or export_payload.get("file_hash_count") != file_hash_count
    ):
        raise ValueError(f"{command.scene} export manifest content binding mismatch")

    output_bindings = [camera_binding, timestamps_binding, trajectory_binding]
    extra_fixture_bindings: list[_FileBinding] = []
    fixture_only = bool(getattr(dataset, "_fixture_only", False))
    if fixture_only:
        computed_combined = combined
        computed_file_count = file_hash_count
        fixture_results = command.source_root / "results"
        if fixture_results.is_dir():
            for path in sorted(fixture_results.iterdir()):
                binding, _ = _snapshot_regular_file(
                    path,
                    field_name=f"{command.scene} fixture RGB-D input",
                )
                extra_fixture_bindings.append(binding)
    else:
        for dataset_record in dataset.records:
            for kind in ("rgb_path", "depth_path"):
                path = Path(getattr(dataset_record, kind))
                binding, _ = _snapshot_regular_file(
                    path,
                    field_name=f"{command.scene} validated RGB-D input",
                )
                output_bindings.append(binding)
        output_root = camera_binding.path.parent
        output_hashes: list[tuple[str, str]] = []
        for binding in output_bindings:
            try:
                relative = str(binding.path.relative_to(output_root))
            except ValueError as exc:
                raise ValueError("validated RGB-D input escapes the export root") from exc
            output_hashes.append((relative, binding.sha256))
        digest = hashlib.sha256()
        for relative, checksum in sorted(output_hashes):
            digest.update(
                relative.encode("utf-8")
                + b"\0"
                + checksum.encode("ascii")
                + b"\n"
            )
        computed_combined = digest.hexdigest()
        computed_file_count = len(output_bindings)
        if (
            computed_combined != combined
            or computed_file_count != file_hash_count
        ):
            raise ValueError(f"{command.scene} exported RGB-D binding mismatch")

    try:
        intrinsics = dataset.intrinsics
        intrinsics_values = {
            name: float(getattr(intrinsics, name))
            for name in ("fx", "fy", "cx", "cy")
        }
        intrinsics_values.update(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
        )
        first = _record_witness(dataset.records[0], role="first dataset record")
        last = _record_witness(dataset.records[-1], role="last dataset record")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"scene {command.scene} validated dataset metadata is invalid") from exc
    if not all(
        np.isfinite(intrinsics_values[name]) for name in ("fx", "fy", "cx", "cy")
    ):
        raise ValueError(f"scene {command.scene} dataset intrinsics are invalid")

    witness = {
        "schema_version": 1,
        "scene": command.scene,
        "root": str(command.source_root),
        "validated_frame_count": expected_count,
        "intrinsics": intrinsics_values,
        "first_record": first,
        "last_record": last,
        "schedule_manifest": {
            "path": str(schedule_binding.path),
            "sha256": schedule_binding.sha256,
        },
        "source_manifest": {
            "path": str(source_binding.path),
            "sha256": source_binding.sha256,
        },
        "camera_manifest": {
            "path": str(camera_binding.path),
            "sha256": camera_binding.sha256,
        },
        "export_manifest": {
            "path": str(export_binding.path),
            "sha256": export_binding.sha256,
            "combined_output_sha256": combined,
            "file_hash_count": file_hash_count,
            "validated_combined_output_sha256": computed_combined,
            "validated_file_hash_count": computed_file_count,
        },
        "timestamps": {
            "path": str(timestamps_binding.path),
            "sha256": timestamps_binding.sha256,
        },
        "trajectory": {
            "path": str(trajectory_binding.path),
            "sha256": trajectory_binding.sha256,
        },
    }
    return _DatasetCapture(
        witness=witness,
        bindings=(
            schedule_binding,
            source_binding,
            export_binding,
            *output_bindings,
            *extra_fixture_bindings,
        ),
        directories=(
            _bind_directory_content(command.source_root, role="TESSE-CD scene root"),
            _bind_directory_content(
                command.source_root / "results",
                role="TESSE-CD RGB-D results",
            ),
        )
        if (command.source_root / "results").is_dir()
        else (
            _bind_directory_content(command.source_root, role="TESSE-CD scene root"),
        ),
        fixture_only=fixture_only,
    )


def _capture_run_snapshot(
    config: Mapping[str, Any],
    commands: Iterable[FrontendCommand],
    input_manifest: _FileBinding,
    manifest: Mapping[str, Any],
    *,
    dataset_factory: DatasetFactory,
) -> _RunSnapshot:
    commands = tuple(commands)
    expected_input = config.get("manifest_sha256")
    if input_manifest.sha256 != expected_input:
        raise ValueError("input manifest hash mismatch")
    frontend = config.get("frontend")
    if not isinstance(frontend, Mapping):
        raise ValueError("frontend config must be an object")
    expected_provenance = frontend.get("provenance_sha256")
    if not isinstance(expected_provenance, Mapping) or set(expected_provenance) != set(
        PROVENANCE_PATH_KEYS
    ):
        raise ValueError("frontend provenance_sha256 keys are not frozen")

    by_path: dict[Path, _FileBinding] = {}
    provenance: dict[str, _FileBinding] = {}
    for name, config_key in PROVENANCE_PATH_KEYS.items():
        path = _resolve_path(frontend.get(config_key))
        binding = by_path.get(path)
        if binding is None:
            binding, _ = _snapshot_regular_file(path, field_name=name)
            by_path[path] = binding
        expected = expected_provenance[name]
        if (
            not isinstance(expected, str)
            or SHA256_PATTERN.fullmatch(expected) is None
            or binding.sha256 != expected
        ):
            raise ValueError(f"{name} hash mismatch")
        provenance[name] = binding

    vocabularies: dict[str, _FileBinding] = {}
    classes: dict[str, tuple[str, ...]] = {}
    for command in commands:
        binding, data = _snapshot_regular_file(
            command.classes_file,
            field_name=f"{command.scene} vocabulary",
            retain_bytes=True,
        )
        if binding.sha256 != command.classes_sha256:
            raise ValueError(f"scene {command.scene} vocabulary hash mismatch")
        assert data is not None
        try:
            lines = data.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"scene {command.scene} vocabulary is not UTF-8") from exc
        normalized = hybrid_cache._classes(
            [line.strip() for line in lines if line.strip()]
        )
        vocabularies[command.scene] = binding
        classes[command.scene] = normalized

    snapshot = _RunSnapshot(
        input_manifest=input_manifest,
        provenance=provenance,
        vocabularies=vocabularies,
        classes=classes,
        datasets={
            command.scene: _capture_dataset_witness(
                manifest,
                command,
                dataset_factory=dataset_factory,
            )
            for command in commands
        },
    )
    return snapshot


def _strict_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _scene_records(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if manifest.get("dataset") != "TESSE-CD":
        raise ValueError("manifest dataset must be TESSE-CD")
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping) or (
        camera.get("height"), camera.get("width")
    ) != IMAGE_SHAPE:
        raise ValueError("TESSE-CD camera must be frozen at 480x720")
    records = manifest.get("scenes")
    if not isinstance(records, Mapping) or tuple(records) != SCENES:
        raise ValueError("manifest must contain the frozen TESSE-CD scene order")
    return records


def build_commands(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    scenes: Iterable[str] | None = None,
    gpu_ids: tuple[int, ...] | None = None,
) -> list[FrontendCommand]:
    records = _scene_records(manifest)
    requested_values = SCENES if scenes is None else tuple(scenes)
    requested = set(requested_values)
    unknown = requested - set(SCENES)
    if unknown:
        raise ValueError(f"unknown TESSE-CD scene: {', '.join(sorted(unknown))}")
    requested_scenes = tuple(scene for scene in SCENES if scene in requested)
    selected_gpu_ids = (
        tuple(SCENE_GPUS[scene] for scene in requested_scenes)
        if gpu_ids is None
        else gpu_ids
    )
    if len(selected_gpu_ids) != len(requested_scenes):
        raise ValueError("exactly one GPU ID per requested scene is required")
    if any(type(value) is not int or value < 0 for value in selected_gpu_ids):
        raise ValueError("GPU IDs must be non-negative integers")
    gpu_by_scene = dict(zip(requested_scenes, selected_gpu_ids, strict=True))
    frontend = config.get("frontend")
    if not isinstance(frontend, Mapping):
        raise ValueError("frontend config must be an object")
    dimensions = (
        _strict_int(frontend.get("desired_height"), name="desired_height"),
        _strict_int(frontend.get("desired_width"), name="desired_width"),
    )
    if dimensions != IMAGE_SHAPE:
        raise ValueError("TESSE-CD frontend image shape must be 480x720")
    logs_root = _resolve_path(config.get("frontend_logs_root"))
    output_root = _resolve_path(config.get("frontend_cache_root"))

    commands: list[FrontendCommand] = []
    for scene in SCENES:
        if scene not in requested:
            continue
        record = records[scene]
        if not isinstance(record, Mapping):
            raise ValueError(f"scene {scene} record must be an object")
        frame_count = _strict_int(record.get("frame_count"), name=f"{scene} frame_count")
        if frame_count != SCENE_FRAME_COUNTS[scene]:
            raise ValueError(f"scene {scene} frame_count is not frozen")
        if tuple(record.get("image_shape", ())) != IMAGE_SHAPE:
            raise ValueError(f"scene {scene} image shape is not 480x720")
        selection = record.get("source_frame_ids")
        if not isinstance(selection, Mapping) or selection != {
            "start": 0,
            "stop_exclusive": frame_count,
            "stride": 1,
        }:
            raise ValueError(f"scene {scene} source frame IDs must equal range(frame_count)")
        root = _resolve_path(record.get("root"))
        if root.is_symlink() or not root.is_dir() or root.name != scene:
            raise ValueError(f"scene {scene} root path must end with the scene name")
        if output_root.is_relative_to(root) or root.is_relative_to(output_root):
            raise ValueError("frontend cache root must be independent of source RGB-D")
        vocabulary = record.get("vocabulary")
        if not isinstance(vocabulary, Mapping):
            raise ValueError(f"scene {scene} vocabulary must be an object")
        classes_file = _resolve_path(vocabulary.get("txt_path"))
        classes_sha256 = str(vocabulary.get("txt_sha256", ""))
        if classes_file.is_symlink() or not classes_file.is_file():
            raise ValueError(
                f"scene {scene} vocabulary must be a regular non-symlink file"
            )
        if _sha256(classes_file) != classes_sha256:
            raise ValueError(f"scene {scene} vocabulary hash mismatch")
        variant = str(frontend["gsa_variant_template"]).format(scene=scene)
        suffix = str(frontend["exp_suffix_template"]).format(scene=scene)
        if not variant or Path(variant).name != variant:
            raise ValueError(f"scene {scene} gsa variant is not a safe path component")
        if not suffix or Path(suffix).name != suffix:
            raise ValueError(f"scene {scene} experiment suffix is not a safe path component")
        log_dir = logs_root / scene
        descriptor = {
            "image_shape": list(IMAGE_SHAPE),
            "frontend_algorithm": {
                "clip_model_card": frontend["clip_model_card"],
                "desired_height": frontend["desired_height"],
                "desired_width": frontend["desired_width"],
                "hydra_config_name": frontend["hydra_config_name"],
                "provenance_sha256": frontend["provenance_sha256"],
            },
        }
        argv = (
            str(frontend["python"]),
            str(frontend["script"]),
            f"--config-name={frontend['hydra_config_name']}",
            f"dataset_root={output_root}",
            f"dataset_config={frontend['dataset_config']}",
            f"scene_id={scene}",
            "start=0",
            f"end={frame_count}",
            "stride=1",
            "desired_height=480",
            "desired_width=720",
            f"classes_file={classes_file}",
            f"gsa_variant={variant}",
            f"exp_suffix={suffix}",
            f"device={frontend['device']}",
            "save_video=false",
            f"yolo_model_path={frontend['yolo_model_path']}",
            f"mobile_sam_model_path={frontend['mobile_sam_model_path']}",
            f"clip_model_card={frontend['clip_model_card']}",
            f"clip_pretrained_path={frontend['clip_pretrained_path']}",
            f"hydra.run.dir={log_dir / 'hydra'}",
            "hydra.job.chdir=false",
        )
        commands.append(
            FrontendCommand(
                scene=scene,
                gpu_id=gpu_by_scene[scene],
                argv=argv,
                cache_dir=output_root / scene / f"gsa_detections_{variant}",
                log_dir=log_dir,
                algorithm_hash=_json_hash(descriptor),
                frame_count=frame_count,
                source_frame_ids=tuple(range(frame_count)),
                image_shape=IMAGE_SHAPE,
                classes_file=classes_file,
                classes_sha256=classes_sha256,
                source_root=root,
            )
        )
    return commands


def build_environment(
    gpu_id: int,
    conceptgraphs_root: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("GPU ID must be a non-negative integer")
    environment = dict(os.environ if base is None else base)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{conceptgraphs_root}{os.pathsep}{existing}"
        if existing
        else str(conceptgraphs_root)
    )
    return environment


def warm_shared_clip_cache(frontend: Mapping[str, Any]) -> str:
    model_path = _resolve_path(frontend.get("yolo_clip_model_path"))
    provenance = frontend.get("provenance_sha256")
    if not isinstance(provenance, Mapping):
        raise ValueError("frontend provenance_sha256 must be an object")
    expected = str(provenance.get("yolo_clip_model", ""))
    if model_path.is_file():
        binding, _ = _snapshot_regular_file(
            model_path,
            field_name="shared YOLO-World CLIP model",
        )
        if binding.sha256 != expected:
            raise ValueError("shared YOLO-World CLIP model checksum mismatch")
        return expected
    code = (
        "from clip.clip import _MODELS, _download; "
        "_download(_MODELS['ViT-B/32'], download_root=r'{}')"
    ).format(model_path.parent)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(frontend["python"]), "-c", code], check=True)
    if not model_path.is_file():
        raise ValueError("shared YOLO-World CLIP model checksum mismatch after warmup")
    binding, _ = _snapshot_regular_file(
        model_path,
        field_name="shared YOLO-World CLIP model",
    )
    if binding.sha256 != expected:
        raise ValueError("shared YOLO-World CLIP model checksum mismatch after warmup")
    return expected


def _regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")


def _cache_prefix_sha256(cache_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for cache_index, checksum in enumerate(cache_hashes.values()):
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_new(
    path: Path,
    payload: Mapping[str, Any],
    *,
    before_link: Callable[[], None] | None = None,
) -> None:
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if before_link is not None:
            before_link()
        os.link(temporary, path, follow_symlinks=False)
        published = True
        try:
            _fsync_directory(path.parent)
        except BaseException as exc:
            raise FrontendManifestPublicationUncertainError(path) from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                if not published:
                    raise


def _read_cache_snapshot(path: Path) -> tuple[object, _FileBinding]:
    source = hybrid_cache._open_regular_input(path, field_name="frontend cache input")
    try:
        with source, tempfile.SpooledTemporaryFile(
            max_size=hybrid_cache._SNAPSHOT_MEMORY_LIMIT_BYTES,
            mode="w+b",
        ) as snapshot:
            before = _file_identity(os.fstat(source.fileno()))
            digest = hashlib.sha256()
            while block := source.read(1024 * 1024):
                digest.update(block)
                snapshot.write(block)
            after = _file_identity(os.fstat(source.fileno()))
            if before != after:
                raise ValueError(f"file changed during snapshot: {path}")
            snapshot.seek(0)
            with gzip.GzipFile(fileobj=snapshot, mode="rb") as compressed:
                payload = _TesseFrontendRestrictedUnpickler(compressed).load()
                if compressed.read(1):
                    raise ValueError("frontend cache payload has trailing data")
    except ValueError:
        raise
    except (
        EOFError,
        OSError,
        pickle.PickleError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
    ) as exc:
        message = str(exc)
        if "unsafe pickle" in message:
            raise ValueError(message) from exc
        raise ValueError("invalid frontend cache gzip or pickle payload") from exc
    binding = _FileBinding(path=path, sha256=digest.hexdigest(), identity=after)
    _assert_binding_unchanged(binding)
    return payload, binding


def _validate_cache_payload(
    payload: object,
    *,
    classes: tuple[str, ...],
    image_shape: tuple[int, int],
    path: Path,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"frontend cache payload must be a dictionary: {path}")
    keys = set(payload)
    if not REQUIRED_CACHE_FIELDS <= keys or not keys <= ALLOWED_CACHE_FIELDS:
        raise ValueError(f"frontend cache payload has invalid required fields: {path}")
    masks = hybrid_cache._numeric_ndarray(payload["mask"], "mask")
    boxes = hybrid_cache._numeric_ndarray(payload["xyxy"], "xyxy")
    confidences = hybrid_cache._numeric_ndarray(payload["confidence"], "confidence")
    class_ids = hybrid_cache._integer_ndarray(payload["class_id"], "class_id")
    image_features = hybrid_cache._numeric_ndarray(payload["image_feats"], "image_feats")
    text_features = hybrid_cache._numeric_ndarray(payload["text_feats"], "text_feats")
    cached_classes = hybrid_cache._classes(payload["classes"])
    count = masks.shape[0] if masks.ndim == 3 else -1
    height, width = image_shape
    if masks.shape != (count, height, width) or boxes.shape != (count, 4):
        raise ValueError(f"frontend cache mask/box shape mismatch: {path}")
    if confidences.shape != (count,) or class_ids.shape != (count,):
        raise ValueError(f"frontend cache vector length mismatch: {path}")
    if cached_classes != classes:
        raise ValueError(f"frontend cache class order mismatch: {path}")
    if count and (class_ids.min() < 0 or class_ids.max() >= len(classes)):
        raise ValueError(f"frontend cache class ID out of range: {path}")
    for name, features in (
        ("image_feats", image_features),
        ("text_feats", text_features),
    ):
        if (
            features.ndim != 2
            or features.shape[0] != count
            or features.shape[1] <= 0
        ):
            raise ValueError(f"frontend cache {name} is invalid: {path}")
        if count and np.any(np.linalg.norm(features.astype(np.float64), axis=1) == 0.0):
            raise ValueError(f"frontend cache {name} contains a zero row: {path}")
    if image_features.shape[1:] != text_features.shape[1:]:
        raise ValueError(f"frontend cache feature dimensions do not match: {path}")
    for name, values in (
        ("mask", masks),
        ("xyxy", boxes),
        ("confidence", confidences),
        ("image_feats", image_features),
        ("text_feats", text_features),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"frontend cache {name} must be finite: {path}")
    crops = payload.get("image_crops", [])
    if not isinstance(crops, (list, tuple)):
        raise ValueError(f"frontend cache image_crops are invalid: {path}")
    for value in crops:
        if isinstance(value, _IgnoredImageCrop):
            continue
        hybrid_cache._numeric_ndarray(value, "image_crops item")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _gpu_inventory() -> dict[int, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        rows = [line.split(",", 1) for line in result.stdout.splitlines() if line.strip()]
        inventory = {int(index.strip()): name.strip() for index, name in rows}
    except (TypeError, ValueError) as exc:
        raise ValueError("nvidia-smi returned an invalid GPU inventory") from exc
    if len(inventory) != len(rows):
        raise ValueError("nvidia-smi returned duplicate GPU indices")
    return inventory


def _validate_output_disk(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _resolve_path(config.get("frontend_cache_root"))
    hybrid_cache._assert_no_symlink(output_root, field_name="frontend cache root")
    current = output_root
    while not current.exists():
        if current == current.parent:
            raise ValueError("frontend cache root has no existing ancestor")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError("frontend cache root ancestor must be a regular directory")
    if not os.access(current, os.W_OK | os.X_OK):
        raise ValueError("frontend cache root ancestor is not writable")
    usage = shutil.disk_usage(current)
    required = _strict_int(
        config.get("required_free_bytes"),
        name="required_free_bytes",
    )
    if required != REQUIRED_FREE_BYTES:
        raise ValueError("required_free_bytes must be exactly 300 GiB")
    if usage.free < required:
        raise ValueError("frontend cache disk has insufficient free space")
    return {
        "output_root": str(output_root),
        "free_bytes": usage.free,
        "required_free_bytes": required,
    }


def _assert_run_snapshot_unchanged(run_snapshot: _RunSnapshot) -> None:
    _assert_bindings_unchanged(run_snapshot.bindings)
    for capture in run_snapshot.datasets.values():
        for binding in capture.directories:
            _verify_directory_content(binding, role="TESSE-CD input directory")


def _preflight(
    commands: list[FrontendCommand],
    config: Mapping[str, Any],
    run_snapshot: _RunSnapshot,
) -> dict[str, Any]:
    requested = {command.gpu_id for command in commands}
    inventory = _gpu_inventory()
    if len(inventory) < 3:
        raise ValueError("TESSE-CD frontend requires at least three visible GPUs")
    if any(inventory.get(index) != "NVIDIA A40" for index in (0, 1, 2)):
        raise ValueError("GPUs 0, 1, and 2 must all be NVIDIA A40")
    missing = requested - set(inventory)
    if missing:
        raise ValueError(f"requested GPU is not visible: {', '.join(map(str, sorted(missing)))}")
    disk = _validate_output_disk(config)
    _assert_run_snapshot_unchanged(run_snapshot)
    return {
        "status": "ready",
        "gpu_ids": [command.gpu_id for command in commands],
        "gpu_inventory": {str(index): name for index, name in sorted(inventory.items())},
        **disk,
    }


def validate_frontend_cache(
    command: FrontendCommand,
    config: Mapping[str, Any],
    *,
    input_manifest_sha256: str,
    run_snapshot: _RunSnapshot | None = None,
    manifest: Mapping[str, Any] | None = None,
    dataset_factory: DatasetFactory = TesseCdRgbdDataset,
) -> dict[str, Any]:
    if command.scene not in SCENES:
        raise ValueError("unknown TESSE-CD scene in frontend command")
    if command.source_frame_ids != tuple(range(command.frame_count)):
        raise ValueError("frontend source frame IDs must equal range(frame_count)")
    if command.image_shape != IMAGE_SHAPE:
        raise ValueError("frontend command image shape must be 480x720")
    recorded_manifest_sha256 = config.get("manifest_sha256")
    if (
        not isinstance(input_manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(input_manifest_sha256) is None
        or input_manifest_sha256 != recorded_manifest_sha256
    ):
        raise ValueError("input manifest hash mismatch")
    if manifest is None:
        manifest, loaded_input_binding = _load_frozen_input_manifest(config)
    else:
        loaded_input_binding = None
    if run_snapshot is None:
        if loaded_input_binding is None:
            loaded_manifest, loaded_input_binding = _load_frozen_input_manifest(config)
            if loaded_manifest != manifest:
                raise ValueError("input manifest changed before snapshot")
        run_snapshot = _capture_run_snapshot(
            config,
            (command,),
            loaded_input_binding,
            manifest,
            dataset_factory=dataset_factory,
        )
    if run_snapshot.input_manifest.sha256 != input_manifest_sha256:
        raise ValueError("input manifest snapshot hash mismatch")
    if command.scene not in run_snapshot.classes:
        raise ValueError("run snapshot does not bind the requested scene vocabulary")
    if command.scene not in run_snapshot.datasets:
        raise ValueError("run snapshot does not bind the requested dataset input")
    vocabulary_binding = run_snapshot.vocabularies[command.scene]
    if (
        vocabulary_binding.path != command.classes_file
        or vocabulary_binding.sha256 != command.classes_sha256
    ):
        raise ValueError("run snapshot vocabulary binding mismatch")
    classes = run_snapshot.classes[command.scene]

    expected_names = [
        f"frame{cache_index:06d}.pkl.gz" for cache_index in range(command.frame_count)
    ]
    hybrid_cache._assert_no_symlink(
        command.cache_dir,
        field_name="frontend cache directory",
    )
    actual_names = sorted(path.name for path in command.cache_dir.glob("*.pkl.gz"))
    if actual_names != expected_names:
        raise ValueError("frontend directory contains a missing or unexpected cache file")

    cache_hashes: dict[str, str] = {}
    cache_bindings: list[_FileBinding] = []
    for name in expected_names:
        path = command.cache_dir / name
        payload, binding = _read_cache_snapshot(path)
        _validate_cache_payload(
            payload,
            classes=classes,
            image_shape=command.image_shape,
            path=path,
        )
        cache_bindings.append(binding)
        cache_hashes[name] = binding.sha256

    provenance = {
        name: binding.sha256 for name, binding in run_snapshot.provenance.items()
    }
    dataset_capture = _capture_dataset_witness(
        manifest,
        command,
        dataset_factory=dataset_factory,
    )
    input_witness = dataset_capture.witness
    if input_witness != run_snapshot.datasets[command.scene].witness:
        raise ValueError("dataset input witness changed after frontend inference")

    result = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "scene": command.scene,
        "frame_count": command.frame_count,
        "source_frame_ids": list(command.source_frame_ids),
        "source_frame_ids_hash": _json_hash(list(command.source_frame_ids)),
        "image_shape": list(command.image_shape),
        "class_count": len(classes),
        "classes": list(classes),
        "vocabulary_sha256": command.classes_sha256,
        "algorithm_hash": command.algorithm_hash,
        "feature_model_id": f"clip-sha256:{provenance['clip_model']}",
        "input_manifest_sha256": input_manifest_sha256,
        "input_witness": input_witness,
        "provenance_sha256": provenance,
        "cache_files_sha256": cache_hashes,
        "cache_prefix_sha256": _cache_prefix_sha256(cache_hashes),
    }
    final_bindings = (*run_snapshot.bindings, *dataset_capture.bindings, *cache_bindings)

    def assert_final_inputs() -> None:
        _assert_bindings_unchanged(final_bindings)
        _assert_run_snapshot_unchanged(run_snapshot)
        for directory in dataset_capture.directories:
            _verify_directory_content(directory, role="TESSE-CD input directory")

    assert_final_inputs()
    manifest_path = command.cache_dir / "frontend_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        binding, data = _snapshot_regular_file(
            manifest_path,
            field_name="frontend manifest",
            retain_bytes=True,
        )
        assert data is not None
        try:
            existing = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing frontend manifest is invalid") from exc
        _assert_binding_unchanged(binding)
        if existing != result:
            raise ValueError("existing frontend manifest does not match validated cache")
        assert_final_inputs()
        return result
    try:
        _publish_json_new(
            manifest_path,
            result,
            before_link=assert_final_inputs,
        )
    except FileExistsError:
        binding, data = _snapshot_regular_file(
            manifest_path,
            field_name="frontend manifest",
            retain_bytes=True,
        )
        assert data is not None
        try:
            existing = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing frontend manifest is invalid") from exc
        _assert_binding_unchanged(binding)
        if existing != result:
            raise ValueError("existing frontend manifest does not match validated cache")
        assert_final_inputs()
    return result


def _require_absent_lstat(path: Path, *, role: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise ValueError(f"{role} must not preexist: {path}")


def _bind_directory_content(path: Path, *, role: str) -> _DirectoryContentBinding:
    hybrid_cache._assert_no_symlink(path, field_name=role)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{role} must be a real directory: {path}")
    return _DirectoryContentBinding(
        path=path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _verify_directory_content(binding: _DirectoryContentBinding, *, role: str) -> None:
    hybrid_cache._assert_no_symlink(binding.path, field_name=role)
    metadata = os.lstat(binding.path)
    current = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    expected = (
        binding.device,
        binding.inode,
        binding.mode,
        binding.size,
        binding.modified_ns,
        binding.changed_ns,
    )
    if current != expected:
        raise ValueError(f"{role} changed after snapshot: {binding.path}")


def _verify_directory(binding: _DirectoryBinding, *, role: str) -> None:
    try:
        hybrid_cache._assert_no_symlink(binding.path, field_name=role)
        metadata = os.lstat(binding.path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{role} changed after creation: {binding.path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != binding.device
        or metadata.st_ino != binding.inode
    ):
        raise ValueError(f"{role} changed after creation: {binding.path}")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _directory_binding_from_fd(
    descriptor: int,
    path: Path,
    *,
    role: str,
) -> _DirectoryBinding:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{role} descriptor is not a directory")
    return _DirectoryBinding(path=path, device=metadata.st_dev, inode=metadata.st_ino)


def _directory_content_binding_from_fd(
    descriptor: int,
    path: Path,
    *,
    role: str,
) -> _DirectoryContentBinding:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{role} descriptor is not a directory")
    return _DirectoryContentBinding(
        path=path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    role: str,
) -> tuple[int, _DirectoryBinding]:
    if not name or Path(name).name != name:
        raise ValueError(f"{role} name is not a safe path component")
    descriptor = os.open(
        name,
        _directory_open_flags(),
        dir_fd=parent_descriptor,
    )
    try:
        binding = _directory_binding_from_fd(descriptor, path, role=role)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, binding


def _open_or_create_absolute_directory(path: Path) -> tuple[int, _DirectoryBinding]:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", _directory_open_flags())
    try:
        for name in absolute.parts[1:]:
            try:
                child = os.open(
                    name,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                staging_name = _private_staging_name(name)
                _require_absent_at(
                    descriptor,
                    staging_name,
                    role="private parent staging",
                )
                os.mkdir(staging_name, dir_fd=descriptor)
                child = os.open(
                    staging_name,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
                try:
                    _rename_directory_no_replace_at(
                        descriptor,
                        staging_name,
                        name,
                    )
                    os.fsync(descriptor)
                except BaseException:
                    os.close(child)
                    raise
            os.close(descriptor)
            descriptor = child
        binding = _directory_binding_from_fd(
            descriptor,
            absolute,
            role="frontend cache parent",
        )
        return descriptor, binding
    except BaseException:
        os.close(descriptor)
        raise


def _require_absent_at(directory_descriptor: int, name: str, *, role: str) -> None:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError(f"{role} must not preexist: {name}")


def _rename_directory_no_replace_at(
    directory_descriptor: int,
    source_name: str,
    target_name: str,
) -> None:
    for value in (source_name, target_name):
        if not value or Path(value).name != value:
            raise ValueError("directory publication name is not a safe path component")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("renameat2 is required for secure staging publication") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        directory_descriptor,
        os.fsencode(source_name),
        directory_descriptor,
        os.fsencode(target_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            errno.EEXIST,
            "frontend staging target already exists",
            target_name,
        )
    raise OSError(
        error_number,
        f"secure staging publication failed: {os.strerror(error_number)}",
        f"{source_name} -> {target_name}",
    )


def _private_staging_name(target_name: str) -> str:
    return f".{target_name}.{secrets.token_hex(16)}.staging"


def _private_cleanup_name(target_name: str) -> str:
    return f".{target_name}.{secrets.token_hex(16)}.cleanup"


def _binding_matches(metadata: os.stat_result, binding: _DirectoryBinding) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_dev == binding.device
        and metadata.st_ino == binding.inode
    )


def _remove_directory_contents_fd(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    for name in os.listdir(descriptor):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            try:
                if not _binding_matches(
                    os.fstat(child),
                    _DirectoryBinding(
                        path=Path(name),
                        device=metadata.st_dev,
                        inode=metadata.st_ino,
                    ),
                ):
                    raise ValueError("cleanup directory changed before recursive open")
                _remove_directory_contents_fd(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _quarantine_and_remove_bound_directory(
    parent_descriptor: int,
    candidate_names: Iterable[str],
    binding: _DirectoryBinding,
) -> None:
    matches: list[str] = []
    for name in candidate_names:
        try:
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if _binding_matches(metadata, binding):
            matches.append(name)
    if len(matches) != 1:
        raise FrontendLayoutCleanupUncertainError(binding.path)
    source_name = matches[0]
    cleanup_name = _private_cleanup_name(binding.path.name)
    _require_absent_at(
        parent_descriptor,
        cleanup_name,
        role="frontend cleanup quarantine",
    )
    try:
        _rename_directory_no_replace_at(
            parent_descriptor,
            source_name,
            cleanup_name,
        )
        os.fsync(parent_descriptor)
        cleanup_descriptor, cleanup_binding = _open_directory_at(
            parent_descriptor,
            cleanup_name,
            binding.path,
            role="frontend cleanup quarantine",
        )
        try:
            if cleanup_binding != binding:
                raise ValueError("frontend cleanup quarantine identity mismatch")
            _remove_directory_contents_fd(cleanup_descriptor)
        finally:
            os.close(cleanup_descriptor)
        os.rmdir(cleanup_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except FrontendLayoutCleanupUncertainError:
        raise
    except BaseException as exc:
        raise FrontendLayoutCleanupUncertainError(binding.path) from exc


def _open_existing_absolute_directory(path: Path) -> tuple[int, _DirectoryBinding]:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open("/", _directory_open_flags())
    try:
        for name in absolute.parts[1:]:
            child = os.open(
                name,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, _directory_binding_from_fd(
            descriptor,
            absolute,
            role="frontend cache parent",
        )
    except BaseException:
        os.close(descriptor)
        raise


def _cleanup_output_layout(layout: _OutputLayout) -> None:
    try:
        parent_descriptor, parent_binding = _open_existing_absolute_directory(
            layout.parent.path
        )
    except BaseException as exc:
        raise FrontendLayoutCleanupUncertainError(layout.root.path) from exc
    try:
        if parent_binding != layout.parent:
            raise FrontendLayoutCleanupUncertainError(layout.root.path)
        _quarantine_and_remove_bound_directory(
            parent_descriptor,
            (layout.root.path.name,),
            layout.root,
        )
    finally:
        os.close(parent_descriptor)


def _copy_bound_file(
    source_binding: _FileBinding,
    destination: Path,
    *,
    destination_directory_fd: int,
) -> _FileBinding:
    source = hybrid_cache._open_regular_input(
        source_binding.path,
        field_name="frozen TESSE-CD staging input",
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_fd: int | None = None
    try:
        before = _file_identity(os.fstat(source.fileno()))
        if before != source_binding.identity:
            raise ValueError(f"TESSE-CD input changed before staging: {source_binding.path}")
        destination_fd = os.open(
            destination.name,
            flags,
            0o400,
            dir_fd=destination_directory_fd,
        )
        digest = hashlib.sha256()
        while block := source.read(1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        after = _file_identity(os.fstat(source.fileno()))
        if after != before or digest.hexdigest() != source_binding.sha256:
            raise ValueError(f"TESSE-CD input changed during staging: {source_binding.path}")
        os.fchmod(destination_fd, 0o444)
        os.fsync(destination_fd)
        destination_identity = _file_identity(os.fstat(destination_fd))
        if (
            destination_identity.device == source_binding.identity.device
            and destination_identity.inode == source_binding.identity.inode
        ):
            raise ValueError("TESSE-CD staging input must not be a hard link")
        return _FileBinding(
            path=destination,
            sha256=source_binding.sha256,
            identity=destination_identity,
        )
    finally:
        source.close()
        if destination_fd is not None:
            os.close(destination_fd)


def _staging_source_bindings(
    command: FrontendCommand,
    run_snapshot: _RunSnapshot | None,
) -> tuple[tuple[_FileBinding, ...], _FileBinding]:
    results_source = command.source_root / "results"
    trajectory_source = command.source_root / "traj.txt"
    if run_snapshot is None:
        result_bindings = tuple(
            _snapshot_regular_file(
                path,
                field_name="fixture RGB-D staging input",
            )[0]
            for path in sorted(results_source.iterdir())
        )
        trajectory_binding = _snapshot_regular_file(
            trajectory_source,
            field_name="fixture trajectory staging input",
        )[0]
    else:
        capture = run_snapshot.datasets.get(command.scene)
        if capture is None:
            raise ValueError("run snapshot does not bind staging inputs")
        result_bindings = tuple(
            binding
            for binding in capture.bindings
            if binding.path.parent == results_source
        )
        trajectory_matches = tuple(
            binding
            for binding in capture.bindings
            if binding.path == trajectory_source
        )
        if not capture.fixture_only and len(result_bindings) != command.frame_count * 2:
            raise ValueError("run snapshot does not bind every RGB-D staging input")
        if len(trajectory_matches) != 1:
            raise ValueError("run snapshot does not bind the trajectory staging input")
        trajectory_binding = trajectory_matches[0]
    names = [binding.path.name for binding in result_bindings]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("RGB-D staging input names are not unique safe components")
    return tuple(sorted(result_bindings, key=lambda value: value.path.name)), trajectory_binding


def _command_output_paths(command: FrontendCommand) -> tuple[Path, ...]:
    variant_prefix = "gsa_detections_"
    if not command.cache_dir.name.startswith(variant_prefix):
        raise ValueError("frontend cache directory does not encode a GSA variant")
    variant = command.cache_dir.name.removeprefix(variant_prefix)
    suffix_values = [
        value.removeprefix("exp_suffix=")
        for value in command.argv
        if value.startswith("exp_suffix=")
    ]
    if len(suffix_values) != 1:
        raise ValueError("frontend command must encode exactly one experiment suffix")
    scene_root = command.cache_dir.parent
    return (
        command.cache_dir,
        scene_root / f"exp_{suffix_values[0]}",
        scene_root / f"gsa_vis_{variant}",
        scene_root / f"gsa_classes_{variant}.json",
    )


def _prepare_output_layout(
    commands: Iterable[FrontendCommand],
    run_snapshot: _RunSnapshot | None = None,
) -> _OutputLayout:
    selected = tuple(commands)
    if not selected:
        raise ValueError("at least one frontend command is required")
    roots = {command.cache_dir.parent.parent for command in selected}
    if len(roots) != 1:
        raise ValueError("frontend commands must share one cache root")
    root = roots.pop()
    hybrid_cache._assert_no_symlink(root, field_name="frontend cache root")
    scenes: dict[str, _DirectoryBinding] = {}
    results: dict[str, _DirectoryContentBinding] = {}
    input_copies: dict[Path, _FileBinding] = {}
    parent_descriptor, parent_binding = _open_or_create_absolute_directory(root.parent)
    root_staging_name: str | None = None
    root_binding: _DirectoryBinding | None = None
    try:
        _require_absent_at(
            parent_descriptor,
            root.name,
            role="frontend cache root",
        )
        root_staging_name = _private_staging_name(root.name)
        _require_absent_at(
            parent_descriptor,
            root_staging_name,
            role="private frontend root staging",
        )
        os.mkdir(root_staging_name, dir_fd=parent_descriptor)
        root_descriptor, root_binding = _open_directory_at(
            parent_descriptor,
            root_staging_name,
            root,
            role="frontend cache root",
        )
        try:
            for command in selected:
                scene_root = command.cache_dir.parent
                _require_absent_at(
                    root_descriptor,
                    scene_root.name,
                    role="frontend scene root",
                )
                scene_staging_name = _private_staging_name(scene_root.name)
                _require_absent_at(
                    root_descriptor,
                    scene_staging_name,
                    role="private frontend scene staging",
                )
                os.mkdir(scene_staging_name, dir_fd=root_descriptor)
                scene_descriptor, scene_binding = _open_directory_at(
                    root_descriptor,
                    scene_staging_name,
                    scene_root,
                    role="frontend scene root",
                )
                try:
                    for output_path in _command_output_paths(command):
                        _require_absent_at(
                            scene_descriptor,
                            output_path.name,
                            role="frontend output path",
                        )
                    _require_absent_at(
                        scene_descriptor,
                        "results",
                        role="staged RGB-D results",
                    )
                    _require_absent_at(
                        scene_descriptor,
                        "traj.txt",
                        role="staged trajectory",
                    )
                    result_sources, trajectory_source = _staging_source_bindings(
                        command,
                        run_snapshot,
                    )
                    results_staging_name = _private_staging_name("results")
                    os.mkdir(results_staging_name, dir_fd=scene_descriptor)
                    staged_results = scene_root / "results"
                    results_descriptor, _ = _open_directory_at(
                        scene_descriptor,
                        results_staging_name,
                        staged_results,
                        role="staged RGB-D results",
                    )
                    try:
                        for source_binding in result_sources:
                            destination = staged_results / source_binding.path.name
                            input_copies[destination] = _copy_bound_file(
                                source_binding,
                                destination,
                                destination_directory_fd=results_descriptor,
                            )
                        os.fchmod(results_descriptor, 0o555)
                        os.fsync(results_descriptor)
                        _rename_directory_no_replace_at(
                            scene_descriptor,
                            results_staging_name,
                            "results",
                        )
                        results[command.scene] = _directory_content_binding_from_fd(
                            results_descriptor,
                            staged_results,
                            role="staged RGB-D results",
                        )
                    finally:
                        os.close(results_descriptor)
                    trajectory_destination = scene_root / "traj.txt"
                    input_copies[trajectory_destination] = _copy_bound_file(
                        trajectory_source,
                        trajectory_destination,
                        destination_directory_fd=scene_descriptor,
                    )
                    os.fsync(scene_descriptor)
                    _rename_directory_no_replace_at(
                        root_descriptor,
                        scene_staging_name,
                        scene_root.name,
                    )
                    scenes[command.scene] = _directory_binding_from_fd(
                        scene_descriptor,
                        scene_root,
                        role="frontend scene root",
                    )
                finally:
                    os.close(scene_descriptor)
            os.fsync(root_descriptor)
            _rename_directory_no_replace_at(
                parent_descriptor,
                root_staging_name,
                root.name,
            )
            root_binding = _directory_binding_from_fd(
                root_descriptor,
                root,
                role="frontend cache root",
            )
        finally:
            os.close(root_descriptor)
        os.fsync(parent_descriptor)
        layout = _OutputLayout(
            parent=parent_binding,
            root=root_binding,
            scenes=scenes,
            results=results,
            input_copies=input_copies,
        )
        _verify_directory(layout.parent, role="frontend cache parent")
        _verify_directory(layout.root, role="frontend cache root")
        for binding in layout.scenes.values():
            _verify_directory(binding, role="frontend scene root")
        for binding in layout.results.values():
            _verify_directory_content(binding, role="staged RGB-D results")
        _assert_bindings_unchanged(layout.input_copies.values())
        return layout
    except BaseException as build_error:
        if root_binding is not None and root_staging_name is not None:
            try:
                _quarantine_and_remove_bound_directory(
                    parent_descriptor,
                    (root.name, root_staging_name),
                    root_binding,
                )
            except FrontendLayoutCleanupUncertainError as cleanup_error:
                raise cleanup_error from build_error
        raise
    finally:
        os.close(parent_descriptor)


def _verify_output_layout(command: FrontendCommand, layout: _OutputLayout) -> None:
    _verify_directory(layout.parent, role="frontend cache parent")
    _verify_directory(layout.root, role="frontend cache root")
    expected_scene_names = {binding.path.name for binding in layout.scenes.values()}
    if {path.name for path in layout.root.path.iterdir()} != expected_scene_names:
        raise ValueError("frontend cache root contains an unexpected path")
    scene_binding = layout.scenes.get(command.scene)
    if scene_binding is None:
        raise ValueError("frontend output layout does not bind the requested scene")
    _verify_directory(scene_binding, role="frontend scene root")
    for output_path in _command_output_paths(command):
        _require_absent_lstat(output_path, role="frontend output path")
    expected_names = {"results", "traj.txt"}
    if {path.name for path in scene_binding.path.iterdir()} != expected_names:
        raise ValueError("frontend scene root contains an unexpected path")
    _verify_source_links(scene_binding, layout)


def _verify_source_links(
    scene_binding: _DirectoryBinding,
    layout: _OutputLayout,
) -> None:
    _verify_directory(layout.parent, role="frontend cache parent")
    _verify_directory(layout.root, role="frontend cache root")
    _verify_directory(scene_binding, role="frontend scene root")
    scene_matches = [
        scene for scene, binding in layout.scenes.items() if binding == scene_binding
    ]
    if len(scene_matches) != 1:
        raise ValueError("staged frontend scene binding is ambiguous")
    scene = scene_matches[0]
    results_binding = layout.results.get(scene)
    if results_binding is None:
        raise ValueError("staged RGB-D results are not bound by this run")
    _verify_directory_content(results_binding, role="staged RGB-D results")
    copies = tuple(
        binding
        for path, binding in layout.input_copies.items()
        if path == scene_binding.path / "traj.txt"
        or path.parent == results_binding.path
    )
    expected_result_names = {
        binding.path.name
        for binding in copies
        if binding.path.parent == results_binding.path
    }
    if {path.name for path in results_binding.path.iterdir()} != expected_result_names:
        raise ValueError("staged RGB-D results contain an unexpected path")
    if sum(binding.path == scene_binding.path / "traj.txt" for binding in copies) != 1:
        raise ValueError("staged trajectory is not uniquely bound")
    _assert_bindings_unchanged(copies)
    for binding in copies:
        metadata = os.lstat(binding.path)
        if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise ValueError(f"staged input is not an immutable private copy: {binding.path}")


def _run_queue(
    commands: list[FrontendCommand],
    config: Mapping[str, Any],
    input_manifest_sha256: str,
    run_snapshot: _RunSnapshot,
    manifest: Mapping[str, Any],
    dataset_factory: DatasetFactory,
    output_layout: _OutputLayout,
) -> None:
    frontend = config["frontend"]
    cwd = _resolve_path(frontend["script"]).parents[2]
    for command in commands:
        _assert_run_snapshot_unchanged(run_snapshot)
        _verify_output_layout(command, output_layout)
        command.log_dir.mkdir(parents=True, exist_ok=True)
        environment = build_environment(command.gpu_id, cwd)
        with (command.log_dir / "frontend.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command.argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        scene_binding = output_layout.scenes[command.scene]
        _verify_source_links(scene_binding, output_layout)
        _assert_run_snapshot_unchanged(run_snapshot)
        validate_frontend_cache(
            command,
            config,
            input_manifest_sha256=input_manifest_sha256,
            run_snapshot=run_snapshot,
            manifest=manifest,
            dataset_factory=dataset_factory,
        )


def _execute_queues(
    commands: Iterable[FrontendCommand],
    config: Mapping[str, Any],
    input_manifest_sha256: str,
    run_snapshot: _RunSnapshot,
    manifest: Mapping[str, Any],
    dataset_factory: DatasetFactory,
    output_layout: _OutputLayout,
) -> None:
    selected = tuple(commands)
    gpu_ids = tuple(dict.fromkeys(command.gpu_id for command in selected))
    queues = [
        [command for command in selected if command.gpu_id == gpu_id]
        for gpu_id in gpu_ids
    ]
    errors: list[BaseException] = []
    try:
        with ThreadPoolExecutor(max_workers=len(queues)) as executor:
            futures = [
                executor.submit(
                    _run_queue,
                    queue,
                    config,
                    input_manifest_sha256,
                    run_snapshot,
                    manifest,
                    dataset_factory,
                    output_layout,
                )
                for queue in queues
            ]
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    errors.append(exc)
    except BaseException as exc:
        errors.append(exc)
    if not errors:
        return
    uncertain = next(
        (
            error
            for error in errors
            if isinstance(error, FrontendManifestPublicationUncertainError)
        ),
        None,
    )
    if uncertain is not None:
        raise uncertain
    try:
        _cleanup_output_layout(output_layout)
    except FrontendLayoutCleanupUncertainError as cleanup_error:
        raise cleanup_error from errors[0]
    raise errors[0]


def main(
    argv: list[str] | None = None,
    *,
    dataset_factory: DatasetFactory = TesseCdRgbdDataset,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/oviv2_tesse_cd_frontend_stage3.json",
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--gpu", type=int, action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    _regular_file(config_path, name="frontend config")
    config = _load_json_object(config_path)
    manifest, input_binding = _load_frozen_input_manifest(config)
    input_manifest_sha256 = input_binding.sha256
    commands = build_commands(
        config,
        manifest,
        scenes=args.scene,
        gpu_ids=None if args.gpu is None else tuple(args.gpu),
    )
    if args.dry_run:
        for command in commands:
            print(
                json.dumps(
                    {
                        "scene": command.scene,
                        "gpu": command.gpu_id,
                        "frame_count": command.frame_count,
                        "argv": command.argv,
                    }
                )
            )
        return 0

    if tuple(command.scene for command in commands) != SCENES:
        raise ValueError("formal frontend preflight must include both TESSE-CD scenes")
    if tuple(command.gpu_id for command in commands) != (0, 1):
        raise ValueError("formal frontend execution must use GPUs 0 and 1")

    run_snapshot = _capture_run_snapshot(
        config,
        commands,
        input_binding,
        manifest,
        dataset_factory=dataset_factory,
    )
    if args.preflight_only:
        print(json.dumps(_preflight(commands, config, run_snapshot), sort_keys=True))
        return 0

    _preflight(commands, config, run_snapshot)
    output_layout = _prepare_output_layout(commands, run_snapshot)
    _execute_queues(
        commands,
        config,
        input_manifest_sha256,
        run_snapshot,
        manifest,
        dataset_factory,
        output_layout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
