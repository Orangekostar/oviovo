#!/usr/bin/env python3
"""Build a causal TESSE-CD OVI-MAP static anchor package."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np


PINNED_OVIMAP_COMMIT = "58a804e2d7c82ba05a489eb071aba3367301fed8"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TesseNativeEnvironment:
    build_root: Path
    frontend_python: Path
    mapping_python: Path
    entity_root: Path
    ovimap_root: Path
    cropformer_weights: Path
    siglip_model: Path
    source_hashes: Path

    def __post_init__(self) -> None:
        for name in (
            "build_root",
            "frontend_python",
            "mapping_python",
            "entity_root",
            "ovimap_root",
            "cropformer_weights",
            "siglip_model",
            "source_hashes",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path")

    @property
    def cropformer_root(self) -> Path:
        return self.entity_root / "Entityv2" / "CropFormer"

    @property
    def cropformer_config(self) -> Path:
        return (
            self.cropformer_root
            / "configs"
            / "entityv2"
            / "entity_segmentation"
            / "cropformer_hornet_3x.yaml"
        )

    @property
    def cropformer_demo(self) -> Path:
        return self.cropformer_root / "demo_cropformer" / "demo_from_dirs.py"

    @property
    def mapping_workspace(self) -> Path:
        return self.ovimap_root / "mapping_ros_ws"


@dataclass(frozen=True)
class TesseNativeCommands:
    scene: str
    frame_ids: tuple[int, ...]
    geometry: tuple[str, ...]
    frontend: tuple[str, ...]
    mapping: tuple[str, ...]
    attempt_root: Path

    def __post_init__(self) -> None:
        if not self.scene or self.scene != self.scene.strip():
            raise ValueError("scene must be normalized and non-empty")
        if self.frame_ids != tuple(range(len(self.frame_ids))) or not self.frame_ids:
            raise ValueError("frame_ids must be contiguous and start at zero")
        for name in ("geometry", "frontend", "mapping"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(f"{name} must be a non-empty argv tuple")
        if not isinstance(self.attempt_root, Path):
            raise TypeError("attempt_root must be a Path")


@dataclass(frozen=True)
class AnchorPackagePaths:
    manifest: Path
    snapshot: Path
    entities: Path

    def __post_init__(self) -> None:
        for name in ("manifest", "snapshot", "entities"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")


@dataclass(frozen=True)
class CausalPrefixContract:
    scene: str
    first_intervention_frame: int
    frame_ids: tuple[int, ...]
    maximum_source_frame: int
    schedule_sha256: str
    rgbd_export_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene, str) or not self.scene.strip():
            raise ValueError("scene must be non-empty")
        if self.scene != self.scene.strip():
            raise ValueError("scene must not contain surrounding whitespace")
        if (
            isinstance(self.first_intervention_frame, bool)
            or not isinstance(self.first_intervention_frame, Integral)
            or int(self.first_intervention_frame) <= 0
        ):
            raise ValueError("first_intervention_frame must be positive")
        expected = tuple(range(int(self.first_intervention_frame)))
        if self.frame_ids != expected:
            raise ValueError("frame_ids must be the complete causal prefix")
        if self.maximum_source_frame != expected[-1]:
            raise ValueError("maximum_source_frame must end the causal prefix")
        for value in (
            self.schedule_sha256,
            self.rgbd_export_manifest_sha256,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("source hashes must be lowercase SHA-256 values")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    _reject_symlink_components(path, label=label)
    try:
        status = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    data = path.read_bytes()
    final = path.stat(follow_symlinks=False)
    if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while being read")
    return data


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    expected = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    observed = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    final = path.stat(follow_symlinks=False)
    current = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    if expected != observed or expected != current or byte_count != before.st_size:
        raise ValueError(f"{label} changed while being hashed")
    return {
        "path": str(path.absolute()),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _directory_record(path: Path, *, label: str) -> dict[str, Any]:
    _reject_symlink_components(path, label=label)
    if not path.is_dir():
        raise ValueError(f"{label} is missing: {path}")
    members: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*")):
        if member.is_dir():
            continue
        record = _file_record(member, label=f"{label} member")
        relative = member.relative_to(path).as_posix()
        member_hash = record["sha256"]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(member_hash.encode("ascii"))
        digest.update(b"\n")
        members.append(
            {
                "path": relative,
                "sha256": member_hash,
                "byte_count": record["byte_count"],
            }
        )
    if not members:
        raise ValueError(f"{label} contains no regular files")
    return {
        "path": str(path.absolute()),
        "sha256": digest.hexdigest(),
        "file_count": len(members),
        "files": members,
    }


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot read OVI-MAP commit: {completed.stderr.strip()}")
    value = completed.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("OVI-MAP checkout returned an invalid commit")
    return value


def _inside(path: Path, parent: Path) -> bool:
    child = path.absolute()
    root = parent.absolute()
    return child == root or root in child.parents


def build_tesse_native_commands(
    *,
    scene: str,
    rgbd_root: Path,
    frame_ids: tuple[int, ...],
    native: TesseNativeEnvironment,
    attempt_root: Path,
) -> TesseNativeCommands:
    """Build exact native OVI-MAP commands for a Replica-layout TESSE export."""

    if not isinstance(native, TesseNativeEnvironment):
        raise TypeError("native must be a TesseNativeEnvironment")
    if not isinstance(scene, str) or not scene.strip() or scene != scene.strip():
        raise ValueError("scene must be normalized and non-empty")
    if not isinstance(frame_ids, tuple) or frame_ids != tuple(range(len(frame_ids))):
        raise ValueError("frame_ids must be contiguous and start at zero")
    if not frame_ids:
        raise ValueError("frame_ids must not be empty")
    rgbd_root = Path(rgbd_root)
    attempt_root = Path(attempt_root)
    if _inside(attempt_root, rgbd_root):
        raise ValueError("native attempt root must be outside RGB-D source")

    scene_root = rgbd_root / scene
    frontend_output = attempt_root / "frontend"
    geometry_output = attempt_root / "geometric_segments"
    mapping_output = attempt_root / "mapping"
    intermediate_output = attempt_root / "intermediate_segments"
    audit_output = attempt_root / "native_audit"
    images = tuple(
        str(scene_root / "results" / f"frame{frame_id:06d}.jpg")
        for frame_id in frame_ids
    )
    geometry = (
        str(native.mapping_python),
        str(REPO_ROOT / "scripts" / "evaluation" / "run_ovimap_native.py"),
        "--internal-geometry",
        "--scene-data",
        str(scene_root),
        "--geometry-output",
        str(geometry_output),
        "--frame-ids",
        ",".join(str(value) for value in frame_ids),
        "--ovimap-root",
        str(native.ovimap_root),
    )
    frontend = (
        str(native.frontend_python),
        str(native.cropformer_demo),
        "--config-file",
        str(native.cropformer_config),
        "--input",
        *images,
        "--output",
        str(frontend_output),
        "--confidence-threshold",
        "0.5",
        "--out-type",
        "0",
        "--opts",
        "MODEL.WEIGHTS",
        str(native.cropformer_weights),
    )
    mapping = (
        str(native.mapping_python),
        str(native.ovimap_root / "scripts" / "panoptic_mapping_.py"),
        "--dataset",
        "replica",
        "--task",
        "Nyu40",
        "--scene_num",
        scene,
        "--data_folder",
        str(rgbd_root),
        "--result_folder",
        str(mapping_output),
        "--start",
        "0",
        "--end",
        str(len(frame_ids)),
        "--step",
        "1",
        "--data_association",
        "2",
        "--inst_association",
        "4",
        "--seg_graph_confidence",
        "3",
        "--use_temp_results",
        "--save_temp_results",
        "--intermediate_seg_folder",
        str(intermediate_output),
        "--temp_panoptics_folder",
        str(frontend_output),
        "--use_temp_geometrics",
        "--save_temp_geometrics",
        "--temp_geometrics_folder",
        str(geometry_output),
        "--num_threads",
        "10",
        "--siglip_model_path",
        str(native.siglip_model),
        "--native_audit_dir",
        str(audit_output),
        "--log",
        "tesse-causal-static-anchor",
    )
    return TesseNativeCommands(
        scene=scene,
        frame_ids=frame_ids,
        geometry=geometry,
        frontend=frontend,
        mapping=mapping,
        attempt_root=attempt_root,
    )


def preflight_tesse_native(
    *,
    scene: str,
    rgbd_root: Path,
    commands: TesseNativeCommands,
    native: TesseNativeEnvironment,
) -> dict[str, Any]:
    """Bind every native source required by one causal TESSE mapping attempt."""

    if not isinstance(commands, TesseNativeCommands):
        raise TypeError("commands must be TesseNativeCommands")
    if not isinstance(native, TesseNativeEnvironment):
        raise TypeError("native must be TesseNativeEnvironment")
    if commands.scene != scene:
        raise ValueError("command scene does not match preflight scene")
    commit = _git_head(native.ovimap_root)
    if commit != PINNED_OVIMAP_COMMIT:
        raise ValueError("native checkout must use the pinned OVI-MAP commit")
    rgbd_root = Path(rgbd_root)
    scene_root = rgbd_root / scene
    extension_root = native.mapping_workspace / "devel" / "lib"
    sources = {
        "camera": _file_record(rgbd_root / "cam_params.json", label="camera"),
        "trajectory": _file_record(scene_root / "traj.txt", label="trajectory"),
        "frontend_python": _file_record(
            native.frontend_python, label="frontend Python"
        ),
        "mapping_python": _file_record(
            native.mapping_python, label="mapping Python"
        ),
        "cropformer_config": _file_record(
            native.cropformer_config, label="CropFormer config"
        ),
        "cropformer_demo": _file_record(
            native.cropformer_demo, label="CropFormer demo"
        ),
        "cropformer_weights": _file_record(
            native.cropformer_weights, label="CropFormer weights"
        ),
        "geometry_helper": _file_record(
            REPO_ROOT / "scripts" / "evaluation" / "run_ovimap_native.py",
            label="geometry helper",
        ),
        "mapper": _file_record(
            native.ovimap_root / "scripts" / "panoptic_mapping_.py",
            label="OVI-MAP mapper",
        ),
        "consistent_gsm_extension": _file_record(
            extension_root / "consistent_gsm.cpython-311-x86_64-linux-gnu.so",
            label="consistent_gsm extension",
        ),
        "depth_segmentation_extension": _file_record(
            extension_root
            / "depth_segmentation_py.cpython-311-x86_64-linux-gnu.so",
            label="depth_segmentation_py extension",
        ),
        "source_hashes": _file_record(
            native.source_hashes, label="native source hashes"
        ),
        "siglip_model": _directory_record(native.siglip_model, label="SigLIP model"),
    }
    frames = []
    for frame_id in commands.frame_ids:
        frames.append(
            {
                "frame_id": frame_id,
                "rgb": _file_record(
                    scene_root / "results" / f"frame{frame_id:06d}.jpg",
                    label=f"RGB frame {frame_id}",
                ),
                "depth": _file_record(
                    scene_root / "results" / f"depth{frame_id:06d}.png",
                    label=f"depth frame {frame_id}",
                ),
            }
        )
    return {
        "status": "PASS",
        "scene": scene,
        "ovimap_commit": commit,
        "frame_ids": list(commands.frame_ids),
        "sources": sources,
        "frames": frames,
    }


def _frontend_environment(native: TesseNativeEnvironment) -> dict[str, str]:
    environment = os.environ.copy()
    prefix = native.frontend_python.parents[1]
    environment["PATH"] = f"{prefix / 'bin'}:{environment.get('PATH', '')}"
    environment["LD_LIBRARY_PATH"] = (
        f"{prefix / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["PYTHONPATH"] = (
        f"{native.cropformer_root}:{environment.get('PYTHONPATH', '')}"
    )
    return environment


def _mapping_environment(native: TesseNativeEnvironment) -> dict[str, str]:
    environment = os.environ.copy()
    prefix = native.mapping_python.parents[1]
    devel = native.mapping_workspace / "devel"
    environment["PATH"] = f"{prefix / 'bin'}:{environment.get('PATH', '')}"
    environment["LD_LIBRARY_PATH"] = (
        f"{devel / 'lib'}:{prefix / 'lib'}:"
        f"{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["PYTHONPATH"] = (
        f"{devel / 'lib'}:{devel / 'lib/python3.11/site-packages'}:"
        f"{native.ovimap_root / 'scripts'}:{environment.get('PYTHONPATH', '')}"
    )
    environment["OVIMAP_SIGLIP_MODEL"] = str(native.siglip_model)
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    return environment


def _run_native_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except FileExistsError as error:
        raise ValueError(f"native command log already exists: {log}") from error
    record = {
        "argv": list(argv),
        "cwd": str(cwd.absolute()),
        "exit_code": completed.returncode,
        "log": _file_record(log, label="native command log"),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"native command failed with exit code {completed.returncode}: {log}"
        )
    return record


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _native_artifacts(commands: TesseNativeCommands) -> dict[str, dict[str, Any]]:
    count = len(commands.frame_ids)
    mapping = commands.attempt_root / "mapping" / "cropformer_inst"
    color_log = commands.attempt_root / "logs" / "mapping.log"
    from src.evaluation.baselines.ovimap import parse_instance_color_log

    parse_instance_color_log(color_log)
    return {
        "instance_mesh": _file_record(
            mapping / f"instance_mesh_{count}.ply", label="OVI-MAP instance mesh"
        ),
        "semantic_features": _file_record(
            mapping / f"inst_sem_siglip-l-16-384_{count}_incre_combine.pkl",
            label="OVI-MAP semantic features",
        ),
        "instance_color_log": _file_record(
            color_log, label="OVI-MAP instance color log"
        ),
    }


def run_tesse_native_mapping(
    *,
    scene: str,
    rgbd_root: Path,
    commands: TesseNativeCommands,
    native: TesseNativeEnvironment,
) -> Path:
    """Execute one native causal-prefix attempt and publish its bound manifest."""

    if not isinstance(commands, TesseNativeCommands):
        raise TypeError("commands must be TesseNativeCommands")
    preflight = preflight_tesse_native(
        scene=scene,
        rgbd_root=rgbd_root,
        commands=commands,
        native=native,
    )
    attempt = commands.attempt_root
    attempt.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise ValueError(f"native attempt root already exists: {attempt}") from error
    for name in (
        "frontend",
        "geometric_segments",
        "intermediate_segments",
        "mapping",
        "native_audit",
        "logs",
    ):
        (attempt / name).mkdir()

    mapping_environment = _mapping_environment(native)
    geometry = _run_native_command(
        commands.geometry,
        cwd=REPO_ROOT,
        env=mapping_environment,
        log=attempt / "logs" / "geometry.log",
    )
    frontend = _run_native_command(
        commands.frontend,
        cwd=native.cropformer_root,
        env=_frontend_environment(native),
        log=attempt / "logs" / "frontend.log",
    )
    for frame_id in commands.frame_ids:
        _file_record(
            attempt / "frontend" / f"frame{frame_id:06d}.png",
            label=f"CropFormer mask {frame_id}",
        )
    mapping = _run_native_command(
        commands.mapping,
        cwd=native.ovimap_root,
        env=mapping_environment,
        log=attempt / "logs" / "mapping.log",
    )
    artifacts = _native_artifacts(commands)
    postflight = preflight_tesse_native(
        scene=scene,
        rgbd_root=rgbd_root,
        commands=commands,
        native=native,
    )
    if postflight != preflight:
        raise RuntimeError("native mapping inputs changed during execution")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "state": "MAPPING_PASS",
        "scene": scene,
        "frame_ids": list(commands.frame_ids),
        "preflight": preflight,
        "commands": {
            "geometry": geometry,
            "frontend": frontend,
            "mapping": mapping,
        },
        "artifacts": artifacts,
        "postflight": postflight,
    }
    manifest_path = attempt / "native_mapping_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _json_hash(value: Mapping[str, Any]) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return normalized


def _scene_events(schedule: Mapping[str, Any], scene: str) -> tuple[Mapping[str, Any], ...]:
    scenes = schedule.get("scenes")
    if not isinstance(scenes, Mapping) or scene not in scenes:
        raise ValueError(f"scene is not declared by causal schedule: {scene}")
    scene_payload = scenes[scene]
    if not isinstance(scene_payload, Mapping):
        raise ValueError("causal schedule scene record must be an object")
    raw_events = scene_payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("causal schedule scene must declare events")
    if any(not isinstance(item, Mapping) for item in raw_events):
        raise ValueError("causal schedule events must be objects")
    events = tuple(raw_events)
    event_ids = tuple(item.get("event_id") for item in events)
    if any(not isinstance(value, str) or not value.strip() for value in event_ids):
        raise ValueError("causal schedule event IDs must be non-empty strings")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("causal schedule event IDs must be unique")
    return events


def load_causal_prefix_contract(
    *,
    scene: str,
    schedule_path: Path,
    rgbd_root: Path,
    configured_cutoff: int,
) -> CausalPrefixContract:
    """Validate the complete RGB-D prefix strictly before the first intervention."""

    if not isinstance(scene, str) or not scene.strip() or scene != scene.strip():
        raise ValueError("scene must be a normalized non-empty string")
    cutoff = _integer(configured_cutoff, label="configured_cutoff", minimum=0)
    schedule_path = Path(schedule_path)
    rgbd_root = Path(rgbd_root)
    schedule_bytes = _regular_file_bytes(
        schedule_path, label="causal schedule"
    )
    schedule = _json_object(schedule_bytes, label="causal schedule")
    if schedule.get("dataset") != "TESSE-CD":
        raise ValueError("causal schedule dataset must be TESSE-CD")
    events = _scene_events(schedule, scene)
    interventions = tuple(
        _integer(
            item.get("intervention_frame_index"),
            label="intervention_frame_index",
            minimum=1,
        )
        for item in events
    )
    first_intervention = min(interventions)
    if cutoff != first_intervention - 1:
        raise ValueError(
            "configured cutoff must be strictly before first intervention "
            "and include the complete prefix"
        )

    scene_root = rgbd_root / scene
    export_manifest = scene_root / "export_manifest.json"
    export_bytes = _regular_file_bytes(
        export_manifest, label="RGB-D export manifest"
    )
    export_payload = _json_object(export_bytes, label="RGB-D export manifest")
    if (
        export_payload.get("dataset") != "TESSE-CD"
        or export_payload.get("scene") != scene
    ):
        raise ValueError("RGB-D export manifest does not match dataset and scene")

    frame_ids = tuple(range(first_intervention))
    results = scene_root / "results"
    for frame_index in frame_ids:
        for prefix, suffix in (("frame", ".jpg"), ("depth", ".png")):
            path = results / f"{prefix}{frame_index:06d}{suffix}"
            try:
                _regular_file_bytes(path, label="causal RGB-D frame")
            except ValueError as error:
                raise ValueError(f"causal RGB-D frame is missing or invalid: {path}") from error

    return CausalPrefixContract(
        scene=scene,
        first_intervention_frame=first_intervention,
        frame_ids=frame_ids,
        maximum_source_frame=cutoff,
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        rgbd_export_manifest_sha256=hashlib.sha256(export_bytes).hexdigest(),
    )


def _declared_artifact(
    native_payload: Mapping[str, Any], role: str, path: Path
) -> dict[str, Any]:
    artifacts = native_payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("native manifest artifacts must be an object")
    declared = artifacts.get(role)
    if not isinstance(declared, Mapping):
        raise ValueError(f"native manifest is missing artifact: {role}")
    observed = _file_record(path, label=f"native {role}")
    if dict(declared) != observed:
        raise ValueError(f"native artifact binding mismatch: {role}")
    return observed


def _vocabulary(path: Path, *, scene: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    data = _regular_file_bytes(path, label="TESSE vocabulary")
    payload = _json_object(data, label="TESSE vocabulary")
    classes = payload.get("classes")
    if (
        payload.get("dataset") != "TESSE-CD"
        or payload.get("scene") != scene
        or not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) or not item.strip() for item in classes)
    ):
        raise ValueError("TESSE vocabulary does not match scene or class schema")
    normalized = tuple(item.strip() for item in classes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("TESSE vocabulary classes must be unique")
    return normalized, {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _encode_text_features(
    values: tuple[str, ...], model: Path, device: str
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not values:
        raise ValueError("text feature inputs must not be empty")
    loaded_model = AutoModel.from_pretrained(
        str(model), local_files_only=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    tokens = tokenizer(
        list(values),
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        encoded = loaded_model.get_text_features(**tokens).float().cpu().numpy()
    return np.asarray(encoded, dtype=np.float32)


def _classify_anchor_entities(
    snapshot: Any,
    *,
    classes: tuple[str, ...],
    siglip_model: Path,
    device: str,
) -> None:
    from src.evaluation.baselines.ovimap import relative_similarity_labels

    eligible = [
        index
        for index, entity in enumerate(snapshot.entities)
        if entity.semantic_embedding is not None
        and int(entity.metadata.get("observation_count", 0)) >= 2
    ]
    if not eligible:
        return
    entity_features = np.vstack(
        [snapshot.entities[index].semantic_embedding for index in eligible]
    )
    text_features = _encode_text_features(classes, siglip_model, device)
    canonical_values = ("object", "things", "stuff", "texture")
    canonical_features = _encode_text_features(
        canonical_values, siglip_model, device
    )
    labels, scores = relative_similarity_labels(
        entity_features,
        text_features,
        canonical_features,
        classes,
    )
    for index, label, score in zip(eligible, labels, scores, strict=True):
        entity = snapshot.entities[index]
        entity.semantic_label = label
        entity.semantic_score = score
        entity.metadata["semantic_match_score"] = score


def _relative_output_record(path: Path, *, root: Path) -> dict[str, Any]:
    record = _file_record(path, label="anchor output")
    record["path"] = path.relative_to(root).as_posix()
    return record


def _validated_causal_sources(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_roles = {"config", "schedule", "rgbd_export_manifest"}
    if not isinstance(value, Mapping) or set(value) != expected_roles:
        raise ValueError("causal_sources must bind config, schedule, and RGB-D export")
    normalized: dict[str, dict[str, Any]] = {}
    for role in sorted(expected_roles):
        record = value[role]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"causal source record is invalid: {role}")
        path = record.get("path")
        sha256 = record.get("sha256")
        byte_count = record.get("byte_count")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, Integral)
            or int(byte_count) < 0
        ):
            raise ValueError(f"causal source record is invalid: {role}")
        observed = _file_record(Path(path), label=f"causal source {role}")
        if dict(record) != observed:
            raise ValueError(f"causal source binding mismatch: {role}")
        normalized[role] = observed
    return normalized


def build_anchor_package(
    *,
    scene: str,
    cutoff_frame: int,
    native_manifest: Path,
    instance_mesh: Path,
    semantic_features: Path,
    instance_color_log: Path,
    vocabulary_json: Path,
    siglip_model: Path,
    device: str,
    output_root: Path,
    causal_sources: Mapping[str, Mapping[str, Any]],
) -> AnchorPackagePaths:
    """Convert verified native OVI-MAP outputs to a pickle-free anchor package."""

    from src.evaluation.baselines.adapters import adapt_ovimap
    from src.evaluation.baselines.contracts import RuntimeBreakdown
    from src.evaluation.baselines.ovimap import (
        bind_mesh_instances,
        load_instance_mesh,
        parse_instance_color_log,
    )
    from src.evaluation.exporters.oviovo import write_map_snapshot

    if not isinstance(scene, str) or not scene.strip() or scene != scene.strip():
        raise ValueError("scene must be normalized and non-empty")
    cutoff = _integer(cutoff_frame, label="cutoff_frame", minimum=0)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be non-empty")
    native_manifest = Path(native_manifest)
    instance_mesh = Path(instance_mesh)
    semantic_features = Path(semantic_features)
    instance_color_log = Path(instance_color_log)
    vocabulary_json = Path(vocabulary_json)
    siglip_model = Path(siglip_model)
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError(f"anchor output root already exists: {output_root}")
    causal_records = _validated_causal_sources(causal_sources)

    native_bytes = _regular_file_bytes(
        native_manifest, label="native mapping manifest"
    )
    native_payload = _json_object(native_bytes, label="native mapping manifest")
    if (
        native_payload.get("status") != "PASS"
        or native_payload.get("state") != "MAPPING_PASS"
        or native_payload.get("scene") != scene
        or native_payload.get("frame_ids") != list(range(cutoff + 1))
    ):
        raise ValueError("native manifest is not a complete causal mapping PASS")
    preflight = native_payload.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("ovimap_commit") != PINNED_OVIMAP_COMMIT
    ):
        raise ValueError("native manifest does not bind the pinned OVI-MAP commit")
    source_records = {
        "instance_mesh": _declared_artifact(
            native_payload, "instance_mesh", instance_mesh
        ),
        "semantic_features": _declared_artifact(
            native_payload, "semantic_features", semantic_features
        ),
        "instance_color_log": _declared_artifact(
            native_payload, "instance_color_log", instance_color_log
        ),
    }
    native_record = {
        "path": str(native_manifest.absolute()),
        "sha256": hashlib.sha256(native_bytes).hexdigest(),
        "byte_count": len(native_bytes),
    }
    classes, vocabulary_record = _vocabulary(vocabulary_json, scene=scene)
    siglip_record = _directory_record(siglip_model, label="SigLIP model")
    anchor_id = _json_hash(
        {
            "schema_version": 1,
            "scene": scene,
            "cutoff_frame": cutoff,
            "ovimap_commit": PINNED_OVIMAP_COMMIT,
            "native_manifest": native_record,
            "vocabulary": vocabulary_record,
            "siglip_model": siglip_record,
            "causal_sources": causal_records,
        }
    )

    semantic_bytes = _regular_file_bytes(
        semantic_features, label="OVI-MAP semantic features"
    )
    try:
        semantic_instances = pickle.loads(semantic_bytes)
    except Exception as error:
        raise ValueError("OVI-MAP semantic features are not a readable pickle") from error
    if not isinstance(semantic_instances, Mapping):
        raise ValueError("OVI-MAP semantic features must contain a mapping")
    colors_by_instance = parse_instance_color_log(instance_color_log)
    points_by_color, background_xyz = load_instance_mesh(
        instance_mesh, colors_by_instance.values()
    )
    instances = bind_mesh_instances(
        semantic_instances, colors_by_instance, points_by_color
    )
    if not instances:
        raise ValueError("OVI-MAP anchor contains no mesh-backed instances")
    artifact = adapt_ovimap(
        instances,
        points_by_color=points_by_color,
        scene_id=scene,
        timestamp=float(cutoff),
        upstream_commit=PINNED_OVIMAP_COMMIT,
        runtime=RuntimeBreakdown(frame_count=cutoff + 1),
        background_xyz=background_xyz,
        semantic_label_source=(
            "method_output.feat SigLIP-L/16-384 canonical-relative zero-shot"
        ),
        protocol_notes=(
            "anchor consumes only the complete pre-intervention RGB-D prefix",
            "runtime consumes repository-owned NPZ and JSONL without pickle",
        ),
    )
    snapshot = artifact.snapshot
    snapshot.method = "OVI-MAP causal static anchor"
    for entity in snapshot.entities:
        entity.metadata.update(
            {
                "anchor_id": anchor_id,
                "anchor_instance_id": int(entity.entity_id.split(":", 1)[1]),
                "authority": "ovimap_anchor",
                "native_manifest_sha256": native_record["sha256"],
            }
        )
    _classify_anchor_entities(
        snapshot,
        classes=classes,
        siglip_model=siglip_model,
        device=device.strip(),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        written = write_map_snapshot(snapshot, staging)
        output_records = {
            name: _relative_output_record(path, root=staging)
            for name, path in written.items()
        }
        manifest_payload = {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_v1",
            "status": "PASS",
            "method": "OVI-MAP causal static anchor",
            "scene": scene,
            "anchor_id": anchor_id,
            "causality": {
                "first_source_frame": 0,
                "last_source_frame": cutoff,
                "maximum_source_frame": cutoff,
                "strictly_pre_intervention": True,
            },
            "sources": {
                **causal_records,
                "native_manifest": native_record,
                **source_records,
                "vocabulary": vocabulary_record,
                "siglip_model": siglip_record,
            },
            "outputs": output_records,
        }
        manifest_path = staging / "anchor_manifest.json"
        _atomic_json(manifest_path, manifest_payload)

        if _file_record(
            native_manifest, label="native mapping manifest"
        ) != native_record:
            raise RuntimeError("native manifest changed during anchor conversion")
        for role, path in (
            ("instance_mesh", instance_mesh),
            ("semantic_features", semantic_features),
            ("instance_color_log", instance_color_log),
        ):
            if _file_record(path, label=f"native {role}") != source_records[role]:
                raise RuntimeError(f"native {role} changed during anchor conversion")
        if _file_record(vocabulary_json, label="TESSE vocabulary") != vocabulary_record:
            raise RuntimeError("TESSE vocabulary changed during anchor conversion")
        if _directory_record(siglip_model, label="SigLIP model") != siglip_record:
            raise RuntimeError("SigLIP model changed during anchor conversion")
        if _validated_causal_sources(causal_records) != causal_records:
            raise RuntimeError("causal sources changed during anchor conversion")
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return AnchorPackagePaths(
        manifest=output_root / "anchor_manifest.json",
        snapshot=output_root / output_records["snapshot"]["path"],
        entities=output_root / output_records["entities"]["path"],
    )


_ANCHOR_CONFIG_FIELDS = {
    "schema_version",
    "method_id",
    "dataset",
    "scene",
    "source_role",
    "first_source_frame",
    "last_source_frame",
    "source_stride",
    "ovimap_commit",
    "vocabulary_json",
    "minimum_spatial_iou",
    "maximum_centroid_distance_m",
    "minimum_semantic_cosine",
    "moved_displacement_m",
    "background_voxel_size_m",
}


def _anchor_config(path: Path) -> dict[str, Any]:
    payload = _json_object(
        _regular_file_bytes(path, label="anchor config"), label="anchor config"
    )
    if set(payload) != _ANCHOR_CONFIG_FIELDS:
        raise ValueError("anchor config fields are not exact")
    if (
        payload.get("schema_version") != 1
        or payload.get("method_id") != "crove_ovimap_static_anchor_v1"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("source_role") != "causal_pre_intervention_initialization"
        or payload.get("first_source_frame") != 0
        or payload.get("source_stride") != 1
        or payload.get("ovimap_commit") != PINNED_OVIMAP_COMMIT
        or not isinstance(payload.get("scene"), str)
        or not str(payload["scene"]).strip()
        or not isinstance(payload.get("vocabulary_json"), str)
        or not str(payload["vocabulary_json"]).strip()
    ):
        raise ValueError("anchor config identity is invalid")
    _integer(
        payload.get("last_source_frame"),
        label="last_source_frame",
        minimum=0,
    )
    for name in (
        "minimum_spatial_iou",
        "maximum_centroid_distance_m",
        "minimum_semantic_cosine",
        "moved_displacement_m",
        "background_voxel_size_m",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"anchor config {name} must be numeric")
        normalized = float(value)
        if not np.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"anchor config {name} must be finite and positive")
    if not 0.0 < float(payload["minimum_semantic_cosine"]) <= 1.0:
        raise ValueError("minimum_semantic_cosine must be in (0, 1]")
    return payload


def _manifest_input_path(payload: Mapping[str, Any], role: str) -> Path:
    artifacts = payload.get("artifacts")
    record = artifacts.get(role) if isinstance(artifacts, Mapping) else None
    path = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"native manifest artifact path is invalid: {role}")
    return Path(path)


def _manifest_siglip_path(payload: Mapping[str, Any]) -> Path:
    preflight = payload.get("preflight")
    sources = preflight.get("sources") if isinstance(preflight, Mapping) else None
    record = sources.get("siglip_model") if isinstance(sources, Mapping) else None
    path = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("native manifest SigLIP model path is invalid")
    return Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--rgbd-root", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    config = _anchor_config(args.config)
    scene = str(config["scene"])
    cutoff = int(config["last_source_frame"])
    contract = load_causal_prefix_contract(
        scene=scene,
        schedule_path=args.schedule,
        rgbd_root=args.rgbd_root,
        configured_cutoff=cutoff,
    )
    causal_sources = {
        "config": _file_record(args.config, label="anchor config"),
        "schedule": _file_record(args.schedule, label="causal schedule"),
        "rgbd_export_manifest": _file_record(
            args.rgbd_root / scene / "export_manifest.json",
            label="RGB-D export manifest",
        ),
    }
    if (
        causal_sources["schedule"]["sha256"] != contract.schedule_sha256
        or causal_sources["rgbd_export_manifest"]["sha256"]
        != contract.rgbd_export_manifest_sha256
    ):
        raise RuntimeError("causal sources changed after prefix validation")

    native_bytes = _regular_file_bytes(
        args.native_manifest, label="native mapping manifest"
    )
    native_payload = _json_object(native_bytes, label="native mapping manifest")
    vocabulary = Path(str(config["vocabulary_json"]))
    if not vocabulary.is_absolute():
        vocabulary = REPO_ROOT / vocabulary
    build_anchor_package(
        scene=scene,
        cutoff_frame=cutoff,
        native_manifest=args.native_manifest,
        instance_mesh=_manifest_input_path(native_payload, "instance_mesh"),
        semantic_features=_manifest_input_path(native_payload, "semantic_features"),
        instance_color_log=_manifest_input_path(
            native_payload, "instance_color_log"
        ),
        vocabulary_json=vocabulary,
        siglip_model=_manifest_siglip_path(native_payload),
        device=args.device,
        output_root=args.output_root,
        causal_sources=causal_sources,
    )
    return 0


__all__ = [
    "PINNED_OVIMAP_COMMIT",
    "AnchorPackagePaths",
    "CausalPrefixContract",
    "TesseNativeCommands",
    "TesseNativeEnvironment",
    "build_anchor_package",
    "build_tesse_native_commands",
    "load_causal_prefix_contract",
    "main",
    "preflight_tesse_native",
    "run_tesse_native_mapping",
]


if __name__ == "__main__":
    raise SystemExit(main())
