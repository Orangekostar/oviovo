#!/usr/bin/env python3
"""Run the frozen OVIV2 Stage3 online mapper on one TESSE-CD scene."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass, replace
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    RUNNER_SCENE_CONFIG_FIELDS,
    build_evaluation_checkpoint_plan,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)

@dataclass(frozen=True)
class TesseCausalCheckpoint:
    frame_index: int
    timestamp_ns: int
    relative_timestamp_ns: int
    event_ids: tuple[str, ...]
    roles: tuple[str, ...]


SCENE_CONFIG_FIELDS = RUNNER_SCENE_CONFIG_FIELDS


@dataclass(frozen=True)
class RunnerDependencies:
    dataset_factory: Callable[[Mapping[str, Any]], Any]
    cache_loader_factory: Callable[[Mapping[str, Any], Any], Any]
    runtime_factory: Callable[[Mapping[str, Any], Any], Any]
    checkpoint_exporter: Callable[
        [Any, TesseCausalCheckpoint, Path, Mapping[str, Any]], Mapping[str, Any] | None
    ]
    provenance_factory: Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class _CheckpointIdentity:
    path: Path
    directory_fingerprint: tuple[int, int, int, int, int]
    checksums_fingerprint: tuple[int, int, int, int, int]
    checksums_sha256: str
    member_fingerprints: tuple[
        tuple[str, tuple[int, int, int, int, int]], ...
    ]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return _load_json_bytes(path.read_bytes(), path)


def _load_json_bytes(data: bytes, path: Path) -> dict[str, Any]:
    payload = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _stat_fingerprint(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_regular_at(
    directory_fd: int,
    name: str,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint member is not a regular file")
        content = b"".join(iter(lambda: os.read(descriptor, 1024 * 1024), b""))
        after = os.fstat(descriptor)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise ValueError("checkpoint member changed while reading")
        return content, _stat_fingerprint(after)
    finally:
        os.close(descriptor)


def _member_fingerprint_at(
    directory_fd: int,
    name: str,
) -> tuple[int, int, int, int, int]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("checkpoint member is not a regular file")
        return _stat_fingerprint(status)
    finally:
        os.close(descriptor)


def _open_checkpoint_directory(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )


def _capture_checkpoint_identity(path: Path) -> _CheckpointIdentity:
    try:
        directory_fd = _open_checkpoint_directory(path)
        try:
            directory_status = os.fstat(directory_fd)
            checksums_content, checksums_fingerprint = _read_regular_at(
                directory_fd, "checksums.json"
            )
            checksums = _load_json_bytes(checksums_content, path / "checksums.json")
            if not checksums:
                raise ValueError("checkpoint checksum inventory must not be empty")
            expected_names = set(checksums) | {"checksums.json"}
            if set(os.listdir(directory_fd)) != expected_names:
                raise ValueError("checkpoint physical inventory does not match checksums")
            member_fingerprints = []
            for name in sorted(checksums):
                if not (
                    isinstance(name, str)
                    and name not in {".", ".."}
                    and Path(name).parts == (name,)
                    and isinstance(checksums[name], str)
                    and _is_sha256(checksums[name])
                ):
                    raise ValueError("checkpoint checksum inventory is invalid")
                member_fingerprints.append(
                    (name, _member_fingerprint_at(directory_fd, name))
                )
            if _stat_fingerprint(os.lstat(path)) != _stat_fingerprint(
                directory_status
            ):
                raise ValueError("checkpoint identity changed while capturing")
            return _CheckpointIdentity(
                path=path,
                directory_fingerprint=_stat_fingerprint(directory_status),
                checksums_fingerprint=checksums_fingerprint,
                checksums_sha256=_sha256_bytes(checksums_content),
                member_fingerprints=tuple(member_fingerprints),
            )
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise ValueError("checkpoint identity could not be captured") from error


def _revalidate_checkpoint_identity(identity: _CheckpointIdentity) -> None:
    try:
        before = os.lstat(identity.path)
        directory_fd = _open_checkpoint_directory(identity.path)
        try:
            directory_status = os.fstat(directory_fd)
            checksums_content, checksums_fingerprint = _read_regular_at(
                directory_fd, "checksums.json"
            )
            actual_names = set(os.listdir(directory_fd))
            member_fingerprints = tuple(
                (name, _member_fingerprint_at(directory_fd, name))
                for name, _ in identity.member_fingerprints
            )
            after = os.lstat(identity.path)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError) as error:
        raise ValueError("checkpoint identity changed before index publication") from error
    expected_names = {name for name, _ in identity.member_fingerprints} | {
        "checksums.json"
    }
    if not (
        _stat_fingerprint(before) == identity.directory_fingerprint
        and _stat_fingerprint(directory_status) == identity.directory_fingerprint
        and _stat_fingerprint(after) == identity.directory_fingerprint
        and checksums_fingerprint == identity.checksums_fingerprint
        and _sha256_bytes(checksums_content) == identity.checksums_sha256
        and actual_names == expected_names
        and member_fingerprints == identity.member_fingerprints
    ):
        raise ValueError("checkpoint identity changed before index publication")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _load_causal_checkpoints(
    schedule: Path,
    *,
    scene: str,
    frame_count: int,
) -> tuple[TesseCausalCheckpoint, ...]:
    return _load_causal_checkpoints_bytes(
        schedule.read_bytes(),
        schedule,
        scene=scene,
        frame_count=frame_count,
    )


def _load_causal_checkpoints_bytes(
    data: bytes,
    path: Path,
    *,
    scene: str,
    frame_count: int,
) -> tuple[TesseCausalCheckpoint, ...]:
    payload = _load_json_bytes(data, path)
    parameters = payload.get("parameters")
    if (
        payload.get("schema_version") != 2
        or payload.get("manifest_id") != "tesse_cd_causal_schedule_v2"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("method_predictions_used") is not False
        or not isinstance(parameters, dict)
        or parameters.get("frame_indexing") != "zero_based"
    ):
        raise ValueError("causal schedule identity mismatch")
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != {"apartment", "office"}:
        raise ValueError("causal schedule must contain apartment and office exactly")
    selected = scenes.get(scene)
    if not isinstance(selected, dict) or selected.get("frame_count") != frame_count:
        raise ValueError("causal schedule frame count mismatch")
    raw_entries = selected.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("causal schedule entries must be a non-empty list")
    checkpoints: list[TesseCausalCheckpoint] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("causal schedule entry must be an object")
        frame_index = raw.get("frame_index")
        timestamp_ns = raw.get("timestamp_ns")
        relative_timestamp_ns = raw.get("relative_timestamp_ns")
        if type(frame_index) is not int or not 0 <= frame_index < frame_count:
            raise ValueError("causal checkpoint frame is outside the declared frame count")
        if type(timestamp_ns) is not int or timestamp_ns <= 0:
            raise ValueError("causal checkpoint timestamp must be positive")
        if type(relative_timestamp_ns) is not int or relative_timestamp_ns < 0:
            raise ValueError("causal checkpoint relative timestamp must be non-negative")
        roles = _string_tuple(raw.get("roles"), "roles")
        if not roles or any(role not in {"official", "common_v2"} for role in roles):
            raise ValueError("causal checkpoint roles are invalid")
        checkpoints.append(
            TesseCausalCheckpoint(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                relative_timestamp_ns=relative_timestamp_ns,
                event_ids=_string_tuple(raw.get("event_ids"), "event_ids"),
                roles=roles,
            )
        )
    checkpoints.sort(key=lambda item: item.frame_index)
    if len({item.frame_index for item in checkpoints}) != len(checkpoints):
        raise ValueError("causal schedule contains duplicate checkpoint frames")
    if any(
        current.timestamp_ns >= following.timestamp_ns
        or current.relative_timestamp_ns >= following.relative_timestamp_ns
        for current, following in zip(checkpoints, checkpoints[1:])
    ):
        raise ValueError("causal checkpoint timestamps must be strictly increasing")
    return tuple(checkpoints)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)


def algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_algorithm_config(config)


def algorithm_hash(config: Mapping[str, Any]) -> str:
    return canonical_algorithm_hash(config)


def apply_frozen_visibility_policy(runtime_config: Any, policy: str) -> Any:
    if policy not in {"signed_depth", "missing_as_absence"}:
        raise ValueError("missing_observation_policy is invalid")
    if not hasattr(runtime_config, "missing_observation_policy"):
        raise ValueError(
            "runtime config does not expose missing_observation_policy"
        )
    return replace(runtime_config, missing_observation_policy=policy)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _byte_record(data: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(data), "byte_count": len(data)}


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _cache_prefix_sha256(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for cache_index, checksum in enumerate(hashes.values()):
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.absolute() if path.is_absolute() else (REPO_ROOT / path).absolute()


def _require_regular_file(path: Path, role: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} cannot be a symlink or reside below one")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular file")


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} cannot contain symlink path components")


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path if relative_to is None else path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _content_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    return {"sha256": record["sha256"], "byte_count": record["byte_count"]}


def _tree_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact tree is empty: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    for item in files:
        relative = str(item.relative_to(path))
        size = item.stat().st_size
        byte_count += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
        digest.update(b"\n")
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[str, int, Path, tuple[int, ...]]:
    if config.get("schema_version") != 1:
        raise ValueError("runner schema_version must be 1")
    if config.get("dataset") != "TESSE-CD" or config.get("method_id") != "OVIV2":
        raise ValueError("runner identity must be OVIV2 on TESSE-CD")
    scene = config.get("scene")
    if scene not in {"apartment", "office"}:
        raise ValueError("scene must be apartment or office")
    frame_count = _positive_integer(config.get("frame_count"), "frame_count")
    if config.get("missing_observation_policy") not in {
        "signed_depth",
        "missing_as_absence",
    }:
        raise ValueError("missing_observation_policy is invalid")
    frozen_algorithm_hash = config.get("algorithm_hash")
    if frozen_algorithm_hash is not None and frozen_algorithm_hash != algorithm_hash(config):
        raise ValueError("configured algorithm_hash does not match mapping parameters")
    schedule = Path(str(config.get("schedule_manifest", "")))
    if not schedule.is_absolute():
        schedule = REPO_ROOT / schedule
    raw_evaluation_frames = config.get("evaluation_checkpoint_frames", [])
    if not isinstance(raw_evaluation_frames, list) or any(
        type(value) is not int or not 0 <= value < frame_count
        for value in raw_evaluation_frames
    ):
        raise ValueError(
            "evaluation_checkpoint_frames must contain in-range integer frames"
        )
    evaluation_frames = tuple(raw_evaluation_frames)
    if list(evaluation_frames) != sorted(set(evaluation_frames)):
        raise ValueError(
            "evaluation_checkpoint_frames must be strictly increasing and unique"
        )
    return str(scene), frame_count, schedule, evaluation_frames


def _checkpoint_plan_from_target_manifest(
    content: bytes,
    path: Path,
) -> dict[str, Any]:
    payload = _load_json_bytes(content, path)
    metadata = payload.get("metadata")
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_occlusion_v1_targets"
        and payload.get("dataset") == "TESSE-CD"
        and isinstance(metadata, Mapping)
    ):
        raise ValueError("occlusion target manifest identity mismatch")
    plan = build_evaluation_checkpoint_plan(arrays={}, metadata=metadata)
    declared_frames = metadata.get("scene_frame_indices")
    if not isinstance(declared_frames, Mapping) or set(declared_frames) != {
        "apartment",
        "office",
    }:
        raise ValueError("occlusion target checkpoint inventory is inconsistent")
    for scene in ("apartment", "office"):
        frames = declared_frames[scene]
        if not (
            isinstance(frames, list)
            and all(type(frame) is int and frame >= 0 for frame in frames)
            and frames == sorted(set(frames))
            and set(plan["evaluation_checkpoint_frames"][scene]).issubset(frames)
        ):
            raise ValueError("occlusion target checkpoint inventory is inconsistent")
    return plan


def _checkpoint_record(
    checkpoint: TesseCausalCheckpoint,
    *,
    scene: str,
    run_root: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    return {
        "scene": scene,
        "frame_index": checkpoint.frame_index,
        "timestamp_ns": checkpoint.timestamp_ns,
        "relative_timestamp_ns": checkpoint.relative_timestamp_ns,
        "consumed_through_frame": checkpoint.frame_index,
        "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
        "event_ids": list(checkpoint.event_ids),
        "roles": list(checkpoint.roles),
        "voxel_snapshot": _tree_record(
            checkpoint_root / "voxel_snapshot",
            relative_to=run_root,
        ),
        "artifact": _tree_record(
            checkpoint_root / "artifact",
            relative_to=run_root,
        ),
    }


def _compact_checkpoint_record(
    checkpoint: TesseCausalCheckpoint,
    *,
    scene: str,
    run_root: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    return {
        "scene": scene,
        "frame_index": checkpoint.frame_index,
        "timestamp_ns": checkpoint.timestamp_ns,
        "relative_timestamp_ns": checkpoint.relative_timestamp_ns,
        "consumed_through_frame": checkpoint.frame_index,
        "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
        "event_ids": list(checkpoint.event_ids),
        "roles": list(checkpoint.roles),
        "format": "oviv2_compact_ownership_checkpoint",
        "ownership_checkpoint": _tree_record(
            checkpoint_root / "ownership_checkpoint",
            relative_to=run_root,
        ),
    }


def _occlusion_index_record(
    checkpoint: TesseCausalCheckpoint,
    *,
    scene: str,
    run_root: Path,
    snapshot_root: Path,
    checkpoint_format: str,
) -> dict[str, Any]:
    checksums = snapshot_root / "checksums.json"
    _require_regular_file(checksums, "checkpoint checksum manifest")
    return {
        "scene": scene,
        "frame_index": checkpoint.frame_index,
        "timestamp_ns": checkpoint.timestamp_ns,
        "relative_timestamp_ns": checkpoint.relative_timestamp_ns,
        "consumed_through_frame": checkpoint.frame_index,
        "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
        "format": checkpoint_format,
        "path": snapshot_root.relative_to(run_root).as_posix(),
        "checksums_sha256": _sha256(checksums),
    }


def _export_paths(
    value: Mapping[str, Any] | None,
    *,
    checkpoint_root: Path,
) -> tuple[Path, Path, list[Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint exporter must return artifact paths")
    snapshot = Path(str(value.get("snapshot", "")))
    entities = Path(str(value.get("entities", "")))
    artifact_root = (checkpoint_root / "artifact").resolve()
    for path, role in ((snapshot, "snapshot"), (entities, "entities")):
        if not path.is_absolute():
            path = checkpoint_root / path
        _require_regular_file(path, f"checkpoint {role}")
        path = path.resolve()
        if not path.is_relative_to(artifact_root):
            raise ValueError(f"checkpoint exporter returned an invalid {role} path")
        if role == "snapshot":
            snapshot = path
        else:
            entities = path
    rows = value.get("trajectory_rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("checkpoint trajectory_rows must be a list of objects")
    return snapshot, entities, rows


class RunPublicationUncertainError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("run staging tree cannot contain symlinks")
        if path.is_dir():
            directories.append(path)
            continue
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise NotImplementedError("renameat2 is unavailable") from exc
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
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(target)
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }:
        raise NotImplementedError("RENAME_NOREPLACE is unavailable")
    raise OSError(error_number, os.strerror(error_number), target)


def _publish_run(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    _fsync_tree(staging)
    try:
        _rename_directory_no_replace(staging, output)
    except NotImplementedError:
        try:
            output.mkdir()
        except FileExistsError:
            raise FileExistsError(output) from None
        try:
            os.replace(staging, output)
        except BaseException as exc:
            raise RunPublicationUncertainError(
                f"run publication state is uncertain: {output}"
            ) from exc
    try:
        _fsync_directory(output.parent)
    except OSError as exc:
        raise RunPublicationUncertainError(
            f"run publication durability is uncertain: {output}"
        ) from exc


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    dependencies: RunnerDependencies | None = None,
) -> dict[str, Any]:
    source_config = Path(config_path).absolute()
    _require_regular_file(source_config, "runner config")
    source_config_bytes = source_config.read_bytes()
    config = _load_json_bytes(source_config_bytes, source_config)
    scene, frame_count, schedule_path, evaluation_frames = _validate_config(config)
    _require_regular_file(schedule_path, "causal schedule")
    source_schedule_bytes = schedule_path.read_bytes()
    official_checkpoints = tuple(
        _load_causal_checkpoints_bytes(
            source_schedule_bytes,
            schedule_path,
            scene=scene,
            frame_count=frame_count,
        )
    )
    target_manifest_path = _resolve_path(config.get("occlusion_target_manifest", ""))
    target_manifest_sha256 = config.get("occlusion_target_manifest_sha256")
    checkpoint_plan_sha256 = config.get("evaluation_checkpoint_frames_sha256")
    if not _is_sha256(target_manifest_sha256):
        raise ValueError("occlusion_target_manifest_sha256 is invalid")
    if not _is_sha256(checkpoint_plan_sha256):
        raise ValueError("evaluation_checkpoint_frames_sha256 is invalid")
    _require_regular_file(target_manifest_path, "occlusion target manifest")
    source_target_manifest_bytes = target_manifest_path.read_bytes()
    if _sha256_bytes(source_target_manifest_bytes) != target_manifest_sha256:
        raise ValueError("occlusion target manifest checksum binding mismatch")
    checkpoint_plan = _checkpoint_plan_from_target_manifest(
        source_target_manifest_bytes,
        target_manifest_path,
    )
    if (
        list(evaluation_frames)
        != checkpoint_plan["evaluation_checkpoint_frames"][scene]
        or checkpoint_plan_sha256
        != checkpoint_plan["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("evaluation checkpoint frame binding mismatch")
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    started = time.perf_counter()
    try:
        if dependencies is None:
            dependencies = _production_dependencies()
        dataset = dependencies.dataset_factory(config)
        if len(dataset) != frame_count:
            raise ValueError("dataset frame count does not match runner config")
        by_frame = {item.frame_index: item for item in official_checkpoints}
        first_timestamp_ns = int(dataset.timestamp_ns(0))
        for frame_index in evaluation_frames:
            if frame_index in by_frame:
                continue
            timestamp_ns = int(dataset.timestamp_ns(frame_index))
            by_frame[frame_index] = TesseCausalCheckpoint(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                relative_timestamp_ns=timestamp_ns - first_timestamp_ns,
                event_ids=(),
                roles=("occlusion_v1",),
            )
        checkpoints = tuple(by_frame[index] for index in sorted(by_frame))
        official_frame_set = {
            checkpoint.frame_index for checkpoint in official_checkpoints
        }
        caches = dependencies.cache_loader_factory(config, dataset)
        runtime = dependencies.runtime_factory(config, caches)
        checkpoint_records: list[dict[str, Any]] = []
        occlusion_records: list[dict[str, Any]] = []
        exported_sources: dict[int, dict[str, Any]] = {}
        trajectory_rows: list[Mapping[str, Any]] = []
        captured: list[int] = []
        checkpoint_identities: list[tuple[Any, _CheckpointIdentity]] = []
        for frame_index in range(frame_count):
            frame = dataset[frame_index]
            if int(frame.frame_id) != frame_index:
                raise ValueError("dataset frame IDs must equal zero-based frame indices")
            observations, dense_semantics = caches.load(frame_index, frame)
            runtime.process_frame(
                frame,
                observations=observations,
                dense_semantics=dense_semantics,
            )
            checkpoint = by_frame.get(frame_index)
            if checkpoint is None:
                continue
            timestamp_ns = int(dataset.timestamp_ns(frame_index))
            if timestamp_ns != checkpoint.timestamp_ns:
                raise ValueError("checkpoint timestamp does not match dataset timestamp")
            checkpoint_root = (
                staging
                / "checkpoints"
                / f"{frame_index:08d}-{checkpoint.timestamp_ns}"
            )
            checkpoint_root.mkdir(parents=True)
            if frame_index in official_frame_set:
                snapshot = runtime.commit_new(checkpoint_root / "voxel_snapshot")
                checkpoint_format = "oviv2_voxel_map_snapshot"
                snapshot_root = checkpoint_root / "voxel_snapshot"
            else:
                commit_compact = getattr(
                    runtime, "commit_compact_ownership_new", None
                )
                if not callable(commit_compact):
                    raise ValueError(
                        "runtime does not support compact ownership checkpoints"
                    )
                snapshot = commit_compact(
                    checkpoint_root / "ownership_checkpoint"
                )
                checkpoint_format = "oviv2_compact_ownership_checkpoint"
                snapshot_root = checkpoint_root / "ownership_checkpoint"
            checkpoint_identities.append(
                (snapshot, _capture_checkpoint_identity(snapshot_root))
            )
            if int(snapshot.metadata.frame_id) != frame_index:
                raise ValueError("committed snapshot frame does not match checkpoint")
            snapshot_timestamp = float(snapshot.metadata.timestamp)
            if snapshot_timestamp != float(checkpoint.timestamp_ns) and not np.isclose(
                snapshot_timestamp,
                checkpoint.timestamp_ns / 1_000_000_000,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("committed snapshot timestamp does not match checkpoint")
            if frame_index in official_frame_set:
                export_result = dependencies.checkpoint_exporter(
                    snapshot,
                    checkpoint,
                    checkpoint_root,
                    {
                        "config": config,
                        "dataset": dataset,
                        "caches": caches,
                    },
                )
                neutral_snapshot, neutral_entities, rows = _export_paths(
                    export_result,
                    checkpoint_root=checkpoint_root,
                )
                trajectory_rows.extend(rows)
                record = _checkpoint_record(
                    checkpoint,
                    scene=scene,
                    run_root=staging,
                    checkpoint_root=checkpoint_root,
                )
            else:
                record = _compact_checkpoint_record(
                    checkpoint,
                    scene=scene,
                    run_root=staging,
                    checkpoint_root=checkpoint_root,
                )
            status_path = checkpoint_root / "checkpoint_status.json"
            _write_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "checkpoint_frame": checkpoint.frame_index,
                    "timestamp_ns": checkpoint.timestamp_ns,
                    "event_ids": list(checkpoint.event_ids),
                    "roles": list(checkpoint.roles),
                    "consumed_through_frame": checkpoint.frame_index,
                    "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                },
            )
            record["checkpoint_status"] = _file_record(
                status_path,
                relative_to=staging,
            )
            if frame_index in official_frame_set:
                record.update(
                    {
                        "neutral_snapshot": _file_record(
                            neutral_snapshot,
                            relative_to=staging,
                        ),
                        "neutral_entities": _file_record(
                            neutral_entities,
                            relative_to=staging,
                        ),
                    }
                )
            checkpoint_records.append(record)
            if frame_index in official_frame_set:
                exported_sources[frame_index] = {
                    "frame_index": checkpoint.frame_index,
                    "timestamp_ns": checkpoint.timestamp_ns,
                    "consumed_through_frame": checkpoint.frame_index,
                    "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                    "checkpoint_status": record["checkpoint_status"],
                    "snapshot": record["neutral_snapshot"],
                    "entities": record["neutral_entities"],
                }
            if frame_index in set(evaluation_frames):
                occlusion_records.append(
                    _occlusion_index_record(
                        checkpoint,
                        scene=scene,
                        run_root=staging,
                        snapshot_root=snapshot_root,
                        checkpoint_format=checkpoint_format,
                    )
                )
            captured.append(frame_index)
        scheduled = [item.frame_index for item in checkpoints]
        if captured != scheduled:
            raise ValueError("captured checkpoints do not exactly match the schedule")
        checkpoint_parent = staging / "checkpoints"
        expected_directories = {
            f"{item.frame_index:08d}-{item.timestamp_ns}" for item in checkpoints
        }
        actual_directories = (
            {item.name for item in checkpoint_parent.iterdir()}
            if checkpoint_parent.is_dir()
            else set()
        )
        if actual_directories != expected_directories:
            raise ValueError("checkpoint directories do not exactly match the schedule")

        inputs_root = staging / "inputs"
        inputs_root.mkdir()
        schedule_copy = inputs_root / "schedule.json"
        schedule_copy.write_bytes(source_schedule_bytes)
        schedule_source_record = _file_record(schedule_copy, relative_to=staging)
        trajectories_path = staging / "trajectories.jsonl"
        trajectories_path.write_text(
            "".join(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in trajectory_rows
            ),
            encoding="utf-8",
        )
        trajectory_source_record = _file_record(
            trajectories_path,
            relative_to=staging,
        )
        official_frames = [item.frame_index for item in official_checkpoints]
        official_sources = [exported_sources[frame] for frame in official_frames]
        capture_path = staging / "capture_status.json"
        _write_json(
            capture_path,
            {
                "schema_version": 1,
                "status": "PASS",
                "scene": scene,
                "mode": "causal_checkpoints",
                "scheduled_frame_indices": official_frames,
                "captured_frame_indices": official_frames,
                "schedule": schedule_source_record,
                "trajectories": trajectory_source_record,
                "checkpoint_statuses": [
                    item["checkpoint_status"] for item in official_sources
                ],
            },
        )
        capture_source_record = _file_record(capture_path, relative_to=staging)
        _write_json(
            staging / "source_index.json",
            {
                "schema_version": 1,
                "dataset": "TESSE-CD",
                "mode": "causal_checkpoint_exports",
                "method": "OVIV2",
                "scene": scene,
                "schedule": schedule_source_record,
                "capture_status": capture_source_record,
                "trajectories": trajectory_source_record,
                "checkpoints": official_sources,
            },
        )

        _require_regular_file(source_config, "runner config")
        if source_config.read_bytes() != source_config_bytes:
            raise ValueError("runner config changed during run")
        _require_regular_file(schedule_path, "causal schedule")
        if schedule_path.read_bytes() != source_schedule_bytes:
            raise ValueError("causal schedule changed during run")
        _require_regular_file(target_manifest_path, "occlusion target manifest")
        if target_manifest_path.read_bytes() != source_target_manifest_bytes:
            raise ValueError("occlusion target manifest changed during run")
        config_record = _byte_record(source_config_bytes)
        schedule_record = _byte_record(source_schedule_bytes)
        cache_bindings = getattr(caches, "bindings", {})
        if not isinstance(cache_bindings, Mapping):
            raise ValueError("cache bindings must be a mapping")
        assert_inputs_unchanged = getattr(caches, "assert_inputs_unchanged", None)
        if assert_inputs_unchanged is not None:
            assert_inputs_unchanged()
        for committed_snapshot, identity in checkpoint_identities:
            revalidate_source = getattr(committed_snapshot, "revalidate_source", None)
            if callable(revalidate_source):
                revalidate_source()
            _revalidate_checkpoint_identity(identity)
        normalized_config_path = staging / "normalized_run_config.json"
        _write_json(normalized_config_path, config)
        normalized_config_record = _file_record(
            normalized_config_path,
            relative_to=staging,
        )
        occlusion_index_path = staging / "occlusion_checkpoint_index.json"
        _write_json(
            occlusion_index_path,
            {
                "schema_version": 2,
                "manifest_id": "oviv2_tesse_cd_occlusion_checkpoints_v1",
                "dataset": "TESSE-CD",
                "method_id": "OVIV2",
                "scene": scene,
                "algorithm_hash": algorithm_hash(config),
                "run_config": normalized_config_record,
                "target_manifest": {
                    "sha256": target_manifest_sha256,
                    "byte_count": len(source_target_manifest_bytes),
                },
                "evaluation_checkpoint_frames_sha256": checkpoint_plan_sha256,
                "snapshots": occlusion_records,
            },
        )
        occlusion_index_record = _file_record(
            occlusion_index_path,
            relative_to=staging,
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "mode": "causal_checkpoints",
            "scene": scene,
            "missing_observation_policy": config["missing_observation_policy"],
            "algorithm_hash": algorithm_hash(config),
            "normalized_algorithm_config": algorithm_config(config),
            "stage3_lineage_commit": config.get("stage3_lineage_commit"),
            "maintenance_parameters": {
                "visibility_depth_tolerance_m": config.get(
                    "visibility_depth_tolerance_m"
                ),
                "absence_negative_support": config.get(
                    "absence_negative_support"
                ),
                "ownership_min_net_support": config.get(
                    "ownership_min_net_support"
                ),
            },
            "processed_frame_count": frame_count,
            "official_schedule_frame_indices": [
                item.frame_index for item in official_checkpoints
            ],
            "evaluation_checkpoint_frames": list(evaluation_frames),
            "scheduled_frame_indices": scheduled,
            "captured_frame_indices": captured,
            "config": {
                "sha256": config_record["sha256"],
                "byte_count": config_record["byte_count"],
            },
            "schedule": {
                "sha256": schedule_record["sha256"],
                "byte_count": schedule_record["byte_count"],
            },
            "source_bindings": dict(cache_bindings),
            "occlusion_checkpoint_index": occlusion_index_record,
            "checkpoints": checkpoint_records,
        }
        _write_json(staging / "run_manifest.json", manifest)
        _write_json(
            staging / "run_provenance.json",
            {
                **dict(dependencies.provenance_factory()),
                "config_path": str(source_config),
                "output": str(destination),
                "python": platform.python_version(),
                "input_paths": {
                    name: str(_resolve_path(config[name]))
                    for name in (
                        "dataset_root",
                        "export_manifest",
                        "schedule_manifest",
                        "frontend_cache_dir",
                        "frontend_manifest",
                        "dense_cache_dir",
                        "dense_manifest",
                        "occlusion_target_manifest",
                        "vocabulary_json",
                        "vocabulary_txt",
                    )
                    if name in config
                },
            },
        )
        _write_json(
            staging / "timing.json",
            {
                "elapsed_sec": time.perf_counter() - started,
                "processed_frame_count": frame_count,
            },
        )
        _publish_run(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_cache_hashes(
    value: object,
    *,
    suffix: str,
    frame_count: int,
    role: str,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} cache_files_sha256 must be an object")
    expected_names = [f"frame{index:06d}.{suffix}" for index in range(frame_count)]
    if list(value) != expected_names:
        raise ValueError(f"{role} cache checksum keys are not canonical")
    hashes: dict[str, str] = {}
    for name in expected_names:
        checksum = value[name]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError(f"{role} cache checksum is invalid: {name}")
        hashes[name] = checksum
    return hashes


def _validate_file_binding(
    value: object,
    *,
    path: Path,
    role: str,
    require_exact_path: bool = True,
) -> None:
    if not isinstance(value, dict) or not _is_sha256(value.get("sha256")):
        raise ValueError(f"{role} binding is invalid")
    if (
        require_exact_path
        and _resolve_path(value.get("path")).resolve() != path.resolve()
    ):
        raise ValueError(f"{role} path binding mismatch")
    if value["sha256"] != _sha256(path):
        raise ValueError(f"{role} checksum binding mismatch")


def _validate_frontend_input_witness(
    value: object,
    *,
    scene: str,
    frame_count: int,
    dataset: Any,
    input_manifest: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "scene",
        "root",
        "validated_frame_count",
        "intrinsics",
        "export_manifest",
        "camera_manifest",
        "schedule_manifest",
        "source_manifest",
        "timestamps",
        "trajectory",
        "first_record",
        "last_record",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("scene") != scene
        or value.get("validated_frame_count") != frame_count
        or Path(str(value.get("root"))).resolve() != dataset.root.resolve()
    ):
        raise ValueError("frontend input witness identity mismatch")
    _validate_file_binding(
        value.get("camera_manifest"),
        path=dataset.camera_path,
        role="frontend witness camera",
    )
    _validate_file_binding(
        value.get("export_manifest"),
        path=dataset.export_manifest_path,
        role="frontend witness export manifest",
    )
    export_binding = value["export_manifest"]
    export_payload = _load_json(dataset.export_manifest_path)
    if (
        export_binding.get("combined_output_sha256")
        != export_payload.get("combined_output_sha256")
        or export_binding.get("validated_combined_output_sha256")
        != export_payload.get("combined_output_sha256")
        or export_binding.get("file_hash_count")
        != export_payload.get("file_hash_count")
        or export_binding.get("validated_file_hash_count")
        != export_payload.get("file_hash_count")
    ):
        raise ValueError("frontend witness export content binding mismatch")
    _validate_file_binding(
        value.get("timestamps"),
        path=dataset.timestamps_path,
        role="frontend witness timestamps",
    )
    _validate_file_binding(
        value.get("trajectory"),
        path=dataset.trajectory_path,
        role="frontend witness trajectory",
    )
    schedule_binding = value.get("schedule_manifest")
    if (
        not isinstance(schedule_binding, dict)
        or schedule_binding.get("sha256") != _sha256(dataset.schedule_manifest_path)
    ):
        raise ValueError("frontend witness schedule binding mismatch")
    source_binding = value.get("source_manifest")
    expected_source = input_manifest.get("source_manifest")
    if (
        not isinstance(source_binding, dict)
        or not isinstance(expected_source, dict)
        or source_binding.get("sha256") != expected_source.get("sha256")
    ):
        raise ValueError("frontend witness source manifest binding mismatch")
    records = getattr(dataset, "records", ())
    if len(records) != frame_count:
        raise ValueError("dataset does not expose the frozen RGB-D frame records")
    intrinsics = dataset.intrinsics
    expected_intrinsics = {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }
    if value.get("intrinsics") != expected_intrinsics:
        raise ValueError("frontend witness intrinsics binding mismatch")
    for raw, record, role in (
        (value.get("first_record"), records[0], "first"),
        (value.get("last_record"), records[-1], "last"),
    ):
        if (
            not isinstance(raw, dict)
            or raw.get("frame_index") != int(record.frame_index)
            or raw.get("timestamp_ns") != int(record.timestamp_ns)
            or raw.get("relative_timestamp_ns") != int(record.relative_timestamp_ns)
            or Path(str(raw.get("rgb_path"))).resolve() != record.rgb_path.resolve()
            or Path(str(raw.get("depth_path"))).resolve() != record.depth_path.resolve()
            or raw.get("camera_to_world_sha256")
            != _json_hash(np.asarray(record.camera_to_world, dtype=np.float64).tolist())
        ):
            raise ValueError(f"frontend witness {role} frame binding mismatch")


def _validate_dense_provenance(
    value: object,
    cache_hashes: Mapping[str, str],
) -> None:
    expected_keys = {
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
        "cache_prefix_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("backend") != "radseg"
        or any(
            not _is_sha256(item)
            for key, item in value.items()
            if key.endswith("sha256")
        )
        or any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in (
                "source_commit",
                "radio_commit",
                "model_id",
                "language_model_id",
                "language_model_revision",
            )
        )
        or value.get("cache_prefix_sha256")
        != _cache_prefix_sha256(cache_hashes)
    ):
        raise ValueError("dense manifest provenance does not match its cache")


@dataclass
class _ProductionCaches:
    frontend: Any
    structure: Any
    dense_cache_dir: Path
    dense_hashes: dict[str, str]
    dense_provenance: Any
    bindings: dict[str, Any]
    class_names: tuple[str, ...]
    object_semantic_ids: frozenset[int]
    timestamp_ns_by_frame: tuple[int, ...]
    semantic_fusion: Any
    input_hashes: dict[Path, str]

    def load(self, frame_index: int, frame: Any) -> tuple[tuple[Any, ...], Any]:
        from src.oviv2.dense_semantics import load_dense_frame

        object_observations = self.frontend.observe(frame, frame_index)
        structure_observations = self.structure.observe(
            frame,
            object_observations=object_observations,
        )
        name = f"frame{frame_index:06d}.npz"
        dense = load_dense_frame(
            self.dense_cache_dir / name,
            expected_sha256=self.dense_hashes[name],
        )
        if dense.cache_frame_id != frame_index or dense.source_frame_id != frame_index:
            raise ValueError("dense cache frame identity mismatch")
        return (*object_observations, *structure_observations), dense

    def assert_inputs_unchanged(self) -> None:
        for path, expected in self.input_hashes.items():
            _require_regular_file(path, "frozen runner input")
            if _sha256(path) != expected:
                raise ValueError(f"frozen runner input changed during run: {path}")


def _production_dataset_factory(config: Mapping[str, Any]) -> Any:
    from src.datasets.tesse_cd import TesseCdRgbdDataset

    return TesseCdRgbdDataset(
        _resolve_path(config.get("dataset_root")),
        str(config.get("scene")),
        _resolve_path(config.get("export_manifest")),
        _resolve_path(config.get("schedule_manifest")),
    )


def _production_cache_loader_factory(
    config: Mapping[str, Any],
    dataset: Any,
) -> _ProductionCaches:
    from scripts.precompute_oviv2_tesse_frontend import (
        _read_cache_snapshot,
        _validate_cache_payload,
    )
    from src.oviv2.dense_semantics import DenseSemanticProvenance
    from src.oviv2.observations import CachedFrontendAdapter, ReplicaVocabulary
    from src.oviv2.runner_config import (
        semantic_fusion_config_from_json,
        structure_config_from_json,
    )
    from src.oviv2.structure import DepthStructureFrontend

    scene = str(config["scene"])
    frame_count = int(config["frame_count"])
    input_manifest_path = _resolve_path(config.get("input_manifest"))
    vocabulary_path = _resolve_path(config.get("vocabulary_json"))
    vocabulary_txt = _resolve_path(config.get("vocabulary_txt"))
    frontend_dir = _resolve_path(config.get("frontend_cache_dir"))
    frontend_manifest_path = _resolve_path(config.get("frontend_manifest"))
    dense_dir = _resolve_path(config.get("dense_cache_dir"))
    dense_manifest_path = _resolve_path(config.get("dense_manifest"))
    if (
        frontend_manifest_path.parent.resolve() != frontend_dir.resolve()
        or frontend_manifest_path.name != "frontend_manifest.json"
    ):
        raise ValueError("frontend manifest must be inside the selected cache directory")
    if (
        dense_manifest_path.parent.resolve() != dense_dir.resolve()
        or dense_manifest_path.name != "dense_manifest.json"
    ):
        raise ValueError("dense manifest must be inside the selected cache directory")
    metadata_paths = (
        input_manifest_path,
        vocabulary_path,
        vocabulary_txt,
        frontend_manifest_path,
        dense_manifest_path,
        dataset.export_manifest_path,
        dataset.schedule_manifest_path,
        dataset.camera_path,
        dataset.timestamps_path,
        dataset.trajectory_path,
    )
    for path in metadata_paths:
        _require_regular_file(path, "frozen runner input")

    input_manifest = _load_json(input_manifest_path)
    if (
        input_manifest.get("schema_version") != 1
        or input_manifest.get("manifest_id") != "oviv2_tesse_cd_cache_v1"
        or input_manifest.get("dataset") != "TESSE-CD"
        or input_manifest.get("stage3_lineage_commit")
        != config.get("stage3_lineage_commit")
    ):
        raise ValueError("runner input manifest identity mismatch")
    scene_lock = input_manifest.get("scenes", {}).get(scene)
    if not isinstance(scene_lock, dict) or scene_lock.get("frame_count") != frame_count:
        raise ValueError("runner input manifest scene binding mismatch")
    vocabulary_lock = scene_lock.get("vocabulary")
    if not isinstance(vocabulary_lock, dict):
        raise ValueError("runner input manifest vocabulary binding is missing")
    for locked_path_key, locked_hash_key, actual_path, role in (
        ("json_path", "json_sha256", vocabulary_path, "JSON vocabulary"),
        ("txt_path", "txt_sha256", vocabulary_txt, "TXT vocabulary"),
    ):
        if (
            _resolve_path(vocabulary_lock.get(locked_path_key)).resolve()
            != actual_path.resolve()
            or vocabulary_lock.get(locked_hash_key) != _sha256(actual_path)
        ):
            raise ValueError(f"runner input manifest {role} binding mismatch")
    vocabulary_payload = _load_json(vocabulary_path)
    raw_classes = vocabulary_payload.get("classes")
    if (
        vocabulary_payload.get("scene") != scene
        or not isinstance(raw_classes, list)
        or not raw_classes
        or any(not isinstance(item, str) or not item.strip() for item in raw_classes)
    ):
        raise ValueError("runner vocabulary identity mismatch")
    classes = tuple(item.strip() for item in raw_classes)
    if vocabulary_txt.read_text(encoding="utf-8").splitlines() != list(classes):
        raise ValueError("runner JSON and TXT vocabularies disagree")

    frontend_manifest = _load_json(frontend_manifest_path)
    frontend_manifest_keys = {
        "schema_version",
        "method",
        "dataset",
        "scene",
        "frame_count",
        "source_frame_ids",
        "source_frame_ids_hash",
        "image_shape",
        "class_count",
        "classes",
        "vocabulary_sha256",
        "algorithm_hash",
        "feature_model_id",
        "input_manifest_sha256",
        "input_witness",
        "provenance_sha256",
        "cache_files_sha256",
        "cache_prefix_sha256",
    }
    frontend_provenance = frontend_manifest.get("provenance_sha256")
    frontend_provenance_keys = {
        "script",
        "hydra_config",
        "dataset_config",
        "yolo_model",
        "yolo_clip_model",
        "mobile_sam_model",
        "clip_model",
    }
    if (
        set(frontend_manifest) != frontend_manifest_keys
        or frontend_manifest.get("schema_version") != 1
        or frontend_manifest.get("method") != "OVIV2"
        or frontend_manifest.get("dataset") != "TESSE-CD"
        or frontend_manifest.get("scene") != scene
        or frontend_manifest.get("frame_count") != frame_count
        or frontend_manifest.get("source_frame_ids") != list(range(frame_count))
        or frontend_manifest.get("source_frame_ids_hash")
        != _json_hash(list(range(frame_count)))
        or frontend_manifest.get("image_shape") != [480, 720]
        or frontend_manifest.get("classes") != list(classes)
        or frontend_manifest.get("class_count") != len(classes)
        or not _is_sha256(frontend_manifest.get("algorithm_hash"))
        or frontend_manifest.get("input_manifest_sha256")
        != _sha256(input_manifest_path)
        or frontend_manifest.get("vocabulary_sha256") != _sha256(vocabulary_txt)
        or not isinstance(frontend_provenance, dict)
        or set(frontend_provenance) != frontend_provenance_keys
        or any(not _is_sha256(value) for value in frontend_provenance.values())
        or frontend_manifest.get("feature_model_id")
        != f"clip-sha256:{frontend_provenance.get('clip_model')}"
    ):
        raise ValueError("frontend manifest does not match the frozen runner inputs")
    _validate_frontend_input_witness(
        frontend_manifest.get("input_witness"),
        scene=scene,
        frame_count=frame_count,
        dataset=dataset,
        input_manifest=input_manifest,
    )
    frontend_hashes = _validate_cache_hashes(
        frontend_manifest.get("cache_files_sha256"),
        suffix="pkl.gz",
        frame_count=frame_count,
        role="frontend",
    )
    if frontend_manifest.get("cache_prefix_sha256") != _cache_prefix_sha256(
        frontend_hashes
    ):
        raise ValueError("frontend manifest cache prefix checksum mismatch")
    expected_frontend_entries = set(frontend_hashes) | {frontend_manifest_path.name}
    if {path.name for path in frontend_dir.iterdir()} != expected_frontend_entries:
        raise ValueError("frontend cache directory has missing or extra entries")

    dense_manifest = _load_json(dense_manifest_path)
    dense_manifest_keys = {
        "schema_version",
        "method",
        "scene",
        "frame_count",
        "source_frame_ids",
        "image_shape",
        "sample_stride",
        "top_k",
        "class_count",
        "vocabulary_sha256",
        "provenance",
        "cache_files_sha256",
    }
    if (
        set(dense_manifest) != dense_manifest_keys
        or dense_manifest.get("schema_version") != 1
        or dense_manifest.get("method") != "OVIV2-dense-semantic-cache"
        or dense_manifest.get("scene") != scene
        or dense_manifest.get("frame_count") != frame_count
        or dense_manifest.get("source_frame_ids") != list(range(frame_count))
        or dense_manifest.get("image_shape") != [480, 720]
        or dense_manifest.get("sample_stride") != config.get("dense_sample_stride")
        or dense_manifest.get("top_k") != config.get("dense_top_k")
        or dense_manifest.get("class_count") != len(classes)
        or dense_manifest.get("vocabulary_sha256") != _sha256(vocabulary_path)
    ):
        raise ValueError("dense manifest does not match the frozen runner inputs")
    dense_hashes = _validate_cache_hashes(
        dense_manifest.get("cache_files_sha256"),
        suffix="npz",
        frame_count=frame_count,
        role="dense",
    )
    if {path.name for path in dense_dir.iterdir()} != set(dense_hashes) | {
        dense_manifest_path.name
    }:
        raise ValueError("dense cache directory has missing or extra entries")
    provenance = dense_manifest.get("provenance")
    _validate_dense_provenance(provenance, dense_hashes)
    assert isinstance(provenance, dict)
    dense_provenance = DenseSemanticProvenance(**provenance)

    vocabulary = ReplicaVocabulary(
        classes=(*classes, "wall", "floor", "ceiling"),
        aliases={},
    )
    feature_model_id = frontend_manifest.get("feature_model_id")
    if not isinstance(feature_model_id, str) or not feature_model_id.strip():
        raise ValueError("frontend manifest feature_model_id must be non-empty")

    class BoundFrontendAdapter(CachedFrontendAdapter):
        def _load(self, cache_frame_id: int) -> tuple[Any, ...]:
            name = f"frame{cache_frame_id:06d}.pkl.gz"
            path = frontend_dir / name
            payload, binding = _read_cache_snapshot(path)
            if binding.sha256 != frontend_hashes[name]:
                raise ValueError("frontend cache checksum mismatch")
            _validate_cache_payload(
                payload,
                classes=classes,
                image_shape=(480, 720),
                path=path,
            )
            assert isinstance(payload, dict)
            masks = np.asarray(payload["mask"], dtype=bool)
            boxes = np.asarray(payload["xyxy"], dtype=np.float32)
            confidences = np.asarray(payload["confidence"], dtype=np.float32)
            class_ids = np.asarray(payload["class_id"], dtype=np.int64)
            image_features = np.asarray(payload["image_feats"], dtype=np.float64)
            text_features = np.asarray(payload["text_feats"], dtype=np.float64)
            labels = [classes[int(index)] for index in class_ids]
            return (
                masks,
                boxes,
                confidences,
                labels,
                image_features,
                text_features,
            )

    frontend = BoundFrontendAdapter(
        frontend_dir,
        vocabulary,
        voxel_size_m=float(config.get("voxel_size_m", 0.05)),
        pixel_stride=int(config.get("pixel_stride", 4)),
        min_valid_points=int(config.get("min_valid_points", 10)),
        feature_model_id=feature_model_id.strip(),
    )
    structure = DepthStructureFrontend(
        vocabulary,
        structure_config_from_json(
            dict(config),
            voxel_size_m=float(config.get("voxel_size_m", 0.05)),
        ),
    )
    semantic_fusion = semantic_fusion_config_from_json(dict(config))
    if semantic_fusion is None:
        raise ValueError("TESSE-CD runner requires Stage3 semantic fusion")
    bindings = {
        "input_manifest": _content_record(input_manifest_path),
        "export_manifest": _content_record(dataset.export_manifest_path),
        "camera": _content_record(dataset.camera_path),
        "trajectory": _content_record(dataset.trajectory_path),
        "timestamps": _content_record(dataset.timestamps_path),
        "vocabulary_json": _content_record(vocabulary_path),
        "vocabulary_txt": _content_record(vocabulary_txt),
        "frontend_manifest": _content_record(frontend_manifest_path),
        "dense_manifest": _content_record(dense_manifest_path),
        "rgbd_combined_output_sha256": _load_json(
            dataset.export_manifest_path
        ).get("combined_output_sha256"),
    }
    records = getattr(dataset, "records", ())
    if len(records) != frame_count:
        raise ValueError("dataset does not expose the frozen RGB-D frame records")
    frame_input_paths = tuple(
        path
        for record in records
        for path in (record.rgb_path, record.depth_path)
    )
    for path in frame_input_paths:
        _require_regular_file(path, "frozen RGB-D input")
    input_hashes = {
        path: _sha256(path) for path in (*metadata_paths, *frame_input_paths)
    }
    export_hash_paths = (
        *frame_input_paths,
        dataset.trajectory_path,
        dataset.timestamps_path,
        dataset.camera_path,
    )
    export_digest = hashlib.sha256()
    for path in sorted(
        export_hash_paths,
        key=lambda item: str(item.relative_to(dataset.camera_path.parent)),
    ):
        relative = str(path.relative_to(dataset.camera_path.parent))
        export_digest.update(
            relative.encode("utf-8")
            + b"\0"
            + input_hashes[path].encode("ascii")
            + b"\n"
        )
    export_manifest = _load_json(dataset.export_manifest_path)
    if (
        export_manifest.get("combined_output_sha256") != export_digest.hexdigest()
        or export_manifest.get("file_hash_count") != len(export_hash_paths)
    ):
        raise ValueError("RGB-D inputs changed after dataset validation")

    return _ProductionCaches(
        frontend=frontend,
        structure=structure,
        dense_cache_dir=dense_dir,
        dense_hashes=dense_hashes,
        dense_provenance=dense_provenance,
        bindings=bindings,
        class_names=("unknown", *classes, "wall", "floor", "ceiling"),
        object_semantic_ids=frozenset(range(1, len(classes) + 1)),
        timestamp_ns_by_frame=tuple(
            int(dataset.timestamp_ns(index)) for index in range(frame_count)
        ),
        semantic_fusion=semantic_fusion,
        input_hashes=input_hashes,
    )


def _production_runtime_factory(
    config: Mapping[str, Any],
    caches: _ProductionCaches,
) -> Any:
    from src.oviv2.runner_config import runtime_config_from_json
    from src.oviv2.runtime import Oviv2Runtime

    runtime_config = runtime_config_from_json(dict(config))
    runtime_config = apply_frozen_visibility_policy(
        runtime_config,
        str(config["missing_observation_policy"]),
    )
    return Oviv2Runtime(
        str(config["scene"]),
        runtime_config,
        dense_semantic_provenance=caches.dense_provenance,
    )


def _production_checkpoint_exporter(
    snapshot: Any,
    checkpoint: TesseCausalCheckpoint,
    destination: Path,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    from src.evaluation.exporters.oviovo import write_map_snapshot
    from src.evaluation.oviv2_tesse import build_neutral_current_snapshot

    caches = context.get("caches")
    if not isinstance(caches, _ProductionCaches):
        raise TypeError("production checkpoint export requires production caches")
    neutral = build_neutral_current_snapshot(
        snapshot,
        timestamp_ns=checkpoint.timestamp_ns,
        class_names=caches.class_names,
        object_semantic_ids=caches.object_semantic_ids,
        fusion=caches.semantic_fusion,
        timestamp_ns_by_frame=caches.timestamp_ns_by_frame,
    )
    paths = write_map_snapshot(neutral, destination / "artifact")
    rows = []
    for entity in neutral.entities:
        points = np.asarray(entity.points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("neutral entity must contain points for trajectory export")
        rows.append(
            {
                "frame_index": checkpoint.frame_index,
                "timestamp_ns": checkpoint.timestamp_ns,
                "entity_id": entity.entity_id,
                "centroid_xyz": [float(value) for value in points.mean(axis=0)],
            }
        )
    return {
        "snapshot": paths["snapshot"],
        "entities": paths["entities"],
        "trajectory_rows": rows,
    }


def _production_provenance() -> Mapping[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    try:
        gpu_inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu_inventory = []
    try:
        nvcc_version = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        nvcc_version = []
    try:
        import torch

        torch_cuda_version = torch.version.cuda or "unavailable"
        cudnn_version = torch.backends.cudnn.version()
    except (AttributeError, ImportError, RuntimeError):
        torch_cuda_version = "unavailable"
        cudnn_version = None

    library_versions: dict[str, str] = {}
    for distribution in ("numpy", "open3d", "torch", "scipy", "pillow"):
        try:
            library_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            library_versions[distribution] = "unavailable"
    return {
        "repository_commit": commit,
        "command": [str(value) for value in sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_version": torch_cuda_version,
        "cudnn_version": cudnn_version,
        "nvcc_version": nvcc_version,
        "gpu_inventory": gpu_inventory,
        "library_versions": library_versions,
    }


def _production_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        dataset_factory=_production_dataset_factory,
        cache_loader_factory=_production_cache_loader_factory,
        runtime_factory=_production_runtime_factory,
        checkpoint_exporter=_production_checkpoint_exporter,
        provenance_factory=_production_provenance,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args.config, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "processed_frame_count": manifest["processed_frame_count"],
                "scene": manifest["scene"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
