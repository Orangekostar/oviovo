#!/usr/bin/env python3
"""Compare cumulative audit artifacts byte-for-byte across OVIV2 runs."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from src.evaluation.baselines.dynamic_metrics import (  # noqa: E402
    DynamicFrameMetrics,
    aggregate_dynamic_metrics,
)


PROVENANCE_FIELDS = {
    "repository_commit",
    "repository_tree",
    "dirty_state_digest",
    "command",
    "hostname",
    "platform",
    "machine",
    "cuda_visible_devices",
    "torch_cuda_version",
    "cudnn_version",
    "nvcc_version",
    "gpu_inventory",
    "library_versions",
}
SCHEMA1_VARIANTS = {"production", "t1_transaction", "minimal"}
_MUTATION_MASK = (
    0x00000002  # IN_MODIFY
    | 0x00000004  # IN_ATTRIB
    | 0x00000008  # IN_CLOSE_WRITE
    | 0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
    | 0x00000400  # IN_DELETE_SELF
    | 0x00000800  # IN_MOVE_SELF
    | 0x00002000  # IN_UNMOUNT
)
PROVENANCE_LIBRARY_FIELDS = {"numpy", "open3d", "torch", "scipy", "pillow"}
SCHEMA1_SUPPORT_FILES = {
    "capture_status.json",
    "inputs/schedule.json",
    "normalized_run_config.json",
    "occlusion_checkpoint_index.json",
    "run_provenance.json",
    "source_index.json",
    "timing.json",
    "trajectories.jsonl",
}
SCHEMA1_PRODUCTION_FIELDS = {
    "schema_version",
    "dataset",
    "method_id",
    "scene",
    "mode",
    "processed_frame_count",
    "scheduled_frame_indices",
    "captured_frame_indices",
    "official_schedule_frame_indices",
    "evaluation_checkpoint_frames",
    "algorithm_hash",
    "normalized_algorithm_config",
    "config",
    "schedule",
    "source_bindings",
    "maintenance_parameters",
    "missing_observation_policy",
    "stage3_lineage_commit",
    "checkpoints",
    "occlusion_checkpoint_index",
}
SCHEMA1_T1_TRANSACTION_FIELDS = {
    "schema_version",
    "scene",
    "mode",
    "algorithm_hash",
    "code_commit",
    "config",
    "schedule",
    "target_manifest",
    "source_bindings",
    "checkpoints",
    "final_artifact",
}
MAX_JSON_BYTES = 16 * 1024 * 1024
SCHEMA2_MINIMAL_FIELDS = {
    "schema_version",
    "protocol_id",
    "algorithm_hash",
    "code_commit",
    "source_bindings",
    "normalized_run_config",
    "checkpoints",
    "final_artifact",
    "artifact_inventory",
}
SCHEMA2_PRODUCTION_FIELDS = {
    "schema_version",
    "protocol_id",
    "dataset",
    "method_id",
    "scene",
    "mode",
    "algorithm_hash",
    "processed_frame_count",
    "covered_frame_count",
    "trajectory_frame_count",
    "first_frame_index",
    "last_frame_index",
    "temporal_export_schema_version",
    "scheduled_frame_indices",
    "captured_frame_indices",
    "config",
    "normalized_run_config",
    "schedule",
    "target_manifest",
    "source_bindings",
    "input_sha256",
    "code_commit",
    "checkpoints",
    "occlusion_checkpoint_index",
    "source_index",
    "artifact_inventory",
}
SCHEMA2_PRODUCTION_CHECKPOINT_FIELDS = {
    "scene",
    "frame_index",
    "timestamp_ns",
    "relative_timestamp_ns",
    "consumed_through_frame",
    "consumed_through_frame_exclusive",
    "event_ids",
    "roles",
    "format",
    "artifact",
    "checksums_sha256",
    "artifacts",
    "cumulative_audit",
    "checkpoint_status",
}


class ArtifactMismatch(ValueError):
    """Raised when cumulative artifacts are incomplete, unsafe, or unequal."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactMismatch(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


@dataclass(frozen=True)
class FileEntry:
    root: Path
    path: PurePosixPath
    sha256: str
    byte_count: int
    identity: tuple[int, int, int, int, int]
    data: bytes | None = field(default=None, repr=False, compare=False)


@dataclass
class _RootHandle:
    path: Path
    descriptors: list[int]
    components: list[tuple[int, str, int, tuple[int, int, int]]]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for parent, name, child, opened_identity in self.components:
            try:
                linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
                current = os.fstat(child)
            except OSError as exc:
                raise ArtifactMismatch(
                    "run root path changed during comparison"
                ) from exc
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or _directory_identity(linked) != opened_identity
                or _directory_identity(current) != opened_identity
            ):
                raise ArtifactMismatch("run root path changed during comparison")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_root(root: Path) -> _RootHandle:
    absolute = Path(os.path.abspath(os.fspath(root)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptors: list[int] = []
    components: list[tuple[int, str, int, tuple[int, int, int]]] = []
    try:
        descriptor = os.open("/", flags)
        descriptors.append(descriptor)
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            descriptors.append(child)
            opened_identity = _directory_identity(os.fstat(child))
            linked = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or _directory_identity(linked) != opened_identity
            ):
                raise ArtifactMismatch("run root path component was replaced")
            components.append((descriptor, part, child, opened_identity))
            descriptor = child
        return _RootHandle(absolute, descriptors, components)
    except (OSError, ArtifactMismatch) as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(exc, ArtifactMismatch):
            raise
        raise ArtifactMismatch(
            "run root is missing, unsafe, or contains a symlink"
        ) from exc


def _open_regular(root: _RootHandle, relative: PurePosixPath, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.dup(root.descriptor)
        for part in relative.parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(relative.parts[-1], flags, dir_fd=descriptor)
        os.close(descriptor)
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise ArtifactMismatch(f"{label} is missing, unsafe, or contains a symlink") from exc
    if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
        os.close(file_descriptor)
        raise ArtifactMismatch(f"{label} is not a regular file")
    return file_descriptor


def _stream_entry(root: _RootHandle, relative: PurePosixPath, label: str) -> FileEntry:
    descriptor = _open_regular(root, relative, label)
    return _stream_descriptor(root, relative, descriptor, label)


def _stream_descriptor(
    root: _RootHandle, relative: PurePosixPath, descriptor: int, label: str
) -> FileEntry:
    digest = hashlib.sha256()
    total = 0
    capture = relative.as_posix() in {
        "run_manifest.json",
        "normalized_run_config.json",
        "execution_receipt.json",
        "t1_exact_receipt.json",
        "source_index.json",
        "capture_status.json",
        "occlusion_checkpoint_index.json",
        "run_provenance.json",
        "timing.json",
        "temporal/temporal_manifest.json",
        "temporal/sidecars/capture_status.json",
        "temporal/sidecars/schedule.json",
        "temporal/sidecars/source_index.json",
        "evaluation/summary.json",
        "evaluation/summary.canonical.json",
        "inputs/schedule.json",
        "trajectories.jsonl",
    }
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if capture:
                if total > MAX_JSON_BYTES:
                    raise ArtifactMismatch(f"{label} exceeds JSON size limit")
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or total != after.st_size:
        raise ArtifactMismatch(f"{label} changed while being read")
    return FileEntry(
        root.path,
        relative,
        digest.hexdigest(),
        total,
        _identity(after),
        b"".join(chunks) if capture else None,
    )


def _open_directory(
    root: _RootHandle, relative: PurePosixPath | None, label: str
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        descriptor = os.dup(root.descriptor)
        for part in (() if relative is None else relative.parts):
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise ArtifactMismatch(f"{label} tree is missing, unsafe, or contains a symlink") from exc


def _add_mutation_watches(descriptor: int, directory: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    add = libc.inotify_add_watch
    add.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    add.restype = ctypes.c_int
    if add(descriptor, os.fsencode(f"/proc/self/fd/{directory}"), _MUTATION_MASK) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    before = os.fstat(directory)
    names = sorted(os.listdir(directory))
    for name in names:
        metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
                dir_fd=directory,
            )
            try:
                if _identity(os.fstat(child)) != _identity(metadata):
                    raise ArtifactMismatch(
                        "artifact tree changed while mutation watches were installed"
                    )
                _add_mutation_watches(descriptor, child)
            finally:
                os.close(child)
        elif stat.S_ISLNK(metadata.st_mode):
            raise ArtifactMismatch("artifact tree contains a symlink")
        elif not stat.S_ISREG(metadata.st_mode):
            raise ArtifactMismatch("artifact tree contains a forbidden entry")
    if _identity(before) != _identity(os.fstat(directory)) or names != sorted(
        os.listdir(directory)
    ):
        raise ArtifactMismatch("artifact tree changed while mutation watches were installed")


def _begin_mutation_watch(root: _RootHandle) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    init = libc.inotify_init1
    init.argtypes = (ctypes.c_int,)
    init.restype = ctypes.c_int
    descriptor = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise ArtifactMismatch("artifact mutation witness is unavailable") from OSError(
            error, os.strerror(error)
        )
    directory = _open_directory(root, None, "artifact mutation witness")
    try:
        _add_mutation_watches(descriptor, directory)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        os.close(directory)


def _verify_mutation_watch(descriptor: int) -> None:
    try:
        data = os.read(descriptor, 64 * 1024)
    except BlockingIOError:
        return
    if data:
        raise ArtifactMismatch("artifact tree changed during comparison")


def _walk_directory(
    root: _RootHandle,
    descriptor: int,
    prefix: PurePosixPath | None,
    label: str,
) -> list[tuple[PurePosixPath, FileEntry]]:
    before = os.fstat(descriptor)
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise ArtifactMismatch(f"{label} tree cannot be listed") from exc
    result: list[tuple[PurePosixPath, FileEntry]] = []
    for name in names:
        if not name or name in {".", ".."} or "/" in name:
            raise ArtifactMismatch(f"{label} tree contains a noncanonical entry")
        path = PurePosixPath(name) if prefix is None else prefix / name
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactMismatch(f"{label} tree changed during traversal") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactMismatch(f"{label} tree contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ArtifactMismatch(f"{label} tree changed during traversal") from exc
            if _identity(os.fstat(file_descriptor)) != _identity(metadata):
                os.close(file_descriptor)
                raise ArtifactMismatch(f"{label} tree entry was replaced")
            result.append(
                (path, _stream_descriptor(root, path, file_descriptor, label))
            )
        elif stat.S_ISDIR(metadata.st_mode):
            try:
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ArtifactMismatch(f"{label} tree changed during traversal") from exc
            try:
                if _identity(os.fstat(child)) != _identity(metadata):
                    raise ArtifactMismatch(f"{label} tree directory was replaced")
                result.extend(_walk_directory(root, child, path, label))
            finally:
                os.close(child)
        else:
            raise ArtifactMismatch(f"{label} tree contains a forbidden entry")
    after = os.fstat(descriptor)
    try:
        final_names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise ArtifactMismatch(f"{label} tree changed during traversal") from exc
    if _identity(before) != _identity(after) or names != final_names:
        raise ArtifactMismatch(f"{label} tree changed during traversal")
    return result


def _relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactMismatch(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactMismatch(f"{label} path is noncanonical")
    if path.as_posix() != value:
        raise ArtifactMismatch(f"{label} path is noncanonical")
    return path


def _regular_bytes(
    all_files: Mapping[str, FileEntry], relative: PurePosixPath, label: str
) -> bytes:
    entry = all_files.get(relative.as_posix())
    if entry is None or entry.data is None:
        raise ArtifactMismatch(f"{label} is missing or cannot be decoded")
    return entry.data


def _record(record: object, label: str) -> tuple[PurePosixPath, str, int]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ArtifactMismatch(f"{label} manifest record fields are invalid")
    path = _relative(record["path"], label)
    digest = record["sha256"]
    count = record["byte_count"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(count) is not int
        or count < 0
    ):
        raise ArtifactMismatch(f"{label} manifest record is invalid")
    return path, digest, count


def _validate_file_record(
    all_files: Mapping[str, FileEntry], record: object, label: str
) -> tuple[PurePosixPath, FileEntry]:
    path, digest, count = _record(record, label)
    entry = all_files.get(path.as_posix())
    if entry is None:
        raise ArtifactMismatch(f"{label} manifest record is missing")
    if entry.byte_count != count or entry.sha256 != digest:
        raise ArtifactMismatch(f"{label} manifest record does not match raw bytes")
    return path, entry


def _validate_tree_record(
    all_files: Mapping[str, FileEntry], record: object, label: str
) -> list[tuple[PurePosixPath, FileEntry]]:
    path, digest, count = _record(record, label)
    prefix = path.as_posix() + "/"
    files = [
        (PurePosixPath(item), entry)
        for item, entry in all_files.items()
        if item.startswith(prefix)
    ]
    files.sort(key=lambda item: item[0].as_posix())
    if not files:
        raise ArtifactMismatch(f"{label} tree is empty")
    tree_digest = hashlib.sha256()
    total = 0
    result: list[tuple[PurePosixPath, FileEntry]] = []
    for item, entry in files:
        local = item.relative_to(path).as_posix()
        total += entry.byte_count
        tree_digest.update(local.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(entry.sha256))
        tree_digest.update(b"\n")
        result.append((item, entry))
    if total != count or tree_digest.hexdigest() != digest:
        raise ArtifactMismatch(f"{label} manifest record does not match raw bytes")
    return result


def _all_regular_files(root: _RootHandle) -> dict[str, FileEntry]:
    descriptor = _open_directory(root, None, "artifact inventory")
    try:
        entries = _walk_directory(root, descriptor, None, "artifact inventory")
    finally:
        os.close(descriptor)
    return {path.as_posix(): entry for path, entry in entries}


def _walk_inventory_paths(
    root: _RootHandle, descriptor: int, prefix: PurePosixPath | None
) -> set[str]:
    before = os.fstat(descriptor)
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise ArtifactMismatch("artifact inventory cannot be listed") from exc
    result: set[str] = set()
    for name in names:
        if not name or name in {".", ".."} or "/" in name:
            raise ArtifactMismatch("artifact inventory contains a noncanonical entry")
        path = PurePosixPath(name) if prefix is None else prefix / name
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactMismatch("artifact inventory changed during verification") from exc
        if stat.S_ISREG(metadata.st_mode):
            result.add(path.as_posix())
        elif stat.S_ISDIR(metadata.st_mode):
            child = _open_directory(root, path, "artifact inventory verification")
            try:
                result.update(_walk_inventory_paths(root, child, path))
            finally:
                os.close(child)
        else:
            raise ArtifactMismatch("artifact inventory contains a forbidden entry")
    after = os.fstat(descriptor)
    try:
        final_names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise ArtifactMismatch("artifact inventory changed during verification") from exc
    if _identity(before) != _identity(after) or names != final_names:
        raise ArtifactMismatch("artifact inventory changed during verification")
    return result


def _verify_full_inventory(
    root: _RootHandle, original: Mapping[str, FileEntry]
) -> None:
    descriptor = _open_directory(root, None, "artifact inventory verification")
    try:
        current_paths = _walk_inventory_paths(root, descriptor, None)
    finally:
        os.close(descriptor)
    if current_paths != set(original):
        raise ArtifactMismatch("artifact inventory changed before comparison completed")
    for path, entry in original.items():
        _verify_entry(root, entry, path)


def _validate_manifest_records(
    all_files: Mapping[str, FileEntry], value: object, label: str = "manifest"
) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "byte_count"}:
            path, _, _ = _record(value, label)
            key = path.as_posix()
            if any(item.startswith(key + "/") for item in all_files):
                _validate_tree_record(all_files, value, label)
            else:
                _validate_file_record(all_files, value, label)
            return
        for key, child in value.items():
            _validate_manifest_records(all_files, child, f"{label}.{key}")
    elif isinstance(value, list):
        for position, child in enumerate(value):
            _validate_manifest_records(all_files, child, f"{label}[{position}]")


def _record_inventory(
    all_files: Mapping[str, FileEntry], value: object
) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "byte_count"}:
            path, _, _ = _record(value, "manifest")
            key = path.as_posix()
            children = {item for item in all_files if item.startswith(key + "/")}
            result.update(children or {key})
        else:
            for child in value.values():
                result.update(_record_inventory(all_files, child))
    elif isinstance(value, list):
        for child in value:
            result.update(_record_inventory(all_files, child))
    return result


def _single_record_inventory(
    all_files: Mapping[str, FileEntry], value: object, label: str
) -> set[str]:
    path, _, _ = _record(value, label)
    key = path.as_posix()
    children = {item for item in all_files if item.startswith(key + "/")}
    return children or {key}


def _reject_source_binding_pseudo_records(value: object) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "byte_count"}:
            raise ArtifactMismatch("run source binding contains a pseudo-record")
        for child in value.values():
            _reject_source_binding_pseudo_records(child)
    elif isinstance(value, list):
        for child in value:
            _reject_source_binding_pseudo_records(child)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _string_list(value: object, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty or bool(value))
        and all(_nonempty_string(item) for item in value)
    )


def _index_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(_nonnegative_integer(item) for item in value)
        and value == sorted(set(value))
    )


def _validate_byte_record(value: object, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"sha256", "byte_count"}
        or not _hex_id(value.get("sha256"), (64,))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] < 0
    ):
        raise ArtifactMismatch(f"{label} schema is invalid")


def _validate_frozen_run_identity(value: object, label: str) -> None:
    expected = {
        "schema_version",
        "freeze_id",
        "protocol_id",
        "dataset",
        "method_id",
        "scene",
        "freeze_manifest",
        "repository",
        "config",
        "algorithm_hash",
        "input_bindings_sha256",
        "formal_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ArtifactMismatch(f"{label} schema is invalid")
    _validate_byte_record(value["freeze_manifest"], f"{label} freeze manifest")
    _validate_byte_record(value["config"], f"{label} config")
    repository = value.get("repository")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or not isinstance(repository, Mapping)
        or set(repository) != {"commit", "tree"}
        or not _hex_id(repository.get("commit"), (40, 64))
        or not _hex_id(repository.get("tree"), (40, 64))
        or any(
            not _hex_id(value.get(key), (64,))
            for key in (
                "algorithm_hash",
                "input_bindings_sha256",
                "formal_evidence_sha256",
            )
        )
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("freeze_id", "protocol_id", "dataset", "method_id", "scene")
        )
    ):
        raise ArtifactMismatch(f"{label} identity is invalid")


def _validate_cumulative_audit_structure(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactMismatch(f"{label} schema is invalid")
    expected = {"format", "artifact", "snapshot", "entities"}
    if "voxel_snapshot" in value:
        expected.add("voxel_snapshot")
    if set(value) != expected or value.get("format") != "oviv2_cumulative_audit_v1":
        raise ArtifactMismatch(f"{label} schema is invalid")
    for key in expected - {"format"}:
        _record(value[key], f"{label} {key}")


def _validate_schema2_structure(manifest: Mapping[str, Any]) -> None:
    fields = set(manifest)
    minimal_variants = {
        frozenset(SCHEMA2_MINIMAL_FIELDS),
        frozenset(SCHEMA2_MINIMAL_FIELDS | {"source_index"}),
        frozenset(SCHEMA2_MINIMAL_FIELDS | {"final_cumulative_audit"}),
        frozenset(
            SCHEMA2_MINIMAL_FIELDS | {"source_index", "final_cumulative_audit"}
        ),
    }
    production_variants = {
        frozenset(SCHEMA2_PRODUCTION_FIELDS),
        frozenset(SCHEMA2_PRODUCTION_FIELDS | {"frozen_run_identity"}),
    }
    if frozenset(fields) not in minimal_variants | production_variants:
        raise ArtifactMismatch("run manifest schema is invalid")
    production = frozenset(fields) in production_variants
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 2
        or not _nonempty_string(manifest.get("protocol_id"))
        or not _hex_id(manifest.get("algorithm_hash"), (64,))
        or not _hex_id(manifest.get("code_commit"), (40, 64))
        or not isinstance(manifest.get("source_bindings"), Mapping)
    ):
        raise ArtifactMismatch("run manifest identity is invalid")
    if production:
        if (
            any(
                not _nonempty_string(manifest.get(key))
                for key in ("dataset", "method_id", "scene", "mode")
            )
            or any(
                not _nonnegative_integer(manifest.get(key))
                for key in (
                    "processed_frame_count",
                    "covered_frame_count",
                    "trajectory_frame_count",
                    "first_frame_index",
                    "last_frame_index",
                    "temporal_export_schema_version",
                )
            )
            or not _index_list(manifest.get("scheduled_frame_indices"))
            or not _index_list(manifest.get("captured_frame_indices"))
            or not _hex_id(manifest.get("input_sha256"), (64,))
        ):
            raise ArtifactMismatch("run manifest production schema is invalid")
        for key in ("config", "schedule", "target_manifest"):
            _validate_byte_record(manifest[key], f"run manifest {key}")
        if "frozen_run_identity" in manifest:
            _validate_frozen_run_identity(
                manifest["frozen_run_identity"], "frozen run identity"
            )
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    minimal_checkpoint_fields = {
        "frame_index",
        "checkpoint_status",
        "cumulative_audit",
    }
    minimal_checkpoint_variants = {
        frozenset(minimal_checkpoint_fields),
        frozenset(minimal_checkpoint_fields | {"neutral_entities"}),
    }
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ArtifactMismatch("checkpoint schema is invalid")
        expected = set(SCHEMA2_PRODUCTION_CHECKPOINT_FIELDS)
        if production and "neutral_snapshot" in checkpoint:
            expected.update({"neutral_snapshot", "neutral_entities"})
        if (production and set(checkpoint) != expected) or (
            not production
            and frozenset(checkpoint) not in minimal_checkpoint_variants
        ):
            raise ArtifactMismatch("checkpoint schema is invalid")
        for key in ("checkpoint_status", "neutral_entities"):
            if key in checkpoint:
                _record(checkpoint[key], f"checkpoint {position} {key}")
        if "neutral_snapshot" in checkpoint:
            _record(
                checkpoint["neutral_snapshot"],
                f"checkpoint {position} neutral_snapshot",
            )
        _validate_cumulative_audit_structure(
            checkpoint.get("cumulative_audit"),
            f"checkpoint {position} cumulative audit",
        )
        if not _nonnegative_integer(checkpoint.get("frame_index")):
            raise ArtifactMismatch("checkpoint schema is invalid")
        if production:
            if (
                not _nonempty_string(checkpoint.get("scene"))
                or not _nonempty_string(checkpoint.get("format"))
                or any(
                    not _nonnegative_integer(checkpoint.get(key))
                    for key in (
                        "timestamp_ns",
                        "relative_timestamp_ns",
                        "consumed_through_frame",
                        "consumed_through_frame_exclusive",
                    )
                )
                or not _string_list(checkpoint.get("event_ids"))
                or not _string_list(checkpoint.get("roles"), nonempty=True)
                or not _hex_id(checkpoint.get("checksums_sha256"), (64,))
            ):
                raise ArtifactMismatch("checkpoint schema is invalid")
            _record(checkpoint["artifact"], f"checkpoint {position} artifact")
            artifacts = checkpoint.get("artifacts")
            if (
                not isinstance(artifacts, Mapping)
                or not artifacts
                or not set(artifacts).issubset(
                    {"temporal_current", "neutral_current", "temporal_compact"}
                )
            ):
                raise ArtifactMismatch("checkpoint artifacts schema is invalid")
            for role, artifact in artifacts.items():
                if (
                    not isinstance(artifact, Mapping)
                    or set(artifact) != {"format", "artifact", "checksums_sha256"}
                    or not isinstance(artifact.get("format"), str)
                    or not _hex_id(artifact.get("checksums_sha256"), (64,))
                ):
                    raise ArtifactMismatch("checkpoint artifact schema is invalid")
                _record(
                    artifact["artifact"],
                    f"checkpoint {position} {role} artifact",
                )
    final_audit = manifest.get("final_cumulative_audit")
    if final_audit is not None:
        _validate_cumulative_audit_structure(final_audit, "final cumulative audit")
    if "source_index" in manifest:
        source_index_path, _, _ = _record(
            manifest["source_index"], "manifest source index"
        )
        if source_index_path != PurePosixPath("source_index.json"):
            raise ArtifactMismatch("manifest source index path is invalid")
    _reject_source_binding_pseudo_records(manifest.get("source_bindings"))


def _schema2_manifest_inventory(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> set[str]:
    result: set[str] = set()
    for key in (
        "normalized_run_config",
        "final_artifact",
        "occlusion_checkpoint_index",
        "source_index",
    ):
        if key in manifest:
            result.update(
                _single_record_inventory(all_files, manifest[key], f"manifest {key}")
            )
    checkpoints = manifest.get("checkpoints")
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                continue
            for key in (
                "artifact",
                "checkpoint_status",
                "neutral_snapshot",
                "neutral_entities",
            ):
                if key in checkpoint:
                    result.update(
                        _single_record_inventory(
                            all_files, checkpoint[key], f"checkpoint {key}"
                        )
                    )
            if "cumulative_audit" in checkpoint:
                result.update(
                    _record_inventory(all_files, checkpoint["cumulative_audit"])
                )
            artifacts = checkpoint.get("artifacts")
            if isinstance(artifacts, Mapping):
                for role in ("temporal_current", "neutral_current", "temporal_compact"):
                    artifact = artifacts.get(role)
                    if isinstance(artifact, Mapping) and "artifact" in artifact:
                        result.update(
                            _single_record_inventory(
                                all_files,
                                artifact["artifact"],
                                f"checkpoint {role} artifact",
                            )
                        )
    if "final_cumulative_audit" in manifest:
        result.update(
            _record_inventory(all_files, manifest["final_cumulative_audit"])
        )
    if "source_index.json" in all_files:
        source_index = _json_object(
            _regular_bytes(
                all_files, PurePosixPath("source_index.json"), "source index"
            ),
            "source index",
        )
        expected_index_fields = {
            "schema_version",
            "dataset",
            "mode",
            "method",
            "scene",
            "schedule",
            "capture_status",
            "trajectories",
            "frame_coverage",
            "lifecycle_transitions",
            "checkpoints",
        }
        expected_index_variants = {
            frozenset(expected_index_fields),
            frozenset(expected_index_fields | {"runtime_diagnostics"}),
        }
        if "frozen_run_identity" in source_index:
            expected_index_variants = {
                fields | {"frozen_run_identity"}
                for fields in expected_index_variants
            }
        if frozenset(source_index) not in expected_index_variants:
            raise ArtifactMismatch("source index schema is invalid")
        if (
            type(source_index.get("schema_version")) is not int
            or source_index["schema_version"] != 1
            or any(
                not _nonempty_string(source_index.get(key))
                for key in ("dataset", "mode", "method", "scene")
            )
        ):
            raise ArtifactMismatch("source index identity is invalid")
        if "frozen_run_identity" in source_index:
            _validate_frozen_run_identity(
                source_index["frozen_run_identity"],
                "source index frozen run identity",
            )
        for key in (
            "schedule",
            "capture_status",
            "trajectories",
            "frame_coverage",
            "lifecycle_transitions",
        ):
            result.update(
                _single_record_inventory(
                    all_files, source_index[key], f"source index {key}"
                )
            )
        if "runtime_diagnostics" in source_index:
            diagnostics_path, _, _ = _record(
                source_index["runtime_diagnostics"],
                "source index runtime diagnostics",
            )
            if diagnostics_path != PurePosixPath("runtime_diagnostics.json"):
                raise ArtifactMismatch("source index runtime diagnostics path is invalid")
            _validate_file_record(
                all_files,
                source_index["runtime_diagnostics"],
                "source index runtime diagnostics",
            )
            result.update(
                _single_record_inventory(
                    all_files,
                    source_index["runtime_diagnostics"],
                    "source index runtime diagnostics",
                )
            )
        indexed_checkpoints = source_index.get("checkpoints")
        if not isinstance(indexed_checkpoints, list):
            raise ArtifactMismatch("source index checkpoint inventory is invalid")
        for checkpoint in indexed_checkpoints:
            if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
                "frame_index",
                "timestamp_ns",
                "consumed_through_frame",
                "consumed_through_frame_exclusive",
                "checkpoint_status",
                "snapshot",
                "entities",
            }:
                raise ArtifactMismatch("source index checkpoint inventory is invalid")
            if any(
                not _nonnegative_integer(checkpoint.get(key))
                for key in (
                    "frame_index",
                    "timestamp_ns",
                    "consumed_through_frame",
                    "consumed_through_frame_exclusive",
                )
            ):
                raise ArtifactMismatch("source index checkpoint schema is invalid")
            for key in ("checkpoint_status", "snapshot", "entities"):
                result.update(
                    _single_record_inventory(
                        all_files, checkpoint[key], f"source index {key}"
                    )
                )
    return result


def _schema1_manifest_inventory(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> set[str]:
    result: set[str] = set()
    checkpoints = manifest.get("checkpoints")
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                continue
            for key in (
                "artifact",
                "voxel_snapshot",
                "ownership_checkpoint",
                "checkpoint_status",
                "neutral_snapshot",
                "neutral_entities",
            ):
                if key in checkpoint:
                    result.update(
                        _single_record_inventory(
                            all_files, checkpoint[key], f"v1 checkpoint {key}"
                        )
                    )
    if "final_artifact" in manifest:
        result.update(
            _single_record_inventory(
                all_files, manifest["final_artifact"], "v1 final artifact"
            )
        )
    return result


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ArtifactMismatch(f"non-finite {label} value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactMismatch(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactMismatch(f"{label} root is invalid")
    return value


def _hex_id(value: object, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _schema_version(value: Mapping[str, Any], expected: int) -> bool:
    return (
        type(value.get("schema_version")) is int
        and value["schema_version"] == expected
    )


def _schema1_checkpoint_variant(manifest: Mapping[str, Any]) -> str:
    minimal = {
        "frame_index", "artifact", "voxel_snapshot", "checkpoint_status",
        "neutral_snapshot", "neutral_entities",
    }
    production_compact = {
        "checkpoint_status", "consumed_through_frame",
        "consumed_through_frame_exclusive", "event_ids", "format", "frame_index",
        "ownership_checkpoint", "relative_timestamp_ns", "roles", "scene",
        "timestamp_ns",
    }
    production_voxel = (minimal | {
        "consumed_through_frame", "consumed_through_frame_exclusive", "event_ids",
        "relative_timestamp_ns", "roles", "scene", "timestamp_ns",
    })
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("schema1 checkpoint inventory is invalid")
    shapes = [frozenset(checkpoint) if isinstance(checkpoint, Mapping) else frozenset() for checkpoint in checkpoints]
    if all(shape == minimal for shape in shapes):
        variant = "minimal"
    elif all(shape in {frozenset(production_compact), frozenset(production_voxel)} for shape in shapes):
        variant = "production"
    else:
        raise ArtifactMismatch("schema1 checkpoint schema is not a known closed-world variant")
    for checkpoint in checkpoints:
        frame = checkpoint["frame_index"]
        status_path, _, _ = _record(checkpoint["checkpoint_status"], "schema1 checkpoint status")
        parent = status_path.parent
        parts = parent.name.split("-", 1)
        if (
            len(parent.parts) != 2
            or parent.parts[0] != "checkpoints"
            or len(parts) != 2
            or len(parts[0]) != 8
            or not all(part.isdigit() for part in parts)
            or not _nonnegative_integer(frame)
            or int(parts[0]) != frame
            or status_path != parent / "checkpoint_status.json"
        ):
            raise ArtifactMismatch("schema1 checkpoint physical path is invalid")
        for key, child in (
            ("artifact", "artifact"), ("voxel_snapshot", "voxel_snapshot"),
            ("ownership_checkpoint", "ownership_checkpoint"),
        ):
            if key in checkpoint and _record(checkpoint[key], f"schema1 checkpoint {key}")[0] != parent / child:
                raise ArtifactMismatch("schema1 checkpoint physical path is invalid")
        for key in ("neutral_snapshot", "neutral_entities"):
            if key in checkpoint:
                path, _, _ = _record(checkpoint[key], f"schema1 checkpoint {key}")
                if not path.is_relative_to(parent / "artifact"):
                    raise ArtifactMismatch("schema1 checkpoint physical path is invalid")
    return variant


def _validate_schema1_structure(manifest: Mapping[str, Any], variant: str) -> bool:
    fields = frozenset(manifest)
    minimal_variants = {
        frozenset({"schema_version", "checkpoints"}),
        frozenset({"schema_version", "checkpoints", "final_artifact"}),
    }
    production_variants = {
        frozenset(SCHEMA1_PRODUCTION_FIELDS),
        frozenset(SCHEMA1_PRODUCTION_FIELDS | {"frozen_run_identity", "run_execution"}),
    }
    checkpoint_variant = _schema1_checkpoint_variant(manifest)
    if fields in minimal_variants:
        if variant != "minimal":
            raise ArtifactMismatch("schema1 manifest does not match caller-trusted variant")
        if checkpoint_variant != "minimal":
            raise ArtifactMismatch("schema1 minimal checkpoint schema is invalid")
        return False
    if fields == frozenset(SCHEMA1_T1_TRANSACTION_FIELDS):
        if variant != "t1_transaction":
            raise ArtifactMismatch("schema1 manifest does not match caller-trusted variant")
        if checkpoint_variant != "minimal":
            raise ArtifactMismatch("schema1 T1 checkpoint schema is invalid")
        if (
            not _schema_version(manifest, 1)
            or not _nonempty_string(manifest.get("scene"))
            or manifest.get("mode") != "causal_checkpoints"
            or not _hex_id(manifest.get("algorithm_hash"), (64,))
            or not _hex_id(manifest.get("code_commit"), (40, 64))
            or not isinstance(manifest.get("source_bindings"), Mapping)
            or not manifest["source_bindings"]
        ):
            raise ArtifactMismatch("schema1 T1 transaction identity is invalid")
        for key in ("config", "schedule", "target_manifest"):
            _validate_byte_record(manifest[key], f"schema1 T1 transaction {key}")
        return False
    if fields not in production_variants:
        raise ArtifactMismatch(
            "schema1 run manifest schema is invalid; artifact inventory is not exact"
        )
    if variant != "production":
        raise ArtifactMismatch("schema1 manifest does not match caller-trusted variant")
    if checkpoint_variant != "production":
        raise ArtifactMismatch("schema1 production checkpoint schema is invalid")
    checkpoints = manifest.get("checkpoints")
    checkpoint_frames = (
        [checkpoint.get("frame_index") for checkpoint in checkpoints]
        if isinstance(checkpoints, list)
        and all(isinstance(checkpoint, Mapping) for checkpoint in checkpoints)
        else None
    )
    scheduled = manifest.get("scheduled_frame_indices")
    captured = manifest.get("captured_frame_indices")
    official = manifest.get("official_schedule_frame_indices")
    evaluation = manifest.get("evaluation_checkpoint_frames")
    if (
        not _schema_version(manifest, 1)
        or any(
            not _nonempty_string(manifest.get(key))
            for key in (
                "dataset",
                "method_id",
                "scene",
                "mode",
                "missing_observation_policy",
            )
        )
        or manifest.get("mode") != "causal_checkpoints"
        or not _nonnegative_integer(manifest.get("processed_frame_count"))
        or any(
            not _index_list(value)
            for value in (scheduled, captured, official, evaluation)
        )
        or scheduled != captured
        or checkpoint_frames != scheduled
        or checkpoint_frames != sorted(set(official) | set(evaluation))
        or not _hex_id(manifest.get("algorithm_hash"), (64,))
        or not _hex_id(manifest.get("stage3_lineage_commit"), (40, 64))
        or not isinstance(manifest.get("normalized_algorithm_config"), Mapping)
        or not isinstance(manifest.get("source_bindings"), Mapping)
        or not manifest["source_bindings"]
        or not isinstance(manifest.get("maintenance_parameters"), Mapping)
    ):
        raise ArtifactMismatch("schema1 production manifest identity is invalid")
    _validate_byte_record(manifest["config"], "schema1 run config")
    _validate_byte_record(manifest["schedule"], "schema1 run schedule")
    _reject_source_binding_pseudo_records(manifest["source_bindings"])
    return True


def _support_json(
    all_files: Mapping[str, FileEntry], path: str, label: str
) -> dict[str, Any]:
    return _json_object(
        _regular_bytes(all_files, PurePosixPath(path), label), label
    )


def _validate_fixed_file_record(
    all_files: Mapping[str, FileEntry], record: object, path: str, label: str
) -> None:
    record_path, _ = _validate_file_record(all_files, record, label)
    if record_path != PurePosixPath(path):
        raise ArtifactMismatch(f"{label} path is invalid")


def _validate_schema1_frozen_identity(value: object, label: str) -> None:
    expected = {
        "schema_version",
        "freeze_id",
        "dataset",
        "method_id",
        "scene",
        "freeze_manifest",
        "repository",
        "config",
        "algorithm_hash",
        "input_bindings_sha256",
        "missing_observation_policy",
    }
    repository = value.get("repository") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not _schema_version(value, 1)
        or any(
            not _nonempty_string(value.get(key))
            for key in (
                "freeze_id",
                "dataset",
                "method_id",
                "scene",
                "missing_observation_policy",
            )
        )
        or not _hex_id(value.get("algorithm_hash"), (64,))
        or not _hex_id(value.get("input_bindings_sha256"), (64,))
        or not isinstance(repository, Mapping)
        or set(repository) != {"commit", "tree"}
        or not _hex_id(repository.get("commit"), (40, 64))
        or not _hex_id(repository.get("tree"), (40, 64))
    ):
        raise ArtifactMismatch(f"{label} schema is invalid")
    _validate_byte_record(value["freeze_manifest"], f"{label} freeze manifest")
    _validate_byte_record(value["config"], f"{label} config")


def _validate_schema1_run_execution(
    value: object, root: _RootHandle, label: str
) -> None:
    expected = {
        "schema_version",
        "run_slot",
        "output_root",
        "root_device",
        "root_inode",
        "execution_id",
    }
    root_metadata = os.fstat(root.descriptor)
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not _schema_version(value, 1)
        or not _nonempty_string(value.get("run_slot"))
        or value.get("output_root") != str(root.path)
        or value.get("root_device") != root_metadata.st_dev
        or value.get("root_inode") != root_metadata.st_ino
        or not _hex_id(value.get("execution_id"), (64,))
    ):
        raise ArtifactMismatch(f"{label} schema is invalid")


def _validate_schema1_source_index(
    root: _RootHandle,
    all_files: Mapping[str, FileEntry],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    base_fields = {
        "schema_version",
        "dataset",
        "method",
        "mode",
        "scene",
        "schedule",
        "capture_status",
        "trajectories",
        "checkpoints",
    }
    frozen_fields = base_fields | {"frozen_run_identity", "run_execution"}
    if set(source) not in (base_fields, frozen_fields):
        raise ArtifactMismatch("source index schema is invalid")
    if (
        not _schema_version(source, 1)
        or source.get("dataset") != manifest.get("dataset")
        or source.get("method") != manifest.get("method_id")
        or source.get("scene") != manifest.get("scene")
        or source.get("mode") != "causal_checkpoint_exports"
    ):
        raise ArtifactMismatch("source index identity is invalid")
    if set(source) == frozen_fields:
        if (
            source["frozen_run_identity"] != manifest.get("frozen_run_identity")
            or source["run_execution"] != manifest.get("run_execution")
        ):
            raise ArtifactMismatch("source index frozen identity mismatch")
        _validate_schema1_frozen_identity(
            source["frozen_run_identity"], "source index frozen identity"
        )
        _validate_schema1_run_execution(
            source["run_execution"], root, "source index run execution"
        )
    _validate_fixed_file_record(
        all_files, source["schedule"], "inputs/schedule.json", "source index schedule"
    )
    _validate_fixed_file_record(
        all_files,
        source["capture_status"],
        "capture_status.json",
        "source index capture status",
    )
    _validate_fixed_file_record(
        all_files,
        source["trajectories"],
        "trajectories.jsonl",
        "source index trajectories",
    )
    checkpoints = source.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("source index checkpoint inventory is invalid")
    manifest_checkpoints = {
        checkpoint.get("frame_index"): checkpoint
        for checkpoint in manifest.get("checkpoints", [])
        if isinstance(checkpoint, Mapping)
    }
    frames: list[int] = []
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "frame_index",
            "timestamp_ns",
            "consumed_through_frame",
            "consumed_through_frame_exclusive",
            "checkpoint_status",
            "snapshot",
            "entities",
        }:
            raise ArtifactMismatch("source index checkpoint schema is invalid")
        frame = checkpoint.get("frame_index")
        if (
            not _nonnegative_integer(frame)
            or frame in frames
            or not _nonnegative_integer(checkpoint.get("timestamp_ns"))
            or checkpoint.get("consumed_through_frame") != frame
            or checkpoint.get("consumed_through_frame_exclusive") != frame + 1
        ):
            raise ArtifactMismatch("source index checkpoint identity is invalid")
        frames.append(frame)
        target = manifest_checkpoints.get(frame)
        if (
            target is None
            or checkpoint["checkpoint_status"] != target.get("checkpoint_status")
            or checkpoint["snapshot"] != target.get("neutral_snapshot")
            or checkpoint["entities"] != target.get("neutral_entities")
        ):
            raise ArtifactMismatch("source index checkpoint binding is invalid")
        for key in ("checkpoint_status", "snapshot", "entities"):
            _validate_file_record(
                all_files, checkpoint[key], f"source index checkpoint {position} {key}"
            )
    if frames != sorted(frames) or frames != manifest.get(
        "official_schedule_frame_indices"
    ):
        raise ArtifactMismatch("source index checkpoint inventory is invalid")
    return checkpoints


def _validate_schema1_capture_status(
    all_files: Mapping[str, FileEntry],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    source_checkpoints: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version",
        "status",
        "scene",
        "mode",
        "scheduled_frame_indices",
        "captured_frame_indices",
        "schedule",
        "trajectories",
        "checkpoint_statuses",
    }
    scheduled = capture.get("scheduled_frame_indices")
    captured = capture.get("captured_frame_indices")
    if (
        set(capture) != expected
        or not _schema_version(capture, 1)
        or capture.get("status") != "PASS"
        or capture.get("scene") != manifest.get("scene")
        or capture.get("mode") != "causal_checkpoints"
        or not _index_list(scheduled)
        or captured != scheduled
        or scheduled != manifest.get("official_schedule_frame_indices")
        or capture.get("schedule") != source.get("schedule")
        or capture.get("trajectories") != source.get("trajectories")
        or capture.get("checkpoint_statuses")
        != [checkpoint["checkpoint_status"] for checkpoint in source_checkpoints]
    ):
        raise ArtifactMismatch("capture status schema or binding is invalid")
    for position, record in enumerate(capture["checkpoint_statuses"]):
        _validate_file_record(
            all_files, record, f"capture status checkpoint {position}"
        )


def _validate_schema1_schedule(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> dict[int, int]:
    schedule = _support_json(all_files, "inputs/schedule.json", "causal schedule")
    expected_fields = {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_predictions_used",
        "source_manifest",
        "parameters",
        "scenes",
    }
    parameter_fields = {
        "frame_indexing",
        "official_stride_frames",
        "common_event_step_frames",
        "common_event_horizon_frames",
        "common_checkpoints_per_event",
        "event_frame_rule",
    }
    scene_fields = {
        "frame_count",
        "entries",
        "events",
        "first_depth_timestamp_ns",
        "last_depth_timestamp_ns",
        "sources",
    }
    entry_fields = {
        "frame_index",
        "timestamp_ns",
        "relative_timestamp_ns",
        "event_ids",
        "roles",
    }
    event_fields = {
        "common_checkpoint_frame_indices",
        "event_id",
        "event_relative_timestamp_ns",
        "intervention_frame_index",
        "intervention_relative_timestamp_ns",
        "intervention_timestamp_ns",
    }
    parameters = schedule.get("parameters")
    scenes = schedule.get("scenes")
    scene = scenes.get(manifest.get("scene")) if isinstance(scenes, Mapping) else None
    if (
        set(schedule) != expected_fields
        or not _schema_version(schedule, 2)
        or schedule.get("manifest_id") != "tesse_cd_causal_schedule_v2"
        or schedule.get("dataset") != manifest.get("dataset")
        or schedule.get("method_predictions_used") is not False
        or not isinstance(parameters, Mapping)
        or set(parameters) != parameter_fields
        or parameters.get("frame_indexing") != "zero_based"
        or any(
            not _nonnegative_integer(parameters.get(key))
            for key in parameter_fields - {"frame_indexing", "event_frame_rule"}
        )
        or parameters.get("common_event_step_frames", 0) <= 0
        or parameters.get("common_checkpoints_per_event", 0) <= 0
        or parameters.get("common_event_horizon_frames")
        != parameters.get("common_event_step_frames")
        * (parameters.get("common_checkpoints_per_event") - 1)
        or not _nonempty_string(parameters.get("event_frame_rule"))
        or not isinstance(scene, Mapping)
        or set(scene) != scene_fields
        or scene.get("frame_count") != manifest.get("processed_frame_count")
        or not isinstance(scene.get("events"), list)
        or not isinstance(scene.get("sources"), Mapping)
        or not _nonnegative_integer(scene.get("first_depth_timestamp_ns"))
        or not _nonnegative_integer(scene.get("last_depth_timestamp_ns"))
    ):
        raise ArtifactMismatch("causal schedule schema or identity is invalid")
    _record(schedule["source_manifest"], "causal schedule source manifest")
    sources = scene["sources"]
    if set(sources) != {"database", "gt_changes"}:
        raise ArtifactMismatch("causal schedule sources schema is invalid")
    for role, record in sources.items():
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "byte_count"}
            or not _nonempty_string(record.get("path"))
            or not _hex_id(record.get("sha256"), (64,))
            or not _nonnegative_integer(record.get("byte_count"))
        ):
            raise ArtifactMismatch(f"causal schedule source {role} is invalid")
    events = scene["events"]
    event_ids: set[str] = set()
    event_frames: dict[str, list[int]] = {}
    for event in events:
        if (
            not isinstance(event, Mapping)
            or set(event) != event_fields
            or not _nonempty_string(event.get("event_id"))
            or event["event_id"] in event_ids
            or any(
                not _nonnegative_integer(event.get(key))
                for key in event_fields
                - {"event_id", "common_checkpoint_frame_indices"}
            )
            or not _index_list(event.get("common_checkpoint_frame_indices"))
            or len(event["common_checkpoint_frame_indices"])
            != parameters["common_checkpoints_per_event"]
            or event["common_checkpoint_frame_indices"]
            != list(
                range(
                    event["intervention_frame_index"],
                    event["intervention_frame_index"]
                    + parameters["common_event_horizon_frames"]
                    + 1,
                    parameters["common_event_step_frames"],
                )
            )
            or event["event_relative_timestamp_ns"]
            > event["intervention_relative_timestamp_ns"]
        ):
            raise ArtifactMismatch("causal schedule event is invalid")
        event_ids.add(event["event_id"])
        event_frames[event["event_id"]] = event["common_checkpoint_frame_indices"]
    entries = scene.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ArtifactMismatch("causal schedule entries are invalid")
    timestamps: dict[int, int] = {}
    previous_frame = -1
    previous_timestamp = -1
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != entry_fields
            or not _nonnegative_integer(entry.get("frame_index"))
            or not _nonnegative_integer(entry.get("timestamp_ns"))
            or not _nonnegative_integer(entry.get("relative_timestamp_ns"))
            or entry["frame_index"] <= previous_frame
            or entry["timestamp_ns"] <= previous_timestamp
            or entry["frame_index"] >= scene["frame_count"]
            or not _string_list(entry.get("event_ids"))
            or not _string_list(entry.get("roles"), nonempty=True)
            or not set(entry["roles"]).issubset({"official", "common_v2"})
            or not set(entry["event_ids"]).issubset(event_ids)
        ):
            raise ArtifactMismatch("causal schedule entry is invalid")
        timestamps[entry["frame_index"]] = entry["timestamp_ns"]
        previous_frame = entry["frame_index"]
        previous_timestamp = entry["timestamp_ns"]
    for event_id, frames in event_frames.items():
        if [
            entry["frame_index"] for entry in entries if event_id in entry["event_ids"]
        ] != frames:
            raise ArtifactMismatch("causal schedule event frame binding mismatch")
        event = next(item for item in events if item["event_id"] == event_id)
        intervention_entry = next(
            item for item in entries
            if item["frame_index"] == event["intervention_frame_index"]
        )
        if (
            intervention_entry["timestamp_ns"] != event["intervention_timestamp_ns"]
            or intervention_entry["relative_timestamp_ns"]
            != event["intervention_relative_timestamp_ns"]
        ):
            raise ArtifactMismatch("causal schedule event intervention binding mismatch")
    if list(timestamps) != manifest.get("official_schedule_frame_indices"):
        raise ArtifactMismatch("causal schedule frame inventory mismatch")
    manifest_checkpoints = {
        checkpoint.get("frame_index"): checkpoint
        for checkpoint in manifest["checkpoints"]
    }
    if any(
        manifest_checkpoints.get(frame, {}).get("timestamp_ns") != timestamp
        for frame, timestamp in timestamps.items()
    ):
        raise ArtifactMismatch("causal schedule checkpoint binding mismatch")
    for entry in entries:
        checkpoint = manifest_checkpoints[entry["frame_index"]]
        if any(
            checkpoint.get(key) != entry[key]
            for key in ("relative_timestamp_ns", "event_ids", "roles")
        ):
            raise ArtifactMismatch("causal schedule checkpoint semantics mismatch")
    return timestamps


def _validate_schema1_trajectories(
    all_files: Mapping[str, FileEntry], schedule_timestamps: Mapping[int, int]
) -> None:
    data = _regular_bytes(
        all_files, PurePosixPath("trajectories.jsonl"), "trajectories"
    )
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ArtifactMismatch("trajectories are not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise ArtifactMismatch("trajectories contain a blank line")
    expected_fields = {"frame_index", "timestamp_ns", "entity_id", "centroid_xyz"}
    seen: set[tuple[int, str]] = set()
    frames: list[int] = []
    previous_frame = -1
    previous_timestamp = -1
    for position, line in enumerate(lines, start=1):
        record = _json_object(line.encode("utf-8"), f"trajectory line {position}")
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        entity_id = record.get("entity_id")
        centroid = record.get("centroid_xyz")
        if (
            set(record) != expected_fields
            or not _nonnegative_integer(frame)
            or not _nonnegative_integer(timestamp)
            or not _nonempty_string(entity_id)
            or entity_id != entity_id.strip()
            or frame < previous_frame
            or timestamp < previous_timestamp
            or (frame == previous_frame and timestamp != previous_timestamp)
            or (frame > previous_frame and timestamp <= previous_timestamp)
            or schedule_timestamps.get(frame) != timestamp
            or (frame, entity_id) in seen
            or not isinstance(centroid, list)
            or len(centroid) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in centroid
            )
        ):
            raise ArtifactMismatch("trajectory record is invalid")
        if frame != previous_frame:
            frames.append(frame)
        seen.add((frame, entity_id))
        previous_frame = frame
        previous_timestamp = timestamp
    if frames != list(schedule_timestamps):
        raise ArtifactMismatch("trajectory frame inventory mismatch")


def _validate_schema1_normalized_config(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = _support_json(
        all_files, "normalized_run_config.json", "normalized run config"
    )
    try:
        algorithm_hash = canonical_algorithm_hash(normalized)
        canonical = canonical_algorithm_config(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactMismatch("normalized run config algorithm identity is invalid") from exc
    if (
        normalized.get("algorithm_hash") != algorithm_hash
        or manifest.get("algorithm_hash") != algorithm_hash
        or manifest.get("normalized_algorithm_config") != canonical
    ):
        raise ArtifactMismatch("normalized run config algorithm identity mismatch")
    return normalized


def _validate_schema1_provenance(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> None:
    provenance = _support_json(all_files, "run_provenance.json", "run provenance")
    expected = PROVENANCE_FIELDS | {"config_path", "output", "python", "input_paths"}
    libraries = provenance.get("library_versions")
    inputs = provenance.get("input_paths")
    output = provenance.get("output")
    if (
        set(provenance) != expected
        or not _hex_id(provenance.get("repository_commit"), (40, 64))
        or not _hex_id(provenance.get("repository_tree"), (40, 64))
        or not _hex_id(provenance.get("dirty_state_digest"), (64,))
        or not _string_list(provenance.get("command"), nonempty=True)
        or not isinstance(provenance.get("config_path"), str)
        or not Path(provenance["config_path"]).is_absolute()
        or not isinstance(output, str)
        or output != str(all_files["run_provenance.json"].root)
        or any(
            not isinstance(provenance.get(key), str)
            for key in ("hostname", "platform", "machine", "python", "torch_cuda_version")
        )
        or (
            provenance.get("cuda_visible_devices") is not None
            and not isinstance(provenance["cuda_visible_devices"], str)
        )
        or (
            provenance.get("cudnn_version") is not None
            and type(provenance["cudnn_version"]) is not int
        )
        or not _string_list(provenance.get("nvcc_version"))
        or not _string_list(provenance.get("gpu_inventory"))
        or not isinstance(inputs, Mapping)
        or not inputs
        or any(not _nonempty_string(key) or not _nonempty_string(value) for key, value in inputs.items())
        or not isinstance(libraries, Mapping)
        or set(libraries) != PROVENANCE_LIBRARY_FIELDS
        or any(not isinstance(value, str) for value in libraries.values())
    ):
        raise ArtifactMismatch("run provenance schema or identity is invalid")
    frozen = manifest.get("frozen_run_identity")
    if isinstance(frozen, Mapping) and (
        provenance["repository_commit"] != frozen["repository"]["commit"]
        or provenance["repository_tree"] != frozen["repository"]["tree"]
    ):
        raise ArtifactMismatch("run provenance frozen identity mismatch")


def _validate_schema1_occlusion_index(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any]
) -> None:
    _validate_fixed_file_record(
        all_files,
        manifest.get("occlusion_checkpoint_index"),
        "occlusion_checkpoint_index.json",
        "manifest occlusion checkpoint index",
    )
    index = _support_json(
        all_files, "occlusion_checkpoint_index.json", "occlusion checkpoint index"
    )
    base_fields = {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "scene",
        "algorithm_hash",
        "evaluation_checkpoint_frames_sha256",
        "run_config",
        "target_manifest",
        "snapshots",
    }
    frozen_fields = base_fields | {"frozen_run_identity", "run_execution"}
    if set(index) not in (base_fields, frozen_fields):
        raise ArtifactMismatch("occlusion checkpoint index schema is invalid")
    if (
        not _schema_version(index, 2)
        or index.get("manifest_id") != "oviv2_tesse_cd_occlusion_checkpoints_v1"
        or index.get("dataset") != manifest.get("dataset")
        or index.get("method_id") != manifest.get("method_id")
        or index.get("scene") != manifest.get("scene")
        or index.get("algorithm_hash") != manifest.get("algorithm_hash")
        or not _hex_id(index.get("evaluation_checkpoint_frames_sha256"), (64,))
    ):
        raise ArtifactMismatch("occlusion checkpoint index identity is invalid")
    if set(index) == frozen_fields and (
        index["frozen_run_identity"] != manifest.get("frozen_run_identity")
        or index["run_execution"] != manifest.get("run_execution")
    ):
        raise ArtifactMismatch("occlusion checkpoint index frozen identity mismatch")
    _validate_fixed_file_record(
        all_files,
        index["run_config"],
        "normalized_run_config.json",
        "occlusion checkpoint index run config",
    )
    _validate_byte_record(index["target_manifest"], "occlusion target manifest")
    snapshots = index.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ArtifactMismatch("occlusion checkpoint inventory is invalid")
    manifest_checkpoints = {
        checkpoint.get("frame_index"): checkpoint
        for checkpoint in manifest.get("checkpoints", [])
        if isinstance(checkpoint, Mapping)
    }
    frames: list[int] = []
    expected_snapshot_fields = {
        "scene",
        "frame_index",
        "timestamp_ns",
        "relative_timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "format",
        "path",
        "checksums_sha256",
    }
    for snapshot in snapshots:
        frame = snapshot.get("frame_index") if isinstance(snapshot, Mapping) else None
        target = manifest_checkpoints.get(frame)
        target_record = None
        if isinstance(target, Mapping):
            if snapshot.get("format") == "oviv2_compact_ownership_checkpoint":
                target_record = target.get("ownership_checkpoint")
            elif snapshot.get("format") == "oviv2_voxel_map_snapshot":
                target_record = target.get("voxel_snapshot")
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != expected_snapshot_fields
            or not _nonnegative_integer(frame)
            or frame in frames
            or snapshot.get("scene") != manifest.get("scene")
            or snapshot.get("format") not in {
                "oviv2_compact_ownership_checkpoint",
                "oviv2_voxel_map_snapshot",
            }
            or any(
                not _nonnegative_integer(snapshot.get(key))
                for key in ("timestamp_ns", "relative_timestamp_ns")
            )
            or snapshot.get("consumed_through_frame") != frame
            or snapshot.get("consumed_through_frame_exclusive") != frame + 1
            or not _hex_id(snapshot.get("checksums_sha256"), (64,))
            or target is None
            or not isinstance(target_record, Mapping)
            or snapshot.get("path") != target_record.get("path")
        ):
            raise ArtifactMismatch("occlusion checkpoint snapshot binding is invalid")
        path = _relative(snapshot["path"], "occlusion checkpoint snapshot")
        checksums = all_files.get((path / "checksums.json").as_posix())
        if checksums is None or checksums.sha256 != snapshot["checksums_sha256"]:
            raise ArtifactMismatch("occlusion checkpoint checksum binding is invalid")
        frames.append(frame)
    if frames != manifest.get("evaluation_checkpoint_frames"):
        raise ArtifactMismatch("occlusion checkpoint inventory is invalid")


def _schema1_support_inventory(
    root: _RootHandle,
    all_files: Mapping[str, FileEntry],
    manifest: Mapping[str, Any],
    *,
    production: bool,
) -> set[str]:
    present = SCHEMA1_SUPPORT_FILES & set(all_files)
    if not production:
        if present:
            raise ArtifactMismatch("schema1 support inventory is not exact")
        return set()
    if present != SCHEMA1_SUPPORT_FILES:
        raise ArtifactMismatch("schema1 support inventory is not exact")
    frozen = manifest.get("frozen_run_identity")
    execution = manifest.get("run_execution")
    if (frozen is None) != (execution is None):
        raise ArtifactMismatch("schema1 frozen run identity is incomplete")
    if frozen is not None:
        _validate_schema1_frozen_identity(frozen, "frozen run identity")
        if (
            frozen.get("dataset") != manifest.get("dataset")
            or frozen.get("method_id") != manifest.get("method_id")
            or frozen.get("scene") != manifest.get("scene")
            or frozen.get("algorithm_hash") != manifest.get("algorithm_hash")
            or frozen.get("missing_observation_policy")
            != manifest.get("missing_observation_policy")
        ):
            raise ArtifactMismatch("frozen run identity does not match manifest")
        _validate_schema1_run_execution(
            execution, root, "run execution"
        )
    source = _support_json(all_files, "source_index.json", "source index")
    source_checkpoints = _validate_schema1_source_index(
        root, all_files, manifest, source
    )
    capture = _support_json(all_files, "capture_status.json", "capture status")
    _validate_schema1_capture_status(
        all_files, manifest, source, source_checkpoints, capture
    )
    schedule_timestamps = _validate_schema1_schedule(all_files, manifest)
    _validate_schema1_trajectories(all_files, schedule_timestamps)
    _validate_schema1_normalized_config(all_files, manifest)
    _validate_schema1_provenance(all_files, manifest)
    timing = _support_json(all_files, "timing.json", "timing")
    elapsed = timing.get("elapsed_sec")
    if (
        set(timing) != {"elapsed_sec", "processed_frame_count"}
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
        or timing.get("processed_frame_count") != manifest.get("processed_frame_count")
    ):
        raise ArtifactMismatch("timing schema or binding is invalid")
    _validate_schema1_occlusion_index(all_files, manifest)
    return set(SCHEMA1_SUPPORT_FILES)


def _validate_prefixed_record(
    all_files: Mapping[str, FileEntry],
    value: object,
    prefix: PurePosixPath,
    label: str,
) -> str:
    path, digest, count = _record(value, label)
    prefixed = prefix / path
    entry = all_files.get(prefixed.as_posix())
    if entry is None or entry.sha256 != digest or entry.byte_count != count:
        raise ArtifactMismatch(f"{label} manifest record does not match raw bytes")
    return prefixed.as_posix()


def _validate_legacy_evaluation_summary(
    summary: Mapping[str, Any],
    canonical: Mapping[str, Any],
    manifest: Mapping[str, Any],
    temporal_source: Mapping[str, Any],
    schedule_events: Mapping[str, Mapping[str, Any]],
) -> None:
    shared_fields = {
        "schema_version",
        "dataset",
        "method",
        "mode",
        "scene",
        "protocol",
        "status",
        "sources",
        "frames",
        "metrics",
        "event_background_prediction_counts",
        "event_region_prediction_counts",
    }
    summary_fields = shared_fields | {"manifest_id"}
    canonical_fields = summary_fields | {"canonicalization", "source_manifest_id"}
    if (
        set(summary) != summary_fields
        or set(canonical) != canonical_fields
        or summary.get("manifest_id") != "tesse_cd_common_v2_scene_summary"
        or canonical.get("manifest_id")
        != "tesse_cd_common_v2_canonical_scene_summary_v1"
        or canonical.get("source_manifest_id") != summary.get("manifest_id")
        or any(
            not _schema_version(value, 1)
            or value.get("dataset") != manifest.get("dataset")
            or value.get("method") != manifest.get("method_id")
            or value.get("scene") != manifest.get("scene")
            or value.get("mode") != "causal_checkpoints"
            or value.get("protocol") != "tesse_cd_common_v2"
            or value.get("status") != "PASS"
            for value in (summary, canonical)
        )
        or any(summary.get(key) != canonical.get(key) for key in shared_fields - {"sources"})
    ):
        raise ArtifactMismatch("evaluation summary identity or schema is invalid")
    canonicalization = canonical.get("canonicalization")
    if (
        not isinstance(canonicalization, Mapping)
        or set(canonicalization)
        != {"schema_version", "source_identity", "physical_paths_serialized"}
        or not _schema_version(canonicalization, 1)
        or canonicalization.get("source_identity")
        != "stable_role_sha256_byte_count"
        or canonicalization.get("physical_paths_serialized") is not False
    ):
        raise ArtifactMismatch("canonical evaluation summary schema is invalid")
    raw_sources = summary.get("sources")
    canonical_sources = canonical.get("sources")
    if (
        not isinstance(raw_sources, Mapping)
        or not raw_sources
        or not isinstance(canonical_sources, Mapping)
        or set(raw_sources) != set(canonical_sources)
    ):
        raise ArtifactMismatch("evaluation summary sources are invalid")
    for role, raw in raw_sources.items():
        stable = canonical_sources[role]
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"path", "sha256", "byte_count"}
            or not _nonempty_string(raw.get("path"))
            or not _hex_id(raw.get("sha256"), (64,))
            or not _nonnegative_integer(raw.get("byte_count"))
            or not isinstance(stable, Mapping)
            or set(stable) != {"role", "sha256", "byte_count"}
            or stable.get("role") != role
            or (
                role != "temporal_index"
                and (
                    stable.get("sha256") != raw.get("sha256")
                    or stable.get("byte_count") != raw.get("byte_count")
                )
            )
            or (role == "temporal_index" and stable != temporal_source)
        ):
            raise ArtifactMismatch("canonical evaluation source binding is invalid")
    frame_fields = {
        "background_f5",
        "background_precision_match_count",
        "background_recall_match_count",
        "current_miou",
        "event_id",
        "frame_id",
        "ghost_count",
        "ghost_rate",
        "ground_truth_revealed_background_count",
        "intervention_frame_id",
        "predicted_changed_object_count",
        "predicted_revealed_background_count",
    }
    frames = summary.get("frames")
    if (
        not isinstance(frames, list)
        or not frames
        or any(not isinstance(frame, Mapping) or set(frame) != frame_fields for frame in frames)
        or not isinstance(summary.get("metrics"), Mapping)
        or not summary["metrics"]
        or not isinstance(summary.get("event_background_prediction_counts"), Mapping)
        or not isinstance(summary.get("event_region_prediction_counts"), Mapping)
    ):
        raise ArtifactMismatch("evaluation summary evidence schema is invalid")
    frame_keys: set[tuple[str, int]] = set()
    for frame in frames:
        event_id = frame.get("event_id")
        frame_id = frame.get("frame_id")
        if (
            not _nonempty_string(event_id)
            or not _nonnegative_integer(frame_id)
            or (event_id, frame_id) in frame_keys
            or not _nonnegative_integer(frame.get("intervention_frame_id"))
            or any(
                not _nonnegative_integer(frame.get(key))
                for key in frame_fields
                - {
                    "event_id", "frame_id", "intervention_frame_id",
                    "background_f5", "current_miou", "ghost_rate",
                }
            )
            or any(
                isinstance(frame.get(key), bool)
                or not isinstance(frame.get(key), (int, float))
                or not math.isfinite(frame[key])
                or not 0 <= frame[key] <= 1
                for key in ("background_f5", "current_miou", "ghost_rate")
            )
        ):
            raise ArtifactMismatch("evaluation frame evidence is invalid")
        frame_keys.add((event_id, frame_id))
    metrics = summary["metrics"]
    metric_fields = {
        "background_f5", "background_observable_event_count", "censored_event_count",
        "checkpoint_step_frames", "current_miou", "event_count", "events", "ghost_rate",
        "recovered_event_count", "recovery_background_f5", "recovery_consecutive",
        "recovery_frames", "recovery_horizon_frames",
        "unobservable_revealed_target_event_count",
    }
    metric_events = metrics.get("events") if isinstance(metrics, Mapping) else None
    event_fields = {
        "background_observable", "censor_frame", "censor_reason", "frame_ids",
        "intervention_frame_id", "overlapping_intervention", "recovered",
        "recovery_frames", "right_censored",
    }
    event_ids = {event_id for event_id, _ in frame_keys}
    if (
        set(metrics) != metric_fields
        or not isinstance(metric_events, Mapping)
        or set(metric_events) != event_ids
        or metrics.get("event_count") != len(metric_events)
        or any(
            not _nonnegative_integer(metrics.get(key))
            for key in (
                "background_observable_event_count", "censored_event_count",
                "checkpoint_step_frames", "event_count", "recovered_event_count",
                "recovery_consecutive", "recovery_horizon_frames",
                "unobservable_revealed_target_event_count",
            )
        )
        or any(
            isinstance(metrics.get(key), bool)
            or not isinstance(metrics.get(key), (int, float))
            or not math.isfinite(metrics[key])
            for key in ("background_f5", "current_miou", "ghost_rate", "recovery_background_f5", "recovery_frames")
        )
    ):
        raise ArtifactMismatch("evaluation metrics evidence is invalid")
    if set(schedule_events) != event_ids:
        raise ArtifactMismatch("evaluation event inventory does not match root schedule")
    for event_id, event in metric_events.items():
        expected_frames = sorted(frame for current, frame in frame_keys if current == event_id)
        schedule_event = schedule_events[event_id]
        if (
            not isinstance(event, Mapping)
            or set(event) != event_fields
            or type(event.get("background_observable")) is not bool
            or type(event.get("overlapping_intervention")) is not bool
            or type(event.get("right_censored")) is not bool
            or (
                event.get("recovered") is not None
                and type(event.get("recovered")) is not bool
            )
            or not _nonnegative_integer(event.get("intervention_frame_id"))
            or event.get("frame_ids") != expected_frames
            or expected_frames != schedule_event.get("common_checkpoint_frame_indices")
            or event.get("intervention_frame_id")
            != schedule_event.get("intervention_frame_index")
            or event.get("censor_frame") is not None
            and not _nonnegative_integer(event.get("censor_frame"))
            or event.get("recovery_frames") is not None
            and not _nonnegative_integer(event.get("recovery_frames"))
            or event.get("censor_reason") is not None
            and not _nonempty_string(event.get("censor_reason"))
        ):
            raise ArtifactMismatch("evaluation event evidence is invalid")
    try:
        evaluations = [DynamicFrameMetrics(**frame) for frame in frames]
        ordered_events = sorted(
            metric_events.items(), key=lambda item: item[1]["intervention_frame_id"]
        )
        overlap_censors = {
            event_id: next_event["intervention_frame_id"]
            for (event_id, event), (_, next_event) in zip(
                ordered_events, ordered_events[1:]
            )
            if event["overlapping_intervention"]
        }
        unobservable = {
            event_id for event_id, event in metric_events.items()
            if not event["background_observable"]
        }
        recomputed = aggregate_dynamic_metrics(
            evaluations,
            recovery_background_f5=metrics["recovery_background_f5"],
            recovery_consecutive=metrics["recovery_consecutive"],
            checkpoint_step_frames=metrics["checkpoint_step_frames"],
            recovery_horizon_frames=metrics["recovery_horizon_frames"],
            overlap_censor_frames=overlap_censors,
            background_unobservable_events=unobservable,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactMismatch("evaluation aggregate evidence is invalid") from exc
    if recomputed != metrics:
        raise ArtifactMismatch("evaluation aggregate evidence binding mismatch")
    for key, frame_field in (
        ("event_background_prediction_counts", "predicted_revealed_background_count"),
        ("event_region_prediction_counts", "predicted_changed_object_count"),
    ):
        counts = summary[key]
        if not isinstance(counts, Mapping) or set(counts) != event_ids:
            raise ArtifactMismatch("evaluation count evidence is invalid")
        for event_id, values in counts.items():
            expected = {
                str(frame["frame_id"]): frame[frame_field]
                for frame in frames if frame["event_id"] == event_id
            }
            if values != expected:
                raise ArtifactMismatch("evaluation count evidence binding mismatch")


def _schema1_frozen_legacy_inventory(
    all_files: Mapping[str, FileEntry], manifest: Mapping[str, Any], validation_stage: str
) -> set[str]:
    actual = {
        path
        for path in all_files
        if path.startswith("temporal/") or path.startswith("evaluation/")
    }
    if not actual:
        if "frozen_run_identity" in manifest and validation_stage == "final":
            raise ArtifactMismatch("schema1 frozen legacy inventory is missing")
        if validation_stage == "pre_legacy" and "frozen_run_identity" not in manifest:
            raise ArtifactMismatch("pre-legacy validation requires frozen identity")
        return set()
    if validation_stage == "pre_legacy":
        raise ArtifactMismatch("pre-legacy validation cannot include legacy inventory")
    if "frozen_run_identity" not in manifest:
        raise ArtifactMismatch("schema1 legacy inventory requires frozen identity")
    temporal = _support_json(
        all_files, "temporal/temporal_manifest.json", "temporal manifest"
    )
    expected_temporal_fields = {
        "schema_version",
        "dataset",
        "method",
        "mode",
        "scene",
        "sources",
        "trajectories",
        "checkpoints",
        "entity_lifecycles",
        "frozen_run_identity",
        "run_execution",
    }
    if (
        set(temporal) != expected_temporal_fields
        or not _schema_version(temporal, 1)
        or temporal.get("dataset") != manifest.get("dataset")
        or temporal.get("method") != manifest.get("method_id")
        or temporal.get("mode") != "causal_checkpoints"
        or temporal.get("scene") != manifest.get("scene")
        or temporal.get("frozen_run_identity") != manifest.get("frozen_run_identity")
        or temporal.get("run_execution") != manifest.get("run_execution")
    ):
        raise ArtifactMismatch("temporal manifest identity is invalid")
    prefix = PurePosixPath("temporal")
    expected = {"temporal/temporal_manifest.json"}
    sources = temporal.get("sources")
    source_fields = {
        "capture_status",
        "checkpoint_statuses",
        "schedule",
        "source_index",
        "trajectories",
    }
    if not isinstance(sources, Mapping) or set(sources) != source_fields:
        raise ArtifactMismatch("temporal manifest sources schema is invalid")
    for key, path in (
        ("capture_status", "sidecars/capture_status.json"),
        ("schedule", "sidecars/schedule.json"),
        ("source_index", "sidecars/source_index.json"),
        ("trajectories", "trajectories.jsonl"),
    ):
        record_path, _, _ = _record(sources[key], f"temporal source {key}")
        if record_path != PurePosixPath(path):
            raise ArtifactMismatch(f"temporal source {key} path is invalid")
        expected.add(
            _validate_prefixed_record(all_files, sources[key], prefix, f"temporal source {key}")
        )
    legacy_schedule = all_files["temporal/sidecars/schedule.json"]
    root_schedule = all_files["inputs/schedule.json"]
    if (
        legacy_schedule.sha256 != root_schedule.sha256
        or legacy_schedule.byte_count != root_schedule.byte_count
    ):
        raise ArtifactMismatch("temporal schedule content binding is invalid")
    capture = _support_json(
        all_files, "temporal/sidecars/capture_status.json", "temporal capture status"
    )
    capture_fields = {
        "schema_version", "status", "scene", "mode", "scheduled_frame_indices",
        "captured_frame_indices", "schedule", "trajectories", "checkpoint_statuses",
    }
    if (
        set(capture) != capture_fields
        or not _schema_version(capture, 1)
        or capture.get("status") != "PASS"
        or capture.get("scene") != manifest.get("scene")
        or capture.get("mode") != "causal_checkpoints"
        or capture.get("scheduled_frame_indices") != manifest.get("official_schedule_frame_indices")
        or capture.get("captured_frame_indices") != manifest.get("official_schedule_frame_indices")
        or capture.get("schedule") != {
            "path": "schedule.json", "sha256": legacy_schedule.sha256,
            "byte_count": legacy_schedule.byte_count,
        }
        or capture.get("trajectories") != {
            "path": "../trajectories.jsonl", "sha256": all_files["trajectories.jsonl"].sha256,
            "byte_count": all_files["trajectories.jsonl"].byte_count,
        }
        or capture.get("checkpoint_statuses") != [
            {
                "path": PurePosixPath(record["path"]).relative_to("sidecars").as_posix(),
                "sha256": record["sha256"], "byte_count": record["byte_count"],
            }
            for record in sources["checkpoint_statuses"]
        ]
    ):
        raise ArtifactMismatch("temporal capture status schema or binding is invalid")
    if temporal.get("trajectories") != sources["trajectories"]:
        raise ArtifactMismatch("temporal trajectory binding is invalid")
    temporal_trajectory = all_files["temporal/trajectories.jsonl"]
    root_trajectory = all_files["trajectories.jsonl"]
    if (
        temporal_trajectory.sha256 != root_trajectory.sha256
        or temporal_trajectory.byte_count != root_trajectory.byte_count
    ):
        raise ArtifactMismatch("temporal trajectory content binding is invalid")
    checkpoints = temporal.get("checkpoints")
    statuses = sources.get("checkpoint_statuses")
    checkpoint_fields = {
        "frame_index",
        "timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "snapshot",
        "entities",
    }
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not isinstance(statuses, list)
        or len(statuses) != len(checkpoints)
    ):
        raise ArtifactMismatch("temporal checkpoint inventory is invalid")
    frames: list[int] = []
    for position, checkpoint in enumerate(checkpoints):
        frame = checkpoint.get("frame_index") if isinstance(checkpoint, Mapping) else None
        if (
            not isinstance(checkpoint, Mapping)
            or set(checkpoint) != checkpoint_fields
            or not _nonnegative_integer(frame)
            or frame in frames
            or not _nonnegative_integer(checkpoint.get("timestamp_ns"))
            or checkpoint.get("consumed_through_frame") != frame
            or checkpoint.get("consumed_through_frame_exclusive") != frame + 1
        ):
            raise ArtifactMismatch("temporal checkpoint schema is invalid")
        frames.append(frame)
        for key, suffix in (("snapshot", "snapshot.npz"), ("entities", "entities.jsonl")):
            expected_path = PurePosixPath(f"checkpoints/{frame:08d}/{suffix}")
            path, _, _ = _record(checkpoint[key], f"temporal checkpoint {position} {key}")
            if path != expected_path:
                raise ArtifactMismatch("temporal checkpoint path is invalid")
            expected.add(
                _validate_prefixed_record(
                    all_files, checkpoint[key], prefix, f"temporal checkpoint {position} {key}"
                )
            )
        status_path = PurePosixPath(f"sidecars/checkpoint_statuses/{frame:08d}.json")
        path, _, _ = _record(statuses[position], f"temporal checkpoint status {position}")
        if path != status_path:
            raise ArtifactMismatch("temporal checkpoint status path is invalid")
        expected.add(
            _validate_prefixed_record(
                all_files, statuses[position], prefix, f"temporal checkpoint status {position}"
            )
        )
    if frames != manifest.get("official_schedule_frame_indices"):
        raise ArtifactMismatch("temporal checkpoint frame inventory mismatch")
    checkpoint_timestamps = {
        checkpoint["frame_index"]: checkpoint["timestamp_ns"]
        for checkpoint in checkpoints
    }
    lifecycles = temporal.get("entity_lifecycles")
    lifecycle_fields = {"entity_id", "semantic_label", "entity_type", "presence_intervals"}
    interval_fields = {
        "first_frame_index",
        "first_timestamp_ns",
        "last_frame_index",
        "last_timestamp_ns",
    }
    if not isinstance(lifecycles, list):
        raise ArtifactMismatch("temporal lifecycle inventory is invalid")
    entity_ids: list[str] = []
    for lifecycle in lifecycles:
        intervals = lifecycle.get("presence_intervals") if isinstance(lifecycle, Mapping) else None
        if (
            not isinstance(lifecycle, Mapping)
            or set(lifecycle) != lifecycle_fields
            or not _nonempty_string(lifecycle.get("entity_id"))
            or lifecycle["entity_id"] in entity_ids
            or not _nonempty_string(lifecycle.get("semantic_label"))
            or not _nonempty_string(lifecycle.get("entity_type"))
            or not isinstance(intervals, list)
            or not intervals
            or any(
                not isinstance(interval, Mapping)
                or set(interval) != interval_fields
                or any(not _nonnegative_integer(interval.get(key)) for key in interval_fields)
                or interval["first_frame_index"] > interval["last_frame_index"]
                or interval["first_timestamp_ns"] > interval["last_timestamp_ns"]
                or checkpoint_timestamps.get(interval["first_frame_index"])
                != interval["first_timestamp_ns"]
                or checkpoint_timestamps.get(interval["last_frame_index"])
                != interval["last_timestamp_ns"]
                for interval in intervals
            )
        ):
            raise ArtifactMismatch("temporal lifecycle schema is invalid")
        entity_ids.append(lifecycle["entity_id"])
    sidecar = _support_json(
        all_files, "temporal/sidecars/source_index.json", "temporal source index"
    )
    sidecar_fields = {
        "schema_version", "dataset", "method", "mode", "scene", "schedule",
        "capture_status", "trajectories", "checkpoints", "frozen_run_identity",
        "run_execution",
    }
    if (
        set(sidecar) != sidecar_fields
        or not _schema_version(sidecar, 1)
        or sidecar.get("dataset") != manifest.get("dataset")
        or sidecar.get("method") != manifest.get("method_id")
        or sidecar.get("mode") != "causal_checkpoint_exports"
        or sidecar.get("scene") != manifest.get("scene")
        or sidecar.get("frozen_run_identity") != manifest.get("frozen_run_identity")
        or sidecar.get("run_execution") != manifest.get("run_execution")
        or sidecar.get("schedule") != capture.get("schedule")
        or sidecar.get("capture_status") != {
            "path": "capture_status.json", "sha256": all_files["temporal/sidecars/capture_status.json"].sha256,
            "byte_count": all_files["temporal/sidecars/capture_status.json"].byte_count,
        }
        or sidecar.get("trajectories") != capture.get("trajectories")
        or not isinstance(sidecar.get("checkpoints"), list)
        or len(sidecar["checkpoints"]) != len(checkpoints)
    ):
        raise ArtifactMismatch("temporal source index identity mismatch")
    for position, (indexed, checkpoint) in enumerate(zip(sidecar["checkpoints"], checkpoints)):
        frame = checkpoint["frame_index"]
        expected_indexed = {
            "frame_index": frame,
            "timestamp_ns": checkpoint["timestamp_ns"],
            "consumed_through_frame": frame,
            "consumed_through_frame_exclusive": frame + 1,
            "checkpoint_status": {
                "path": f"checkpoint_statuses/{frame:08d}.json",
                "sha256": statuses[position]["sha256"],
                "byte_count": statuses[position]["byte_count"],
            },
            "snapshot": {
                "path": f"../checkpoints/{frame:08d}/snapshot.npz",
                "sha256": checkpoint["snapshot"]["sha256"],
                "byte_count": checkpoint["snapshot"]["byte_count"],
            },
            "entities": {
                "path": f"../checkpoints/{frame:08d}/entities.jsonl",
                "sha256": checkpoint["entities"]["sha256"],
                "byte_count": checkpoint["entities"]["byte_count"],
            },
        }
        if indexed != expected_indexed:
            raise ArtifactMismatch("temporal source index checkpoint binding mismatch")
    root_source = _support_json(all_files, "source_index.json", "root source index")
    root_checkpoints = {
        checkpoint.get("frame_index"): checkpoint
        for checkpoint in root_source.get("checkpoints", [])
        if isinstance(checkpoint, Mapping)
    }
    for position, (indexed, checkpoint) in enumerate(zip(sidecar["checkpoints"], checkpoints)):
        root_checkpoint = root_checkpoints.get(checkpoint["frame_index"])
        if root_checkpoint is None or any(
            indexed.get(key) != root_checkpoint.get(key)
            for key in (
                "frame_index", "timestamp_ns", "consumed_through_frame",
                "consumed_through_frame_exclusive",
            )
        ):
            raise ArtifactMismatch("temporal checkpoint root identity mismatch")
        for key in ("checkpoint_status", "snapshot", "entities"):
            if any(
                indexed[key].get(field) != root_checkpoint[key].get(field)
                for field in ("sha256", "byte_count")
            ):
                raise ArtifactMismatch("temporal checkpoint root content mismatch")
    projected_sidecar = dict(sidecar)
    projected_sidecar.pop("run_execution")
    projected_sidecar_bytes = (
        json.dumps(projected_sidecar, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    projected_temporal = dict(temporal)
    projected_temporal.pop("run_execution")
    projected_sources = dict(sources)
    projected_sources["source_index"] = {
        "path": "sidecars/source_index.json",
        "sha256": _sha256(projected_sidecar_bytes),
        "byte_count": len(projected_sidecar_bytes),
    }
    projected_temporal["sources"] = projected_sources
    projected_temporal_bytes = (
        json.dumps(projected_temporal, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporal_source = {
        "role": "temporal_index",
        "sha256": _sha256(projected_temporal_bytes),
        "byte_count": len(projected_temporal_bytes),
    }
    evaluation_files = {
        "evaluation/summary.json",
        "evaluation/summary.canonical.json",
    }
    summary = _support_json(all_files, "evaluation/summary.json", "evaluation summary")
    canonical = _support_json(
        all_files,
        "evaluation/summary.canonical.json",
        "canonical evaluation summary",
    )
    _validate_legacy_evaluation_summary(
        summary,
        canonical,
        manifest,
        temporal_source,
        {
            event["event_id"]: event
            for event in _support_json(
                all_files, "inputs/schedule.json", "root causal schedule"
            )["scenes"][manifest["scene"]]["events"]
        },
    )
    expected.update(evaluation_files)
    if actual != expected:
        raise ArtifactMismatch("schema1 frozen legacy inventory is not exact")
    return expected


def _validate_production_provenance(
    value: object, *, manifest_commit: object
) -> None:
    if not isinstance(value, Mapping) or set(value) != PROVENANCE_FIELDS:
        raise ArtifactMismatch("execution receipt provenance schema is invalid")
    libraries = value.get("library_versions")
    if (
        not _hex_id(value.get("repository_commit"), (40, 64))
        or value.get("repository_commit") != manifest_commit
        or not _hex_id(value.get("repository_tree"), (40, 64))
        or not _hex_id(value.get("dirty_state_digest"), (64,))
        or not isinstance(value.get("command"), list)
        or not value["command"]
        or any(not isinstance(item, str) for item in value["command"])
        or any(
            not isinstance(value.get(field), str)
            for field in ("hostname", "platform", "machine", "torch_cuda_version")
        )
        or (
            value.get("cuda_visible_devices") is not None
            and not isinstance(value["cuda_visible_devices"], str)
        )
        or (
            value.get("cudnn_version") is not None
            and type(value["cudnn_version"]) is not int
        )
        or any(
            not isinstance(value.get(field), list)
            or any(not isinstance(item, str) for item in value[field])
            for field in ("nvcc_version", "gpu_inventory")
        )
        or not isinstance(libraries, Mapping)
        or set(libraries) != PROVENANCE_LIBRARY_FIELDS
        or any(not isinstance(item, str) for item in libraries.values())
    ):
        raise ArtifactMismatch("execution receipt provenance identity is invalid")


def _schema2_run_identity(
    manifest: Mapping[str, Any], all_files: Mapping[str, FileEntry]
) -> None:
    algorithm_hash = manifest.get("algorithm_hash")
    if (
        not isinstance(algorithm_hash, str)
        or len(algorithm_hash) != 64
        or any(character not in "0123456789abcdef" for character in algorithm_hash)
    ):
        raise ArtifactMismatch("run algorithm identity is invalid")
    normalized_record = manifest.get("normalized_run_config")
    normalized_path, normalized_data = _validate_file_record(
        all_files, normalized_record, "normalized run config"
    )
    if normalized_path.as_posix() != "normalized_run_config.json":
        raise ArtifactMismatch("normalized run config path is invalid")
    normalized = _json_object(
        _regular_bytes(all_files, normalized_path, "normalized run config"),
        "normalized run config",
    )
    try:
        recomputed_algorithm_hash = canonical_algorithm_hash(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactMismatch("normalized run config algorithm identity is invalid") from exc
    if (
        normalized.get("algorithm_hash") != recomputed_algorithm_hash
        or algorithm_hash != recomputed_algorithm_hash
    ):
        raise ArtifactMismatch("normalized run config algorithm identity mismatch")
    if "execution_receipt.json" not in all_files:
        raise ArtifactMismatch("execution receipt is missing")
    receipt = _json_object(
        _regular_bytes(
            all_files, PurePosixPath("execution_receipt.json"), "execution receipt"
        ),
        "execution receipt",
    )
    base_fields = {"schema_version", "provenance", "environment"}
    frozen_fields = base_fields | {"frozen_run_identity", "run_execution"}
    if (
        set(receipt) not in (base_fields, frozen_fields)
        or receipt.get("schema_version") != 1
        or not isinstance(receipt.get("environment"), Mapping)
    ):
        raise ArtifactMismatch("execution receipt schema is invalid")
    manifest_commit = manifest.get("code_commit")
    if not _hex_id(manifest_commit, (40, 64)):
        raise ArtifactMismatch("run code identity is invalid")
    _validate_production_provenance(
        receipt.get("provenance"), manifest_commit=manifest_commit
    )
    if set(receipt) == frozen_fields and (
        not isinstance(receipt["frozen_run_identity"], Mapping)
        or not isinstance(receipt["run_execution"], Mapping)
        or manifest.get("frozen_run_identity") != receipt["frozen_run_identity"]
        or receipt["frozen_run_identity"].get("algorithm_hash")
        != recomputed_algorithm_hash
    ):
        raise ArtifactMismatch("execution receipt frozen identity mismatch")
    source_bindings = manifest.get("source_bindings")
    if source_bindings is not None and not isinstance(source_bindings, Mapping):
        raise ArtifactMismatch("run source binding is invalid")


def _cumulative_entries(
    all_files: Mapping[str, FileEntry],
    audit: object,
    *,
    logical_prefix: str,
    label: str,
) -> tuple[dict[str, FileEntry], set[str]]:
    if (
        not isinstance(audit, Mapping)
        or set(audit) not in (
            {"format", "artifact", "snapshot", "entities"},
            {"format", "artifact", "snapshot", "entities", "voxel_snapshot"},
        )
        or audit.get("format") != "oviv2_cumulative_audit_v1"
    ):
        raise ArtifactMismatch("cumulative audit manifest is invalid")
    entries = _validate_tree_record(
        all_files, audit["artifact"], f"{label} artifact"
    )
    entries.extend(
        (
            _validate_file_record(
                all_files, audit["snapshot"], f"{label} snapshot"
            ),
            _validate_file_record(
                all_files, audit["entities"], f"{label} entities"
            ),
        )
    )
    if "voxel_snapshot" in audit:
        entries.extend(
            _validate_tree_record(
                all_files, audit["voxel_snapshot"], f"{label} voxel snapshot"
            )
        )
    projection: dict[str, FileEntry] = {}
    physical: set[str] = set()
    for path, entry in entries:
        parts = path.parts
        if parts.count("cumulative_audit") != 1:
            raise ArtifactMismatch("cumulative audit path is outside its audit root")
        offset = parts.index("cumulative_audit") + 1
        if offset == len(parts):
            raise ArtifactMismatch("cumulative audit file path is invalid")
        local = PurePosixPath(*parts[offset:]).as_posix()
        logical = f"{logical_prefix}/{local}"
        previous = projection.setdefault(logical, entry)
        if (
            previous.sha256,
            previous.byte_count,
        ) != (entry.sha256, entry.byte_count):
            raise ArtifactMismatch("cumulative inventory aliases unequal raw bytes")
        physical.add(path.as_posix())
    return projection, physical


def _schema2_projection(
    manifest: Mapping[str, Any],
    all_files: Mapping[str, FileEntry],
) -> tuple[list[int], dict[str, FileEntry]]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    frames: list[int] = []
    projection: dict[str, FileEntry] = {}
    projected_files: set[str] = set()
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ArtifactMismatch("checkpoint inventory record is invalid")
        frame = checkpoint.get("frame_index")
        if type(frame) is not int or frame < 0 or frame in frames:
            raise ArtifactMismatch("checkpoint inventory is invalid")
        frames.append(frame)
        entries, physical = _cumulative_entries(
            all_files,
            checkpoint.get("cumulative_audit"),
            logical_prefix=f"checkpoint/{position:08d}/{frame:08d}",
            label=f"checkpoint {frame} cumulative audit",
        )
        for path, data in entries.items():
            if path in projection:
                raise ArtifactMismatch("cumulative projection contains duplicate paths")
            projection[path] = data
        projected_files.update(physical)
    final_audit = manifest.get("final_cumulative_audit")
    if final_audit is not None:
        entries, physical = _cumulative_entries(
            all_files,
            final_audit,
            logical_prefix="final",
            label="final cumulative audit",
        )
        for path, data in entries.items():
            if path in projection:
                raise ArtifactMismatch("cumulative projection contains duplicate paths")
            projection[path] = data
        projected_files.update(physical)
    actual_cumulative = {
        path
        for path in all_files
        if "cumulative_audit" in PurePosixPath(path).parts
    }
    if actual_cumulative != projected_files:
        raise ArtifactMismatch("cumulative artifact inventory is not exact")
    if frames != sorted(frames):
        raise ArtifactMismatch("checkpoint inventory is not ordered")
    return frames, projection


def _schema1_projection(
    manifest: Mapping[str, Any], all_files: Mapping[str, FileEntry]
) -> tuple[list[int], dict[str, FileEntry]]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    frames: list[int] = []
    projection: dict[str, FileEntry] = {}
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ArtifactMismatch("checkpoint inventory record is invalid")
        frame = checkpoint.get("frame_index")
        if type(frame) is not int or frame < 0 or frame in frames:
            raise ArtifactMismatch("checkpoint inventory is invalid")
        frames.append(frame)
        roles = [
            role
            for role in ("artifact", "voxel_snapshot", "ownership_checkpoint")
            if role in checkpoint
        ]
        if not roles or ("voxel_snapshot" in roles and "artifact" not in roles):
            raise ArtifactMismatch("v1 cumulative checkpoint manifest is incomplete")
        for role in roles:
            logical_role = "voxel_snapshot" if role == "ownership_checkpoint" else role
            record_path, _, _ = _record(checkpoint[role], f"v1 {role}")
            for path, data in _validate_tree_record(
                all_files, checkpoint[role], f"v1 {role}"
            ):
                local = path.relative_to(record_path).as_posix()
                logical = (
                    f"checkpoint/{position:08d}/{frame:08d}/{logical_role}/{local}"
                )
                if logical in projection:
                    raise ArtifactMismatch("cumulative projection contains duplicate paths")
                projection[logical] = data
    if frames != sorted(frames):
        raise ArtifactMismatch("checkpoint inventory is not ordered")
    return frames, projection


def _projection_summary(
    frames: list[int], projection: Mapping[str, FileEntry]
) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for path in sorted(projection):
        entry = projection[path]
        records.append(
            {"path": path, "sha256": entry.sha256, "byte_count": entry.byte_count}
        )
        root_digest.update(path.encode("utf-8"))
        root_digest.update(b"\0")
        root_digest.update(bytes.fromhex(entry.sha256))
        root_digest.update(b"\n")
    return records, root_digest.hexdigest()


def _valid_receipt_inventory(value: object) -> bool:
    if not isinstance(value, list):
        return False
    try:
        for position, record in enumerate(value):
            _record(record, f"T1 exact receipt inventory[{position}]")
    except ArtifactMismatch:
        return False
    return True


def _validate_t1_receipt(
    root: _RootHandle,
    all_files: Mapping[str, FileEntry],
    frames: list[int],
    projection: Mapping[str, FileEntry],
) -> None:
    receipt = _json_object(
        _regular_bytes(
            all_files, PurePosixPath("t1_exact_receipt.json"), "T1 exact receipt"
        ),
        "T1 exact receipt",
    )
    expected_fields = {
        "schema_version",
        "format",
        "execution",
        "source_manifest",
        "artifact_inventory",
        "checkpoint_frames",
        "cumulative_root_sha256",
    }
    execution = receipt.get("execution")
    source = receipt.get("source_manifest")
    frozen_execution_fields = {
        "profile",
        "argv",
        "pid",
        "code_commit",
        "source_manifest_sha256",
        "input_fingerprints",
        "output_root",
    }
    development_execution_fields = frozen_execution_fields | {"mode"}
    source_fields = {"path", "sha256", "byte_count"}
    records, root_digest = _projection_summary(frames, projection)
    fingerprints = (
        execution.get("input_fingerprints")
        if isinstance(execution, Mapping)
        else None
    )
    receipt_inventory = receipt.get("artifact_inventory")
    receipt_frames = receipt.get("checkpoint_frames")
    if (
        set(receipt) != expected_fields
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("format") != "oviv2_t1_exact_execution_receipt_v1"
        or not isinstance(execution, Mapping)
        or frozenset(execution) not in {
            frozenset(frozen_execution_fields),
            frozenset(development_execution_fields),
        }
        or (
            "mode" in execution
            and execution.get("mode") != "apartment_development_unfrozen"
        )
        or execution.get("profile") != "reference"
        or not isinstance(execution.get("argv"), list)
        or not execution["argv"]
        or any(not isinstance(item, str) or not item for item in execution["argv"])
        or type(execution.get("pid")) is not int
        or execution["pid"] <= 0
        or not _hex_id(execution.get("code_commit"), (40,))
        or not _hex_id(execution.get("source_manifest_sha256"), (64,))
        or not isinstance(fingerprints, Mapping)
        or not fingerprints
        or any(
            not isinstance(key, str)
            or not key
            or not _hex_id(value, (64,))
            for key, value in fingerprints.items()
        )
        or execution.get("output_root") != str(root.path)
        or not isinstance(source, Mapping)
        or set(source) != source_fields
        or not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
        or source.get("sha256") != execution.get("source_manifest_sha256")
        or type(source.get("byte_count")) is not int
        or source["byte_count"] < 0
        or not _valid_receipt_inventory(receipt_inventory)
        or receipt_inventory != records
        or not isinstance(receipt_frames, list)
        or any(type(frame) is not int or frame < 0 for frame in receipt_frames)
        or receipt_frames != frames
        or receipt.get("cumulative_root_sha256") != root_digest
    ):
        raise ArtifactMismatch("T1 exact receipt binding is invalid")


def _load_inventory(
    root: _RootHandle, validation_stage: str, schema1_variant: str | None,
) -> tuple[list[int], dict[str, FileEntry], dict[str, FileEntry]]:
    all_files = _all_regular_files(root)
    manifest_data = _regular_bytes(
        all_files, PurePosixPath("run_manifest.json"), "run manifest"
    )
    manifest = _json_object(manifest_data, "run manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in (1, 2):
        raise ArtifactMismatch("run manifest identity is invalid")
    checkpoints = manifest.get("checkpoints")
    declared_inventory = manifest.get("artifact_inventory")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    if manifest["schema_version"] == 2:
        if validation_stage != "final":
            raise ArtifactMismatch("pre-legacy validation requires schema1 production")
        if not isinstance(declared_inventory, list) or any(
            not isinstance(item, str) for item in declared_inventory
        ):
            raise ArtifactMismatch("artifact inventory is invalid")
        if declared_inventory != sorted(set(declared_inventory)):
            raise ArtifactMismatch("artifact inventory is noncanonical")
    if manifest["schema_version"] == 2:
        _validate_schema2_structure(manifest)
        if "t1_exact_receipt.json" in all_files:
            raise ArtifactMismatch("artifact inventory is not exact")
        if set(declared_inventory) != _schema2_manifest_inventory(
            all_files, manifest
        ):
            raise ArtifactMismatch("artifact inventory is not exact")
        allowed_root_files = {"run_manifest.json", "execution_receipt.json"}
        expected_files = set(declared_inventory) | allowed_root_files
        if set(all_files) != expected_files:
            raise ArtifactMismatch("artifact inventory is not exact")
        _validate_manifest_records(all_files, manifest)
        _schema2_run_identity(manifest, all_files)
        frames, projection = _schema2_projection(manifest, all_files)
        return frames, projection, all_files
    if schema1_variant is None:
        raise ArtifactMismatch("schema1 variant must be supplied by the caller")
    production = _validate_schema1_structure(manifest, schema1_variant)
    if validation_stage == "pre_legacy" and not production:
        raise ArtifactMismatch("pre-legacy validation requires schema1 production")
    _validate_manifest_records(all_files, manifest)
    allowed_root_files = {"run_manifest.json"}
    if "t1_exact_receipt.json" in all_files:
        allowed_root_files.add("t1_exact_receipt.json")
    support_files = _schema1_support_inventory(
        root, all_files, manifest, production=production
    )
    legacy_files = _schema1_frozen_legacy_inventory(
        all_files, manifest, validation_stage
    )
    expected_files = (
        _schema1_manifest_inventory(all_files, manifest)
        | support_files
        | legacy_files
        | allowed_root_files
    )
    if set(all_files) != expected_files:
        raise ArtifactMismatch("artifact inventory is not exact")
    frames, projection = _schema1_projection(manifest, all_files)
    if "t1_exact_receipt.json" in all_files:
        _validate_t1_receipt(root, all_files, frames, projection)
    return frames, projection, all_files


def _verify_entry(root: _RootHandle, entry: FileEntry, label: str) -> None:
    descriptor = _open_regular(root, entry.path, label)
    try:
        if _identity(os.fstat(descriptor)) != entry.identity:
            raise ArtifactMismatch(f"artifact was replaced before comparison: {label}")
    finally:
        os.close(descriptor)


def compare_cumulative_artifacts(
    left: str | Path,
    right: str | Path,
    *,
    validation_stage: str = "final",
    left_schema1_variant: str | None = None,
    right_schema1_variant: str | None = None,
) -> dict[str, Any]:
    """Compare with O(chunk + directory depth + output inventory) memory."""
    if validation_stage not in {"final", "pre_legacy"}:
        raise ValueError("validation_stage must be 'final' or 'pre_legacy'")
    if (
        left_schema1_variant not in SCHEMA1_VARIANTS | {None}
        or right_schema1_variant not in SCHEMA1_VARIANTS | {None}
    ):
        raise ValueError("schema1 variant is invalid")
    left_path = Path(os.path.abspath(os.fspath(left)))
    right_path = Path(os.path.abspath(os.fspath(right)))
    handles: dict[str, _RootHandle] = {}
    watches: dict[str, int] = {}
    loaded: dict[
        tuple[str, str | None],
        tuple[list[int], dict[str, FileEntry], dict[str, FileEntry]],
    ] = {}

    def load(value: str | Path, variant: str | None) -> tuple[
        str, _RootHandle, tuple[list[int], dict[str, FileEntry], dict[str, FileEntry]]
    ]:
        key = os.path.abspath(os.fspath(value))
        if key not in handles:
            handles[key] = _open_root(Path(value))
            watches[key] = _begin_mutation_watch(handles[key])
        cache_key = (key, variant)
        if cache_key not in loaded:
            loaded[cache_key] = _load_inventory(handles[key], validation_stage, variant)
        return key, handles[key], loaded[cache_key]

    try:
        if left_path == right_path and left_schema1_variant != right_schema1_variant:
            raise ValueError("the same root must use the same schema1 variant")
        left_key, left_root, (left_frames, left_inventory, left_all) = load(
            left_path, left_schema1_variant
        )
        right_key, right_root, (right_frames, right_inventory, right_all) = load(
            right_path, right_schema1_variant
        )
        if left_frames != right_frames:
            raise ArtifactMismatch("checkpoint inventory differs")
        if set(left_inventory) != set(right_inventory):
            raise ArtifactMismatch("cumulative inventory differs")
        records: list[dict[str, Any]] = []
        root_digest = hashlib.sha256()
        for path in sorted(left_inventory):
            left_entry = left_inventory[path]
            right_entry = right_inventory[path]
            _verify_entry(left_root, left_entry, path)
            if right_key != left_key:
                _verify_entry(right_root, right_entry, path)
            if (
                left_entry.sha256 != right_entry.sha256
                or left_entry.byte_count != right_entry.byte_count
            ):
                raise ArtifactMismatch(f"raw bytes differ: {path}")
            record = {
                "path": path,
                "sha256": left_entry.sha256,
                "byte_count": left_entry.byte_count,
            }
            records.append(record)
            root_digest.update(path.encode("utf-8"))
            root_digest.update(b"\0")
            root_digest.update(bytes.fromhex(left_entry.sha256))
            root_digest.update(b"\n")
        _verify_full_inventory(left_root, left_all)
        if right_key != left_key:
            _verify_full_inventory(right_root, right_all)
        for handle in handles.values():
            handle.verify()
        for descriptor in watches.values():
            _verify_mutation_watch(descriptor)
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": left_frames,
            "inventory": records,
            "root_sha256": root_digest.hexdigest(),
        }
    finally:
        try:
            for descriptor in watches.values():
                os.close(descriptor)
        finally:
            for handle in handles.values():
                handle.close()


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("comparison output write made no progress")
        view = view[written:]


def _publish_output(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name:
        raise ValueError("comparison output path is invalid")
    parent = _open_root(absolute.parent)
    descriptor: int | None = None
    owned_identity: tuple[int, int, int] | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            absolute.name, flags, 0o644, dir_fd=parent.descriptor
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError("comparison output temporary is not regular")
        owned_identity = _directory_identity(opened)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        linked = os.stat(
            absolute.name, dir_fd=parent.descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(linked.st_mode)
            or _directory_identity(linked) != owned_identity
        ):
            raise RuntimeError(
                "comparison output publication state is uncertain: owner changed"
            )
        os.fsync(parent.descriptor)
        parent.verify()
        linked = os.stat(
            absolute.name, dir_fd=parent.descriptor, follow_symlinks=False
        )
        if _directory_identity(linked) != owned_identity:
            raise RuntimeError(
                "comparison output publication state is uncertain: owner changed"
            )
    except FileExistsError:
        raise
    except Exception as exc:
        if descriptor is None:
            raise
        raise RuntimeError(
            "comparison output publication state is uncertain; output was preserved"
        ) from exc
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            parent.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--left-schema1-variant", choices=sorted(SCHEMA1_VARIANTS))
    parser.add_argument("--right-schema1-variant", choices=sorted(SCHEMA1_VARIANTS))
    args = parser.parse_args(argv)
    result = compare_cumulative_artifacts(
        args.left,
        args.right,
        left_schema1_variant=args.left_schema1_variant,
        right_schema1_variant=args.right_schema1_variant,
    )
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        _publish_output(args.output, payload.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
