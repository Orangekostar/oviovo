#!/usr/bin/env python3
"""Compare cumulative audit artifacts byte-for-byte across OVIV2 runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    canonical_algorithm_hash,
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
PROVENANCE_LIBRARY_FIELDS = {"numpy", "open3d", "torch", "scipy", "pillow"}
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
        if "frozen_run_identity" in source_index:
            expected_index_fields.add("frozen_run_identity")
        if set(source_index) != expected_index_fields:
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
    execution_fields = {
        "profile",
        "argv",
        "pid",
        "code_commit",
        "source_manifest_sha256",
        "input_fingerprints",
        "output_root",
    }
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
        or set(execution) != execution_fields
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
    root: _RootHandle,
) -> tuple[list[int], dict[str, FileEntry]]:
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
        return _schema2_projection(manifest, all_files)
    _validate_manifest_records(all_files, manifest)
    allowed_root_files = {"run_manifest.json"}
    if "t1_exact_receipt.json" in all_files:
        allowed_root_files.add("t1_exact_receipt.json")
    expected_files = _schema1_manifest_inventory(
        all_files, manifest
    ) | allowed_root_files
    if set(all_files) != expected_files:
        raise ArtifactMismatch("artifact inventory is not exact")
    frames, projection = _schema1_projection(manifest, all_files)
    if "t1_exact_receipt.json" in all_files:
        _validate_t1_receipt(root, all_files, frames, projection)
    return frames, projection


def _verify_entry(root: _RootHandle, entry: FileEntry, label: str) -> None:
    descriptor = _open_regular(root, entry.path, label)
    try:
        if _identity(os.fstat(descriptor)) != entry.identity:
            raise ArtifactMismatch(f"artifact was replaced before comparison: {label}")
    finally:
        os.close(descriptor)


def compare_cumulative_artifacts(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare with O(chunk + directory depth + output inventory) memory."""
    left_path = Path(os.path.abspath(os.fspath(left)))
    right_path = Path(os.path.abspath(os.fspath(right)))
    handles: dict[str, _RootHandle] = {}
    loaded: dict[str, tuple[list[int], dict[str, FileEntry]]] = {}

    def load(value: str | Path) -> tuple[
        str, _RootHandle, tuple[list[int], dict[str, FileEntry]]
    ]:
        key = os.path.abspath(os.fspath(value))
        if key not in handles:
            handles[key] = _open_root(Path(value))
            loaded[key] = _load_inventory(handles[key])
        return key, handles[key], loaded[key]

    try:
        left_key, left_root, (left_frames, left_inventory) = load(left_path)
        right_key, right_root, (right_frames, right_inventory) = load(right_path)
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
        for handle in handles.values():
            handle.verify()
        return {
            "format": "oviv2_cumulative_exact_v1",
            "checkpoint_frames": left_frames,
            "inventory": records,
            "root_sha256": root_digest.hexdigest(),
        }
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
    args = parser.parse_args(argv)
    result = compare_cumulative_artifacts(args.left, args.right)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        _publish_output(args.output, payload.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
