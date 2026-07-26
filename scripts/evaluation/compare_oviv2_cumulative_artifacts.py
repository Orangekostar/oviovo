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
        allowed_root_files = {"run_manifest.json"}
        if "execution_receipt.json" in all_files:
            allowed_root_files.add("execution_receipt.json")
        if "t1_exact_receipt.json" in all_files:
            allowed_root_files.add("t1_exact_receipt.json")
        expected_files = set(declared_inventory) | allowed_root_files
        if set(all_files) != expected_files:
            raise ArtifactMismatch("artifact inventory is not exact")
        _validate_manifest_records(all_files, manifest)
        _schema2_run_identity(manifest, all_files)
        return _schema2_projection(manifest, all_files)
    _validate_manifest_records(all_files, manifest)
    return _schema1_projection(manifest, all_files)


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
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
