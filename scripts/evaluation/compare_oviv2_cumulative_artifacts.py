#!/usr/bin/env python3
"""Compare cumulative audit artifacts byte-for-byte across OVIV2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence


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


def _relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactMismatch(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactMismatch(f"{label} path is noncanonical")
    if path.as_posix() != value:
        raise ArtifactMismatch(f"{label} path is noncanonical")
    return path


def _regular_bytes(root: Path, relative: PurePosixPath, label: str) -> bytes:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ArtifactMismatch(f"{label} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactMismatch(f"{label} contains a symlink")
    if not stat.S_ISREG(os.lstat(current).st_mode):
        raise ArtifactMismatch(f"{label} is not a regular file")
    return current.read_bytes()


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


def _files_below(root: Path, relative: PurePosixPath, label: str) -> list[PurePosixPath]:
    directory = root.joinpath(*relative.parts)
    try:
        metadata = os.lstat(directory)
    except OSError as exc:
        raise ArtifactMismatch(f"{label} tree is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactMismatch(f"{label} tree is not a regular directory")
    files: list[PurePosixPath] = []
    for path in sorted(directory.rglob("*")):
        entry = os.lstat(path)
        if stat.S_ISLNK(entry.st_mode):
            raise ArtifactMismatch(f"{label} tree contains a symlink")
        if stat.S_ISREG(entry.st_mode):
            files.append(PurePosixPath(path.relative_to(root).as_posix()))
        elif not stat.S_ISDIR(entry.st_mode):
            raise ArtifactMismatch(f"{label} tree contains a forbidden entry")
    if not files:
        raise ArtifactMismatch(f"{label} tree is empty")
    return files


def _validate_file_record(
    root: Path, record: object, label: str
) -> tuple[PurePosixPath, bytes]:
    path, digest, count = _record(record, label)
    data = _regular_bytes(root, path, label)
    if len(data) != count or _sha256(data) != digest:
        raise ArtifactMismatch(f"{label} manifest record does not match raw bytes")
    return path, data


def _validate_tree_record(
    root: Path, record: object, label: str
) -> list[tuple[PurePosixPath, bytes]]:
    path, digest, count = _record(record, label)
    files = _files_below(root, path, label)
    tree_digest = hashlib.sha256()
    total = 0
    result: list[tuple[PurePosixPath, bytes]] = []
    for item in files:
        data = _regular_bytes(root, item, label)
        local = item.relative_to(path).as_posix()
        total += len(data)
        tree_digest.update(local.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(_sha256(data)))
        tree_digest.update(b"\n")
        result.append((item, data))
    if total != count or tree_digest.hexdigest() != digest:
        raise ArtifactMismatch(f"{label} manifest record does not match raw bytes")
    return result


def _all_regular_files(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        metadata = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactMismatch("artifact inventory contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            result[relative] = path.read_bytes()
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactMismatch("artifact inventory contains a forbidden entry")
    return result


def _validate_manifest_records(root: Path, value: object, label: str = "manifest") -> None:
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "byte_count"}:
            path, _, _ = _record(value, label)
            target = root.joinpath(*path.parts)
            try:
                metadata = os.lstat(target)
            except OSError as exc:
                raise ArtifactMismatch(f"{label} manifest record is missing") from exc
            if stat.S_ISDIR(metadata.st_mode):
                _validate_tree_record(root, value, label)
            else:
                _validate_file_record(root, value, label)
            return
        for key, child in value.items():
            _validate_manifest_records(root, child, f"{label}.{key}")
    elif isinstance(value, list):
        for position, child in enumerate(value):
            _validate_manifest_records(root, child, f"{label}[{position}]")


def _load_inventory(root: Path) -> tuple[list[int], dict[str, bytes]]:
    absolute = root.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise ArtifactMismatch("run root contains a symlink path component")
        except FileNotFoundError:
            break
    if root.is_symlink() or not root.is_dir():
        raise ArtifactMismatch("run root is not a regular directory")
    manifest_data = _regular_bytes(root, PurePosixPath("run_manifest.json"), "run manifest")
    try:
        manifest = json.loads(
            manifest_data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactMismatch(f"non-finite manifest value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactMismatch("run manifest is invalid JSON") from exc
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
    all_files = _all_regular_files(root)
    if manifest["schema_version"] == 2:
        allowed_root_files = {"run_manifest.json"}
        if "execution_receipt.json" in all_files:
            allowed_root_files.add("execution_receipt.json")
        if "t1_exact_receipt.json" in all_files:
            allowed_root_files.add("t1_exact_receipt.json")
        expected_files = set(declared_inventory) | allowed_root_files
        if set(all_files) != expected_files:
            raise ArtifactMismatch("artifact inventory is not exact")
        selected_paths = sorted(expected_files - {"t1_exact_receipt.json"})
    else:
        selected_paths = sorted(
            path
            for path in all_files
            if path not in {"run_provenance.json", "timing.json", "t1_exact_receipt.json"}
        )
    _validate_manifest_records(root, manifest)

    frames: list[int] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ArtifactMismatch("checkpoint inventory record is invalid")
        frame = checkpoint.get("frame_index")
        audit = checkpoint.get("cumulative_audit")
        if type(frame) is not int or frame < 0 or frame in frames:
            raise ArtifactMismatch("checkpoint inventory is invalid")
        frames.append(frame)
        if manifest["schema_version"] == 2:
            if (
                not isinstance(audit, Mapping)
                or set(audit) not in (
                    {"format", "artifact", "snapshot", "entities"},
                    {"format", "artifact", "snapshot", "entities", "voxel_snapshot"},
                )
                or audit.get("format") != "oviv2_cumulative_audit_v1"
            ):
                raise ArtifactMismatch("cumulative audit manifest is invalid")
        else:
            roles = [name for name in ("artifact", "voxel_snapshot", "ownership_checkpoint") if name in checkpoint]
            if not roles or ("voxel_snapshot" in roles and "artifact" not in roles):
                raise ArtifactMismatch("v1 cumulative checkpoint manifest is incomplete")
    if frames != sorted(frames):
        raise ArtifactMismatch("checkpoint inventory is not ordered")
    return frames, {path: all_files[path] for path in selected_paths}


def compare_cumulative_artifacts(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Validate and compare every cumulative audit file with no normalization."""
    left_frames, left_inventory = _load_inventory(Path(left))
    right_frames, right_inventory = _load_inventory(Path(right))
    if left_frames != right_frames:
        raise ArtifactMismatch("checkpoint inventory differs")
    if set(left_inventory) != set(right_inventory):
        raise ArtifactMismatch("cumulative inventory differs")
    records: list[dict[str, Any]] = []
    root_digest = hashlib.sha256()
    for path in sorted(left_inventory):
        left_data = left_inventory[path]
        right_data = right_inventory[path]
        if left_data != right_data:
            raise ArtifactMismatch(f"raw bytes differ: {path}")
        digest = _sha256(left_data)
        record = {"path": path, "sha256": digest, "byte_count": len(left_data)}
        records.append(record)
        root_digest.update(path.encode("utf-8"))
        root_digest.update(b"\0")
        root_digest.update(bytes.fromhex(digest))
        root_digest.update(b"\n")
    return {
        "format": "oviv2_cumulative_exact_v1",
        "checkpoint_frames": left_frames,
        "inventory": records,
        "root_sha256": root_digest.hexdigest(),
    }


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
