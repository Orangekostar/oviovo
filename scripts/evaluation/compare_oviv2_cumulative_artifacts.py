#!/usr/bin/env python3
"""Compare cumulative audit artifacts byte-for-byte across OVIV2 runs."""

from __future__ import annotations

import argparse
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
    root: Path, manifest: Mapping[str, Any], all_files: Mapping[str, bytes]
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
        root, normalized_record, "normalized run config"
    )
    if normalized_path.as_posix() != "normalized_run_config.json":
        raise ArtifactMismatch("normalized run config path is invalid")
    normalized = _json_object(normalized_data, "normalized run config")
    try:
        recomputed_algorithm_hash = canonical_algorithm_hash(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactMismatch("normalized run config algorithm identity is invalid") from exc
    if (
        normalized.get("algorithm_hash") != recomputed_algorithm_hash
        or algorithm_hash != recomputed_algorithm_hash
    ):
        raise ArtifactMismatch("normalized run config algorithm identity mismatch")
    receipt = _json_object(
        all_files.get("execution_receipt.json", b""), "execution receipt"
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
    root: Path,
    audit: object,
    *,
    logical_prefix: str,
    label: str,
) -> tuple[dict[str, bytes], set[str]]:
    if (
        not isinstance(audit, Mapping)
        or set(audit) not in (
            {"format", "artifact", "snapshot", "entities"},
            {"format", "artifact", "snapshot", "entities", "voxel_snapshot"},
        )
        or audit.get("format") != "oviv2_cumulative_audit_v1"
    ):
        raise ArtifactMismatch("cumulative audit manifest is invalid")
    entries = _validate_tree_record(root, audit["artifact"], f"{label} artifact")
    entries.extend(
        (
            _validate_file_record(root, audit["snapshot"], f"{label} snapshot"),
            _validate_file_record(root, audit["entities"], f"{label} entities"),
        )
    )
    if "voxel_snapshot" in audit:
        entries.extend(
            _validate_tree_record(
                root, audit["voxel_snapshot"], f"{label} voxel snapshot"
            )
        )
    projection: dict[str, bytes] = {}
    physical: set[str] = set()
    for path, data in entries:
        parts = path.parts
        if parts.count("cumulative_audit") != 1:
            raise ArtifactMismatch("cumulative audit path is outside its audit root")
        offset = parts.index("cumulative_audit") + 1
        if offset == len(parts):
            raise ArtifactMismatch("cumulative audit file path is invalid")
        local = PurePosixPath(*parts[offset:]).as_posix()
        logical = f"{logical_prefix}/{local}"
        previous = projection.setdefault(logical, data)
        if previous != data:
            raise ArtifactMismatch("cumulative inventory aliases unequal raw bytes")
        physical.add(path.as_posix())
    return projection, physical


def _schema2_projection(
    root: Path,
    manifest: Mapping[str, Any],
    all_files: Mapping[str, bytes],
) -> tuple[list[int], dict[str, bytes]]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    frames: list[int] = []
    projection: dict[str, bytes] = {}
    projected_files: set[str] = set()
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ArtifactMismatch("checkpoint inventory record is invalid")
        frame = checkpoint.get("frame_index")
        if type(frame) is not int or frame < 0 or frame in frames:
            raise ArtifactMismatch("checkpoint inventory is invalid")
        frames.append(frame)
        entries, physical = _cumulative_entries(
            root,
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
            root, final_audit, logical_prefix="final", label="final cumulative audit"
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
    root: Path, manifest: Mapping[str, Any]
) -> tuple[list[int], dict[str, bytes]]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ArtifactMismatch("checkpoint inventory is empty")
    frames: list[int] = []
    projection: dict[str, bytes] = {}
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
            for path, data in _validate_tree_record(root, checkpoint[role], f"v1 {role}"):
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
        _validate_manifest_records(root, manifest)
        _schema2_run_identity(root, manifest, all_files)
        return _schema2_projection(root, manifest, all_files)
    _validate_manifest_records(root, manifest)
    return _schema1_projection(root, manifest)


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
