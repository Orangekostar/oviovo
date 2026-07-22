#!/usr/bin/env python3
"""Evaluate OVIV2 ownership retention on frozen TESSE-CD occlusion targets."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

import numpy as np

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from scripts.evaluation.derive_tesse_cd_occlusion_v1 import (
    REPO_ROOT,
    validate_generated_target,
)
from src.oviv2.snapshot import VoxelMapSnapshot
from src.oviv2.compact_checkpoint import (
    COMPACT_OWNERSHIP_FORMAT,
    CompactOwnershipCheckpoint,
    CompactOwnershipMetadata,
)


SnapshotKey = tuple[str, int]
SourceFrameTime = tuple[int, int]
FULL_SNAPSHOT_FORMAT = "oviv2_voxel_map_snapshot"
RUNNER_SCENE_CONFIG_FIELDS = frozenset(
    {
        "algorithm_hash",
        "scene",
        "frame_count",
        "dataset_root",
        "export_manifest",
        "frontend_cache_dir",
        "frontend_manifest",
        "dense_cache_dir",
        "dense_manifest",
        "evaluation_checkpoint_frames",
        "occlusion_target_manifest",
        "vocabulary_json",
        "vocabulary_txt",
    }
)
_REPOSITORY_SOURCE_ROLES = frozenset(
    {"source_manifest", "schedule", "rgbd_lock"}
)
_SCENE_SOURCE_SUFFIXES = (
    "changes",
    "dsg_with_mesh",
    "export_manifest",
    "timestamps",
    "trajectory",
)


def canonical_algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in sorted(config)
        if key not in RUNNER_SCENE_CONFIG_FIELDS
    }


def canonical_algorithm_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_algorithm_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _LazyCheckpointSnapshots(Mapping[SnapshotKey, Any]):
    def __init__(
        self,
        bindings: Mapping[SnapshotKey, "_SnapshotBinding"],
        *,
        cache_size: int = 1,
    ) -> None:
        self._bindings = dict(bindings)
        self._cache_size = cache_size
        self._cache: OrderedDict[SnapshotKey, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self._bindings)

    def __iter__(self) -> Iterator[SnapshotKey]:
        return iter(self._bindings)

    def __getitem__(self, key: SnapshotKey) -> Any:
        if key not in self._bindings:
            raise KeyError(key)
        cached = self._cache.pop(key, None)
        if cached is None:
            binding = self._bindings[key]
            _revalidate_snapshot_binding(binding)
            cached = (
                CompactOwnershipCheckpoint.load(binding.path)
                if binding.format == COMPACT_OWNERSHIP_FORMAT
                else VoxelMapSnapshot.load(binding.path)
            )
            if cached.checksums != binding.checksums:
                raise ValueError(f"snapshot checksums changed after index check: {key}")
            loaded_metadata = asdict(cached.metadata)
            if (
                binding.format == FULL_SNAPSHOT_FORMAT
                and cached.metadata.schema_version in {1, 2}
            ):
                loaded_metadata.pop("dense_semantic_provenance")
            if loaded_metadata != binding.metadata:
                raise ValueError(f"snapshot metadata changed after index check: {key}")
            if not (
                cached.metadata.scene_id == key[0]
                and cached.metadata.frame_id == key[1]
                and cached.metadata.timestamp == binding.timestamp_ns / 1_000_000_000
            ):
                raise ValueError(f"snapshot timestamp or identity mismatch: {key}")
            _revalidate_snapshot_binding(binding)
        self._cache[key] = cached
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return cached

    def revalidate_all(self) -> None:
        for binding in self._bindings.values():
            _revalidate_snapshot_binding(binding)


@dataclass(frozen=True)
class _FileWitness:
    path: Path
    fingerprint: tuple[int, int, int, int, int]
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _SnapshotBinding:
    path: Path
    directory_fingerprint: tuple[int, int, int, int, int]
    file_fingerprints: dict[str, tuple[int, int, int, int, int]]
    checksums: dict[str, str]
    metadata: dict[str, Any]
    timestamp_ns: int
    format: str


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _fingerprint(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_bound_file(
    path: Path,
    *,
    label: str,
    capture_content: bool = True,
) -> tuple[bytes, _FileWitness]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if capture_content:
                chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise ValueError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    return content, _FileWitness(
        path=path,
        fingerprint=_fingerprint(after),
        sha256=digest.hexdigest(),
        byte_count=byte_count,
    )


def _revalidate_witness(witness: _FileWitness, *, label: str) -> None:
    try:
        status = os.lstat(witness.path)
    except OSError as error:
        raise ValueError(f"{label} changed after validation") from error
    if not stat.S_ISREG(status.st_mode) or _fingerprint(status) != witness.fingerprint:
        raise ValueError(f"{label} changed after validation")


def _strict_jsonl(content: bytes, *, label: str) -> None:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not lines:
        raise ValueError(f"{label} must not be empty")
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{label} contains an empty line at {line_number}")
        _load_json_bytes(line.encode("utf-8"), label=f"{label} line {line_number}")


def _validate_snapshot_metadata_types(metadata: Mapping[str, Any]) -> None:
    required = {
        "scene_id",
        "frame_id",
        "timestamp",
        "revision",
        "voxel_size_m",
        "block_resolution",
        "schema_version",
    }
    if frozenset(metadata) not in {
        frozenset(required),
        frozenset({*required, "dense_semantic_provenance"}),
    }:
        raise ValueError("snapshot metadata fields are not exact")
    if not isinstance(metadata["scene_id"], str) or not metadata["scene_id"]:
        raise ValueError("snapshot scene_id is invalid")
    for name in ("frame_id", "revision"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError(f"snapshot {name} must be a non-negative integer")
    if type(metadata["schema_version"]) is not int:
        raise ValueError("snapshot schema_version must be an integer")
    if type(metadata["block_resolution"]) is not int:
        raise ValueError("snapshot block_resolution must be an integer")
    for name in ("timestamp", "voxel_size_m"):
        value = metadata[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"snapshot {name} must be finite numeric")


def _validate_compact_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        CompactOwnershipMetadata(**dict(metadata))
    except (TypeError, ValueError) as error:
        raise ValueError(f"compact snapshot metadata is invalid: {error}") from error


def _capture_snapshot_binding(
    path: Path,
    *,
    timestamp_ns: int,
    expected_checksums_sha256: str,
    checkpoint_format: str,
) -> _SnapshotBinding:
    try:
        directory_status = os.lstat(path)
    except OSError as error:
        raise ValueError(f"snapshot directory is missing: {path}") from error
    if not stat.S_ISDIR(directory_status.st_mode):
        raise ValueError(f"snapshot path is not a real directory: {path}")
    checksums_content, checksums_witness = _read_bound_file(
        path / "checksums.json",
        label="snapshot checksum manifest",
    )
    if checksums_witness.sha256 != expected_checksums_sha256:
        raise ValueError(f"snapshot checksum binding mismatch: {path}")
    checksums_payload = _load_json_bytes(
        checksums_content,
        label="snapshot checksum manifest",
    )
    allowed_files_v1 = {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
    }
    allowed_files_v2 = {*allowed_files_v1, "entities.jsonl"}
    if checkpoint_format == FULL_SNAPSHOT_FORMAT:
        allowed_checksum_sets = {
            frozenset(allowed_files_v1),
            frozenset(allowed_files_v2),
        }
    elif checkpoint_format == COMPACT_OWNERSHIP_FORMAT:
        allowed_checksum_sets = {
            frozenset({"metadata.json", "ownership.npz"})
        }
    else:
        raise ValueError("checkpoint format is invalid")
    if frozenset(checksums_payload) not in allowed_checksum_sets:
        raise ValueError("snapshot checksum manifest has unexpected files")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in checksums_payload.values()
    ):
        raise ValueError("snapshot checksum manifest has an invalid hash")
    expected_physical = {*checksums_payload, "checksums.json"}
    try:
        physical = {item.name for item in path.iterdir()}
    except OSError as error:
        raise ValueError(f"snapshot inventory is unreadable: {path}") from error
    if physical != expected_physical:
        raise ValueError("snapshot physical files do not match checksum manifest")

    file_fingerprints: dict[str, tuple[int, int, int, int, int]] = {}
    metadata_payload: dict[str, Any] | None = None
    for name in sorted(expected_physical):
        source = path / name
        try:
            source_status = os.lstat(source)
        except OSError as error:
            raise ValueError(f"snapshot file is missing: {source}") from error
        if not stat.S_ISREG(source_status.st_mode):
            raise ValueError(f"snapshot file is not regular: {source}")
        file_fingerprints[name] = _fingerprint(source_status)
        if name == "checksums.json":
            file_fingerprints[name] = checksums_witness.fingerprint
        if name not in {"metadata.json", "entities.jsonl"}:
            continue
        content, witness = _read_bound_file(source, label=f"snapshot {name}")
        file_fingerprints[name] = witness.fingerprint
        if witness.sha256 != checksums_payload[name]:
            raise ValueError(f"snapshot checksum mismatch for {name}")
        if name == "metadata.json":
            metadata_payload = _load_json_bytes(content, label="snapshot metadata")
            if checkpoint_format == COMPACT_OWNERSHIP_FORMAT:
                _validate_compact_metadata(metadata_payload)
            else:
                _validate_snapshot_metadata_types(metadata_payload)
        else:
            _strict_jsonl(content, label="snapshot entities")
    if metadata_payload is None:
        raise ValueError("snapshot metadata is missing")
    binding = _SnapshotBinding(
        path=path,
        directory_fingerprint=_fingerprint(directory_status),
        file_fingerprints=file_fingerprints,
        checksums={str(key): str(value) for key, value in checksums_payload.items()},
        metadata=metadata_payload,
        timestamp_ns=timestamp_ns,
        format=checkpoint_format,
    )
    _revalidate_snapshot_binding(binding)
    return binding


def _revalidate_snapshot_binding(binding: _SnapshotBinding) -> None:
    try:
        directory_status = os.lstat(binding.path)
    except OSError as error:
        raise ValueError(f"snapshot changed after index check: {binding.path}") from error
    if not stat.S_ISDIR(directory_status.st_mode) or (
        _fingerprint(directory_status) != binding.directory_fingerprint
    ):
        raise ValueError(f"snapshot identity changed after index check: {binding.path}")
    try:
        physical = {item.name for item in binding.path.iterdir()}
    except OSError as error:
        raise ValueError(f"snapshot changed after index check: {binding.path}") from error
    if physical != set(binding.file_fingerprints):
        raise ValueError(f"snapshot inventory changed after index check: {binding.path}")
    for name, expected in binding.file_fingerprints.items():
        try:
            status = os.lstat(binding.path / name)
        except OSError as error:
            raise ValueError(
                f"snapshot file changed after index check: {binding.path / name}"
            ) from error
        if not stat.S_ISREG(status.st_mode) or _fingerprint(status) != expected:
            raise ValueError(
                f"snapshot file identity changed after index check: {binding.path / name}"
            )


def _expected_source_roles(metadata: Mapping[str, Any]) -> set[str]:
    expected = {*_REPOSITORY_SOURCE_ROLES, "camera"}
    for scene in ("apartment", "office"):
        expected.update(f"{scene}.{suffix}" for suffix in _SCENE_SOURCE_SUFFIXES)
        expected.update(
            f"{scene}.depth.{frame_index:06d}"
            for frame_index in metadata["scene_frame_indices"][scene]
        )
    return expected


def _parse_source_frame_times(
    content: bytes,
    *,
    scene: str,
) -> dict[SnapshotKey, SourceFrameTime]:
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8")), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{scene} source timestamps are invalid") from error
    if not rows or rows[0] != [
        "frame_index",
        "sensor_timestamp_ns",
        "relative_timestamp_ns",
    ]:
        raise ValueError(f"{scene} source timestamp header is invalid")
    result: dict[SnapshotKey, SourceFrameTime] = {}
    previous_sensor = -1
    previous_relative = -1
    for expected_frame, row in enumerate(rows[1:]):
        if len(row) != 3 or any(
            not value or any(character not in "0123456789" for character in value)
            for value in row
        ):
            raise ValueError(f"{scene} source timestamp row is invalid")
        frame_index, sensor_timestamp_ns, relative_timestamp_ns = map(int, row)
        if (
            frame_index != expected_frame
            or sensor_timestamp_ns <= previous_sensor
            or relative_timestamp_ns <= previous_relative
        ):
            raise ValueError(f"{scene} source timestamps are not strictly ordered")
        result[(scene, frame_index)] = (
            sensor_timestamp_ns,
            relative_timestamp_ns,
        )
        previous_sensor = sensor_timestamp_ns
        previous_relative = relative_timestamp_ns
    if not result:
        raise ValueError(f"{scene} source timestamps are empty")
    return result


def _source_path(
    raw_path: object,
    *,
    role: str,
    dataset_root: Path,
    formal: bool,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("target source path is invalid")
    path = Path(raw_path)
    if formal and path.is_absolute():
        raise ValueError(f"formal target source path must be relative: {role}")
    if path.is_absolute():
        return path
    if "\\" in raw_path or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"target source path is not canonical: {role}")
    if path.as_posix() != raw_path:
        raise ValueError(f"target source path is not canonical: {role}")
    root = REPO_ROOT if role in _REPOSITORY_SOURCE_ROLES else dataset_root
    return root / path


def _load_target_package(
    target_dir: Path,
    *,
    dataset_root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    _FileWitness,
    list[_FileWitness],
    dict[SnapshotKey, SourceFrameTime],
]:
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise ValueError(f"dataset_root is not a real directory: {dataset_root}")
    manifest_content, manifest_witness = _read_bound_file(
        target_dir / "manifest.json", label="target manifest"
    )
    manifest = _load_json_bytes(manifest_content, label="target manifest")
    if not (
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1
        and manifest.get("manifest_id") == "tesse_cd_occlusion_v1_targets"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("status") in {"FIXTURE", "SMOKE", "GENERATED"}
        and manifest.get("targets_generated") is True
        and manifest.get("prediction_inputs_used") is False
    ):
        raise ValueError("target manifest identity mismatch")

    array_record = manifest.get("target_arrays")
    if (
        not isinstance(array_record, Mapping)
        or set(array_record) != {"path", "sha256", "byte_count", "count", "arrays"}
        or array_record.get("path") != "targets.npz"
    ):
        raise ValueError("target array binding is invalid")
    target_content, target_witness = _read_bound_file(
        target_dir / "targets.npz", label="target arrays"
    )
    if not (
        type(array_record.get("byte_count")) is int
        and array_record["byte_count"] == target_witness.byte_count
        and array_record.get("sha256") == target_witness.sha256
    ):
        raise ValueError("target array hash mismatch")
    try:
        with np.load(io.BytesIO(target_content), allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, ValueError) as error:
        raise ValueError("target arrays are not a valid safe NPZ") from error
    declared_arrays = array_record.get("arrays")
    if not isinstance(declared_arrays, Mapping) or set(declared_arrays) != set(arrays):
        raise ValueError("target array inventory mismatch")
    if type(array_record.get("count")) is not int or array_record.get(
        "count"
    ) != len(arrays):
        raise ValueError("target array count mismatch")
    for name, values in arrays.items():
        declaration = declared_arrays[name]
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "shape",
            "dtype",
            "element_count",
        }:
            raise ValueError(f"target array declaration mismatch: {name}")
        shape = declaration["shape"]
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or shape != list(values.shape)
            or not isinstance(declaration["dtype"], str)
            or declaration["dtype"] != str(values.dtype)
            or type(declaration["element_count"]) is not int
            or declaration["element_count"] != values.size
        ):
            raise ValueError(f"target array declaration mismatch: {name}")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("target metadata is missing")
    validate_generated_target(arrays, metadata)

    source_witnesses: list[_FileWitness] = []
    source_frame_times: dict[SnapshotKey, SourceFrameTime] = {}
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("target source bindings are missing")
    expected_roles = _expected_source_roles(metadata)
    if set(sources) != expected_roles:
        raise ValueError("target sources do not match the exact role allowlist")
    formal = manifest["status"] == "GENERATED"
    for role, raw_record in sorted(sources.items()):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"target source binding is invalid: {role}")
        timestamp_role = role.endswith(".timestamps")
        source_content, witness = _read_bound_file(
            _source_path(
                raw_record["path"],
                role=role,
                dataset_root=dataset_root,
                formal=formal,
            ),
            label=f"target source {role}",
            capture_content=timestamp_role,
        )
        if not (
            isinstance(raw_record["sha256"], str)
            and raw_record["sha256"] == witness.sha256
            and type(raw_record["byte_count"]) is int
            and raw_record["byte_count"] == witness.byte_count
        ):
            raise ValueError(f"target source hash mismatch: {role}")
        if timestamp_role:
            scene = role.removesuffix(".timestamps")
            source_frame_times.update(
                _parse_source_frame_times(source_content, scene=scene)
            )
        source_witnesses.append(witness)
    expected_frame_keys = {
        (scene, frame_index)
        for scene in ("apartment", "office")
        for frame_index in metadata["scene_frame_indices"][scene]
    }
    if set(source_frame_times) != expected_frame_keys:
        raise ValueError("source timestamp frames do not match the frozen target")
    return (
        arrays,
        dict(metadata),
        manifest_witness,
        [target_witness, *source_witnesses],
        source_frame_times,
    )


def _canonical_relative_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError("snapshot path must be canonical and relative")
    path = Path(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("snapshot path must be canonical and relative")
    if path.as_posix() != raw_path:
        raise ValueError("snapshot path must be canonical and relative")
    return path


def _plain_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _load_bound_run_config(
    record: object,
    *,
    index_root: Path,
) -> tuple[dict[str, Any], _FileWitness]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError("run config binding is invalid")
    relative = _canonical_relative_path(record["path"])
    path = index_root / relative
    if path.is_symlink():
        raise ValueError("run config path must not be a symlink")
    content, witness = _read_bound_file(path, label="normalized run config")
    if not (
        isinstance(record["sha256"], str)
        and record["sha256"] == witness.sha256
        and _plain_nonnegative_int(record["byte_count"])
        and record["byte_count"] == witness.byte_count
    ):
        raise ValueError("run config hash binding mismatch")
    config = _load_json_bytes(content, label="normalized run config")
    policy = config.get("missing_observation_policy")
    if not isinstance(policy, str) or policy not in {
        "signed_depth",
        "missing_as_absence",
    }:
        raise ValueError("normalized run config missing_observation_policy is invalid")
    return config, witness


def _required_checkpoint_times(metadata: Mapping[str, Any]) -> dict[SnapshotKey, int]:
    required: dict[SnapshotKey, int] = {}
    for episode in metadata["episodes"]:
        scene = str(episode["scene"])
        records = [episode["anchor"], *episode["checkpoints"]]
        for record in records:
            key = (scene, int(record["frame_index"]))
            timestamp = int(record["relative_timestamp_ns"])
            previous = required.setdefault(key, timestamp)
            if previous != timestamp:
                raise ValueError(f"conflicting checkpoint timestamps for {key}")
    return required


def build_evaluation_checkpoint_plan(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen snapshot-frame plan from an already validated target."""
    del arrays
    required = _required_checkpoint_times(metadata)
    frames = {
        scene: sorted(
            frame_index
            for candidate_scene, frame_index in required
            if candidate_scene == scene
        )
        for scene in ("apartment", "office")
    }
    binding = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_checkpoint_frames",
        "evaluation_checkpoint_frames": frames,
    }
    binding_bytes = json.dumps(
        binding,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **binding,
        "evaluation_checkpoint_frames_sha256": hashlib.sha256(
            binding_bytes
        ).hexdigest(),
    }


def _load_single_checkpoint_index(
    checkpoint_index: Path,
    *,
    metadata: Mapping[str, Any],
    target_manifest_witness: _FileWitness,
    checkpoint_plan: Mapping[str, Any],
    source_frame_times: Mapping[SnapshotKey, SourceFrameTime],
) -> tuple[
    Mapping[SnapshotKey, Any],
    dict[str, Any],
    _FileWitness,
    _FileWitness,
    str,
]:
    content, index_witness = _read_bound_file(
        checkpoint_index, label="checkpoint index"
    )
    payload = _load_json_bytes(content, label="checkpoint index")
    index_schema_version = payload.get("schema_version")
    base_fields = {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "run_config",
        "target_manifest",
        "evaluation_checkpoint_frames_sha256",
        "snapshots",
    }
    expected_fields = (
        base_fields
        if index_schema_version == 1
        else base_fields | {"scene", "algorithm_hash"}
    )
    if set(payload) != expected_fields or not (
        type(index_schema_version) is int
        and index_schema_version in {1, 2}
        and payload["manifest_id"] == "oviv2_tesse_cd_occlusion_checkpoints_v1"
        and payload["dataset"] == "TESSE-CD"
        and payload["method_id"] == "OVIV2"
    ):
        raise ValueError("checkpoint index identity mismatch")
    target_binding = payload["target_manifest"]
    if not isinstance(target_binding, Mapping) or not (
        set(target_binding) == {"sha256", "byte_count"}
        and target_binding["sha256"] == target_manifest_witness.sha256
        and _plain_nonnegative_int(target_binding["byte_count"])
        and target_binding["byte_count"] == target_manifest_witness.byte_count
    ):
        raise ValueError("checkpoint target manifest binding mismatch")
    run_config, run_config_witness = _load_bound_run_config(
        payload["run_config"],
        index_root=checkpoint_index.parent,
    )
    policy = run_config["missing_observation_policy"]
    if not isinstance(payload["evaluation_checkpoint_frames_sha256"], str) or (
        payload["evaluation_checkpoint_frames_sha256"]
        != checkpoint_plan["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("evaluation checkpoint frame binding mismatch")
    index_scene: str | None = None
    if index_schema_version == 2:
        index_scene = payload["scene"]
        index_algorithm_hash = payload["algorithm_hash"]
        if index_scene not in {"apartment", "office"}:
            raise ValueError("checkpoint index scene is invalid")
        if not (
            isinstance(index_algorithm_hash, str)
            and len(index_algorithm_hash) == 64
            and all(character in "0123456789abcdef" for character in index_algorithm_hash)
        ):
            raise ValueError("checkpoint index algorithm hash is invalid")
        if not (
            run_config.get("dataset") == "TESSE-CD"
            and run_config.get("method_id") == "OVIV2"
            and run_config.get("scene") == index_scene
        ):
            raise ValueError("normalized run config scene or identity mismatch")
        if not (
            run_config.get("algorithm_hash") == index_algorithm_hash
            and canonical_algorithm_hash(run_config) == index_algorithm_hash
        ):
            raise ValueError("normalized run config algorithm hash mismatch")
        if (
            run_config.get("occlusion_target_manifest_sha256")
            != target_manifest_witness.sha256
        ):
            raise ValueError("normalized run config target manifest binding mismatch")
        if (
            run_config.get("evaluation_checkpoint_frames_sha256")
            != checkpoint_plan["evaluation_checkpoint_frames_sha256"]
            or run_config.get("evaluation_checkpoint_frames")
            != checkpoint_plan["evaluation_checkpoint_frames"][index_scene]
        ):
            raise ValueError("normalized run config checkpoint plan binding mismatch")
    records = payload["snapshots"]
    if not isinstance(records, list):
        raise ValueError("checkpoint snapshot inventory must be a list")
    required = _required_checkpoint_times(metadata)
    snapshot_bindings: dict[SnapshotKey, _SnapshotBinding] = {}
    record_fields = {
        "scene",
        "frame_index",
        "timestamp_ns",
        "relative_timestamp_ns",
        "consumed_through_frame",
        "consumed_through_frame_exclusive",
        "path",
        "checksums_sha256",
    }
    if index_schema_version == 2:
        record_fields.add("format")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("checkpoint record fields are not exact")
        scene = record["scene"]
        frame_index = record["frame_index"]
        if (
            scene not in {"apartment", "office"}
            or type(frame_index) is not int
            or frame_index < 0
        ):
            raise ValueError("checkpoint scene or frame is invalid")
        if index_scene is not None and scene != index_scene:
            raise ValueError("checkpoint record violates index scene membership")
        key = (scene, frame_index)
        if key in snapshot_bindings:
            raise ValueError(f"duplicate checkpoint: {key}")
        if key not in required:
            raise ValueError(f"unexpected checkpoint: {key}")
        causal_values = (
            record["timestamp_ns"],
            record["relative_timestamp_ns"],
            record["consumed_through_frame"],
            record["consumed_through_frame_exclusive"],
        )
        source_time = source_frame_times.get(key)
        if not all(_plain_nonnegative_int(value) for value in causal_values) or not (
            source_time is not None
            and record["timestamp_ns"] == source_time[0]
            and record["relative_timestamp_ns"] == source_time[1]
            and record["relative_timestamp_ns"] == required[key]
            and record["consumed_through_frame"] == frame_index
            and record["consumed_through_frame_exclusive"] == frame_index + 1
        ):
            raise ValueError(
                f"future, source timestamp, or invalid causal boundary for checkpoint {key}"
            )
        relative = _canonical_relative_path(record["path"])
        snapshot_path = checkpoint_index.parent / relative
        current = checkpoint_index.parent
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"snapshot path contains a symlink: {relative}")
        if not isinstance(record["checksums_sha256"], str):
            raise ValueError(f"snapshot checksum binding mismatch: {key}")
        checkpoint_format = (
            FULL_SNAPSHOT_FORMAT
            if index_schema_version == 1
            else record["format"]
        )
        if not isinstance(checkpoint_format, str) or checkpoint_format not in {
            FULL_SNAPSHOT_FORMAT,
            COMPACT_OWNERSHIP_FORMAT,
        }:
            raise ValueError(f"checkpoint format is invalid: {key}")
        snapshot_bindings[key] = _capture_snapshot_binding(
            snapshot_path,
            timestamp_ns=record["timestamp_ns"],
            expected_checksums_sha256=record["checksums_sha256"],
            checkpoint_format=checkpoint_format,
        )
    expected_keys = (
        set(required)
        if index_scene is None
        else {key for key in required if key[0] == index_scene}
    )
    missing = sorted(expected_keys.difference(snapshot_bindings))
    if missing:
        scene, frame_index = missing[0]
        raise ValueError(f"missing checkpoint for {scene} frame {frame_index}")
    return (
        _LazyCheckpointSnapshots(snapshot_bindings),
        payload,
        index_witness,
        run_config_witness,
        policy,
    )


def _load_checkpoint_snapshots(
    checkpoint_index: str | Path | Sequence[str | Path],
    *,
    metadata: Mapping[str, Any],
    target_manifest_witness: _FileWitness,
    checkpoint_plan: Mapping[str, Any],
    source_frame_times: Mapping[SnapshotKey, SourceFrameTime],
) -> tuple[
    Mapping[SnapshotKey, Any],
    dict[str, Any],
    _FileWitness | tuple[_FileWitness, ...],
    _FileWitness | tuple[_FileWitness, ...],
    str,
]:
    if isinstance(checkpoint_index, (str, Path)):
        paths = (Path(checkpoint_index),)
    else:
        paths = tuple(Path(path) for path in checkpoint_index)
    if not paths:
        raise ValueError("at least one checkpoint index is required")

    loaded = [
        _load_single_checkpoint_index(
            path,
            metadata=metadata,
            target_manifest_witness=target_manifest_witness,
            checkpoint_plan=checkpoint_plan,
            source_frame_times=source_frame_times,
        )
        for path in paths
    ]
    if len(loaded) == 1 and loaded[0][1]["schema_version"] == 1:
        return loaded[0]
    if len(loaded) != 2 or any(
        item[1]["schema_version"] != 2 for item in loaded
    ):
        raise ValueError(
            "schema2 checkpoint indexes must contain apartment and office exactly"
        )

    loaded.sort(key=lambda item: item[1]["scene"])
    scenes = [item[1]["scene"] for item in loaded]
    if scenes != ["apartment", "office"]:
        raise ValueError(
            "schema2 checkpoint indexes must contain apartment and office exactly"
        )
    algorithm_hashes = {item[1]["algorithm_hash"] for item in loaded}
    if len(algorithm_hashes) != 1:
        raise ValueError("checkpoint index algorithm hash mismatch across scenes")
    policies = {item[4] for item in loaded}
    if len(policies) != 1:
        raise ValueError("normalized run config policy mismatch across scenes")

    combined_bindings: dict[SnapshotKey, _SnapshotBinding] = {}
    for snapshots, _, _, _, _ in loaded:
        bindings = snapshots._bindings  # type: ignore[attr-defined]
        duplicates = set(combined_bindings).intersection(bindings)
        if duplicates:
            raise ValueError(f"duplicate checkpoint across indexes: {min(duplicates)}")
        combined_bindings.update(bindings)
    required = set(_required_checkpoint_times(metadata))
    if set(combined_bindings) != required:
        missing = sorted(required.difference(combined_bindings))
        if missing:
            raise ValueError(f"missing checkpoint for {missing[0][0]} frame {missing[0][1]}")
        raise ValueError("checkpoint inventory contains unexpected records")

    payload = {
        "method_id": "OVIV2",
        "schema_version": 2,
        "scene_indexes": [item[1] for item in loaded],
    }
    return (
        _LazyCheckpointSnapshots(combined_bindings),
        payload,
        tuple(item[2] for item in loaded),
        tuple(item[3] for item in loaded),
        next(iter(policies)),
    )


def _render_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("occlusion result is not finite canonical JSON") from error


def _witness_tuple(
    witness: _FileWitness | tuple[_FileWitness, ...],
) -> tuple[_FileWitness, ...]:
    return witness if isinstance(witness, tuple) else (witness,)


def _witness_result(
    witness: _FileWitness | tuple[_FileWitness, ...],
) -> dict[str, int | str] | list[dict[str, int | str]]:
    records = [
        {"sha256": item.sha256, "byte_count": item.byte_count}
        for item in _witness_tuple(witness)
    ]
    return records[0] if len(records) == 1 else records


def _publish_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise ValueError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"output already exists: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _required_snapshot(
    snapshots: Mapping[SnapshotKey, Any], scene: str, frame_index: int
) -> Any:
    key = (scene, frame_index)
    if key not in snapshots:
        raise ValueError(f"missing checkpoint for {scene} frame {frame_index}")
    snapshot = snapshots[key]
    metadata = snapshot.metadata
    if metadata.scene_id != scene:
        raise ValueError(f"checkpoint scene mismatch for {scene} frame {frame_index}")
    if metadata.frame_id > frame_index:
        raise ValueError(f"future snapshot for {scene} frame {frame_index}")
    if metadata.frame_id < frame_index:
        raise ValueError(f"missing exact checkpoint for {scene} frame {frame_index}")
    return snapshot


def _voxel_keys(values: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    array = np.asarray(values)
    if array.dtype != np.int64 or array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("occlusion target arrays must have shape (N, 3) and dtype int64")
    return tuple(tuple(int(value) for value in row) for row in array)


def _majority_owner(
    snapshot: Any,
    voxel_keys: tuple[tuple[int, int, int], ...],
) -> tuple[int | None, int, bool]:
    counts = Counter(
        record.entity_id
        for key in voxel_keys
        if (record := snapshot.ownership.owner_of(key)) is not None
    )
    if not counts:
        return None, 0, False
    majority_count = max(counts.values())
    tied = sorted(entity_id for entity_id, count in counts.items() if count == majority_count)
    return tied[0], majority_count, len(tied) > 1


def _empty_counts() -> dict[str, int]:
    return {
        "all_gt_occluded_target_voxels": 0,
        "anchor_owned_target_voxels": 0,
        "retained_owner_count": 0,
        "false_release_count": 0,
        "false_reassignment_count": 0,
        "object_checkpoint_count": 0,
        "anchor_mapped_object_checkpoint_count": 0,
        "retained_object_checkpoint_count": 0,
        "episode_count": 0,
        "anchor_mapped_episode_count": 0,
        "zero_release_episode_count": 0,
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finalize_metrics(counts: Mapping[str, int]) -> dict[str, int | float]:
    result: dict[str, int | float] = dict(counts)
    anchor_count = counts["anchor_owned_target_voxels"]
    gt_count = counts["all_gt_occluded_target_voxels"]
    mapped_objects = counts["anchor_mapped_object_checkpoint_count"]
    result.update(
        {
            "false_release_rate": _safe_rate(
                counts["false_release_count"], anchor_count
            ),
            "false_reassignment_rate": _safe_rate(
                counts["false_reassignment_count"], anchor_count
            ),
            "retained_ownership_recall": _safe_rate(
                counts["retained_owner_count"], anchor_count
            ),
            "gt_retained_object_recall": _safe_rate(
                counts["retained_owner_count"], gt_count
            ),
            "retained_object_recall": _safe_rate(
                counts["retained_object_checkpoint_count"], mapped_objects
            ),
            "zero_release_episode_rate": _safe_rate(
                counts["zero_release_episode_count"], counts["episode_count"]
            ),
        }
    )
    return result


def evaluate_fixed_anchor_ownership(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    snapshots: Mapping[SnapshotKey, Any],
    missing_observation_policy: str,
) -> dict[str, Any]:
    """Map at each anchor once, then score every later checkpoint without rematching."""
    if missing_observation_policy not in {"signed_depth", "missing_as_absence"}:
        raise ValueError("missing_observation_policy is invalid")
    raw_episodes = metadata.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("occlusion metadata has no episodes")
    episodes = {str(item["episode_id"]): item for item in raw_episodes}
    if len(episodes) != len(raw_episodes):
        raise ValueError("occlusion episode IDs must be unique")

    mappings: list[dict[str, Any]] = []
    episode_counts: dict[str, dict[str, int]] = {}
    episode_scenes: dict[str, str] = {}
    for episode_id, episode in episodes.items():
        scene = str(episode["scene"])
        episode_scenes[episode_id] = scene
        anchor = episode["anchor"]
        anchor_snapshot = _required_snapshot(
            snapshots, scene, int(anchor["frame_index"])
        )
        anchor_keys = _voxel_keys(arrays[str(anchor["array"])])
        mapped_owner, majority_count, tie = _majority_owner(
            anchor_snapshot, anchor_keys
        )
        mappings.append(
            {
                "episode_id": episode_id,
                "gt_object_id": episode["object_id"],
                "predicted_owner_id": mapped_owner,
                "majority_count": majority_count,
                "tie": tie,
            }
        )

        counts = _empty_counts()
        counts["episode_count"] = 1
        episode_anchor_voxels = 0
        episode_releases = 0
        for checkpoint in episode["checkpoints"]:
            counts["object_checkpoint_count"] += 1
            checkpoint_snapshot = _required_snapshot(
                snapshots, scene, int(checkpoint["frame_index"])
            )
            target_keys = _voxel_keys(arrays[str(checkpoint["array"])])
            counts["all_gt_occluded_target_voxels"] += len(target_keys)
            retained_at_checkpoint = 0
            eligible_at_checkpoint = 0
            if mapped_owner is not None:
                for key in target_keys:
                    anchor_record = anchor_snapshot.ownership.owner_of(key)
                    if (
                        anchor_record is None
                        or anchor_record.entity_id != mapped_owner
                    ):
                        continue
                    eligible_at_checkpoint += 1
                    current_record = checkpoint_snapshot.ownership.owner_of(key)
                    if current_record is None:
                        counts["false_release_count"] += 1
                        episode_releases += 1
                    elif current_record.entity_id == mapped_owner:
                        counts["retained_owner_count"] += 1
                        retained_at_checkpoint += 1
                    else:
                        counts["false_reassignment_count"] += 1
            counts["anchor_owned_target_voxels"] += eligible_at_checkpoint
            episode_anchor_voxels += eligible_at_checkpoint
            if eligible_at_checkpoint:
                counts["anchor_mapped_object_checkpoint_count"] += 1
                if retained_at_checkpoint:
                    counts["retained_object_checkpoint_count"] += 1
        if episode_anchor_voxels:
            counts["anchor_mapped_episode_count"] = 1
            if episode_releases == 0:
                counts["zero_release_episode_count"] = 1
        episode_counts[episode_id] = counts

    stress_layers: dict[str, dict[str, int | float]] = {}
    layers = metadata.get("stress_layers")
    if not isinstance(layers, Mapping):
        raise ValueError("occlusion stress layers are missing")
    for label in ("all", "0.50", "0.75", "0.90"):
        layer = layers.get(label)
        if not isinstance(layer, Mapping) or not isinstance(
            layer.get("episode_ids"), list
        ):
            raise ValueError(f"occlusion stress layer {label} is invalid")
        counts = _empty_counts()
        for episode_id in layer["episode_ids"]:
            if episode_id not in episode_counts:
                raise ValueError(f"unknown episode in stress layer {label}")
            for name, value in episode_counts[episode_id].items():
                counts[name] += value
        stress_layers[label] = _finalize_metrics(counts)

    headline = stress_layers["0.90"]
    headline_ids = layers["0.90"]["episode_ids"]
    scene_coverage = {
        scene: {
            "episode_count": sum(
                episode_scenes[episode_id] == scene for episode_id in headline_ids
            ),
            "anchor_mapped_episode_count": sum(
                episode_scenes[episode_id] == scene
                and episode_counts[episode_id]["anchor_mapped_episode_count"] == 1
                for episode_id in headline_ids
            ),
        }
        for scene in ("apartment", "office")
    }
    both_scenes_covered = all(
        coverage["episode_count"] > 0
        and coverage["anchor_mapped_episode_count"] == coverage["episode_count"]
        for coverage in scene_coverage.values()
    )
    return {
        "fixed_anchor_mapping_rule": "majority_owner_then_lowest_entity_id",
        "fixed_anchor_mappings": mappings,
        "stress_layers": stress_layers,
        "headline_gate": {
            "stress_layer": "0.90",
            "missing_observation_policy": missing_observation_policy,
            "episode_count": headline["episode_count"],
            "anchor_mapped_episode_count": headline[
                "anchor_mapped_episode_count"
            ],
            "anchor_owned_target_voxels": headline[
                "anchor_owned_target_voxels"
            ],
            "false_release_count": headline["false_release_count"],
            "false_reassignment_count": headline["false_reassignment_count"],
            "retained_ownership_recall": headline[
                "retained_ownership_recall"
            ],
            "scene_coverage": scene_coverage,
            "passed": bool(
                missing_observation_policy == "signed_depth"
                and both_scenes_covered
                and headline["anchor_owned_target_voxels"] > 0
                and headline["anchor_mapped_episode_count"]
                == headline["episode_count"]
                and headline["false_release_count"] == 0
                and headline["false_reassignment_count"] == 0
                and headline["retained_ownership_recall"] == 1.0
            ),
        },
    }


def evaluate_occlusion_package(
    *,
    target_dir: str | Path,
    checkpoint_index: str | Path | Sequence[str | Path],
    dataset_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate frozen inputs, evaluate fixed anchors, and optionally publish JSON."""
    output = Path(output_path) if output_path is not None else None
    if output is not None and os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    (
        arrays,
        metadata,
        target_manifest_witness,
        target_witnesses,
        source_frame_times,
    ) = (
        _load_target_package(
            Path(target_dir),
            dataset_root=Path(dataset_root),
        )
    )
    checkpoint_plan = build_evaluation_checkpoint_plan(
        arrays=arrays,
        metadata=metadata,
    )
    (
        snapshots,
        checkpoint_payload,
        checkpoint_witness,
        run_config_witness,
        missing_observation_policy,
    ) = _load_checkpoint_snapshots(
        checkpoint_index,
        metadata=metadata,
        target_manifest_witness=target_manifest_witness,
        checkpoint_plan=checkpoint_plan,
        source_frame_times=source_frame_times,
    )
    metrics = evaluate_fixed_anchor_ownership(
        arrays=arrays,
        metadata=metadata,
        snapshots=snapshots,
        missing_observation_policy=missing_observation_policy,
    )
    result = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_evaluation_v1",
        "dataset": "TESSE-CD",
        "method_id": checkpoint_payload["method_id"],
        "missing_observation_policy": missing_observation_policy,
        "target_manifest": {
            "sha256": target_manifest_witness.sha256,
            "byte_count": target_manifest_witness.byte_count,
        },
        "checkpoint_count": len(snapshots),
        "checkpoint_index": _witness_result(checkpoint_witness),
        "run_config": _witness_result(run_config_witness),
        "evaluation_checkpoint_frames_sha256": checkpoint_plan[
            "evaluation_checkpoint_frames_sha256"
        ],
        **metrics,
    }
    encoded = _render_json(result)
    _revalidate_witness(target_manifest_witness, label="target manifest")
    for witness in target_witnesses:
        _revalidate_witness(witness, label="target input")
    for witness in _witness_tuple(checkpoint_witness):
        _revalidate_witness(witness, label="checkpoint index")
    for witness in _witness_tuple(run_config_witness):
        _revalidate_witness(witness, label="normalized run config")
    snapshots.revalidate_all()  # type: ignore[attr-defined]
    if output is not None:
        _publish_no_replace(output, encoded)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_occlusion_package(
        target_dir=args.targets,
        checkpoint_index=args.checkpoints,
        dataset_root=args.dataset_root,
        output_path=args.output,
    )
    print(json.dumps(result["headline_gate"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
