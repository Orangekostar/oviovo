#!/usr/bin/env python3
"""Evaluate OVIV2 temporal lifecycle behavior on frozen TESSE-CD occlusion targets."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.oviv2_temporal_occlusion import (  # noqa: E402
    EVALUATION_FORMAT,
    evaluate_temporal_occlusion,
    mechanism_telemetry_from_sources,
)
from src.oviv2.temporal_snapshot import (  # noqa: E402
    TEMPORAL_COMPACT_FORMAT,
    TemporalCompactCheckpoint,
)
from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    _expected_source_roles,
    _load_target_package,
    _revalidate_witness,
)


_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_TARGET_BYTES = 512 * 1024 * 1024
_MAX_ENTITIES = 4096
_MAX_OBJECT_VOXELS = 1_000_000
_INDEX_FIELDS = {
    "schema_version", "format", "protocol_id", "dataset", "method_id", "scene",
    "algorithm_hash", "schedule", "target_manifest", "input_sha256", "code_commit",
    "source_bindings", "checkpoints",
}
_CHECKPOINT_FIELDS = {
    "scene", "frame_index", "timestamp_ns", "relative_timestamp_ns",
    "consumed_through_frame", "consumed_through_frame_exclusive", "event_ids",
    "roles", "format", "maximum_entities", "maximum_object_voxels", "artifact",
    "checksums_sha256",
}


@dataclass(frozen=True)
class _CompactBinding:
    key: tuple[str, int]
    path: Path
    run_root: Path
    tree_record: Mapping[str, Any]
    checksums_sha256: str
    maximum_entities: int
    maximum_object_voxels: int
    timestamp_ns: int
    relative_timestamp_ns: int
    algorithm_hash: str
    _source_witness: Any = field(default=None, init=False, repr=False, compare=False)

    def load(self) -> TemporalCompactCheckpoint:
        if _tree_record(self.path, self.run_root) != self.tree_record:
            raise ValueError("compact artifact changed")
        loaded = TemporalCompactCheckpoint.load(
            self.path,
            maximum_entities=self.maximum_entities,
            maximum_object_voxels=self.maximum_object_voxels,
        )
        if (
            loaded.metadata.frame_id != self.key[1]
            or loaded.metadata.scene_id != self.key[0]
            or round(loaded.metadata.timestamp * 1e9) != self.timestamp_ns
            or loaded.metadata.config_sha256 != self.algorithm_hash
        ):
            raise ValueError("compact metadata authority mismatch")
        checksum_content, _ = _read(
            self.path / "checksums.json", "compact checksums"
        )
        if _sha(checksum_content) != self.checksums_sha256:
            raise ValueError("compact checksum binding mismatch")
        loaded.revalidate_source()
        object.__setattr__(self, "_source_witness", getattr(loaded, "_source_witness"))
        return loaded

    def revalidate(self) -> None:
        if self._source_witness is None:
            raise ValueError("compact artifact was not evaluated")
        try:
            self._source_witness.revalidate()
        except ValueError as error:
            raise ValueError("compact artifact changed") from error


class _LazyCheckpoints(Mapping[tuple[str, int], TemporalCompactCheckpoint]):
    def __init__(self, bindings: Mapping[tuple[str, int], _CompactBinding]) -> None:
        self.bindings = dict(bindings)
        self._cached_key: tuple[str, int] | None = None
        self._cached_value: TemporalCompactCheckpoint | None = None
        self.maximum_cached_checkpoints = 0

    def __len__(self) -> int:
        return len(self.bindings)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self.bindings)

    def __getitem__(self, key: tuple[str, int]) -> TemporalCompactCheckpoint:
        if key != self._cached_key:
            self._cached_key = None
            self._cached_value = None
            self._cached_value = self.bindings[key].load()
            self._cached_key = key
            self.maximum_cached_checkpoints = max(self.maximum_cached_checkpoints, 1)
        assert self._cached_value is not None
        return self._cached_value

    def revalidate_all(self) -> None:
        self._cached_key = None
        self._cached_value = None
        for binding in self.bindings.values():
            binding.revalidate()


def _directory_identity_chain(path: Path) -> tuple[tuple[int, int], ...]:
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    identities = []
    try:
        status = os.fstat(descriptor)
        identities.append((status.st_dev, status.st_ino))
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            status = os.fstat(descriptor)
            identities.append((status.st_dev, status.st_ino))
        return tuple(identities)
    except OSError as error:
        raise ValueError("validated directory changed during evaluation") from error
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _DirectoryIdentityWitness:
    path: Path
    identities: tuple[tuple[int, int], ...]

    @classmethod
    def capture(cls, path: Path) -> "_DirectoryIdentityWitness":
        return cls(path.absolute(), _directory_identity_chain(path))

    def revalidate(self) -> None:
        if _directory_identity_chain(self.path) != self.identities:
            raise ValueError("validated directory changed during evaluation")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path, label: str, limit: int = _MAX_JSON_BYTES) -> tuple[bytes, tuple[int, int, int, int, int]]:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try: status = os.lstat(component)
        except FileNotFoundError: continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} contains a symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode): raise ValueError(f"{label} is not regular")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > limit:
                raise ValueError(f"{label} exceeds size limit")
        content = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if fingerprint(before) != fingerprint(after): raise ValueError(f"{label} changed while reading")
    return content, fingerprint(after)


def _json(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode(), object_pairs_hook=_strict_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict): raise ValueError(f"{label} must be an object")
    return value


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _content_record(content: bytes) -> dict[str, Any]:
    return {"sha256": _sha(content), "byte_count": len(content)}


def _valid_record(value: object, content: bytes) -> bool:
    return isinstance(value, Mapping) and set(value) >= {"sha256", "byte_count"} and value["sha256"] == _sha(content) and type(value["byte_count"]) is int and value["byte_count"] == len(content)


def _hex(value: object, lengths: set[int]) -> bool:
    return isinstance(value, str) and len(value) in lengths and all(char in "0123456789abcdef" for char in value)


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} path is invalid")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} path traversal is forbidden")
    return path


def _source_path(raw: object, role: str, dataset_root: Path) -> Path:
    if not isinstance(raw, str) or not raw: raise ValueError("source path is invalid")
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve(strict=True)
        root = dataset_root.resolve(strict=True)
        if not resolved.is_relative_to(root): raise ValueError("absolute source escapes dataset root")
        return resolved
    relative = _relative(raw, "source")
    root = REPO_ROOT if role in {"source_manifest", "schedule", "rgbd_lock"} else dataset_root
    return root / relative


def _asset_root(dataset_root: Path) -> Path:
    resolved = dataset_root.resolve(strict=True)
    if resolved.name in {"apartment", "office"} and resolved.parent.name == "rgbd_v1" and resolved.parent.parent.name == "derived":
        return resolved.parents[2]
    return resolved


def _load_target(target_path: Path, dataset_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], list[Any], dict[str, Any], dict[tuple[str, int], tuple[int, int]]]:
    if target_path.name != "manifest.json":
        raise ValueError("targets must name manifest.json")
    for path, limit, label in ((target_path, _MAX_JSON_BYTES, "target manifest"),
                               (target_path.parent / "targets.npz", _MAX_TARGET_BYTES, "target arrays")):
        status = os.lstat(path)
        if not stat.S_ISREG(status.st_mode) or status.st_size > limit:
            raise ValueError(f"{label} exceeds size or type limit")
    arrays, metadata, manifest_witness, target_witnesses, source_frame_times = _load_target_package(
        target_path.parent, dataset_root=_asset_root(dataset_root)
    )
    roles = sorted(_expected_source_roles(metadata))
    if len(target_witnesses) != len(roles) + 1:
        raise ValueError("target source witness inventory mismatch")
    sources = {
        role: {"sha256": witness.sha256, "byte_count": witness.byte_count}
        for role, witness in zip(roles, target_witnesses[1:])
    }
    return (
        arrays,
        metadata,
        {"sha256": manifest_witness.sha256, "byte_count": manifest_witness.byte_count},
        [manifest_witness, *target_witnesses],
        sources,
        source_frame_times,
    )


def _tree_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir(): raise ValueError("compact artifact is not a real directory")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256(); count = 0
    for item in files:
        relative = item.relative_to(path).as_posix(); content, _ = _read(item, "compact member", _MAX_TARGET_BYTES)
        count += len(content); digest.update(relative.encode()); digest.update(b"\0")
        digest.update(bytes.fromhex(_sha(content))); digest.update(b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "byte_count": count}


def _json_lines(content: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = content.decode().splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid {label}") from error
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _json(line.encode(), f"{label} line {number}")
        records.append(value)
    return records


def _load_mechanism_sources(
    run_root: Path,
    run: Mapping[str, Any],
    scene: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[Any]]:
    source_index_record = run.get("source_index")
    if not isinstance(source_index_record, Mapping) or set(source_index_record) != {
        "path", "sha256", "byte_count"
    }:
        raise ValueError("run manifest mechanism source index is missing")
    source_index_path = run_root / _relative(
        source_index_record["path"], "mechanism source index"
    )
    source_index_content, source_index_witness = _read(
        source_index_path, "mechanism source index"
    )
    if not _valid_record(source_index_record, source_index_content):
        raise ValueError("mechanism source index binding mismatch")
    source_index = _json(source_index_content, "mechanism source index")
    if not (
        source_index.get("schema_version") == 1
        and source_index.get("dataset") == "TESSE-CD"
        and source_index.get("method") == "OVIV2"
        and source_index.get("scene") == scene
    ):
        raise ValueError("mechanism source index identity mismatch")
    contents: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    witnesses: list[Any] = [(source_index_path, source_index_witness)]
    for role in (
        "trajectories", "lifecycle_transitions", "frame_coverage",
        "runtime_diagnostics",
    ):
        record = source_index.get(role)
        if not isinstance(record, Mapping) or set(record) != {
            "path", "sha256", "byte_count"
        }:
            raise ValueError(f"{role} source record is not exact")
        path = run_root / _relative(record["path"], role)
        content, witness = _read(path, role)
        if not _valid_record(record, content):
            raise ValueError(f"{role} source binding mismatch")
        contents[role] = content
        records[role] = dict(record)
        witnesses.append((path, witness))
    telemetry = mechanism_telemetry_from_sources(
        trajectories=_json_lines(contents["trajectories"], "trajectories"),
        lifecycle_transitions=_json_lines(
            contents["lifecycle_transitions"], "lifecycle transitions"
        ),
        frame_coverage=_json_lines(
            contents["frame_coverage"], "frame coverage"
        ),
        runtime_diagnostics=_json(
            contents["runtime_diagnostics"], "runtime diagnostics"
        ),
        source_records=records,
    )
    return telemetry, {"scene": scene, **dict(source_index_record)}, witnesses


def _load_index(index_path: Path, target_record: Mapping[str, Any], metadata: Mapping[str, Any], sources: Mapping[str, Any], source_frame_times: Mapping[tuple[str, int], tuple[int, int]]) -> tuple[dict[tuple[str, int], _CompactBinding], dict[tuple[str, int], int], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], list[Any]]:
    run_root = index_path.parent
    index_content, index_witness = _read(index_path, "occlusion checkpoint index")
    index = _json(index_content, "occlusion checkpoint index")
    if set(index) != _INDEX_FIELDS or not (
        index.get("schema_version") == 1 and index.get("format") == EVALUATION_FORMAT
        and index.get("protocol_id") == "oviv2-tessecd-v2" and index.get("dataset") == "TESSE-CD"
        and index.get("method_id") == "OVIV2" and index.get("scene") in {"apartment", "office"}
        and _hex(index.get("algorithm_hash"), {64})
        and _hex(index.get("input_sha256"), {64})
        and _hex(index.get("code_commit"), {40, 64})
        and isinstance(index.get("source_bindings"), Mapping)
    ): raise ValueError("occlusion checkpoint index identity mismatch")
    run_content, run_witness = _read(run_root / "run_manifest.json", "sibling run manifest")
    run = _json(run_content, "sibling run manifest")
    if not (
        run.get("schema_version") == 2
        and run.get("protocol_id") == "oviv2-tessecd-v2"
        and run.get("dataset") == "TESSE-CD"
        and run.get("method_id") == "OVIV2"
        and run.get("scene") == index["scene"]
    ):
        raise ValueError("sibling run manifest identity mismatch")
    mechanism_telemetry, mechanism_source_index, mechanism_witnesses = _load_mechanism_sources(
        run_root, run, index["scene"]
    )
    record = run.get("occlusion_checkpoint_index")
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"} or record["path"] != index_path.name or not _valid_record(record, index_content):
        raise ValueError("run manifest index binding mismatch")
    common = ("protocol_id", "dataset", "method_id", "scene", "algorithm_hash", "schedule", "target_manifest", "input_sha256", "code_commit", "source_bindings")
    if any(run.get(key) != index.get(key) for key in common): raise ValueError("run/index authority mismatch")
    if not (isinstance(index["target_manifest"], Mapping) and set(index["target_manifest"]) == {"sha256", "byte_count"} and index["target_manifest"] == target_record):
        raise ValueError("index target binding mismatch")
    schedule_source = sources.get("schedule")
    if not isinstance(schedule_source, Mapping) or {key: schedule_source[key] for key in ("sha256", "byte_count")} != index["schedule"]:
        raise ValueError("index schedule binding mismatch")
    required: dict[int, int] = {}
    for episode in metadata["episodes"]:
        if episode.get("scene") != index["scene"]: continue
        for event in [episode["anchor"], *episode["checkpoints"]]:
            frame = int(event["frame_index"]); timestamp = int(event["relative_timestamp_ns"])
            if frame in required and required[frame] != timestamp: raise ValueError("target event timestamp conflict")
            required[frame] = timestamp
    records = index.get("checkpoints")
    if not isinstance(records, list): raise ValueError("checkpoint inventory is invalid")
    checkpoints: dict[tuple[str, int], _CompactBinding] = {}
    relative_times: dict[tuple[str, int], int] = {}
    witnesses = [
        _DirectoryIdentityWitness.capture(run_root),
        (index_path, index_witness),
        (run_root / "run_manifest.json", run_witness),
        *mechanism_witnesses,
    ]
    for item in records:
        if not isinstance(item, Mapping) or set(item) != _CHECKPOINT_FIELDS: raise ValueError("checkpoint fields are not exact")
        frame = item["frame_index"]
        causal = item["consumed_through_frame"] == frame and item["consumed_through_frame_exclusive"] == frame + 1
        if not (type(frame) is int and frame >= 0 and type(item["timestamp_ns"]) is int and item["timestamp_ns"] >= 0
                and type(item["relative_timestamp_ns"]) is int and item["scene"] == index["scene"]
                and item["format"] == TEMPORAL_COMPACT_FORMAT and causal
                and item["relative_timestamp_ns"] == required.get(frame)):
            raise ValueError("checkpoint is outside target or causal boundary")
        if not (
            isinstance(item["event_ids"], list)
            and all(isinstance(value, str) and value for value in item["event_ids"])
            and len(item["event_ids"]) == len(set(item["event_ids"]))
            and isinstance(item["roles"], list)
            and "occlusion_v1" in item["roles"]
            and all(isinstance(value, str) and value for value in item["roles"])
            and len(item["roles"]) == len(set(item["roles"]))
        ):
            raise ValueError("checkpoint event/role binding is invalid")
        key = (index["scene"], frame)
        if source_frame_times.get(key) != (item["timestamp_ns"], item["relative_timestamp_ns"]):
            raise ValueError("checkpoint timestamps disagree with frozen source authority")
        if key in checkpoints: raise ValueError("duplicate checkpoint frame")
        maximum_entities = item["maximum_entities"]; maximum_voxels = item["maximum_object_voxels"]
        if not (type(maximum_entities) is int and 0 < maximum_entities <= _MAX_ENTITIES and type(maximum_voxels) is int and 0 < maximum_voxels <= _MAX_OBJECT_VOXELS):
            raise ValueError("checkpoint capacity is invalid")
        artifact_record = item["artifact"]
        if not isinstance(artifact_record, Mapping) or set(artifact_record) != {"path", "sha256", "byte_count"}: raise ValueError("artifact binding is invalid")
        artifact = run_root / _relative(artifact_record["path"], "artifact")
        if _tree_record(artifact, run_root) != artifact_record: raise ValueError("compact artifact changed")
        if not _hex(item["checksums_sha256"], {64}):
            raise ValueError("compact checksum digest is invalid")
        checkpoints[key] = _CompactBinding(
            key=key, path=artifact, run_root=run_root, tree_record=dict(artifact_record),
            checksums_sha256=item["checksums_sha256"], maximum_entities=maximum_entities,
            maximum_object_voxels=maximum_voxels, timestamp_ns=item["timestamp_ns"],
            relative_timestamp_ns=item["relative_timestamp_ns"], algorithm_hash=index["algorithm_hash"],
        )
        relative_times[key] = item["relative_timestamp_ns"]
    if [item["frame_index"] for item in records] != sorted(required):
        raise ValueError("checkpoint frames must exactly match target order")
    return (
        checkpoints, relative_times, index, _content_record(index_content),
        mechanism_telemetry, mechanism_source_index, witnesses,
    )


def _revalidate(witnesses: Sequence[Any]) -> None:
    for witness in witnesses:
        if isinstance(witness, _DirectoryIdentityWitness):
            witness.revalidate()
            continue
        if not isinstance(witness, tuple):
            _revalidate_witness(witness, label="evaluation input")
            continue
        path, expected = witness
        status = os.lstat(path)
        observed = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        if not stat.S_ISREG(status.st_mode) or observed != expected: raise ValueError("validated input changed during evaluation")


def _open_output_parent(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError("output parent is not a stable real directory") from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _parent_identity(path: Path) -> tuple[int, int]:
    descriptor = _open_output_parent(path)
    try:
        status = os.fstat(descriptor)
        return status.st_dev, status.st_ino
    finally:
        os.close(descriptor)


def _named_identity(parent: int, name: str) -> tuple[int, int] | None:
    try:
        status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _unlink_if_identity(
    parent: int, name: str, expected: tuple[int, int] | None
) -> None:
    if expected is not None and _named_identity(parent, name) == expected:
        os.unlink(name, dir_fd=parent)


def _publish(path: Path, content: bytes) -> None:
    path = path.absolute()
    parent = _open_output_parent(path)
    parent_identity = (os.fstat(parent).st_dev, os.fstat(parent).st_ino)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}"
    temporary_created = False
    linked = False
    published = False
    temporary_identity: tuple[int, int] | None = None
    destination_identity: tuple[int, int] | None = None
    descriptor = -1
    try:
        try:
            os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(path)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        temporary_status = os.fstat(descriptor)
        temporary_identity = (temporary_status.st_dev, temporary_status.st_ino)
        if _named_identity(parent, temporary_name) != temporary_identity:
            raise ValueError("temporary output changed during publication")
        if _parent_identity(path) != parent_identity:
            raise ValueError("output parent changed during publication")
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(path)
        linked = True
        destination_identity = _named_identity(parent, path.name)
        if (
            destination_identity != temporary_identity
            or _named_identity(parent, temporary_name) != temporary_identity
        ):
            raise ValueError("temporary output changed during publication")
        if _parent_identity(path) != parent_identity:
            raise ValueError("output parent changed during publication")
        os.fsync(parent)
        if _parent_identity(path) != parent_identity:
            raise ValueError("output parent changed during publication")
        published = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if linked and not published:
            try:
                _unlink_if_identity(parent, path.name, destination_identity)
            except FileNotFoundError:
                pass
        if temporary_created:
            try:
                _unlink_if_identity(parent, temporary_name, temporary_identity)
            except FileNotFoundError:
                pass
        os.close(parent)


def evaluate_temporal_occlusion_package(*, targets: str | Path, checkpoints: Sequence[str | Path], dataset_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    target_path = Path(targets).absolute()
    dataset = Path(dataset_root).absolute()
    if not dataset.is_dir() or dataset.is_symlink(): raise ValueError("dataset_root must be a real directory")
    arrays, metadata, target_record, witnesses, sources, source_frame_times = _load_target(target_path, dataset)
    if not checkpoints: raise ValueError("at least one compact index is required")
    bindings: dict[tuple[str, int], _CompactBinding] = {}
    relative_times: dict[tuple[str, int], int] = {}
    indexes = []
    index_records = []
    telemetry_by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    mechanism_source_indexes: list[dict[str, Any]] = []
    cross_scene_authority = (
        "schema_version", "format", "protocol_id", "dataset", "method_id",
        "algorithm_hash", "schedule", "target_manifest", "input_sha256",
        "code_commit", "source_bindings",
    )
    for candidate in checkpoints:
        values, times, index, index_record, mechanism_telemetry, mechanism_source_index, index_witnesses = _load_index(
            Path(candidate).absolute(), target_record, metadata, sources, source_frame_times
        )
        if indexes and any(
            indexes[0].get(key) != index.get(key) for key in cross_scene_authority
        ):
            raise ValueError("cross-scene authority mismatch")
        if set(bindings) & set(values): raise ValueError("duplicate checkpoint indexes")
        bindings.update(values); indexes.append(index); index_records.append(index_record)
        telemetry_by_scene[index["scene"]] = mechanism_telemetry
        mechanism_source_indexes.append(mechanism_source_index)
        witnesses.extend(index_witnesses)
        relative_times.update(times)
    loaded = _LazyCheckpoints(bindings)
    results = [evaluate_temporal_occlusion(arrays=arrays, metadata=metadata, checkpoints=loaded,
               checkpoint_relative_timestamp_ns=relative_times, scene=index["scene"],
               mechanism_telemetry=telemetry_by_scene[index["scene"]]) for index in indexes]
    result = results[0] if len(results) == 1 else {"format": EVALUATION_FORMAT, "scenes": results}
    result["input_bindings"] = {"target_manifest": dict(target_record),
                                "indexes": index_records,
                                "source_indexes": mechanism_source_indexes,
                                "maximum_cached_checkpoints": loaded.maximum_cached_checkpoints}
    loaded.revalidate_all()
    _revalidate(witnesses)
    rendered = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if output is not None: _publish(Path(output).absolute(), rendered)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path, nargs="+")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_temporal_occlusion_package(targets=args.targets, checkpoints=args.checkpoints, dataset_root=args.dataset_root, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
