#!/usr/bin/env python3
"""Build verified source bundles for the Apartment A0-A4 search preflight."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import contextvars
import ctypes
import csv
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
from typing import Any, Sequence

import numpy as np

from scripts.evaluation.verify_oviv2_dual_readout_development_gates import (
    verify_exact_profile_runs,
    verify_source_manifest,
)
from scripts.evaluation.evaluate_oviv2_tesse_temporal_occlusion import (
    evaluate_temporal_occlusion_package,
)
from scripts.evaluation.build_oviv2_tesse_search_preflight import (
    _anchor_coverage,
    build_preflight,
)
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    _materialize_config,
    _revalidate_preflight_witnesses,
    _validate_preflight_gate_evidence,
)


_EXACT_SEQUENCE = ("reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4")
_CANDIDATE_POSITIONS = {"a0": 1, "a1": 2, "a2": 4, "a3": 6, "a4": 8}
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUSTED_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
)
_AUDIT_WITNESSES: contextvars.ContextVar[list[_FileWitness] | None] = contextvars.ContextVar(
    "oviv2_preflight_audit_witnesses", default=None
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    identity: tuple[int, int, int, int, int]

    def revalidate(self) -> None:
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != self.identity:
            raise ValueError(f"source changed before publication: {self.path}")


@dataclass(frozen=True)
class _FileWitness:
    path: Path
    identity: tuple[int, int, int, int, int]

    def revalidate(self) -> None:
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != self.identity:
            raise ValueError(f"source changed before publication: {self.path}")


@dataclass(frozen=True)
class _DirectoryWitness:
    path: Path
    identity: tuple[int, int]

    def revalidate(self) -> None:
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != self.identity:
            raise ValueError(f"source directory changed before publication: {self.path}")


@dataclass(frozen=True)
class _ExactTransactionWitness:
    executions: tuple[dict[str, Any], ...]
    expected: bytes

    def revalidate(self) -> None:
        reopened = verify_exact_profile_runs([dict(item) for item in self.executions])
        if _canonical(reopened) != self.expected:
            raise ValueError("development exact transaction changed before publication")


def _code_identity(repo: Path) -> tuple[str, str]:
    values: list[str] = []
    for revision in ("HEAD", "HEAD^{tree}"):
        completed = subprocess.run(
            ("git", "rev-parse", revision), cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        value = completed.stdout.decode("ascii").strip()
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("git returned an invalid code identity")
        values.append(value)
    return values[0], values[1]


@dataclass(frozen=True)
class _DevelopmentSourcesWitness:
    manifest: dict[str, Any]
    protected: tuple[dict[str, Any], ...]
    commit: str
    tree: str
    repo: Path
    verifier: Callable[..., list[dict[str, Any]]]
    code_identity: Callable[[Path], tuple[str, str]]

    def revalidate(self) -> None:
        current = self.verifier(dict(self.manifest), repo=self.repo)
        if current != [dict(item) for item in self.protected]:
            raise ValueError("development protected source records changed")
        if self.code_identity(self.repo) != (self.commit, self.tree):
            raise ValueError("development code commit/tree changed")


def _verify_development_sources(
    deterministic: Mapping[str, Any],
    source_manifest_snapshot: _Snapshot,
    *,
    repo: Path = _REPO_ROOT,
    verifier: Callable[..., list[dict[str, Any]]] = verify_source_manifest,
    code_identity: Callable[[Path], tuple[str, str]] = _code_identity,
) -> _DevelopmentSourcesWitness:
    manifest = _decode_json(source_manifest_snapshot.data, "development source manifest")
    protected = deterministic.get("protected_files")
    commit = deterministic.get("code_commit")
    tree = deterministic.get("code_tree")
    if not isinstance(protected, list) or any(not isinstance(item, dict) for item in protected):
        raise ValueError("development protected source records are invalid")
    if not isinstance(commit, str) or not isinstance(tree, str):
        raise ValueError("development code commit/tree is invalid")
    witness = _DevelopmentSourcesWitness(
        manifest, tuple(dict(item) for item in protected), commit, tree,
        repo, verifier, code_identity,
    )
    witness.revalidate()
    return witness


@dataclass(frozen=True)
class PreservedArtifact:
    name: str
    logical_path: str
    parent_device: int | None
    parent_inode: int | None
    artifact_device: int | None
    artifact_inode: int | None
    st_mode: int | None
    ownership: str


class PreflightPublicationUncertain(RuntimeError):
    def __init__(self, preserved: Sequence[PreservedArtifact]):
        self.preserved = tuple(preserved)
        detail = ", ".join(
            f"{item.logical_path}[{item.ownership},dev={item.artifact_device},ino={item.artifact_inode},mode={item.st_mode}]"
            for item in self.preserved
        ) or "identity unavailable"
        super().__init__(f"preflight publication uncertain; preserved: {detail}")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} path contains a symlink")


def _snapshot(path: str | Path, label: str) -> _Snapshot:
    absolute = Path(path).absolute()
    _reject_symlink_components(absolute, label)
    before_path = os.lstat(absolute)
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"{label} must be a regular file")
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_SOURCE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_SOURCE_BYTES:
                raise ValueError(f"{label} exceeds size limit")
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(absolute, follow_symlinks=False)
    if not (_identity(before_path) == _identity(opened) == _identity(finished) == _identity(current)):
        raise ValueError(f"{label} changed while reading")
    data = b"".join(chunks)
    if len(data) != finished.st_size:
        raise ValueError(f"{label} changed size while reading")
    snapshot = _Snapshot(absolute, data, _identity(finished))
    captured = _AUDIT_WITNESSES.get()
    if captured is not None:
        captured.append(_FileWitness(snapshot.path, snapshot.identity))
    return snapshot


def _decode_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line:
            raise ValueError(f"{label} contains blank line {line_number}")
        rows.append(_decode_json(line, f"{label} line {line_number}"))
    return rows


def _relative_path(record: object, root: Path, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} record schema is not exact")
    raw = record.get("path")
    relative = Path(raw) if isinstance(raw, str) else Path(".")
    if not isinstance(raw, str) or not raw or relative.is_absolute() or relative == Path(".") or ".." in relative.parts or raw != relative.as_posix():
        raise ValueError(f"{label} path is unsafe")
    path = (root / relative).absolute()
    try:
        path.relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes run root") from exc
    return path


def _bound_snapshot(record: object, root: Path, label: str) -> _Snapshot:
    path = _relative_path(record, root, label)
    snapshot = _snapshot(path, label)
    assert isinstance(record, Mapping)
    if record.get("sha256") != hashlib.sha256(snapshot.data).hexdigest() or record.get("byte_count") != len(snapshot.data):
        raise ValueError(f"{label} source binding mismatch")
    return snapshot


def _absolute_bound_snapshot(record: object, label: str) -> _Snapshot:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} record schema is not exact")
    raw = record.get("path")
    if not isinstance(raw, str) or not Path(raw).is_absolute() or raw != str(Path(raw).absolute()) or ".." in Path(raw).parts:
        raise ValueError(f"{label} path is not canonical absolute")
    snapshot = _snapshot(Path(raw), label)
    if record.get("sha256") != hashlib.sha256(snapshot.data).hexdigest() or record.get("byte_count") != len(snapshot.data):
        raise ValueError(f"{label} source binding mismatch")
    return snapshot


def _record(path: Path, data: bytes, *, relative_to: Path) -> dict[str, Any]:
    relative = path.relative_to(relative_to)
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError("output record path is unsafe")
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _mkdirs_at(root_fd: int, relative: Path) -> None:
    current = os.dup(root_fd)
    try:
        for part in relative.parts:
            try:
                os.mkdir(part, 0o755, dir_fd=current)
            except FileExistsError:
                pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = next_fd
        os.fsync(current)
    finally:
        os.close(current)


def _write_bytes_at(root_fd: int, relative: Path, data: bytes) -> None:
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError("staged output path is unsafe")
    _mkdirs_at(root_fd, relative.parent)
    descriptor = os.open(
        relative.as_posix(), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o644, dir_fd=root_fd,
    )
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("staged output write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_at(root_fd: int, relative: Path, value: object) -> bytes:
    data = _canonical(value) + b"\n"
    _write_bytes_at(root_fd, relative, data)
    return data


def _preserved(parent_fd: int, parent: Path, name: str, logical: str, expected: tuple[int, int] | None, owned: bool) -> PreservedArtifact:
    try:
        parent_status = os.fstat(parent_fd)
        parent_identity: tuple[int | None, int | None] = (parent_status.st_dev, parent_status.st_ino)
    except OSError:
        parent_identity = (None, None)
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (status.st_dev, status.st_ino)
        ownership = "owned" if owned and expected == identity else "unknown"
        return PreservedArtifact(name, logical, parent_identity[0], parent_identity[1], status.st_dev, status.st_ino, status.st_mode, ownership)
    except OSError:
        return PreservedArtifact(name, logical, parent_identity[0], parent_identity[1], None, None, None, "unbound")


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), destination)


def _root_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError("staged bundle contains a symlink")
        data = _snapshot(path, "staged bundle member").data
        records.append(_record(path, data, relative_to=root))
    return records, hashlib.sha256(_canonical(records)).hexdigest()


@dataclass(frozen=True)
class _StagingTreeWitness:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    inventory: tuple[dict[str, Any], ...]
    root_sha256: str

    @classmethod
    def capture(cls, path: Path, descriptor: int) -> _StagingTreeWitness:
        opened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise ValueError("staged bundle must be a directory")
        identity = (opened.st_dev, opened.st_ino)
        if identity != (current.st_dev, current.st_ino):
            raise ValueError("staged bundle directory changed")
        inventory, digest = _root_inventory(path)
        return cls(path, descriptor, identity, tuple(inventory), digest)

    def revalidate(self) -> None:
        opened = os.fstat(self.descriptor)
        current = os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (opened.st_dev, opened.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise ValueError("staged bundle changed before publication")
        inventory, digest = _root_inventory(self.path)
        if inventory != list(self.inventory) or digest != self.root_sha256:
            raise ValueError("staged bundle changed before publication")


def _fsync_staged_tree(root: Path) -> None:
    files = sorted(item for item in root.rglob("*") if item.is_file())
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in files:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("staged bundle member must be a regular file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in (*directories, root):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _tree_binding(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint artifact tree is empty")
    for item in files:
        if item.is_symlink():
            raise ValueError("checkpoint artifact tree contains symlink")
        data = _snapshot(item, "checkpoint artifact member").data
        relative = item.relative_to(path).as_posix()
        byte_count += len(data)
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest()); digest.update(b"\n")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "byte_count": byte_count}


def _violation(candidate_id: str, frame: object, role: str, observed: object, bound: object) -> ValueError:
    return ValueError(
        f"future leakage violation candidate={candidate_id} frame={frame} role={role} "
        f"observed={observed!r} bound={bound!r}"
    )


def _audit_future_leakage(
    *,
    candidate_id: str,
    run_root: Path,
    run: Mapping[str, Any],
    source_index: Mapping[str, Any],
    checkpoint_index: Mapping[str, Any],
) -> dict[str, Any]:
    processed = run.get("processed_frame_count")
    expected = list(range(processed)) if type(processed) is int and processed > 0 else []
    if not expected or (run.get("covered_frame_count"), run.get("first_frame_index"), run.get("last_frame_index")) != (processed, 0, processed - 1):
        raise _violation(candidate_id, None, "frame_coverage.summary", (run.get("covered_frame_count"), run.get("first_frame_index"), run.get("last_frame_index")), (processed, 0, processed - 1))
    coverage_snapshot = _bound_snapshot(source_index.get("frame_coverage"), run_root, "frame coverage")
    coverage = _jsonl(coverage_snapshot.data, "frame coverage")
    observed = [row.get("frame_index") for row in coverage]
    if observed != expected:
        mismatch = next((frame for frame, value in enumerate(observed) if value != frame), len(observed))
        raise _violation(candidate_id, mismatch, "frame_coverage.frame_index", observed, expected)
    timestamps = [row.get("timestamp_ns") for row in coverage]
    if any(type(value) is not int or value < 0 for value in timestamps):
        raise _violation(candidate_id, None, "frame_coverage.timestamp_ns", timestamps, "nonnegative integers")
    trajectories_record = source_index.get("trajectories")
    if trajectories_record is not None:
        trajectories = _jsonl(_bound_snapshot(trajectories_record, run_root, "trajectories").data, "trajectories")
        trajectory_timestamps: dict[int, set[int]] = {}
        for row in trajectories:
            trajectory_timestamps.setdefault(row.get("frame_index"), set()).add(row.get("timestamp_ns"))
        for frame, observed_timestamps in trajectory_timestamps.items():
            bound = timestamps[frame] if type(frame) is int and frame in expected else "processed frame"
            if observed_timestamps != {bound}:
                raise _violation(candidate_id, frame, "trajectories.timestamp_ns", sorted(observed_timestamps), bound)

    config_snapshot = _bound_snapshot(run.get("normalized_run_config"), run_root, "normalized run config")
    config = _decode_json(config_snapshot.data, "normalized run config")
    raw_dataset_root = config.get("dataset_root")
    if isinstance(raw_dataset_root, str) and Path(raw_dataset_root).is_absolute():
        dataset_root = Path(raw_dataset_root)
        scene_root = dataset_root if (dataset_root / "results").is_dir() else dataset_root / "apartment"
        timestamp_snapshot = _snapshot(scene_root / "timestamps.csv", "dataset timestamps")
        timestamp_binding = run.get("source_bindings", {}).get("timestamps") if isinstance(run.get("source_bindings"), Mapping) else None
        expected_timestamp_binding = {
            "sha256": hashlib.sha256(timestamp_snapshot.data).hexdigest(),
            "byte_count": len(timestamp_snapshot.data),
        }
        if timestamp_binding != expected_timestamp_binding:
            raise _violation(candidate_id, None, "source_bindings.timestamps", timestamp_binding, expected_timestamp_binding)
        reader = csv.DictReader(io.StringIO(timestamp_snapshot.data.decode("utf-8")))
        if reader.fieldnames != ["frame_index", "sensor_timestamp_ns", "relative_timestamp_ns"]:
            raise _violation(candidate_id, None, "dataset_timestamps.header", reader.fieldnames, ["frame_index", "sensor_timestamp_ns", "relative_timestamp_ns"])
        dataset_timestamps: list[int] = []
        for expected_frame, row in enumerate(reader):
            try:
                frame = int(row["frame_index"])
                timestamp = int(row["sensor_timestamp_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _violation(candidate_id, expected_frame, "dataset_timestamps.row", row, "integer frame/timestamp") from exc
            if frame != expected_frame:
                raise _violation(candidate_id, expected_frame, "dataset_timestamps.frame_index", frame, expected_frame)
            dataset_timestamps.append(timestamp)
        if len(dataset_timestamps) < processed:
            raise _violation(candidate_id, len(dataset_timestamps), "dataset_timestamps.count", len(dataset_timestamps), processed)
        for frame, timestamp in enumerate(timestamps):
            if timestamp != dataset_timestamps[frame]:
                raise _violation(candidate_id, frame, "frame_coverage.timestamp_ns", timestamp, dataset_timestamps[frame])
    for role, suffix in (("frontend_manifest", "pkl.gz"), ("dense_manifest", "npz")):
        raw_manifest = config.get(role)
        if not isinstance(raw_manifest, str) or not Path(raw_manifest).is_absolute():
            raise _violation(candidate_id, None, role, raw_manifest, "absolute manifest path")
        manifest_path = Path(raw_manifest)
        manifest_snapshot = _snapshot(manifest_path, role)
        manifest = _decode_json(manifest_snapshot.data, role)
        source_binding = run.get("source_bindings", {}).get(role) if isinstance(run.get("source_bindings"), Mapping) else None
        expected_binding = {
            "sha256": hashlib.sha256(manifest_snapshot.data).hexdigest(),
            "byte_count": len(manifest_snapshot.data),
        }
        if source_binding != expected_binding:
            raise _violation(candidate_id, None, f"source_bindings.{role}", source_binding, expected_binding)
        if manifest.get("frame_count") != processed or manifest.get("source_frame_ids") != expected:
            raise _violation(candidate_id, None, f"{role}.source_frame_ids", manifest.get("source_frame_ids"), expected)
        hashes = manifest.get("cache_files_sha256")
        expected_names = [f"frame{frame:06d}.{suffix}" for frame in expected]
        if not isinstance(hashes, Mapping) or list(hashes) != expected_names:
            raise _violation(candidate_id, None, f"{role}.cache_inventory", list(hashes) if isinstance(hashes, Mapping) else hashes, expected_names)
        for frame, name in enumerate(expected_names):
            cache = _snapshot(manifest_path.parent / name, f"{role} cache frame {frame}")
            digest = hashlib.sha256(cache.data).hexdigest()
            if hashes[name] != digest:
                raise _violation(candidate_id, frame, f"{role}.cache_sha256", hashes[name], digest)
            if role == "dense_manifest":
                try:
                    with np.load(io.BytesIO(cache.data), allow_pickle=False) as archive:
                        cache_frame = int(np.asarray(archive["cache_frame_id"]).item())
                        source_frame = int(np.asarray(archive["source_frame_id"]).item())
                except (KeyError, ValueError, OSError) as exc:
                    raise _violation(candidate_id, frame, "dense_cache.metadata", type(exc).__name__, "cache_frame_id/source_frame_id") from exc
                if cache_frame != frame or source_frame > frame or source_frame != manifest["source_frame_ids"][frame]:
                    raise _violation(candidate_id, frame, "dense_cache.source_frame_id", (cache_frame, source_frame), (frame, f"<= {frame}"))

    for field in ("algorithm_hash", "schedule", "target_manifest", "input_sha256", "code_commit", "source_bindings"):
        if checkpoint_index.get(field) != run.get(field):
            raise _violation(candidate_id, None, f"checkpoint_index.{field}", checkpoint_index.get(field), run.get(field))
    checkpoints = checkpoint_index.get("checkpoints")
    if not isinstance(checkpoints, list) or any(not isinstance(item, Mapping) for item in checkpoints):
        raise _violation(candidate_id, None, "checkpoint_index.checkpoints", checkpoints, "checkpoint records")
    checkpoint_frames = [item.get("frame_index") for item in checkpoints]
    if (
        not checkpoint_frames
        or any(type(frame) is not int or frame not in expected for frame in checkpoint_frames)
        or checkpoint_frames != sorted(set(checkpoint_frames))
    ):
        raise _violation(candidate_id, None, "checkpoint_index.frames", checkpoint_frames, "unique increasing processed frames")
    run_checkpoints = {item.get("frame_index"): item for item in run.get("checkpoints", []) if isinstance(item, Mapping)}
    source_checkpoints = {item.get("frame_index"): item for item in source_index.get("checkpoints", []) if isinstance(item, Mapping)}
    for checkpoint in checkpoints:
        frame = checkpoint.get("frame_index")
        timestamp = checkpoint.get("timestamp_ns")
        if type(frame) is not int or frame not in expected or timestamp != timestamps[frame]:
            raise _violation(candidate_id, frame, "checkpoint_index.timestamp_ns", timestamp, timestamps[frame] if type(frame) is int and frame in expected else "processed frame")
        for field, bound in (("consumed_through_frame", frame), ("consumed_through_frame_exclusive", frame + 1)):
            if checkpoint.get(field) != bound:
                raise _violation(candidate_id, frame, f"checkpoint_index.{field}", checkpoint.get(field), bound)
        run_record = run_checkpoints.get(frame)
        if not isinstance(run_record, Mapping):
            raise _violation(candidate_id, frame, "run_manifest.checkpoint", run_record, "present")
        inventories = [("run_manifest", run_record)]
        source_record = source_checkpoints.get(frame)
        if source_record is not None:
            if not isinstance(source_record, Mapping):
                raise _violation(candidate_id, frame, "source_index.checkpoint", source_record, "checkpoint record")
            inventories.append(("source_index", source_record))
        for inventory_role, record in inventories:
            for field in ("timestamp_ns", "consumed_through_frame", "consumed_through_frame_exclusive"):
                if record.get(field) != checkpoint.get(field):
                    raise _violation(candidate_id, frame, f"{inventory_role}.{field}", record.get(field), checkpoint.get(field))
        status_record = run_record.get("checkpoint_status")
        if source_record is not None and source_record.get("checkpoint_status") != status_record:
            raise _violation(candidate_id, frame, "source_index.checkpoint_status", source_record.get("checkpoint_status"), status_record)
        status = _decode_json(_bound_snapshot(status_record, run_root, "checkpoint status").data, "checkpoint status")
        for field, bound in (("checkpoint_frame", frame), ("timestamp_ns", timestamp), ("consumed_through_frame", frame), ("consumed_through_frame_exclusive", frame + 1)):
            if status.get(field) != bound:
                raise _violation(candidate_id, frame, f"checkpoint_status.{field}", status.get(field), bound)
        artifact = checkpoint.get("artifact")
        artifact_path = _relative_path(artifact, run_root, "compact artifact")
        if not artifact_path.is_dir() or _tree_binding(artifact_path, run_root) != artifact:
            raise _violation(candidate_id, frame, "compact.artifact", artifact, _tree_binding(artifact_path, run_root) if artifact_path.is_dir() else "directory")
        checksums_snapshot = _snapshot(artifact_path / "checksums.json", "compact checksums")
        if checkpoint.get("checksums_sha256") != hashlib.sha256(checksums_snapshot.data).hexdigest():
            raise _violation(candidate_id, frame, "compact.checksums_sha256", checkpoint.get("checksums_sha256"), hashlib.sha256(checksums_snapshot.data).hexdigest())
        checksums = _decode_json(checksums_snapshot.data, "compact checksums")
        metadata_snapshot = _snapshot(artifact_path / "metadata.json", "compact metadata")
        if checksums.get("metadata.json") != hashlib.sha256(metadata_snapshot.data).hexdigest():
            raise _violation(candidate_id, frame, "compact.metadata.sha256", checksums.get("metadata.json"), hashlib.sha256(metadata_snapshot.data).hexdigest())
        metadata = _decode_json(metadata_snapshot.data, "compact metadata")
        for field, bound in (("frame_id", frame), ("revision", frame + 1), ("timestamp", timestamp / 1_000_000_000)):
            if metadata.get(field) != bound:
                raise _violation(candidate_id, frame, f"compact.metadata.{field}", metadata.get(field), bound)
        run_compact = run_record.get("artifacts", {}).get("temporal_compact") if isinstance(run_record.get("artifacts"), Mapping) else None
        expected_compact = {"artifact": artifact, "checksums_sha256": checkpoint.get("checksums_sha256")}
        if not isinstance(run_compact, Mapping) or {
            "artifact": run_compact.get("artifact"),
            "checksums_sha256": run_compact.get("checksums_sha256"),
        } != expected_compact:
            raise _violation(candidate_id, frame, "run_manifest.temporal_compact", run_compact, expected_compact)
    return {
        "schema_version": 1, "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
        "scene": "apartment", "candidate_id": candidate_id,
        "source_index": run["source_index"], "records": [],
    }


def _select_candidate_executions(
    exact: Mapping[str, Any],
    *,
    verifier: Callable[[list[dict[str, object]]], Mapping[str, Any]] = verify_exact_profile_runs,
) -> dict[str, Mapping[str, Any]]:
    executions = exact.get("executions")
    if (
        exact.get("format") != "oviv2_t1_exact_transaction_v1"
        or exact.get("sequence") != list(_EXACT_SEQUENCE)
        or not isinstance(executions, list)
        or len(executions) != len(_EXACT_SEQUENCE)
        or any(not isinstance(item, Mapping) for item in executions)
        or [item.get("profile") for item in executions] != list(_EXACT_SEQUENCE)
    ):
        raise ValueError("development exact transaction identity is invalid")
    copied = [dict(item) for item in executions]
    reopened = verifier(copied)
    if _canonical(reopened) != _canonical(dict(exact)):
        raise ValueError("development exact transaction differs after re-verification")
    return {
        candidate_id: executions[position]
        for candidate_id, position in _CANDIDATE_POSITIONS.items()
    }


def build_preflight_sources(
    *,
    development_evidence: str | Path,
    search_manifest: str | Path,
    apartment_base_config: str | Path,
    targets: str | Path,
    dataset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    output_path = Path(output).absolute()
    parent = output_path.parent
    _reject_symlink_components(parent, "output parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent, "output parent")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    stage_fd = -1
    stage_name = f".{output_path.name}.staging-{secrets.token_hex(12)}"
    stage_path = parent / stage_name
    stage_identity: tuple[int, int] | None = None
    renamed = False
    rename_collision = False
    witnesses: list[
        _Snapshot | _FileWitness | _DirectoryWitness | _ExactTransactionWitness
        | _DevelopmentSourcesWitness
    ] = []
    try:
        parent_status = os.fstat(parent_fd)
        try:
            os.stat(output_path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"output already exists: {output_path}")

        evidence_snapshot = _snapshot(development_evidence, "development evidence")
        manifest_snapshot = _snapshot(search_manifest, "search manifest")
        base_snapshot = _snapshot(apartment_base_config, "Apartment base config")
        targets_snapshot = _snapshot(targets, "occlusion targets")
        dataset_path = Path(dataset_root).absolute()
        _reject_symlink_components(dataset_path, "dataset root")
        dataset_status = os.stat(dataset_path, follow_symlinks=False)
        if not stat.S_ISDIR(dataset_status.st_mode):
            raise ValueError("dataset root must be a directory")
        witnesses.extend((evidence_snapshot, manifest_snapshot, base_snapshot, targets_snapshot))
        witnesses.append(_DirectoryWitness(dataset_path, (dataset_status.st_dev, dataset_status.st_ino)))
        evidence = _decode_json(evidence_snapshot.data, "development evidence")
        manifest = _decode_json(manifest_snapshot.data, "search manifest")
        base_config = _decode_json(base_snapshot.data, "Apartment base config")
        if set(evidence) != {"schema_version", "manifest_id", "deterministic_evidence", "receipt"} or evidence.get("schema_version") != 1 or evidence.get("manifest_id") != "oviv2_dual_readout_development_gates_v1":
            raise ValueError("development evidence identity/schema is invalid")
        deterministic = evidence.get("deterministic_evidence")
        expected_deterministic = {"base_commit", "code_commit", "code_tree", "protected_files", "test_sources", "source_manifest", "cumulative_exact", "gates"}
        if not isinstance(deterministic, Mapping) or set(deterministic) != expected_deterministic:
            raise ValueError("development deterministic evidence schema is invalid")
        code_commit = deterministic.get("code_commit")
        if not isinstance(code_commit, str) or len(code_commit) != 40:
            raise ValueError("development code commit is invalid")
        source_manifest_snapshot = _absolute_bound_snapshot(deterministic.get("source_manifest"), "development source manifest")
        if source_manifest_snapshot.path != _TRUSTED_SOURCE_MANIFEST.absolute():
            raise ValueError("development source manifest identity mismatch")
        witnesses.append(source_manifest_snapshot)
        witnesses.append(
            _verify_development_sources(deterministic, source_manifest_snapshot)
        )
        source_manifest_sha = hashlib.sha256(source_manifest_snapshot.data).hexdigest()
        exact = deterministic.get("cumulative_exact")
        if not isinstance(exact, Mapping):
            raise ValueError("development exact transaction is missing")
        selected = _select_candidate_executions(exact, verifier=verify_exact_profile_runs)
        exact_executions = exact.get("executions")
        assert isinstance(exact_executions, list)
        witnesses.append(
            _ExactTransactionWitness(
                tuple(dict(item) for item in exact_executions), _canonical(dict(exact))
            )
        )
        declarations_list = manifest.get("candidates")
        if not isinstance(declarations_list, list) or [item.get("candidate_id") for item in declarations_list if isinstance(item, Mapping)][:5] != list(_CANDIDATE_POSITIONS):
            raise ValueError("search manifest A0-A4 declaration order is invalid")
        declarations = {item["candidate_id"]: item for item in declarations_list if isinstance(item, Mapping)}

        candidate_sources: list[dict[str, Any]] = []
        source_inodes: set[tuple[int, int]] = set()
        os.mkdir(stage_name, 0o755, dir_fd=parent_fd)
        stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        stage_status = os.fstat(stage_fd)
        stage_identity = (stage_status.st_dev, stage_status.st_ino)
        os.fsync(parent_fd)

        for candidate_id in _CANDIDATE_POSITIONS:
            execution = selected[candidate_id]
            if execution.get("profile") != candidate_id or execution.get("code_commit") != code_commit or execution.get("source_manifest_sha256") != source_manifest_sha:
                raise ValueError(f"{candidate_id} execution development binding mismatch")
            root_raw = execution.get("output_root")
            if not isinstance(root_raw, str) or not Path(root_raw).is_absolute() or root_raw != str(Path(root_raw).absolute()):
                raise ValueError(f"{candidate_id} execution root is invalid")
            run_root = Path(root_raw)
            root_status = os.stat(run_root, follow_symlinks=False)
            if not stat.S_ISDIR(root_status.st_mode):
                raise ValueError(f"{candidate_id} execution root is not a directory")
            witnesses.append(_DirectoryWitness(run_root, (root_status.st_dev, root_status.st_ino)))
            observation_snapshot = _absolute_bound_snapshot(execution.get("observation_receipt"), f"{candidate_id} observation receipt")
            witnesses.append(observation_snapshot)
            observation = _decode_json(observation_snapshot.data, f"{candidate_id} observation receipt")
            if observation.get("profile") != candidate_id or observation.get("output_root") != str(run_root):
                raise ValueError(f"{candidate_id} observation identity mismatch")
            if observation.get("source_manifest") != deterministic["source_manifest"]:
                raise ValueError(f"{candidate_id} observation source manifest mismatch")
            run_snapshot = _absolute_bound_snapshot(observation.get("run_manifest"), f"{candidate_id} run manifest")
            try:
                run_relative = run_snapshot.path.relative_to(run_root)
            except ValueError as exc:
                raise ValueError(f"{candidate_id} run manifest escapes execution root") from exc
            if run_relative != Path("run_manifest.json"):
                raise ValueError(f"{candidate_id} run manifest path is not canonical")
            run = _decode_json(run_snapshot.data, f"{candidate_id} run manifest")
            config_snapshot = _absolute_bound_snapshot(observation.get("config"), f"{candidate_id} materialized config")
            materialized = _materialize_config(base_config, declarations[candidate_id])
            if _decode_json(config_snapshot.data, f"{candidate_id} materialized config") != materialized:
                raise ValueError(f"{candidate_id} materialized config mismatch")
            if run.get("algorithm_hash") != materialized.get("algorithm_hash") or run.get("code_commit") != code_commit:
                raise ValueError(f"{candidate_id} run config/code binding mismatch")
            configured_dataset = materialized.get("dataset_root")
            if configured_dataset is not None and Path(str(configured_dataset)).absolute() != dataset_path:
                raise ValueError(f"{candidate_id} dataset root differs from requested root")
            witnesses.extend((run_snapshot, config_snapshot))
            source_snapshot = _bound_snapshot(run.get("source_index"), run_root, f"{candidate_id} source index")
            source_index = _decode_json(source_snapshot.data, f"{candidate_id} source index")
            witnesses.append(source_snapshot)
            required_snapshots: list[tuple[Path, _Snapshot]] = [(run_relative, run_snapshot)]
            source_relative = source_snapshot.path.relative_to(run_root)
            required_snapshots.append((source_relative, source_snapshot))
            for role in ("trajectories", "lifecycle_transitions", "frame_coverage", "runtime_diagnostics"):
                role_snapshot = _bound_snapshot(source_index.get(role), source_snapshot.path.parent, f"{candidate_id} {role}")
                try:
                    role_relative = role_snapshot.path.relative_to(run_root)
                except ValueError as exc:
                    raise ValueError(f"{candidate_id} {role} escapes execution root") from exc
                required_snapshots.append((role_relative, role_snapshot))
                witnesses.append(role_snapshot)
            if len({relative for relative, _ in required_snapshots}) != len(required_snapshots):
                raise ValueError(f"{candidate_id} required source paths collide")
            for _, snapshot in required_snapshots:
                inode = snapshot.identity[:2]
                if inode in source_inodes:
                    raise ValueError("required source paths alias the same inode")
                source_inodes.add(inode)
            for relative, snapshot in required_snapshots:
                _write_bytes_at(stage_fd, Path(candidate_id) / relative, snapshot.data)
                copied = _snapshot(stage_path / candidate_id / relative, f"copied {candidate_id} source")
                if copied.data != snapshot.data:
                    raise ValueError(f"{candidate_id} copied source digest mismatch")

            checkpoint_snapshot = _bound_snapshot(run.get("occlusion_checkpoint_index"), run_root, f"{candidate_id} occlusion checkpoint index")
            if checkpoint_snapshot.identity[:2] in source_inodes:
                raise ValueError("required source paths alias the same inode")
            source_inodes.add(checkpoint_snapshot.identity[:2])
            checkpoint_index = _decode_json(checkpoint_snapshot.data, f"{candidate_id} occlusion checkpoint index")
            witnesses.append(checkpoint_snapshot)
            audit_witnesses: list[_FileWitness] = []
            audit_token = _AUDIT_WITNESSES.set(audit_witnesses)
            try:
                leakage = _audit_future_leakage(
                    candidate_id=candidate_id, run_root=run_root, run=run,
                    source_index=source_index, checkpoint_index=checkpoint_index,
                )
            finally:
                _AUDIT_WITNESSES.reset(audit_token)
            witnesses.extend(audit_witnesses)
            leakage_relative = Path(candidate_id) / "future_leakage.json"
            leakage_data = _write_json_at(stage_fd, leakage_relative, leakage)
            occlusion_relative = Path(candidate_id) / "temporal_occlusion_result.json"
            occlusion_result = evaluate_temporal_occlusion_package(
                targets=targets_snapshot.path, checkpoints=[checkpoint_snapshot.path],
                dataset_root=dataset_path, output=stage_path / occlusion_relative,
            )
            anchor = _anchor_coverage(occlusion_result)
            if anchor.get("eligible_count") != 66 or anchor.get("mapped_count", 0) < 53:
                raise ValueError(f"{candidate_id} Apartment anchor coverage requires at least 53/66")
            occlusion_data = _snapshot(stage_path / occlusion_relative, f"{candidate_id} temporal occlusion result").data
            run_copy = stage_path / candidate_id / run_relative
            candidate_sources.append({
                "candidate_id": candidate_id,
                "run_manifest": _record(run_copy, run_snapshot.data, relative_to=stage_path),
                "temporal_occlusion_result": _record(stage_path / occlusion_relative, occlusion_data, relative_to=stage_path),
                "future_leakage_evidence": _record(stage_path / leakage_relative, leakage_data, relative_to=stage_path),
            })

        sources_payload = {"schema_version": 1, "manifest_id": "oviv2_tesse_search_preflight_sources_v1", "candidates": candidate_sources}
        _write_json_at(stage_fd, Path("candidate_sources.json"), sources_payload)
        preflight_path = stage_path / "preflight.json"
        build_preflight(
            search_manifest=manifest_snapshot.path, apartment_base_config=base_snapshot.path,
            candidate_sources=stage_path / "candidate_sources.json", output=preflight_path,
        )
        preflight_record = _validate_preflight_gate_evidence(
            preflight_path, manifest_bytes=manifest_snapshot.data, apartment_bytes=base_snapshot.data,
            apartment=base_config, declarations=declarations, selected_ids=tuple(_CANDIDATE_POSITIONS),
        )
        inventory, root_digest = _root_inventory(stage_path)
        receipt = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_search_preflight_publication_receipt_v1",
            "inputs": {
                "development_evidence": {"path": str(evidence_snapshot.path), "sha256": hashlib.sha256(evidence_snapshot.data).hexdigest(), "byte_count": len(evidence_snapshot.data)},
                "search_manifest": {"path": str(manifest_snapshot.path), "sha256": hashlib.sha256(manifest_snapshot.data).hexdigest(), "byte_count": len(manifest_snapshot.data)},
                "apartment_base_config": {"path": str(base_snapshot.path), "sha256": hashlib.sha256(base_snapshot.data).hexdigest(), "byte_count": len(base_snapshot.data)},
                "targets": {"path": str(targets_snapshot.path), "sha256": hashlib.sha256(targets_snapshot.data).hexdigest(), "byte_count": len(targets_snapshot.data)},
                "dataset_root": {"path": str(dataset_path), "device": dataset_status.st_dev, "inode": dataset_status.st_ino},
            },
            "output_inventory": inventory, "output_root_sha256": root_digest,
        }
        _write_json_at(stage_fd, Path("publication_receipt.json"), receipt)
        _fsync_staged_tree(stage_path)
        staging_witness = _StagingTreeWitness.capture(stage_path, stage_fd)
        current_parent = os.stat(parent, follow_symlinks=False)
        if (current_parent.st_dev, current_parent.st_ino) != (parent_status.st_dev, parent_status.st_ino):
            raise ValueError("output parent changed before publication")
        for witness in witnesses:
            witness.revalidate()
        _revalidate_preflight_witnesses(preflight_record)
        staging_witness.revalidate()
        try:
            _rename_noreplace(parent_fd, stage_name, output_path.name)
        except FileExistsError:
            rename_collision = True
            raise
        renamed = True
        os.fsync(parent_fd)
        published_status = os.stat(output_path.name, dir_fd=parent_fd, follow_symlinks=False)
        if stage_identity != (published_status.st_dev, published_status.st_ino):
            raise ValueError("published bundle inode differs from staging bundle")
        current_parent = os.stat(parent, follow_symlinks=False)
        if (current_parent.st_dev, current_parent.st_ino) != (parent_status.st_dev, parent_status.st_ino):
            raise ValueError("output parent changed after publication")
        return sources_payload
    except FileExistsError as exc:
        if stage_identity is None or rename_collision:
            if stage_identity is None and not rename_collision:
                try:
                    os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
                except OSError:
                    raise exc
                raise PreflightPublicationUncertain(
                    (_preserved(parent_fd, parent, stage_name, str(stage_path), None, False),)
                )
            raise
        raise PreflightPublicationUncertain(
            (_preserved(parent_fd, parent, stage_name, str(stage_path), stage_identity, True),)
        )
    except PreflightPublicationUncertain:
        raise
    except BaseException as exc:
        if stage_identity is None:
            try:
                os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                raise exc
            raise PreflightPublicationUncertain(
                (_preserved(parent_fd, parent, stage_name, str(stage_path), None, False),)
            ) from exc
        name = output_path.name if renamed else stage_name
        logical = str(output_path if renamed else stage_path)
        raise PreflightPublicationUncertain(
            (_preserved(parent_fd, parent, name, logical, stage_identity, True),)
        ) from exc
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-evidence", required=True, type=Path)
    parser.add_argument("--search-manifest", required=True, type=Path)
    parser.add_argument("--apartment-base-config", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    build_preflight_sources(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
