#!/usr/bin/env python3
"""Collect source-bound OVIV2 TESSE-CD T4 measurements."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextvars
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import numpy as np


METRICS = (
    "total_runtime_s_per_frame", "query_mean_ms", "query_p95_ms",
    "peak_gpu_gb", "peak_ram_gb", "final_map_mb",
)
UNITS = {
    "total_runtime_s_per_frame": "seconds/frame",
    "query_mean_ms": "milliseconds",
    "query_p95_ms": "milliseconds",
    "peak_gpu_gb": "GB (decimal)",
    "peak_ram_gb": "GB (decimal)",
    "final_map_mb": "MB (decimal)",
}
_TIME = Path("/usr/bin/time")
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_PROTOCOL_PATH = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json"
_ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)\s*$", re.MULTILINE)
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)\s*$", re.MULTILINE)
_READ_SNAPSHOT: contextvars.ContextVar[dict[Path, bytes] | None] = contextvars.ContextVar(
    "oviv2_t4_read_snapshot", default=None
)
_TRUSTED_FD_ROOTS: contextvars.ContextVar[tuple[Path, ...]] = contextvars.ContextVar(
    "oviv2_t4_trusted_fd_roots", default=()
)


class T4CollectionError(ValueError):
    pass


@dataclass(frozen=True)
class PreservedArtifact:
    logical_path: Path
    name: str
    parent_device: int
    parent_inode: int
    device: int
    inode: int
    mode: int
    ownership: str


class T4PublicationUncertain(RuntimeError):
    def __init__(self, preserved: Sequence[PreservedArtifact]):
        self.preserved = tuple(preserved)
        details = ", ".join(
            f"{item.name}[{item.ownership},dev={item.device},ino={item.inode}]"
            for item in self.preserved
        ) or "identity unavailable"
        super().__init__(f"T4 publication failed; preserved artifacts: {details}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trusted_fd_root(path: Path) -> Path | None:
    return next(
        (root for root in _TRUSTED_FD_ROOTS.get() if path == root or root in path.parents),
        None,
    )


def _read(path: Path, label: str) -> bytes:
    path = path.absolute()
    snapshot = _READ_SNAPSHOT.get()
    if snapshot is not None and path in snapshot:
        return snapshot[path]
    trusted_root = _trusted_fd_root(path)
    for component in (path, *path.parents):
        if component == trusted_root:
            break
        try:
            if stat.S_ISLNK(os.lstat(component).st_mode):
                raise T4CollectionError(f"{label} path contains a symlink")
        except FileNotFoundError:
            continue
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise T4CollectionError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise T4CollectionError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) != (opened.st_dev, opened.st_ino):
            raise T4CollectionError(f"{label} changed before it was opened")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    witness = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if witness(status) != witness(opened) or witness(opened) != witness(finished) or witness(finished) != witness(current):
        raise T4CollectionError(f"{label} changed while being read")
    data = b"".join(chunks)
    if len(data) != finished.st_size:
        raise T4CollectionError(f"{label} changed size while being read")
    if snapshot is not None:
        snapshot[path] = data
    return data


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read(path, label)
    try:
        def strict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(data, object_pairs_hook=strict, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise T4CollectionError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise T4CollectionError(f"{label} must be a JSON object")
    return value, data


def _record(path: Path, data: bytes | None = None) -> dict[str, Any]:
    absolute = path.absolute()
    payload = _read(absolute, str(absolute)) if data is None else data
    return {"path": str(absolute), "sha256": _sha256(payload), "byte_count": len(payload)}


def _write_new(
    path: Path, value: object, *, parent_fd: int | None = None
) -> tuple[int, int]:
    path = path.absolute()
    if parent_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        trusted_root = _trusted_fd_root(path)
        for component in (path.parent, *path.parent.parents):
            if component == trusted_root:
                break
            if stat.S_ISLNK(os.lstat(component).st_mode):
                raise T4CollectionError("T4 output parent contains a symlink")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    else:
        directory = os.dup(parent_fd)
    data = _canonical(value) + b"\n"
    temporary = f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    descriptor = -1
    temporary_created = False
    published = False
    file_witness: tuple[int, int] | None = None
    try:
        parent_status = os.fstat(directory)
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(path)
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644, dir_fd=directory,
        )
        temporary_created = True
        try:
            file_status = os.fstat(descriptor)
            file_witness = (file_status.st_dev, file_status.st_ino)
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("T4 output write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = -1
        current_parent = os.lstat(path.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise T4CollectionError("T4 output parent changed during publication")
        os.link(
            temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
        current_parent = os.lstat(path.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise T4CollectionError("T4 output parent changed during publication")
        return file_status.st_dev, file_status.st_ino
    except T4PublicationUncertain:
        raise
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            raise T4PublicationUncertain(
                _preserved_artifacts(
                    directory,
                    path.parent,
                    [
                        (temporary, file_witness, True),
                        (path.name, file_witness, published),
                    ],
                )
            ) from exc
        raise
    finally:
        os.close(directory)


def _replace_staged(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.rewrite-{os.getpid()}-{time.time_ns()}")
    _write_new(temporary, value)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _preserved_artifacts(
    parent_fd: int,
    logical_parent: Path,
    targets: Sequence[tuple[str, tuple[int, int] | None, bool]],
) -> tuple[PreservedArtifact, ...]:
    parent = os.fstat(parent_fd)
    try:
        inventory = os.listdir(parent_fd)
    except OSError:
        inventory = []
    records: dict[tuple[str, int, int], PreservedArtifact] = {}
    priority = {"unknown": 0, "unbound": 1, "owned": 2}
    for preferred, witness, created in targets:
        names = [preferred, *(name for name in inventory if name != preferred)]
        for candidate in names:
            try:
                status = os.stat(candidate, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                continue
            identity = (status.st_dev, status.st_ino)
            if candidate == preferred:
                ownership = (
                    "owned" if witness == identity else "unbound" if created and witness is None else "unknown"
                )
            elif witness != identity:
                continue
            else:
                ownership = "owned"
            key = (candidate, status.st_dev, status.st_ino)
            record = PreservedArtifact(
                logical_path=logical_parent / candidate,
                name=candidate,
                parent_device=parent.st_dev,
                parent_inode=parent.st_ino,
                device=status.st_dev,
                inode=status.st_ino,
                mode=status.st_mode,
                ownership=ownership,
            )
            if key not in records or priority[ownership] > priority[records[key].ownership]:
                records[key] = record
    return tuple(records.values())


def _rename_directory_new(
    source: Path, destination: Path, *, parent_fd: int | None = None
) -> tuple[int, int]:
    source, destination = source.absolute(), destination.absolute()
    if parent_fd is None and source.parent != destination.parent:
        raise T4CollectionError("staging and destination must share a parent")
    directory = (
        os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if parent_fd is None else os.dup(parent_fd)
    )
    renamed = False
    source_witness: tuple[int, int] | None = None
    try:
        parent_status = os.fstat(directory)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise T4CollectionError("atomic no-clobber directory publication is unavailable")
        current_parent = os.lstat(destination.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise T4CollectionError("T4 output parent changed during publication")
        source_status = os.stat(source.name, dir_fd=directory, follow_symlinks=False)
        source_witness = (source_status.st_dev, source_status.st_ino)
        result = renameat2(
            ctypes.c_int(directory), os.fsencode(source.name), ctypes.c_int(directory),
            os.fsencode(destination.name), ctypes.c_uint(1),
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
        renamed = True
        os.fsync(directory)
        destination_status = os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
        current_parent = os.lstat(destination.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            cause = T4CollectionError("T4 output parent changed during publication")
            raise T4PublicationUncertain(
                _preserved_artifacts(
                    directory,
                    destination.parent,
                    [(destination.name, (destination_status.st_dev, destination_status.st_ino), True)],
                )
            ) from cause
        return destination_status.st_dev, destination_status.st_ino
    except T4PublicationUncertain:
        raise
    except BaseException as exc:
        if renamed:
            raise T4PublicationUncertain(
                _preserved_artifacts(
                    directory,
                    destination.parent,
                    [(destination.name, source_witness, True)],
                )
            ) from exc
        raise
    finally:
        os.close(directory)


def _parent_matches(path: Path, parent_fd: int) -> bool:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return False
    opened = os.fstat(parent_fd)
    return (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise T4CollectionError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, label: str) -> float:
    converted = _finite(value, label)
    if converted < 0.0:
        raise T4CollectionError(f"{label} must be non-negative")
    return converted


def parse_time_v(path: Path) -> tuple[float, float]:
    text = _read(path, "time log").decode("utf-8", errors="replace")
    elapsed_match, rss_match = _ELAPSED_RE.search(text), _RSS_RE.search(text)
    if elapsed_match is None or rss_match is None:
        raise T4CollectionError("time log lacks elapsed time or maximum RSS")
    fields = [float(item) for item in elapsed_match.group(1).split(":")]
    if len(fields) == 2:
        elapsed = fields[0] * 60.0 + fields[1]
    elif len(fields) == 3:
        elapsed = fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    else:
        raise T4CollectionError("time log elapsed format is invalid")
    if elapsed < 0.0:
        raise T4CollectionError("time log elapsed value is negative")
    return elapsed, int(rss_match.group(1)) / 1_000_000.0


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise T4CollectionError("query latencies are empty")
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def collect_gpu_sample(
    rows: Sequence[Mapping[str, object]], *, process_group_id: int, phase: str
) -> dict[str, object]:
    matched = []
    for row in rows:
        if row.get("process_group_id") != process_group_id:
            continue
        pid = row.get("pid")
        memory = row.get("used_memory_mib")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise T4CollectionError("GPU sample PID is invalid")
        matched.append((pid, _nonnegative(memory, "GPU sample memory")))
    if not matched:
        raise T4CollectionError("GPU query returned no process-group-bound sample")
    if len(matched) == 1:
        return {"phase": phase, "pid": matched[0][0], "process_group_id": process_group_id, "used_memory_mib": matched[0][1]}
    return {
        "phase": phase, "pid": min(pid for pid, _ in matched),
        "process_group_id": process_group_id,
        "used_memory_mib": sum(memory for _, memory in matched),
    }


def build_final_map_inventory(run_root: Path, files: Mapping[str, Path]) -> dict[str, Any]:
    if set(files) != {"snapshot", "entities"}:
        raise T4CollectionError("final-map inventory roles must be snapshot and entities")
    root = Path(os.path.abspath(run_root))
    records = []
    identities = set()
    for role in ("snapshot", "entities"):
        path = Path(os.path.abspath(files[role]))
        if path != root and root not in path.parents:
            raise T4CollectionError(f"final-map {role} is outside the whole-profile run")
        data = _read(path, f"final-map {role}")
        identity = (os.lstat(path).st_dev, os.lstat(path).st_ino)
        if identity in identities:
            raise T4CollectionError("final-map files must be distinct")
        identities.add(identity)
        records.append({"role": role, **_record(path, data)})
    return {
        "schema_version": 1, "manifest_id": "oviv2_tesse_t4_final_map_inventory_v1",
        "files": records, "total_bytes": sum(item["byte_count"] for item in records),
    }


def _internal_artifact_path(
    raw_path: object, run_root: Path, inventory: list[object], label: str
) -> Path:
    if not (
        isinstance(raw_path, str)
        and raw_path not in {"", "."}
        and not Path(raw_path).is_absolute()
        and Path(raw_path).as_posix() == raw_path
        and ".." not in Path(raw_path).parts
        and raw_path in inventory
    ):
        raise T4CollectionError(f"runner {label} path is invalid")
    root = Path(os.path.abspath(run_root))
    path = Path(os.path.abspath(root / raw_path))
    if root not in path.parents:
        raise T4CollectionError(f"runner {label} is outside the run directory")
    return path


def _run_end_coverage(run_manifest: Mapping[str, Any], run_root: Path) -> tuple[int, int]:
    inventory = run_manifest.get("artifact_inventory")
    if not isinstance(inventory, list):
        raise T4CollectionError("runner artifact inventory is invalid")

    def bound_path(record: object, label: str) -> Path:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"}:
            raise T4CollectionError(f"runner {label} record is invalid")
        path = _internal_artifact_path(record.get("path"), run_root, inventory, label)
        if _record(path) != {**record, "path": str(path)}:
            raise T4CollectionError(f"runner {label} binding is stale")
        return path

    source_path = bound_path(run_manifest.get("source_index"), "source index")
    source_index, _ = _json(source_path, "runner source index")
    coverage_path = bound_path(source_index.get("frame_coverage"), "frame coverage")
    data = _read(coverage_path, "runner frame coverage")
    rows: list[dict[str, Any]] = []
    try:
        def strict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            row: dict[str, Any] = {}
            for key, value in pairs:
                if key in row:
                    raise ValueError(f"duplicate frame coverage key: {key}")
                row[key] = value
            return row

        for raw in data.splitlines():
            if not raw:
                raise ValueError("blank frame coverage line")
            row = json.loads(
                raw,
                object_pairs_hook=strict,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            if not isinstance(row, dict):
                raise ValueError("frame coverage row is not an object")
            rows.append(row)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise T4CollectionError("runner frame coverage is invalid") from exc
    if not rows:
        raise T4CollectionError("runner frame coverage is empty")
    last = rows[-1]
    frame_index = last.get("frame_index")
    timestamp_ns = last.get("timestamp_ns")
    if type(frame_index) is not int or type(timestamp_ns) is not int:
        raise T4CollectionError("runner frame coverage run-end identity is invalid")
    return frame_index, timestamp_ns


def _resolve_final_map(run_manifest: Mapping[str, Any], run_root: Path) -> dict[str, Path]:
    declared = run_manifest.get("final_current_map")
    if not isinstance(declared, Mapping) or set(declared) != {
        "frame_index", "timestamp_ns", "scope", "snapshot", "entities",
        "background_storage",
    }:
        raise T4CollectionError("run manifest lacks exact run-end final current-map inventory")
    if not (
        declared.get("frame_index") == run_manifest.get("last_frame_index")
        and isinstance(declared.get("timestamp_ns"), int)
        and not isinstance(declared.get("timestamp_ns"), bool)
        and declared.get("scope") == "current"
        and declared.get("background_storage") == "snapshot.npz:background_xyz"
    ):
        raise T4CollectionError("run manifest final current-map progress is invalid")
    if (declared["frame_index"], declared["timestamp_ns"]) != _run_end_coverage(
        run_manifest, run_root
    ):
        raise T4CollectionError("final current-map does not match run-end frame coverage")
    inventory = run_manifest.get("artifact_inventory")
    if not isinstance(inventory, list):
        raise T4CollectionError("runner artifact inventory is invalid")
    result = {}
    for role in ("snapshot", "entities"):
        record = declared.get(role)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"}:
            raise T4CollectionError(f"run manifest final current-map {role} is invalid")
        path = _internal_artifact_path(
            record.get("path"), run_root, inventory, f"final current-map {role}"
        )
        if _record(path) != {**record, "path": str(path.absolute())}:
            raise T4CollectionError(f"run manifest final current-map {role} binding is stale")
        result[role] = path
    try:
        with np.load(
            io.BytesIO(_read(result["snapshot"], "final current-map snapshot")),
            allow_pickle=False,
        ) as arrays:
            if not {"background_xyz", "timestamp", "scope", "scene_id"} <= set(arrays.files):
                raise T4CollectionError("final current-map snapshot metadata is incomplete")
            metadata = tuple(
                (arrays[name].shape, arrays[name].item())
                for name in ("timestamp", "scope", "scene_id")
            )
    except T4CollectionError:
        raise
    except (OSError, ValueError) as exc:
        raise T4CollectionError("final current-map snapshot is not a valid NPZ") from exc
    if metadata != (
        ((), declared["timestamp_ns"]),
        ((), "current"),
        ((), "apartment"),
    ):
        raise T4CollectionError("final current-map snapshot metadata is stale")
    return result


def _validate_run_inventory(root: Path, inventory: object) -> None:
    if not isinstance(inventory, list) or inventory != sorted(set(inventory)) or any(
        not isinstance(item, str) or not item for item in inventory
    ):
        raise T4CollectionError("runner artifact inventory is invalid")
    actual_files: list[str] = []
    actual_directories: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise T4CollectionError("runner artifact inventory contains a symlink")
        if path.is_file():
            if relative not in {"run_manifest.json", "execution_receipt.json"}:
                actual_files.append(relative)
        elif path.is_dir():
            actual_directories.append(relative)
        else:
            raise T4CollectionError("runner artifact inventory contains a non-file entry")
    if inventory != sorted(actual_files):
        raise T4CollectionError("runner artifact inventory differs from run directory")
    expected_directories = sorted({
        parent.as_posix()
        for item in inventory
        for parent in Path(item).parents
        if parent.as_posix() != "."
    })
    if sorted(actual_directories) != expected_directories:
        raise T4CollectionError("runner directory inventory differs from artifact paths")


def _actual_runner(
    run: Mapping[str, Any], run_path: Path, config_path: Path,
    config: Mapping[str, Any], config_data: bytes,
    candidate_id: str, config_sha: str,
) -> tuple[dict[str, Any], Path]:
    actual = dict(run)
    actual_path = run_path
    if run.get("manifest_id") == "oviv2_tesse_t4_measurement_run_v1":
        if run.get("candidate_config") != _record(config_path, config_data):
            raise T4CollectionError("measurement run candidate config binding mismatch")
        runner_record = run.get("runner_manifest")
        if not isinstance(runner_record, Mapping) or not isinstance(runner_record.get("path"), str):
            raise T4CollectionError("measurement run runner manifest binding is invalid")
        actual_path = Path(runner_record["path"])
        actual, actual_data = _json(actual_path, "actual runner manifest")
        if dict(runner_record) != _record(actual_path, actual_data):
            raise T4CollectionError("measurement run runner manifest binding is stale")
        if run.get("processed_frame_count") != actual.get("processed_frame_count"):
            raise T4CollectionError("measurement run processed-frame binding mismatch")
    if not (
        actual.get("schema_version") == 2
        and actual.get("dataset") == "TESSE-CD"
        and actual.get("method_id") == "OVIV2"
        and actual.get("protocol_id") == "oviv2-tessecd-v2"
        and actual.get("scene") == "apartment"
    ):
        raise T4CollectionError("run manifest is not an Apartment OVIV2 runner output")
    from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash

    algorithm_hash = canonical_algorithm_hash(config)
    canonical_config_sha = _sha256(_canonical(config))
    temporal = config.get("temporal_readout")
    if not (
        canonical_config_sha == config_sha
        and config.get("algorithm_hash") == algorithm_hash
        and isinstance(temporal, Mapping)
        and temporal.get("execution_profile") == candidate_id
        and actual.get("algorithm_hash") == algorithm_hash
    ):
        raise T4CollectionError("candidate identity does not match normalized runner config")
    source_record = actual.get("config")
    if not isinstance(source_record, Mapping) or {
        "sha256": _sha256(config_data), "byte_count": len(config_data)
    } != {key: source_record.get(key) for key in ("sha256", "byte_count")}:
        raise T4CollectionError("runner source config binding mismatch")
    normalized_record = actual.get("normalized_run_config")
    if not isinstance(normalized_record, Mapping) or set(normalized_record) != {"path", "sha256", "byte_count"}:
        raise T4CollectionError("runner normalized config record is invalid")
    normalized_path = Path(str(normalized_record["path"]))
    normalized_path = (
        normalized_path if normalized_path.is_absolute()
        else actual_path.parent / normalized_path
    ).absolute()
    if actual_path.parent.absolute() not in normalized_path.parents:
        raise T4CollectionError("runner normalized config is outside the run directory")
    normalized, normalized_data = _json(normalized_path, "normalized runner config")
    if normalized != dict(config) or _record(normalized_path, normalized_data) != {
        **normalized_record, "path": str(normalized_path.absolute())
    }:
        raise T4CollectionError("normalized runner config differs from requested config")
    _validate_run_inventory(actual_path.parent, actual.get("artifact_inventory"))
    return actual, actual_path


def _bind_shortlist_candidate(
    shortlist_path: Path,
    *,
    candidate_id: str,
    config_path: Path,
    config: Mapping[str, Any],
    config_data: bytes,
    config_sha: str,
) -> dict[str, Any]:
    shortlist, _ = _json(shortlist_path, "shortlist")
    rows = shortlist.get("shortlisted_candidates")
    if not isinstance(rows, list):
        raise T4CollectionError("shortlist candidate inventory is invalid")
    matched = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
    ]
    if len(matched) != 1:
        raise T4CollectionError("shortlist does not bind exactly one requested candidate")
    row = matched[0]
    if set(row) != {
        "candidate_id", "profile", "config_sha256", "algorithm_hash", "result",
        "selected_config", "selected_config_record",
    }:
        raise T4CollectionError("shortlist candidate schema is not exact")
    from scripts.evaluation.evaluate_oviv2_tesse_occlusion import canonical_algorithm_hash

    selected = row.get("selected_config")
    result_record = row.get("result")
    if not (
        row.get("profile") == candidate_id
        and row.get("config_sha256") == config_sha
        and row.get("algorithm_hash") == canonical_algorithm_hash(config)
        and isinstance(selected, Mapping)
        and dict(selected) == dict(config)
        and row.get("selected_config_record") == _record(config_path, config_data)
        and isinstance(result_record, Mapping)
        and isinstance(result_record.get("path"), str)
    ):
        raise T4CollectionError("shortlist candidate identity binding is invalid")
    result_path = Path(result_record["path"])
    if dict(result_record) != _record(result_path):
        raise T4CollectionError("shortlist candidate result binding is stale")
    return dict(row)


def _validate_protocol(protocol: Mapping[str, Any], path: Path) -> None:
    frozen_path = FROZEN_PROTOCOL_PATH.absolute()
    if path.absolute() != frozen_path or _read(path, "collection protocol") != _read(frozen_path, "frozen collection protocol"):
        raise T4CollectionError("collection protocol is not the controlled frozen manifest")
    if set(protocol) != {
        "schema_version", "manifest_id", "dataset", "method_id", "protocol_id",
        "scene", "office_data_permitted", "sample_interval_ms", "time_collector",
        "gpu_collector", "query", "final_map_inventory", "units", "bounds",
        "mapping_argv", "query_argv",
    } or not (
        protocol.get("schema_version") == 1
        and protocol.get("manifest_id") == "oviv2_tesse_t4_collection_protocol_v1"
        and protocol.get("dataset") == "TESSE-CD" and protocol.get("method_id") == "OVIV2"
        and protocol.get("protocol_id") == "oviv2-tessecd-v2" and protocol.get("scene") == "apartment"
        and protocol.get("office_data_permitted") is False and protocol.get("sample_interval_ms") == 200
        and protocol.get("units") == UNITS
    ):
        raise T4CollectionError("collection protocol identity, scope, or units are invalid")
    if protocol.get("final_map_inventory") != {
        "scope": "final_current_map_after_all_processed_frames",
        "roles": ["snapshot", "entities"],
        "background_storage": "snapshot.npz:background_xyz",
        "include": "only_unique_regular_source_files_for_snapshot_and_entities",
        "exclude": ["cumulative", "checkpoints", "diagnostics", "logs", "metrics", "office"],
    }:
        raise T4CollectionError("collection final-map inventory schema is invalid")
    if protocol.get("time_collector") != "/usr/bin/time -v" or protocol.get("gpu_collector") != {
        "command": "nvidia-smi",
        "identity_fields": ["gpu_uuid", "pid", "proc_starttime_ticks", "process_group_id"],
        "aggregation": "sum_pid_memory_per_sample_then_peak_across_mapping_and_queries",
        "source_unit": "MiB", "output_unit": "GB (decimal)", "conversion_divisor": 1000.0,
    }:
        raise T4CollectionError("collection resource collectors are not frozen")
    expected_commands = {
        "mapping_argv": [
            "{python}", "{repo_root}/scripts/evaluation/run_oviv2_tesse_cd_v2.py",
            "--config", "{config}", "--output", "{run}",
        ],
        "query_argv": [
            "{python}", "{repo_root}/scripts/evaluation/measure_baseline_queries.py",
            "--baseline", "oviv2", "--snapshot", "{snapshot}", "--entities",
            "{entities}", "--queries", "{queries}", "--clip-weight", "{checkpoint}",
            "--device", "cuda:{gpu}", "--warmup", "10", "--repeats", "5",
            "--protocol", "{protocol}", "--output", "{query_output}",
        ],
    }
    if any(protocol.get(key) != value for key, value in expected_commands.items()):
        raise T4CollectionError("collection command templates are not frozen")
    query = protocol.get("query")
    if not isinstance(query, Mapping) or set(query) != {
        "vocabulary_path", "vocabulary_sha256", "text_model_id", "checkpoint_sha256",
        "operation_order", "warmup_count", "measured_repeats",
        "cuda_synchronize_each_query", "precomputed_query_embeddings",
    } or not (
        query.get("text_model_id") == "ViT-H-14" and query.get("warmup_count") == 10
        and query.get("measured_repeats") == 5
        and query.get("operation_order") == [
            "full_tokenization", "text_encoding", "l2_normalization", "current_map_entity_ranking"
        ]
        and query.get("cuda_synchronize_each_query") is True
        and query.get("precomputed_query_embeddings") is False
    ):
        raise T4CollectionError("collection query protocol is invalid")
    vocabulary = query.get("vocabulary_path")
    vocabulary_hash = query.get("vocabulary_sha256")
    if not isinstance(vocabulary, str) or not isinstance(vocabulary_hash, str):
        raise T4CollectionError("collection query vocabulary binding is invalid")
    vocabulary_path = Path(__file__).resolve().parents[2] / vocabulary
    if hashlib.sha256(_read(vocabulary_path, "query vocabulary")).hexdigest() != vocabulary_hash:
        raise T4CollectionError("collection query vocabulary hash mismatch")
    if protocol.get("bounds") != {
        "total_runtime_s_per_frame": 6.42, "query_mean_ms": 11.92,
        "query_p95_ms": 12.12, "peak_gpu_gb": 12.76,
        "peak_ram_gb": 9.36, "final_map_mb": 46.77,
    }:
        raise T4CollectionError("collection bounds are not frozen")


def _validate_gpu_evidence(gpu: Mapping[str, Any]) -> tuple[object, list[float]]:
    samples = gpu.get("samples")
    events = gpu.get("phase_events")
    if gpu.get("sample_interval_ms") != 200 or not isinstance(samples, list):
        raise T4CollectionError("GPU samples violate the 200 ms protocol")
    if not isinstance(events, list) or len(events) != 4:
        raise T4CollectionError("GPU phase-event sidecar is invalid")
    expected = [("mapping", "start"), ("mapping", "end"), ("queries", "start"), ("queries", "end")]
    identities: dict[str, tuple[int, int, int]] = {}
    bounds: dict[str, tuple[int, int]] = {}
    for event, identity in zip(events, expected):
        if not isinstance(event, Mapping) or set(event) != {
            "phase", "event", "timestamp_ns", "pid", "process_group_id",
            "proc_starttime_ticks",
        } or (event.get("phase"), event.get("event")) != identity:
            raise T4CollectionError("GPU phase-event sidecar schema is invalid")
        values = tuple(event.get(key) for key in ("pid", "process_group_id", "proc_starttime_ticks"))
        timestamp = event.get("timestamp_ns")
        if any(type(value) is not int or value <= 0 for value in (*values, timestamp)):
            raise T4CollectionError("GPU phase-event process identity is invalid")
        phase = identity[0]
        if identity[1] == "start":
            identities[phase] = values
            bounds[phase] = (timestamp, -1)
        elif values != identities.get(phase) or timestamp <= bounds[phase][0]:
            raise T4CollectionError("GPU phase-event PID starttime changed")
        else:
            bounds[phase] = (bounds[phase][0], timestamp)
    event_timestamps = [event["timestamp_ns"] for event in events]
    if event_timestamps != sorted(event_timestamps):
        raise T4CollectionError("GPU phase-event timestamps are not ordered")
    peaks: list[float] = []
    for phase in ("mapping", "queries"):
        phase_samples = [sample for sample in samples if isinstance(sample, Mapping) and sample.get("phase") == phase]
        if len(phase_samples) < 2:
            raise T4CollectionError(f"GPU {phase} phase requires at least two samples")
        timestamps = [sample.get("timestamp_ns") for sample in phase_samples]
        if any(type(value) is not int or value <= 0 for value in timestamps) or timestamps != sorted(set(timestamps)):
            raise T4CollectionError("GPU sample timestamps are not strictly monotonic")
        start, end = bounds[phase]
        if timestamps[0] - start > 200_000_000 or end - timestamps[-1] > 200_000_000 or any(
            right - left > 200_000_000 for left, right in zip(timestamps, timestamps[1:])
        ):
            raise T4CollectionError("GPU sampling interval or phase endpoint coverage exceeds 200 ms")
        pid, pgid, _starttime = identities[phase]
        process_starttimes: dict[int, int] = {}
        for sample in phase_samples:
            if sample.get("pid") != pid or sample.get("process_group_id") != pgid:
                raise T4CollectionError("GPU sample is not bound to the phase process")
            processes = sample.get("processes")
            if not isinstance(processes, list):
                raise T4CollectionError("GPU sample process inventory is missing")
            memory = 0.0
            for process in processes:
                if not isinstance(process, Mapping) or set(process) != {
                    "gpu_uuid", "pid", "proc_starttime_ticks", "process_group_id", "used_memory_mib"
                }:
                    raise T4CollectionError("GPU process identity schema is invalid")
                if process.get("process_group_id") != pgid:
                    raise T4CollectionError("GPU sample contains a foreign process group")
                process_pid = process.get("pid")
                process_starttime = process.get("proc_starttime_ticks")
                if (
                    not isinstance(process.get("gpu_uuid"), str) or not process["gpu_uuid"]
                    or type(process_pid) is not int or process_pid <= 0
                    or type(process_starttime) is not int or process_starttime <= 0
                ):
                    raise T4CollectionError("GPU process identity is invalid")
                if process_pid in process_starttimes and process_starttimes[process_pid] != process_starttime:
                    raise T4CollectionError("GPU process starttime indicates PID reuse")
                process_starttimes[process_pid] = process_starttime
                memory += _nonnegative(process.get("used_memory_mib"), "GPU process memory")
            if memory != _nonnegative(sample.get("used_memory_mib"), "GPU sample memory"):
                raise T4CollectionError("GPU process memory aggregate is stale")
            peaks.append(memory)
    all_timestamps = [sample["timestamp_ns"] for sample in samples if isinstance(sample, Mapping)]
    if len(all_timestamps) != len(samples) or any(
        sample.get("phase") not in {"mapping", "queries"}
        for sample in samples if isinstance(sample, Mapping)
    ):
        raise T4CollectionError("GPU sample phase inventory is not exact")
    if all_timestamps != sorted(set(all_timestamps)):
        raise T4CollectionError("GPU sample timestamps are not globally monotonic")
    pgids = [identities[phase][1] for phase in ("mapping", "queries")]
    if gpu.get("process_group_ids", sorted(set(pgids))) != sorted(set(pgids)):
        raise T4CollectionError("GPU process-group inventory is stale")
    if gpu.get("process_group_id") != min(pgids):
        raise T4CollectionError("GPU primary process group is stale")
    return gpu.get("process_group_id"), peaks


def measure_t4(request: Mapping[str, object]) -> dict[str, Any]:
    required = {"candidate_id", "config", "config_sha256", "run_manifest", "time_log", "gpu_samples", "query_measurements", "protocol", "shortlist", "output"}
    missing = required - set(request)
    if missing:
        raise T4CollectionError(f"measurement request missing {sorted(missing)}")
    candidate_id = request["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise T4CollectionError("candidate ID is invalid")
    config_path = Path(request["config"])
    config, config_data = _json(config_path, "candidate config")
    config_sha = request["config_sha256"]
    source_config_sha = request.get("config_source_sha256", config_sha)
    if not isinstance(config_sha, str) or _sha256(config_data) != source_config_sha:
        raise T4CollectionError("candidate config hash mismatch")
    run_path = Path(request["run_manifest"])
    run, run_data = _json(run_path, "run manifest")
    actual_runner, actual_runner_path = _actual_runner(
        run, run_path, config_path, config, config_data, candidate_id, config_sha
    )
    _bind_shortlist_candidate(
        Path(request["shortlist"]),
        candidate_id=candidate_id,
        config_path=config_path,
        config=config,
        config_data=config_data,
        config_sha=config_sha,
    )
    frames = actual_runner.get("processed_frame_count")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise T4CollectionError("processed-frame count is invalid")
    elapsed, peak_ram = parse_time_v(Path(request["time_log"]))
    gpu, _ = _json(Path(request["gpu_samples"]), "gpu samples")
    pgid, peaks = _validate_gpu_evidence(gpu)
    query, _ = _json(Path(request["query_measurements"]), "query measurements")
    latencies = query.get("latencies_ms")
    if not isinstance(latencies, list) or not latencies:
        raise T4CollectionError("query measurements lack raw latencies")
    values = [_nonnegative(value, "query latency") for value in latencies]
    samples = query.get("samples")
    if not (
        query.get("schema_version") == 1 and query.get("manifest_id") == "oviv2-query-measurement-v1"
        and query.get("baseline") == "oviv2" and query.get("model") == "ViT-H-14"
        and query.get("warmup_count") == 10 and query.get("measured_repeats") == 5
        and query.get("repeat_count") == 5 and isinstance(samples, list)
        and len(samples) == len(values)
        and [_nonnegative(item.get("latency_ms"), "query sample latency") for item in samples if isinstance(item, Mapping)] == values
    ):
        raise T4CollectionError("query measurements violate the full OVIV2 protocol")
    if query.get("query_protocol") != _record(Path(request["protocol"])):
        raise T4CollectionError("query measurements use a different frozen protocol")
    output = Path(request["output"]).absolute()
    supplied_inventory = request.get("final_map_inventory")
    final_files = _resolve_final_map(actual_runner, actual_runner_path.parent)
    expected_inventory = build_final_map_inventory(actual_runner_path.parent, final_files)
    if supplied_inventory is None:
        inventory = expected_inventory
        inventory_path = output.with_name(f"{candidate_id}-final-map-inventory.json")
        _write_new(inventory_path, inventory)
        created_inventory = True
    else:
        inventory_path = Path(supplied_inventory)
        inventory, _ = _json(inventory_path, "final map inventory")
        created_inventory = False
        if inventory != expected_inventory:
            raise T4CollectionError("final map inventory differs from the actual runner")
    inventory_by_role = {
        item.get("role"): {key: item.get(key) for key in ("path", "sha256", "byte_count")}
        for item in inventory.get("files", []) if isinstance(item, Mapping)
    }
    query_sources = query.get("sources")
    if not isinstance(query_sources, Mapping) or set(query_sources) != {
        "snapshot", "entities", "queries", "checkpoint"
    } or query_sources["snapshot"] != inventory_by_role.get("snapshot") or query_sources["entities"] != inventory_by_role.get("entities"):
        raise T4CollectionError("query measurements are not bound to the final current map")
    sources = {
        "config": _record(config_path, config_data),
        "run_manifest": _record(run_path, run_data),
        "time_log": _record(Path(request["time_log"])),
        "gpu_samples": _record(Path(request["gpu_samples"])),
        "query_measurements": _record(Path(request["query_measurements"])),
        "final_map_inventory": _record(inventory_path),
        "protocol": _record(Path(request["protocol"])),
        "shortlist": _record(Path(request["shortlist"])),
    }
    sources.update({f"{name}_sha256": record["sha256"] for name, record in tuple(sources.items())})
    metrics = {
        "total_runtime_s_per_frame": elapsed / frames,
        "query_mean_ms": sum(values) / len(values),
        "query_p95_ms": _percentile(values, 95.0),
        "peak_gpu_gb": max(peaks) / 1000.0,
        "peak_ram_gb": peak_ram,
        "final_map_mb": inventory["total_bytes"] / 1_000_000.0,
    }
    payload = {
        "schema_version": 1, "manifest_id": "oviv2_tesse_t4_evidence_v1",
        "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
        "scene": "apartment", "candidate_id": candidate_id, "config_sha256": config_sha,
        "process_group_id": pgid, "units": dict(UNITS), "metrics": metrics, "sources": sources,
    }
    try:
        _write_new(output, payload)
    except BaseException:
        if created_inventory:
            inventory_path.unlink(missing_ok=True)
        raise
    return payload


def _proc_starttime(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
    return int(fields[21])


def _gpu_rows(gpu: str) -> list[dict[str, object]]:
    command = ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[1].isdigit():
            pid = int(fields[1])
            try:
                pgid = os.getpgid(pid)
                starttime = _proc_starttime(pid)
            except (ProcessLookupError, FileNotFoundError):
                continue
            rows.append({"gpu_uuid": fields[0], "pid": pid, "proc_starttime_ticks": starttime, "process_group_id": pgid, "used_memory_mib": float(fields[2])})
    return rows


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process: subprocess.Popen[Any], pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 5.0
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait()


def _run_sampled(
    argv: Sequence[str], *, gpu: str, phase: str, time_log: Path | None = None
) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    command = [str(_TIME), "-v", "-o", str(time_log), *argv] if time_log else list(argv)
    process = subprocess.Popen(command, start_new_session=True, cwd=REPO_ROOT)
    # start_new_session makes the child PID the process-group ID before exec.
    pgid = process.pid
    try:
        starttime = _proc_starttime(process.pid)
        events = [{
            "phase": phase, "event": "start", "timestamp_ns": time.time_ns(),
            "pid": process.pid, "process_group_id": pgid,
            "proc_starttime_ticks": starttime,
        }]
        samples: list[dict[str, object]] = []
        while process.poll() is None:
            iteration_started = time.monotonic()
            rows = [row for row in _gpu_rows(gpu) if row["process_group_id"] == pgid]
            samples.append({
                "timestamp_ns": time.time_ns(), "phase": phase, "pid": process.pid,
                "process_group_id": pgid, "used_memory_mib": sum(float(row["used_memory_mib"]) for row in rows),
                "processes": rows,
            })
            # Leave headroom for scheduler jitter while enforcing a 200 ms maximum gap.
            time.sleep(max(0.0, 0.18 - (time.monotonic() - iteration_started)))
        process.wait()
        final_timestamp = time.time_ns()
        if not samples or samples[-1]["timestamp_ns"] < final_timestamp:
            samples.append({
                "timestamp_ns": final_timestamp, "phase": phase, "pid": process.pid,
                "process_group_id": pgid, "used_memory_mib": 0.0, "processes": [],
        })
        if _process_group_exists(pgid):
            raise T4CollectionError(f"{phase} process group remained alive after leader exit")
        events.append({
            "phase": phase, "event": "end",
            "timestamp_ns": max(time.time_ns(), final_timestamp + 1),
            "pid": process.pid, "process_group_id": pgid,
            "proc_starttime_ticks": starttime,
        })
        return process.returncode, samples, events
    except BaseException:
        _terminate_process_group(process, pgid)
        raise


def _collect_shortlist_unpublished(
    protocol_path: Path, shortlist_path: Path, gpu: str, output: Path
) -> list[Path]:
    protocol, _ = _json(protocol_path, "T4 protocol")
    _validate_protocol(protocol, protocol_path)
    shortlist, _ = _json(shortlist_path, "shortlist")
    if shortlist.get("office_results_read") not in (None, False) or shortlist.get("scenes_read") not in (None, ["apartment"]):
        raise T4CollectionError("Office evidence is forbidden")
    candidates = shortlist.get("shortlisted_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise T4CollectionError("shortlist contains no candidates")
    evidence_paths = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise T4CollectionError("shortlist candidate is invalid")
        candidate_id = candidate.get("candidate_id")
        config_record = candidate.get("selected_config_record")
        if not isinstance(candidate_id, str) or not isinstance(config_record, Mapping) or not isinstance(config_record.get("path"), str):
            raise T4CollectionError("shortlist candidate config binding is invalid")
        config_path = Path(config_record["path"])
        config_payload, config_data = _json(config_path, "candidate config")
        if dict(config_record) != _record(config_path, config_data):
            raise T4CollectionError("shortlist candidate config source binding mismatch")
        _bind_shortlist_candidate(
            shortlist_path,
            candidate_id=candidate_id,
            config_path=config_path,
            config=config_payload,
            config_data=config_data,
            config_sha=str(candidate.get("config_sha256")),
        )
        query_protocol = protocol.get("query")
        checkpoint = config_payload.get("clip_pretrained_path")
        vocabulary = query_protocol.get("vocabulary_path") if isinstance(query_protocol, Mapping) else None
        checkpoint_sha = query_protocol.get("checkpoint_sha256") if isinstance(query_protocol, Mapping) else None
        if not isinstance(checkpoint, str) or not isinstance(vocabulary, str) or not isinstance(checkpoint_sha, str):
            raise T4CollectionError("query checkpoint or vocabulary binding is unavailable")
        if hashlib.sha256(_read(Path(checkpoint), "query checkpoint")).hexdigest() != checkpoint_sha:
            raise T4CollectionError("query checkpoint hash mismatch")
        candidate_root = output / candidate_id
        run_root = candidate_root / "run"
        candidate_root.mkdir()
        mapping_template = protocol.get("mapping_argv")
        query_template = protocol.get("query_argv")
        if not isinstance(mapping_template, list) or not isinstance(query_template, list):
            raise T4CollectionError("protocol command templates are invalid")
        repo_root = REPO_ROOT
        replacements = {
            "{python}": str(Path(sys.executable).resolve()), "{repo_root}": str(repo_root),
            "{config}": config_record["path"], "{run}": str(run_root), "{gpu}": gpu,
            "{protocol}": str(protocol_path), "{query_output}": str(candidate_root / "query.json"),
        }
        def expand(values: Sequence[object]) -> list[str]:
            return [
                replacements.get(str(value), str(value).replace("{repo_root}", str(repo_root)))
                for value in values
            ]
        code, mapping_samples, mapping_events = _run_sampled(
            expand(mapping_template), gpu=gpu, phase="mapping", time_log=candidate_root / "time.txt"
        )
        if code:
            raise subprocess.CalledProcessError(code, expand(mapping_template))
        runner_manifest_path = run_root / "run_manifest.json"
        runner_manifest, runner_manifest_data = _json(runner_manifest_path, "runner manifest")
        final_files = _resolve_final_map(runner_manifest, run_root)
        inventory_path = candidate_root / "final-map-inventory.json"
        _write_new(inventory_path, build_final_map_inventory(run_root, final_files))
        replacements.update({
            "{snapshot}": str(final_files["snapshot"]), "{entities}": str(final_files["entities"]),
            "{queries}": str((Path(__file__).resolve().parents[2] / vocabulary).absolute()),
            "{checkpoint}": checkpoint,
        })
        code, query_samples, query_events = _run_sampled(
            expand(query_template), gpu=gpu, phase="queries"
        )
        if code:
            raise subprocess.CalledProcessError(code, expand(query_template))
        all_samples = mapping_samples + query_samples
        if not all_samples:
            raise T4CollectionError("no PID-bound GPU samples collected")
        pgids = {sample["process_group_id"] for sample in all_samples}
        gpu_path = candidate_root / "gpu.json"
        _write_new(gpu_path, {
            "sample_interval_ms": 200, "process_group_id": min(pgids),
            "process_group_ids": sorted(pgids),
            "phase_events": [*mapping_events, *query_events], "samples": all_samples,
        })
        wrapper_path = candidate_root / "measurement-run.json"
        wrapper = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_t4_measurement_run_v1",
            "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment", "candidate_id": candidate_id,
            "config_sha256": candidate.get("config_sha256"),
            "processed_frame_count": runner_manifest.get("processed_frame_count"),
            "collection_protocol": _record(protocol_path), "shortlist": _record(shortlist_path),
            "candidate_config": _record(config_path, config_data),
            "runner_manifest": _record(runner_manifest_path, runner_manifest_data),
            "time_log": _record(candidate_root / "time.txt"), "gpu_samples": _record(gpu_path),
            "query_measurements": _record(candidate_root / "query.json"),
            "final_map_inventory": _record(inventory_path),
        }
        _write_new(wrapper_path, wrapper)
        evidence_path = candidate_root / "evidence.json"
        measure_t4({
            "candidate_id": candidate_id, "config": Path(config_record["path"]),
            "config_sha256": candidate.get("config_sha256"),
            "config_source_sha256": config_record.get("sha256"), "run_manifest": wrapper_path,
            "time_log": candidate_root / "time.txt", "gpu_samples": gpu_path,
            "query_measurements": candidate_root / "query.json", "protocol": protocol_path,
            "shortlist": shortlist_path, "final_map_inventory": inventory_path, "output": evidence_path,
        })
        evidence_paths.append(evidence_path)
    return evidence_paths


def _rebase_candidate(candidate_root: Path, staging: Path, destination: Path) -> None:
    def final_path(path: str) -> str:
        source = Path(path)
        try:
            relative = source.relative_to(staging)
        except ValueError:
            return path
        return str(destination / relative)

    inventory_path = candidate_root / "final-map-inventory.json"
    inventory, _ = _json(inventory_path, "staged final map inventory")
    for item in inventory["files"]:
        item["path"] = final_path(item["path"])
    _replace_staged(inventory_path, inventory)

    query_path = candidate_root / "query.json"
    query, _ = _json(query_path, "staged query measurements")
    for role in ("snapshot", "entities"):
        query["sources"][role]["path"] = final_path(query["sources"][role]["path"])
    _replace_staged(query_path, query)

    wrapper_path = candidate_root / "measurement-run.json"
    wrapper, _ = _json(wrapper_path, "staged measurement run")
    final_candidate = destination / candidate_root.relative_to(staging)
    staged_files = {
        "runner_manifest": candidate_root / "run" / "run_manifest.json",
        "time_log": candidate_root / "time.txt",
        "gpu_samples": candidate_root / "gpu.json",
        "query_measurements": query_path,
        "final_map_inventory": inventory_path,
    }
    for role, staged_path in staged_files.items():
        wrapper[role] = _record(staged_path)
        wrapper[role]["path"] = str(final_candidate / staged_path.relative_to(candidate_root))
    _replace_staged(wrapper_path, wrapper)

    evidence_path = candidate_root / "evidence.json"
    evidence, _ = _json(evidence_path, "staged T4 evidence")
    evidence_files = {
        "run_manifest": wrapper_path,
        "time_log": candidate_root / "time.txt",
        "gpu_samples": candidate_root / "gpu.json",
        "query_measurements": query_path,
        "final_map_inventory": inventory_path,
    }
    for role, staged_path in evidence_files.items():
        record = _record(staged_path)
        record["path"] = str(final_candidate / staged_path.relative_to(candidate_root))
        evidence["sources"][role] = record
        evidence["sources"][f"{role}_sha256"] = record["sha256"]
    _replace_staged(evidence_path, evidence)


def collect_shortlist(
    protocol_path: Path, shortlist_path: Path, gpu: str, output: Path
) -> list[Path]:
    destination = output.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for component in (destination.parent, *destination.parent.parents):
        if stat.S_ISLNK(os.lstat(component).st_mode):
            raise T4CollectionError("T4 output parent contains a symlink")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    staging = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}-{time.time_ns()}"
    )
    publication_directory = -1
    staging_directory = -1
    owned_witness: tuple[int, int] | None = None
    staging_created = False
    published = False
    try:
        publication_directory = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        os.mkdir(staging.name, dir_fd=publication_directory)
        staging_created = True
        status = os.stat(
            staging.name, dir_fd=publication_directory, follow_symlinks=False
        )
        owned_witness = (status.st_dev, status.st_ino)
        staging_directory = os.open(
            staging.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=publication_directory,
        )
        opened_staging = os.fstat(staging_directory)
        if (opened_staging.st_dev, opened_staging.st_ino) != owned_witness:
            raise T4CollectionError("T4 staging directory changed before open")
        stable_staging = Path(f"/proc/self/fd/{staging_directory}")
        if not _parent_matches(destination.parent, publication_directory):
            raise T4CollectionError("T4 output parent changed before collection")
        trusted_token = _TRUSTED_FD_ROOTS.set((stable_staging,))
        try:
            staged_paths = _collect_shortlist_unpublished(
                protocol_path, shortlist_path, gpu, stable_staging
            )
            for staged_path in staged_paths:
                _rebase_candidate(staged_path.parent, stable_staging, destination)
        finally:
            _TRUSTED_FD_ROOTS.reset(trusted_token)
        os.fsync(staging_directory)
        if not _parent_matches(destination.parent, publication_directory):
            raise T4CollectionError("T4 output parent changed during collection")
        owned_witness = _rename_directory_new(
            staging, destination, parent_fd=publication_directory
        )
        published = True
        if not _parent_matches(destination.parent, publication_directory):
            raise T4CollectionError("T4 output parent changed during publication")
        return [destination / path.relative_to(stable_staging) for path in staged_paths]
    except BaseException as exc:
        if publication_directory >= 0:
            preferred = destination.name if published else staging.name
            preserved = _preserved_artifacts(
                publication_directory,
                destination.parent,
                [(preferred, owned_witness, True)],
            )
            if staging_created or preserved:
                if isinstance(exc, T4PublicationUncertain):
                    preserved = (*exc.preserved, *preserved)
                raise T4PublicationUncertain(
                    preserved
                ) from (
                    exc.__cause__
                    if isinstance(exc, T4PublicationUncertain) and exc.__cause__ is not None
                    else exc
                )
        if isinstance(exc, T4PublicationUncertain):
            raise
        raise
    finally:
        if staging_directory >= 0:
            os.close(staging_directory)
        if publication_directory >= 0:
            os.close(publication_directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = collect_shortlist(args.protocol.absolute(), args.shortlist.absolute(), args.gpu, args.output.absolute())
    print(json.dumps({"status": "COLLECTED", "evidence": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
