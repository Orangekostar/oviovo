#!/usr/bin/env python3
"""Recompute OVIV2 TESSE-CD T4 gates from raw, hash-bound sources."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import contextvars
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import shutil
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.measure_oviv2_tesse_t4 import (
    METRICS, UNITS, T4CollectionError, _actual_runner, _bind_shortlist_candidate,
    _percentile, _validate_gpu_evidence,
    _resolve_final_map, _validate_protocol, build_final_map_inventory,
    parse_time_v,
)
import scripts.evaluation.measure_oviv2_tesse_t4 as _collector


BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}
RAW_SOURCE_NAMES = (
    "config", "run_manifest", "time_log", "gpu_samples", "query_measurements",
    "final_map_inventory", "protocol", "shortlist",
)
_READ_SNAPSHOT: contextvars.ContextVar[dict[Path, bytes] | None] = contextvars.ContextVar(
    "oviv2_t4_verify_read_snapshot", default=None
)


class T4GateError(ValueError):
    pass


class T4PublicationUncertain(RuntimeError):
    def __init__(self, preserved: Sequence[_collector.PreservedArtifact]):
        self.preserved = tuple(preserved)
        details = ", ".join(
            f"{item.name}[{item.ownership},dev={item.device},ino={item.inode}]"
            for item in self.preserved
        ) or "identity unavailable"
        super().__init__(f"T4 verification publication failed; preserved artifacts: {details}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path, label: str) -> bytes:
    path = path.absolute()
    snapshot = _READ_SNAPSHOT.get()
    if snapshot is not None and path in snapshot:
        return snapshot[path]
    for component in (path, *path.parents):
        try:
            if stat.S_ISLNK(os.lstat(component).st_mode):
                raise T4GateError(f"{label} path contains a symlink")
        except FileNotFoundError:
            continue
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise T4GateError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise T4GateError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise T4GateError(f"{label} changed before it was opened")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        finished = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    witness = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if witness(before) != witness(opened) or witness(opened) != witness(finished) or witness(finished) != witness(after):
        raise T4GateError(f"{label} changed while being read")
    data = b"".join(chunks)
    if len(data) != finished.st_size:
        raise T4GateError(f"{label} changed size while being read")
    if snapshot is not None:
        snapshot[path] = data
    return data


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
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
        raise T4GateError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise T4GateError(f"{label} must be a JSON object")
    return value, data


def _record(path: Path, data: bytes | None = None) -> dict[str, Any]:
    absolute = path.absolute()
    payload = _read(absolute, str(absolute)) if data is None else data
    return {"path": str(absolute), "sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)}


def _bound(record: object, label: str) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "byte_count"}:
        raise T4GateError(f"{label} source record is invalid")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise T4GateError(f"{label} source path is not absolute")
    path = Path(os.path.abspath(path_value))
    if str(path) != path_value:
        raise T4GateError(f"{label} source path is not canonical")
    data = _read(path, label)
    if hashlib.sha256(data).hexdigest() != record.get("sha256") or len(data) != record.get("byte_count"):
        raise T4GateError(f"{label} source binding mismatch")
    return path, data


def _write_new(
    path: Path, value: object, *, parent_fd: int | None = None
) -> tuple[int, int]:
    path = path.absolute()
    if parent_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for component in (path.parent, *path.parent.parents):
            if stat.S_ISLNK(os.lstat(component).st_mode):
                raise T4GateError("T4 output parent contains a symlink")
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
            raise T4GateError("T4 output parent changed during publication")
        os.link(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        published = True
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
        current_parent = os.lstat(path.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise T4GateError("T4 output parent changed during publication")
        return file_status.st_dev, file_status.st_ino
    except T4PublicationUncertain:
        raise
    except BaseException as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            raise T4PublicationUncertain(
                _collector._preserved_artifacts(
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


def _rename_directory_new(
    source: Path, destination: Path, *, parent_fd: int | None = None
) -> tuple[int, int]:
    source, destination = source.absolute(), destination.absolute()
    if parent_fd is None and source.parent != destination.parent:
        raise T4GateError("staging and destination must share a parent")
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
            raise T4GateError("atomic no-clobber directory publication is unavailable")
        current_parent = os.lstat(destination.parent)
        if (parent_status.st_dev, parent_status.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise T4GateError("T4 output parent changed during publication")
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
            cause = T4GateError("T4 output parent changed during publication")
            raise T4PublicationUncertain(
                _collector._preserved_artifacts(
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
                _collector._preserved_artifacts(
                    directory,
                    destination.parent,
                    [(destination.name, source_witness, True)],
                )
            ) from exc
        raise
    finally:
        os.close(directory)


def _write_staged_bytes(path: Path, data: bytes, *, parent_fd: int | None = None) -> None:
    descriptor = os.open(
        path.name if parent_fd is not None else path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=parent_fd,
    )
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("T4 staged source write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise T4GateError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, label: str) -> float:
    converted = _finite(value, label)
    if converted < 0.0:
        raise T4GateError(f"{label} must be non-negative")
    return converted


def _raw_metrics_uncached(evidence: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    sources = evidence.get("sources")
    if not isinstance(sources, Mapping):
        raise T4GateError("raw sources are missing")
    required = RAW_SOURCE_NAMES
    labels = {
        "config": "candidate config", "run_manifest": "run manifest", "time_log": "time log",
        "gpu_samples": "gpu samples", "query_measurements": "query measurements",
        "final_map_inventory": "final map inventory", "protocol": "collection protocol", "shortlist": "shortlist",
    }
    bound: dict[str, tuple[Path, bytes]] = {}
    for key in required:
        if key not in sources:
            raise T4GateError(f"{labels[key]} raw source is missing")
        bound[key] = _bound(sources[key], labels[key])
        flat_hash = sources.get(f"{key}_sha256")
        if flat_hash != sources[key]["sha256"]:
            raise T4GateError(f"{labels[key]} hash binding mismatch")
    expected_source_keys = set(required) | {f"{key}_sha256" for key in required}
    if set(sources) != expected_source_keys:
        raise T4GateError("raw source inventory is not exact")
    candidate = evidence.get("candidate_id")
    config_sha = evidence.get("config_sha256")
    run, _ = _load(bound["run_manifest"][0], "run manifest")
    config, config_data = _load(bound["config"][0], "candidate config")
    if not isinstance(candidate, str) or not isinstance(config_sha, str):
        raise T4GateError("evidence candidate identity is invalid")
    try:
        actual_runner, actual_runner_path = _actual_runner(
            run,
            bound["run_manifest"][0],
            bound["config"][0],
            config,
            config_data,
            candidate,
            config_sha,
        )
    except (T4CollectionError, OSError) as exc:
        raise T4GateError("evidence is not bound to one whole-profile run") from exc
    if run.get("manifest_id") == "oviv2_tesse_t4_measurement_run_v1":
        if run.get("candidate_config") != sources["config"]:
            raise T4GateError("measurement run candidate config binding mismatch")
        for role, source_key in (
            ("collection_protocol", "protocol"), ("shortlist", "shortlist"),
            ("time_log", "time_log"), ("gpu_samples", "gpu_samples"),
            ("query_measurements", "query_measurements"),
            ("final_map_inventory", "final_map_inventory"),
        ):
            if run.get(role) != sources[source_key]:
                raise T4GateError(f"measurement run {role} binding mismatch")
    frames = actual_runner.get("processed_frame_count")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise T4GateError("run manifest processed-frame count is invalid")
    elapsed, peak_ram = parse_time_v(bound["time_log"][0])

    gpu, _ = _load(bound["gpu_samples"][0], "gpu samples")
    try:
        _, memories = _validate_gpu_evidence(gpu)
    except T4CollectionError as exc:
        raise T4GateError(str(exc).replace("GPU", "gpu")) from exc

    query, _ = _load(bound["query_measurements"][0], "query measurements")
    expected_query_keys = {
        "schema_version", "manifest_id", "baseline", "scene_id", "query_count",
        "warmup_count", "measured_repeats", "repeat_count", "entity_count", "model",
        "text_model", "query_vocabulary_sha256", "device", "initialization_s",
        "evaluation_io_s", "latencies_ms", "samples", "query_mean_ms", "query_p50_ms",
        "query_p95_ms", "summary", "sources", "protocol", "query_protocol",
    }
    if set(query) != expected_query_keys:
        raise T4GateError("OVIV2 query evidence schema is not exact")
    raw_latencies = query.get("latencies_ms")
    if query.get("warmup_count") != 10 or query.get("measured_repeats") != 5 or not isinstance(raw_latencies, list) or not raw_latencies:
        raise T4GateError("query raw latency protocol is invalid")
    latencies = [_nonnegative(item, "query latency") for item in raw_latencies]
    if not (
        query.get("schema_version") == 1
        and query.get("manifest_id") == "oviv2-query-measurement-v1"
        and query.get("baseline") == "oviv2" and query.get("model") == "ViT-H-14"
        and query.get("warmup_count") == 10 and query.get("measured_repeats") == 5
        and query.get("repeat_count") == 5
    ):
        raise T4GateError("OVIV2 query evidence identity is invalid")
    samples_rows = query.get("samples")
    if not isinstance(samples_rows, list) or len(samples_rows) != len(latencies) or [
        _nonnegative(item.get("latency_ms"), "query sample latency")
        for item in samples_rows if isinstance(item, Mapping)
    ] != latencies:
        raise T4GateError("query samples disagree with raw latencies")
    query_sources = query.get("sources")
    if not isinstance(query_sources, Mapping) or set(query_sources) != {
        "snapshot", "entities", "queries", "checkpoint"
    }:
        raise T4GateError("query raw source inventory is invalid")
    for role in query_sources:
        _bound(query_sources[role], f"query {role}")
    vocabulary, _ = _load(Path(query_sources["queries"]["path"]), "query vocabulary")
    nested = vocabulary.get("vocabulary")
    classes = vocabulary.get("classes") if "classes" in vocabulary else (
        nested.get("classes") if isinstance(nested, Mapping) else None
    )
    query_count = query.get("query_count")
    entity_count = query.get("entity_count")
    if (
        not isinstance(classes, list) or not classes
        or isinstance(query_count, bool) or not isinstance(query_count, int) or query_count != len(classes)
        or isinstance(entity_count, bool) or not isinstance(entity_count, int) or entity_count <= 0
        or len(latencies) != query_count * 5
    ):
        raise T4GateError("query raw coverage is invalid")
    for index, sample in enumerate(samples_rows):
        if not isinstance(sample, Mapping) or set(sample) != {
            "repeat_index", "query_index", "latency_ms", "top_entity_index"
        } or not (
            sample.get("repeat_index") == index // query_count
            and sample.get("query_index") == index % query_count
            and isinstance(sample.get("top_entity_index"), int)
            and not isinstance(sample.get("top_entity_index"), bool)
            and 0 <= sample["top_entity_index"] < entity_count
        ):
            raise T4GateError("query repeat-major samples are invalid")

    inventory, _ = _load(bound["final_map_inventory"][0], "final map inventory")
    files = inventory.get("files")
    if inventory.get("manifest_id") != "oviv2_tesse_t4_final_map_inventory_v1" or not isinstance(files, list):
        raise T4GateError("final map inventory schema is invalid")
    if [item.get("role") for item in files if isinstance(item, Mapping)] != ["snapshot", "entities"]:
        raise T4GateError("final map inventory must contain exact snapshot/entities")
    total_bytes = 0
    source_paths = set()
    source_identities = set()
    for item in files:
        path, data = _bound({key: item[key] for key in ("path", "sha256", "byte_count")}, f"final map {item['role']}")
        if path in source_paths or bound["run_manifest"][0].parent not in path.parents:
            raise T4GateError("final map inventory is spliced or aliases a source")
        identity = (os.lstat(path).st_dev, os.lstat(path).st_ino)
        if identity in source_identities:
            raise T4GateError("final map inventory roles alias the same inode")
        source_paths.add(path)
        source_identities.add(identity)
        total_bytes += len(data)
    if inventory.get("total_bytes") != total_bytes:
        raise T4GateError("final map inventory byte total is stale")
    by_role = {item["role"]: {key: item[key] for key in ("path", "sha256", "byte_count")} for item in files}
    try:
        final_files = _resolve_final_map(actual_runner, actual_runner_path.parent)
        expected_inventory = build_final_map_inventory(
            actual_runner_path.parent, final_files
        )
    except (T4CollectionError, OSError) as exc:
        raise T4GateError("actual runner final current map is invalid") from exc
    if inventory != expected_inventory:
        raise T4GateError("final current map is not bound to the actual runner")
    if query["sources"]["snapshot"] != by_role["snapshot"] or query["sources"]["entities"] != by_role["entities"]:
        raise T4GateError("query evidence is not bound to the final current map")
    collection_protocol, _ = _load(bound["protocol"][0], "collection protocol")
    query_protocol = collection_protocol.get("query")
    if not isinstance(query_protocol, Mapping) or not (
        query.get("query_protocol") == evidence["sources"]["protocol"]
        and query["sources"]["queries"]["sha256"] == query_protocol.get("vocabulary_sha256")
        and query["sources"]["checkpoint"]["sha256"] == query_protocol.get("checkpoint_sha256")
        and query.get("query_vocabulary_sha256") == query_protocol.get("vocabulary_sha256")
        and query.get("text_model") == {"model_id": "ViT-H-14", "checkpoint_sha256": query_protocol.get("checkpoint_sha256")}
    ):
        raise T4GateError("query sources differ from frozen protocol")
    metrics = {
        "total_runtime_s_per_frame": elapsed / frames,
        "query_mean_ms": sum(latencies) / len(latencies),
        "query_p95_ms": _percentile(latencies, 95.0),
        "peak_gpu_gb": max(memories) / 1000.0,
        "peak_ram_gb": peak_ram,
        "final_map_mb": total_bytes / 1_000_000.0,
    }
    return metrics, {"bound": bound, "run": run}


def _raw_metrics(evidence: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    snapshot: dict[Path, bytes] = {}
    local_token = _READ_SNAPSHOT.set(snapshot)
    collector_token = _collector._READ_SNAPSHOT.set(snapshot)
    try:
        return _raw_metrics_uncached(evidence)
    finally:
        _collector._READ_SNAPSHOT.reset(collector_token)
        _READ_SNAPSHOT.reset(local_token)


def verify_t4_gate(evidence_paths: Sequence[Path] | Mapping[str, Any], shortlist_path: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    if isinstance(evidence_paths, Mapping):
        sources = evidence_paths.get("sources", {})
        if shortlist_path is None:
            shortlist_path = Path(sources.get("shortlist", {}).get("path", ""))
        if shortlist_path is None or not shortlist_path.is_file():
            raise T4GateError("evidence mapping lacks a bound shortlist")
        temporary = Path(tempfile.mkdtemp(prefix="oviv2-t4-verify-"))
        succeeded = False
        try:
            evidence_path = temporary / "evidence.json"
            _write_new(evidence_path, evidence_paths)
            destination = output or temporary / "matrix.json"
            result = verify_t4_gate([evidence_path], shortlist_path, destination)
            succeeded = True
            return result
        finally:
            if succeeded:
                shutil.rmtree(temporary)
    paths = [Path(path).absolute() for path in evidence_paths]
    if shortlist_path is None or output is None:
        raise T4GateError("shortlist and matrix output are required")
    shortlist_path, output = shortlist_path.absolute(), output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    for component in (output.parent, *output.parent.parents):
        if stat.S_ISLNK(os.lstat(component).st_mode):
            raise T4GateError("T4 output parent contains a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    shortlist, shortlist_data = _load(shortlist_path, "shortlist")
    ids = shortlist.get("shortlisted_candidate_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise T4GateError("shortlist candidate inventory is invalid")
    shortlist_rows = shortlist.get("shortlisted_candidates")
    if not isinstance(shortlist_rows, list) or [
        row.get("candidate_id") for row in shortlist_rows if isinstance(row, Mapping)
    ] != ids:
        raise T4GateError("shortlist candidate bindings are invalid")
    if len(paths) != len(ids):
        raise T4GateError("evidence must exactly cover shortlist")
    loaded: dict[str, tuple[dict[str, Any], bytes, Path]] = {}
    for path in paths:
        evidence, data = _load(path, "T4 evidence")
        candidate = evidence.get("candidate_id")
        if not isinstance(candidate, str) or candidate in loaded:
            raise T4GateError("duplicate or invalid evidence candidate")
        loaded[candidate] = (evidence, data, path)
    if set(loaded) != set(ids):
        raise T4GateError("evidence candidate inventory differs from shortlist")

    protocol_hash = None
    rows, protocol_candidates = {}, {}
    seen_run_sources: set[tuple[str, str]] = set()
    source_dir = output.with_name(f"{output.stem}.sources")
    staging_dir = output.with_name(
        f".{output.stem}.sources.staging-{os.getpid()}-{time.time_ns()}"
    )
    protocol_path = output.with_name(f"{output.stem}.protocol.json")
    if source_dir.exists() or source_dir.is_symlink() or protocol_path.exists() or protocol_path.is_symlink() or staging_dir.exists():
        raise FileExistsError(source_dir if source_dir.exists() else protocol_path)
    source_published = False
    protocol_published = False
    output_published = False
    publication_directory = os.open(
        output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    staging_directory = -1
    staging_created = False
    staging_witness: tuple[int, int] | None = None
    source_witness: tuple[int, int] | None = None
    protocol_witness: tuple[int, int] | None = None
    output_witness: tuple[int, int] | None = None
    try:
        os.mkdir(staging_dir.name, dir_fd=publication_directory)
        staging_created = True
        staging_directory = os.open(
            staging_dir.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=publication_directory,
        )
        status = os.stat(staging_dir.name, dir_fd=publication_directory, follow_symlinks=False)
        staging_witness = (status.st_dev, status.st_ino)
        for candidate in ids:
            evidence, _, _ = loaded[candidate]
            sources = evidence.get("sources")
            if not isinstance(sources, Mapping):
                raise T4GateError(f"{candidate} shortlist config source binding mismatch")
            config_path, config_data = _bound(sources.get("config"), "candidate config")
            selected_config, _ = _load(config_path, "candidate config")
            try:
                _bind_shortlist_candidate(
                    shortlist_path,
                    candidate_id=candidate,
                    config_path=config_path,
                    config=selected_config,
                    config_data=config_data,
                    config_sha=str(evidence.get("config_sha256")),
                )
            except (T4CollectionError, OSError) as exc:
                raise T4GateError(
                    f"{candidate} shortlist config candidate identity binding mismatch"
                ) from exc
            if evidence.get("units") != UNITS:
                raise T4GateError(f"{candidate} units are not exact")
            source_records = evidence.get("sources")
            shortlist_record = source_records.get("shortlist") if isinstance(source_records, Mapping) else None
            protocol_record = source_records.get("protocol") if isinstance(source_records, Mapping) else None
            if shortlist_record != _record(shortlist_path, shortlist_data):
                raise T4GateError(f"{candidate} shortlist hash binding mismatch")
            _, protocol_data = _bound(protocol_record, "collection protocol")
            current_protocol_hash = hashlib.sha256(protocol_data).hexdigest()
            if protocol_hash is None:
                protocol_hash = current_protocol_hash
            elif protocol_hash != current_protocol_hash:
                raise T4GateError("candidates use different collection protocol hashes")
            collection_protocol = json.loads(protocol_data)
            try:
                _validate_protocol(collection_protocol, Path(protocol_record["path"]))
            except Exception as exc:
                raise T4GateError("collection protocol is not frozen") from exc
            if not (
                collection_protocol.get("dataset") == "TESSE-CD"
                and collection_protocol.get("method_id") == "OVIV2"
                and collection_protocol.get("protocol_id") == "oviv2-tessecd-v2"
                and collection_protocol.get("scene") == "apartment"
                and collection_protocol.get("sample_interval_ms") == 200
                and collection_protocol.get("query", {}).get("warmup_count") == 10
                and collection_protocol.get("query", {}).get("measured_repeats") == 5
                and collection_protocol.get("units") == UNITS
                and collection_protocol.get("bounds") == BOUNDS
            ):
                raise T4GateError("collection protocol is not the frozen Apartment protocol")
            metrics, raw_context = _raw_metrics(evidence)
            config_sha = evidence["config_sha256"]
            raw_records: dict[str, Any] = {}
            for raw_name in RAW_SOURCE_NAMES:
                if raw_name == "shortlist":
                    raw_records[raw_name] = _record(shortlist_path, shortlist_data)
                    continue
                raw_data = raw_context["bound"][raw_name][1]
                staged_raw = staging_dir / f"{candidate}-raw-{raw_name}"
                logical_raw = source_dir / staged_raw.name
                _write_staged_bytes(staged_raw, raw_data, parent_fd=staging_directory)
                raw_records[raw_name] = _record(logical_raw, raw_data)
            raw_sources = {
                **raw_records,
                **{
                    f"{name}_sha256": record["sha256"]
                    for name, record in raw_records.items()
                },
            }
            run_record = dict(raw_sources["run_manifest"])
            run_hash = run_record["sha256"]
            run_identity = (run_record["path"], run_hash)
            if run_identity in seen_run_sources:
                raise T4GateError("shortlisted candidates splice the same whole-profile run")
            seen_run_sources.add(run_identity)
            metric_records = {}
            for metric in METRICS:
                metric_path = source_dir / f"{candidate}-{metric}.json"
                staging_path = staging_dir / metric_path.name
                payload = {
                    "schema_version": 1, "manifest_id": "oviv2_tesse_t4_metric_v1",
                    "scene": "apartment", "candidate_id": candidate,
                    "config_sha256": config_sha, "run_manifest_sha256": run_hash,
                    "metric": metric, "value": metrics[metric],
                }
                metric_data = _canonical(payload) + b"\n"
                _write_new(staging_path, payload, parent_fd=staging_directory)
                metric_records[metric] = _record(metric_path, metric_data)
            gates = {metric: metrics[metric] <= BOUNDS[metric] for metric in METRICS}
            rows[candidate] = {
                "status": "PASS" if all(gates.values()) else "FAIL",
                "config_sha256": config_sha, "run_manifest_sha256": run_hash,
                "metrics": metrics, "gates": gates,
            }
            protocol_candidates[candidate] = {
                "config_sha256": config_sha, "run_manifest": run_record,
                "metric_sources": metric_records,
                "raw_sources": raw_sources,
            }
        protocol_payload = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_t4_protocol_v1",
            "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment", "bounds": BOUNDS, "candidates": protocol_candidates,
        }
        source_witness = _rename_directory_new(
            staging_dir, source_dir, parent_fd=publication_directory
        )
        source_published = True
        protocol_witness = _write_new(
            protocol_path, protocol_payload, parent_fd=publication_directory
        )
        protocol_published = True
        matrix = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_t4_matrix_v1",
            "status": "PASS" if any(row["status"] == "PASS" for row in rows.values()) else "FAIL",
            "shortlist": _record(shortlist_path, shortlist_data), "protocol": _record(protocol_path),
            "candidates": rows,
        }
        matrix["root_sha256"] = hashlib.sha256(_canonical(matrix)).hexdigest()
        output_witness = _write_new(output, matrix, parent_fd=publication_directory)
        output_published = True
        return matrix
    except BaseException as exc:
        if staging_created:
            directory_name = source_dir.name if source_published else staging_dir.name
            directory_witness = source_witness if source_published else staging_witness
            preserved = _collector._preserved_artifacts(
                publication_directory,
                output.parent,
                [
                    (directory_name, directory_witness, True),
                    (protocol_path.name, protocol_witness, protocol_published),
                    (output.name, output_witness, output_published),
                ],
            )
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
        os.close(publication_directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = sorted(args.evidence_root.glob("*/evidence.json"))
    matrix = verify_t4_gate(evidence, args.shortlist, args.output)
    print(json.dumps({"status": matrix["status"], "output": str(args.output.absolute())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
