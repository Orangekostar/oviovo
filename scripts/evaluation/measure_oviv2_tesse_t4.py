#!/usr/bin/env python3
"""Collect source-bound OVIV2 TESSE-CD T4 measurements."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
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
FROZEN_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json"
_ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)\s*$", re.MULTILINE)
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)\s*$", re.MULTILINE)


class T4CollectionError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, label: str) -> bytes:
    path = path.absolute()
    for component in (path, *path.parents):
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
    data = path.read_bytes()
    current = os.lstat(path)
    if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) != (
        current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns
    ):
        raise T4CollectionError(f"{label} changed while being read")
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


def _write_new(path: Path, value: object) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    for component in (path.parent, *path.parent.parents):
        if stat.S_ISLNK(os.lstat(component).st_mode):
            raise T4CollectionError("T4 output parent contains a symlink")
    data = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("T4 output write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    finally:
        os.close(descriptor)


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
        with np.load(result["snapshot"], allow_pickle=False) as arrays:
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
    if not (
        protocol.get("schema_version") == 1
        and protocol.get("manifest_id") == "oviv2_tesse_t4_collection_protocol_v1"
        and protocol.get("dataset") == "TESSE-CD" and protocol.get("method_id") == "OVIV2"
        and protocol.get("protocol_id") == "oviv2-tessecd-v2" and protocol.get("scene") == "apartment"
        and protocol.get("office_data_permitted") is False and protocol.get("sample_interval_ms") == 200
        and protocol.get("units") == UNITS
    ):
        raise T4CollectionError("collection protocol identity, scope, or units are invalid")
    query = protocol.get("query")
    if not isinstance(query, Mapping) or not (
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
    if gpu.get("sample_interval_ms") != 200 or not isinstance(gpu.get("samples"), list) or not gpu["samples"]:
        raise T4CollectionError("GPU samples violate the 200 ms protocol")
    pgid = gpu.get("process_group_id")
    permitted_pgids = set(gpu.get("process_group_ids", [pgid]))
    peaks = []
    phases = set()
    for sample in gpu["samples"]:
        if not isinstance(sample, Mapping) or sample.get("process_group_id") not in permitted_pgids:
            raise T4CollectionError("GPU samples are not PID/process-group bound")
        if isinstance(sample.get("pid"), bool) or not isinstance(sample.get("pid"), int):
            raise T4CollectionError("GPU sample PID is invalid")
        sample_memory = _nonnegative(sample.get("used_memory_mib"), "GPU sample memory")
        processes = sample.get("processes")
        if not isinstance(processes, list) or not processes:
            raise T4CollectionError("GPU sample process inventory is missing")
        process_memory = 0.0
        for process in processes:
            if not isinstance(process, Mapping) or set(process) != {
                "gpu_uuid", "pid", "proc_starttime_ticks", "process_group_id", "used_memory_mib"
            }:
                raise T4CollectionError("GPU process identity schema is invalid")
            if (
                not isinstance(process.get("gpu_uuid"), str) or not process["gpu_uuid"]
                or isinstance(process.get("pid"), bool) or not isinstance(process.get("pid"), int) or process["pid"] <= 0
                or isinstance(process.get("proc_starttime_ticks"), bool)
                or not isinstance(process.get("proc_starttime_ticks"), int) or process["proc_starttime_ticks"] <= 0
                or process.get("process_group_id") != sample.get("process_group_id")
            ):
                raise T4CollectionError("GPU process identity is invalid")
            process_memory += _nonnegative(process.get("used_memory_mib"), "GPU process memory")
        if process_memory != sample_memory:
            raise T4CollectionError("GPU process memory aggregate is stale")
        peaks.append(sample_memory)
        phases.add(sample.get("phase"))
    if not {"mapping", "queries"} <= phases:
        raise T4CollectionError("GPU samples must cover mapping and queries")
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
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
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


def _run_sampled(argv: Sequence[str], *, gpu: str, phase: str, time_log: Path | None = None) -> tuple[int, list[dict[str, object]]]:
    command = [str(_TIME), "-v", "-o", str(time_log), *argv] if time_log else list(argv)
    process = subprocess.Popen(command, start_new_session=True)
    pgid = os.getpgid(process.pid)
    samples: list[dict[str, object]] = []
    while process.poll() is None:
        iteration_started = time.monotonic()
        rows = [row for row in _gpu_rows(gpu) if row["process_group_id"] == pgid]
        if rows:
            samples.append({
                "timestamp_ns": time.time_ns(), "phase": phase, "pid": process.pid,
                "process_group_id": pgid, "used_memory_mib": sum(float(row["used_memory_mib"]) for row in rows),
                "processes": rows,
            })
        time.sleep(max(0.0, 0.2 - (time.monotonic() - iteration_started)))
    return process.returncode, samples


def collect_shortlist(protocol_path: Path, shortlist_path: Path, gpu: str, output: Path) -> list[Path]:
    protocol, _ = _json(protocol_path, "T4 protocol")
    _validate_protocol(protocol, protocol_path)
    shortlist, _ = _json(shortlist_path, "shortlist")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if shortlist.get("office_results_read") not in (None, False) or shortlist.get("scenes_read") not in (None, ["apartment"]):
        raise T4CollectionError("Office evidence is forbidden")
    candidates = shortlist.get("shortlisted_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise T4CollectionError("shortlist contains no candidates")
    output.mkdir(parents=True)
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
        replacements = {"{config}": config_record["path"], "{run}": str(run_root), "{gpu}": gpu, "{protocol}": str(protocol_path), "{query_output}": str(candidate_root / "query.json")}
        expand = lambda values: [replacements.get(str(value), str(value)) for value in values]
        code, mapping_samples = _run_sampled(expand(mapping_template), gpu=gpu, phase="mapping", time_log=candidate_root / "time.txt")
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
        code, query_samples = _run_sampled(expand(query_template), gpu=gpu, phase="queries")
        if code:
            raise subprocess.CalledProcessError(code, expand(query_template))
        all_samples = mapping_samples + query_samples
        if not all_samples:
            raise T4CollectionError("no PID-bound GPU samples collected")
        pgids = {sample["process_group_id"] for sample in all_samples}
        gpu_path = candidate_root / "gpu.json"
        _write_new(gpu_path, {"sample_interval_ms": 200, "process_group_id": min(pgids), "process_group_ids": sorted(pgids), "samples": all_samples})
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
