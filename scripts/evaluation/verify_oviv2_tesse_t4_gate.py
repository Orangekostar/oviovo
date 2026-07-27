#!/usr/bin/env python3
"""Recompute OVIV2 TESSE-CD T4 gates from raw, hash-bound sources."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.measure_oviv2_tesse_t4 import (
    METRICS, UNITS, T4CollectionError, _actual_runner, _bind_shortlist_candidate,
    _percentile,
    _resolve_final_map, _validate_protocol, build_final_map_inventory,
    parse_time_v,
)


BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}


class T4GateError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path: Path, label: str) -> bytes:
    path = path.absolute()
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
    data = path.read_bytes()
    after = os.lstat(path)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise T4GateError(f"{label} changed while being read")
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


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for component in (path.parent, *path.parent.parents):
        if stat.S_ISLNK(os.lstat(component).st_mode):
            raise T4GateError("T4 output parent contains a symlink")
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
        raise T4GateError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, label: str) -> float:
    converted = _finite(value, label)
    if converted < 0.0:
        raise T4GateError(f"{label} must be non-negative")
    return converted


def _raw_metrics(evidence: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    sources = evidence.get("sources")
    if not isinstance(sources, Mapping):
        raise T4GateError("raw sources are missing")
    required = ("config", "run_manifest", "time_log", "gpu_samples", "query_measurements", "final_map_inventory", "protocol", "shortlist")
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
        if flat_hash is not None and flat_hash != sources[key]["sha256"]:
            raise T4GateError(f"{labels[key]} hash binding mismatch")
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
    samples = gpu.get("samples")
    pgid = gpu.get("process_group_id")
    if gpu.get("sample_interval_ms") != 200 or not isinstance(samples, list) or not samples:
        raise T4GateError("gpu samples violate sampling protocol")
    phases, memories = set(), []
    timestamps: list[int] = []
    permitted_pgids = set(gpu.get("process_group_ids", [pgid]))
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("process_group_id") not in permitted_pgids:
            raise T4GateError("gpu samples are not PID/process-group bound")
        pid = sample.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise T4GateError("gpu samples contain invalid PID")
        timestamp = sample.get("timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
            raise T4GateError("gpu sample timestamp is invalid")
        timestamps.append(timestamp)
        phases.add(sample.get("phase"))
        sample_memory = _nonnegative(sample.get("used_memory_mib"), "gpu memory")
        processes = sample.get("processes")
        if not isinstance(processes, list) or not processes:
            raise T4GateError("gpu sample process inventory is invalid")
        process_memory = 0.0
        for process in processes:
            if not isinstance(process, Mapping) or set(process) != {
                "gpu_uuid", "pid", "proc_starttime_ticks", "process_group_id", "used_memory_mib"
            }:
                raise T4GateError("gpu process identity schema is invalid")
            if (
                not isinstance(process.get("gpu_uuid"), str) or not process["gpu_uuid"]
                or isinstance(process.get("proc_starttime_ticks"), bool)
                or not isinstance(process.get("proc_starttime_ticks"), int)
                or process.get("proc_starttime_ticks") <= 0
                or process.get("process_group_id") != sample.get("process_group_id")
                or isinstance(process.get("pid"), bool)
                or not isinstance(process.get("pid"), int) or process.get("pid") <= 0
            ):
                raise T4GateError("gpu process identity is invalid")
            process_memory += _nonnegative(process.get("used_memory_mib"), "gpu process memory")
        if process_memory != sample_memory:
            raise T4GateError("gpu process memory aggregate is stale")
        memories.append(sample_memory)
    if not {"mapping", "queries"} <= phases:
        raise T4GateError("gpu samples do not cover mapping and queries")
    if timestamps != sorted(set(timestamps)):
        raise T4GateError("gpu sample timestamps are not strictly monotonic")
    for phase in ("mapping", "queries"):
        phase_times = [sample["timestamp_ns"] for sample in samples if sample["phase"] == phase]
        if any(right - left > 400_000_000 for left, right in zip(phase_times, phase_times[1:])):
            raise T4GateError("gpu sampling interval exceeds the 200 ms protocol tolerance")

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


def verify_t4_gate(evidence_paths: Sequence[Path] | Mapping[str, Any], shortlist_path: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    if isinstance(evidence_paths, Mapping):
        sources = evidence_paths.get("sources", {})
        if shortlist_path is None:
            shortlist_path = Path(sources.get("shortlist", {}).get("path", ""))
        if shortlist_path is None or not shortlist_path.is_file():
            raise T4GateError("evidence mapping lacks a bound shortlist")
        with tempfile.TemporaryDirectory(prefix="oviv2-t4-verify-") as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            _write_new(evidence_path, evidence_paths)
            destination = output or Path(temporary) / "matrix.json"
            return verify_t4_gate([evidence_path], shortlist_path, destination)
    paths = [Path(path).absolute() for path in evidence_paths]
    if shortlist_path is None or output is None:
        raise T4GateError("shortlist and matrix output are required")
    shortlist_path, output = shortlist_path.absolute(), output.absolute()
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
    created: list[Path] = []
    source_dir = output.with_name(f"{output.stem}.sources")
    protocol_path = output.with_name(f"{output.stem}.protocol.json")
    if source_dir.exists() or source_dir.is_symlink() or protocol_path.exists() or protocol_path.is_symlink():
        raise FileExistsError(source_dir if source_dir.exists() else protocol_path)
    try:
        source_dir.mkdir(parents=True)
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
            metrics, _ = _raw_metrics(evidence)
            config_sha = evidence["config_sha256"]
            run_record = dict(evidence["sources"]["run_manifest"])
            run_hash = run_record["sha256"]
            run_identity = (run_record["path"], run_hash)
            if run_identity in seen_run_sources:
                raise T4GateError("shortlisted candidates splice the same whole-profile run")
            seen_run_sources.add(run_identity)
            metric_records = {}
            for metric in METRICS:
                metric_path = source_dir / f"{candidate}-{metric}.json"
                payload = {
                    "schema_version": 1, "manifest_id": "oviv2_tesse_t4_metric_v1",
                    "scene": "apartment", "candidate_id": candidate,
                    "config_sha256": config_sha, "run_manifest_sha256": run_hash,
                    "metric": metric, "value": metrics[metric],
                }
                _write_new(metric_path, payload)
                created.append(metric_path)
                metric_records[metric] = _record(metric_path)
            gates = {metric: metrics[metric] <= BOUNDS[metric] for metric in METRICS}
            rows[candidate] = {
                "status": "PASS" if all(gates.values()) else "FAIL",
                "config_sha256": config_sha, "run_manifest_sha256": run_hash,
                "metrics": metrics, "gates": gates,
            }
            protocol_candidates[candidate] = {
                "config_sha256": config_sha, "run_manifest": run_record,
                "metric_sources": metric_records,
            }
        for candidate in ids:
            evidence, original_data, evidence_path = loaded[candidate]
            if _read(evidence_path, "T4 evidence") != original_data:
                raise T4GateError(f"{candidate} evidence changed during verification")
            recomputed, _ = _raw_metrics(evidence)
            if recomputed != rows[candidate]["metrics"]:
                raise T4GateError(f"{candidate} raw sources changed during verification")
        protocol_payload = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_t4_protocol_v1",
            "dataset": "TESSE-CD", "method_id": "OVIV2", "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment", "bounds": BOUNDS, "candidates": protocol_candidates,
        }
        _write_new(protocol_path, protocol_payload)
        created.append(protocol_path)
        matrix = {
            "schema_version": 1, "manifest_id": "oviv2_tesse_t4_matrix_v1",
            "status": "PASS" if any(row["status"] == "PASS" for row in rows.values()) else "FAIL",
            "shortlist": _record(shortlist_path, shortlist_data), "protocol": _record(protocol_path),
            "candidates": rows,
        }
        matrix["root_sha256"] = hashlib.sha256(_canonical(matrix)).hexdigest()
        _write_new(output, matrix)
        return matrix
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        try:
            source_dir.rmdir()
        except OSError:
            pass
        raise


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
