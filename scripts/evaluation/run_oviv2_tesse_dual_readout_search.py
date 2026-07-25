#!/usr/bin/env python3
"""Run the predeclared OVIV2 dual-readout ablations on Apartment only."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    RUNNER_SCENE_CONFIG_FIELDS,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from src.oviv2.temporal_config import (  # noqa: E402
    ExecutionProfile,
    temporal_config_from_json,
    temporal_config_to_json,
)


CommandBuilder = Callable[[Path, Path, str], tuple[str, ...]]
_CANDIDATE_IDS = ("a0", "a1", "a2", "a3", "a4")
_METRIC_DIRECTIONS = {
    "background_f5_cm": "maximize",
    "change_f1": "maximize",
    "current_miou": "maximize",
    "dynamic_f1": "maximize",
    "ghost_rate": "minimize",
    "object_f1": "maximize",
    "recovery_frames": "minimize",
    "runtime_seconds": "minimize",
}
_PROMOTION_ORDER = [
    "t1_exact_gate",
    "current_miou_floor",
    "object_f1_floor",
    "ghost_rate",
    "background_f5_cm",
    "recovery_frames",
    "dynamic_f1",
    "change_f1",
    "runtime_seconds",
    "config_sha256",
]
_INPUT_BINDING_FIELDS = (
    "dense_manifest",
    "evaluation_checkpoint_frames_sha256",
    "export_manifest",
    "frontend_manifest",
    "input_manifest",
    "occlusion_target_manifest_sha256",
    "schedule_manifest",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


def _read_regular_bytes(path: Path, label: str) -> bytes:
    absolute = path.absolute()
    _reject_symlink_components(absolute, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    witness = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    current = os.stat(absolute, follow_symlinks=False)
    if witness(before) != witness(after) or witness(after) != witness(current):
        raise ValueError(f"{label} changed while it was read")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return data


def _decode_json(data: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON must be UTF-8: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular_bytes(path, label)
    return _decode_json(data, path), data


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _profile_components(profile: ExecutionProfile) -> dict[str, str]:
    return {
        "association_mode": profile.association_mode,
        "background_mode": profile.background_mode,
        "geometry_mode": profile.geometry_mode,
        "lifecycle_mode": profile.lifecycle_mode,
        "motion_mode": profile.motion_mode,
    }


def _validate_bounds(bounds: object) -> dict[str, list[float | int]]:
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("parameter_bounds must be a non-empty object")
    result: dict[str, list[float | int]] = {}
    for name, interval in bounds.items():
        if not isinstance(name, str) or "." not in name:
            raise ValueError("parameter bound names must be group.field strings")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"parameter bound must have two endpoints: {name}")
        lower, upper = interval
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in interval
        ):
            raise ValueError(f"parameter bounds must be finite numbers: {name}")
        if lower > upper:
            raise ValueError(f"parameter bound is reversed: {name}")
        result[name] = interval
    return result


def _validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "manifest_id",
            "dataset",
            "method_id",
            "protocol_id",
            "development_scene",
            "transfer_scene",
            "transfer_policy",
            "minimum_available_ram_bytes_per_candidate",
            "parameter_bounds",
            "metric_directions",
            "metric_policy",
            "promotion_order",
            "hard_gates",
            "candidates",
        },
        "search manifest",
    )
    if (
        payload["schema_version"] != 1
        or payload["manifest_id"] != "oviv2-tesse-dual-readout-search-v1"
        or payload["dataset"] != "TESSE-CD"
        or payload["method_id"] != "OVIV2"
        or payload["protocol_id"] != "oviv2-tessecd-v2"
        or payload["development_scene"] != "apartment"
        or payload["transfer_scene"] != "office"
        or payload["transfer_policy"] != "bind_only_never_execute"
    ):
        raise ValueError("search manifest identity or scene policy mismatch")
    ram = payload["minimum_available_ram_bytes_per_candidate"]
    if isinstance(ram, bool) or not isinstance(ram, int) or ram <= 0:
        raise ValueError("minimum_available_ram_bytes_per_candidate must be positive")
    if payload["metric_directions"] != _METRIC_DIRECTIONS:
        raise ValueError("metric_directions do not match the frozen contract")
    if payload["metric_policy"] != {
        "structured_fields": ["available", "value", "reason", "source"],
        "selection_required": [
            "current_miou",
            "object_f1",
            "ghost_rate",
            "background_f5_cm",
            "recovery_frames",
            "runtime_seconds",
        ],
        "optional_tie_axes": ["dynamic_f1", "change_f1"],
        "optional_availability": (
            "must_match_a0_all_candidates;skip_axis_if_all_unavailable;"
            "reject_partial_availability"
        ),
    }:
        raise ValueError("metric_policy does not match the frozen contract")
    if payload["promotion_order"] != _PROMOTION_ORDER:
        raise ValueError("promotion_order does not match the frozen contract")
    hard_gates = payload["hard_gates"]
    if not isinstance(hard_gates, dict):
        raise ValueError("hard_gates must be an object")
    _exact_keys(
        hard_gates,
        {
            "current_miou_not_below_a0",
            "object_f1_not_below_a0",
            "required_result_gates",
        },
        "hard_gates",
    )
    if (
        hard_gates["current_miou_not_below_a0"] is not True
        or hard_gates["object_f1_not_below_a0"] is not True
        or hard_gates["required_result_gates"]
        != ["correctness", "causality", "determinism", "t1_exact"]
    ):
        raise ValueError("hard_gates do not match the frozen promotion contract")
    bounds = _validate_bounds(payload["parameter_bounds"])
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or [
        item.get("candidate_id") if isinstance(item, dict) else None
        for item in candidates
    ] != list(_CANDIDATE_IDS):
        raise ValueError("candidates must predeclare exactly a0 through a4 in order")
    expected_bound_names: set[str] | None = None
    for candidate in candidates:
        _exact_keys(
            candidate,
            {"candidate_id", "components", "temporal_readout"},
            f"candidate {candidate['candidate_id']}",
        )
        profile = ExecutionProfile.from_id(candidate["candidate_id"])
        if candidate["components"] != _profile_components(profile):
            raise ValueError(f"candidate {profile.profile_id} components are not canonical")
        parsed = temporal_config_from_json(
            {"temporal_readout": candidate["temporal_readout"]}
        )
        if parsed.execution_profile is not profile:
            raise ValueError(f"candidate {profile.profile_id} execution_profile mismatch")
        canonical_temporal = temporal_config_to_json(parsed)["temporal_readout"]
        if candidate["temporal_readout"] != canonical_temporal:
            raise ValueError(f"candidate {profile.profile_id} temporal config is not canonical")
        candidate_bound_names: set[str] = set()
        for group in ("lifecycle", "association", "geometry"):
            values = candidate["temporal_readout"][group]
            for name, value in values.items():
                bound_name = f"{group}.{name}"
                candidate_bound_names.add(bound_name)
                if bound_name not in bounds:
                    raise ValueError(f"candidate parameter has no bound: {bound_name}")
                lower, upper = bounds[bound_name]
                if not lower <= value <= upper:
                    raise ValueError(f"candidate parameter is outside bounds: {bound_name}")
        if expected_bound_names is None:
            expected_bound_names = candidate_bound_names
        elif candidate_bound_names != expected_bound_names:
            raise ValueError("candidate parameter sets differ")
    if set(bounds) != expected_bound_names:
        raise ValueError("parameter_bounds must cover exactly all temporal numeric fields")
    return payload


def load_search_manifest(path: str | Path) -> dict[str, Any]:
    payload, _ = _load_json(Path(path).absolute(), "search manifest")
    return _validate_manifest(payload)


def _validate_base_config(
    path: Path,
    expected_scene: str,
) -> tuple[dict[str, Any], bytes]:
    config, data = _load_json(path.absolute(), f"{expected_scene} base config")
    if (
        config.get("schema_version") != 2
        or config.get("dataset") != "TESSE-CD"
        or config.get("method_id") != "OVIV2"
        or config.get("protocol_id") != "oviv2-tessecd-v2"
        or config.get("scene") != expected_scene
    ):
        raise ValueError(f"{expected_scene} base config identity mismatch")
    temporal_config_from_json({"temporal_readout": config.get("temporal_readout")})
    if config.get("algorithm_hash") != canonical_algorithm_hash(config):
        raise ValueError(f"{expected_scene} base config algorithm_hash is stale")
    return config, data


def _non_temporal_algorithm(config: Mapping[str, Any]) -> dict[str, Any]:
    result = canonical_algorithm_config(config)
    result.pop("temporal_readout", None)
    return result


def non_temporal_config_sha256(config: Mapping[str, Any]) -> str:
    result = dict(config)
    result.pop("algorithm_hash", None)
    result.pop("temporal_readout", None)
    return hashlib.sha256(_canonical_json(result)).hexdigest()


def input_binding_values_sha256(config: Mapping[str, Any]) -> str:
    missing = [name for name in _INPUT_BINDING_FIELDS if name not in config]
    if missing:
        raise ValueError(f"base config is missing input bindings: {missing}")
    bindings = {name: config[name] for name in _INPUT_BINDING_FIELDS}
    return hashlib.sha256(_canonical_json(bindings)).hexdigest()


def _materialize_config(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    config = dict(base)
    config["temporal_readout"] = candidate["temporal_readout"]
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    temporal_config_from_json({"temporal_readout": config["temporal_readout"]})
    if _non_temporal_algorithm(config) != _non_temporal_algorithm(base):
        raise ValueError("candidate changed non-temporal algorithm fields")
    if input_binding_values_sha256(config) != input_binding_values_sha256(base):
        raise ValueError("candidate changed input binding values")
    return config


def _available_ram_bytes(
    meminfo_path: Path = Path("/proc/meminfo"),
) -> int:
    try:
        for line in meminfo_path.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields[:1] != ["MemAvailable:"]:
                continue
            if len(fields) != 3 or fields[2] != "kB":
                raise ValueError("MemAvailable has an invalid unit")
            available = int(fields[1]) * 1024
            if available <= 0:
                raise ValueError("MemAvailable must be positive")
            return available
    except (OSError, UnicodeError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        value = int(pages) * int(page_size)
    except (OSError, ValueError):
        value = 0
    if value <= 0:
        raise RuntimeError("could not determine available RAM")
    return value


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    data = _canonical_json(payload) + b"\n"
    _write_exclusive(temporary, data)
    os.replace(temporary, path)


def _file_record(path: Path) -> dict[str, Any]:
    data = _read_regular_bytes(path, "search artifact")
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _default_command(config: Path, output: Path, candidate_id: str) -> tuple[str, ...]:
    del candidate_id
    return (
        sys.executable,
        str(REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"),
        "--config",
        str(config),
        "--output",
        str(output),
    )


def _validate_gpu_ids(gpu_ids: Sequence[str], max_parallel: int) -> tuple[str, ...]:
    if isinstance(gpu_ids, (str, bytes)):
        raise ValueError("gpu_ids must be a sequence of individual GPU identifiers")
    values = tuple(gpu_ids)
    if (
        not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value or "," in value for value in values)
    ):
        raise ValueError("gpu_ids must be unique non-empty identifiers without commas")
    if len(values) < max_parallel:
        raise ValueError("gpu_ids must provide at least max_parallel devices")
    return values


def run_search(
    *,
    manifest_path: str | Path,
    apartment_base_config: str | Path,
    office_base_config: str | Path,
    output_root: str | Path,
    gpu_ids: Sequence[str],
    max_parallel: int = 1,
    candidate_ids: Sequence[str] | None = None,
    available_ram_bytes: int | None = None,
    command_builder: CommandBuilder | None = None,
) -> Path:
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int):
        raise ValueError("max_parallel must be an integer")
    if not 1 <= max_parallel <= 3:
        raise ValueError("max_parallel must be in 1..3")
    gpus = _validate_gpu_ids(gpu_ids, max_parallel)
    manifest_file = Path(manifest_path).absolute()
    manifest, manifest_bytes = _load_json(manifest_file, "search manifest")
    manifest = _validate_manifest(manifest)
    apartment, apartment_bytes = _validate_base_config(
        Path(apartment_base_config).absolute(), "apartment"
    )
    office, office_bytes = _validate_base_config(
        Path(office_base_config).absolute(), "office"
    )
    if _non_temporal_algorithm(apartment) != _non_temporal_algorithm(office):
        raise ValueError("Apartment and Office non-temporal algorithm configs differ")

    declared = {item["candidate_id"]: item for item in manifest["candidates"]}
    selected_ids = tuple(candidate_ids) if candidate_ids is not None else _CANDIDATE_IDS
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("candidate_ids contain duplicates")
    undeclared = [name for name in selected_ids if name not in declared]
    if undeclared:
        raise ValueError(f"undeclared candidate: {undeclared[0]}")
    if not selected_ids:
        raise ValueError("candidate_ids cannot be empty")

    supplied_ram = (
        _available_ram_bytes() if available_ram_bytes is None else available_ram_bytes
    )
    if isinstance(supplied_ram, bool) or not isinstance(supplied_ram, int) or supplied_ram < 0:
        raise ValueError("available_ram_bytes must be a nonnegative integer")
    required_ram = manifest["minimum_available_ram_bytes_per_candidate"] * min(
        max_parallel, len(selected_ids)
    )
    if supplied_ram < required_ram:
        raise RuntimeError(
            f"available RAM gate failed: need {required_ram} bytes, have {supplied_ram}"
        )

    destination = Path(output_root).absolute()
    _reject_symlink_components(destination.parent, "output parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(f"output root already exists; resume is forbidden: {destination}")
    (destination / "candidates").mkdir()

    office_binding = {
        "scene": "office",
        "executed": False,
        "config_path": str(Path(office_base_config).absolute()),
        "file_sha256": hashlib.sha256(office_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(_canonical_json(office)).hexdigest(),
        "algorithm_hash": office["algorithm_hash"],
    }
    records: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "schema_version": 1,
        "manifest": {
            "path": str(manifest_file),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "apartment_base_config": {
            "path": str(Path(apartment_base_config).absolute()),
            "file_sha256": hashlib.sha256(apartment_bytes).hexdigest(),
        },
        "office_binding": office_binding,
        "max_parallel": max_parallel,
        "gpu_ids": list(gpus),
        "required_available_ram_bytes": required_ram,
        "observed_available_ram_bytes": supplied_ram,
        "status": "RUNNING",
        "candidates": records,
        "unscheduled_candidate_ids": list(selected_ids),
    }
    status_path = destination / "search_status.json"
    _write_status(status_path, status)

    pending = list(selected_ids)
    active: list[dict[str, Any]] = []
    failed = False
    fatal_error: Exception | None = None
    requires_run_manifest = command_builder is None
    builder = command_builder or _default_command
    available_gpus = list(gpus)
    while pending or active:
        while pending and not failed and len(active) < max_parallel:
            candidate_id = pending.pop(0)
            gpu = available_gpus.pop(0)
            candidate_root = destination / "candidates" / candidate_id / "apartment"
            candidate_root.mkdir(parents=True)
            config_path = candidate_root / "config.json"
            run_root = candidate_root / "run"
            config = _materialize_config(apartment, declared[candidate_id])
            config_bytes = _canonical_json(config)
            config_file_bytes = config_bytes + b"\n"
            _write_exclusive(config_path, config_file_bytes)
            command = tuple(builder(config_path, run_root, candidate_id))
            if not command or any(not isinstance(value, str) or not value for value in command):
                raise ValueError("candidate command must be non-empty strings")
            stdout_path = candidate_root / "stdout.log"
            stderr_path = candidate_root / "stderr.log"
            stdout = stdout_path.open("xb")
            stderr = stderr_path.open("xb")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            try:
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    close_fds=True,
                )
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            record = {
                "candidate_id": candidate_id,
                "scene": "apartment",
                "status": "RUNNING",
                "command": list(command),
                "pid": process.pid,
                "exit_code": None,
                "cuda_visible_devices": gpu,
                "config_path": str(config_path),
                "config_file": {
                    "path": str(config_path),
                    "sha256": hashlib.sha256(config_file_bytes).hexdigest(),
                    "byte_count": len(config_file_bytes),
                },
                "output_root": str(run_root),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_file": None,
                "stderr_file": None,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "algorithm_hash": config["algorithm_hash"],
                "non_temporal_config_sha256": non_temporal_config_sha256(config),
                "input_binding_values_sha256": input_binding_values_sha256(config),
                "input_hashes": None,
                "run_identity": None,
                "runtime_seconds": None,
                "failure_reason": None,
            }
            records.append(record)
            active.append(
                {
                    "process": process,
                    "record": record,
                    "stdout": stdout,
                    "stderr": stderr,
                    "gpu": gpu,
                    "started_monotonic": time.monotonic(),
                }
            )
            status["unscheduled_candidate_ids"] = list(pending)
            _write_status(status_path, status)

        if not active:
            break
        if all(item["process"].poll() is None for item in active):
            try:
                active[0]["process"].wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        completed = [item for item in active if item["process"].poll() is not None]
        completed_monotonic = time.monotonic()
        for item in completed:
            process = item["process"]
            record = item["record"]
            exit_code = int(process.returncode)
            item["stdout"].close()
            item["stderr"].close()
            record["exit_code"] = exit_code
            record["status"] = "PASS" if exit_code == 0 else "FAIL"
            record["runtime_seconds"] = max(
                0.0, completed_monotonic - item["started_monotonic"]
            )
            record["stdout_file"] = _file_record(Path(record["stdout_path"]))
            record["stderr_file"] = _file_record(Path(record["stderr_path"]))
            if exit_code == 0:
                run_manifest_path = Path(record["output_root"]) / "run_manifest.json"
                try:
                    if requires_run_manifest and (
                        not run_manifest_path.is_file() or run_manifest_path.is_symlink()
                    ):
                        raise ValueError(
                            f"missing candidate run manifest: {run_manifest_path}"
                        )
                    if run_manifest_path.is_file() and not run_manifest_path.is_symlink():
                        run_manifest, _ = _load_json(
                            run_manifest_path, "candidate run manifest"
                        )
                        source_bindings = run_manifest.get("source_bindings")
                        if not isinstance(source_bindings, dict):
                            raise ValueError(
                                "candidate run manifest source_bindings are invalid"
                            )
                        record["input_hashes"] = source_bindings
                        record["run_identity"] = {
                            name: run_manifest.get(name)
                            for name in (
                                "algorithm_hash",
                                "input_sha256",
                                "code_commit",
                                "source_bindings",
                            )
                        }
                except (FileNotFoundError, ValueError) as exc:
                    record["status"] = "FAIL"
                    record["failure_reason"] = str(exc)
                    failed = True
                    if fatal_error is None:
                        fatal_error = exc
            available_gpus.append(item["gpu"])
            available_gpus.sort(key=gpus.index)
            active.remove(item)
            if exit_code != 0:
                failed = True
        status["unscheduled_candidate_ids"] = list(pending)
        _write_status(status_path, status)

    status["status"] = "FAIL" if failed else "PASS"
    status["unscheduled_candidate_ids"] = list(pending)
    _write_status(status_path, status)
    if fatal_error is not None:
        raise fatal_error
    return status_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--apartment-config", required=True, type=Path)
    parser.add_argument("--office-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", action="append", required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_search(
        manifest_path=args.manifest,
        apartment_base_config=args.apartment_config,
        office_base_config=args.office_config,
        output_root=args.output,
        gpu_ids=tuple(args.gpu),
        max_parallel=args.max_parallel,
    )
    print(json.dumps({"status_path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
