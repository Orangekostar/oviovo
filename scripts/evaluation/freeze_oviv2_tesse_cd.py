#!/usr/bin/env python3
"""Freeze the selected OVIV2 Stage3 TESSE-CD configuration and its inputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shlex
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SCENES = ("apartment", "office")
TUNED_PARAMETERS = {
    "visibility_depth_tolerance_m": (0.05, 0.10, 0.15),
    "absence_negative_support": (0.5, 1.0),
    "ownership_min_net_support": (0.000001, 0.5, 1.0),
}
ALGORITHM_EXCLUDED_FIELDS = frozenset(
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
SCENE_DIFFERENCE_FIELDS = ALGORITHM_EXCLUDED_FIELDS - {"algorithm_hash"}
_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_CONFIG_MARKERS = ("route3", "surface-observation", "stage4", "scannet200")
_OFFICE_METRIC_KEYS = frozenset(
    {"metric_source", "metrics_path", "evaluation_result", "official_metrics"}
)
_ACTIVE_SNAPSHOTS: dict[
    Path, tuple[tuple[int, int, int, int, int], bytes | None, str, int]
] | None = None


class PublicationUncertainError(ValueError):
    pass


@dataclass(frozen=True)
class _TemporaryPublication:
    parent: Path
    name: str
    descriptor: int
    device: int
    inode: int
    byte_count: int
    sha256: str


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_record(path: Path, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, digest, byte_count = _read_snapshot(path, role, collect=True)
    assert raw is not None
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value, {
        "path": str(path),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _load_json(
    path: Path, role: str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    value, record = _load_json_record(path, role)
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise ValueError(f"{role} changed after its binding was verified")
    return value


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = _absolute_path(path)
    for component in (absolute, *absolute.parents):
        try:
            component_status = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(component_status.st_mode):
            raise ValueError(f"{role} must not contain symlink components: {path}")


def _require_regular_file(path: Path, role: str) -> None:
    _reject_symlink_components(path, f"{role} regular non-symlink file")
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{role} must be a regular non-symlink file: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_snapshot(
    path: Path, role: str, *, collect: bool
) -> tuple[bytes | None, str, int]:
    absolute = Path(os.path.abspath(path))
    _require_regular_file(path, role)
    current = os.lstat(absolute)
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if _ACTIVE_SNAPSHOTS is not None and absolute in _ACTIVE_SNAPSHOTS:
        identity, cached_raw, cached_digest, cached_size = _ACTIVE_SNAPSHOTS[absolute]
        if current_identity != identity:
            raise ValueError(f"{role} changed during freeze: {path}")
        if not collect or cached_raw is not None:
            return cached_raw if collect else None, cached_digest, cached_size
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open stable {role} snapshot: {path}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if collect else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{role} must be a regular file")
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{role} changed during freeze: {path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if (
        identity_before != identity_after
        or identity_after != identity_current
        or byte_count != before.st_size
    ):
        raise ValueError(f"{role} changed during freeze: {path}")
    raw = b"".join(chunks) if chunks is not None else None
    digest_hex = digest.hexdigest()
    if _ACTIVE_SNAPSHOTS is not None:
        previous = _ACTIVE_SNAPSHOTS.get(absolute)
        if previous is not None and (
            previous[0] != identity_current
            or previous[2] != digest_hex
            or previous[3] != byte_count
        ):
            raise ValueError(f"{role} changed during freeze: {path}")
        _ACTIVE_SNAPSHOTS[absolute] = (
            identity_current,
            raw if raw is not None else (previous[1] if previous else None),
            digest_hex,
            byte_count,
        )
    return raw, digest_hex, byte_count


def _sha256(path: Path, role: str = "file") -> str:
    _, digest, _ = _read_snapshot(path, role, collect=False)
    return digest


def _json_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value).rstrip(b"\n"))


def _cache_prefix_sha256(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for index, checksum in enumerate(hashes.values()):
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _resolve_path(value: object, repo_root: Path, role: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} path must be a non-empty string")
    path = Path(value).expanduser()
    candidate = repo_root / path if not path.is_absolute() else path
    return _absolute_path(candidate)


def _verify_binding(
    value: object,
    *,
    repo_root: Path,
    role: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} binding must be an object")
    path = _resolve_path(value.get("path"), repo_root, role)
    if expected_path is not None and path != Path(os.path.abspath(expected_path)):
        raise ValueError(f"{role} path binding mismatch")
    expected_hash = value.get("sha256")
    if not _is_sha256(expected_hash):
        raise ValueError(f"{role} binding requires a sha256 hash")
    _, actual_hash, byte_count = _read_snapshot(path, role, collect=False)
    if actual_hash != expected_hash:
        raise ValueError(f"{role} checksum binding mismatch")
    if "byte_count" in value and (
        type(value["byte_count"]) is not int or value["byte_count"] != byte_count
    ):
        raise ValueError(f"{role} byte_count binding mismatch")
    return {"path": str(path), "sha256": actual_hash, "byte_count": byte_count}


def _frozen_file_binding(path: Path, role: str) -> dict[str, Any]:
    absolute = _absolute_path(path)
    _, digest, byte_count = _read_snapshot(absolute, role, collect=False)
    return {
        "path": str(absolute),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _require_complete_file_binding(value: object, role: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not _is_sha256(value.get("sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] < 0
    ):
        raise ValueError(f"{role} artifact binding is incomplete")


def _verify_target_schedule_binding(
    value: object,
    *,
    repo_root: Path,
    target_manifest_dir: Path,
    expected_path: Path,
) -> dict[str, Any]:
    role = "common target schedule"
    if (
        not isinstance(value, dict)
        or set(value) not in (
            {"path", "sha256", "byte_count"},
            {"path", "path_base", "sha256", "byte_count"},
        )
        or not isinstance(value.get("path"), str)
        or not value["path"].strip()
        or not _is_sha256(value.get("sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] < 0
    ):
        raise ValueError(f"{role} artifact binding is incomplete")

    raw_path = Path(value["path"]).expanduser()
    path_base = value.get("path_base")
    if path_base is None:
        path = _resolve_path(raw_path.as_posix(), target_manifest_dir, role)
    elif path_base == "repository":
        if raw_path.is_absolute():
            raise ValueError(f"{role} repository path must be relative")
        if ".." in raw_path.parts:
            raise ValueError(f"{role} repository path must not escape repository root")
        resolved_root = repo_root.resolve(strict=True)
        if resolved_root != repo_root.absolute() or not resolved_root.is_dir():
            raise ValueError("repository root must be a direct directory")
        candidate = resolved_root / raw_path
        try:
            candidate.resolve(strict=True).relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{role} repository path must not escape repository root"
            ) from exc
        path = _absolute_path(candidate)
    elif path_base == "absolute":
        if not raw_path.is_absolute():
            raise ValueError(f"{role} absolute path_base requires an absolute path")
        path = _absolute_path(raw_path)
    else:
        raise ValueError(f"{role} has unknown path_base: {path_base!r}")

    return _verify_binding(
        {
            "path": str(path),
            "sha256": value["sha256"],
            "byte_count": value["byte_count"],
        },
        repo_root=repo_root,
        role=role,
        expected_path=expected_path,
    )


def _git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _inspect_repository(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    parents = _git(repo_root, "show", "-s", "--format=%P", "HEAD").stdout.split()
    clean = not _git(
        repo_root, "status", "--porcelain", "--untracked-files=all"
    ).stdout.strip()
    lineage = _git(repo_root, "rev-parse", STAGE3_LINEAGE_COMMIT).stdout.strip()
    ancestry = _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        STAGE3_LINEAGE_COMMIT,
        "HEAD",
        check=False,
    ).returncode == 0
    return {
        "commit": commit,
        "parents": parents,
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "commit_time_utc": _git(repo_root, "show", "-s", "--format=%cI", "HEAD").stdout.strip(),
        "clean": clean,
        "stage3_lineage_commit": lineage,
        "stage3_is_ancestor": ancestry,
    }


def _validate_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("clean") is not True:
        raise ValueError("freeze requires a clean repository commit")
    if (
        value.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        or value.get("stage3_is_ancestor") is not True
    ):
        raise ValueError("freeze commit must descend from Stage3 commit 47962fb")
    for field in ("commit", "tree"):
        raw = value.get(field)
        if not isinstance(raw, str) or len(raw) != 40:
            raise ValueError(f"repository {field} must be a full git object ID")
    parents = value.get("parents")
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or len(item) != 40 for item in parents
    ):
        raise ValueError("repository parents must be full git object IDs")
    if not isinstance(value.get("commit_time_utc"), str) or not value["commit_time_utc"]:
        raise ValueError("repository commit time is missing")
    return {
        "clean": True,
        "commit": value["commit"],
        "parents": list(value["parents"]),
        "tree": value["tree"],
        "commit_time_utc": value["commit_time_utc"],
        "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
        "stage3_is_ancestor": True,
    }


def _default_environment() -> dict[str, Any]:
    libraries: dict[str, str] = {}
    for package in ("numpy", "open3d", "torch"):
        try:
            libraries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            libraries[package] = "UNAVAILABLE"
    def command_output(command: list[str]) -> list[str]:
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return ["UNAVAILABLE"]
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines if result.returncode == 0 and lines else ["UNAVAILABLE"]

    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    cuda = command_output(["nvcc", "--version"])
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "cuda": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": sorted(gpu),
        "libraries": dict(sorted(libraries.items())),
    }


def _algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in sorted(config)
        if key not in ALGORITHM_EXCLUDED_FIELDS
    }


def _algorithm_hash(config: Mapping[str, Any]) -> str:
    return _json_hash(_algorithm_config(config))


def _find_forbidden(value: object, path: str = "config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_forbidden(key, f"{path}.<key>") or _find_forbidden(
                item, f"{path}.{key}"
            )
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_forbidden(item, f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _FORBIDDEN_CONFIG_MARKERS):
            return path
    return None


def _validate_source_configs(
    apartment: dict[str, Any], office: dict[str, Any]
) -> None:
    for scene, config in zip(SCENES, (apartment, office), strict=True):
        if (
            config.get("schema_version") != 1
            or config.get("method_id") != "OVIV2"
            or config.get("dataset") != "TESSE-CD"
            or config.get("scene") != scene
            or config.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        ):
            raise ValueError(f"{scene} config identity or Stage3 lineage mismatch")
        forbidden = _find_forbidden(config)
        if forbidden is not None:
            raise ValueError(f"forbidden route3/Stage4/ScanNet200 reference at {forbidden}")
        calculated = _algorithm_hash(config)
        if config.get("algorithm_hash") != calculated:
            raise ValueError(f"{scene} config algorithm_hash is stale")
    if set(apartment) != set(office):
        raise ValueError("scene configs must expose the same configuration keys")
    differences = {
        key for key in apartment if apartment.get(key) != office.get(key)
    }
    if not differences <= SCENE_DIFFERENCE_FIELDS:
        invalid = sorted(differences - SCENE_DIFFERENCE_FIELDS)
        raise ValueError(f"Apartment and Office algorithm configuration differs: {invalid}")
    if _algorithm_config(apartment) != _algorithm_config(office):
        raise ValueError("Apartment and Office normalized algorithm configuration differs")


def _finite_metric(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric metric")
    return float(value)


def _candidate_from_config(
    base_config: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = dict(base_config)
    candidate.update(parameters)
    candidate["algorithm_hash"] = _algorithm_hash(candidate)
    return candidate


def _validate_target_manifest(
    binding: object, repo_root: Path, schedule_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_complete_file_binding(binding, "common target manifest")
    target_binding = _verify_binding(
        binding, repo_root=repo_root, role="common target manifest"
    )
    target_path = Path(target_binding["path"])
    target = _load_json(
        target_path,
        "common target manifest",
        expected_sha256=target_binding["sha256"],
    )
    metadata = target.get("metadata")
    arrays = target.get("target_arrays")
    if (
        type(target.get("schema_version")) is not int
        or target.get("schema_version") != 1
        or target.get("manifest_id") != "tesse_cd_common_v2_targets"
        or target.get("dataset") != "TESSE-CD"
        or target.get("status") != "GENERATED"
        or target.get("targets_generated") is not True
        or target.get("prediction_inputs_used") is not False
        or not isinstance(metadata, dict)
        or metadata.get("protocol_complete") is not True
        or type(metadata.get("window_frames")) is not int
        or metadata.get("window_frames") != 450
        or type(metadata.get("voxel_size_m")) not in (int, float)
        or float(metadata.get("voxel_size_m")) != 0.05
        or not isinstance(metadata.get("scenes"), list)
        or set(metadata["scenes"]) != set(SCENES)
        or not isinstance(metadata.get("schedule"), dict)
        or not isinstance(arrays, dict)
        or set(arrays) != {"path", "sha256", "byte_count", "count", "arrays"}
        or not _is_sha256(arrays.get("sha256"))
        or type(arrays.get("byte_count")) is not int
        or type(arrays.get("count")) is not int
        or arrays["count"] <= 0
        or not isinstance(arrays.get("arrays"), dict)
    ):
        raise ValueError("common target manifest is incomplete or unbound")
    target_schedule = _verify_target_schedule_binding(
        metadata["schedule"],
        repo_root=repo_root,
        target_manifest_dir=target_path.parent,
        expected_path=Path(str(schedule_binding["path"])),
    )
    if (
        target_schedule["sha256"] != schedule_binding.get("sha256")
        or target_schedule["byte_count"] != schedule_binding.get("byte_count")
    ):
        raise ValueError("common target schedule binding mismatch")
    arrays_path = _resolve_path(
        arrays.get("path"), target_path.parent, "common target arrays"
    )
    arrays_raw, arrays_sha256, arrays_byte_count = _read_snapshot(
        arrays_path, "common target arrays", collect=True
    )
    assert arrays_raw is not None
    if (
        arrays_sha256 != arrays["sha256"]
        or arrays_byte_count != arrays["byte_count"]
    ):
        raise ValueError("common target arrays checksum binding mismatch")
    declarations = arrays["arrays"]
    try:
        with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as bundle:
            if set(bundle.files) != set(declarations):
                raise ValueError("common target array names disagree with manifest")
            for name in sorted(bundle.files):
                value = np.asarray(bundle[name])
                declaration = declarations.get(name)
                if (
                    not isinstance(declaration, dict)
                    or set(declaration) != {"shape", "dtype", "element_count"}
                    or list(value.shape) != declaration.get("shape")
                    or str(value.dtype) != declaration.get("dtype")
                    or int(value.size) != declaration.get("element_count")
                    or not np.issubdtype(value.dtype, np.integer)
                    or (value.size and not np.all(np.isfinite(value)))
                ):
                    raise ValueError(f"common target array declaration mismatch: {name}")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("common target array"):
            raise
        raise ValueError("common target arrays are not a valid NPZ package") from exc
    if len(declarations) != arrays["count"]:
        raise ValueError("common target array count disagrees with manifest")
    arrays_binding = {
        "path": str(arrays_path),
        "sha256": arrays["sha256"],
        "byte_count": arrays["byte_count"],
        "count": arrays["count"],
        "arrays": {name: declarations[name] for name in sorted(declarations)},
    }
    return target_binding, arrays_binding


def _validate_selection(
    selection_path: Path,
    selection: dict[str, Any],
    *,
    apartment_config: dict[str, Any],
    apartment_config_sha256: str,
    repo_root: Path,
    schedule_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected_grid = {key: list(values) for key, values in TUNED_PARAMETERS.items()}
    selection_keys = {
        "schema_version",
        "manifest_id",
        "dataset",
        "method",
        "method_id",
        "status",
        "scene",
        "base_config_sha256",
        "grid",
        "parameter_grid",
        "candidate_count",
        "selection_rule",
        "common_target_manifest",
        "common_v2_target_manifest_sha256",
        "freeze_bindings",
        "selected_candidate_id",
        "selected_config",
        "selected_config_sha256",
        "selected_maintenance_parameters",
        "selected_metrics",
        "candidates",
    }
    expected_rule = [
        "maximize current_miou",
        "minimize ghost_rate",
        "maximize background_f5_cm",
        "minimize recovery_frames",
        "minimize config_sha256",
    ]
    if (
        set(selection) != selection_keys
        or type(selection.get("schema_version")) is not int
        or selection.get("schema_version") != 1
        or selection.get("manifest_id")
        != "oviv2_tesse_cd_apartment_selection_v1"
        or selection.get("dataset") != "TESSE-CD"
        or selection.get("method") != "OVIV2"
        or selection.get("method_id") != "OVIV2"
        or selection.get("status") != "PASS"
        or selection.get("scene") != "apartment"
        or selection.get("base_config_sha256") != apartment_config_sha256
        or selection.get("grid") != expected_grid
        or type(selection.get("candidate_count")) is not int
        or selection.get("candidate_count") != 18
        or selection.get("selection_rule") != expected_rule
    ):
        raise ValueError("Apartment selection identity mismatch")
    raw_grid = selection.get("parameter_grid")
    if (
        not isinstance(raw_grid, dict)
        or set(raw_grid) != set(expected_grid)
        or any(
            not isinstance(raw_grid[name], list)
            or len(raw_grid[name]) != len(expected_grid[name])
            or any(type(value) is not float for value in raw_grid[name])
            or raw_grid[name] != expected_grid[name]
            for name in expected_grid
        )
    ):
        raise ValueError("selection does not use the predeclared parameter grid")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 18:
        raise ValueError("Apartment selection must contain exactly 18 candidates")
    target_binding, arrays_binding = _validate_target_manifest(
        selection.get("common_target_manifest"), repo_root, schedule_binding
    )
    if (
        selection.get("common_v2_target_manifest_sha256")
        != target_binding["sha256"]
    ):
        raise ValueError("Apartment selection target identity mismatch")

    expected_combinations = {
        tuple((name, value) for name, value in zip(TUNED_PARAMETERS, values, strict=True))
        for values in itertools.product(*TUNED_PARAMETERS.values())
    }
    observed_combinations: set[tuple[tuple[str, Any], ...]] = set()
    validated: list[dict[str, Any]] = []
    candidate_keys = {
        "schema_version",
        "status",
        "scene",
        "candidate_id",
        "parameters",
        "config_sha256",
        "common_target_manifest_sha256",
        "metrics",
        "config",
        "run_identity",
        "evaluator_summary",
        "summary",
    }
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
            raise ValueError(f"candidate {index} must be an object")
        candidate_id = f"candidate-{index:02d}"
        if candidate.get("candidate_id") != candidate_id:
            raise ValueError(f"candidate {index} identity mismatch")
        for artifact_role in (
            "config",
            "run_identity",
            "evaluator_summary",
            "summary",
        ):
            _require_complete_file_binding(
                candidate.get(artifact_role),
                f"candidate {index} {artifact_role}",
            )
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != set(TUNED_PARAMETERS):
            raise ValueError(f"candidate {index} parameter grid is invalid")
        if any(type(parameters[name]) is not float for name in TUNED_PARAMETERS):
            raise ValueError(f"candidate {index} parameter grid requires float values")
        combination = tuple((name, parameters[name]) for name in TUNED_PARAMETERS)
        if combination not in expected_combinations or combination in observed_combinations:
            raise ValueError(f"candidate {index} parameter grid is outside the predeclared sweep")
        observed_combinations.add(combination)
        candidate_config = _candidate_from_config(apartment_config, parameters)
        config_sha256 = _json_hash(candidate_config)
        if candidate.get("config_sha256") != config_sha256:
            raise ValueError(f"candidate {index} config hash mismatch")
        config_binding = _verify_binding(
            candidate.get("config"),
            repo_root=selection_path.parent,
            role=f"candidate config {index}",
        )
        bound_config = _load_json(
            Path(config_binding["path"]),
            f"candidate config {index}",
            expected_sha256=config_binding["sha256"],
        )
        if (
            config_binding["sha256"] != config_sha256
            or bound_config != candidate_config
        ):
            raise ValueError(f"candidate {index} config identity mismatch")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "current_miou",
            "ghost_rate",
            "background_f5_cm",
            "recovery_frames",
        }:
            raise ValueError(f"candidate {index} metric set mismatch")
        normalized_metrics = {
            key: _finite_metric(value, f"candidate {index} {key}")
            for key, value in metrics.items()
        }
        if any(
            not 0.0 <= normalized_metrics[name] <= 1.0
            for name in ("current_miou", "ghost_rate", "background_f5_cm")
        ) or not 0.0 <= normalized_metrics["recovery_frames"] <= 450.0:
            raise ValueError(f"candidate {index} metric domain is invalid")
        if (
            type(candidate.get("schema_version")) is not int
            or candidate.get("schema_version") != 1
            or candidate.get("status") != "PASS"
            or candidate.get("scene") != "apartment"
            or candidate.get("common_target_manifest_sha256") != target_binding["sha256"]
        ):
            raise ValueError(f"candidate {index} summary identity mismatch")
        run_binding = _verify_binding(
            candidate.get("run_identity"),
            repo_root=selection_path.parent,
            role=f"candidate run identity {index}",
        )
        run_identity = _load_json(
            Path(run_binding["path"]),
            f"candidate run identity {index}",
            expected_sha256=run_binding["sha256"],
        )
        if (
            type(run_identity.get("schema_version")) is not int
            or run_identity.get("schema_version") != 1
            or run_identity.get("dataset") != "TESSE-CD"
            or run_identity.get("method_id") != "OVIV2"
            or run_identity.get("mode") != "causal_checkpoints"
            or run_identity.get("scene") != "apartment"
            or run_identity.get("algorithm_hash")
            != candidate_config["algorithm_hash"]
            or run_identity.get("config")
            != {
                "sha256": config_sha256,
                "byte_count": config_binding["byte_count"],
            }
            or run_identity.get("maintenance_parameters")
            != {name: parameters[name] for name in sorted(TUNED_PARAMETERS)}
        ):
            raise ValueError(f"candidate {index} run identity mismatch")
        evaluator_binding = _verify_binding(
            candidate.get("evaluator_summary"),
            repo_root=selection_path.parent,
            role=f"candidate evaluator summary {index}",
        )
        evaluator = _load_json(
            Path(evaluator_binding["path"]),
            f"candidate evaluator summary {index}",
            expected_sha256=evaluator_binding["sha256"],
        )
        evaluator_metrics = evaluator.get("metrics")
        evaluator_sources = evaluator.get("sources")
        if (
            type(evaluator.get("schema_version")) is not int
            or evaluator.get("schema_version") != 1
            or evaluator.get("manifest_id")
            != "tesse_cd_common_v2_scene_summary"
            or evaluator.get("dataset") != "TESSE-CD"
            or evaluator.get("protocol") != "tesse_cd_common_v2"
            or evaluator.get("status") != "PASS"
            or evaluator.get("method") != "OVIV2"
            or evaluator.get("mode") != "causal_checkpoints"
            or evaluator.get("scene") != "apartment"
            or not isinstance(evaluator_metrics, dict)
            or not isinstance(evaluator_sources, dict)
        ):
            raise ValueError(f"candidate {index} evaluator identity mismatch")
        _require_complete_file_binding(
            evaluator_sources.get("target_manifest"),
            f"candidate {index} evaluator target",
        )
        evaluator_target = _verify_binding(
            evaluator_sources.get("target_manifest"),
            repo_root=selection_path.parent,
            role=f"candidate evaluator target {index}",
            expected_path=Path(target_binding["path"]),
        )
        if (
            evaluator_target["sha256"] != target_binding["sha256"]
            or evaluator_target["byte_count"] != target_binding["byte_count"]
        ):
            raise ValueError(f"candidate {index} evaluator target mismatch")
        expected_evaluator_metrics = {
            "current_miou": normalized_metrics["current_miou"],
            "ghost_rate": normalized_metrics["ghost_rate"],
            "background_f5": normalized_metrics["background_f5_cm"],
            "recovery_frames": normalized_metrics["recovery_frames"],
        }
        if any(
            _finite_metric(
                evaluator_metrics.get(name), f"candidate {index} evaluator {name}"
            )
            != value
            for name, value in expected_evaluator_metrics.items()
        ):
            raise ValueError(f"candidate {index} evaluator metric mismatch")
        summary_binding = _verify_binding(
            candidate.get("summary"),
            repo_root=repo_root,
            role=f"candidate summary {index}",
        )
        summary = _load_json(
            Path(summary_binding["path"]),
            f"candidate summary {index}",
            expected_sha256=summary_binding["sha256"],
        )
        expected_summary = {key: value for key, value in candidate.items() if key != "summary"}
        if summary != expected_summary:
            raise ValueError(f"candidate summary {index} content mismatch")
        validated.append(
            {
                "schema_version": 1,
                "status": "PASS",
                "scene": "apartment",
                "candidate_id": candidate_id,
                "parameters": dict(parameters),
                "metrics": normalized_metrics,
                "config_sha256": config_sha256,
                "common_target_manifest_sha256": target_binding["sha256"],
                "config": config_binding,
                "run_identity": run_binding,
                "evaluator_summary": evaluator_binding,
                "summary": summary_binding,
            }
        )
    if observed_combinations != expected_combinations:
        raise ValueError("Apartment selection does not cover the full parameter grid")
    winner = min(
        validated,
        key=lambda item: (
            -item["metrics"]["current_miou"],
            item["metrics"]["ghost_rate"],
            -item["metrics"]["background_f5_cm"],
            item["metrics"]["recovery_frames"],
            item["config_sha256"],
        ),
    )
    if selection.get("selected_config_sha256") != winner["config_sha256"]:
        raise ValueError("selected configuration is not the predeclared lexicographic winner")
    selected_raw = candidates[int(winner["candidate_id"].rsplit("-", 1)[1])]
    if (
        selection.get("selected_candidate_id") != winner["candidate_id"]
        or selection.get("selected_config") != selected_raw["config"]
        or selection.get("selected_maintenance_parameters")
        != winner["parameters"]
        or selection.get("selected_metrics") != selected_raw["metrics"]
    ):
        raise ValueError("selected configuration evidence is inconsistent")
    return winner, validated, target_binding, arrays_binding


def _validate_declared_freeze_bindings(
    selection: Mapping[str, Any],
    *,
    repo_root: Path,
    expected: Mapping[str, Mapping[str, Path]],
) -> dict[str, dict[str, dict[str, Any]]]:
    raw = selection.get("freeze_bindings")
    if not isinstance(raw, dict) or set(raw) != {"shared", "scenes"}:
        raise ValueError("selection freeze_bindings are incomplete")
    raw_shared = raw.get("shared")
    raw_scenes = raw.get("scenes")
    if not isinstance(raw_shared, dict) or not isinstance(raw_scenes, dict):
        raise ValueError("selection freeze_bindings are invalid")
    if set(raw_shared) != set(expected["shared"]) or set(raw_scenes) != set(SCENES):
        raise ValueError("selection freeze_bindings do not cover exact artifacts")
    result: dict[str, dict[str, dict[str, Any]]] = {"shared": {}, "scenes": {}}
    for role, path in expected["shared"].items():
        result["shared"][role] = _verify_binding(
            raw_shared.get(role),
            repo_root=repo_root,
            role=role.replace("_", " "),
            expected_path=path,
        )
    for scene in SCENES:
        declared = raw_scenes.get(scene)
        scene_expected = expected[scene]
        if not isinstance(declared, dict) or set(declared) != set(scene_expected):
            raise ValueError(f"{scene} freeze bindings do not cover exact artifacts")
        result["scenes"][scene] = {}
        for role, path in scene_expected.items():
            result["scenes"][scene][role] = _verify_binding(
                declared.get(role),
                repo_root=repo_root,
                role=f"{scene} {role.replace('_', ' ')}",
                expected_path=path,
            )
    return result


def _expected_binding_paths(
    configs: Mapping[str, Mapping[str, Any]], repo_root: Path
) -> dict[str, Mapping[str, Path]]:
    input_paths = {
        _resolve_path(config.get("input_manifest"), repo_root, f"{scene} input manifest")
        for scene, config in configs.items()
    }
    if len(input_paths) != 1:
        raise ValueError("Apartment and Office must share one input manifest")
    input_manifest = next(iter(input_paths))
    result: dict[str, Mapping[str, Path]] = {
        "shared": {
            "input_manifest": input_manifest,
            "source_manifest": repo_root / "configs/evaluation/manifests/tesse_cd.json",
            "schedule": _resolve_path(
                configs["apartment"].get("schedule_manifest"), repo_root, "schedule"
            ),
            "camera": repo_root / "configs/evaluation/manifests/oviv2_tesse_cd_cache.json",
            "alias_map": repo_root
            / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml",
            "evaluator": repo_root / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
            "common_finalizer": repo_root
            / "scripts/evaluation/finalize_tesse_common_v2.py",
            "official_finalizer": repo_root / "scripts/evaluation/finalize_tesse_t2.py",
        }
    }
    for scene, config in configs.items():
        result[scene] = {
            "database": Path("."),
            "timestamps": Path("."),
            "trajectory": Path("."),
            "export_manifest": _resolve_path(
                config.get("export_manifest"), repo_root, f"{scene} export manifest"
            ),
            "frontend_manifest": _resolve_path(
                config.get("frontend_manifest"), repo_root, f"{scene} frontend manifest"
            ),
            "dense_manifest": _resolve_path(
                config.get("dense_manifest"), repo_root, f"{scene} dense manifest"
            ),
            "vocabulary_json": _resolve_path(
                config.get("vocabulary_json"), repo_root, f"{scene} vocabulary JSON"
            ),
            "vocabulary_txt": _resolve_path(
                config.get("vocabulary_txt"), repo_root, f"{scene} vocabulary TXT"
            ),
        }
    return result


def _validate_input_manifest_and_bindings(
    configs: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _expected_binding_paths(configs, repo_root)
    input_path = paths["shared"]["input_manifest"]
    input_manifest = _load_json(input_path, "runner input manifest")
    if (
        input_manifest.get("schema_version") != 1
        or input_manifest.get("dataset") != "TESSE-CD"
        or input_manifest.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
    ):
        raise ValueError("runner input manifest identity mismatch")
    scenes = input_manifest.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(SCENES):
        raise ValueError("runner input manifest scene set mismatch")
    source_record = input_manifest.get("source_manifest")
    schedule_record = input_manifest.get("schedule_manifest")
    camera_record = input_manifest.get("camera")
    source = _verify_binding(
        source_record, repo_root=repo_root, role="source manifest"
    )
    schedule = _verify_binding(
        schedule_record,
        repo_root=repo_root,
        role="schedule",
        expected_path=paths["shared"]["schedule"],
    )
    camera = _verify_binding(camera_record, repo_root=repo_root, role="camera")
    paths["shared"] = {**paths["shared"], "source_manifest": Path(source["path"]), "camera": Path(camera["path"])}

    scene_records: dict[str, Any] = {}
    for scene in SCENES:
        config = configs[scene]
        locked = scenes[scene]
        if not isinstance(locked, dict):
            raise ValueError(f"{scene} runner input lock must be an object")
        if (
            type(locked.get("frame_count")) is not int
            or locked["frame_count"] != config.get("frame_count")
            or _resolve_path(locked.get("root"), repo_root, f"{scene} RGB-D root")
            != _resolve_path(config.get("dataset_root"), repo_root, f"{scene} dataset root")
        ):
            raise ValueError(f"{scene} RGB-D source identity mismatch")
        export = _verify_binding(
            locked.get("export_manifest"),
            repo_root=repo_root,
            role=f"{scene} export manifest",
            expected_path=paths[scene]["export_manifest"],
        )
        export_payload = _load_json(
            Path(export["path"]),
            f"{scene} export manifest",
            expected_sha256=export["sha256"],
        )
        database_path = _resolve_path(
            export_payload.get("source_database"), repo_root, f"{scene} source database"
        )
        _require_regular_file(database_path, f"{scene} source database")
        database_sha256 = _sha256(database_path)
        if (
            export_payload.get("dataset") != "TESSE-CD"
            or export_payload.get("scene") != scene
            or export_payload.get("frame_count") != locked["frame_count"]
            or not _is_sha256(export_payload.get("combined_output_sha256"))
            or type(export_payload.get("file_hash_count")) is not int
            or export_payload.get("source_database_sha256") != database_sha256
            or locked.get("source_database_sha256") != database_sha256
            or locked["export_manifest"].get("combined_output_sha256")
            != export_payload["combined_output_sha256"]
            or locked["export_manifest"].get("file_hash_count")
            != export_payload["file_hash_count"]
        ):
            raise ValueError(f"{scene} RGB-D export or source database hash mismatch")
        export_source = _resolve_path(
            export_payload.get("source_manifest"), repo_root, f"{scene} export source manifest"
        )
        _require_regular_file(export_source, f"{scene} export source manifest")
        if _sha256(export_source) != source["sha256"]:
            raise ValueError(f"{scene} export source manifest hash mismatch")
        timestamps = _verify_binding(
            locked.get("timestamps"), repo_root=repo_root, role=f"{scene} timestamps"
        )
        trajectory = _verify_binding(
            locked.get("trajectory"), repo_root=repo_root, role=f"{scene} trajectory"
        )
        vocabulary = locked.get("vocabulary")
        if not isinstance(vocabulary, dict):
            raise ValueError(f"{scene} vocabulary lock is missing")
        vocab_json_binding = {
            "path": vocabulary.get("json_path"),
            "sha256": vocabulary.get("json_sha256"),
        }
        vocab_txt_binding = {
            "path": vocabulary.get("txt_path"),
            "sha256": vocabulary.get("txt_sha256"),
        }
        vocab_json = _verify_binding(
            vocab_json_binding,
            repo_root=repo_root,
            role=f"{scene} vocabulary JSON",
            expected_path=paths[scene]["vocabulary_json"],
        )
        vocab_txt = _verify_binding(
            vocab_txt_binding,
            repo_root=repo_root,
            role=f"{scene} vocabulary TXT",
            expected_path=paths[scene]["vocabulary_txt"],
        )
        paths[scene] = {
            **paths[scene],
            "database": database_path,
            "timestamps": Path(timestamps["path"]),
            "trajectory": Path(trajectory["path"]),
        }
        scene_records[scene] = {
            "rgbd": {
                "root": str(_resolve_path(locked["root"], repo_root, f"{scene} root")),
                "frame_count": locked["frame_count"],
                "export_manifest": export,
                "combined_output_sha256": export_payload["combined_output_sha256"],
                "file_hash_count": export_payload["file_hash_count"],
                "source_database": {
                    "path": str(database_path),
                    "sha256": database_sha256,
                    "byte_count": database_path.stat().st_size,
                },
                "timestamps": timestamps,
                "trajectory": trajectory,
                "camera": camera,
            },
            "vocabulary": {"json": vocab_json, "txt": vocab_txt},
        }

    declared = _validate_declared_freeze_bindings(
        selection, repo_root=repo_root, expected=paths
    )
    return (
        {"input_manifest": declared["shared"]["input_manifest"], "source": source, "schedule": schedule, "camera": camera},
        scene_records,
        declared,
    )


def _validate_cache_manifest(
    scene: str,
    config: Mapping[str, Any],
    scene_binding: Mapping[str, Mapping[str, Any]],
    shared_binding: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frontend_path = _resolve_path(config.get("frontend_manifest"), repo_root, "frontend manifest")
    dense_path = _resolve_path(config.get("dense_manifest"), repo_root, "dense manifest")
    frontend_dir = _resolve_path(
        config.get("frontend_cache_dir"), repo_root, "frontend cache directory"
    )
    dense_dir = _resolve_path(
        config.get("dense_cache_dir"), repo_root, "dense cache directory"
    )
    if frontend_path.parent != frontend_dir or dense_path.parent != dense_dir:
        raise ValueError(f"{scene} cache directory does not own its declared manifest")
    frontend = _load_json(
        frontend_path,
        f"{scene} frontend manifest",
        expected_sha256=scene_binding["frontend_manifest"]["sha256"],
    )
    dense = _load_json(
        dense_path,
        f"{scene} dense manifest",
        expected_sha256=scene_binding["dense_manifest"]["sha256"],
    )
    frame_count = config.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError(f"{scene} frame_count must be a positive integer")
    frontend_keys = {
        "schema_version",
        "method",
        "dataset",
        "scene",
        "frame_count",
        "source_frame_ids",
        "source_frame_ids_hash",
        "image_shape",
        "class_count",
        "classes",
        "vocabulary_sha256",
        "algorithm_hash",
        "feature_model_id",
        "input_manifest_sha256",
        "input_witness",
        "provenance_sha256",
        "cache_files_sha256",
        "cache_prefix_sha256",
    }
    dense_keys = {
        "schema_version",
        "method",
        "scene",
        "frame_count",
        "source_frame_ids",
        "image_shape",
        "sample_stride",
        "top_k",
        "class_count",
        "vocabulary_sha256",
        "provenance",
        "cache_files_sha256",
    }
    frontend_provenance_keys = {
        "script",
        "hydra_config",
        "dataset_config",
        "yolo_model",
        "yolo_clip_model",
        "mobile_sam_model",
        "clip_model",
    }
    dense_provenance_keys = {
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
        "cache_prefix_sha256",
    }
    witness_keys = {
        "schema_version",
        "scene",
        "root",
        "validated_frame_count",
        "intrinsics",
        "export_manifest",
        "camera_manifest",
        "schedule_manifest",
        "source_manifest",
        "timestamps",
        "trajectory",
        "first_record",
        "last_record",
    }
    vocabulary_json = _resolve_path(
        config.get("vocabulary_json"), repo_root, f"{scene} vocabulary JSON"
    )
    vocabulary_txt = _resolve_path(
        config.get("vocabulary_txt"), repo_root, f"{scene} vocabulary TXT"
    )
    vocabulary = _load_json(
        vocabulary_json,
        f"{scene} vocabulary JSON",
        expected_sha256=scene_binding["vocabulary_json"]["sha256"],
    )
    classes = vocabulary.get("classes")
    input_manifest = _resolve_path(
        config.get("input_manifest"), repo_root, f"{scene} input manifest"
    )
    witness = frontend.get("input_witness")
    frontend_provenance = frontend.get("provenance_sha256")
    export_payload = _load_json(
        Path(str(scene_binding["export_manifest"]["path"])),
        f"{scene} export manifest",
        expected_sha256=scene_binding["export_manifest"]["sha256"],
    )
    camera_payload = _load_json(
        Path(str(shared_binding["camera"]["path"])),
        "TESSE-CD camera manifest",
        expected_sha256=shared_binding["camera"]["sha256"],
    )

    def witness_matches(
        role: str, expected: Mapping[str, Any], *, require_exact_path: bool = True
    ) -> bool:
        if not isinstance(witness, dict):
            return False
        raw = witness.get(role)
        extra_keys = (
            {
                "combined_output_sha256",
                "validated_combined_output_sha256",
                "file_hash_count",
                "validated_file_hash_count",
            }
            if role == "export_manifest"
            else set()
        )
        return (
            isinstance(raw, dict)
            and {"path", "sha256"} <= set(raw)
            and set(raw) <= {"path", "sha256", "byte_count"} | extra_keys
            and raw.get("sha256") == expected.get("sha256")
            and (
                "byte_count" not in raw
                or raw.get("byte_count") == expected.get("byte_count")
            )
            and (
                not require_exact_path
                or _resolve_path(
                    raw.get("path"), repo_root, f"{scene} witness {role}"
                )
                == Path(str(expected.get("path")))
            )
        )

    witness_valid = isinstance(witness, dict)
    timestamp_rows: list[tuple[int, int, int]] = []
    trajectory_rows: list[list[float]] = []
    expected_intrinsics: dict[str, float | int] | None = None
    if witness_valid:
        try:
            timestamp_raw, _, _ = _read_snapshot(
                Path(str(scene_binding["timestamps"]["path"])),
                f"{scene} timestamps",
                collect=True,
            )
            trajectory_raw, _, _ = _read_snapshot(
                Path(str(scene_binding["trajectory"]["path"])),
                f"{scene} trajectory",
                collect=True,
            )
            assert timestamp_raw is not None and trajectory_raw is not None
            reader = csv.DictReader(io.StringIO(timestamp_raw.decode("utf-8")))
            timestamp_fields = [
                "frame_index",
                "sensor_timestamp_ns",
                "relative_timestamp_ns",
            ]
            if reader.fieldnames != timestamp_fields:
                raise ValueError("unexpected timestamp columns")
            for row in reader:
                if set(row) != set(timestamp_fields) or any(
                    row[field] is None for field in timestamp_fields
                ):
                    raise ValueError("invalid timestamp row")
                timestamp_rows.append(
                    (
                        int(row["frame_index"]),
                        int(row["sensor_timestamp_ns"]),
                        int(row["relative_timestamp_ns"]),
                    )
                )
            for line in trajectory_raw.decode("utf-8").splitlines():
                if not line.strip():
                    raise ValueError("blank trajectory row")
                values = [float(item) for item in line.split()]
                if len(values) != 16 or any(not math.isfinite(item) for item in values):
                    raise ValueError("invalid trajectory row")
                trajectory_rows.append(values)

            camera = camera_payload.get("camera", camera_payload)
            if not isinstance(camera, dict):
                raise ValueError("invalid camera manifest")
            width = camera.get("w", camera.get("width"))
            height = camera.get("h", camera.get("height"))
            if type(width) is not int or type(height) is not int:
                raise ValueError("invalid camera dimensions")
            intrinsic_values = {
                key: camera.get(key) for key in ("fx", "fy", "cx", "cy")
            }
            if any(
                type(item) not in (int, float) or not math.isfinite(float(item))
                for item in intrinsic_values.values()
            ):
                raise ValueError("invalid camera intrinsics")
            expected_intrinsics = {
                "fx": float(intrinsic_values["fx"]),
                "fy": float(intrinsic_values["fy"]),
                "cx": float(intrinsic_values["cx"]),
                "cy": float(intrinsic_values["cy"]),
                "width": width,
                "height": height,
            }
        except (UnicodeError, ValueError):
            witness_valid = False

    if (
        len(timestamp_rows) != frame_count
        or [row[0] for row in timestamp_rows] != list(range(frame_count))
        or len(trajectory_rows) != frame_count
    ):
        witness_valid = False

    record_keys = {
        "camera_to_world_sha256",
        "depth_path",
        "frame_index",
        "relative_timestamp_ns",
        "rgb_path",
        "timestamp_ns",
    }
    dataset_root = _resolve_path(
        config.get("dataset_root"), repo_root, f"{scene} dataset root"
    )

    def record_matches(index: int, raw: object) -> bool:
        if not witness_valid or not isinstance(raw, dict) or set(raw) != record_keys:
            return False
        frame_index, timestamp_ns, relative_timestamp_ns = timestamp_rows[index]
        values = trajectory_rows[index]
        pose = [values[offset : offset + 4] for offset in range(0, 16, 4)]
        rgb_path = dataset_root / "results" / f"frame{index:06d}.jpg"
        depth_path = dataset_root / "results" / f"depth{index:06d}.png"
        if (
            _resolve_path(raw.get("rgb_path"), repo_root, f"{scene} witness RGB")
            != rgb_path
            or _resolve_path(raw.get("depth_path"), repo_root, f"{scene} witness depth")
            != depth_path
        ):
            return False
        try:
            _read_snapshot(rgb_path, f"{scene} witness RGB", collect=False)
            _read_snapshot(depth_path, f"{scene} witness depth", collect=False)
        except ValueError:
            return False
        return (
            raw.get("frame_index") == frame_index
            and raw.get("timestamp_ns") == timestamp_ns
            and raw.get("relative_timestamp_ns") == relative_timestamp_ns
            and raw.get("camera_to_world_sha256") == _json_hash(pose)
        )

    export_witness = witness.get("export_manifest") if witness_valid else None
    export_witness_keys = {
        "path",
        "sha256",
        "combined_output_sha256",
        "validated_combined_output_sha256",
        "file_hash_count",
        "validated_file_hash_count",
    }
    export_witness_valid = (
        isinstance(export_witness, dict)
        and set(export_witness) in (
            export_witness_keys,
            export_witness_keys | {"byte_count"},
        )
        and export_witness.get("combined_output_sha256")
        == export_payload.get("combined_output_sha256")
        and export_witness.get("validated_combined_output_sha256")
        == export_payload.get("combined_output_sha256")
        and export_witness.get("file_hash_count") == export_payload.get("file_hash_count")
        and export_witness.get("validated_file_hash_count")
        == export_payload.get("file_hash_count")
    )

    if (
        set(frontend) != frontend_keys
        or type(frontend.get("schema_version")) is not int
        or frontend.get("schema_version") != 1
        or frontend.get("method") != "OVIV2"
        or frontend.get("dataset") != "TESSE-CD"
        or frontend.get("scene") != scene
        or frontend.get("frame_count") != frame_count
        or frontend.get("source_frame_ids") != list(range(frame_count))
        or frontend.get("source_frame_ids_hash") != _json_hash(list(range(frame_count)))
        or frontend.get("image_shape") != [480, 720]
        or not isinstance(classes, list)
        or frontend.get("classes") != classes
        or frontend.get("class_count") != len(classes)
        or frontend.get("vocabulary_sha256") != _sha256(vocabulary_txt)
        or frontend.get("input_manifest_sha256") != _sha256(input_manifest)
        or not _is_sha256(frontend.get("algorithm_hash"))
        or not _is_sha256(frontend.get("cache_prefix_sha256"))
        or not isinstance(frontend_provenance, dict)
        or frontend.get("feature_model_id")
        != f"clip-sha256:{frontend_provenance.get('clip_model')}"
        or set(frontend_provenance) != frontend_provenance_keys
        or any(not _is_sha256(value) for value in frontend_provenance.values())
        or not witness_valid
        or set(witness) != witness_keys
        or type(witness.get("schema_version")) is not int
        or witness.get("schema_version") != 1
        or witness.get("scene") != scene
        or witness.get("validated_frame_count") != frame_count
        or _resolve_path(witness.get("root"), repo_root, f"{scene} witness root")
        != _resolve_path(config.get("dataset_root"), repo_root, f"{scene} dataset root")
        or not export_witness_valid
        or not witness_matches("export_manifest", scene_binding["export_manifest"])
        or not witness_matches("camera_manifest", shared_binding["camera"])
        or not witness_matches(
            "schedule_manifest", shared_binding["schedule"], require_exact_path=False
        )
        or not witness_matches(
            "source_manifest",
            shared_binding["source_manifest"],
            require_exact_path=False,
        )
        or not witness_matches("timestamps", scene_binding["timestamps"])
        or not witness_matches("trajectory", scene_binding["trajectory"])
        or witness.get("intrinsics") != expected_intrinsics
        or not record_matches(0, witness.get("first_record"))
        or not record_matches(frame_count - 1, witness.get("last_record"))
    ):
        raise ValueError(f"{scene} frontend cache manifest is invalid")
    provenance = dense.get("provenance")
    if (
        set(dense) != dense_keys
        or type(dense.get("schema_version")) is not int
        or dense.get("schema_version") != 1
        or dense.get("method") != "OVIV2-dense-semantic-cache"
        or dense.get("scene") != scene
        or dense.get("frame_count") != frame_count
        or dense.get("source_frame_ids") != list(range(frame_count))
        or dense.get("image_shape") != [480, 720]
        or dense.get("sample_stride") != config.get("dense_sample_stride")
        or dense.get("top_k") != config.get("dense_top_k")
        or dense.get("class_count") != len(classes)
        or dense.get("vocabulary_sha256") != _sha256(vocabulary_json)
        or not isinstance(provenance, dict)
        or set(provenance) != dense_provenance_keys
        or provenance.get("backend") != "radseg"
        or not isinstance(provenance.get("model_id"), str)
        or not isinstance(provenance.get("language_model_id"), str)
        or not isinstance(provenance.get("language_model_revision"), str)
        or any(
            not _is_sha256(provenance.get(key))
            for key in (
                "model_sha256",
                "auxiliary_model_sha256",
                "language_model_sha256",
                "vocabulary_sha256",
                "prompt_sha256",
                "inference_config_sha256",
                "cache_prefix_sha256",
            )
        )
    ):
        raise ValueError(f"{scene} dense cache manifest is invalid")
    frontend_hashes = frontend.get("cache_files_sha256")
    dense_hashes = dense.get("cache_files_sha256")
    expected_frontend_names = [
        f"frame{index:06d}.pkl.gz" for index in range(frame_count)
    ]
    expected_dense_names = [f"frame{index:06d}.npz" for index in range(frame_count)]
    for hashes, names, directory, role in (
        (frontend_hashes, expected_frontend_names, frontend_dir, "frontend"),
        (dense_hashes, expected_dense_names, dense_dir, "dense"),
    ):
        if not isinstance(hashes, dict) or list(hashes) != names:
            raise ValueError(f"{scene} {role} cache checksum keys are not canonical")
        for name in names:
            path = directory / name
            if not _is_sha256(hashes[name]) or _sha256(path) != hashes[name]:
                raise ValueError(f"{scene} {role} cache file checksum mismatch: {name}")
        if {path.name for path in directory.iterdir()} != set(names) | {
            f"{role}_manifest.json"
        }:
            raise ValueError(f"{scene} {role} cache inventory mismatch")
    if (
        frontend["cache_prefix_sha256"] != _cache_prefix_sha256(frontend_hashes)
        or provenance["cache_prefix_sha256"] != _cache_prefix_sha256(dense_hashes)
    ):
        raise ValueError(f"{scene} cache prefix checksum mismatch")
    cache = {
        "frontend_manifest": scene_binding["frontend_manifest"],
        "frontend_algorithm_sha256": frontend["algorithm_hash"],
        "frontend_cache_prefix_sha256": frontend["cache_prefix_sha256"],
        "dense_manifest": scene_binding["dense_manifest"],
        "dense_cache_prefix_sha256": provenance["cache_prefix_sha256"],
    }
    frontend_models = {
        "feature_model_id": frontend["feature_model_id"],
        "provenance_sha256": dict(sorted(frontend["provenance_sha256"].items())),
    }
    dense_models = {
        key: provenance[key]
        for key in (
            "backend",
            "source_commit",
            "radio_commit",
            "model_id",
            "model_sha256",
            "auxiliary_model_sha256",
            "language_model_id",
            "language_model_revision",
            "language_model_sha256",
            "vocabulary_sha256",
            "prompt_sha256",
            "inference_config_sha256",
        )
    }
    return cache, frontend_models, dense_models


def _prepared_manifest_path(output_manifest: Path) -> Path:
    return output_manifest.with_name(
        f"{output_manifest.stem}.prepared{output_manifest.suffix}"
    )


def _preflight_outputs(args: argparse.Namespace) -> tuple[str, Path]:
    apartment = _absolute_path(args.output_apartment_config)
    office = _absolute_path(args.output_office_config)
    final_manifest = _absolute_path(args.output_manifest)
    prepared_manifest = _prepared_manifest_path(final_manifest)
    destinations = (apartment, office, prepared_manifest, final_manifest)
    if len(set(destinations)) != 4:
        raise ValueError("freeze output paths must be distinct")
    for destination in destinations:
        _reject_symlink_components(destination, "freeze output path")
    if os.path.lexists(final_manifest):
        raise FileExistsError(f"freeze output already exists: {final_manifest}")
    config_exists = (os.path.lexists(apartment), os.path.lexists(office))
    output_root = final_manifest.parent
    existing_root = set(output_root.iterdir()) if output_root.exists() else set()
    if config_exists == (False, False):
        if os.path.lexists(prepared_manifest) or existing_root:
            raise FileExistsError(f"prior run output exists under {output_root}")
        return "prepare", prepared_manifest
    if config_exists != (True, True):
        raise FileExistsError("partial frozen config output already exists")
    _require_regular_file(apartment, "prepared Apartment frozen config")
    _require_regular_file(office, "prepared Office frozen config")
    _require_regular_file(prepared_manifest, "prepared freeze manifest")
    if existing_root != {prepared_manifest}:
        raise FileExistsError(f"prior run output exists under {output_root}")
    return "finalize", prepared_manifest


def _verify_configs_in_commit(
    repo_root: Path,
    expected: Mapping[Path, bytes],
    injected_state: Mapping[str, Any] | None,
) -> None:
    if injected_state is not None:
        tracked = injected_state.get("tracked_frozen_config_sha256")
        if not isinstance(tracked, dict) or set(tracked) != {
            str(path) for path in expected
        }:
            raise ValueError("clean commit does not declare both tracked frozen configs")
        if any(
            tracked.get(str(path)) != _sha256_bytes(content)
            for path, content in expected.items()
        ):
            raise ValueError("tracked frozen config hash does not match clean commit")
        return
    for path, content in expected.items():
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError("frozen configs must be tracked inside the repository") from exc
        result = _git(repo_root, "show", f"HEAD:{relative}", check=False)
        if result.returncode != 0:
            raise ValueError(f"tracked frozen config is absent from clean commit: {relative}")
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if committed != content:
            raise ValueError(f"tracked frozen config differs from clean commit: {relative}")


def _office_metric_sources(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    allowed_files: frozenset[Path] = frozenset(),
) -> list[str]:
    found: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_path = f"{path}.{key}"
                lowered = key.lower()
                if key.lower() in _OFFICE_METRIC_KEYS or any(
                    marker in lowered for marker in ("metric", "result", "summary")
                ):
                    found.append(key_path)
                visit(nested, key_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
        elif isinstance(value, str) and any(
            marker in value.lower() for marker in ("metric", "result", "summary")
        ):
            found.append(f"{path}=<string>")

    visit(config, "office_config")
    if output_root.exists():
        found.extend(
            str(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and _absolute_path(path) not in allowed_files
        )
    return sorted(set(found))


def _mapping_commands(
    args: argparse.Namespace, repo_root: Path
) -> tuple[list[str], dict[str, str]]:
    commands: list[str] = []
    roots: dict[str, str] = {}
    root = _absolute_path(args.output_manifest).parent
    config_paths = {
        "apartment": _absolute_path(args.output_apartment_config),
        "office": _absolute_path(args.output_office_config),
    }
    for scene in SCENES:
        for repeat in (1, 2):
            output = root / scene / f"run{repeat}"
            roots[f"{scene}_run{repeat}"] = str(output)
            commands.append(
                shlex.join(
                    [
                        sys.executable,
                        str(repo_root / "scripts/evaluation/run_oviv2_tesse_cd.py"),
                        "--config",
                        str(config_paths[scene]),
                        "--output",
                        str(output),
                        "--freeze-manifest",
                        str(_absolute_path(args.output_manifest)),
                        "--run-slot",
                        f"{scene}_run{repeat}",
                    ]
                )
            )
    return commands, roots


def _link_no_replace(
    source: str,
    destination: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
) -> None:
    os.link(
        source,
        destination,
        src_dir_fd=source_dir_fd,
        dst_dir_fd=destination_dir_fd,
        follow_symlinks=False,
    )


def _revalidate_file_records(
    value: object, *, excluded_paths: frozenset[Path] = frozenset()
) -> None:
    records: dict[Path, tuple[str, int]] = {}

    def visit(item: object) -> None:
        if isinstance(item, dict):
            if (
                isinstance(item.get("path"), str)
                and _is_sha256(item.get("sha256"))
                and type(item.get("byte_count")) is int
            ):
                path = Path(item["path"])
                if path not in excluded_paths:
                    expected = (item["sha256"], item["byte_count"])
                    previous = records.setdefault(path, expected)
                    if previous != expected:
                        raise ValueError(f"conflicting file bindings for {path}")
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    for path, (expected_hash, expected_size) in records.items():
        _, actual_hash, actual_size = _read_snapshot(
            path, f"bound file {path}", collect=False
        )
        if actual_hash != expected_hash or actual_size != expected_size:
            raise ValueError(f"bound file changed during freeze: {path}")


def _revalidate_snapshot_session() -> None:
    if _ACTIVE_SNAPSHOTS is None:
        return
    for path in tuple(_ACTIVE_SNAPSHOTS):
        _read_snapshot(path, f"snapshotted file {path}", collect=False)


def _publish_json_transaction(
    publications: Sequence[tuple[Path, bytes]],
    *,
    empty_root_guard: Path,
    allowed_root_entries: frozenset[Path] = frozenset(),
) -> None:
    guard = _absolute_path(empty_root_guard)
    normalized = tuple((_absolute_path(path), content) for path, content in publications)
    parents = {path.parent for path, _ in normalized} | {guard}
    bound: dict[Path, tuple[int, int, int]] = {}
    directory_descriptors: set[int] = set()
    temp_descriptors: set[int] = set()
    temporary: list[_TemporaryPublication] = []
    active_temporary: set[tuple[Path, str]] = set()
    published: list[tuple[_TemporaryPublication, str]] = []

    def assert_bound(parent: Path) -> None:
        descriptor, device, inode = bound[parent]
        try:
            current = os.lstat(parent)
        except OSError as exc:
            raise ValueError(f"publication parent directory changed: {parent}") from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != device
            or current.st_ino != inode
            or os.fstat(descriptor).st_dev != device
            or os.fstat(descriptor).st_ino != inode
        ):
            raise ValueError(f"publication parent directory changed: {parent}")

    def assert_all_bound() -> None:
        for parent in bound:
            assert_bound(parent)

    def guard_names() -> set[str]:
        return set(os.listdir(bound[guard][0]))

    def descriptor_hash(descriptor: int) -> tuple[str, int]:
        position = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        os.lseek(descriptor, position, os.SEEK_SET)
        return digest.hexdigest(), byte_count

    def retained_file_is_expected(item: _TemporaryPublication) -> bool:
        try:
            status = os.fstat(item.descriptor)
            digest, byte_count = descriptor_hash(item.descriptor)
        except OSError:
            return False
        return (
            status.st_dev == item.device
            and status.st_ino == item.inode
            and status.st_size == item.byte_count
            and byte_count == item.byte_count
            and digest == item.sha256
        )

    def classify_name(item: _TemporaryPublication, name: str) -> str:
        parent_fd = bound[item.parent][0]
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return "missing"
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != item.device
            or before.st_ino != item.inode
            or before.st_size != item.byte_count
        ):
            return "foreign"
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except OSError:
            return "foreign"
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return "foreign"
        identities = {
            (value.st_dev, value.st_ino, value.st_size)
            for value in (before, opened, after, current)
        }
        if (
            identities != {(item.device, item.inode, item.byte_count)}
            or byte_count != item.byte_count
            or digest.hexdigest() != item.sha256
        ):
            return "foreign"
        return "owned"

    def unlink_if_owned(item: _TemporaryPublication, name: str) -> bool:
        if classify_name(item, name) != "owned":
            return False
        try:
            os.unlink(name, dir_fd=bound[item.parent][0])
        except FileNotFoundError:
            return False
        return True

    def uncertain(message: str) -> PublicationUncertainError:
        return PublicationUncertainError(f"publication uncertain: {message}")

    allowed_names: set[str] = set()
    for path in allowed_root_entries:
        absolute = _absolute_path(path)
        if absolute.parent != guard:
            raise ValueError("allowed output entry must be inside the guarded root")
        allowed_names.add(absolute.name)

    def expected_guard_names() -> set[str]:
        return (
            allowed_names
            | {name for parent, name in active_temporary if parent == guard}
            | {
                name
                for item, name in published
                if item.parent == guard
            }
        )

    def assert_guard_clean() -> None:
        if guard_names() != expected_guard_names():
            raise FileExistsError(f"prior run output appeared under {guard}")

    primary_error: BaseException | None = None
    primary_traceback = None
    rollback_uncertainty: list[str] = []
    temp_cleanup_uncertainty: list[str] = []
    close_uncertainty: list[str] = []
    try:
        for parent in sorted(parents):
            parent.mkdir(parents=True, exist_ok=True)
            _reject_symlink_components(parent, "publication parent")
            descriptor = os.open(
                parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            directory_descriptors.add(descriptor)
            status = os.fstat(descriptor)
            bound[parent] = (descriptor, status.st_dev, status.st_ino)
            assert_bound(parent)
        assert_guard_clean()
        for destination, content in normalized:
            parent_fd = bound[destination.parent][0]
            while True:
                temp_name = (
                    f".{destination.name}.{secrets.token_hex(8)}.freeze-tmp"
                )
                try:
                    descriptor = os.open(
                        temp_name,
                        os.O_RDWR
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=parent_fd,
                    )
                    temp_descriptors.add(descriptor)
                    break
                except FileExistsError:
                    continue
            status = os.fstat(descriptor)
            item = _TemporaryPublication(
                parent=destination.parent,
                name=temp_name,
                descriptor=descriptor,
                device=status.st_dev,
                inode=status.st_ino,
                byte_count=len(content),
                sha256=_sha256_bytes(content),
            )
            temporary.append(item)
            active_temporary.add((item.parent, item.name))
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("failed to write publication temporary file")
                view = view[written:]
            os.fsync(descriptor)
            digest, byte_count = descriptor_hash(descriptor)
            if byte_count != item.byte_count or digest != item.sha256:
                raise uncertain(f"temporary content mismatch for {destination}")
        assert_guard_clean()
        for (destination, _), item in zip(normalized, temporary, strict=True):
            assert_all_bound()
            assert_guard_clean()
            if (
                not retained_file_is_expected(item)
                or classify_name(item, item.name) != "owned"
            ):
                raise uncertain(f"temporary file changed before link: {destination}")
            parent_fd = bound[destination.parent][0]
            try:
                _link_no_replace(
                    item.name,
                    destination.name,
                    source_dir_fd=bound[item.parent][0],
                    destination_dir_fd=parent_fd,
                )
            except FileExistsError:
                raise
            except BaseException as exc:
                state = classify_name(item, destination.name)
                if state == "owned":
                    published.append((item, destination.name))
                elif state == "foreign":
                    raise uncertain(
                        f"destination identity after failed link: {destination}"
                    ) from exc
                raise
            if classify_name(item, destination.name) != "owned":
                raise uncertain(f"destination identity after link: {destination}")
            published.append((item, destination.name))
            assert_all_bound()
            if not unlink_if_owned(item, item.name):
                raise uncertain(f"temporary cleanup identity: {destination}")
            active_temporary.remove((item.parent, item.name))
            assert_guard_clean()
        assert_all_bound()
        assert_guard_clean()
        if any(
            not retained_file_is_expected(item)
            or classify_name(item, name) != "owned"
            for item, name in published
        ):
            raise uncertain("published destination changed before commit")
        for descriptor, _, _ in bound.values():
            os.fsync(descriptor)
    except BaseException as publication_error:
        primary_error = publication_error
        primary_traceback = publication_error.__traceback__
        for item, name in reversed(published):
            try:
                if not unlink_if_owned(item, name):
                    rollback_uncertainty.append(
                        f"{item.parent / name} ({classify_name(item, name)})"
                    )
            except OSError as cleanup_error:
                rollback_uncertainty.append(
                    f"{item.parent / name} (cleanup error: {cleanup_error})"
                )
    for item in temporary:
        if (item.parent, item.name) not in active_temporary:
            continue
        try:
            if not unlink_if_owned(item, item.name):
                temp_cleanup_uncertainty.append(
                    f"{item.parent / item.name} ({classify_name(item, item.name)})"
                )
        except OSError as cleanup_error:
            temp_cleanup_uncertainty.append(
                f"{item.parent / item.name} (cleanup error: {cleanup_error})"
            )
    try:
        for descriptor in temp_descriptors:
            try:
                os.close(descriptor)
            except OSError as close_error:
                close_uncertainty.append(
                    f"temporary fd {descriptor} (close error: {close_error})"
                )
    finally:
        for descriptor in directory_descriptors:
            try:
                os.close(descriptor)
            except OSError as close_error:
                close_uncertainty.append(
                    f"directory fd {descriptor} (close error: {close_error})"
                )

    uncertainty_sections: list[str] = []
    if rollback_uncertainty:
        uncertainty_sections.append(
            "rollback uncertainty: " + ", ".join(rollback_uncertainty)
        )
    if temp_cleanup_uncertainty:
        uncertainty_sections.append(
            "temp cleanup uncertainty: " + ", ".join(temp_cleanup_uncertainty)
        )
    if close_uncertainty:
        uncertainty_sections.append(
            "fd close uncertainty: " + ", ".join(close_uncertainty)
        )

    if primary_error is not None and uncertainty_sections:
        if isinstance(primary_error, PublicationUncertainError):
            message = str(primary_error)
            cause = primary_error.__cause__ or primary_error
        else:
            message = (
                "publication uncertain: publication error: "
                f"{type(primary_error).__name__}: {primary_error}"
            )
            cause = primary_error
        raise PublicationUncertainError(
            f"{message}; {'; '.join(uncertainty_sections)}"
        ) from cause
    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    if uncertainty_sections:
        raise PublicationUncertainError(
            f"publication uncertain: {'; '.join(uncertainty_sections)}"
        )


def _freeze_impl(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    repository_state: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _absolute_path(repo_root or REPO_ROOT)
    _reject_symlink_components(root, "repository root")
    phase, prepared_manifest_path = _preflight_outputs(args)
    repository = _validate_repository(
        repository_state if repository_state is not None else _inspect_repository(root)
    )
    apartment_path = _absolute_path(args.apartment_config)
    office_path = _absolute_path(args.office_config)
    selection_path = _absolute_path(args.apartment_selection)
    apartment, apartment_record = _load_json_record(apartment_path, "Apartment config")
    office, office_record = _load_json_record(office_path, "Office config")
    selection, selection_record = _load_json_record(
        selection_path, "Apartment selection"
    )
    office_metrics = _office_metric_sources(
        office,
        _absolute_path(args.output_manifest).parent,
        allowed_files=(
            frozenset({_absolute_path(prepared_manifest_path)})
            if phase == "finalize"
            else frozenset()
        ),
    )
    if office_metrics:
        raise ValueError(f"Office metric source exists before freeze: {office_metrics}")
    _validate_source_configs(apartment, office)

    shared, scene_records, declared = _validate_input_manifest_and_bindings(
        {"apartment": apartment, "office": office}, selection, repo_root=root
    )
    winner, candidates, target_binding, target_arrays = _validate_selection(
        selection_path,
        selection,
        apartment_config=apartment,
        apartment_config_sha256=apartment_record["sha256"],
        repo_root=root,
        schedule_binding=shared["schedule"],
    )
    frozen_configs: dict[str, dict[str, Any]] = {}
    for scene, source in (("apartment", apartment), ("office", office)):
        frozen = dict(source)
        frozen.update(winner["parameters"])
        frozen["algorithm_hash"] = _algorithm_hash(frozen)
        frozen_configs[scene] = frozen
    if frozen_configs["apartment"]["algorithm_hash"] != frozen_configs["office"]["algorithm_hash"]:
        raise ValueError("frozen normalized algorithm hashes differ across scenes")
    if _json_hash(frozen_configs["apartment"]) != winner["config_sha256"]:
        raise ValueError("selected candidate does not reproduce the frozen Apartment config")

    models: dict[str, dict[str, Any]] = {"frontend": {}, "dense": {}}
    for scene, config in frozen_configs.items():
        cache, frontend_models, dense_models = _validate_cache_manifest(
            scene,
            config,
            declared["scenes"][scene],
            declared["shared"],
            root,
        )
        scene_records[scene]["cache"] = cache
        models["frontend"][scene] = frontend_models
        models["dense"][scene] = dense_models
    if models["frontend"]["apartment"] != models["frontend"]["office"]:
        raise ValueError("Apartment and Office must use the same frozen model provenance")
    dense_shared_fields = (
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
    )
    if any(
        models["dense"]["apartment"][field]
        != models["dense"]["office"][field]
        for field in dense_shared_fields
    ):
        raise ValueError("Apartment and Office must use the same frozen model weights")

    apartment_bytes = _canonical_json_bytes(frozen_configs["apartment"])
    office_bytes = _canonical_json_bytes(frozen_configs["office"])
    mapping_commands, output_roots = _mapping_commands(args, root)
    shared_bindings = {
        "input_manifest": shared["input_manifest"],
        "source_manifest": shared["source"],
        "schedule": shared["schedule"],
        "camera": shared["camera"],
        "common_target_manifest": target_binding,
        "common_target_arrays": target_arrays,
        "alias_map": declared["shared"]["alias_map"],
        "evaluator": declared["shared"]["evaluator"],
        "finalizers": {
            "common_v2": declared["shared"]["common_finalizer"],
            "official_t2": declared["shared"]["official_finalizer"],
        },
    }
    release_bindings = {
        "canonical_summary_generator": _frozen_file_binding(
            root
            / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py",
            "canonical summary generator",
        ),
        "release_finalizer": _frozen_file_binding(
            root
            / "scripts/evaluation/finalize_oviv2_tesse_common_v2_release.py",
            "OVIV2 common-v2 release finalizer",
        ),
        "label_spaces": {
            "apartment": _frozen_file_binding(
                _absolute_path(args.apartment_label_space),
                "Apartment label space",
            ),
            "office": _frozen_file_binding(
                _absolute_path(args.office_label_space),
                "Office label space",
            ),
        },
    }
    manifest = {
        "schema_version": 1,
        "freeze_id": "oviv2-tessecd-v1",
        "status": "PREPARED" if phase == "prepare" else "FROZEN",
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "repository": repository,
        "algorithm": {
            "sha256": frozen_configs["apartment"]["algorithm_hash"],
            "normalized_config": _algorithm_config(frozen_configs["apartment"]),
        },
        "selection": {
            **selection_record,
            "candidate_count": len(candidates),
            "selected_config_sha256": winner["config_sha256"],
            "selected_parameters": winner["parameters"],
            "selection_rule": [
                "maximize_current_miou",
                "minimize_ghost_rate",
                "maximize_background_f5_cm",
                "minimize_recovery_frames",
                "minimize_config_sha256",
            ],
            "candidates": candidates,
        },
        "scenes": {
            "apartment": {
                **scene_records["apartment"],
                "source_config": {
                    **apartment_record,
                },
                "frozen_config": {
                    "path": str(_absolute_path(args.output_apartment_config)),
                    "sha256": _sha256_bytes(apartment_bytes),
                    "byte_count": len(apartment_bytes),
                },
            },
            "office": {
                **scene_records["office"],
                "source_config": {
                    **office_record,
                },
                "frozen_config": {
                    "path": str(_absolute_path(args.output_office_config)),
                    "sha256": _sha256_bytes(office_bytes),
                    "byte_count": len(office_bytes),
                },
            },
        },
        "shared_bindings": shared_bindings,
        "release_bindings": release_bindings,
        "models": models,
        "environment": dict(environment) if environment is not None else _default_environment(),
        "commands": {
            "cwd": str(root),
            "python": sys.executable,
            "mapping": mapping_commands,
        },
        "output_roots": output_roots,
        "office_pre_freeze_audit": {
            "metric_sources_found": [],
            "output_root_was_empty": phase == "prepare",
            "output_root_had_only_preparation": phase == "finalize",
            "scope": {
                "office_config_recursive": True,
                "output_root": str(_absolute_path(args.output_manifest).parent),
                "selection_scene": "apartment",
            },
        },
    }
    if phase == "finalize":
        expected_configs = {
            _absolute_path(args.output_apartment_config): apartment_bytes,
            _absolute_path(args.output_office_config): office_bytes,
        }
        for path, expected_bytes in expected_configs.items():
            if path.read_bytes() != expected_bytes:
                raise ValueError(f"prepared frozen config bytes changed: {path}")
        prepared, prepared_record = _load_json_record(
            prepared_manifest_path, "prepared freeze manifest"
        )
        if (
            prepared.get("status") != "PREPARED"
            or prepared.get("freeze_id") != "oviv2-tessecd-v1"
            or prepared.get("algorithm") != manifest["algorithm"]
            or prepared.get("selection") != manifest["selection"]
            or prepared.get("shared_bindings") != manifest["shared_bindings"]
            or prepared.get("release_bindings") != manifest["release_bindings"]
            or prepared.get("models") != manifest["models"]
            or prepared.get("commands") != manifest["commands"]
            or prepared.get("output_roots") != manifest["output_roots"]
            or any(
                prepared.get("scenes", {}).get(scene, {}).get("frozen_config")
                != manifest["scenes"][scene]["frozen_config"]
                for scene in SCENES
            )
        ):
            raise ValueError("prepared freeze manifest does not match final inputs")
        _verify_configs_in_commit(root, expected_configs, repository_state)
        manifest["preparation"] = {
            "manifest": prepared_record,
            "repository": prepared["repository"],
        }
    excluded_records = (
        frozenset(
            {
                _absolute_path(args.output_apartment_config),
                _absolute_path(args.output_office_config),
            }
        )
        if phase == "prepare"
        else frozenset()
    )
    _revalidate_file_records(manifest, excluded_paths=excluded_records)
    _revalidate_snapshot_session()
    if repository_state is None:
        final_repository = _validate_repository(_inspect_repository(root))
        if final_repository != repository:
            raise ValueError("repository commit or tree changed during freeze")
    manifest_bytes = _canonical_json_bytes(manifest)
    if phase == "prepare":
        publications = (
            (_absolute_path(args.output_apartment_config), apartment_bytes),
            (_absolute_path(args.output_office_config), office_bytes),
            (prepared_manifest_path, manifest_bytes),
        )
        allowed_root_entries: frozenset[Path] = frozenset()
    else:
        publications = ((_absolute_path(args.output_manifest), manifest_bytes),)
        allowed_root_entries = frozenset({_absolute_path(prepared_manifest_path)})
    _publish_json_transaction(
        publications,
        empty_root_guard=_absolute_path(args.output_manifest).parent,
        allowed_root_entries=allowed_root_entries,
    )
    return manifest


def freeze(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    repository_state: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    global _ACTIVE_SNAPSHOTS

    if _ACTIVE_SNAPSHOTS is not None:
        raise RuntimeError("nested freeze snapshot sessions are not supported")
    _ACTIVE_SNAPSHOTS = {}
    try:
        return _freeze_impl(
            args,
            repo_root=repo_root,
            repository_state=repository_state,
            environment=environment,
        )
    finally:
        _ACTIVE_SNAPSHOTS = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-selection", type=Path, required=True)
    parser.add_argument("--apartment-config", type=Path, required=True)
    parser.add_argument("--office-config", type=Path, required=True)
    parser.add_argument("--output-apartment-config", type=Path, required=True)
    parser.add_argument("--output-office-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--apartment-label-space", type=Path, required=True)
    parser.add_argument("--office-label-space", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = freeze(args)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"{manifest['status']} {manifest['freeze_id']} "
        f"algorithm={manifest['algorithm']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
