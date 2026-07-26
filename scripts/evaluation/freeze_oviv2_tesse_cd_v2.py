#!/usr/bin/env python3
"""Freeze the selected OVIV2 TESSE-CD dual-readout v2 evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import stat
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.run_oviv2_tesse_cd_v2 import (  # noqa: E402
    V2_FREEZE_COMMAND_KEYS,
    V2_FREEZE_ENVIRONMENT_KEYS,
    V2_FREEZE_MODEL_KEYS,
    V2_FREEZE_OFFICE_AUDIT_KEYS,
    V2_FREEZE_RELEASE_KEYS,
    V2_FREEZE_REPOSITORY_KEYS,
    V2_FREEZE_SCENE_KEYS,
    V2_FREEZE_SELECTION_KEYS,
    V2_FREEZE_SHARED_KEYS,
    V2_FREEZE_TOP_KEYS,
    V2_T4_METRIC_KEYS,
    _V2_CONFIG_KEYS,
    _validate_config,
)
from scripts.evaluation.verify_oviv2_dual_readout_development_gates import (  # noqa: E402
    verify_exact_profile_runs,
)


FREEZE_ID = "oviv2-tessecd-v2"
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SCENES = ("apartment", "office")
CANDIDATES = tuple(f"a{index}" for index in range(5))
_OFFICE_PATH_TOKEN = re.compile(r"(^|[^a-z0-9])office([^a-z0-9]|$)")
SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "phase",
        "dataset",
        "method_id",
        "protocol_id",
        "status",
        "development_scene",
        "transfer_scene",
        "scenes_read",
        "office_results_read",
        "manifest",
        "shortlist",
        "t4_matrix",
        "t4_protocol",
        "t4_root_sha256",
        "profile_fallback_order",
        "selected_candidate_id",
        "selected_config",
        "selected_config_record",
        "selected_config_sha256",
        "algorithm_hash",
        "t4_ledger",
    }
)
LEDGER_KEYS = frozenset(
    {
        "candidate_id",
        "profile",
        "passed",
        "selected",
        "failed_gates",
        "config_sha256",
        "run_manifest_sha256",
    }
)
ENVIRONMENT_LIBRARY_KEYS = frozenset(
    {"numpy", "open3d", "scipy", "torch", "pillow"}
)
SCENE_FILE_FIELDS = (
    "export_manifest",
    "frontend_manifest",
    "dense_manifest",
    "vocabulary_json",
    "vocabulary_txt",
)
SHARED_CONFIG_FIELDS = {
    "input_manifest": "input_manifest",
    "schedule": "schedule_manifest",
    "occlusion_target_manifest": "occlusion_target_manifest",
}
DEFAULT_RELEASE_PATHS = {
    "temporal_evaluator": REPO_ROOT
    / "scripts/evaluation/evaluate_oviv2_tesse_temporal_occlusion.py",
    "result_finalizer": REPO_ROOT / "scripts/evaluation/finalize_tesse_t2.py",
}
_HEX = frozenset("0123456789abcdef")
T4_BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}
SHORTLIST_KEYS = frozenset(
    {
        "schema_version", "manifest_id", "phase", "dataset", "method_id",
        "protocol_id", "status", "development_scene", "transfer_scene",
        "office_results_read", "scenes_read", "result_contract", "results_root",
        "manifest", "result_files", "profile_fallback_order", "floors",
        "shortlisted_candidate_ids", "shortlisted_candidates", "rejection_ledger",
    }
)


@dataclass(frozen=True)
class FreezeDependencies:
    repository_inspector: Callable[[Path], Mapping[str, Any]]
    environment_collector: Callable[[], Mapping[str, Any]]
    release_paths: Mapping[str, Path]
    python_executable: Path
    t1_transaction_verifier: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    identity: tuple[int, int, int, int, int]
    content: bytes
    sha256: str

    @property
    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "byte_count": len(self.content),
        }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value).rstrip(b"\n"))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _is_git_id(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _HEX


def _absolute(path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or Path.cwd()) / candidate
    return Path(os.path.abspath(candidate))


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = _absolute(path)
    for component in (absolute, *absolute.parents):
        try:
            status = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{role} must not contain symlink components: {path}")


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _snapshot(path: Path, role: str) -> _Snapshot:
    absolute = _absolute(path)
    _reject_symlink_components(absolute, role)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{role} must be a regular non-symlink file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{role} changed while being read: {path}") from exc
    if (
        _identity(before) != _identity(after)
        or _identity(after) != _identity(current)
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError(f"{role} changed while being read: {path}")
    content = b"".join(chunks)
    if len(content) != current.st_size:
        raise ValueError(f"{role} changed while being read: {path}")
    return _Snapshot(absolute, _identity(current), content, _sha256_bytes(content))


def _revalidate_snapshot(snapshot: _Snapshot, role: str) -> None:
    current = _snapshot(snapshot.path, role)
    if current.identity != snapshot.identity or current.sha256 != snapshot.sha256:
        raise ValueError(f"{role} changed during freeze: {snapshot.path}")


def _load_json(snapshot: _Snapshot, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    _reject_nonfinite(value, role)
    return value


def _reject_nonfinite(value: object, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item, role)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, role)


def _binding(
    path_value: object,
    *,
    repo_root: Path,
    role: str,
    snapshots: dict[Path, _Snapshot],
) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{role} path must be a non-empty string")
    path = _absolute(path_value, base=repo_root)
    snapshot = snapshots.get(path)
    if snapshot is None:
        snapshot = _snapshot(path, role)
        snapshots[path] = snapshot
    if path.suffix.lower() == ".json":
        _load_json(snapshot, role)
    return snapshot.record


def _verify_record(
    value: object,
    *,
    repo_root: Path,
    role: str,
    snapshots: dict[Path, _Snapshot],
) -> tuple[dict[str, Any], _Snapshot]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{role} binding schema is invalid")
    if not _is_sha256(value.get("sha256")) or type(value.get("byte_count")) is not int:
        raise ValueError(f"{role} binding identity is invalid")
    record = _binding(
        value.get("path"), repo_root=repo_root, role=role, snapshots=snapshots
    )
    if dict(value) != record:
        raise ValueError(f"{role} binding changed")
    return record, snapshots[Path(record["path"])]


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _inspect_repository(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            STAGE3_LINEAGE_COMMIT,
            commit,
        ],
        check=False,
        capture_output=True,
    )
    return {
        "clean": _git(repo_root, "status", "--porcelain", "--untracked-files=all") == "",
        "commit": commit,
        "parents": _git(repo_root, "show", "-s", "--format=%P", "HEAD").split(),
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "commit_time_utc": _git(repo_root, "show", "-s", "--format=%cI", "HEAD"),
        "stage3_lineage_commit": STAGE3_LINEAGE_COMMIT,
        "stage3_is_ancestor": ancestor.returncode == 0,
    }


def _validate_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_REPOSITORY_KEYS:
        raise ValueError("repository schema is invalid")
    parents = value.get("parents")
    if not (
        value.get("clean") is True
        and _is_git_id(value.get("commit"))
        and _is_git_id(value.get("tree"))
        and type(parents) is list
        and all(_is_git_id(parent) for parent in parents)
        and isinstance(value.get("commit_time_utc"), str)
        and bool(value["commit_time_utc"].strip())
        and value.get("stage3_lineage_commit") == STAGE3_LINEAGE_COMMIT
        and value.get("stage3_is_ancestor") is True
    ):
        raise ValueError("repository must be a clean v2 lineage commit")
    return dict(value)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _default_environment() -> dict[str, Any]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda: list[str] = []
    gpu: list[str] = []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            if line.strip():
                gpu.append(line.split(",", 1)[0].strip())
                cuda.append(line.strip())
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "host": socket.gethostname(),
        "cuda": cuda or ["unavailable"],
        "cuda_visible_devices": cuda_visible,
        "gpu": gpu or ["unavailable"],
        "libraries": {
            "numpy": _version("numpy"),
            "open3d": _version("open3d"),
            "scipy": _version("scipy"),
            "torch": _version("torch"),
            "pillow": _version("pillow"),
        },
    }


def _validate_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != V2_FREEZE_ENVIRONMENT_KEYS:
        raise ValueError("environment schema is invalid")
    for key in ("python", "python_implementation", "platform", "machine", "host"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"environment.{key} is invalid")
    for key in ("cuda", "gpu"):
        items = value.get(key)
        if type(items) is not list or not items or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"environment.{key} is invalid")
    visible = value.get("cuda_visible_devices")
    if visible is not None and (not isinstance(visible, str) or not visible.strip()):
        raise ValueError("environment.cuda_visible_devices is invalid")
    libraries = value.get("libraries")
    if not isinstance(libraries, Mapping) or set(libraries) != ENVIRONMENT_LIBRARY_KEYS:
        raise ValueError("environment libraries schema is invalid")
    if any(not isinstance(item, str) or not item.strip() for item in libraries.values()):
        raise ValueError("environment library versions are invalid")
    return dict(value)


def _load_configs(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    snapshots: dict[Path, _Snapshot],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    configs: dict[str, dict[str, Any]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        path = _absolute(getattr(args, f"{scene}_config"), base=repo_root)
        snapshot = _snapshot(path, f"{scene} frozen config")
        snapshots[path] = snapshot
        config = _load_json(snapshot, f"{scene} frozen config")
        if set(config) != _V2_CONFIG_KEYS:
            raise ValueError(f"{scene} config schema has missing or surplus fields")
        parsed_scene, *_ = _validate_config(config)
        if parsed_scene != scene or config.get("protocol_id") != FREEZE_ID:
            raise ValueError(f"{scene} config is not a {FREEZE_ID} config")
        if config.get("algorithm_hash") != canonical_algorithm_hash(config):
            raise ValueError(f"{scene} config algorithm_hash is stale")
        scene_bindings = {"frozen_config": snapshot.record}
        for field in SCENE_FILE_FIELDS:
            scene_bindings[field] = _binding(
                config.get(field),
                repo_root=repo_root,
                role=f"{scene} {field}",
                snapshots=snapshots,
            )
        configs[scene] = config
        scenes[scene] = scene_bindings
    algorithm_hashes = {config["algorithm_hash"] for config in configs.values()}
    normalized = [canonical_algorithm_config(config) for config in configs.values()]
    if len(algorithm_hashes) != 1 or normalized[0] != normalized[1]:
        raise ValueError("Apartment and Office normalized algorithm configs differ")
    return configs, scenes


def _shared_bindings(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
    snapshots: dict[Path, _Snapshot],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, field in SHARED_CONFIG_FIELDS.items():
        paths = {_absolute(config[field], base=repo_root) for config in configs.values()}
        if len(paths) != 1:
            raise ValueError(f"shared {role} paths differ across scenes")
        result[role] = _binding(
            str(next(iter(paths))),
            repo_root=repo_root,
            role=f"shared {role}",
            snapshots=snapshots,
        )
    expected_target_hash = result["occlusion_target_manifest"]["sha256"]
    if any(
        config.get("occlusion_target_manifest_sha256") != expected_target_hash
        for config in configs.values()
    ):
        raise ValueError("occlusion target config hash drift")
    return result


def _validate_selection(
    path: Path,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
    snapshots: dict[Path, _Snapshot],
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = _snapshot(path, "Apartment final selection")
    snapshots[path] = snapshot
    selection = _load_json(snapshot, "Apartment final selection")
    apartment = configs["apartment"]
    selected_hash = _json_hash(apartment)
    algorithm_hash = apartment["algorithm_hash"]
    if set(selection) != SELECTION_KEYS or not (
        selection.get("schema_version") == 1
        and selection.get("manifest_id")
        == "oviv2_tesse_dual_readout_selection_v2"
        and selection.get("phase") == "final"
        and selection.get("dataset") == "TESSE-CD"
        and selection.get("method_id") == "OVIV2"
        and selection.get("protocol_id") == FREEZE_ID
        and selection.get("status") == "PASS"
        and selection.get("development_scene") == "apartment"
        and selection.get("transfer_scene") == "office"
        and selection.get("scenes_read") == ["apartment"]
        and selection.get("office_results_read") is False
        and selection.get("profile_fallback_order") == ["a4", "a3", "a2"]
        and selection.get("selected_config") == apartment
        and selection.get("selected_config_sha256") == selected_hash
        and selection.get("algorithm_hash") == algorithm_hash
    ):
        raise ValueError("final selection identity or selected config drift")
    selected_candidate = selection.get("selected_candidate_id")
    if selected_candidate not in CANDIDATES:
        raise ValueError("final selection candidate is undeclared")
    selected_record, selected_snapshot = _verify_record(
        selection.get("selected_config_record"),
        repo_root=repo_root,
        role="selected candidate config",
        snapshots=snapshots,
    )
    if _load_json(selected_snapshot, "selected candidate config") != apartment:
        raise ValueError("selected config record differs from frozen Apartment config")
    bound_records: dict[str, dict[str, Any]] = {}
    for role in ("manifest", "shortlist", "t4_matrix", "t4_protocol"):
        bound_records[role], _ = _verify_record(
            selection.get(role),
            repo_root=repo_root,
            role=f"selection {role}",
            snapshots=snapshots,
        )
    bound_paths = [Path(record["path"]) for record in bound_records.values()]
    if len(set(bound_paths)) != len(bound_paths):
        raise ValueError("final selection source records must be distinct")
    if any(_OFFICE_PATH_TOKEN.search(path.as_posix().lower()) for path in bound_paths):
        raise ValueError("final selection contains an Office result source path")
    for role, record in bound_records.items():
        source = _load_json(snapshots[Path(record["path"])], f"selection {role}")
        if _has_office_read_evidence(source):
            raise ValueError("final selection source contains Office result evidence")
    shortlist_snapshot = snapshots[Path(bound_records["shortlist"]["path"])]
    shortlist = _load_json(shortlist_snapshot, "selection shortlist")
    shortlist_ids = shortlist.get("shortlisted_candidate_ids")
    shortlist_candidates = shortlist.get("shortlisted_candidates")
    if not (
        set(shortlist) == SHORTLIST_KEYS
        and shortlist.get("schema_version") == 1
        and shortlist.get("manifest_id") == "oviv2_tesse_dual_readout_shortlist_v1"
        and shortlist.get("phase") == "shortlist"
        and shortlist.get("dataset") == "TESSE-CD"
        and shortlist.get("method_id") == "OVIV2"
        and shortlist.get("protocol_id") == FREEZE_ID
        and shortlist.get("status") == "PASS"
        and shortlist.get("development_scene") == "apartment"
        and shortlist.get("transfer_scene") == "office"
        and shortlist.get("office_results_read") is False
        and shortlist.get("scenes_read") == ["apartment"]
        and shortlist.get("manifest") == bound_records["manifest"]
        and shortlist.get("profile_fallback_order") == ["a4", "a3", "a2"]
        and isinstance(shortlist_ids, list)
        and bool(shortlist_ids)
        and len(shortlist_ids) == len(set(shortlist_ids))
        and all(candidate in {"a4", "a3", "a2"} for candidate in shortlist_ids)
        and shortlist_ids == [
            candidate for candidate in ("a4", "a3", "a2") if candidate in shortlist_ids
        ]
        and isinstance(shortlist_candidates, list)
        and len(shortlist_candidates) == len(shortlist_ids)
        and all(
            isinstance(item, Mapping)
            and set(item) == {
                "candidate_id", "profile", "config_sha256", "algorithm_hash",
                "result", "selected_config", "selected_config_record",
            }
            and item.get("candidate_id") == candidate
            and item.get("profile") == candidate
            and _is_sha256(item.get("config_sha256"))
            and _is_sha256(item.get("algorithm_hash"))
            for candidate, item in zip(shortlist_ids, shortlist_candidates, strict=True)
        )
        and selected_candidate in shortlist_ids
    ):
        raise ValueError("selection shortlist binding or Apartment scope is invalid")
    matrix_snapshot = snapshots[Path(bound_records["t4_matrix"]["path"])]
    matrix = _load_json(matrix_snapshot, "selection T4 matrix")
    matrix_unhashed = dict(matrix)
    matrix_root = matrix_unhashed.pop("root_sha256", None)
    t4_row = (
        matrix.get("candidates", {}).get(selected_candidate)
        if isinstance(matrix.get("candidates"), Mapping)
        else None
    )
    if not (
        matrix.get("manifest_id") == "oviv2_tesse_t4_matrix_v1"
        and matrix.get("status") == "PASS"
        and matrix.get("shortlist") == bound_records["shortlist"]
        and matrix.get("protocol") == bound_records["t4_protocol"]
        and matrix_root == selection.get("t4_root_sha256")
        and matrix_root == _json_hash(matrix_unhashed)
        and isinstance(matrix.get("candidates"), Mapping)
        and set(matrix["candidates"]) == set(shortlist_ids)
        and isinstance(t4_row, Mapping)
        and t4_row.get("status") == "PASS"
        and t4_row.get("config_sha256") == selected_hash
        and isinstance(t4_row.get("gates"), Mapping)
        and set(t4_row["gates"]) == V2_T4_METRIC_KEYS
        and all(value is True for value in t4_row["gates"].values())
    ):
        raise ValueError("selected candidate T4 matrix binding is invalid")
    ledger = selection.get("t4_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(shortlist_ids):
        raise ValueError("final selection T4 ledger is missing")
    selected_entries = 0
    first_passed: str | None = None
    for candidate, shortlist_item, entry in zip(
        shortlist_ids, shortlist_candidates, ledger, strict=True
    ):
        if not isinstance(entry, Mapping) or set(entry) != LEDGER_KEYS:
            raise ValueError("final selection T4 ledger schema is invalid")
        if not (
            entry.get("candidate_id") == candidate
            and entry.get("profile") == shortlist_item["profile"]
            and type(entry.get("passed")) is bool
            and type(entry.get("selected")) is bool
            and isinstance(entry.get("failed_gates"), list)
            and all(name in V2_T4_METRIC_KEYS for name in entry["failed_gates"])
            and entry.get("config_sha256") == shortlist_item["config_sha256"]
            and entry.get("run_manifest_sha256")
            == matrix["candidates"][candidate]["run_manifest_sha256"]
        ):
            raise ValueError("final selection T4 ledger row is invalid")
        expected_failures = [
            name
            for name, bound in T4_BOUNDS.items()
            if matrix["candidates"][candidate]["metrics"][name] > bound
        ]
        if (
            entry["failed_gates"] != expected_failures
            or entry["passed"] is not (not expected_failures)
        ):
            raise ValueError("final selection T4 ledger disagrees with matrix gates")
        if first_passed is None and entry["passed"]:
            first_passed = candidate
        if entry["selected"] is not (candidate == first_passed and entry["passed"]):
            raise ValueError("final selection does not follow fallback order")
        if entry["selected"]:
            selected_entries += 1
            if not (
                entry["candidate_id"] == selected_candidate
                and entry["passed"] is True
                and entry["failed_gates"] == []
                and entry["config_sha256"] == selected_hash
                and entry["run_manifest_sha256"] == t4_row["run_manifest_sha256"]
                and shortlist_item["selected_config"] == apartment
                and shortlist_item["selected_config_record"] == selected_record
                and shortlist_item["algorithm_hash"] == algorithm_hash
            ):
                raise ValueError("selected T4 ledger row does not prove PASS")
    if selected_entries != 1 or first_passed != selected_candidate or _has_office_read_evidence(selection):
        raise ValueError("final selection is not an Apartment-only unique T4 PASS")
    del selected_record
    return selection, {
        "artifact": snapshot.record,
        "development_scene": "apartment",
        "selected_config_sha256": selected_hash,
        "selected_algorithm_hash": algorithm_hash,
    }


def _has_office_read_evidence(value: object) -> bool:
    if isinstance(value, Mapping):
        if value.get("scene") == "office" or value.get("office_results_read") is True:
            return True
        scenes_read = value.get("scenes_read")
        if isinstance(scenes_read, list) and "office" in scenes_read:
            return True
        return any(_has_office_read_evidence(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_office_read_evidence(item) for item in value)
    return False


def _models(
    scenes: Mapping[str, Mapping[str, Any]], snapshots: Mapping[Path, _Snapshot]
) -> dict[str, Any]:
    result: dict[str, Any] = {"frontend": {}, "dense": {}}
    for scene in SCENES:
        for branch in ("frontend", "dense"):
            manifest_record = scenes[scene][f"{branch}_manifest"]
            snapshot = snapshots[Path(manifest_record["path"])]
            manifest = _load_json(snapshot, f"{scene} {branch} manifest")
            if branch == "frontend":
                provenance = manifest.get("provenance_sha256")
                model_id = manifest.get("feature_model_id")
                model_hash = (
                    provenance.get("clip_model")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if model_id != f"clip-sha256:{model_hash}" or not _is_sha256(model_hash):
                    raise ValueError(f"{scene} frontend model identity is invalid")
            else:
                provenance = manifest.get("provenance")
                model_id = (
                    provenance.get("model_id")
                    if isinstance(provenance, Mapping)
                    else None
                )
                model_hash = (
                    provenance.get("model_sha256")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if not isinstance(model_id, str) or not model_id or not _is_sha256(model_hash):
                    raise ValueError(f"{scene} dense model identity is invalid")
            model = {
                "manifest_sha256": manifest_record["sha256"],
                "model_id": model_id,
                "model_sha256": model_hash,
            }
            if set(model) != V2_FREEZE_MODEL_KEYS:
                raise AssertionError("runner model schema changed")
            result[branch][scene] = model

    for branch in ("frontend", "dense"):
        identities = {
            (
                result[branch][scene]["model_id"],
                result[branch][scene]["model_sha256"],
            )
            for scene in SCENES
        }
        if len(identities) != 1:
            raise ValueError(f"Apartment and Office {branch} model identities differ")
    return result


def _release_bindings(
    paths: Mapping[str, Path], snapshots: dict[Path, _Snapshot]
) -> dict[str, Any]:
    if set(paths) != V2_FREEZE_RELEASE_KEYS:
        raise ValueError("release path schema is invalid")
    return {
        role: _binding(
            str(_absolute(paths[role])),
            repo_root=REPO_ROOT,
            role=f"release {role}",
            snapshots=snapshots,
        )
        for role in sorted(paths)
    }


def _output_roots(output: Path, seeds: Sequence[int]) -> dict[str, Path]:
    return {
        "apartment_run1": output.parent / "apartment" / "run1",
        "apartment_run2": output.parent / "apartment" / "run2",
        **{
            f"office_seed_{seed}": output.parent / f"office_seed_{seed}"
            for seed in seeds
        },
    }


def _validate_outputs(
    output: Path,
    roots: Mapping[str, Path],
    *,
    input_paths: set[Path],
) -> dict[str, str]:
    _reject_symlink_components(output, "freeze output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    expected_slots = {"apartment_run1", "apartment_run2"} | {
        slot for slot in roots if re.fullmatch(r"office_seed_(0|17|29|43|71|101)", slot)
    }
    if set(roots) != expected_slots:
        raise ValueError("run slots are incomplete or duplicated")
    normalized = {slot: _absolute(path) for slot, path in roots.items()}
    values = list(normalized.values())
    if len(set(values)) != len(values):
        raise ValueError("output roots must be distinct")
    protected = input_paths | {output}
    for path in values:
        _reject_symlink_components(path, "run output root")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        for other in values:
            if path != other and (path in other.parents or other in path.parents):
                raise ValueError("output roots must not have parent/child overlap")
        if any(path == item or path in item.parents or item in path.parents for item in protected):
            raise ValueError("output root aliases or overlaps a frozen input")
    if output.parent.exists():
        entries = list(output.parent.iterdir())
        if entries:
            raise ValueError("formal output root must be new and empty")
    return {slot: str(path) for slot, path in normalized.items()}


def _office_evidence(value: object, path: str = "config") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {
                "metrics",
                "metric_source",
                "metrics_path",
                "official_metrics",
                "evaluation_result",
                "run_output",
            }:
                found.append(f"{path}.{key}")
            found.extend(_office_evidence(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_office_evidence(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        if "office" in lowered and any(
            marker in lowered for marker in ("/result", "/output", "/metric")
        ):
            found.append(path)
    return found


def _mapping_commands(
    *,
    python: Path,
    configs: Mapping[str, Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
    output: Path,
    roots: Mapping[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    del configs
    runner = REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"
    mapping: list[dict[str, Any]] = []
    for scene, slots in (
        ("apartment", ("apartment_run1", "apartment_run2")),
        ("office", tuple(sorted(slot for slot in roots if slot.startswith("office_seed_")))),
    ):
        config_path = scenes[scene]["frozen_config"]["path"]
        for slot in slots:
            argv = [
                str(python),
                str(runner),
                "--config",
                config_path,
                "--output",
                roots[slot],
                "--freeze-manifest",
                str(output),
                "--run-slot",
                slot,
            ]
            mapping.append(
                {"scene": scene, "run_slot": slot, "output": roots[slot], "argv": argv}
            )
    result = {"cwd": str(repo_root), "python": str(python), "mapping": mapping}
    if set(result) != V2_FREEZE_COMMAND_KEYS:
        raise AssertionError("runner command schema changed")
    return result


def _required_evidence_path(args: argparse.Namespace, name: str, root: Path) -> Path:
    value = getattr(args, name, None)
    if value is None:
        raise ValueError("T1 and T4 evidence are required")
    return _absolute(value, base=root)


def _freeze_t1_evidence(
    path: Path,
    *,
    repo_root: Path,
    snapshots: dict[Path, _Snapshot],
    transaction_verifier: Callable[
        [Sequence[Mapping[str, Any]]], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    artifact = _snapshot(path, "T1 evidence")
    snapshots[path] = artifact
    payload = _load_json(artifact, "T1 evidence")
    if set(payload) != {
        "schema_version", "manifest_id", "deterministic_evidence", "receipt"
    } or not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id")
        == "oviv2_dual_readout_development_gates_v1"
    ):
        raise ValueError("T1 evidence schema or identity is invalid")
    deterministic = payload.get("deterministic_evidence")
    if not isinstance(deterministic, Mapping):
        raise ValueError("T1 deterministic evidence is missing")
    gates = deterministic.get("gates")
    if not (
        isinstance(gates, Mapping)
        and set(gates) == {"t1_exact", "determinism"}
        and all(
            isinstance(gates[name], Mapping) and gates[name].get("status") == "PASS"
            for name in gates
        )
    ):
        raise ValueError("T1 exactness and determinism gates must PASS")
    exact = deterministic.get("cumulative_exact")
    expected_sequence = [
        "reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4"
    ]
    if not (
        isinstance(exact, Mapping)
        and set(exact) == {"format", "sequence", "executions", "profiles"}
        and exact.get("format") == "oviv2_t1_exact_transaction_v1"
        and exact.get("sequence") == expected_sequence
        and isinstance(exact.get("executions"), list)
        and bool(exact["executions"])
        and isinstance(exact.get("profiles"), Mapping)
        and set(exact["profiles"]) == set(CANDIDATES)
    ):
        raise ValueError("T1 cumulative exact transaction is invalid")
    try:
        recomputed_exact = transaction_verifier(exact["executions"])
    except Exception as exc:
        raise ValueError("T1 exact transaction recomputation failed") from exc
    if recomputed_exact != exact:
        raise ValueError("T1 exact transaction recomputation disagrees")
    roots = {
        profile.get("cumulative_root_sha256")
        for profile in exact["profiles"].values()
        if isinstance(profile, Mapping)
    }
    if len(roots) != 1 or not _is_sha256(next(iter(roots), None)):
        raise ValueError("T1 profile cumulative roots disagree")
    source_record, _ = _verify_record(
        deterministic.get("source_manifest"),
        repo_root=repo_root,
        role="T1 source manifest",
        snapshots=snapshots,
    )
    return {
        "artifact": artifact.record,
        "source_manifest": source_record,
        "root_sha256": next(iter(roots)),
    }


def _freeze_t4_evidence(
    path: Path,
    *,
    selected_candidate: str,
    selected_config_sha256: str,
    repo_root: Path,
    snapshots: dict[Path, _Snapshot],
) -> dict[str, Any]:
    artifact = _snapshot(path, "T4 evidence")
    snapshots[path] = artifact
    payload = _load_json(artifact, "T4 evidence")
    if set(payload) != {
        "schema_version", "manifest_id", "status", "shortlist", "protocol",
        "candidates", "root_sha256",
    } or not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "oviv2_tesse_t4_matrix_v1"
        and payload.get("status") == "PASS"
    ):
        raise ValueError("T4 matrix schema or identity is invalid")
    claimed_root = payload.get("root_sha256")
    unhashed = dict(payload)
    unhashed.pop("root_sha256")
    if not _is_sha256(claimed_root) or claimed_root != _json_hash(unhashed):
        raise ValueError("T4 matrix root hash is stale")
    records: dict[str, Any] = {}
    for role in ("shortlist", "protocol"):
        records[role], _ = _verify_record(
            payload.get(role), repo_root=repo_root, role=f"T4 {role}", snapshots=snapshots
        )
    candidates = payload.get("candidates")
    shortlist_payload = _load_json(
        snapshots[Path(records["shortlist"]["path"])], "T4 shortlist"
    )
    shortlist_ids = shortlist_payload.get("shortlisted_candidate_ids")
    if (
        not isinstance(shortlist_ids, list)
        or not isinstance(candidates, Mapping)
        or set(candidates) != set(shortlist_ids)
    ):
        raise ValueError("T4 matrix candidate inventory is invalid")
    for candidate, candidate_row in candidates.items():
        if not (
            isinstance(candidate_row, Mapping)
            and set(candidate_row) == {
                "status", "config_sha256", "run_manifest_sha256", "metrics", "gates"
            }
            and _is_sha256(candidate_row.get("config_sha256"))
            and _is_sha256(candidate_row.get("run_manifest_sha256"))
            and isinstance(candidate_row.get("metrics"), Mapping)
            and set(candidate_row["metrics"]) == V2_T4_METRIC_KEYS
            and all(
                type(metric_value) in {int, float} and math.isfinite(metric_value)
                for metric_value in candidate_row["metrics"].values()
            )
            and isinstance(candidate_row.get("gates"), Mapping)
            and set(candidate_row["gates"]) == V2_T4_METRIC_KEYS
        ):
            raise ValueError(f"T4 matrix row {candidate} is invalid")
        expected_gates = {
            metric: candidate_row["metrics"][metric] <= T4_BOUNDS[metric]
            for metric in V2_T4_METRIC_KEYS
        }
        expected_status = "PASS" if all(expected_gates.values()) else "FAIL"
        if candidate_row["gates"] != expected_gates or candidate_row["status"] != expected_status:
            raise ValueError(f"T4 matrix row {candidate} has stale gates or status")
    if payload["status"] != (
        "PASS" if any(row["status"] == "PASS" for row in candidates.values()) else "FAIL"
    ):
        raise ValueError("T4 matrix aggregate status is stale")
    row = candidates.get(selected_candidate) if isinstance(candidates, Mapping) else None
    if not (
        isinstance(row, Mapping)
        and set(row) == {
            "status", "config_sha256", "run_manifest_sha256", "metrics", "gates"
        }
        and row.get("status") == "PASS"
        and row.get("config_sha256") == selected_config_sha256
        and _is_sha256(row.get("run_manifest_sha256"))
        and isinstance(row.get("metrics"), Mapping)
        and set(row["metrics"]) == V2_T4_METRIC_KEYS
        and all(
            type(value) in {int, float} and math.isfinite(value)
            for value in row["metrics"].values()
        )
        and isinstance(row.get("gates"), Mapping)
        and set(row["gates"]) == V2_T4_METRIC_KEYS
        and all(value is True for value in row["gates"].values())
    ):
        raise ValueError("selected candidate is not a source-bound T4 PASS")
    protocol_snapshot = snapshots[Path(records["protocol"]["path"])]
    protocol = _load_json(protocol_snapshot, "T4 protocol")
    if set(protocol) != {
        "schema_version", "manifest_id", "dataset", "method_id", "protocol_id",
        "scene", "bounds", "candidates",
    } or not (
        protocol.get("schema_version") == 1
        and protocol.get("manifest_id") == "oviv2_tesse_t4_protocol_v1"
        and protocol.get("dataset") == "TESSE-CD"
        and protocol.get("method_id") == "OVIV2"
        and protocol.get("protocol_id") == FREEZE_ID
        and protocol.get("scene") == "apartment"
        and protocol.get("bounds") == T4_BOUNDS
        and isinstance(protocol.get("candidates"), Mapping)
        and set(protocol["candidates"]) == set(candidates)
    ):
        raise ValueError("T4 protocol identity or Apartment scope is invalid")
    source_paths: set[Path] = set()
    for candidate, matrix_row in candidates.items():
        protocol_row = protocol["candidates"][candidate]
        if not isinstance(protocol_row, Mapping) or set(protocol_row) != {
            "config_sha256", "run_manifest", "metric_sources"
        } or protocol_row.get("config_sha256") != matrix_row.get("config_sha256"):
            raise ValueError("T4 protocol candidate binding is invalid")
        run_record, run_snapshot = _verify_record(
            protocol_row.get("run_manifest"),
            repo_root=repo_root,
            role=f"T4 {candidate} run manifest",
            snapshots=snapshots,
        )
        run_path = Path(run_record["path"])
        run_manifest = _load_json(run_snapshot, f"T4 {candidate} run manifest")
        if run_path in source_paths or not (
            run_record["sha256"] == matrix_row.get("run_manifest_sha256")
            and run_manifest.get("dataset") == "TESSE-CD"
            and run_manifest.get("method_id") == "OVIV2"
            and run_manifest.get("protocol_id") == FREEZE_ID
            and run_manifest.get("scene") == "apartment"
            and run_manifest.get("candidate_id") == candidate
            and run_manifest.get("config_sha256") == matrix_row.get("config_sha256")
        ):
            raise ValueError("T4 whole-profile run manifest binding is invalid")
        source_paths.add(run_path)
        metric_sources = protocol_row.get("metric_sources")
        if not isinstance(metric_sources, Mapping) or set(metric_sources) != V2_T4_METRIC_KEYS:
            raise ValueError("T4 metric source inventory is invalid")
        for metric, metric_record_value in metric_sources.items():
            metric_record, metric_snapshot = _verify_record(
                metric_record_value,
                repo_root=repo_root,
                role=f"T4 {candidate} {metric}",
                snapshots=snapshots,
            )
            metric_path = Path(metric_record["path"])
            metric_payload = _load_json(metric_snapshot, f"T4 {candidate} {metric}")
            if metric_path in source_paths or set(metric_payload) != {
                "schema_version", "manifest_id", "scene", "candidate_id",
                "config_sha256", "run_manifest_sha256", "metric", "value",
            } or not (
                metric_payload.get("schema_version") == 1
                and metric_payload.get("manifest_id") == "oviv2_tesse_t4_metric_v1"
                and metric_payload.get("scene") == "apartment"
                and metric_payload.get("candidate_id") == candidate
                and metric_payload.get("config_sha256") == matrix_row.get("config_sha256")
                and metric_payload.get("run_manifest_sha256") == run_record["sha256"]
                and metric_payload.get("metric") == metric
                and type(metric_payload.get("value")) in {int, float}
                and math.isfinite(metric_payload["value"])
                and metric_payload.get("value") == matrix_row["metrics"][metric]
            ):
                raise ValueError("T4 metric is not whole-profile source-bound")
            source_paths.add(metric_path)
    return {"artifact": artifact.record, **records, "root_sha256": claimed_root}


def _seed_policy(declared: object = None) -> dict[str, Any]:
    if declared is None:
        policy = {"behavior": "deterministic", "seeds": [0]}
    elif isinstance(declared, str):
        policy = {
            "behavior": declared,
            "seeds": [0] if declared == "deterministic" else [17, 29, 43, 71, 101],
        }
    elif isinstance(declared, Mapping):
        policy = {
            "behavior": declared.get("behavior", declared.get("mode")),
            "seeds": declared.get("seeds"),
        }
    else:
        raise ValueError("seed policy declaration is invalid")
    expected = (
        [0]
        if policy["behavior"] == "deterministic"
        else [17, 29, 43, 71, 101]
    )
    if (
        policy["behavior"] not in {"deterministic", "stochastic"}
        or policy["seeds"] != expected
    ):
        raise ValueError("seed policy does not use the pre-registered seeds")
    return {**policy, "sha256": _json_hash(policy)}


def _default_dependencies() -> FreezeDependencies:
    return FreezeDependencies(
        repository_inspector=_inspect_repository,
        environment_collector=_default_environment,
        release_paths=DEFAULT_RELEASE_PATHS,
        python_executable=Path(sys.executable).resolve(),
        t1_transaction_verifier=verify_exact_profile_runs,
    )


def _publish_no_replace(
    output: Path,
    content: bytes,
    *,
    snapshots: Mapping[Path, _Snapshot],
    repository_before: Mapping[str, Any],
    inspect_repository: Callable[[Path], Mapping[str, Any]],
    repo_root: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    name = f".{output.name}.tmp-{secrets.token_hex(12)}"
    descriptor: int | None = None
    parent_descriptor: int | None = None
    published = False
    try:
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        parent_status = os.fstat(parent_descriptor)
        parent_identity = (parent_status.st_dev, parent_status.st_ino)
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
            dir_fd=parent_descriptor,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while staging freeze manifest")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        for path, snapshot in snapshots.items():
            _revalidate_snapshot(snapshot, f"frozen input {path}")
        current_repository = _validate_repository(inspect_repository(repo_root))
        if current_repository != dict(repository_before):
            raise ValueError("repository changed during freeze")
        os.link(
            name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = True
        for path, snapshot in snapshots.items():
            _revalidate_snapshot(snapshot, f"frozen input {path}")
        current_repository = _validate_repository(inspect_repository(repo_root))
        if current_repository != dict(repository_before):
            raise ValueError("repository changed during freeze")
        current_parent = os.stat(output.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current_parent.st_mode)
            or (current_parent.st_dev, current_parent.st_ino) != parent_identity
        ):
            raise ValueError("freeze output parent changed during publication")
        os.fsync(parent_descriptor)
    except BaseException:
        if published and parent_descriptor is not None:
            try:
                os.unlink(output.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.close(parent_descriptor)


def freeze(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    dependencies: FreezeDependencies | None = None,
) -> dict[str, Any]:
    root = _absolute(repo_root or REPO_ROOT)
    _reject_symlink_components(root, "repository root")
    deps = dependencies or _default_dependencies()
    repository = _validate_repository(deps.repository_inspector(root))
    environment = _validate_environment(deps.environment_collector())
    python = _absolute(deps.python_executable)
    python_snapshot = _snapshot(python, "active Python interpreter")
    if not os.access(python, os.X_OK):
        raise ValueError("active Python interpreter is not executable")
    snapshots: dict[Path, _Snapshot] = {python: python_snapshot}
    configs, scenes = _load_configs(args, repo_root=root, snapshots=snapshots)
    if _office_evidence(configs["office"]):
        raise ValueError("Office result evidence exists before freeze")
    shared = _shared_bindings(configs, repo_root=root, snapshots=snapshots)
    selection_path = _absolute(args.selection, base=root)
    selection_payload, selection = _validate_selection(
        selection_path, configs, repo_root=root, snapshots=snapshots
    )
    t1 = _freeze_t1_evidence(
        _required_evidence_path(args, "t1_evidence", root),
        repo_root=root,
        snapshots=snapshots,
        transaction_verifier=deps.t1_transaction_verifier,
    )
    t4 = _freeze_t4_evidence(
        _required_evidence_path(args, "t4_evidence", root),
        selected_candidate=selection_payload["selected_candidate_id"],
        selected_config_sha256=selection["selected_config_sha256"],
        repo_root=root,
        snapshots=snapshots,
    )
    if not (
        t4["artifact"] == selection_payload["t4_matrix"]
        and t4["protocol"] == selection_payload["t4_protocol"]
        and t4["root_sha256"] == selection_payload["t4_root_sha256"]
    ):
        raise ValueError("T4 evidence input differs from final selection bindings")
    evidence = {"t1": t1, "t4": t4}
    seed_policy = _seed_policy(getattr(args, "seed_policy", None))
    models = _models(scenes, snapshots)
    release = _release_bindings(deps.release_paths, snapshots)
    output = _absolute(args.output, base=root)
    roots = _validate_outputs(
        output, _output_roots(output, seed_policy["seeds"]), input_paths=set(snapshots)
    )
    commands = _mapping_commands(
        python=python,
        configs=configs,
        scenes=scenes,
        output=output,
        roots=roots,
        repo_root=root,
    )
    algorithm_hash = configs["apartment"]["algorithm_hash"]
    frozen_hashes = {
        "code_sha256": _json_hash(
            {"commit": repository["commit"], "tree": repository["tree"]}
        ),
        "config_sha256": {
            scene: scenes[scene]["frozen_config"]["sha256"] for scene in SCENES
        },
        "evaluator_sha256": {
            role: record["sha256"] for role, record in release.items()
        },
        "ground_truth_sha256": shared["occlusion_target_manifest"]["sha256"],
        "schedule_sha256": shared["schedule"]["sha256"],
        "seed_policy_sha256": seed_policy["sha256"],
    }
    frozen_hashes_sha256 = _json_hash(frozen_hashes)
    office_authorizations: dict[str, Any] = {}
    for seed in seed_policy["seeds"]:
        slot = f"office_seed_{seed}"
        authorization = {
            "authorization_id": slot,
            "scene": "office",
            "seed": seed,
            "config_sha256": _json_hash(configs["office"]),
            "algorithm_hash": algorithm_hash,
            "output_root": roots[slot],
            "t1_root_sha256": t1["root_sha256"],
            "t4_root_sha256": t4["root_sha256"],
            "frozen_hashes_sha256": frozen_hashes_sha256,
        }
        office_authorizations[slot] = {
            **authorization,
            "authorization_sha256": _json_hash(authorization),
        }
    manifest = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "status": "FROZEN",
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "repository": repository,
        "algorithm": {
            "sha256": algorithm_hash,
            "normalized_config": canonical_algorithm_config(configs["apartment"]),
        },
        "scenes": scenes,
        "shared_bindings": shared,
        "output_roots": roots,
        "selection": selection,
        "environment": environment,
        "commands": commands,
        "models": models,
        "release_bindings": release,
        "office_pre_freeze_audit": {
            "selection_scene": "apartment",
            "metric_sources_found": [],
            "office_outputs_read": False,
        },
        "evidence": evidence,
        "seed_policy": seed_policy,
        "frozen_hashes": frozen_hashes,
        "office_authorizations": office_authorizations,
    }
    if set(manifest) != V2_FREEZE_TOP_KEYS:
        raise AssertionError("runner freeze schema changed")
    if set(manifest["office_pre_freeze_audit"]) != V2_FREEZE_OFFICE_AUDIT_KEYS:
        raise AssertionError("runner Office audit schema changed")
    if any(set(scene) != V2_FREEZE_SCENE_KEYS for scene in scenes.values()):
        raise AssertionError("runner scene schema changed")
    if set(shared) != V2_FREEZE_SHARED_KEYS or set(selection) != V2_FREEZE_SELECTION_KEYS:
        raise AssertionError("runner binding schema changed")
    if set(release) != V2_FREEZE_RELEASE_KEYS:
        raise AssertionError("runner release schema changed")
    content = _canonical_json_bytes(manifest)
    _publish_no_replace(
        output,
        content,
        snapshots=snapshots,
        repository_before=repository,
        inspect_repository=deps.repository_inspector,
        repo_root=root,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-config", type=Path, required=True)
    parser.add_argument("--office-config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--t1-evidence", type=Path, required=True)
    parser.add_argument("--t4-evidence", type=Path, required=True)
    parser.add_argument(
        "--seed-policy",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = freeze(args)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"freeze failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"FROZEN {manifest['freeze_id']} algorithm={manifest['algorithm']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
