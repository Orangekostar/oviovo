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
    _V2_CONFIG_KEYS,
    _validate_config,
)


FREEZE_ID = "oviv2-tessecd-v2"
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SCENES = ("apartment", "office")
CANDIDATES = tuple(f"a{index}" for index in range(5))
SOURCE_ROLES = frozenset(
    {
        "search_manifest",
        "search_status",
        "candidate_config",
        "run_manifest",
        "common_v2_summary",
        "temporal_occlusion_result",
        "official_metrics",
        "t1_exact_evidence",
        "determinism_evidence",
    }
)
CANDIDATE_SOURCE_ROLES = frozenset(
    {
        "candidate_config",
        "run_manifest",
        "common_v2_summary",
        "temporal_occlusion_result",
        "official_metrics",
    }
)
_OFFICE_PATH_TOKEN = re.compile(r"(^|[^a-z0-9])office([^a-z0-9]|$)")
SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "protocol_id",
        "status",
        "development_scene",
        "transfer_scene",
        "scenes_read",
        "office_results_read",
        "selected_candidate_id",
        "selected_config",
        "selected_config_record",
        "selected_config_sha256",
        "algorithm_hash",
        "manifest",
        "result_contract",
        "results_root",
        "result_files",
        "promotion_order",
        "metric_policy",
        "floors",
        "skipped_optional_tie_axes",
        "rejection_ledger",
    }
)
LEDGER_KEYS = frozenset(
    {
        "candidate_id",
        "eligible",
        "selected",
        "reasons",
        "notes",
        "config_sha256",
        "algorithm_hash",
        "gates",
        "metrics",
        "sources",
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


@dataclass(frozen=True)
class FreezeDependencies:
    repository_inspector: Callable[[Path], Mapping[str, Any]]
    environment_collector: Callable[[], Mapping[str, Any]]
    release_paths: Mapping[str, Path]
    python_executable: Path


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
    snapshot = _snapshot(path, "Apartment selection")
    snapshots[path] = snapshot
    selection = _load_json(snapshot, "Apartment selection")
    if set(selection) != SELECTION_KEYS:
        raise ValueError("selection schema has missing or surplus fields")
    apartment = configs["apartment"]
    selected_hash = _json_hash(apartment)
    algorithm_hash = apartment["algorithm_hash"]
    if not (
        selection.get("schema_version") == 1
        and selection.get("manifest_id") == "oviv2_tesse_cd_v2_selection"
        and selection.get("dataset") == "TESSE-CD"
        and selection.get("method_id") == "OVIV2"
        and selection.get("protocol_id") == FREEZE_ID
        and selection.get("status") == "PASS"
        and selection.get("development_scene") == "apartment"
        and selection.get("transfer_scene") == "office"
        and selection.get("scenes_read") == ["apartment"]
        and selection.get("office_results_read") is False
        and selection.get("selected_config") == apartment
        and selection.get("selected_config_sha256") == selected_hash
        and selection.get("algorithm_hash") == algorithm_hash
    ):
        raise ValueError("selection identity or selected config drift")
    selected_record, selected_snapshot = _verify_record(
        selection.get("selected_config_record"),
        repo_root=repo_root,
        role="selected candidate config",
        snapshots=snapshots,
    )
    if _load_json(selected_snapshot, "selected candidate config") != apartment:
        raise ValueError("selected config record differs from frozen Apartment config")
    manifest_record, _ = _verify_record(
        selection.get("manifest"),
        repo_root=repo_root,
        role="selection search manifest",
        snapshots=snapshots,
    )
    del selected_record, manifest_record
    if not (
        selection.get("result_contract")
        == "candidates/<candidate_id>/apartment/result.json"
        and isinstance(selection.get("results_root"), str)
        and Path(selection["results_root"]).is_absolute()
        and type(selection.get("promotion_order")) is list
        and bool(selection["promotion_order"])
        and isinstance(selection.get("metric_policy"), Mapping)
        and isinstance(selection.get("floors"), Mapping)
        and isinstance(selection.get("skipped_optional_tie_axes"), Mapping)
    ):
        raise ValueError("selection policy evidence is invalid")
    selected_candidate = selection.get("selected_candidate_id")
    if selected_candidate not in CANDIDATES:
        raise ValueError("selection candidate is undeclared")
    ledger = selection.get("rejection_ledger")
    if type(ledger) is not list or len(ledger) != len(CANDIDATES):
        raise ValueError("selection must bind the complete A0-A4 ledger")
    result_files = selection.get("result_files")
    if type(result_files) is not list or len(result_files) != len(CANDIDATES):
        raise ValueError("selection must bind all A0-A4 result files")
    results_root = _absolute(Path(selection["results_root"]))
    result_candidates: list[str] = []
    result_payloads: dict[str, dict[str, Any]] = {}
    result_paths: set[Path] = set()
    for item in result_files:
        if not isinstance(item, Mapping) or set(item) != {
            "candidate_id",
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError("selection result file schema is invalid")
        candidate = item.get("candidate_id")
        if candidate not in CANDIDATES:
            raise ValueError("selection result file candidate is undeclared")
        verified, result_snapshot = _verify_record(
            {key: item[key] for key in ("path", "sha256", "byte_count")},
            repo_root=repo_root,
            role=f"{candidate} packaged result",
            snapshots=snapshots,
        )
        result_path = Path(verified["path"])
        expected_path = (
            results_root
            / "candidates"
            / candidate
            / "apartment"
            / "result.json"
        )
        if result_path != expected_path or result_path in result_paths:
            raise ValueError("selection result path does not bind its candidate")
        result_paths.add(result_path)
        result_payload = _load_json(result_snapshot, f"{candidate} packaged result")
        if not (
            result_payload.get("schema_version") == 1
            and result_payload.get("manifest_id")
            == "oviv2-tesse-dual-readout-candidate-result-v1"
            and result_payload.get("candidate_id") == candidate
            and result_payload.get("scene") == "apartment"
            and result_payload.get("status") == "PASS"
            and isinstance(result_payload.get("sources"), Mapping)
            and set(result_payload["sources"]) == SOURCE_ROLES
        ):
            raise ValueError("packaged result identity or source roles are invalid")
        result_payloads[candidate] = result_payload
        result_candidates.append(candidate)
    if tuple(result_candidates) != CANDIDATES:
        raise ValueError("selection result files must bind A0-A4 exactly once")
    seen: list[str] = []
    selected_entries = 0
    candidate_source_paths: dict[Path, tuple[str, str]] = {}
    for entry in ledger:
        if not isinstance(entry, Mapping) or set(entry) != LEDGER_KEYS:
            raise ValueError("selection ledger schema is invalid")
        candidate = entry.get("candidate_id")
        reasons = entry.get("reasons")
        if (
            candidate not in CANDIDATES
            or type(entry.get("eligible")) is not bool
            or type(entry.get("selected")) is not bool
            or type(reasons) is not list
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or type(entry.get("notes")) is not list
            or any(
                not isinstance(note, str) or not note for note in entry["notes"]
            )
            or not _is_sha256(entry.get("config_sha256"))
            or not _is_sha256(entry.get("algorithm_hash"))
            or not isinstance(entry.get("gates"), Mapping)
            or not isinstance(entry.get("metrics"), Mapping)
        ):
            raise ValueError("selection ledger contains undeclared or Office evidence")
        sources = entry.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != SOURCE_ROLES:
            raise ValueError("selection ledger source schema is invalid")
        if sources != result_payloads[candidate]["sources"]:
            raise ValueError("selection ledger sources differ from packaged result")
        candidate_root = results_root / "candidates" / candidate / "apartment"
        for role, record in sources.items():
            verified, source_snapshot = _verify_record(
                record,
                repo_root=repo_root,
                role=f"{candidate} {role}",
                snapshots=snapshots,
            )
            source_path = Path(verified["path"])
            if _OFFICE_PATH_TOKEN.search(source_path.as_posix().lower()):
                raise ValueError("selection ledger contains Office result evidence")
            if role in CANDIDATE_SOURCE_ROLES:
                if candidate_root not in source_path.parents:
                    raise ValueError(
                        f"{candidate} {role} is outside its candidate evidence root"
                    )
                previous = candidate_source_paths.get(source_path)
                if previous is not None:
                    raise ValueError(
                        f"candidate-specific source is reused: {previous} and "
                        f"{(candidate, role)}"
                    )
                candidate_source_paths[source_path] = (candidate, role)
            source_payload = _load_json(source_snapshot, f"{candidate} {role}")
            audited_payload = source_payload
            if role == "search_status":
                office_binding = source_payload.get("office_binding")
                if not (
                    isinstance(office_binding, Mapping)
                    and office_binding.get("scene") == "office"
                    and office_binding.get("executed") is False
                ):
                    raise ValueError("search status Office binding is not bind-only")
                audited_payload = dict(source_payload)
                audited_payload.pop("office_binding")
            if _has_office_read_evidence(audited_payload):
                raise ValueError("selection ledger contains Office result evidence")
            if source_payload.get("candidate_id") not in {None, candidate}:
                raise ValueError("candidate source payload identity differs")
            if source_payload.get("role") not in {None, role}:
                raise ValueError("candidate source payload role differs")
            if role == "candidate_config" and candidate == selected_candidate:
                if verified != selection["selected_config_record"]:
                    raise ValueError("selected config source record differs")
                if _load_json(source_snapshot, "selected candidate config") != apartment:
                    raise ValueError("selected candidate config content differs")
        is_selected = entry["selected"]
        if is_selected:
            selected_entries += 1
            if (
                candidate != selected_candidate
                or entry["config_sha256"] != selected_hash
                or entry["algorithm_hash"] != algorithm_hash
                or entry["eligible"] is not True
            ):
                raise ValueError("selected ledger result does not bind selected config")
        elif not reasons:
            raise ValueError("rejected candidates require ledger reasons")
        seen.append(candidate)
    if tuple(seen) != CANDIDATES or selected_entries != 1:
        raise ValueError("selection ledger must bind A0-A4 exactly once")
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


def _output_roots(output: Path) -> dict[str, Path]:
    return {
        f"{scene}_run{repeat}": output.parent / scene / f"run{repeat}"
        for scene in SCENES
        for repeat in (1, 2)
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
    expected_slots = {
        f"{scene}_run{repeat}" for scene in SCENES for repeat in (1, 2)
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
    for scene in SCENES:
        config_path = scenes[scene]["frozen_config"]["path"]
        for repeat in (1, 2):
            slot = f"{scene}_run{repeat}"
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


def _default_dependencies() -> FreezeDependencies:
    return FreezeDependencies(
        repository_inspector=_inspect_repository,
        environment_collector=_default_environment,
        release_paths=DEFAULT_RELEASE_PATHS,
        python_executable=Path(sys.executable).resolve(),
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
    _, selection = _validate_selection(
        selection_path, configs, repo_root=root, snapshots=snapshots
    )
    models = _models(scenes, snapshots)
    release = _release_bindings(deps.release_paths, snapshots)
    output = _absolute(args.output, base=root)
    roots = _validate_outputs(
        output, _output_roots(output), input_paths=set(snapshots)
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
