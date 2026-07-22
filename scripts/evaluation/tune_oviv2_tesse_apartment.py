#!/usr/bin/env python3
"""Tune the predeclared OVIV2 Stage3 maintenance grid on TESSE-CD Apartment."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_oviv2_tesse_cd import algorithm_hash  # noqa: E402


TUNABLE_FIELDS = frozenset(
    {
        "visibility_depth_tolerance_m",
        "absence_negative_support",
        "ownership_min_net_support",
    }
)
GRID = {
    "visibility_depth_tolerance_m": (0.05, 0.10, 0.15),
    "absence_negative_support": (0.5, 1.0),
    "ownership_min_net_support": (0.000001, 0.5, 1.0),
}
STAGE3_LINEAGE_COMMIT = "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
SELECTION_RULE = (
    "maximize current_miou",
    "minimize ghost_rate",
    "maximize background_f5_cm",
    "minimize recovery_frames",
    "minimize config_sha256",
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_line(payload: object) -> bytes:
    return _canonical_json(payload) + b"\n"


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    config: dict[str, Any]
    config_bytes: bytes
    config_sha256: str


@dataclass(frozen=True)
class CandidateExecution:
    candidate: CandidateConfig
    config_path: Path
    artifact_root: Path
    commands: tuple[tuple[str, ...], ...]


def _validate_base_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("dataset") != "TESSE-CD"
        or config.get("method_id") != "OVIV2"
        or config.get("scene") != "apartment"
        or config.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
    ):
        raise ValueError("base config must be the TESSE-CD Apartment Stage3 config")
    if not TUNABLE_FIELDS <= config.keys():
        raise ValueError("base config is missing maintenance parameters")
    serialized = _canonical_json(config).lower()
    if any(token in serialized for token in (b"route3", b"scannet200", b"stage4")):
        raise ValueError("base config contains a forbidden non-Stage3 implementation")
    frozen_hash = config.get("algorithm_hash")
    if frozen_hash != algorithm_hash(config):
        raise ValueError("base config algorithm_hash does not match Stage3 parameters")
    canonical_path = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v1.json"
    canonical, _ = _load_json(canonical_path, "canonical Apartment Stage3 config")
    if dict(config) != canonical:
        raise ValueError("base config does not match the canonical Stage3 base")


def build_candidate_configs(config: Mapping[str, Any]) -> tuple[CandidateConfig, ...]:
    _validate_base_config(config)
    candidates: list[CandidateConfig] = []
    combinations = itertools.product(
        GRID["visibility_depth_tolerance_m"],
        GRID["absence_negative_support"],
        GRID["ownership_min_net_support"],
    )
    for index, (tolerance, absence, ownership) in enumerate(combinations):
        candidate = dict(config)
        candidate.update(
            {
                "visibility_depth_tolerance_m": tolerance,
                "absence_negative_support": absence,
                "ownership_min_net_support": ownership,
            }
        )
        candidate["algorithm_hash"] = algorithm_hash(candidate)
        encoded = _canonical_json(candidate)
        candidates.append(
            CandidateConfig(
                candidate_id=f"candidate-{index:02d}",
                config=candidate,
                config_bytes=encoded,
                config_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
    return tuple(candidates)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, path: Path) -> dict[str, Any]:
    payload = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _reject_symlink_components(path, label)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {label}: {path}")
    data = path.read_bytes()
    return _load_json_bytes(data, path), data


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    _reject_symlink_components(path, "result artifact")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing result artifact: {path}")
    data = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _stable_json_record(
    path: Path,
    label: str,
    *,
    relative_to: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def witness(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    try:
        path_status = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} changed while it was read") from exc
    if witness(before) != witness(after) or witness(after) != witness(path_status):
        raise ValueError(f"{label} changed while it was read")
    _reject_symlink_components(path, label)
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    payload = _load_json_bytes(data, path)
    return payload, {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _absolute_file_record(path: Path) -> dict[str, Any]:
    _reject_symlink_components(path, "freeze binding")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing freeze binding: {path}")
    data = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _resolve_repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path must be a non-empty string")
    raw = Path(value).expanduser()
    return (REPO_ROOT / raw if not raw.is_absolute() else raw).absolute()


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


def _declared_file_record(
    value: object,
    label: str,
    *,
    verify_bytes: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding must be an object")
    path = _resolve_repo_path(value.get("path"), label)
    _reject_symlink_components(path, label)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {label}: {path}")
    declared_hash = _sha256(value.get("sha256"), f"{label} sha256")
    if verify_bytes and hashlib.sha256(path.read_bytes()).hexdigest() != declared_hash:
        raise ValueError(f"{label} checksum binding mismatch")
    byte_count = path.stat().st_size
    declared_size = value.get("byte_count")
    if declared_size is not None and (
        type(declared_size) is not int or declared_size != byte_count
    ):
        raise ValueError(f"{label} byte_count binding mismatch")
    return {
        "path": str(path),
        "sha256": declared_hash,
        "byte_count": byte_count,
    }


def _config_file_record(config: Mapping[str, Any], field: str, scene: str) -> dict[str, Any]:
    return _absolute_file_record(_resolve_repo_path(config.get(field), f"{scene} {field}"))


def _build_freeze_bindings(
    apartment_config: Mapping[str, Any],
    *,
    office_config_path: Path,
    aliases_path: Path,
) -> dict[str, Any]:
    office_config, _ = _load_json(office_config_path, "Office input config")
    if (
        office_config.get("schema_version") != 1
        or office_config.get("dataset") != "TESSE-CD"
        or office_config.get("method_id") != "OVIV2"
        or office_config.get("scene") != "office"
        or office_config.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
    ):
        raise ValueError("Office input config identity mismatch")
    if office_config.get("algorithm_hash") != algorithm_hash(office_config):
        raise ValueError("Office input config algorithm_hash is stale")
    input_path = _resolve_repo_path(
        apartment_config.get("input_manifest"), "Apartment input manifest"
    )
    if input_path != _resolve_repo_path(
        office_config.get("input_manifest"), "Office input manifest"
    ):
        raise ValueError("Apartment and Office must share one input manifest")
    input_manifest, _ = _load_json(input_path, "runner input manifest")
    scenes = input_manifest.get("scenes")
    if (
        input_manifest.get("schema_version") != 1
        or input_manifest.get("dataset") != "TESSE-CD"
        or input_manifest.get("stage3_lineage_commit") != STAGE3_LINEAGE_COMMIT
        or not isinstance(scenes, Mapping)
        or set(scenes) != {"apartment", "office"}
    ):
        raise ValueError("runner input manifest identity mismatch")
    expected_aliases = (
        REPO_ROOT / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml"
    ).absolute()
    if aliases_path.absolute() != expected_aliases:
        raise ValueError("Apartment tuning must use the checked common-v2 alias map")

    shared = {
        "input_manifest": _absolute_file_record(input_path),
        "source_manifest": _declared_file_record(
            input_manifest.get("source_manifest"), "source manifest"
        ),
        "schedule": _declared_file_record(
            input_manifest.get("schedule_manifest"), "causal schedule"
        ),
        "camera": _declared_file_record(input_manifest.get("camera"), "camera"),
        "alias_map": _absolute_file_record(expected_aliases),
        "evaluator": _absolute_file_record(
            REPO_ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"
        ),
        "common_finalizer": _absolute_file_record(
            REPO_ROOT / "scripts/evaluation/finalize_tesse_common_v2.py"
        ),
        "official_finalizer": _absolute_file_record(
            REPO_ROOT / "scripts/evaluation/finalize_tesse_t2.py"
        ),
    }
    scene_bindings: dict[str, Any] = {}
    for scene, config in (
        ("apartment", apartment_config),
        ("office", office_config),
    ):
        locked = scenes.get(scene)
        if not isinstance(locked, Mapping):
            raise ValueError(f"{scene} input binding is missing")
        export_path = _resolve_repo_path(config.get("export_manifest"), f"{scene} export")
        export_manifest, _ = _load_json(export_path, f"{scene} export manifest")
        database_path = _resolve_repo_path(
            export_manifest.get("source_database"), f"{scene} source database"
        )
        database_hash = _sha256(
            export_manifest.get("source_database_sha256"),
            f"{scene} source database sha256",
        )
        if locked.get("source_database_sha256") != database_hash:
            raise ValueError(f"{scene} source database binding mismatch")
        database = _declared_file_record(
            {"path": str(database_path), "sha256": database_hash},
            f"{scene} source database",
            verify_bytes=False,
        )
        scene_bindings[scene] = {
            "database": database,
            "timestamps": _declared_file_record(
                locked.get("timestamps"), f"{scene} timestamps"
            ),
            "trajectory": _declared_file_record(
                locked.get("trajectory"), f"{scene} trajectory"
            ),
            "export_manifest": _absolute_file_record(export_path),
            "frontend_manifest": _config_file_record(
                config, "frontend_manifest", scene
            ),
            "dense_manifest": _config_file_record(config, "dense_manifest", scene),
            "vocabulary_json": _config_file_record(config, "vocabulary_json", scene),
            "vocabulary_txt": _config_file_record(config, "vocabulary_txt", scene),
        }
    return {"shared": shared, "scenes": scene_bindings}


def _write_exclusive(path: Path, data: bytes) -> None:
    _reject_symlink_components(path.parent, "output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path.parent, "output parent")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _write_exclusive(temporary, _canonical_json_line(payload))
    os.replace(temporary, path)


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic no-clobber directory publication is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(error_number, os.strerror(error_number), target)
    descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commands_for_candidate(
    candidate: CandidateConfig,
    *,
    target_manifest: Path,
    aliases: Path,
    label_space: Path,
) -> tuple[tuple[str, ...], ...]:
    candidate_id = candidate.candidate_id
    config = f"configs/{candidate_id}.json"
    artifact = f"candidates/{candidate_id}"
    return (
        (
            str(Path(sys.executable).absolute()),
            str(REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd.py"),
            "--config",
            config,
            "--output",
            f"{artifact}/run",
        ),
        (
            str(Path(sys.executable).absolute()),
            str(REPO_ROOT / "scripts/evaluation/export_tesse_temporal_artifact.py"),
            "--source-index",
            f"{artifact}/run/source_index.json",
            "--output",
            f"{artifact}/temporal",
        ),
        (
            str(Path(sys.executable).absolute()),
            str(REPO_ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"),
            "--temporal-index",
            f"{artifact}/temporal/temporal_manifest.json",
            "--target-manifest",
            str(target_manifest.absolute()),
            "--aliases",
            str(aliases.absolute()),
            "--label-space",
            str(label_space.absolute()),
            "--output",
            f"{artifact}/evaluation",
        ),
    )


def _plan_payload(
    executions: Sequence[CandidateExecution],
    *,
    status: str,
    max_parallel: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_apartment_tuning_commands_v1",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "status": status,
        "scene": "apartment",
        "candidate_count": len(executions),
        "max_parallel": max_parallel,
        "grid": {name: list(values) for name, values in GRID.items()},
        "candidates": [
            {
                "candidate_id": item.candidate.candidate_id,
                "config": f"configs/{item.candidate.candidate_id}.json",
                "config_sha256": item.candidate.config_sha256,
                "artifact_root": f"candidates/{item.candidate.candidate_id}",
                "commands": [list(command) for command in item.commands],
            }
            for item in executions
        ],
    }


def _default_execute_candidate(item: CandidateExecution) -> None:
    if item.artifact_root.exists() or item.artifact_root.is_symlink():
        raise FileExistsError(item.artifact_root)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    cwd = item.artifact_root.parents[1]
    for command in item.commands:
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate metric is invalid: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite candidate metric: {name}")
    return result


def selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("candidate selection record is invalid")
    return (
        -_metric(metrics, "current_miou"),
        _metric(metrics, "ghost_rate"),
        -_metric(metrics, "background_f5_cm"),
        _metric(metrics, "recovery_frames"),
        _sha256(record.get("config_sha256"), "config sha256"),
    )


def _candidate_record(
    item: CandidateExecution,
    *,
    output: Path,
    published_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_record = _file_record(item.config_path, relative_to=output)
    if (
        config_record["sha256"] != item.candidate.config_sha256
        or config_record["byte_count"] != len(item.candidate.config_bytes)
    ):
        raise ValueError("candidate config changed during tuning")
    run_path = item.artifact_root / "run/run_manifest.json"
    run_manifest, run_record = _stable_json_record(
        run_path,
        "candidate run manifest",
        relative_to=output,
    )
    if (
        run_manifest.get("schema_version") != 1
        or run_manifest.get("dataset") != "TESSE-CD"
        or run_manifest.get("method_id") != "OVIV2"
        or run_manifest.get("mode") != "causal_checkpoints"
        or run_manifest.get("scene") != "apartment"
        or run_manifest.get("algorithm_hash")
        != item.candidate.config["algorithm_hash"]
        or run_manifest.get("config")
        != {
            "sha256": item.candidate.config_sha256,
            "byte_count": len(item.candidate.config_bytes),
        }
        or run_manifest.get("maintenance_parameters")
        != {
            field: item.candidate.config[field]
            for field in sorted(TUNABLE_FIELDS)
        }
    ):
        raise ValueError("candidate run identity does not match its Apartment config")

    summary_path = item.artifact_root / "evaluation/summary.json"
    summary, evaluator_record = _stable_json_record(
        summary_path,
        "candidate evaluator summary",
        relative_to=output,
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("manifest_id")
        != "tesse_cd_common_v2_scene_summary"
        or summary.get("dataset") != "TESSE-CD"
        or summary.get("protocol") != "tesse_cd_common_v2"
        or summary.get("status") != "PASS"
        or summary.get("method") != "OVIV2"
        or summary.get("mode") != "causal_checkpoints"
    ):
        raise ValueError("candidate evaluator summary must be common-v2 PASS")
    if summary.get("scene") != "apartment":
        raise ValueError("candidate evaluator summary must be Apartment only")
    raw_metrics = summary.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("candidate evaluator metrics are missing")
    metrics = {
        "current_miou": _metric(raw_metrics, "current_miou"),
        "ghost_rate": _metric(raw_metrics, "ghost_rate"),
        "background_f5_cm": _metric(raw_metrics, "background_f5"),
        "recovery_frames": _metric(raw_metrics, "recovery_frames"),
    }
    sources = summary.get("sources")
    target = sources.get("target_manifest") if isinstance(sources, Mapping) else None
    if not isinstance(target, Mapping):
        raise ValueError("candidate summary is missing target manifest binding")
    target_path = target.get("path")
    target_size = target.get("byte_count")
    if (
        not isinstance(target_path, str)
        or not target_path.strip()
        or type(target_size) is not int
        or target_size <= 0
    ):
        raise ValueError("candidate target manifest binding is invalid")
    target_record = {
        "path": target_path,
        "sha256": _sha256(target.get("sha256"), "target manifest sha256"),
        "byte_count": target_size,
    }
    summary_payload = {
        "schema_version": 1,
        "status": "PASS",
        "scene": "apartment",
        "candidate_id": item.candidate.candidate_id,
        "parameters": {
            field: item.candidate.config[field] for field in sorted(TUNABLE_FIELDS)
        },
        "config_sha256": item.candidate.config_sha256,
        "common_target_manifest_sha256": target_record["sha256"],
        "metrics": metrics,
        "config": config_record,
        "run_identity": run_record,
        "evaluator_summary": evaluator_record,
    }
    candidate_summary_path = item.artifact_root / "candidate_summary.json"
    _write_atomic(candidate_summary_path, summary_payload)
    summary_record = _absolute_file_record(candidate_summary_path)
    summary_record["path"] = str(
        published_output / candidate_summary_path.relative_to(output)
    )
    record = {
        **summary_payload,
        "summary": summary_record,
    }
    return record, target_record


def _build_selection(
    executions: Sequence[CandidateExecution],
    *,
    output: Path,
    published_output: Path,
    base_config_sha256: str,
    freeze_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = {item.candidate.candidate_id for item in executions}
    candidates_root = output / "candidates"
    entries = list(candidates_root.iterdir()) if candidates_root.is_dir() else []
    if any(path.is_symlink() or not path.is_dir() for path in entries):
        raise ValueError("candidate result path contains a symlink or non-directory")
    actual_ids = {path.name for path in entries}
    if actual_ids != expected_ids:
        raise ValueError("missing or extra candidate result directories")
    records: list[dict[str, Any]] = []
    target_records: dict[bytes, dict[str, Any]] = {}
    for item in executions:
        record, target_record = _candidate_record(
            item,
            output=output,
            published_output=published_output,
        )
        records.append(record)
        target_records[_canonical_json(target_record)] = target_record
    if len(records) != 18 or len({item["candidate_id"] for item in records}) != 18:
        raise ValueError("candidate results are missing or duplicated")
    if len(target_records) != 1:
        raise ValueError("candidate target manifest hash mismatch")
    records.sort(key=lambda item: str(item["candidate_id"]))
    selected = min(records, key=selection_key)
    target_record = next(iter(target_records.values()))
    return {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_apartment_selection_v1",
        "dataset": "TESSE-CD",
        "method": "OVIV2",
        "method_id": "OVIV2",
        "status": "PASS",
        "scene": "apartment",
        "base_config_sha256": base_config_sha256,
        "grid": {name: list(values) for name, values in GRID.items()},
        "parameter_grid": {name: list(values) for name, values in GRID.items()},
        "candidate_count": len(records),
        "selection_rule": list(SELECTION_RULE),
        "common_target_manifest": target_record,
        "common_v2_target_manifest_sha256": target_record["sha256"],
        "freeze_bindings": dict(freeze_bindings),
        "selected_candidate_id": selected["candidate_id"],
        "selected_config": selected["config"],
        "selected_config_sha256": selected["config_sha256"],
        "selected_maintenance_parameters": selected["parameters"],
        "selected_metrics": selected["metrics"],
        "candidates": records,
    }


def run_tuning(
    *,
    base_config: str | Path,
    output: str | Path,
    target_manifest: str | Path,
    aliases: str | Path,
    label_space: str | Path,
    office_config: str | Path = REPO_ROOT / "configs/oviv2_tesse_cd_office_v1.json",
    mode: str = "run",
    max_parallel: int = 3,
    dry_run: bool = False,
    execute_candidate: Callable[[CandidateExecution], None] | None = None,
) -> Path | dict[str, Any]:
    if mode not in {"run", "commands"}:
        raise ValueError("mode must be run or commands")
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int):
        raise ValueError("max_parallel must be an integer")
    if not 1 <= max_parallel <= 3:
        raise ValueError("max_parallel must be between one and three")
    base_path = Path(base_config).absolute()
    base, base_bytes = _load_json(base_path, "Apartment base config")
    candidates = build_candidate_configs(base)
    destination = Path(output).absolute()
    target_path = Path(target_manifest).absolute()
    aliases_path = Path(aliases).absolute()
    label_space_path = Path(label_space).absolute()
    office_config_path = Path(office_config).absolute()

    def executions_for(root: Path) -> tuple[CandidateExecution, ...]:
        return tuple(
            CandidateExecution(
                candidate=candidate,
                config_path=root / f"configs/{candidate.candidate_id}.json",
                artifact_root=root / f"candidates/{candidate.candidate_id}",
                commands=_commands_for_candidate(
                    candidate,
                    target_manifest=target_path,
                    aliases=aliases_path,
                    label_space=label_space_path,
                ),
            )
            for candidate in candidates
        )

    if dry_run:
        return _plan_payload(
            executions_for(destination),
            status="DRY_RUN",
            max_parallel=max_parallel,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    working = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    executions = executions_for(working)
    try:
        (working / "configs").mkdir()
        (working / "candidates").mkdir()
        for item in executions:
            _write_exclusive(item.config_path, item.candidate.config_bytes)
        commands_payload = _plan_payload(
            executions,
            status="COMMANDS",
            max_parallel=max_parallel,
        )
        _write_atomic(working / "commands.json", commands_payload)
        if mode == "commands":
            _write_atomic(
                working / "sweep_status.json",
                {
                    "schema_version": 1,
                    "status": "COMMANDS",
                    "scene": "apartment",
                    "candidate_count": 18,
                },
            )
            _publish_directory_no_replace(working, destination)
            return destination / "commands.json"

        freeze_bindings = _build_freeze_bindings(
            base,
            office_config_path=office_config_path,
            aliases_path=aliases_path,
        )
        runner = execute_candidate or _default_execute_candidate
        for offset in range(0, len(executions), max_parallel):
            batch = executions[offset : offset + max_parallel]
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = [executor.submit(runner, item) for item in batch]
                for future in futures:
                    future.result()
        selection = _build_selection(
            executions,
            output=working,
            published_output=destination,
            base_config_sha256=hashlib.sha256(base_bytes).hexdigest(),
            freeze_bindings=freeze_bindings,
        )
        _write_atomic(working / "selection.json", selection)
        _write_atomic(
            working / "sweep_status.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "scene": "apartment",
                "candidate_count": 18,
                "selection_sha256": hashlib.sha256(
                    (working / "selection.json").read_bytes()
                ).hexdigest(),
            },
        )
        _publish_directory_no_replace(working, destination)
        return destination / "selection.json"
    except BaseException:
        shutil.rmtree(working, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--label-space", type=Path, required=True)
    parser.add_argument(
        "--office-config",
        type=Path,
        default=REPO_ROOT / "configs/oviv2_tesse_cd_office_v1.json",
        help="Office input config used only to bind freeze inputs; no Office result is read",
    )
    parser.add_argument("--mode", choices=("run", "commands"), default="run")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tuning(
        base_config=args.base_config,
        output=args.output,
        target_manifest=args.target_manifest,
        aliases=args.aliases,
        label_space=args.label_space,
        office_config=args.office_config,
        mode=args.mode,
        max_parallel=args.max_parallel,
        dry_run=args.dry_run,
    )
    if isinstance(result, Path):
        payload: object = {"output": str(result)}
    else:
        payload = result
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
