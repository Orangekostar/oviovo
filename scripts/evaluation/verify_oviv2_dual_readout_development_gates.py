#!/usr/bin/env python3
"""Produce shared T1-exact and determinism development-gate evidence."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.compare_oviv2_cumulative_artifacts import (
    ArtifactMismatch,
    compare_cumulative_artifacts,
)


CUMULATIVE_BASE_COMMIT = "8034e79d9cb853166222610981a7e6893f6cca70"
SCOPE = "shared_code_and_A0-A4_fixture"
SCHEMA_VERSION = 1
MANIFEST_ID = "oviv2_dual_readout_development_gates_v1"
DEVELOPMENT_MODE = "apartment_development_unfrozen"
DEVELOPMENT_EXECUTION_CONTEXT = {
    "mode": DEVELOPMENT_MODE,
    "authorization": "development_only",
    "scene": "apartment",
}
DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024
PROCESS_REAP_TIMEOUT_SECONDS = 5.0
LOCAL_PROCESS_TRUST_MODEL = {
    "pid_semantics": "trusted_local_orchestrator_observation",
    "observation_basis": "parent_popen_and_waitpid",
    "root_ownership": "after_successful_receipt_manifest_observation_reopen_only",
    "audit_authentication": "unsigned_local_audit",
    "stability_scope": "verification_interval_only",
    "excluded_adversaries": ["same_uid_process", "root"],
}

CUMULATIVE_ROOTS = (
    "src/oviv2/runtime.py",
    "src/oviv2/runner_config.py",
    "scripts/run_oviv2_replica.py",
    "scripts/evaluation/run_oviv2_tesse_cd.py",
    "configs/oviv2_tesse_cd_apartment_v1.json",
    "configs/oviv2_tesse_cd_office_v1.json",
)
DEFAULT_SOURCE_MANIFEST = Path(
    "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
)
SOURCE_MANIFEST_ID = "oviv2_t1_transitive_sources_v1"
LOCAL_IMPORT_PREFIXES = ("src", "scripts", "configs")

T1_TEST_FILES = ("tests/oviv2/test_t1_noninterference.py",)
DETERMINISM_TEST_FILES = (
    "tests/oviv2/test_temporal_config.py",
    "tests/oviv2/test_temporal_lifecycle.py",
    "tests/oviv2/test_temporal_association.py",
    "tests/oviv2/test_temporal_geometry.py",
    "tests/oviv2/test_temporal_background.py",
    "tests/oviv2/test_temporal_runtime.py",
    "tests/oviv2/test_temporal_snapshot.py",
    "tests/oviv2/test_dual_readout.py",
    "tests/oviv2/test_reference_readout.py",
    "tests/evaluation/test_oviv2_temporal_tesse.py",
    "tests/evaluation/test_run_oviv2_tesse_cd_v2.py",
)
TEST_FILES = tuple(dict.fromkeys((*T1_TEST_FILES, *DETERMINISM_TEST_FILES)))
SMOKE_TEST = "tests/oviv2/test_temporal_config.py::test_serialization_is_deterministic"

GitRunner = Callable[[tuple[str, ...], Path], bytes]
CommandRunner = Callable[
    [tuple[str, ...], Path], subprocess.CompletedProcess[bytes]
]
PopenFactory = Callable[..., subprocess.Popen[bytes]]
ExactTransactionRunner = Callable[..., dict[str, Any]]


class GateVerificationError(RuntimeError):
    """Raised when evidence cannot be established without ambiguity."""


EXACT_PROFILE_SEQUENCE = (
    "reference",
    "a0",
    "a1",
    "a0",
    "a2",
    "a0",
    "a3",
    "a0",
    "a4",
)
EXACT_EXECUTION_FIELDS = {
    "profile",
    "argv",
    "pid",
    "returncode",
    "code_commit",
    "source_manifest_sha256",
    "profile_config_binding",
    "common_input_fingerprints",
    "output_root",
    "receipt_sha256",
    "run_manifest_sha256",
    "observation_receipt",
}
PROFILE_CONFIG_BINDING_FIELDS = {
    "mode", "config_sha256", "algorithm_hash", "profile_sha256"
}
COMMON_INPUT_FINGERPRINT_FIELDS = {
    "source_manifest_sha256", "input_manifest_sha256",
    "schedule_sha256", "ground_truth_sha256", "source_bindings_sha256",
}


@dataclass(frozen=True)
class _ExactProfileSpec:
    profile: str
    config: Path
    output_root: Path
    source_manifest: Path
    argv: tuple[str, ...]


def _exact_execution(record: object) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != EXACT_EXECUTION_FIELDS:
        raise GateVerificationError("exact execution binding fields are invalid")
    argv = record.get("argv")
    pid = record.get("pid")
    commit = record.get("code_commit")
    source = record.get("source_manifest_sha256")
    returncode = record.get("returncode")
    profile_binding = record.get("profile_config_binding")
    common_inputs = record.get("common_input_fingerprints")
    output = record.get("output_root")
    receipt_sha256 = record.get("receipt_sha256")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise GateVerificationError("exact execution argv is invalid")
    if type(pid) is not int or pid <= 0:
        raise GateVerificationError("exact execution PID is invalid")
    if returncode != 0:
        raise GateVerificationError("exact execution return code is invalid")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(source, str)
        or len(source) != 64
        or any(character not in "0123456789abcdef" for character in source)
    ):
        raise GateVerificationError("exact execution binding is invalid")
    for value, expected, label in ((common_inputs, COMMON_INPUT_FINGERPRINT_FIELDS, "common input"),):
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or any(not _is_sha256(item) for item in value.values())
        ):
            raise GateVerificationError(f"exact execution {label} binding is invalid")
    if (
        not isinstance(profile_binding, dict)
        or set(profile_binding) != PROFILE_CONFIG_BINDING_FIELDS
        or profile_binding.get("mode") != DEVELOPMENT_MODE
        or any(
            not _is_sha256(profile_binding.get(name))
            for name in ("config_sha256", "algorithm_hash", "profile_sha256")
        )
    ):
        raise GateVerificationError("exact execution profile config binding is invalid")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise GateVerificationError("exact execution output binding is invalid")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise GateVerificationError("exact execution receipt binding is invalid")
    if not _is_sha256(record.get("run_manifest_sha256")):
        raise GateVerificationError("exact execution run manifest binding is invalid")
    observation = record.get("observation_receipt")
    if (
        not isinstance(observation, dict)
        or set(observation) != {"path", "sha256", "byte_count"}
        or not isinstance(observation.get("path"), str)
        or not Path(observation["path"]).is_absolute()
        or not _is_sha256(observation.get("sha256"))
        or type(observation.get("byte_count")) is not int
        or observation["byte_count"] <= 0
    ):
        raise GateVerificationError("exact execution observation binding is invalid")
    return dict(record)


def _validate_exact_argv(record: Mapping[str, Any], root: Path) -> None:
    profile = str(record["profile"])
    argv = record["argv"]
    expected_runner = (
        REPO_ROOT / "scripts/evaluation/run_oviv2_t1_reference.py"
        if profile == "reference"
        else REPO_ROOT / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"
    ).resolve()
    expected_flags = ("--config", "--output")
    expected_length = 10 if profile == "reference" else 6
    if (
        len(argv) != expected_length
        or not Path(argv[0]).is_absolute()
        or Path(argv[1]) != expected_runner
        or tuple(argv[2:6:2]) != expected_flags
        or argv[5] != str(root)
        or any(not Path(argv[index]).is_absolute() for index in (3, 5))
    ):
        raise GateVerificationError("exact execution argv is not canonical")
    if profile == "reference" and (
        tuple(argv[6:10:2]) != ("--receipt", "--source-manifest")
        or argv[7] != str(root / "t1_exact_receipt.json")
        or not Path(argv[9]).is_absolute()
    ):
        raise GateVerificationError("reference execution argv is not canonical")


def _bind_exact_receipt(
    record: dict[str, Any],
    *,
    compare: Callable[[Path, Path], dict[str, Any]],
    expected_position: int | None = None,
    audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_root = record["output_root"]
    try:
        root = Path(raw_root).resolve(strict=True)
    except OSError as exc:
        raise GateVerificationError("exact execution root is missing") from exc
    if str(root) != raw_root or not root.is_dir():
        raise GateVerificationError("exact execution root is not canonical")
    _validate_exact_argv(record, root)
    observation_record = record["observation_receipt"]
    observation_path = Path(observation_record["path"])
    observation_data = _regular_file_bytes(
        observation_path.parent, observation_path.name, DEFAULT_MAX_INPUT_BYTES
    )
    if (
        _sha256(observation_data) != observation_record["sha256"]
        or len(observation_data) != observation_record["byte_count"]
    ):
        raise GateVerificationError("exact execution observation receipt hash mismatch")
    try:
        observation = json.loads(
            observation_data, object_pairs_hook=_strict_json_object
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateVerificationError("exact execution observation receipt is invalid") from exc
    expected_observation_fields = {
        "schema_version", "format", "position", "profile", "argv", "pid",
        "returncode", "config", "source_manifest",
        "output_root", "root_device", "root_inode", "run_manifest",
        "production_receipt", "completed_execution", "trust_model",
        "execution_context",
    }
    root_status = os.stat(root, follow_symlinks=False)
    if (
        not isinstance(observation, dict)
        or set(observation) != expected_observation_fields
        or observation.get("schema_version") != 1
        or observation.get("format") != "oviv2_exact_process_observation_v1"
        or (
            expected_position is not None
            and observation.get("position") != expected_position
        )
        or observation.get("profile") != record["profile"]
        or observation.get("argv") != record["argv"]
        or observation.get("pid") != record["pid"]
        or observation.get("returncode") != record["returncode"]
        or observation.get("output_root") != str(root)
        or observation.get("root_device") != root_status.st_dev
        or observation.get("root_inode") != root_status.st_ino
        or observation.get("trust_model") != LOCAL_PROCESS_TRUST_MODEL
        or observation.get("execution_context") != DEVELOPMENT_EXECUTION_CONTEXT
    ):
        raise GateVerificationError("exact execution observation binding mismatch")
    for key in (
        "config", "source_manifest", "run_manifest",
        "production_receipt",
    ):
        bound = observation.get(key)
        if not isinstance(bound, dict) or _absolute_file_record(Path(str(bound.get("path", "")))) != bound:
            raise GateVerificationError(f"exact execution observed {key} changed")
    derived = _reopen_completed_execution(
        record["profile"],
        root,
        record["argv"],
        record["pid"],
        record["returncode"],
        Path(observation["source_manifest"]["path"]),
    )
    expected_record = {**derived, "observation_receipt": observation_record}
    if observation.get("completed_execution") != derived or record != expected_record:
        raise GateVerificationError("exact execution reopened binding mismatch")
    if audit is None:
        try:
            audit = compare(root, root)
        except (ArtifactMismatch, KeyError, TypeError) as exc:
            raise GateVerificationError(f"exact execution artifact mismatch: {exc}") from exc
    if (
        record["profile"] == "reference"
    ):
        receipt_data = _regular_file_bytes(root, "t1_exact_receipt.json", DEFAULT_MAX_INPUT_BYTES)
        receipt = json.loads(receipt_data, object_pairs_hook=_strict_json_object)
        if (
            receipt.get("artifact_inventory") != audit.get("inventory")
            or receipt.get("checkpoint_frames") != audit.get("checkpoint_frames")
            or receipt.get("cumulative_root_sha256") != audit.get("root_sha256")
        ):
            raise GateVerificationError("exact execution receipt artifact mismatch")
    return record, audit


def verify_exact_profile_runs(
    executions: list[dict[str, Any]],
    *,
    compare: Callable[[Path, Path], dict[str, Any]] = compare_cumulative_artifacts,
    audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify an independently executed, interleaved T1/A0-A4 transaction."""
    if audits is not None and len(audits) != len(executions):
        raise GateVerificationError("exact cumulative audit inventory is invalid")
    records_and_audits = [
        _bind_exact_receipt(
            _exact_execution(record),
            compare=compare,
            expected_position=position,
            audit=None if audits is None else audits[position],
        )
        for position, record in enumerate(executions)
    ]
    records = [record for record, _ in records_and_audits]
    sequence = tuple(record["profile"] for record in records)
    if sequence != EXACT_PROFILE_SEQUENCE:
        raise GateVerificationError("exact execution profile sequence is invalid")
    if len({record["pid"] for record in records}) != len(records):
        raise GateVerificationError("exact executions must use independent PIDs")
    roots = [Path(record["output_root"]) for record in records]
    if len(set(roots)) != len(roots) or any(
        left in right.parents or right in left.parents
        for position, left in enumerate(roots)
        for right in roots[position + 1 :]
    ):
        raise GateVerificationError("exact execution roots must be independent")
    binding = (
        records[0]["code_commit"],
        records[0]["source_manifest_sha256"],
        records[0]["common_input_fingerprints"],
    )
    if any(
        (
            record["code_commit"],
            record["source_manifest_sha256"],
            record["common_input_fingerprints"],
        )
        != binding
        for record in records[1:]
    ):
        raise GateVerificationError("exact execution binding disagreement")

    profiles: dict[str, Any] = {}
    try:
        audit_records = [audit for _, audit in records_and_audits]
        baseline = {
            "cumulative_root_sha256": audit_records[0]["root_sha256"],
            "checkpoint_frames": audit_records[0]["checkpoint_frames"],
            "inventory": audit_records[0]["inventory"],
        }
        if any(
            {
                "cumulative_root_sha256": audit["root_sha256"],
                "checkpoint_frames": audit["checkpoint_frames"],
                "inventory": audit["inventory"],
            }
            != baseline
            for audit in audit_records[1:]
        ):
            raise GateVerificationError("cumulative audit root disagreement")
        for record in records[1:]:
            profiles[record["profile"]] = dict(baseline)
    except (ArtifactMismatch, KeyError, TypeError) as exc:
        raise GateVerificationError(f"exact cumulative artifact mismatch: {exc}") from exc
    return {
        "format": "oviv2_t1_exact_transaction_v1",
        "sequence": list(sequence),
        "executions": records,
        "profiles": profiles,
    }


def _reopen_completed_execution(
    profile: str,
    root: Path,
    argv: list[str],
    pid: int,
    returncode: int,
    source_manifest: Path,
) -> dict[str, object]:
    config_record = _absolute_file_record(Path(argv[3]))
    source_record = _absolute_file_record(source_manifest)
    try:
        config = json.loads(
            _regular_file_bytes(Path(argv[3]).parent, Path(argv[3]).name, DEFAULT_MAX_INPUT_BYTES),
            object_pairs_hook=_strict_json_object,
        )
        manifest_data = _regular_file_bytes(root, "run_manifest.json", DEFAULT_MAX_INPUT_BYTES)
        manifest = json.loads(manifest_data, object_pairs_hook=_strict_json_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateVerificationError("completed execution JSON binding is invalid") from exc
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise GateVerificationError("completed execution JSON root is invalid")
    if config.get("scene") != "apartment" or manifest.get("scene") != "apartment":
        raise GateVerificationError("development execution requires Apartment config")
    if "frozen_run_identity" in manifest:
        raise GateVerificationError("development execution unexpectedly contains frozen identity")

    def config_input(name: str) -> dict[str, object]:
        raw = config.get(name)
        if not isinstance(raw, str) or not raw:
            raise GateVerificationError(f"completed execution {name} binding is invalid")
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        record = _absolute_file_record(path.absolute())
        return record

    input_record = config_input("input_manifest")
    schedule_record = config_input("schedule_manifest")
    target_record = config_input("occlusion_target_manifest")
    if config.get("occlusion_target_manifest_sha256") != target_record["sha256"]:
        raise GateVerificationError("completed execution target manifest hash is invalid")
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, dict):
        raise GateVerificationError("production source bindings are invalid")
    algorithm_hash = manifest.get("algorithm_hash")
    code_commit = manifest.get("code_commit")
    manifest_config = manifest.get("config")
    manifest_schedule = manifest.get("schedule")
    manifest_target = manifest.get("target_manifest")

    def byte_binding(record: Mapping[str, object]) -> dict[str, object]:
        return {
            "sha256": record["sha256"],
            "byte_count": record["byte_count"],
        }

    if manifest_target is None and profile == "reference":
        index_record = manifest.get("occlusion_checkpoint_index")
        index_path_value = (
            index_record.get("path") if isinstance(index_record, dict) else None
        )
        if not isinstance(index_path_value, str) or not index_path_value:
            raise GateVerificationError(
                "completed execution target manifest binding is invalid"
            )
        index_path = root / index_path_value
        if index_path.resolve(strict=False) != index_path or root not in index_path.parents:
            raise GateVerificationError(
                "completed execution target manifest binding is invalid"
            )
        index_file = _absolute_file_record(index_path)
        if index_record != {
            "path": index_path_value,
            "sha256": index_file["sha256"],
            "byte_count": index_file["byte_count"],
        }:
            raise GateVerificationError(
                "completed execution target manifest index changed"
            )
        try:
            index_value = json.loads(
                _regular_file_bytes(
                    index_path.parent, index_path.name, DEFAULT_MAX_INPUT_BYTES
                ),
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GateVerificationError(
                "completed execution target manifest index is invalid"
            ) from exc
        manifest_target = (
            index_value.get("target_manifest")
            if isinstance(index_value, dict)
            else None
        )
    if (
        not _is_sha256(algorithm_hash)
        or config.get("algorithm_hash") != algorithm_hash
        or manifest_config != byte_binding(config_record)
        or manifest_schedule != byte_binding(schedule_record)
        or manifest_target != byte_binding(target_record)
    ):
        raise GateVerificationError("completed execution identity is invalid")
    if source_bindings.get("input_manifest") != byte_binding(input_record):
        raise GateVerificationError(
            "production input source binding does not match verified config"
        )
    receipt_name = (
        "t1_exact_receipt.json" if profile == "reference" else "execution_receipt.json"
    )
    receipt_data = _regular_file_bytes(root, receipt_name, DEFAULT_MAX_INPUT_BYTES)
    try:
        receipt = json.loads(receipt_data, object_pairs_hook=_strict_json_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateVerificationError("completed execution receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise GateVerificationError("completed execution receipt root is invalid")
    if profile == "reference":
        execution = receipt.get("execution")
        code_commit = execution.get("code_commit") if isinstance(execution, dict) else None
        input_fingerprints = (
            execution.get("input_fingerprints")
            if isinstance(execution, dict)
            else None
        )
        if (
            receipt.get("format") != "oviv2_t1_exact_execution_receipt_v1"
            or not isinstance(execution, dict)
            or execution.get("mode") != DEVELOPMENT_MODE
            or execution.get("pid") != pid
            or execution.get("argv") != argv
            or execution.get("output_root") != str(root)
            or not isinstance(code_commit, str)
            or receipt.get("source_manifest") != source_record
            or not isinstance(input_fingerprints, dict)
            or input_fingerprints.get("config") != config_record["sha256"]
        ):
            raise GateVerificationError("reference process receipt binding mismatch")
    else:
        provenance = receipt.get("provenance")
        command = provenance.get("command") if isinstance(provenance, dict) else None
        normalized_command = (
            [argv[0], *command] if isinstance(command, list) else None
        )
        if (
            receipt.get("schema_version") != 1
            or not isinstance(provenance, dict)
            or provenance.get("repository_commit") != code_commit
            or normalized_command != argv
        ):
            raise GateVerificationError("dual process receipt command/commit binding mismatch")
    canonical = lambda value: _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return {
        "profile": profile,
        "argv": argv,
        "pid": pid,
        "returncode": returncode,
        "code_commit": code_commit,
        "source_manifest_sha256": source_record["sha256"],
        "profile_config_binding": {
            "mode": DEVELOPMENT_MODE,
            "config_sha256": config_record["sha256"],
            "algorithm_hash": algorithm_hash,
            "profile_sha256": _sha256(profile.encode()),
        },
        "common_input_fingerprints": {
            "source_manifest_sha256": source_record["sha256"],
            "input_manifest_sha256": input_record["sha256"],
            "schedule_sha256": schedule_record["sha256"],
            "ground_truth_sha256": target_record["sha256"],
            "source_bindings_sha256": canonical(source_bindings),
        },
        "output_root": str(root),
        "receipt_sha256": _sha256(receipt_data),
        "run_manifest_sha256": _sha256(manifest_data),
    }


def _absolute_file_record(path: Path) -> dict[str, object]:
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise GateVerificationError(f"observed file is missing: {path}") from exc
    if str(canonical) != str(path):
        raise GateVerificationError(f"observed file path is not canonical: {path}")
    data = _regular_file_bytes(canonical.parent, canonical.name, DEFAULT_MAX_INPUT_BYTES)
    return {
        "path": str(canonical),
        "sha256": _sha256(data),
        "byte_count": len(data),
    }


TreeEntry = tuple[str, int, int, int, int, int, int]
OwnedPath = tuple[Path, int, int, int, int, tuple[TreeEntry, ...] | None]


def _tree_inventory(
    descriptor: int, prefix: str = ""
) -> tuple[TreeEntry, ...]:
    entries: list[TreeEntry] = []
    for name in sorted(os.listdir(descriptor)):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        relative = f"{prefix}/{name}" if prefix else name
        entries.append(
            (
                relative,
                metadata.st_dev,
                metadata.st_ino,
                stat.S_IFMT(metadata.st_mode),
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise GateVerificationError(
                        "owned tree changed while capturing inventory"
                    )
                entries.extend(_tree_inventory(child, relative))
            finally:
                os.close(child)
    return tuple(entries)


def _open_parent_directory(path: Path) -> tuple[int, str]:
    absolute = Path(os.path.abspath(path))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def _owned_path(path: Path, *, capture_tree: bool = False) -> OwnedPath:
    descriptor, name = _open_parent_directory(path)
    owned_descriptor: int | None = None
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        owned_descriptor = os.open(
            name,
            (
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                if stat.S_ISDIR(metadata.st_mode)
                else getattr(os, "O_PATH", os.O_RDONLY)
            )
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
        opened = os.fstat(owned_descriptor)
    finally:
        os.close(descriptor)
    if owned_descriptor is None or (opened.st_dev, opened.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        if owned_descriptor is not None:
            os.close(owned_descriptor)
        raise GateVerificationError(f"owned path changed while observing: {path}")
    inventory = (
        _tree_inventory(owned_descriptor)
        if capture_tree and stat.S_ISDIR(metadata.st_mode)
        else None
    )
    return (
        Path(os.path.abspath(path)),
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        owned_descriptor,
        inventory,
    )


def _create_owned_directory(path: Path) -> OwnedPath:
    absolute = Path(os.path.abspath(path))
    descriptor, name = _open_parent_directory(absolute)
    try:
        os.mkdir(name, dir_fd=descriptor)
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        owned_descriptor = os.open(
            name,
            getattr(os, "O_PATH", os.O_RDONLY)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
    finally:
        os.close(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise GateVerificationError(f"created path is not a directory: {absolute}")
    return (
        absolute,
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        owned_descriptor,
        None,
    )


def _owned_path_matches(witness: OwnedPath) -> bool:
    path, device, inode, kind, owned_descriptor, inventory = witness
    try:
        opened = os.fstat(owned_descriptor)
        descriptor, name = _open_parent_directory(path)
        try:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        finally:
            os.close(descriptor)
    except OSError:
        return False
    identity_matches = (
        opened.st_dev,
        opened.st_ino,
        current.st_dev,
        current.st_ino,
        stat.S_IFMT(current.st_mode),
    ) == (device, inode, device, inode, kind)
    if not identity_matches:
        return False
    try:
        return inventory is None or _tree_inventory(owned_descriptor) == inventory
    except (GateVerificationError, OSError):
        return False


def _open_inventory_parent(descriptor: int, relative: str) -> tuple[int, str]:
    parts = relative.split("/")
    parent = os.dup(descriptor)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
            os.close(parent)
            parent = child
        return parent, parts[-1]
    except BaseException:
        os.close(parent)
        raise


def _remove_owned_inventory(
    descriptor: int, inventory: tuple[TreeEntry, ...], label: Path
) -> None:
    for entry in sorted(
        inventory,
        key=lambda item: (item[0].count("/"), stat.S_ISDIR(item[3])),
        reverse=True,
    ):
        relative, device, inode, kind, size, modified, changed = entry
        parent, name = _open_inventory_parent(descriptor, relative)
        try:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            identity = (
                current.st_dev,
                current.st_ino,
                stat.S_IFMT(current.st_mode),
            )
            metadata = (
                current.st_size, current.st_mtime_ns, current.st_ctime_ns
            )
            if identity != (device, inode, kind) or (
                not stat.S_ISDIR(kind) and metadata != (size, modified, changed)
            ):
                raise GateVerificationError(
                    f"cleanup unsafe: owned tree entry replaced: {label / relative}"
                )
            if stat.S_ISDIR(kind):
                os.rmdir(name, dir_fd=parent)
            else:
                os.unlink(name, dir_fd=parent)
        finally:
            os.close(parent)


def _remove_owned_path(witness: OwnedPath, *, recursive: bool = True) -> None:
    path, device, inode, kind, owned_descriptor, inventory = witness
    opened_owner = os.fstat(owned_descriptor)
    if (opened_owner.st_dev, opened_owner.st_ino) != (device, inode):
        raise GateVerificationError(f"cleanup unsafe: ownership changed: {path}")
    descriptor, name = _open_parent_directory(path)
    child: int | None = None
    try:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            current.st_dev,
            current.st_ino,
            stat.S_IFMT(current.st_mode),
        ) != (device, inode, kind):
            raise GateVerificationError(f"cleanup unsafe: owned path replaced: {path}")
        if stat.S_ISDIR(current.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (device, inode):
                raise GateVerificationError(f"cleanup unsafe: owned path replaced: {path}")
            if recursive:
                if inventory is None:
                    raise GateVerificationError(
                        f"cleanup unsafe: owned tree inventory is missing: {path}"
                    )
                _remove_owned_inventory(child, inventory, path)
            elif os.listdir(child):
                raise GateVerificationError(
                    f"cleanup unsafe: owned directory has unknown entries: {path}"
                )
            os.close(child)
            child = None
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (device, inode):
                raise GateVerificationError(f"cleanup unsafe: owned path replaced: {path}")
            os.rmdir(name, dir_fd=descriptor)
        else:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                current.st_dev,
                current.st_ino,
                stat.S_IFMT(current.st_mode),
            ) != (device, inode, kind):
                raise GateVerificationError(f"cleanup unsafe: owned path replaced: {path}")
            os.unlink(name, dir_fd=descriptor)
    except FileNotFoundError as exc:
        raise GateVerificationError(f"cleanup unsafe: owned path disappeared: {path}") from exc
    finally:
        if child is not None:
            os.close(child)
        os.close(descriptor)


def _cleanup_exact_transaction(
    roots: list[OwnedPath],
    observations: list[OwnedPath],
    receipts: OwnedPath | None,
    transaction: OwnedPath | None,
    unproven_roots: list[Path] | None = None,
) -> list[str]:
    problems: list[str] = []
    unsafe_paths: list[Path] = []
    witnesses = [
        *roots,
        *observations,
        *([receipts] if receipts else []),
        *([transaction] if transaction else []),
    ]
    try:
        if unproven_roots:
            problems.extend(
                f"cleanup unsafe: preserving unproven output root: {path}"
                for path in unproven_roots
            )
            if transaction is not None:
                problems.append(
                    f"cleanup unsafe: preserving transaction with unproven output: {transaction[0]}"
                )
            return problems
        mismatched = [
            witness for witness in witnesses if not _owned_path_matches(witness)
        ]
        if mismatched:
            unsafe_paths.extend(witness[0] for witness in mismatched)
            problems.extend(
                f"cleanup unsafe: owned path replaced: {witness[0]}"
                for witness in mismatched
            )
            if transaction is not None:
                problems.append(
                    f"cleanup unsafe: preserving transaction after ownership mismatch: {transaction[0]}"
                )
            return problems
        for witness in reversed(observations):
            try:
                _remove_owned_path(witness)
            except (GateVerificationError, OSError) as exc:
                problems.append(str(exc))
                unsafe_paths.append(witness[0])
                break
        if not unsafe_paths:
            for witness in reversed(roots):
                try:
                    _remove_owned_path(witness)
                except (GateVerificationError, OSError) as exc:
                    problems.append(str(exc))
                    unsafe_paths.append(witness[0])
                    break
        if receipts is not None and not unsafe_paths:
            receipts_path = receipts[0]
            try:
                _remove_owned_path(receipts, recursive=False)
            except (GateVerificationError, OSError) as exc:
                problems.append(str(exc))
                unsafe_paths.append(receipts_path)
        if transaction is not None:
            transaction_path = transaction[0]
            if unsafe_paths:
                problems.append(
                    f"cleanup unsafe: preserving transaction after ownership mismatch: {transaction_path}"
                )
            else:
                try:
                    _remove_owned_path(transaction, recursive=False)
                except (GateVerificationError, OSError) as exc:
                    problems.append(str(exc))
    finally:
        for witness in witnesses:
            try:
                os.close(witness[4])
            except OSError:
                pass
    return problems


def _close_owned_paths(witnesses: list[OwnedPath]) -> None:
    for witness in witnesses:
        os.close(witness[4])


def _preflight_exact_profile_specs(
    specs: list[dict[str, object]],
    *,
    repo: Path,
    python_executable: str,
    transaction_dir: Path,
) -> tuple[_ExactProfileSpec, ...]:
    if len(specs) != len(EXACT_PROFILE_SEQUENCE):
        raise GateVerificationError("exact execution spec inventory is invalid")
    if not Path(python_executable).is_absolute():
        raise GateVerificationError("exact execution Python path must be absolute")
    if (
        not transaction_dir.is_absolute()
        or str(transaction_dir.resolve(strict=False)) != str(transaction_dir)
    ):
        raise GateVerificationError("exact transaction directory is not canonical")
    reserved_transaction_paths = (transaction_dir, transaction_dir / "receipts")
    expected_fields = {"profile", "config", "output_root", "source_manifest"}
    prepared: list[_ExactProfileSpec] = []
    roots: list[Path] = []
    for profile, spec in zip(EXACT_PROFILE_SEQUENCE, specs, strict=True):
        if not isinstance(spec, dict) or set(spec) != expected_fields:
            raise GateVerificationError("exact execution spec fields are invalid")
        if spec.get("profile") != profile:
            raise GateVerificationError("exact execution spec sequence is invalid")
        config = Path(str(spec["config"]))
        root = Path(str(spec["output_root"]))
        source = Path(str(spec["source_manifest"]))
        if any(not path.is_absolute() for path in (config, root, source)):
            raise GateVerificationError("exact execution spec paths must be absolute")
        _absolute_file_record(config)
        _absolute_file_record(source)
        try:
            config_value = json.loads(
                _regular_file_bytes(
                    config.parent, config.name, DEFAULT_MAX_INPUT_BYTES
                ),
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GateVerificationError("exact execution config is invalid") from exc
        if not isinstance(config_value, dict) or config_value.get("scene") != "apartment":
            raise GateVerificationError(
                "exact development transaction requires Apartment config"
            )
        if str(root.resolve(strict=False)) != str(root):
            raise GateVerificationError("exact execution output root is not canonical")
        if root.exists() or root.is_symlink():
            raise GateVerificationError("exact execution output root would clobber data")
        if any(
            root == reserved
            or root in reserved.parents
            or reserved in root.parents
            for reserved in reserved_transaction_paths
        ):
            raise GateVerificationError(
                "exact execution output root overlaps transaction or receipts"
            )
        if any(
            root == previous
            or root in previous.parents
            or previous in root.parents
            for previous in roots
        ):
            raise GateVerificationError("exact execution output roots are not independent")
        runner = (
            repo / "scripts/evaluation/run_oviv2_t1_reference.py"
            if profile == "reference"
            else repo / "scripts/evaluation/run_oviv2_tesse_cd_v2.py"
        ).resolve()
        argv = (
            python_executable,
            str(runner),
            "--config",
            str(config),
            "--output",
            str(root),
            *(
                (
                    "--receipt",
                    str(root / "t1_exact_receipt.json"),
                    "--source-manifest",
                    str(source),
                )
                if profile == "reference"
                else ()
            ),
        )
        prepared.append(
            _ExactProfileSpec(profile, config, root, source, argv)
        )
        roots.append(root)
    return tuple(prepared)


def _reap_process_after_interruption(
    process: subprocess.Popen[bytes],
) -> str | None:
    try:
        running = process.poll() is None
    except BaseException as exc:
        raise GateVerificationError(f"process poll failed: {exc}") from exc
    if not running:
        try:
            process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except BaseException as exc:
            raise GateVerificationError(f"process wait/reap failed: {exc}") from exc
        return None

    try:
        process.terminate()
    except BaseException as terminate_error:
        try:
            process.kill()
            process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except BaseException as kill_error:
            raise GateVerificationError(
                f"process terminate failed: {terminate_error}; "
                f"process kill/reap failed: {kill_error}"
            ) from kill_error
        return f"process terminate failed: {terminate_error}"

    try:
        process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except BaseException as exc:
            raise GateVerificationError(f"process kill/reap failed: {exc}") from exc
    except BaseException as exc:
        raise GateVerificationError(f"process wait/reap failed: {exc}") from exc
    return None


def execute_exact_profile_transaction(
    specs: list[dict[str, object]],
    *,
    repo: Path,
    python_executable: str,
    transaction_dir: Path,
    popen_factory: PopenFactory = subprocess.Popen,
    compare: Callable[[Path, Path], dict[str, Any]] = compare_cumulative_artifacts,
) -> dict[str, Any]:
    """Launch and immediately verify the interleaved exact-profile transaction."""
    transaction_dir = Path(os.path.abspath(transaction_dir))
    if transaction_dir.exists() or transaction_dir.is_symlink():
        raise GateVerificationError("exact transaction directory would clobber data")
    prepared_specs = _preflight_exact_profile_specs(
        specs,
        repo=repo,
        python_executable=python_executable,
        transaction_dir=transaction_dir,
    )
    receipts_dir = transaction_dir / "receipts"
    environment = os.environ.copy()
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    roots: list[Path] = []
    root_witnesses: list[OwnedPath] = []
    observation_witnesses: list[OwnedPath] = []
    transaction_witness: OwnedPath | None = None
    receipts_witness: OwnedPath | None = None
    unproven_root: Path | None = None
    process_reap_failed = False
    try:
        try:
            transaction_witness = _create_owned_directory(transaction_dir)
            receipts_witness = _create_owned_directory(receipts_dir)
        except FileExistsError as exc:
            raise GateVerificationError(
                "exact transaction directory would clobber data"
            ) from exc
        for position, prepared in enumerate(prepared_specs):
            profile = prepared.profile
            config = prepared.config
            root = prepared.output_root
            source = prepared.source_manifest
            argv = list(prepared.argv)
            unproven_root = root
            process: subprocess.Popen[bytes] | None = None
            process = popen_factory(
                argv,
                cwd=repo,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate()
            except BaseException as communication_error:
                try:
                    recovery_problem = _reap_process_after_interruption(process)
                except BaseException as recovery_error:
                    process_reap_failed = True
                    raise GateVerificationError(
                        f"exact execution process reap failed after "
                        f"{type(communication_error).__name__}: {recovery_error}"
                    ) from communication_error
                if recovery_problem is not None:
                    raise GateVerificationError(
                        f"exact execution process recovery failed after "
                        f"{type(communication_error).__name__}: {recovery_problem}"
                    ) from communication_error
                raise
            returncode = process.returncode
            if returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise GateVerificationError(
                    f"exact execution failed ({profile}, {returncode}): {detail}"
                )
            if not root.is_dir() or root.is_symlink():
                raise GateVerificationError("exact execution did not publish its output root")
            record = _reopen_completed_execution(
                profile,
                root.resolve(strict=True),
                argv,
                process.pid,
                returncode,
                source,
            )
            root_status = os.stat(root, follow_symlinks=False)
            production_receipt = root / (
                "t1_exact_receipt.json"
                if profile == "reference"
                else "execution_receipt.json"
            )
            observation = {
                "schema_version": 1,
                "format": "oviv2_exact_process_observation_v1",
                "position": position,
                "profile": profile,
                "argv": argv,
                "pid": process.pid,
                "returncode": returncode,
                "config": _absolute_file_record(config),
                "source_manifest": _absolute_file_record(source),
                "output_root": str(root.resolve(strict=True)),
                "root_device": root_status.st_dev,
                "root_inode": root_status.st_ino,
                "trust_model": LOCAL_PROCESS_TRUST_MODEL,
                "execution_context": DEVELOPMENT_EXECUTION_CONTEXT,
                "run_manifest": _absolute_file_record(root / "run_manifest.json"),
                "production_receipt": _absolute_file_record(production_receipt),
                "completed_execution": record,
            }
            observation_path = receipts_dir / f"{position:03d}-{profile}.json"
            _atomic_json_no_replace(observation_path, observation)
            try:
                reopened_observation = json.loads(
                    _regular_file_bytes(
                        observation_path.parent,
                        observation_path.name,
                        DEFAULT_MAX_INPUT_BYTES,
                    ),
                    object_pairs_hook=_strict_json_object,
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise GateVerificationError(
                    "exact execution observation receipt is invalid"
                ) from exc
            if reopened_observation != observation:
                raise GateVerificationError(
                    "exact execution observation receipt changed during reopen"
                )
            observation_witnesses.append(_owned_path(observation_path))
            root_witnesses.append(_owned_path(root, capture_tree=True))
            unproven_root = None
            record = {
                **record,
                "observation_receipt": _absolute_file_record(observation_path),
            }
            records.append(dict(record))
            roots.append(root)
            audits.append(compare(root, root))
            del stdout
        result = verify_exact_profile_runs(records, compare=compare, audits=audits)
        _close_owned_paths(
            [
                *root_witnesses,
                *observation_witnesses,
                receipts_witness,
                transaction_witness,
            ]
        )
        return result
    except BaseException as error:
        if process_reap_failed:
            raise
        problems = _cleanup_exact_transaction(
            root_witnesses,
            observation_witnesses,
            receipts_witness,
            transaction_witness,
            [unproven_root]
            if unproven_root is not None
            and (unproven_root.exists() or unproven_root.is_symlink())
            else None,
        )
        if problems:
            raise GateVerificationError(
                f"{error}; cleanup unsafe: {'; '.join(problems)}"
            ) from error
        raise


def _default_git(argv: tuple[str, ...], cwd: Path) -> bytes:
    try:
        return subprocess.run(
            ("git", *argv),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise GateVerificationError(
            f"git {' '.join(argv)} failed: {detail or error.returncode}"
        ) from error


def _default_run(
    argv: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_PLUGINS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _module_candidates(module: str) -> tuple[str, str]:
    stem = module.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def _module_for_path(path: str) -> tuple[str, bool]:
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")].replace("/", "."), True
    if path.endswith(".py"):
        return path[:-3].replace("/", "."), False
    return "", False


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _top_level_exports(payload: bytes, *, path: str) -> tuple[set[str], set[str] | None]:
    try:
        tree = ast.parse(payload, filename=path)
    except (SyntaxError, ValueError) as error:
        raise GateVerificationError(f"cannot parse source imports: {path}") from error
    exports: set[str] = set()
    explicit_all: set[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                exports.update(_assigned_names(target))
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)) and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.value.elts
                ):
                    explicit_all = {item.value for item in node.value.elts}
        elif isinstance(node, ast.AnnAssign):
            exports.update(_assigned_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exports.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    exports.add(alias.asname or alias.name)
    return exports, explicit_all


def _dynamic_import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    functions = {"__import__"}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    modules.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    functions.add(alias.asname or alias.name)
    return functions, modules


def _local_import_closure(
    roots: tuple[str, ...], read: Callable[[str], bytes | None]
) -> dict[str, bytes]:
    pending = list(roots)
    closure: dict[str, bytes] = {}

    def resolve(module: str, *, required: bool) -> str | None:
        if not module or module.split(".", 1)[0] not in LOCAL_IMPORT_PREFIXES:
            return None
        for candidate in _module_candidates(module):
            payload = read(candidate)
            if payload is not None:
                if candidate not in closure and candidate not in pending:
                    pending.append(candidate)
                return candidate
        if required:
            raise GateVerificationError(f"unresolved local import: {module}")
        return None

    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        payload = read(relative)
        if payload is None:
            raise GateVerificationError(f"required source is missing: {relative}")
        closure[relative] = payload
        module, is_package = _module_for_path(relative)
        if not module:
            continue
        parts = module.split(".")
        for index in range(1, len(parts) if is_package else len(parts)):
            resolve(".".join(parts[:index]), required=False)
        try:
            tree = ast.parse(payload, filename=relative)
        except (SyntaxError, ValueError) as error:
            raise GateVerificationError(f"cannot parse source imports: {relative}") from error
        dynamic_functions, importlib_modules = _dynamic_import_aliases(tree)
        package = module if is_package else module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                recognized = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in dynamic_functions
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in importlib_modules
                )
                if not recognized:
                    continue
                if (
                    not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)
                ):
                    raise GateVerificationError(
                        f"non-literal dynamic import in {relative}"
                    )
                target = node.args[0].value
                if target.startswith("."):
                    raise GateVerificationError(
                        f"unresolved dynamic import in {relative}: {target}"
                    )
                if target.split(".", 1)[0] in LOCAL_IMPORT_PREFIXES:
                    resolve(target, required=True)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name, required=True)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    if node.level > len(package_parts):
                        raise GateVerificationError(
                            f"unresolved local import in {relative}"
                        )
                    prefix = package_parts[: len(package_parts) - node.level + 1]
                    base = ".".join((*prefix, *(node.module or "").split("."))).strip(".")
                else:
                    base = node.module or ""
                if not base or base.split(".", 1)[0] not in LOCAL_IMPORT_PREFIXES:
                    continue
                base_found = resolve(base, required=False)
                base_exports: set[str] = set()
                explicit_all: set[str] | None = None
                if base_found is not None:
                    base_payload = read(base_found)
                    if base_payload is None:
                        raise GateVerificationError(
                            f"source disappeared while resolving import: {base}"
                        )
                    base_exports, explicit_all = _top_level_exports(
                        base_payload, path=base_found
                    )
                for alias in node.names:
                    if alias.name == "*":
                        if (
                            base_found is None
                            or explicit_all is None
                            or not explicit_all <= base_exports
                        ):
                            raise GateVerificationError(
                                f"unresolved local import: {base}.*"
                            )
                        continue
                    child_found = resolve(
                        f"{base}.{alias.name}", required=False
                    )
                    if child_found is None and (
                        base_found is None or alias.name not in base_exports
                    ):
                        raise GateVerificationError(
                            f"unresolved local import: {base}.{alias.name}"
                        )
    return dict(sorted(closure.items()))


def _git_source_reader(
    *, repo: Path, base_commit: str, git: GitRunner
) -> Callable[[str], bytes | None]:
    cache: dict[str, bytes | None] = {}
    raw_paths = git(("ls-tree", "-rz", "--name-only", base_commit), repo)
    try:
        paths = {
            item.decode("utf-8")
            for item in raw_paths.split(b"\0")
            if item
        }
    except UnicodeDecodeError as error:
        raise GateVerificationError("git tree contains a non-UTF-8 path") from error

    def read(relative: str) -> bytes | None:
        if relative not in paths:
            return None
        if relative not in cache:
            cache[relative] = git(("show", f"{base_commit}:{relative}"), repo)
        return cache[relative]

    return read


def _worktree_source_reader(
    *, repo: Path, maximum: int
) -> Callable[[str], bytes | None]:
    def read(relative: str) -> bytes | None:
        path = repo / relative
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        return _regular_file_bytes(repo, relative, maximum)

    return read


def build_source_manifest(
    *,
    repo: Path,
    base_commit: str,
    roots: tuple[str, ...] = CUMULATIVE_ROOTS,
    git: GitRunner = _default_git,
) -> dict[str, Any]:
    if base_commit != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError(
            f"base commit must equal cumulative base {CUMULATIVE_BASE_COMMIT}"
        )
    closure = _local_import_closure(
        roots, _git_source_reader(repo=repo, base_commit=base_commit, git=git)
    )
    return {
        "schema_version": 1,
        "manifest_id": SOURCE_MANIFEST_ID,
        "base_commit": base_commit,
        "roots": list(roots),
        "files": {path: _sha256(payload) for path, payload in closure.items()},
    }


def verify_source_manifest(
    manifest: dict[str, Any],
    *,
    repo: Path,
    git: GitRunner = _default_git,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> list[dict[str, Any]]:
    if set(manifest) != {"schema_version", "manifest_id", "base_commit", "roots", "files"}:
        raise GateVerificationError("source manifest has invalid fields")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["manifest_id"] != SOURCE_MANIFEST_ID
    ):
        raise GateVerificationError("source manifest identity is invalid")
    if manifest["base_commit"] != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError("source manifest cumulative base is invalid")
    roots_value = manifest["roots"]
    files_value = manifest["files"]
    if not isinstance(roots_value, list) or not all(isinstance(item, str) for item in roots_value):
        raise GateVerificationError("source manifest roots are invalid")
    if not all(_is_canonical_repo_path(path) for path in roots_value):
        raise GateVerificationError(
            "source manifest paths must be canonical relative POSIX paths"
        )
    if tuple(roots_value) != CUMULATIVE_ROOTS:
        raise GateVerificationError("source manifest cumulative roots are invalid")
    if not isinstance(files_value, dict) or not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= set("0123456789abcdef")
        for path, digest in files_value.items()
    ):
        raise GateVerificationError("source manifest hashes are invalid")
    if not all(_is_canonical_repo_path(path) for path in files_value):
        raise GateVerificationError(
            "source manifest paths must be canonical relative POSIX paths"
        )
    for path in files_value:
        current_path = repo / path
        try:
            metadata = current_path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {path}"
            )
        if metadata.st_size > max_input_bytes:
            raise GateVerificationError(
                f"required file exceeds maximum {max_input_bytes} bytes: {path}"
            )
    roots = tuple(roots_value)
    trusted = build_source_manifest(
        repo=repo,
        base_commit=CUMULATIVE_BASE_COMMIT,
        roots=roots,
        git=git,
    )
    trusted_files = trusted["files"]
    if set(files_value) != set(trusted_files):
        raise GateVerificationError("trusted source path set does not match manifest")
    if files_value != trusted_files:
        raise GateVerificationError("trusted source hash does not match manifest")
    current = _local_import_closure(
        roots, _worktree_source_reader(repo=repo, maximum=max_input_bytes)
    )
    if set(current) != set(files_value):
        raise GateVerificationError("current source path set does not match manifest")
    records = []
    for path, payload in current.items():
        digest = _sha256(payload)
        if digest != files_value[path]:
            raise GateVerificationError(f"current source hash does not match: {path}")
        records.append({"path": path, "sha256": digest, "bytes": len(payload)})
    return records


def _load_source_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateVerificationError(f"cannot read source manifest: {path}") from error
    if not isinstance(payload, dict):
        raise GateVerificationError("source manifest root must be an object")
    return payload


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_canonical_repo_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path or "\0" in path:
        return False
    parts = path.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _regular_file_bytes(repo: Path, relative: str, maximum: int) -> bytes:
    if not _is_canonical_repo_path(relative):
        raise GateVerificationError(
            f"required file path must be canonical relative POSIX: {relative!r}"
        )
    parts = relative.split("/")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_descriptor = os.open(repo, directory_flags)
    except FileNotFoundError as error:
        raise GateVerificationError(f"repository root is missing: {repo}") from error
    except OSError as error:
        raise GateVerificationError(
            f"repository root must be a regular non-symlink directory: {repo}"
        ) from error
    directory_descriptor = root_descriptor
    descriptor: int | None = None
    try:
        for part in parts[:-1]:
            try:
                next_descriptor = os.open(
                    part, directory_flags, dir_fd=directory_descriptor
                )
            except FileNotFoundError as error:
                raise GateVerificationError(
                    f"required file is missing: {relative}"
                ) from error
            except OSError as error:
                raise GateVerificationError(
                    f"required file ancestors must be regular non-symlink directories: {relative}"
                ) from error
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            descriptor = os.open(
                parts[-1], file_flags, dir_fd=directory_descriptor
            )
        except FileNotFoundError as error:
            raise GateVerificationError(
                f"required file is missing: {relative}"
            ) from error
        except OSError as error:
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {relative}"
            ) from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateVerificationError(
                f"required file must be a regular non-symlink: {relative}"
            )
        if before.st_size > maximum:
            raise GateVerificationError(
                f"required file exceeds maximum {maximum} bytes: {relative}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise GateVerificationError(
                    f"required file exceeds maximum {maximum} bytes: {relative}"
                )
        after = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)
        os.close(root_descriptor)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise GateVerificationError(f"required file changed while hashing: {relative}")
    return b"".join(chunks)


def _regular_file_record(repo: Path, relative: str, maximum: int) -> dict[str, Any]:
    payload = _regular_file_bytes(repo, relative, maximum)
    return {"path": relative, "sha256": _sha256(payload), "bytes": len(payload)}


def _snapshot_files(
    repo: Path, paths: tuple[str, ...], maximum: int
) -> list[dict[str, Any]]:
    return [_regular_file_record(repo, path, maximum) for path in paths]


def _git_state(repo: Path, git: GitRunner) -> tuple[str, str]:
    if git(("status", "--porcelain", "--untracked-files=all"), repo):
        raise GateVerificationError("repository must be clean")
    commit = git(("rev-parse", "HEAD"), repo).decode("ascii").strip()
    tree = git(("rev-parse", "HEAD^{tree}"), repo).decode("ascii").strip()
    if len(commit) != 40 or len(tree) != 40:
        raise GateVerificationError("git returned an invalid commit or tree identity")
    return commit, tree


def _command_record(
    *, repo: Path, argv: tuple[str, ...], run: CommandRunner
) -> dict[str, Any]:
    completed = run(argv, repo)
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    if isinstance(stdout, str) or isinstance(stderr, str):
        raise GateVerificationError("test runner must capture stdout and stderr as bytes")
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace").lower()
    if completed.returncode != 0:
        raise GateVerificationError(f"test command failed with return code {completed.returncode}")
    forbidden = (
        " skipped",
        " xfailed",
        " xpassed",
        " deselected",
        "skip=",
        "xfail=",
        "xpass=",
    )
    progress_has_nonpass = False
    for line in combined.splitlines():
        prefix = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if prefix and set(prefix) <= set(".efxs") and any(
            marker in prefix for marker in "sx"
        ):
            progress_has_nonpass = True
            break
    if any(token in combined for token in forbidden) or progress_has_nonpass:
        raise GateVerificationError(
            "test command reported skip, xfail/xpass, or deselected outcomes"
        )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_sha256": _sha256(stdout),
        "stdout_bytes": len(stdout),
        "stderr_sha256": _sha256(stderr),
        "stderr_bytes": len(stderr),
    }


def _atomic_json_no_replace(
    path: Path,
    payload: dict[str, Any],
    validate_before_publish: Callable[[], None] = lambda: None,
) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise GateVerificationError("output parent must be a regular directory")
    encoded = _canonical_json_bytes(payload)
    validate_before_publish()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
        temporary.unlink()
    except BaseException:
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError:
                pass
        if linked:
            for _ in range(2):
                try:
                    path.unlink(missing_ok=True)
                    break
                except OSError:
                    continue
        for _ in range(2):
            try:
                temporary.unlink(missing_ok=True)
                break
            except OSError:
                continue
        raise


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_json_replace(
    path: Path,
    payload: dict[str, Any],
    validate_before_replace: Callable[[], None],
) -> None:
    encoded = _canonical_json_bytes(payload)
    destination_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    directory_descriptor: int | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), destination_mode)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        validate_before_replace()
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        temporary.unlink(missing_ok=True)


def _checked_manifest_destination(repo: Path, destination: Path) -> tuple[Path, str]:
    raw = destination if destination.is_absolute() else repo / destination
    if raw.is_symlink():
        raise GateVerificationError("source manifest destination must not be a symlink")
    try:
        lexical_relative = raw.relative_to(repo)
    except ValueError as error:
        raise GateVerificationError("source manifest destination must be inside repo") from error
    current = repo
    for part in lexical_relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise GateVerificationError(
                "source manifest destination ancestor must not be a symlink"
            )
    resolved = raw.resolve(strict=False)
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as error:
        raise GateVerificationError("source manifest destination must be inside repo") from error
    return resolved, relative


def write_source_manifest(
    *,
    repo: Path,
    destination: Path,
    payload: dict[str, Any],
    git: GitRunner = _default_git,
) -> Path:
    repo = repo.resolve(strict=True)
    destination, relative = _checked_manifest_destination(repo, destination)
    if git(("status", "--porcelain", "--", relative), repo):
        raise GateVerificationError("source manifest destination is dirty")
    encoded = _canonical_json_bytes(payload)
    if destination.exists():
        current = _regular_file_bytes(
            destination.parent, destination.name, DEFAULT_MAX_INPUT_BYTES
        )
        if current == encoded:
            return destination

        def validate_before_replace() -> None:
            if git(("status", "--porcelain", "--", relative), repo):
                raise GateVerificationError("source manifest destination became dirty")
            if (
                _regular_file_bytes(
                    destination.parent, destination.name, DEFAULT_MAX_INPUT_BYTES
                )
                != current
            ):
                raise GateVerificationError(
                    "source manifest destination changed before replacement"
                )

        _atomic_json_replace(destination, payload, validate_before_replace)
        return destination
    _atomic_json_no_replace(destination, payload)
    return destination


def generate_evidence(
    *,
    repo: Path,
    output: Path,
    base_commit: str,
    python_executable: str,
    git: GitRunner = _default_git,
    run: CommandRunner = _default_run,
    now_utc: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    ),
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    exact_profile_specs: list[dict[str, object]] | None = None,
    transaction_dir: Path | None = None,
    execute_transaction: ExactTransactionRunner = execute_exact_profile_transaction,
    compare: Callable[[Path, Path], dict[str, Any]] = compare_cumulative_artifacts,
) -> Path:
    repo = repo.resolve(strict=True)
    if base_commit != CUMULATIVE_BASE_COMMIT:
        raise GateVerificationError(
            f"base commit must equal cumulative base {CUMULATIVE_BASE_COMMIT}"
        )
    if output.is_symlink() or output.exists():
        raise FileExistsError(output)

    code_commit, code_tree = _git_state(repo, git)
    source_manifest = _load_source_manifest(repo / DEFAULT_SOURCE_MANIFEST)
    source_manifest_bytes = _regular_file_bytes(
        repo, DEFAULT_SOURCE_MANIFEST.as_posix(), max_input_bytes
    )
    source_manifest_record = {
        "path": str((repo / DEFAULT_SOURCE_MANIFEST).resolve()),
        "sha256": _sha256(source_manifest_bytes),
        "byte_count": len(source_manifest_bytes),
    }
    protected = verify_source_manifest(
        source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
    )
    test_sources = _snapshot_files(repo, TEST_FILES, max_input_bytes)

    pytest_prefix = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-rA",
        "-o",
        "addopts=",
    )
    t1_argv = (*pytest_prefix, *T1_TEST_FILES)
    deterministic_argv = (
        *pytest_prefix,
        *DETERMINISM_TEST_FILES,
    )
    t1_record = _command_record(repo=repo, argv=t1_argv, run=run)
    determinism_record = _command_record(repo=repo, argv=deterministic_argv, run=run)
    tested_commit, tested_tree = _git_state(repo, git)
    if (tested_commit, tested_tree) != (code_commit, code_tree) or _snapshot_files(
        repo, TEST_FILES, max_input_bytes
    ) != test_sources:
        raise GateVerificationError("protected or test source file changed during tests")
    if exact_profile_specs is None or transaction_dir is None:
        raise GateVerificationError("exact profile execution specs and transaction directory are required")
    cumulative_exact = execute_transaction(
        exact_profile_specs,
        repo=repo,
        python_executable=python_executable,
        transaction_dir=transaction_dir,
        compare=compare,
    )
    if (
        not isinstance(cumulative_exact, dict)
        or cumulative_exact.get("format") != "oviv2_t1_exact_transaction_v1"
        or not isinstance(cumulative_exact.get("executions"), list)
    ):
        raise GateVerificationError("exact profile transaction result is invalid")
    if verify_exact_profile_runs(
        cumulative_exact["executions"], compare=compare
    ) != cumulative_exact:
        raise GateVerificationError("exact profile transaction verification disagrees")

    final_commit, final_tree = _git_state(repo, git)
    final_protected = verify_source_manifest(
        source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
    )
    final_tests = _snapshot_files(repo, TEST_FILES, max_input_bytes)
    if (final_commit, final_tree) != (code_commit, code_tree):
        raise GateVerificationError("git commit or tree changed during verification")
    if final_protected != protected or final_tests != test_sources:
        raise GateVerificationError("protected or test source file changed during verification")

    common = {
        "scope": SCOPE,
        "status": "PASS",
        "code_commit": code_commit,
        "code_tree": code_tree,
        "protected_records": protected,
    }
    deterministic_evidence = {
        "base_commit": base_commit,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "protected_files": protected,
        "test_sources": test_sources,
        "source_manifest": source_manifest_record,
        "cumulative_exact": cumulative_exact,
        "gates": {
            "t1_exact": {**common, "test_records": [t1_record]},
            "determinism": {**common, "test_records": [determinism_record]},
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "deterministic_evidence": deterministic_evidence,
        "receipt": {"created_at_utc": now_utc()},
    }

    def validate_before_publish() -> None:
        publish_commit, publish_tree = _git_state(repo, git)
        publish_protected = verify_source_manifest(
            source_manifest, repo=repo, git=git, max_input_bytes=max_input_bytes
        )
        if (publish_commit, publish_tree) != (code_commit, code_tree):
            raise GateVerificationError("git commit or tree changed before publication")
        if (
            publish_protected != protected
            or _snapshot_files(repo, TEST_FILES, max_input_bytes) != test_sources
        ):
            raise GateVerificationError(
                "protected or test source file changed before publication"
            )
        if verify_exact_profile_runs(
            cumulative_exact["executions"], compare=compare
        ) != cumulative_exact:
            raise GateVerificationError(
                "exact profile transaction changed before publication"
            )

    _atomic_json_no_replace(output, payload, validate_before_publish)
    return output


def run_focused_smoke(
    *,
    repo: Path,
    python_executable: str = sys.executable,
    run: CommandRunner = _default_run,
) -> subprocess.CompletedProcess[bytes]:
    return run((python_executable, "-m", "pytest", "-q", SMOKE_TEST), repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--write-source-manifest", type=Path)
    mode.add_argument("--verify-source-manifest", type=Path)
    parser.add_argument("--base-commit", default=CUMULATIVE_BASE_COMMIT)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--exact-profile-specs", type=Path)
    parser.add_argument("--transaction-dir", type=Path)
    args = parser.parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        if args.write_source_manifest is not None:
            manifest = build_source_manifest(repo=repo, base_commit=args.base_commit)
            write_source_manifest(
                repo=repo,
                destination=args.write_source_manifest,
                payload=manifest,
            )
        elif args.verify_source_manifest is not None:
            source = args.verify_source_manifest
            if not source.is_absolute():
                source = repo / source
            verify_source_manifest(_load_source_manifest(source), repo=repo)
        else:
            if args.exact_profile_specs is None or args.transaction_dir is None:
                raise GateVerificationError(
                    "--exact-profile-specs and --transaction-dir are required for development evidence"
                )
            specs = _load_source_manifest(args.exact_profile_specs)
            if not isinstance(specs, dict) or set(specs) != {"specs"} or not isinstance(specs["specs"], list):
                raise GateVerificationError("exact profile execution specs are invalid")
            generate_evidence(
                repo=repo,
                output=args.output,
                base_commit=args.base_commit,
                python_executable=sys.executable,
                exact_profile_specs=specs["specs"],
                transaction_dir=args.transaction_dir,
            )
    except (GateVerificationError, FileExistsError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
