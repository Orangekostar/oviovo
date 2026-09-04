#!/usr/bin/env python3
"""Run the source-bound ReScene pair backend or publish its exact blocker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.rescene_backend import ReSceneBackend
from src.oviv2.two_visit_contracts import TemporalQueryEvidence

BACKEND_ID = "RESCENE_CONCERTO_TWO_VISIT_V1"
BLOCKED_STATUS = "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "backend_id",
    "status",
    "ranking_eligible",
    "source",
    "adapter_contract",
    "checkpoint",
    "environment",
    "executor",
    "random_initialization",
}


class BackendRunError(ValueError):
    """Raised when a backend run cannot satisfy its frozen contract."""


@dataclass(frozen=True, slots=True)
class BackendRunPaths:
    root: Path
    evidence_manifest: Path
    evidence_arrays: Path | None
    receipt: Path


@dataclass(frozen=True, slots=True)
class _BackendSpec:
    config_path: Path
    config_content: bytes
    status: str
    ranking_eligible: bool
    checkout: Path
    source_commit: str
    source_manifest: Path
    source_manifest_content: bytes
    feature_schema: str
    neural_voxel_size_m: float
    checkpoint: Path
    checkpoint_sha256: str | None
    executor_command: tuple[str, ...] | None
    timeout_s: float


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise BackendRunError(f"symlink path component is not allowed: {current}")


def _regular_bytes(path: Path, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(absolute)
    try:
        before = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise BackendRunError(f"{label} is missing") from error
    if not stat.S_ISREG(before.st_mode):
        raise BackendRunError(f"{label} must be a regular file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(content) != before.st_size:
        raise BackendRunError(f"{label} changed while being read")
    return content


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackendRunError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BackendRunError(f"{label} must contain a JSON object")
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolve_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BackendRunError(f"{label} must be a non-empty path")
    declared = Path(value)
    if ".." in declared.parts:
        raise BackendRunError(f"{label} cannot contain parent traversal")
    return declared if declared.is_absolute() else REPO_ROOT / declared


def _bound_file(record: object, *, label: str) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise BackendRunError(f"{label} binding schema is invalid")
    path = _resolve_path(record.get("path"), label=f"{label} path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise BackendRunError(f"{label} binding values are invalid")
    content = _regular_bytes(path, label)
    if len(content) != byte_count or _sha256(content) != digest:
        raise BackendRunError(f"{label} binding mismatch")
    return path, content


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendRunError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BackendRunError(f"{label} must be finite and positive")
    return result


def _load_spec(config_path: str | Path) -> _BackendSpec:
    path = Path(os.path.abspath(os.fspath(config_path)))
    content = _regular_bytes(path, "backend config")
    config = _json_object(content, "backend config")
    if set(config) != _TOP_LEVEL_KEYS:
        raise BackendRunError("backend config schema is invalid")
    status = config.get("status")
    ranking_eligible = config.get("ranking_eligible")
    if (
        config.get("schema_version") != 1
        or config.get("backend_id") != BACKEND_ID
        or status not in {"PASS", BLOCKED_STATUS}
        or type(ranking_eligible) is not bool
        or ranking_eligible != (status == "PASS")
    ):
        raise BackendRunError("backend config identity or status is invalid")

    source = config.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "checkout",
        "commit",
        "manifest",
    }:
        raise BackendRunError("backend source schema is invalid")
    checkout = _resolve_path(source.get("checkout"), label="source checkout")
    commit = source.get("commit")
    if not isinstance(commit, str) or _GIT_SHA.fullmatch(commit) is None:
        raise BackendRunError("source commit is invalid")
    source_manifest, source_manifest_content = _bound_file(
        source.get("manifest"), label="source manifest"
    )
    source_payload = _json_object(source_manifest_content, "source manifest")
    try:
        declared_commit = source_payload["sources"]["rescene"]["commit"]
        selected_source_backbone = source_payload["checkpoint"][
            "selected_backbone"
        ]
    except (KeyError, TypeError) as error:
        raise BackendRunError("source manifest ReScene declaration is invalid") from error
    if declared_commit != commit or selected_source_backbone != "Concerto":
        raise BackendRunError("source manifest ReScene identity mismatch")

    adapter = config.get("adapter_contract")
    if not isinstance(adapter, Mapping) or set(adapter) != {
        "feature_schema",
        "neural_voxel_size_m",
    }:
        raise BackendRunError("adapter contract schema is invalid")
    feature_schema = adapter.get("feature_schema")
    if not isinstance(feature_schema, str) or not feature_schema:
        raise BackendRunError("adapter feature schema is invalid")
    neural_voxel_size_m = _finite_positive(
        adapter.get("neural_voxel_size_m"), "adapter neural voxel size"
    )

    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "status",
        "path",
        "sha256",
        "selected_backbone",
        "selection_policy",
    }:
        raise BackendRunError("checkpoint schema is invalid")
    if (
        checkpoint.get("status") != status
        or checkpoint.get("selected_backbone") != "Concerto"
        or checkpoint.get("selection_policy")
        != "official_or_source_bound_training_only"
    ):
        raise BackendRunError("checkpoint identity or policy is invalid")
    checkpoint_path = _resolve_path(checkpoint.get("path"), label="checkpoint path")
    checkpoint_sha256 = checkpoint.get("sha256")

    environment = config.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != {
        "status",
        "python",
    }:
        raise BackendRunError("environment schema is invalid")
    environment_status = environment.get("status")
    environment_python = environment.get("python")
    if not isinstance(environment_status, str) or not environment_status:
        raise BackendRunError("environment status is invalid")
    if environment_python is not None and (
        not isinstance(environment_python, str) or not environment_python
    ):
        raise BackendRunError("environment Python path is invalid")

    executor = config.get("executor")
    if not isinstance(executor, Mapping) or set(executor) != {"command", "timeout_s"}:
        raise BackendRunError("executor schema is invalid")
    raw_command = executor.get("command")
    if raw_command is None:
        command = None
    elif (
        not isinstance(raw_command, list)
        or not raw_command
        or any(not isinstance(item, str) or not item for item in raw_command)
    ):
        raise BackendRunError("executor command is invalid")
    else:
        command = tuple(raw_command)
    timeout_s = _finite_positive(executor.get("timeout_s"), "executor timeout")

    random_initialization = config.get("random_initialization")
    if not isinstance(random_initialization, Mapping) or set(
        random_initialization
    ) != {"allowed_for_plumbing", "ranking_eligible"}:
        raise BackendRunError("random initialization schema is invalid")
    if (
        type(random_initialization.get("allowed_for_plumbing")) is not bool
        or random_initialization.get("ranking_eligible") is not False
    ):
        raise BackendRunError("random initialization cannot be ranking eligible")

    if status == BLOCKED_STATUS:
        if checkpoint_sha256 is not None or command is not None:
            raise BackendRunError("blocked checkpoint config cannot enable execution")
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            raise BackendRunError("blocked checkpoint config is stale: file now exists")
    else:
        if (
            not isinstance(checkpoint_sha256, str)
            or _SHA256.fullmatch(checkpoint_sha256) is None
            or command is None
            or not isinstance(environment_python, str)
        ):
            raise BackendRunError("runnable backend identities are incomplete")

    return _BackendSpec(
        config_path=path,
        config_content=content,
        status=status,
        ranking_eligible=ranking_eligible,
        checkout=checkout,
        source_commit=commit,
        source_manifest=source_manifest,
        source_manifest_content=source_manifest_content,
        feature_schema=feature_schema,
        neural_voxel_size_m=neural_voxel_size_m,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        executor_command=command,
        timeout_s=timeout_s,
    )


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _record(path: Path) -> dict[str, object]:
    content = _regular_bytes(path, "published artifact")
    return {
        "path": str(path),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }


def _evidence_payload(
    evidence: TemporalQueryEvidence,
    arrays_record: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_id": "RESCENE_TEMPORAL_QUERY_EVIDENCE_V1",
        "status": evidence.status,
        "backend_name": evidence.backend_name,
        "backend_config_sha256": evidence.backend_config_sha256,
        "pair_sha256": evidence.pair_sha256,
        "temporal_query_ids": list(evidence.temporal_query_ids),
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "ranking_eligible": evidence.ranking_eligible,
        "runtime_s": evidence.runtime_s,
        "peak_memory_bytes": evidence.peak_memory_bytes,
        "diagnostics": dict(evidence.diagnostics),
        "prediction_sha256": evidence.prediction_sha256(),
        "content_sha256": evidence.content_sha256(),
        "arrays": arrays_record,
    }


def run_rescene_pair_backend(
    config_path: str | Path,
    pair_root: str | Path,
    output_root: str | Path,
) -> BackendRunPaths:
    """Execute one pair under the frozen backend identity and publish atomically."""

    spec = _load_spec(config_path)
    pair_path = Path(os.path.abspath(os.fspath(pair_root)))
    pair = load_neural_sample_artifact(pair_path)
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise BackendRunError(f"backend output already exists: {output}")
    _reject_symlink_components(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)

    backend = ReSceneBackend(
        checkout=spec.checkout,
        checkpoint=spec.checkpoint,
        checkpoint_sha256=spec.checkpoint_sha256,
        source_manifest=spec.source_manifest,
        executor_command=spec.executor_command,
        expected_feature_schema=spec.feature_schema,
        expected_neural_voxel_size_m=spec.neural_voxel_size_m,
        timeout_s=spec.timeout_s,
    )
    evidence = backend.infer(pair)
    if evidence.status != spec.status:
        raise BackendRunError(
            f"backend result {evidence.status} contradicts frozen status {spec.status}"
        )
    if evidence.ranking_eligible != spec.ranking_eligible:
        raise BackendRunError("backend ranking eligibility contradicts frozen config")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path: Path | None = None
        arrays_record: dict[str, object] | None = None
        if evidence.status == "PASS":
            if (
                evidence.query_masks is None
                or evidence.token_scores is None
                or evidence.query_scores is None
            ):
                raise BackendRunError("PASS evidence is missing prediction arrays")
            arrays_path = staging / "query_evidence.npz"
            with arrays_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    query_masks=evidence.query_masks,
                    token_scores=evidence.token_scores,
                    query_scores=evidence.query_scores,
                )
                stream.flush()
                os.fsync(stream.fileno())
            arrays_record = _record(arrays_path)
            arrays_record["path"] = arrays_path.name

        evidence_path = staging / "evidence.json"
        _write_file(evidence_path, _json_bytes(_evidence_payload(evidence, arrays_record)))
        pair_manifest = pair_path / "manifest.json"
        pair_arrays = pair_path / "arrays.npz"
        evidence_record = _record(evidence_path)
        evidence_record["path"] = evidence_path.name
        receipt = {
            "schema_version": 1,
            "artifact_id": "RESCENE_PAIR_BACKEND_RUN_V1",
            "backend_id": BACKEND_ID,
            "status": evidence.status,
            "ranking_eligible": evidence.ranking_eligible,
            "pair_sha256": pair.content_sha256(),
            "source_commit": spec.source_commit,
            "config": {
                "path": str(spec.config_path),
                "sha256": _sha256(spec.config_content),
                "byte_count": len(spec.config_content),
            },
            "pair_manifest": _record(pair_manifest),
            "pair_arrays": _record(pair_arrays),
            "source_manifest": {
                "path": str(spec.source_manifest),
                "sha256": _sha256(spec.source_manifest_content),
                "byte_count": len(spec.source_manifest_content),
            },
            "checkpoint": {
                "path": str(spec.checkpoint),
                "sha256": spec.checkpoint_sha256,
                "status": spec.status,
            },
            "evidence": evidence_record,
            "evidence_arrays": arrays_record,
        }
        receipt_path = staging / "receipt.json"
        _write_file(receipt_path, _json_bytes(receipt))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise BackendRunError(f"backend output already exists: {output}")
        staging.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return BackendRunPaths(
        root=output,
        evidence_manifest=output / "evidence.json",
        evidence_arrays=(
            output / "query_evidence.npz" if evidence.status == "PASS" else None
        ),
        receipt=output / "receipt.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/rescene_two_visit_backend.json",
    )
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        paths = run_rescene_pair_backend(
            arguments.config, arguments.pair_root, arguments.output_root
        )
    except (BackendRunError, OSError, ValueError) as error:
        parser.error(str(error))
    print(paths.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
