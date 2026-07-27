#!/usr/bin/env python3
"""Package one immutable Apartment A0-A4 dual-readout candidate result."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    canonical_algorithm_hash,
)
from scripts.evaluation.evaluate_tesse_cd_common_v2 import (  # noqa: E402
    evaluate_common_v2,
)
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (  # noqa: E402
    input_binding_values_sha256,
    load_search_manifest,
    non_temporal_config_sha256,
)
from scripts.evaluation.verify_oviv2_dual_readout_development_gates import (  # noqa: E402
    GateVerificationError,
    verify_exact_profile_runs,
)
from src.evaluation.baselines.tesse_cd import (  # noqa: E402
    summarize_khronos_official_metrics_partial,
)
from src.evaluation.oviv2_temporal_occlusion import (  # noqa: E402
    mechanism_telemetry_from_sources,
)
from src.oviv2.temporal_config import ExecutionProfile  # noqa: E402


MANIFEST_ID = "oviv2-tesse-dual-readout-candidate-result-v1"
CUMULATIVE_BASE_COMMIT = "8034e79d9cb853166222610981a7e6893f6cca70"
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
EXACT_PROFILE_SEQUENCE = (
    "reference", "a0", "a1", "a0", "a2", "a0", "a3", "a0", "a4",
)
RUN_MANIFEST_FIELDS = {
    "schema_version", "protocol_id", "dataset", "method_id", "scene", "mode",
    "algorithm_hash", "processed_frame_count", "covered_frame_count",
    "trajectory_frame_count", "first_frame_index", "last_frame_index",
    "temporal_export_schema_version", "scheduled_frame_indices",
    "captured_frame_indices", "config", "normalized_run_config", "schedule",
    "target_manifest", "source_bindings", "input_sha256", "code_commit",
    "checkpoints", "occlusion_checkpoint_index", "source_index",
    "artifact_inventory", "final_current_map",
}
LEGACY_RUN_MANIFEST_FIELDS = RUN_MANIFEST_FIELDS - {"final_current_map"}
MAX_JSON_BYTES = 8 * 1024 * 1024
SOURCE_NAMES = (
    "search_manifest", "search_status", "candidate_config", "run_manifest",
    "common_v2_summary", "temporal_occlusion_result", "official_metrics",
    "t1_exact_evidence", "determinism_evidence", "short_gate_evidence",
    "anchor_evidence", "baseline_evidence",
)
METRIC_NAMES = (
    "current_miou", "object_f1", "ghost_rate", "background_f5_cm",
    "recovery_frames", "dynamic_f1", "change_f1", "runtime_seconds",
)
RESULT_KEYS = {
    "schema_version", "manifest_id", "candidate_id", "scene", "status",
    "evidence_scope", "sources", "bindings", "run_identity", "gates", "metrics",
    "profile", "component_map", "parameter_values", "mechanism_telemetry",
    "anchor_coverage_gate", "promotion_evidence",
}
EVIDENCE_SCOPE = {
    "publication": "pre_and_post_link_revalidated",
    "snapshot": "point_in_time_not_permanent",
}
T2_DIRECTIONS = {
    "dynamic_f1": "maximize_strict",
    "change_f1": "maximize_strict",
    "ghost_rate": "minimize_strict",
    "background_f5_cm": "maximize_strict",
    "recovery_frames": "minimize_strict",
    "current_miou": "maximize_noninferior",
    "object_f1": "maximize_noninferior",
}
MECHANISMS_BY_PROFILE = {
    "a0": (),
    "a1": ("absence", "readout_invalidation"),
    "a2": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
    ),
    "a3": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
        "background_release", "background_reclaim",
    ),
    "a4": (
        "absence", "readout_invalidation", "proposal_recovery", "epoch_reset",
        "background_release", "background_reclaim", "eligible_reid",
        "icp_attempt", "icp_accept", "motion_rejection",
    ),
}
_COMMON_REPLAY_CACHE: dict[
    tuple[Path, tuple[int, int, int, int, int]],
    tuple[dict[str, Any], tuple[FileIdentityWitness, ...]],
] = {}


class PublicationUncertainError(RuntimeError):
    """Publication may not have been rolled back to a known absent state."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _witness(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: dict[str, Any]
    data: bytes
    identity: tuple[int, int, int, int, int]

    @property
    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": hashlib.sha256(self.data).hexdigest(), "byte_count": len(self.data)}

    def revalidate(self) -> None:
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"source changed before publication: {self.path}") from exc
        if _witness(current) != self.identity or not stat.S_ISREG(current.st_mode):
            raise ValueError(f"source changed before publication: {self.path}")
        descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            data = _read_bounded(descriptor, self.path.name)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if _witness(after) != self.identity or data != self.data:
            raise ValueError(f"source changed before publication: {self.path}")


@dataclass(frozen=True)
class FileIdentityWitness:
    path: Path
    identity: tuple[int, int, int, int, int]

    def revalidate(self) -> None:
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"source changed before publication: {self.path}") from exc
        if not stat.S_ISREG(current.st_mode) or _witness(current) != self.identity:
            raise ValueError(f"source changed before publication: {self.path}")


@dataclass(frozen=True)
class MissingPathWitness:
    path: Path

    def revalidate(self) -> None:
        try:
            _reject_symlink_components(self.path, "missing official source")
            os.lstat(self.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"source changed before publication: {self.path}") from exc
        except ValueError as exc:
            raise ValueError(f"source changed before publication: {self.path}") from exc
        raise ValueError(f"source changed before publication: {self.path}")


ExactTransactionKey = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class ExactGateWitness:
    key: ExactTransactionKey
    executions: tuple[dict[str, Any], ...]
    expected: bytes

    def revalidate(self) -> None:
        try:
            current = verify_exact_profile_runs([dict(item) for item in self.executions])
        except (GateVerificationError, OSError) as exc:
            raise ValueError("exact execution changed before publication") from exc
        if _canonical(current) != self.expected:
            raise ValueError("exact execution changed before publication")


PublicationWitness = Snapshot | FileIdentityWitness | MissingPathWitness | ExactGateWitness


def _exact_transaction_key(
    exact: Mapping[str, Any], executions: Sequence[Mapping[str, Any]]
) -> ExactTransactionKey:
    bindings: list[tuple[str, str]] = []
    for execution in executions:
        root = execution.get("output_root")
        observation = execution.get("observation_receipt")
        if not isinstance(root, str) or not isinstance(observation, Mapping):
            raise ValueError("cumulative execution observation/root binding is invalid")
        bindings.append(
            (root, hashlib.sha256(_canonical(dict(observation))).hexdigest())
        )
    return hashlib.sha256(_canonical(dict(exact))).hexdigest(), tuple(bindings)


def _identity_witness(path: Path, label: str) -> FileIdentityWitness:
    absolute = path.absolute()
    _reject_symlink_components(absolute, label)
    try:
        status = os.stat(absolute, follow_symlinks=False)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return FileIdentityWitness(absolute, _witness(status))


def _read_bounded(descriptor: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= MAX_JSON_BYTES:
        chunk = os.read(descriptor, min(1024 * 1024, MAX_JSON_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > MAX_JSON_BYTES:
        raise ValueError(f"{label} is too large")
    return b"".join(chunks)


def _snapshot(path: str | Path, label: str, *, parse_json: bool = True) -> Snapshot:
    absolute = Path(path).absolute()
    _reject_symlink_components(absolute, label)
    try:
        before_path = os.lstat(absolute)
    except OSError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"{label} must be a regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        data = _read_bounded(descriptor, label)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(absolute, follow_symlinks=False)
    if _witness(before_path) != _witness(before) or _witness(before) != _witness(after) or _witness(after) != _witness(current):
        raise ValueError(f"{label} changed while it was read")
    payload: dict[str, Any] = {}
    if parse_json:
        try:
            parsed = json.loads(
                data.decode("utf-8"), object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")),
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} is not UTF-8") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} root must be an object")
        payload = parsed
    return Snapshot(absolute, payload, data, _witness(after))


def _sha(value: object, label: str, lengths: set[int] = {64}) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase hash")
    return value


def _record_matches(record: object, snap: Snapshot, label: str, *, path_required: bool = False, base: Path | None = None) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} source record is invalid")
    if record.get("sha256") != hashlib.sha256(snap.data).hexdigest():
        raise ValueError(f"{label} source record SHA256 mismatch")
    if "byte_count" in record and record["byte_count"] != len(snap.data):
        raise ValueError(f"{label} source record byte_count mismatch")
    if path_required or "path" in record:
        raw = record.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{label} source record path is invalid")
        resolved = Path(raw) if Path(raw).is_absolute() else (base or snap.path.parent) / raw
        if Path(os.path.abspath(resolved)) != snap.path:
            raise ValueError(f"{label} source record path mismatch")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {name} is non-finite")
    bounded = name not in {"recovery_frames", "runtime_seconds"}
    if bounded and not 0.0 <= result <= 1.0:
        raise ValueError(f"metric {name} must be in [0, 1]")
    if not bounded and result < 0.0:
        raise ValueError(f"metric {name} must be nonnegative")
    return result


def _same_content_record(left: object, right: object, label: str) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError(f"{label} record is missing")
    if {key: left.get(key) for key in ("sha256", "byte_count")} != {
        key: right.get(key) for key in ("sha256", "byte_count")
    }:
        raise ValueError(f"{label} content binding mismatch")


def _compare_source_indexes(run_index: Mapping[str, Any], exported: Mapping[str, Any], temporal: Mapping[str, Any]) -> None:
    for name, expected in (("dataset", "TESSE-CD"), ("method", "OVIV2"), ("scene", "apartment")):
        if run_index.get(name) != expected or exported.get(name) != expected or temporal.get(name) != expected:
            raise ValueError("common-v2 temporal source identity differs from candidate run")
    _same_content_record(run_index.get("schedule"), exported.get("schedule"), "export schedule")
    run_checkpoints = run_index.get("checkpoints")
    export_checkpoints = exported.get("checkpoints")
    temporal_checkpoints = temporal.get("checkpoints")
    if not isinstance(run_checkpoints, list) or not isinstance(export_checkpoints, list) or not isinstance(temporal_checkpoints, list):
        raise ValueError("common-v2 temporal checkpoint inventories are missing")
    if len(run_checkpoints) != len(export_checkpoints) or len(run_checkpoints) != len(temporal_checkpoints):
        raise ValueError("common-v2 temporal checkpoint inventory differs from candidate run")
    identity = ("frame_index", "timestamp_ns", "consumed_through_frame", "consumed_through_frame_exclusive")
    for run_item, export_item, temporal_item in zip(run_checkpoints, export_checkpoints, temporal_checkpoints):
        if not all(isinstance(item, Mapping) for item in (run_item, export_item, temporal_item)):
            raise ValueError("common-v2 temporal checkpoint is invalid")
        values = [tuple(item.get(key) for key in identity) for item in (run_item, export_item, temporal_item)]
        if values[0] != values[1] or values[0] != values[2]:
            raise ValueError("common-v2 temporal checkpoint identity differs from candidate run")
        frame = run_item.get("frame_index")
        if type(frame) is not int or run_item.get("consumed_through_frame") != frame or run_item.get("consumed_through_frame_exclusive") != frame + 1:
            raise ValueError("common-v2 temporal checkpoint is non-causal")
        for role in ("checkpoint_status", "snapshot", "entities"):
            _same_content_record(run_item.get(role), export_item.get(role), f"export checkpoint {role}")
        for role in ("snapshot", "entities"):
            _same_content_record(run_item.get(role), temporal_item.get(role), f"temporal checkpoint {role}")


def _source_record_path(
    record: object,
    *,
    base: Path,
    label: str,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} source record is not exact")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} source path is invalid")
    _sha(record.get("sha256"), f"{label} source sha256")
    if type(record.get("byte_count")) is not int or record["byte_count"] < 0:
        raise ValueError(f"{label} source byte_count is invalid")
    path = Path(raw_path)
    return (path if path.is_absolute() else base / path).absolute()


def _recompute_common_v2_metrics(
    common_snapshot: Snapshot,
    witnesses: list[PublicationWitness],
) -> dict[str, Any]:
    cache_key = (common_snapshot.path, common_snapshot.identity)
    cached = _COMMON_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        metrics, cached_witnesses = cached
        try:
            for witness in cached_witnesses:
                witness.revalidate()
        except ValueError:
            _COMMON_REPLAY_CACHE.pop(cache_key, None)
        else:
            witnesses.extend(cached_witnesses)
            return dict(metrics)

    common = common_snapshot.payload
    sources = common.get("sources")
    required = {"temporal_index", "target_manifest", "aliases", "label_space", "evaluator"}
    if not isinstance(sources, Mapping) or not required <= set(sources):
        raise ValueError("common-v2 replay sources are incomplete")
    paths = {
        name: _source_record_path(
            sources[name], base=common_snapshot.path.parent, label=f"common-v2 {name}"
        )
        for name in required
    }
    expected_evaluator = (
        REPO_ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py"
    ).absolute()
    if paths["evaluator"] != expected_evaluator:
        raise ValueError("common-v2 evaluator source is not the repository evaluator")

    with tempfile.TemporaryDirectory(prefix=".oviv2-common-v2-replay-") as temporary:
        replay_path = evaluate_common_v2(
            paths["temporal_index"],
            paths["target_manifest"],
            paths["aliases"],
            paths["label_space"],
            Path(temporary) / "output",
        )
        replay = _snapshot(replay_path, "replayed common-v2 summary").payload

    for field in (
        "metrics",
        "frames",
        "event_region_prediction_counts",
        "event_background_prediction_counts",
        "sources",
    ):
        if common.get(field) != replay.get(field):
            if field == "metrics":
                raise ValueError("common-v2 metrics differ from replay")
            raise ValueError(f"common-v2 {field} differs from replay")

    replay_sources = replay.get("sources")
    if not isinstance(replay_sources, Mapping):
        raise ValueError("replayed common-v2 sources are invalid")
    replay_witnesses: list[FileIdentityWitness] = []
    for name, record in replay_sources.items():
        path = _source_record_path(
            record,
            base=common_snapshot.path.parent,
            label=f"replayed common-v2 {name}",
        )
        witness = _identity_witness(path, f"replayed common-v2 {name}")
        if witness.identity[2] != record["byte_count"]:
            raise ValueError(f"replayed common-v2 {name} byte_count mismatch")
        replay_witnesses.append(witness)
    metrics = replay.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("replayed common-v2 metrics are invalid")
    witnesses.extend(replay_witnesses)
    _COMMON_REPLAY_CACHE[cache_key] = (dict(metrics), tuple(replay_witnesses))
    return metrics


def _recompute_official_metrics(
    official_snapshot: Snapshot,
    witnesses: list[PublicationWitness],
) -> tuple[dict[str, Any], dict[str, str]]:
    official = official_snapshot.payload
    sources = official.get("sources")
    names = ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv")
    if not isinstance(sources, list) or len(sources) != len(names):
        raise ValueError("official metrics must declare the exact three CSV sources")
    if (
        official_snapshot.path.name != "official_metrics.json"
        or official_snapshot.path.parent.name != "evaluation"
        or official_snapshot.path.parent.parent.name != "khronos"
    ):
        raise ValueError("official metrics path is not canonical")
    results_dir = official_snapshot.path.parent.parent / "map" / "results"
    source_snapshots: dict[str, Snapshot] = {}
    for name, record in zip(names, sources):
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError(f"official source record is invalid: {name}")
        raw_path = Path(record["path"])
        path = (
            raw_path
            if raw_path.is_absolute()
            else official_snapshot.path.parent / raw_path
        )
        path = Path(os.path.abspath(path))
        if path != Path(os.path.abspath(results_dir / name)):
            raise ValueError(f"official source path is not canonical: {name}")
        if record.get("status") == "MISSING":
            if set(record) != {"path", "status"}:
                raise ValueError(f"missing official source record is not exact: {name}")
            missing = MissingPathWitness(path)
            try:
                missing.revalidate()
            except ValueError as exc:
                raise ValueError(
                    f"official source declared missing but exists: {name}"
                ) from exc
            witnesses.append(missing)
            continue
        if set(record) != {"path", "sha256", "byte_count"}:
            raise ValueError(f"official source record is not exact: {name}")
        source = _snapshot(path, f"official source {name}", parse_json=False)
        _record_matches(
            record,
            source,
            f"official source {name}",
            path_required=True,
            base=official_snapshot.path.parent,
        )
        witnesses.append(source)
        source_snapshots[name] = source

    with tempfile.TemporaryDirectory(prefix="oviv2-official-metrics-") as directory:
        # Keep source filenames and rebase path-bearing diagnostics after replay.
        snapshot_results = Path(directory) / results_dir.name
        snapshot_results.mkdir()
        for name, source in source_snapshots.items():
            (snapshot_results / name).write_bytes(source.data)
        recomputed = summarize_khronos_official_metrics_partial(snapshot_results)
    metrics = {
        "state_count": recomputed["state_count"],
        **recomputed["metrics"],
    }
    recomputed_unavailable = recomputed["unavailable"]
    declared_unavailable = official.get("unavailable", {})
    if not isinstance(metrics, dict) or not isinstance(recomputed_unavailable, dict):
        raise ValueError("recomputed official metrics are invalid")
    recomputed_unavailable = {
        name: (
            reason.replace(str(snapshot_results), str(results_dir))
            if isinstance(reason, str)
            else reason
        )
        for name, reason in recomputed_unavailable.items()
    }
    if (
        not isinstance(declared_unavailable, dict)
        or declared_unavailable != recomputed_unavailable
    ):
        raise ValueError("official unavailable metrics differ from recomputed CSV")
    if any(
        not isinstance(reason, str) or not reason.strip()
        for reason in declared_unavailable.values()
    ):
        raise ValueError("official unavailable metric reason is invalid")
    expected_status = "PARTIAL" if recomputed_unavailable else "PASS"
    if official.get("status") != expected_status:
        raise ValueError("official metrics status differs from recomputed CSV")
    if official.get("metrics") != metrics:
        raise ValueError("official metrics differ from recomputed CSV")
    return metrics, declared_unavailable


def _metric(value: object, name: str, source: str) -> dict[str, Any]:
    return {"available": True, "value": _finite(value, name), "reason": "available", "source": source}


def _validate_evidence_file_records(
    records: object,
    expected_paths: Sequence[str],
    label: str,
    witnesses: list[PublicationWitness],
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> None:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ValueError(f"{label} inventory is not exact")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError(f"{label} record schema is not exact")
        path = record.get("path")
        if not isinstance(path, str):
            raise ValueError(f"{label} path is invalid")
        paths.append(path)
    if paths != list(expected_paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} path inventory is not canonical")
    for record, relative in zip(records, expected_paths, strict=True):
        snapshot = _snapshot(REPO_ROOT / relative, label, parse_json=False)
        digest = hashlib.sha256(snapshot.data).hexdigest()
        if (
            record["sha256"] != digest
            or record["bytes"] != len(snapshot.data)
            or (
                expected_hashes is not None
                and digest != expected_hashes[relative]
            )
        ):
            raise ValueError(f"{label} hash/byte binding mismatch")
        witnesses.append(snapshot)


def _gate_evidence(
    snapshot: Snapshot,
    name: str,
    run: Mapping[str, Any],
    witnesses: list[PublicationWitness],
    exact_transactions: dict[ExactTransactionKey, ExactGateWitness],
) -> None:
    root = snapshot.payload
    evidence = root.get("deterministic_evidence")
    if set(root) != {"schema_version", "manifest_id", "deterministic_evidence", "receipt"} or root.get("schema_version") != 1 or root.get("manifest_id") != "oviv2_dual_readout_development_gates_v1" or not isinstance(evidence, Mapping):
        raise ValueError(f"{name} evidence identity mismatch")
    if set(evidence) != {
        "base_commit", "code_commit", "code_tree", "protected_files",
        "test_sources", "source_manifest", "cumulative_exact", "gates",
    }:
        raise ValueError(f"{name} evidence schema is not exact")
    if evidence.get("base_commit") != CUMULATIVE_BASE_COMMIT:
        raise ValueError(f"{name} evidence base commit is invalid")
    if evidence.get("code_commit") != run["code_commit"] or not isinstance(evidence.get("code_tree"), str):
        raise ValueError(f"{name} evidence commit/tree differs from run code")
    protected = evidence.get("protected_files")
    tests = evidence.get("test_sources")
    gate = evidence.get("gates", {}).get(name) if isinstance(evidence.get("gates"), Mapping) else None
    if not isinstance(protected, list) or not protected or not isinstance(tests, list) or not tests or not isinstance(gate, Mapping):
        raise ValueError(f"{name} evidence records are missing")
    if set(gate) != {"scope", "status", "code_commit", "code_tree", "protected_records", "test_records"}:
        raise ValueError(f"{name} evidence gate schema is not exact")
    if gate.get("scope") != "shared_code_and_A0-A4_fixture" or gate.get("status") != "PASS":
        raise ValueError(f"{name} evidence gate did not PASS")
    if gate.get("code_commit") != evidence["code_commit"] or gate.get("code_tree") != evidence["code_tree"] or gate.get("protected_records") != protected:
        raise ValueError(f"{name} evidence protected hashes differ from run code evidence")
    commands = gate.get("test_records")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"{name} evidence has no named test record")
    if len(commands) != 1:
        raise ValueError(f"{name} evidence has an inexact test command list")
    expected_files = T1_TEST_FILES if name == "t1_exact" else DETERMINISM_TEST_FILES
    expected_tail = ["-m", "pytest", "-q", "-rA", "-o", "addopts=", *expected_files]
    for command in commands:
        if (
            not isinstance(command, Mapping)
            or set(command) != {"argv", "returncode", "stdout_sha256", "stdout_bytes", "stderr_sha256", "stderr_bytes"}
            or command.get("returncode") != 0
            or not isinstance(command.get("argv"), list)
            or len(command["argv"]) != len(expected_tail) + 1
            or command["argv"][1:] != expected_tail
            or not isinstance(command["argv"][0], str)
            or not command["argv"][0]
        ):
            raise ValueError(f"{name} evidence pytest argv is invalid")
        if command.get("returncode") != 0:
            raise ValueError(f"{name} evidence test did not PASS")
        _sha(command.get("stdout_sha256"), f"{name} stdout")
        _sha(command.get("stderr_sha256"), f"{name} stderr")
        if type(command.get("stdout_bytes")) is not int or type(command.get("stderr_bytes")) is not int:
            raise ValueError(f"{name} evidence test byte counts are invalid")

    source_record = evidence["source_manifest"]
    trusted_path = REPO_ROOT / "configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json"
    trusted = _snapshot(trusted_path, "trusted T1 source manifest")
    _record_matches(source_record, trusted, "trusted T1 source manifest", path_required=True)
    witnesses.append(trusted)
    trusted_files = trusted.payload.get("files")
    if not isinstance(trusted_files, Mapping) or any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in trusted_files.items()
    ):
        raise ValueError("trusted source manifest files are invalid")
    _validate_evidence_file_records(
        protected,
        tuple(trusted_files),
        f"{name} protected files",
        witnesses,
        expected_hashes=trusted_files,
    )
    _validate_evidence_file_records(
        tests,
        (*T1_TEST_FILES, *DETERMINISM_TEST_FILES),
        f"{name} test sources",
        witnesses,
    )
    exact = evidence["cumulative_exact"]
    if (
        not isinstance(exact, Mapping)
        or set(exact) != {"format", "sequence", "executions", "profiles"}
        or exact.get("format") != "oviv2_t1_exact_transaction_v1"
        or exact.get("sequence") != list(EXACT_PROFILE_SEQUENCE)
    ):
        raise ValueError(f"{name} cumulative exact schema is invalid")
    profiles = exact.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {"a0", "a1", "a2", "a3", "a4"}:
        raise ValueError(f"{name} cumulative profile inventory is incomplete")
    roots: set[str] = set()
    frames: set[tuple[int, ...]] = set()
    inventories: set[bytes] = set()
    for profile, audit in profiles.items():
        if not isinstance(audit, Mapping) or set(audit) != {"cumulative_root_sha256", "checkpoint_frames", "inventory"}:
            raise ValueError(f"{name} cumulative profile schema is invalid: {profile}")
        roots.add(_sha(audit.get("cumulative_root_sha256"), f"{name} cumulative root"))
        raw_frames = audit.get("checkpoint_frames")
        if not isinstance(raw_frames, list) or not raw_frames or raw_frames != sorted(set(raw_frames)) or any(type(frame) is not int or frame < 0 for frame in raw_frames):
            raise ValueError(f"{name} cumulative checkpoint inventory is invalid")
        frames.add(tuple(raw_frames))
        inventory = audit.get("inventory")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError(f"{name} cumulative artifact inventory is empty")
        previous_path = ""
        for item in inventory:
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "byte_count"}:
                raise ValueError(f"{name} cumulative artifact inventory schema is invalid")
            path = item.get("path")
            if not isinstance(path, str) or not path or path <= previous_path or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError(f"{name} cumulative artifact inventory path is invalid")
            previous_path = path
            _sha(item.get("sha256"), f"{name} cumulative artifact")
            if type(item.get("byte_count")) is not int or item["byte_count"] < 0:
                raise ValueError(f"{name} cumulative artifact byte count is invalid")
        inventories.add(_canonical(inventory))
    if len(roots) != 1 or len(frames) != 1 or len(inventories) != 1:
        raise ValueError(f"{name} cumulative profile roots disagree")
    executions = exact.get("executions")
    if not isinstance(executions, list) or len(executions) != len(EXACT_PROFILE_SEQUENCE):
        raise ValueError(f"{name} cumulative execution inventory is invalid")
    if any(not isinstance(item, Mapping) for item in executions):
        raise ValueError(f"{name} cumulative execution inventory is invalid")
    copied_executions = tuple(dict(item) for item in executions)
    source_sha = str(source_record["sha256"])
    if any(
        execution.get("code_commit") != evidence["code_commit"]
        or execution.get("source_manifest_sha256") != source_sha
        for execution in executions
    ):
        raise ValueError(f"{name} cumulative execution binding mismatch")
    key = _exact_transaction_key(exact, copied_executions)
    witness = exact_transactions.get(key)
    if witness is None:
        try:
            reopened = verify_exact_profile_runs(
                [dict(item) for item in copied_executions]
            )
        except (GateVerificationError, OSError) as exc:
            raise ValueError(
                f"{name} exact execution root/receipt/manifest is invalid: {exc}"
            ) from exc
        if reopened != exact:
            raise ValueError(
                f"{name} cumulative evidence differs from reopened executions"
            )
        witness = ExactGateWitness(key, copied_executions, _canonical(exact))
        exact_transactions[key] = witness
        witnesses.append(witness)


def _nested_value(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"candidate parameter is absent: {dotted}")
        current = current[part]
    return current


def _parameter_values(
    manifest: Mapping[str, Any], profile: str, temporal: Mapping[str, Any]
) -> dict[str, Any]:
    spaces = manifest.get("parameter_spaces")
    if not isinstance(spaces, Mapping):
        raise ValueError("search manifest parameter_spaces are missing")
    result: dict[str, Any] = {}
    for name, space in spaces.items():
        if (
            not isinstance(name, str)
            or not isinstance(space, Mapping)
            or set(space) != {"profiles", "values"}
            or not isinstance(space.get("profiles"), list)
            or not isinstance(space.get("values"), list)
        ):
            raise ValueError("search manifest parameter space schema is invalid")
        if profile not in space["profiles"]:
            continue
        selected = _nested_value(temporal, name)
        if selected not in space["values"]:
            raise ValueError(f"candidate parameter is outside its declared space: {name}")
        result[name] = selected
    return result


def _bound_source_snapshot(
    record: object,
    *,
    base: Path,
    label: str,
    witnesses: list[PublicationWitness],
    parse_json: bool = True,
) -> Snapshot:
    path = _source_record_path(record, base=base, label=label)
    snapshot = _snapshot(path, label, parse_json=parse_json)
    _record_matches(record, snapshot, label, path_required=True, base=base)
    witnesses.append(snapshot)
    return snapshot


def _parse_jsonl(snapshot: Snapshot, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(snapshot.data.splitlines(), 1):
        if not raw:
            raise ValueError(f"{label} contains a blank line")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} line {line_number} is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(value)
    return rows


def _mechanism_telemetry(
    artifact: Mapping[str, Any],
    *,
    profile: str,
    run_root: Path,
    run_source_index: Mapping[str, Any],
    witnesses: list[PublicationWitness],
) -> dict[str, Any]:
    declared = artifact.get("mechanism_telemetry")
    macro = artifact.get("macro")
    if (
        not isinstance(declared, Mapping)
        or not isinstance(macro, Mapping)
        or macro.get("mechanism_telemetry") != declared
        or set(declared) != set(MECHANISMS_BY_PROFILE[profile])
    ):
        raise ValueError("mechanism telemetry inventory is not exact")
    roles = (
        "trajectories", "lifecycle_transitions", "frame_coverage",
        "runtime_diagnostics",
    )
    source_records: dict[str, dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    for role in roles:
        record = run_source_index.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"{role} mechanism source is missing")
        snapshot = _bound_source_snapshot(
            record, base=run_root, label=f"{role} mechanism source",
            witnesses=witnesses, parse_json=role == "runtime_diagnostics",
        )
        source_records[role] = dict(record)
        payloads[role] = (
            snapshot.payload
            if role == "runtime_diagnostics"
            else _parse_jsonl(snapshot, role)
        )
    derived = mechanism_telemetry_from_sources(
        trajectories=payloads["trajectories"],
        lifecycle_transitions=payloads["lifecycle_transitions"],
        frame_coverage=payloads["frame_coverage"],
        runtime_diagnostics=payloads["runtime_diagnostics"],
        source_records=source_records,
    )
    if derived != declared:
        differences = sorted(
            name
            for name in set(derived) | set(declared)
            if derived.get(name) != declared.get(name)
        )
        raise ValueError(
            ",".join(differences)
            + " mechanism telemetry differs from source-backed evaluator replay"
        )
    failed = [name for name, item in derived.items() if item["passed"] is not True]
    if failed:
        raise ValueError(
            "mechanism has no usable opportunity/trigger: " + ",".join(failed)
        )
    return derived


def _anchor_coverage_gate(artifact: Mapping[str, Any]) -> dict[str, Any]:
    mappings = artifact.get("anchor_mappings")
    macro = artifact.get("macro")
    declared = macro.get("anchor_coverage_gate") if isinstance(macro, Mapping) else None
    expected_mapping_fields = {
        "scene", "object_id", "lifecycle_index", "anchor_frame_index",
        "anchor_relative_timestamp_ns", "eligible", "target_voxel_count",
        "mapped_temporal_id", "overlap_voxel_count", "ambiguous",
    }
    if not isinstance(mappings, list) or not isinstance(declared, Mapping):
        raise ValueError("anchor mapping evidence is missing")
    identities: set[tuple[str, int]] = set()
    eligible: list[Mapping[str, Any]] = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping) or set(mapping) != expected_mapping_fields:
            raise ValueError("anchor mapping schema is not exact")
        object_id = mapping.get("object_id")
        lifecycle_index = mapping.get("lifecycle_index")
        if not (
            mapping.get("scene") == "apartment"
            and isinstance(object_id, (str, int))
            and type(lifecycle_index) is int
            and type(mapping.get("anchor_frame_index")) is int
            and mapping["anchor_frame_index"] >= 0
            and type(mapping.get("anchor_relative_timestamp_ns")) is int
            and mapping["anchor_relative_timestamp_ns"] >= 0
            and type(mapping.get("target_voxel_count")) is int
            and mapping["target_voxel_count"] >= 0
        ):
            raise ValueError("anchor mapping identity is invalid")
        identity = (str(object_id), lifecycle_index)
        if identity in identities:
            raise ValueError("anchor mapping identity is duplicated")
        identities.add(identity)
        if mapping.get("eligible") is True:
            eligible.append(mapping)
        elif mapping.get("eligible") is not False:
            raise ValueError("anchor mapping eligible flag is invalid")
    mapped = 0
    zero_overlap = 0
    ambiguous = 0
    for mapping in eligible:
        overlap = mapping.get("overlap_voxel_count")
        is_ambiguous = mapping.get("ambiguous")
        if type(overlap) is not int or overlap < 0 or type(is_ambiguous) is not bool:
            raise ValueError("anchor mapping evidence value is invalid")
        temporal_id = mapping.get("mapped_temporal_id")
        if temporal_id is not None and (type(temporal_id) is not int or temporal_id < 0):
            raise ValueError("anchor temporal ID is invalid")
        unique = temporal_id is not None and overlap > 0 and not is_ambiguous
        mapped += int(unique)
        zero_overlap += int(overlap == 0)
        ambiguous += int(is_ambiguous)
    eligible_count = len(eligible)
    available = eligible_count > 0
    passed = available and eligible_count >= 66 and mapped >= 53
    if not available:
        reason = "no_eligible_anchor_mappings"
    elif eligible_count < 66:
        reason = "insufficient_eligible_anchor_mappings"
    elif mapped < 53:
        reason = "insufficient_unique_anchor_mappings"
    else:
        reason = None
    result = {
        "scene": "apartment",
        "eligible_count": eligible_count,
        "uniquely_mapped_count": mapped,
        "zero_overlap_count": zero_overlap,
        "ambiguous_count": ambiguous,
        "required_eligible_count": 66,
        "required_mapped_count": 53,
        "available": available,
        "passed": passed,
        "reason": reason,
    }
    if set(declared) != set(result) or dict(declared) != result:
        raise ValueError("anchor coverage gate differs from source mappings")
    if not passed:
        raise ValueError("Apartment anchor coverage requires at least 53/66")
    return result


def _baseline_values(
    snapshot: Snapshot, witnesses: list[PublicationWitness]
) -> dict[str, float]:
    root = snapshot.payload
    if set(root) != {
        "schema_version", "manifest_id", "dataset", "protocol_id", "scene",
        "baselines",
    } or not (
        root.get("schema_version") == 1
        and root.get("manifest_id")
        == "oviv2-tesse-dual-readout-baseline-evidence-v1"
        and root.get("dataset") == "TESSE-CD"
        and root.get("protocol_id") == "oviv2-tessecd-v2"
        and root.get("scene") == "apartment"
    ):
        raise ValueError("baseline evidence identity/schema is invalid")
    baselines = root.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(T2_DIRECTIONS):
        raise ValueError("baseline metric inventory is not exact")
    result: dict[str, float] = {}
    cache: dict[Path, Snapshot] = {}
    expected_package_fields = {
        "schema_version", "manifest_id", "dataset", "protocol_id", "scene",
        "status", "method_id", "oracle", "metrics",
    }
    for name in T2_DIRECTIONS:
        entry = baselines[name]
        if not isinstance(entry, Mapping) or set(entry) != {"metric", "source"} or entry.get("metric") != name:
            raise ValueError(f"baseline evidence entry is invalid: {name}")
        path = _source_record_path(
            entry["source"], base=snapshot.path.parent, label=f"{name} baseline"
        )
        package = cache.get(path)
        if package is None:
            package = _bound_source_snapshot(
                entry["source"], base=snapshot.path.parent,
                label=f"{name} baseline", witnesses=witnesses,
            )
            cache[path] = package
        else:
            _record_matches(
                entry["source"], package, f"{name} baseline", path_required=True,
                base=snapshot.path.parent,
            )
        payload = package.payload
        if set(payload) != expected_package_fields or not (
            payload.get("schema_version") == 1
            and payload.get("manifest_id") == "tesse-cd-frozen-baseline-scene-result-v1"
            and payload.get("dataset") == "TESSE-CD"
            and payload.get("protocol_id") == "oviv2-tessecd-v2"
            and payload.get("scene") == "apartment"
            and payload.get("status") == "PASS"
            and payload.get("oracle") is False
        ):
            raise ValueError(f"baseline package identity/schema is invalid: {name}")
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or name not in metrics:
            raise ValueError(f"baseline package metric is missing: {name}")
        result[name] = _finite(metrics[name], name)
    return result


def _t2_gates(
    metrics: Mapping[str, Mapping[str, Any]], baselines: Mapping[str, float]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, direction in T2_DIRECTIONS.items():
        metric = metrics[name]
        available = metric.get("available") is True
        value = metric.get("value") if available else None
        baseline = baselines[name]
        passed = False
        if available:
            numeric = _finite(value, name)
            if direction == "maximize_strict":
                passed = numeric > baseline
            elif direction == "minimize_strict":
                passed = numeric < baseline
            else:
                passed = numeric >= baseline
            value = numeric
        result[name] = {
            "direction": direction,
            "baseline": baseline,
            "value": value,
            "available": available,
            "passed": passed,
            "source": metric.get("source"),
        }
    return result


def _derive(
    candidate_id: str,
    snapshots: Mapping[str, Snapshot],
    *,
    auxiliary: list[PublicationWitness] | None = None,
) -> dict[str, Any]:
    if set(snapshots) != set(SOURCE_NAMES):
        raise ValueError("source set is not exact")
    witnesses = auxiliary if auxiliary is not None else []
    exact_transactions: dict[ExactTransactionKey, ExactGateWitness] = {}

    def take(path: str | Path, label: str, *, parse_json: bool = True) -> Snapshot:
        snapshot = _snapshot(path, label, parse_json=parse_json)
        witnesses.append(snapshot)
        return snapshot
    manifest = snapshots["search_manifest"].payload
    if load_search_manifest(snapshots["search_manifest"].path) != manifest:
        raise ValueError("search manifest does not satisfy the production contract")
    if not (manifest.get("schema_version") == 1 and manifest.get("manifest_id") == "oviv2-tesse-dual-readout-search-v1"
            and manifest.get("dataset") == "TESSE-CD" and manifest.get("method_id") == "OVIV2"
            and manifest.get("protocol_id") == "oviv2-tessecd-v2" and manifest.get("development_scene") == "apartment"):
        raise ValueError("search manifest identity mismatch")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len({item.get("candidate_id") for item in candidates if isinstance(item, Mapping)}) != len(candidates):
        raise ValueError("search manifest candidates are invalid")
    declaration = next((item for item in candidates if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id), None)
    if declaration is None or candidate_id not in {"a0", "a1", "a2", "a3", "a4"}:
        raise ValueError("candidate is not declared in A0-A4 manifest")
    profile = ExecutionProfile.from_id(candidate_id)
    if declaration.get("components") != profile.components:
        raise ValueError("candidate manifest component map is profile-incompatible")

    status = snapshots["search_status"].payload
    if status.get("status") != "PASS" or not isinstance(status.get("candidates"), list):
        raise ValueError("search status is not PASS")
    manifest_record = status.get("manifest")
    _record_matches(manifest_record, snapshots["search_manifest"], "search manifest", path_required=True)
    records = [item for item in status["candidates"] if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id]
    if len(records) != 1:
        raise ValueError("candidate is absent or duplicated in search status")
    record = records[0]
    if record.get("status") != "PASS" or record.get("exit_code") != 0 or record.get("scene") != "apartment":
        raise ValueError("candidate search status is not an Apartment PASS")
    config_snap = snapshots["candidate_config"]
    if Path(str(record.get("config_path"))).absolute() != config_snap.path:
        raise ValueError("search status config path mismatch")
    config = config_snap.payload
    canonical_config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    if record.get("config_sha256") != canonical_config_hash:
        raise ValueError("search status config hash mismatch")
    _record_matches(record.get("config_file"), config_snap, "search status config", path_required=True)
    if config.get("scene") != "apartment" or config.get("temporal_readout") != declaration.get("temporal_readout"):
        raise ValueError("candidate config does not match Apartment manifest declaration")
    temporal_config = config.get("temporal_readout")
    if (
        not isinstance(temporal_config, Mapping)
        or temporal_config.get("execution_profile") != candidate_id
        or temporal_config.get("components") != profile.components
    ):
        raise ValueError("candidate profile/components are incompatible")
    parameter_values = _parameter_values(manifest, candidate_id, temporal_config)
    algorithm_hash = config.get("algorithm_hash")
    if algorithm_hash != canonical_algorithm_hash(config) or record.get("algorithm_hash") != algorithm_hash:
        raise ValueError("candidate algorithm hash mismatch")
    non_temporal = non_temporal_config_sha256(config)
    if record.get("non_temporal_config_sha256") != non_temporal:
        raise ValueError("candidate non-temporal binding mismatch")
    if record.get("input_binding_values_sha256") != input_binding_values_sha256(config):
        raise ValueError("candidate input binding values hash mismatch")
    for stream in ("stdout", "stderr"):
        raw_path = record.get(f"{stream}_path")
        if not isinstance(raw_path, str):
            raise ValueError(f"search status {stream} path is missing")
        stream_snap = take(raw_path, f"candidate {stream}", parse_json=False)
        _record_matches(record.get(f"{stream}_file"), stream_snap, f"candidate {stream}", path_required=True)
    runtime = _finite(record.get("runtime_seconds"), "runtime_seconds")

    run_snap = snapshots["run_manifest"]
    run = run_snap.payload
    run_root = run_snap.path.parent
    if set(run) not in (
        LEGACY_RUN_MANIFEST_FIELDS,
        LEGACY_RUN_MANIFEST_FIELDS | {"frozen_run_identity"},
        RUN_MANIFEST_FIELDS,
        RUN_MANIFEST_FIELDS | {"frozen_run_identity"},
    ):
        raise ValueError("candidate run manifest field inventory is not exact")
    if Path(str(record.get("output_root"))).absolute() != run_root:
        raise ValueError("search status output root mismatch")
    if not (run.get("schema_version") == 2 and run.get("protocol_id") == "oviv2-tessecd-v2"
            and run.get("dataset") == "TESSE-CD" and run.get("method_id") == "OVIV2"
            and run.get("scene") == "apartment" and run.get("mode") == "dual_readout_causal_checkpoints"):
        raise ValueError("candidate run manifest identity mismatch")
    _record_matches(run.get("config"), config_snap, "run config")
    normalized_record = run.get("normalized_run_config")
    if not isinstance(normalized_record, Mapping) or not isinstance(normalized_record.get("path"), str):
        raise ValueError("normalized run config record is invalid")
    normalized_snap = take(run_root / normalized_record["path"], "normalized run config")
    _record_matches(normalized_record, normalized_snap, "normalized run config", path_required=True, base=run_root)
    if normalized_snap.payload != config:
        raise ValueError("normalized run config differs from candidate config")
    final_current = run.get("final_current_map")
    if final_current is None and any(
        isinstance(item, str) and item.startswith("final_current_map/")
        for item in run.get("artifact_inventory", ())
    ):
        raise ValueError("new runner artifact lacks final current map binding")
    if final_current is not None:
        if not isinstance(final_current, Mapping) or set(final_current) != {
            "frame_index", "timestamp_ns", "scope", "snapshot", "entities",
            "background_storage",
        }:
            raise ValueError("final current map schema is not exact")
        if not (
            final_current.get("frame_index") == run.get("last_frame_index")
            and isinstance(final_current.get("timestamp_ns"), int)
            and not isinstance(final_current.get("timestamp_ns"), bool)
            and final_current.get("scope") == "current"
            and final_current.get("background_storage")
            == "snapshot.npz:background_xyz"
        ):
            raise ValueError("final current map progress binding is invalid")
        artifact_inventory = run.get("artifact_inventory")
        if (
            not isinstance(artifact_inventory, list)
            or artifact_inventory != sorted(set(artifact_inventory))
            or any(not isinstance(item, str) or not item for item in artifact_inventory)
        ):
            raise ValueError("run artifact inventory is invalid")

        def internal_artifact(record: Mapping[str, Any], label: str) -> Path:
            raw_path = record.get("path")
            if not (
                isinstance(raw_path, str)
                and raw_path not in {"", "."}
                and not Path(raw_path).is_absolute()
                and Path(raw_path).as_posix() == raw_path
                and ".." not in Path(raw_path).parts
                and raw_path in artifact_inventory
            ):
                raise ValueError(f"{label} path is not a canonical run artifact")
            path = Path(os.path.abspath(run_root / raw_path))
            if Path(os.path.abspath(run_root)) not in path.parents:
                raise ValueError(f"{label} is outside the run")
            return path

        source_index_record = run.get("source_index")
        if not isinstance(source_index_record, Mapping) or not isinstance(
            source_index_record.get("path"), str
        ):
            raise ValueError("run source index record is invalid")
        final_source_index = take(
            internal_artifact(source_index_record, "run source index for final map"),
            "run source index for final map",
        )
        _record_matches(
            source_index_record, final_source_index, "run source index for final map",
            path_required=True, base=run_root,
        )
        coverage_record = final_source_index.payload.get("frame_coverage")
        if not isinstance(coverage_record, Mapping) or not isinstance(
            coverage_record.get("path"), str
        ):
            raise ValueError("run frame coverage record is invalid")
        coverage_snapshot = take(
            internal_artifact(coverage_record, "run frame coverage for final map"),
            "run frame coverage for final map",
            parse_json=False,
        )
        _record_matches(
            coverage_record, coverage_snapshot, "run frame coverage for final map",
            path_required=True, base=run_root,
        )
        coverage_rows = _parse_jsonl(coverage_snapshot, "run frame coverage for final map")
        if not coverage_rows or (
            coverage_rows[-1].get("frame_index"), coverage_rows[-1].get("timestamp_ns")
        ) != (final_current["frame_index"], final_current["timestamp_ns"]):
            raise ValueError("final current map does not match run-end frame coverage")
        actual_inventory = []
        actual_directories = []
        for path in run_root.rglob("*"):
            relative = path.relative_to(run_root).as_posix()
            if path.is_symlink():
                raise ValueError("run artifact inventory contains a symlink")
            if path.is_file():
                if relative not in {"run_manifest.json", "execution_receipt.json"}:
                    actual_inventory.append(relative)
            elif path.is_dir():
                actual_directories.append(relative)
            else:
                raise ValueError("run artifact inventory contains a non-file entry")
        actual_inventory.sort()
        if artifact_inventory != actual_inventory:
            raise ValueError("run artifact inventory differs from run directory")
        expected_directories = sorted({
            parent.as_posix()
            for item in artifact_inventory
            for parent in Path(item).parents
            if parent.as_posix() != "."
        })
        if sorted(actual_directories) != expected_directories:
            raise ValueError("run directory inventory differs from artifact paths")
        final_snaps = []
        for role in ("snapshot", "entities"):
            record = final_current.get(role)
            if not isinstance(record, Mapping) or set(record) != {
                "path", "sha256", "byte_count"
            }:
                raise ValueError(f"final current map {role} record is invalid")
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or raw_path not in artifact_inventory:
                raise ValueError(f"final current map {role} is absent from artifact inventory")
            path = run_root / raw_path
            if run_root not in path.absolute().parents:
                raise ValueError(f"final current map {role} is outside the run")
            snap = take(path, f"final current map {role}", parse_json=False)
            _record_matches(
                record, snap, f"final current map {role}",
                path_required=True, base=run_root,
            )
            final_snaps.append(snap)
        if final_snaps[0].path == final_snaps[1].path or (
            final_snaps[0].identity[0], final_snaps[0].identity[1]
        ) == (final_snaps[1].identity[0], final_snaps[1].identity[1]):
            raise ValueError("final current map files alias")
        try:
            with np.load(final_snaps[0].path, allow_pickle=False) as arrays:
                if not {"background_xyz", "timestamp", "scope", "scene_id"} <= set(arrays.files):
                    raise ValueError("final current map snapshot metadata is incomplete")
                metadata = tuple(
                    (arrays[name].shape, arrays[name].item())
                    for name in ("timestamp", "scope", "scene_id")
                )
        except (OSError, ValueError) as exc:
            raise ValueError("final current map snapshot is invalid") from exc
        if metadata != (
            ((), final_current["timestamp_ns"]),
            ((), "current"),
            ((), "apartment"),
        ):
            raise ValueError("final current map snapshot metadata is stale")
    if run.get("algorithm_hash") != algorithm_hash or not isinstance(run.get("source_bindings"), Mapping) or not run["source_bindings"]:
        raise ValueError("run algorithm/input/source bindings differ from candidate")
    _sha(run.get("input_sha256"), "run input_sha256")
    _sha(run.get("code_commit"), "run code_commit", {40, 64})
    index_record = run.get("occlusion_checkpoint_index")
    if not isinstance(index_record, Mapping) or not isinstance(index_record.get("path"), str):
        raise ValueError("run occlusion index record is invalid")
    index_snap = take(run_root / index_record["path"], "run occlusion index")
    _record_matches(index_record, index_snap, "run occlusion index", path_required=True, base=run_root)
    index = index_snap.payload
    identity_fields = ("protocol_id", "dataset", "method_id", "scene", "algorithm_hash", "input_sha256", "code_commit", "source_bindings")
    if any(index.get(name) != run.get(name) for name in identity_fields):
        raise ValueError("run occlusion index identity mismatch")
    checkpoints = index.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("run occlusion index has no causal checkpoints")
    for item in checkpoints:
        frame = item.get("frame_index") if isinstance(item, Mapping) else None
        if type(frame) is not int or item.get("consumed_through_frame") != frame or item.get("consumed_through_frame_exclusive") != frame + 1:
            raise ValueError("run checkpoint violates causal boundary")
    source_index_record = run.get("source_index")
    if not isinstance(source_index_record, Mapping) or not isinstance(source_index_record.get("path"), str):
        raise ValueError("run source_index record is invalid")
    run_source_snap = take(run_root / source_index_record["path"], "run source index")
    _record_matches(source_index_record, run_source_snap, "run source index", path_required=True, base=run_root)

    common = snapshots["common_v2_summary"].payload
    if not (common.get("manifest_id") == "tesse_cd_common_v2_scene_summary" and common.get("status") == "PASS"
            and common.get("dataset") == "TESSE-CD" and common.get("protocol") == "tesse_cd_common_v2"
            and common.get("method") == "OVIV2" and common.get("mode") == "causal_checkpoints"
            and common.get("scene") == "apartment"):
        raise ValueError("common-v2 summary is not a PASS OVIV2 Apartment result")
    temporal_record = common.get("sources", {}).get("temporal_index") if isinstance(common.get("sources"), Mapping) else None
    if not isinstance(temporal_record, Mapping) or not isinstance(temporal_record.get("path"), str):
        raise ValueError("common-v2 temporal source is missing")
    temporal_snap = take(temporal_record["path"], "common-v2 temporal source")
    _record_matches(temporal_record, temporal_snap, "common-v2 temporal source", path_required=True)
    temporal = temporal_snap.payload
    export_record = temporal.get("sources", {}).get("source_index") if isinstance(temporal.get("sources"), Mapping) else None
    if not isinstance(export_record, Mapping) or not isinstance(export_record.get("path"), str):
        raise ValueError("common-v2 temporal export source_index is missing")
    export_path = Path(export_record["path"])
    if not export_path.is_absolute():
        export_path = temporal_snap.path.parent / export_path
    export_snap = take(export_path, "exported temporal source index")
    _record_matches(export_record, export_snap, "exported temporal source index", path_required=True, base=temporal_snap.path.parent)
    _compare_source_indexes(run_source_snap.payload, export_snap.payload, temporal)
    common_metrics = _recompute_common_v2_metrics(
        snapshots["common_v2_summary"], witnesses
    )
    if common.get("metrics") != common_metrics:
        raise ValueError("common-v2 metrics differ from replay")

    occlusion = snapshots["temporal_occlusion_result"].payload
    if set(occlusion) != {
        "format", "mapping_rule", "anchor_mappings", "events", "macro",
        "mechanism_telemetry", "input_bindings",
    } or not (
        occlusion.get("format") == "oviv2_temporal_compact_v1"
        and occlusion.get("mapping_rule")
        == "maximum_world_voxel_overlap_unique_winner_minimum_one_voxel"
        and isinstance(occlusion.get("events"), list)
    ):
        raise ValueError("temporal occlusion result format mismatch")
    occlusion_macro = occlusion.get("macro")
    if not isinstance(occlusion_macro, Mapping) or set(occlusion_macro) != {
        "anchor_coverage_gate", "anchor_mapping_coverage",
        "occluded_retention_rate", "stale_removal_accuracy",
        "reactivation_identity_accuracy", "mechanism_telemetry",
    }:
        raise ValueError("temporal occlusion macro schema is not exact")
    input_bindings = occlusion.get("input_bindings")
    indexes = input_bindings.get("indexes") if isinstance(input_bindings, Mapping) else None
    source_indexes = (
        input_bindings.get("source_indexes")
        if isinstance(input_bindings, Mapping)
        else None
    )
    if (
        not isinstance(input_bindings, Mapping)
        or set(input_bindings) != {
            "target_manifest", "indexes", "source_indexes",
            "maximum_cached_checkpoints",
        }
        or not isinstance(indexes, list)
        or len(indexes) != 1
        or not isinstance(source_indexes, list)
        or len(source_indexes) != 1
        or input_bindings.get("maximum_cached_checkpoints") != 1
    ):
        raise ValueError("temporal occlusion input index is not exact")
    if not isinstance(indexes[0], Mapping) or set(indexes[0]) != {
        "sha256", "byte_count"
    }:
        raise ValueError("temporal occlusion checkpoint index binding is invalid")
    _record_matches(indexes[0], index_snap, "temporal occlusion index")
    if input_bindings.get("target_manifest") != run.get("target_manifest"):
        raise ValueError("temporal occlusion target binding differs from run")
    source_index_binding = source_indexes[0]
    if (
        not isinstance(source_index_binding, Mapping)
        or set(source_index_binding) != {"scene", "path", "sha256", "byte_count"}
        or source_index_binding.get("scene") != "apartment"
    ):
        raise ValueError("temporal occlusion source-index binding is invalid")
    _record_matches(
        {name: source_index_binding[name] for name in ("path", "sha256", "byte_count")},
        run_source_snap,
        "temporal occlusion run source index", path_required=True, base=run_root,
    )
    for evidence_name in ("short_gate_evidence", "anchor_evidence"):
        evidence_snapshot = snapshots[evidence_name]
        if (
            evidence_snapshot.path != snapshots["temporal_occlusion_result"].path
            or evidence_snapshot.data != snapshots["temporal_occlusion_result"].data
        ):
            raise ValueError(
                f"{evidence_name} must bind the formal temporal occlusion evaluator artifact"
            )
    mechanism_telemetry = _mechanism_telemetry(
        occlusion, profile=candidate_id, run_root=run_root,
        run_source_index=run_source_snap.payload, witnesses=witnesses
    )
    anchor_coverage = _anchor_coverage_gate(occlusion)

    official = snapshots["official_metrics"].payload
    if not (official.get("status") in {"PASS", "PARTIAL"} and official.get("dataset") == "TESSE-CD"
            and official.get("scene") == "apartment" and official.get("method") == "OVIV2"
            and official.get("mode") == "causal_checkpoints"):
        raise ValueError("official metrics identity mismatch")
    official_identity = official.get("run_identity")
    if not isinstance(official_identity, Mapping) or official_identity.get("config_sha256") != canonical_config_hash:
        raise ValueError("official metrics run_identity/config mismatch")
    official_metrics, unavailable = _recompute_official_metrics(
        snapshots["official_metrics"], witnesses
    )

    _gate_evidence(
        snapshots["t1_exact_evidence"],
        "t1_exact",
        run,
        witnesses,
        exact_transactions,
    )
    _gate_evidence(
        snapshots["determinism_evidence"],
        "determinism",
        run,
        witnesses,
        exact_transactions,
    )
    first_evidence = snapshots["t1_exact_evidence"].payload["deterministic_evidence"]
    second_evidence = snapshots["determinism_evidence"].payload["deterministic_evidence"]
    if first_evidence != second_evidence:
        raise ValueError("T1 and determinism evidence transactions differ")

    if not isinstance(common_metrics, Mapping) or not isinstance(official_metrics, Mapping):
        raise ValueError("metric source mappings are missing")
    metrics = {
        "current_miou": _metric(common_metrics.get("current_miou"), "current_miou", "common_v2_summary"),
        "object_f1": _metric(official_metrics.get("object_f1"), "object_f1", "official_metrics"),
        "ghost_rate": _metric(common_metrics.get("ghost_rate"), "ghost_rate", "common_v2_summary"),
        "background_f5_cm": _metric(common_metrics.get("background_f5"), "background_f5_cm", "common_v2_summary"),
        "recovery_frames": _metric(common_metrics.get("recovery_frames"), "recovery_frames", "common_v2_summary"),
        "runtime_seconds": _metric(runtime, "runtime_seconds", "search_status"),
    }
    for name in ("dynamic_f1", "change_f1"):
        value = official_metrics.get(name)
        if value is None:
            reason_value = unavailable.get(name) if isinstance(unavailable, Mapping) else None
            reason = reason_value.get("reason") if isinstance(reason_value, Mapping) else reason_value
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"optional metric {name} has no unavailable reason")
            metrics[name] = {"available": False, "value": None, "reason": reason, "source": "official_metrics"}
        else:
            metrics[name] = _metric(value, name, "official_metrics")
    ordered_metrics = {name: metrics[name] for name in METRIC_NAMES}
    baseline_values = _baseline_values(snapshots["baseline_evidence"], witnesses)
    t2_metrics = _t2_gates(ordered_metrics, baseline_values)
    failed_promotion_gates = [
        f"t2_metrics.{name}"
        for name, gate in t2_metrics.items()
        if gate["passed"] is not True
    ]
    gates = {
        "correctness": {"passed": True, "reason": "all_source_and_identity_checks_passed", "source": "derived"},
        "causality": {"passed": True, "reason": "checkpoint_boundaries_and_temporal_links_validated", "source": "run_manifest"},
        "determinism": {"passed": True, "reason": "named_determinism_evidence_passed", "source": "determinism_evidence"},
        "t1_exact": {"passed": True, "reason": "named_t1_exact_evidence_passed", "source": "t1_exact_evidence"},
        "mechanisms": mechanism_telemetry,
        "anchor_coverage": anchor_coverage,
        "t2_metrics": t2_metrics,
    }
    return {
        "schema_version": 1, "manifest_id": MANIFEST_ID, "candidate_id": candidate_id,
        "scene": "apartment", "status": "PASS",
        "evidence_scope": dict(EVIDENCE_SCOPE),
        "sources": {name: snapshots[name].record for name in SOURCE_NAMES},
        "bindings": {"candidate": {"declaration": dict(declaration), "config": config},
            "config_sha256": canonical_config_hash, "non_temporal_config_sha256": non_temporal,
            "input_hashes": run["source_bindings"]},
        "run_identity": {name: run[name] for name in ("algorithm_hash", "input_sha256", "code_commit", "source_bindings")},
        "profile": {
            "candidate_id": candidate_id,
            "kind": "main",
            "execution_profile": candidate_id,
            "selectable": True,
        },
        "component_map": dict(profile.components),
        "parameter_values": parameter_values,
        "mechanism_telemetry": mechanism_telemetry,
        "anchor_coverage_gate": anchor_coverage,
        "promotion_evidence": {
            "selectable": True,
            "passed": not failed_promotion_gates,
            "failed_gates": failed_promotion_gates,
            "source": "derived_from_source_backed_gates",
        },
        "gates": gates, "metrics": ordered_metrics,
    }


def _publish(
    path: Path,
    payload: Mapping[str, Any],
    snapshots: Sequence[PublicationWitness],
) -> None:
    output = path.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent, "output parent")
    parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    opened_parent = os.fstat(parent_fd)
    parent_identity = (opened_parent.st_dev, opened_parent.st_ino)

    def revalidate_parent() -> None:
        current = os.stat(output.parent, follow_symlinks=False)
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != parent_identity:
            raise ValueError("output parent changed during publication")

    temporary = f".{output.name}.{secrets.token_hex(12)}.tmp"
    temp_identity: tuple[int, int] | None = None

    def name_identity(name: str) -> tuple[int, int]:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return current.st_dev, current.st_ino

    def unlink_known(name: str, identity: tuple[int, int]) -> None:
        try:
            current = name_identity(name)
        except FileNotFoundError:
            return
        if current != identity:
            raise PublicationUncertainError(
                f"publication state is uncertain: {name} no longer identifies the published inode"
            )
        os.unlink(name, dir_fd=parent_fd)

    def rollback_publication() -> None:
        if temp_identity is None:
            raise PublicationUncertainError(
                "publication state is uncertain: published inode identity is unavailable"
            )
        try:
            unlink_known(output.name, temp_identity)
            unlink_known(temporary, temp_identity)
            os.fsync(parent_fd)
            try:
                revalidate_parent()
            except ValueError as exc:
                raise PublicationUncertainError(
                    "publication state is uncertain: output parent changed during rollback"
                ) from exc
            try:
                name_identity(output.name)
            except FileNotFoundError:
                return
            raise PublicationUncertainError(
                "publication state is uncertain: output name was recreated during rollback"
            )
        except PublicationUncertainError:
            raise
        except OSError as exc:
            raise PublicationUncertainError(
                "publication state is uncertain: rollback could not be persisted"
            ) from exc

    try:
        revalidate_parent()
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"output already exists: {output}")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=parent_fd)
        try:
            st = os.fstat(fd); temp_identity = (st.st_dev, st.st_ino)
            data = _canonical(payload) + b"\n"
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        for snapshot in snapshots:
            snapshot.revalidate()
        revalidate_parent()
        os.link(temporary, output.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        try:
            if temp_identity != name_identity(output.name):
                raise ValueError("output changed after publication link")
            os.fsync(parent_fd)
            for snapshot in snapshots:
                snapshot.revalidate()
            revalidate_parent()
            if temp_identity != name_identity(output.name):
                raise ValueError("output changed after publication link")
            unlink_known(temporary, temp_identity)
            os.fsync(parent_fd)
        except BaseException as exc:
            try:
                rollback_publication()
            except PublicationUncertainError as uncertain:
                raise uncertain from exc
            raise
    finally:
        try:
            st = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
            if temp_identity == (st.st_dev, st.st_ino):
                os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def package_result(*, manifest: str | Path, search_status: str | Path, candidate_id: str,
                   candidate_config: str | Path, run_manifest: str | Path, common_v2_summary: str | Path,
                   temporal_occlusion_result: str | Path, official_metrics: str | Path,
                   t1_exact_evidence: str | Path, determinism_evidence: str | Path,
                   short_gate_evidence: str | Path, anchor_evidence: str | Path,
                   baseline_evidence: str | Path,
                   output: str | Path) -> dict[str, Any]:
    paths = {"search_manifest": manifest, "search_status": search_status, "candidate_config": candidate_config,
             "run_manifest": run_manifest, "common_v2_summary": common_v2_summary,
             "temporal_occlusion_result": temporal_occlusion_result, "official_metrics": official_metrics,
             "t1_exact_evidence": t1_exact_evidence, "determinism_evidence": determinism_evidence,
             "short_gate_evidence": short_gate_evidence, "anchor_evidence": anchor_evidence,
             "baseline_evidence": baseline_evidence}
    snapshots = {name: _snapshot(path, name.replace("_", " ")) for name, path in paths.items()}
    auxiliary: list[PublicationWitness] = []
    result = _derive(candidate_id, snapshots, auxiliary=auxiliary)
    _publish(Path(output), result, [*snapshots.values(), *auxiliary])
    return result


def load_and_revalidate_result(path: str | Path, *, manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    result_snapshot = _snapshot(path, "candidate result")
    payload = result_snapshot.payload
    if set(payload) != RESULT_KEYS or payload.get("manifest_id") != MANIFEST_ID or payload.get("schema_version") != 1:
        raise ValueError("candidate result schema is not exact")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_NAMES):
        raise ValueError("candidate result source records are not exact")
    snapshots: dict[str, Snapshot] = {}
    for name in SOURCE_NAMES:
        source_record = sources[name]
        if not isinstance(source_record, Mapping) or set(source_record) != {"path", "sha256", "byte_count"}:
            raise ValueError(f"source record is not exact: {name}")
        snap = _snapshot(source_record.get("path"), f"result source {name}")
        _record_matches(source_record, snap, name, path_required=True)
        snapshots[name] = snap
    if isinstance(manifest, Mapping):
        if dict(manifest) != snapshots["search_manifest"].payload:
            raise ValueError("supplied manifest differs from result source")
    elif Path(manifest).absolute() != snapshots["search_manifest"].path:
        raise ValueError("supplied manifest path differs from result source")
    auxiliary: list[PublicationWitness] = []
    derived = _derive(str(payload.get("candidate_id")), snapshots, auxiliary=auxiliary)
    if payload != derived:
        raise ValueError("candidate result does not match revalidated sources")
    for snapshot in [*snapshots.values(), *auxiliary]:
        snapshot.revalidate()
    result_snapshot.revalidate()
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCE_NAMES:
        if name == "search_manifest":
            parser.add_argument("--manifest", required=True, type=Path)
        else:
            parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--candidate-id", required=True, choices=("a0", "a1", "a2", "a3", "a4"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    package_result(manifest=args.manifest, search_status=args.search_status, candidate_id=args.candidate_id,
                   candidate_config=args.candidate_config, run_manifest=args.run_manifest,
                   common_v2_summary=args.common_v2_summary, temporal_occlusion_result=args.temporal_occlusion_result,
                   official_metrics=args.official_metrics, t1_exact_evidence=args.t1_exact_evidence,
                   determinism_evidence=args.determinism_evidence,
                   short_gate_evidence=args.short_gate_evidence,
                   anchor_evidence=args.anchor_evidence,
                   baseline_evidence=args.baseline_evidence, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
