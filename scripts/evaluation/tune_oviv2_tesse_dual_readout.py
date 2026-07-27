#!/usr/bin/env python3
"""Build an Apartment shortlist, then select it against bound T4 evidence."""

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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.package_oviv2_tesse_dual_readout_result import (  # noqa: E402
    load_and_revalidate_result,
)
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (  # noqa: E402
    load_search_manifest,
)


RESULT_CONTRACT = "candidates/<candidate_id>/apartment/result.json"
_MAIN_IDS = ("a0", "a1", "a2", "a3", "a4")
_FALLBACK_ORDER = ("a4", "a3", "a2")
_REQUIRED_RESULT_GATES = ("correctness", "causality", "determinism", "t1_exact")
_T2_DIRECTIONS = {
    "dynamic_f1": "maximize_strict",
    "change_f1": "maximize_strict",
    "ghost_rate": "minimize_strict",
    "background_f5_cm": "maximize_strict",
    "recovery_frames": "minimize_strict",
}
_T4_BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}
_T4_RAW_SOURCE_NAMES = {
    "config", "run_manifest", "time_log", "gpu_samples", "query_measurements",
    "final_map_inventory", "protocol", "shortlist",
}
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_SELECTION_BYTES = 4 * 1024 * 1024
_MAX_T4_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def _stable_bytes(path: Path, label: str, *, max_bytes: int) -> bytes:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
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
        if before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} byte limit")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - byte_count + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} byte limit")
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


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path.parent, "selection output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, "selection output parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = _canonical_json(payload) + b"\n"
    created = False
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("selection output write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
        published = True
        temporary.unlink()
        created = False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    data = _stable_bytes(path, label, max_bytes=max_bytes)
    try:
        value = json.loads(
            data,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, data


def _record(path: Path, data: bytes, **extra: object) -> dict[str, Any]:
    return {
        **extra,
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} schema is not exact: {actual}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _metric_value(result: Mapping[str, Any], name: str) -> float:
    metric = result.get("metrics", {}).get(name)
    if not isinstance(metric, Mapping) or metric.get("available") is not True:
        raise ValueError(f"metric {name} is unavailable")
    return _finite_number(metric.get("value"), f"metric {name}")


def _passed(gate: object) -> bool:
    return gate is True or (isinstance(gate, Mapping) and gate.get("passed") is True)


def _source_binding(record: object, label: str) -> tuple[Path, bytes]:
    bound = _exact_keys(record, {"path", "sha256", "byte_count"}, label)
    path = Path(bound["path"]).absolute() if isinstance(bound["path"], str) else Path()
    if not isinstance(bound["path"], str) or str(path) != bound["path"]:
        raise ValueError(f"{label} path must be absolute and canonical")
    data = _stable_bytes(path, label, max_bytes=_MAX_SOURCE_BYTES)
    if (
        not isinstance(bound["sha256"], str)
        or hashlib.sha256(data).hexdigest() != bound["sha256"]
        or len(data) != bound["byte_count"]
    ):
        raise ValueError(f"{label} source binding mismatch")
    return path, data


def _manifest_and_record(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    manifest_path = Path(path).absolute()
    data = _stable_bytes(manifest_path, "search manifest", max_bytes=_MAX_MANIFEST_BYTES)
    manifest = load_search_manifest(manifest_path)
    if tuple(manifest.get("profile_fallback_order", _FALLBACK_ORDER)) != _FALLBACK_ORDER:
        raise ValueError("manifest profile fallback order must be a4, a3, a2")
    bounds = manifest.get("t4_bounds", _T4_BOUNDS)
    if not isinstance(bounds, Mapping) or set(bounds) != set(_T4_BOUNDS):
        raise ValueError("manifest T4 bounds schema is not exact")
    for name, expected in _T4_BOUNDS.items():
        if _finite_number(bounds[name], f"T4 bound {name}") != expected:
            raise ValueError(f"manifest T4 bound differs for {name}")
    return manifest, _record(manifest_path, data), data


def _profile(result: Mapping[str, Any]) -> tuple[str, bool, str]:
    profile = _exact_keys(
        result.get("profile"),
        {"candidate_id", "kind", "execution_profile", "selectable"},
        "candidate profile",
    )
    candidate_id = result.get("candidate_id")
    if profile["candidate_id"] != candidate_id:
        raise ValueError("candidate profile identity mismatch")
    execution_profile = profile["execution_profile"]
    if execution_profile not in _MAIN_IDS:
        raise ValueError("candidate execution profile is invalid")
    if profile["kind"] not in {"main", "diagnostic"} or not isinstance(
        profile["selectable"], bool
    ):
        raise ValueError("candidate profile kind/selectable fields are invalid")
    return execution_profile, profile["selectable"], profile["kind"]


def _candidate_reasons(result: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if (
        result.get("manifest_id")
        != "oviv2-tesse-dual-readout-candidate-result-v1"
        or result.get("scene") != "apartment"
        or result.get("status") != "PASS"
    ):
        reasons.append("development_scene_mismatch")
    profile, selectable, kind = _profile(result)
    if kind != "main" or not selectable:
        reasons.append("diagnostic_candidate_not_selectable")
        return reasons
    if profile not in _FALLBACK_ORDER:
        reasons.append("profile_not_in_fallback_order")
        return reasons

    gates = result.get("gates")
    if not isinstance(gates, Mapping):
        return [*reasons, "result_gates_missing"]
    for name in _REQUIRED_RESULT_GATES:
        if not _passed(gates.get(name)):
            reasons.append(f"{name}_gate_failed")

    anchor = gates.get("anchor_coverage")
    if not isinstance(anchor, Mapping) or not (
        anchor.get("available") is True
        and anchor.get("passed") is True
        and anchor.get("required_eligible_count") == 66
        and anchor.get("required_mapped_count") == 53
        and anchor.get("eligible_count") == 66
        and isinstance(anchor.get("uniquely_mapped_count"), int)
        and anchor["uniquely_mapped_count"] >= 53
    ):
        reasons.append("anchor_coverage_gate_failed")

    mechanisms = gates.get("mechanisms")
    if not isinstance(mechanisms, Mapping) or not mechanisms:
        reasons.append("mechanism_evidence_missing")
    else:
        for name, evidence in sorted(mechanisms.items()):
            if not isinstance(evidence, Mapping):
                reasons.append(f"mechanism_{name}_invalid")
                continue
            opportunities = evidence.get("opportunities")
            triggers = evidence.get("triggers")
            if (
                isinstance(opportunities, bool)
                or not isinstance(opportunities, int)
                or isinstance(triggers, bool)
                or not isinstance(triggers, int)
                or opportunities < 0
                or triggers < 0
                or triggers > opportunities
            ):
                reasons.append(f"mechanism_{name}_invalid")
            elif opportunities == 0:
                reasons.append(f"mechanism_{name}_no_opportunity")
            elif triggers == 0 or evidence.get("available") is not True or evidence.get("passed") is not True:
                reasons.append(f"mechanism_{name}_claim_not_supported")

    current_floor = _metric_value(baseline, "current_miou")
    object_floor = _metric_value(baseline, "object_f1")
    if _metric_value(result, "current_miou") < current_floor:
        reasons.append("current_miou_below_a0_floor")
    if _metric_value(result, "object_f1") < object_floor:
        reasons.append("object_f1_below_a0_floor")

    t2_evidence = gates.get("t2_metrics")
    if not isinstance(t2_evidence, Mapping):
        reasons.append("t2_metric_evidence_missing")
        t2_evidence = {}
    for name, direction in _T2_DIRECTIONS.items():
        value = _metric_value(result, name)
        evidence = t2_evidence.get(name)
        evidence_matches = (
            isinstance(evidence, Mapping)
            and evidence.get("passed") is True
            and evidence.get("available") is True
            and evidence.get("direction") == direction
            and math.isfinite(
                _finite_number(evidence.get("baseline"), f"{name} evidence baseline")
            )
            and _finite_number(evidence.get("value"), f"{name} evidence value") == value
        )
        if not evidence_matches:
            reasons.append(f"{name}_t2_hard_gate_failed")
    return reasons


def _shortlist_entry(result: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    profile, _, _ = _profile(result)
    return {
        "candidate_id": result["candidate_id"],
        "profile": profile,
        "config_sha256": result["bindings"]["config_sha256"],
        "algorithm_hash": result["run_identity"]["algorithm_hash"],
        "result": dict(record),
        "selected_config": result["bindings"]["candidate"]["config"],
        "selected_config_record": result["sources"]["candidate_config"],
    }


def tune_shortlist(
    manifest_path: str | Path,
    result_paths: Sequence[str | Path],
    output: str | Path,
    *,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest, manifest_record, manifest_bytes = _manifest_and_record(manifest_path)
    if isinstance(result_paths, (str, bytes)) or not result_paths:
        raise ValueError("result_paths must be a non-empty sequence")

    loaded: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for raw in result_paths:
        path = Path(raw).absolute()
        before = _stable_bytes(path, "packaged candidate result", max_bytes=_MAX_RESULT_BYTES)
        result = load_and_revalidate_result(path, manifest=manifest)
        after = _stable_bytes(path, "packaged candidate result", max_bytes=_MAX_RESULT_BYTES)
        if before != after:
            raise ValueError("packaged candidate result changed during ingestion")
        candidate_id = result.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in loaded:
            raise ValueError(f"duplicate or invalid candidate result: {candidate_id!r}")
        loaded[candidate_id] = result
        records[candidate_id] = _record(path, after, candidate_id=candidate_id)

    declared_main = tuple(item["candidate_id"] for item in manifest["candidates"])
    if declared_main != _MAIN_IDS:
        raise ValueError("manifest main candidate order must be exactly a0 through a4")
    diagnostic_ids = {
        item["candidate_id"] for item in manifest.get("diagnostic_candidates", [])
    }
    unknown = sorted(set(loaded) - set(declared_main) - diagnostic_ids)
    if unknown:
        raise ValueError(f"candidate inventory has unknown results: {unknown}")
    if "a0" not in loaded:
        raise ValueError("A0 packaged result is required as the shortlist baseline")
    baseline = loaded["a0"]
    baseline_profile, baseline_selectable, baseline_kind = _profile(baseline)
    if (baseline_profile, baseline_selectable, baseline_kind) != ("a0", True, "main"):
        raise ValueError("A0 must be the selectable main baseline")
    if any(not _passed(baseline.get("gates", {}).get(name)) for name in _REQUIRED_RESULT_GATES):
        raise ValueError("A0 must pass protocol and T1 evidence gates")
    for name in (*_T2_DIRECTIONS, "current_miou", "object_f1"):
        _metric_value(baseline, name)

    baseline_non_temporal = baseline["bindings"]["non_temporal_config_sha256"]
    baseline_inputs = baseline["bindings"]["input_hashes"]
    ledger: list[dict[str, Any]] = []
    eligible_by_profile: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in _FALLBACK_ORDER
    }
    candidate_order = [*declared_main, *sorted(set(loaded) & diagnostic_ids)]
    for candidate_id in candidate_order:
        result = loaded.get(candidate_id)
        if result is None:
            ledger.append(
                {
                    "candidate_id": candidate_id,
                    "profile": candidate_id,
                    "eligible": False,
                    "shortlisted": False,
                    "reasons": ["candidate_result_unavailable"],
                    "config_sha256": None,
                    "algorithm_hash": None,
                    "result": None,
                }
            )
            continue
        if result["bindings"]["non_temporal_config_sha256"] != baseline_non_temporal:
            raise ValueError(f"candidate {candidate_id} non-temporal config hash differs from A0")
        if result["bindings"]["input_hashes"] != baseline_inputs:
            raise ValueError(f"candidate {candidate_id} input hash bindings differ from A0")
        profile, _, _ = _profile(result)
        reasons = (
            ["baseline_or_nonpromotable_profile"]
            if candidate_id in {"a0", "a1"}
            else _candidate_reasons(result, baseline)
        )
        entry = {
            "candidate_id": candidate_id,
            "profile": profile,
            "eligible": not reasons,
            "shortlisted": False,
            "reasons": reasons,
            "config_sha256": result["bindings"]["config_sha256"],
            "algorithm_hash": result["run_identity"]["algorithm_hash"],
            "result": records[candidate_id],
        }
        ledger.append(entry)
        if not reasons:
            eligible_by_profile[profile].append(result)

    shortlisted: list[dict[str, Any]] = []
    for profile in _FALLBACK_ORDER:
        eligible = eligible_by_profile[profile]
        if not eligible:
            continue
        # All axes above are mandatory. The digest is only a deterministic tie break
        # between equally eligible configurations of the same execution profile.
        selected = min(eligible, key=lambda item: item["bindings"]["config_sha256"])
        shortlisted.append(_shortlist_entry(selected, records[selected["candidate_id"]]))
        for entry in ledger:
            if entry["candidate_id"] == selected["candidate_id"]:
                entry["shortlisted"] = True
            elif entry["eligible"] and entry["profile"] == profile:
                entry["reasons"].append("same_profile_pareto_pruned")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_dual_readout_shortlist_v1",
        "phase": "shortlist",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "protocol_id": "oviv2-tessecd-v2",
        "status": "PASS" if shortlisted else "NO_ELIGIBLE_CANDIDATE",
        "development_scene": "apartment",
        "transfer_scene": "office",
        "office_results_read": False,
        "scenes_read": ["apartment"],
        "result_contract": RESULT_CONTRACT,
        "results_root": str(Path(results_root).absolute()) if results_root else None,
        "manifest": manifest_record,
        "result_files": [records[item] for item in candidate_order if item in records],
        "profile_fallback_order": list(_FALLBACK_ORDER),
        "floors": {
            "current_miou_from_a0": _metric_value(baseline, "current_miou"),
            "object_f1_from_a0": _metric_value(baseline, "object_f1"),
        },
        "shortlisted_candidate_ids": [item["candidate_id"] for item in shortlisted],
        "shortlisted_candidates": shortlisted,
        "rejection_ledger": ledger,
    }

    current_manifest = _stable_bytes(Path(manifest_record["path"]), "search manifest", max_bytes=_MAX_MANIFEST_BYTES)
    if current_manifest != manifest_bytes or load_search_manifest(manifest_record["path"]) != manifest:
        raise ValueError("search manifest changed during shortlist selection")
    for candidate_id, original in loaded.items():
        record = records[candidate_id]
        path = Path(record["path"])
        data = _stable_bytes(path, "packaged candidate result", max_bytes=_MAX_RESULT_BYTES)
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["byte_count"]:
            raise ValueError("packaged candidate result changed during shortlist selection")
        if load_and_revalidate_result(path, manifest=manifest) != original:
            raise ValueError("packaged candidate result changed during shortlist selection")
    _atomic_write_new(destination, payload)
    return payload


def tune_shortlist_results_root(
    manifest_path: str | Path, results_root: str | Path, output: str | Path
) -> dict[str, Any]:
    manifest = load_search_manifest(manifest_path)
    root = Path(results_root).absolute()
    _reject_symlink_components(root, "results root")
    if not root.is_dir():
        raise FileNotFoundError(f"missing results root: {root}")
    paths = []
    for candidate in manifest["candidates"]:
        relative = Path(f"candidates/{candidate['candidate_id']}/apartment/result.json")
        path = root / relative
        if (not path.is_file() or path.is_symlink()) and candidate["candidate_id"] == "a0":
            raise FileNotFoundError(f"missing immutable candidate result: {relative}")
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return tune_shortlist(manifest_path, paths, output, results_root=root)


def _load_shortlist(path: Path) -> tuple[dict[str, Any], bytes]:
    value, data = _load_json(path, "shortlist", max_bytes=_MAX_SELECTION_BYTES)
    expected = {
        "schema_version", "manifest_id", "phase", "dataset", "method_id",
        "protocol_id", "status", "development_scene", "transfer_scene",
        "office_results_read", "scenes_read", "result_contract", "results_root",
        "manifest", "result_files", "profile_fallback_order", "floors",
        "shortlisted_candidate_ids", "shortlisted_candidates", "rejection_ledger",
    }
    _exact_keys(value, expected, "shortlist")
    if (
        value["schema_version"] != 1
        or value["manifest_id"] != "oviv2_tesse_dual_readout_shortlist_v1"
        or value["phase"] != "shortlist"
        or value["profile_fallback_order"] != list(_FALLBACK_ORDER)
        or value["office_results_read"] is not False
        or value["scenes_read"] != ["apartment"]
    ):
        raise ValueError("shortlist protocol is invalid")
    candidates = value["shortlisted_candidates"]
    if not isinstance(candidates, list) or [item.get("candidate_id") for item in candidates] != value["shortlisted_candidate_ids"]:
        raise ValueError("shortlist candidate inventory mismatch")
    if [item.get("profile") for item in candidates] != [
        profile for profile in _FALLBACK_ORDER if profile in {item.get("profile") for item in candidates}
    ]:
        raise ValueError("shortlist fallback order mismatch")
    return value, data


def _load_t4(
    path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], Path, bytes]:
    value, data = _load_json(path, "T4 matrix", max_bytes=_MAX_T4_BYTES)
    _exact_keys(
        value,
        {
            "schema_version", "manifest_id", "status", "shortlist", "protocol",
            "candidates", "root_sha256",
        },
        "T4 matrix",
    )
    if (
        value["schema_version"] != 1
        or value["manifest_id"] != "oviv2_tesse_t4_matrix_v1"
        or value["status"] != "PASS"
    ):
        raise ValueError("T4 matrix protocol is invalid")
    protocol_path, protocol_bytes = _source_binding(value["protocol"], "T4 protocol")
    protocol, reread_protocol = _load_json(
        protocol_path, "T4 protocol", max_bytes=_MAX_SOURCE_BYTES
    )
    if reread_protocol != protocol_bytes:
        raise ValueError("T4 protocol changed during ingestion")
    _exact_keys(
        protocol,
        {
            "schema_version", "manifest_id", "dataset", "method_id",
            "protocol_id", "scene", "bounds", "candidates",
        },
        "T4 protocol",
    )
    if not (
        protocol["schema_version"] == 1
        and protocol["manifest_id"] == "oviv2_tesse_t4_protocol_v1"
        and protocol["dataset"] == "TESSE-CD"
        and protocol["method_id"] == "OVIV2"
        and protocol["protocol_id"] == "oviv2-tessecd-v2"
        and protocol["scene"] == "apartment"
        and protocol["bounds"] == _T4_BOUNDS
        and isinstance(protocol["candidates"], Mapping)
    ):
        raise ValueError("T4 protocol identity or Apartment scope is invalid")
    root = value["root_sha256"]
    if not _is_sha256(root):
        raise ValueError("T4 root_sha256 is invalid")
    unhashed = dict(value)
    unhashed.pop("root_sha256")
    if hashlib.sha256(_canonical_json(unhashed)).hexdigest() != root:
        raise ValueError("T4 matrix root hash is stale")
    return value, data, protocol, protocol_path, protocol_bytes


def tune_final(
    manifest_path: str | Path,
    shortlist_path: str | Path,
    t4_matrix_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest, manifest_record, manifest_bytes = _manifest_and_record(manifest_path)
    shortlist_file = Path(shortlist_path).absolute()
    shortlist, shortlist_bytes = _load_shortlist(shortlist_file)
    if shortlist["manifest"] != manifest_record:
        raise ValueError("shortlist manifest binding mismatch")
    matrix_file = Path(t4_matrix_path).absolute()
    matrix, matrix_bytes, protocol, protocol_path, protocol_bytes = _load_t4(
        matrix_file
    )
    bound_shortlist_path, bound_shortlist_bytes = _source_binding(
        matrix["shortlist"], "T4 shortlist"
    )
    if bound_shortlist_path != shortlist_file or bound_shortlist_bytes != shortlist_bytes:
        raise ValueError("shortlist source binding mismatch")

    rows = matrix["candidates"]
    if not isinstance(rows, Mapping) or set(rows) != set(shortlist["shortlisted_candidate_ids"]):
        raise ValueError("T4 candidate matrix must exactly cover the shortlist")
    protocol_candidates = protocol["candidates"]
    if set(protocol_candidates) != set(rows):
        raise ValueError("T4 protocol must exactly cover the shortlisted candidates")
    shortlisted = {
        item["candidate_id"]: item for item in shortlist["shortlisted_candidates"]
    }
    t4_ledger: list[dict[str, Any]] = []
    selected_id: str | None = None
    seen_run_manifests: set[str] = set()
    source_witnesses: list[tuple[Path, bytes, str]] = []
    for candidate_id in shortlist["shortlisted_candidate_ids"]:
        row = _exact_keys(
            rows[candidate_id],
            {"status", "config_sha256", "run_manifest_sha256", "metrics", "gates"},
            f"T4 candidate {candidate_id}",
        )
        if row["config_sha256"] != shortlisted[candidate_id]["config_sha256"]:
            raise ValueError(f"T4 config binding mismatch for {candidate_id}")
        run_hash = row["run_manifest_sha256"]
        if not _is_sha256(run_hash):
            raise ValueError(f"T4 run manifest hash is invalid for {candidate_id}")
        if run_hash in seen_run_manifests:
            raise ValueError("T4 candidates must use a single whole-profile source each")
        seen_run_manifests.add(run_hash)
        protocol_row = _exact_keys(
            protocol_candidates[candidate_id],
            {"config_sha256", "run_manifest", "metric_sources", "raw_sources"},
            f"T4 protocol candidate {candidate_id}",
        )
        if protocol_row["config_sha256"] != row["config_sha256"]:
            raise ValueError(f"T4 protocol config binding mismatch for {candidate_id}")
        run_path, run_bytes = _source_binding(
            protocol_row["run_manifest"], f"T4 run manifest {candidate_id}"
        )
        if hashlib.sha256(run_bytes).hexdigest() != run_hash:
            raise ValueError(
                f"T4 candidate {candidate_id} does not bind one whole-profile run"
            )
        run_manifest, reread_run = _load_json(
            run_path, f"T4 run manifest {candidate_id}", max_bytes=_MAX_SOURCE_BYTES
        )
        if reread_run != run_bytes or not (
            run_manifest.get("dataset") == "TESSE-CD"
            and run_manifest.get("method_id") == "OVIV2"
            and run_manifest.get("protocol_id") == "oviv2-tessecd-v2"
            and run_manifest.get("scene") == "apartment"
            and run_manifest.get("candidate_id") == candidate_id
            and run_manifest.get("config_sha256") == row["config_sha256"]
        ):
            raise ValueError(f"T4 run manifest scope mismatch for {candidate_id}")
        source_witnesses.append((run_path, run_bytes, f"T4 run manifest {candidate_id}"))
        raw_sources = _exact_keys(
            protocol_row["raw_sources"],
            _T4_RAW_SOURCE_NAMES | {f"{name}_sha256" for name in _T4_RAW_SOURCE_NAMES},
            f"T4 raw sources {candidate_id}",
        )
        if raw_sources["run_manifest"] != protocol_row["run_manifest"]:
            raise ValueError(f"T4 raw run manifest binding mismatch for {candidate_id}")
        if raw_sources["shortlist"] != matrix["shortlist"]:
            raise ValueError(f"T4 raw shortlist binding mismatch for {candidate_id}")
        for raw_name in sorted(_T4_RAW_SOURCE_NAMES):
            raw_path, raw_bytes = _source_binding(
                raw_sources[raw_name], f"T4 raw {candidate_id} {raw_name}"
            )
            if raw_sources[f"{raw_name}_sha256"] != raw_sources[raw_name]["sha256"]:
                raise ValueError(f"T4 raw {raw_name} flat hash mismatch for {candidate_id}")
            source_witnesses.append(
                (raw_path, raw_bytes, f"T4 raw {candidate_id} {raw_name}")
            )
        metrics = _exact_keys(row["metrics"], set(_T4_BOUNDS), f"T4 metrics {candidate_id}")
        gates = _exact_keys(row["gates"], set(_T4_BOUNDS), f"T4 gates {candidate_id}")
        metric_sources = _exact_keys(
            protocol_row["metric_sources"],
            set(_T4_BOUNDS),
            f"T4 metric sources {candidate_id}",
        )
        failures = []
        for name, bound in _T4_BOUNDS.items():
            value = _finite_number(metrics[name], f"T4 {candidate_id} {name}")
            metric_path, metric_bytes = _source_binding(
                metric_sources[name], f"T4 metric {candidate_id} {name}"
            )
            metric_payload, reread_metric = _load_json(
                metric_path,
                f"T4 metric {candidate_id} {name}",
                max_bytes=_MAX_SOURCE_BYTES,
            )
            _exact_keys(
                metric_payload,
                {
                    "schema_version", "manifest_id", "scene", "candidate_id",
                    "config_sha256", "run_manifest_sha256", "metric", "value",
                },
                f"T4 metric {candidate_id} {name}",
            )
            if reread_metric != metric_bytes or not (
                metric_payload["schema_version"] == 1
                and metric_payload["manifest_id"] == "oviv2_tesse_t4_metric_v1"
                and metric_payload["scene"] == "apartment"
                and metric_payload["candidate_id"] == candidate_id
                and metric_payload["config_sha256"] == row["config_sha256"]
                and metric_payload["run_manifest_sha256"] == run_hash
                and metric_payload["metric"] == name
                and _finite_number(metric_payload["value"], f"T4 source {name}")
                == value
            ):
                raise ValueError(
                    f"T4 metric {candidate_id} {name} is not whole-profile source-bound"
                )
            source_witnesses.append(
                (metric_path, metric_bytes, f"T4 metric {candidate_id} {name}")
            )
            actual_pass = value <= bound
            if not isinstance(gates[name], bool) or gates[name] is not actual_pass:
                raise ValueError(f"T4 gate does not match bound for {candidate_id} {name}")
            if not actual_pass:
                failures.append(name)
        expected_status = "PASS" if not failures else "FAIL"
        if row["status"] != expected_status:
            raise ValueError(f"T4 candidate status mismatch for {candidate_id}")
        t4_ledger.append(
            {
                "candidate_id": candidate_id,
                "profile": shortlisted[candidate_id]["profile"],
                "passed": not failures,
                "selected": False,
                "failed_gates": failures,
                "config_sha256": row["config_sha256"],
                "run_manifest_sha256": run_hash,
            }
        )
        if selected_id is None and not failures:
            selected_id = candidate_id
    for item in t4_ledger:
        item["selected"] = item["candidate_id"] == selected_id

    selected = shortlisted.get(selected_id) if selected_id else None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_dual_readout_selection_v2",
        "phase": "final",
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "protocol_id": "oviv2-tessecd-v2",
        "status": "PASS" if selected else "NO_T4_ELIGIBLE_CANDIDATE",
        "development_scene": "apartment",
        "transfer_scene": "office",
        "office_results_read": False,
        "scenes_read": ["apartment"],
        "manifest": manifest_record,
        "shortlist": _record(shortlist_file, shortlist_bytes),
        "t4_matrix": _record(matrix_file, matrix_bytes),
        "t4_protocol": dict(matrix["protocol"]),
        "t4_root_sha256": matrix["root_sha256"],
        "profile_fallback_order": list(_FALLBACK_ORDER),
        "selected_candidate_id": selected_id,
        "selected_config": selected["selected_config"] if selected else None,
        "selected_config_record": selected["selected_config_record"] if selected else None,
        "selected_config_sha256": selected["config_sha256"] if selected else None,
        "algorithm_hash": selected["algorithm_hash"] if selected else None,
        "t4_ledger": t4_ledger,
    }

    if _stable_bytes(Path(manifest_record["path"]), "search manifest", max_bytes=_MAX_MANIFEST_BYTES) != manifest_bytes or load_search_manifest(manifest_record["path"]) != manifest:
        raise ValueError("search manifest changed during final selection")
    if _stable_bytes(shortlist_file, "shortlist", max_bytes=_MAX_SELECTION_BYTES) != shortlist_bytes:
        raise ValueError("shortlist changed during final selection")
    if _stable_bytes(matrix_file, "T4 matrix", max_bytes=_MAX_T4_BYTES) != matrix_bytes:
        raise ValueError("T4 matrix changed during final selection")
    if _stable_bytes(protocol_path, "T4 protocol", max_bytes=_MAX_SOURCE_BYTES) != protocol_bytes:
        raise ValueError("T4 protocol changed during final selection")
    for source_path, source_bytes, label in source_witnesses:
        if _stable_bytes(source_path, label, max_bytes=_MAX_SOURCE_BYTES) != source_bytes:
            raise ValueError(f"{label} changed during final selection")
    _atomic_write_new(destination, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("shortlist", "final"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--t4-matrix", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.phase == "shortlist" and args.results_root is None:
        parser.error("--results-root is required for --phase shortlist")
    if args.phase == "shortlist" and (args.shortlist is not None or args.t4_matrix is not None):
        parser.error("--shortlist/--t4-matrix are only valid for --phase final")
    if args.phase == "final" and (args.shortlist is None or args.t4_matrix is None):
        parser.error("--shortlist and --t4-matrix are required for --phase final")
    if args.phase == "final" and args.results_root is not None:
        parser.error("--results-root is only valid for --phase shortlist")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "shortlist":
        result = tune_shortlist_results_root(args.manifest, args.results_root, args.output)
    else:
        result = tune_final(args.manifest, args.shortlist, args.t4_matrix, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "phase": args.phase,
                "selected_candidate_id": result.get("selected_candidate_id"),
                "shortlisted_candidate_ids": result.get("shortlisted_candidate_ids"),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
