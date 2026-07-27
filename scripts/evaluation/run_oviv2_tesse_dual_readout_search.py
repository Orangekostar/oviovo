#!/usr/bin/env python3
"""Run the predeclared OVIV2 dual-readout ablations on Apartment only."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
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
from src.evaluation.oviv2_runtime_diagnostics import (  # noqa: E402
    validate_runtime_diagnostics,
)


CommandBuilder = Callable[[Path, Path, str], tuple[str, ...]]
_CANDIDATE_IDS = ("a0", "a1", "a2", "a3", "a4")
_GPU_LANES = {
    "lane0": ["a0", "a1"],
    "lane1": ["a2"],
    "lane2": ["a3", "a4"],
}
_LANE_BY_CANDIDATE = {
    candidate_id: lane
    for lane, candidate_ids in enumerate(_GPU_LANES.values())
    for candidate_id in candidate_ids
}
_PARAMETER_SPACE_PROFILES = {
    "lifecycle.minimum_absent_streak": ("a1", "a2", "a3", "a4"),
    "proposal.minimum_depth_residual_m": ("a2", "a3", "a4"),
    "dynamic_state.minimum_motion_confidence": ("a2", "a3", "a4"),
    "identity.minimum_reid_similarity": ("a4",),
    "identity.maximum_reid_distance_m": ("a4",),
    "motion.minimum_translation_confidence": ("a2", "a3", "a4"),
    "motion.maximum_translation_residual_m": ("a2", "a3", "a4"),
    "background_ledger.commit_support_frames": ("a3", "a4"),
    "background_ledger.commit_distinct_view_bins": ("a3", "a4"),
}
_MECHANISMS_BY_PROFILE = {
    "a0": (),
    "a1": ("absence", "readout_invalidation"),
    "a2": (
        "absence",
        "readout_invalidation",
        "proposal_recovery",
        "epoch_reset",
        "motion_rejection",
    ),
    "a3": (
        "absence",
        "readout_invalidation",
        "proposal_recovery",
        "epoch_reset",
        "motion_rejection",
        "background_release",
        "background_reclaim",
    ),
    "a4": (
        "absence",
        "readout_invalidation",
        "proposal_recovery",
        "epoch_reset",
        "motion_rejection",
        "background_release",
        "background_reclaim",
        "eligible_reid",
        "icp",
    ),
}
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
_METRIC_GATES = {
    "apartment_anchor_coverage": {
        "minimum_mapped_anchors": 53,
        "eligible_anchors": 66,
    },
    "non_inferiority": ["object_f1", "current_miou"],
    "strict_improvement": [
        "dynamic_f1",
        "change_f1",
        "ghost_rate",
        "background_f5_cm",
        "recovery_frames",
    ],
    "weighted_compensation_allowed": False,
}
_T4_BOUNDS = {
    "total_runtime_s_per_frame": 6.42,
    "query_mean_ms": 11.92,
    "query_p95_ms": 12.12,
    "peak_gpu_gb": 12.76,
    "peak_ram_gb": 9.36,
    "final_map_mb": 46.77,
}
_LAUNCH_GATES = {
    "preflight_manifest_id": "oviv2_dual_readout_search_preflight_v1",
    "short_prefix_required": True,
    "required_evidence": [
        "frame_coverage",
        "future_leakage",
        "anchor_coverage",
        "mechanisms",
    ],
    "future_leakage_max_count": 0,
}
_DIAGNOSTIC_CANDIDATES = [
    {
        "candidate_id": "diag_a2_no_proposal_recovery",
        "base_profile": "a2",
        "ablation": "without_proposal_recovery",
        "diagnostic": True,
        "selectable": False,
        "runnable": True,
        "lane": "lane1",
        "diagnostic_controls": {"proposal_recovery_enabled": False},
    },
    {
        "candidate_id": "diag_a3_masking_only_no_ledger",
        "base_profile": "a3",
        "ablation": "masking_only_without_reversible_ledger",
        "diagnostic": True,
        "selectable": False,
        "runnable": True,
        "lane": "lane2",
        "diagnostic_controls": {"background_mode": "masking_only"},
    },
    {
        "candidate_id": "diag_a4_no_dormant_candidates",
        "base_profile": "a4",
        "ablation": "without_dormant_candidates",
        "diagnostic": True,
        "selectable": False,
        "runnable": True,
        "lane": "lane2",
        "diagnostic_controls": {"dormant_reid_enabled": False},
    },
    {
        "candidate_id": "diag_a4_translation_only_no_icp",
        "base_profile": "a4",
        "ablation": "translation_only_without_icp",
        "diagnostic": True,
        "selectable": False,
        "runnable": True,
        "lane": "lane2",
        "diagnostic_controls": {"icp_enabled": False},
    },
]
_DISABLED_MECHANISMS_BY_DIAGNOSTIC = {
    "diag_a2_no_proposal_recovery": {"proposal_recovery"},
    "diag_a3_masking_only_no_ledger": {"background_release", "background_reclaim"},
    "diag_a4_no_dormant_candidates": {"eligible_reid"},
    "diag_a4_translation_only_no_icp": {"icp"},
}
_INPUT_BINDING_FIELDS = (
    "dense_manifest",
    "evaluation_checkpoint_frames_sha256",
    "export_manifest",
    "frontend_manifest",
    "input_manifest",
    "occlusion_target_manifest_sha256",
    "schedule_manifest",
)
_SOURCE_EVIDENCE_ROLES = (
    "run_manifest",
    "source_index",
    "trajectories",
    "lifecycle_transitions",
    "frame_coverage",
    "runtime_diagnostics",
    "temporal_occlusion_result",
    "future_leakage_evidence",
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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _EvidenceWitness:
    path: Path
    data: bytes
    identity: tuple[int, int, int, int, int]

    @property
    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "byte_count": len(self.data),
        }

    def revalidate(self, label: str) -> None:
        try:
            current = _read_evidence_witness(self.path, label)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} changed before launch") from exc
        if current.identity != self.identity or current.record != self.record:
            raise ValueError(f"{label} changed before launch")


class _PreflightRecord(dict[str, Any]):
    def __init__(self, value: Mapping[str, Any], witnesses: Sequence[_EvidenceWitness]):
        super().__init__(value)
        self.witnesses = tuple(witnesses)


def _read_evidence_witness(path: Path, label: str) -> _EvidenceWitness:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(current)
    ):
        raise ValueError(f"{label} changed while it was read")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return _EvidenceWitness(path, data, _file_identity(after))


def _source_witness(
    record: object, label: str, *, preflight_parent: Path
) -> _EvidenceWitness:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} source record must be an object")
    _exact_keys(record, {"path", "sha256", "byte_count"}, f"{label} source record")
    raw_path = record.get("path")
    relative = Path(raw_path) if isinstance(raw_path, str) else Path(".")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or raw_path != relative.as_posix()
    ):
        raise ValueError(f"{label} source path is an alias")
    path = (preflight_parent / relative).absolute()
    try:
        path.relative_to(preflight_parent.absolute())
    except ValueError as exc:
        raise ValueError(f"{label} source path escapes preflight bundle") from exc
    witness = _read_evidence_witness(path, label)
    if (
        record.get("sha256") != witness.record["sha256"]
        or record.get("byte_count") != witness.record["byte_count"]
    ):
        raise ValueError(f"{label} source binding mismatch")
    return witness


def _jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(_decode_json(line, Path(f"{label} line {line_number}")))
    return rows


def _require_nested_source_record(
    record: object,
    *,
    base: Path,
    witness: _EvidenceWitness,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} source record must be an object")
    _exact_keys(record, {"path", "sha256", "byte_count"}, f"{label} source record")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} source path is invalid")
    relative = Path(raw)
    if (
        raw != str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"{label} source path is an alias")
    path = (base / relative).absolute()
    if path != witness.path or record.get("sha256") != witness.record["sha256"] or record.get(
        "byte_count"
    ) != witness.record["byte_count"]:
        raise ValueError(f"{label} source binding mismatch")


def _revalidate_preflight_witnesses(record: _PreflightRecord) -> None:
    for witness in record.witnesses:
        witness.revalidate(f"preflight witness {witness.path}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _profile_components(profile: ExecutionProfile) -> dict[str, str]:
    return profile.components


def _validate_parameter_spaces(spaces: object) -> dict[str, dict[str, Any]]:
    if not isinstance(spaces, dict) or set(spaces) != set(_PARAMETER_SPACE_PROFILES):
        unknown = (
            sorted(set(spaces) - set(_PARAMETER_SPACE_PROFILES))
            if isinstance(spaces, dict)
            else []
        )
        missing = (
            sorted(set(_PARAMETER_SPACE_PROFILES) - set(spaces))
            if isinstance(spaces, dict)
            else []
        )
        raise ValueError(
            f"parameter_spaces keys mismatch; missing={missing}, unknown={unknown}"
        )
    for name, space in spaces.items():
        if not isinstance(space, dict):
            raise ValueError(f"parameter_spaces entry must be an object: {name}")
        _exact_keys(space, {"profiles", "values"}, f"parameter_spaces.{name}")
        expected_profiles = list(_PARAMETER_SPACE_PROFILES[name])
        if space["profiles"] != expected_profiles:
            raise ValueError(f"parameter_spaces field is profile-incompatible: {name}")
        values = space["values"]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 3
            or len({_canonical_json(value) for value in values}) != len(values)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError(
                f"parameter_spaces values are not narrow finite discrete values: {name}"
            )
    return spaces


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
            "parameter_spaces",
            "metric_directions",
            "metric_policy",
            "metric_gates",
            "t4_bounds",
            "profile_fallback_order",
            "promotion_order",
            "hard_gates",
            "launch_gates",
            "gpu_lanes",
            "diagnostic_candidates",
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
    if payload["metric_gates"] != _METRIC_GATES:
        raise ValueError("metric_gates do not match the frozen contract")
    if payload["t4_bounds"] != _T4_BOUNDS:
        raise ValueError("t4_bounds do not match the frozen contract")
    if payload["profile_fallback_order"] != ["a4", "a3", "a2"]:
        raise ValueError("profile_fallback_order does not match the frozen contract")
    if payload["launch_gates"] != _LAUNCH_GATES:
        raise ValueError("launch_gates do not match the frozen contract")
    if payload["gpu_lanes"] != _GPU_LANES:
        raise ValueError("gpu_lanes do not match the frozen contract")
    if payload["diagnostic_candidates"] != _DIAGNOSTIC_CANDIDATES:
        raise ValueError("diagnostic_candidates do not match the preregistered contract")
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
    spaces = _validate_parameter_spaces(payload["parameter_spaces"])
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or [
        item.get("candidate_id") if isinstance(item, dict) else None
        for item in candidates
    ] != list(_CANDIDATE_IDS):
        raise ValueError("candidates must predeclare exactly a0 through a4 in order")
    declarations: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        _exact_keys(
            candidate,
            {"candidate_id", "components", "temporal_readout"},
            f"candidate {candidate['candidate_id']}",
        )
        profile = ExecutionProfile.from_id(candidate["candidate_id"])
        declarations[profile.profile_id] = candidate
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
    for diagnostic in payload["diagnostic_candidates"]:
        base = declarations[diagnostic["base_profile"]]
        temporal = copy.deepcopy(base["temporal_readout"])
        temporal["diagnostic_controls"] = diagnostic["diagnostic_controls"]
        parsed = temporal_config_from_json({"temporal_readout": temporal})
        if parsed.diagnostic_identity != diagnostic["candidate_id"]:
            raise ValueError("diagnostic candidate identity/control mismatch")
    for field_name, space in spaces.items():
        group, name = field_name.split(".", 1)
        for profile_id in space["profiles"]:
            declaration = declarations[profile_id]
            value = declaration["temporal_readout"][group][name]
            if value not in space["values"]:
                raise ValueError(
                    f"candidate parameter is outside discrete space: {profile_id}.{field_name}"
                )
            for alternative in space["values"]:
                temporal = copy.deepcopy(declaration["temporal_readout"])
                temporal[group][name] = alternative
                try:
                    temporal_config_from_json({"temporal_readout": temporal})
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"parameter_spaces value is not a legal production config: "
                        f"{profile_id}.{field_name}={alternative!r}"
                    ) from exc
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
    temporal = copy.deepcopy(candidate["temporal_readout"])
    if "diagnostic_controls" in candidate:
        temporal["diagnostic_controls"] = copy.deepcopy(
            candidate["diagnostic_controls"]
        )
    config["temporal_readout"] = temporal
    config["algorithm_hash"] = canonical_algorithm_hash(config)
    temporal_config_from_json({"temporal_readout": config["temporal_readout"]})
    if _non_temporal_algorithm(config) != _non_temporal_algorithm(base):
        raise ValueError("candidate changed non-temporal algorithm fields")
    if input_binding_values_sha256(config) != input_binding_values_sha256(base):
        raise ValueError("candidate changed input binding values")
    return config


def _exact_integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        or len(value) != len(set(value))
        or value != sorted(value)
    ):
        raise ValueError(f"{label} must be unique sorted nonnegative integers")
    return value


def _validate_mechanism_record(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_keys(
        value,
        {"opportunity_count", "trigger_count", "opportunity_records", "trigger_records"},
        label,
    )
    for count_name, records_name in (
        ("opportunity_count", "opportunity_records"),
        ("trigger_count", "trigger_records"),
    ):
        count = value[count_name]
        records = value[records_name]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(records, list)
            or len(records) != count
            or len(records) != len(set(records))
            or any(not isinstance(item, str) or not item for item in records)
        ):
            raise ValueError(f"{label}.{count_name} does not match nonempty records")
    if value["trigger_count"] > value["opportunity_count"]:
        raise ValueError(f"{label}.trigger_count exceeds opportunity_count")


def _validate_runtime_mechanism_records(
    runtime: Mapping[str, Any],
    label: str,
    *,
    expected_temporal_readout: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
) -> None:
    validate_runtime_diagnostics(
        runtime,
        label=label,
        expected_temporal_readout=expected_temporal_readout,
        expected_candidate_id=expected_candidate_id,
    )


def _diagnostic_recompute_payloads(
    payloads: Mapping[str, Any],
    disabled: set[str],
    label: str,
    *,
    expected_temporal_readout: Mapping[str, Any] | None = None,
    expected_candidate_id: str | None = None,
) -> dict[str, Any]:
    runtime = payloads["runtime_diagnostics"]
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{label} runtime diagnostics are invalid")
    _validate_runtime_mechanism_records(
        runtime,
        label,
        expected_temporal_readout=expected_temporal_readout,
        expected_candidate_id=expected_candidate_id,
    )
    disabled_counters = {
        "proposal_recovery": {
            "proposal_opportunity_count",
            "proposal_trigger_count",
        },
        "background_release": {"ledger_stage_count", "ledger_commit_count"},
        "background_reclaim": {"ledger_commit_count", "ledger_reclaim_count"},
        "eligible_reid": {"reid_opportunity_count", "reid_trigger_count"},
        "icp": {
            "icp_opportunity_count",
            "icp_accept_count",
            "icp_reject_count",
        },
        "motion_rejection": {"motion_rejection_count"},
    }
    names = set().union(*(disabled_counters[name] for name in disabled))
    counters = runtime["counters"]
    records = runtime["mechanism_records"]
    if any(counters[name] != 0 or records[name] != [] for name in names):
        raise ValueError(f"{label} disabled mechanism counters/records must be zero")

    return copy.deepcopy(dict(payloads))


def _validate_preflight_gate_evidence(
    path: Path,
    *,
    manifest_bytes: bytes,
    apartment_bytes: bytes,
    apartment: Mapping[str, Any],
    declarations: Mapping[str, Mapping[str, Any]],
    selected_ids: tuple[str, ...],
) -> dict[str, Any]:
    from scripts.evaluation.build_oviv2_tesse_search_preflight import (
        _anchor_coverage as recompute_anchor_coverage,
        _mechanisms as recompute_mechanisms,
    )

    evidence_witness = _read_evidence_witness(
        path.absolute(), "preflight gate evidence"
    )
    evidence_bytes = evidence_witness.data
    evidence = _decode_json(evidence_bytes, evidence_witness.path)
    _exact_keys(
        evidence,
        {"schema_version", "manifest_id", "scene", "bindings", "candidates"},
        "preflight gate evidence",
    )
    if (
        evidence["schema_version"] != 1
        or evidence["manifest_id"] != _LAUNCH_GATES["preflight_manifest_id"]
        or evidence["scene"] != "apartment"
    ):
        raise ValueError("preflight gate evidence identity mismatch")
    bindings = evidence["bindings"]
    if not isinstance(bindings, dict):
        raise ValueError("preflight gate evidence bindings must be an object")
    _exact_keys(
        bindings,
        {"search_manifest_sha256", "apartment_base_config_file_sha256"},
        "preflight gate evidence bindings",
    )
    expected_bindings = {
        "search_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "apartment_base_config_file_sha256": hashlib.sha256(apartment_bytes).hexdigest(),
    }
    if bindings != expected_bindings:
        raise ValueError("preflight gate evidence source bindings mismatch")
    candidates = evidence["candidates"]
    if not isinstance(candidates, list) or [
        item.get("candidate_id") if isinstance(item, dict) else None for item in candidates
    ] != list(selected_ids):
        raise ValueError("preflight gate evidence candidate order mismatch")
    witnesses: list[_EvidenceWitness] = [evidence_witness]
    inventory_inodes: set[tuple[int, int]] = set()
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        _exact_keys(
            candidate,
            {
                "candidate_id",
                "candidate_config_sha256",
                "frame_coverage",
                "future_leakage",
                "anchor_coverage",
                "mechanisms",
                "source_evidence",
            },
            f"preflight candidate {candidate_id}",
        )
        materialized = _materialize_config(apartment, declarations[candidate_id])
        expected_config_sha256 = hashlib.sha256(_canonical_json(materialized)).hexdigest()
        if candidate["candidate_config_sha256"] != expected_config_sha256:
            raise ValueError(f"preflight candidate {candidate_id} config binding mismatch")
        declaration = declarations[candidate_id]
        base_profile = declaration.get("base_profile", candidate_id)
        source_evidence = candidate["source_evidence"]
        if not isinstance(source_evidence, Mapping):
            raise ValueError(f"preflight candidate {candidate_id} source_evidence is invalid")
        _exact_keys(
            source_evidence,
            set(_SOURCE_EVIDENCE_ROLES),
            f"preflight candidate {candidate_id} source_evidence",
        )
        source_witnesses = {
            role: _source_witness(
                source_evidence[role],
                f"preflight candidate {candidate_id} {role}",
                preflight_parent=evidence_witness.path.parent,
            )
            for role in _SOURCE_EVIDENCE_ROLES
        }
        source_root = source_witnesses["run_manifest"].path.parent
        for role, witness in source_witnesses.items():
            try:
                witness.path.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(
                    f"preflight candidate {candidate_id} {role} escapes source root"
                ) from exc
            inode = witness.identity[:2]
            if inode in inventory_inodes:
                raise ValueError("preflight source evidence roles alias the same inode")
            inventory_inodes.add(inode)
            witnesses.append(witness)

        run = _decode_json(
            source_witnesses["run_manifest"].data,
            source_witnesses["run_manifest"].path,
        )
        if not (
            run.get("dataset") == "TESSE-CD"
            and run.get("protocol_id") == "oviv2-tessecd-v2"
            and run.get("scene") == "apartment"
            and run.get("method_id") == "OVIV2"
            and run.get("algorithm_hash") == materialized["algorithm_hash"]
        ):
            raise ValueError(f"preflight candidate {candidate_id} run identity mismatch")
        _require_nested_source_record(
            run.get("source_index"),
            base=source_witnesses["run_manifest"].path.parent,
            witness=source_witnesses["source_index"],
            label=f"preflight candidate {candidate_id} source_index",
        )
        source_index = _decode_json(
            source_witnesses["source_index"].data,
            source_witnesses["source_index"].path,
        )
        if not (
            source_index.get("dataset") == "TESSE-CD"
            and source_index.get("method") == "OVIV2"
            and source_index.get("scene") == "apartment"
        ):
            raise ValueError(f"preflight candidate {candidate_id} source_index identity mismatch")
        for role in (
            "trajectories",
            "lifecycle_transitions",
            "frame_coverage",
            "runtime_diagnostics",
        ):
            _require_nested_source_record(
                source_index.get(role),
                base=source_witnesses["source_index"].path.parent,
                witness=source_witnesses[role],
                label=f"preflight candidate {candidate_id} {role}",
            )

        payloads: dict[str, Any] = {
            role: _jsonl(
                source_witnesses[role].data,
                f"preflight candidate {candidate_id} {role}",
            )
            for role in ("trajectories", "lifecycle_transitions", "frame_coverage")
        }
        payloads["runtime_diagnostics"] = _decode_json(
            source_witnesses["runtime_diagnostics"].data,
            source_witnesses["runtime_diagnostics"].path,
        )
        if payloads["runtime_diagnostics"].get("execution_profile") != base_profile:
            raise ValueError(f"preflight candidate {candidate_id} profile binding mismatch")
        processed_frame_count = run.get("processed_frame_count")
        expected_frames = (
            list(range(processed_frame_count))
            if type(processed_frame_count) is int and processed_frame_count > 0
            else None
        )
        observed_frames = [row.get("frame_index") for row in payloads["frame_coverage"]]
        if not (
            isinstance(expected_frames, list)
            and expected_frames
            and observed_frames == expected_frames
            and run.get("covered_frame_count") == processed_frame_count
            and run.get("first_frame_index") == 0
            and run.get("last_frame_index") == processed_frame_count - 1
        ):
            raise ValueError(
                f"preflight candidate {candidate_id} raw frame_coverage did not PASS"
            )
        recomputed_coverage = {
            "expected_frame_indices": expected_frames,
            "observed_frame_indices": observed_frames,
            "expected_count": len(expected_frames),
            "observed_count": len(observed_frames),
        }
        if candidate["frame_coverage"] != recomputed_coverage:
            raise ValueError(
                f"preflight candidate {candidate_id} frame_coverage differs from raw source"
            )

        occlusion = _decode_json(
            source_witnesses["temporal_occlusion_result"].data,
            source_witnesses["temporal_occlusion_result"].path,
        )
        bindings = occlusion.get("input_bindings")
        bound_indexes = bindings.get("source_indexes") if isinstance(bindings, Mapping) else None
        if bound_indexes != [{"scene": "apartment", **run["source_index"]}]:
            raise ValueError(
                f"preflight candidate {candidate_id} occlusion source_index binding mismatch"
            )
        recomputed_anchor = recompute_anchor_coverage(occlusion)
        if candidate["anchor_coverage"] != recomputed_anchor:
            raise ValueError(
                f"preflight candidate {candidate_id} anchor coverage differs from "
                "raw source; 53/66 required"
            )

        leakage = _decode_json(
            source_witnesses["future_leakage_evidence"].data,
            source_witnesses["future_leakage_evidence"].path,
        )
        if set(leakage) != {
            "schema_version",
            "manifest_id",
            "scene",
            "candidate_id",
            "source_index",
            "records",
        } or not (
            leakage.get("schema_version") == 1
            and leakage.get("manifest_id") == "oviv2_tesse_future_leakage_evidence_v1"
            and leakage.get("scene") == "apartment"
            and leakage.get("candidate_id") == candidate_id
            and leakage.get("source_index") == run.get("source_index")
            and isinstance(leakage.get("records"), list)
        ):
            raise ValueError(f"preflight candidate {candidate_id} future_leakage source is invalid")
        recomputed_leakage = {
            "count": len(leakage["records"]),
            "records": leakage["records"],
        }
        if candidate["future_leakage"] != recomputed_leakage or leakage["records"]:
            raise ValueError(
                f"preflight candidate {candidate_id} future_leakage must PASS with zero raw records"
            )

        source_records = {
            role: source_index[role]
            for role in (
                "trajectories",
                "lifecycle_transitions",
                "frame_coverage",
                "runtime_diagnostics",
            )
        }
        disabled_mechanisms = set(
            _DISABLED_MECHANISMS_BY_DIAGNOSTIC.get(candidate_id, ())
        )
        recompute_payloads = (
            _diagnostic_recompute_payloads(
                payloads,
                disabled_mechanisms,
                f"preflight candidate {candidate_id}",
                expected_temporal_readout=materialized["temporal_readout"],
                expected_candidate_id=candidate_id,
            )
            if disabled_mechanisms
            else payloads
        )
        recomputed_mechanisms = recompute_mechanisms(
            candidate_id=base_profile,
            source_records=source_records,
            payloads=recompute_payloads,
            expected_temporal_readout=materialized["temporal_readout"],
            expected_candidate_id=candidate_id,
            disabled_mechanisms=disabled_mechanisms,
        )
        if candidate["mechanisms"] != recomputed_mechanisms:
            raise ValueError(
                f"preflight candidate {candidate_id} mechanism records/counts "
                "differ from raw source"
            )
    return _PreflightRecord({
        "path": str(evidence_witness.path),
        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "byte_count": len(evidence_bytes),
    }, witnesses)


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


def _validate_gpu_ids(gpu_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gpu_ids, (str, bytes)):
        raise ValueError("gpu_ids must be a sequence of individual GPU identifiers")
    values = tuple(gpu_ids)
    if (
        not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value or "," in value for value in values)
    ):
        raise ValueError("gpu_ids must be unique non-empty identifiers without commas")
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
    scene: str = "apartment",
    office_freeze_authorization: str | Path | None = None,
    preflight_gate_evidence: str | Path | None = None,
    available_ram_bytes: int | None = None,
    command_builder: CommandBuilder | None = None,
) -> Path:
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int):
        raise ValueError("max_parallel must be an integer")
    if not 1 <= max_parallel <= 3:
        raise ValueError("max_parallel must be in 1..3")
    if scene != "apartment":
        if scene == "office" and office_freeze_authorization is None:
            raise ValueError("Office search requires frozen authorization")
        if scene == "office":
            raise ValueError("Office search is not supported by the development search runner")
        raise ValueError(f"search scene is unsupported: {scene!r}")
    if office_freeze_authorization is not None:
        raise ValueError("Office freeze authorization is invalid for Apartment development")
    gpus = _validate_gpu_ids(gpu_ids)
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

    main_declared = {item["candidate_id"]: item for item in manifest["candidates"]}
    selected_ids = tuple(candidate_ids) if candidate_ids is not None else _CANDIDATE_IDS
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("candidate_ids contain duplicates")
    diagnostics = {
        item["candidate_id"]: item for item in manifest["diagnostic_candidates"]
    }
    requested_diagnostics = tuple(name for name in selected_ids if name in diagnostics)
    requested_main = tuple(name for name in selected_ids if name in main_declared)
    if requested_diagnostics and requested_main:
        raise ValueError("main and diagnostic candidates cannot run in one search")
    undeclared = [
        name for name in selected_ids
        if name not in main_declared and name not in diagnostics
    ]
    if undeclared:
        raise ValueError(f"undeclared candidate: {undeclared[0]}")
    if not selected_ids:
        raise ValueError("candidate_ids cannot be empty")
    diagnostic_ids = tuple(item["candidate_id"] for item in _DIAGNOSTIC_CANDIDATES)
    canonical_order = diagnostic_ids if requested_diagnostics else _CANDIDATE_IDS
    if selected_ids != tuple(name for name in canonical_order if name in selected_ids):
        raise ValueError("candidate_ids must preserve canonical candidate order")
    declared = dict(main_declared)
    for candidate_id, diagnostic in diagnostics.items():
        declared[candidate_id] = {
            **main_declared[diagnostic["base_profile"]],
            **diagnostic,
        }
    lane_by_candidate = dict(_LANE_BY_CANDIDATE)
    lane_by_candidate.update(
        {
            item["candidate_id"]: int(item["lane"].removeprefix("lane"))
            for item in diagnostics.values()
        }
    )
    required_lane = max(lane_by_candidate[name] for name in selected_ids)
    if len(gpus) <= required_lane:
        raise ValueError(
            f"gpu_ids must bind fixed lane {required_lane} for selected candidates"
        )

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

    if preflight_gate_evidence is None:
        raise ValueError("preflight gate evidence is required before full launch")
    preflight_record = _validate_preflight_gate_evidence(
        Path(preflight_gate_evidence),
        manifest_bytes=manifest_bytes,
        apartment_bytes=apartment_bytes,
        apartment=apartment,
        declarations=declared,
        selected_ids=selected_ids,
    )

    destination = Path(output_root).absolute()
    _revalidate_preflight_witnesses(preflight_record)
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
        "preflight_gate_evidence": preflight_record,
        "max_parallel": max_parallel,
        "gpu_ids": list(gpus),
        "required_available_ram_bytes": required_ram,
        "observed_available_ram_bytes": supplied_ram,
        "status": "RUNNING",
        "candidates": records if not requested_diagnostics else [],
        "unscheduled_candidate_ids": list(selected_ids),
    }
    if requested_diagnostics:
        status["diagnostic_candidates"] = records
    status_path = destination / "search_status.json"
    _write_status(status_path, status)

    pending = list(selected_ids)
    active: list[dict[str, Any]] = []
    failed = False
    fatal_error: Exception | None = None
    requires_run_manifest = command_builder is None
    builder = command_builder or _default_command
    while pending or active:
        while pending and not failed and len(active) < max_parallel:
            active_lanes = {
                int(item["lane"])
                for item in active
            }
            launch_position = next(
                (
                    position
                    for position, name in enumerate(pending)
                    if lane_by_candidate[name] not in active_lanes
                ),
                None,
            )
            if launch_position is None:
                break
            candidate_id = pending.pop(launch_position)
            lane = lane_by_candidate[candidate_id]
            gpu = gpus[lane]
            inventory = "diagnostics" if requested_diagnostics else "candidates"
            candidate_root = destination / inventory / candidate_id / "apartment"
            candidate_root.parent.parent.mkdir(exist_ok=True)
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
            _revalidate_preflight_witnesses(preflight_record)
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
            if requested_diagnostics:
                record["diagnostic_identity"] = candidate_id
                record["base_profile"] = declared[candidate_id]["base_profile"]
                record["selectable"] = False
            records.append(record)
            active.append(
                {
                    "process": process,
                    "record": record,
                    "stdout": stdout,
                    "stderr": stderr,
                    "gpu": gpu,
                    "lane": lane,
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
            active.remove(item)
            if exit_code != 0:
                failed = True
        status["unscheduled_candidate_ids"] = list(pending)
        _write_status(status_path, status)

    records.sort(key=lambda item: canonical_order.index(item["candidate_id"]))
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
    parser.add_argument("--scene", choices=("apartment", "office"), default="apartment")
    parser.add_argument("--office-freeze-authorization", type=Path)
    parser.add_argument("--preflight-gate-evidence", required=True, type=Path)
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
        scene=args.scene,
        office_freeze_authorization=args.office_freeze_authorization,
        preflight_gate_evidence=args.preflight_gate_evidence,
    )
    print(json.dumps({"status_path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
