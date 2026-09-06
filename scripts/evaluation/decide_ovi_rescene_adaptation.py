#!/usr/bin/env python3
"""Select one OVI/ReScene adaptation action from hash-bound measured evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_STATUS = {"MEASURED", "MISSING_ASSET"}
_EVIDENCE_FIELDS = {
    "native_r": ("temporal_ap", "temporal_recall"),
    "d2": ("d0_paired_f1", "d2_paired_f1"),
    "raw_resolver": (
        "raw_nonempty_fraction",
        "legacy_paired_f1",
        "supported_paired_f1",
    ),
    "registration_oracle": ("o1_f_score", "o2_f_score"),
    "completion_opportunity": ("opportunity_fraction",),
    "identity_geometry": ("identity_gain", "geometry_gain"),
}
_CONFIG_KEYS = {
    "schema_version",
    "config_id",
    "matrix_result",
    "ownership_completion_result",
    "output",
}
_RECORD_KEYS = {"path", "sha256", "byte_count"}


class AdaptationDecisionError(ValueError):
    """Raised when adaptation evidence is missing, synthetic, or malformed."""


def _metric(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AdaptationDecisionError(f"{label} must be a finite measured value")
    metric = float(value)
    if not 0.0 <= metric <= 1.0:
        raise AdaptationDecisionError(f"{label} must be in [0, 1]")
    return metric


def _validated_evidence(value: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if value.get("schema_version") != 1 or set(value) != {
        "schema_version",
        *_EVIDENCE_FIELDS,
    }:
        raise AdaptationDecisionError("adaptation evidence schema is invalid")
    normalized: dict[str, dict[str, object]] = {}
    for group_name, fields in _EVIDENCE_FIELDS.items():
        group = value.get(group_name)
        if not isinstance(group, Mapping) or set(group) != {"status", *fields}:
            raise AdaptationDecisionError(f"{group_name} evidence schema is invalid")
        status = group.get("status")
        if status not in _STATUS:
            raise AdaptationDecisionError(f"{group_name} evidence status is invalid")
        if status == "MISSING_ASSET":
            if any(group.get(field) is not None for field in fields):
                raise AdaptationDecisionError(
                    f"{group_name} missing evidence cannot carry a numeric value"
                )
            normalized[group_name] = {"status": status, **{field: None for field in fields}}
        else:
            normalized[group_name] = {
                "status": status,
                **{
                    field: _metric(group.get(field), label=f"{group_name}.{field}")
                    for field in fields
                },
            }
    return normalized


def decide_adaptation(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the six preregistered rules without treating missing values as zero."""

    if not isinstance(evidence, Mapping):
        raise AdaptationDecisionError("adaptation evidence must be a mapping")
    measured = _validated_evidence(evidence)
    native = measured["native_r"]
    if native["status"] != "MEASURED":
        action = "REPORT_MODEL_LIMIT"
        rule = "native_r_missing"
    elif native["temporal_ap"] < 0.05 and native["temporal_recall"] < 0.10:
        action = "REPORT_MODEL_LIMIT"
        rule = "native_r_weak"
    else:
        d2 = measured["d2"]
        raw = measured["raw_resolver"]
        oracle = measured["registration_oracle"]
        opportunity = measured["completion_opportunity"]
        identity = measured["identity_geometry"]
        if (
            d2["status"] == "MEASURED"
            and d2["d2_paired_f1"] + 0.05 < d2["d0_paired_f1"]
        ):
            action = "ADAPT_DECODER_MASK_HEAD"
            rule = "native_strong_d2_weak"
        elif (
            raw["status"] == "MEASURED"
            and raw["raw_nonempty_fraction"] >= 0.50
            and raw["legacy_paired_f1"] >= 0.05
            and raw["supported_paired_f1"] + 0.02 <= raw["legacy_paired_f1"]
        ):
            action = "TUNE_RESOLVER"
            rule = "raw_masks_strong_resolver_weak"
        elif (
            oracle["status"] == "MEASURED"
            and oracle["o1_f_score"] < 0.10
            and oracle["o2_f_score"] >= oracle["o1_f_score"] + 0.10
        ):
            action = "FIX_REGISTRATION"
            rule = "estimated_registration_bottleneck"
        elif (
            opportunity["status"] == "MEASURED"
            and opportunity["opportunity_fraction"] <= 0.01
        ):
            action = "CHANGE_PAIR_BUDGET"
            rule = "completion_opportunity_negligible"
        elif (
            identity["status"] == "MEASURED"
            and identity["identity_gain"] > 0.0
            and identity["geometry_gain"] <= 0.0
        ):
            action = "RETAIN_IDENTITY_ONLY"
            rule = "identity_gain_without_geometry_gain"
        else:
            action = "REPORT_MODEL_LIMIT"
            rule = "no_measured_actionable_gap"
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_ADAPTATION_DECISION_V1",
        "status": "PASS",
        "evidence_status": "MEASURED_ONLY",
        "action": action,
        "selected_rule": rule,
        "ordered_rules": [
            "native_r_weak",
            "native_strong_d2_weak",
            "raw_masks_strong_resolver_weak",
            "estimated_registration_bottleneck",
            "completion_opportunity_negligible",
            "identity_gain_without_geometry_gain",
        ],
        "evidence": measured,
    }


def _regular_bytes(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise AdaptationDecisionError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise AdaptationDecisionError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise AdaptationDecisionError(f"{label} changed while being read")
    return absolute, content


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdaptationDecisionError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise AdaptationDecisionError(f"{label} must contain an object")
    return value


def _load_bound(record: object, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise AdaptationDecisionError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise AdaptationDecisionError(f"{label} path must be absolute")
    path, content = _regular_bytes(raw_path, label=label)
    observed = {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    if dict(record) != observed:
        raise AdaptationDecisionError(f"{label} binding mismatch")
    return path, _json_object(content, label=label)


def _f_score(metrics: Mapping[str, object], *, label: str) -> float:
    precision = _metric(metrics.get("paired_precision"), label=f"{label}.precision")
    recall = _metric(metrics.get("paired_recall"), label=f"{label}.recall")
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def build_measured_evidence(
    matrix: Mapping[str, object],
    ownership: Mapping[str, object],
    pair_results: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Derive the decision surface from real result schemas."""

    if (
        matrix.get("artifact_id") != "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1"
        or matrix.get("status") != "PASS"
    ):
        raise AdaptationDecisionError("matrix result identity is invalid")
    official = matrix.get("official_stmetrics")
    methods = matrix.get("methods")
    domains = matrix.get("domain_coverage")
    if not all(isinstance(value, Mapping) for value in (official, methods, domains)):
        raise AdaptationDecisionError("matrix evidence schema is invalid")
    official_metrics = official.get("metrics")
    legacy = methods.get("R_legacy")
    supported = methods.get("R_supported")
    if not all(isinstance(value, Mapping) for value in (official_metrics, legacy, supported)):
        raise AdaptationDecisionError("matrix method evidence is invalid")
    legacy_primary = legacy.get("pooled", {}).get("iou_0_50")
    supported_primary = supported.get("pooled", {}).get("iou_0_50")
    if not isinstance(legacy_primary, Mapping) or not isinstance(supported_primary, Mapping):
        raise AdaptationDecisionError("matrix pooled association evidence is invalid")
    raw_total = raw_nonempty = 0
    for payload in pair_results:
        raw = payload.get("native_raw_queries")
        if not isinstance(raw, Mapping):
            raise AdaptationDecisionError("pair raw-query evidence is invalid")
        total = raw.get("total_count")
        nonempty = raw.get("nonempty_count")
        if type(total) is not int or total <= 0 or type(nonempty) is not int or not 0 <= nonempty <= total:
            raise AdaptationDecisionError("pair raw-query counts are invalid")
        raw_total += total
        raw_nonempty += nonempty
    if ownership.get("artifact_id") != "OVI_RESCENE_OWNERSHIP_COMPLETION_V1":
        raise AdaptationDecisionError("ownership result identity is invalid")
    completion = ownership.get("completion")
    ownership_rows = ownership.get("ownership")
    if not isinstance(completion, list) or not isinstance(ownership_rows, list):
        raise AdaptationDecisionError("ownership result schema is invalid")
    completion_by_id = {
        row.get("variant_id"): row for row in completion if isinstance(row, Mapping)
    }
    ownership_by_id = {
        row.get("variant_id"): row for row in ownership_rows if isinstance(row, Mapping)
    }
    oracle_available = all(
        isinstance(completion_by_id.get(name), Mapping)
        and completion_by_id[name].get("completion_surface") is not None
        for name in ("O1", "O2")
    )
    opportunity_surface = completion_by_id.get("C1", {}).get("completion_surface")
    identity_available = all(
        isinstance(ownership_by_id.get(name), Mapping)
        and ownership_by_id[name].get("instance_ap") is not None
        for name in ("A1", "A2")
    )
    d2_available = domains.get("D2_OVI_RECONSTRUCTION") == "PASS"
    return {
        "schema_version": 1,
        "native_r": {
            "status": "MEASURED",
            "temporal_ap": official_metrics.get("dev_mean_t-AP"),
            "temporal_recall": official_metrics.get("dev_mean_t-REC"),
        },
        "d2": {
            "status": "MEASURED" if d2_available else "MISSING_ASSET",
            "d0_paired_f1": _f_score(legacy_primary, label="D0") if d2_available else None,
            "d2_paired_f1": None,
        },
        "raw_resolver": {
            "status": "MEASURED",
            "raw_nonempty_fraction": raw_nonempty / raw_total,
            "legacy_paired_f1": _f_score(legacy_primary, label="legacy"),
            "supported_paired_f1": _f_score(supported_primary, label="supported"),
        },
        "registration_oracle": {
            "status": "MEASURED" if oracle_available else "MISSING_ASSET",
            "o1_f_score": completion_by_id.get("O1", {}).get("completion_surface", {}).get("f_score") if oracle_available else None,
            "o2_f_score": completion_by_id.get("O2", {}).get("completion_surface", {}).get("f_score") if oracle_available else None,
        },
        "completion_opportunity": {
            "status": "MEASURED" if isinstance(opportunity_surface, Mapping) else "MISSING_ASSET",
            "opportunity_fraction": opportunity_surface.get("opportunity_fraction") if isinstance(opportunity_surface, Mapping) else None,
        },
        "identity_geometry": {
            "status": "MEASURED" if identity_available else "MISSING_ASSET",
            "identity_gain": ownership_by_id.get("A2", {}).get("instance_ap") - ownership_by_id.get("A1", {}).get("instance_ap") if identity_available else None,
            "geometry_gain": 0.0 if identity_available else None,
        },
    }


def decide_from_config(config_path: str | Path) -> tuple[Path, dict[str, object]]:
    """Audit configured results and atomically publish their selected action."""

    config_path, config_content = _regular_bytes(config_path, label="decision config")
    config = _json_object(config_content, label="decision config")
    if set(config) != _CONFIG_KEYS or config.get("schema_version") != 1 or config.get("config_id") != "OVI_RESCENE_ADAPTATION_DECISION_CONFIG_V1":
        raise AdaptationDecisionError("decision config identity is invalid")
    _matrix_path, matrix = _load_bound(config.get("matrix_result"), label="matrix result")
    _ownership_path, ownership = _load_bound(
        config.get("ownership_completion_result"), label="ownership result"
    )
    bindings = matrix.get("source_bindings")
    pair_records = bindings.get("pair_results") if isinstance(bindings, Mapping) else None
    if not isinstance(pair_records, Mapping) or not pair_records:
        raise AdaptationDecisionError("matrix pair bindings are invalid")
    pair_payloads = tuple(
        _load_bound(record, label=f"pair result {pair_id}")[1]
        for pair_id, record in sorted(pair_records.items())
    )
    evidence = build_measured_evidence(matrix, ownership, pair_payloads)
    result = decide_adaptation(evidence)
    result["source_bindings"] = {
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_content).hexdigest(),
            "byte_count": len(config_content),
        },
        "matrix_result": dict(config["matrix_result"]),
        "ownership_completion_result": dict(config["ownership_completion_result"]),
        "pair_results": dict(sorted(pair_records.items())),
        "decision_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "byte_count": Path(__file__).stat().st_size,
        },
    }
    output = config.get("output")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise AdaptationDecisionError("decision output path must be absolute")
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise AdaptationDecisionError(f"decision output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        output, result = decide_from_config(arguments.config)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"action": result["action"], "output": str(output)}, sort_keys=True))
    return 0


__all__ = [
    "AdaptationDecisionError",
    "build_measured_evidence",
    "decide_adaptation",
    "decide_from_config",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
