#!/usr/bin/env python3
"""Compare OVIV2 metrics against the Route 2 or final Pareto gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.oviv2.hybrid_cache import _publish_bytes


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "miou": ("semantic", "miou"),
    "macc": ("semantic", "macc"),
    "f_miou": ("semantic", "f_miou"),
    "ap25": ("instance", "class_agnostic", "ap25"),
    "ap50": ("instance", "class_agnostic", "ap50"),
    "f5": ("geometry", "f5"),
}
ROUTE2_IMPROVEMENT_METRICS = frozenset(METRIC_PATHS) - {"f5"}
VALID_MODES = frozenset({"route2", "final"})
PROTOCOL_COMPARISON_EXCLUSIONS = frozenset({"snapshot_checksums"})


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _exact_number(value: float) -> str:
    return value.hex()


def _read_metric(payload: Mapping[str, Any], name: str, label: str) -> float:
    current: object = payload
    for component in METRIC_PATHS[name]:
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"{label} is missing metric {name}")
        current = current[component]
    value = _finite_number(current, f"{label} metric {name}")

    aliases: list[tuple[str, object]] = []
    if name in payload:
        aliases.append((name, payload[name]))
    if name in {"ap25", "ap50"}:
        instance = payload.get("instance")
        if isinstance(instance, Mapping) and name in instance:
            aliases.append((f"instance.{name}", instance[name]))
    for alias_name, alias_value in aliases:
        alias = _finite_number(alias_value, f"{label} alias {alias_name}")
        if _exact_number(alias) != _exact_number(value):
            raise ValueError(f"{label} has ambiguous metric field {name}")
    return value


def _protocol_context(
    payload: Mapping[str, Any], label: str
) -> tuple[str, dict[str, Any], str]:
    protocol = _object(payload.get("protocol"), f"{label} protocol")
    scene = protocol.get("scene_id")
    if not isinstance(scene, str) or not scene:
        raise ValueError(f"{label} protocol.scene_id must be a non-empty string")
    for alias_name in ("scene", "scene_id"):
        if alias_name in payload and payload[alias_name] != scene:
            raise ValueError(
                f"{label} {alias_name} does not match protocol.scene_id"
            )

    full_protocol = copy.deepcopy(dict(protocol))
    comparison_protocol = {
        key: value
        for key, value in full_protocol.items()
        if key not in PROTOCOL_COMPARISON_EXCLUSIONS
    }
    try:
        canonical = json.dumps(
            comparison_protocol,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} protocol is not valid JSON: {error}") from error
    return scene, full_protocol, canonical


def _reject_ambiguous_wrapper(payload: Mapping[str, Any], label: str) -> None:
    wrapped = payload.get("metrics")
    if isinstance(wrapped, Mapping) and any(
        key in wrapped for key in ("semantic", "instance", "geometry")
    ):
        raise ValueError(f"{label} has ambiguous metrics wrapper")


def compare_metrics(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Return a complete PASS/FAIL audit for an OVIV2 Pareto comparison."""

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    baseline_object = _object(baseline, "baseline")
    candidate_object = _object(candidate, "candidate")
    _reject_ambiguous_wrapper(baseline_object, "baseline")
    _reject_ambiguous_wrapper(candidate_object, "candidate")

    baseline_scene, baseline_protocol, baseline_contract = _protocol_context(
        baseline_object, "baseline"
    )
    candidate_scene, candidate_protocol, candidate_contract = _protocol_context(
        candidate_object, "candidate"
    )
    if baseline_scene != candidate_scene:
        raise ValueError(
            f"scene mismatch: baseline={baseline_scene!r}, candidate={candidate_scene!r}"
        )
    if baseline_contract != candidate_contract:
        raise ValueError("protocol mismatch between baseline and candidate")

    checks: dict[str, dict[str, Any]] = {}
    for name in METRIC_PATHS:
        baseline_value = _read_metric(baseline_object, name, "baseline")
        candidate_value = _read_metric(candidate_object, name, "candidate")
        requires_improvement = mode == "final" or name in ROUTE2_IMPROVEMENT_METRICS
        relation = "strictly_greater" if requires_improvement else "byte_identical"
        passed = (
            candidate_value > baseline_value
            if requires_improvement
            else _exact_number(candidate_value) == _exact_number(baseline_value)
        )
        check: dict[str, Any] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
            "passed": passed,
            "relation": relation,
        }
        if not requires_improvement:
            check["baseline_ieee754"] = _exact_number(baseline_value)
            check["candidate_ieee754"] = _exact_number(candidate_value)
        checks[name] = check

    return {
        "schema_version": 1,
        "mode": mode,
        "scene": baseline_scene,
        "status": "PASS" if all(check["passed"] for check in checks.values()) else "FAIL",
        "checks": checks,
        "protocol": {
            "baseline": baseline_protocol,
            "candidate": candidate_protocol,
            "comparison_excludes": sorted(PROTOCOL_COMPARISON_EXCLUSIONS),
            "matched": True,
        },
    }


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"invalid non-finite JSON number: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read JSON input {path}: {error}") from error
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _publish_audit(path: Path, audit: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    def write_audit(stream: Any) -> None:
        stream.write(encoded)

    _publish_bytes(Path(path), write_audit, field_name="Pareto audit")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        baseline, baseline_sha256 = _load_json(args.baseline)
        candidate, candidate_sha256 = _load_json(args.candidate)
        audit = compare_metrics(baseline, candidate, args.mode)
        audit["inputs"] = {
            "baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": baseline_sha256,
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "sha256": candidate_sha256,
            },
        }
        _publish_audit(args.output, audit)
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
