#!/usr/bin/env python3
"""Audit OVIV2 Replica headline instance metric compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


_PROTOCOL_FIELDS = (
    "distance_threshold_m",
    "projection_comparator",
    "min_instance_vertices",
    "headline_instance_protocol",
    "semantic_instance_protocol",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _read_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            result.extend(_nonfinite_paths(child, f"{path}.{key}"))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_nonfinite_paths(child, f"{path}[{index}]"))
        return result
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    return []


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    old = _read_object(args.old_metrics, "old metrics")
    new = _read_object(args.new_metrics, "new metrics")
    reasons = []
    for name, metrics in (("old", old), ("new", new)):
        nonfinite = _nonfinite_paths(metrics)
        if nonfinite:
            reasons.append(f"{name} metrics must contain only finite values: {nonfinite}")

    protocol = new.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}
        reasons.append("new metrics protocol must be an object")
    for name in _PROTOCOL_FIELDS:
        if name not in protocol:
            reasons.append(f"new protocol is missing {name}")
    if protocol.get("headline_instance_protocol") != "class_agnostic":
        reasons.append("headline_instance_protocol must be class_agnostic")
    if protocol.get("semantic_instance_protocol") != "diagnostic_only":
        reasons.append("semantic_instance_protocol must be diagnostic_only")

    instance = new.get("instance")
    class_agnostic = (
        instance.get("class_agnostic") if isinstance(instance, dict) else None
    )
    if not isinstance(class_agnostic, dict):
        class_agnostic = {}
        reasons.append("new metrics instance.class_agnostic must be an object")
    for metric_name in ("ap25", "ap50"):
        if metric_name not in old or metric_name not in new:
            reasons.append(f"old and new metrics require headline {metric_name}")
        if class_agnostic.get(metric_name) != new.get(metric_name):
            reasons.append(
                f"headline {metric_name} must equal instance.class_agnostic.{metric_name}"
            )

    def _headline(metrics: dict[str, Any]) -> dict[str, float | None]:
        result = {}
        for name in ("ap25", "ap50"):
            value = metrics.get(name)
            normalized = float(value) if isinstance(value, (int, float)) else None
            result[name] = (
                normalized
                if normalized is not None and math.isfinite(normalized)
                else None
            )
        return result

    old_headline = _headline(old)
    new_headline = _headline(new)
    delta = {
        name: (
            new_headline[name] - old_headline[name]
            if new_headline[name] is not None
            and old_headline[name] is not None
            else None
        )
        for name in ("ap25", "ap50")
    }
    minimum = protocol.get("min_instance_vertices")
    audit = {
        "schema_version": 1,
        "headline_compatible": not reasons,
        "reasons": reasons,
        "protocol": {
            "distance_threshold_m": protocol.get("distance_threshold_m"),
            "headline_accepted_view_filter": False,
            "headline_instance_protocol": protocol.get("headline_instance_protocol"),
            "min_gt_instance_vertices": minimum,
            "min_predicted_instance_vertices": minimum,
            "projection_comparator": protocol.get("projection_comparator"),
            "semantic_diagnostic_accepted_view_filter": True,
        },
        "headline_metrics": {
            "old": old_headline,
            "new": new_headline,
            "delta": delta,
        },
        "sources": {
            "old_metrics": _source(args.old_metrics),
            "new_metrics": _source(args.new_metrics),
        },
    }
    if args.ovimap_diagnostics is not None:
        diagnostics = _read_object(args.ovimap_diagnostics, "OVI-MAP diagnostics")
        if diagnostics.get("headline_token_eligible") is not False:
            reasons.append("OVI-MAP diagnostics must be non-headline")
            audit["headline_compatible"] = False
        audit["sources"]["ovimap_diagnostics"] = _source(args.ovimap_diagnostics)
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--ovimap-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        audit = _audit(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        audit = {
            "schema_version": 1,
            "headline_compatible": False,
            "reasons": [str(error)],
        }
    _write_json_atomic(args.output, audit)
    if not audit["headline_compatible"]:
        print("; ".join(audit["reasons"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
