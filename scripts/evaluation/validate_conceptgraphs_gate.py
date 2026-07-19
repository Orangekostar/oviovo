#!/usr/bin/env python3
"""Validate the canonical ConceptGraphs room gate and its provenance."""

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


_IDENTITY = {
    "frontend": "sam_segment_all",
    "gsa_variant": "none",
    "class_agnostic": True,
    "mask_conf_threshold": 0.95,
    "sim_threshold": 1.2,
    "dbscan_eps": 0.1,
    "merge_interval": 20,
    "merge_visual_sim_thresh": 0.8,
    "merge_text_sim_thresh": 0.8,
    "headline_stride": 10,
    "official_diagnostic_stride": 5,
}
_REQUIRED_PROVENANCE = {
    "sam_vit_h_checkpoint",
    "groundingdino_checkpoint",
    "ram_checkpoint",
    "mapping_config",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_config(config: dict[str, Any]) -> None:
    for name, expected in _IDENTITY.items():
        if config.get(name) != expected:
            raise ValueError(f"canonical {name} must equal {expected!r}")
    headline = config.get("headline_evaluation")
    if not isinstance(headline, dict):
        raise ValueError("headline_evaluation must be an object")
    if headline.get("gt_only_class_suppression") is not False:
        raise ValueError("headline GT-only class suppression must be disabled")
    if headline.get("n_exclude") != 0:
        raise ValueError("headline n_exclude must be 0")
    commits = config.get("source_commits")
    if not isinstance(commits, dict) or set(commits) != {
        "conceptgraphs",
        "grounded_segment_anything",
    }:
        raise ValueError("canonical source commits are incomplete")
    if any(
        not isinstance(value, str) or len(value) != 40
        for value in commits.values()
    ):
        raise ValueError("canonical source commits must be full hashes")
    if set(config.get("required_provenance", ())) != _REQUIRED_PROVENANCE:
        raise ValueError("canonical required provenance set is incomplete")


def _positive_count(counts: dict[str, Any], name: str) -> int:
    value = counts.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_provenance(status: dict[str, Any]) -> dict[str, dict[str, str]]:
    provenance = status.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("status provenance must be an object")
    missing = sorted(_REQUIRED_PROVENANCE - set(provenance))
    if missing:
        raise ValueError(f"status provenance is missing records: {missing}")
    verified = {}
    for name in sorted(_REQUIRED_PROVENANCE):
        record = provenance[name]
        if not isinstance(record, dict):
            raise ValueError(f"provenance {name} must be an object")
        path_value = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise ValueError(f"provenance {name} requires path and sha256")
        path = Path(path_value)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"provenance {name} hash verification failed")
        verified[name] = {"path": str(path.resolve()), "sha256": digest}
    return verified


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_object(args.config, "config")
    _validate_config(config)
    status = _read_object(args.status, "status")
    if status.get("source_commits") != config["source_commits"]:
        raise ValueError("status source commits do not match canonical config")
    stages = status.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("status stages must be an object")
    for name in ("frontend", "mapper"):
        stage = stages.get(name)
        if not isinstance(stage, dict) or stage.get("exit_status") != 0:
            raise ValueError(f"{name} exit_status must be 0")
    counts = status.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("status counts must be an object")
    normalized_counts = {
        "mapped_object_count": _positive_count(counts, "mapped_object_count"),
        "mask_count": _positive_count(counts, "mask_count"),
    }
    if not args.detections.is_dir():
        raise ValueError("detections directory is missing")
    detection_files = sorted(path for path in args.detections.rglob("*") if path.is_file())
    if not detection_files or not any(path.stat().st_size > 0 for path in detection_files):
        raise ValueError("detections contain no non-empty artifacts")
    if not args.map.is_file() or args.map.stat().st_size <= 0:
        raise ValueError("map artifact must be non-empty")
    metrics = _read_object(args.neutral_metrics, "neutral metrics")
    try:
        matched_ratio = float(metrics["metrics"]["semantic"]["matched_point_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("neutral matched_point_ratio is missing") from error
    if not math.isfinite(matched_ratio) or not 0.0 <= matched_ratio <= 1.0:
        raise ValueError("neutral matched_point_ratio must be finite and within [0, 1]")
    provenance = _validate_provenance(status)
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "headline_result_eligible": True,
        "counts": normalized_counts,
        "neutral_matched_point_ratio": matched_ratio,
        "source_commits": config["source_commits"],
        "provenance": provenance,
        "sources": {
            "canonical_config": {
                "path": str(args.config.resolve()),
                "sha256": _sha256(args.config),
            },
            "status": {
                "path": str(args.status.resolve()),
                "sha256": _sha256(args.status),
            },
            "map": {"path": str(args.map.resolve()), "sha256": _sha256(args.map)},
            "neutral_metrics": {
                "path": str(args.neutral_metrics.resolve()),
                "sha256": _sha256(args.neutral_metrics),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--neutral-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = _validate(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _write_json_atomic(
            args.output,
            {
                "schema_version": 1,
                "status": "BLOCKED",
                "headline_result_eligible": False,
                "error": str(error),
            },
        )
        print(str(error), file=sys.stderr)
        return 1
    _write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
