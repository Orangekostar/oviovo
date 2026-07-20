#!/usr/bin/env python3
"""Aggregate native OVI-MAP evidence and publish eligible diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.baselines.ovimap_paper_audit import (
    NATIVE_INPUT_HASHES,
    PAPER_PROTOCOL,
)


class FinalizationError(ValueError):
    """Raised when native OVI-MAP evidence is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"evidence file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    _source(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalizationError(f"evidence is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"evidence must contain an object: {path}")
    return value


def _validate_artifact(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise FinalizationError(f"{label} artifact record is missing")
    path = Path(str(record.get("path", "")))
    source = _source(path)
    if record.get("sha256") != source["sha256"]:
        raise FinalizationError(f"{label} artifact hash mismatch")
    return source


def aggregate_native_mapping(
    scene_manifests: Mapping[str, Path],
    *,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Verify and aggregate the exact native Replica-8 mapping artifacts."""
    scenes = tuple(PAPER_PROTOCOL["scene_ids"])
    if set(scene_manifests) != set(scenes):
        missing = sorted(set(scenes) - set(scene_manifests))
        extra = sorted(set(scene_manifests) - set(scenes))
        raise FinalizationError(
            f"scene manifests must contain Replica-8; missing={missing}, extra={extra}"
        )
    if set(input_hashes) != set(NATIVE_INPUT_HASHES) or any(
        not _is_sha256(input_hashes.get(name)) for name in NATIVE_INPUT_HASHES
    ):
        raise FinalizationError("native input hashes are incomplete")

    expected_frames = list(
        range(
            0,
            PAPER_PROTOCOL["frame_count_per_scene"] * PAPER_PROTOCOL["frame_step"],
            PAPER_PROTOCOL["frame_step"],
        )
    )
    shared_sources: dict[str, str] | None = None
    shared_environments: dict[str, str] | None = None
    scene_records: dict[str, Any] = {}
    frame_ids_by_scene: dict[str, list[int]] = {}
    for scene in scenes:
        path = Path(scene_manifests[scene])
        manifest = _load_json(path)
        if (
            manifest.get("status") != "PASS"
            or manifest.get("scene") != scene
            or manifest.get("audit", {}).get("status") != "PASS"
        ):
            raise FinalizationError(f"{scene} native mapping gate did not pass")
        if manifest.get("frame_ids") != expected_frames:
            raise FinalizationError(f"{scene} frame protocol mismatch")
        if manifest.get("audit", {}).get("frame_count") != len(expected_frames):
            raise FinalizationError(f"{scene} audit frame count mismatch")

        sources = manifest.get("preflight", {}).get("sources", {})
        source_hashes = {
            name: record.get("sha256") if isinstance(record, Mapping) else None
            for name, record in sources.items()
        }
        required_sources = {
            "cropformer_config",
            "cropformer_weights",
            "mapper",
            "source_hashes",
        }
        if set(source_hashes) != required_sources or any(
            not _is_sha256(value) for value in source_hashes.values()
        ):
            raise FinalizationError(f"{scene} source hashes are incomplete")
        if shared_sources is None:
            shared_sources = source_hashes
        elif source_hashes != shared_sources:
            raise FinalizationError(f"{scene} source hash mismatch")

        frontend_argv = manifest.get("frontend", {}).get("argv", [])
        geometry_argv = manifest.get("mapping", {}).get("geometry", {}).get("argv", [])
        mapping_argv = manifest.get("mapping", {}).get("mapping", {}).get("argv", [])
        if not all(isinstance(argv, Sequence) and argv for argv in (frontend_argv, geometry_argv, mapping_argv)):
            raise FinalizationError(f"{scene} command provenance is incomplete")
        environments = {
            "frontend_python": str(frontend_argv[0]),
            "geometry_python": str(geometry_argv[0]),
            "mapping_python": str(mapping_argv[0]),
        }
        if shared_environments is None:
            shared_environments = environments
        elif environments != shared_environments:
            raise FinalizationError(f"{scene} environment mismatch")

        artifacts = manifest.get("artifacts", {})
        verified_artifacts = {
            name: _validate_artifact(artifacts.get(name), label=f"{scene} {name}")
            for name in ("instance_mesh", "semantic_features")
        }
        scene_records[scene] = {
            "manifest": _source(path),
            "artifacts": verified_artifacts,
            "full_frame_bbox_ratio": manifest.get("audit", {}).get(
                "full_frame_bbox_ratio"
            ),
        }
        frame_ids_by_scene[scene] = expected_frames

    return {
        "schema_version": 1,
        "status": "COMPLETE_NATIVE_MAPPING",
        "protocol_name": PAPER_PROTOCOL["name"],
        "scene_ids": list(scenes),
        "frame_ids_by_scene": frame_ids_by_scene,
        "input_hashes": dict(input_hashes),
        "shared_source_hashes": shared_sources,
        "runtime_environments": shared_environments,
        "scenes": scene_records,
    }


def _finite_metric(metrics: Mapping[str, Any], name: str, *, label: str) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise FinalizationError(f"{label} metric is missing or non-finite: {name}")
    return float(value)


def finalize_native_result(evidence: Mapping[str, Path]) -> dict[str, Any]:
    """Publish released metrics as diagnostics without inventing paper AP."""
    released_path = evidence.get("released_evaluation_manifest")
    if released_path is None:
        raise FinalizationError("released evaluation manifest is required")
    released = _load_json(Path(released_path))
    if (
        released.get("status") != "COMPLETE_RELEASED_EVALUATION"
        or released.get("source_protocol") != "released_ovimap_replica51"
        or released.get("scene_ids") != list(PAPER_PROTOCOL["scene_ids"])
        or released.get("frame_count_per_scene")
        != PAPER_PROTOCOL["frame_count_per_scene"]
    ):
        raise FinalizationError("released evaluation protocol is incomplete")
    outputs = released.get("output_artifacts", {})
    vertex = outputs.get("semantic_vertex_metrics", {})
    semantic_instance = outputs.get("semantic_instance_metrics", {})
    if not isinstance(vertex, Mapping):
        vertex = {}
    if not isinstance(semantic_instance, Mapping):
        semantic_instance = {}
    miou = _finite_metric(vertex, "miou", label="semantic vertex")
    macc = _finite_metric(vertex, "macc", label="semantic vertex")
    apall = _finite_metric(semantic_instance, "apall", label="semantic-instance")
    ap50 = _finite_metric(semantic_instance, "ap50", label="semantic-instance")
    ap25 = _finite_metric(semantic_instance, "ap25", label="semantic-instance")

    protocol_evidence = {
        name: _source(Path(path)) for name, path in evidence.items()
    }
    has_paper_ap = "class_agnostic_ap_manifest" in evidence
    if has_paper_ap:
        paper_ap = _load_json(Path(evidence["class_agnostic_ap_manifest"]))
        if paper_ap.get("status") != "COMPLETE_PAPER_AP_EVALUATION":
            raise FinalizationError("class-agnostic AP manifest is incomplete")

    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "method": {"key": "OVIMAP", "display_label": "OVI-MAP", "mode": "native"},
        "protocol": {
            **PAPER_PROTOCOL,
            "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
        },
        "protocol_evidence": protocol_evidence,
        "metrics": {
            "replica_8_compat": {
                "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
                "scene_count": len(PAPER_PROTOCOL["scene_ids"]),
                "semantic": {"miou": miou, "macc": macc},
            },
            "released_replica51": {
                "semantic_miou": miou,
                "semantic_macc": macc,
                "semantic_instance_apall": apall,
                "semantic_instance_ap50": ap50,
                "semantic_instance_ap25": ap25,
            },
        },
        "availability": {
            "released_semantic_miou": "VERIFIED_DIAGNOSTIC",
            "released_semantic_macc": "VERIFIED_DIAGNOSTIC",
            "t1_replica8_miou": "UNFILLED_PAPER_AUDIT_INCOMPLETE",
            "t1_replica8_macc": "UNFILLED_PAPER_AUDIT_INCOMPLETE",
            "class_agnostic_ap25": (
                "PRESENT_REQUIRES_PAPER_AUDIT"
                if has_paper_ap
                else "UNFILLED_NO_RELEASED_EVALUATOR"
            ),
            "class_agnostic_ap50": (
                "PRESENT_REQUIRES_PAPER_AUDIT"
                if has_paper_ap
                else "UNFILLED_NO_RELEASED_EVALUATOR"
            ),
            "fmiou": "UNFILLED_NOT_DEFINED_BY_RELEASED_PROTOCOL",
            "f5": "UNFILLED_NOT_DEFINED_BY_RELEASED_PROTOCOL",
        },
        "token_bindings": [],
    }


def _parse_mapping(values: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise FinalizationError(f"invalid {label}: {value}")
        key, item = value.split("=", 1)
        if not key or key in parsed:
            raise FinalizationError(f"duplicate or empty key in {label}: {key}")
        parsed[key] = item
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--scene-manifest", action="append", required=True)
    aggregate.add_argument("--input-hash", action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    result = subparsers.add_parser("result")
    result.add_argument("--native", type=Path, required=True)
    result.add_argument("--released", type=Path, required=True)
    result.add_argument("--class-agnostic-ap", type=Path)
    result.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "aggregate":
            scene_paths = {
                key: Path(value)
                for key, value in _parse_mapping(
                    args.scene_manifest, label="--scene-manifest"
                ).items()
            }
            payload = aggregate_native_mapping(
                scene_paths,
                input_hashes=_parse_mapping(args.input_hash, label="--input-hash"),
            )
        else:
            evidence = {
                "native_mapping_manifest": args.native,
                "released_evaluation_manifest": args.released,
            }
            if args.class_agnostic_ap is not None:
                evidence["class_agnostic_ap_manifest"] = args.class_agnostic_ap
            payload = finalize_native_result(evidence)
        _atomic_json(args.output, payload)
    except (OSError, FinalizationError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
