#!/usr/bin/env python3
"""Validate frozen protocol and prediction bindings before two-visit evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.freeze_tesse_two_visit_protocol import (
    PROTOCOL_ID,
    ProtocolError,
    authorize_scene_run,
    compute_window_input_sha256,
    protocol_content_sha256,
)

_RECORD_KEYS = {"path", "sha256", "byte_count"}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_regular(path: Path, *, label: str) -> bytes:
    before = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    data = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise ProtocolError(f"{label} changed while reading")
    return data


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular(path, label=label)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain a JSON object")
    return value, data


def _validate_record(record: object, *, label: str) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ProtocolError(f"{label} binding fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ProtocolError(f"{label} binding path is invalid")
    declared_path = Path(raw_path)
    if ".." in declared_path.parts:
        raise ProtocolError(f"{label} binding path is invalid")
    path = declared_path if declared_path.is_absolute() else REPO_ROOT / declared_path
    data = _read_regular(path, label=label)
    observed = {
        "path": raw_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }
    if observed != dict(record):
        raise ProtocolError(f"{label} binding mismatch")
    return path, observed


def validate_evaluation_inputs(
    *,
    protocol_path: Path,
    scene: str,
    method_input_manifest_path: Path,
    prediction_manifest_path: Path,
    prediction_loader: Callable[[Path], object],
    office_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate every declared byte before invoking the prediction loader."""

    protocol, protocol_bytes = _load_json(protocol_path, label="protocol")
    expected_method_input = authorize_scene_run(
        protocol, scene, office_release=office_release
    )
    method_input, method_input_bytes = _load_json(
        method_input_manifest_path, label="method input manifest"
    )
    if method_input != expected_method_input:
        raise ProtocolError("method input manifest does not match frozen protocol")
    protocol_sources = protocol.get("source_bindings")
    method_sources = method_input.get("source_bindings")
    if not isinstance(protocol_sources, Mapping) or not isinstance(method_sources, Mapping):
        raise ProtocolError("protocol source bindings are invalid")
    source_records: list[tuple[Path, dict[str, object], str]] = []
    method_source_paths: dict[str, Path] = {}
    for prefix, records in (
        ("protocol", protocol_sources),
        ("method", method_sources),
    ):
        for role in sorted(records):
            path, observed = _validate_record(
                records[role], label=f"source binding {prefix}.{role}"
            )
            source_records.append((path, observed, f"source binding {prefix}.{role}"))
            if prefix == "method":
                method_source_paths[str(role)] = path
    if set(method_source_paths) != {
        "rgbd_export_manifest",
        "camera",
        "trajectory",
        "timestamps",
    }:
        raise ProtocolError("method source binding roles are invalid")
    export_path = method_source_paths["rgbd_export_manifest"]
    scene_root = export_path.parent
    rgbd_root = scene_root.parent
    if (
        scene_root.name != scene
        or method_source_paths["camera"] != rgbd_root / "cam_params.json"
        or method_source_paths["trajectory"] != scene_root / "traj.txt"
        or method_source_paths["timestamps"] != scene_root / "timestamps.csv"
    ):
        raise ProtocolError("method source binding layout is invalid")
    visits = method_input.get("visits")
    if not isinstance(visits, Mapping) or set(visits) != {"t0", "t1"}:
        raise ProtocolError("method visit inputs are invalid")

    def validate_visit_inputs() -> None:
        for visit_id in ("t0", "t1"):
            visit = visits[visit_id]
            if not isinstance(visit, Mapping):
                raise ProtocolError(f"{visit_id} visit input is invalid")
            observed = compute_window_input_sha256(
                rgbd_root,
                scene=scene,
                start_frame=int(visit["start_frame"]),
                end_frame=int(visit["end_frame"]),
            )
            if observed != visit.get("input_sha256"):
                raise ProtocolError(f"{visit_id} visit input hash mismatch")

    validate_visit_inputs()
    prediction_manifest, prediction_bytes = _load_json(
        prediction_manifest_path, label="prediction manifest"
    )
    if (
        set(prediction_manifest)
        != {
            "schema_version",
            "status",
            "protocol_id",
            "protocol_content_sha256",
            "scene",
            "method_input_manifest",
            "method_config",
            "outputs",
        }
        or prediction_manifest.get("schema_version") != 1
        or prediction_manifest.get("status") != "PASS"
        or prediction_manifest.get("protocol_id") != PROTOCOL_ID
        or prediction_manifest.get("scene") != scene
        or prediction_manifest.get("protocol_content_sha256")
        != protocol_content_sha256(protocol)
    ):
        raise ProtocolError("prediction manifest identity is invalid")
    declared_method = prediction_manifest["method_input_manifest"]
    _, observed_method = _validate_record(
        declared_method, label="method input manifest"
    )
    expected_method_record = {
        "path": str(method_input_manifest_path),
        "sha256": hashlib.sha256(method_input_bytes).hexdigest(),
        "byte_count": len(method_input_bytes),
    }
    if observed_method != expected_method_record:
        raise ProtocolError("method input manifest path is not the validated input")
    _config_path, config_record = _validate_record(
        prediction_manifest["method_config"], label="method config"
    )
    outputs = prediction_manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ProtocolError("prediction outputs are invalid")
    validated_outputs: dict[str, dict[str, object]] = {}
    output_paths: dict[str, Path] = {}
    for role in sorted(outputs):
        path, record = _validate_record(outputs[role], label=f"output binding {role}")
        output_paths[str(role)] = path
        validated_outputs[str(role)] = record

    loaded = {
        role: prediction_loader(path) for role, path in sorted(output_paths.items())
    }
    if len(loaded) != len(output_paths):
        raise ProtocolError("prediction loader did not consume every validated output")
    for path, expected, label in source_records:
        _, observed = _validate_record(expected, label=label)
        if observed != expected:
            raise ProtocolError(f"{label} changed during prediction loading")
    _, observed_config = _validate_record(config_record, label="method config")
    if observed_config != config_record:
        raise ProtocolError("method config changed during prediction loading")
    for role, path in sorted(output_paths.items()):
        _, observed = _validate_record(validated_outputs[role], label=f"output binding {role}")
        if observed != validated_outputs[role]:
            raise ProtocolError(f"output binding {role} changed during prediction loading")
    validate_visit_inputs()
    return {
        "schema_version": 1,
        "status": "EVALUATION_INPUTS_PASS",
        "protocol": {
            "path": str(protocol_path),
            "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "byte_count": len(protocol_bytes),
            "content_sha256": protocol_content_sha256(protocol),
        },
        "prediction_manifest": {
            "path": str(prediction_manifest_path),
            "sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "byte_count": len(prediction_bytes),
        },
        "outputs": validated_outputs,
    }


def _atomic_json(path: Path, value: object) -> None:
    data = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--method-input-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--office-release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    release = (
        _load_json(args.office_release, label="Office release")[0]
        if args.office_release
        else None
    )
    receipt = validate_evaluation_inputs(
        protocol_path=args.protocol,
        scene=args.scene,
        method_input_manifest_path=args.method_input_manifest,
        prediction_manifest_path=args.prediction_manifest,
        prediction_loader=lambda path: path.stat().st_size,
        office_release=release,
    )
    _atomic_json(args.output, receipt)
    print("EVALUATION_INPUTS_PASS")
    return 0


__all__ = ["main", "validate_evaluation_inputs"]


if __name__ == "__main__":
    raise SystemExit(main())
