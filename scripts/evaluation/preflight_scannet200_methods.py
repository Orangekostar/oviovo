#!/usr/bin/env python3
"""Preflight ScanNet200 method assets without executing a baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
METHODS = {"OPENFUSION", "OVIMAP", "CONCEPTGRAPHS", "DUALMAP"}
SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _resolve_asset(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _validate_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if config.get("schema_version") != 1 or config.get("dataset") != "ScanNet200":
        raise ValueError("method config must be ScanNet200 schema version 1")
    methods = config.get("methods")
    if not isinstance(methods, dict) or set(methods) != METHODS:
        raise ValueError("method config must contain exactly four frozen methods")
    required = {
        "upstream_commit",
        "mode",
        "runner_kind",
        "environment",
        "required_assets",
        "supported_metrics",
        "unsupported_metrics",
        "gpu_hint",
        "argv_template",
    }
    for name, record in methods.items():
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"method config is incomplete: {name}")
        if len(str(record["upstream_commit"])) != 40:
            raise ValueError(f"method upstream commit must be full: {name}")
    return methods


def _asset_report(method: dict[str, Any]) -> dict[str, Any]:
    verified = []
    failures = []
    for asset in method["required_assets"]:
        name = str(asset.get("name", ""))
        path = _resolve_asset(str(asset.get("path", "")))
        expected = str(asset.get("sha256", ""))
        if not path.is_file():
            failures.append({"name": name, "path": str(path), "kind": "missing"})
            continue
        actual = _sha256(path)
        if actual != expected:
            failures.append(
                {
                    "name": name,
                    "path": str(path),
                    "kind": "hash_mismatch",
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
            continue
        verified.append({"name": name, "path": str(path), "sha256": actual})
    return {
        "asset_status": "READY" if not failures else "BLOCKED",
        "verified_assets": verified,
        "failures": failures,
    }


def _manifest_status(path: Path) -> tuple[bool, dict[str, Any] | None]:
    if not path.is_file():
        return False, None
    manifest = _read_object(path, "manifest")
    if (
        manifest.get("manifest_id") != "scannet200_5_static_v1"
        or manifest.get("dataset") != "ScanNet200"
        or tuple(item.get("scene_id") for item in manifest.get("scenes", ())) != SCENES
    ):
        raise ValueError("manifest does not match the frozen ScanNet200 five-scene contract")
    return True, manifest


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
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.method_config.resolve()
    config = _read_object(config_path, "method config")
    methods = _validate_config(config)
    manifest_ready, _ = _manifest_status(args.manifest.resolve())
    reports = {}
    assets_ready = True
    for name in sorted(methods):
        asset_report = _asset_report(methods[name])
        assets_ready &= asset_report["asset_status"] == "READY"
        reports[name] = {
            "upstream_commit": methods[name]["upstream_commit"],
            "mode": methods[name]["mode"],
            "runner_kind": methods[name]["runner_kind"],
            **asset_report,
        }
    if not manifest_ready:
        status = "BLOCKED"
        reason = "SCANNET_DATA_NOT_VALIDATED"
    elif not assets_ready:
        status = "BLOCKED"
        reason = "METHOD_ASSETS_NOT_READY"
    else:
        status = "READY"
        reason = None
    report = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "method_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "manifest": (
            {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest)}
            if manifest_ready
            else {"path": str(args.manifest.resolve()), "sha256": None}
        ),
        "methods": reports,
    }
    _write_json_atomic(args.output.resolve(), report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = preflight(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {"schema_version": 1, "status": "BLOCKED", "reason": str(error)}
        _write_json_atomic(args.output.resolve(), report)
    if report["status"] != "READY":
        print(str(report["reason"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
