#!/usr/bin/env python3
"""Generate deterministic, non-executing ScanNet200 baseline run specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


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
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_sources(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    config_path = args.method_config.resolve()
    manifest_path = args.manifest.resolve()
    config = _read_object(config_path, "method config")
    manifest = _read_object(manifest_path, "manifest")
    preflight = _read_object(args.preflight.resolve(), "preflight")
    methods = config.get("methods")
    if not isinstance(methods, dict) or set(methods) != METHODS:
        raise ValueError("method config must contain the exact four-method set")
    if tuple(item.get("scene_id") for item in manifest.get("scenes", ())) != SCENES:
        raise ValueError("manifest must contain the frozen five-scene order")
    if preflight.get("status") != "READY":
        raise ValueError("method preflight must be READY before generating run specs")
    if preflight.get("method_config", {}).get("sha256") != _sha256(config_path):
        raise ValueError("preflight method config hash mismatch")
    if preflight.get("manifest", {}).get("sha256") != _sha256(manifest_path):
        raise ValueError("preflight manifest hash mismatch")
    preflight_methods = preflight.get("methods")
    if not isinstance(preflight_methods, dict) or set(preflight_methods) != METHODS:
        raise ValueError("preflight method set is incomplete")
    blocked = sorted(
        name
        for name, record in preflight_methods.items()
        if record.get("asset_status") != "READY"
    )
    if blocked:
        raise ValueError(f"all method assets must be READY: {blocked}")
    return config, manifest, preflight


def generate(args: argparse.Namespace) -> None:
    config, manifest, _ = _validate_sources(args)
    target = args.spec_dir.resolve()
    if target.exists():
        if any(target.iterdir()):
            raise ValueError(f"spec directory is non-empty: {target}")
        if not args.replace_empty:
            raise ValueError(f"spec directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    manifest_path = args.manifest.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    try:
        for name in sorted(METHODS):
            method = config["methods"][name]
            method_output = output_root / name.lower()
            argv = [
                str(value).format(
                    manifest=str(manifest_path),
                    output=str(method_output),
                )
                for value in method["argv_template"]
            ]
            spec = {
                "schema_version": 1,
                "status": "READY",
                "method": name,
                "mode": method["mode"],
                "upstream_commit": method["upstream_commit"],
                "runner_kind": method["runner_kind"],
                "environment": method["environment"],
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                },
                "scene_order": list(SCENES),
                "input_root": str(input_root),
                "output_root": str(method_output),
                "command_argv": argv,
                "required_metric_contract": {
                    "supported": method["supported_metrics"],
                    "unsupported": method["unsupported_metrics"],
                },
                "gpu_allocation_hint": method["gpu_hint"],
            }
            _write_json(temporary / f"{name.lower()}.json", spec)
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--replace-empty", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        generate(_parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
