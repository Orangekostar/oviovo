"""Run one real historical integration boundary in its model environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.static_ovmap.module_validation.boundary_jobs import (
    capture_inputs,
    file_identity,
    geometry_smoke,
    query_smoke,
    semantic_smoke,
    verify_capture,
)
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("capture", "semantic", "geometry", "query"), required=True
    )
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.resolved_config.read_text())
    resolved = config.get("resolved_config", config)
    config = resolved["historical_smoke"]
    os.environ["HF_MODULES_CACHE"] = str(config.get("hf_modules_cache", args.output.parent / "hf_modules"))
    manifest_path = config["capture_manifest"]
    started = time.monotonic()
    manifest = verify_capture(manifest_path)
    if args.phase == "capture":
        result = {
            "status": "COMPLETE",
            "scientific_result": False,
            "scope": "verify existing native capture payload and frame schedule",
            "scene_count": 1,
            "frame_count": len(manifest["frames"]),
            "native_replay_performed": False,
        }
    elif args.phase == "geometry":
        result = geometry_smoke(manifest_path)
    elif args.phase == "query":
        result = query_smoke(
            manifest_path, config["feature_root"], config["text_cache"]
        )
    else:
        result = semantic_smoke(manifest_path, config)
    result.update(
        {
            "artifact_type": "OVIMAP_EXECUTED_BOUNDARY_SMOKE",
            "schema_version": 1,
            "phase": args.phase,
            "elapsed_seconds": time.monotonic() - started,
            "config_identity": canonical_digest(resolved),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "Pillow")},
            },
            "command": [sys.executable, *sys.argv],
            "capture_manifest": file_identity(manifest_path),
        }
    )
    result["input_identities"] = [
        file_identity(row["path"]) for row in capture_inputs(manifest_path, manifest)
    ] + result.get("input_identities", [])
    if config.get("native_replay_receipt"):
        replay_path = Path(config["native_replay_receipt"])
        replay = json.loads(replay_path.read_text())
        if replay.get("status") != "COMPLETE" or replay[
            "capture_manifest"
        ] != file_identity(manifest_path):
            raise ValueError("native replay receipt does not bind this capture")
        result["input_identities"].append(file_identity(replay_path))
        result["native_replay_evidence"] = str(replay_path)
    if args.phase == "semantic":
        # Hash actual files. A checkpoint hash is never described as a directory hash.
        for key in (
            "native_model",
            "siglip2_model",
            "wow_model",
            "name_model",
            "wow_code",
        ):
            for path in sorted(Path(config[key]).rglob("*")):
                if path.is_file() and not any(
                    part in {".git", ".cache", "__pycache__"} for part in path.parts
                ):
                    result["input_identities"].append(file_identity(path))
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": result["status"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
