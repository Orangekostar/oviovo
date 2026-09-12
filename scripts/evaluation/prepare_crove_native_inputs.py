"""Resume original CropFormer input preparation, with optional native mapping.

CPU frontend execution preserves the original input resolution, crops and weights.
Completed frontend evidence is reused only after per-frame file validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_ovimap_native import (
    _artifact,
    _atomic_json,
    audit_scene,
    build_scene_commands,
    default_config,
    preflight,
    run_frontend,
    run_mapping,
    write_scene_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("room0", "room1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = default_config(args.output)
    commands = build_scene_commands(
        config, scene=args.scene, start=0, end=2000, step=10, attempt_root=args.output
    )
    commands = replace(
        commands, frontend=(*commands.frontend, "MODEL.DEVICE", args.frontend_device)
    )
    for name in (
        "frontend",
        "geometric_segments",
        "mapping",
        "intermediate_segments",
        "native_audit",
        "logs",
    ):
        (args.output / name).mkdir(exist_ok=True)
    settings = {
        "scene": args.scene,
        "source_frame_ids": commands.frame_ids,
        "frontend_device": args.frontend_device,
        "frontend_command": list(commands.frontend),
    }
    registry = args.output / "preparation_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("existing preparation differs; use a separate output")
    else:
        _atomic_json(registry, settings)
    preflight_record = preflight(config, commands, scene=args.scene)
    frontend_receipt = args.output / "frontend_receipt.json"
    if frontend_receipt.exists():
        frontend_record = json.loads(frontend_receipt.read_text())
        for item in frontend_record["artifacts"]:
            if _artifact(Path(item["path"])) != item:
                raise ValueError("cached frontend artifact changed")
    else:
        frontend_record = run_frontend(config, commands)
        artifacts = []
        for frame in commands.frame_ids:
            artifacts.append(
                _artifact(args.output / "frontend" / f"frame{frame:06d}.png")
            )
            artifacts.append(
                _artifact(
                    args.output
                    / "frontend/frame_diagnostics"
                    / f"frame{frame:06d}.json"
                )
            )
        frontend_record["artifacts"] = artifacts
        _atomic_json(frontend_receipt, frontend_record)
    print("FRONTEND_PASS", frontend_receipt, flush=True)
    if not args.map:
        return
    mapping_receipt = args.output / "mapping_receipt.json"
    if mapping_receipt.exists():
        mapping_record = json.loads(mapping_receipt.read_text())
    else:
        mapping_record = run_mapping(config, commands)
        _atomic_json(mapping_receipt, mapping_record)
    audit_record = audit_scene(commands)
    if audit_record["status"] != "PASS":
        raise ValueError("native scene audit failed")
    result = write_scene_manifest(
        commands,
        scene=args.scene,
        stage="room",
        state="MAPPING_PASS",
        preflight_record=preflight_record,
        frontend_record=frontend_record,
        mapping_record=mapping_record,
        audit_record=audit_record,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
