"""Preserve validated CPU frames and finish only missing native masks on GPU."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_ovimap_native import (
    _artifact,
    _atomic_json,
    build_scene_commands,
    default_config,
    run_frontend,
)


def validated_frame(root, scene_root, frame, weights, config_hash):
    mask = root / "frontend" / f"frame{frame:06d}.png"
    diagnostic = root / "frontend/frame_diagnostics" / f"frame{frame:06d}.json"
    if not mask.exists() or not diagnostic.exists():
        return None
    try:
        record = json.loads(diagnostic.read_text())
    except json.JSONDecodeError:
        return None  # A running producer may not yet have closed its diagnostic.
    if (
        record["weights_sha256"] != weights
        or record["config_sha256"] != config_hash
        or record["confidence_threshold"] != 0.5
        or record["input"] != str(scene_root / "results" / f"frame{frame:06d}.jpg")
    ):
        raise ValueError("native frame model/input binding differs")
    pixels = np.asarray(Image.open(mask))
    if (
        pixels.ndim != 2
        or list(pixels.shape) != record["shape"]
        or str(pixels.dtype) != record["dtype"]
    ):
        raise ValueError("native frame mask/diagnostic mismatch")
    if sorted(np.unique(pixels[pixels > 0]).tolist()) != record["instance_ids"]:
        raise ValueError("native frame mask IDs differ from diagnostic")
    return [_artifact(mask), _artifact(diagnostic)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("room0", "room1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    config = default_config(args.output)
    weights, config_hash = (
        _artifact(config.cropformer_weights)["sha256"],
        _artifact(config.cropformer_config)["sha256"],
    )
    scene_root = config.data_root / args.scene
    ready = {}
    for frame in range(0, 2000, 10):
        artifacts = validated_frame(
            args.output, scene_root, frame, weights, config_hash
        )
        if artifacts is not None:
            ready[frame] = artifacts
    missing = [frame for frame in range(0, 2000, 10) if frame not in ready]
    print(
        args.scene,
        len(ready),
        "validated retained frames;",
        len(missing),
        "remaining",
        flush=True,
    )
    if args.inspect_only:
        return
    # A second producer in this canonical output must already have stopped.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if (
            any(part.endswith("demo_from_dirs.py") for part in command[:3])
            and str(args.output / "frontend") in command
        ):
            raise RuntimeError(
                "canonical CPU producer is still live; stop it before GPU completion"
            )
    staging = args.output / "gpu_completion"
    staging.mkdir(exist_ok=True)
    (staging / "frontend").mkdir(exist_ok=True)
    (staging / "logs").mkdir(exist_ok=True)
    record_file = staging / "completion_plan.json"
    if record_file.exists():
        original = json.loads(record_file.read_text())
        for item in original["retained_artifacts"]:
            if _artifact(Path(item["path"])) != item:
                raise ValueError("retained CPU evidence changed")
        original_missing = original["gpu_frames"]
    else:
        original_missing = missing
        _atomic_json(
            record_file,
            {
                "scene": args.scene,
                "retained_frames": sorted(ready),
                "gpu_frames": missing,
                "retained_artifacts": [
                    item for values in ready.values() for item in values
                ],
                "reason": "GPU capacity recovered; retain all completed CPU evidence, same config, weights and source frames",
            },
        )
    pending = [
        frame
        for frame in missing
        if validated_frame(staging, scene_root, frame, weights, config_hash) is None
    ]
    if pending:
        commands = build_scene_commands(
            config, scene=args.scene, start=0, end=2000, step=10, attempt_root=staging
        )
        command = list(commands.frontend)
        begin, end = command.index("--input") + 1, command.index("--output")
        command[begin:end] = [
            str(scene_root / "results" / f"frame{frame:06d}.jpg") for frame in pending
        ]
        commands = replace(
            commands, frame_ids=pending, frontend=(*command, "MODEL.DEVICE", "cuda")
        )
        record = run_frontend(config, commands)
        _atomic_json(
            staging / f"execution_{pending[0]:06d}_{pending[-1]:06d}.json", record
        )
    for frame in missing:
        artifacts = validated_frame(staging, scene_root, frame, weights, config_hash)
        if artifacts is None:
            raise ValueError("GPU continuation did not produce a complete frame")
        for item in artifacts:
            source = Path(item["path"])
            target = args.output / source.relative_to(staging)
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_name(target.name + ".gpu-partial")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    for values in ready.values():
        for item in values:
            if _artifact(Path(item["path"])) != item:
                raise ValueError("retained completed evidence changed during continuation")
    artifacts = []
    for frame in range(0, 2000, 10):
        checked = validated_frame(args.output, scene_root, frame, weights, config_hash)
        if checked is None:
            raise ValueError("combined frontend is incomplete")
        artifacts.extend(checked)
    _atomic_json(
        args.output / "frontend_receipt.json",
        {
            "status": "PASS",
            "mask_count": 200,
            "artifacts": artifacts,
            "execution_mode": "VALIDATED_CPU_PREFIX_GPU_COMPLETION",
            "cpu_frames": [f for f in range(0, 2000, 10) if f not in original_missing],
            "gpu_frames": original_missing,
            "completion_plan": _artifact(record_file),
            "execution_records": [
                _artifact(p) for p in sorted(staging.glob("execution_*.json"))
            ],
            "same_model_config_weights_and_input_grid": True,
            "original_cpu_command": json.loads(
                (args.output / "preparation_registry.json").read_text()
            )["frontend_command"],
            "cpu_log": _artifact(args.output / "logs/frontend.log"),
        },
    )
    print(args.scene, "FRONTEND_PASS", 200, flush=True)


if __name__ == "__main__":
    main()
