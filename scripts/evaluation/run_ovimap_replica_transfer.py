#!/usr/bin/env python3
"""Execute the frozen six-method Replica transfer and report actual results."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001"))
    parser.add_argument("--phase", required=True, choices=("bind", "capture", "text", "semantic", "prepare", "query", "fuse", "evaluate", "report", "all"))
    parser.add_argument("--scene")
    parser.add_argument("--method")
    parser.add_argument("--model", choices=("native", "siglip2"))
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ[name] = "8"
    from src.static_ovmap.replica_transfer.protocol import bind, require_transfer

    path = args.config or bind(args.output_root)
    config = require_transfer(path, args.scene, args.method)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(config["runtime"]["cuda_device"]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_MODULES_CACHE=config["runtime"]["hf_modules_cache"])
    from src.static_ovmap.replica_transfer.parallel import delegated_scene

    delegated = delegated_scene(config, path, args.scene, args.phase)
    if delegated is not None:
        result = delegated
    elif args.phase == "bind":
        result = {"status": "COMPLETE", "config": str(path)}
    elif args.phase == "capture":
        import fcntl

        from src.static_ovmap.composition_study.io import write_once
        from src.static_ovmap.replica_transfer.capture import capture_one
        from src.static_ovmap.replica_transfer.protocol import bind_capture_inputs

        rows = []
        for scene in ([args.scene] if args.scene else config["scenes"]):
            row, native = bind_capture_inputs(config, scene)
            with Path(config["gpu_lock"]).open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                receipt = capture_one(config["runtime"], row, native, Path(config["attempt_root"]) / "native")
            rows.append({"scene_id": scene, "status": receipt["status"], "completed_count": receipt["completed_count"], "receipt": str(Path(config["attempt_root"]) / "native" / scene / "mapping_job/receipt.json")})
        result = {"status": "COMPLETE", "captures": rows}
        if args.scene is None:
            write_once(Path(config["attempt_root"]) / "capture_receipt.json", result)
    else:
        from src.static_ovmap.replica_transfer.jobs import run_phase

        result = run_phase(config, path, args.phase, scene=args.scene, method=args.method, model=args.model)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
