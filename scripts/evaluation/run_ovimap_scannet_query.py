"""Execute the frozen causal native ScanNet query study, one GPU at a time."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "calibrate", "select", "all"), required=True)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(runtime["cuda_device"]), "HF_MODULES_CACHE": runtime["hf_modules_cache"],
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ[name] = "8"
    from src.static_ovmap.module_validation.query_pipeline import (
        CONTROLS,
        calibrate_query,
        run_query_scene,
        run_query_select,
    )
    from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
    from src.static_ovmap.module_validation.study_execution import roles

    split, _ = roles(runtime)
    if args.scenes and args.phase != "prepare":
        raise ValueError("query scene subsets are limited to FIT/CAL preparation")
    lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
    context = lock.open("a") if args.phase != "calibrate" else nullcontext()
    with context as handle:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
            require_idle_gpu(str(runtime["cuda_device"]), Path(config["study_root"]) / "query")
        if args.phase in {"prepare", "all"}:
            scenes = args.scenes or [*split["fit"], *split["cal"]]
            if not set(scenes) <= set(split["fit"] + split["cal"]):
                raise ValueError("query preparation is limited to FIT/CAL")
            for scene in scenes:
                role = "fit" if scene in split["fit"] else "cal"
                result = run_query_scene(scene, role, "Q_RANDOM_TRACE", runtime, config, config_path,
                                         budget=512 if role == "fit" else 256)
                print(json.dumps({"scene_id": scene, "status": result["status"], "logical_cost": result["logical_cost"]}), flush=True)
                if role == "cal":
                    for method in CONTROLS:
                        result = run_query_scene(scene, role, method, runtime, config, config_path)
                        print(json.dumps({"scene_id": scene, "method_id": method, "status": result["status"]}), flush=True)
        if args.phase in {"calibrate", "all"}:
            result = calibrate_query(runtime, config, config_path)
            print(json.dumps({key: result[key] for key in ("status", "learned_status", "comparator_id")}), flush=True)
        if args.phase in {"select", "all"}:
            result = run_query_select(runtime, config, config_path)
            print(json.dumps({key: result[key] for key in ("status", "learned_status", "comparator_id")}), flush=True)


if __name__ == "__main__":
    main()
