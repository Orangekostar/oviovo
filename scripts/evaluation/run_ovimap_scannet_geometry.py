"""Run the frozen native-surface ScanNet geometry experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"

from src.static_ovmap.module_validation.geometry_pipeline import (
    calibrate_geometry,
    prepare_geometry_scene,
    run_geometry_select,
)
from src.static_ovmap.module_validation.study_execution import roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "calibrate", "select", "all"), required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--reuse-inference", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    split, _ = roles(runtime)
    if args.scenes and args.phase != "prepare":
        raise ValueError("G scene subsets are only allowed for FIT/CAL preparation")
    if args.phase in {"prepare", "all"}:
        scenes = args.scenes or [*split["fit"], *split["cal"]]
        if not set(scenes) <= set(split["fit"] + split["cal"]):
            raise ValueError("initial G preparation only accepts FIT/CAL scenes")
        for scene in scenes:
            result = prepare_geometry_scene(scene, "fit" if scene in split["fit"] else "cal", runtime, config,
                config_path, run_models=not args.reuse_inference)
            print(json.dumps({"phase": "prepare", "scene_id": scene, "status": result["status"]}), flush=True)
    if args.phase in {"calibrate", "all"}:
        result = calibrate_geometry(runtime, config, config_path, run_models=not args.reuse_inference)
        print(json.dumps({key: result[key] for key in ("status", "learned_status", "margin")}), flush=True)
    if args.phase in {"select", "all"}:
        result = run_geometry_select(runtime, config, config_path, run_models=not args.reuse_inference)
        print(json.dumps({key: result[key] for key in ("status", "learned_status", "margin")}), flush=True)


if __name__ == "__main__":
    main()
