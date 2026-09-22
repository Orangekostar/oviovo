"""Execute the frozen ScanNet semantic study, with explicit resumable phases."""

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

from src.static_ovmap.module_validation.semantic_pipeline import (
    calibrate_semantic,
    prepare_semantic_scene,
    run_semantic_select,
    study_roles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "calibrate", "select", "all"), required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--reuse-inference", action="store_true", help="Require existing model receipts; perform CPU readout only")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    roles, _ = study_roles(runtime)
    if args.scenes and args.phase != "prepare":
        raise ValueError("scene subsets are allowed only for FIT/CAL evidence preparation")
    if args.phase in {"prepare", "all"}:
        scenes = args.scenes or [*roles["fit"], *roles["cal"]]
        if not set(scenes) <= set(roles["fit"] + roles["cal"]):
            raise ValueError("prepare accepts only the frozen FIT/CAL scenes")
        for scene in scenes:
            result = prepare_semantic_scene(scene, "fit" if scene in roles["fit"] else "cal",
                runtime, config, config_path, run_models=not args.reuse_inference)
            print(json.dumps({"phase": "prepare", "scene_id": scene, "status": result["status"]}), flush=True)
    if args.phase in {"calibrate", "all"}:
        result = calibrate_semantic(runtime, config, config_path)
        print(json.dumps({key: result[key] for key in ("status", "teacher_id", "learned_status")}), flush=True)
    if args.phase in {"select", "all"}:
        result = run_semantic_select(runtime, config, config_path, run_models=not args.reuse_inference)
        print(json.dumps({key: result[key] for key in ("status", "teacher_id", "learned_status")}), flush=True)


if __name__ == "__main__":
    main()
