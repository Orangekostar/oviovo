#!/usr/bin/env python3
"""Test area fallback beside the frozen compact-table v1 experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from static_ovmap.cvpr_compact.area_fallback_experiment import (
    lock_predictions, pool_results, preflight, run_regions, score_scene,
)
from static_ovmap.recovery_wave2.binding import read


def score_task(arguments):
    return score_scene(*arguments)["scene"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "regions", "predictions", "evaluation", "pools", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("workers must be in 1..3")
    if args.phase == "all":
        binding = read(args.parent / "resolved_inputs.json")
        interpreters = {"preflight": binding["fc"]["python"], "regions": binding["fc"]["python"]}
        for phase in ("preflight", "regions", "predictions", "evaluation", "pools", "report"):
            python = interpreters.get(phase, "/home/ww/miniconda3/envs/ovimap-map/bin/python")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "3" if phase == "regions" else "1",
                   "MKL_NUM_THREADS": "3" if phase == "regions" else "1", "OPENBLAS_NUM_THREADS": "3" if phase == "regions" else "1"}
            subprocess.run([python, str(Path(__file__).resolve()), "--parent", str(args.parent), "--output", str(args.output),
                            "--phase", phase, "--workers", str(args.workers)], cwd=REPO, env=env, check=True)
        return
    if args.phase == "preflight":
        preflight(args.parent, args.output)
    elif args.phase == "regions":
        run_regions(args.parent, args.output)
    elif args.phase == "predictions":
        for scene in read(args.output / "experiment.json")["scenes"]:
            lock_predictions(args.parent, args.output, scene)
    elif args.phase == "evaluation":
        scenes = list(read(args.output / "experiment.json")["scenes"])
        for scene in scenes:
            if read(args.output / "predictions" / scene / "receipt.json")["status"] != "PREDICTIONS_LOCKED":
                raise ValueError("all 26 prediction locks must exist before v2 evaluation starts")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(score_task, [(args.parent, args.output, scene) for scene in scenes]))
    elif args.phase == "pools":
        pool_results(args.parent, args.output)
    else:
        from static_ovmap.cvpr_compact.area_fallback_report import build_report
        build_report(args.parent, args.output)


if __name__ == "__main__":
    main()
