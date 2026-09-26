#!/usr/bin/env python3
"""Run the bound complementary composition study and its real leaf jobs."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT
        / "docs/paper/static_ovmap/complementary_composition_v1/spec/PROTOCOL_SPEC.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/shared/ww/ovimap-complementary-composition-v1"),
    )
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "bind",
            "prepare-cal",
            "calibrate",
            "compose-cal",
            "freeze",
            "regression",
            "confirm",
            "report",
            "publish",
            "all",
        ),
    )
    parser.add_argument(
        "--job",
        choices=("query", "static", "evaluate"),
        help="Internal real leaf; also accepts --resolved-config",
    )
    parser.add_argument("--scene")
    parser.add_argument("--method")
    args = parser.parse_args()
    # Set thread and device environment before importing Torch/NumPy-backed jobs.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        os.environ[name] = "8"
    from src.static_ovmap.composition_study.binding import bind
    from src.static_ovmap.composition_study.io import read_json

    config_path = args.resolved_config or bind(args.spec, args.output_root)
    config = read_json(config_path)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(config["runtime"]["cuda_device"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_MODULES_CACHE=config["runtime"]["hf_modules_cache"],
    )
    if args.phase == "bind":
        print(
            json.dumps(
                {
                    "resolved_config": str(config_path),
                    "binding_key": config["binding_key"],
                }
            ),
            flush=True,
        )
        return 0
    from src.static_ovmap.composition_study.pipeline import run_leaf, run_phase

    if args.job:
        if args.scene is None or (args.job != "static" and args.method is None):
            parser.error(
                "leaf jobs require --scene and query/evaluate require --method"
            )
        result = run_leaf(config, args.phase, args.job, args.scene, args.method)
    else:
        if args.scene is not None or args.method is not None:
            parser.error(
                "scene/method subsets are only accepted for authorized leaf jobs"
            )
        result = run_phase(config, Path(config_path), args.phase)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETE", "FROZEN", "PUSH_VERIFIED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
