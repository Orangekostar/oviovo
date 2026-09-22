"""Prepare finite native geometry pools for completed FIT/CAL captures on CPU."""

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

from src.static_ovmap.module_validation.geometry_study import prepare_geometry_pool
from src.static_ovmap.module_validation.study_scene import bind_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    lock = json.loads((Path(runtime["data_root"]) / "acquisition_lock.json").read_text())
    allowed = {row["scene_id"] for row in lock["selected"] if row["role"] in {"fit", "cal"}}
    if not set(args.scenes) <= allowed:
        raise ValueError("initial geometry preparation is limited to FIT/CAL scenes")
    for scene in args.scenes:
        data, _targets = bind_scene(scene, runtime, config)
        receipt = prepare_geometry_pool(data, data["output"] / "geometry/pool")
        print(json.dumps({"scene_id": scene, "receipt": str(receipt), "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
