"""Encode only the actual final-mask native region requests for one G row."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(runtime["cuda_device"]), "HF_MODULES_CACHE": runtime["hf_modules_cache"],
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "8"
    from src.static_ovmap.module_validation.geometry_models import (
        encode_static_requests,
    )
    from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu

    lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(runtime["cuda_device"]), args.output)
        result = encode_static_requests(args.request_manifest, config,
            Path(config["study_root"]) / "geometry/native_request_cache", args.output)
    print(json.dumps({key: result[key] for key in ("status", "request_count", "physical_attempts_this_invocation")}))


if __name__ == "__main__":
    main()
