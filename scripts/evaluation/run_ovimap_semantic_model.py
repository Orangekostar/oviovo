"""Encode the bound ScanNet S requests with exactly one frozen visual model."""

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
    parser.add_argument("--model", choices=("native", "siglip2", "wow"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = Path(config["runtime_config"])
    if not runtime_path.is_absolute():
        runtime_path = ROOT / runtime_path
    runtime = json.loads(runtime_path.read_text())
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(runtime["cuda_device"]),
        "HF_MODULES_CACHE": runtime["hf_modules_cache"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "8"
    from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
    from src.static_ovmap.module_validation.semantic_models import (
        encode_semantic_requests,
    )

    lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(runtime["cuda_device"]), args.output)
        result = encode_semantic_requests(args.request_manifest, args.model, config, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
