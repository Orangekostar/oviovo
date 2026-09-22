"""Prepare locked development annotations or per-model ScanNet200 text caches."""

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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.static_ovmap.module_validation.contracts import atomic_write_json
from src.static_ovmap.module_validation.scannet_runtime import development_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("annotations", "texts"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ["HF_MODULES_CACHE"] = config["hf_modules_cache"]
    upstream = Path(config["upstream"])
    if args.phase == "texts":
        from src.static_ovmap.module_validation.scannet_study import prepare_text_cache

        if args.model_path is None:
            parser.error("--model-path is required for texts")
        # CPU text-only preprocessing leaves occupied study GPUs untouched.
        path = prepare_text_cache(upstream, args.model_path, args.output, device="cpu")
        print(json.dumps({"status": "COMPLETE", "text_cache": str(path)}))
        return
    from src.static_ovmap.module_validation.boundary_jobs import file_identity
    from src.static_ovmap.module_validation.scannet_ground_truth import (
        prepare_ground_truth,
    )

    data = Path(config["data_root"])
    lock_path = data / "acquisition_lock.json"
    locked = json.loads(lock_path.read_text())
    receipts = {}
    for row in development_rows(locked):
        scene = row["scene_id"]
        path = prepare_ground_truth(upstream, data / "scans" / scene, scene,
            data / "scannetv2-labels.combined.tsv", args.output / scene)
        receipts[scene] = {"role": row["role"], **file_identity(path)}
        print(f"{scene}: unchanged native annotation conversion complete", flush=True)
    atomic_write_json(args.output / "receipt.json", {"status": "COMPLETE", "scenes": receipts,
        "acquisition_lock": file_identity(lock_path), "confirmation": "DEFERRED_UNTIL_FROZEN_SELECTION"})


if __name__ == "__main__":
    main()
