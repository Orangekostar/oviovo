"""Execute the frozen at-most-four-row ScanNet confirmation package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(runtime["cuda_device"]), "HF_MODULES_CACHE": runtime["hf_modules_cache"],
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ[name] = "8"
    from src.static_ovmap.module_validation.confirmation_pipeline import (
        run_confirmation,
    )

    result = run_confirmation(runtime, config, config_path,
        args.selection_receipt or Path(config["study_root"]) / "selection/receipt.json")
    print(json.dumps({"status": result["status"], "confirmation_status": result["confirmation_status"], "rows": len(result["rows"])}), flush=True)


if __name__ == "__main__":
    main()
