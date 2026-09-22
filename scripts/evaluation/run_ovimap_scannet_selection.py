"""Apply frozen SELECT decisions, eligible budget curves, and final holdout lock."""

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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("modules", "combinations", "curves", "freeze", "all"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    runtime_path = Path(config["runtime_config"])
    runtime = json.loads((runtime_path if runtime_path.is_absolute() else ROOT / runtime_path).read_text())
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(runtime["cuda_device"]), "HF_MODULES_CACHE": runtime["hf_modules_cache"],
                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ[key] = "8"
    from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
    from src.static_ovmap.module_validation.selection_execution import (
        freeze_selection,
        prepare_selection,
        run_budget_curves,
    )
    from src.static_ovmap.module_validation.study_execution import verify_receipt

    output = Path(config["study_root"]) / "selection"
    if args.phase in {"modules", "all"}:
        result = prepare_selection(runtime, config, config_path)
        print(json.dumps(result["decision"]), flush=True)
    if args.phase in {"combinations", "all"}:
        from src.static_ovmap.module_validation.combination_pipeline import (
            run_combinations,
        )

        results = run_combinations(runtime, config, config_path)
        print(json.dumps({"combinations": [{"method_id": row["method_id"], "status": row["disposition"]}
                                           for row in results]}), flush=True)
    if args.phase in {"curves", "all"}:
        decision = verify_receipt(output / "module_receipt.json")["decision"]
        if decision["budget_curves"]["status"] == "REQUIRED":
            lock = Path(runtime["output_root"]).parent / f".visual-gpu-{runtime['cuda_device']}.lock"
            with lock.open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                require_idle_gpu(str(runtime["cuda_device"]), output)
                result = run_budget_curves(runtime, config, config_path)
        else:
            result = run_budget_curves(runtime, config, config_path)
        print(json.dumps({"budget_curves": result["gate_status"]}), flush=True)
    if args.phase in {"freeze", "all"}:
        decision = verify_receipt(output / "module_receipt.json")["decision"]
        pending = [method for method in decision["combination_plan"]["required"]
                   if not (output / "combinations" / method / "receipt.json").is_file()]
        if pending:
            print(json.dumps({"status": "PENDING_REQUIRED_COMBINATIONS", "methods": pending}), flush=True)
            return 2
        result = freeze_selection(runtime, config, config_path)
        print(json.dumps({key: result[key] for key in ("status", "final_candidate", "science_status", "confirmation_status")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
