"""Bind and evaluate completed native FIT/CAL captures using ScanNet200."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"

from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.contracts import atomic_write_json
from src.static_ovmap.module_validation.semantic_study import prepare_semantic_manifest
from src.static_ovmap.module_validation.study_scene import (
    audit_native_export,
    bind_scene,
    evaluate_predictions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = Path(config["runtime_config"])
    if not runtime_path.is_absolute():
        runtime_path = ROOT / runtime_path
    runtime = json.loads(runtime_path.read_text())
    lock_path = Path(runtime["data_root"]) / "acquisition_lock.json"
    lock = json.loads(lock_path.read_text())
    allowed = [row["scene_id"] for row in lock["selected"] if row["role"] in {"fit", "cal"}]
    scenes = allowed if args.scenes is None else args.scenes
    if not set(scenes) <= set(allowed):
        raise ValueError("initial baseline evaluation is restricted to FIT/CAL; SELECT follows frozen branch settings")
    receipts = {}
    for scene in scenes:
        data, targets = bind_scene(scene, runtime, config)
        output = data["output"]
        parity = audit_native_export(data["native"], targets, Path(runtime["upstream"]), output / "native_export_parity")
        rows = evaluate_predictions([data["native"]], {scene: targets}, Path(runtime["upstream"]), output / "baseline_evaluation")
        semantic = prepare_semantic_manifest(data["capture_path"], data["native"], output / "semantic_requests.json")
        receipts[scene] = {"native_manifest": file_identity(data["native_manifest_path"]),
            "parity": parity, "metrics": rows, "semantic_request_count": len(semantic["requests"])}
        print(json.dumps({"scene_id": scene, "status": "COMPLETE", "metrics": rows,
            "semantic_request_count": len(semantic["requests"])}), flush=True)
    atomic_write_json(Path(config["study_root"]) / "baseline_fitcal_receipt.json", {
        "status": "COMPLETE", "scenes": receipts, "requested_scenes": scenes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": file_identity(args.config), "runtime_config": file_identity(runtime_path),
        "acquisition_lock": file_identity(lock_path), "command": sys.argv})


if __name__ == "__main__":
    main()
