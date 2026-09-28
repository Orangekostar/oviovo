"""Hardware-only scene scheduling; all scientific settings remain frozen."""

import copy
import json
import time
from pathlib import Path

from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)

from .protocol import require_transfer


def worker_config(original, gpu):
    result = copy.deepcopy(original)
    result["runtime"]["cuda_device"] = str(gpu)
    result["runtime"]["ipc_prefix"] = "replica_" + original["identity"][:12]
    result["gpu_lock"] = str(Path(original["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock"))
    result.pop("identity")
    result["identity"] = canonical_digest(result)
    return result


def delegated_scene(config, path, scene, phase):
    root = Path(config["attempt_root"])
    plan_path = root / "parallel/execution_plan.json"
    if not scene or phase not in ("capture", "prepare", "query", "fuse", "evaluate") or not plan_path.exists():
        return None
    plan = json.loads(plan_path.read_text())
    if Path(path).resolve() != Path(plan["original_config"]).resolve() or scene not in plan["assignments"]:
        return None
    receipt = root / "parallel" / f"{scene}.json"
    gpu = plan["assignments"][scene]
    while not receipt.exists():
        failure = root / "parallel" / f"gpu{gpu}_failed.json"
        if failure.exists():
            raise RuntimeError(f"parallel worker failed: {failure}")
        time.sleep(2)
    result = json.loads(receipt.read_text())
    if result["status"] != "COMPLETE":
        raise RuntimeError(f"incomplete parallel scene: {scene}")
    return result


def launch_plan(path):
    config = require_transfer(path)
    root = Path(config["attempt_root"])
    if (root / "parallel/execution_plan.json").exists():
        raise RuntimeError("parallel execution plan already exists; do not overwrite")
    assignments = {"office0": 0, "office2": 0, "office1": 1, "office3": 1, "office4": 2}
    for scene in assignments:
        if (root / "native" / scene / "frontend_job").exists():
            raise RuntimeError(f"scene already started: {scene}")
    plan = {"original_config": str(Path(path).resolve()), "original_identity": config["identity"], "assignments": assignments, "authorization": "user authorized parallel scenes on multiple GPUs", "gpu2_gate": "room2 six evaluation rows"}
    for gpu in (0, 1, 2):
        atomic_write_json(root / "parallel" / f"gpu{gpu}.json", worker_config(config, gpu))
    atomic_write_json(root / "parallel/execution_plan.json", plan)
    return plan


def run_worker(path):
    from .jobs import invoke

    config = require_transfer(path)
    root = Path(config["attempt_root"])
    plan = json.loads((root / "parallel/execution_plan.json").read_text())
    original = require_transfer(Path(plan["original_config"]))
    gpu = int(config["runtime"]["cuda_device"])
    if config != worker_config(original, gpu):
        raise ValueError("worker changed settings beyond authorized hardware fields")
    try:
        if gpu == 2:
            rows = root / "scenes/room2/rows/replica_transfer/room2"
            while len(list(rows.glob("*.json"))) != 6:
                time.sleep(2)
        for scene, assigned in plan["assignments"].items():
            if assigned != gpu:
                continue
            for phase, method in (("capture", None), ("prepare", None), ("query", "Q_GAIN"), ("query", "CP_M4_GAIN_S2"), ("fuse", None), ("evaluate", None)):
                invoke(config, path, phase, scene=scene, method=method)
            rows = list((root / "scenes" / scene / "rows/replica_transfer" / scene).glob("*.json"))
            if len(rows) != 6:
                raise RuntimeError(f"expected six result rows: {scene}")
            atomic_write_json(root / "parallel" / f"{scene}.json", {"status": "COMPLETE", "scene": scene, "gpu": gpu, "config_identity": config["identity"]})
    except Exception as error:
        atomic_write_json(root / "parallel" / f"gpu{gpu}_failed.json", {"status": "FAILED", "error": repr(error)})
        raise
