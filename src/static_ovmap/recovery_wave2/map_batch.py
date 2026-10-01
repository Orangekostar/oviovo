"""Two bounded CPU mapper processes for a fixed, complete map cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import read
from .maps import run_map


def _job(binding, build, scene, map_id):
    return run_map(binding, build, scene, map_id, threads=8, resume=True)


def run(binding, build, cohort, map_ids):
    spec = read(binding["spec"])
    scene_order = spec["datasets"][cohort]
    root = Path(binding["output_root"]) / "execution"
    name = cohort + "_" + "_".join(map_ids)
    started = time.monotonic()
    rows, failures = {}, []
    with ProcessPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(_job, binding, build, scene, map_id): (scene, map_id)
                   for map_id in map_ids for scene in scene_order}
        for future in as_completed(pending):
            scene, map_id = pending[future]
            try:
                rows[scene + "/" + map_id] = future.result()
            except Exception as exc:
                failures.append({"scene": scene, "map_id": map_id, "status": "FAILED",
                                 "error": f"{type(exc).__name__}: {exc}"})
            atomic_write_json(root / (name + ".running.json"), {"status": "RUNNING", "completed": len(rows),
                "total": len(pending), "failures": failures, "workers": 2, "threads_per_worker": 8})
    result = {"status": "MAPS_COMPLETE" if not failures else "INCOMPLETE_COHORT", "cohort": cohort,
        "scene_order": scene_order, "map_ids": map_ids, "maps": rows, "failures": failures,
        "workers": 2, "threads_per_worker": 8, "elapsed_seconds": time.monotonic() - started,
        "semantics_status": "NOT_YET_RUN", "GT_input": False, "new_visual_inference": 0}
    result["identity"] = canonical_digest({key: value for key, value in result.items() if key != "elapsed_seconds"})
    atomic_write_json(root / (name + ".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--cohort", choices=("development", "replica"), default="development")
    parser.add_argument("--map-id", action="append", required=True)
    args = parser.parse_args()
    result = run(read(args.binding), read(args.build), args.cohort, args.map_id)
    print(result["status"], "maps", len(result["maps"]), "failures", len(result["failures"]), flush=True)
    if result["failures"]:
        raise SystemExit(1)
