"""Actual native trace and SAM video preflight, separate from measurements."""

import json
import os
from pathlib import Path

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.boundary_jobs import file_identity
from .binding import read
from .runtime import execute


def native_traces(binding, spec, build):
    root = Path(binding["output_root"]) / "prepare/native_trace"
    scene = spec["datasets"]["development"][0]
    data = binding["scenes"][scene]
    extension = build["extension"]["path"]
    old_extension = data["runtime"]["native_extension"]
    repo = Path(binding["repository_root"])
    variants = [("old_native", "native", False), ("new_native", "native", True),
                ("forward", "object_forward", True), ("bidir", "object_bidirectional", True),
                ("old_native_repeat", "native", False), ("old_native_serial", "native", False),
                ("new_native_serial", "native", True)]
    receipts = {}
    for name, mode, new in variants:
        output = root / name
        job = {"output_root": str(output), "capture_manifest": data["parent_capture"],
               "geometric_root": data["geometric_root"], "extension": extension if new else old_extension,
               "mode": mode, "new_binary": new, "threads": 1 if name.endswith("serial") else 8}
        job_path = root / (name + ".json")
        atomic_write_json(job_path, job)
        cached = read(output / "receipt.json") if (output / "receipt.json").is_file() else None
        if cached and cached.get("extension_identity") != file_identity(job["extension"]):
            number = 1
            while output.with_name(f"{name}.previous_{number:03d}").exists():
                number += 1
            output.rename(output.with_name(f"{name}.previous_{number:03d}"))
            cached = None
        if cached is None:
            env = dict(os.environ, **data["parent_command"]["environment"])
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTHONPATH"] = ":".join([str(Path(job["extension"]).parent),
                str(Path(spec["upstream_worktree"]) / "scripts"), str(repo / "src"),
                str(repo),
                str(Path(data["runtime"]["baseline_build"]) / "mapping_ros_ws/devel/lib")])
            execute([spec["runtime_default"], "-m", "static_ovmap.backbone_wave1.native_trace", "--job", job_path],
                    spec["upstream_worktree"], root / (name + ".log"), env=env)
        receipts[name] = read(output / "receipt.json")
    baseline_keys = ("input_instance_label", "registered_label", "point_count", "semantic_label")
    for before, after in zip(receipts["old_native"]["frames"], receipts["new_native"]["frames"]):
        if before["frame_id"] != after["frame_id"]:
            raise ValueError("native trace schedules differ")
        for key in ("aliases", "label_instances"):
            if before["state"][key] != after["state"][key]:
                raise ValueError(f"native-off state differs: {key}")
        select = lambda frame: [{k: row[k] for k in baseline_keys} for row in frame["state"]["segments"]]
        if select(before) != select(after):
            raise ValueError("native-off assigned segments differ")
        filename = f"{before['frame_id']:06d}.npz"
        with np.load(root / "old_native" / filename) as old, np.load(root / "new_native" / filename) as new:
            np.testing.assert_array_equal(old["owner_projection"], new["owner_projection"])
    tsdf_comparison = {}
    with np.load(root / "old_native_serial/tsdf.npz") as old, np.load(root / "new_native_serial/tsdf.npz") as new:
        if old.files != new.files:
            raise ValueError("native-off TSDF exports differ")
        for key in old.files:
            exact = np.array_equal(old[key], new[key])
            if not exact:
                raise ValueError(f"single-thread native-off TSDF differs: {key}")
            tsdf_comparison[key] = {"exact": exact}
    parallel_variation = {}
    for first, second in [("old_native", "old_native_repeat"), ("old_native", "new_native")]:
        with np.load(root / first / "tsdf.npz") as a, np.load(root / second / "tsdf.npz") as b:
            parallel_variation[first + "__" + second] = {
                key: {"different_elements": int(np.sum(a[key] != b[key])),
                      "max_absolute_difference": float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))}
                for key in a.files}
    summary = {"status": "COMPLETE", "acquired_frame_count": 3, "native_off_partition_parity": True,
               "native_off_tsdf_comparison": tsdf_comparison, "variants": list(receipts),
               "serial_diagnostic_threads": 1, "production_mapping_threads": 8,
               "native_parallel_variation": parallel_variation,
               "probe_read_only": all(frame["probe_state_before"] == frame["probe_state_after"]
                    for name in ("new_native", "forward", "bidir") for frame in receipts[name]["frames"]),
               "candidate_vetoes": {name: sum(f["state"]["association"].get("candidate_vetoes", 0)
                    for f in receipts[name]["frames"]) for name in ("forward", "bidir")},
               "receipts": {name: str(root / name / "receipt.json") for name in receipts}}
    atomic_write_json(root / "summary.json", summary)
    return summary


def sam_preflight(binding, *, gpu):
    root = Path(binding["output_root"]) / "prepare/sam2_preflight"
    sam = binding["sam2"]
    job = {"output_root": str(root), "capture_manifest": binding["scenes"]["scene0056_00"]["parent_capture"],
           "checkpoint": sam["checkpoint"], "sam2_repo": sam["code"], "sam2_commit": sam["commit"],
           "sam2_config": sam["config"], "preflight": True,
           "gpu_lock": f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{gpu}.lock"}
    atomic_write_json(root.parent / "sam2_preflight_job.json", job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=sam["code"] + ":" + str(Path(binding["repository_root"]) / "src"))
    execute([sam["python"], "-m", "static_ovmap.backbone_wave1.frontend_sam2", "--job",
             root.parent / "sam2_preflight_job.json"], binding["repository_root"], root / "run.log", env=env)
    result = read(root / "receipt.json")
    if result["status"] != "COMPLETE" or len(result["frames"]) != 3 or result["counters"]["physical_image_encodings"] < 3:
        raise ValueError("real SAM video preflight was incomplete")
    return result
