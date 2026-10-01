"""Bounded own-map reconstruction with unchanged schedules and real native actions."""

import argparse
import os
from pathlib import Path
import subprocess
import time

from static_ovmap.backbone_wave1.binding import set_option
from static_ovmap.backbone_wave1.runtime import execute, exclusive_lock
from static_ovmap.module_validation.boundary_jobs import capture_inputs, verify_capture
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, read
from .light import require_transfer_freeze


def recipe_for(spec, map_id):
    originals = {row["id"]: row for row in spec["map_variants"]}
    if map_id not in originals:
        raise ValueError("map recipe is not a fixed registered recovery arm")
    mode = {"RW_A1_NATIVE_FALLBACK": "A1", "RW_A2_MULTI_FREE": "A2", "RW_A3_MULTI_UNION": "A3",
            "RW_S1_CROP_PRIORITY": "native", "RW_S2_CONFLICT": "native"}[map_id]
    frontend = "S1" if map_id == "RW_S1_CROP_PRIORITY" else "S2" if map_id == "RW_S2_CONFLICT" else "cropformer"
    return {"id": map_id, "association": mode, "frontend": frontend,
            "source_spec_recipe": originals[map_id], "depth_fusion": "native_sequential",
            "inst_association": 4, "data_association": 2, "ratio_threshold": 0.0}


def run_map(binding, build, scene, map_id, *, threads=8, resume=False):
    require_transfer_freeze(binding, scene)
    spec = read(binding["spec"])
    recipe = recipe_for(spec, map_id)
    if scene not in binding["scenes"] or threads < 1 or threads > 8:
        raise ValueError("map scene or thread allowance differs from the task")
    root = Path(binding["output_root"]) / "maps" / scene / map_id
    data = binding["scenes"][scene]
    validation = read(Path(binding["output_root"]) / "validation/native/validation_receipt.json")
    if validation["status"] != "VERIFIED" or validation["new_extension"] != build["extension"]:
        raise ValueError("full maps require validated native off/fallback/follower behavior")
    if scene in spec["datasets"]["replica"]:
        freeze = read(Path(binding["output_root"]) / "freeze/receipt.json")
        if freeze["nominated_map"] != map_id:
            raise ValueError("Replica mapping requires the one committed qualifying map")
    command = list(data["actual_parent_command"]["command"])
    upstream = Path(build["upstream_worktree"])
    repo = Path(binding["repository_root"])
    command[1] = str(upstream / "scripts/panoptic_mapping_.py")
    set_option(command, "--result_folder", root / "mapper")
    set_option(command, "--intermediate_seg_folder", root / "segments")
    set_option(command, "--num_threads", threads)
    set_option(command, "--perception_worker", repo / "scripts/evaluation/ovimap_module_perception_worker.py")
    if not {"--use_temp_geometrics", "--skip_feature_extraction", "--use_temp_panoptics"} <= set(command):
        raise ValueError("actual inherited command does not use the complete cached frontend/depth path")
    recipe["diagnostic_root"] = str(root / "diagnostics")
    if recipe["frontend"] != "cropformer":
        frontend_root = Path(binding["output_root"]) / "frontend" / scene / "S1"
        frontend = read(frontend_root / "receipt.json")
        if (frontend["status"] != "COMPLETE"
                or frontend["identity"] != canonical_digest({key: value for key, value in frontend.items() if key != "identity"})
                or frontend["scheduled_frame_ids"] != data["schedule"]
                or frontend["completed_frame_ids"] != data["completed_frame_ids"]):
            raise ValueError("SAM maps require a complete locked CropFormer-priority frontend")
        recipe.update(frontend_root=str(frontend_root / "rasters"),
                      frontend_receipt=str(frontend_root / "receipt.json"),
                      frontend_identity=frontend["identity"],
                      parent_sam_receipt=data["sam_receipt"], capture_manifest=data["capture_manifest"])
    env = dict(os.environ, **data["actual_parent_command"]["environment"])
    binary_root = str(Path(build["extension"]["path"]).parent)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = ":".join([binary_root, str(upstream / "scripts"), str(repo / "src"), str(repo),
        str(Path(data["inherited"]["runtime"]["baseline_build"]) / "mapping_ros_ws/devel/lib")])
    env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"].replace(str(Path(data["native_extension"]["path"]).parent), binary_root)
    env.pop("OVIMAP_BACKBONE_RECIPE", None)
    env.update(OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT=str(root / "capture"),
        OVIMAP_MODULE_VALIDATION_HELPER_ROOT=str(repo), OVIMAP_RECOVERY_RECIPE=str(root / "recipe.json"),
        OVIMAP_NATIVE_UPSTREAM=str(upstream))
    safe_env = {key: value for key, value in env.items() if key.startswith("OVIMAP_") or key in
                {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}}
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    for name in ("maps.py", "mapping_hooks.py", "association.py"):
        index.identity(Path(__file__).with_name(name))
    if recipe["frontend"] != "cropformer":
        for path in (recipe["frontend_receipt"], recipe["parent_sam_receipt"], recipe["capture_manifest"]):
            index.identity(path)
    if recipe["frontend"] == "S2":
        index.identity(Path(__file__).with_name("sam_completion.py"))
    for path in (upstream / "scripts/panoptic_mapping_.py", upstream / "scripts/utils/common_scannet_nyu.py",
                 upstream / "scripts/utils/common_utils.py", upstream / "scripts/utils/data_loaders.py",
                 upstream / "scripts/view_selection.py", build["extension"]["path"],
                 Path(binding["output_root"]) / "validation/native/validation_receipt.json"):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "recipe": recipe,
        "command": command, "environment": safe_env, "inputs": index.entries(), "build": build["recovery_identity"]})
    with exclusive_lock(root.parent / ("." + map_id + ".lock")):
        receipt_path = root / "map_receipt.json"
        if receipt_path.is_file() and read(receipt_path)["status"] == "COMPLETE":
            previous = read(receipt_path)
            if not resume or previous["input_identity"] != identity:
                raise ValueError("completed full map can resume only with its exact input identity")
            for row in previous["outputs"]:
                index.identity(row["path"], row)
            return previous
        if root.exists():
            number = 1
            while root.with_name(root.name + f".failed_{number:03d}").exists():
                number += 1
            root.rename(root.with_name(root.name + f".failed_{number:03d}"))
        root.mkdir(parents=True)
        atomic_write_json(root / "recipe.json", recipe)
        atomic_write_json(root / "running.json", {"status": "RUNNING", "input_identity": identity,
            "command": command, "environment": safe_env, "started_at_unix": time.time(),
            "partial_tsdf_resumption": False, "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()})
        source_inputs, started = index.entries(), time.monotonic()
        try:
            actual = execute(command, upstream, root / "mapping.log", env=env)
            capture_path = root / "capture" / scene / "manifest.json"
            capture = verify_capture(capture_path, allow_skipped=True)
            if (capture["scheduled_frame_ids"] != data["schedule"]
                    or capture["completed_frame_ids"] != data["completed_frame_ids"]
                    or capture["native_extension"]["sha256"] != build["extension"]["sha256"]):
                raise ValueError("map changed the inherited schedule, frame policy, or extension")
            deferred = root / "diagnostics/native_deferred_metadata.json"
            outputs = [index.identity(row["path"], row) for row in capture_inputs(capture_path, capture)]
            outputs.append(index.identity(deferred))
            receipt = {"status": "COMPLETE", "scene": scene, "map_id": map_id, "input_identity": identity,
                "recipe": recipe, "native_build_identity": build["recovery_identity"], "inputs": source_inputs,
                "outputs": outputs, "capture_manifest": str(capture_path), "deferred_metadata": str(deferred),
                "geometry_locked_before_semantics_and_labels": True, "GT_input": False,
                "scheduled_count": len(capture["scheduled_frame_ids"]), "completed_count": len(capture["frames"]),
                "command": actual, "elapsed_seconds": time.monotonic() - started, "new_visual_inference": 0,
                "native_pickle_status": "DEFERRED_FEATURES_REQUIRED"}
            index.write_memo(root / "input_verifications.json")
            atomic_write_json(receipt_path, receipt)
            print(scene, map_id, receipt["status"], flush=True)
            return receipt
        except BaseException as exc:
            atomic_write_json(receipt_path, {"status": "FAILED", "input_identity": identity, "scene": scene,
                "map_id": map_id, "error": f"{type(exc).__name__}: {exc}", "new_visual_inference": 0,
                "partial_tsdf_resumption": False, "elapsed_seconds": time.monotonic() - started})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_map(read(args.binding), read(args.build), args.scene, args.map_id, resume=args.resume)
