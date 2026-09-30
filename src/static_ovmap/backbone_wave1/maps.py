"""Fresh CPU reconstruction jobs with strict map-specific captures."""

import os
from pathlib import Path
import time

from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.module_validation.boundary_jobs import capture_inputs, verify_capture
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from .binding import option, read, set_option
from .runtime import execute, exclusive_lock


PRE_C6_VALIDATION_CONTROLLER = "98710ee02de36869cd4a647eaadb6f0e85115e196ceae11de55c52b7e8c8d96a"


def unchanged_bridge_controller_alias(root, saved, command, env, recipe, build, source_inputs, identity):
    """Reuse the measured BB00 after a validation-only controller correction."""
    if recipe["id"] != "BB00_NATIVE":
        return False
    old = {item["path"]: item for item in saved["inputs"]}
    new = {item["path"]: item for item in source_inputs}
    controller = str(Path(__file__).resolve())
    if old.get(controller, {}).get("sha256") != PRE_C6_VALIDATION_CONTROLLER:
        return False
    if {k: v for k, v in old.items() if k != controller} != {k: v for k, v in new.items() if k != controller}:
        return False
    selected_env = {k: v for k, v in env.items() if k.startswith("OVIMAP_") or k in
        ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    running = read(root / "running.json")
    if (saved["command"]["argv"] != [str(x) for x in command] or saved["recipe"] != recipe or
            saved["native_build_identity"] != build["identity"] or running["environment"] != selected_env):
        return False
    bridge = root.parents[2] / "bridge_parity.json"
    if not bridge.is_file() or read(bridge)["status"] != "VERIFIED":
        return False
    atomic_write_json(root / "controller_reuse_alias.json", {
        "status": "VERIFIED_UNCHANGED_BB00_CONTROLLER_ALIAS", "original_input_identity": saved["input_identity"],
        "requested_input_identity": identity, "original_controller": old[controller], "current_controller": new[controller],
        "unchanged_command_environment_recipe_binary_and_kernel_inputs": True,
        "reason": "C6 post-update validation now records rare discrepancies; BB00 disables C6",
        "new_mapping_work": 0, "original_measured_receipt_preserved": True})
    return True


def mapping_job(binding, spec, build, scene, recipe, *, threads=8, phase="maps"):
    if scene not in binding["scenes"]:
        raise ValueError("mapping scene is outside the exposed task cohort")
    root = Path(binding["output_root"]) / phase / scene / recipe["id"]
    data = binding["scenes"][scene]
    command = list(data["parent_command"]["command"])
    command[1] = str(Path(spec["upstream_worktree"]) / "scripts/panoptic_mapping_.py")
    set_option(command, "--result_folder", root / "mapper")
    set_option(command, "--intermediate_seg_folder", root / "segments")
    set_option(command, "--num_threads", threads)
    set_option(command, "--temp_geometrics_folder", data["geometric_root"])
    command.remove("--save_temp_geometrics")
    command.extend(["--use_temp_geometrics", "--skip_feature_extraction"])
    set_option(command, "--perception_worker", Path(binding["repository_root"]) / "scripts/evaluation/ovimap_module_perception_worker.py")
    paired_root = Path(binding["output_root"]) / "frontend" / scene / "SAM2_PAIRED"
    if recipe["frontend"] != "cropformer":
        if read(paired_root / "receipt.json")["status"] != "COMPLETE":
            raise ValueError("SAM maps require the actual complete paired frontend")
    hook_recipe = dict(recipe, diagnostic_root=str(root / "diagnostics"),
                       paired_frontend_root=str(paired_root), order_diagnostics=scene in spec["datasets"]["development"])
    hook_path = root / "recipe.json"
    index = InputIndex()
    repo, upstream = Path(binding["repository_root"]), Path(spec["upstream_worktree"])
    for name in ("maps.py", "mapping_hooks.py", "fusion.py", "association.py"):
        index.identity(Path(__file__).with_name(name))
    for path in (upstream / "scripts/panoptic_mapping_.py", upstream / "scripts/view_selection.py",
                 upstream / "scripts/utils/common_scannet_nyu.py", upstream / "scripts/utils/common_utils.py",
                 upstream / "scripts/utils/data_loaders.py", Path(build["extension"]["path"])):
        index.identity(path)
    if recipe["frontend"] != "cropformer":
        index.identity(paired_root / "receipt.json")
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "recipe": hook_recipe,
                                "command": command, "inputs": index.entries(), "native_build": build["identity"]})
    env = dict(os.environ, **data["parent_command"]["environment"])
    binary_root = str(Path(build["extension"]["path"]).parent)
    old_root = str(Path(data["runtime"]["native_extension"]).parent)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = ":".join([binary_root, str(upstream / "scripts"), str(repo / "src"), str(repo),
        str(Path(data["runtime"]["baseline_build"]) / "mapping_ros_ws/devel/lib")])
    env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"].replace(old_root, binary_root)
    env.update(OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT=str(root / "capture"),
               OVIMAP_MODULE_VALIDATION_HELPER_ROOT=str(repo), OVIMAP_BACKBONE_RECIPE=str(hook_path),
               OVIMAP_NATIVE_UPSTREAM=str(upstream))
    return root, command, env, hook_recipe, identity, index


def run_map(binding, spec, build, scene, recipe, *, threads=8, resume=False):
    root, command, env, hook_recipe, identity, index = mapping_job(binding, spec, build, scene, recipe, threads=threads)
    lock_path = root.parent / f".{recipe['id']}.lock"
    with exclusive_lock(lock_path):
        source_inputs = index.entries()
        receipt_path = root / "map_receipt.json"
        if resume and receipt_path.is_file():
            saved = read(receipt_path)
            if saved["status"] == "COMPLETE":
                if saved["input_identity"] != identity and not unchanged_bridge_controller_alias(
                        root, saved, command, env, hook_recipe, build, source_inputs, identity):
                    raise ValueError("completed map identity changed")
                for item in saved["outputs"]:
                    index.identity(item["path"], item)
                return saved
        if root.exists():
            number = 1
            while root.with_name(f"{root.name}.failed_{number:03d}").exists():
                number += 1
            root.rename(root.with_name(f"{root.name}.failed_{number:03d}"))
        root.mkdir(parents=True)
        atomic_write_json(root / "recipe.json", hook_recipe)
        atomic_write_json(root / "running.json", {"status": "RUNNING", "input_identity": identity,
            "command": command, "environment": {k: v for k, v in env.items() if k.startswith("OVIMAP_") or k in
                 ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "started_at_unix": time.time(), "partial_tsdf_resumption": False})
        start = time.monotonic()
        try:
            command_receipt = execute(command, spec["upstream_worktree"], root / "mapping.log", env=env)
            capture_path = root / "capture" / scene / "manifest.json"
            capture = verify_capture(capture_path, allow_skipped=True)
            data = binding["scenes"][scene]
            if capture["scheduled_frame_ids"] != data["schedule"]:
                raise ValueError("actual map changed the 200-slot schedule")
            if capture["completed_frame_ids"] != data["parent_completed_frame_ids"]:
                raise ValueError("actual map changed the inherited invalid/missing-frame policy")
            if capture["native_extension"]["sha256"] != build["extension"]["sha256"]:
                raise ValueError("map loaded an unbound native extension")
            owner_discrepancies = []
            if recipe["association"].startswith("object_"):
                for frame in capture["frames"]:
                    snapshot = read(capture_path.parent / frame["native_state"]["path"])["native_state"]
                    owner_discrepancies.extend({"frame_id": frame["frame_id"], **segment}
                        for segment in snapshot["segments"] if segment.get("owner_discrepancy", False))
            outputs = [index.identity(item["path"]) for item in capture_inputs(capture_path, capture)]
            deferred = root / "diagnostics/native_deferred_metadata.json"
            outputs.append(index.identity(deferred))
            result = {"status": "COMPLETE", "scene": scene, "map_id": recipe["id"],
                "input_identity": identity, "recipe": hook_recipe, "native_build_identity": build["identity"],
                "inputs": source_inputs, "outputs": outputs, "capture_manifest": str(capture_path),
                "deferred_metadata": str(deferred), "geometry_locked_before_semantics_and_labels": True,
                "scheduled_count": len(capture["scheduled_frame_ids"]), "completed_count": len(capture["frames"]),
                "command": command_receipt, "elapsed_seconds": time.monotonic() - start,
                "new_visual_inference": 0, "native_pickle_status": "DEFERRED_FEATURES_REQUIRED"}
            result["post_update_owner_discrepancies"] = owner_discrepancies
            atomic_write_json(receipt_path, result)
            return result
        except BaseException as exc:
            atomic_write_json(receipt_path, {"status": "FAILED", "input_identity": identity,
                "scene": scene, "map_id": recipe["id"], "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - start, "new_visual_inference": 0,
                "partial_tsdf_resumption": False})
            raise
