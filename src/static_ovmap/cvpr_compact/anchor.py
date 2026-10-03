"""Fixed CF18 native captures and verified aliases of the eight Replica maps."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import time

from static_ovmap.backbone_wave1.binding import option, set_option
from static_ovmap.backbone_wave1.runtime import check_gpu_once, exclusive_lock
from static_ovmap.module_validation.boundary_jobs import capture_inputs, verify_capture
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .projected_views import _verified_identity
from .protocol import load_spec
from .runtime import execute_leaf


def anchor_plan(binding, scene, preparation):
    if scene not in binding["cohorts"]["scannet_cf18"]:
        raise ValueError("new anchors may only use the fixed CF18 captures")
    if preparation["status"] != "COMPLETE" or preparation["scene"] != scene:
        raise ValueError("new anchors require the exact complete same-scene inputs")
    template = binding["scenes"]["scene0056_00"]
    root, repo = Path(binding["output_root"]), Path(binding["repository_root"])
    runtime, original = template["runtime"], template["actual_capture_command"]
    mapping = list(original["command"])
    expected = {"--dataset": "scannet_nyu", "--task": "Nyu40", "--data_association": "2",
        "--inst_association": "4", "--seg_graph_confidence": "3", "--use_inst_label_connect": "1",
        "--connection_ratio_th": "0.2", "--num_threads": "8"}
    if any(option(mapping, key) != value or mapping.count(key) != 1 for key, value in expected.items()):
        raise ValueError("parent command differs from the prescribed actual Native options")
    schedule = preparation["schedule"]
    if len(schedule["frame_ids"]) != 200 or schedule["frame_ids"] != list(range(schedule["start"], schedule["end"], schedule["step"])):
        raise ValueError("new Native maps require their exact original 200-slot schedule")
    map_root, front_root = root / "maps" / scene / "BB00_NATIVE", root / "frontend" / scene / "cropformer"
    exported = Path(preparation["export_root"])
    invalid = set(preparation["invalid_pose_frame_ids"])
    if not invalid <= set(schedule["frame_ids"]):
        raise ValueError("invalid pose IDs leave the original scheduled slots")
    images = [str(exported / "color" / f"{frame}.jpg") for frame in schedule["frame_ids"] if frame not in invalid]
    if not images:
        raise RuntimeError("BLOCKED_INPUT: all original pose slots are invalid")
    frontend = [runtime["frontend_python"], str(Path(runtime["cropformer_root"]) / "demo_cropformer/demo_from_dirs.py"),
        "--config-file", runtime["cropformer_config"], "--input", *images, "--output", str(front_root),
        "--confidence-threshold", "0.5", "--out-type", "0", "--opts", "MODEL.WEIGHTS", runtime["cropformer_weights"]]
    replacements = {"--scene_num": scene, "--data_folder": exported.parent, "--result_folder": map_root / "mapper",
        "--temp_panoptics_folder": front_root, "--temp_geometrics_folder": map_root / "geometrics",
        "--intermediate_seg_folder": map_root / "segments", "--perception_worker": repo / "scripts/evaluation/ovimap_module_perception_worker.py",
        **{"--" + key: schedule[key] for key in ("start", "end", "step")}}
    for key, value in replacements.items():
        set_option(mapping, key, value)
    if mapping.count("--use_temp_geometrics") != 1 or "--save_temp_geometrics" in mapping or mapping.count("--skip_feature_extraction") != 1:
        raise ValueError("parent geometric-cache/deferred-feature options changed")
    mapping.remove("--use_temp_geometrics")
    mapping.append("--save_temp_geometrics")
    recipe = {**template["original_map_options"], "diagnostic_root": str(map_root / "diagnostics"),
        "paired_frontend_root": str(front_root), "order_diagnostics": False}
    if any(recipe[key] != value for key, value in {
            "id": "BB00_NATIVE", "association": "native", "depth_fusion": "native", "frontend": "cropformer"}.items()):
        raise ValueError("new anchor recipe differs from the common BB00_NATIVE")
    extension_root = Path(template["native_extension"]["path"]).parent
    baseline_lib = Path(runtime["baseline_build"]) / "mapping_ros_ws/devel/lib"
    upstream = original["environment"]["OVIMAP_NATIVE_UPSTREAM"]
    map_env = {**original["environment"], "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "PYTHONPATH": ":".join(map(str, (extension_root, Path(upstream) / "scripts", repo / "src", repo, baseline_lib))),
        "LD_LIBRARY_PATH": ":".join(map(str, (extension_root, baseline_lib, Path(mapping[0]).parents[1] / "lib"))),
        "OVIMAP_BACKBONE_RECIPE": str(map_root / "recipe.json"), "OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT": str(map_root / "capture"),
        "OVIMAP_MODULE_VALIDATION_HELPER_ROOT": str(repo), "OVIMAP_NATIVE_MODEL": runtime["native_model"],
        "OVIMAP_NATIVE_UPSTREAM": str(upstream)}
    frontend_python = Path(runtime["frontend_python"])
    front_env = {"CUDA_VISIBLE_DEVICES": str(binding["gpu"]), "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8", "OPENCV_FOR_THREADS_NUM": "8",
        "PYTHONPATH": runtime["cropformer_root"], "LD_LIBRARY_PATH": str(frontend_python.parents[1] / "lib"),
        "PATH": str(frontend_python.parent) + ":" + os.environ.get("PATH", ""), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_MODULES_CACHE": runtime["hf_modules_cache"]}
    return {"scene": scene, "map_root": str(map_root), "frontend_root": str(front_root),
        "mapping_command": mapping, "mapping_environment": map_env, "mapping_cwd": str(upstream),
        "frontend_command": frontend, "frontend_environment": front_env, "frontend_cwd": runtime["cropformer_root"],
        "recipe": recipe, "schedule": schedule, "invalid_pose_frame_ids": sorted(invalid),
        "geometric_cache_mode": "ORIGINAL_OPERATOR_COMPUTE_AND_SAVE", "geometry_variant": False}


def _native_inputs(binding, index):
    template = binding["scenes"]["scene0056_00"]
    parent = read(template["parent_map_receipt"])
    repo = Path(binding["repository_root"])
    inputs = []
    for item in parent["inputs"]:
        path = Path(item["path"])
        if path.name == "maps.py":
            continue
        if path.parent.name == "backbone_wave1":
            path = repo / "src/static_ovmap/backbone_wave1" / path.name
        inputs.append(index.identity(path, item))
    baseline_lib = Path(template["runtime"]["baseline_build"]) / "mapping_ros_ws/devel/lib"
    for path in (baseline_lib / "depth_segmentation_py.cpython-311-x86_64-linux-gnu.so",
                 baseline_lib / "libdepth_segmentation.so", repo / "src/static_ovmap/module_validation/native_capture.py",
                 repo / "scripts/evaluation/ovimap_module_perception_worker.py", Path(__file__)):
        inputs.append(index.identity(path))
    return inputs


def _ensure_frontend(binding, scene, preparation, plan, index, native_inputs):
    root = Path(plan["frontend_root"])
    runtime = binding["scenes"]["scene0056_00"]["runtime"]
    front_root = Path(runtime["cropformer_root"])
    inputs = [index.identity(runtime[key]) for key in ("cropformer_config", "cropformer_weights")]
    inputs += [index.identity(path) for path in sorted(front_root.rglob("*.py"))]
    inputs += [index.identity(path) for path in sorted((front_root / "configs").rglob("*.yaml"))]
    export = read(preparation["export_receipt"])
    for relative, digest in export["file_hashes"].items():
        index.identity(Path(preparation["export_root"]) / relative, {"sha256": digest})
    identity = canonical_digest({"plan": plan["frontend_command"], "environment": plan["frontend_environment"],
        "inputs": inputs, "export": index.identity(preparation["export_receipt"]), "producer": native_inputs[-1]})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file() and read(receipt_path)["status"] == "COMPLETE":
        previous = read(receipt_path)
        _verified_identity(previous)
        if previous["input_identity"] != identity:
            raise ValueError("completed CropFormer inputs changed; invalidate explicitly")
        for item in previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    with exclusive_lock(binding["gpu_lock"]), exclusive_lock(root / ".execution.lock"):
        resource = check_gpu_once(binding["gpu"], root / "gpu_check.json")
        if resource["occupants"]:
            raise RuntimeError("RESOURCE_BLOCK: CropFormer GPU is occupied")
        print(f"CF18 frontend {scene}: {200 - len(plan['invalid_pose_frame_ids'])} original slots", flush=True)
        command = execute_leaf(plan["frontend_command"], plan["frontend_cwd"], root / "frontend.log",
            env={**os.environ, **plan["frontend_environment"]}, input_identity=identity)
    valid_frames = [frame for frame in plan["schedule"]["frame_ids"] if frame not in plan["invalid_pose_frame_ids"]]
    outputs = [index.identity(root / f"{frame}.png") for frame in valid_frames]
    outputs += [index.identity(root / "frame_diagnostics" / f"{frame}.json") for frame in valid_frames]
    receipt = {"status": "COMPLETE", "scene": scene, "input_identity": identity, "command": command,
        "physical_image_inputs": len(valid_frames), "inference_counts_source": "ORIGINAL_CROPFORMER_FRAME_DIAGNOSTICS",
        "inputs": inputs, "outputs": outputs, "GT_input": False, "original_out_type": 0}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def _run_map(binding, scene, preparation, plan, front, native_inputs):
    root = Path(plan["map_root"])
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    for item in native_inputs + front["outputs"]:
        index.identity(item["path"], item)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "plan": plan,
        "inputs": native_inputs, "frontend": front["identity"], "preparation": preparation["identity"]})
    receipt_path = root / "map_receipt.json"
    with exclusive_lock(root.parent / ".BB00_NATIVE.lock"):
        if receipt_path.is_file() and read(receipt_path)["status"] == "COMPLETE":
            previous = read(receipt_path)
            _verified_identity(previous)
            if previous["input_identity"] != identity:
                raise ValueError("completed Native anchor changed; invalidate explicitly")
            for item in previous["outputs"]:
                index.identity(item["path"], item)
            return previous
        if root.exists():
            root.rename(root.with_name(root.name + ".failed_" + str(time.time_ns())))
        root.mkdir(parents=True)
        if shutil.disk_usage(root).free < 2 * 1024**3:
            raise OSError("RESOURCE_BLOCK: less than 2GiB free for the next fixed anchor")
        atomic_write_json(root / "recipe.json", plan["recipe"])
        running = {"status": "RUNNING", "input_identity": identity, "command": plan["mapping_command"],
            "environment": plan["mapping_environment"], "partial_tsdf_resumption": False, "GT_input": False}
        atomic_write_json(root / "running.json", running)
        started = time.monotonic()
        try:
            print(f"CF18 native map {scene}: CPU 8 threads, no semantic encoder", flush=True)
            command = execute_leaf(plan["mapping_command"], plan["mapping_cwd"],
                Path(binding["output_root"]) / "execution/maps" / scene / "mapping.log",
                env={**os.environ, **plan["mapping_environment"]}, input_identity=identity)
            capture_path = root / "capture" / scene / "manifest.json"
            capture = verify_capture(capture_path, allow_skipped=True)
            if (capture["scheduled_frame_ids"] != plan["schedule"]["frame_ids"] or not capture["frames"]
                    or capture["native_extension"]["sha256"] != binding["scenes"]["scene0056_00"]["native_extension"]["sha256"]):
                raise ValueError("actual Native capture changed its schedule, extension or completed-frame availability")
            outputs = [index.identity(item["path"]) for item in capture_inputs(capture_path, capture)]
            metadata = root / "diagnostics/native_deferred_metadata.json"
            outputs.append(index.identity(metadata))
            result = {"status": "COMPLETE", "scene": scene, "map_id": "BB00_NATIVE", "input_identity": identity,
                "recipe": plan["recipe"], "command": command, "actual_capture_command": running,
                "capture_manifest": str(capture_path), "deferred_metadata": str(metadata),
                "geometry_locked_before_semantics_and_labels": True, "GT_input": False,
                "scheduled_count": 200, "completed_count": len(capture["frames"]),
                "missing_frame_ids": sorted(set(plan["schedule"]["frame_ids"]) - set(capture["completed_frame_ids"])),
                "invalid_pose_frame_ids": plan["invalid_pose_frame_ids"], "inputs": native_inputs,
                "outputs": outputs, "geometric_cache_mode": plan["geometric_cache_mode"], "geometry_variant": False,
                "new_visual_inference": 0, "elapsed_seconds": time.monotonic() - started,
                "native_pickle_status": "DEFERRED_FEATURES_REQUIRED"}
        except BaseException as exc:
            result = {"status": "FAILED", "scene": scene, "input_identity": identity,
                "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started,
                "partial_tsdf_resumption": False, "GT_input": False}
            result["identity"] = canonical_digest(result)
            atomic_write_json(receipt_path, result)
            raise
        result["identity"] = canonical_digest(result)
        atomic_write_json(receipt_path, result)
        index.write_memo(root / "input_verifications.json")
        print(f"CF18 native map {scene}: COMPLETE {result['completed_count']}/200", flush=True)
        return result


def _write_context(binding, scene, mapping, preparation, plan):
    data = dict(binding["scenes"][scene])
    data.update(capture_manifest=mapping["capture_manifest"], deferred_metadata=mapping["deferred_metadata"],
        parent_map_receipt=str(Path(plan["map_root"]) / "map_receipt.json"), actual_capture_command=mapping["actual_capture_command"],
        schedule=plan["schedule"]["frame_ids"], completed_frame_ids=read(mapping["capture_manifest"])["completed_frame_ids"],
        preparation_receipt=str(Path(binding["output_root"]) / "prepare" / scene / "receipt.json"),
        original_map_options=plan["recipe"], availability="LABEL_FREE_CAPTURE_COMPLETE", config=None,
        runtime={**data["runtime"], "data_root": str(Path(preparation["export_root"]).parents[1]),
                 "output_root": binding["output_root"], "upstream": plan["mapping_cwd"]},
        map_input_identity=mapping["input_identity"], GT_input=False)
    data["identity"] = canonical_digest(data)
    atomic_write_json(Path(binding["output_root"]) / "anchor_contexts" / (scene + ".json"), data)
    return data


def run_anchors(binding, *, scenes=None):
    spec = load_spec(binding["spec"])
    root = Path(binding["output_root"])
    selected = list(scenes if scenes is not None else [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]])
    if len(set(selected)) != len(selected) or not set(selected) <= set(sum(spec["cohorts"].values(), [])):
        raise ValueError("anchor phase must keep the fixed main captures")
    results, jobs = {}, {}
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    with exclusive_lock(root / "execution/.anchors_controller.lock"), ThreadPoolExecutor(max_workers=2) as pool:
        native_inputs = _native_inputs(binding, index)
        for scene in selected:
            if scene in spec["cohorts"]["replica8"]:
                data = binding["scenes"][scene]
                index.identity(data["parent_map_receipt"])
                mapping = read(data["parent_map_receipt"])
                if mapping["status"] != "COMPLETE" or mapping["map_id"] != "BB00_NATIVE":
                    raise ValueError("Replica anchor alias requires the actual complete native map")
                context = {**data, "binding_identity": binding["identity"], "GT_input": False,
                           "alias_kind": "EXACT_VERIFIED_IMMUTABLE_PARENT_MAP"}
                context["identity"] = canonical_digest(context)
                atomic_write_json(root / "anchor_contexts" / (scene + ".json"), context)
                results[scene] = {"status": "COMPLETE", "source": data["parent_map_receipt"], "new_mapping_work": 0}
                continue
            try:
                prep_path = root / "prepare" / scene / "receipt.json"
                index.identity(prep_path)
                preparation = read(prep_path)
                _verified_identity(preparation)
                plan = anchor_plan(binding, scene, preparation)
                if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=plan["mapping_cwd"], text=True).strip() != spec["upstream_commit"]:
                    raise ValueError("actual native upstream revision changed")
                front = _ensure_frontend(binding, scene, preparation, plan, index, native_inputs)
                future = pool.submit(_run_map, binding, scene, preparation, plan, front, native_inputs)
                jobs[future] = (scene, preparation, plan)
            except Exception as exc:
                results[scene] = {"status": "BLOCKED_ANCHOR_LEAF", "error": f"{type(exc).__name__}: {exc}", "capture_not_dropped": True}
                print(f"CF18 anchor {scene}: {results[scene]['status']} {exc}", flush=True)
        for future in as_completed(jobs):
            scene, preparation, plan = jobs[future]
            try:
                mapping = future.result()
                context = _write_context(binding, scene, mapping, preparation, plan)
                results[scene] = {"status": "COMPLETE", "context_identity": context["identity"], "new_mapping_work": 1}
            except Exception as exc:
                results[scene] = {"status": "BLOCKED_ANCHOR_LEAF", "error": f"{type(exc).__name__}: {exc}", "capture_not_dropped": True}
                print(f"CF18 anchor {scene}: {results[scene]['status']} {exc}", flush=True)
    index.write_memo(root / "validation/input_verifications.json")
    coverage = {"status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in results.values()) else "INCOMPLETE",
        "selected": selected, "results": results, "required_main_anchors": 26,
        "complete_main_contexts": sum((root / "anchor_contexts" / (scene + ".json")).exists()
                                      for scene in sum(spec["cohorts"].values(), []))}
    atomic_write_json(root / "anchors/coverage.json", coverage)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", action="append")
    args = parser.parse_args()
    result = run_anchors(read(args.binding), scenes=args.scene)
    print("ANCHORS", result["status"], result["complete_main_contexts"], "/", result["required_main_anchors"], flush=True)
