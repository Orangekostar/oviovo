"""Hash-bound execution of the preselected ScanNet development captures."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from .assets import sha256_file
from .boundary_jobs import capture_inputs, file_identity, verify_capture
from .contracts import atomic_write_json, canonical_digest

ROOT = Path(__file__).resolve().parents[3]
PIN = "f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424"


def development_rows(locked: dict, requested: list[str] | None = None) -> list[dict]:
    rows = [row for row in locked["selected"] if row["role"] in {"fit", "cal", "select"}]
    if requested is not None:
        allowed = {row["scene_id"] for row in rows}
        if not set(requested) <= allowed:
            raise ValueError("capture is restricted to locked development scenes")
        rows = [row for row in rows if row["scene_id"] in requested]
    return rows


def reusable_job(path: Path, identity: str) -> bool:
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    return (receipt.get("status") == "COMPLETE" and receipt.get("input_identity") == identity
            and bool(receipt.get("outputs")) and all(
                Path(row["path"]).is_file() and sha256_file(row["path"]) == row["sha256"]
                for row in receipt["outputs"]))


def _tree_inputs(root: Path, pattern: str = "*") -> list[dict]:
    return [file_identity(path) for path in sorted(root.rglob(pattern)) if path.is_file()
            and not any(part in {".git", ".cache", "__pycache__"} for part in path.parts)]


def require_idle_gpu(device: str, output: Path, *, sample: str | None = None) -> None:
    if sample is None:
        # NVML can briefly report the just-exited visual process's context and
        # utilization. Recheck for at most 15 seconds; never waive occupancy.
        deadline = time.monotonic() + 15
        while True:
            sample = subprocess.check_output(["nvidia-smi", "-i", device,
                "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            memory, utilization = (int(value.strip()) for value in sample.split(","))
            if (memory <= 512 and utilization <= 5) or time.monotonic() >= deadline:
                break
            time.sleep(1)
    memory, utilization = (int(value.strip()) for value in sample.split(","))
    if memory > 512 or utilization > 5:
        atomic_write_json(output / "resource_status.json", {"status": "BLOCKED_GPU_BUSY",
            "device": device, "memory_used_mib": memory, "utilization_percent": utilization,
            "checked_unix": time.time()})
        raise RuntimeError(f"GPU_BUSY: CUDA device {device}: {memory} MiB, {utilization}% utilization")
    atomic_write_json(output / "resource_status.json", {"status": "AVAILABLE", "device": device,
        "checked_unix": time.time()})


def _execute(command: list[str], cwd: Path, env: dict, job: Path, identity: str) -> float:
    job.mkdir(parents=True, exist_ok=True)
    running = job / "running.json"
    if running.exists() and json.loads(running.read_text())["input_identity"] != identity:
        raise ValueError(f"interrupted job input identity differs; use a new output root: {job}")
    started = time.monotonic()
    atomic_write_json(job / "running.json", {"command": command, "cwd": str(cwd),
        "input_identity": identity, "started_unix": time.time(),
        "environment": {key: env.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "LD_LIBRARY_PATH", "PYTHONPATH",
            "OVIMAP_NATIVE_MODEL", "OVIMAP_NATIVE_UPSTREAM", "HF_MODULES_CACHE")}})
    with (job / "runtime.log").open("a") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"job failed ({result.returncode}); see {job / 'runtime.log'}")
    return time.monotonic() - started


def preserve_interrupted_replay(base: Path, identity: str) -> Path | None:
    """Native TSDF has no checkpoint: preserve the failed attempt before restart."""
    if not (base / "capture").exists():
        return None
    running = base / "mapping_job/running.json"
    if not running.is_file() or json.loads(running.read_text())["input_identity"] != identity:
        raise ValueError("interrupted native replay input identity differs")
    index = 1
    while (base / f"interrupted_mapping_{index:03d}").exists():
        index += 1
    archive = base / f"interrupted_mapping_{index:03d}"
    archive.mkdir()
    for name in ("capture", "mapper", "mapping_job"):
        path = base / name
        if path.exists():
            path.rename(archive / name)
    return archive


def capture_development(config: dict, *, scenes: list[str] | None = None, confirmation_lock=None) -> dict:
    """Run CropFormer then the pinned native CPU mapper/FP32 visual worker."""
    data = Path(config["data_root"])
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    # This lock also serializes visual jobs sharing the designated study GPU.
    with (output.parent / f".visual-gpu-{config['cuda_device']}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _capture_development(config, data, output, scenes, confirmation_lock=confirmation_lock)


def _capture_development(config: dict, data: Path, output: Path, scenes, *, confirmation_lock=None) -> dict:
    lock_path = data / "acquisition_lock.json"
    locked = json.loads(lock_path.read_text())
    if confirmation_lock is not None:
        from .confirmation_access import confirmation_contract, confirmation_rows

        contract = confirmation_contract(confirmation_lock, config)
        rows = confirmation_rows(locked, contract, scenes)
    else:
        rows = development_rows(locked, scenes)
    downloads = json.loads((data / "download_receipt.json").read_text())
    if downloads["status"] != "RAW_DOWNLOADS_COMPLETE" or downloads["lock_sha256"] != sha256_file(lock_path):
        raise ValueError("raw data receipt does not match the frozen acquisition lock")
    if not rows:
        raise ValueError("no locked development captures selected")
    upstream = Path(config["upstream"])
    actual_pin = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != PIN:
        raise ValueError("native upstream commit differs from the frozen protocol")
    patch = ROOT / "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch"
    import hashlib
    difference = subprocess.check_output(["git", "-C", str(upstream), "diff", "--binary", "--unified=50", "HEAD"])
    if hashlib.sha256(difference).hexdigest() != sha256_file(patch):
        raise ValueError("native upstream patch identity differs")
    extension = Path(config["native_extension"])
    build_receipt = json.loads(Path(config["native_build_receipt"]).read_text())
    if build_receipt["status"] != "COMPLETE":
        raise ValueError("native extension has not passed the bound replay")
    if build_receipt["extension"] != file_identity(extension):
        raise ValueError("native extension differs from the verified build")
    front_root = Path(config["cropformer_root"])
    front_py = Path(config["frontend_python"])
    map_py = Path(config["mapping_python"])
    wrapper = ROOT / "scripts/evaluation/ovimap_module_perception_worker.py"
    helper = Path(__file__).with_name("native_capture.py")
    front_inputs = [file_identity(config[key]) for key in ("cropformer_config", "cropformer_weights")]
    front_inputs += _tree_inputs(front_root, "*.py") + _tree_inputs(front_root / "configs", "*.yaml")
    map_inputs = _tree_inputs(upstream / "scripts", "*.py")
    map_inputs += [file_identity(path) for path in (extension, patch, wrapper, helper)]
    map_inputs += _tree_inputs(Path(config["native_model"]))
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(config["cuda_device"]), "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "OPENCV_FOR_THREADS_NUM": "8",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_MODULES_CACHE": config["hf_modules_cache"]})
    front_env = dict(env)
    front_env.update({"PATH": f"{front_py.parent}:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{front_py.parents[1] / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        "PYTHONPATH": str(front_root)})
    baseline_lib = Path(config["baseline_build"]) / "mapping_ros_ws/devel/lib"
    map_env = dict(env)
    map_env.update({"LD_LIBRARY_PATH": ":".join(map(str, (extension.parent, baseline_lib, map_py.parents[1] / "lib"))),
        "PYTHONPATH": ":".join(map(str, (extension.parent, baseline_lib, upstream / "scripts", ROOT))),
        "OVIMAP_MODULE_VALIDATION_HELPER_ROOT": str(ROOT),
        "OVIMAP_NATIVE_MODEL": config["native_model"], "OVIMAP_NATIVE_UPSTREAM": str(upstream)})
    completed = {}
    for row in rows:
        scene = row["scene_id"]
        native_root = data / "exported" / scene
        export_path = native_root / "export_receipt.json"
        export = json.loads(export_path.read_text())
        if export["schedule"] != row["schedule"] or export["status"] != "EXPORTED_NATIVE_INPUTS":
            raise ValueError(f"export schedule mismatch: {scene}")
        for relative, digest in export["file_hashes"].items():
            if sha256_file(native_root / relative) != digest:
                raise ValueError(f"export changed: {scene}/{relative}")
        export_inputs = [{"path": str(native_root / relative), "sha256": digest,
                          "bytes": (native_root / relative).stat().st_size}
                         for relative, digest in export["file_hashes"].items()]
        base = output / scene
        frontend = base / "frontend"
        frontend.mkdir(parents=True, exist_ok=True)
        schedule = row["schedule"]
        invalid = set(export["invalid_pose_frame_ids"])
        images = [str(native_root / "color" / f"{frame}.jpg") for frame in schedule["frame_ids"] if frame not in invalid]
        front_command = [str(front_py), str(front_root / "demo_cropformer/demo_from_dirs.py"),
            "--config-file", config["cropformer_config"], "--input", *images, "--output", str(frontend),
            "--confidence-threshold", "0.5", "--out-type", "0", "--opts", "MODEL.WEIGHTS", config["cropformer_weights"]]
        front_identity = canonical_digest({"command": front_command, "source": front_inputs,
            "export": file_identity(export_path), "environment": front_env["PYTHONPATH"], "schema": 1})
        front_receipt = base / "frontend_job/receipt.json"
        if not reusable_job(front_receipt, front_identity):
            if front_receipt.exists():
                raise ValueError(f"changed completed frontend requires a new output root: {scene}")
            require_idle_gpu(str(config["cuda_device"]), output)
            print(f"{scene}: CropFormer {len(images)} scheduled valid frames", flush=True)
            elapsed = _execute(front_command, front_root, front_env, front_receipt.parent, front_identity)
            outputs = [file_identity(frontend / f"{frame}.png") for frame in schedule["frame_ids"] if frame not in invalid]
            outputs += [file_identity(frontend / "frame_diagnostics" / f"{frame}.json")
                        for frame in schedule["frame_ids"] if frame not in invalid]
            atomic_write_json(front_receipt, {"status": "COMPLETE", "input_identity": front_identity,
                "elapsed_seconds": elapsed, "outputs": outputs})
        capture_root = base / "capture"
        map_env["OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT"] = str(capture_root)
        command = [str(map_py), str(upstream / "scripts/panoptic_mapping_.py"),
            "--dataset", "scannet_nyu", "--task", "Nyu40", "--scene_num", scene,
            "--data_folder", str(native_root.parent), "--result_folder", str(base / "mapper"),
            "--start", str(schedule["start"]), "--end", str(schedule["end"]), "--step", str(schedule["step"]),
            "--num_threads", "8", "--data_association", "2", "--inst_association", "4",
            "--seg_graph_confidence", "3", "--use_inst_label_connect", "1", "--connection_ratio_th", "0.2",
            "--use_temp_panoptics", "--temp_panoptics_folder", str(frontend),
            "--save_temp_geometrics", "--temp_geometrics_folder", str(base / "geometrics"),
            "--save_temp_results", "--intermediate_seg_folder", str(base / "segments"),
            "--perception_python", config["native_perception_python"], "--perception_worker", str(wrapper)]
        identity = canonical_digest({"command": command, "source": map_inputs, "frontend": file_identity(front_receipt),
            "export": file_identity(export_path), "schema": 1, "precision": "float32"})
        receipt_path = base / "mapping_job/receipt.json"
        if not reusable_job(receipt_path, identity):
            if receipt_path.exists():
                raise ValueError(f"changed completed native capture requires a new output root: {scene}")
            require_idle_gpu(str(config["cuda_device"]), output)
            preserve_interrupted_replay(base, identity)
            print(f"{scene}: native mapping, 200 scheduled slots, FP32 SigLIP", flush=True)
            elapsed = _execute(command, upstream, map_env, receipt_path.parent, identity)
            manifest_path = capture_root / scene / "manifest.json"
            manifest = verify_capture(manifest_path, allow_skipped=True)
            if manifest["scheduled_frame_ids"] != schedule["frame_ids"]:
                raise ValueError("native mapper changed the frozen frame schedule")
            if manifest["native_extension"]["sha256"] != sha256_file(extension):
                raise ValueError("native mapper loaded an unbound extension")
            feature_path = base / "mapper/cropformer_inst/inst_sem_siglip-l-16-384_200_incre_combine.pkl"
            outputs = [file_identity(item["path"]) for item in capture_inputs(manifest_path, manifest)]
            outputs.append(file_identity(feature_path))
            atomic_write_json(receipt_path, {"status": "COMPLETE", "scene_id": scene, "role": row["role"],
                "input_identity": identity, "elapsed_seconds": elapsed, "outputs": outputs,
                "input_identities": map_inputs + front_inputs + export_inputs
                + json.loads(front_receipt.read_text())["outputs"]
                + [file_identity(front_receipt), file_identity(export_path), file_identity(config["native_build_receipt"])],
                "capture_manifest": str(manifest_path), "native_features": str(feature_path),
                "scheduled_count": 200, "completed_count": len(manifest["frames"]),
                "missing_frame_ids": sorted(set(schedule["frame_ids"]) - set(manifest["completed_frame_ids"])),
                "invalid_pose_frame_ids": sorted(invalid), "precision": "float32"})
        completed[scene] = file_identity(receipt_path)
        progress_name = "confirmation_capture_progress.json" if confirmation_lock is not None else "capture_progress.json"
        atomic_write_json(output / progress_name, {"status": "IN_PROGRESS", "scenes": completed,
            "requested": [r["scene_id"] for r in rows], "acquisition_lock": file_identity(lock_path)})
        print(f"{scene}: native capture verified", flush=True)
    result = {"status": "COMPLETE", "scenes": completed, "acquisition_lock": file_identity(lock_path),
        "confirmation_processing": "AUTHORIZED_FROZEN_SELECTION" if confirmation_lock is not None
        else "DEFERRED_UNTIL_FROZEN_SELECTION"}
    receipt_name = "confirmation_capture_receipt.json" if confirmation_lock is not None else "capture_receipt.json"
    atomic_write_json(output / receipt_name, result)
    return result
