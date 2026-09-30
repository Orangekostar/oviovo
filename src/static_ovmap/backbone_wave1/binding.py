"""Bind consumed historical inputs without importing historical predictions."""

import json
import os
from pathlib import Path
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.m2_reviewer_study.binding import InputIndex


def read(path):
    return json.loads(Path(path).read_text())


def option(command, name):
    return command[command.index(name) + 1]


def set_option(command, name, value):
    command[command.index(name) + 1] = str(value)


def bind_inputs(spec_path, repo_root, output_root):
    spec_path, repo_root, root = Path(spec_path), Path(repo_root), Path(output_root)
    spec, index = read(spec_path), InputIndex()
    index.identity(spec_path)
    wave, paired = read(spec["parent_wave1_binding"]), read(spec["parent_paired_binding"])
    index.identity(spec["parent_wave1_binding"])
    index.identity(spec["parent_paired_binding"])
    reviewer = read(wave["reviewer_binding"])
    index.identity(wave["reviewer_binding"])
    composition = Path(reviewer["composition_root"])
    scan_path = composition / "resolved_config.json"
    scan_config = read(scan_path)
    calibration_paths = {"original_folds": composition / "calibration/folds.json",
        "original_final": Path(reviewer["transfer"]),
        "native_folds": Path(reviewer["output_root"]) / "calibration/new_folds.json",
        "native_final": Path(reviewer["output_root"]) / "calibration/new_final.json",
        "fc": Path(wave["output_root"]) / "calibration/AW_E03_FC_FROZEN.json"}
    calibrations = {name: read(path) for name, path in calibration_paths.items()}
    for path in calibration_paths.values():
        index.identity(path)
    scenes = {}
    for dataset, names in [("ScanNet", spec["datasets"]["development"]), ("Replica", spec["datasets"]["replica"])]:
        for scene in names:
            config_path = scan_path if dataset == "ScanNet" else Path(wave["scenes"][scene]["config"])
            index.identity(config_path)
            config = scan_config if dataset == "ScanNet" else read(config_path)
            data = config["scenes"][scene]
            capture_path, mapping_path = Path(data["capture"]), Path(data["mapping_receipt"])
            index.identity(capture_path)
            index.identity(mapping_path)
            running_path = mapping_path.with_name("running.json")
            index.identity(running_path)
            command = read(running_path)
            capture, mapping = read(capture_path), read(mapping_path)
            schedule = capture["scheduled_frame_ids"]
            start, end, step = [int(option(command["command"], x)) for x in ("--start", "--end", "--step")]
            if len(schedule) != 200 or schedule != list(range(start, end, step)):
                raise ValueError(f"historical 200-slot command/capture mismatch: {scene}")
            if mapping["status"] != "COMPLETE":
                raise ValueError(f"parent native map is incomplete: {scene}")
            for flag, value in [("--data_association", 2), ("--inst_association", 4),
                                ("--seg_graph_confidence", 3), ("--use_inst_label_connect", 1)]:
                if int(option(command["command"], flag)) != value:
                    raise ValueError(f"inherited native mode differs: {scene}/{flag}")
            foreground = Path(option(command["command"], "--temp_panoptics_folder"))
            geometry = Path(option(command["command"], "--temp_geometrics_folder"))
            # Only original inputs used by this task are checked, not parent result trees.
            front_receipt = mapping_path.parent.parent / "frontend_job/receipt.json"
            index.identity(front_receipt)
            for item in read(front_receipt)["outputs"]:
                if Path(item["path"]).suffix == ".png":
                    index.identity(item["path"], item)
            for frame in capture["frames"]:
                index.identity(geometry / f"{frame['frame_id']:05d}_mask.png")
            native_model = config["models"]["native"]
            for item in native_model["files"]:
                index.identity(item["path"], item)
            index.identity(native_model["text"]["path"], native_model["text"])
            if scene in spec["datasets"]["parent_calibration"]:
                old = calibrations["original_folds"]["folds"][scene]
                temperatures = {"N": calibrations["native_folds"]["folds"][scene]["cosine_N0"]["temperature"],
                                "Q": old["Q_GAIN"]["temperature"], "F": calibrations["fc"]["folds"][scene]["temperature"]}
            else:
                temperatures = {"N": calibrations["native_final"]["cosine_N0"]["temperature"],
                                "Q": calibrations["original_final"]["temperatures"]["Q_GAIN"],
                                "F": calibrations["fc"]["final"]["temperature"]}
            if dataset == "ScanNet":
                exported = Path(option(command["command"], "--data_folder")) / scene
                export_path = exported / "export_receipt.json"
                index.identity(export_path)
                export = read(export_path)
                if export["schedule"]["frame_ids"] != schedule:
                    raise ValueError("export schedule differs")
                for relative, sha in export["file_hashes"].items():
                    index.identity(exported / relative, {"sha256": sha})
            else:
                # Bind acquired RGB/depth/poses through the actual loader input paths.
                data_root = Path(option(command["command"], "--data_folder")) / scene
                for frame in capture["frames"]:
                    for key in ("rgb", "depth"):
                        source = frame["source_paths"][key]
                        index.identity(source)
                index.identity(data_root / "traj.txt")
                index.identity(data_root.parent / "cam_params.json")
            scenes[scene] = {"dataset": dataset, "parent_config": str(config_path),
                "parent_scene": data, "parent_capture": str(capture_path), "parent_mapping": str(mapping_path),
                "parent_command": command, "schedule": schedule,
                "invalid_pose_frame_ids": mapping.get("invalid_pose_frame_ids", []),
                "parent_completed_frame_ids": capture["completed_frame_ids"],
                "cropformer_root": str(foreground), "geometric_root": str(geometry),
                "models": {"native": native_model}, "temperatures": temperatures,
                "checkpoint": config["checkpoint"], "runtime": config["runtime"],
                "exposure": "EXPOSED_DEVELOPMENT" if dataset == "ScanNet" else "EXPOSED_REPLICA"}
    asset_root = Path(wave["output_root"]).parent / "assets"
    sam_path = Path(spec["frontend"]["assets_receipt"])
    resolved_sam = sam_path if sam_path.is_file() else asset_root / "sam2/download_receipt.json"
    sam = read(resolved_sam)
    index.identity(resolved_sam)
    index.identity(sam["checkpoint"]["path"], sam["checkpoint"])
    fc_path = asset_root / "fc_frozen/download_receipt.json"
    index.identity(fc_path)
    fc = read(fc_path)
    for item in fc["files"]:
        index.identity(item["path"], item)
    workers_path = Path(wave["output_root"]) / "workers.json"
    index.identity(workers_path)
    workers = read(workers_path)
    sam_repo = "/mnt/shared/ww/ovimap-backbone-wave1-v1/tooling/sam2-source-pin"
    sam_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sam_repo, text=True).strip()
    if sam_commit != spec["frontend"]["sam2_commit"]:
        raise ValueError("SAM2 code pin differs")
    for code_path in sorted((Path(sam_repo) / "sam2").rglob("*")):
        if code_path.suffix in (".py", ".yaml"):
            relative = code_path.relative_to(sam_repo)
            pinned = subprocess.check_output(["git", "show", f"{sam_commit}:{relative}"], cwd=sam_repo)
            content = os.readlink(code_path).encode() if code_path.is_symlink() else code_path.read_bytes()
            if content != pinned:
                raise ValueError(f"SAM2 pinned source changed: {relative}")
            index.identity(code_path)
    result = {"status": "BOUND", "schema_version": 1, "spec": str(spec_path.resolve()),
        "output_root": str(root.resolve()), "repository_root": str(repo_root.resolve()),
        "deployment": "N0_UNCHANGED", "scenes": scenes, "assets_root": str(asset_root),
        "sam2": {"declared_receipt": str(sam_path), "resolved_receipt": str(resolved_sam),
                 "checkpoint": sam["checkpoint"], "code": sam_repo, "commit": sam_commit,
                 "python": workers["sam2"], "config": "configs/sam2.1/sam2.1_hiera_l.yaml"},
        "fc": {"receipt": str(fc_path), "model_root": str(asset_root / "fc_frozen/model"),
               "operator_root": str(asset_root / "ovrcoat/code"), "python": workers["region"]},
        "inherited_ratio_threshold": 0.0, "calibration_receipts": {k: str(v) for k, v in calibration_paths.items()},
        "inputs": index.entries(), "map_variants": spec["map_variants"],
        "map_scene_budget": 64, "primary_row_budget": 192, "scene_role_remap_authorized": True}
    result["identity"] = canonical_digest(result)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "resolved_inputs.json"
    if path.is_file() and read(path) != result:
        raise ValueError("resolved binding changed; use a fresh attempt root")
    atomic_write_json(path, result)
    atomic_write_json(root / "matrix.json", {"development": [
        {"scene": scene, "map_id": arm["id"], "readouts": spec["semantics"]["readouts"]}
        for scene in spec["datasets"]["development"] for arm in spec["map_variants"]],
        "replica": "UNDEFINED_UNTIL_FINAL_SELECTION_FREEZE", "input_identity": result["identity"]})
    return result
