"""Fixed cross-dataset settings and read-only source/input binding."""

from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import SourceIndex, read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.released_loader import load_released_module

SCENES = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
METHODS = ("N0", "Q_GAIN", "S_SIGLIP2_AREA", "CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL", "CP_M4_GAIN_S2")
SOURCE_ROOT = Path("/mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001")
DEFAULT_ROOT = Path("/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001")
DATA_ROOT = Path("/home/ww/vv/dataset/Replica")


def vocabulary(upstream):
    source = Path(upstream) / "scripts"
    constants = load_released_module(source / "utils/semantic_const.py")
    names = tuple(constants["REPLICA_51"])
    ids = tuple(range(1, len(names) + 1))
    evaluator = load_released_module(source / "eval_utils.py")
    evaluator["init"]("Replica")
    ap_ids = tuple(map(int, evaluator["VALID_CLASS_IDS"]))
    if len(names) != 51 or len(set(names)) != 51 or len(ap_ids) != 48 or not set(ap_ids) <= set(ids):
        raise ValueError("released Replica vocabulary changed")
    return names, ids, ap_ids


def validate_transfer(config, frozen):
    if tuple(config["scenes"]) != SCENES:
        raise ValueError("fixed Replica scene set changed")
    if tuple(config["methods"]) != METHODS:
        raise ValueError("fixed transfer method set changed")
    if config["temperatures"] != frozen["final_temperatures"]["temperatures"]:
        raise ValueError("transferred temperatures changed")
    if config["checkpoint"] != frozen["checkpoint"]:
        raise ValueError("transferred checkpoint/scaler changed")
    if config["frame_ids"] != list(range(0, 2000, 10)) or config["budget"] != 200:
        raise ValueError("fixed frame schedule or query budget changed")


def bind(output=DEFAULT_ROOT):
    output = Path(output).resolve()
    frozen_path = SOURCE_ROOT / "selection.json"
    frozen = read_json(frozen_path)
    source = read_json(SOURCE_ROOT / "resolved_config.json")
    index = SourceIndex()
    selection = index.identity(frozen_path)
    index.identity(frozen["checkpoint"]["path"], frozen["checkpoint"])
    for model in frozen["models"].values():
        for entry in model["files"]:
            index.identity(entry["path"], entry)
    names, ids, ap_ids = vocabulary(source["runtime"]["upstream"])
    runtime = {**source["runtime"], "data_root": str(DATA_ROOT), "output_root": str(output / "native")}
    config = {
        "schema": 1, "dataset": "Replica", "role": "frozen_transfer",
        "attempt_root": str(output), "source_selection": selection,
        "scenes": list(SCENES), "methods": list(METHODS),
        "frame_ids": list(range(0, 2000, 10)), "budget": 200,
        "checkpoint": frozen["checkpoint"],
        "temperatures": frozen["final_temperatures"]["temperatures"],
        "runtime": runtime, "gpu_lock": source["gpu_lock"],
        "source_models": frozen["models"],
        "class_names": names, "valid_ids": ids, "ap_valid_ids": ap_ids,
        "no_replica_fitting": True, "historical_replica_exposure": True,
        "source_commit": "04e0287b0d05c946d7519f0b3100ca2e4716713f",
    }
    validate_transfer(config, frozen)
    config["identity"] = canonical_digest(config)
    write_once(output / "transfer.json", config)
    write_once(output / "transfer_source_manifest.json", index.manifest())
    return output / "transfer.json"


def require_transfer(config_path, scene=None, method=None):
    config = read_json(config_path)
    if canonical_digest({k: v for k, v in config.items() if k != "identity"}) != config["identity"]:
        raise ValueError("transfer contract changed")
    SourceIndex().identity(config["source_selection"]["path"], config["source_selection"])
    validate_transfer(config, read_json(config["source_selection"]["path"]))
    if scene is not None and scene not in config["scenes"]:
        raise ValueError("unlisted Replica scene")
    if method is not None and method not in config["methods"]:
        raise ValueError("unlisted Replica method")
    return config


def validate_scene_binding(config, contract):
    for key in ("checkpoint", "temperatures", "runtime", "gpu_lock"):
        if config[key] != contract[key]:
            raise ValueError(f"frozen transfer {key} changed")
    if set(config["models"]) != set(contract["source_models"]):
        raise ValueError("transfer model set changed")
    for name, model in config["models"].items():
        original = contract["source_models"][name]
        if model["identity"] != original["identity"] or model["files"] != original["files"]:
            raise ValueError("transferred visual model changed")
        if model["valid_ids"] != contract["valid_ids"] or model["class_names"] != contract["class_names"]:
            raise ValueError("Replica source vocabulary changed")


def require_access(config, scene, phase, method=None):
    if phase != "replica":
        raise ValueError("Replica leaves require their own transfer phase")
    contract = require_transfer(config["transfer_config"], scene, method)
    validate_scene_binding(config, contract)
    for name, model in config["models"].items():
        bound = read_json(Path(contract["attempt_root"]) / "text" / name / "model_binding.json")
        if model != bound:
            raise ValueError("transferred Replica text binding changed")
        SourceIndex().identity(model["text"]["path"], model["text"])
    if scene not in config["scenes"] or config["scenes"][scene]["schedule"] != contract["frame_ids"]:
        raise ValueError("unbound Replica scene or changed schedule")
    SourceIndex().identity(config["checkpoint"]["path"], config["checkpoint"])
    return "replica_transfer"


def kernel_identity():
    from src.static_ovmap.composition_study.execution import kernel_identity as original

    result = original()
    index = SourceIndex()
    entries = result["files"] + [index.identity(Path(__file__).with_name(name))
                                 for name in ("protocol.py", "query.py")]
    return {"files": entries, "digest": canonical_digest(entries)}


def bind_capture_inputs(config, scene):
    if scene not in SCENES:
        raise ValueError("unlisted Replica capture")
    native = Path(config["runtime"]["data_root"]) / scene
    frames = config["frame_ids"]
    poses = np.loadtxt(native / "traj.txt").reshape(-1, 4, 4)
    if len(poses) <= max(frames) or not np.isfinite(poses[frames]).all():
        raise ValueError("invalid scheduled Replica camera poses")
    camera = read_json(native.parent / "cam_params.json")["camera"]
    if camera["scale"] != 6553.5:
        raise ValueError("Replica depth scale differs from source protocol")
    paths = ["traj.txt", "../cam_params.json"] + [
        f"results/{kind}{frame:06d}.{ext}" for frame in frames
        for kind, ext in (("frame", "jpg"), ("depth", "png"))]
    index = SourceIndex()
    hashes = {relative: index.identity(native / relative)["sha256"] for relative in paths}
    schedule = {"start": 0, "end": 2000, "step": 10, "frame_ids": frames}
    row = {"scene_id": scene, "role": "frozen_transfer", "schedule": schedule}
    write_once(Path(config["attempt_root"]) / "native" / scene / "input_receipt.json", {
        "status": "BOUND_REPLICA_INPUTS", "schedule": schedule, "file_hashes": hashes,
        "invalid_pose_frame_ids": [], "pose_convention": "unchanged ReplicaLoader world c2w",
        "depth_scale": camera["scale"], "GT_input": False})
    return row, native
