"""Dataset adapter and ordered execution of frozen Replica transfer jobs."""

import fcntl
import os
import subprocess
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.binding import _bind_scene
from src.static_ovmap.composition_study.calibration_jobs import sources_for_scene
from src.static_ovmap.composition_study.execution import verified_receipt
from src.static_ovmap.composition_study.io import (
    ROOT,
    SourceIndex,
    read_json,
    write_once,
)
from src.static_ovmap.composition_study.label_fusion import fuse_labels
from src.static_ovmap.composition_study.object_evidence import (
    owner_labels,
    prepare_static_sources,
)
from src.static_ovmap.composition_study.pipeline import method_inputs
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_ground_truth import load_ground_truth
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.scannet_study import (
    build_native_prediction,
    freeze_projection,
    load_prediction,
    relabel_prediction,
    save_prediction,
)
from src.static_ovmap.module_validation.semantic_models import load_semantic_records
from src.static_ovmap.module_validation.semantic_study import (
    direct_readouts,
    prepare_semantic_manifest,
)
from src.static_ovmap.module_validation.study_scene import audit_native_export

from .ground_truth import prepare_ground_truth
from .protocol import METHODS, require_access


def invoke(config, path, phase, scene=None, method=None, model=None):
    semantic = phase == "semantic" or (phase == "query" and method == "CP_M4_GAIN_S2") or (phase == "text" and model == "siglip2")
    python = config["runtime"]["semantic_python" if semantic else "native_perception_python"]
    command = [python, str(ROOT / "scripts/evaluation/run_ovimap_replica_transfer.py"), "--config", str(path), "--phase", phase]
    for key, value in (("scene", scene), ("method", method), ("model", model)):
        if value is not None:
            command.extend(("--" + key, value))
    destination = Path(config["attempt_root"]) / "logs" / (scene or "shared")
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / (phase + "_" + (method or model or "scene") + ".log")
    print(f"Replica: {scene or 'shared'} {phase} {method or model or ''}", flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Replica {phase} failed ({result.returncode}): {log_path}")


def scene_root(config, scene):
    return Path(config["attempt_root"]) / "scenes" / scene


def prepare_scene(config, path, scene):
    root = scene_root(config, scene)
    runtime = config["runtime"]
    models = {name: read_json(Path(config["attempt_root"]) / "text" / name / "model_binding.json") for name in ("native", "siglip2")}
    study = root / "study"
    annotations = prepare_ground_truth(runtime["upstream"], runtime["data_root"], scene, study / "annotations" / scene)
    mapping = verified_receipt(Path(runtime["output_root"]) / scene / "mapping_job/receipt.json")
    capture_path = Path(mapping["capture_manifest"])
    capture = read_json(capture_path)
    targets = load_ground_truth(annotations)
    base = study / "scenes" / scene
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        projection = freeze_projection(arrays["surface_xyz"], targets["xyz"], base / "projection")
    native_path = build_native_prediction(capture_path, Path(mapping["native_features"]), Path(models["native"]["text"]["path"]), projection, base / "baseline")
    native = load_prediction(native_path)
    targets.update(nearest=projection["nearest"], matched=projection["matched"])
    parity = audit_native_export(native, targets, Path(runtime["upstream"]), base / "native_export_parity")
    write_once(base / "native_export_parity.json", parity)
    manifest = prepare_semantic_manifest(capture_path, native, base / "semantic_requests.json")
    source = {"study_root": str(study), "native_text_cache": models["native"]["text"]["path"], "siglip2_model": models["siglip2"]["path"]}
    write_once(root / "semantic_config.json", source)
    invoke(config, path, "semantic", scene)
    semantic_root = base / "semantic_models/siglip2"
    records = load_semantic_records(semantic_root)
    with np.load(models["siglip2"]["text"]["path"], allow_pickle=False) as arrays:
        suggestions = direct_readouts(manifest, records, arrays["text_embeddings"], tuple(map(int, arrays["valid_ids"])), owner_labels(native), model="siglip2")["S_SIGLIP2_AREA"]
    labels = {owner: int(value["label_id"]) for owner, value in suggestions.items()}
    receipt = read_json(semantic_root / "receipt.json")
    costs = {**dict(native.logical_cost), "added_attempts": len(records), "added_crop_inputs": 6 * len(records), "added_background_inputs": 0, "generations": 0, "added_seconds": receipt["request_seconds"]}
    payload = relabel_prediction(native, "S_SIGLIP2_AREA", "S", labels, costs, {"request_manifest": manifest["identity"], "readout_id": "S_SIGLIP2_AREA"})
    save_prediction(payload, base / "semantic/direct/S_SIGLIP2_AREA")
    index = SourceIndex()
    for model in models.values():
        index.identity(model["text"]["path"], model["text"])
    bound = _bind_scene(index, scene, "replica_transfer", runtime, source)
    child = {"transfer_config": str(Path(path).resolve()), "attempt_root": str(root), "binding_key": config["identity"], "runtime": runtime,
             "source": source, "scenes": {scene: bound}, "models": models,
             "checkpoint": config["checkpoint"], "temperatures": config["temperatures"], "gpu_lock": config["gpu_lock"]}
    require_access(child, scene, "replica")
    write_once(root / "source_manifest.json", index.manifest())
    write_once(root / "resolved_config.json", child)
    prepare_static_sources(child, scene, root / "sources" / scene)
    return child


def semantic(config, scene):
    from src.static_ovmap.module_validation.semantic_models import (
        encode_semantic_requests,
    )

    root = scene_root(config, scene)
    base = root / "study/scenes" / scene
    output = base / "semantic_models/siglip2"
    with Path(config["gpu_lock"]).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(config["runtime"]["cuda_device"]), output)
        return encode_semantic_requests(base / "semantic_requests.json", "siglip2", read_json(root / "semantic_config.json"), output)


def compose(config, scene):
    require_access(config, scene, "replica")
    sources = sources_for_scene(config, scene)
    root = Path(config["attempt_root"])
    native = load_prediction(config["scenes"][scene]["native_prediction"])
    query = verified_receipt(root / "query" / scene / "Q_GAIN/receipt.json")
    for method in ("CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL"):
        temperatures = config["temperatures"] if method == "CP_M2_EQUAL_CAL" else {name: 0.07 for name in sources}
        target, decisions = fuse_labels(owner_labels(native), sources, config["models"]["native"]["valid_ids"], method, temperatures)
        costs = {key: sum(source["required_logical"][key] for source in sources.values()) for key in ("native_requests", "siglip2_requests", "crop_inputs")}
        payload = relabel_prediction(native, method, "S", target, costs, {"source_identities": {name: source["identity"] for name, source in sources.items()}, "temperatures": temperatures})
        output = root / "compositions" / scene / method
        prediction = save_prediction(payload, output / "prediction")
        write_once(output / "decisions.json", {"scene_id": scene, "method_id": method, "objects": decisions})
        index = SourceIndex()
        inputs = [index.identity(root / "sources" / scene / (name + ".json")) for name in ("N0", "S_SIGLIP2_AREA")]
        inputs += [index.identity(query["evidence_path"]), index.identity(Path(__file__)), index.identity(config["transfer_config"])]
        outputs = [index.identity(p) for p in (prediction, prediction.parent / "prediction.npz", output / "decisions.json")]
        write_once(output / "receipt.json", {"status": "COMPLETE", "input_identity": canonical_digest({"prediction": payload.record_key, "inputs": inputs}),
            "method_id": method, "scene_id": scene, "prediction_manifest": str(prediction), "required_logical": costs,
            "physical": {"Q_GAIN": query["physical"], "static_sources_reused": True, "fusion_model_forwards": 0, "shared_dependency_count_once_globally": True}, "inputs": inputs, "outputs": outputs})


def run_phase(config, path, phase, *, scene=None, method=None, model=None):
    scenes = [scene] if scene else config["scenes"]
    if phase == "text":
        from .text import prepare

        if model is None:
            for name in ("native", "siglip2"):
                invoke(config, path, "text", model=name)
        else:
            prepare(config, model)
    elif phase == "semantic":
        if scene is None:
            raise ValueError("semantic worker requires one scene")
        semantic(config, scene)
    elif phase == "prepare":
        for name in scenes:
            prepare_scene(config, path, name)
    elif phase in ("query", "fuse", "evaluate"):
        for name in scenes:
            child = read_json(scene_root(config, name) / "resolved_config.json")
            require_access(child, name, "replica", method)
            if phase == "query":
                from .query import run_query

                if method not in ("Q_GAIN", "CP_M4_GAIN_S2"):
                    raise ValueError("query leaf requires Q_GAIN or CP_M4_GAIN_S2")
                run_query(child, name, method, "replica")
            elif phase == "fuse":
                compose(child, name)
            else:
                from .evaluation import evaluate_method

                for current in ([method] if method else METHODS):
                    prediction, logical, physical, reuse = method_inputs(child, name, current)
                    evaluate_method(child, name, current, "replica", prediction, logical=logical, physical=physical, reuse_kind=reuse)
    elif phase == "all":
        for name in ("native", "siglip2"):
            invoke(config, path, "text", model=name)
        for name in scenes:
            invoke(config, path, "capture", name)
            invoke(config, path, "prepare", name)
            for current in ("Q_GAIN", "CP_M4_GAIN_S2"):
                invoke(config, path, "query", name, current)
            invoke(config, path, "fuse", name)
            invoke(config, path, "evaluate", name)
        return run_phase(config, path, "report")
    elif phase == "report":
        from .reporting import report

        return report(config)
    else:
        raise ValueError("unknown Replica phase")
    return {"status": "COMPLETE", "phase": phase, "scenes": scenes}
