"""Ordered composition jobs: predictions, evaluation, CAL freeze, then holdout."""

import os
import subprocess
from pathlib import Path

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    relabel_prediction,
    save_prediction,
)

from .calibration_jobs import (
    fit_folds,
    load_source,
    refit_after_nomination,
    sources_for_scene,
)
from .evaluation import evaluate_method
from .execution import READERS, require_access, run_query, verified_receipt
from .io import ROOT, SourceIndex, code_identity, read_json, write_once
from .label_fusion import METHODS, fuse_labels
from .object_evidence import owner_labels, prepare_static_sources
from .selection import COMPOSITIONS, CONTROLS, m6_gate, nominate


def _root(config):
    return Path(config["attempt_root"])


def _sources(config, scene):
    base = _root(config) / "sources" / scene
    return {
        name: load_source(base / (name + ".json")) for name in ("N0", "S_SIGLIP2_AREA")
    }


def method_inputs(config, scene, method):
    sources = _sources(config, scene)
    common = sources["N0"]["required_logical"]["native_requests"]
    if method in ("N0", "S_SIGLIP2_AREA"):
        path = config["scenes"][scene]["controls"][method]["prediction"]
        logical = dict(sources[method]["required_logical"])
        if method == "S_SIGLIP2_AREA":
            logical["native_requests"] += common
            logical["crop_inputs"] += 6 * common
        physical = {
            "model_loads": 0,
            "model_forwards": 0,
            "crop_inputs": 0,
            "reused_source": True,
        }
        reuse_kind = "EXACT_FROZEN_SOURCE_PREDICTION"
    elif method in METHODS:
        receipt = verified_receipt(
            _root(config) / "compositions" / scene / method / "receipt.json"
        )
        path, logical, physical = (
            receipt["prediction_manifest"],
            receipt["required_logical"],
            receipt["physical"],
        )
        reuse_kind = "NEW_STATIC_LABEL_FUSION"
    else:
        receipt = verified_receipt(
            _root(config) / "query" / scene / method / "receipt.json"
        )
        path, logical, physical = (
            receipt["prediction_manifest"],
            dict(receipt["required_logical"]),
            receipt["physical"],
        )
        reuse_kind = receipt["execution_mode"]
        if method in READERS:
            parent = verified_receipt(
                _root(config) / "query" / scene / READERS[method] / "receipt.json"
            )
            physical = {
                "reader": physical,
                "controller": parent["physical"],
                "shared_dependency_count_once_globally": True,
            }
    logical = {
        **logical,
        "accounting": "CONSERVATIVE_ADDITIVE_SOURCE_REQUESTS",
        "common_native_map_requests": common,
    }
    return path, logical, physical, reuse_kind


def run_leaf(config, phase, job, scene, method):
    require_access(config, scene, phase, method)
    if job == "query":
        row = run_query(config, scene, method, phase)
        return {
            key: row[key]
            for key in ("status", "scene_id", "method_id", "logical", "physical")
        }
    if job == "static":
        result = prepare_static_sources(
            config, scene, _root(config) / "sources" / scene
        )
        return {
            "status": "COMPLETE",
            "scene_id": scene,
            "sources": {key: value["identity"] for key, value in result.items()},
        }
    if job == "evaluate":
        path, logical, physical, reuse_kind = method_inputs(config, scene, method)
        return evaluate_method(
            config,
            scene,
            method,
            phase,
            path,
            logical=logical,
            physical=physical,
            reuse_kind=reuse_kind,
        )
    raise ValueError("unknown leaf job")


def invoke_leaf(config, config_path, phase, job, scene, method=None):
    python = (
        config["runtime"]["semantic_python"]
        if job == "query" and method in READERS
        else config["runtime"]["native_perception_python"]
    )
    command = [
        python,
        str(ROOT / "scripts/evaluation/run_ovimap_composition_study.py"),
        "--resolved-config",
        str(config_path),
        "--phase",
        phase,
        "--job",
        job,
        "--scene",
        scene,
    ]
    if method is not None:
        command += ["--method", method]
    destination = _root(config) / "logs" / phase / scene
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / (job + "_" + (method or "sources") + ".log")
    print(f"{phase}: {scene} {job} {method or 'sources'}", flush=True)
    with log_path.open("a") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"real leaf failed ({result.returncode}): {log_path}")


def compose_static(config, scene, phase):
    require_access(config, scene, phase)
    sources = sources_for_scene(config, scene)
    root = _root(config)
    native = load_prediction(config["scenes"][scene]["native_prediction"])
    labels = owner_labels(native)
    if phase == "compose-cal":
        fits = read_json(root / "calibration/folds.json")["folds"][scene]
        temperatures = {name: fit["temperature"] for name, fit in fits.items()}
    else:
        temperatures = read_json(root / "selection.json")["final_temperatures"][
            "temperatures"
        ]
    allowed = list(METHODS)
    if phase == "confirm":
        allowed = [
            name
            for name in METHODS
            if name
            in read_json(root / "selection.json")["nomination"]["confirmation_methods"]
        ]
    for method in allowed:
        require_access(config, scene, phase, method)
        target, decisions = fuse_labels(
            labels,
            sources,
            config["models"]["native"]["valid_ids"],
            method,
            temperatures,
        )
        costs = {
            key: sum(source["required_logical"][key] for source in sources.values())
            for key in ("native_requests", "siglip2_requests", "crop_inputs")
        }
        payload = relabel_prediction(
            native,
            method,
            "S",
            target,
            costs,
            {
                "source_identities": {
                    name: source["identity"] for name, source in sources.items()
                },
                "temperatures": temperatures
                if method == METHODS[2]
                else {name: 0.07 for name in sources},
            },
        )
        output = root / "compositions" / scene / method
        prediction_path = save_prediction(payload, output / "prediction")
        write_once(
            output / "decisions.json",
            {"scene_id": scene, "method_id": method, "objects": decisions},
        )
        query = verified_receipt(root / "query" / scene / "Q_GAIN/receipt.json")
        index = SourceIndex()
        inputs = [
            index.identity(root / "sources" / scene / (name + ".json"))
            for name in ("N0", "S_SIGLIP2_AREA")
        ]
        inputs.append(index.identity(query["evidence_path"]))
        outputs = [
            index.identity(path)
            for path in (
                prediction_path,
                prediction_path.parent / "prediction.npz",
                output / "decisions.json",
            )
        ]
        receipt = {
            "status": "COMPLETE",
            "input_identity": canonical_digest(
                {"prediction": payload.record_key, "inputs": inputs}
            ),
            "method_id": method,
            "scene_id": scene,
            "prediction_manifest": str(prediction_path),
            "required_logical": costs,
            "physical": {
                "Q_GAIN": query["physical"],
                "static_sources_reused": True,
                "fusion_model_forwards": 0,
                "shared_dependency_count_once_globally": True,
            },
            "inputs": inputs,
            "outputs": outputs,
        }
        write_once(output / "receipt.json", receipt)


def rows_for_role(config, role):
    paths = sorted((_root(config) / "rows" / role).glob("*/*.json"))
    return [read_json(path) for path in paths]


def freeze(config):
    root = _root(config)
    rows = rows_for_role(config, "compose_cal")
    gate = read_json(root / "calibration/m6_gate.json")
    nomination = nominate(rows, m6_enabled=gate["enabled"])
    final = refit_after_nomination(config, nomination)
    code = code_identity()
    if (root / "selection.json").is_file():
        existing = read_json(root / "selection.json")
        expected = {
            "binding_key": config["binding_key"],
            "nomination": nomination,
            "m6_gate": gate,
            "final_temperatures": final,
            "code": code,
        }
        if any(existing[key] != value for key, value in expected.items()):
            raise ValueError("frozen composition scientific inputs changed")
        return existing  # Preserve implementation A across later report-only commits.
    # All inference/evaluation code must be committed before confirmation exposure.
    for row in code["files"]:
        relative = str(Path(row["path"]).relative_to(ROOT))
        content = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        import hashlib

        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(
                "commit implementation A before freezing composition selection"
            )
    result = {
        "status": "FROZEN",
        "binding_key": config["binding_key"],
        "nomination": nomination,
        "m6_gate": gate,
        "final_temperatures": final,
        "code": code,
        "experiment_commit_A": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "models": config["models"],
        "checkpoint": config["checkpoint"],
        "runtime": config["runtime"],
        "source": config["source"],
        "gpu_lock": config["gpu_lock"],
        "source_manifest": SourceIndex().identity(root / "source_manifest.json"),
        "confirmation_rows": config["confirmation_rows"],
        "confirmation_exposure": config["confirmation_exposure"],
        "spec": config["spec"],
        "cal_rows": rows,
        "schedules": {
            scene: row["schedule"] for scene, row in config["scenes"].items()
        },
        "cost_definition": "common N0 map separated; required branches add primitive six-crop requests conservatively",
    }
    result["identity"] = canonical_digest(result)
    write_once(root / "selection.json", result)
    return result


def run_phase(config, config_path, phase):
    root = _root(config)
    if phase == "all":
        result = None
        statuses = {}
        for item in (
            "prepare-cal",
            "calibrate",
            "compose-cal",
            "freeze",
            "regression",
            "confirm",
            "report",
            "publish",
        ):
            result = run_phase(config, config_path, item)
            statuses[item] = result["status"]
        complete = all(
            value
            in {
                "COMPLETE",
                "FROZEN",
                "PUSH_VERIFIED",
                "NOT_REQUIRED_NO_EFFECTIVE_INTERVENTION",
            }
            for value in statuses.values()
        )
        return {
            "status": "COMPLETE" if complete else "PARTIAL",
            "phases": statuses,
            "publication": result,
        }
    if phase == "prepare-cal":
        scenes = config["spec"]["data"]["compose_cal"]
        for scene in scenes:
            invoke_leaf(config, config_path, phase, "static", scene)
            for method in ("N0", "S_SIGLIP2_AREA"):
                invoke_leaf(config, config_path, phase, "evaluate", scene, method)
        for scene in scenes:
            for method in ("Q_COMBINE", "Q_GAIN"):
                invoke_leaf(config, config_path, phase, "query", scene, method)
                invoke_leaf(config, config_path, phase, "evaluate", scene, method)
        return {"status": "COMPLETE", "rows": rows_for_role(config, "compose_cal")}
    if phase == "calibrate":
        return {"status": "COMPLETE", "calibration": fit_folds(config)}
    if phase == "compose-cal":
        scenes = config["spec"]["data"]["compose_cal"]
        for scene in scenes:
            compose_static(config, scene, phase)
            for method in METHODS:
                invoke_leaf(config, config_path, phase, "evaluate", scene, method)
            for method in COMPOSITIONS[3:6]:
                invoke_leaf(config, config_path, phase, "query", scene, method)
                invoke_leaf(config, config_path, phase, "evaluate", scene, method)
        gate = m6_gate(rows_for_role(config, "compose_cal"))
        write_once(root / "calibration/m6_gate.json", gate)
        if gate["enabled"]:
            for scene in scenes:
                invoke_leaf(config, config_path, phase, "query", scene, COMPOSITIONS[6])
                invoke_leaf(
                    config, config_path, phase, "evaluate", scene, COMPOSITIONS[6]
                )
        return {
            "status": "COMPLETE",
            "rows": rows_for_role(config, "compose_cal"),
            "m6_gate": gate,
        }
    if phase == "freeze":
        return freeze(config)
    if phase == "regression":
        selection = read_json(root / "selection.json")
        methods = COMPOSITIONS if selection["m6_gate"]["enabled"] else COMPOSITIONS[:-1]
        for scene in config["spec"]["data"]["regression_only"]:
            require_access(config, scene, phase)
            invoke_leaf(config, config_path, phase, "static", scene)
            for method in ("Q_COMBINE", "Q_GAIN", *COMPOSITIONS[3:6]):
                invoke_leaf(config, config_path, phase, "query", scene, method)
            if selection["m6_gate"]["enabled"]:
                invoke_leaf(config, config_path, phase, "query", scene, COMPOSITIONS[6])
            compose_static(config, scene, phase)
            for method in (*CONTROLS, *methods):
                invoke_leaf(config, config_path, phase, "evaluate", scene, method)
        return {"status": "COMPLETE", "rows": rows_for_role(config, "regression_only")}
    if phase == "confirm":
        from .capture_bridge import run_confirmation

        return run_confirmation(config, config_path)
    if phase == "report":
        from .reporting import report

        return report(config)
    if phase == "publish":
        from .publication import publish

        return publish(config)
    raise ValueError("unknown composition phase")
