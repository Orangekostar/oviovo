"""Evaluation-side whole-content cache and immutable per-method result rows."""

import gzip
import json
from pathlib import Path

import numpy as np

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_ground_truth import load_ground_truth
from src.static_ovmap.module_validation.scannet_study import load_prediction
from src.static_ovmap.module_validation.study_scene import evaluate_predictions
from src.static_ovmap.module_validation.study_targets import semantic_events

from .execution import require_access, verified_receipt
from .io import ROOT, SourceIndex, read_json, write_once
from .object_evidence import owner_labels, same_source_surface


def evaluation_identity(payload, protocol_files):
    if not payload.locked:
        raise ValueError("prediction must be locked before evaluation inputs are read")
    if set(protocol_files) != {"evaluator", "projection", "ground_truth", "vocabulary"}:
        raise ValueError("incomplete persistent evaluator protocol identity")
    index = SourceIndex()
    context = {
        key: [index.identity(path) for path in paths]
        for key, paths in protocol_files.items()
    }
    return canonical_digest(
        {
            "scene_id": payload.scene_id,
            "prediction_key": payload.prediction_key,
            "protocol_content": context,
        }
    )


def protocol_files(config, scene):
    data = config["scenes"][scene]
    upstream = Path(config["runtime"]["upstream"]) / "scripts"
    annotation = read_json(data["annotations"])
    return {
        "evaluator": [
            upstream / "eval_utils.py",
            upstream / "eval_sem_seg.py",
            ROOT / "scripts/evaluation/diagnose_static_t1_attribution.py",
            ROOT / "scripts/evaluation/evaluate_static_ovmap_readout.py",
            ROOT / "src/static_ovmap/released_loader.py",
            ROOT / "src/static_ovmap/released_trace.py",
            ROOT / "src/static_ovmap/module_validation/evaluation.py",
            ROOT / "src/static_ovmap/module_validation/scannet_ground_truth.py",
            ROOT / "src/evaluation/static_projected_instances.py",
            Path(__file__),
        ],
        "projection": [
            Path(data["projection"]),
            Path(data["projection"]).with_name("projection.npz"),
        ],
        "ground_truth": [
            Path(data["annotations"]),
            *[Path(row["path"]) for row in annotation["outputs"]],
        ],
        "vocabulary": [
            upstream / "utils/semantic_const.py",
            Path(config["models"]["native"]["text"]["path"]),
        ],
    }


def load_targets(config, scene, native):
    if not native.locked:
        raise ValueError("targets require a locked scene prediction")
    data = config["scenes"][scene]
    index = SourceIndex()
    bound = {
        row["path"]: row
        for row in read_json(Path(config["attempt_root"]) / "source_manifest.json")[
            "entries"
        ]
    }
    for name in ("projection", "annotations"):
        index.identity(data[name], bound[data[name]])
    projection = read_json(data["projection"])
    if projection["identity"] != native.geometry.projection_identity:
        raise ValueError("native geometry and frozen projection identity differ")
    path = Path(data["projection"]).with_name("projection.npz")
    index.identity(path, projection)
    targets = load_ground_truth(Path(data["annotations"]))
    with np.load(path, allow_pickle=False) as arrays:
        targets.update(nearest=arrays["nearest"], matched=arrays["matched"])
    if tuple(map(int, targets["valid_ids"])) != tuple(
        config["models"]["native"]["valid_ids"]
    ):
        raise ValueError("annotation vocabulary differs")
    return targets


def evaluate_method(
    config, scene, method, phase, prediction_path, *, logical, physical, reuse_kind
):
    role = require_access(config, scene, phase, method)
    # Load and validate locked prediction before any GT arrays are opened.
    prediction = load_prediction(prediction_path)
    if prediction.method_id != method:
        raise ValueError("method record and prediction differ")
    native = load_prediction(config["scenes"][scene]["native_prediction"])
    same_source_surface(native, prediction)
    files = protocol_files(config, scene)
    identity = evaluation_identity(prediction, files)
    root = Path(config["attempt_root"])
    cache = root / "evaluation_cache" / identity
    receipt_path = cache / "receipt.json"
    reused = receipt_path.is_file()
    if reused:
        receipt = verified_receipt(receipt_path, identity=identity)
    else:
        targets = load_targets(config, scene, native)
        row = evaluate_predictions(
            [prediction], {scene: targets}, Path(config["runtime"]["upstream"]), cache
        )[0]
        matches = list(cache.rglob("released_matches.json.gz"))
        traces = list(cache.rglob("released_trace.json.gz"))
        if len(matches) != 1 or len(traces) != 1:
            raise ValueError(
                "released evaluator did not produce one actual scene trace"
            )
        with gzip.open(matches[0], "rt") as handle:
            relationships = json.load(handle)
        count = sum(
            len(values)
            for match in relationships.values()
            for values in match["pred"].values()
        )
        index = SourceIndex()
        inputs = [index.identity(path) for paths in files.values() for path in paths]
        outputs = [
            index.identity(path) for path in sorted(cache.rglob("*")) if path.is_file()
        ]
        receipt = {
            "status": "COMPLETE",
            "input_identity": identity,
            "metrics": row["metrics"],
            "prediction_key": prediction.prediction_key,
            "first_method_id": method,
            "evaluator_prediction_count": count,
            "trace_path": str(traces[0]),
            "matches_path": str(matches[0]),
            "inputs": inputs,
            "outputs": outputs,
        }
        write_once(receipt_path, receipt)
    labels, reference = owner_labels(prediction), owner_labels(native)
    record = {
        "status": "COMPLETE",
        "scene_id": scene,
        "role": role,
        "method_id": method,
        "metrics": receipt["metrics"],
        "prediction_key": prediction.prediction_key,
        "record_key": prediction.record_key,
        "prediction_manifest": str(prediction_path),
        "owners": len(labels),
        "changed_owners": sum(labels[key] != reference[key] for key in labels),
        "positive_owners": sum(value > 0 for value in labels.values()),
        "unknown_owners": sum(value == 0 for value in labels.values()),
        "evaluator_prediction_count": receipt["evaluator_prediction_count"],
        "logical": logical,
        "logical_requests": logical["native_requests"] + logical["siglip2_requests"],
        "physical": physical,
        "source_reuse": reuse_kind,
        "evaluation_identity": identity,
        "evaluation_receipt": str(receipt_path),
        "trace_path": receipt["trace_path"],
        "matches_path": receipt["matches_path"],
        "shared_evaluation_first_method": receipt["first_method_id"],
    }
    destination = root / "rows" / role / scene / (method + ".json")
    # Resume must not alter an existing row merely because it is now a cache hit.
    write_once(destination, record)
    return record


def object_correspondence(config, scene, native):
    targets = load_targets(config, scene, native)
    suggestions = {
        owner: {"label_id": label, "technical_fallback": False}
        for owner, label in owner_labels(native).items()
    }
    combined = np.load(targets["gt_instance_path"], allow_pickle=False)
    return semantic_events(
        native,
        suggestions,
        targets["nearest"],
        targets["matched"],
        combined,
        tuple(map(int, targets["valid_ids"])),
    )
