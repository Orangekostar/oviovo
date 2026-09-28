"""Recover frozen image aggregates without image inference or query replay."""

import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.native_capture import _write_npz
from src.static_ovmap.module_validation.semantic_models import load_semantic_records
from src.static_ovmap.module_validation.semantic_study import direct_readouts

from .binding import InputIndex
from .scores import read_scene


def recover_aggregates(binding, scene):
    started, index = time.monotonic(), InputIndex()
    evidence = read_scene(binding, scene, index)
    root = Path(binding["output_root"]) / "robustness/aggregates" / scene
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for entry in receipt["inputs"] + receipt["outputs"]:
            index.identity(entry["path"], entry)
        return receipt
    config = evidence.config
    data = config["scenes"][scene]
    arrays = {f"N0_{owner}": value for owner, value in evidence.native_aggregates.items()}
    provenance = {"N0": {str(owner): evidence.sources["N0"]["objects"][str(owner)]["used_request_ids"]
                         for owner in evidence.native_aggregates}, "Q_GAIN": {}, "S_SIGLIP2_AREA": {}}
    query_path = Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/receipt.json"
    index.identity(query_path)
    query = read_json(query_path)
    index.expected_output(query["decisions_path"], query)
    decisions = read_json(query["decisions_path"])
    requests = {r["request"]["request_id"]: r["request"] for r in decisions["paid_requests"]}
    cache = {}
    for entry in query["inputs"]:
        path = Path(entry["path"])
        if path.suffix != ".json" or not ("cache" in path.parts or "native_request_cache" in path.parts):
            continue
        index.identity(path, entry)
        row = read_json(path)
        if "request" in row and "arrays_path" in row:
            request_id = row["request"]["request_id"]
            if request_id in cache and cache[request_id]["arrays_path"] != row["arrays_path"]:
                raise ValueError("query request has conflicting cached feature payloads")
            cache[request_id] = row
    for owner, source in evidence.sources["Q_GAIN"]["objects"].items():
        if not source["available"]:
            continue
        retained = source["retained_request_ids"]
        if retained != decisions["retained_features"][owner]:
            raise ValueError("frozen final query retention differs")
        features, areas = [], []
        for request_id in retained:
            row = cache.get(request_id)
            if row is None:
                raise ValueError(f"query aggregate has no bound cached feature: {scene}/{request_id}")
            index.expected_output(row["arrays_path"], row)
            with np.load(row["arrays_path"], allow_pickle=False) as payload:
                feature = np.asarray(payload["feature"], np.float64)
                vectors = payload["crop_features"] if "crop_features" in payload else payload["vectors"]
                if not np.array_equal(vectors[:6].mean(0), feature):
                    raise ValueError("query feature differs from raw six-crop mean")
            features.append(feature)
            areas.append(requests[request_id]["visible_target_pixels"])
        aggregate = np.average(np.stack(features), axis=0, weights=np.asarray(areas, np.float64))
        arrays[f"Q_GAIN_{owner}"] = aggregate / np.linalg.norm(aggregate)
        provenance["Q_GAIN"][owner] = retained
    static_root = Path(data["source_directory"]) / "semantic_models/siglip2"
    records = load_semantic_records(static_root)
    static_receipt = read_json(static_root / "receipt.json")
    index.identity(static_root / "receipt.json")
    for entry in static_receipt["outputs"]:
        index.identity(entry["path"], entry)
    manifest_path = Path(data["source_directory"]) / "semantic_requests.json"
    index.identity(manifest_path)
    manifest = read_json(manifest_path)
    model = config["models"]["siglip2"]
    index.identity(model["text"]["path"], model["text"])
    with np.load(model["text"]["path"], allow_pickle=False) as payload:
        text = payload["text_embeddings"]
    incumbent = {int(k): row["label"] for k, row in evidence.sources["N0"]["objects"].items()}
    readouts = direct_readouts(manifest, records, text, tuple(model["valid_ids"]), incumbent, model="siglip2")["S_SIGLIP2_AREA"]
    for owner, row in readouts.items():
        source = evidence.sources["S_SIGLIP2_AREA"]["objects"][str(owner)]
        if source["available"]:
            arrays[f"S_SIGLIP2_AREA_{owner}"] = np.asarray(row["aggregate_feature"], np.float64)
            provenance["S_SIGLIP2_AREA"][str(owner)] = source["used_request_ids"]
    parity = {}
    for name, model_name in (("Q_GAIN", "native"), ("S_SIGLIP2_AREA", "siglip2")):
        model = config["models"][model_name]
        with np.load(model["text"]["path"], allow_pickle=False) as payload:
            text = np.asarray(payload["text_embeddings"], np.float64)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        errors = []
        for owner, source in evidence.sources[name]["objects"].items():
            if not source["available"]:
                continue
            aggregate = arrays[f"{name}_{owner}"]
            scores = text @ (aggregate / np.linalg.norm(aggregate))
            expected = np.asarray(source["scores"])
            if not np.allclose(scores, expected, atol=1e-12, rtol=0) or int(scores.argmax()) != int(expected.argmax()):
                raise ValueError(f"aggregate does not reproduce frozen scores: {scene}/{name}/{owner}")
            errors.append(float(np.max(np.abs(scores - expected))))
        parity[name] = {"objects": len(errors), "maximum_absolute_score_error": max(errors, default=0), "top1_exact": True}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "aggregates.npz"
    _write_npz(path, arrays)
    receipt = {"status": "COMPLETE", "scene": scene, "dataset": evidence.dataset, "binding": binding["identity"],
               "models": {name: config["models"]["siglip2" if name == "S_SIGLIP2_AREA" else "native"]["identity"] for name in provenance},
               "valid_ids": model["valid_ids"], "used_requests": provenance, "score_parity": parity,
               "N0_parity": "exact FP32 canonical-relative reconstruction in read_scene",
               "query_replayed": False, "physical_image_forwards": 0,
               "inputs": index.entries(), "outputs": [index.identity(path)], "elapsed_seconds": time.monotonic() - started}
    write_once(receipt_path, receipt)
    return receipt
