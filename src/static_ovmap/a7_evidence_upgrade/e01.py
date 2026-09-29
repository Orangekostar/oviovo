"""Recover actual final retained Q observations and export all three readouts."""

import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _write_npz

from .readouts import query_readout

MODES = ("RAW_EQ", "UNIT_EQ", "GMED")


def generate_sources(binding, scene):
    started, index = time.monotonic(), InputIndex()
    if scene not in binding["scenes"]:
        raise ValueError("scene outside frozen wave1 scope")
    parent = read_json(binding["reviewer_binding"])
    index.identity(binding["reviewer_binding"])
    if parent["identity"] != binding["reviewer_identity"]:
        raise ValueError("parent binding differs")
    output = Path(binding["output_root"]) / "e01" / scene
    for file in (Path(__file__), Path(__file__).with_name("readouts.py")):
        index.identity(file)
    evidence = read_scene(parent, scene, index)
    config = evidence.config
    query_path = Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/receipt.json"
    index.identity(query_path)
    query = read_json(query_path)
    index.expected_output(query["decisions_path"], query)
    decisions = read_json(query["decisions_path"])
    requests = {r["request"]["request_id"]: r["request"] for r in decisions["paid_requests"]}
    cache = {}
    for entry in query["inputs"]:
        path = Path(entry["path"])
        if path.suffix == ".json" and ("cache" in path.parts or "native_request_cache" in path.parts):
            index.identity(path, entry)
            record = read_json(path)
            if "request" in record and "arrays_path" in record:
                request_id = record["request"]["request_id"]
                if request_id in cache and cache[request_id]["arrays_path"] != record["arrays_path"]:
                    raise ValueError("conflicting retained request payloads")
                cache[request_id] = record
    model = config["models"]["native"]
    index.identity(model["text"]["path"], model["text"])
    with np.load(model["text"]["path"], allow_pickle=False) as payload:
        text = np.asarray(payload["text_embeddings"], np.float64)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    ids = model["valid_ids"]
    objects, arrays, reconstruction = {mode: {} for mode in MODES}, {}, []
    for owner, original in evidence.sources["Q_GAIN"]["objects"].items():
        retained = original["retained_request_ids"]
        if retained != decisions["retained_features"].get(owner, []):
            raise ValueError("final retained request membership/order changed")
        if not original["available"]:
            for mode in MODES:
                objects[mode][owner] = {**original, "scores": None, "available": False,
                                       "aggregate_feature_key": None, "readout_diagnostics": None}
            continue
        features, areas = [], []
        for request_id in retained:
            if request_id not in cache or request_id not in requests:
                raise ValueError(f"missing actual retained input: {scene}/{request_id}")
            record = cache[request_id]
            index.expected_output(record["arrays_path"], record)
            with np.load(record["arrays_path"], allow_pickle=False) as payload:
                feature = np.asarray(payload["feature"], np.float64)
                crops = payload["crop_features"] if "crop_features" in payload else payload["vectors"]
                if not np.array_equal(crops[:6].mean(0), feature):
                    raise ValueError("retained feature is not the actual raw six-crop mean")
            features.append(feature)
            areas.append(requests[request_id]["visible_target_pixels"])
        features, areas = np.stack(features), np.asarray(areas, np.float64)
        area = np.average(features, axis=0, weights=areas)
        area /= np.linalg.norm(area)
        scores = text @ area
        original_scores = np.asarray(original["scores"])
        if not np.allclose(scores, original_scores, atol=1e-12, rtol=0) or ids[int(scores.argmax())] != original["label"]:
            raise ValueError("original retained-area readout does not reproduce Q evidence")
        reconstruction.append(float(np.abs(scores - original_scores).max()))
        arrays[f"retained_{owner}"] = features
        arrays[f"areas_{owner}"] = areas
        for mode in MODES:
            aggregate, diagnostics = query_readout(features, areas, mode)
            scores = text @ aggregate
            key = f"{mode}_{owner}"
            arrays[key] = aggregate
            objects[mode][owner] = {**original, "scores": scores.tolist(), "label": ids[int(scores.argmax())],
                "aggregate_feature_key": key, "readout_diagnostics": diagnostics,
                "original_visibility_weights": areas.tolist(), "model_identity": model["identity"],
                "mask_mode": "UNCHANGED_PAID_Q_SIX_CROP", "score_representation": "unit_aggregate_text_cosine"}
    arrays_path = output / "features.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError("immutable E01 feature export changed")
    else:
        _write_npz(arrays_path, arrays)
    outputs = [index.identity(arrays_path)]
    for mode in MODES:
        variant = "AW_E01_" + mode
        source = {"scene": scene, "variant": variant, "slot": "Q_GAIN", "valid_ids": ids,
                  "native_record_key": evidence.native.record_key, "objects": objects[mode],
                  "model_identity": model["identity"], "text_identity": model["text"],
                  "arrays": outputs[0], "parent_source_identity": evidence.sources["Q_GAIN"]["identity"],
                  "query_replayed": False, "physical_image_forwards": 0}
        source["identity"] = canonical_digest(source)
        path = output / (variant + ".json")
        write_once(path, source)
        outputs.append(index.identity(path))
    receipt_path = output / "receipt.json"
    receipt = {"status": "SOURCES_COMPLETE", "scene": scene, "binding": binding["identity"],
               "owners": len(objects[MODES[0]]), "available": len(reconstruction),
               "maximum_original_score_error": max(reconstruction, default=0),
               "retention_order_exact": True, "physical_image_forwards": 0,
               "inputs": [row for row in index.entries() if row["path"] not in {r["path"] for r in outputs}],
               "outputs": outputs, "elapsed_seconds": time.monotonic() - started}
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["inputs"] != receipt["inputs"] or previous["outputs"] != receipt["outputs"]:
            raise ValueError("E01 resume dependency changed")
        return previous
    write_once(receipt_path, receipt)
    return receipt
