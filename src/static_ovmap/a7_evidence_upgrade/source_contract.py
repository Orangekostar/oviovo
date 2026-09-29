"""Supplement immutable measured sources with explicit aggregate provenance."""

from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _write_npz

from .calibration import source_path
from .recognition_worker import aggregate_views


def export_contract(binding, scene, variant):
    root, index = Path(binding["output_root"]), InputIndex()
    index.identity(__file__)
    path = source_path(binding, scene, variant)
    source_identity = index.identity(path)
    source = read_json(path)
    if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
        raise ValueError("measured source changed before contract export")
    family = next(v["family"].lower() for v in read_json(binding["spec"])["source_variants"] if v["id"] == variant)
    manifest_identity = binding["scenes"][scene]["static_manifest"]
    index.identity(manifest_identity["path"], manifest_identity)
    manifest = read_json(manifest_identity["path"])
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    text_identity = source["text_identity"]
    index.identity(text_identity["path"], text_identity)
    with np.load(text_identity["path"], allow_pickle=False) as data:
        text = np.asarray(data["text_embeddings"], np.float64)
    if family in {"e01", "e02"}:
        text /= np.linalg.norm(text, axis=1, keepdims=True)
    arrays, owners, errors = {}, {}, []
    for owner, obj in source["objects"].items():
        used = obj["retained_request_ids"] if family == "e01" else obj["used_request_ids"]
        extra = {"available": obj["available"], "source_row": owner, "used_request_ids": used,
                 "original_visibility_weights": [], "aggregate_feature_key": None, "view_records": []}
        if not obj["available"]:
            if obj["scores"] is not None:
                raise ValueError("unavailable measured source contains a score vector")
            if family == "e01":
                reason = obj.get("fallback_reason", "ORIGINAL_Q_UNAVAILABLE")
            elif f"owner:{owner}" in manifest["excluded_targets"]:
                reason = "OUTSIDE_STATIC_CAP"
            elif not obj.get("attempted_request_ids"):
                reason = "NO_ORIGINAL_STATIC_VIEW"
            else:
                reason = "NO_SUCCESSFUL_RECOGNITION_VIEW"
            extra["explicit_fallback_reason"] = reason
            owners[owner] = extra
            continue
        if family == "e01":
            index.identity(source["arrays"]["path"], source["arrays"])
            with np.load(source["arrays"]["path"], allow_pickle=False) as data:
                aggregate = data[obj["aggregate_feature_key"]]
                weights = data["areas_" + owner].tolist()
            extra["view_feature_artifact"] = source["arrays"]
            extra["readout_diagnostics"] = obj["readout_diagnostics"]
        else:
            features, weights = [], []
            request_root = root / ("c0/requests" if family == "c0" else family) / scene
            if family != "c0":
                request_root /= variant + "/requests"
            for request_id in used:
                record_path = request_root / (request_id + ".json")
                record_identity = index.identity(record_path)
                record = read_json(record_path)
                if record["status"] != "COMPLETE":
                    raise ValueError("available owner uses unsuccessful view")
                index.expected_output(record["arrays_path"], record)
                with np.load(record["arrays_path"], allow_pickle=False) as data:
                    features.append(data["feature"])
                weights.append(requests[request_id]["visible_target_pixels"])
                extra["view_records"].append(record_identity)
            aggregate = aggregate_views(features, weights)
        scores = text @ aggregate
        error = float(np.max(np.abs(scores - np.asarray(obj["scores"]))))
        if error > 1e-12 or source["valid_ids"][int(scores.argmax())] != obj["label"]:
            raise ValueError("exported aggregate does not reproduce measured source cosine/label")
        key = "owner_" + owner
        arrays[key] = aggregate
        errors.append(error)
        extra.update(original_visibility_weights=weights, aggregate_feature_key=key,
                     explicit_fallback_reason=None, maximum_score_reconstruction_error=error)
        owners[owner] = extra
    output = root / "source_contracts" / scene / variant
    arrays_path = output / "aggregates.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(not np.array_equal(previous[k], v) for k, v in arrays.items()):
                raise ValueError("source aggregate supplement changed")
    else:
        _write_npz(arrays_path, arrays)
    result = {"status": "SOURCE_CONTRACT_VERIFIED", "scene": scene, "variant": variant,
              "measured_source": source_identity, "measured_source_identity": source["identity"],
              "actual_model_identity": source["model_identity"], "text_identity": text_identity,
              "mask_mode": variant, "score_representation": "pre_temperature_region_text_cosine",
              "owners": owners, "aggregate_features": index.identity(arrays_path),
              "maximum_score_reconstruction_error": max(errors, default=0),
              "source_and_metric_bytes_rewritten": False, "inputs": index.entries()}
    result["identity"] = canonical_digest(result)
    write_once(output / "contract.json", result)
    return result
