"""Recover real full-score evidence without opening annotations or changing maps."""

import pickle
from pathlib import Path

import numpy as np

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.query_state import export_query_labels
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    native_readout,
)
from src.static_ovmap.module_validation.semantic_models import load_semantic_records
from src.static_ovmap.module_validation.semantic_study import direct_readouts

from .io import SourceIndex, read_json, write_once


def same_source_surface(native, source):
    if not native.locked or not source.locked:
        raise ValueError("source prediction is not locked")
    if native.scene_id != source.scene_id or native.geometry != source.geometry:
        raise ValueError("source scene or geometry differs")
    if not np.array_equal(native.owner_ids, source.owner_ids):
        raise ValueError("source owner rows differ")
    if native.instance_ranks != source.instance_ranks:
        raise ValueError("source ranks differ")


def owner_labels(prediction):
    owners, indices, inverse = np.unique(
        prediction.owner_ids, return_index=True, return_inverse=True
    )
    labels = prediction.semantic_labels[indices]
    if not np.array_equal(labels[inverse], prediction.semantic_labels):
        raise ValueError("source does not assign a single semantic label per owner")
    return {
        int(owner): int(label)
        for owner, label in zip(owners, labels, strict=True)
        if owner > 0
    }


def native_scores(feature, text, canonical):
    """Match the native Torch FP32 canonical-relative classifier, including ties."""
    import torch

    feature, text, canonical = (
        torch.as_tensor(value, dtype=torch.float32)
        for value in (feature, text, canonical)
    )
    cosine = torch.nn.CosineSimilarity(dim=-1)
    query = cosine(feature, text).unsqueeze(0)
    references = cosine(feature, canonical).unsqueeze(1)
    relative = torch.exp(query) / (torch.exp(query) + torch.exp(references))
    result = torch.min(relative, dim=0).values.numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite native canonical-relative scores")
    return result


def native_request_ids(saved, frames):
    """Resolve saved observations by unique paid frame/bbox/area/pose, not owner ID."""
    resolved = []
    for position, frame_id in enumerate(saved["frame_id"]):
        frame = frames[int(frame_id)]
        if not np.array_equal(
            np.asarray(saved["pose"][position], np.float32),
            np.asarray(frame["pose_c2w"], np.float32),
        ):
            raise ValueError("saved native observation pose differs from capture")
        paid = set(frame["native_selected_request_ids"])
        matches = [
            row["request_id"]
            for row in frame["requests"]
            if row["request_id"] in paid
            and tuple(row["bbox_xyxy"]) == tuple(saved["box_2d"][position])
            and row["visible_target_pixels"] == saved["vis_area"][position]
        ]
        if len(matches) != 1:
            raise ValueError(
                "native observation lacks a unique paid frame/bbox/area match"
            )
        resolved.append(matches[0])
    return resolved


def static_objects(manifest, records, text, valid_ids, native_labels):
    readouts = direct_readouts(
        manifest, records, text, tuple(valid_ids), native_labels, model="siglip2"
    )["S_SIGLIP2_AREA"]
    objects = {}
    for owner, row in readouts.items():
        attempted = manifest["views"].get(f"owner:{owner}", [])
        used = [key for key in attempted if records[key]["status"] == "COMPLETE"]
        available = not row["technical_fallback"] and row["successful_views"] > 0
        objects[owner] = {
            "owner_id": owner,
            "label": int(row["label_id"]),
            "available": available,
            "fallback_reason": None if available else "NO_SUCCESSFUL_STATIC_VIEW",
            "attempted_request_ids": attempted,
            "used_request_ids": used,
            "retained_request_ids": used,
            "scores": list(row["scores"]) if available else None,
        }
    return objects


def query_objects(state, owners, text, valid_ids):
    # Caller must reconcile final numerical membership before calling this.
    labels = export_query_labels(state, owners, text, valid_class_ids=valid_ids)
    objects = {}
    for owner, label in labels.items():
        obj = state.object_state(owner)
        retained = [row.request_id for row in obj.features]
        available = bool(retained) and label > 0
        objects[owner] = {
            "owner_id": owner,
            "label": label,
            "available": available,
            "fallback_reason": None if available else "NO_RETAINED_QUERY_EVIDENCE",
            "used_request_ids": retained,
            "retained_request_ids": retained,
            "scores": obj.cached_scores.tolist() if available else None,
        }
    return objects


def source_bundle(scene, method, native, prediction, objects, model, receipts, costs):
    same_source_surface(native, prediction)
    labels = owner_labels(prediction)
    ids = tuple(model["valid_ids"])
    if set(objects) != set(labels):
        raise ValueError("source evidence must preserve every native owner")
    for owner, row in objects.items():
        if row["label"] != labels[owner]:
            raise ValueError(f"reconstructed {method} top1 differs for owner {owner}")
        if row["available"]:
            scores = np.asarray(row["scores"])
            if (
                scores.shape != (len(ids),)
                or not np.isfinite(scores).all()
                or ids[int(np.argmax(scores))] != labels[owner]
            ):
                raise ValueError("source full scores do not reproduce frozen top1")
        elif row["scores"] is not None:
            raise ValueError("unavailable source has fabricated scores")
    result = {
        "schema_version": 1,
        "scene_id": scene,
        "source": method,
        "native_record_key": native.record_key,
        "prediction_key": prediction.prediction_key,
        "valid_ids": list(ids),
        "model_identity": model["identity"],
        "text_identity": model["text"],
        "source_receipts": receipts,
        "readout_identity": {
            "N0": "native_FP32_canonical_relative_saved_top10_last8_min2",
            "S_SIGLIP2_AREA": "static_top3_raw_six_crop_mean_area_then_L2",
        }.get(method, "query_post_final_membership_top10_raw_mean_area_then_L2"),
        "objects": objects,
        "required_logical": costs,
    }
    result["identity"] = canonical_digest(result)
    return result


def prepare_static_sources(config, scene, output):
    """Read only bound source assets. Phase authorization belongs to the runner."""
    index = SourceIndex()
    bound = {
        row["path"]: row
        for row in read_json(Path(config["attempt_root"]) / "source_manifest.json")[
            "entries"
        ]
    }

    def verify(path):
        path = str(Path(path).resolve())
        return index.identity(path, bound[path])

    data = config["scenes"][scene]
    base = Path(data["source_directory"])
    verify(data["native_prediction"])
    native = load_prediction(data["native_prediction"])
    labels = owner_labels(native)
    capture = read_json(data["capture"])
    verify(data["capture"])
    feature_identity = verify(data["native_features"])
    with open(data["native_features"], "rb") as handle:
        saved = pickle.load(handle)  # Only the receipt-bound local mapper pickle.
    model = config["models"]["native"]
    verify(model["text"]["path"])
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, canonical, ids = (
            arrays["text_embeddings"],
            arrays["canonical_embeddings"],
            tuple(map(int, arrays["valid_ids"])),
        )
    if ids != tuple(model["valid_ids"]):
        raise ValueError("native text vocabulary changed")
    readouts = native_readout(saved, text, canonical, ids)
    verify(base / "baseline/native_readout.npz")
    with np.load(base / "baseline/native_readout.npz", allow_pickle=False) as arrays:
        aggregates = dict(
            zip(map(int, arrays["owner_ids"]), arrays["features"], strict=True)
        )
    frames = {row["frame_id"]: row for row in capture["frames"]}
    objects = {}
    for owner in labels:
        row = readouts[owner]
        retained = native_request_ids(saved[owner], frames)
        available = row["status"] == "AVAILABLE"
        if available and not np.array_equal(row["feature"], aggregates[owner]):
            raise ValueError("saved native aggregate changed")
        objects[owner] = {
            "owner_id": owner,
            "label": row["class_id"],
            "available": available,
            "fallback_reason": None if available else row["status"],
            "used_request_ids": retained[-8:] if available else [],
            "retained_request_ids": retained,
            "used_frames": row["used_frames"],
            "scores": native_scores(row["feature"], text, canonical).tolist()
            if available
            else None,
            "request_provenance": "UNIQUE_PAID_FRAME_BBOX_AREA_POSE_MATCH",
        }
    attempts = sum(len(row["native_selected_request_ids"]) for row in capture["frames"])
    if native.logical_cost["attempts"] != attempts:
        raise ValueError("N0 attempt ledger differs from capture")
    n0 = source_bundle(
        scene,
        "N0",
        native,
        native,
        objects,
        model,
        [feature_identity, verify(base / "baseline/native_readout.json")],
        {
            "native_requests": attempts,
            "siglip2_requests": 0,
            "crop_inputs": attempts * 6,
        },
    )
    output = Path(output)
    write_once(output / "N0.json", n0)

    model = config["models"]["siglip2"]
    verify(model["text"]["path"])
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, s2_ids = arrays["text_embeddings"], tuple(map(int, arrays["valid_ids"]))
    if s2_ids != ids or s2_ids != tuple(model["valid_ids"]):
        raise ValueError("source vocabularies differ")
    evidence_root = base / "semantic_models/siglip2"
    receipt_path = evidence_root / "receipt.json"
    receipt = read_json(receipt_path)
    for entry in receipt["inputs"]:
        index.identity(entry["path"], entry)
    records = load_semantic_records(evidence_root)
    manifest_path = base / "semantic_requests.json"
    manifest = read_json(manifest_path)
    prediction_path = data["controls"]["S_SIGLIP2_AREA"]["prediction"]
    verify(prediction_path)
    prediction = load_prediction(prediction_path)
    if prediction.metadata["request_manifest"] != manifest["identity"]:
        raise ValueError("static S2 source manifest differs from frozen prediction")
    if set(records) != {key for values in manifest["views"].values() for key in values}:
        raise ValueError("static request coverage differs")
    objects = static_objects(manifest, records, text, ids, labels)
    s2 = source_bundle(
        scene,
        "S_SIGLIP2_AREA",
        native,
        prediction,
        objects,
        model,
        [index.identity(receipt_path), index.identity(manifest_path)],
        {
            "native_requests": 0,
            "siglip2_requests": len(records),
            "crop_inputs": len(records) * 6,
        },
    )
    write_once(output / "S_SIGLIP2_AREA.json", s2)
    write_once(
        output / "static_source_receipt.json",
        {
            "status": "COMPLETE",
            "source_identities": [n0["identity"], s2["identity"]],
            "physical": {
                "model_loads": 0,
                "model_forwards": 0,
                "crop_inputs": 0,
                "reused_static_siglip2_requests": len(records),
            },
            "inputs": index.manifest()["entries"],
            "outputs": [
                index.identity(output / name)
                for name in ("N0.json", "S_SIGLIP2_AREA.json")
            ],
        },
    )
    return {"N0": n0, "S_SIGLIP2_AREA": s2}
