"""Restored native/Q evidence and pre-inference FC acquisition plans."""

from pathlib import Path
import pickle
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, PathResolver, read
from .recovery_registry import physical_request_identity


def single_native_classifier(feature, text, canonical, valid_ids):
    import torch

    feature = torch.as_tensor(feature, dtype=torch.float32)
    text = torch.as_tensor(text, dtype=torch.float32)
    canonical = torch.as_tensor(canonical, dtype=torch.float32)
    if (feature.ndim != 1 or text.shape != (len(valid_ids), len(feature))
            or canonical.ndim != 2 or canonical.shape[1] != len(feature)
            or not len(canonical) or not all(torch.isfinite(v).all() for v in (feature, text, canonical))):
        raise ValueError("native single-view feature/text spaces are invalid")
    cosine = torch.nn.CosineSimilarity(dim=-1)
    query = cosine(feature, text).unsqueeze(0)
    reference = cosine(feature, canonical).unsqueeze(1)
    relative = torch.exp(query) / (torch.exp(query) + torch.exp(reference))
    scores = torch.min(relative, dim=0).values
    return {"label": int(valid_ids[int(scores.argmax())]), "scores": scores.tolist(),
            "cosine_scores": query[0].tolist(), "precision": "float32",
            "score_kind": "ORIGINAL_CANONICAL_RELATIVE_NATIVE"}


def _unavailable(owner, reason):
    return {"owner_id": int(owner), "available": False, "scores": None, "label": None,
            "used_request_ids": [], "reason": reason}


def native_single_sources(registry, manifest, saved, receipt, frames, text, canonical, valid_ids):
    if receipt["status"] != "COMPLETE":
        raise ValueError("native recovery requires a completed original feature receipt")
    successful = {row["request_id"]: row for row in receipt["native_requests"]
                  if row["status"] == "COMPLETE"}
    selected = set(manifest["native_selected_request_ids"])
    result = {}
    for candidate in registry["candidates"]:
        owner = candidate["raw_owner"]
        row = saved.get(owner)
        result[str(owner)] = _unavailable(owner, "NO_SINGLE_RETAINED_NATIVE_ROW")
        if row is None or len(row["frame_id"]) != 1:
            continue
        frame_id = int(row["frame_id"][0])
        matches = []
        for rid in selected:
            request = manifest["requests"][rid]
            actual = successful.get(rid)
            if (request["frame_id"] != frame_id or actual is None
                    or manifest["lineage_proofs"][rid]["target_id"] != f"owner:{owner}"
                    or int(actual["owner"]) != owner or actual["frame_id"] != frame_id
                    or list(actual["lineage"]) != list(request["lineage"])
                    or actual["source_map_version"] != request["source_map_version"]
                    or tuple(request["bbox_xyxy"]) != tuple(row["box_2d"][0])
                    or actual["encoder_content"] is None):
                continue
            if not np.array_equal(np.asarray(row["pose"][0], np.float32),
                                  np.asarray(frames[frame_id]["pose_c2w"], np.float32)):
                continue
            matches.append(rid)
        if len(matches) != 1:
            result[str(owner)] = _unavailable(owner, "NO_UNIQUE_MATCHING_NATIVE_REQUEST_PROOF")
            continue
        feature = np.asarray(row["feat"], np.float32)
        if feature.shape != (1, np.shape(text)[1]) or not np.isfinite(feature).all():
            raise ValueError("stored successful native single feature is malformed")
        rid = matches[0]
        result[str(owner)] = {"owner_id": owner, "available": True,
            **single_native_classifier(feature[0], text, canonical, valid_ids),
            "used_request_ids": [rid], "native_retained_frame_id": frame_id,
            "native_encoder_content": successful[rid]["encoder_content"],
            "reason": "VERIFIED_SINGLE_NATIVE_OBSERVATION"}
    return result


def paid_query_sources(registry, manifest, arrays, trace, valid_ids):
    if (not trace["final_surface_read_after_last_barrier"]
            or not np.array_equal(arrays["valid_ids"], valid_ids)):
        raise ValueError("Q recovery requires final reconciliation and the original class order")
    positions = {int(owner): i for i, owner in enumerate(arrays["owner_ids"])}
    if len(positions) != len(arrays["owner_ids"]):
        raise ValueError("stored Q owner rows are ambiguous")
    result = {}
    for candidate in registry["candidates"]:
        owner = candidate["raw_owner"]
        result[str(owner)] = _unavailable(owner, "NO_AVAILABLE_PAID_QUERY_SCORE")
        position = positions.get(owner)
        if position is None or not bool(arrays["available"][position]):
            continue
        requests = trace["retained_feature_requests"].get(str(owner), [])
        verified = bool(requests)
        for rid in requests:
            request = manifest["requests"].get(rid)
            alias = trace["aliases"].get(rid)
            proof = manifest["lineage_proofs"].get(rid)
            if (request is None or alias is None or proof is None
                    or proof["target_id"] != f"owner:{owner}"
                    or alias["frame_id"] != request["frame_id"]
                    or list(alias["lineage"]) != list(request["lineage"])
                    or alias["encoder_content"] is None):
                verified = False
                break
        if not verified:
            result[str(owner)] = _unavailable(owner, "UNRESOLVED_PAID_QUERY_LINEAGE")
            continue
        scores = np.asarray(arrays["scores"][position])
        if scores.shape != (len(valid_ids),) or not np.isfinite(scores).all():
            raise ValueError("available paid query scores are malformed")
        result[str(owner)] = {"owner_id": owner, "available": True, "scores": scores.tolist(),
            "label": int(valid_ids[int(scores.argmax())]), "used_request_ids": list(requests),
            "reason": "VERIFIED_FINAL_RECONCILED_PAID_QUERY", "score_kind": "ORIGINAL_Q_COSINE"}
    return result


def fc_image_identity(request, model_identity):
    return canonical_digest({"model": model_identity, "rgb": request["image_sha256"],
                             "preprocess": "original_800_1333_bilinear_32pad_FP32"})


def allocate_fc_requests(u2_requests, u3_requests, requests, model_identity, cached_images, *, max_new_frames=32):
    if not 0 <= max_new_frames <= 32:
        raise ValueError("recovery frame allowance exceeds the fixed protocol")
    cached_images = set(cached_images)
    admitted, all_images, rows, physical = {}, {}, {}, {}
    for rid in [*u2_requests, *u3_requests]:
        if rid in rows:
            continue
        request = requests[rid]
        key = fc_image_identity(request, model_identity)
        physical_key = physical_request_identity(request)
        all_images[key] = 1
        cached = key in cached_images
        if not cached and key not in admitted and len(admitted) < max_new_frames:
            admitted[key] = 1
        authorized = cached or key in admitted
        original = physical.setdefault(physical_key, rid)
        rows[rid] = {"request_id": rid, "physical_request_identity": physical_key,
            "physical_request_alias": None if original == rid else original,
            "image_content_identity": key, "cached_in_initial_inventory": cached,
            "authorized": authorized,
            "reason": ("INITIAL_DENSE_CACHE" if cached else "AUTHORIZED_NEW_FRAME"
                       if authorized else "NEW_FC_FRAME_BUDGET_EXCLUDED")}
    result = {"requests": rows, "authorized_new_image_contents": admitted,
        "required_image_contents": all_images, "new_frame_cap": max_new_frames,
        "initial_cached_image_contents": sorted(cached_images),
        "view_substitution_allowed": False, "budget_assigned_before_inference": True,
        "authorization_order": "U2_THEN_U3_IN_LOCKED_CANDIDATE_VIEW_ORDER", "GT_input": False}
    result["identity"] = canonical_digest(result)
    return result


def prepare_cached_sources(binding, scene, registry_receipt, *, context=None):
    from .light import require_transfer_freeze
    require_transfer_freeze(binding, scene)
    data = context or binding["scenes"][scene]
    root = Path(registry_receipt["registry"]).parent
    resolver = PathResolver(binding["path_map"])
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    started = time.monotonic()
    for item in registry_receipt["outputs"]:
        index.identity(item["path"], item)
    registry, manifest = read(registry_receipt["registry"]), read(registry_receipt["requests"])
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path, manifest["capture"])
    capture = read(capture_path)
    nq_root = Path(data["native_query_root"])
    index.identity(nq_root / "native_query_receipt.json")
    nq = resolver.rewrite(read(nq_root / "native_query_receipt.json"))
    for name in ("native_features", "query_scores", "query_decisions"):
        index.identity(nq[name]["path"], nq[name])
    with Path(nq["native_features"]["path"]).open("rb") as stream:
        saved = pickle.load(stream)
    with np.load(nq["query_scores"]["path"], allow_pickle=False) as arrays:
        query = {key: arrays[key] for key in arrays.files}
    trace = read(nq["query_decisions"]["path"])
    native_model = data["inherited"]["models"]["native"]
    index.identity(native_model["text"]["path"], native_model["text"])
    with np.load(native_model["text"]["path"], allow_pickle=False) as arrays:
        text, canonical, ids = arrays["text_embeddings"], arrays["canonical_embeddings"], arrays["valid_ids"]
    if not np.array_equal(ids, native_model["valid_ids"]):
        raise ValueError("restored native text class order differs")
    frames = {frame["frame_id"]: frame for frame in capture["frames"]}
    sources = {"U1": native_single_sources(registry, manifest, saved, nq, frames, text, canonical, ids),
               "UQ": paid_query_sources(registry, manifest, query, trace, ids)}
    u2 = [sources["U1"][str(row["raw_owner"])]["used_request_ids"][0]
          for row in registry["candidates"] if sources["U1"][str(row["raw_owner"])]["available"]]
    u3 = [rid for row in registry["candidates"] for rid in manifest["views"][f"owner:{row['raw_owner']}"]]
    index.identity(binding["parent_cache_inventory"])
    inventory = read(binding["parent_cache_inventory"])
    dense_inventory = dict(inventory["dense"])
    if context is not None:
        own_fc_path = Path(data["fc_root"]) / "receipt.json"
        index.identity(own_fc_path)
        own_fc = resolver.rewrite(read(own_fc_path))
        if (own_fc["status"] != "COMPLETE"
                or own_fc["physical_model_identity"] != data["FC_physical_model_identity"]):
            raise ValueError("new-map recovery requires its own locked standard FC readout")
        dense_inventory.update(own_fc["required_dense_receipts"])
    cached = set()
    for rid in {*u2, *u3}:
        key = fc_image_identity(manifest["requests"][rid], data["FC_physical_model_identity"])
        path = dense_inventory.get(key)
        if path is not None and Path(path).is_file():
            index.identity(path)
            dense = resolver.rewrite(read(path))
            if Path(dense["arrays"]["path"]).is_file():
                cached.add(key)
    plan = allocate_fc_requests(u2, u3, manifest["requests"], data["FC_physical_model_identity"], cached)
    source = {"scene": scene, "map_id": registry_receipt["map_id"], "sources": sources,
        "valid_ids": ids.tolist(), "U2_request_ids": u2, "U3_request_ids": u3,
        "registry_identity": registry["identity"], "request_manifest_identity": manifest["identity"],
        "query_final_reconciled": True, "new_query_acquisitions": 0, "new_native_image_encodings": 0,
        "parent_native_query_receipt": index.identity(nq_root / "native_query_receipt.json"), "GT_input": False}
    source["identity"] = canonical_digest(source)
    source_path, plan_path = root / "cached_sources.json", root / "fc_request_plan.json"
    for path, value in ((source_path, source), (plan_path, plan)):
        if path.is_file() and read(path)["identity"] != value["identity"]:
            raise ValueError("completed recovery source/allowance changed")
        atomic_write_json(path, value)
    receipt = {"status": "CACHED_SOURCES_AND_FC_PLAN_LOCKED", "scene": scene,
        "sources": str(source_path), "FC_plan": str(plan_path),
        "available_U1": sum(row["available"] for row in sources["U1"].values()),
        "available_UQ": sum(row["available"] for row in sources["UQ"].values()),
        "U2_planned_requests": len(u2), "U3_planned_requests": len(u3),
        "authorized_new_FC_frames": len(plan["authorized_new_image_contents"]),
        "initial_cached_FC_frames": len(cached), "inputs": index.entries(),
        "outputs": [index.identity(source_path), index.identity(plan_path)],
        "elapsed_seconds": time.monotonic() - started, "new_neural_inference": 0, "GT_input": False}
    atomic_write_json(root / "cached_source_receipt.json", receipt)
    return receipt
