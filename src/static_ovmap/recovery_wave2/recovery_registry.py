"""Geometry-only omitted-owner registry and captured-request lineage proofs."""

from pathlib import Path
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import RegionRequest, _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.module_validation.semantic_study import reconcile_semantic_request

from .binding import ConsumptionIndex, read


def support_digest(xyz):
    coordinates = np.array(xyz, dtype=np.asarray(xyz).dtype.newbyteorder("<"), copy=True)
    coordinates[coordinates == 0] = 0
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    return _array_digest(np.ascontiguousarray(coordinates[order]))


def build_registry(xyz, raw, painted, *, minimum_rows=100, candidate_cap=128):
    xyz, raw, painted = np.asarray(xyz), np.asarray(raw), np.asarray(painted)
    if (xyz.shape != (len(raw), 3) or raw.ndim != 1 or painted.shape != raw.shape
            or not np.issubdtype(raw.dtype, np.integer)
            or not np.issubdtype(painted.dtype, np.integer)
            or not np.isfinite(xyz).all() or np.any(raw < 0) or np.any(painted < 0)):
        raise ValueError("raw and painted owners must align with finite source coordinates")
    if minimum_rows < 1 or not 1 <= candidate_cap <= 128:
        raise ValueError("invalid fixed recovery registry bounds")
    active = set(map(int, np.unique(painted[painted > 0])))
    raw_ids, counts = np.unique(raw[raw > 0], return_counts=True)
    candidates, excluded = [], []
    for owner, total in zip(raw_ids, counts, strict=True):
        owner, total = int(owner), int(total)
        if owner in active:
            excluded.append({"raw_owner": owner, "raw_source_rows": total,
                             "reason": "ALREADY_ACTIVE_NATIVE_PAINTED"})
            continue
        indices = np.flatnonzero((raw == owner) & (painted == 0))
        row = {"raw_owner": owner, "raw_source_rows": total,
               "residual_source_rows": len(indices), "clipping_ratio": len(indices) / total}
        if len(indices) < minimum_rows:
            excluded.append({**row, "reason": "RESIDUAL_SOURCE_ROWS_BELOW_MINIMUM"})
            continue
        candidates.append({**row, "support_sha256": support_digest(xyz[indices])})
    candidates.sort(key=lambda row: (-row["residual_source_rows"], row["support_sha256"]))
    excluded.extend({**row, "reason": "CANDIDATE_CAP_EXCLUDED"} for row in candidates[candidate_cap:])
    result = {"candidate_selection_uses_GT": False, "source_row_count": len(raw),
        "raw_positive_owner_count": len(raw_ids), "native_painted_owner_count": len(active),
        "minimum_residual_source_rows": minimum_rows, "candidate_cap": candidate_cap,
        "raw_owner_sha256": _array_digest(raw), "painted_owner_sha256": _array_digest(painted),
        "xyz_sha256": _array_digest(xyz), "candidates": candidates[:candidate_cap], "excluded": excluded}
    result["identity"] = canonical_digest(result)
    return result


def prepare_registry(binding, scene, *, context=None, recovery_settings=None):
    """Lock a raw registry and original request proofs without opening target data."""
    data = context or binding["scenes"][scene]
    map_id = data.get("map_id", "BB00_NATIVE")
    root = Path(binding["output_root"]) / "recovery" / scene / map_id
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    baseline_path = Path(data["predictions"]["D2"])
    index.identity(baseline_path)
    baseline_manifest = read(baseline_path)
    index.identity(baseline_path.parent / baseline_manifest["arrays"]["path"], baseline_manifest["arrays"])
    surface_path = capture_path.parent / capture["surface"]["path"]
    index.identity(surface_path, capture["surface"])
    settings = recovery_settings or read(binding["spec"])["recovery"]
    identity = canonical_digest({"capture": capture["identity"],
        "baseline": baseline_manifest["prediction_key"], "surface": capture["surface"]["sha256"],
        "minimum_source_rows": settings["minimum_residual_source_rows"],
        "cap": settings["candidate_cap_per_scene"], "max_views": settings["max_views"],
        "operator": index.identity(__file__)["sha256"]})
    receipt_path = root / "registry_receipt.json"
    if receipt_path.is_file():
        receipt = read(receipt_path)
        if receipt["input_identity"] != identity:
            raise ValueError("completed recovery registry inputs changed")
        for item in receipt["outputs"]:
            index.identity(item["path"], item)
        return receipt
    started = time.monotonic()
    baseline = load_prediction(baseline_path)
    with np.load(surface_path, allow_pickle=False) as surface:
        xyz, raw = surface["surface_xyz"], surface["original_owner"]
        if (_array_digest(xyz) != baseline.geometry.xyz_sha256
                or _array_digest(surface["surface_faces"]) != baseline.geometry.faces_sha256
                or capture["tsdf"]["sha256"] != baseline.geometry.tsdf_sha256
                or baseline.scene_id != scene):
            raise ValueError("raw registry geometry differs from its parent anchor")
        registry = build_registry(xyz, raw, baseline.owner_ids,
            minimum_rows=settings["minimum_residual_source_rows"],
            candidate_cap=settings["candidate_cap_per_scene"])
        requests = reconcile_requests(capture, surface["segment_labels"], raw, registry,
                                      max_views=settings["max_views"])
    registry_path, requests_path = root / "registry.json", root / "captured_requests.json"
    atomic_write_json(registry_path, registry)
    requests.update(capture=index.identity(capture_path), native_record_key=baseline.record_key)
    requests["identity"] = canonical_digest({k: v for k, v in requests.items() if k != "identity"})
    atomic_write_json(requests_path, requests)
    receipt = {"status": "COMPLETE", "scene": scene, "map_id": map_id,
        "input_identity": identity, "registry": str(registry_path), "requests": str(requests_path),
        "candidate_count": len(registry["candidates"]),
        "candidates_with_legal_views": sum(bool(v) for v in requests["views"].values()),
        "native_selected_candidate_requests": len(requests["native_selected_request_ids"]),
        "outputs": [index.identity(registry_path), index.identity(requests_path)],
        "inputs": index.entries(), "elapsed_seconds": time.monotonic() - started,
        "GT_input": False, "new_neural_inference": 0, "new_mapping": 0}
    atomic_write_json(receipt_path, receipt)
    return receipt


def physical_request_identity(request):
    if isinstance(request, RegionRequest):
        request = request.to_dict()
    return canonical_digest({key: request[key] for key in (
        "image_sha256", "target_mask_sha256", "native_union_mask_sha256",
        "bbox_xyxy", "crop_convention")})


def reconcile_requests(capture, segment_labels, raw, registry, *, max_views=3):
    if not 1 <= max_views <= 3 or np.shape(segment_labels) != np.shape(raw):
        raise ValueError("invalid captured request support or view bound")
    if _array_digest(np.asarray(raw)) != registry["raw_owner_sha256"]:
        raise ValueError("request registry belongs to different raw owners")
    pairs = np.unique(np.column_stack((segment_labels, raw)), axis=0)
    segment_owners = {}
    for segment, owner in pairs:
        segment_owners.setdefault(int(segment), set()).add(int(owner))
    active = set(map(int, np.unique(np.asarray(raw)[np.asarray(raw) > 0])))
    candidates = {f"owner:{row['raw_owner']}": [] for row in registry["candidates"]}
    requests, proofs, unavailable, duplicates, native_selected = {}, {}, [], [], []
    for frame in capture["frames"]:
        selected = set(frame["native_selected_request_ids"])
        for value in frame["requests"]:
            request = RegionRequest.from_dict(value)
            if (request.scene_id != capture["scene_id"] or request.frame_id != frame["frame_id"]
                    or request.source_map_version != frame["map_state_id"]):
                raise ValueError("captured request/frame identity differs")
            rid = request.request_id
            if rid in requests:
                if requests[rid] != request.to_dict():
                    raise ValueError("duplicate request has conflicting content")
                duplicates.append(rid)
                continue
            requests[rid] = request.to_dict()
            proof = reconcile_semantic_request(request, segment_owners, capture["alias_table"], active)
            proofs[rid] = proof
            if proof["target_id"] in candidates:
                candidates[proof["target_id"]].append(request)
                if rid in selected:
                    native_selected.append(rid)
            else:
                unavailable.append({"request_id": rid, "target_id": proof["target_id"],
                    "reason": ("TARGET_NOT_RECOVERY_CANDIDATE" if proof["target_id"] is not None
                               else proof["reason"])})
    views = {target: [request.request_id for request in sorted(values, key=lambda request: (
        -request.visible_target_pixels, request.frame_id, physical_request_identity(request)))[:max_views]]
        for target, values in candidates.items()}
    result = {"scene_id": capture["scene_id"], "registry_identity": registry["identity"],
        "requests": requests, "views": views, "lineage_proofs": proofs,
        "native_selected_request_ids": native_selected, "unavailable": unavailable,
        "duplicate_request_ids": duplicates,
        "no_view_candidates": [{"target_id": target, "reason": "NO_LEGAL_CAPTURED_VIEW"}
                               for target, values in views.items() if not values], "GT_input": False}
    result["identity"] = canonical_digest(result)
    return result
