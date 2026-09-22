"""ScanNet S request manifests and direct readouts; no annotation inputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity, verify_capture
from .contracts import atomic_write_json, canonical_digest
from .native_capture import RegionRequest, _array_digest
from .region_evidence import stable_target_subset
from .semantic_selector import area_readout, vote_readout


def prepare_semantic_manifest(capture_path: Path, native, output: Path) -> dict:
    """Bind <=128 final targets and <=3 original requests with explicit proofs."""
    if not native.locked:
        raise ValueError("N0 must be locked before semantic request selection")
    capture_path = Path(capture_path)
    capture = verify_capture(capture_path, allow_skipped=True)
    if capture["scene_id"] != native.scene_id:
        raise ValueError("semantic requests and N0 belong to different scenes")
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        if (_array_digest(arrays["surface_xyz"]) != native.geometry.xyz_sha256
                or _array_digest(arrays["surface_faces"]) != native.geometry.faces_sha256
                or capture["tsdf"]["sha256"] != native.geometry.tsdf_sha256):
            raise ValueError("semantic native geometry identity differs")
        pairs = np.unique(np.column_stack((arrays["segment_labels"], arrays["original_owner"])), axis=0)
    segment_owners = {}
    for segment, owner in pairs:
        segment_owners.setdefault(int(segment), set()).add(int(owner))
    active = set(map(int, np.unique(native.owner_ids))) - {0}
    selected, excluded = stable_target_subset(native.scene_id, [f"owner:{owner}" for owner in sorted(active)])
    candidates = {target: [] for target in selected}
    requests, proofs, unavailable, duplicates = {}, {}, [], []
    for frame in capture["frames"]:
        for value in frame["requests"]:
            request = RegionRequest.from_dict(value)
            if (request.scene_id != native.scene_id or request.frame_id != frame["frame_id"]
                    or request.source_map_version != frame["map_state_id"]):
                raise ValueError("captured request and frame identity differ")
            if request.request_id in requests:
                duplicates.append(request.request_id)
                continue
            requests[request.request_id] = request.to_dict()
            proof = reconcile_semantic_request(request, segment_owners, capture["alias_table"], active)
            proofs[request.request_id] = proof
            target = proof["target_id"]
            if target in candidates:
                candidates[target].append(request)
            else:
                unavailable.append({"request_id": request.request_id, "target_id": target,
                    "reason": "TARGET_CAP_EXCLUDED" if target in excluded else proof["reason"]})
    views = {target: [request.request_id for request in sorted(values,
        key=lambda request: (-request.visible_target_pixels, request.frame_id, request.request_id))[:3]]
        for target, values in candidates.items()}
    chosen = {request for values in views.values() for request in values}
    result = {"schema_version": 1, "artifact_type": "OVIMAP_SEMANTIC_REQUEST_MANIFEST",
        "scene_id": native.scene_id, "capture": file_identity(capture_path),
        "native_record_key": native.record_key, "selected_targets": list(selected), "excluded_targets": list(excluded),
        "views": views, "requests": {key: requests[key] for key in sorted(chosen)},
        "lineage_proofs": {key: proofs[key] for key in sorted(chosen)},
        "unavailable": unavailable, "duplicate_request_ids": duplicates, "GT_input": False,
        "source": file_identity(Path(__file__))}
    result["identity"] = canonical_digest(result)
    output = Path(output)
    if output.exists():
        import json

        if json.loads(output.read_text())["identity"] != result["identity"]:
            raise ValueError("semantic request inputs changed; choose a new output root")
    else:
        atomic_write_json(output, result)
    return result


def reconcile_semantic_request(request: RegionRequest, final_segment_owners: dict[int, set[int]],
                               aliases: list[tuple[int, int]], active_owners: set[int]) -> dict:
    """Require every recorded segment to converge to the same active final owner.

    Preserve the original captured request identity even when a proven merge
    changes the final target ID. An equal numeric owner ID alone proves nothing.
    """
    edges = {}
    for old, new in aliases:
        if old != new:
            edges.setdefault(int(old), set()).add(int(new))

    def follow(label, seen):
        if label in seen:
            return None
        destinations = edges.get(label, set())
        if not destinations:
            owners = final_segment_owners.get(label, set())
            return [(int(owner), [label]) for owner in sorted(owners)] or None
        results = []
        for destination in sorted(destinations):
            reached = follow(destination, seen | {label})
            if reached is None:
                return None
            results.extend((owner, [label, *path]) for owner, path in reached)
        return results

    segments = [int(value.split(":")[1]) for value in request.lineage if value.startswith("segment:")]
    proof = {"request_id": request.request_id, "captured_target_id": request.target_id,
             "source_map_version": request.source_map_version, "target_id": None, "segment_paths": []}
    if not segments:
        return {**proof, "reason": "MISSING_CAPTURED_SEGMENT_LINEAGE"}
    owner_set = set()
    for segment in segments:
        paths = follow(segment, set())
        if paths is None:
            return {**proof, "reason": "UNRESOLVED_SEGMENT_LINEAGE"}
        owner_set.update(owner for owner, _ in paths)
        proof["segment_paths"].extend(path for _, path in paths)
    if len(owner_set) != 1:
        return {**proof, "reason": "AMBIGUOUS_FINAL_OWNER"}
    owner = next(iter(owner_set))
    if owner <= 0 or owner not in active_owners:
        return {**proof, "reason": "TARGET_OUTSIDE_FINAL_REGISTRY"}
    return {**proof, "target_id": f"owner:{owner}", "reason": "VERIFIED_NATIVE_SEGMENT_LINEAGE"}


def direct_readouts(manifest: dict, records: dict, text: np.ndarray,
                    valid_ids: tuple[int, ...], incumbent: dict[int, int], *, model: str) -> dict:
    """All targets retain their incumbent unless at least one real request succeeds."""
    if model not in {"native", "siglip2", "wow"}:
        raise ValueError("unlisted semantic teacher")
    prefix = {"native": "S_NATIVE", "siglip2": "S_SIGLIP2", "wow": "S_WOW"}[model]
    methods = (prefix + "_VOTE",) if model == "wow" else (prefix + "_AREA", prefix + "_VOTE")
    result = {method: {} for method in methods}
    normalized_text = np.asarray(text, np.float64)
    normalized_text = normalized_text / np.linalg.norm(normalized_text, axis=1, keepdims=True)
    for owner, label in incumbent.items():
        requests = manifest["views"].get(f"owner:{owner}", [])
        successful = [(rank, request_id, records[request_id]) for rank, request_id in enumerate(requests)
                      if records[request_id]["status"] == "COMPLETE"]
        if not successful:
            for method in methods:
                result[method][owner] = {"label_id": label, "successful_views": 0,
                    "technical_fallback": True, "scores": [], "aggregate_feature": None}
            continue
        if model == "wow":
            view_labels = [valid_ids[row["mapping"]["class_index"]] for _, _, row in successful]
        else:
            features = np.stack([row["feature"] for _, _, row in successful])
            areas = [manifest["requests"][request_id]["visible_target_pixels"] for _, request_id, _ in successful]
            area = area_readout(features, areas, text, valid_ids=valid_ids)
            result[prefix + "_AREA"][owner] = {**asdict(area), "technical_fallback": False}
            view_features = features / np.linalg.norm(features, axis=1, keepdims=True)
            view_labels = [valid_ids[index] for index in np.argmax(view_features @ normalized_text.T, axis=1)]
        vote = vote_readout(label_ids=view_labels, request_ranks=[rank for rank, _, _ in successful], valid_ids=valid_ids)
        result[prefix + "_VOTE"][owner] = {**asdict(vote), "technical_fallback": False}
    return result
