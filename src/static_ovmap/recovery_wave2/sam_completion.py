"""Cached winner-preserving SAM completion and own-map causal conflict checks."""

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image

from static_ovmap.backbone_wave1.diagnostics import canonical_partition
from static_ovmap.backbone_wave1.frontend_sam2 import warp_visible_support
from static_ovmap.backbone_wave1.fusion import spatial_key
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _write_npz

from .binding import ConsumptionIndex, PathResolver, read
from .light import require_transfer_freeze


def crop_priority_raster(crop, masks, winners):
    crop, masks, winners = np.asarray(crop), np.asarray(masks, bool), np.asarray(winners, bool)
    if (crop.ndim != 2 or masks.ndim != 3 or masks.shape[1:] != crop.shape
            or winners.shape != masks.shape or not np.issubdtype(crop.dtype, np.integer)
            or np.any(crop < 0) or np.any(winners & ~masks) or np.any(winners.sum(0) > 1)):
        raise ValueError("aligned exclusive cached winner supports and nonnegative CropFormer labels required")
    groups = sorted((int(x) for x in np.unique(crop) if x > 0), key=lambda x: spatial_key(crop == x))
    area = np.asarray([np.count_nonzero(crop == group) for group in groups], np.int64)
    binary_area = masks.sum((1, 2))
    intersections = np.zeros((len(masks), len(groups)), np.int64)
    for i, mask in enumerate(masks):
        values, counts = np.unique(crop[mask], return_counts=True)
        count_by_group = dict(zip(values.tolist(), counts.tolist()))
        intersections[i] = [count_by_group.get(group, 0) for group in groups]
    union = binary_area[:, None] + area[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(union, float), where=union > 0)
    result, rows, next_group = crop.astype(np.int32, copy=True), [], int(crop.max(initial=0)) + 1
    for i, mask in enumerate(masks):
        best = np.flatnonzero(iou[i] == iou[i].max()) if groups else np.empty(0, int)
        j = int(best[0]) if len(best) == 1 else None
        column_best = np.flatnonzero(iou[:, j] == iou[:, j].max()) if j is not None else []
        ambiguous = len(best) > 1 or (j is not None and len(column_best) > 1)
        mutual = j is not None and len(column_best) == 1 and int(column_best[0]) == i
        coverage = float(intersections[i, j] / area[j]) if j is not None else None
        best_iou = float(iou[i, j]) if j is not None else None
        overlap = float(intersections[i].sum() / binary_area[i]) if binary_area[i] else 0.
        additions = winners[i] & (crop == 0)
        attached = mutual and coverage >= .8 and best_iou >= .5
        action, output_group = "OMITTED", None
        if attached:
            action, output_group = "ATTACHED", groups[j]
        elif binary_area[i] and overlap < .2:
            action = "STANDALONE"
            if additions.any():
                output_group, next_group = next_group, next_group + 1
        if output_group is not None:
            if output_group > np.iinfo(np.uint16).max:
                raise ValueError("completion labels exceed the unchanged native uint16 domain")
            result[additions] = output_group
        rows.append({"track_index": i, "action": action, "output_group": output_group,
            "attached_crop_group": groups[j] if attached else None,
            "binary_pixels": int(binary_area[i]), "winning_pixels": int(winners[i].sum()),
            "added_pixels": int(additions.sum()) if output_group is not None else 0,
            "crop_positive_overlap": overlap, "coverage": coverage, "iou": best_iou,
            "ambiguous_best": bool(ambiguous)})
    if not np.array_equal(result[crop > 0], crop[crop > 0]):
        raise RuntimeError("completion modified protected CropFormer pixels")
    return result, rows


class CachedTracks:
    def __init__(self, sam_receipt, capture_manifest, *, resolver=None, memo=None):
        self.resolver, self.index = resolver or PathResolver(), ConsumptionIndex(memo)
        self.sam_path = Path(self.resolver.resolve(sam_receipt))
        self.capture_path = Path(self.resolver.resolve(capture_manifest))
        self.sam, self.capture = read(self.sam_path), read(self.capture_path)
        self.index.identity(self.sam_path)
        self.index.identity(self.capture_path)
        source_path = Path(self.resolver.resolve(self.sam["source_capture"]))
        recorded = {self.resolver.resolve(row["path"]): row for row in self.sam["inputs_and_outputs"]}
        self.index.identity(source_path, recorded.get(str(source_path)))
        source_capture = read(source_path)
        if (self.sam["status"] != "COMPLETE"
                or [row["frame_id"] for row in self.sam["frames"]] != self.capture["completed_frame_ids"]
                or source_capture["scheduled_frame_ids"] != self.capture["scheduled_frame_ids"]
                or source_capture["completed_frame_ids"] != self.capture["completed_frame_ids"]):
            raise ValueError("cached SAM tracks differ from the exact inherited completed frame universe")
        keys = ("frame_id", "rgb_sha256", "depth_sha256", "panoptic_sha256", "pose_c2w", "intrinsics")
        for old, current in zip(source_capture["frames"], self.capture["frames"], strict=True):
            if any(old[key] != current[key] for key in keys):
                raise ValueError("SAM source capture changed actual RGB-D, CropFormer, pose, or intrinsics")
        self.frames = {row["frame_id"]: row for row in self.sam["frames"]}
        self.capture_frames = {row["frame_id"]: row for row in self.capture["frames"]}
        self.seed_keys = {}

    def crop(self, frame_id):
        row = self.capture_frames[frame_id]
        path = self.capture_path.parent / row["panoptic_path"]
        self.index.identity(path, {"sha256": row["panoptic_sha256"]})
        return np.asarray(Image.open(path), np.int32)

    def load(self, frame_id):
        row, crop = self.frames[frame_id], self.crop(frame_id)
        chunk = row["chunk"]
        if chunk not in self.seed_keys:
            first = next(frame for frame in self.sam["frames"] if frame["chunk"] == chunk)
            seed = self.crop(first["frame_id"])
            groups = sorted((int(x) for x in np.unique(seed) if x > 0), key=lambda x: spatial_key(seed == x))
            self.seed_keys[chunk] = {i + 1: spatial_key(seed == group) for i, group in enumerate(groups)}
        track_path = Path(self.resolver.resolve(row["tracks"]["path"]))
        self.index.identity(track_path, row["tracks"])
        with np.load(track_path, allow_pickle=False) as arrays:
            ids = np.asarray(arrays["track_ids"], np.int64)
            shape = tuple(int(x) for x in arrays["shape"])
            bits = np.asarray(arrays["raw_masks_bits"], np.uint8)
        if shape != crop.shape or len(np.unique(ids)) != len(ids) or any(int(x) not in self.seed_keys[chunk] for x in ids):
            raise ValueError("cached SAM identity or shape cannot be reconciled with original chunk seeds")
        if len(ids):
            if bits.shape != (len(ids), (crop.size + 7) // 8):
                raise ValueError("cached binary SAM support has an invalid packed shape")
            masks = np.unpackbits(bits, axis=1, count=crop.size).reshape((len(ids),) + shape).astype(bool)
        else:
            masks = np.empty((0,) + shape, bool)
        raw_path = Path(self.resolver.resolve(row["outputs"]["raw"]["path"]))
        self.index.identity(raw_path, row["outputs"]["raw"])
        raw = np.asarray(Image.open(raw_path), np.int32)
        winners = np.zeros_like(masks)
        exact = row["diagnostic"].get("state") in {"EXACT_CHUNK_FIRST", "NO_SEED"}
        if exact:
            if not np.array_equal(raw, crop):
                raise ValueError("chunk-first or no-seed cached frontend is not exact CropFormer")
            if row["current_index"] == 0 and len(ids):
                for i, track in enumerate(ids):
                    if spatial_key(masks[i]) != self.seed_keys[chunk][int(track)]:
                        raise ValueError("chunk-first cached binary track differs from its original seed")
                winners[:] = masks
        else:
            by_seed = {self.seed_keys[chunk][int(track)]: i for i, track in enumerate(ids)}
            seen = set()
            for remap in row["diagnostic"]["raw_remap"]:
                if remap["kind"] != "SAM_TRACK":
                    continue
                key = tuple(remap["seed_key"])
                if key not in by_seed or key in seen:
                    raise ValueError("RAW winner remap has missing or ambiguous cached track identity")
                seen.add(key)
                winners[by_seed[key]] = raw == remap["dense_group"]
            if (raw.shape != crop.shape or np.any(winners & ~masks) or np.any(winners.sum(0) > 1)
                    or not np.array_equal(winners.any(0), masks.any(0))):
                raise ValueError("recorded RAW winners cannot reconstruct the cached binary track union")
        return {"source": row, "crop": crop, "masks": masks, "winners": winners, "ids": ids,
                "seed_keys": [self.seed_keys[chunk][int(track)] for track in ids], "exact": exact,
                "track_artifact": self.index.identity(track_path, row["tracks"]),
                "raw_artifact": self.index.identity(raw_path, row["outputs"]["raw"])}


def build_s1(binding, scene, *, resume=False):
    require_transfer_freeze(binding, scene)
    data, root = binding["scenes"][scene], Path(binding["output_root"]) / "frontend" / scene / "S1"
    root.mkdir(parents=True, exist_ok=True)
    loader = CachedTracks(data["sam_receipt"], data["capture_manifest"], resolver=PathResolver(binding["path_map"]),
                          memo=root / "input_verifications.json")
    identity = canonical_digest({"binding": binding["identity"], "scene": scene,
        "operator": loader.index.identity(__file__), "sources": loader.index.entries(), "recipe": "S1_FIXED"})
    path = root / "receipt.json"
    if path.is_file():
        old = read(path)
        if not resume or old["input_identity"] != identity or old["status"] != "COMPLETE":
            raise ValueError("S1 can resume only a complete frontend with the exact consumed input identity")
        for frame in old["frames"]:
            loader.load(frame["frame_id"])
            loader.index.identity(frame["output"]["path"], frame["output"])
        return old
    (root / "rasters").mkdir(exist_ok=True)
    started, frames = time.monotonic(), []
    for source in loader.sam["frames"]:
        frame_id, tracks = source["frame_id"], loader.load(source["frame_id"])
        crop = tracks["crop"]
        if tracks["exact"]:
            raster, rows = crop.copy(), []
        else:
            raster, rows = crop_priority_raster(crop, tracks["masks"], tracks["winners"])
        for row in rows:
            row.update(track_id=int(tracks["ids"][row["track_index"]]), seed_key=tracks["seed_keys"][row["track_index"]])
        output = root / "rasters" / f"{frame_id:06d}.png"
        Image.fromarray(raster.astype(np.uint16)).save(output)
        union = tracks["masks"].any(0)
        coverage = [{"crop_group": int(group), "SAM_union_coverage": float(union[crop == group].mean())}
                    for group in np.unique(crop) if group > 0]
        frames.append({"frame_id": frame_id, "chunk": source["chunk"], "current_index": source["current_index"],
            "state": source["diagnostic"].get("state", "CROP_PRIORITY_COMPLETION"), "tracks": rows,
            "track_artifact": tracks["track_artifact"], "raw_artifact": tracks["raw_artifact"],
            "output": loader.index.identity(output), "protected_crop_pixels": int((crop > 0).sum()),
            "protected_crop_groups": int(len(np.unique(crop[crop > 0]))),
            "added_pixels": int(((crop == 0) & (raster > 0)).sum()),
            "partial_union_coverage_groups": [row for row in coverage if .2 <= row["SAM_union_coverage"] < .8],
            "changed_canonical_raster": bool(not np.array_equal(canonical_partition(crop), canonical_partition(raster)))})
    result = {"status": "COMPLETE", "scene": scene, "frontend": "S1", "input_identity": identity,
        "source_sam_identity": loader.sam["identity"], "frames": frames,
        "scheduled_frame_ids": loader.capture["scheduled_frame_ids"],
        "completed_frame_ids": loader.capture["completed_frame_ids"],
        "changed_canonical_frames": sum(row["changed_canonical_raster"] for row in frames),
        "added_pixels": sum(row["added_pixels"] for row in frames),
        "protected_crop_pixels": sum(row["protected_crop_pixels"] for row in frames),
        "all_positive_crop_pixels_unchanged": True, "original_separate_crop_groups_preserved": True,
        "new_SAM_forwards": 0, "new_CropFormer_forwards": 0, "GT_input": False,
        "inputs_and_outputs": loader.index.entries(), "elapsed_seconds": time.monotonic() - started}
    result["identity"] = canonical_digest(result)
    loader.index.write_memo(root / "input_verifications.json")
    atomic_write_json(path, result)
    print(scene, "S1", result["changed_canonical_frames"], "changed frames", flush=True)
    return result


def known_stable_support(probe, depth, snapshots):
    depth = np.asarray(depth)
    labels, owners = np.asarray(probe["prior_label"]), np.asarray(probe["prior_owner"])
    hit, valid = np.asarray(probe["hit_depth"]), np.asarray(probe["validity"], bool)
    if any(array.shape != depth.shape for array in (labels, owners, hit, valid)):
        raise ValueError("current own-map probe must align with current measured depth")
    result = np.zeros_like(depth, bool)
    if len(snapshots) < 2:
        return result
    result = valid & (labels > 0) & (owners > 0) & np.isfinite(hit) & (hit > 0)
    result &= np.isfinite(depth) & (depth > 0) & (np.abs(hit - depth) <= .03 + .01 * depth)
    for snapshot in snapshots[-2:]:
        lookup = np.zeros(labels.shape, np.int64)
        keys = np.asarray(sorted(snapshot), np.int64)
        if not len(keys):
            return np.zeros_like(result)
        at = np.searchsorted(keys, labels)
        exists = (at < len(keys)) & (keys[np.minimum(at, len(keys) - 1)] == labels)
        values = np.asarray([snapshot[int(k)] for k in keys], np.int64)
        lookup[exists] = values[at[exists]]
        result &= exists & (lookup > 0) & (lookup == owners)
    return result


def filter_additions(additions, proposed, stable, owners, self_owner):
    additions, proposed, stable, owners = (np.asarray(additions, bool), np.asarray(proposed, bool),
                                            np.asarray(stable, bool), np.asarray(owners))
    if any(x.shape != additions.shape for x in (proposed, stable, owners)) or np.any(additions & ~proposed):
        raise ValueError("SAM additions must be an aligned subset of the actual proposed support")
    known = proposed & stable & (owners > 0)
    count = int(known.sum())
    own = int((known & (owners == self_owner)).sum())
    other = count - own
    q = float(other / count) if count else None
    reject = self_owner is not None and self_owner > 0 and count >= 100 and q > .2
    kept = np.zeros_like(additions) if reject else additions.copy()
    return kept, {"known_proposed_pixels": count, "known_self": own, "known_other": other,
        "q_other": q, "decision": "DROP_SAM_ADDITIONS" if reject else "KEEP_S1",
        "removed_additions": int(additions.sum()) if reject else 0}


def _hook_tracks(hook, frame_id):
    if not hasattr(hook, "recovery_tracks"):
        receipt_path = Path(hook.recipe["frontend_receipt"])
        receipt = read(receipt_path)
        if receipt["identity"] != canonical_digest({k: v for k, v in receipt.items() if k != "identity"}):
            raise ValueError("locked S1 receipt content changed")
        hook.recovery_s1_frames = {row["frame_id"]: row for row in receipt["frames"]}
        hook.recovery_tracks = CachedTracks(hook.recipe["parent_sam_receipt"], hook.recipe["capture_manifest"],
                                            memo=receipt_path.parent / "input_verifications.json")
    return hook.recovery_tracks.load(frame_id), hook.recovery_s1_frames[frame_id]


def causal_s2_raster(hook, frame_id, depth, intrinsics, pose, pano, frame_root):
    tracks, s1 = _hook_tracks(hook, frame_id)
    crop, result = tracks["crop"], np.asarray(pano).copy()
    if (hook.frontend_context["frame_id"] != frame_id or not np.array_equal(crop, hook.frontend_context["original"])
            or not np.array_equal(result, hook.frontend_context["s1"])):
        raise ValueError("S2 current frontend context differs from the locked original CropFormer/S1 inputs")
    probe = hook.gsm.exportAssociationProbe(np.asarray(pose, np.float32), np.asarray(depth, np.float32))
    if probe["state_before"] != probe["state_after"]:
        raise RuntimeError("S2 own-map preinsert probe mutated native state")
    stable = known_stable_support(probe, depth, [row["membership"] for row in hook.history])
    owners, diagnostics = np.asarray(probe["prior_owner"]), []
    previous = hook.history[-1] if hook.history else None
    for row in s1["tracks"]:
        i, group = row["track_index"], row["output_group"]
        additions = tracks["winners"][i] & (crop == 0) if group is not None else np.zeros_like(crop, bool)
        if additions.any() and not np.all(result[additions] == group):
            raise ValueError("S2 additions disagree with the actual recorded S1 winner insertion")
        proposed = additions | (crop == row["attached_crop_group"]) if row["attached_crop_group"] is not None else additions
        self_owner, purity, known_past = None, None, 0
        reason = "NO_SAM_ADDITIONS" if not additions.any() else "HISTORY_OR_TRACK_UNKNOWN"
        if (additions.any() and len(hook.history) >= 2 and previous["chunk"] == s1["chunk"]
                and row["track_id"] in previous["binary_tracks"]):
            past = previous["binary_tracks"][row["track_id"]]
            visible = warp_visible_support(past, previous["depth"], previous["pose"], previous["intrinsics"],
                                           depth, pose, intrinsics)
            visible_known = visible & stable
            values, counts = np.unique(owners[visible_known], return_counts=True)
            known_past = int(counts.sum())
            if known_past >= 100:
                best = np.flatnonzero(counts == counts.max())
                purity = float(counts.max() / known_past)
                if len(best) == 1 and purity >= .8 and values[best[0]] > 0:
                    self_owner, reason = int(values[best[0]]), "SELF_RESOLVED"
                else:
                    reason = "SELF_PURITY_UNKNOWN"
            else:
                reason = "INSUFFICIENT_STABLE_VISIBLE_PAST"
        kept, diagnostic = filter_additions(additions, proposed, stable, owners, self_owner)
        if self_owner is None:
            kept, diagnostic["decision"], diagnostic["removed_additions"] = additions, "KEEP_S1", 0
        result[additions & ~kept] = 0
        diagnostic.update(track_id=row["track_id"], seed_key=row["seed_key"], action=row["action"],
            self_owner=self_owner, self_purity=purity, known_visible_past_pixels=known_past,
            self_resolution=reason, additions=int(additions.sum()), proposed_pixels=int(proposed.sum()),
            history_abstention=self_owner is None and bool(additions.any()))
        diagnostics.append(diagnostic)
    if not np.array_equal(result[crop > 0], crop[crop > 0]):
        raise RuntimeError("S2 conflict rejection changed a protected CropFormer pixel")
    _write_npz(frame_root / "s2_preinsert_probe.npz", {name: np.asarray(probe[name])
               for name in ("prior_label", "prior_owner", "hit_depth", "validity")})
    atomic_write_json(frame_root / "s2_conflict.json", {"frame_id": frame_id, "chunk": s1["chunk"],
        "prior_state_id": canonical_digest(probe["state_before"]), "state_before": probe["state_before"],
        "state_after": probe["state_after"], "past_snapshot_frames": [row["frame_id"] for row in hook.history],
        "stable_known_pixels": int(stable.sum()), "tracks": diagnostics,
        "removed_additions": sum(row["removed_additions"] for row in diagnostics),
        "protected_crop_pixels": int((crop > 0).sum()), "GT_input": False, "new_SAM_forwards": 0})
    return result


def capture_s2_history(hook, frame_id, depth, intrinsics, pose, pano):
    tracks, s1 = _hook_tracks(hook, frame_id)
    state = hook.gsm.exportStudyFrameState()
    membership, ambiguous = {}, set()
    for row in state["label_instances"]:
        label, owner = int(row["segment_label"]), int(row["raycast_instance_label"])
        if label in membership and membership[label] != owner:
            ambiguous.add(label)
        membership[label] = owner
    membership = {label: owner for label, owner in membership.items() if label not in ambiguous and label > 0 and owner > 0}
    atomic_write_json(hook.root / f"{frame_id:06d}" / "s2_membership_snapshot.json", {
        "frame_id": frame_id, "chunk": s1["chunk"], "membership_factor": 0.0,
        "membership": {str(k): v for k, v in sorted(membership.items())}, "ambiguous_labels": sorted(ambiguous),
        "after_insertion_state_id": canonical_digest(state), "original_binary_tracks": tracks["track_artifact"],
        "track_context_resets_at_chunk_boundary": True, "native_history_resets_at_chunk_boundary": False})
    hook.history.append({"frame_id": frame_id, "chunk": s1["chunk"], "membership": membership,
        "depth": np.asarray(depth).copy(), "pose": np.asarray(pose).copy(), "intrinsics": np.asarray(intrinsics).copy(),
        "binary_tracks": {int(track): tracks["masks"][i].copy() for i, track in enumerate(tracks["ids"])}})
    hook.history[:] = hook.history[-2:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build_s1(read(args.binding), args.scene, resume=args.resume)
