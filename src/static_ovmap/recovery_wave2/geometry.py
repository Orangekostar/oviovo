"""Locked raw-map diagnosis and complete-cohort resource screening."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from static_ovmap.backbone_wave1.diagnostics import canonical_partition, instance_diagnostics, nearest_squared
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.scannet_ground_truth import load_ground_truth
from static_ovmap.module_validation.scannet_study import freeze_projection, project_values

from .binding import ConsumptionIndex, PathResolver, read


def screen_cohort(baseline, variants, scene_order):
    if (len(baseline) != len(scene_order) or len(variants) != len(scene_order)
            or {row["scene"] for row in baseline} != set(scene_order)
            or {row["scene"] for row in variants} != set(scene_order)
            or any(row["status"] != "COMPLETE" for row in baseline + variants)):
        raise ValueError("geometry screening requires the exact complete cohort")
    base, variant = {row["scene"]: row for row in baseline}, {row["scene"]: row for row in variants}
    for scene in scene_order:
        if [row["gt_id"] for row in base[scene]["per_gt"]] != [row["gt_id"] for row in variant[scene]["per_gt"]]:
            raise ValueError("geometry cohorts differ in eligible GT object universe")

    def aggregate(rows):
        objects = [obj for scene in scene_order for obj in rows[scene]["per_gt"]]
        if not objects:
            raise ValueError("geometry screening cannot use an empty GT universe")
        return {"eligible_gt": len(objects), "mean_best_gt_iou": float(np.mean([obj["best_iou"] for obj in objects])),
            "R50_count": sum(rows[scene]["matches"]["0.5"]["matched_gt"] for scene in scene_order),
            "substantial_fragments": sum(obj["substantial_fragments"] for obj in objects)}

    b, v = aggregate(base), aggregate(variant)
    drop, loss = b["mean_best_gt_iou"] - v["mean_best_gt_iou"], b["R50_count"] - v["R50_count"]
    iou_rule = drop >= .03 and loss >= 2
    fragment_rule = v["substantial_fragments"] >= 1.75 * b["substantial_fragments"] and loss >= 1
    catastrophic = bool(iou_rule or fragment_rule)
    return {"status": "SCREENED_OUT_GEOMETRY" if catastrophic else "GEOMETRY_PASS", "scene_order": scene_order,
        "baseline": b, "variant": v, "mean_best_iou_drop_pp": 100 * drop, "R50_count_loss": loss,
        "fragment_ratio": v["substantial_fragments"] / b["substantial_fragments"] if b["substantial_fragments"] else None,
        "iou_rule_triggered": bool(iou_rule), "fragment_rule_triggered": bool(fragment_rule),
        "semantic_status": "NOT_RUN_RESOURCE_SCREEN" if catastrophic else "NOT_YET_RUN",
        "aggregation": "ALL_ELIGIBLE_GT_OBJECTS_CONCATENATED_AND_R50_COUNTS_SUMMED",
        "decision_uses_only_complete_raw_geometry": True}


def partition_comparison(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("map comparison requires identical whole-scene target rows")
    ids_a, ids_b = np.unique(first[first > 0]), np.unique(second[second > 0])
    ai, bi = {int(k): i for i, k in enumerate(ids_a)}, {int(k): i for i, k in enumerate(ids_b)}
    intersections = np.zeros((len(ai), len(bi)), np.int64)
    pairs, counts = np.unique(np.column_stack((first, second)), axis=0, return_counts=True)
    for (p, q), count in zip(pairs, counts):
        if int(p) in ai and int(q) in bi:
            intersections[ai[int(p)], bi[int(q)]] = count
    a, b = linear_sum_assignment(intersections, maximize=True)
    same, associations = (first == 0) & (second == 0), []
    for i, j in zip(a, b):
        if not intersections[i, j]:
            continue
        p, q = int(ids_a[i]), int(ids_b[j])
        same |= (first == p) & (second == q)
        union = int((first == p).sum() + (second == q).sum() - intersections[i, j])
        associations.append({"baseline_owner": p, "variant_owner": q, "intersection": int(intersections[i, j]),
                             "iou": float(intersections[i, j] / union)})
    return {"changed_target_partition_rows": int((~same).sum()),
        "real_raw_partition_intervention": bool((~same).any()), "predicted_support_associations": associations,
        "association_uses_GT_labels": False}


def diagnose_scene(binding, scene, map_id, *, resume=False):
    root = Path(binding["output_root"]) / "geometry" / scene / map_id
    map_path = Path(binding["output_root"]) / "maps" / scene / map_id / "map_receipt.json"
    mapping = read(map_path)
    if (mapping["status"] != "COMPLETE" or mapping["scene"] != scene or mapping["map_id"] != map_id
            or not mapping["geometry_locked_before_semantics_and_labels"] or mapping["GT_input"]):
        raise ValueError("geometry labels may be opened only after a complete raw own-map lock")
    resolver = PathResolver(binding["path_map"])
    index = ConsumptionIndex(root / "input_verifications.json")
    index.identity(map_path)
    capture_path = Path(mapping["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    surface_path = capture_path.parent / capture["surface"]["path"]
    surface_id = index.identity(surface_path, capture["surface"])
    config = resolver.rewrite(read(binding["scenes"][scene]["config"]))
    index.identity(binding["scenes"][scene]["config"])
    annotations_path = Path(config["scenes"][scene]["annotations"])
    annotations = resolver.rewrite(read(annotations_path))
    index.identity(annotations_path)
    for row in annotations["outputs"]:
        index.identity(row["path"], row)
    for path in (Path(__file__), Path(__file__).parents[1] / "backbone_wave1/diagnostics.py",
                 Path(__file__).parents[1] / "module_validation/scannet_study.py",
                 Path(__file__).parents[1] / "module_validation/scannet_ground_truth.py"):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "map_id": map_id,
        "map_input_identity": mapping["input_identity"], "surface": surface_id, "inputs": index.entries()})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = read(receipt_path)
        if not resume or receipt["status"] != "COMPLETE" or receipt["input_identity"] != identity:
            raise ValueError("geometry diagnosis can resume only with the complete exact input identity")
        for row in receipt["outputs"]:
            index.identity(row["path"], row)
        return read(root / "diagnostic.json")
    started = time.monotonic()
    # The raw geometry receipt is already locked; target labels are first consumed here.
    targets = load_ground_truth(annotations_path)
    if tuple(map(int, targets["valid_ids"])) != tuple(config["models"]["native"]["valid_ids"]):
        raise ValueError("geometry diagnostics changed the inherited class vocabulary")
    combined = np.load(targets["gt_instance_path"], allow_pickle=False).reshape(-1)
    with np.load(surface_path, allow_pickle=False) as arrays:
        xyz, raw, faces = arrays["surface_xyz"], arrays["original_owner"], arrays["surface_faces"]
        projection = freeze_projection(xyz, targets["xyz"], root / "projection")
        projected = project_values(raw, projection["nearest"], projection["matched"])
        geometry_identity = {"xyz": _array_digest(xyz), "faces": _array_digest(faces),
                             "raw_canonical_partition": _array_digest(canonical_partition(raw))}
    diagnostic = instance_diagnostics(projected, combined, targets["valid_ids"])
    valid_target = targets["xyz"][np.isin(targets["gt_semantic"], targets["valid_ids"])]
    precision = float(np.mean(nearest_squared(valid_target, xyz) < .05 ** 2))
    completeness = float(np.mean(nearest_squared(xyz, valid_target) < .05 ** 2))
    fscore = 2 * precision * completeness / (precision + completeness) if precision + completeness else 0.
    parent_root = Path(binding["parent_root"]) / "readouts" / scene / "BB00_NATIVE"
    baseline = read(parent_root / "raw_geometry_diagnostics.json")
    index.identity(parent_root / "raw_geometry_diagnostics.json")
    index.identity(parent_root / "raw_projected_owners.npz")
    with np.load(parent_root / "raw_projected_owners.npz", allow_pickle=False) as arrays:
        comparison = partition_comparison(arrays["owners"], projected)
    diagnostic.update(status="COMPLETE", scene=scene, map_id=map_id, geometry_identity=geometry_identity,
        geometry_exact_identity=geometry_identity == baseline["geometry_identity"],
        raw_numeric_owner_count=len(np.unique(raw[raw > 0])), projection_identity=projection["identity"],
        annotation_receipt=str(annotations_path), map_lock_input_identity=mapping["input_identity"],
        map_locked_before_target_labels=True, semantic_predictions_required_for_geometry=False,
        surface={"precision_5cm": precision, "completeness_5cm": completeness, "fscore_5cm": fscore,
            "sampling": "ALL_NATIVE_EXPORT_VERTICES_DUPLICATES_PRESERVED_ALL_VALID_TARGET_VERTICES",
            "distance_rule": "OPEN3D_FP32_1NN_STRICT_SQUARED_DISTANCE_LT_0.05_SQUARED",
            "predicted_rows": len(xyz), "valid_target_rows": len(valid_target)},
        intervention=comparison, elapsed_seconds=time.monotonic() - started)
    old = {row["gt_id"]: row for row in baseline["per_gt"]}
    diagnostic["GT_indexed_best_iou_changes"] = [{"gt_id": row["gt_id"],
        "baseline_best_iou": old[row["gt_id"]]["best_iou"], "variant_best_iou": row["best_iou"],
        "delta": row["best_iou"] - old[row["gt_id"]]["best_iou"]} for row in diagnostic["per_gt"]]
    diagnostic["identity"] = canonical_digest(diagnostic)
    _write_npz(root / "raw_projected_owners.npz", {"owners": projected})
    atomic_write_json(root / "diagnostic.json", diagnostic)
    outputs = [index.identity(path) for path in (root / "diagnostic.json", root / "raw_projected_owners.npz",
                                                root / "projection/manifest.json", root / "projection/projection.npz")]
    atomic_write_json(receipt_path, {"status": "COMPLETE", "scene": scene, "map_id": map_id,
        "input_identity": identity, "diagnostic_identity": diagnostic["identity"], "inputs": index.entries(),
        "outputs": outputs, "GT_use": "POST_RAW_MAP_LOCK_GEOMETRY_DIAGNOSTICS", "elapsed_seconds": time.monotonic() - started})
    index.write_memo(root / "input_verifications.json")
    print(scene, map_id, "geometry", diagnostic["mean_best_gt_iou"], flush=True)
    return diagnostic


def gate_development(binding, map_id):
    scenes, root = binding["datasets"]["development"], Path(binding["output_root"])
    variants = [read(root / "geometry" / scene / map_id / "diagnostic.json") for scene in scenes]
    baseline = [read(Path(binding["parent_root"]) / "readouts" / scene / "BB00_NATIVE/raw_geometry_diagnostics.json")
                for scene in scenes]
    result = screen_cohort(baseline, variants, scenes)
    result.update(map_id=map_id, diagnostic_identities=[row["identity"] for row in variants],
                  baseline_geometry_identities=[row["geometry_identity"] for row in baseline])
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "geometry/screens" / (map_id + ".json"), result)
    print(map_id, result["status"], "bestIoU drop pp", result["mean_best_iou_drop_pp"],
          "R50 loss", result["R50_count_loss"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    binding = read(args.binding)
    if args.scene:
        diagnose_scene(binding, args.scene, args.map_id, resume=args.resume)
    else:
        gate_development(binding, args.map_id)
