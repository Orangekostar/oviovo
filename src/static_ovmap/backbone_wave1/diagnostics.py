"""Post-lock raw numeric instance and observed-surface diagnostics."""

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction, project_values
from static_ovmap.composition_study.evaluation import load_targets
from static_ovmap.released_loader import load_released_module
from .binding import read


def instance_diagnostics(projected, gt_combined, valid_ids, *, minimum=100):
    projected, gt = np.asarray(projected), np.asarray(gt_combined)
    if projected.shape != gt.shape:
        raise ValueError("raw instance masks and target rows must align")
    pids, pc = np.unique(projected, return_counts=True)
    pids, pc = pids[(pids > 0) & (pc >= minimum)], pc[(pids > 0) & (pc >= minimum)]
    gids, gc = np.unique(gt, return_counts=True)
    keep = (gids >= 1000) & np.isin(gids // 1000, valid_ids) & (gc >= minimum)
    gids, gc = gids[keep], gc[keep]
    intersections = np.zeros((len(gids), len(pids)), np.int64)
    if len(gids) and len(pids):
        pair, count = np.unique(np.column_stack((gt, projected)), axis=0, return_counts=True)
        gi = {int(k): i for i, k in enumerate(gids)}
        pi = {int(k): i for i, k in enumerate(pids)}
        for (g, p), n in zip(pair, count):
            if int(g) in gi and int(p) in pi:
                intersections[gi[int(g)], pi[int(p)]] = n
    union = gc[:, None] + pc[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(union, float), where=union > 0)
    matches = {}
    for threshold in (.25, .5, .75):
        matching = maximum_bipartite_matching(csr_matrix(iou > threshold), perm_type="column")
        pairs = [{"gt_id": int(gids[g]), "owner": int(pids[p]), "iou": float(iou[g, p])}
                 for g, p in enumerate(matching) if p >= 0]
        matches[str(threshold)] = {"matched_gt": len(pairs), "eligible_gt": len(gids),
            "recall": len(pairs) / len(gids) if len(gids) else None, "pairs": pairs}
    per_gt = []
    for g, gid in enumerate(gids):
        substantial = np.flatnonzero((intersections[g] >= minimum) & (intersections[g] / gc[g] >= .01))
        per_gt.append({"gt_id": int(gid), "pixels": int(gc[g]),
            "best_iou": float(iou[g].max()) if len(pids) else 0.,
            "substantial_fragments": len(substantial), "intersections": [
                {"owner": int(pids[p]), "pixels": int(intersections[g, p]),
                 "coverage": float(intersections[g, p] / gc[g]),
                 "purity": float(intersections[g, p] / pc[p]), "iou": float(iou[g, p])}
                for p in substantial]})
    return {"predicted_min100_raw_owners": len(pids), "eligible_gt": len(gids),
        "matching": "MAXIMUM_CARDINALITY_ONE_TO_ONE_STRICT_IOU_GT_THRESHOLD",
        "substantial_intersection_rule": "pixels_ge_100_AND_GT_coverage_ge_0.01_FIXED",
        "matches": matches, "per_gt": per_gt,
        "mean_best_gt_iou": float(np.mean([r["best_iou"] for r in per_gt])) if per_gt else None}


def nearest_squared(source, target):
    import open3d.core as o3c
    source, target = np.asarray(source, np.float32), np.asarray(target, np.float32)
    search = o3c.nns.NearestNeighborSearch(o3c.Tensor(np.ascontiguousarray(source)))
    search.knn_index()
    values = []
    for begin in range(0, len(target), 250000):
        _, distance = search.knn_search(o3c.Tensor(np.ascontiguousarray(target[begin:begin + 250000])), 1)
        values.append(distance.numpy().reshape(-1))
    return np.concatenate(values)


def canonical_partition(owners):
    owners = np.asarray(owners)
    unique, first, inverse = np.unique(owners, return_index=True, return_inverse=True)
    ordered = sorted((int(i) for i in range(len(unique)) if unique[i] > 0), key=lambda i: first[i])
    labels = np.zeros(len(unique), np.int64)
    labels[ordered] = np.arange(1, len(ordered) + 1)
    return labels[inverse]


def diagnose_map(binding, spec, scene, receipt):
    root = Path(receipt["config"]).parent
    path = root / "raw_geometry_diagnostics.json"
    if path.is_file():
        return read(path)
    native = load_prediction(receipt["predictions"]["NATIVE_READOUT"])
    if any(not load_prediction(p).locked for p in receipt["predictions"].values()):
        raise ValueError("all readouts must lock before GT diagnosis")
    config = read(receipt["config"])
    targets = load_targets(config, scene, native)
    capture_path = Path(config["scenes"][scene]["capture"])
    capture = read(capture_path)
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        raw_owners, xyz = arrays["original_owner"], arrays["surface_xyz"]
        raw = project_values(raw_owners, targets["nearest"], targets["matched"])
        geometry_identity = {"xyz": _array_digest(xyz), "faces": _array_digest(arrays["surface_faces"]),
                             "raw_canonical_partition": _array_digest(canonical_partition(raw_owners))}
    combined = np.load(targets["gt_instance_path"], allow_pickle=False).reshape(-1)
    diagnostic = instance_diagnostics(raw, combined, targets["valid_ids"])
    valid_target = targets["xyz"][np.isin(targets["gt_semantic"], targets["valid_ids"])]
    precision = float(np.mean(nearest_squared(valid_target, xyz) < .05 ** 2))
    completeness = float(np.mean(nearest_squared(xyz, valid_target) < .05 ** 2))
    fscore = 2 * precision * completeness / (precision + completeness) if precision + completeness else 0.
    diagnostic.update(status="COMPLETE", scene=scene, map_id=receipt["map_id"],
        geometry_identity=geometry_identity, raw_numeric_owner_count=len(np.unique(raw_owners[raw_owners > 0])),
        native_eligible_owner_count=receipt["native_eligible_owners"],
        raw_vs_native_painted_source_rows=int(np.count_nonzero(raw_owners != native.owner_ids)),
        surface={"precision_5cm": precision, "completeness_5cm": completeness, "fscore_5cm": fscore,
            "sampling": "ALL_NATIVE_EXPORT_VERTICES_DUPLICATES_PRESERVED_ALL_VALID_TARGET_VERTICES",
            "distance_rule": "OPEN3D_FP32_1NN_STRICT_SQUARED_DISTANCE_LT_0.05_SQUARED",
            "predicted_rows": len(xyz), "valid_target_rows": len(valid_target)})
    np.savez_compressed(root / "raw_projected_owners.npz", owners=raw)
    atomic_write_json(path, diagnostic)
    return diagnostic


def compare_maps(binding, scene, map_id):
    root = Path(binding["output_root"]) / "readouts" / scene
    a, b = root / "BB00_NATIVE", root / map_id
    with np.load(a / "raw_projected_owners.npz") as x, np.load(b / "raw_projected_owners.npz") as y:
        first, second = x["owners"], y["owners"]
    positive_a, positive_b = sorted(set(first.tolist()) - {0}), sorted(set(second.tolist()) - {0})
    ai, bi = {k: i for i, k in enumerate(positive_a)}, {k: i for i, k in enumerate(positive_b)}
    intersections = np.zeros((len(ai), len(bi)), np.int64)
    pairs, counts = np.unique(np.column_stack((first, second)), axis=0, return_counts=True)
    for (p, q), count in zip(pairs, counts):
        if p in ai and q in bi:
            intersections[ai[p], bi[q]] = count
    indices_a, indices_b = linear_sum_assignment(intersections, maximize=True)
    mapping = {positive_b[j]: positive_a[i] for i, j in zip(indices_a, indices_b) if intersections[i, j] > 0}
    same = (first == 0) & (second == 0)
    associations = []
    for q, p in mapping.items():
        same |= (first == p) & (second == q)
        overlap = int(intersections[ai[p], bi[q]])
        union = int(np.sum(first == p) + np.sum(second == q) - overlap)
        associations.append({"baseline_owner": p, "variant_owner": q, "intersection": overlap, "iou": overlap / union})
    da, db = read(a / "raw_geometry_diagnostics.json"), read(b / "raw_geometry_diagnostics.json")
    old = {row["gt_id"]: row for row in da["per_gt"]}
    gt_changes = [{"gt_id": row["gt_id"], "baseline_best_iou": old[row["gt_id"]]["best_iou"],
                   "variant_best_iou": row["best_iou"], "delta": row["best_iou"] - old[row["gt_id"]]["best_iou"]}
                  for row in db["per_gt"]]
    result = {"scene": scene, "map_id": map_id, "changed_target_partition_rows": int((~same).sum()),
        "geometry_exact_identity": da["geometry_identity"] == db["geometry_identity"],
        "real_raw_partition_intervention": bool((~same).any()), "predicted_support_associations": associations,
        "association_uses_GT_labels": False, "GT_indexed_best_iou_changes": gt_changes,
        "surface_delta": {key: db["surface"][key] - da["surface"][key] for key in
                          ("precision_5cm", "completeness_5cm", "fscore_5cm")}}
    atomic_write_json(b / "intervention.json", result)
    return result
