"""Export all frozen restored rows for transparent local success/failure display."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_tesse_cd_common_v2 import _voxel_centers
from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    selected = json.loads((compact / "selected_configs.json").read_text())["FINAL_CURRENT_READOUT"]
    if selected["state_variant"] != "H2":
        raise ValueError("this restored-row diagnostic requires the selected H2 state")
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    with np.load(run / "bridge/B3_legacy_prediction.npz") as data:
        b3 = data["source_indices"]
    native_path = run / "dev/apartment/native_cached_batch/H2_MV_NATIVE_CACHED.npz"
    with np.load(native_path) as data:
        h2 = data["source_indices"]
        rows = np.setdiff1d(h2, b3, assume_unique=True)
        local = np.searchsorted(h2, rows)
        native = data["semantic_ids"][local]
    prediction = run / selected["prediction"]
    with np.load(prediction) as data:
        if not np.array_equal(data["source_indices"], h2):
            raise ValueError("selected readout source changed")
        final = data["semantic_ids"][local]
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz = data["vertices_xyz"][rows]
    with np.load(run / "bridge/evaluation_context.npz") as data:
        targets = data["current_semantic_voxels"]
    distances, nearest = cKDTree(_voxel_centers(targets[:, :3])).query(xyz, k=1)
    supported = distances < 0.05
    gt = np.where(supported, targets[nearest, 3], -1)
    audit = json.loads((compact / "apartment_H2_recovered_semantic_attribution.json").read_text())
    if len(rows) != audit["source_rows"]:
        raise ValueError("restored-row population differs from audited population")
    for method, ids in (("MV_NATIVE_CACHED", native), (selected["exact_implementation"], final)):
        for key, mask in (("GT_supported_correct", supported & (ids == gt)),
                          ("GT_supported_incorrect", supported & (ids != gt))):
            expected = audit["methods"][method][key]
            if int(mask.sum()) != expected["source_rows"] or len(np.unique(xyz[mask], axis=0)) != expected["unique_physical_samples"]:
                raise ValueError(f"local examples disagree with full recovery audit: {method}/{key}")
    categories = np.full(len(rows), "GT_unmatched", dtype=object)
    old_correct, new_correct = native == gt, final == gt
    for label, mask in (("retained_correct", old_correct & new_correct),
                        ("improved", ~old_correct & new_correct),
                        ("regressed", old_correct & ~new_correct),
                        ("remaining_error", ~old_correct & ~new_correct)):
        categories[supported & mask] = label
    destination = compact / "figures/recovery_examples.source.csv"
    temporary = destination.with_suffix(".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["canonical_source_row", "x", "y", "z", "native_id", "final_id", "diagnostic_GT_id", "category"])
        writer.writerows((int(r), *map(float, p), int(a), int(b), int(g), str(c)) for r,p,a,b,g,c in zip(rows,xyz,native,final,gt,categories))
    temporary.replace(destination)
    _atomic_json(compact / "figures/recovery_examples.manifest.json", {
        "status": "ALL_RESTORED_ROWS_EXPORTED", "source_rows": len(rows),
        "selection_rule": "all H2 minus B3 source rows; no outcome-dependent ROI selection",
        "GT_rule": "source-to-GT nearest current semantic voxel center strictly within 0.05m; diagnostic, not official GT-to-prediction mIoU",
        "choice": selected["config_id"], "prediction_sha256": _sha256(prediction),
        "native_sha256": _sha256(native_path), "source_csv_sha256": _sha256(destination),
        "audit_sha256": _sha256(compact / "apartment_H2_recovered_semantic_attribution.json"),
        "categories": {str(k): int(np.count_nonzero(categories == k)) for k in np.unique(categories)},
        "GT_used_for_prediction_or_selection": False,
    })
    print(len(rows), "restored source rows exported; both method audit counts match", flush=True)


if __name__ == "__main__":
    main()
