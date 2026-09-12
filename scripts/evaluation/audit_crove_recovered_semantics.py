"""Post-prediction attribution for the exact 2,939 H2-restored source rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_tesse_cd_common_v2 import _voxel_centers
from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json


def histogram(columns, physical_weights, names):
    values, inverse, counts = np.unique(
        np.column_stack(columns), axis=0, return_inverse=True, return_counts=True
    )
    masses = np.bincount(inverse, weights=physical_weights, minlength=len(values))
    return [
        dict(
            zip(names, row.tolist()),
            source_rows=int(count),
            physical_sample_mass=float(mass),
        )
        for row, count, mass in zip(values, counts, masses)
    ]


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    room = run / "dev/apartment"
    methods = {
        "native_cached_batch": ["MV_NATIVE_CACHED", "MV_SINGLE", "MV_TOPK_MEAN"],
        "native_diverse_quality": ["MV_DIVERSE_MEAN", "MV_QUALITY"],
        "adapter_projected_top4": ["ADAPTER_CLIP_MEAN", "ADAPTER_CLIP_LEARNED"],
        "graph_mv_quality": ["MV_QUALITY_PATCH_ONLY", "MV_QUALITY_GRAPH_GEOM"],
        "graph_mv_quality_boundary": ["MV_QUALITY_GRAPH_BOUNDARY"],
        "consensus": ["INST_PAIRWISE", "INST_CONSENSUS_OWNER"],
        "reencoded_regions": [
            "B_SEM_OVI_REENCODE",
            "MV_REENCODE_DIVERSE",
            "MV_REENCODE_QUALITY",
            "INST_CONSENSUS_RESEM",
        ],
    }
    # Diagnostic GT is opened only after both states' complete predictions exist.
    for directory, names in methods.items():
        for name in names:
            for state in ("B3", "H2"):
                if not (room / directory / f"{state}_{name}.npz").exists():
                    raise RuntimeError(
                        "both full-state predictions must precede attribution"
                    )
    with np.load(run / "bridge/B3_legacy_prediction.npz") as data:
        b3 = data["source_indices"]
    with np.load(run / "bridge/H2_legacy_prediction.npz") as data:
        h2 = data["source_indices"]
        recovered = np.setdiff1d(h2, b3, assume_unique=True)
        if len(recovered) != 2939 or len(np.setdiff1d(b3, h2, assume_unique=True)):
            raise ValueError("H2 is not the frozen 2,939-row restoration")
        local = np.searchsorted(h2, recovered)
        original = data["semantic_ids"][local]
        original_roles = data["eval_role"][local]
        original_owners = data["owner_ids"][local]
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        points = data["vertices_xyz"][recovered]
        visits = data["source_visit_ids"][recovered]
    unique, inverse, count = np.unique(
        points, axis=0, return_inverse=True, return_counts=True
    )
    weights = 1.0 / count[inverse]
    with np.load(run / "bridge/evaluation_context.npz") as data:
        semantic = data["current_semantic_voxels"]
        free = _voxel_centers(data["confirmed_free_voxels"])
    distance, nearest = cKDTree(_voxel_centers(semantic[:, :3])).query(points, k=1)
    supported = distance < 0.05
    gt = np.where(supported, semantic[nearest, 3], -1)
    free_distance, _ = cKDTree(free).query(points, k=1)
    conflict = free_distance < 0.05

    def counts(mask):
        return {
            "source_rows": int(mask.sum()),
            "unique_physical_samples": len(np.unique(inverse[mask])),
        }

    report = {
        "case": pair["pair_id"],
        "status": "POST_PREDICTION_DIAGNOSTIC",
        "source_rows": len(recovered),
        "unique_physical_samples": len(unique),
        "unique_5cm_voxels": len(
            np.unique(np.floor(points / 0.05).astype(np.int64), axis=0)
        ),
        "source_visits": np.unique(visits).tolist(),
        "source_owner_counts": histogram([original_owners], weights, ["owner"]),
        "GT_support": counts(supported),
        "confirmed_free": counts(conflict),
        "GT_support_and_confirmed_free": counts(supported & conflict),
        "GT_unmatched": counts(~supported),
        "GT_rule": "nearest current semantic voxel center strictly within 0.05 m; -1 means unmatched. Source-to-GT diagnostic, not the official GT-to-prediction mIoU mapping.",
        "physical_rule": "exact XYZ deduplication. Transition/confusion mass weights each duplicate row by 1/multiplicity; sums equal unique physical samples even if duplicate predictions disagree.",
        "selection_use": "none; this audit cannot choose regions, parameters or predictions",
        "methods": {},
    }
    for directory, names in methods.items():
        for name in names:
            with np.load(room / directory / f"H2_{name}.npz") as data:
                if not np.array_equal(data["source_indices"], h2):
                    raise ValueError("readout source identity changed")
                predicted, roles = data["semantic_ids"][local], data["eval_role"][local]
            report["methods"][name] = {
                "semantic_transitions": histogram(
                    [original, predicted], weights, ["original", "predicted"]
                ),
                "role_transitions": histogram(
                    [original_roles, roles], weights, ["original", "predicted"]
                ),
                "GT_confusion": histogram(
                    [gt, predicted], weights, ["ground_truth", "predicted"]
                ),
                "GT_conditioned_transitions": histogram(
                    [gt, original, predicted],
                    weights,
                    ["ground_truth", "original", "predicted"],
                ),
                "changed_semantics": counts(original != predicted),
                "GT_supported_correct": counts(supported & (gt == predicted)),
                "GT_supported_incorrect": counts(supported & (gt != predicted)),
            }
    _atomic_json(compact / "apartment_H2_recovered_semantic_attribution.json", report)
    print(
        {
            key: report[key]
            for key in [
                "source_rows",
                "unique_physical_samples",
                "GT_support",
                "confirmed_free",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
