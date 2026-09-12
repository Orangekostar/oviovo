"""Separate all-geometry confirmed-free conflicts from role-conditioned Ghost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    _crop_points_to_voxels,
    _voxel_centers,
)
from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.dynamic_metrics import _match_count


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    run = path(config["run_root"])
    room = run / "dev/apartment"
    methods = {
        "native_cached_batch": ["MV_NATIVE_CACHED", "MV_SINGLE", "MV_TOPK_MEAN"],
        "adapter_projected_top4": ["ADAPTER_CLIP_MEAN", "ADAPTER_CLIP_LEARNED"],
        "native_diverse_quality": ["MV_DIVERSE_MEAN", "MV_QUALITY"],
        "graph_mv_quality": ["MV_QUALITY_PATCH_ONLY", "MV_QUALITY_GRAPH_GEOM"],
    }
    # Refuse an audit before every complete current-state prediction exists.
    for state in ("B3", "H2"):
        for directory, names in methods.items():
            for name in names:
                if not (room / directory / f"{state}_{name}.npz").exists():
                    raise RuntimeError(
                        "complete paired predictions are required before evaluator-only geometry audit"
                    )
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz = data["vertices_xyz"]
    with np.load(run / "bridge/evaluation_context.npz") as data:
        changed, free = (
            data["changed_region_voxels"],
            _voxel_centers(data["confirmed_free_voxels"]),
        )
    report = {
        "case": pair["pair_id"],
        "scope": "all current source geometry cropped to existing changed-region voxels, strict 0.05 m confirmed-free match; independent of semantic roles",
        "counter_reuse": "one measured geometry count per state; exact source-row equality proves identical counts for readouts",
        "states": {},
    }
    for state in ("B3", "H2"):
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            canonical, owners = data["source_indices"], data["owner_ids"]
        points = _crop_points_to_voxels([xyz[canonical]], changed)
        conflict = _match_count(points, free, 0.05)
        record = {
            "source_rows": len(canonical),
            "all_geometry_changed_region_rows": len(points),
            "all_geometry_confirmed_free_conflict_rows": int(conflict),
            "readouts": {},
        }
        for directory, names in methods.items():
            for name in names:
                with np.load(room / directory / f"{state}_{name}.npz") as data:
                    if not np.array_equal(
                        data["source_indices"], canonical
                    ) or not np.array_equal(data["owner_ids"], owners):
                        raise ValueError(
                            "readout changed frozen source geometry or owner"
                        )
                    roles = data["eval_role"]
                    role_counts = {
                        str(role): int(np.count_nonzero(roles == role))
                        for role in (0, 1, 2)
                    }
                record["readouts"][name] = {
                    "source_indices_exact_match": True,
                    "owners_exact_match": True,
                    "all_geometry_conflict_count_delta": 0,
                    "eval_role_counts": role_counts,
                }
        report["states"][state] = record
        print(
            state,
            "all-geometry confirmed-free conflicts",
            conflict,
            "of",
            len(points),
            "changed-region rows",
            flush=True,
        )
    _atomic_json(
        path(config["compact_output_root"]) / "apartment_readout_geometry_audit.json",
        report,
    )


if __name__ == "__main__":
    main()
