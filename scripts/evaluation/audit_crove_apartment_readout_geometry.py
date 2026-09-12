"""Separate all-geometry confirmed-free conflicts from role-conditioned Ghost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    _crop_points_to_voxels,
    _voxel_centers,
)
from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.dynamic_metrics import _match_count


def full_geometry_counts(points, free, legacy_sources):
    """Count all current geometry independently of newly predicted labels."""
    points, free = np.asarray(points), np.asarray(free)
    sources = np.asarray(legacy_sources)
    if points.ndim != 2 or points.shape[1] != 3 or sources.shape != (len(points),):
        raise ValueError("aligned current geometry and legacy provenance required")
    keys = np.floor(points.astype(np.float64) / 0.05).astype(np.int64)
    _, voxel = np.unique(keys, axis=0, return_inverse=True)
    hit = np.zeros(len(points), bool)
    if len(free):
        tree = cKDTree(free)
        for start in range(0, len(points), 1_000_000):
            distances, _ = tree.query(points[start : start + 1_000_000], workers=8)
            hit[start : start + 1_000_000] = distances < 0.05

    def subset(mask):
        return {
            "source_rows": int(mask.sum()),
            "confirmed_free_conflict_rows": int((mask & hit).sum()),
            "unique_5cm_voxels": len(np.unique(voxel[mask])),
            "conflicting_5cm_voxels": len(np.unique(voxel[mask & hit])),
        }

    result = subset(np.ones(len(points), bool))
    result["fixed_legacy_sources"] = {
        str(kind): subset(sources == kind) for kind in (0, 1, 2, 3)
    }
    return result


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
        "graph_mv_quality_boundary": ["MV_QUALITY_GRAPH_BOUNDARY"],
        "graph_mv_topk": ["MV_TOPK_PATCH_ONLY", "MV_TOPK_GRAPH_GEOM"],
        "graph_mv_topk_boundary": ["MV_TOPK_GRAPH_BOUNDARY"],
        "local_background": ["MV_BG_OWNER_QUALITY", "MV_LOCAL_BG_QUALITY"],
        "consensus": ["INST_PAIRWISE", "INST_CONSENSUS_OWNER"],
        "current_aware": ["MV_CURRENT_AWARE"],
        "graph_s2_dense_replay": [
            "S2_DENSE_REPLAY_PATCH_ONLY",
            "S2_DENSE_REPLAY_GRAPH_GEOM",
        ],
        "graph_s2_dense_replay_boundary": ["S2_DENSE_REPLAY_GRAPH_BOUNDARY"],
        "graph_adapter_learned": [
            "ADAPTER_LEARNED_PATCH_ONLY",
            "ADAPTER_LEARNED_GRAPH_GEOM",
        ],
        "local_semantic_controls": [
            "B_SEM_CROVE_S0_DENSE_REPLAY",
            "B_SEM_CROVE_S2_DENSE_REPLAY",
        ],
        "reencoded_regions": [
            "B_SEM_OVI_REENCODE",
            "MV_REENCODE_DIVERSE",
            "MV_REENCODE_QUALITY",
            "INST_CONSENSUS_RESEM",
        ],
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
        "full_current_scope": "additional uncropped all-current strict 0.05 m free-conflict counts and floor(XYZ/0.05) physical voxels; fixed original source strata 0 explicit background, 1 stuff, 2 thing, 3 unknown entity",
        "states": {},
    }
    for state in ("B3", "H2"):
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            canonical, owners = data["source_indices"], data["owner_ids"]
            old_ids, old_roles, provenance = (
                data["semantic_ids"],
                data["eval_role"],
                data["legacy_eval_source"],
            )
        points = _crop_points_to_voxels([xyz[canonical]], changed)
        conflict = _match_count(points, free, 0.05)
        record = {
            "source_rows": len(canonical),
            "all_geometry_changed_region_rows": len(points),
            "all_geometry_confirmed_free_conflict_rows": int(conflict),
            "full_current_geometry": full_geometry_counts(
                xyz[canonical], free, provenance
            ),
            "readouts": {},
        }
        for directory, names in methods.items():
            for name in names:
                with np.load(room / directory / f"{state}_{name}.npz") as data:
                    owners_equal = np.array_equal(data["owner_ids"], owners)
                    if not np.array_equal(data["source_indices"], canonical) or (
                        directory != "consensus"
                        and name != "INST_CONSENSUS_RESEM"
                        and not owners_equal
                    ):
                        raise ValueError(
                            "readout changed frozen source geometry or owner"
                        )
                    roles = data["eval_role"]
                    ids = data["semantic_ids"]
                    if name == "INST_CONSENSUS_RESEM":
                        with np.load(
                            room / "consensus" / f"{state}_INST_CONSENSUS_OWNER.npz"
                        ) as owner_data:
                            if not np.array_equal(
                                data["owner_ids"], owner_data["owner_ids"]
                            ):
                                raise ValueError(
                                    "semantic re-estimation changed consensus owners"
                                )
                    if directory == "consensus" and (
                        not np.array_equal(ids, old_ids)
                        or not np.array_equal(roles, old_roles)
                    ):
                        raise ValueError(
                            "owner-only readout changed semantics or roles"
                        )
                    role_counts = {
                        str(role): int(np.count_nonzero(roles == role))
                        for role in (0, 1, 2)
                    }
                    object_changed = _crop_points_to_voxels(
                        [xyz[canonical[roles != 0]]], changed
                    )
                    ghost_numerator = int(_match_count(object_changed, free, 0.05))
                    ghost_denominator = len(object_changed)
                    ghost_rate = (
                        ghost_numerator / ghost_denominator
                        if ghost_denominator
                        else 0.0
                    )
                    scored = json.loads(
                        (
                            path(config["compact_output_root"])
                            / f"apartment_{state}_{name}.json"
                        ).read_text()
                    )
                    if not np.isclose(
                        ghost_rate, scored["metrics"]["ghost"], rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            "role-conditioned Ghost counts disagree with scored readout"
                        )
                record["readouts"][name] = {
                    "source_indices_exact_match": True,
                    "owners_exact_match": owners_equal,
                    "all_geometry_conflict_count_delta": 0,
                    "role_conditioned_ghost": {
                        "numerator": ghost_numerator,
                        "denominator": ghost_denominator,
                        "rate": ghost_rate,
                        "scope": "roles != BACKGROUND, existing changed-region crop, strict 0.05 m confirmed-free match",
                        "matches_scored_rate": True,
                    },
                    "eval_role_counts": role_counts,
                    "object_to_background_rows": int(
                        np.count_nonzero((old_roles != 0) & (roles == 0))
                    ),
                    "background_to_object_rows": int(
                        np.count_nonzero((old_roles == 0) & (roles != 0))
                    ),
                    "known_to_unknown_rows": int(
                        np.count_nonzero((old_ids > 0) & (ids == 0))
                    ),
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
