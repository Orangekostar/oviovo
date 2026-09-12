"""Frozen Apartment M1-posterior patch/geometry graph controls, both states."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_readout import resolve_semantic_update
from src.oviv2.surface_readout_graph import graph_labels, posterior_unary


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    classes = np.array(
        sorted({v.semantic_id for v in crosswalk.aliases.values() if v.matched}),
        np.int32,
    )
    run = path(config["run_root"])
    room = run / "dev/apartment"
    output = room / "graph_mv_quality"
    output.mkdir(exist_ok=True)
    variants = [("MV_QUALITY_PATCH_ONLY", 0.0), ("MV_QUALITY_GRAPH_GEOM", 0.2)]
    settings = {
        "states": ["B3", "H2"],
        "variants": [v[0] for v in variants],
        "unary": "M1_REAL_POSTERIOR_PHYSICAL_MEAN_NEG_LOG",
        "point_readout": "native_diverse_quality/MV_QUALITY",
        "scope": "eligible DEV candidate control, not final family selection",
        "topology": "graph_topology_apartment_registry.json",
        "normalization": "symmetric_degree",
        "lambda": 0.2,
        "iterations": 5,
        "damping": 0.5,
        "reliability": "unit for observed owner; quality already applied in M1",
        "missing": "retain exact pointwise semantic, confidence and role",
        "confidence": "inherited pointwise confidence, not a new calibrated graph posterior",
        "GT_read_before_all_predictions_saved": False,
    }
    registry = (
        path(config["compact_output_root"]) / "graph_mv_quality_apartment_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen graph settings differ")
    else:
        _atomic_json(registry, settings)
    for state in ("B3", "H2"):
        with np.load(room / "graph_topology" / state / "patch_mapping.npz") as data:
            source, owners = data["source_indices"], data["owner_ids"]
            patch, physical = data["source_patch"], data["physical_weight"]
        adjacency = sparse.load_npz(
            room / "graph_topology" / state / "geometry_edges.npz"
        )
        with np.load(
            room / "native_diverse_quality" / f"{state}_MV_QUALITY.npz"
        ) as data:
            if not np.array_equal(source, data["source_indices"]) or not np.array_equal(
                owners, data["owner_ids"]
            ):
                raise ValueError("pointwise readout differs from frozen topology")
            baseline = {key: data[key] for key in data.files}
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            if not np.array_equal(source, data["source_indices"]) or not np.array_equal(
                owners, data["owner_ids"]
            ):
                raise ValueError("topology differs from original bridge")
            old_ids, old_roles = data["semantic_ids"], data["eval_role"]
        with np.load(
            room / "native_diverse_quality" / f"{state}_MV_QUALITY_owner_features.npz"
        ) as data:
            if not np.array_equal(classes, data["class_ids"]):
                raise ValueError("posterior vocabulary differs")
            unary, valid, tie = posterior_unary(
                patch, owners, physical, data["feature_owners"], data["owner_posterior"]
            )
        patch_labels = None
        for method, strength in variants:
            target = output / f"{state}_{method}.npz"
            if target.exists():
                if strength == 0:
                    with np.load(target) as data:
                        patch_labels = data["semantic_ids"]
                continue
            started = time.monotonic()
            labels = graph_labels(unary, adjacency, valid, tie, strength=strength)
            labels = labels[patch]
            covered = baseline["feature_covered"] & (labels >= 0)
            proposed = baseline["semantic_ids"].copy()
            proposed[covered] = classes[labels[covered]]
            ids, roles = resolve_semantic_update(
                baseline["semantic_ids"],
                baseline["eval_role"],
                proposed,
                covered.astype(np.uint8),
                crosswalk,
            )
            if strength == 0:
                patch_labels = ids
            prediction = dict(baseline)
            prediction.update(semantic_ids=ids, eval_role=roles)
            atomic_npz(target, **prediction)
            record = {
                "state": state,
                "method": method,
                "source_rows": len(source),
                "feature_coverage": float(covered.mean()),
                "changed_semantic_rows": int(np.count_nonzero(ids != old_ids)),
                "changed_role_rows": int(np.count_nonzero(roles != old_roles)),
                "changed_from_point_count": int(
                    np.count_nonzero(ids != baseline["semantic_ids"])
                ),
                "changed_from_patch_count": int(np.count_nonzero(ids != patch_labels)),
                "prediction_and_write_seconds": time.monotonic() - started,
                "geometry_and_owner_fixed": True,
                "nodes": len(unary),
                "undirected_edges": adjacency.nnz // 2,
                "backprojection_coverage": 1.0,
            }
            _atomic_json(output / f"{state}_{method}_prediction.json", record)
            print(state, method, record, flush=True)
        del baseline, unary, adjacency, patch, physical
    evaluate_saved_readouts(
        config, state_config, pair, crosswalk, output, [v[0] for v in variants]
    )


if __name__ == "__main__":
    main()
