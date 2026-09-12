"""Frozen Apartment M1-posterior patch/geometry graph controls, both states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_mask_evidence import boundary_graph
from src.oviv2.surface_readout import resolve_semantic_update
from src.oviv2.surface_readout_graph import (
    graph_labels,
    posterior_unary,
    s2_pseudo_unary,
)


def load_boundary_observations(config, pair, state):
    """Read only complete hash-bound frame evidence on the frozen state's nodes."""
    room = path(config["run_root"]) / "dev/apartment"
    compact = path(config["compact_output_root"])
    registry = json.loads(
        (compact / "independent_mask_bank_apartment_registry.json").read_text()
    )
    status = json.loads(
        (compact / "independent_mask_bank_apartment_status.json").read_text()
    )
    frames = [
        frame
        for visit in (0, 1)
        for frame in range(
            pair["visit_frame_ranges"][f"t{visit}"][0],
            pair["visit_frame_ranges"][f"t{visit}"][1] + 1,
        )
    ]
    if any(status.get(s, {}).get("completed_frames") != frames for s in ("B3", "H2")):
        raise RuntimeError("boundary control requires complete paired 512-frame banks")
    topology = room / "graph_topology" / state / "patch_mapping.npz"
    if file_hash(topology) != registry["patch_sha256"][state]:
        raise ValueError("boundary topology binding changed")
    root = room / "independent_mask_bank" / state
    with np.load(root / "representatives.npz") as data:
        visits = data["visit_ids"]
    labels = np.zeros((len(visits), len(frames)), np.uint16, order="F")
    colors = np.zeros((len(visits), 3), np.float64)
    counts = np.zeros(len(visits), np.int32)
    for column, frame_id in enumerate(frames):
        visit = int(frame_id >= pair["visit_frame_ranges"]["t1"][0])
        local_frame = frame_id - pair["visit_frame_ranges"][f"t{visit}"][0]
        native = path(
            f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}/frontend/frame{local_frame:06d}.png"
        )
        with np.load(root / f"{frame_id:06d}.npz") as data:
            nodes, mask_ids = data["node_ids"], data["mask_ids"]
            if (
                len(nodes) != len(np.unique(nodes))
                or np.any(nodes < 0)
                or np.any(nodes >= len(visits))
            ):
                raise ValueError("invalid boundary observation node identity")
            digest = file_hash(native)
            if (
                np.any(visits[nodes] != visit)
                or int(data["frame_id"]) != frame_id
                or int(data["visit_id"]) != visit
                or int(data["local_frame_id"]) != local_frame
                or str(data["mask_sha256"]) != digest
                or digest != registry["mask_sha256"][str(frame_id)]
            ):
                raise ValueError("boundary mask/frame binding changed")
            labels[nodes, column] = mask_ids
            colors[nodes] += data["rgb"] / 255.0
            counts[nodes] += 1
    return labels, colors / np.maximum(counts[:, None], 1), counts >= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--unary", choices=["mv_quality", "s2"], default="mv_quality")
    args = parser.parse_args()
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
    point_graph_output = output
    if args.boundary:
        output = room / "graph_mv_quality_boundary"
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
    if args.boundary:
        variants = [("MV_QUALITY_GRAPH_BOUNDARY", 0.2)]
        settings.update(
            variants=[v[0] for v in variants],
            boundary_evidence="independent_mask_bank_apartment_registry.json",
            minimum_joint_frames=2,
            boundary_factor="1 - 0.8 * disagreeing_views / jointly_observed_views; unobserved unchanged",
            rgb_factor="exp(-mean_channel_squared_difference / 0.25**2); real RGB means from at least two observations; otherwise 1",
        )
        registry = registry.with_name(
            "graph_mv_quality_boundary_apartment_registry.json"
        )
    prefix = "MV_QUALITY"
    point_directory, point_method = "native_diverse_quality", "MV_QUALITY"
    if args.unary == "s2":
        prefix = "S2_DENSE_REPLAY"
        point_directory, point_method = (
            "local_semantic_controls",
            "B_SEM_CROVE_S2_DENSE_REPLAY",
        )
        point_graph_output = room / "graph_s2_dense_replay"
        output = room / (
            "graph_s2_dense_replay_boundary"
            if args.boundary
            else "graph_s2_dense_replay"
        )
        output.mkdir(exist_ok=True)
        variants = [
            (name.replace("MV_QUALITY", prefix), strength)
            for name, strength in variants
        ]
        settings.update(
            variants=[name for name, _ in variants],
            unary="S2_PSEUDO_UNARY_V1",
            point_readout=f"{point_directory}/{point_method}",
            reliability="bounded point confidence only; no second local support factor; known native fallback remains valid",
        )
        registry = registry.with_name(
            f"graph_s2_dense_replay{'_boundary' if args.boundary else ''}_apartment_registry.json"
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
        with np.load(room / point_directory / f"{state}_{point_method}.npz") as data:
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
        if args.unary == "s2":
            unary, valid, tie = s2_pseudo_unary(
                patch,
                baseline["semantic_ids"],
                baseline["semantic_confidence"],
                physical,
                classes,
            )
        else:
            with np.load(
                room
                / "native_diverse_quality"
                / f"{state}_MV_QUALITY_owner_features.npz"
            ) as data:
                if not np.array_equal(classes, data["class_ids"]):
                    raise ValueError("posterior vocabulary differs")
                unary, valid, tie = posterior_unary(
                    patch,
                    owners,
                    physical,
                    data["feature_owners"],
                    data["owner_posterior"],
                )
        patch_labels = None
        boundary_stats = {}
        if args.boundary:
            start_boundary = time.monotonic()
            observations, colors, color_valid = load_boundary_observations(
                config, pair, state
            )
            boundary, boundary_stats = boundary_graph(
                adjacency, observations, rgb=colors, rgb_valid=color_valid
            )
            if boundary.shape != adjacency.shape or boundary.nnz != adjacency.nnz:
                raise ValueError("boundary control changed graph nodes or edges")
            adjacency = boundary
            boundary_stats["boundary_load_and_construction_seconds"] = (
                time.monotonic() - start_boundary
            )
            del observations, colors, color_valid
            with np.load(
                point_graph_output / f"{state}_{prefix}_PATCH_ONLY.npz"
            ) as data:
                if not np.array_equal(data["source_indices"], source):
                    raise ValueError("patch-only control source identity differs")
                patch_labels = data["semantic_ids"]
            print(state, "boundary", boundary_stats, flush=True)
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
            point_supported = (
                np.isin(baseline["semantic_ids"], classes)
                if args.unary == "s2"
                else baseline["feature_covered"]
            )
            covered = point_supported & (labels >= 0)
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
                **boundary_stats,
            }
            _atomic_json(output / f"{state}_{method}_prediction.json", record)
            print(state, method, record, flush=True)
        del baseline, unary, adjacency, patch, physical
    evaluate_saved_readouts(
        config, state_config, pair, crosswalk, output, [v[0] for v in variants]
    )


if __name__ == "__main__":
    main()
