"""Initial S2 patch-only and geometry-Potts full-map readouts; boundary run pending."""

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

from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_consensus_room0 import load_mask_observations
from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    evaluate_replica_voxel_map,
    load_ovimap_semantic_mapping,
    load_replica_ground_truth,
)
from src.oviv2.surface_mask_evidence import boundary_graph
from src.oviv2.surface_readout_graph import (
    graph_labels,
    posterior_unary,
    s2_pseudo_unary,
    surface_patches,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unary", choices=("s2", "mv_quality"), default="s2")
    parser.add_argument("--boundary", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    class_ids = np.arange(1, len(classes) + 1)
    room = path(config["run_root"]) / "dev/room0"
    shared_graph = room / "graph_s2"
    root = shared_graph if args.unary == "s2" else room / "graph_mv_quality"
    point_graph_root = root
    if args.boundary:
        root = root.with_name(root.name + "_boundary")
    prefix = "S2" if args.unary == "s2" else "MV_QUALITY"
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "case": "room0",
        "patch_size_m": 0.02,
        "patches": "welded compatible source-triangle connected components within cells",
        "unary": "S2_PSEUDO_UNARY_V1",
        "physical_weights": "inverse count of same XYZ/normal-bin/visit duplicates",
        "normal_cosine_min": 0.95,
        "distance_weight_sigma_m": 0.03,
        "normalization": "symmetric_degree",
        "lambda": 0.2,
        "iterations": 5,
        "damping": 0.5,
        "variants": ["S2_PATCH_ONLY", "S2_GRAPH_GEOM"],
        "scope": "initial_geometry_comparison; real boundary and M1 unary still required",
    }
    registry = path(config["compact_output_root"]) / "graph_room0_registry.json"
    if args.unary == "mv_quality":
        settings.update(
            unary="M1_REAL_POSTERIOR_PHYSICAL_MEAN_NEG_LOG",
            variants=[f"{prefix}_PATCH_ONLY", f"{prefix}_GRAPH_GEOM"],
            scope="current room0 DEV M1 leader, not final cross-protocol selection; boundary still required",
            reliability="unit for observed owner; quality already used in M1 aggregation; missing rows preserve M1 fallback",
            shared_patch_mapping="dev/room0/graph_s2/patch_mapping.npz",
        )
        registry = registry.with_name("graph_mv_quality_room0_registry.json")
    if args.boundary:
        settings.update(
            variants=[f"{prefix}_GRAPH_BOUNDARY"],
            scope="same frozen patches and unary, repeated independent mask/depth boundary attenuation",
            boundary_evidence="independent_mask_bank_room0_registry.json",
            minimum_joint_frames=2,
            boundary_factor="1 - 0.8 * disagreeing_views / jointly_observed_views; unobserved unchanged",
            rgb_factor="exp(-mean_channel_squared_difference / 0.25**2); mean real RGB from at least two valid source observations; otherwise factor 1",
        )
        registry = registry.with_name(
            f"graph_{args.unary}_boundary_room0_registry.json"
        )
        status = json.loads(
            (
                path(config["compact_output_root"])
                / "independent_mask_bank_room0_status.json"
            ).read_text()
        )
        if status["completed_frames"] != list(range(0, 2000, 10)):
            raise RuntimeError(
                "boundary graph requires the complete 200-frame independent mask bank"
            )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen graph settings differ")
    else:
        with registry.open("x") as f:
            json.dump(settings, f, indent=2)
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
    ) as d:
        xyz, owners, source = (
            d["vertices_xyz"],
            d["owner_entity_ids"],
            d["source_vertex_indices"],
        )
        normals, triangles = d["normals_xyz"], d["triangles"]
        if not d["current_valid"].all():
            raise ValueError("static current set differs")
    prediction = room / (
        "semantic_controls/B_SEM_CROVE_S2.npz"
        if args.unary == "s2"
        else "native_diverse_quality/MV_QUALITY.npz"
    )
    with np.load(prediction) as d:
        if not np.array_equal(d["source_indices"], source) or not np.array_equal(
            d["owner_ids"], owners
        ):
            raise ValueError("point prediction source mismatch")
        ids, conf = d["semantic_ids"], d["semantic_confidence"]
        if args.unary == "mv_quality":
            feature_owners, posterior = d["feature_owners"], d["owner_posterior"]
            original_supported = d["feature_covered"]
            if not np.array_equal(d["class_ids"], class_ids):
                raise ValueError("M1 class vocabulary differs")
    cache = shared_graph / "patch_mapping.npz"
    graph_path = shared_graph / "geometry_edges.npz"
    if cache.exists() and graph_path.exists():
        with np.load(cache) as d:
            patch = {k: d[k] for k in d.files}
        adjacency = sparse.load_npz(graph_path)
    else:
        print("Building welded source-topology patches", flush=True)
        start = time.monotonic()
        patch = surface_patches(xyz, normals, triangles)
        adjacency = patch.pop("adjacency")
        patch["construction_seconds"] = time.monotonic() - start
        atomic_npz(cache, **patch)
        sparse.save_npz(graph_path, adjacency)
    del normals, triangles
    if args.unary == "s2":
        unary, valid, tie = s2_pseudo_unary(
            patch["source_patch"], ids, conf, patch["physical_weight"], class_ids
        )
        original_supported = np.isin(ids, class_ids) & (conf > 0)
    else:
        unary, valid, tie = posterior_unary(
            patch["source_patch"],
            owners,
            patch["physical_weight"],
            feature_owners,
            posterior,
        )
    stats = {
        "nodes": len(unary),
        "undirected_edges": adjacency.nnz // 2,
        "mean_degree": adjacency.nnz / max(1, len(unary)),
        "physical_samples": int(patch["physical_samples"]),
        "source_rows": len(xyz),
        "backprojection_coverage": 1.0,
        "construction_seconds": float(patch["construction_seconds"]),
    }
    patch_labels = None
    runs = [
        (f"{prefix}_PATCH_ONLY", 0.0),
        (f"{prefix}_GRAPH_GEOM", 0.2),
    ]
    if args.boundary:
        observations, colors, color_valid = load_mask_observations(
            room, list(range(0, 2000, 10)), with_rgb=True
        )
        adjacency, boundary_stats = boundary_graph(
            adjacency, observations, rgb=colors, rgb_valid=color_valid
        )
        stats.update(boundary_stats)
        del observations
        with np.load(point_graph_root / f"{prefix}_PATCH_ONLY.npz") as data:
            patch_labels = data["semantic_ids"]
        runs = [(f"{prefix}_GRAPH_BOUNDARY", 0.2)]
    print(stats, flush=True)
    for name, strength in runs:
        target = root / f"{name}.npz"
        if target.exists():
            with np.load(target) as d:
                if name == f"{prefix}_PATCH_ONLY":
                    patch_labels = d["semantic_ids"]
            continue
        start = time.monotonic()
        node_labels = graph_labels(unary, adjacency, valid, tie, strength=strength)
        source_labels = node_labels[patch["source_patch"]]
        changed = original_supported & (source_labels >= 0)
        result = ids.copy()
        result[changed] = class_ids[source_labels[changed]]
        if name == f"{prefix}_PATCH_ONLY":
            patch_labels = result
        atomic_npz(
            target,
            semantic_ids=result,
            semantic_confidence=conf,
            owner_ids=owners,
            source_indices=source,
            changed_from_point_count=np.count_nonzero(result != ids),
            changed_from_patch_count=np.count_nonzero(result != patch_labels),
            inference_seconds=time.monotonic() - start,
        )
        print(
            name, "saved prediction", int(np.count_nonzero(result != ids)), flush=True
        )
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    vocabulary = {n: i + 1 for i, n in enumerate(classes)}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
        aliases=manifest["aliases"],
    )
    mapping = load_ovimap_semantic_mapping(
        path(case["ovimap_semantic_mapping"]),
        source_vocabulary=tuple(case["ovimap_source_vocabulary"]),
        target_vocabulary=tuple(classes),
    )
    info = [
        EntityEvaluationInfo(int(o), mapping[o][0], 2, mapping[o][1])
        for o in np.unique(owners)
        if mapping.get(o, (0, 0))[0] > 0
    ]
    for name in settings["variants"]:
        target = path(config["compact_output_root"]) / f"room0_{name}.json"
        if target.exists():
            continue
        with np.load(root / f"{name}.npz") as d:
            pred = d["semantic_ids"]
            changes = {
                k: int(d[k])
                for k in ["changed_from_point_count", "changed_from_patch_count"]
            }
            inference_seconds = float(d["inference_seconds"])
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            pred,
            owners,
            conf,
            (owners > 0).astype(np.float32),
        )
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=class_ids,
            instance_semantic_ids={
                i for n, i in vocabulary.items() if n not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=100,
            distance_threshold_m=0.05,
        )
        record = {
            "case": "room0",
            "method": name,
            "status": "FULL_MAP_EVALUATED",
            "geometry_fixed": True,
            "owner_changed": False,
            "metrics": measured,
            "graph": stats,
            "changes": changes,
            "inference_seconds": inference_seconds,
            "semantic_confidence": "inherited point baseline scalar; not graph posterior calibrated confidence",
        }
        with target.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(
            name, {k: measured[k] for k in ["miou", "f_miou", "ap50", "f5"]}, flush=True
        )


if __name__ == "__main__":
    main()
