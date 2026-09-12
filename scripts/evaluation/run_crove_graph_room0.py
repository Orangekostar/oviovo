"""Initial S2 patch-only and geometry-Potts full-map readouts; boundary run pending."""

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
from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    evaluate_replica_voxel_map,
    load_ovimap_semantic_mapping,
    load_replica_ground_truth,
)
from src.oviv2.surface_readout_graph import (
    graph_labels,
    s2_pseudo_unary,
    surface_patches,
)


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    class_ids = np.arange(1, len(classes) + 1)
    root = path(config["run_root"]) / "dev/room0/graph_s2"
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
    with np.load(
        path(config["run_root"]) / "dev/room0/semantic_controls/B_SEM_CROVE_S2.npz"
    ) as d:
        if not np.array_equal(d["source_indices"], source) or not np.array_equal(
            d["owner_ids"], owners
        ):
            raise ValueError("S2 source mismatch")
        ids, conf = d["semantic_ids"], d["semantic_confidence"]
    cache = root / "patch_mapping.npz"
    graph_path = root / "geometry_edges.npz"
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
    unary, valid, tie = s2_pseudo_unary(
        patch["source_patch"], ids, conf, patch["physical_weight"], class_ids
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
    print(stats, flush=True)
    original_supported = np.isin(ids, class_ids) & (conf > 0)
    patch_labels = None
    for name, strength in [("S2_PATCH_ONLY", 0.0), ("S2_GRAPH_GEOM", 0.2)]:
        target = root / f"{name}.npz"
        if target.exists():
            with np.load(target) as d:
                if name == "S2_PATCH_ONLY":
                    patch_labels = d["semantic_ids"]
            continue
        start = time.monotonic()
        node_labels = graph_labels(unary, adjacency, valid, tie, strength=strength)
        source_labels = node_labels[patch["source_patch"]]
        changed = original_supported & (source_labels >= 0)
        result = ids.copy()
        result[changed] = class_ids[source_labels[changed]]
        if name == "S2_PATCH_ONLY":
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
            "semantic_confidence": "inherited S2 scalar; not posterior calibrated confidence",
        }
        with target.open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        print(
            name, {k: measured[k] for k in ["miou", "f_miou", "ap50", "f5"]}, flush=True
        )


if __name__ == "__main__":
    main()
