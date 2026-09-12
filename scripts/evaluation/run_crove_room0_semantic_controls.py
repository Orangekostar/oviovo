"""Re-evaluate saved S0/S2 on the identical canonical room0 surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    classes = manifest["vocabulary"]["classes"]
    root = path(config["run_root"]) / "dev/room0/semantic_controls"
    root.mkdir(parents=True, exist_ok=True)
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
        if not d["current_valid"].all() or not np.array_equal(
            source, np.arange(len(xyz))
        ):
            raise ValueError("canonical full static source order changed")
    names = {
        "B_SEM_CROVE_S0": "s0_nearest",
        "B_SEM_CROVE_S2": "s2_reliable_surface_owner_fallback",
    }
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static/semantic_transfer_cache.npz"
        )
    ) as d:
        if not np.array_equal(d["owner_entity_ids"], owners):
            raise ValueError("cached S0/S2 ownership differs from canonical surface")
        for method, prefix in names.items():
            target = root / f"{method}.npz"
            if not target.exists():
                atomic_npz(
                    target,
                    semantic_ids=d[f"{prefix}_semantic_ids"],
                    semantic_confidence=d[f"{prefix}_semantic_confidences"],
                    owner_ids=owners,
                    source_indices=source,
                    support_reliability=d[f"{prefix}_support_reliabilities"],
                )
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    class_ids = {n: i + 1 for i, n in enumerate(classes)}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=class_ids,
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
    for method in names:
        target = path(config["compact_output_root"]) / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(root / f"{method}.npz") as d:
            mesh = LabeledMesh(
                xyz,
                np.empty((0, 3), np.int64),
                np.zeros_like(xyz),
                d["semantic_ids"],
                owners,
                d["semantic_confidence"],
                (owners > 0).astype(np.float32),
            )
        metrics = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=class_ids.values(),
            instance_semantic_ids={
                i for n, i in class_ids.items() if n not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=100,
            distance_threshold_m=0.05,
        )
        with target.open("x") as f:
            json.dump(
                {
                    "method": method,
                    "case": "room0",
                    "status": "FULL_MAP_EVALUATED",
                    "metrics": metrics,
                    "geometry_fixed": True,
                    "owner_changed": False,
                    "semantic_origin": "frozen_existing_transfer_cache",
                },
                f,
                indent=2,
                allow_nan=False,
            )
        print(
            method,
            {k: metrics[k] for k in ["miou", "f_miou", "ap50", "f5"]},
            flush=True,
        )


if __name__ == "__main__":
    main()
