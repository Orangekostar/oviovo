"""Pairwise/consensus owner-only readouts on a frozen native surface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_fine_current_map import (
    NON_INSTANCE_CLASSES,
    EntityEvaluationInfo,
    LabeledMesh,
    evaluate_replica_voxel_map,
    load_replica_ground_truth,
)
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.oviv2.surface_mask_consensus import (
    apply_owner_proposals,
    arbitrate_candidates,
    cluster_masks,
    construct_mask_graph,
)


def load_mask_observations(
    room, frames, *, with_rgb=False, frontend_subdir="cropformer_cpu/frontend"
):
    with np.load(room / "independent_mask_bank/representatives.npz") as data:
        count = len(data["source_rows"])
    labels = np.zeros((count, len(frames)), np.uint16)
    color_sum = np.zeros((count, 3), np.float64) if with_rgb else None
    color_count = np.zeros(count, np.int32) if with_rgb else None
    for column, frame in enumerate(frames):
        target = room / "independent_mask_bank" / f"{frame:06d}.npz"
        with np.load(target) as data:
            nodes, mask_ids = data["node_ids"], data["mask_ids"]
            if (
                len(nodes) != len(np.unique(nodes))
                or np.any(nodes < 0)
                or np.any(nodes >= count)
            ):
                raise ValueError(
                    "mask bank contains duplicate or out-of-range patch nodes"
                )
            if int(data["frame_id"]) != frame or int(data["visit_id"]) != 0:
                raise ValueError("independent observation frame binding differs")
            mask_file = room / frontend_subdir / f"frame{frame:06d}.png"
            if file_hash(mask_file) != str(data["mask_sha256"]):
                raise ValueError("original independent mask artifact changed")
            labels[nodes, column] = mask_ids
            if with_rgb:
                color_sum[nodes] += data["rgb"] / 255.0
                color_count[nodes] += 1
    if with_rgb:
        return labels, color_sum / np.maximum(color_count[:, None], 1), color_count >= 2
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-available",
        action="store_true",
        help="No GT or benchmark scores; exercise available frames only in a separate diagnostic directory",
    )
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    case = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    compact = path(config["compact_output_root"])
    room = path(config["run_root"]) / "dev/room0"
    bank_status = json.loads(
        (compact / "independent_mask_bank_room0_status.json").read_text()
    )
    frames = bank_status["completed_frames"]
    if not args.diagnostic_available and frames != list(range(0, 2000, 10)):
        raise RuntimeError(
            "M3 full-map protocol requires all 200 independent frames; input preparation is incomplete"
        )
    if len(frames) != len(set(frames)) or any(
        f < 0 or f >= 2000 or f % 10 for f in frames
    ):
        raise ValueError("duplicate or unauthorized independent frames")
    methods = {"INST_PAIRWISE": "pairwise", "INST_CONSENSUS_OWNER": "consensus"}
    output = room / (
        f"consensus_diagnostic_{len(frames):04d}"
        if args.diagnostic_available
        else "consensus"
    )
    output.mkdir(exist_ok=True)
    settings = {
        "case": "room0",
        "original_source_commit": "eb2d41c2267cde21966f4340c37b8cf94b05c23c",
        "implementation": "CROVE_LOCAL_CONSENSUS fixed-patch adaptation",
        "source_modules": ["graph/construction.py", "graph/iterative_clustering.py"],
        "mask_visible_fraction": 0.3,
        "contained_fraction": 0.8,
        "undersegment_split_fraction": 0.3,
        "minimum_cloud_patches": 8,
        "pairwise_overlap_over_smaller_cloud": 0.5,
        "consensus_rate": 0.9,
        "minimum_joint_views": 2,
        "maximum_iterations": 3,
        "readout": config["initial_method_parameters"]["consensus"],
        "adaptation": "one physical representative per shared local patch; per-frame boundary exclusion; no 500-point visibility override; fixed three iterations instead of percentile schedule; per-frame union support; same undersegment filter and readout for both controls",
        "ROI": "supported candidate spans multiple source parents or one parent has multiple supported candidates; otherwise keep owner",
        "new_IDs": "ascending candidate order above maximum original owner; original residual IDs preserved",
        "semantics": "frozen B_SEM_OVI_NATIVE point semantics and confidence",
        "methods": list(methods),
        "resem_status": "PENDING_NEW_FEATURE_EXTRACTION",
    }
    registry = compact / "consensus_room0_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen consensus configuration changed")
    else:
        _atomic_json(registry, settings)
    labels = load_mask_observations(room, frames)
    started = time.monotonic()
    graph = construct_mask_graph(labels)
    construction_seconds = time.monotonic() - started
    print(
        "mask graph",
        len(graph["mask_keys"]),
        "masks;",
        int(graph["undersegmented"].sum()),
        "filtered",
        flush=True,
    )
    with np.load(room / "graph_s2/patch_mapping.npz") as data:
        source_patch = data["source_patch"]
    with np.load(room / "native_cached_batch/B_SEM_OVI_NATIVE.npz") as data:
        source, owners = data["source_indices"], data["owner_ids"]
        semantics, confidence = data["semantic_ids"], data["semantic_confidence"]
    if len(source_patch) != len(owners):
        raise ValueError("source inverse does not match native rows")
    for method, mode in methods.items():
        target = output / f"{method}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        assignment, history = cluster_masks(graph, mode=mode)
        votes = np.full(labels.shape, -1, np.int32)
        for frame in range(len(frames)):
            lookup = np.full(int(labels[:, frame].max()) + 1, -1, np.int32)
            for mask_index, (mask_frame, mask_id) in enumerate(graph["mask_keys"]):
                if mask_frame == frame:
                    lookup[mask_id] = assignment[mask_index]
            votes[:, frame] = lookup[labels[:, frame]]
        proposed = arbitrate_candidates(votes)
        result, lineage = apply_owner_proposals(owners, source_patch, proposed)
        accepted = proposed >= 0
        atomic_npz(
            target,
            source_indices=source,
            owner_ids=result,
            semantic_ids=semantics,
            semantic_confidence=confidence,
            patch_candidate=proposed,
        )
        parent_children = {}
        for entry in lineage:
            for parent in entry["parents"]:
                parent_children[parent] = parent_children.get(parent, 0) + 1
        merge_parents = {
            parent
            for entry in lineage
            if len(entry["parents"]) > 1
            for parent in entry["parents"]
        }
        remaining_parent_rows = (result == owners) & np.isin(
            owners, list(parent_children)
        )
        record = {
            "method": method,
            "frames": frames,
            "GT_read": False,
            "status": "DIAGNOSTIC_PARTIAL_INPUT_NO_BENCHMARK_SCORE"
            if args.diagnostic_available
            else "FULL_MAP_PREDICTION_SAVED",
            "mask_nodes": len(graph["mask_keys"]),
            "filtered_mask_nodes": int(graph["undersegmented"].sum()),
            "construction_seconds_shared": construction_seconds,
            "inference_and_write_seconds": time.monotonic() - started,
            "cluster_history": history,
            "lineage": lineage,
            "new_ID_count": len(lineage),
            "merge_candidate_count": sum(len(e["parents"]) > 1 for e in lineage),
            "merge_parent_count": len(merge_parents),
            "split_parent_count": sum(count > 1 for count in parent_children.values()),
            "residual_parent_ID_count": len(np.unique(owners[remaining_parent_rows])),
            "residual_parent_source_rows": int(remaining_parent_rows.sum()),
            "changed_source_rows": int(np.count_nonzero(result != owners)),
            "unsupported_or_ambiguous_patch_count": int((~accepted).sum()),
            "unsupported_or_ambiguous_source_rows": int(
                (~accepted[source_patch]).sum()
            ),
            "source_rows": len(source),
            "backprojection_coverage": 1.0,
            "source_semantics_unchanged": True,
        }
        _atomic_json(output / f"{method}_actions.json", record)
        print(
            method,
            record["changed_source_rows"],
            "changed source rows;",
            len(lineage),
            "new IDs",
            flush=True,
        )
    if args.diagnostic_available:
        print(
            "Diagnostic only: no GT opened and no screening metrics produced",
            flush=True,
        )
        return
    # The complete pair of predictions is saved before any GT is opened.
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
    ) as data:
        xyz = data["vertices_xyz"]
        if (
            not np.array_equal(data["source_vertex_indices"], source)
            or not np.array_equal(data["owner_entity_ids"], owners)
            or not data["current_valid"].all()
        ):
            raise ValueError("native source geometry/state differs")
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    vocabulary = {c: i + 1 for i, c in enumerate(manifest["vocabulary"]["classes"])}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
        aliases=manifest["aliases"],
    )
    baseline = json.loads((compact / "room0_B_SEM_OVI_NATIVE.json").read_text())[
        "metrics"
    ]
    for method in methods:
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        with np.load(output / f"{method}.npz") as data:
            result = data["owner_ids"]
            if not np.array_equal(
                data["semantic_ids"], semantics
            ) or not np.array_equal(data["source_indices"], source):
                raise ValueError(
                    "owner-only readout altered source semantics or row mapping"
                )
        info = []
        for owner in np.unique(result):
            owner_rows = result == owner
            values = semantics[owner_rows]
            values = values[values > 0]
            if owner > 0 and len(values):
                info.append(
                    EntityEvaluationInfo(
                        int(owner),
                        int(np.bincount(values).argmax()),
                        2,
                        float(confidence[owner_rows].mean()),
                    )
                )
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            semantics,
            result,
            confidence,
            (owners > 0).astype(np.float32),
        )
        measured = evaluate_replica_voxel_map(
            mesh,
            truth,
            info,
            valid_semantic_ids=vocabulary.values(),
            instance_semantic_ids={
                i for c, i in vocabulary.items() if c not in NON_INSTANCE_CLASSES
            },
            min_instance_vertices=100,
            distance_threshold_m=0.05,
        )
        for metric in ("miou", "f_miou", "f5"):
            if abs(measured[metric] - baseline[metric]) > 1e-12:
                raise ValueError("owner-only change altered semantic/geometry score")
        record = json.loads((output / f"{method}_actions.json").read_text())
        record.update(
            status="FULL_MAP_EVALUATED",
            GT_read=True,
            metrics=measured,
            instance_ranking="existing class-agnostic size-based protocol for all methods; no new confidence ranking",
            semantic_constrained_AP="diagnostic only: majority frozen source semantics, mean inherited point confidence and common two-view gate",
        )
        _atomic_json(target, record)
        print(
            method, {k: measured[k] for k in ("miou", "ap25", "ap50", "f5")}, flush=True
        )


if __name__ == "__main__":
    main()
