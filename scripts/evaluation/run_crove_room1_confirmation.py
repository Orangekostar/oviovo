"""Run the selected static confirmation matrix without retuning."""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
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
    _owner_row_groups,
    evaluate_replica_voxel_map,
    load_replica_ground_truth,
)
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256
from src.oviv2.surface_mask_consensus import (
    apply_owner_proposals,
    arbitrate_candidates,
    cluster_masks,
    construct_mask_graph,
)
from src.oviv2.surface_mask_evidence import boundary_graph
from src.oviv2.surface_multiview_semantics import aggregate_views, diverse_views
from src.oviv2.surface_readout_graph import (
    graph_labels,
    posterior_unary,
    s2_pseudo_unary,
)


def validate_selection(selection, policy):
    confirmation = selection["static_confirmation"]
    if (
        selection["status"] != "DEV_SELECTION_FROZEN"
        or selection["selection_policy"] != policy
    ):
        raise ValueError("confirmation requires the unchanged frozen DEV policy")
    if confirmation["scene"] != "room1" or confirmation["retuning_allowed"]:
        raise ValueError("confirmation scene or no-retuning rule changed")
    entries = [
        e
        for e in selection["task_selections"]
        if e["task"].startswith("STATIC_") and e["family"] in ("M1", "M2", "M3", "M4")
    ]
    if len(entries) != 4 or {e["family"] for e in entries} != {"M1", "M2", "M3", "M4"}:
        raise ValueError("all four static family selections are required")
    methods = confirmation["methods"]
    if len(methods) != len(set(methods)):
        raise ValueError("duplicate confirmation method")
    for entry in entries:
        for key in ("metric_winner", "best_new_candidate"):
            row = entry[key]
            if row is None or row["exact_implementation"] not in methods:
                raise ValueError("selected family winner or new representative missing")
            if row["direct_control"] and row["direct_control"] not in methods:
                raise ValueError("selected representative direct control missing")
    return methods


def validate_scored_invariants(records):
    baseline = records["B_SEM_OVI_NATIVE"]["metrics"]
    checks = {}
    for method, record in records.items():
        fields = ["f5"] + (
            ["miou", "f_miou"] if method.startswith("INST_") else ["ap25", "ap50"]
        )
        for field in fields:
            if not np.isclose(
                record["metrics"][field], baseline[field], rtol=0, atol=1e-12
            ):
                raise ValueError(f"confirmation invariant changed: {method}/{field}")
        checks[method] = {field: True for field in fields}
    return checks


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    compact, run = path(config["compact_output_root"]), path(config["run_root"])
    selection_file = compact / "selected_configs.json"
    if not selection_file.exists():
        raise RuntimeError(
            "DEV selected_configs.json must exist before confirmation execution"
        )
    selection = json.loads(selection_file.read_text())
    methods = validate_selection(selection, config["selection_policy"])
    if (
        _sha256(compact / selection["source_summary_file"])
        != selection["source_summary_sha256"]
    ):
        raise ValueError("frozen DEV result evidence changed")
    for name, record in selection["method_registries"].items():
        if _sha256(compact / name) != record["sha256"]:
            raise ValueError(f"frozen method registry changed: {name}")
    supported = {
        "B_SEM_OVI_NATIVE",
        "B_SEM_CROVE_S0",
        "B_SEM_CROVE_S2",
        "MV_NATIVE_CACHED",
        "MV_SINGLE",
        "MV_TOPK_MEAN",
        "MV_DIVERSE_MEAN",
        "MV_QUALITY",
        "ADAPTER_CLIP_MEAN",
        "ADAPTER_CLIP_LEARNED",
        "INST_PAIRWISE",
        "INST_CONSENSUS_OWNER",
        *[
            f"{p}_{v}"
            for p in ("S2", "MV_QUALITY", "ADAPTER_LEARNED")
            for v in ("PATCH_ONLY", "GRAPH_GEOM", "GRAPH_BOUNDARY")
        ],
    }
    if set(methods) - supported:
        raise RuntimeError(
            f"selected confirmation heads need an exact input implementation: {sorted(set(methods) - supported)}"
        )
    room = run / "confirm/room1"
    output = room / "selected_readouts"
    fine = json.loads(path(config["source_config"]).read_text())["cases"][
        "replica_room0_static"
    ]
    manifest = json.loads(path(fine["benchmark_manifest"]).read_text())
    vocabulary = {
        name: i + 1 for i, name in enumerate(manifest["vocabulary"]["classes"])
    }
    class_ids = np.array(list(vocabulary.values()))
    geometry_file = room / "inputs/current_surface.npz"
    binding = {
        "selection_sha256": _sha256(selection_file),
        "geometry_sha256": _sha256(geometry_file),
        "methods": methods,
        "scene": "room1",
        "GT_before_all_selected_predictions": False,
    }
    registry = compact / "room1_confirmation_binding.json"
    if registry.exists() and json.loads(registry.read_text()) != binding:
        raise ValueError("confirmation inputs or selected methods changed")
    _atomic_json(registry, binding)
    output.mkdir(exist_ok=True)
    with np.load(geometry_file) as data:
        xyz, source, native_owners = (
            data["vertices_xyz"],
            data["source_vertex_indices"],
            data["owner_entity_ids"],
        )
        if not data["current_valid"].all() or np.any(data["source_visit_ids"] != 0):
            raise ValueError("room1 current geometry differs")
    with np.load(room / "native_cached_batch/B_SEM_OVI_NATIVE.npz") as data:
        if not np.array_equal(source, data["source_indices"]) or not np.array_equal(
            native_owners, data["owner_ids"]
        ):
            raise ValueError("native source binding differs")
        native_ids, native_conf = data["semantic_ids"], data["semantic_confidence"]
    groups = _owner_row_groups(native_owners)
    mapping_manifest = json.loads(
        (room / "native_cpu/native_mapping_manifest.json").read_text()
    )
    with Path(mapping_manifest["artifacts"]["semantic_features"]["path"]).open(
        "rb"
    ) as handle:
        bank = pickle.load(handle)
    with np.load(room / "native_cached_batch/text_features.npz") as data:
        text, scale = data["text"], float(data["logit_scale"])
    with np.load(room / "graph_s2/patch_mapping.npz") as data:
        patch = {k: data[k] for k in data.files}
    if not np.array_equal(patch["source_indices"], source) or not np.array_equal(
        patch["owner_ids"], native_owners
    ):
        raise ValueError("confirmation topology source/owner binding differs")
    adjacency = sparse.load_npz(room / "graph_s2/geometry_edges.npz")
    point_cache = {}

    def point_head(method):
        if method in point_cache:
            return point_cache[method]
        if method.startswith("B_SEM_"):
            directory = (
                "native_cached_batch"
                if method == "B_SEM_OVI_NATIVE"
                else "semantic_controls"
            )
            with np.load(room / directory / f"{method}.npz") as data:
                if not np.array_equal(
                    source, data["source_indices"]
                ) or not np.array_equal(native_owners, data["owner_ids"]):
                    raise ValueError("semantic control source changed")
                result = (
                    data["semantic_ids"],
                    data["semantic_confidence"],
                    None,
                    None,
                    None,
                )
        else:
            features = {}
            selected_text, selected_scale = text, scale
            if method.startswith("ADAPTER_"):
                directory = room / "adapter_projected_top4"
                status = json.loads(
                    (compact / "room1_adapter_feature_preparation.json").read_text()
                )
                with np.load(directory / "text_features.npz") as data:
                    selected_text, selected_scale = (
                        data["text"],
                        float(data["logit_scale"]),
                    )
                    if not np.array_equal(data["class_ids"], class_ids):
                        raise ValueError("adapter text vocabulary differs")
                kind = "mean" if method == "ADAPTER_CLIP_MEAN" else "learned"
                observations = defaultdict(list)
                for frame in status["frames"]:
                    with np.load(directory / "view_bank" / f"{frame:06d}.npz") as data:
                        for owner, z in zip(
                            data["feature_owner_ids"], data[f"{kind}_embeddings"]
                        ):
                            observations[int(owner)].append(z)
                for owner, zs in observations.items():
                    z = np.mean(zs, axis=0)
                    features[owner] = z / max(np.linalg.norm(z), 1e-12)
            else:
                quality = json.loads(
                    (room / "native_diverse_quality/geometric_quality.json").read_text()
                )["evidence"]
                for owner, entry in sorted(bank.items()):
                    if owner not in groups or len(entry["frame_id"]) < 2:
                        continue
                    areas, zs = np.asarray(entry["vis_area"]), np.asarray(entry["feat"])
                    if method == "MV_NATIVE_CACHED":
                        indices = np.arange(max(0, len(areas) - 8), len(areas))
                        if areas[indices].sum() <= 0:
                            continue
                        z = np.average(zs[indices], axis=0, weights=areas[indices])
                        z /= max(np.linalg.norm(z), 1e-12)
                    else:
                        if method in ("MV_SINGLE", "MV_TOPK_MEAN"):
                            indices = np.argsort(-areas, kind="stable")[
                                : 1 if method == "MV_SINGLE" else 4
                            ]
                        else:
                            center = np.unique(xyz[groups[owner]], axis=0).mean(
                                axis=0, dtype=np.float64
                            )
                            directions = np.asarray(entry["pose"])[:, :3, 3] - center
                            indices = diverse_views(directions, areas, k=4)
                        weights = (
                            np.array(
                                [quality[f"{owner}:{i}"]["weight"] for i in indices]
                            )
                            if method == "MV_QUALITY"
                            else np.ones(len(indices))
                        )
                        z = aggregate_views(zs[indices], weights)
                    if z is not None:
                        features[int(owner)] = z
            ids, conf = native_ids.copy(), native_conf.copy()
            covered = np.zeros(len(source), bool)
            feature_owners, posterior = [], []
            for owner, z in sorted(features.items()):
                logits = selected_scale * (z @ selected_text.T)
                p = np.exp(logits - logits.max())
                p /= p.sum()
                ids[groups[owner]], conf[groups[owner]] = class_ids[p.argmax()], p.max()
                covered[groups[owner]] = True
                feature_owners.append(owner)
                posterior.append(p)
            result = (
                ids,
                conf,
                covered,
                np.array(feature_owners),
                np.asarray(posterior).reshape(-1, len(class_ids)),
            )
        point_cache[method] = result
        return result

    observations = colors = color_valid = mask_graph = None
    for method in methods:
        target = output / f"{method}.npz"
        if target.exists():
            continue
        started = time.monotonic()
        owners = native_owners
        details = {}
        if observations is None and (
            method.startswith("INST_") or method.endswith("GRAPH_BOUNDARY")
        ):
            status = json.loads(
                (compact / "independent_mask_bank_room1_status.json").read_text()
            )
            if status["completed_frames"] != list(range(0, 2000, 10)):
                raise RuntimeError("complete independent room1 mask bank required")
            observations, colors, color_valid = load_mask_observations(
                room,
                status["completed_frames"],
                with_rgb=True,
                frontend_subdir="native_cpu/frontend",
            )
        if method.startswith("INST_"):
            if mask_graph is None:
                mask_graph = construct_mask_graph(observations)
            assignments, history = cluster_masks(
                mask_graph,
                mode="pairwise" if method == "INST_PAIRWISE" else "consensus",
            )
            votes = np.full(observations.shape, -1, np.int32)
            for frame in range(observations.shape[1]):
                lookup = np.full(int(observations[:, frame].max()) + 1, -1, np.int32)
                for index, (column, mask) in enumerate(mask_graph["mask_keys"]):
                    if column == frame:
                        lookup[mask] = assignments[index]
                votes[:, frame] = lookup[observations[:, frame]]
            proposed = arbitrate_candidates(votes)
            owners, lineage = apply_owner_proposals(
                native_owners, patch["source_patch"], proposed
            )
            ids, conf = native_ids, native_conf
            details.update(
                lineage=lineage, cluster_history=history, new_ID_count=len(lineage)
            )
        elif method.endswith(("PATCH_ONLY", "GRAPH_GEOM", "GRAPH_BOUNDARY")):
            head = (
                "B_SEM_CROVE_S2"
                if method.startswith("S2_")
                else "MV_QUALITY"
                if method.startswith("MV_QUALITY_")
                else "ADAPTER_CLIP_LEARNED"
            )
            ids, conf, covered, feature_owners, posterior = point_head(head)
            if head == "B_SEM_CROVE_S2":
                unary, valid, tie = s2_pseudo_unary(
                    patch["source_patch"],
                    ids,
                    conf,
                    patch["physical_weight"],
                    class_ids,
                )
                covered = np.isin(ids, class_ids) & (conf > 0)
            else:
                unary, valid, tie = posterior_unary(
                    patch["source_patch"],
                    native_owners,
                    patch["physical_weight"],
                    feature_owners,
                    posterior,
                )
            graph = adjacency
            if method.endswith("GRAPH_BOUNDARY"):
                graph, details = boundary_graph(
                    adjacency, observations, rgb=colors, rgb_valid=color_valid
                )
            labels = graph_labels(
                unary,
                graph,
                valid,
                tie,
                strength=0.0 if method.endswith("PATCH_ONLY") else 0.2,
            )
            backprojected = labels[patch["source_patch"]]
            changed = covered & (backprojected >= 0)
            result = ids.copy()
            result[changed] = class_ids[backprojected[changed]]
            details["changed_from_point_count"] = int(np.count_nonzero(result != ids))
            ids = result
        else:
            ids, conf, covered, _, _ = point_head(method)
            if covered is not None:
                details["feature_coverage"] = float(covered.mean())
        atomic_npz(
            target,
            source_indices=source,
            owner_ids=owners,
            semantic_ids=ids,
            semantic_confidence=conf,
        )
        _atomic_json(
            output / f"{method}_prediction.json",
            {
                "method": method,
                "source_rows": len(source),
                "inference_and_write_seconds": time.monotonic() - started,
                "runtime_scope": "this method execution; cached point-head and observation construction shared in sorted method order, not a standalone cross-method cost",
                "changed_owner_rows": int(np.count_nonzero(owners != native_owners)),
                "GT_read": False,
                **details,
            },
        )
        print("room1 prediction saved", method, flush=True)
    # Validate every selected output before any GT load, including cache reuse.
    for method in methods:
        with np.load(output / f"{method}.npz") as data:
            if not np.array_equal(data["source_indices"], source):
                raise ValueError("confirmation prediction changed source rows")
            if method.startswith("INST_"):
                if not np.array_equal(
                    data["semantic_ids"], native_ids
                ) or not np.array_equal(data["semantic_confidence"], native_conf):
                    raise ValueError("owner-only confirmation changed native semantics")
            elif not np.array_equal(data["owner_ids"], native_owners):
                raise ValueError("semantic confirmation changed native owners")
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room1")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
        aliases=manifest["aliases"],
    )
    for method in methods:
        target = compact / f"room1_{method}.json"
        if target.exists():
            if (
                json.loads(target.read_text())["selection_sha256"]
                != binding["selection_sha256"]
            ):
                raise ValueError(
                    "cached confirmation score belongs to another selection"
                )
            continue
        with np.load(output / f"{method}.npz") as data:
            ids, owners, conf = (
                data["semantic_ids"],
                data["owner_ids"],
                data["semantic_confidence"],
            )
        info = []
        for owner, owner_rows in _owner_row_groups(owners).items():
            values = native_ids[owner_rows]
            values = values[values > 0]
            if owner > 0 and len(values):
                info.append(
                    EntityEvaluationInfo(
                        int(owner),
                        int(np.bincount(values).argmax()),
                        2,
                        float(native_conf[owner_rows].mean()),
                    )
                )
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            ids,
            owners,
            conf,
            (native_owners > 0).astype(np.float32),
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
        record = json.loads((output / f"{method}_prediction.json").read_text())
        record.update(
            case="room1",
            split="confirm",
            status="FULL_MAP_EVALUATED",
            metrics=measured,
            geometry_fixed=True,
            owner_changed=bool(record["changed_owner_rows"]),
            selection_sha256=binding["selection_sha256"],
            GT_read=True,
            instance_ranking="same class-agnostic size-based protocol and two-view gate as DEV",
        )
        _atomic_json(target, record)
        print(
            "room1",
            method,
            {k: measured[k] for k in ("miou", "ap50", "f5")},
            flush=True,
        )
    scored = {
        method: json.loads((compact / f"room1_{method}.json").read_text())
        for method in methods
    }
    invariants = validate_scored_invariants(scored)
    _atomic_json(
        compact / "room1_confirmation_invariants.json",
        {
            "selection_sha256": binding["selection_sha256"],
            "checks": invariants,
            "all_source_arrays_verified": True,
        },
    )
    _atomic_json(
        compact / "room1_confirmation_status.json",
        {
            "status": "SELECTED_FOUR_FAMILY_MATRIX_EVALUATED",
            "scene": "room1",
            "methods": methods,
            "selection_sha256": binding["selection_sha256"],
            "retuning": False,
        },
    )


if __name__ == "__main__":
    main()
