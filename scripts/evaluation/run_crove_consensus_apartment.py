"""Per-visit independent-mask owner-only controls on frozen Apartment B3/H2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_mask_consensus import (
    apply_owner_proposals,
    arbitrate_candidates,
    cluster_masks,
    construct_mask_graph,
)


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
    run = path(config["run_root"])
    room, compact = run / "dev/apartment", path(config["compact_output_root"])
    output = room / "consensus"
    output.mkdir(exist_ok=True)
    methods = {"INST_PAIRWISE": "pairwise", "INST_CONSENSUS_OWNER": "consensus"}
    settings = json.loads((compact / "consensus_room0_registry.json").read_text())
    settings.update(
        case=pair["pair_id"],
        states=["B3", "H2"],
        semantics="exact frozen native current point semantics, confidence and eval_role",
        selection="initial frozen room0 parameters; no dynamic AP selection; static selection remains pending",
        visit_policy="separate mask graphs per visit; deterministic disjoint candidate IDs; no cross-visit union",
        instance_AP="N/A: this common-v2 context has no corresponding instance GT",
    )
    registry = compact / "consensus_apartment_registry.json"
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen consensus settings changed")
    else:
        _atomic_json(registry, settings)
    status = json.loads(
        (compact / "independent_mask_bank_apartment_status.json").read_text()
    )
    mask_registry = json.loads(
        (compact / "independent_mask_bank_apartment_registry.json").read_text()
    )
    expected_frames = [
        frame
        for visit in (0, 1)
        for frame in range(
            pair["visit_frame_ranges"][f"t{visit}"][0],
            pair["visit_frame_ranges"][f"t{visit}"][1] + 1,
        )
    ]
    for state in ("B3", "H2"):
        if status.get(state, {}).get("completed_frames") != expected_frames:
            raise RuntimeError(
                "M3 requires both complete 256-frame visit banks in each frozen state"
            )
        root = room / "independent_mask_bank" / state
        topology = room / "graph_topology" / state / "patch_mapping.npz"
        if file_hash(topology) != mask_registry["patch_sha256"][state]:
            raise ValueError("mask/topology binding changed")
        with np.load(topology) as data:
            source, owners, patch = (
                data["source_indices"],
                data["owner_ids"],
                data["source_patch"],
            )
            visits = data["source_visit_ids"]
        with np.load(root / "representatives.npz") as data:
            node_visits = data["visit_ids"]
        if not np.array_equal(node_visits[patch], visits):
            raise ValueError("patch spans visits")
        proposals = {
            method: np.full(len(node_visits), -1, np.int64) for method in methods
        }
        histories, offsets = (
            {method: {} for method in methods},
            {method: 0 for method in methods},
        )
        started = time.monotonic()
        for visit in (0, 1):
            nodes = np.flatnonzero(node_visits == visit)
            node_inverse = np.full(len(node_visits), -1, np.int64)
            node_inverse[nodes] = np.arange(len(nodes))
            start, stop = pair["visit_frame_ranges"][f"t{visit}"]
            frames = list(range(start, stop + 1))
            labels = np.zeros((len(nodes), len(frames)), np.uint16, order="F")
            native = path(
                f"$HOME/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/native-runs/apartment/t{visit}/2f638911509f-apartment-t{visit}/frontend"
            )
            for column, frame in enumerate(frames):
                with np.load(root / f"{frame:06d}.npz") as data:
                    observed = data["node_ids"]
                    if (
                        int(data["frame_id"]) != frame
                        or int(data["visit_id"]) != visit
                        or len(observed) != len(np.unique(observed))
                        or np.any(observed < 0)
                        or np.any(observed >= len(node_visits))
                    ):
                        raise ValueError("invalid observation identity")
                    local_nodes = node_inverse[observed]
                    digest = file_hash(native / f"frame{frame - start:06d}.png")
                    if (
                        np.any(local_nodes < 0)
                        or str(data["mask_sha256"]) != digest
                        or digest != mask_registry["mask_sha256"][str(frame)]
                    ):
                        raise ValueError(
                            "observation crosses visit or differs from original mask"
                        )
                    labels[local_nodes, column] = data["mask_ids"]
            graph_started = time.monotonic()
            graph = construct_mask_graph(labels)
            graph_seconds = time.monotonic() - graph_started
            print(state, visit, len(graph["mask_keys"]), "masks", flush=True)
            for method, mode in methods.items():
                method_started = time.monotonic()
                assignment, history = cluster_masks(graph, mode=mode)
                votes = np.full(labels.shape, -1, np.int32, order="F")
                for column in range(len(frames)):
                    lookup = np.full(int(labels[:, column].max()) + 1, -1, np.int32)
                    for mask_index, (mask_frame, mask_id) in enumerate(
                        graph["mask_keys"]
                    ):
                        if mask_frame == column:
                            lookup[mask_id] = assignment[mask_index]
                    votes[:, column] = lookup[labels[:, column]]
                proposed = arbitrate_candidates(votes)
                supported = proposed >= 0
                proposals[method][nodes[supported]] = (
                    proposed[supported] + offsets[method]
                )
                offsets[method] += int(assignment.max()) + 1 if len(assignment) else 0
                histories[method][str(visit)] = {
                    "history": history,
                    "shared_graph_construction_seconds": graph_seconds,
                    "clustering_and_arbitration_seconds": time.monotonic()
                    - method_started,
                    "input_masks": len(assignment),
                    "undersegmented_masks": int(graph["undersegmented"].sum()),
                    "supported_patches": int(supported.sum()),
                }
                del votes
            del labels, graph
        with np.load(
            room / "native_cached_batch" / f"{state}_MV_NATIVE_CACHED.npz"
        ) as data:
            baseline = {key: data[key] for key in data.files}
        if not np.array_equal(source, baseline["source_indices"]) or not np.array_equal(
            owners, baseline["owner_ids"]
        ):
            raise ValueError("native readout differs from topology")
        unique_owners, first_rows = np.unique(owners, return_index=True)
        parent_visits = dict(zip(unique_owners.tolist(), visits[first_rows].tolist()))
        shared_owners = np.intersect1d(owners[visits == 0], owners[visits == 1])
        if np.any(shared_owners > 0):
            raise ValueError("source entity owner spans visits")
        for method in methods:
            result, lineage = apply_owner_proposals(owners, patch, proposals[method])
            for item in lineage:
                if len({parent_visits[parent] for parent in item["parents"]}) != 1:
                    raise ValueError("candidate merges visits")
            prediction = dict(baseline)
            prediction["owner_ids"] = result
            atomic_npz(
                output / f"{state}_{method}.npz",
                **prediction,
                patch_candidate=proposals[method],
            )
            children = {}
            for item in lineage:
                for parent in item["parents"]:
                    children[parent] = children.get(parent, 0) + 1
            revised_parents = np.array(sorted(children), np.int64)
            record = {
                "state": state,
                "method": method,
                "source_rows": len(source),
                "changed_owner_rows": int(np.count_nonzero(result != owners)),
                "new_owner_count": len(lineage),
                "original_instance_count": int(np.count_nonzero(unique_owners > 0)),
                "output_instance_count": int(np.count_nonzero(np.unique(result) > 0)),
                "split_parent_count": sum(v > 1 for v in children.values()),
                "merge_candidate_count": sum(len(v["parents"]) > 1 for v in lineage),
                "merge_parent_count": len(
                    {
                        parent
                        for item in lineage
                        if len(item["parents"]) > 1
                        for parent in item["parents"]
                    }
                ),
                "residual_parent_ID_count": len(
                    np.intersect1d(result, revised_parents)
                ),
                "residual_source_rows": int(np.isin(result, revised_parents).sum()),
                "unsupported_or_ambiguous_source_rows": int(
                    (proposals[method][patch] < 0).sum()
                ),
                "changed_semantic_rows": 0,
                "changed_role_rows": 0,
                "geometry_fixed": True,
                "owner_fixed": False,
                "prediction_and_write_seconds": time.monotonic() - started,
                "timing_scope": "cumulative shared state load, both candidates and writes; not isolated method cost",
                "CA_AP25": None,
                "CA_AP50": None,
                "AP_status": "N/A: no corresponding instance GT in frozen dynamic protocol",
            }
            _atomic_json(
                output / f"{state}_{method}_lineage.json",
                {"lineage": lineage, "per_visit": histories[method]},
            )
            _atomic_json(output / f"{state}_{method}_prediction.json", record)
            with np.load(output / f"{state}_{method}.npz") as data:
                for key, value in baseline.items():
                    if key != "owner_ids" and not np.array_equal(data[key], value):
                        raise ValueError(f"owner-only readout changed {key}")
            print(state, method, record, flush=True)
    evaluate_saved_readouts(
        config,
        state_config,
        pair,
        crosswalk,
        output,
        list(methods),
        allow_owner_changes=True,
    )
    for state in ("B3", "H2"):
        original = json.loads(
            (compact / f"apartment_{state}_MV_NATIVE_CACHED.json").read_text()
        )["metrics"]
        for method in methods:
            measured = json.loads(
                (compact / f"apartment_{state}_{method}.json").read_text()
            )["metrics"]
            if measured != original:
                raise ValueError("owner-only readout changed frozen pointwise metrics")
    print(
        "All four owner-only readouts reproduce all nine pointwise metrics exactly",
        flush=True,
    )


if __name__ == "__main__":
    main()
