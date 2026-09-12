#!/usr/bin/env python3
"""Reproduce saved B3/H2 through legacy and pointwise common-v2 inputs.

Predictions are written before evaluator targets are loaded. Source indices stay
canonical in the sidecar; a separately saved legacy ordering preserves KD-tree
tie behavior. No neural relation generation or state update is repeated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import _evaluation_context
from scripts.evaluation.run_crove_fine_current_map import (
    _dynamic_snapshot,
    _entity_records,
    _owner_row_groups,
)
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.evaluation.two_visit_snapshot_metrics import (
    TwoVisitEvaluationContext,
    evaluate_two_visit_points,
    evaluate_two_visit_snapshot,
)
from src.oviv2.fine_dynamic_policy import CoarseSurfaceState, load_b3_surface_policy
from src.oviv2.surface_readout import BRIDGE_VERSION, legacy_roles
from src.oviv2.two_visit_execution import SignedVisibilityConfig


def path(value):
    p = Path(os.path.expandvars(str(value)))
    return p if p.is_absolute() else ROOT / p


def write_json(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/evaluation/crove_multimethod_readout_v1.json"
    )
    args = parser.parse_args()
    config = json.loads(path(args.config).read_text())
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    output = path(config["run_root"]) / "bridge"
    output.mkdir(parents=True, exist_ok=True)
    compact = path(config["compact_output_root"])
    print("Loading fixed surface arrays", flush=True)
    surface_path = path(pair["current_map_root"]) / "current_surface.npz"
    with np.load(surface_path) as data:
        xyz = data["vertices_xyz"]
        owners = data["owner_entity_ids"]
        visits = data["source_visit_ids"]
        source_rows = data["source_vertex_indices"]
    n0 = int(np.count_nonzero(visits == 0))
    if not (np.all(visits[:n0] == 0) and np.all(visits[n0:] == 1)):
        raise ValueError("canonical visit order changed")
    groups = [_owner_row_groups(owners[:n0]), _owner_row_groups(owners[n0:])]
    records = [
        _entity_records(path(pair["b0_entities"])),
        {
            k + int(pair["t1_owner_offset"]): v
            for k, v in _entity_records(path(pair["b2_entities"])).items()
        },
    ]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    ids = np.zeros(len(xyz), np.int32)
    covered = np.zeros(len(xyz), bool)
    ordered_chunks = []
    background_chunks = []
    for offset, grouping, record_set in zip([0, n0], groups, records, strict=True):
        bg = grouping.get(0, np.empty(0, np.int64)) + offset
        covered[bg] = True
        background_chunks.append(bg)
        for owner in sorted(record_set):
            rows = grouping.get(owner, np.empty(0, np.int64)) + offset
            label = record_set[owner]["semantic_label"]
            lookup = crosswalk.lookup(label) if label is not None else crosswalk.unknown
            ids[rows] = lookup.semantic_id
            covered[rows] = True
            ordered_chunks.append(rows)
    order = np.concatenate(ordered_chunks + background_chunks)
    if len(np.unique(order)) != len(order):
        raise ValueError("legacy snapshot repeats source rows")
    sources, roles = legacy_roles(ids, owners == 0, crosswalk)
    base = path(state_config["run_root"]) / "dev/pairs" / pair["pair_id"] / "hypotheses"
    states = {}
    for name, hypothesis, variant in [
        ("B3", "H1_B3", "D1_B3"),
        ("H2", "H2_INHERIT", "D2_INHERIT"),
    ]:
        with np.load(
            base / hypothesis / variant / "entity_epoch_state/entity_epoch_state.npz"
        ) as state:
            if not np.array_equal(state["source_visit_ids"], visits):
                raise ValueError("state visit keys differ")
            if not np.array_equal(state["source_vertex_indices"], source_rows):
                raise ValueError("state source row keys differ")
            current = state["current_valid_after"]
        if not current[n0:].all():
            raise ValueError("legacy t1 all-current assumption does not hold")
        states[name] = current
        target = output / f"{name}_legacy_prediction.npz"
        if not target.exists():
            with target.open("xb") as f:
                np.savez_compressed(
                    f,
                    source_indices=np.flatnonzero(current),
                    semantic_ids=ids[current],
                    eval_role=roles[current],
                    legacy_eval_source=sources[current],
                    owner_ids=owners[current],
                    legacy_covered=covered[current],
                    semantic_update_kind=np.zeros(np.count_nonzero(current), np.uint8),
                    legacy_eval_order=order[current[order]],
                )
        else:
            with np.load(target) as saved:
                if not np.array_equal(saved["source_indices"], np.flatnonzero(current)):
                    raise ValueError("saved prediction state changed")
        print(name, "prediction saved", int(current.sum()), "rows", flush=True)
    # All predicted labels/states are now on disk, prior to opening GT.
    context_path = output / "evaluation_context.npz"
    if context_path.exists():
        with np.load(context_path) as cache:
            values = {f.name: cache[f.name] for f in fields(TwoVisitEvaluationContext)}
        for key in ["event_id", "frame_id", "intervention_frame_id"]:
            values[key] = values[key].item()
        context = TwoVisitEvaluationContext(**values)
    else:
        print("Preparing frozen evaluator visibility", flush=True)
        dataset = TesseCdRgbdDataset(
            path(pair["rgbd_root"]),
            pair["scene"],
            path(pair["rgbd_export_manifest"]),
            path(pair["causal_schedule"]),
        )
        start, stop = pair["visit_frame_ranges"]["t1"]
        frames = tuple(dataset[i] for i in range(start, stop + 1))
        context, _ = _evaluation_context(
            protocol=json.loads(path(pair["two_visit_protocol"]).read_text()),
            scene=pair["scene"],
            final_frame=stop,
            schedule_path=path(pair["causal_schedule"]),
            target_manifest_path=path(pair["common_v2_target_manifest"]),
            frames=frames,
            visibility_config=SignedVisibilityConfig(
                **state_config["evaluation_visibility"]
            ),
        )
        with context_path.open("xb") as f:
            np.savez_compressed(
                f, **{v.name: getattr(context, v.name) for v in fields(context)}
            )
        del frames, dataset
    policy = load_b3_surface_policy(path(pair["b3_provenance"]), owners[:n0])
    observed = policy.states != int(CoarseSurfaceState.RETAINED_UNOBSERVED)
    for name, current in states.items():
        target = compact / f"bridge_{name}.json"
        if target.exists():
            print(name, "existing receipt retained", flush=True)
            continue
        begin = time.monotonic()
        snapshot = _dynamic_snapshot(
            t0_xyz=xyz[:n0],
            t1_xyz=xyz[n0:],
            t0_groups=groups[0],
            t1_groups=groups[1],
            t0_records=records[0],
            t1_records=records[1],
            t0_current=current[:n0],
            final_frame=context.frame_id,
        )
        retained = {
            "retained_t0_xyz": xyz[:n0][current[:n0]],
            "retained_t0_t1_observed_mask": observed[current[:n0]],
        }
        print(name, "legacy evaluation", flush=True)
        old = evaluate_two_visit_snapshot(snapshot, context, crosswalk, **retained)
        del snapshot
        # Consume saved prediction values, not a regenerated label lookup.
        with np.load(output / f"{name}_legacy_prediction.npz") as saved:
            canonical = saved["source_indices"]
            eval_indices = saved["legacy_eval_order"]
            inverse = np.searchsorted(canonical, eval_indices)
            predicted_ids = saved["semantic_ids"][inverse]
            predicted_roles = saved["eval_role"][inverse]
            predicted_owners = saved["owner_ids"][inverse]
        print(name, "pointwise evaluation", flush=True)
        new = evaluate_two_visit_points(
            points_xyz=xyz[eval_indices],
            semantic_ids=predicted_ids,
            eval_role=predicted_roles,
            owner_ids=predicted_owners,
            scene_id=pair["scene"],
            timestamp=float(context.frame_id),
            context=context,
            crosswalk=crosswalk,
            **retained,
        )
        delta = {k: float(new[k] - old[k]) for k in old}
        passed = all(abs(v) <= 1e-6 for v in delta.values())
        receipt = {
            "bridge_version": BRIDGE_VERSION,
            "state": name,
            "status": "PASS" if passed else "FAIL",
            "legacy": old,
            "pointwise": new,
            "delta": delta,
            "current_rows": int(current.sum()),
            "legacy_covered_rows": len(eval_indices),
            "legacy_uncovered_rows": int(np.count_nonzero(current & ~covered)),
            "role_counts": {
                str(i): int(np.count_nonzero(predicted_roles == i)) for i in range(3)
            },
            "elapsed_seconds": time.monotonic() - begin,
        }
        write_json(target, receipt)
        print(json.dumps(receipt), flush=True)
        if not passed:
            raise ValueError("pointwise bridge changes legacy metrics")


if __name__ == "__main__":
    main()
