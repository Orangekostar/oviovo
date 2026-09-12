"""Full native-fallback local-background and original-owner semantic controls."""

from __future__ import annotations

import json
import sys
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
    _owner_row_groups,
    evaluate_replica_voxel_map,
    load_replica_ground_truth,
)
from scripts.evaluation.run_crove_multiview_apartment import evaluate_saved_readouts
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_multiview_semantics import owner_posteriors_to_rows
from src.oviv2.surface_readout import resolve_semantic_update


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    fine = json.loads(path(config["source_config"]).read_text())
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    methods = ["MV_BG_OWNER_QUALITY", "MV_LOCAL_BG_QUALITY"]
    settings = {
        "methods": methods,
        "cases": ["room0", "apartment_B3", "apartment_H2"],
        "fallback": "original native semantics/confidence/roles everywhere not supported by the declared head",
        "local": "real local-region full-class posterior on all source rows of its frozen region; no owner changes",
        "owner_control": "same rows with a usable local region observation, but read the matched re-encoded original-owner quality head",
        "coverage": "local regions with no original-owner feature retain native in owner control; counted separately, not attributed solely to locality",
        "GT_before_all_six_predictions": False,
    }
    registry = compact / "local_background_readout_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen local readout policy changed")
    _atomic_json(registry, settings)
    for name in settings["cases"]:
        if not (
            run / "dev/local_background_regions" / name / "local_posteriors.npz"
        ).exists():
            raise RuntimeError(f"actual local feature bank not complete: {name}")
    for name in settings["cases"]:
        static = name == "room0"
        state = None if static else name.split("_")[1]
        room = run / ("dev/room0" if static else "dev/apartment")
        output = room / "local_background"
        output.mkdir(exist_ok=True)
        root = run / "dev/local_background_regions" / name
        baseline_file = (
            room
            / "native_cached_batch"
            / ("B_SEM_OVI_NATIVE.npz" if static else f"{state}_MV_NATIVE_CACHED.npz")
        )
        owner_file = (
            room
            / "reencoded_regions"
            / (
                "MV_REENCODE_QUALITY.npz"
                if static
                else f"{state}_MV_REENCODE_QUALITY.npz"
            )
        )
        with np.load(baseline_file) as data:
            keep = (
                ("source_indices", "owner_ids", "semantic_ids", "semantic_confidence")
                if static
                else data.files
            )
            baseline = {key: data[key] for key in keep}
        source = baseline["source_indices"]
        with np.load(root / "regions.npz") as data:
            region_source, regions = data["source_indices"], data["source_region"]
        local_rows = np.searchsorted(source, region_source)
        if not np.array_equal(source[local_rows], region_source):
            raise ValueError("local region not on native current source")
        with np.load(root / "local_posteriors.npz") as data:
            proposed, confidence, covered = owner_posteriors_to_rows(
                regions, data["region_ids"], data["posterior"], data["class_ids"]
            )
        rows = local_rows[covered]
        local_ids, local_conf = proposed[covered], confidence[covered]
        with np.load(owner_file) as data:
            if not np.array_equal(data["source_indices"], source) or not np.array_equal(
                data["owner_ids"], baseline["owner_ids"]
            ):
                raise ValueError("owner control source differs")
            owner_ids, owner_conf, owner_covered = (
                data["semantic_ids"][rows],
                data["semantic_confidence"][rows],
                data["feature_covered"][rows],
            )
        binding = {
            "native": file_hash(baseline_file),
            "owner_head": file_hash(owner_file),
            "regions": file_hash(root / "regions.npz"),
            "local_posterior": file_hash(root / "local_posteriors.npz"),
            "policy": file_hash(registry),
        }
        binding_file = output / f"{name}_input_binding.json"
        if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
            raise ValueError("frozen local readout inputs changed")
        _atomic_json(binding_file, binding)
        for method in methods:
            stem = method if static else f"{state}_{method}"
            prediction = dict(baseline)
            update = np.zeros(len(source), np.uint8)
            ids, conf = (
                baseline["semantic_ids"].copy(),
                baseline["semantic_confidence"].copy(),
            )
            if method == methods[0]:
                active, values, scores = (
                    rows[owner_covered],
                    owner_ids[owner_covered],
                    owner_conf[owner_covered],
                )
            else:
                active, values, scores = rows, local_ids, local_conf
            update[active] = 1
            ids[active] = values
            conf[active] = scores
            if not static:
                ids, roles = resolve_semantic_update(
                    baseline["semantic_ids"],
                    baseline["eval_role"],
                    ids,
                    update,
                    crosswalk,
                )
                prediction["eval_role"] = roles
            prediction.update(
                semantic_ids=ids,
                semantic_confidence=conf,
                semantic_update_kind=update,
                feature_covered=update.astype(bool),
            )
            atomic_npz(output / f"{stem}.npz", **prediction)
            _atomic_json(
                output / f"{stem}_prediction.json",
                {
                    "case": name,
                    "state": state,
                    "method": method,
                    "source_rows": len(source),
                    "updated_source_rows": len(active),
                    "local_supported_source_rows": len(rows),
                    "local_supported_rows_without_owner_features": int(
                        (~owner_covered).sum()
                    ),
                    "changed_semantic_rows": int(
                        np.count_nonzero(ids != baseline["semantic_ids"])
                    ),
                    "geometry_and_owner_fixed": True,
                    "fallback": "original native",
                },
            )
            print(name, method, "full prediction saved", flush=True)
    # All six complete predictions are present before either GT evaluator.
    evaluate_saved_readouts(
        config,
        state_config,
        pair,
        crosswalk,
        run / "dev/apartment/local_background",
        methods,
    )
    case = fine["cases"]["replica_room0_static"]
    manifest = json.loads(path(case["benchmark_manifest"]).read_text())
    scene = next(s for s in manifest["scenes"] if s["scene"] == "room0")
    gt = path(manifest["ground_truth_root"]) / scene["ground_truth_scene"] / "habitat"
    vocabulary = {c: i + 1 for i, c in enumerate(manifest["vocabulary"]["classes"])}
    truth = load_replica_ground_truth(
        gt / "mesh_semantic.ply",
        gt / "info_semantic.json",
        class_to_id=vocabulary,
        aliases=manifest["aliases"],
    )
    with np.load(
        path(
            "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
        )
    ) as data:
        xyz = data["vertices_xyz"]
    for method in methods:
        target = compact / f"room0_{method}.json"
        if target.exists():
            continue
        root = run / "dev/room0/local_background"
        with np.load(root / f"{method}.npz") as data:
            ids, owners, conf = (
                data["semantic_ids"],
                data["owner_ids"],
                data["semantic_confidence"],
            )
            if not np.array_equal(data["source_indices"], np.arange(len(xyz))):
                raise ValueError("static source row mapping differs")
        info = []
        for owner, rows in _owner_row_groups(owners).items():
            values = ids[rows]
            values = values[values > 0]
            if owner > 0 and len(values):
                info.append(
                    EntityEvaluationInfo(
                        owner,
                        int(np.bincount(values).argmax()),
                        2,
                        float(conf[rows].mean()),
                    )
                )
        mesh = LabeledMesh(
            xyz,
            np.empty((0, 3), np.int64),
            np.zeros_like(xyz),
            ids,
            owners,
            conf,
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
        record = json.loads((root / f"{method}_prediction.json").read_text())
        record.update(status="FULL_MAP_EVALUATED", metrics=measured)
        _atomic_json(target, record)
        print(
            method,
            {k: measured[k] for k in ("miou", "f_miou", "ap50", "f5")},
            flush=True,
        )


if __name__ == "__main__":
    main()
