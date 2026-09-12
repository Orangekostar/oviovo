"""Freeze connected local structural-background crop regions without GT."""

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
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.oviv2.surface_readout_graph import current_surface_topology, surface_patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all_dev", "room1"), default="all_dev")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    run, compact = path(config["run_root"]), path(config["compact_output_root"])
    fine = json.loads(path(config["source_config"]).read_text())
    state = json.loads(path(config["state_config"]).read_text())
    pair = state["splits"]["dev"]["pairs"][0]
    structural = ["wall", "floor", "ceiling", "stairs"]
    settings = {
        "cases": ["room0", "apartment_B3", "apartment_H2"],
        "structural_class_names": structural,
        "explicit_background": "include owner 0 as a separate original source group",
        "selection": "original native labels and original current rows only; never new trial labels or GT",
        "patch_size_m": 0.5,
        "connectivity": "actual source triangles inside each .5m cell; same owner, same visit and compatible normals (cosine >=.95); disconnected components stay separate",
        "purpose": "local crop regions only; no owner mutation and no removal of any full-map row",
        "uncovered": "retain native/S2 semantic fallback in later readout",
        "GT_read": False,
    }
    registry = compact / "local_background_regions_registry.json"
    if args.case == "room1":
        settings["cases"] = ["room1"]
        registry = compact / "local_background_regions_room1_registry.json"
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen local background policy changed")
    _atomic_json(registry, settings)
    replica_manifest = json.loads(
        path(fine["cases"]["replica_room0_static"]["benchmark_manifest"]).read_text()
    )
    replica_ids = [
        i + 1
        for i, name in enumerate(replica_manifest["vocabulary"]["classes"])
        if name.lower() in structural
    ]
    crosswalk = load_tesse_semantic_crosswalk(
        path(pair["semantic_aliases"]),
        pair["scene"],
        path(pair["semantic_label_space"]),
    )
    apartment_ids = sorted(
        {
            v.semantic_id
            for v in crosswalk.aliases.values()
            if v.matched and v.native_name.lower() in structural
        }
    )
    cases = [
        (
            "room0",
            run / "dev/room0/native_cached_batch/B_SEM_OVI_NATIVE.npz",
            path(
                "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/current_surface.npz"
            ),
            replica_ids,
        )
    ]
    cases += [
        (
            f"apartment_{s}",
            run / f"dev/apartment/native_cached_batch/{s}_MV_NATIVE_CACHED.npz",
            path(pair["current_map_root"]) / "current_surface.npz",
            apartment_ids,
        )
        for s in ("B3", "H2")
    ]
    if args.case == "room1":
        cases = [
            (
                "room1",
                run / "confirm/room1/native_cached_batch/B_SEM_OVI_NATIVE.npz",
                run / "confirm/room1/inputs/current_surface.npz",
                replica_ids,
            )
        ]
    for name, baseline, geometry, classes in cases:
        root = (
            run
            / (
                "confirm/local_background_regions"
                if name == "room1"
                else "dev/local_background_regions"
            )
            / name
        )
        root.mkdir(parents=True, exist_ok=True)
        binding = {
            "native_prediction": file_hash(baseline),
            "geometry": file_hash(geometry),
            "policy": file_hash(registry),
            "class_ids": classes,
        }
        binding_file = root / "input_binding.json"
        if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
            raise ValueError("local background source changed")
        _atomic_json(binding_file, binding)
        target = root / "regions.npz"
        if target.exists():
            print(name, "frozen regions already exist", flush=True)
            continue
        started = time.monotonic()
        with np.load(baseline) as data:
            source, owners, labels = (
                data["source_indices"],
                data["owner_ids"],
                data["semantic_ids"],
            )
        selected = source[np.isin(labels, classes) | (owners == 0)]
        with np.load(geometry) as data:
            xyz, normals, faces, visits = (
                data[k]
                for k in (
                    "vertices_xyz",
                    "normals_xyz",
                    "triangles",
                    "source_visit_ids",
                )
            )
            if not np.array_equal(data["owner_entity_ids"][source], owners):
                raise ValueError("native owner/source binding differs")
        surface = current_surface_topology(xyz, normals, faces, visits, selected)
        local = np.searchsorted(source, selected)
        # The grouping argument prevents welding across owner or visit; the
        # original visit IDs are saved independently and never replaced.
        _, compatibility = np.unique(
            np.column_stack((visits[selected], owners[local])),
            axis=0,
            return_inverse=True,
        )
        patches = surface_patches(
            surface["xyz"],
            surface["normals"],
            surface["triangles"],
            visit_ids=compatibility,
            patch_size=0.5,
        )
        regions = patches["source_patch"]
        count = len(patches["centers"])
        first = np.full(count, len(selected), np.int64)
        np.minimum.at(first, regions, np.arange(len(selected)))
        if not np.array_equal(compatibility[first][regions], compatibility):
            raise ValueError("local region crossed an original owner/visit boundary")
        atomic_npz(
            target,
            source_indices=selected,
            source_region=regions,
            centers=patches["centers"],
            normals=patches["normals"],
            physical_weight=patches["physical_weight"],
            region_owner_ids=owners[local[first]],
            region_visit_ids=visits[selected[first]],
        )
        record = {
            "case": name,
            "status": "REGIONS_BUILT_OBSERVATIONS_PENDING",
            "full_state_source_rows": len(source),
            "structural_source_rows": len(selected),
            "region_count": count,
            "physical_samples": int(patches["physical_samples"]),
            "construction_seconds": time.monotonic() - started,
            "GT_read": False,
        }
        _atomic_json(compact / f"{name}_local_background_regions.json", record)
        print(record, flush=True)


if __name__ == "__main__":
    main()
