"""Freeze complete per-state native surface topology for dynamic M2/M3."""

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
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.oviv2.surface_readout_graph import current_surface_topology, surface_patches


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    run = path(config["run_root"])
    output = run / "dev/apartment/graph_topology"
    output.mkdir(exist_ok=True)
    settings = {
        "case": pair["pair_id"],
        "states": ["B3", "H2"],
        "patch_size_m": 0.02,
        "normals": "same fixed 0.95 cosine compatibility as room0",
        "edges": "actual welded source triangle adjacency; Gaussian distance sigma 0.03 m",
        "current_filter": "all source rows from each frozen state; keep only faces with all three vertices current; retain isolated current rows",
        "visit_boundary": "never weld or propagate across visits; this source has one geometry epoch/surface per visit and is checked explicitly",
        "shared_representation": "all unary and boundary controls in a state share this exact patch inverse",
        "GT_read": False,
    }
    registry = (
        path(config["compact_output_root"]) / "graph_topology_apartment_registry.json"
    )
    if registry.exists():
        if json.loads(registry.read_text()) != settings:
            raise ValueError("frozen topology policy changed")
    else:
        _atomic_json(registry, settings)
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz, normals, triangles = (
            data["vertices_xyz"],
            data["normals_xyz"],
            data["triangles"],
        )
        owners, visits, source_rows = (
            data["owner_entity_ids"],
            data["source_visit_ids"],
            data["source_vertex_indices"],
        )
        if not np.array_equal(
            data["source_surface_indices"], visits
        ) or not np.array_equal(data["geometry_epochs"], visits):
            raise ValueError(
                "source has multiple geometry episodes per visit; requires explicit episode compatibility groups"
            )
    hypotheses = (
        path(state_config["run_root"]) / "dev/pairs" / pair["pair_id"] / "hypotheses"
    )
    for state, hypothesis, variant in [
        ("B3", "H1_B3", "D1_B3"),
        ("H2", "H2_INHERIT", "D2_INHERIT"),
    ]:
        root = output / state
        root.mkdir(exist_ok=True)
        target = root / "patch_mapping.npz"
        edge_file = root / "geometry_edges.npz"
        with np.load(
            hypotheses
            / hypothesis
            / variant
            / "entity_epoch_state/entity_epoch_state.npz"
        ) as data:
            if not np.array_equal(
                data["source_visit_ids"], visits
            ) or not np.array_equal(data["source_vertex_indices"], source_rows):
                raise ValueError("state source identity differs")
            current = np.flatnonzero(data["current_valid_after"])
        with np.load(run / "bridge" / f"{state}_legacy_prediction.npz") as data:
            if not np.array_equal(
                data["source_indices"], current
            ) or not np.array_equal(data["owner_ids"], owners[current]):
                raise ValueError("state/bridge source identity differs")
        if target.exists() and edge_file.exists():
            with np.load(target) as data:
                if not np.array_equal(data["source_indices"], current):
                    raise ValueError("cached topology state changed")
            print(state, "existing topology retained", flush=True)
            continue
        started = time.monotonic()
        surface = current_surface_topology(xyz, normals, triangles, visits, current)
        face_count = len(surface["triangles"])
        print(
            state, len(current), "source rows;", face_count, "current faces", flush=True
        )
        patches = surface_patches(
            surface["xyz"],
            surface["normals"],
            surface["triangles"],
            visit_ids=surface["visit_ids"],
        )
        adjacency = patches.pop("adjacency")
        if len(patches["source_patch"]) != len(current):
            raise ValueError("patch mapping lost current source rows")
        stats = {
            "state": state,
            "source_rows": len(current),
            "current_faces": face_count,
            "patch_nodes": len(patches["centers"]),
            "physical_samples": int(patches["physical_samples"]),
            "undirected_edges": adjacency.nnz // 2,
            "mean_degree": adjacency.nnz / max(1, len(patches["centers"])),
            "backprojection_coverage": 1.0,
            "construction_seconds": time.monotonic() - started,
            "GT_read": False,
            "status": "FROZEN_TOPOLOGY_BUILT_NOT_YET_SCORED",
        }
        atomic_npz(
            target,
            source_indices=current,
            owner_ids=owners[current],
            source_visit_ids=visits[current],
            **patches,
        )
        temporary = root / "geometry_edges.partial.npz"
        sparse.save_npz(temporary, adjacency)
        temporary.replace(edge_file)
        _atomic_json(
            path(config["compact_output_root"])
            / f"apartment_{state}_graph_topology.json",
            stats,
        )
        print(stats, flush=True)
        del surface, patches, adjacency


if __name__ == "__main__":
    main()
