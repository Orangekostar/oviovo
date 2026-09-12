"""Unscored room1 topology shared by confirmation readouts."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.build_crove_room0_mask_bank import file_hash
from scripts.evaluation.run_crove_adapter_room0 import atomic_npz, path
from scripts.evaluation.run_ovimap_native import _atomic_json
from src.oviv2.surface_readout_graph import surface_patches


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    room = path(config["run_root"]) / "confirm/room1"
    source_file = room / "inputs/current_surface.npz"
    output = room / "graph_s2"
    output.mkdir(exist_ok=True)
    settings = {
        "scene": "room1",
        "source_sha256": file_hash(source_file),
        "patch_size_m": 0.02,
        "normal_cosine": 0.95,
        "edge_distance_sigma_m": 0.03,
        "implementation": "same surface_patches as room0; actual welded triangle adjacency and physical duplicate weighting",
        "GT_read": False,
        "semantic_selection_used": False,
    }
    registry = (
        path(config["compact_output_root"]) / "room1_graph_topology_registry.json"
    )
    if registry.exists() and json.loads(registry.read_text()) != settings:
        raise ValueError("frozen room1 topology changed")
    _atomic_json(registry, settings)
    if (output / "patch_mapping.npz").exists() and (
        output / "geometry_edges.npz"
    ).exists():
        return
    started = time.monotonic()
    with np.load(source_file) as data:
        xyz, normals, faces, owners, source = (
            data[k]
            for k in (
                "vertices_xyz",
                "normals_xyz",
                "triangles",
                "owner_entity_ids",
                "source_vertex_indices",
            )
        )
        if not data["current_valid"].all() or not np.array_equal(
            source, np.arange(len(xyz))
        ):
            raise ValueError("room1 source state changed")
    patches = surface_patches(xyz, normals, faces)
    adjacency = patches.pop("adjacency")
    atomic_npz(
        output / "patch_mapping.npz", source_indices=source, owner_ids=owners, **patches
    )
    temporary = output / "geometry_edges.partial.npz"
    sparse.save_npz(temporary, adjacency)
    temporary.replace(output / "geometry_edges.npz")
    record = {
        "scene": "room1",
        "source_rows": len(source),
        "nodes": len(patches["centers"]),
        "edges": adjacency.nnz // 2,
        "construction_seconds": time.monotonic() - started,
        "GT_read": False,
    }
    _atomic_json(
        path(config["compact_output_root"]) / "room1_graph_topology.json", record
    )
    print(record, flush=True)


if __name__ == "__main__":
    main()
