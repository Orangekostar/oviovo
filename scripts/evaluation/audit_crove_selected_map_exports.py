"""Verify every exported current vertex against frozen prediction/source arrays."""

import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    compact, run = path(config["compact_output_root"]), path(config["run_root"])
    manifest = json.loads((compact / "selected_apartment_map_exports.json").read_text())
    selected = json.loads((compact / "selected_configs.json").read_text())["FINAL_CURRENT_READOUT"]
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    with np.load(run / selected["prediction"]) as data:
        rows, owners, ids = data["source_indices"], data["owner_ids"], data["semantic_ids"]
    with np.load(path(pair["current_map_root"]) / "current_surface.npz") as data:
        xyz = data["vertices_xyz"][rows]
        rgb, rgb_valid = data["observed_rgb_uint8"][rows], data["rgb_valid"][rows]
        triangles = data["triangles"]
        inverse = np.full(len(data["current_valid"]), -1, np.int64)
    inverse[rows] = np.arange(len(rows))
    mapped = inverse[triangles]
    expected_faces = mapped[np.all(mapped >= 0, axis=1)]
    rgb[~rgb_valid] = 128
    checks = {}
    for mode, record in manifest["views"].items():
        target = run / record["path"]
        if _sha256(target) != record["sha256"]:
            raise ValueError(f"export hash changed: {mode}")
        ply = PlyData.read(target, mmap="r")
        vertices = ply["vertex"]
        if len(vertices) != len(rows):
            raise ValueError(f"export lost source rows: {mode}")
        for key, expected in (("canonical_source_row", rows), ("owner_id", owners), ("semantic_id", ids), ("rgb_valid", rgb_valid)):
            if not np.array_equal(vertices[key], expected):
                raise ValueError(f"export field differs: {mode}/{key}")
        for axis, expected in zip(("x", "y", "z"), xyz.T):
            if not np.array_equal(vertices[axis], expected):
                raise ValueError(f"export geometry differs: {mode}/{axis}")
        if mode == "rgb":
            for channel, expected in zip(("red", "green", "blue"), rgb.T):
                if not np.array_equal(vertices[channel], expected):
                    raise ValueError("RGB is not original observed RGB with declared missing-value gray")
        faces = ply["face"]["vertex_indices"]
        if len(faces) != len(expected_faces):
            raise ValueError(f"export triangle count differs: {mode}")
        for start in range(0, len(faces), 100_000):
            if not np.array_equal(np.stack(faces[start:start + 100_000]), expected_faces[start:start + 100_000]):
                raise ValueError(f"export connectivity differs: {mode}")
        checks[mode] = {"all_vertices_exact": True, "all_source_keys_exact": True,
                        "all_owner_and_semantic_ids_exact": True, "all_triangles_exact": True}
        print(mode, "full geometry and labels PASS", flush=True)
    _atomic_json(compact / "selected_apartment_map_export_audit.json", {
        "status": "PASS", "source_rows": len(rows), "triangles": len(expected_faces),
        "checks": checks, "GT_read": False, "sampled": False,
        "export_manifest_sha256": _sha256(compact / "selected_apartment_map_exports.json"),
    })


if __name__ == "__main__":
    main()
