"""Export four full-row views of the frozen final Apartment readout, without GT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256
from src.oviv2.current_surface import _stable_entity_colors, _write_ply


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    compact, run = path(config["compact_output_root"]), path(config["run_root"])
    selected_file = compact / "selected_configs.json"
    selected = json.loads(selected_file.read_text())
    choice = selected["FINAL_CURRENT_READOUT"]
    if selected["status"] != "DEV_SELECTION_FROZEN" or choice is None:
        raise RuntimeError("a frozen eligible final current readout is required")
    state = choice["state_variant"]
    if state not in ("B3", "H2"):
        raise ValueError("unsupported frozen state")
    state_config = json.loads(path(config["state_config"]).read_text())
    pair = state_config["splits"]["dev"]["pairs"][0]
    geometry = path(pair["current_map_root"]) / "current_surface.npz"
    hypothesis, variant = ("H1_B3", "D1_B3") if state == "B3" else ("H2_INHERIT", "D2_INHERIT")
    state_file = path(state_config["run_root"]) / "dev/pairs" / pair["pair_id"] / "hypotheses" / hypothesis / variant / "entity_epoch_state/entity_epoch_state.npz"
    prediction = run / choice["prediction"]
    output = run / "visualizations" / choice["config_id"].replace("@", "_")
    output.mkdir(parents=True, exist_ok=True)
    binding = {"selection_sha256": _sha256(selected_file), "geometry_sha256": _sha256(geometry),
               "state_sha256": _sha256(state_file), "prediction_sha256": _sha256(prediction)}
    binding_file = output / "input_binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("selected visualization inputs changed")
    _atomic_json(binding_file, binding)
    with np.load(state_file) as data:
        valid, before = data["current_valid_after"], data["current_valid_before"]
        visits, vertex_indices = data["source_visit_ids"], data["source_vertex_indices"]
    rows = np.flatnonzero(valid)
    with np.load(prediction) as data:
        if not np.array_equal(data["source_indices"], rows):
            raise ValueError("prediction does not contain the frozen current rows")
        owners, semantic = data["owner_ids"], data["semantic_ids"]
    with np.load(geometry) as data:
        if not np.array_equal(data["source_visit_ids"], visits) or not np.array_equal(data["source_vertex_indices"], vertex_indices):
            raise ValueError("geometry and state source keys differ")
        if not np.array_equal(data["owner_entity_ids"][rows], owners):
            raise ValueError("final semantic readout changed owner IDs")
        xyz = data["vertices_xyz"][rows]
        rgb, rgb_valid = data["observed_rgb_uint8"][rows], data["rgb_valid"][rows]
        triangles = data["triangles"]
    inverse = np.full(len(valid), -1, np.int64)
    inverse[rows] = np.arange(len(rows))
    triangles = inverse[triangles[np.all(valid[triangles], axis=1)]]
    # Display categories derive solely from the frozen before/after state and visit.
    state_codes = np.where(visits[rows] == 1, 1, np.where(before[rows], 2, 3)).astype(np.uint8)
    state_palette = {1: [30, 160, 75], 2: [45, 105, 190], 3: [235, 175, 30]}
    semantic_keys = np.unique(semantic)
    semantic_colors = _stable_entity_colors(semantic_keys)
    rgb[~rgb_valid] = 128
    for values in (rows, owners):
        if values.min() < 0 or values.max() > np.iinfo(np.int32).max:
            raise ValueError("source rows or owner IDs exceed lossless PLY int32 storage")
    vertices = np.empty(len(rows), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("canonical_source_row", "<i4"),
        ("owner_id", "<i4"), ("semantic_id", "<i4"), ("display_state", "u1"), ("rgb_valid", "u1")])
    for i, axis in enumerate(("x", "y", "z")):
        vertices[axis] = xyz[:, i]
    for field, values in (("canonical_source_row", rows), ("owner_id", owners),
                          ("semantic_id", semantic), ("display_state", state_codes), ("rgb_valid", rgb_valid)):
        vertices[field] = values
    receipts = {}
    for mode in ("rgb", "instance", "semantic", "state"):
        colors = (rgb if mode == "rgb" else _stable_entity_colors(owners) if mode == "instance"
                  else _stable_entity_colors(semantic) if mode == "semantic"
                  else np.asarray([[128, 128, 128], *state_palette.values()], np.uint8)[state_codes])
        target = output / f"current_{mode}.ply"
        if not target.exists():
            for i, channel in enumerate(("red", "green", "blue")):
                vertices[channel] = colors[:, i]
            temporary = target.with_suffix(".partial.ply")
            _write_ply(temporary, vertices, triangles)
            temporary.replace(target)
        receipts[mode] = {"path": str(target.relative_to(run)), "sha256": _sha256(target), "bytes": target.stat().st_size}
        print(mode, len(rows), "full source rows exported", flush=True)
    _atomic_json(compact / "selected_apartment_map_exports.json", {
        "status": "FOUR_FULL_CURRENT_VIEWS_EXPORTED", "upload_status": "LOCAL_ONLY_POLICY",
        "choice": choice["config_id"], "binding": binding, "source_rows": len(rows),
        "triangles": len(triangles), "GT_read": False, "geometry_edited": False,
        "rgb_source": "original canonical observed_rgb_uint8; invalid RGB shown gray",
        "rgb_valid_rows": int(rgb_valid.sum()), "views": receipts,
        "state_legend": {"1": "current visit", "2": "retained historical visit", "3": "restored historical visit"},
        "state_colors": state_palette,
        "state_counts": {str(k): int(np.count_nonzero(state_codes == k)) for k in state_palette},
        "semantic_colors": {str(k): c.tolist() for k, c in zip(semantic_keys, semantic_colors)},
        "instance_colors": "existing stable numeric owner-ID hash; zero is gray",
        "source_mapping": "canonical_source_row stored in every PLY; all current rows included, no subsampling",
    })


if __name__ == "__main__":
    main()
