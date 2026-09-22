"""Annotation conversion/evaluation only; never imported by prediction adapters."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _write_npz
from .scannet_runtime import reusable_job
from .scannet_study import scannet_vocabulary


def _unchanged_functions(path: Path, names: set[str], namespace: dict) -> dict:
    """Compile exact function ASTs, avoiding unrelated visualization/model imports."""
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != names:
        raise ValueError(f"pinned native interface changed: {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- pinned local native code, unchanged AST
    return namespace


def prepare_ground_truth(upstream: Path, raw: Path, scene: str, label_table: Path,
                         output: Path) -> Path:
    """Run the unchanged ScanNet200 conversion and native semantic-instance export.

    Object 0 intentionally remains 0 and is ignored by the released exporter.
    No annotation is projected/cropped to predicted support.
    """
    from scripts.evaluation.evaluate_static_ovmap_instances import load_exports

    upstream, raw, label_table, output = [Path(path).resolve() for path in (upstream, raw, label_table, output)]
    source = upstream / "scripts"
    raw_paths = [raw / f"{scene}{suffix}" for suffix in (
        "_vh_clean_2.labels.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json")]
    source_paths = [source / name for name in (
        "datasets/preprocess_gt_mesh.py", "utils/semantic_const.py", "visualizations/vis_utils.py", "eval_sem_seg.py")]
    inputs = [file_identity(path) for path in [*raw_paths, label_table, *source_paths, Path(__file__)]]
    identity = canonical_digest({"scene_id": scene, "inputs": inputs, "schema": 1})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return receipt_path
    if receipt_path.exists():
        raise ValueError("ground-truth input or converted output changed; choose a new output root")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / "native_input"
    staged_scene = staging / scene
    staged_scene.mkdir(parents=True, exist_ok=True)
    for original, link in [(path, staged_scene / path.name) for path in raw_paths] + [
            (label_table, staging / "scannetv2-labels.combined.tsv")]:
        if link.is_symlink():
            if link.resolve() != original:
                raise ValueError("native input staging points to different annotations")
        else:
            link.symlink_to(original)

    # The converter's internal '..utils.semantic_const' import needs its actual
    # scripts package boundary, without importing unrelated top-level modules.
    package_name = "_ovimap_scannet_gt_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    palette = _unchanged_functions(source / "visualizations/vis_utils.py", {"get_new_pallete"}, {"np": np})
    namespace = {"__package__": package_name + ".datasets", "np": np, "copy": copy,
        "json": json, "PlyData": PlyData, "pjoin": os.path.join, "get_new_pallete": palette["get_new_pallete"]}
    converter = _unchanged_functions(source / "datasets/preprocess_gt_mesh.py", {"convert_scannet_mesh"}, namespace)
    converter["convert_scannet_mesh"](str(staged_scene), scene, str(output), map_scannet200=True)
    combined_path = load_exports(source / "eval_sem_seg.py")["map_gt_mesh"]({
        "res_folder": str(output), "inst_mesh_f": str(output / "gt_instance_mesh.ply"),
        "sem_mesh_f": str(output / "gt_semantic_mesh.ply")})
    semantic = PlyData.read(output / "gt_semantic_mesh.ply")["vertex"].data
    instance = PlyData.read(output / "gt_instance_mesh.ply")["vertex"].data
    original = PlyData.read(raw_paths[0])["vertex"].data
    xyz = np.column_stack([semantic[axis] for axis in "xyz"]).astype(np.float32)
    if not np.array_equal(xyz, np.column_stack([original[axis] for axis in "xyz"]).astype(np.float32)):
        raise ValueError("native GT conversion changed the whole-scene source coordinates")
    _, valid_ids = scannet_vocabulary(upstream)
    array_path = output / "annotations.npz"
    _write_npz(array_path, {"xyz": xyz, "gt_semantic": semantic["label"],
        "gt_instance": instance["label"], "valid_ids": np.asarray(valid_ids, np.int64)})
    atomic_write_json(receipt_path, {"status": "COMPLETE", "scene_id": scene, "input_identity": identity,
        "inputs": inputs, "gt_instance_path": str(combined_path), "arrays_path": str(array_path),
        "outputs": [file_identity(output / name) for name in (
            "gt_semantic_mesh.ply", "gt_instance_mesh.ply", "gt_sem_inst_id.npy", "annotations.npz")],
        "source_rows": len(xyz), "object_zero_policy": "PRESERVE_NATIVE_IGNORE", "cropped_to_prediction": False})
    return receipt_path


def canonical_metrics(payload, projected_owners: np.ndarray, ground_truth: dict) -> dict:
    from src.evaluation.static_projected_instances import projected_instance_metrics

    return projected_instance_metrics(projected_owners, ground_truth["gt_instance"],
                                       dict(payload.instance_ranks), min_region=100)


def load_ground_truth(receipt_path: Path) -> dict:
    receipt = json.loads(Path(receipt_path).read_text())
    if not reusable_job(Path(receipt_path), receipt["input_identity"]):
        raise ValueError("ground-truth receipt or converted payload changed")
    for row in receipt["inputs"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("ground-truth source changed after conversion")
    with np.load(receipt["arrays_path"], allow_pickle=False) as arrays:
        return {**{key: arrays[key] for key in arrays.files},
                "gt_instance_path": Path(receipt["gt_instance_path"]),
                "canonical_metrics": canonical_metrics}
