"""Evaluation-only unchanged released Replica mesh/label conversion."""

import hashlib
import os
import sys
import types
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.evaluation.evaluate_static_ovmap_instances import load_exports
from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)
from src.static_ovmap.module_validation.native_capture import _write_npz
from src.static_ovmap.module_validation.scannet_ground_truth import _unchanged_functions
from src.static_ovmap.module_validation.scannet_runtime import reusable_job

from .protocol import vocabulary


def prepare_ground_truth(upstream, raw, scene, output):
    upstream, raw, output = (Path(p).resolve() for p in (upstream, raw, output))
    source = upstream / "scripts"
    semantic_dir = source / "datasets/replica_gt_semantics"
    instance_dir = source / "datasets/replica_gt_instances"
    mesh_path = raw / f"{scene}_mesh.ply"
    inputs = [file_identity(path) for path in (
        mesh_path, semantic_dir / f"semantic_labels_{scene}.txt",
        instance_dir / f"instance_labels_{scene}.txt",
        source / "datasets/preprocess_gt_mesh.py", source / "utils/semantic_const.py",
        source / "utils/mesh_postprocess_utils.py", source / "visualizations/vis_utils.py",
        source / "eval_sem_seg.py", Path(__file__),
    )]
    identity = canonical_digest({"scene_id": scene, "inputs": inputs, "dataset": "Replica"})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return receipt_path
    if receipt_path.exists():
        raise ValueError("Replica annotation source/output changed; use a new root")
    output.mkdir(parents=True, exist_ok=True)
    package_name = "_ovimap_replica_gt_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    palette = _unchanged_functions(source / "visualizations/vis_utils.py", {"get_new_pallete"}, {"np": np})
    writer = _unchanged_functions(source / "utils/mesh_postprocess_utils.py", {"write_ply_with_labels"}, {"np": np, "PlyData": PlyData, "PlyElement": PlyElement})
    namespace = {"__package__": package_name + ".datasets", "np": np, "PlyData": PlyData,
                 "pjoin": os.path.join, "get_new_pallete": palette["get_new_pallete"],
                 "write_ply_with_labels": writer["write_ply_with_labels"]}
    converter = _unchanged_functions(source / "datasets/preprocess_gt_mesh.py", {"convert_replica_sem_inst_mesh"}, namespace)
    converter["convert_replica_sem_inst_mesh"](str(raw / scene), scene, str(output), str(semantic_dir), str(instance_dir))
    combined_path = load_exports(source / "eval_sem_seg.py")["map_gt_mesh"]({
        "res_folder": str(output), "inst_mesh_f": str(output / "gt_instance_mesh.ply"),
        "sem_mesh_f": str(output / "gt_semantic_mesh.ply")})
    semantic = PlyData.read(output / "gt_semantic_mesh.ply")["vertex"].data
    instance = PlyData.read(output / "gt_instance_mesh.ply")["vertex"].data
    original = PlyData.read(mesh_path)["vertex"].data
    xyz = np.column_stack([semantic[axis] for axis in "xyz"]).astype(np.float32)
    if not np.array_equal(xyz, np.column_stack([original[axis] for axis in "xyz"]).astype(np.float32)):
        raise ValueError("released Replica conversion changed source coordinates")
    _, valid_ids, _ = vocabulary(upstream)
    if not set(np.unique(semantic["label"])) <= {0, *valid_ids}:
        raise ValueError("released Replica labels outside declared vocabulary")
    array_path = output / "annotations.npz"
    _write_npz(array_path, {"xyz": xyz, "gt_semantic": semantic["label"],
        "gt_instance": instance["label"], "valid_ids": np.asarray(valid_ids, np.int64)})
    atomic_write_json(receipt_path, {"status": "COMPLETE", "scene_id": scene,
        "input_identity": identity, "inputs": inputs, "gt_instance_path": str(combined_path),
        "arrays_path": str(array_path), "outputs": [file_identity(output / name) for name in (
            "gt_semantic_mesh.ply", "gt_instance_mesh.ply", "gt_sem_inst_id.npy", "annotations.npz")],
        "source_rows": len(xyz), "object_zero_policy": "PRESERVE_NATIVE_IGNORE",
        "cropped_to_prediction": False})
    return receipt_path
