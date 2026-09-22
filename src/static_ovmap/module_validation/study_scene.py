"""Shared scene binding and released evaluation after prediction locking."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .boundary_jobs import file_identity, verify_capture
from .contracts import atomic_write_json
from .evaluation import (
    PinnedReleasedEvaluator,
    ReleasedEvaluationAdapter,
    evaluation_rows_json,
)
from .scannet_ground_truth import load_ground_truth
from .scannet_runtime import reusable_job
from .scannet_study import (
    build_native_prediction,
    freeze_projection,
    load_prediction,
    project_values,
)


def bind_scene(scene: str, runtime: dict, config: dict) -> tuple[dict, dict]:
    """Return prediction-side inputs and evaluator-only targets as separate values."""
    output = Path(config["study_root"]) / "scenes" / scene
    mapping_path = Path(runtime["output_root"]) / scene / "mapping_job/receipt.json"
    if not mapping_path.is_file():
        raise RuntimeError(f"MISSING_NATIVE_CAPTURE:{scene}")
    mapping = json.loads(mapping_path.read_text())
    if not reusable_job(mapping_path, mapping["input_identity"]):
        raise ValueError("native capture mapping receipt is incomplete or changed")
    capture_path = Path(mapping["capture_manifest"])
    capture = verify_capture(capture_path, allow_skipped=True)
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as arrays:
        surface = {key: arrays[key] for key in arrays.files}
    targets = load_ground_truth(Path(config["study_root"]) / "annotations" / scene / "receipt.json")
    projection = freeze_projection(surface["surface_xyz"], targets["xyz"], output / "projection")
    native_path = build_native_prediction(capture_path, Path(mapping["native_features"]),
        Path(config["native_text_cache"]), projection, output / "baseline")
    native = load_prediction(native_path)
    targets.update(nearest=projection["nearest"], matched=projection["matched"])
    prediction_inputs = {"scene_id": scene, "capture_path": capture_path, "capture": capture,
        "surface": surface, "native": native, "native_manifest_path": native_path,
        "mapping_receipt_path": mapping_path, "output": output}
    return prediction_inputs, targets


def released_evaluator(upstream: Path, output: Path) -> ReleasedEvaluationAdapter:
    from src.static_ovmap.released_loader import load_released_module

    namespace = load_released_module(Path(upstream) / "scripts/eval_utils.py")
    namespace["init"]("Scannet200")
    return ReleasedEvaluationAdapter(PinnedReleasedEvaluator(namespace, output))


def evaluate_predictions(payloads, targets: dict[str, dict], upstream: Path, output: Path) -> list[dict]:
    if any(not payload.locked for payload in payloads):
        raise ValueError("all scene predictions must be locked before evaluation")
    adapter = released_evaluator(upstream, output / "released")
    rows = adapter.evaluate_many(payloads, targets)
    result = json.loads(evaluation_rows_json(rows))
    atomic_write_json(output / "metrics.json", {"rows": result})
    return result


def audit_native_export(payload, targets: dict, upstream: Path, output: Path) -> dict:
    """Check the actual unchanged native map_pred_mesh mask/rank export against N0."""
    from plyfile import PlyData

    from scripts.evaluation.evaluate_static_ovmap_instances import load_exports

    if not payload.locked or payload.method_id != "N0":
        raise ValueError("native export parity requires a locked N0 prediction")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    owners = project_values(payload.owner_ids, targets["nearest"], targets["matched"])
    semantic = project_values(payload.semantic_labels, targets["nearest"], targets["matched"])
    if np.any(owners > np.iinfo(np.uint16).max):
        raise ValueError("native N0 registry exceeds the unchanged mesh export dtype")
    template = PlyData.read(Path(targets["gt_instance_path"]).parent / "gt_instance_mesh.ply")
    for name, values in (("instance", owners), ("semantic", semantic)):
        mesh = copy.deepcopy(template)
        mesh["vertex"]["label"] = values.astype(np.uint16)
        mesh.write(output / f"predicted_{name}.ply")
    source = Path(upstream) / "scripts/eval_sem_seg.py"
    manifest = load_exports(source)["map_pred_mesh"]({"res_folder": str(output),
        "inst_mesh_f": str(output / "predicted_instance.ply"), "sem_mesh_f": str(output / "predicted_semantic.ply")})
    exported = {}
    ranks = dict(payload.instance_ranks)
    for line in Path(manifest).read_text().splitlines():
        relative, label, rank = line.split()
        owner = int(relative.split("_label-")[0].removeprefix("inst-"))
        mask = np.load(output / relative, allow_pickle=False).reshape(-1)
        if not np.array_equal(mask, owners == owner) or rank != f"{ranks[owner]:.6f}":
            raise ValueError("N0 differs from unchanged native mask/rank export")
        expected = int(np.unique(payload.semantic_labels[payload.owner_ids == owner])[0])
        if int(label) != expected:
            raise ValueError("N0 semantic label differs from native export")
        exported[owner] = {"class_id": expected, "rank": rank, "mask_pixels": int(mask.sum())}
    expected_owners = {int(owner) for owner, count in zip(*np.unique(owners, return_counts=True), strict=True)
                       if owner > 0 and count >= 100 and np.any(semantic[owners == owner] > 0)}
    if set(exported) != expected_owners:
        raise ValueError("N0 export eligibility differs from unchanged native export")
    result = {"status": "COMPLETE", "exact_mask_and_serialized_rank_parity": True,
        "prediction_key": payload.prediction_key, "scene_id": payload.scene_id,
        "exported_instances": len(exported), "instances": exported, "source": file_identity(source),
        "native_export_manifest": file_identity(manifest)}
    atomic_write_json(output / "receipt.json", result)
    return result
