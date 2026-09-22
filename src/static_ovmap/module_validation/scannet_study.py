"""Prediction-side ScanNet adapters; annotation targets live in a separate module."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .assets import sha256_file
from .contracts import atomic_write_json, canonical_digest
from .evaluation import (
    GeometryIdentity,
    PredictionPayload,
    validate_prediction_invariants,
)
from .native_capture import _array_digest, _write_npz


def save_prediction(payload: PredictionPayload, output: Path) -> Path:
    manifest = payload.manifest()  # Requires an already locked, GT-free payload.
    output = Path(output)
    path = output / "manifest.json"
    if path.exists():
        if load_prediction(path).record_key != payload.record_key:
            raise ValueError("completed prediction changed; choose a new output root")
        return path
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "prediction.npz"
    _write_npz(arrays, {"owner_ids": payload.owner_ids, "semantic_labels": payload.semantic_labels})
    manifest["arrays"] = {"path": arrays.name, "sha256": sha256_file(arrays)}
    atomic_write_json(path, manifest)
    return path


def load_prediction(path: Path) -> PredictionPayload:
    path = Path(path)
    manifest = json.loads(path.read_text())
    arrays_path = path.parent / manifest["arrays"]["path"]
    if sha256_file(arrays_path) != manifest["arrays"]["sha256"]:
        raise ValueError("prediction array hash differs from the locked manifest")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        payload = PredictionPayload(manifest["method_id"], manifest["branch"], manifest["scene_id"],
            GeometryIdentity(**manifest["geometry"]), arrays["owner_ids"], arrays["semantic_labels"],
            tuple(tuple(row) for row in manifest["instance_ranks"]), manifest["logical_cost"], manifest["metadata"])
    payload.lock()
    if payload.record_key != manifest["record_key"] or payload.prediction_key != manifest["prediction_key"]:
        raise ValueError("prediction manifest identity changed")
    return payload


def relabel_prediction(native: PredictionPayload, method: str, branch: str,
                       labels: dict[int, int], costs: dict, metadata: dict | None = None) -> PredictionPayload:
    if not native.locked:
        raise ValueError("native reference must be locked before relabeling")
    if not set(labels) <= set(map(int, np.unique(native.owner_ids))) - {0}:
        raise ValueError("semantic replacement leaves the frozen native registry")
    semantic = native.semantic_labels.copy()
    for owner, label in labels.items():
        semantic[native.owner_ids == owner] = label
    payload = replace(native, method_id=method, branch=branch, semantic_labels=semantic,
                      logical_cost=costs, metadata={"native_record_key": native.record_key, **(metadata or {})})
    validate_prediction_invariants(payload, native)
    payload.lock()
    return payload


def prepare_text_cache(upstream: Path, model_path: Path, output: Path, *, device: str = "cpu") -> Path:
    """Compute official raw-name/canonical embeddings in this model's own FP32 space."""
    import torch
    import transformers

    from .boundary_jobs import file_identity
    from .region_evidence import FrozenSiglipBackend
    from .scannet_runtime import _tree_inputs, reusable_job

    names, valid_ids = scannet_vocabulary(upstream)
    canonical = ("object", "things", "stuff", "texture")
    inputs = _tree_inputs(Path(model_path)) + [file_identity(path) for path in (
        Path(upstream) / "scripts/utils/semantic_const.py", Path(upstream) / "scripts/utils/text_embedding.py",
        Path(__file__), Path(__file__).with_name("region_evidence.py"))]
    identity = canonical_digest({"inputs": inputs, "names": names, "canonical": canonical,
        "valid_ids": valid_ids, "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": "float32", "padding": "max_length", "max_length": 64, "batch_size": 16})
    output = Path(output)
    receipt = output / "receipt.json"
    if reusable_job(receipt, identity):
        return output / "text.npz"
    if receipt.exists():
        raise ValueError("frozen text inputs changed; choose a new output root")
    started = time.monotonic()
    backend = FrozenSiglipBackend.from_local(str(model_path), device=device)
    if any(parameter.dtype != torch.float32 for parameter in backend.model.parameters() if parameter.is_floating_point()):
        raise ValueError("text model did not load in frozen FP32 precision")
    text = np.concatenate([backend.encode_texts(names[index:index + 16]) for index in range(0, len(names), 16)])
    canon = backend.encode_texts(canonical)
    if not np.isfinite(text).all() or not np.isfinite(canon).all():
        raise ValueError("text model returned nonfinite embeddings")
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "text.npz"
    _write_npz(array_path, {"text_embeddings": text, "canonical_embeddings": canon,
        "class_names": np.asarray(names), "canonical_phrases": np.asarray(canonical), "valid_ids": np.asarray(valid_ids)})
    atomic_write_json(receipt, {"status": "COMPLETE", "input_identity": identity, "inputs": inputs,
        "outputs": [file_identity(array_path)], "model_path": str(Path(model_path).resolve()),
        "class_count": len(names), "feature_dimension": text.shape[1], "dtype": "float32", "device": device,
        "elapsed_seconds": time.monotonic() - started, "text_inputs": len(names) + len(canonical),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__})
    return array_path


def scannet_vocabulary(upstream: Path) -> tuple[tuple[str, ...], tuple[int, ...]]:
    from src.static_ovmap.released_loader import load_released_module

    constants = load_released_module(Path(upstream) / "scripts/utils/semantic_const.py")
    names = tuple(constants["CLASS_LABELS_200"])
    ids = tuple(int(value) for value in constants["VALID_CLASS_IDS_200"])
    if len(names) != 200 or len(ids) != 200 or len(set(ids)) != 200 or 0 in ids:
        raise ValueError("pinned ScanNet200 vocabulary is inconsistent")
    return names, ids


def native_readout(saved: dict, text: np.ndarray, canonical: np.ndarray,
                   valid_ids: tuple[int, ...]) -> dict[int, dict]:
    """Read the mapper's already filtered/top10 pickle in its saved native order.

    Preserve FP32 Torch aggregation and the canonical-relative classifier from
    mesh_postprocess_utils. Re-sorting here would change native tied-area order.
    """
    import torch

    text = torch.as_tensor(text, dtype=torch.float32)
    canonical = torch.as_tensor(canonical, dtype=torch.float32)
    if text.ndim != 2 or text.shape[0] != len(valid_ids) or canonical.ndim != 2 or canonical.shape[1] != text.shape[1]:
        raise ValueError("native text spaces do not align")
    if not torch.isfinite(text).all() or not torch.isfinite(canonical).all():
        raise ValueError("native text spaces must be finite")
    cosine = torch.nn.CosineSimilarity(dim=-1)
    result = {}
    for owner, row in saved.items():
        frames = list(map(int, row["frame_id"]))
        features = torch.as_tensor(row["feat"], dtype=torch.float32)
        areas = torch.as_tensor(row["vis_area"])
        if (not 1 <= len(frames) <= 10 or features.shape != (len(frames), text.shape[1])
                or areas.shape != (len(frames),) or not torch.isfinite(features).all()
                or not torch.isfinite(areas).all() or torch.any(areas <= 0)):
            raise ValueError("native saved observations are malformed")
        output = {"class_id": 0, "retained_observations": len(frames), "used_frames": [],
                  "feature": None, "status": "INSUFFICIENT_NATIVE_OBSERVATIONS"}
        if len(frames) >= 2:
            weights = areas[-8:]
            weights = weights / (torch.sum(weights) + 1e-6)
            feature = torch.sum(features[-8:] * weights.unsqueeze(-1), dim=0)
            query_similarity = cosine(feature, text).unsqueeze(0)
            canonical_similarity = cosine(feature, canonical).unsqueeze(1)
            relative = torch.exp(query_similarity) / (torch.exp(query_similarity) + torch.exp(canonical_similarity))
            index = int(torch.argmax(torch.min(relative, dim=0).values))
            output.update(class_id=int(valid_ids[index]), feature=feature.numpy(),
                          used_frames=frames[-8:], status="AVAILABLE")
        result[int(owner)] = output
    return result


def project_values(values: np.ndarray, nearest: np.ndarray, matched: np.ndarray) -> np.ndarray:
    values, nearest, matched = np.asarray(values), np.asarray(nearest), np.asarray(matched)
    if nearest.ndim != 1 or matched.shape != nearest.shape or matched.dtype != bool:
        raise ValueError("frozen projection arrays are misaligned")
    if np.any(nearest[matched] < 0) or np.any(nearest[matched] >= len(values)):
        raise ValueError("frozen projection leaves source rows")
    result = np.zeros(nearest.shape, dtype=values.dtype)
    result[matched] = values[nearest[matched]]
    return result


def native_surface_readout(colors: np.ndarray, faces: np.ndarray, saved: dict,
                           readout: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Exactly preserve the native triangle-first-color paint and dict order.

    Numerical voxel owners are separately captured for diagnostics. Substituting
    them here changes native instance masks at mixed-label triangle boundaries.
    """
    colors, faces = np.asarray(colors), np.asarray(faces, dtype=np.int64)
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.dtype != np.uint8:
        raise ValueError("native surface colors must be uint8 RGB")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(colors)):
        raise ValueError("native triangle indices leave source rows")
    color_to_instance = {}
    for owner, row in saved.items():
        evidence = readout[int(owner)]
        if evidence["status"] == "AVAILABLE":
            color_to_instance[tuple(map(int, row["color"]))] = (int(owner), evidence["class_id"])
    owners = np.zeros(len(colors), dtype=np.int64)
    labels = np.zeros(len(colors), dtype=np.int64)
    triangle_colors = colors[faces[:, 0]]
    for color, (owner, label) in color_to_instance.items():
        vertices = faces[np.all(triangle_colors == color, axis=1)].reshape(-1)
        owners[vertices] = owner
        labels[vertices] = label
    return owners, labels


def native_ranks(owners: np.ndarray, labels: dict[int, int], nearest: np.ndarray,
                 matched: np.ndarray) -> tuple[tuple[int, float], ...]:
    """Released map_pred_mesh ranks, frozen after its exact .6f serialization.

    Owners excluded by native class0/min100 export retain a deterministic zero
    rank in the full numerical registry. Relabeling never recalculates ranks.
    """
    projected = project_values(owners, nearest, matched)
    ids, counts = np.unique(projected, return_counts=True)
    areas = {int(owner): int(count) for owner, count in zip(ids, counts) if owner > 0 and count >= 100}
    maxima = {}
    for owner, area in areas.items():
        label = labels.get(owner, 0)
        maxima[label] = max(maxima.get(label, 0), area)
    return tuple((int(owner), float(f"{areas[int(owner)] / maxima[labels[int(owner)]]:.6f}")
                  if int(owner) in areas and labels.get(int(owner), 0) != 0 else 0.0)
                 for owner in np.unique(owners) if owner > 0)


def freeze_projection(xyz: np.ndarray, target_xyz: np.ndarray, output: Path) -> dict:
    """Bind unchanged Open3D float32 1NN and strict squared-distance < .05²."""
    import open3d.core as o3c

    xyz = np.asarray(xyz, dtype=np.float32)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    for array in (xyz, target_xyz):
        if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.isfinite(array).all():
            raise ValueError("projection coordinates must be nonempty finite [N,3]")
    identity = canonical_digest({"source": _array_digest(xyz), "target": _array_digest(target_xyz),
        "rule": "Open3D_float32_1NN_squared_distance_strict_lt_0.05_squared"})
    output = Path(output)
    manifest_path, array_path = output / "manifest.json", output / "projection.npz"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("frozen projection input changed")
        if sha256_file(array_path) != manifest["sha256"]:
            raise ValueError("frozen projection payload changed")
        with np.load(array_path, allow_pickle=False) as arrays:
            return {"identity": identity, **{key: arrays[key] for key in arrays.files}}
    search = o3c.nns.NearestNeighborSearch(o3c.Tensor(np.ascontiguousarray(xyz)))
    search.knn_index()
    indices, squared = search.knn_search(o3c.Tensor(np.ascontiguousarray(target_xyz)), 1)
    arrays = {"nearest": indices.numpy().reshape(-1), "matched": (squared < 0.05**2).numpy().reshape(-1),
              "distance_squared": squared.numpy().reshape(-1)}
    output.mkdir(parents=True, exist_ok=True)
    _write_npz(array_path, arrays)
    atomic_write_json(manifest_path, {"identity": identity, "sha256": sha256_file(array_path),
        "source_rows": len(xyz), "target_rows": len(target_xyz), "matched_rows": int(arrays["matched"].sum())})
    return {"identity": identity, **arrays}


def build_native_prediction(capture_path: Path, feature_path: Path, text_path: Path,
                            projection: dict, output: Path) -> Path:
    """Lock N0 from original full native observations and exact native mesh paint."""
    from plyfile import PlyData

    from .boundary_jobs import capture_inputs, file_identity, verify_capture
    from .scannet_runtime import reusable_job

    capture_path, feature_path, text_path, output = map(Path, (capture_path, feature_path, text_path, output))
    capture = verify_capture(capture_path, allow_skipped=True)
    if "tsdf" not in capture or not capture.get("native_owner_mesh_parity", {}).get("exact"):
        raise ValueError("native baseline requires true TSDF identity and exact owner/mesh parity")
    text_receipt = json.loads(text_path.with_name("receipt.json").read_text())
    if not reusable_job(text_path.with_name("receipt.json"), text_receipt["input_identity"]):
        raise ValueError("native text cache is not bound to a complete receipt")
    for row in text_receipt["inputs"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("native text cache source changed")
    inputs = [file_identity(row["path"]) for row in capture_inputs(capture_path, capture)]
    inputs.extend(file_identity(path) for path in (feature_path, text_path, Path(__file__)))
    identity = canonical_digest({"inputs": inputs, "projection": projection["identity"], "schema": 1})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return output / "N0/manifest.json"
    if receipt_path.exists():
        raise ValueError("native prediction input changed; choose a new output root")
    with feature_path.open("rb") as handle:
        saved = pickle.load(handle)  # Hash-bound, locally produced native mapper artifact.
    with np.load(text_path, allow_pickle=False) as text:
        readout = native_readout(saved, text["text_embeddings"], text["canonical_embeddings"], tuple(text["valid_ids"]))
    with np.load(capture_path.parent / capture["surface"]["path"], allow_pickle=False) as surface:
        xyz, faces, voxel_owners = surface["surface_xyz"], surface["surface_faces"], surface["original_owner"]
    mesh = PlyData.read(capture["surface"]["source_mesh_path"])["vertex"].data
    colors = np.column_stack([mesh[name] for name in ("red", "green", "blue")])
    owners, semantic = native_surface_readout(colors, faces, saved, readout)
    owner_labels = {int(owner): int(np.unique(semantic[owners == owner])[0]) for owner in np.unique(owners) if owner > 0}
    ranks = native_ranks(owners, owner_labels, projection["nearest"], projection["matched"])
    geometry = GeometryIdentity(_array_digest(xyz), _array_digest(faces), capture["tsdf"]["sha256"],
                                projection["identity"], len(xyz))
    attempts = sum(len(frame["native_selected_request_ids"]) for frame in capture["frames"])
    payload = PredictionPayload("N0", "N0", capture["scene_id"], geometry, owners, semantic, ranks,
        {"attempts": attempts, "crop_inputs": 6 * attempts}, {"capture_identity": capture["identity"],
        "native_features_sha256": sha256_file(feature_path), "text_cache_sha256": sha256_file(text_path),
        "readout": "native_full_observations_top10_last8_canonical_triangle_first_color"})
    payload.lock()
    manifest = save_prediction(payload, output / "N0")
    available = [owner for owner in readout if readout[owner]["status"] == "AVAILABLE"]
    readout_arrays = output / "native_readout.npz"
    width = next((len(readout[owner]["feature"]) for owner in available), 0)
    _write_npz(readout_arrays, {"owner_ids": np.asarray(available, np.int64),
        "features": np.stack([readout[owner]["feature"] for owner in available]) if available else np.empty((0, width), np.float32)})
    ledger = output / "native_readout.json"
    atomic_write_json(ledger, {"instances": {str(owner): {key: value for key, value in row.items() if key != "feature"}
        for owner, row in readout.items()}, "numerical_voxel_owner_count": len(set(voxel_owners.tolist()) - {0}),
        "native_readout_owner_count": len(owner_labels), "owner_rows_changed_by_native_readout": int(np.sum(owners != voxel_owners)),
        "mixed_numerical_owner_triangles": int(np.any(voxel_owners[faces] != voxel_owners[faces[:, :1]], axis=1).sum()),
        "GT_input": False})
    atomic_write_json(receipt_path, {"status": "COMPLETE", "input_identity": identity, "inputs": inputs,
        "prediction_manifest": str(manifest), "outputs": [file_identity(path) for path in (
            manifest, manifest.parent / "prediction.npz", readout_arrays, ledger)], "scene_id": capture["scene_id"]})
    return manifest
