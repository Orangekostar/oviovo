"""Actual final-mask native six-crop inference and G semantic readout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import sha256_file
from .boundary_jobs import file_identity, verify_capture
from .contracts import atomic_write_json, canonical_digest
from .native_capture import RegionRequest, _array_digest, _write_npz
from .region_evidence import FrozenSiglipBackend, native_crops
from .scannet_runtime import _tree_inputs, reusable_job
from .semantic_selector import area_readout


def encode_static_requests(manifest_path, config, cache_root, output, *, device="cuda"):
    import torch
    import transformers

    manifest_path, cache_root, output = map(Path, (manifest_path, cache_root, output))
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"] != canonical_digest({key: value for key, value in manifest.items() if key != "identity"}):
        raise ValueError("static request manifest identity changed")
    capture_path = Path(manifest["capture"]["path"])
    if sha256_file(capture_path) != manifest["capture"]["sha256"]:
        raise ValueError("static region capture changed")
    capture = verify_capture(capture_path, allow_skipped=True)
    frames = {row["frame_id"]: row for row in capture["frames"]}
    model_inputs = _tree_inputs(Path(config["native_model"])) + [file_identity(path) for path in (
        Path(__file__), Path(__file__).with_name("region_evidence.py"))]
    model_identity = canonical_digest({"inputs": model_inputs, "torch": torch.__version__,
        "transformers": transformers.__version__, "device": device, "dtype": "float32", "schema": "native_six_v1"})
    inputs = [file_identity(manifest_path), *model_inputs]
    identity = canonical_digest({"manifest": manifest["identity"], "model": model_identity})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return json.loads(receipt_path.read_text())
    if receipt_path.exists():
        raise ValueError("completed G inference changed; choose a new output root")
    backend = None
    load_seconds = 0.
    records, paths = {}, []
    physical_attempts = 0
    for request_id, item in manifest["requests"].items():
        request = RegionRequest.from_dict(item["request"])
        frame = frames[request.frame_id]
        masks_path = Path(item["masks"]["path"])
        if sha256_file(masks_path) != item["masks"]["sha256"]:
            raise ValueError("static request masks changed")
        request_identity = canonical_digest({"model_identity": model_identity,
            "request": request.to_dict(), "masks_sha256": item["masks"]["sha256"]})
        path = cache_root / model_identity / f"{request_identity}.json"
        cached = reusable_job(path, request_identity)
        if not cached:
            if path.exists():
                raise ValueError("G request cache changed")
            with np.load(masks_path, allow_pickle=False) as arrays:
                target, union = arrays["target"], arrays["union"]
            if _array_digest(target) != request.target_mask_sha256 or _array_digest(union) != request.native_union_mask_sha256:
                raise ValueError("G request mask identities differ")
            image_path = capture_path.parent / frame["rgb_path"]
            if sha256_file(image_path) != request.image_sha256:
                raise ValueError("G request image identity changed")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            crops = native_crops(rgb, target, union, request.bbox_xyxy)
            if backend is None:
                start = time.monotonic()
                backend = FrozenSiglipBackend.from_local(config["native_model"], device=device)
                if any(row.dtype != torch.float32 for row in backend.model.parameters() if row.is_floating_point()):
                    raise ValueError("G native encoder must use FP32")
                load_seconds += time.monotonic() - start
            started = time.monotonic()
            physical_attempts += 1
            row = {"status": "COMPLETE", "input_identity": request_identity, "request_id": request_id,
                "crop_inputs": 6, "inference_status": "COMPLETE"}
            payload = {}
            try:
                vectors = np.asarray(backend.encode_images(crops.legacy_six), np.float64)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if vectors.ndim != 2 or vectors.shape[0] != 6 or not np.isfinite(vectors).all() or np.any(norms <= 0):
                    raise RuntimeError("native six-crop feature is invalid")
                vectors = vectors / norms
                payload = {"vectors": vectors, "feature": vectors.mean(axis=0)}
            except RuntimeError as error:
                row.update(inference_status="UNAVAILABLE_TECHNICAL_FAILURE", error=str(error))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["elapsed_seconds"] = time.monotonic() - started
            arrays_path = path.with_suffix(".npz")
            _write_npz(arrays_path, payload)
            row.update(arrays_path=str(arrays_path), outputs=[file_identity(arrays_path)])
            atomic_write_json(path, row)
        else:
            row = json.loads(path.read_text())
        records[request_id] = {"receipt": file_identity(path), "physical_reuse": cached}
        paths += [file_identity(path), *row["outputs"]]
    index_path = output / "request_index.json"
    atomic_write_json(index_path, {"requests": records})
    paths.append(file_identity(index_path))
    result = {"status": "COMPLETE", "input_identity": identity, "inputs": inputs, "outputs": paths,
        "model_identity": model_identity, "request_count": len(records), "logical_crop_inputs": 6 * len(records),
        "physical_attempts_this_invocation": physical_attempts, "model_load_seconds_this_invocation": load_seconds}
    atomic_write_json(receipt_path, result)
    return result


def load_static_records(output):
    output = Path(output)
    receipt = json.loads((output / "receipt.json").read_text())
    if not reusable_job(output / "receipt.json", receipt["input_identity"]):
        raise ValueError("G static inference receipt changed")
    for row in receipt["inputs"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("G inference source changed")
    index = json.loads((output / "request_index.json").read_text())
    result = {}
    for request_id, item in index["requests"].items():
        row = json.loads(Path(item["receipt"]["path"]).read_text())
        with np.load(row["arrays_path"], allow_pickle=False) as arrays:
            result[request_id] = {**row, "status": row["inference_status"], **{key: arrays[key] for key in arrays.files}}
    return result


def final_mask_labels(native, owner_ids, manifest, records, text, valid_ids):
    if not native.locked:
        raise ValueError("G semantic readout requires locked native reference")
    labels, ledger = {}, {}
    for owner in sorted(set(map(int, np.unique(owner_ids))) - {0}):
        unchanged = np.array_equal(owner_ids == owner, native.owner_ids == owner)
        incumbent = int(native.semantic_labels[np.flatnonzero(native.owner_ids == owner)[0]]) if unchanged else 0
        ids = manifest["views"].get(f"owner:{owner}", [])
        good = [key for key in ids if records[key]["status"] == "COMPLETE"]
        if good:
            result = area_readout(np.stack([records[key]["feature"] for key in good]),
                [manifest["requests"][key]["request"]["visible_target_pixels"] for key in good], text, valid_ids=valid_ids)
            label, status = result.label_id, "COMPLETE"
        else:
            label, status = incumbent, "UNAVAILABLE_UNCHANGED_KEEP_NATIVE" if unchanged else "UNAVAILABLE_NEW_COMPONENT_UNKNOWN"
        labels[owner] = label
        ledger[owner] = {"label_id": label, "status": status, "successful_views": len(good),
            "request_ids": ids, "unchanged_source_mask": unchanged}
    return labels, ledger
