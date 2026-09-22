"""Lazy native six-crop acquisition; persistent cache access follows debit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .native_capture import _array_digest, _write_npz
from .query_state import AcquisitionPayload
from .region_evidence import FrozenSiglipBackend, native_crops
from .scannet_runtime import _tree_inputs, reusable_job


class NativeQueryLoader:
    """Pass this callable only to FeatureStore; it has no ranking interface."""

    def __init__(self, frames, config, cache_root, *, device="cuda"):
        import torch
        import transformers

        self.frames, self.config, self.cache_root, self.device = frames, config, Path(cache_root), device
        self.inputs = _tree_inputs(Path(config["native_model"])) + [file_identity(path) for path in (
            Path(__file__), Path(__file__).with_name("region_evidence.py"))]
        self.model_identity = canonical_digest({"inputs": self.inputs, "dtype": "float32", "device": device,
            "torch": torch.__version__, "transformers": transformers.__version__, "native_crops": 6})
        self.backend = None
        self.model_loads, self.model_load_seconds = 0, 0.
        self.used_receipts = {}

    def __call__(self, candidate):
        import torch

        current = self.frames.current
        if current is None or candidate.frame_id != current["frame"]["frame_id"]:
            raise ValueError("query loader may read only a paid request from the current frame")
        request = current["requests"][candidate.request_id]
        if (request.target_mask_sha256 != _array_digest(candidate.global_mask)
                or request.native_union_mask_sha256 != _array_digest(candidate.union_mask)):
            raise ValueError("paid request masks differ from native capture")
        identity = canonical_digest({"model_identity": self.model_identity, "request": request.to_dict()})
        path = self.cache_root / self.model_identity / f"{identity}.json"
        cached = reusable_job(path, identity)
        if not cached:
            if path.exists():
                raise ValueError("paid query cache payload changed")
            image_path = self.frames.path.parent / current["frame"]["rgb_path"]
            if sha256_file(image_path) != request.image_sha256:
                raise ValueError("paid query RGB image changed")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            crops = native_crops(rgb, candidate.global_mask, candidate.union_mask, candidate.bbox_xyxy)
            if self.backend is None:
                start = time.monotonic()
                self.backend = FrozenSiglipBackend.from_local(self.config["native_model"], device=self.device)
                if any(parameter.dtype != torch.float32 for parameter in self.backend.model.parameters() if parameter.is_floating_point()):
                    raise ValueError("query native encoder must remain FP32")
                self.model_load_seconds += time.monotonic() - start
                self.model_loads += 1
            start = time.monotonic()
            arrays, error = {}, None
            try:
                vectors = np.asarray(self.backend.encode_images(crops.legacy_six), np.float64)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if vectors.ndim != 2 or vectors.shape[0] != 6 or not np.isfinite(vectors).all() or np.any(norms <= 0):
                    raise RuntimeError("native six-crop features are nonfinite or malformed")
                vectors = vectors / norms
                feature = vectors.mean(axis=0)
                if np.linalg.norm(feature) <= 0:
                    raise RuntimeError("native six-crop mean is zero")
                arrays = {"crop_features": vectors, "feature": feature}
            except RuntimeError as exc:
                error = str(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = time.monotonic() - start
            array_path = path.with_suffix(".npz")
            _write_npz(array_path, arrays)
            row = {"status": "COMPLETE", "input_identity": identity, "model_identity": self.model_identity,
                "request": request.to_dict(), "crop_inputs": 6, "elapsed_seconds": elapsed,
                "inference_status": "COMPLETE" if error is None else "UNAVAILABLE_TECHNICAL_FAILURE", "error": error,
                "inputs": self.inputs + [file_identity(image_path)], "outputs": [file_identity(array_path)],
                "arrays_path": str(array_path)}
            atomic_write_json(path, row)
        else:
            row = json.loads(path.read_text())
        self.used_receipts[candidate.request_id] = file_identity(path)
        feature = None
        if row["inference_status"] == "COMPLETE":
            with np.load(row["arrays_path"], allow_pickle=False) as arrays:
                feature = arrays["feature"]
        return AcquisitionPayload(feature, row["crop_inputs"], 0. if cached else row["elapsed_seconds"],
                                  row["error"], physical_cache_hit=cached)
