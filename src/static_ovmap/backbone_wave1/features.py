"""Content-addressed raw encoder vectors with separate lineage receipts."""

from pathlib import Path
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.m2_reviewer_study.binding import InputIndex
from .binding import read
from .runtime import exclusive_lock


class TensorEncoderCache:
    """Cache a whole unchanged batch; preserve each caller's normalization."""

    def __init__(self, model, model_identity, root):
        self.model, self.model_identity, self.root = model, model_identity, Path(root)
        self.original = model.get_image_features
        self.index = InputIndex()
        self.physical_encodings = 0
        self.physical_calls = 0
        self.forward_attempts = 0
        self.physical_seconds = 0.
        self.required_content = {}
        self.calls = []
        self.last_call = None
        model.get_image_features = self.encode
        self.hook = model.vision_model.register_forward_pre_hook(self._forward, with_kwargs=True)

    def _forward(self, module, args, kwargs):
        tensor = kwargs.get("pixel_values", args[0] if args else None)
        self.physical_encodings += int(tensor.shape[0])
        self.physical_calls += 1

    def encode(self, **kwargs):
        import torch

        values = {}
        for key, value in kwargs.items():
            if hasattr(value, "detach"):
                if value.dtype != torch.float32:
                    raise ValueError("frozen native encoder input must remain FP32")
                values[key] = _array_digest(value.detach().cpu().numpy())
            else:
                values[key] = value
        pixels = kwargs["pixel_values"]
        identity = canonical_digest({"model": self.model_identity, "inputs": values})
        path = self.root / self.model_identity / (identity + ".json")
        self.required_content[identity] = int(pixels.shape[0])
        with exclusive_lock(path.with_suffix(".lock")):
            cached = path.is_file()
            start = time.monotonic()
            if cached:
                receipt = read(path)
                self.index.identity(receipt["arrays"]["path"], receipt["arrays"])
                with np.load(receipt["arrays"]["path"], allow_pickle=False) as arrays:
                    vectors = arrays["raw_vectors"].copy()
                result = torch.from_numpy(vectors).to(pixels.device)
            else:
                self.forward_attempts += 1
                result = self.original(**kwargs)
                torch.cuda.synchronize() if pixels.is_cuda else None
                vectors = result.detach().float().cpu().numpy()
                if vectors.ndim != 2 or len(vectors) != len(pixels) or not np.isfinite(vectors).all():
                    raise RuntimeError("raw frozen encoder vectors are malformed")
                array_path = path.with_suffix(".npz")
                _write_npz(array_path, {"raw_vectors": vectors})
                receipt = {"status": "COMPLETE", "content_identity": identity,
                    "model_identity": self.model_identity, "tensor_inputs": values,
                    "arrays": self.index.identity(array_path), "crop_inputs": len(pixels),
                    "elapsed_seconds": time.monotonic() - start}
                atomic_write_json(path, receipt)
                self.physical_seconds += receipt["elapsed_seconds"]
            self.last_call = {"content_identity": identity, "physical_cache_hit": cached,
                              "required_crop_encodings": len(pixels), "cache_receipt": str(path),
                              "call_elapsed_seconds": time.monotonic() - start,
                              "required_encoder_seconds": receipt["elapsed_seconds"]}
            self.calls.append(self.last_call)
            return result

    def close(self):
        self.model.get_image_features = self.original
        self.hook.remove()
