"""Paid-only, model-separated six-crop acquisition and proven cache imports."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from src.static_ovmap.module_validation.query_state import AcquisitionPayload
from src.static_ovmap.module_validation.region_evidence import native_crops
from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend

from .io import ROOT, SourceIndex, read_json, write_once


def operation_identity(model_identity, request):
    required = {
        "image_sha256",
        "target_mask_sha256",
        "native_union_mask_sha256",
        "bbox_xyxy",
        "crop_convention",
    }
    if not model_identity or not required <= set(request):
        raise ValueError("visual operation identity lacks actual crop inputs")
    return canonical_digest(
        {
            "model": model_identity,
            "request": request,
            "crop_count": 6,
            "dtype": "float32",
        }
    )


class VisualRequestLoader:
    """FeatureStore calls this only after the frame's complete batch debit."""

    def __init__(self, frames, config, model, cache_root, *, cache_only=False):
        import torch
        import transformers

        self.frames, self.config, self.model = frames, config, model
        self.bound = config["models"][model]
        if self.bound["status"] != "BOUND":
            raise ValueError(f"unavailable visual model: {model}")
        if (torch.__version__, transformers.__version__) != (
            self.bound["torch_version"],
            self.bound["transformers_version"],
        ):
            raise ValueError(
                f"{model} must run in its original isolated library environment"
            )
        self.index = SourceIndex()
        for row in self.bound["files"]:
            self.index.identity(row["path"], row)
        self.cache_root = Path(cache_root) / self.bound["identity"]
        self.cache_only = cache_only
        self.backend, self.model_loads, self.model_load_seconds = None, 0, 0.0
        self.used_receipts, self.import_map = {}, []
        self.native_paths, self.namespaces = {}, set()
        self.static_receipt = self.static_manifest = None
        scene = config["scenes"][frames.scene_id]
        source_directory = Path(scene["source_directory"])
        if model == "native":
            for receipt_path in sorted(
                (source_directory / "query").glob("**/receipt.json")
            ):
                self.index.identity(receipt_path)
                receipt = read_json(receipt_path)
                if receipt.get("status") != "COMPLETE":
                    continue
                for row in receipt.get("outputs", []):
                    path = Path(row["path"])
                    if "native_request_cache" in path.parts and path.suffix == ".json":
                        self.native_paths[str(path)] = row
                        self.namespaces.add(path.parent)
        else:
            receipt_path = source_directory / "semantic_models/siglip2/receipt.json"
            if receipt_path.is_file():
                self.index.identity(receipt_path)
                self.static_receipt = read_json(receipt_path)
                manifest_path = source_directory / "semantic_requests.json"
                expected = next(
                    row
                    for row in self.static_receipt["inputs"]
                    if Path(row["path"]) == manifest_path
                )
                self.index.identity(manifest_path, expected)
                self.static_manifest = read_json(manifest_path)
                self._verify_model_inputs(self.static_receipt["inputs"])
                identity = canonical_digest(
                    {
                        "inputs": self.static_receipt["inputs"],
                        "model_id": "siglip2",
                        "device": "cuda",
                        "schema": 1,
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "numpy": np.__version__,
                        "precision": "float32",
                    }
                )
                if identity != self.static_receipt["input_identity"]:
                    raise ValueError(
                        "static SigLIP2 model/processor/precision identity differs"
                    )

    def _verify_model_inputs(self, inputs):
        actual = {str(Path(row["path"]).resolve()): row for row in inputs}
        for expected in self.bound["files"]:
            row = actual.get(expected["path"])
            if row is None or row["sha256"] != expected["sha256"]:
                raise ValueError("cache model/processor inputs differ")
        for name in ("region_evidence.py", "rgb_siglip.py"):
            rows = [row for row in inputs if Path(row["path"]).name == name]
            if (
                len(rows) != 1
                or self.index.identity(
                    ROOT / "src/static_ovmap/module_validation" / name
                )["sha256"]
                != rows[0]["sha256"]
            ):
                raise ValueError("cache crop/backend source differs")

    def _import_native(self, request):
        import torch
        import transformers

        for namespace in sorted(self.namespaces):
            identity = canonical_digest(
                {"model_identity": namespace.name, "request": request}
            )
            path = namespace / f"{identity}.json"
            expected = self.native_paths.get(str(path))
            if expected is None:
                continue
            self.index.identity(path, expected)
            row = read_json(path)
            self._verify_model_inputs(row["inputs"])
            model_inputs = [
                item for item in row["inputs"] if Path(item["path"]).suffix != ".png"
            ]
            model_key = canonical_digest(
                {
                    "inputs": model_inputs,
                    "dtype": "float32",
                    "device": "cuda",
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "native_crops": 6,
                }
            )
            if (
                row["model_identity"] != model_key
                or row["input_identity"] != identity
                or row["request"] != request
                or row["crop_inputs"] != 6
            ):
                raise ValueError("native import is not the exact original operation")
            return path, row
        return None

    def _import_static(self, request):
        if (
            self.static_manifest is None
            or request["request_id"] not in self.static_manifest["requests"]
        ):
            return None
        original = self.static_manifest["requests"][request["request_id"]]
        if original != request:
            raise ValueError("static import crop/RGB/mask identity differs")
        directory = Path(
            self.config["scenes"][self.frames.scene_id]["source_directory"]
        )
        path = (
            directory
            / "semantic_models/siglip2/requests"
            / (request["request_id"] + ".json")
        )
        self.index.expected_output(path, self.static_receipt)
        row = read_json(path)
        identity = canonical_digest(
            {
                "model_identity": self.static_receipt["input_identity"],
                "request": request,
            }
        )
        if (
            row["input_identity"] != identity
            or row["crop_inputs"] != 6
            or row["background_crop_inputs"] != 0
        ):
            raise ValueError("static import is not the exact six-crop operation")
        return path, row

    def _feature(self, row):
        self.index.expected_output(row["arrays_path"], row)
        if row["inference_status"] != "COMPLETE":
            return None
        with np.load(row["arrays_path"], allow_pickle=False) as arrays:
            feature = np.array(arrays["feature"], copy=True)
            vectors = (
                arrays["crop_features"]
                if "crop_features" in arrays
                else arrays["vectors"]
            )
            if vectors.shape[0] < 6 or not np.array_equal(
                vectors[:6].mean(axis=0), feature
            ):
                raise ValueError("cache feature is not the unnormalized six-crop mean")
        return feature

    def __call__(self, candidate):
        import torch

        current = self.frames.current
        if current is None or current["frame"]["frame_id"] != candidate.frame_id:
            raise ValueError(
                "visual feature access must concern the paid current frame"
            )
        request = current["requests"][candidate.request_id].to_dict()
        if (
            request["target_mask_sha256"] != _array_digest(candidate.global_mask)
            or request["native_union_mask_sha256"]
            != _array_digest(candidate.union_mask)
            or tuple(request["bbox_xyxy"]) != candidate.bbox_xyxy
        ):
            raise ValueError("paid request differs from captured crop inputs")
        identity = operation_identity(self.bound["identity"], request)
        path = self.cache_root / f"{identity}.json"
        cached = path.is_file()
        if cached:
            row = read_json(path)
            if row["input_identity"] != identity:
                raise ValueError("new visual cache identity changed")
        else:
            imported = (
                self._import_native(request)
                if self.model == "native"
                else self._import_static(request)
            )
            if imported is not None:
                original_path, original = imported
                feature = self._feature(original)
                source = self.index.identity(original_path)
                row = {
                    "status": "COMPLETE",
                    "input_identity": identity,
                    "model_identity": self.bound["identity"],
                    "request": request,
                    "inference_status": original["inference_status"],
                    "arrays_path": original["arrays_path"],
                    "outputs": original["outputs"],
                    "crop_inputs": 6,
                    "elapsed_seconds": 0.0,
                    "error": original.get("error"),
                    "imported_from": source,
                    "canonical_input_reconciled": True,
                }
                write_once(path, row)
                self.import_map.append(
                    {
                        "request_id": candidate.request_id,
                        "operation_identity": identity,
                        "source": source,
                    }
                )
                cached = True
            else:
                if self.cache_only:
                    raise ValueError(
                        f"native parity requires an existing cache: {candidate.request_id}"
                    )
                image_path = self.frames.path.parent / current["frame"]["rgb_path"]
                self.index.identity(image_path, {"sha256": request["image_sha256"]})
                rgb = np.asarray(Image.open(image_path).convert("RGB"))
                crops = native_crops(
                    rgb,
                    candidate.global_mask,
                    candidate.union_mask,
                    candidate.bbox_xyxy,
                )
                if self.backend is None:
                    started = time.monotonic()
                    self.backend = FrozenSiglipBackend.from_local(
                        self.bound["path"], device="cuda"
                    )
                    if any(
                        parameter.dtype != torch.float32
                        for parameter in self.backend.model.parameters()
                        if parameter.is_floating_point()
                    ):
                        raise ValueError("composition encoders must remain FP32")
                    self.model_load_seconds += time.monotonic() - started
                    self.model_loads += 1
                started, arrays, error = time.monotonic(), {}, None
                try:
                    vectors = np.asarray(
                        self.backend.encode_images(crops.legacy_six), np.float64
                    )
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if (
                        vectors.ndim != 2
                        or len(vectors) != 6
                        or not np.isfinite(vectors).all()
                        or np.any(norms <= 0)
                    ):
                        raise RuntimeError("invalid six-crop encoder output")
                    vectors /= norms
                    arrays = {"crop_features": vectors, "feature": vectors.mean(axis=0)}
                    if np.linalg.norm(arrays["feature"]) <= 0:
                        raise RuntimeError("zero six-crop mean")
                except RuntimeError as exception:
                    arrays, error = {}, str(exception)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elapsed = time.monotonic() - started
                arrays_path = path.with_suffix(".npz")
                _write_npz(arrays_path, arrays)
                row = {
                    "status": "COMPLETE",
                    "input_identity": identity,
                    "model_identity": self.bound["identity"],
                    "request": request,
                    "inference_status": "COMPLETE"
                    if error is None
                    else "UNAVAILABLE_TECHNICAL_FAILURE",
                    "arrays_path": str(arrays_path),
                    "outputs": [self.index.identity(arrays_path)],
                    "crop_inputs": 6,
                    "elapsed_seconds": elapsed,
                    "error": error,
                    "imported_from": None,
                }
                write_once(path, row)
        feature = self._feature(row)
        self.used_receipts[candidate.request_id] = self.index.identity(path)
        return AcquisitionPayload(
            feature,
            6,
            0.0 if cached else row["elapsed_seconds"],
            row.get("error"),
            physical_cache_hit=cached,
        )
