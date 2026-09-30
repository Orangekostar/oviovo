"""Frozen FC static-area readout on fresh native lineage and content caches."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time

import numpy as np

from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
from static_ovmap.a7_evidence_upgrade.region_adapter import (
    image_tensor, load_model, operator_nodes, region_vector, signed_mask,
)
from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.semantic_models import _load_request
from .binding import read
from .runtime import exclusive_lock


def run_region(job):
    import open_clip
    import timm
    import torch
    from torch.nn import functional as F

    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    index, start = InputIndex(), time.monotonic()
    manifest = read(job["request_manifest"])
    index.identity(job["request_manifest"])
    index.identity(manifest["capture"]["path"], manifest["capture"])
    capture = read(manifest["capture"]["path"])
    frames = {row["frame_id"]: row for row in capture["frames"]}
    assets = Path(job["assets_root"])
    for leaf in ("ovrcoat", "fc_frozen"):
        receipt = read(assets / leaf / "download_receipt.json")
        index.identity(assets / leaf / "download_receipt.json")
        for item in receipt.get("files", [receipt.get("checkpoint")]):
            index.identity(item["path"], item)
    operators, prompts = operator_nodes(assets / "ovrcoat/code")
    for path in (Path(__file__), Path(__file__).parents[1] / "a7_evidence_upgrade/region_adapter.py",
                 Path(__file__).parents[1] / "a7_evidence_upgrade/recognition_worker.py"):
        index.identity(path)
    model_key = canonical_digest({"assets": index.entries(), "torch": torch.__version__,
                                 "open_clip": open_clip.__version__, "timm": timm.__version__,
                                 "model": "convnext_large_d_320", "precision": "float32"})
    # Geometry/manifest identities belong to lineage, not the physical model key.
    physical_model_key = canonical_digest({"model": "convnext_large_d_320", "precision": "float32",
        "weights": [item for item in index.entries() if str(assets / "fc_frozen") in item["path"]],
        "operators": [item for item in index.entries() if "region_adapter.py" in item["path"]],
        "torch": torch.__version__, "open_clip": open_clip.__version__, "timm": timm.__version__})
    receipt = {"status": "RUNNING", "map_id": job["map_id"], "request_manifest": manifest["identity"],
        "model_identity": model_key, "physical_model_identity": physical_model_key,
        "physical_image_encodings": 0, "physical_region_poolings": 0, "physical_text_inputs": 0,
        "required_image_contents": {}, "required_dense_receipts": {}, "required_region_receipts": {},
        "requests": {}, "GT_input": False}
    receipt["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3], text=True).strip()
    cache = Path(job["cache_root"]) / physical_model_key
    with exclusive_lock(job["gpu_lock"]), torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        try:
            begin = time.monotonic()
            model, tokenizer, audit = load_model(assets / "ovrcoat", assets / "fc_frozen", "FC_FROZEN", "cuda")
            atomic_write_json(root / "weight_audit.json", audit)
            receipt["model_load_seconds"] = time.monotonic() - begin
            text_key = canonical_digest({"model": physical_model_key, "names": job["class_names"], "prompts": prompts})
            text_path = cache / "text" / (text_key + ".npz")
            if text_path.is_file():
                with np.load(text_path, allow_pickle=False) as arrays:
                    text = arrays["text_embeddings"]
            else:
                prototypes = []
                for name in job["class_names"]:
                    tokens = tokenizer([p.format(name) for p in prompts]).to("cuda")
                    receipt["physical_text_inputs"] += len(prompts)
                    encoded = model.encode_text(tokens, normalize=True)
                    prototypes.append(F.normalize(encoded.mean(0), dim=0))
                text = torch.stack(prototypes).cpu().numpy().astype(np.float64)
                _write_npz(text_path, {"text_embeddings": text, "valid_ids": np.asarray(job["valid_ids"])})
                atomic_write_json(text_path.with_suffix(".json"), {"arrays": index.identity(text_path), "identity": text_key})
            text_receipt = read(text_path.with_suffix(".json"))
            index.identity(text_path, text_receipt["arrays"])
            receipt["text_identity"] = index.identity(text_path)
            current, dense, image, shape = None, None, None, None
            features = {}
            requests = manifest["requests"]
            for rid in sorted(requests, key=lambda k: (requests[k]["frame_id"], k)):
                request = requests[rid]
                frame = frames[request["frame_id"]]
                value = _load_request(Path(manifest["capture"]["path"]).parent, frame, request)
                image_key = canonical_digest({"model": physical_model_key, "rgb": request["image_sha256"],
                                              "preprocess": "original_800_1333_bilinear_32pad_FP32"})
                receipt["required_image_contents"][image_key] = 1
                feature_key = canonical_digest({"image": image_key, "mask": _array_digest(value["target"]),
                                                "region": "original_signed_mask_pooling"})
                feature_path = cache / "regions" / (feature_key + ".npz")
                row = {"request_id": rid, "frame_id": frame["frame_id"], "lineage": request["lineage"],
                       "content_identity": feature_key, "image_content_identity": image_key,
                       "physical_cache_hit": feature_path.is_file(), "status": "RUNNING"}
                begin = time.monotonic()
                try:
                    if feature_path.is_file():
                        record = read(feature_path.with_suffix(".json"))
                        index.identity(feature_path, record["arrays"])
                        with np.load(feature_path, allow_pickle=False) as arrays:
                            vector = arrays["feature"]
                    else:
                        if current != frame["frame_id"]:
                            current = frame["frame_id"]
                            dense = None
                            image, shape = image_tensor(value["rgb"], "cuda")
                            actual_key = canonical_digest({"model": physical_model_key,
                                                         "tensor": _array_digest(image.cpu().numpy())})
                            dense_path = cache / "dense" / (actual_key + ".npz")
                            if dense_path.is_file():
                                index.identity(dense_path, read(dense_path.with_suffix(".json"))["arrays"])
                                with np.load(dense_path, allow_pickle=False) as arrays:
                                    dense = torch.from_numpy(arrays["dense"]).to("cuda")
                            else:
                                dense_begin = time.monotonic()
                                receipt["physical_image_encodings"] += 1
                                dense = operators["extract_features_convnext"](SimpleNamespace(clip_model=model), image)["clip_vis_dense"]
                                _write_npz(dense_path, {"dense": dense.cpu().numpy()})
                                atomic_write_json(dense_path.with_suffix(".json"), {
                                    "arrays": index.identity(dense_path), "input_tensor_key": actual_key,
                                    "image_content_key": image_key, "elapsed_seconds": time.monotonic() - dense_begin})
                        if dense is None:
                            raise RuntimeError("current frame FC encoding is unavailable")
                        signed, support, _ = signed_mask(value["target"], shape, image.shape[-2:], dense.shape[-2:], "cuda")
                        if not int(support.sum()):
                            raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                        receipt["physical_region_poolings"] += 1
                        region_begin = time.monotonic()
                        vector, _ = region_vector(model, operators, dense, signed)
                        vector = vector.cpu().numpy().astype(np.float64)
                        _write_npz(feature_path, {"feature": vector})
                        atomic_write_json(feature_path.with_suffix(".json"), {
                            "arrays": index.identity(feature_path), "content_identity": feature_key,
                            "dense_receipt": str(dense_path.with_suffix(".json")),
                            "elapsed_seconds": time.monotonic() - region_begin})
                    region_receipt = read(feature_path.with_suffix(".json"))
                    dense_receipt_path = region_receipt["dense_receipt"]
                    receipt["required_dense_receipts"][image_key] = dense_receipt_path
                    receipt["required_region_receipts"][feature_key] = str(feature_path.with_suffix(".json"))
                    features[rid] = vector
                    row.update(status="COMPLETE", feature=index.identity(feature_path))
                except (RuntimeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc) not in {"EMPTY_DENSE_MASK_SUPPORT", "INVALID_REGION_FEATURE"}:
                        raise
                    row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", error=str(exc))
                    torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
                receipt["requests"][rid] = row
            if requests and not features:
                raise RuntimeError("all FC region requests failed; core evidence is unavailable")
            objects = {}
            for owner in job["owners"]:
                attempted = manifest["views"].get(f"owner:{owner}", [])
                used = [rid for rid in attempted if rid in features]
                row = {"owner_id": owner, "available": bool(used), "scores": None, "label": None,
                       "attempted_request_ids": attempted, "used_request_ids": used}
                if used:
                    aggregate = aggregate_views([features[rid] for rid in used],
                        [requests[rid]["visible_target_pixels"] for rid in used])
                    scores = text @ aggregate
                    row.update(scores=scores.tolist(), label=job["valid_ids"][int(scores.argmax())])
                objects[str(owner)] = row
            source = {"map_id": job["map_id"], "source": "F", "objects": objects,
                      "model_identity": physical_model_key, "text_identity": receipt["text_identity"],
                      "request_manifest_identity": manifest["identity"], "valid_ids": job["valid_ids"]}
            source["identity"] = canonical_digest(source)
            atomic_write_json(root / "F.json", source)
            receipt.update(status="COMPLETE", source=index.identity(root / "F.json"), inputs=index.entries())
            receipt["attributable_standalone_seconds"] = receipt["model_load_seconds"] + sum(
                read(path)["elapsed_seconds"] for path in receipt["required_dense_receipts"].values()) + sum(
                read(path)["elapsed_seconds"] for path in receipt["required_region_receipts"].values())
            receipt["standalone_timing_missing_components"] = ["FC_text_and_CPU_preprocessing_not_separately_attributed"]
        except BaseException as exc:
            receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - start
            receipt["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            atomic_write_json(root / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run_region(read(parser.parse_args().job))
