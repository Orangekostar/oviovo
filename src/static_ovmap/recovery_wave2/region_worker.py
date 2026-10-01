"""Fresh FC static-area readout with exact physical operators and cached text."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest

from .binding import ConsumptionIndex, PathResolver, read
from .recovery_fc_worker import ContentCache, _model_and_text
from .recovery_sources import fc_image_identity


def run_region(job):
    import torch
    from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
    from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, load_model, region_vector, signed_mask
    from static_ovmap.backbone_wave1.runtime import check_gpu_once, exclusive_lock
    from static_ovmap.module_validation.semantic_models import _load_request

    binding, scene = read(job["binding"]), job["scene"]
    from .readouts import require_semantic_gate

    require_semantic_gate(binding, scene, job["map_id"])
    data, root = binding["scenes"][scene], Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "input_verifications.json" if (root / "input_verifications.json").is_file()
                             else Path(binding["output_root"]) / "validation/input_verifications.json")
    index.identity(job["request_manifest"])
    manifest = read(job["request_manifest"])
    if manifest["identity"] != canonical_digest({k: v for k, v in manifest.items() if k != "identity"}):
        raise ValueError("fresh FC request manifest identity changed")
    capture_path = Path(manifest["capture"]["path"])
    index.identity(capture_path, manifest["capture"])
    frames = {row["frame_id"]: row for row in read(capture_path)["frames"]}
    model_key, operators, text, ids, text_identity = _model_and_text(binding, data, index)
    for path in (__file__, Path(__file__).with_name("recovery_fc_worker.py"),
                 Path(__file__).parents[1] / "a7_evidence_upgrade/recognition_worker.py"):
        index.identity(path)
    identity = canonical_digest({"job": job, "request_manifest": manifest["identity"], "inputs": index.entries()})
    path = root / "receipt.json"
    if path.is_file():
        old = read(path)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed own-map FC readout inputs changed")
            for row in old["inputs"] + [old["source"]]:
                index.identity(row["path"], row)
            return old
        path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    cache = ContentCache([binding["parent_fc_cache_root"]], Path(binding["output_root"]) / "content_cache/fc",
                         model_key, index=index, resolver=PathResolver(binding["path_map"]))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "scene": scene, "map_id": job["map_id"], "input_identity": identity,
        "request_manifest": manifest["identity"], "physical_model_identity": model_key, "text_identity": text_identity,
        "physical_image_encodings": 0, "physical_region_poolings": 0, "physical_text_inputs": 0,
        "encoder_batch_calls": 0, "dense_cache_hits": 0, "region_cache_hits": 0,
        "required_image_contents": {}, "required_dense_receipts": {}, "required_region_receipts": {},
        "requests": {}, "GT_input": False, "parent_caches_read_only": True, "model_load_seconds": 0.}
    requests, features, model = manifest["requests"], {}, None
    current, image, size, dense, dense_row = None, None, None, None, None
    try:
        with exclusive_lock(job["gpu_lock"]), torch.inference_mode():
            for rid in sorted(requests, key=lambda key: (requests[key]["frame_id"], key)):
                request, begin = requests[rid], time.monotonic()
                frame = frames[request["frame_id"]]
                index.identity(capture_path.parent / frame["rgb_path"], {"sha256": request["image_sha256"]})
                index.identity(capture_path.parent / frame["request_arrays"]["path"], frame["request_arrays"])
                for name in ("depth_path", "panoptic_path"):
                    index.identity(capture_path.parent / frame[name], {"sha256": frame[name.replace("_path", "_sha256")]})
                value = _load_request(capture_path.parent, frame, request)
                image_key = fc_image_identity(request, model_key)
                key = canonical_digest({"image": image_key, "mask": _array_digest(value["target"]),
                                        "region": "original_signed_mask_pooling"})
                receipt["required_image_contents"][image_key] = 1
                row = {"request_id": rid, "frame_id": request["frame_id"], "lineage": request["lineage"],
                    "content_identity": key, "image_content_identity": image_key, "status": "RUNNING",
                    "physical_image_encodings": 0, "physical_region_poolings": 0}
                receipt["requests"][rid] = row
                record = cache.lookup("regions", key)
                try:
                    if record is not None:
                        receipt["region_cache_hits"] += 1
                        with np.load(record["arrays"]["path"], allow_pickle=False) as arrays:
                            vector = arrays["feature"]
                        required_dense_path = Path(cache.resolver.resolve(record["dense_receipt"]))
                        index.identity(required_dense_path)
                        required_dense = cache.resolver.rewrite(read(required_dense_path))
                        index.identity(required_dense["arrays"]["path"], required_dense["arrays"])
                        if required_dense["image_content_key"] != image_key:
                            raise ValueError("FC region cache refers to a different physical image")
                    else:
                        if model is None:
                            resource = check_gpu_once(job["gpu"], root / "gpu_check.json")
                            foreign = [line for line in resource["occupants"] if int(line.split(",")[1].strip()) != os.getpid()]
                            if foreign:
                                raise RuntimeError("RESOURCE_BLOCK: FC GPU has another process")
                            load_start = time.monotonic()
                            torch.cuda.reset_peak_memory_stats()
                            model, _, audit = load_model(Path(binding["assets_root"]) / "ovrcoat",
                                Path(binding["assets_root"]) / "fc_frozen", "FC_FROZEN", "cuda")
                            atomic_write_json(root / "weight_audit.json", audit)
                            receipt["model_load_seconds"] = time.monotonic() - load_start
                        if current != image_key:
                            current = image_key
                            image, size = image_tensor(value["rgb"], "cuda")
                            actual_key = canonical_digest({"model": model_key, "tensor": _array_digest(image.cpu().numpy())})
                            dense_row = cache.lookup("dense", actual_key)
                            if dense_row is not None:
                                if dense_row["image_content_key"] != image_key:
                                    raise ValueError("FC dense cache differs from the exact image/tensor input")
                                receipt["dense_cache_hits"] += 1
                                with np.load(dense_row["arrays"]["path"], allow_pickle=False) as arrays:
                                    array = arrays["dense"]
                                if array.dtype != np.float32 or array.ndim != 4 or not np.isfinite(array).all():
                                    raise ValueError("FC dense cache has malformed FP32 features")
                                dense = torch.from_numpy(array).to("cuda")
                            else:
                                encode_start = time.monotonic()
                                receipt["physical_image_encodings"] += len(image)
                                receipt["encoder_batch_calls"] += 1
                                row["physical_image_encodings"] += len(image)
                                dense = operators["extract_features_convnext"](SimpleNamespace(clip_model=model), image)["clip_vis_dense"]
                                torch.cuda.synchronize()
                                dense_row = cache.write("dense", actual_key, {"dense": dense.cpu().numpy()},
                                    {"input_tensor_key": actual_key, "image_content_key": image_key,
                                     "elapsed_seconds": time.monotonic() - encode_start})
                        signed, support, _ = signed_mask(value["target"], size, image.shape[-2:], dense.shape[-2:], "cuda")
                        if not int(support.sum()):
                            raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                        pool_start = time.monotonic()
                        receipt["physical_region_poolings"] += 1
                        row["physical_region_poolings"] += 1
                        vector, _ = region_vector(model, operators, dense, signed)
                        torch.cuda.synchronize()
                        vector = vector.cpu().numpy().astype(np.float64)
                        record = cache.write("regions", key, {"feature": vector},
                            {"dense_receipt": dense_row["receipt_path"], "elapsed_seconds": time.monotonic() - pool_start})
                        required_dense_path = Path(dense_row["receipt_path"])
                    if vector.shape != (text.shape[1],) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
                        raise ValueError("INVALID_REGION_FEATURE")
                    features[rid] = vector
                    receipt["required_dense_receipts"][image_key] = str(required_dense_path)
                    receipt["required_region_receipts"][key] = record["receipt_path"]
                    row.update(status="COMPLETE", feature=index.identity(record["arrays"]["path"], record["arrays"]),
                               physical_cache_hit=row["physical_region_poolings"] == 0)
                except (RuntimeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc) not in {"EMPTY_DENSE_MASK_SUPPORT", "INVALID_REGION_FEATURE"}:
                        raise
                    if str(exc).startswith("RESOURCE_BLOCK"):
                        raise
                    row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", error=str(exc))
                    if model is not None:
                        torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
            if requests and not features:
                raise RuntimeError("all own-map FC region requests failed")
            objects = {}
            for owner in job["owners"]:
                attempted = manifest["views"].get(f"owner:{owner}", [])
                used = [rid for rid in attempted if rid in features]
                row = {"owner_id": owner, "available": bool(used), "scores": None, "label": None,
                       "attempted_request_ids": attempted, "used_request_ids": used}
                if used:
                    aggregate = aggregate_views([features[rid] for rid in used], [requests[rid]["visible_target_pixels"] for rid in used])
                    scores = text @ aggregate
                    row.update(scores=scores.tolist(), label=int(ids[int(scores.argmax())]))
                objects[str(owner)] = row
            source = {"map_id": job["map_id"], "source": "F", "objects": objects, "valid_ids": ids,
                "model_identity": model_key, "text_identity": text_identity, "request_manifest_identity": manifest["identity"]}
            source["identity"] = canonical_digest(source)
            atomic_write_json(root / "F.json", source)
            receipt.update(status="COMPLETE", source=index.identity(root / "F.json"), inputs=index.entries(),
                attributable_standalone_seconds=receipt["model_load_seconds"] + sum(read(path)["elapsed_seconds"]
                    for path in receipt["required_dense_receipts"].values()) + sum(read(path)["elapsed_seconds"]
                    for path in receipt["required_region_receipts"].values()),
                standalone_timing_missing_components=["FC_text_and_CPU_preprocessing_not_separately_attributed"],
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if model is not None else 0)
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", inputs=index.entries())
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(path, receipt)
        index.write_memo(root / "input_verifications.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    result = run_region(read(parser.parse_args().job))
    print(result["scene"], result["map_id"], result["status"], "FC image inputs", result["physical_image_encodings"], flush=True)
