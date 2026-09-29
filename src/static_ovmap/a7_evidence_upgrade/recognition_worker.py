"""E02 frozen S2 recognition with unchanged requests and explicit mask overrides."""

import argparse
import fcntl
import os
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from src.static_ovmap.module_validation.region_evidence import native_crops
from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.semantic_models import (
    _load_request,
    _mask_support,
    _unit,
)

from .crops import recognition_crops


def aggregate_views(features, areas):
    """Raw six-crop view means retain their norms until final area aggregation."""
    values, weights = np.asarray(features, np.float64), np.asarray(areas, np.float64)
    if values.ndim != 2 or weights.shape != (len(values),) or not len(values):
        raise ValueError("unaligned or empty recognition views")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid recognition views")
    result = np.average(values, axis=0, weights=weights)
    norm = np.linalg.norm(result)
    if norm <= 1e-12:
        raise ValueError("zero recognition aggregate")
    return result / norm


def run(binding, scene, mode, gpu):
    import torch
    import transformers

    if mode not in {"GLOBAL", "SAM2"} or scene not in binding["scenes"]:
        raise ValueError("unregistered E02 job")
    started, index = time.monotonic(), InputIndex()
    bound = binding["scenes"][scene]
    config = read_json(bound["config"])
    model = config["models"]["siglip2"]
    for item in model["files"]:
        index.identity(item["path"], item)
    index.identity(model["text"]["path"], model["text"])
    index.identity(bound["static_manifest"]["path"], bound["static_manifest"])
    manifest = read_json(bound["static_manifest"]["path"])
    if canonical_digest({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("static manifest changed")
    index.identity(manifest["capture"]["path"], manifest["capture"])
    capture = read_json(manifest["capture"]["path"])
    root = Path(manifest["capture"]["path"]).parent
    frames = {f["frame_id"]: f for f in capture["frames"]}
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    original_path = bound["sources"]["S_SIGLIP2_AREA"]
    index.identity(original_path)
    original = read_json(original_path)
    if original["model_identity"] != model["identity"]:
        raise ValueError("original S2 model identity differs")
    old_root = Path(config["scenes"][scene]["source_directory"]) / "semantic_models/siglip2"
    old_receipt = read_json(old_root / "receipt.json")
    index.identity(old_root / "receipt.json")
    old_inputs = {v["path"]: v for v in old_receipt["inputs"]}
    for item in model["files"]:
        if old_inputs.get(item["path"]) != item:
            raise ValueError("cached crop model/processor identity differs")
    for name in ("semantic_models.py", "region_evidence.py", "rgb_siglip.py"):
        path = Path(__file__).parents[1] / "module_validation" / name
        matches = [v for v in old_receipt["inputs"] if Path(v["path"]).name == name]
        if len(matches) != 1:
            raise ValueError("ambiguous historical image adapter identity")
        index.identity(path, matches[0])
    sam_root = Path(binding["output_root"]) / "e02/sam2_masks" / scene
    if mode == "SAM2":
        index.identity(sam_root / "receipt.json")
        sam_receipt = read_json(sam_root / "receipt.json")
        if sam_receipt["status"] != "MASK_REQUESTS_COMPLETE" or sam_receipt["requests"] != len(requests):
            raise ValueError("SAM2 request set incomplete")
    for path in (Path(__file__), Path(__file__).with_name("crops.py")):
        index.identity(path)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "mode": mode,
        "inputs": index.entries(), "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "precision": "float32"})
    output = Path(binding["output_root"]) / "e02" / scene
    variant = "AW_E02_" + mode
    work = output / variant
    receipt_path = work / "receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["identity"] != identity:
            raise ValueError("completed recognition job changed")
        for item in previous["inputs"] + previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    outputs, records, features = [], {}, {}
    physical_crops, calls, reused = 0, 0, 0
    lock_path = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), work)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        backend = FrozenSiglipBackend.from_local(model["path"], device="cuda")
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_start
        for request_id in sorted(requests, key=lambda k: (requests[k]["frame_id"], k)):
            request = requests[request_id]
            path = work / "requests" / (request_id + ".json")
            if path.exists():
                row = read_json(path)
                if row["job_identity"] != identity:
                    raise ValueError("partial recognition job changed")
                for item in row["inputs"] + row["outputs"]:
                    index.identity(item["path"], item)
            else:
                req_index = InputIndex()
                frame = frames[request["frame_id"]]
                req_index.identity(root / frame["rgb_path"], {"sha256": request["image_sha256"]})
                value = _load_request(root, frame, request, fresh_masks=manifest["requests"][request_id].get("masks"))
                old_crops = native_crops(value["rgb"], value["target"], value["union"], value["bbox"])
                mask, rejection = value["target"], None
                if mode == "SAM2":
                    mask_path = sam_root / "requests" / (request_id + ".json")
                    req_index.expected_output(mask_path, sam_receipt)
                    mask_row = read_json(mask_path)
                    if mask_row["status"] != "MASK_COMPLETE":
                        rejection = "SAM2_TECHNICAL_FAILURE"
                    else:
                        artifact = mask_row["outputs"][0]
                        req_index.identity(artifact["path"], artifact)
                        with np.load(artifact["path"], allow_pickle=False) as data:
                            shape = tuple(data["shape"])
                            mask = np.unpackbits(data["mask_bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)
                        if _array_digest(mask) != mask_row["semantic_mask_sha256"] or mask_row["original_request"] != request:
                            raise ValueError("SAM2 mask/request identity mismatch")
                row = {"job_identity": identity, "request_id": request_id, "status": "UNAVAILABLE",
                       "physical_crop_inputs": 0, "encoder_calls": 0, "reused_raw_crops": 0, "outputs": []}
                attempt_start = time.monotonic()
                try:
                    if rejection:
                        raise ValueError(rejection)
                    crops = recognition_crops(value["rgb"], value["bbox"], mask)
                    if any(not np.array_equal(crops["six"][i], old_crops.legacy_six[i]) for i in (0, 2, 4)):
                        raise ValueError("original raw crop pixel parity failed")
                    support = _mask_support(backend, mask, crops["geometries"])
                    if min(support) <= 0:
                        raise ValueError("EMPTY_FOREGROUND_AFTER_PROCESSOR")
                    row.update(processor_support=support, raw_crop_pixel_parity=True,
                               semantic_mask_sha256=_array_digest(mask), crop_geometries=crops["geometries"])
                    old_path = old_root / "requests" / (request_id + ".json")
                    raw_vectors = None
                    if old_path.exists():
                        req_index.expected_output(old_path, old_receipt)
                        cached = read_json(old_path)
                        expected = canonical_digest({"model_identity": old_receipt["input_identity"], "request": request})
                        if cached["input_identity"] != expected:
                            raise ValueError("cached crop request mismatch")
                        if cached.get("inference_status") == "COMPLETE":
                            req_index.expected_output(cached["arrays_path"], cached)
                            with np.load(cached["arrays_path"], allow_pickle=False) as data:
                                vectors = np.asarray(data["vectors"], np.float64)
                            if vectors.shape != (6, 1024) or not np.isfinite(vectors).all():
                                raise ValueError("cached S2 six-crop vectors malformed")
                            if cached["crop_geometries"] != [list(g) for g in crops["geometries"]]:
                                raise ValueError("cached crop geometry mismatch")
                            raw_vectors = vectors[[0, 2, 4]]
                    positions = [1, 3, 5] if raw_vectors is not None else list(range(6))
                    images = [crops["six"][i] for i in positions]
                    row["physical_crop_inputs"] += len(images)
                    row["encoder_calls"] += 1
                    try:
                        encoded = _unit(backend.encode_images(images))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        row["oom_retry_batch_size"] = 1
                        row["physical_crop_inputs"] += len(images)
                        row["encoder_calls"] += len(images)
                        encoded = np.concatenate([_unit(backend.encode_images([im])) for im in images])
                    vectors = np.empty((6, encoded.shape[1]), np.float64)
                    vectors[positions] = encoded
                    if raw_vectors is not None:
                        vectors[[0, 2, 4]] = raw_vectors
                        row["reused_raw_crops"] = 3
                    torch.cuda.synchronize()
                    arrays_path = path.with_suffix(".npz")
                    _write_npz(arrays_path, {"vectors": vectors, "feature": vectors.mean(0)})
                    row.update(status="COMPLETE", arrays_path=str(arrays_path), outputs=[req_index.identity(arrays_path)])
                except (ValueError, RuntimeError) as error:
                    # Identity/parity failures are corruption, never an ordinary unavailable observation.
                    if isinstance(error, ValueError) and str(error) not in {
                        "SAM2_TECHNICAL_FAILURE", "EMPTY_FOREGROUND_AFTER_PROCESSOR", "empty raw or semantic foreground crop"}:
                        raise
                    row["error"] = str(error)
                    torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - attempt_start
                row["inputs"] = [v for v in req_index.entries() if v["path"] not in {o["path"] for o in row["outputs"]}]
                write_once(path, row)
                physical_crops += row["physical_crop_inputs"]
                calls += row["encoder_calls"]
                reused += row["reused_raw_crops"]
            records[request_id] = row
            outputs += [index.identity(path), *row["outputs"]]
            for item in row["inputs"]:
                index.identity(item["path"], item)
            if row["status"] == "COMPLETE":
                with np.load(row["arrays_path"], allow_pickle=False) as data:
                    features[request_id] = data["feature"]
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    with np.load(model["text"]["path"], allow_pickle=False) as data:
        text = _unit(data["text_embeddings"])
    objects = {}
    for owner in original["objects"]:
        attempted = manifest["views"].get("owner:" + owner, [])
        used = [r for r in attempted if r in features]
        obj = {"owner_id": int(owner), "available": False, "scores": None, "label": None,
               "attempted_request_ids": attempted, "used_request_ids": used,
               "retained_request_ids": used, "fallback_reason": "NO_GENUINE_RECOGNITION_VIEW"}
        if used:
            aggregate = aggregate_views([features[r] for r in used], [requests[r]["visible_target_pixels"] for r in used])
            scores = text @ aggregate
            obj.update(available=True, scores=scores.tolist(), label=model["valid_ids"][int(scores.argmax())], fallback_reason=None)
        objects[owner] = obj
    source = {"scene": scene, "variant": variant, "slot": "S_SIGLIP2_AREA", "valid_ids": model["valid_ids"],
              "native_record_key": original["native_record_key"], "objects": objects,
              "model_identity": model["identity"], "text_identity": model["text"],
              "parent_source_identity": original["identity"], "request_job_identity": identity}
    source["identity"] = canonical_digest(source)
    source_path = output / (variant + ".json")
    write_once(source_path, source)
    outputs.append(index.identity(source_path))
    receipt = {"status": "SOURCES_COMPLETE", "identity": identity, "scene": scene, "variant": variant,
               "requests": len(records), "successful_requests": len(features), "available_owners": sum(v["available"] for v in objects.values()),
               "physical_crop_inputs_this_invocation": physical_crops, "encoder_calls_this_invocation": calls,
               "reused_raw_crops_this_invocation": reused, "model_load_seconds": load_seconds,
               "wall_seconds": time.monotonic() - started, "peak_gpu_allocated_bytes": allocated,
               "peak_gpu_reserved_bytes": reserved, "outputs": outputs,
               "inputs": [v for v in index.entries() if v["path"] not in {o["path"] for o in outputs}]}
    write_once(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", choices=["GLOBAL", "SAM2"], required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = run(read_json(args.binding), args.scene, args.mode, args.gpu)
    print({k: result[k] for k in ("status", "scene", "variant", "requests", "successful_requests", "available_owners")}, flush=True)


if __name__ == "__main__":
    main()
