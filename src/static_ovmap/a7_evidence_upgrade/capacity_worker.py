"""Pinned SO400M capacity control on the original union six-crop requests."""

import argparse
import fcntl
import os
import time
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _write_npz
from src.static_ovmap.module_validation.region_evidence import native_crops
from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.semantic_models import (
    _load_request,
    _mask_support,
    _unit,
)

from .recognition_worker import aggregate_views


def run(binding, scene, assets, gpu, *, smoke=False):
    import torch
    import transformers

    index, started = InputIndex(), time.monotonic()
    spec = read_json(binding["spec"])
    if scene not in binding["scenes"] or (smoke and scene not in spec["datasets"]["calibration"]):
        raise ValueError("capacity scene outside frozen scope")
    bound = binding["scenes"][scene]
    bound_inputs = {r["path"]: r for r in binding["inputs"]}
    index.identity(bound["config"], bound_inputs[bound["config"]])
    config = read_json(bound["config"])
    index.identity(bound["static_manifest"]["path"], bound["static_manifest"])
    manifest = read_json(bound["static_manifest"]["path"])
    if canonical_digest({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("original request manifest changed")
    index.identity(manifest["capture"]["path"], manifest["capture"])
    root = Path(manifest["capture"]["path"]).parent
    frames = {f["frame_id"]: f for f in read_json(manifest["capture"]["path"])["frames"]}
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    ordered = sorted(requests, key=lambda k: (requests[k]["frame_id"], k))
    if smoke:
        ordered = ordered[:1]
    original_path = bound["sources"]["S_SIGLIP2_AREA"]
    index.identity(original_path, bound_inputs[original_path])
    original = read_json(original_path)
    model = config["models"]["siglip2"]
    if original["valid_ids"] != model["valid_ids"]:
        raise ValueError("vocabulary order mismatch")
    assets = Path(assets)
    index.identity(assets / "download_receipt.json")
    download = read_json(assets / "download_receipt.json")
    if download["repo"] != spec["c0"]["model"] or download["revision"] != spec["c0"]["revision"]:
        raise ValueError("capacity checkpoint does not match protocol")
    for row in download["files"]:
        index.identity(row["path"], row)
    for name in ("capacity_worker.py", "recognition_worker.py"):
        index.identity(Path(__file__).with_name(name))
    for name in ("region_evidence.py", "rgb_siglip.py", "semantic_models.py"):
        index.identity(Path(__file__).parents[1] / "module_validation" / name)
    identity = canonical_digest({"binding": binding["identity"], "scene": scene,
        "requests": ordered, "inputs": index.entries(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "precision": "float32"})
    output = Path(binding["output_root"]) / "c0" / ("smoke" if smoke else "requests") / scene
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["identity"] != identity:
            raise ValueError("completed capacity job changed")
        for item in previous["inputs"] + previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    outputs, features, records = [], {}, []
    calls, physical_crops, text_calls, text_inputs = 0, 0, 0, 0
    lock = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        backend = FrozenSiglipBackend.from_local(str(assets / "model"), device="cuda")
        if any(p.dtype != torch.float32 for p in backend.model.parameters()):
            raise ValueError("capacity model is not FP32")
        parameter_count = sum(p.numel() for p in backend.model.parameters())
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_start
        text_start = time.monotonic()
        names = model["class_names"]
        prototypes = []
        for start in range(0, len(names), 32):
            batch = names[start:start + 32]
            text_calls += 1
            text_inputs += len(batch)
            try:
                prototypes.append(_unit(backend.encode_texts(batch)))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for name in batch:
                    text_calls += 1
                    text_inputs += 1
                    prototypes.append(_unit(backend.encode_texts([name])))
        text = np.concatenate(prototypes)
        torch.cuda.synchronize()
        text_seconds = time.monotonic() - text_start
        text_path = output / "text.npz"
        if text_path.exists():
            with np.load(text_path, allow_pickle=False) as previous:
                if not np.array_equal(previous["text_embeddings"], text):
                    raise ValueError("capacity text replay changed")
        else:
            _write_npz(text_path, {"text_embeddings": text, "valid_ids": np.asarray(model["valid_ids"]),
                                   "class_names": np.asarray(names)})
        outputs.append(index.identity(text_path))
        for request_id in ordered:
            request = requests[request_id]
            path = output / (request_id + ".json")
            if path.exists():
                row = read_json(path)
                if row["job_identity"] != identity:
                    raise ValueError("partial capacity job changed")
                for item in row["inputs"] + row["outputs"]:
                    index.identity(item["path"], item)
            else:
                req_index = InputIndex()
                frame = frames[request["frame_id"]]
                req_index.identity(root / frame["rgb_path"], {"sha256": request["image_sha256"]})
                value = _load_request(root, frame, request, fresh_masks=manifest["requests"][request_id].get("masks"))
                crops = native_crops(value["rgb"], value["target"], value["union"], value["bbox"])
                row = {"job_identity": identity, "request_id": request_id, "status": "UNAVAILABLE",
                       "physical_crop_inputs": 0, "encoder_calls": 0, "outputs": []}
                begin = time.monotonic()
                try:
                    support = _mask_support(backend, value["union"], crops.geometries)
                    row["processor_support"] = support
                    if min(support) <= 0:
                        raise RuntimeError("EMPTY_FOREGROUND_AFTER_PROCESSOR")
                    row["physical_crop_inputs"], row["encoder_calls"] = 6, 1
                    try:
                        vectors = _unit(backend.encode_images(crops.legacy_six))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        row["oom_retry_batch_size"] = 1
                        parts = []
                        for crop in crops.legacy_six:
                            row["physical_crop_inputs"] += 1
                            row["encoder_calls"] += 1
                            parts.append(_unit(backend.encode_images([crop])))
                        vectors = np.concatenate(parts)
                    if vectors.shape != (6, text.shape[1]):
                        raise ValueError("own image/text feature dimension mismatch")
                    torch.cuda.synchronize()
                    arrays_path = path.with_suffix(".npz")
                    _write_npz(arrays_path, {"vectors": vectors, "feature": vectors.mean(0)})
                    row.update(status="COMPLETE", arrays_path=str(arrays_path),
                               crop_geometries=crops.geometries, outputs=[req_index.identity(arrays_path)])
                except RuntimeError as error:
                    row["error"] = str(error)
                    torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
                row["inputs"] = [v for v in req_index.entries() if v["path"] not in {o["path"] for o in row["outputs"]}]
                write_once(path, row)
                physical_crops += row["physical_crop_inputs"]
                calls += row["encoder_calls"]
            records.append(row)
            outputs += [index.identity(path), *row["outputs"]]
            for item in row["inputs"]:
                index.identity(item["path"], item)
            if row["status"] == "COMPLETE":
                with np.load(row["arrays_path"], allow_pickle=False) as arrays:
                    features[request_id] = arrays["feature"]
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    model_identity = canonical_digest({"repo": download["repo"], "revision": download["revision"], "files": download["files"]})
    available = None
    if not smoke:
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
        source = {"scene": scene, "variant": "AW_C0_SO400M", "slot": "S_SIGLIP2_AREA", "valid_ids": model["valid_ids"],
                  "native_record_key": original["native_record_key"], "objects": objects,
                  "model_identity": model_identity, "text_identity": index.identity(text_path),
                  "parent_source_identity": original["identity"], "request_job_identity": identity}
        source["identity"] = canonical_digest(source)
        source_path = Path(binding["output_root"]) / "c0" / scene / "AW_C0_SO400M.json"
        write_once(source_path, source)
        outputs.append(index.identity(source_path))
        available = sum(o["available"] for o in objects.values())
    elif len(features) != 1:
        raise RuntimeError("real capacity smoke failed; inspect persisted request")
    receipt = {"status": "SMOKE_COMPLETE" if smoke else "SOURCES_COMPLETE", "identity": identity,
               "scene": scene, "requests": len(records), "successful_requests": len(features), "available_owners": available,
               "model_identity": model_identity, "feature_dimension": text.shape[1], "parameter_count": parameter_count,
               "physical_crop_inputs_this_invocation": physical_crops, "encoder_calls_this_invocation": calls,
               "physical_text_calls_this_invocation": text_calls, "physical_text_inputs_this_invocation": text_inputs,
               "model_load_seconds": load_seconds, "text_seconds": text_seconds,
               "wall_seconds": time.monotonic() - started, "peak_gpu_allocated_bytes": allocated,
               "peak_gpu_reserved_bytes": reserved, "outputs": outputs,
               "inputs": [v for v in index.entries() if v["path"] not in {o["path"] for o in outputs}]}
    write_once(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = run(read_json(args.binding), args.scene, args.assets, args.gpu, smoke=args.smoke)
    print({k: result[k] for k in ("status", "scene", "requests", "successful_requests", "feature_dimension")}, flush=True)
