"""Original static global-mask FC/OVR sources, sharing dense frame encodings."""

import argparse
import fcntl
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _write_npz
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.semantic_models import _load_request

from .recognition_worker import aggregate_views
from .region_adapter import (
    image_tensor,
    load_model,
    operator_nodes,
    region_vector,
    signed_mask,
)


def run(binding, scene, assets, branch, gpu):
    import open_clip
    import timm
    import torch
    from torch.nn import functional as F

    if branch not in {"FC_FROZEN", "OVR"} or scene not in binding["scenes"]:
        raise ValueError("unregistered E03 worker")
    started, index = time.monotonic(), InputIndex()
    bound = binding["scenes"][scene]
    bound_inputs = {r["path"]: r for r in binding["inputs"]}
    index.identity(bound["config"], bound_inputs[bound["config"]])
    config = read_json(bound["config"])
    index.identity(bound["static_manifest"]["path"], bound["static_manifest"])
    manifest = read_json(bound["static_manifest"]["path"])
    if canonical_digest({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("original static manifest changed")
    index.identity(manifest["capture"]["path"], manifest["capture"])
    root = Path(manifest["capture"]["path"]).parent
    frames = {f["frame_id"]: f for f in read_json(manifest["capture"]["path"])["frames"]}
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    original_path = bound["sources"]["S_SIGLIP2_AREA"]
    index.identity(original_path, bound_inputs[original_path])
    original = read_json(original_path)
    assets = Path(assets)
    for leaf in ("ovrcoat", "fc_frozen"):
        receipt_path = assets / leaf / "download_receipt.json"
        index.identity(receipt_path)
        receipt = read_json(receipt_path)
        for item in receipt.get("files", [receipt.get("checkpoint")]):
            index.identity(item["path"], item)
    operators, prompts = operator_nodes(assets / "ovrcoat/code")
    for name in ("region_worker.py", "region_adapter.py", "recognition_worker.py"):
        index.identity(Path(__file__).with_name(name))
    spec = read_json(binding["spec"])
    smoke_path = Path(binding["output_root"]) / "e03/smoke" / spec["datasets"]["calibration"][0] / branch / "receipt.json"
    index.identity(smoke_path)
    if read_json(smoke_path)["status"] != "SMOKE_COMPLETE":
        raise ValueError("matched branch real smoke has not passed")
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "branch": branch,
        "inputs": index.entries(), "torch": torch.__version__, "open_clip": open_clip.__version__,
        "timm": timm.__version__, "precision": "float32"})
    variant = "AW_E03_" + branch
    output = Path(binding["output_root"]) / "e03" / scene / variant
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["identity"] != identity:
            raise ValueError("completed E03 worker changed")
        for item in previous["inputs"] + previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    records, features, outputs = {}, {}, []
    image_encodings, poolings, text_calls, text_inputs = 0, 0, 0, 0
    lock = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        model, tokenizer, audit = load_model(assets / "ovrcoat", assets / "fc_frozen", branch, "cuda")
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_start
        audit_path = output / "weight_audit.json"
        write_once(audit_path, audit)
        outputs.append(index.identity(audit_path))
        names, ids = config["models"]["siglip2"]["class_names"], original["valid_ids"]
        if ids != config["models"]["siglip2"]["valid_ids"]:
            raise ValueError("region vocabulary order differs")
        begin = time.monotonic()
        with torch.inference_mode():
            prototypes = []
            for name in names:
                tokens = tokenizer([p.format(name) for p in prompts]).to("cuda")
                text_calls += 1
                text_inputs += len(prompts)
                try:
                    encoded = model.encode_text(tokens, normalize=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows = []
                    for token in tokens:
                        text_calls += 1
                        text_inputs += 1
                        rows.append(model.encode_text(token[None], normalize=True))
                    encoded = torch.cat(rows)
                prototypes.append(F.normalize(encoded.mean(0), dim=0))
            text = torch.stack(prototypes).cpu().numpy().astype(np.float64)
        torch.cuda.synchronize()
        text_seconds = time.monotonic() - begin
        text_path = output / "text.npz"
        if text_path.exists():
            with np.load(text_path, allow_pickle=False) as previous:
                if not np.array_equal(previous["text_embeddings"], text):
                    raise ValueError("region text replay changed")
        else:
            _write_npz(text_path, {"text_embeddings": text, "valid_ids": np.asarray(ids), "class_names": np.asarray(names)})
        outputs.append(index.identity(text_path))
        current_frame, frame_error, dense = None, None, None
        for request_id in sorted(requests, key=lambda k: (requests[k]["frame_id"], k)):
            request = requests[request_id]
            path = output / "requests" / (request_id + ".json")
            if path.exists():
                row = read_json(path)
                if row["job_identity"] != identity:
                    raise ValueError("partial region request changed")
                for item in row["inputs"] + row["outputs"]:
                    index.identity(item["path"], item)
            else:
                req_index = InputIndex()
                frame = frames[request["frame_id"]]
                req_index.identity(root / frame["rgb_path"], {"sha256": request["image_sha256"]})
                value = _load_request(root, frame, request, fresh_masks=manifest["requests"][request_id].get("masks"))
                row = {"job_identity": identity, "request_id": request_id, "frame_id": request["frame_id"],
                       "status": "UNAVAILABLE", "image_encodings": 0, "region_poolings": 0, "outputs": []}
                begin = time.monotonic()
                with torch.inference_mode():
                    if current_frame != request["frame_id"]:
                        current_frame, frame_error, dense = request["frame_id"], None, None
                        try:
                            image, shape = image_tensor(value["rgb"], "cuda")
                            row["image_encodings"] = 1
                            dense = operators["extract_features_convnext"](SimpleNamespace(clip_model=model), image)["clip_vis_dense"]
                        except RuntimeError as error:
                            frame_error = str(error)
                            torch.cuda.empty_cache()
                    try:
                        if frame_error is not None:
                            raise RuntimeError("FRAME_ENCODING_FAILED: " + frame_error)
                        signed, support, resized = signed_mask(value["target"], shape, image.shape[-2:], dense.shape[-2:], "cuda")
                        row["dense_support"] = int(support.sum().item())
                        if row["dense_support"] == 0:
                            raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                        row["region_poolings"] = 1
                        vector, count = region_vector(model, operators, dense, signed)
                        vector = vector.cpu().numpy().astype(np.float64)
                        torch.cuda.synchronize()
                        arrays = {"feature": vector}
                        for label, mask in (("original", value["target"]), ("resized", resized), ("dense", support[0, 0].cpu().numpy())):
                            arrays[label + "_bits"] = np.packbits(mask)
                            arrays[label + "_shape"] = np.asarray(mask.shape)
                        arrays_path = path.with_suffix(".npz")
                        _write_npz(arrays_path, arrays)
                        row.update(status="COMPLETE", arrays_path=str(arrays_path), outputs=[req_index.identity(arrays_path)],
                                   original_pixels=int(value["target"].sum()), resized_pixels=int(resized.sum()), dense_support=count)
                    except (ValueError, RuntimeError) as error:
                        if isinstance(error, ValueError) and str(error) not in {"EMPTY_DENSE_MASK_SUPPORT", "INVALID_REGION_FEATURE"}:
                            raise
                        row["error"] = str(error)
                        torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
                row["inputs"] = [v for v in req_index.entries() if v["path"] not in {o["path"] for o in row["outputs"]}]
                write_once(path, row)
                image_encodings += row["image_encodings"]
                poolings += row["region_poolings"]
            records[request_id] = row
            outputs += [index.identity(path), *row["outputs"]]
            for item in row["inputs"]:
                index.identity(item["path"], item)
            if row["status"] == "COMPLETE":
                with np.load(row["arrays_path"], allow_pickle=False) as data:
                    features[request_id] = data["feature"]
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    objects = {}
    for owner in original["objects"]:
        attempted = manifest["views"].get("owner:" + owner, [])
        used = [r for r in attempted if r in features]
        obj = {"owner_id": int(owner), "available": False, "scores": None, "label": None,
               "attempted_request_ids": attempted, "used_request_ids": used,
               "retained_request_ids": used, "fallback_reason": "NO_GENUINE_REGION_VIEW"}
        if used:
            aggregate = aggregate_views([features[r] for r in used], [requests[r]["visible_target_pixels"] for r in used])
            scores = text @ aggregate
            obj.update(available=True, scores=scores.tolist(), label=ids[int(scores.argmax())], fallback_reason=None)
        objects[owner] = obj
    source = {"scene": scene, "variant": variant, "slot": "S_SIGLIP2_AREA", "valid_ids": ids,
              "native_record_key": original["native_record_key"], "objects": objects,
              "model_identity": canonical_digest({"branch": branch, "audit": index.identity(audit_path), "job": identity}),
              "text_identity": index.identity(text_path), "parent_source_identity": original["identity"], "request_job_identity": identity}
    source["identity"] = canonical_digest(source)
    source_path = output.parent / (variant + ".json")
    write_once(source_path, source)
    outputs.append(index.identity(source_path))
    receipt = {"status": "SOURCES_COMPLETE", "identity": identity, "scene": scene, "variant": variant,
               "requests": len(records), "successful_requests": len(features), "available_owners": sum(v["available"] for v in objects.values()),
               "physical_image_encodings_this_invocation": image_encodings, "physical_region_poolings_this_invocation": poolings,
               "physical_text_calls_this_invocation": text_calls, "physical_text_inputs_this_invocation": text_inputs,
               "model_load_seconds": load_seconds, "text_seconds": text_seconds, "wall_seconds": time.monotonic() - started,
               "peak_gpu_allocated_bytes": allocated, "peak_gpu_reserved_bytes": reserved, "outputs": outputs,
               "inputs": [v for v in index.entries() if v["path"] not in {o["path"] for o in outputs}]}
    write_once(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--branch", choices=["FC_FROZEN", "OVR"], required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = run(read_json(args.binding), args.scene, args.assets, args.branch, args.gpu)
    print({k: result[k] for k in ("status", "scene", "variant", "requests", "successful_requests", "available_owners")}, flush=True)
