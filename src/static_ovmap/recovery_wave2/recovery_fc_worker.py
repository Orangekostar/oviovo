"""Original FC operators on locked recovery views, with read-only parent caches."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz

from .binding import ConsumptionIndex, PathResolver, read
from .recovery_sources import fc_image_identity


class ContentCache:
    def __init__(self, parent_roots, output_root, model_identity, *, index=None, resolver=None):
        self.parents = [Path(p).resolve() for p in parent_roots]
        self.output = Path(output_root).resolve()
        if any(self.output == p or self.output.is_relative_to(p) for p in self.parents):
            raise ValueError("FC cache writes cannot target a parent cache")
        self.model = model_identity
        self.index = index or ConsumptionIndex()
        self.resolver = resolver or PathResolver()

    def lookup(self, kind, key):
        if kind not in {"dense", "regions", "text"} or Path(key).name != key:
            raise ValueError("invalid content cache namespace")
        for root in [self.output, *self.parents]:
            path = root / self.model / kind / (key + ".json")
            if not path.is_file():
                continue
            self.index.identity(path)
            row = self.resolver.rewrite(read(path))
            recorded = row.get("content_identity", row.get("input_tensor_key", row.get("identity")))
            if recorded != key:
                raise ValueError("FC cache content identity differs from its key")
            self.index.identity(row["arrays"]["path"], row["arrays"])
            return {**row, "receipt_path": str(path), "cache_origin": "TASK" if root == self.output else "PARENT"}
        return None

    def write(self, kind, key, arrays, metadata):
        existing = self.lookup(kind, key)
        if existing is not None:
            return existing
        path = self.output / self.model / kind / (key + ".npz")
        _write_npz(path, arrays)
        row = {**metadata, "content_identity": key, "arrays": self.index.identity(path)}
        atomic_write_json(path.with_suffix(".json"), row)
        return {**row, "receipt_path": str(path.with_suffix(".json")), "cache_origin": "TASK"}


def _model_and_text(binding, data, index):
    import open_clip
    import timm
    import torch
    from static_ovmap.a7_evidence_upgrade.region_adapter import operator_nodes

    resolver = PathResolver(binding["path_map"])
    path = Path(data.get("FC_model_reference_root", data["fc_root"])) / "receipt.json"
    index.identity(path)
    original = read(path)
    operators = [row for row in original["inputs"] if Path(row["path"]).name == "region_adapter.py"]
    weights = [row for row in original["inputs"] if "/assets/fc_frozen/" in row["path"]]
    if len(operators) != 1 or not weights:
        raise ValueError("parent FC physical model lacks effective operator/weight evidence")
    current = Path(__file__).parents[1] / "a7_evidence_upgrade/region_adapter.py"
    index.identity(current, operators[0])
    for row in weights:
        index.identity(resolver.resolve(row["path"]), row)
    model_key = canonical_digest({"model": "convnext_large_d_320", "precision": "float32",
        "weights": weights, "operators": operators, "torch": torch.__version__,
        "open_clip": open_clip.__version__, "timm": timm.__version__})
    if model_key != data["FC_physical_model_identity"] or model_key != original["physical_model_identity"]:
        raise ValueError("effective FC weights/operators/runtime differ from the inherited physical model")
    ops, prompts = operator_nodes(binding["fc"]["operator_root"])
    config = resolver.rewrite(read(data["config"]))
    names = config["models"]["native"]["class_names"]
    ids = config["models"]["native"]["valid_ids"]
    text_key = canonical_digest({"model": model_key, "names": names, "prompts": prompts})
    text_path = Path(resolver.resolve(data["FC_text"]["path"]))
    if text_path.stem != text_key:
        raise ValueError("inherited FC text names/templates/model identity differ")
    index.identity(text_path, data["FC_text"])
    with np.load(text_path, allow_pickle=False) as arrays:
        text, text_ids = arrays["text_embeddings"], arrays["valid_ids"]
    if not np.array_equal(text_ids, ids) or text.ndim != 2 or not np.isfinite(text).all():
        raise ValueError("inherited FC text space is invalid")
    return model_key, ops, text, ids, index.identity(text_path)


def run_scene(binding, scene, *, map_id="BB00_NATIVE", gpu="2", context=None):
    import torch
    from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
    from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, load_model, region_vector, signed_mask
    from static_ovmap.backbone_wave1.runtime import exclusive_lock, check_gpu_once
    from static_ovmap.module_validation.semantic_models import _load_request

    data = context or binding["scenes"][scene]
    root = Path(binding["output_root"]) / "recovery" / scene / map_id
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    started = time.monotonic()
    documents = {}
    for name in ("registry", "captured_requests", "cached_sources", "fc_request_plan"):
        path = root / (name + ".json")
        index.identity(path)
        documents[name] = read(path)
        if canonical_digest({k: v for k, v in documents[name].items() if k != "identity"}) != documents[name]["identity"]:
            raise ValueError("locked FC prerequisite changed")
    registry, manifest, restored, plan = (documents[k] for k in
        ("registry", "captured_requests", "cached_sources", "fc_request_plan"))
    identity = canonical_digest({"scene": scene, "map_id": map_id,
        "prerequisites": {k: v["identity"] for k, v in documents.items()},
        "model": data["FC_physical_model_identity"], "text": data["FC_text"]["sha256"],
        "worker": index.identity(__file__)["sha256"]})
    receipt_path = root / "fc_recovery_receipt.json"
    if receipt_path.is_file():
        previous = read(receipt_path)
        if previous["status"] == "COMPLETE":
            if previous["identity"] != identity:
                raise ValueError("completed recovery FC job changed")
            for row in previous["inputs"] + previous["outputs"]:
                index.identity(row["path"], row)
            return previous
        archived = root / ("fc_recovery_receipt.failed_" + str(time.time_ns()) + ".json")
        receipt_path.rename(archived)
    capture_path = Path(manifest["capture"]["path"])
    index.identity(capture_path, manifest["capture"])
    frames = {row["frame_id"]: row for row in read(capture_path)["frames"]}
    cache = ContentCache([binding["parent_fc_cache_root"]],
        Path(binding["output_root"]) / "content_cache/fc", data["FC_physical_model_identity"],
        index=index, resolver=PathResolver(binding["path_map"]))
    receipt = {"status": "RUNNING", "identity": identity, "scene": scene, "map_id": map_id,
        "physical_image_encodings": 0, "encoder_batch_calls": 0, "physical_region_poolings": 0,
        "physical_text_inputs": 0, "dense_cache_hits": 0, "region_cache_hits": 0,
        "required_image_contents": {}, "required_dense_receipts": {}, "required_region_receipts": {},
        "request_plan_identity": plan["identity"], "requests": {}, "GT_input": False,
        "budget_assigned_before_inference": True, "model_load_seconds": 0.}
    features, model = {}, None
    requests = manifest["requests"]
    ordered = list(dict.fromkeys(restored["U2_request_ids"] + restored["U3_request_ids"]))
    lock_path = Path("/mnt/shared/ww/ovimap-module-validation-v1") / f".visual-gpu-{gpu}.lock"
    try:
        if ordered:
            model_key, operators, text, ids, text_identity = _model_and_text(binding, data, index)
        else:
            ids, text, text_identity = restored["valid_ids"], None, data["FC_text"]
        with exclusive_lock(lock_path), torch.inference_mode():
            for rid in ordered:
                request, allowance = requests[rid], plan["requests"][rid]
                row = {"request_id": rid, "status": "UNAVAILABLE", "physical_image_encodings": 0,
                    "physical_region_poolings": 0, "frame_id": request["frame_id"]}
                receipt["requests"][rid] = row
                if not allowance["authorized"]:
                    row["reason"] = allowance["reason"]
                    continue
                begin = time.monotonic()
                frame = frames[request["frame_id"]]
                index.identity(capture_path.parent / frame["rgb_path"], {"sha256": request["image_sha256"]})
                index.identity(capture_path.parent / frame["request_arrays"]["path"], frame["request_arrays"])
                for name in ("depth_path", "panoptic_path"):
                    index.identity(capture_path.parent / frame[name],
                                   {"sha256": frame[name.replace("_path", "_sha256")]})
                value = _load_request(capture_path.parent, frame, request)
                image_key = fc_image_identity(request, data["FC_physical_model_identity"])
                if image_key != allowance["image_content_identity"]:
                    raise ValueError("locked FC image allowance changed")
                feature_key = canonical_digest({"image": image_key, "mask": _array_digest(value["target"]),
                    "region": "original_signed_mask_pooling"})
                receipt["required_image_contents"][image_key] = 1
                record = cache.lookup("regions", feature_key)
                try:
                    if record is not None:
                        receipt["region_cache_hits"] += 1
                        with np.load(record["arrays"]["path"], allow_pickle=False) as arrays:
                            vector = arrays["feature"]
                        dense_path = Path(cache.resolver.resolve(record["dense_receipt"]))
                        index.identity(dense_path)
                        dense_row = cache.resolver.rewrite(read(dense_path))
                        index.identity(dense_row["arrays"]["path"], dense_row["arrays"])
                        if dense_row["image_content_key"] != image_key:
                            raise ValueError("region cache references another FC image")
                    else:
                        if model is None:
                            resource = check_gpu_once(gpu, root / "fc_gpu_check.json")
                            foreign = [line for line in resource["occupants"]
                                       if int(line.split(",")[1].strip()) != os.getpid()]
                            if foreign:
                                raise RuntimeError("RESOURCE_BLOCK: recovery FC GPU is occupied")
                            resource.update(status="AVAILABLE", own_process_id=os.getpid(),
                                            foreign_occupants=foreign)
                            atomic_write_json(root / "fc_gpu_check.json", resource)
                            load_begin = time.monotonic()
                            model, _, audit = load_model(Path(binding["assets_root"]) / "ovrcoat",
                                Path(binding["assets_root"]) / "fc_frozen", "FC_FROZEN", "cuda")
                            atomic_write_json(root / "fc_weight_audit.json", audit)
                            receipt["model_load_seconds"] = time.monotonic() - load_begin
                        image, size = image_tensor(value["rgb"], "cuda")
                        actual_key = canonical_digest({"model": model_key, "tensor": _array_digest(image.cpu().numpy())})
                        dense_row = cache.lookup("dense", actual_key)
                        if dense_row is not None:
                            if dense_row["image_content_key"] != image_key:
                                raise ValueError("dense FC cache image/tensor identity differs")
                            receipt["dense_cache_hits"] += 1
                            with np.load(dense_row["arrays"]["path"], allow_pickle=False) as arrays:
                                array = arrays["dense"]
                            if array.dtype != np.float32 or array.ndim != 4 or not np.isfinite(array).all():
                                raise ValueError("dense FC cache precision or shape differs")
                            dense = torch.from_numpy(array).to("cuda")
                        else:
                            if (allowance["cached_in_initial_inventory"]
                                    or image_key not in plan["authorized_new_image_contents"]):
                                raise ValueError("authorized initial dense cache disappeared; no budget replacement")
                            encode_begin = time.monotonic()
                            receipt["physical_image_encodings"] += 1
                            receipt["encoder_batch_calls"] += 1
                            row["physical_image_encodings"] += 1
                            dense = operators["extract_features_convnext"](SimpleNamespace(clip_model=model), image)["clip_vis_dense"]
                            dense_row = cache.write("dense", actual_key, {"dense": dense.cpu().numpy()},
                                {"input_tensor_key": actual_key, "image_content_key": image_key,
                                 "elapsed_seconds": time.monotonic() - encode_begin})
                        dense_path = Path(dense_row["receipt_path"])
                        signed, support, _ = signed_mask(value["target"], size, image.shape[-2:], dense.shape[-2:], "cuda")
                        if not int(support.sum()):
                            raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                        receipt["physical_region_poolings"] += 1
                        row["physical_region_poolings"] += 1
                        pool_begin = time.monotonic()
                        vector, _ = region_vector(model, operators, dense, signed)
                        vector = vector.cpu().numpy().astype(np.float64)
                        record = cache.write("regions", feature_key, {"feature": vector},
                            {"dense_receipt": str(dense_path), "elapsed_seconds": time.monotonic() - pool_begin})
                    if vector.shape != (text.shape[1],) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
                        raise ValueError("INVALID_REGION_FEATURE")
                    features[rid] = vector
                    row.update(status="COMPLETE", content_identity=feature_key,
                        feature=index.identity(record["arrays"]["path"], record["arrays"]),
                        physical_cache_hit=row["physical_region_poolings"] == 0)
                    receipt["required_dense_receipts"][image_key] = str(dense_path)
                    receipt["required_region_receipts"][feature_key] = record["receipt_path"]
                except (ValueError, RuntimeError) as exc:
                    if isinstance(exc, ValueError) and str(exc) not in {"EMPTY_DENSE_MASK_SUPPORT", "INVALID_REGION_FEATURE"}:
                        raise
                    if str(exc).startswith("RESOURCE_BLOCK"):
                        raise
                    row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=str(exc))
                    if model is not None:
                        torch.cuda.empty_cache()
                row["elapsed_seconds"] = time.monotonic() - begin
        sources = {}
        for arm in ("U2", "U3"):
            arm_ids = set(restored[arm + "_request_ids"])
            objects = {}
            for candidate in registry["candidates"]:
                owner = candidate["raw_owner"]
                attempted = [rid for rid in (restored["sources"]["U1"][str(owner)]["used_request_ids"]
                    if arm == "U2" else manifest["views"][f"owner:{owner}"]) if rid in arm_ids]
                used = [rid for rid in attempted if rid in features]
                row = {"owner_id": owner, "available": bool(used), "scores": None, "label": None,
                    "attempted_request_ids": attempted, "used_request_ids": used,
                    "reason": "ORIGINAL_FC_STATIC_AREA" if used else "NO_SUCCESSFUL_LOCKED_FC_VIEW"}
                if used:
                    aggregate = aggregate_views([features[rid] for rid in used],
                        [requests[rid]["visible_target_pixels"] for rid in used])
                    scores = text @ aggregate
                    row.update(scores=scores.tolist(), label=int(ids[int(scores.argmax())]))
                objects[str(owner)] = row
            source = {"source": arm, "scene": scene, "map_id": map_id, "objects": objects,
                "valid_ids": ids, "model_identity": data["FC_physical_model_identity"],
                "text_identity": text_identity, "request_manifest_identity": manifest["identity"], "GT_input": False}
            source["identity"] = canonical_digest(source)
            path = root / (arm + "_FC.json")
            atomic_write_json(path, source)
            sources[arm] = index.identity(path)
        receipt.update(status="COMPLETE", sources=sources, outputs=list(sources.values()),
            inputs=index.entries(), physical_model_identity=data["FC_physical_model_identity"],
            required_text_identity=text_identity, standalone_required_image_inputs=len(receipt["required_image_contents"]),
            standalone_required_region_inputs=len(receipt["required_region_receipts"]),
            standalone_timing_missing_components=["FC_text_and_CPU_preprocessing_not_separately_attributed"])
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", inputs=index.entries(), outputs=[])
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--map-id", default="BB00_NATIVE")
    parser.add_argument("--context")
    args = parser.parse_args()
    if args.context and len(args.scene) != 1:
        parser.error("a map-specific context requires exactly one scene")
    for scene in args.scene:
        result = run_scene(read(args.binding), scene, gpu=args.gpu, map_id=args.map_id,
                           context=read(args.context) if args.context else None)
        print(scene, result["status"], "images", result["physical_image_encodings"],
              "poolings", result["physical_region_poolings"], flush=True)
