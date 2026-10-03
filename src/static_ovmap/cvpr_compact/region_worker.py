"""One FC execution path for projected and genuine archived recovery masks."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache, _model_and_text
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity

from .projected_views import _verified_identity, load_projected_request


def region_feature_key(tensor_key, mask):
    return canonical_digest({"image": tensor_key, "mask": _array_digest(mask),
                             "region": "original_signed_mask_pooling"})


def projected_plan(registry, manifest, arms):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {"G1", "G3"}:
        raise ValueError("projected arms must be distinct G1/G3")
    owners = [str(row["raw_owner"]) for row in registry["candidates"]]
    if (manifest["registry_identity"] != registry["identity"]
            or set(manifest["views"]) != {"owner:" + owner for owner in owners}
            or set(manifest["g1"]) != set(manifest["views"])):
        raise ValueError("projected arms differ from the common candidate registry")
    result = {arm: {} for arm in arms}
    for owner in owners:
        selected = manifest["views"]["owner:" + owner]
        if manifest["g1"]["owner:" + owner] != selected[:1]:
            raise ValueError("G1 must equal the fixed G3 prefix")
        rows = [manifest["requests"][rid] for rid in selected]
        if len(selected) > 3 or len({row["frame_id"] for row in rows}) != len(rows):
            raise ValueError("G3 requires at most three distinct original frames")
        keys = [(-row["visible_pixels"], row["frame_id"], row["target_mask_sha256"]) for row in rows]
        if keys != sorted(keys) or any(int(row["raw_owner"]) != int(owner)
                                     or row["selection_rank"] != rank for rank, row in enumerate(rows)):
            raise ValueError("projected view owner/ranking changed")
        for arm in arms:
            result[arm][owner] = list(selected[:1] if arm == "G1" else selected)
    return result


def classify_regions(registry, plan, requests, features, text, valid_ids):
    text, ids = np.asarray(text), list(map(int, valid_ids))
    if (text.ndim != 2 or len(text) != len(ids) or not np.isfinite(text).all()
            or len(set(ids)) != len(ids) or any(value <= 0 for value in ids)):
        raise ValueError("FC classification requires the complete frozen text vocabulary")
    owners = [str(row["raw_owner"]) for row in registry["candidates"]]
    for arm in plan.values():
        if set(arm) != set(owners):
            raise ValueError("FC classification omitted a common candidate")
    for vector in features.values():
        if (np.shape(vector) != (text.shape[1],) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1., rtol=0, atol=1e-5)):
            raise ValueError("FC evidence must preserve finite unit per-view vectors")
    result = {}
    for arm, selections in plan.items():
        objects = {}
        for owner in owners:
            attempted = selections[owner]
            used = [rid for rid in attempted if rid in features]
            row = {"owner_id": int(owner), "available": False, "scores": None, "label": None,
                "attempted_request_ids": list(attempted), "used_request_ids": used,
                "failed_request_ids": [rid for rid in attempted if rid not in features],
                "reason": "NO_SUCCESSFUL_PRESELECTED_FC_VIEW", "score_kind": "ORIGINAL_FC_COSINE"}
            if used:
                try:
                    vector = aggregate_views([features[rid] for rid in used],
                        [requests[rid]["visible_pixels"] for rid in used])
                except ValueError as exc:
                    if str(exc) != "zero recognition aggregate":
                        raise
                    row["reason"] = "ZERO_REGION_AGGREGATE"
                else:
                    scores = text @ vector
                    row.update(available=True, scores=scores.tolist(), label=ids[int(scores.argmax())],
                               reason="ORIGINAL_FC_STATIC_AREA", aggregate_vector_sha256=_array_digest(vector))
            objects[owner] = row
        result[arm] = objects
    return result


def validate_request_outcomes(requests, features, stats, *, require_success=True):
    if not set(features) <= set(requests):
        raise ValueError("visual features leave the fixed preselected request set")
    for rid in set(requests) - set(features):
        row = stats.get("requests", {}).get(rid, {})
        if row.get("status") != "UNAVAILABLE_TECHNICAL_FAILURE" or not row.get("reason"):
            raise RuntimeError("BLOCKED_VISUAL_WORKER: missing selected feature has no recorded technical failure")
    if require_success and requests and not features:
        raise RuntimeError("BLOCKED_ALL_SEMANTIC_REQUESTS_FAILED: selected requests exist but no inference succeeded")


def archived_u2_plan(binding, scene, registry, *, index):
    root = Path(binding["parent_recovery_root"]) / "recovery" / scene / "BB00_NATIVE"
    resolver = PathResolver(binding["path_map"])
    documents = {}
    for name in ("registry", "captured_requests", "cached_sources"):
        path = root / (name + ".json")
        index.identity(path)
        document = read(path)
        _verified_identity(document)
        documents[name] = resolver.rewrite(document)
    original, manifest, sources = (documents[name] for name in ("registry", "captured_requests", "cached_sources"))
    if (original["identity"] != registry["identity"] or manifest["registry_identity"] != registry["identity"]
            or sources["registry_identity"] != registry["identity"]
            or sources["request_manifest_identity"] != manifest["identity"]):
        raise ValueError("archived U2 evidence differs from the common raw-owner registry")
    owners = {str(row["raw_owner"]) for row in registry["candidates"]}
    if set(sources["sources"]["U1"]) != owners:
        raise ValueError("archived U2 source omitted a common candidate")
    plan = {}
    for owner, row in sources["sources"]["U1"].items():
        selected = list(row["used_request_ids"]) if row["available"] else []
        if len(selected) > 1 or (row["available"] and len(selected) != 1):
            raise ValueError("U2 requires the genuinely retained single Native observation")
        for rid in selected:
            if (rid not in manifest["native_selected_request_ids"]
                    or manifest["lineage_proofs"][rid]["target_id"] != "owner:" + owner):
                raise ValueError("archived U2 single Native proof is inconsistent")
        plan[owner] = selected
    if [rid for owner in sorted(owners, key=int) for rid in plan[owner]] != sorted(sources["U2_request_ids"],
            key=lambda rid: int(manifest["lineage_proofs"][rid]["target_id"].split(":")[1])):
        raise ValueError("archived U2 selected request list changed")
    requests = {rid: {**manifest["requests"][rid], "visible_pixels": manifest["requests"][rid]["visible_target_pixels"],
                     "evidence_kind": "GENUINE_ARCHIVED_U2"} for values in plan.values() for rid in values}
    return {"U2": plan}, requests, manifest


def archived_loader(manifest, *, index):
    from static_ovmap.module_validation.semantic_models import _load_request

    path = Path(manifest["capture"]["path"])
    index.identity(path, manifest["capture"])
    capture = read(path)
    _verified_identity(capture)
    frames = {row["frame_id"]: row for row in capture["frames"]}
    def load(rid):
        request = manifest["requests"][rid]
        frame = frames[request["frame_id"]]
        index.identity(path.parent / frame["rgb_path"], {"sha256": request["image_sha256"]})
        index.identity(path.parent / frame["request_arrays"]["path"], frame["request_arrays"])
        value = _load_request(path.parent, frame, request)
        return {**value, "image": value["rgb"], "request": request}
    return load


class FCSession:
    """Cache=None forbids persistent feature reads/writes, including parent aliases."""

    def __init__(self, binding, data, index, *, cache=None, device="cuda"):
        self.binding, self.data, self.index = binding, data, index
        self.cache, self.device = cache, device
        model_data = {**data, "fc_root": data.get("fc_root", data.get("FC_model_reference_root"))}
        self.model_key, self.operators, self.text, self.ids, self.text_identity = _model_and_text(binding, model_data, index)
        self.model, self.weight_audit, self.model_load_seconds = None, None, 0.

    def load_model(self):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import load_model

        if self.model is None:
            torch.cuda.synchronize() if self.device == "cuda" else None
            begin = time.monotonic()
            self.model, _, self.weight_audit = load_model(Path(self.binding["assets_root"]) / "ovrcoat",
                Path(self.binding["assets_root"]) / "fc_frozen", "FC_FROZEN", self.device)
            torch.cuda.synchronize() if self.device == "cuda" else None
            self.model_load_seconds = time.monotonic() - begin
        return self.model

    def _region_hit(self, key, legacy_key, tensor_key, image_key):
        if self.cache is None:
            return None
        for candidate_key in dict.fromkeys((key, legacy_key)):
            row = self.cache.lookup("regions", candidate_key)
            if row is None:
                continue
            path = Path(self.cache.resolver.resolve(row["dense_receipt"]))
            self.index.identity(path)
            dense = self.cache.resolver.rewrite(read(path))
            if dense["input_tensor_key"] != tensor_key or dense["image_content_key"] != image_key:
                raise ValueError("cached FC region refers to different actual RGB/preprocessing tensors")
            self.index.identity(dense["arrays"]["path"], dense["arrays"])
            with np.load(row["arrays"]["path"], allow_pickle=False) as arrays:
                vector = arrays["feature"].copy()
            storage_dtype = vector.dtype.str
            restoration = "NONE"
            if vector.dtype == np.float64:
                restored = vector.astype(np.float32)
                if not np.array_equal(vector, restored.astype(np.float64)):
                    raise ValueError("legacy FC vector is not lossless FP64 storage of original FP32")
                vector, restoration = restored, "LOSSLESS_FP64_STORAGE_OF_ORIGINAL_FP32"
            if (vector.dtype != np.float32 or vector.shape != (self.text.shape[1],)
                    or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12):
                raise ValueError("cached FC region is not a finite nonzero original FP32 vector")
            row = {**row, "cache_storage_dtype": storage_dtype, "precision_restoration": restoration,
                   "effective_FP32_feature_sha256": _array_digest(vector)}
            return vector, row, str(path), candidate_key != key
        return None

    def encode(self, requests, loaders):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, region_vector, signed_mask

        groups = {}
        for rid, request in requests.items():
            groups.setdefault(fc_image_identity(request, self.model_key), []).append(rid)
        stats = {"physical_image_encodings": 0, "encoder_batch_calls": 0, "physical_region_poolings": 0,
            "dense_cache_hits": 0, "region_cache_hits": 0, "legacy_region_cache_hits": 0,
            "persistent_feature_cache_enabled": self.cache is not None,
            "required_dense_receipts": {}, "required_region_receipts": {}, "requests": {}}
        self.last_stats = stats
        features = {}
        with torch.inference_mode():
            for image_key, rids in groups.items():
                dense = image = values = None
                pooled = {}
                try:
                    values = {rid: loaders[rid](rid) for rid in rids}
                    image, size = image_tensor(values[rids[0]]["image"], self.device)
                    tensor_key = canonical_digest({"model": self.model_key, "tensor": _array_digest(image.cpu().numpy())})
                    pending = []
                    for rid in rids:
                        value = values[rid]
                        if not np.array_equal(value["image"], values[rids[0]]["image"]):
                            raise ValueError("one physical RGB digest produced different decoded images")
                        mask = value["target"]
                        if mask.dtype != np.bool_ or mask.shape != value["image"].shape[:2]:
                            raise ValueError("FC target mask must be unchanged full-resolution boolean evidence")
                        key = region_feature_key(tensor_key, mask)
                        stats["requests"][rid] = {"request_id": rid, "frame_id": requests[rid]["frame_id"],
                            "status": "PENDING", "image_content_key": image_key, "input_tensor_key": tensor_key,
                            "feature_content_key": key, "target_mask_sha256": _array_digest(mask)}
                        hit = self._region_hit(key, region_feature_key(image_key, mask), tensor_key, image_key)
                        if hit is None:
                            pending.append(rid)
                            continue
                        vector, row, dense_path, legacy = hit
                        features[rid] = vector
                        stats["region_cache_hits"] += 1
                        stats["legacy_region_cache_hits"] += int(legacy)
                        stats["required_dense_receipts"][image_key] = dense_path
                        stats["required_region_receipts"][key] = row["receipt_path"]
                        stats["requests"][rid].update(status="COMPLETE", physical_cache_hit=True,
                            cache_storage_dtype=row["cache_storage_dtype"], precision_restoration=row["precision_restoration"],
                            effective_FP32_feature_sha256=row["effective_FP32_feature_sha256"],
                            feature=self.index.identity(row["arrays"]["path"], row["arrays"]))
                    if not pending:
                        continue
                    self.load_model()
                    dense_row = self.cache.lookup("dense", tensor_key) if self.cache is not None else None
                    if dense_row is not None:
                        if dense_row["image_content_key"] != image_key:
                            raise ValueError("dense FC content cache has a different physical RGB image")
                        with np.load(dense_row["arrays"]["path"], allow_pickle=False) as arrays:
                            array = arrays["dense"]
                        if array.dtype != np.float32 or array.ndim != 4 or not np.isfinite(array).all():
                            raise ValueError("dense FC cache requires finite FP32 BCHW tensors")
                        dense = torch.from_numpy(array).to(self.device)
                        stats["dense_cache_hits"] += 1
                    else:
                        stats["physical_image_encodings"] += 1
                        stats["encoder_batch_calls"] += 1
                        dense = self.operators["extract_features_convnext"](SimpleNamespace(clip_model=self.model), image)["clip_vis_dense"]
                        if dense.dtype != torch.float32 or dense.ndim != 4 or not torch.isfinite(dense).all():
                            raise ValueError("FC dense operator returned invalid precision/shape/values")
                        if self.cache is not None:
                            dense_row = self.cache.write("dense", tensor_key, {"dense": dense.cpu().numpy()},
                                {"input_tensor_key": tensor_key, "image_content_key": image_key})
                    if dense_row is not None:
                        stats["required_dense_receipts"][image_key] = dense_row["receipt_path"]
                    for rid in pending:
                        row = stats["requests"][rid]
                        if row["feature_content_key"] in pooled:
                            original_rid = pooled[row["feature_content_key"]]
                            features[rid] = features[original_rid]
                            row.update(status="COMPLETE", physical_cache_hit=False,
                                       within_run_region_alias=original_rid)
                            if "feature" in stats["requests"][original_rid]:
                                row["feature"] = stats["requests"][original_rid]["feature"]
                            continue
                        try:
                            signed, support, _ = signed_mask(values[rid]["target"], size,
                                image.shape[-2:], dense.shape[-2:], self.device)
                            if not int(support.sum()):
                                raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                            stats["physical_region_poolings"] += 1
                            vector, dense_support = region_vector(self.model, self.operators, dense, signed)
                            features[rid] = vector.cpu().numpy().copy()
                            pooled[row["feature_content_key"]] = rid
                            row.update(status="COMPLETE", physical_cache_hit=False, dense_mask_support=dense_support)
                            if self.cache is not None:
                                record = self.cache.write("regions", row["feature_content_key"], {"feature": features[rid]},
                                    {"dense_receipt": dense_row["receipt_path"], "target_mask_sha256": row["target_mask_sha256"],
                                     "input_tensor_key": tensor_key, "pooling_operator": "original_signed_mask_pooling"})
                                stats["required_region_receipts"][row["feature_content_key"]] = record["receipt_path"]
                                row["feature"] = self.index.identity(record["arrays"]["path"], record["arrays"])
                        except ValueError as exc:
                            if str(exc) not in {"EMPTY_DENSE_MASK_SUPPORT", "INVALID_REGION_FEATURE"}:
                                raise
                            row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=str(exc))
                except torch.cuda.OutOfMemoryError as exc:
                    for rid in rids:
                        if rid not in features:
                            stats["requests"].setdefault(rid, {"request_id": rid, "frame_id": requests[rid]["frame_id"]})
                            stats["requests"][rid].update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=f"CUDA_OUT_OF_MEMORY: {exc}")
                except BaseException as exc:
                    for rid in rids:
                        if rid not in features:
                            stats["requests"].setdefault(rid, {"request_id": rid, "frame_id": requests[rid]["frame_id"]})
                            stats["requests"][rid].update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    del dense, image, values, pooled
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
        validate_request_outcomes(requests, features, stats, require_success=False)
        return features, stats


def run_regions(binding, scene, projected_receipt, output_root, *, arms=("G1", "G3"), context=None):
    from .runtime import require_frozen_execution

    require_frozen_execution(binding, scene)
    data, root = context or binding["scenes"][scene], Path(output_root)
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {"G1", "G3", "U2"}:
        raise ValueError("FC recovery scope must contain distinct fixed G1/G3/U2 arms")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"]):
        raise ValueError("FC worker must be launched with the single bound CUDA_VISIBLE_DEVICES")
    index = ConsumptionIndex(root / "input_verifications.json")
    for row in projected_receipt["outputs"]:
        index.identity(row["path"], row)
    registry, manifest = read(projected_receipt["registry"]), read(projected_receipt["manifest"])
    _verified_identity(registry)
    _verified_identity(manifest)
    if manifest["scene_id"] != scene:
        raise ValueError("FC recovery views belong to a different scene")
    plan, requests, loaders, archival_identity = {}, {}, {}, None
    projected_arms = [arm for arm in arms if arm != "U2"]
    if projected_arms:
        plan.update(projected_plan(registry, manifest, projected_arms))
        selected = dict.fromkeys(rid for selections in plan.values() for values in selections.values() for rid in values)
        for rid in selected:
            requests[rid] = manifest["requests"][rid]
            loaders[rid] = lambda request_id: load_projected_request(projected_receipt["manifest"], request_id, index=index)
    if "U2" in arms:
        archived_plan, archived_requests, archival = archived_u2_plan(binding, scene, registry, index=index)
        plan.update(archived_plan)
        requests.update(archived_requests)
        loader = archived_loader(archival, index=index)
        loaders.update({rid: loader for rid in archived_requests})
        archival_identity = archival["identity"]
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "views": manifest["identity"],
        "archival": archival_identity, "arms": arms, "plan": plan, "model": data["FC_physical_model_identity"],
        "text": data["FC_text"]["sha256"], "producer": index.identity(__file__)["sha256"]})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed FC recovery job changed; invalidate affected descendants explicitly")
            for row in old["inputs"] + old["outputs"]:
                index.identity(row["path"], row)
            return old
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "input_identity": identity, "scene": scene, "arms": arms,
        "selected_requests": len(requests), "plan": plan, "GT_input": False, "failed_view_replacement": False}
    atomic_write_json(receipt_path, receipt)
    session = None
    try:
        cache = ContentCache(binding["parent_fc_cache_roots"], Path(binding["writable_content_cache"]) / "fc",
            data["FC_physical_model_identity"], index=index, resolver=PathResolver(binding["path_map"]))
        with exclusive_lock(binding["gpu_lock"]):
            from static_ovmap.backbone_wave1.runtime import check_gpu_once
            resource = check_gpu_once(binding["gpu"], root / "gpu_check.json")
            foreign = [line for line in resource["occupants"] if int(line.split(",")[1].strip()) != os.getpid()]
            if foreign:
                raise RuntimeError("RESOURCE_BLOCK: bound FC GPU is occupied")
            session = FCSession(binding, data, index, cache=cache)
            features, stats = session.encode(requests, loaders)
        receipt.update(stats, model_load_seconds=session.model_load_seconds)
        validate_request_outcomes(requests, features, stats)
        for targets in plan.values():
            ids = {rid for selected in targets.values() for rid in selected}
            validate_request_outcomes(dict.fromkeys(ids), {rid: features[rid] for rid in ids if rid in features}, stats)
        classified = classify_regions(registry, plan, requests, features, session.text, session.ids)
        sources, outputs = {}, []
        for arm, objects in classified.items():
            source = {"source": arm, "scene": scene, "objects": objects, "valid_ids": list(map(int, session.ids)),
                "model_identity": session.model_key, "text_identity": session.text_identity,
                "registry_identity": registry["identity"], "request_manifest_identity": manifest["identity"],
                "score_kind": "ORIGINAL_FC_COSINE", "GT_input": False}
            source["identity"] = canonical_digest(source)
            path = root / (arm + "_FC.json")
            atomic_write_json(path, source)
            sources[arm] = index.identity(path)
            outputs.append(sources[arm])
        if session.weight_audit is not None:
            atomic_write_json(root / "weight_audit.json", session.weight_audit)
            outputs.append(index.identity(root / "weight_audit.json"))
        receipt.update(status="COMPLETE", sources=sources, outputs=outputs,
            physical_model_identity=session.model_key, required_text_identity=session.text_identity)
    except BaseException as exc:
        if session is not None:
            receipt.update(getattr(session, "last_stats", {}), model_load_seconds=session.model_load_seconds)
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=[])
        raise
    finally:
        receipt.update(inputs=index.entries(), elapsed_seconds=time.monotonic() - started)
        receipt["identity"] = canonical_digest({k: v for k, v in receipt.items() if k != "identity"})
        atomic_write_json(receipt_path, receipt)
        index.write_memo(root / "input_verifications.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arm", action="append", choices=("G1", "G3", "U2"))
    args = parser.parse_args()
    result = run_regions(read(args.binding), args.scene, read(args.views), args.output_root,
        arms=args.arm or ("G1", "G3"))
    print(args.scene, result["status"], "images", result["physical_image_encodings"],
          "poolings", result["physical_region_poolings"], flush=True)
