"""Original Native six-crop recovery on the geometry-selected G1 evidence."""

import argparse
import importlib.util
import os
from pathlib import Path
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import check_gpu_once, exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.native_worker import ReadOnlyNativeCache
from static_ovmap.recovery_wave2.recovery_sources import single_native_classifier

from .projected_views import _verified_identity, load_projected_request
from .region_worker import projected_plan, validate_request_outcomes


NATIVE_VL_SHA256 = "d4ba96438bc4c0d116afb0ce4a1cc19b7e37c0f54e51562153705bbb41385216"


def native_evidence_key(request, model_identity, operator_identity):
    return canonical_digest({"encoder": "ORIGINAL_NATIVE_SIX_CROPS", "model": model_identity,
        "operators": operator_identity, "image": request["image_sha256"],
        "mask": request["target_mask_sha256"], "union": request["target_mask_sha256"],
        "bbox": request["legacy_native_bbox"], "expansion": .1,
        "crop_convention": "native_global_bbox_union_exclusive_upper_v1"})


def native_sources(registry, plan, features, text, canonical, valid_ids):
    owners = {str(row["raw_owner"]) for row in registry["candidates"]}
    if set(plan) != owners or any(len(selected) > 1 for selected in plan.values()):
        raise ValueError("Native G1 recovery requires exactly the fixed single-view plan")
    result = {}
    for owner, selected in plan.items():
        used = [rid for rid in selected if rid in features]
        row = {"owner_id": int(owner), "available": False, "label": None, "scores": None,
            "attempted_request_ids": selected, "used_request_ids": used,
            "failed_request_ids": [rid for rid in selected if rid not in features],
            "reason": "NO_SUCCESSFUL_PRESELECTED_NATIVE_VIEW"}
        if used:
            feature = np.asarray(features[used[0]])
            if feature.dtype != np.float32 or np.linalg.norm(feature) <= 1e-12:
                raise ValueError("Native recovery requires its original nonzero FP32 six-crop mean")
            row.update(available=True, **single_native_classifier(feature, text, canonical, valid_ids),
                       reason="ORIGINAL_SAME_VIEW_NATIVE_SIX_CROPS")
        result[owner] = row
    return result


def run_native_regions(binding, scene, projected_receipt, output_root, *, context=None):
    import torch
    import transformers
    from .runtime import require_frozen_execution

    require_frozen_execution(binding, scene)

    data, root = context or binding["scenes"][scene], Path(output_root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(binding["gpu"]):
        raise ValueError("Native worker must be launched with the single bound CUDA_VISIBLE_DEVICES")
    index = ConsumptionIndex(root / "input_verifications.json")
    for item in projected_receipt["outputs"]:
        index.identity(item["path"], item)
    registry, manifest = read(projected_receipt["registry"]), read(projected_receipt["manifest"])
    _verified_identity(registry)
    _verified_identity(manifest)
    if manifest["scene_id"] != scene:
        raise ValueError("Native recovery views belong to a different scene")
    plan = projected_plan(registry, manifest, ("G1",))["G1"]
    selected = list(dict.fromkeys(rid for values in plan.values() for rid in values))
    model = data["models"]["native"]
    if (model["dtype"] != "float32" or model["torch_version"] != torch.__version__
            or model["transformers_version"] != transformers.__version__):
        raise ValueError("Native runtime/precision differs from the inherited actual encoder")
    for item in model["files"] + [model["text"]]:
        index.identity(item["path"], item)
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, canonical, ids = arrays["text_embeddings"], arrays["canonical_embeddings"], arrays["valid_ids"]
    if not np.array_equal(ids, model["valid_ids"]):
        raise ValueError("Native G1 text vocabulary/class ordering changed")
    command = data["actual_capture_command"]
    upstream = Path(command["environment"]["OVIMAP_NATIVE_UPSTREAM"])
    operator_path = upstream / "scripts/vl_models.py"
    operator = index.identity(operator_path, {"sha256": NATIVE_VL_SHA256})
    physical_model = canonical_digest({"files": model["files"], "torch": torch.__version__,
        "precision": "float32", "encoder": "SIGLIP_LARGE_PATCH16_384"})
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "views": manifest["identity"],
        "plan": plan, "model": physical_model, "operators": operator, "text": model["text"],
        "producer": index.identity(__file__)["sha256"]})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        old = read(receipt_path)
        if old["status"] == "COMPLETE":
            if old["input_identity"] != identity:
                raise ValueError("completed Native recovery changed; invalidate affected descendants explicitly")
            for item in old["inputs"] + old["outputs"]:
                index.identity(item["path"], item)
            return old
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "input_identity": identity, "scene": scene, "plan": plan,
        "requests": {}, "model_identity": physical_model, "operator_identity": operator,
        "physical_image_encodings": 0, "encoder_batch_calls": 0, "model_load_seconds": 0.,
        "GT_input": False, "failed_view_replacement": False}
    atomic_write_json(receipt_path, receipt)
    encoder, features, outputs = None, {}, []
    try:
        with exclusive_lock(binding["gpu_lock"]), torch.inference_mode():
            if selected:
                resource = check_gpu_once(binding["gpu"], root / "gpu_check.json")
                foreign = [line for line in resource["occupants"] if int(line.split(",")[1].strip()) != os.getpid()]
                if foreign:
                    raise RuntimeError("RESOURCE_BLOCK: bound Native GPU is occupied")
                module_spec = importlib.util.spec_from_file_location("_cvpr_compact_inherited_native_vl", operator_path)
                operators = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(operators)
                operators.siglip_model_list["siglip-l-16-384"] = model["path"]
                capture = read(data["capture_manifest"])
                begin = time.monotonic()
                native = operators.VLModel("siglip-l-16-384", tuple(capture["frames"][0]["image_size_hw"]),
                    "cuda", precision="fp32")
                native.siglip_model.requires_grad_(False)
                if native.k_expand != .1 or any(p.dtype != torch.float32 for p in native.siglip_model.parameters() if p.is_floating_point()):
                    raise ValueError("original Native crop expansion/precision changed")
                torch.cuda.synchronize()
                receipt["model_load_seconds"] = time.monotonic() - begin
                encoder = ReadOnlyNativeCache(native.siglip_model, physical_model,
                    Path(binding["writable_content_cache"]) / "native", binding["parent_native_cache_roots"],
                    resolver=PathResolver(binding["path_map"]), memo=root / "input_verifications.json")
                for rid in selected:
                    request = manifest["requests"][rid]
                    value = load_projected_request(projected_receipt["manifest"], rid, index=index)
                    if not np.array_equal(value["target"], value["union"]):
                        raise ValueError("Native projected union must equal the exact G1 target mask")
                    row = {"request_id": rid, "frame_id": request["frame_id"], "raw_owner": request["raw_owner"],
                        "canonical_bbox": request["canonical_bbox"], "legacy_native_bbox": request["legacy_native_bbox"],
                        "target_mask_sha256": request["target_mask_sha256"],
                        "content_identity": native_evidence_key(request, physical_model, operator["sha256"])}
                    receipt["requests"][rid] = row
                    begin = time.monotonic()
                    try:
                        feature = native.encode_image_with_bbox(value["image"], value["union"], tuple(request["legacy_native_bbox"]))
                        feature = np.asarray(feature)
                        if (feature.dtype != np.float32 or feature.shape != (text.shape[1],)
                                or not np.isfinite(feature).all() or np.linalg.norm(feature) <= 1e-12):
                            raise ValueError("INVALID_NATIVE_REGION_FEATURE")
                        features[rid] = feature
                        path = root / "features" / (row["content_identity"] + ".npz")
                        _write_npz(path, {"feature": feature})
                        output = index.identity(path)
                        outputs.append(output)
                        row.update(status="COMPLETE", feature=output, encoder_content=encoder.last_call,
                                   six_crop_inputs=6, mean_vector_sha256=_array_digest(feature))
                    except (ValueError, torch.cuda.OutOfMemoryError) as exc:
                        if isinstance(exc, ValueError) and str(exc) != "INVALID_NATIVE_REGION_FEATURE":
                            raise
                        row.update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=str(exc))
                        torch.cuda.empty_cache()
                    row["elapsed_seconds"] = time.monotonic() - begin
        validate_request_outcomes(dict.fromkeys(selected), features, receipt)
        objects = native_sources(registry, plan, features, text, canonical, ids)
        source = {"source": "G1_NATIVE", "scene": scene, "objects": objects, "valid_ids": ids.tolist(),
            "model_identity": physical_model, "operator_identity": operator, "text_identity": model["text"],
            "registry_identity": registry["identity"], "request_manifest_identity": manifest["identity"],
            "GT_input": False}
        source["identity"] = canonical_digest(source)
        path = root / "G1_NATIVE.json"
        atomic_write_json(path, source)
        output = index.identity(path)
        outputs.append(output)
        receipt.update(status="COMPLETE", source=output, outputs=outputs)
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", outputs=outputs)
        raise
    finally:
        if encoder is not None:
            encoder.close()
            for item in encoder.index.entries():
                index.identity(item["path"], item)
            receipt.update(physical_image_encodings=encoder.physical_encodings, encoder_batch_calls=encoder.physical_calls,
                forward_attempts=encoder.forward_attempts, required_crop_inputs=sum(encoder.required_content.values()),
                required_content=encoder.required_content, parent_cache_hits=encoder.parent_hits)
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
    parser.add_argument("--context")
    args = parser.parse_args()
    result = run_native_regions(read(args.binding), args.scene, read(args.views), args.output_root,
        context=read(args.context) if args.context else None)
    print(args.scene, result["status"], "native crops", result["physical_image_encodings"], flush=True)
