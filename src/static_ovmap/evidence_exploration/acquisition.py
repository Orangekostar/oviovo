"""One original FC environment worker; frozen encoders and shared AnyUp chunks.

Scientific runs may read bound dense caches. Cold timing uses the same operators
with freshly encoded FC inputs; it is implemented by the separate timing adapter.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
from static_ovmap.a7_evidence_upgrade.region_adapter import signed_mask
from static_ovmap.cvpr_compact.area_fallback import region_vector
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.runtime_parity.runner import execution_config

from .anyup_adapter import aligned_inputs, owner_grids, encode_qkv, stream_regions


def load_anyup(root, device):
    assets = read(Path(root) / "assets.json")
    _verified_identity(assets)
    checkout = Path(root).parent / "assets" / "anyup"
    checkpoint = Path(root).parent / "assets" / "anyup_paper.pth"
    sys.path.insert(0, str(checkout))
    from anyup.model import AnyUp
    model = AnyUp(use_natten=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def project_raw(session, pooled, signed):
    projected = session.operators["visual_prediction_forward_convnext"](
        SimpleNamespace(clip_model=session.model), pooled[None, None], signed)
    if not torch.isfinite(projected).all():
        raise ValueError("nonfinite frozen FC projection is an invariant failure")
    if torch.linalg.vector_norm(projected).item() <= 1e-12:
        return None
    return F.normalize(projected, dim=-1)[0, 0].cpu().numpy().copy()


def cosine_record(session, vector, pixels):
    if vector is None:
        return {"feature_available": False, "fallback_reason": "REPRESENTATION_UNAVAILABLE_KEEP_B1"}
    aggregate = aggregate_views([vector], [int(pixels)])
    scores = session.text @ aggregate
    return {"feature_available": True, "scores": scores.tolist(), "class": session.ids[int(scores.argmax())],
            "vector_sha256": _array_digest(vector), "aggregate_vector_sha256": _array_digest(aggregate)}


@torch.inference_mode()
def validate_neutral(model, session, guidance, dense, qkv, regions, key_hw, output_hw, chunk, root, dependency):
    path = Path(root) / "pilots" / "neutral_full_frame.json"
    if path.exists():
        result = read(path)
        _verified_identity(result)
        if result["dependency_identity"] != dependency:
            raise ValueError("neutral validation inputs/operator changed")
        return result
    begin = time.monotonic()
    official = model(guidance, dense, output_size=output_hw, q_chunk_size=chunk)
    neutral = {r: {**item, "occupancy": torch.zeros_like(item["occupancy"]),
                  "key_depth": torch.full_like(item["key_depth"], torch.nan),
                  "query_depth": torch.full_like(item["query_depth"], torch.nan)} for r, item in regions.items()}
    pooled, counters, full = stream_regions(model, qkv, key_hw, output_hw, neutral,
        torch.ones(dense.shape[-2]*dense.shape[-1], device=dense.device), chunk_size=chunk, return_full=True)
    torch.testing.assert_close(full, official, atol=1e-5, rtol=1e-5)
    comparisons = {}
    for r, item in regions.items():
        raw = (official[0] * item["weights"].reshape(1, *output_hw)).sum((1, 2))/(item["weights"].sum()+1e-8)
        reference = project_raw(session, raw, item["signed"])
        variants = {}
        for name, value in pooled[r].items():
            torch.testing.assert_close(value, raw, atol=1e-5, rtol=1e-5)
            vector = project_raw(session, value, item["signed"])
            if reference is None or vector is None:
                raise ValueError("real neutral pilot projection is unavailable")
            np.testing.assert_allclose(vector, reference, atol=1e-5, rtol=1e-5)
            actual = cosine_record(session, vector, item["pixels"])
            expected = cosine_record(session, reference, item["pixels"])
            if actual["class"] != expected["class"]:
                raise ValueError("neutral adaptation changes the original region class")
            variants[name] = {"class": actual["class"], "max_vector_error": float(np.max(np.abs(vector-reference)))}
        comparisons[r] = variants
    result = seal({"status": "REAL_FULL_FRAME_NEUTRALITY_VERIFIED", "dependency_identity": dependency,
        "shape": list(full.shape), "atol": 1e-5, "rtol": 1e-5, "max_dense_error": float((full-official).abs().max()),
        "region_classes": comparisons, "extra_official_AnyUp_QK_computations": 1,
        "extra_stream_attention": counters, "elapsed_seconds": time.monotonic()-begin,
        "GT_input": False, "chunk": chunk})
    atomic_write_json(path, result)
    print("NEUTRAL_FULL_FRAME_VERIFIED", result["max_dense_error"], flush=True)
    return result


@torch.inference_mode()
def acquire_frame(session, model, prepared, fid, *, root, validate=False):
    group = prepared["frames"][str(fid)]
    owners = group["triggered_owners"]
    if not owners:
        return None
    dest = Path(root) / "encoded" / prepared["scene"] / str(fid)
    index = ConsumptionIndex()
    producer = {name: index.identity(Path(__file__).with_name(name))["sha256"]
                for name in ("acquisition.py", "anyup_adapter.py")}
    dependency = canonical_digest({"prepared": prepared["identity"], "frame": str(fid), "producer": producer,
                                   "assets": read(Path(root)/"assets.json")["identity"], "text": session.text_identity})
    path = dest / "receipt.json"
    if path.exists():
        previous = read(path)
        _verified_identity(previous)
        if previous["input_identity"] != dependency:
            raise ValueError("successful acquisition identity changed; explicitly invalidate dependent frame")
        index.identity(previous["vectors"]["path"], previous["vectors"])
        print("ENCODE_REUSE", prepared["scene"], fid, flush=True)
        return previous
    begin = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for item in (group["rgb"], group["evidence"], group["dense_arrays"]):
        index.identity(item["path"], item)
    image = cv2.cvtColor(cv2.imread(group["rgb"]["path"], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    with np.load(group["dense_arrays"]["path"], allow_pickle=False) as arrays:
        dense = torch.from_numpy(arrays["dense"].copy()).to(session.device)
    with np.load(group["evidence"]["path"], allow_pickle=False) as arrays:
        original_owners, depth = arrays["visible_owners"], arrays["depth_m"]
        masks = {k: arrays[k].copy() for k in arrays.files if k.endswith("_target") or "_control_" in k}
    first = prepared["candidates"][owners[0]]
    size, padded, output_hw = first["resized_size"], first["padded_size"], tuple(first["output_size"])
    key_hw = tuple(dense.shape[-2:])
    if list(dense.shape) != group["dense_shape"]:
        raise ValueError("bound cached dense grid changed")
    guidance, aligned_owner, aligned_depth, valid = aligned_inputs(image, original_owners, depth, size, padded, device=session.device)
    unknown = F.interpolate((aligned_owner == 0).float(), size=key_hw, mode="area").flatten()
    regions, target_keys = {}, {}
    for owner in owners:
        candidate = prepared["candidates"][owner]
        rid = candidate["request_id"]
        target_keys[owner] = rid + "_target"
        definitions = [(rid+"_target", int(owner), masks[rid+"_target"])]
        definitions += [(item["array_key"], item["owner"], masks[item["array_key"]]) for item in candidate["controls"]]
        for rkey, region_owner, mask in definitions:
            signed, support, _ = signed_mask(mask, size, padded, key_hw, session.device)
            if rkey.endswith("_target") and int(support.sum()) != candidate["original_support"]:
                raise ValueError("acquisition signed support differs from frozen trigger")
            region = owner_grids(aligned_owner, aligned_depth, valid, region_owner, key_hw, output_hw)
            region.update(weights=F.interpolate((signed > 0).float(), size=output_hw, mode="area").flatten(),
                          signed=signed, owner=region_owner, pixels=int(mask.sum()))
            regions[rkey] = region
    qkv = encode_qkv(model, guidance, dense, output_hw)
    attempts = []
    for chunk in (256, 128, 64):
        attempt = time.monotonic()
        try:
            if validate:
                validate_neutral(model, session, guidance, dense, qkv, regions, key_hw, output_hw, chunk, root, dependency)
            pooled, counters, _ = stream_regions(model, qkv, key_hw, output_hw, regions, unknown, chunk_size=chunk)
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({"chunk": chunk, "status": "TRUE_CUDA_OOM", "error": str(exc), "seconds": time.monotonic()-attempt})
            atomic_write_json(dest/"oom_attempts.json", {"attempts": attempts, "input_identity": dependency})
            torch.cuda.empty_cache()
            if chunk == 64:
                raise
        else:
            attempts.append({"chunk": chunk, "status": "COMPLETE", "seconds": time.monotonic()-attempt})
            break
    vectors, records = {}, {}
    fine_bilinear = F.interpolate(dense, size=output_hw, mode="bilinear", align_corners=False)
    counts = Counter(FC_image_inputs=0, dense_cache_hits=1, AnyUp_QK_computations=1)
    for rkey, item in regions.items():
        representations = dict(pooled[rkey])
        is_target = rkey.endswith("_target")
        if is_target:
            representations["bilinear"] = (fine_bilinear[0]*item["weights"].reshape(1, *output_hw)).sum((1, 2))/(item["weights"].sum()+1e-8)
        records[rkey] = {}
        for name, raw in representations.items():
            # Scientific acquisition shares all three readouts, including control
            # representations for transparent audit; only declared arms consume them.
            vector = project_raw(session, raw, item["signed"])
            counts["head_projections"] += 1
            counts[("target" if is_target else "control")+"_"+name+"_pools"] += 1
            records[rkey][name] = cosine_record(session, vector, item["pixels"])
            if vector is not None:
                vectors[rkey+"__"+name] = vector
        if not is_target:
            vector, audit = region_vector(session.model, session.operators, dense, item["signed"])
            array = vector.cpu().numpy().copy()
            records[rkey]["coarse"] = {**cosine_record(session, array, item["pixels"]), "pool_audit": audit}
            vectors[rkey+"__coarse"] = array
            counts["coarse_control_pools"] += 1
            counts["head_projections"] += 1
    counts.update(counters)
    path_vectors = dest / "vectors.npz"
    _write_npz(path_vectors, vectors)
    torch.cuda.synchronize()
    result = seal({"status": "ENCODED", "input_identity": dependency, "scene": prepared["scene"], "frame_id": int(fid),
        "prepared_identity": prepared["identity"], "representations": records, "target_keys": target_keys,
        "vectors": index.identity(path_vectors), "counts": dict(counts), "chunk_attempts": attempts,
        "elapsed_seconds": time.monotonic()-begin, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "dense_shape": list(dense.shape),
        "guidance_shape": list(guidance.shape), "output_size": list(output_hw), "GT_input": False, "inputs": index.entries()})
    atomic_write_json(path, result)
    print("ENCODED", prepared["scene"], fid, "targets", len(owners), "seconds", round(result["elapsed_seconds"], 2), flush=True)
    return result


def encode(binding, root, scenes=None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = execution_config(binding["reference"])
    session = model = None
    results = []
    load_times = {}
    for scene in (scenes or binding["scenes"]):
        prepared = read(Path(root)/"prepared"/scene/"receipt.json")
        _verified_identity(prepared)
        if not prepared["triggered_frames"]:
            continue
        data = read(binding["scenes"][scene]["context"]["path"])
        new_session = FCSession(config, data, ConsumptionIndex(), device="cuda")
        if session is None:
            session = new_session
            session.load_model()
            load_times["FC"] = session.model_load_seconds
            start = time.monotonic()
            model = load_anyup(root, "cuda")
            torch.cuda.synchronize()
            load_times["AnyUp"] = time.monotonic()-start
        else:
            if session.model_key != new_session.model_key:
                raise ValueError("cohort silently changes the frozen FC model")
            session.text, session.ids, session.text_identity = new_session.text, new_session.ids, new_session.text_identity
        for fid, frame in prepared["frames"].items():
            if frame["triggered_owners"]:
                validate = scene == binding["specification"]["pilot_scenes"][0] and not (Path(root)/"pilots/neutral_full_frame.json").exists()
                results.append(acquire_frame(session, model, prepared, fid, root=root, validate=validate))
    if scenes is None:
        counts = Counter()
        for row in results:
            counts.update(row["counts"])
        result = seal({"status": "ENCODED", "frames": [{"scene": x["scene"], "frame_id": x["frame_id"], "identity": x["identity"]} for x in results],
            "counts": dict(counts), "model_load_seconds": load_times, "new_Native_NQ_training_frontend_map_calls": 0})
        atomic_write_json(Path(root)/"encoded/summary.json", result)
        return result
    atomic_write_json(Path(root)/"pilots/acquisition.json", seal({"scenes": list(scenes), "frames": [x["identity"] for x in results], "model_load_seconds": load_times}))
    return results
