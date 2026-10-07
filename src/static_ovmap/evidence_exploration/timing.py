"""Resident models, cold views/features/results; synchronized incremental latency."""

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import gc
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from static_ovmap.a7_evidence_upgrade.recognition_worker import aggregate_views
from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.area_fallback import region_vector
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.outputs import construct_output
from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, _verified_identity
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.runtime_parity.instrumentation import Stages
from static_ovmap.runtime_parity.runner import execution_config, load_common
from static_ovmap.runtime_parity.views import build_views, CallLoader

from .acquisition import load_anyup, project_raw, cosine_record
from .anyup_adapter import aligned_inputs, owner_grids, encode_qkv, geometry_factor, reweight_attention
from .evidence import occupancy_trigger, competing_regions
from .outputs import prediction_content
from .verifier import decide


@torch.inference_mode()
def stream_selected(model, qkv, key_hw, output_hw, regions, unknown, variant, counters):
    """Same frozen attention/readout; standalone cost computes the requested arm."""
    from anyup.layers.attention.attention_masking import window2d
    q, k, v = qkv
    cross = model.cross_decode.cross_attn
    weights = torch.stack([x["weights"].flatten() for x in regions.values()])
    if (weights.sum(1) <= 0).any() or not torch.isfinite(weights).all():
        raise ValueError("nonempty cold bound region has invalid area mass")
    active = torch.nonzero(weights.sum(0) > 0).flatten()
    windows = window2d(tuple(key_hw), tuple(output_hw), model.cross_decode.window_ratio).reshape(-1, 4).to(q.device)
    ky = torch.arange(key_hw[0], device=q.device).repeat_interleave(key_hw[1])
    kx = torch.arange(key_hw[1], device=q.device).repeat(key_hw[0])
    qnorm, knorm = cross.norm_q(q), cross.norm_k(k)
    attempts = []
    for chunk in (256, 128, 64):
        start_attempt = time.monotonic()
        sums = {name: v.new_zeros(v.shape[-1]) for name in regions}
        attempt_counts = Counter()
        try:
            for start in range(0, len(active), chunk):
                ids = active[start:start+chunk]
                bounds = windows[ids]
                mask = ~((ky[None] >= bounds[:, :1]) & (ky[None] < bounds[:, 1:2])
                         & (kx[None] >= bounds[:, 2:3]) & (kx[None] < bounds[:, 3:4])) if model.cross_decode.window_ratio > 0 else None
                if mask is not None and mask.all(1).any():
                    raise ValueError("original AnyUp all-masked query: representation unavailable")
                _, attention = cross.attention(qnorm[:, ids], knorm, k, average_attn_weights=True, attn_mask=mask)
                attention = attention[0]
                if not torch.isfinite(attention).all() or (attention.sum(1) <= 0).any():
                    raise ValueError("invalid cold original attention")
                attempt_counts.update(query_chunks=1, original_attention_applications=1, query_locations=len(ids))
                for name, region in regions.items():
                    mass = region["weights"].flatten()[ids]
                    positive = mass > 0
                    if not positive.any():
                        continue
                    a = attention[positive]
                    if variant != "anyup":
                        factor = geometry_factor(region["occupancy"], unknown, region["query_depth"][ids[positive]],
                            region["key_depth"], use_depth=variant != "owner_only")
                        a = reweight_attention(a, factor)
                        attempt_counts["geometry_attention_applications"] += 1
                    sums[name] += (mass[positive] @ a) @ v[0]
        except torch.cuda.OutOfMemoryError as exc:
            counters.update(attempt_counts)
            attempts.append({"chunk": chunk, "status": "TRUE_CUDA_OOM", "reason": str(exc), "seconds": time.monotonic()-start_attempt})
            torch.cuda.empty_cache()
            if chunk == 64:
                raise
        else:
            counters.update(attempt_counts)
            attempts.append({"chunk": chunk, "status": "COMPLETE", "seconds": time.monotonic()-start_attempt})
            return {name: value/(regions[name]["weights"].sum()+1e-8) for name, value in sums.items()}, attempts


class Timer:
    """Five exclusive host stages, plus nonadditive CUDA event annotations."""
    def __init__(self):
        self.seconds = dict.fromkeys(("view_search", "additional_geometry", "FC", "AnyUp", "output"), 0.)
        self.events = {"FC": [], "AnyUp": []}
        self.counts = Counter()

    @contextmanager
    def stage(self, name, cuda=False):
        start = time.perf_counter()
        if cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        try:
            yield
        finally:
            if cuda:
                b.record()
                self.events[name].append((a, b))
            self.seconds[name] += time.perf_counter()-start


@torch.inference_mode()
def cold_call(binding, root, inputs, session, anyup, method, repeat):
    root = Path(root)
    name, scene = method["id"], inputs.scene
    dest = root/"timing"/"calls"/(str(repeat)+"_"+scene+"_"+name)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("cold recovery output directory must be empty")
    index, timer, stages = ConsumptionIndex(), Timer(), Stages()
    config = execution_config(binding["reference"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with timer.stage("view_search"):
        registry, manifest = build_views(inputs, dest/"views", "R2_ROI_EXACT", index, stages)
        loader = CallLoader(dest/"views/manifest.json", index, stages, reuse=True)
        selected = [x[0] for x in manifest["g1"].values() if x]
        groups = {}
        for rid in selected:
            request = manifest["requests"][rid]
            groups.setdefault(str(request["frame_id"]), []).append(rid)
        sources = {str(x["raw_owner"]): {"available": False, "label": None, "scores": None} for x in registry["candidates"]}
    requires_full_owners = name in ("EV03_CONTRAST", "EV06_OWNER_ANYUP", "EV07_COMBINATION", "EV08_NO_DEPTH")
    fine = method["representation"] in ("bilinear", "anyup")
    with timer.stage("additional_geometry"):
        projector = FullSceneProjector(inputs.xyz, inputs.faces, inputs.raw) if requires_full_owners and groups else None
    decisions, frame_details = {}, {}
    for fid, rids in groups.items():
        with timer.stage("FC", cuda=True):
            values = {rid: loader(rid) for rid in rids}
            rgb = values[rids[0]]["image"]
            image, size = image_tensor(rgb, session.device)
            padded = tuple(image.shape[-2:])
            dense = session.operators["extract_features_convnext"](SimpleNamespace(clip_model=session.model), image)["clip_vis_dense"]
            if dense.dtype != torch.float32 or not torch.isfinite(dense).all():
                raise ValueError("invalid cold FC dense tensor")
            timer.counts["FC_image_inputs"] += 1
            candidates, signed_masks = {}, {}
            for rid in rids:
                request, mask = manifest["requests"][rid], values[rid]["target"]
                owner = str(request["raw_owner"])
                signed, support, _ = signed_mask(mask, size, padded, dense.shape[-2:], session.device)
                vector, audit = region_vector(session.model, session.operators, dense, signed)
                vector = vector.cpu().numpy().copy()
                record = cosine_record(session, vector, int(mask.sum()))
                sources[owner] = {"available": True, "label": record["class"], "scores": record["scores"],
                                  "aggregate_vector_sha256": record["aggregate_vector_sha256"]}
                w = F.interpolate((signed > 0).float(), size=dense.shape[-2:], mode="area")[0, 0].cpu().numpy()
                triggered, purity = occupancy_trigger(w, int(support.sum()))
                candidates[owner] = {"request_id": rid, "triggered": triggered, "purity": purity,
                    "controls": [], "mask": mask, "signed": signed, "fallback": audit["fallback"]}
                signed_masks[rid+"_target"] = signed
                timer.counts["coarse_target_pools"] += 1
                timer.counts["head_projections"] += 1
        with timer.stage("additional_geometry"):
            if projector is not None:
                frame = loader.frames[int(fid)]
                depth_path = loader.capture_path.parent/frame["depth_path"]
                index.identity(depth_path, {"sha256": frame["depth_sha256"]})
                with np.load(depth_path, allow_pickle=False) as arrays:
                    depth = arrays["depth_m"]
                projection = projector.project_frame(np.asarray(frame["intrinsics"]), np.asarray(frame["pose_c2w"]),
                    depth, [int(x) for x in candidates])
                owner_map = projection["owners"]
                timer.counts["additional_full_scene_projection_calls"] += 1
                for owner, item in candidates.items():
                    if not np.array_equal(item["mask"], owner_map == int(owner)):
                        raise ValueError("cold full-scene selected mask differs from R2 G1")
                    if item["triggered"] and method["decision"] == "contrast_margin":
                        item["controls"] = competing_regions(item["mask"], owner_map, manifest["requests"][item["request_id"]]["canonical_bbox"])
        active = {owner: item for owner, item in candidates.items() if item["triggered"]}
        readouts, region_items = {}, {}
        with timer.stage("FC", cuda=True):
            for owner, item in active.items():
                definitions = [(item["request_id"]+"_target", int(owner), item["mask"], item["signed"])]
                for control in item["controls"]:
                    key = item["request_id"]+"_control_"+str(control["owner"])
                    signed, _, _ = signed_mask(control["mask"], size, padded, dense.shape[-2:], session.device)
                    definitions.append((key, control["owner"], control["mask"], signed))
                    if not fine:
                        vector, _ = region_vector(session.model, session.operators, dense, signed)
                        readouts[key] = cosine_record(session, vector.cpu().numpy().copy(), control["pixels"])
                        timer.counts.update(coarse_control_pools=1, head_projections=1)
                if fine:
                    for key, region_owner, mask, signed in definitions:
                        region_items[key] = {"owner": region_owner, "pixels": int(mask.sum()), "signed": signed,
                            "weights": F.interpolate((signed > 0).float(), size=tuple(x//4 for x in padded), mode="area").flatten()}
        if fine and active:
            output_hw = tuple(x//4 for x in padded)
            if method["representation"] == "bilinear":
                with timer.stage("FC", cuda=True):
                    up = F.interpolate(dense, size=output_hw, mode="bilinear", align_corners=False)
                    for key, item in region_items.items():
                        raw = (up[0]*item["weights"].reshape(1, *output_hw)).sum((1, 2))/(item["weights"].sum()+1e-8)
                        readouts[key] = cosine_record(session, project_raw(session, raw, item["signed"]), item["pixels"])
                        timer.counts.update(bilinear_target_pools=1, head_projections=1)
                    del up
            else:
                with timer.stage("AnyUp", cuda=True):
                    if projector is None:
                        owner_map = np.zeros(rgb.shape[:2], np.int64)
                        depth = np.zeros(rgb.shape[:2], np.float32)
                    guidance, own, dep, valid = aligned_inputs(rgb, owner_map, depth, size, padded, device=session.device)
                    key_hw = tuple(dense.shape[-2:])
                    unknown = F.interpolate((own == 0).float(), size=key_hw, mode="area").flatten()
                    for item in region_items.values():
                        item.update(owner_grids(own, dep, valid, item["owner"], key_hw, output_hw))
                    qkv = encode_qkv(anyup, guidance, dense, output_hw)
                    timer.counts["AnyUp_QK_computations"] += 1
                    variant = "owner_depth" if name in ("EV06_OWNER_ANYUP", "EV07_COMBINATION") else "owner_only" if name == "EV08_NO_DEPTH" else "anyup"
                    pooled, attempts = stream_selected(anyup, qkv, key_hw, output_hw, region_items, unknown, variant, timer.counts)
                    for key, raw in pooled.items():
                        item = region_items[key]
                        readouts[key] = cosine_record(session, project_raw(session, raw, item["signed"]), item["pixels"])
                        timer.counts.update(AnyUp_target_or_control_pools=1, head_projections=1)
                    frame_details[fid] = {"chunk_attempts": attempts, "variant": variant}
                    del qkv, guidance, own, dep, valid, pooled
        with timer.stage("output"):
            for owner, item in candidates.items():
                target = readouts.get(item["request_id"]+"_target")
                controls = [readouts[item["request_id"]+"_control_"+str(x["owner"])] for x in item["controls"]]
                decisions[owner] = decide(sources[owner], inputs.valid_ids,
                    triggered=item["triggered"] and name != "EV01_G1_V2", purity=item["purity"],
                    target_scores=target.get("scores") if target else None,
                    controls=[x["scores"] for x in controls if x["feature_available"]], decision=method["decision"],
                    representation_fallback=target.get("fallback_reason") if target else None)
        # Within-call decoded-image reuse is bounded to this one active frame.
        loader.images.clear()
        del dense, image, values, candidates, region_items, signed_masks, readouts
    with timer.stage("output"):
        for owner, baseline in sources.items():
            if owner not in decisions:
                decisions[owner] = decide(baseline, inputs.valid_ids, triggered=False, purity=1., decision=method["decision"])
        d2, _ = fuse_readout(inputs.sources, config["final_temperatures"], "D2", inputs.valid_ids, owner_labels(inputs.baseline))
        accepted = {int(owner): item["proposed_class"] for owner, item in decisions.items() if item["accepted"]}
        payload = construct_output(inputs.baseline, inputs.raw, registry, accepted, inputs.valid_ids,
            inputs.nearest, inputs.matched, name, incumbent_labels=d2, metadata={"method_recipe": method, "cold": True})
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-begin
    memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    gpu_seconds = {name: sum(a.elapsed_time(b) for a, b in events)/1000 for name, events in timer.events.items()}
    # All parity and content hashing occur after the timed recovery returns.
    acquired = read(root/"predictions"/scene/"receipt.json")["outcomes"][name]
    content = prediction_content(payload)
    if content != acquired["content_identity"]:
        atomic_write_json(dest/"parity_failure.json", {"method": name, "scene": scene, "timed": content,
            "acquired": acquired["content_identity"], "decisions": decisions, "seconds": elapsed, "counts": dict(timer.counts)})
        raise ValueError("cold prediction differs from fixed scientific acquisition: " + scene + "/" + name)
    if sum(timer.seconds.values()) > elapsed+1e-8:
        raise ValueError("exclusive stage sum exceeds synchronized wall")
    result = seal({"status": "COLD_CALL_PARITY_VERIFIED", "scene": scene, "method": name, "repeat": repeat,
        "seconds_per_scene": elapsed, "exclusive_host_seconds": timer.seconds,
        "boundary_sync_and_overhead_seconds": elapsed-sum(timer.seconds.values()),
        "nonadditive_cuda_event_seconds": gpu_seconds, "counts": dict(timer.counts), "frame_details": frame_details,
        "prediction_content_identity": content, "correctness_checked_after_timer": True, **memory,
        "models_common_geometry_NQF_resident": True, "recovery_views_features_results_cold": True,
        "persistent_feature_view_result_cache_hits": 0, "OS_page_cache": "UNCONTROLLED_NOT_CLEARED"})
    atomic_write_json(dest/"receipt.json", result)
    print("COLD_CALL", repeat, scene, name, round(elapsed, 3), "PARITY_VERIFIED", flush=True)
    return result


def time_study(binding, root):
    root = Path(root)
    selection = read(root/"selection.json")
    _verified_identity(selection)
    if selection["status"] != "RESEARCH_SELECTION_FROZEN":
        raise ValueError("paired latency cannot influence metric selection")
    methods = {x["id"]: x for x in binding["specification"]["methods"]}
    selected = selection["timing_methods"]
    if len(selected) > 4:
        raise ValueError("timing slate exceeds the declared bound")
    source = ConsumptionIndex().identity(__file__)
    timing_binding = seal({"selection_identity": selection["identity"], "source": source,
        "method_order": selected, "repeats": 2, "scene_order": binding["cohorts"]["replica8"]})
    freeze_path = root/"timing/binding.json"
    if freeze_path.exists():
        if read(freeze_path) != timing_binding:
            raise ValueError("completed timing operator/slate changed")
    else:
        atomic_write_json(freeze_path, timing_binding)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = execution_config(binding["reference"])
    data = read(binding["scenes"][selection["timing_scene_order_round1"][0]]["context"]["path"])
    session = FCSession(config, data, ConsumptionIndex(), device="cuda")
    session.load_model()
    start = time.perf_counter()
    anyup = load_anyup(root, "cuda")
    torch.cuda.synchronize()
    anyup_load = time.perf_counter()-start
    calls = []
    for repeat in (1, 2):
        order = binding["cohorts"]["replica8"] if repeat == 1 else list(reversed(binding["cohorts"]["replica8"]))
        method_order = selected if repeat == 1 else list(reversed(selected))
        for scene in order:
            inputs = load_common(binding["reference"], scene, root/"timing/common"/scene)
            for name in method_order:
                path = root/"timing/calls"/(str(repeat)+"_"+scene+"_"+name)/"receipt.json"
                if path.exists():
                    row = read(path)
                    _verified_identity(row)
                    if row["status"] != "COLD_CALL_PARITY_VERIFIED":
                        raise ValueError("timing receipt is not a valid completed call")
                    calls.append(row)
                else:
                    gc.collect()
                    calls.append(cold_call(binding, root, inputs, session, anyup, methods[name], repeat))
            del inputs
    if len(calls) != 16*len(selected) or len(calls) > 64:
        raise ValueError("paired scene/repeat/method coverage incomplete")
    summary = []
    for name in selected:
        rows = [x for x in calls if x["method"] == name]
        counts = Counter()
        for row in rows:
            counts.update(row["counts"])
        summary.append({"method": name, "calls": len(rows), "mean_seconds_per_scene": float(np.mean([x["seconds_per_scene"] for x in rows])),
            "median_seconds_per_scene": float(np.median([x["seconds_per_scene"] for x in rows])),
            "peak_cuda_allocated_bytes": max(x["peak_cuda_allocated_bytes"] for x in rows),
            "peak_cuda_reserved_bytes": max(x["peak_cuda_reserved_bytes"] for x in rows), "counts": dict(counts),
            "mean_exclusive_host_seconds": {stage: float(np.mean([x["exclusive_host_seconds"][stage] for x in rows])) for stage in rows[0]["exclusive_host_seconds"]}})
    result = seal({"status": "PAIRED_COLD_TIMING_COMPLETE", "binding_identity": timing_binding["identity"], "summary": summary,
        "calls": calls, "model_load_seconds": {"FC": session.model_load_seconds, "AnyUp": anyup_load},
        "GPU": {"name": torch.cuda.get_device_name(), "cuda": torch.version.cuda}, "CPU": platform.processor(),
        "threads": {"torch": torch.get_num_threads(), "raycasting": 4}, "all_valid_repeats_included": True,
        "online_30FPS_validated": False})
    atomic_write_json(root/"timing/summary.json", result)
    return result
