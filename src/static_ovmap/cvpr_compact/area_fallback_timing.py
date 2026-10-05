"""Independent v2 cold measurements using frozen projection/export production."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity

from .area_fallback import PROTOCOL, region_vector
from .area_fallback_experiment import document, environment, seal
from .recovery_run import ARM_METHOD, gpu_lease, load_recovery_inputs, recover_fc
from .region_worker import FCSession, validate_request_outcomes
from .runtime import require_frozen_execution
from .timing import (FEATURE_TOLERANCE, PRODUCTION_CALLABLE, _finish_parity, _hardware,
                     aggregate_timings, assert_serial_execution, reserve_measurement)


SESSION = "static_ovmap.cvpr_compact.area_fallback_timing.AreaFallbackSession"
ARMS = ("G1_FC", "G3_FC")


class IndependentMemoIndex(ConsumptionIndex):
    """Keep the frozen loader's verification memo inside the independent attempt."""

    def __init__(self, destination, *, memo=None):
        super().__init__(memo or destination)
        self.destination = Path(destination)

    def write_memo(self, path):
        super().write_memo(self.destination)


class AreaFallbackSession(FCSession):
    def encode(self, requests, loaders):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask

        if self.cache is not None or self.model is None:
            raise ValueError("v2 cold encoding requires a resident model and no persistent cache")
        groups = {}
        for rid, request in requests.items():
            groups.setdefault(fc_image_identity(request, self.model_key), []).append(rid)
        stats = {"physical_image_encodings": 0, "encoder_batch_calls": 0, "physical_region_poolings": 0,
            "area_fallback_poolings": 0, "dense_cache_hits": 0, "region_cache_hits": 0,
            "legacy_region_cache_hits": 0, "persistent_feature_cache_enabled": False,
            "required_dense_receipts": {}, "required_region_receipts": {}, "requests": {},
            "protocol": PROTOCOL, "region_session": SESSION}
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
                    stats["physical_image_encodings"] += 1
                    stats["encoder_batch_calls"] += 1
                    dense = self.operators["extract_features_convnext"](SimpleNamespace(clip_model=self.model), image)["clip_vis_dense"]
                    if dense.dtype != torch.float32 or dense.ndim != 4 or not torch.isfinite(dense).all():
                        raise ValueError("FC dense operator returned invalid precision/shape/values")
                    for rid in rids:
                        value, mask = values[rid], values[rid]["target"]
                        if not np.array_equal(value["image"], values[rids[0]]["image"]):
                            raise ValueError("one physical RGB digest produced different decoded images")
                        if mask.dtype != np.bool_ or mask.shape != value["image"].shape[:2]:
                            raise ValueError("FC target mask must remain full-resolution boolean evidence")
                        key = canonical_digest({"image": tensor_key, "mask": _array_digest(mask), "region": PROTOCOL})
                        row = {"request_id": rid, "frame_id": requests[rid]["frame_id"], "status": "PENDING",
                            "image_content_key": image_key, "input_tensor_key": tensor_key,
                            "feature_content_key": key, "target_mask_sha256": _array_digest(mask), "physical_cache_hit": False}
                        stats["requests"][rid] = row
                        if key in pooled:
                            previous = pooled[key]
                            features[rid] = features[previous]
                            original = stats["requests"][previous]
                            row.update({k: original[k] for k in ("fallback", "original_support", "dense_hw", "area_support", "area_mass") if k in original})
                            row.update(status="COMPLETE", within_run_region_alias=previous)
                            continue
                        signed, _, _ = signed_mask(mask, size, image.shape[-2:], dense.shape[-2:], self.device)
                        stats["physical_region_poolings"] += 1
                        vector, audit = region_vector(self.model, self.operators, dense, signed)
                        features[rid] = vector.cpu().numpy().copy()
                        stats["area_fallback_poolings"] += int(audit["fallback"])
                        pooled[key] = rid
                        row.update(status="COMPLETE", **audit)
                except BaseException as exc:
                    for rid in rids:
                        if rid not in features:
                            stats["requests"].setdefault(rid, {"request_id": rid})
                            stats["requests"][rid].update(status="UNAVAILABLE_TECHNICAL_FAILURE", reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    del dense, image, values, pooled
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
        validate_request_outcomes(requests, features, stats, require_success=False)
        return features, stats


def scientific_parent(output, scene, inputs, index):
    regions = document(output / "regions" / scene / "receipt.json", index)
    lock = document(output / "predictions" / scene / "receipt.json", index)
    if (regions["protocol"] != PROTOCOL or regions["status"] != "COMPLETE"
            or lock["status"] != "PREDICTIONS_LOCKED" or regions["scene"] != scene or lock["scene"] != scene):
        raise ValueError("v2 timing requires complete same-scene scientific sources and locked predictions")
    registry_ids, exports = set(), {}
    for arm, item in regions["sources"].items():
        source = document(item["path"], index, item)
        if source["model_identity"] != inputs.data["FC_physical_model_identity"] or source["valid_ids"] != inputs.valid_ids:
            raise ValueError("scientific v2 FC model/vocabulary differs from resident timing inputs")
        registry_ids.add(source["registry_identity"])
        path = Path(lock["predictions"][ARM_METHOD[arm]])
        manifest = document(path, index, verify=False)
        exports[ARM_METHOD[arm]] = {"manifest": str(path), "files": [index.identity(path),
            index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"])]}
    if len(registry_ids) != 1 or set(regions["sources"]) != {"G1", "G3"}:
        raise ValueError("v2 timing requires both fixed projected arms on one registry")
    index.identity(regions["features"]["path"], regions["features"])
    return seal({"status": "COMPLETE", "scene": scene, "protocol": PROTOCOL,
        "resident_input_identity": inputs.identity, "registry_identity": registry_ids.pop(),
        "plan": regions["plan"], "sources": regions["sources"], "features": regions["features"],
        "exports": exports, "requests": regions["requests"], "regions_identity": regions["identity"],
        "prediction_lock_identity": lock["identity"]})


def validate_mask_parity(scientific, cold, arm):
    selected = {rid for rows in scientific["plan"][arm].values() for rid in rows}
    if set(cold["requests"]) != selected:
        raise ValueError("cold recovery changed selected masks")
    for rid in selected:
        first, second = scientific["requests"][rid], cold["requests"][rid]
        for key in ("target_mask_sha256", "input_tensor_key", "image_content_key", "fallback"):
            if first[key] != second[key]:
                raise ValueError("cold recovery changed RGB preprocessing, mask or fallback scope")
        support = first.get("original_support", first.get("dense_mask_support"))
        if support != second["original_support"]:
            raise ValueError("cold recovery changed original hard-mask support")


def finish_parity(row, path, scientific, index):
    cold = document(row["recovery_receipt"], index)
    try:
        validate_mask_parity(scientific, cold, row["arm"].removesuffix("_FC"))
    except BaseException as exc:
        row.update(status="PARITY_FAILED", parity={"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(path, seal(row))
        raise
    _finish_parity(row, path, scientific, index)


def aggregate_v2_timings(spec, receipts):
    scenes = spec["cohorts"]["replica8"]
    expected = {(scene, arm) for scene in scenes for arm in ARMS}
    if len(scenes) != 8 or len(receipts) != 16 or {(row["scene"], row["arm"]) for row in receipts} != expected:
        raise ValueError("v2 timing requires exactly 16 distinct leaves across all eight scenes")
    for row in receipts:
        if (row.get("protocol") != PROTOCOL or row.get("region_session") != SESSION
                or row.get("model_load_included") is not False):
            raise ValueError("cold measurement changed the v2 operator or model-resident contract")
        if row.get("dense_cache_hits") != 0 or row.get("region_cache_hits") != 0:
            raise ValueError("cold measurement read a persistent feature cache")
    narrowed = copy.deepcopy(spec)
    narrowed["timing"]["arms"] = list(ARMS)
    result = aggregate_timings(narrowed, receipts)
    result.update(protocol=PROTOCOL, region_session=SESSION)
    return seal(result)


def run_timings(parent, output):
    import torch

    parent, output, binding, spec, _ = environment(parent, output)
    root = output / "timing"
    index = IndependentMemoIndex(root / "input_verifications.json", memo=output / "input_verifications.json")
    experiment = document(output / "experiment.json", index)
    for item in (experiment["producer"], experiment["driver"]):
        index.identity(item["path"], item)
    scenes = spec["cohorts"]["replica8"]
    assert_serial_execution(binding)
    hardware = _hardware(binding["gpu"])
    producer = index.identity(__file__)
    plan = seal({"protocol": PROTOCOL, "status": "PREFLIGHT_COMPLETE", "scene_order": scenes,
        "arms": list(ARMS), "repeats_per_scene_arm": 1, "planned_calls": 16, "binding_identity": binding["identity"],
        "scientific_experiment_identity": experiment["identity"], "producer": producer,
        "pooling_producer": experiment["producer"], "feature_tolerance": FEATURE_TOLERANCE,
        "discrete_output_parity": "EXACT", "include": spec["timing"]["include"], "exclude": spec["timing"]["exclude"]})
    plan_path = root / "plan.json"
    if plan_path.exists() and document(plan_path, index) != plan:
        raise ValueError("reserved v2 timing plan changed; measured calls cannot be replayed")
    atomic_write_json(plan_path, plan)
    measurements, resident = [], None
    with exclusive_lock(root / ".controller.lock"), gpu_lease(binding, root) as lease:
        for scene in scenes:
            require_frozen_execution(binding, scene)
            inputs = load_recovery_inputs(binding, scene, index=index)
            scientific = scientific_parent(output, scene, inputs, index)
            session = None
            for arm in ARMS:
                core = arm.removesuffix("_FC")
                leaf, path = root / scene / arm, root / scene / arm / "receipt.json"
                identity = canonical_digest({"plan": plan["identity"], "scientific": scientific["identity"],
                    "resident_inputs": inputs.identity, "arm": arm, "hardware": hardware})
                if path.exists():
                    row = document(path, index, verify=False)
                    if row["input_identity"] != identity:
                        raise ValueError("reserved cold measurement changed inputs; replay is forbidden")
                    if row["status"] == "MEASURED":
                        finish_parity(row, path, scientific, index)
                    elif row["status"] != "COMPLETE":
                        raise RuntimeError(f"cold measurement is {row['status']}; a second physical call is forbidden")
                    row = document(path, index)
                    for item in row["outputs"]:
                        index.identity(item["path"], item)
                    measurements.append(row)
                    continue
                assert_serial_execution(binding)
                if session is None:
                    session = AreaFallbackSession(binding, inputs.data, index, cache=None)
                    if resident is None:
                        session.load_model()
                        event = seal({"model_identity": session.model_key, "text_identity": session.text_identity,
                            "seconds": session.model_load_seconds, "hardware": hardware, "weight_audit": session.weight_audit,
                            "excluded_from_incremental_timer": True, "loaded_at_unix": time.time()})
                        model_path = root / "model_loading" / (str(time.time_ns()) + ".json")
                        atomic_write_json(model_path, event)
                        resident = (session.model_key, session.text_identity["sha256"], list(session.ids),
                            session.model, session.weight_audit, index.identity(model_path))
                    else:
                        if (session.model_key, session.text_identity["sha256"], list(session.ids)) != resident[:3]:
                            raise ValueError("resident FC model/text identity changed between Replica scenes")
                        session.model, session.weight_audit = resident[3:5]
                row = reserve_measurement(path, binding["identity"], scene, arm, identity)
                row.update(protocol=PROTOCOL, region_session=SESSION, hardware=hardware, resident_model=True,
                    model_load_reference=resident[5], model_load_included=False, production_callable=PRODUCTION_CALLABLE,
                    persistent_feature_cache_enabled=False, persistent_view_cache_enabled=False, persistent_result_cache_enabled=False,
                    frozen_common_input_identity=inputs.identity, scientific_identity=scientific["identity"], producer=producer,
                    include=plan["include"], exclude=plan["exclude"], status="CALL_STARTED", started_at_unix=time.time())
                atomic_write_json(path, row)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    result = recover_fc(binding, inputs, leaf / "call", arms=(core,), session=session, cold=True, lease=lease)
                    torch.cuda.synchronize()
                    row.update(status="MEASURED", measured_seconds=time.perf_counter() - started,
                        recovery_receipt=str(leaf / "call/receipt.json"), recovery_identity=result["identity"],
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), candidate_count=result["candidate_count"],
                        source_available_additions=result["source_available_additions"][core],
                        inputs=index.entries(), outputs=[index.identity(leaf / "call/receipt.json"), *result["outputs"]])
                    for key in ("physical_image_encodings", "encoder_batch_calls", "physical_region_poolings",
                                "area_fallback_poolings", "dense_cache_hits", "region_cache_hits"):
                        row[key] = result[key]
                    atomic_write_json(path, row)
                except BaseException as exc:
                    row.update(status="FAILED_MEASUREMENT", observed_elapsed_seconds=time.perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}", costs=getattr(session, "last_stats", {}), inputs=index.entries(), outputs=[])
                    failed = leaf / "call/receipt.json"
                    if failed.exists():
                        row["outputs"] = [index.identity(failed), *read(failed).get("outputs", [])]
                    atomic_write_json(path, seal(row))
                    raise
                finish_parity(row, path, scientific, index)
                measurements.append(document(path, index))
                index.write_memo(root / "input_verifications.json")
                print(f"Cold v2 {scene}/{arm}: {row['measured_seconds']:.3f}s, parity PASS", flush=True)
    result = aggregate_v2_timings(spec, measurements)
    events = [document(path, index) for path in sorted((root / "model_loading").glob("*.json"))]
    result.update(model_loading_events=events, model_loading_seconds_total=sum(row["seconds"] for row in events),
        physical_image_encodings=sum(row["physical_image_encodings"] for row in measurements),
        physical_region_poolings=sum(row["physical_region_poolings"] for row in measurements),
        area_fallback_poolings=sum(row["area_fallback_poolings"] for row in measurements),
        plan_identity=plan["identity"], producer=producer, inputs=index.entries(),
        outputs=[index.identity(root / row["scene"] / row["arm"] / "receipt.json") for row in measurements]
            + [item for row in measurements for item in row["outputs"]])
    atomic_write_json(root / "pool.json", seal(result))
    index.write_memo(root / "input_verifications.json")
    print("V2 COLD COMPLETE", {arm: row["mean_seconds"] for arm, row in result["arms"].items()}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_timings(args.parent, args.output)
