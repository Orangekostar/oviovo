"""Method-specific standalone image requirements versus the measured baseline."""

from pathlib import Path

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import read
from .light import METHODS, RECOVERIES


def light_costs(binding, scene, *, map_id="BB00_NATIVE", context=None):
    data = context or binding["scenes"][scene]
    baseline = read(data["parent_readout_receipt"])
    baseline_fc = read(Path(data["fc_root"]) / "receipt.json")
    root = Path(binding["output_root"]) / "recovery" / scene / map_id
    restored, plan, fc = (read(root / (name + ".json")) for name in
                          ("cached_sources", "fc_request_plan", "fc_recovery_receipt"))
    baseline_images = set(baseline_fc["required_image_contents"])
    baseline_regions = set(baseline_fc["required_region_receipts"])
    result = {}
    for method in METHODS:
        arm = RECOVERIES.get(method)
        images, regions, attempted, physical_images, physical_poolings = set(), set(), [], 0, 0
        if arm in {"U2", "U3"}:
            for rid in restored[arm + "_request_ids"]:
                allowance, row = plan["requests"][rid], fc["requests"][rid]
                if not allowance["authorized"]:
                    continue
                attempted.append(rid)
                if row["status"] in {"COMPLETE", "UNAVAILABLE_TECHNICAL_FAILURE"}:
                    images.add(allowance["image_content_identity"])
                if row.get("content_identity"):
                    regions.add(row["content_identity"])
                physical_images += row["physical_image_encodings"]
                physical_poolings += row["physical_region_poolings"]
        additional = images - baseline_images
        result[method] = {"scene": scene, "map_id": map_id, "method": method,
            "standalone_new_encoder_inputs": len(additional), "standalone_new_FC_image_contents": sorted(additional),
            "standalone_required_FC_image_contents": sorted(baseline_images | images),
            "standalone_additional_region_poolings": len(regions - baseline_regions),
            "standalone_required_image_inputs_including_baseline": baseline["required_image_encodings"] + len(additional),
            "method_FC_attempted_request_ids": attempted, "method_recovery_FC_image_contents": sorted(images),
            "method_recovery_region_contents": sorted(regions),
            "shared_run_physical_FC_image_inputs": physical_images,
            "shared_run_physical_FC_region_poolings": physical_poolings,
            "new_native_or_Q_acquisitions": 0, "new_text_inputs": 0,
            "U1_UQ_evidence_already_paid_by_baseline": arm in {"U1", "UQ"},
            "physical_counts_are_shared_operation_attribution_not_cold_method_runs": True,
            "timing_kind": "WORKER_WALL_TIME_WITH_EXPLICIT_MISSING_COMPONENTS",
            "timing_missing_components": fc["standalone_timing_missing_components"]}
    receipt = {"status": "COMPLETE", "scene": scene, "map_id": map_id, "methods": result,
        "baseline_readout_identity": baseline["identity"], "FC_plan_identity": plan["identity"],
        "FC_producer_identity": fc["identity"], "cold_runs_performed": False}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(Path(binding["output_root"]) / "costs" / scene / map_id / "light.json", receipt)
    return receipt


def map_cost(binding, scene, map_id):
    root = Path(binding["output_root"])
    current = read(root / "readouts" / scene / map_id / "receipt.json")
    baseline = read(binding["scenes"][scene]["parent_readout_receipt"])
    additions = {key: value for key, value in current["standalone_required_contents"].items()
                 if key not in baseline["standalone_required_contents"]}
    mapping = read(root / "maps" / scene / map_id / "map_receipt.json")
    sam_inputs = read(binding["scenes"][scene]["sam_receipt"])["counters"]["physical_image_encodings"] \
        if mapping["recipe"]["frontend"] != "cropformer" else 0
    return {"scene": scene, "map_id": map_id, "standalone_new_encoder_inputs": sum(additions.values()) + sam_inputs,
        "standalone_new_N_Q_F_contents": additions, "standalone_SAM_image_inputs": sam_inputs,
        "actual_new_SAM_image_inputs": 0, "actual_new_CropFormer_image_inputs": 0,
        "physical_native_Q_image_inputs": current["physical_native_Q_image_inputs"],
        "physical_FC_image_inputs": current["physical_FC_image_inputs"],
        "physical_FC_region_poolings": current["physical_FC_region_poolings"],
        "standalone_required_image_encodings": current["required_image_encodings"],
        "attributable_map_plus_readout_seconds": current["attributable_map_plus_readout_seconds"],
        "timing_kind": current["timing_kind"], "missing_timings": current["standalone_timing_missing_components"]}
