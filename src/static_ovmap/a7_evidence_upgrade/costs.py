"""Required operation unions and separately attributed measured worker work."""

from collections import Counter
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

from .audit import measured_methods
from .operations import method_operations, scene_operations


def summarize_operations(operations):
    """Count heterogeneous units separately; never turn their sum into FLOPs."""
    kinds, models = Counter(), {}
    for operation in operations.values():
        kind = operation["kind"]
        kinds[kind] += 1
        # Decodes and poolings refer to their image operation in the same union.
        parent = operations[operation["image"]] if "image" in operation else operation
        model = parent.get("model_processor", parent.get("model"))
        if model is None:
            raise ValueError("operation lacks an attributable model")
        key = canonical_digest(model)
        entry = models.setdefault(key, {"model": model, "counts": Counter()})
        entry["counts"][kind] += 1
    return {"counts_by_kind": dict(kinds), "by_model": models,
            "unique_operation_count_for_prescribed_tiebreak_only": len(operations)}


def scene_costs(binding, scene):
    root, index = Path(binding["output_root"]), InputIndex()
    spec = read_json(binding["spec"])
    variants = [v["id"] for v in spec["source_variants"]]
    methods, blocked = measured_methods(binding)
    pair = read_json(root / "composition.json")
    index.identity(__file__)
    index.identity(root / "composition.json")
    operations = scene_operations(binding, scene, variants)
    operation_file = root / "operations" / scene / (operations["identity"] + ".json")
    index.identity(operation_file)
    logical = {m: summarize_operations(method_operations(operations["sources"], m, pair)) for m in methods}
    receipts = {
        "E01_shared_readout": root / "e01" / scene / "receipt.json",
        "E02_SAM2_masks": root / "e02/sam2_masks" / scene / "receipt.json",
        "AW_E02_GLOBAL": root / "e02" / scene / "AW_E02_GLOBAL/receipt.json",
        "AW_E02_SAM2": root / "e02" / scene / "AW_E02_SAM2/receipt.json",
        "AW_C0_SO400M": root / "c0/requests" / scene / "receipt.json",
        "AW_E03_FC_FROZEN": root / "e03" / scene / "AW_E03_FC_FROZEN/receipt.json",
        "AW_E03_OVR": root / "e03" / scene / "AW_E03_OVR/receipt.json",
    }
    physical = {}
    for worker, path in receipts.items():
        receipt_identity = index.identity(path)
        receipt = read_json(path)
        values = {k: v for k, v in receipt.items() if (
            k.startswith(("physical_", "peak_gpu_", "reused_", "encoder_calls"))
            or k.endswith("_seconds")
            or k in {"requests", "successful_requests", "successful_masks", "available_owners", "parameter_count"})}
        physical[worker] = {"receipt": receipt_identity, "measured": values}
    result = {
        "status": "SCENE_COSTS_RECORDED", "scene": scene, "binding": binding["identity"],
        "logical_required_visual_operations": logical, "operation_manifest": index.identity(operation_file),
        "physical_completed_worker_invocations": physical, "blocked_methods": blocked,
        "accounting_limits": [
            "Every method requires the original geometry/frontend and N0 map; historical frontend time is not remeasured here.",
            "Logical unions identify required visual work, not FLOPs, latency, or measured physical calls.",
            "Completed receipt invocation counters are not a complete accounting of interrupted or failed earlier invocations.",
            "Worker wall times include loading, I/O and cache validation; parallel wall times must not be summed as end-to-end latency.",
            "Peak memory is per worker, not summed across sequential workers.",
            "E01 uses existing Q image features; E01 shared readout time must not be charged three times.",
            "DIRECT/A7/composition reuse the same source worker work; these are dependencies, not additional inference runs.",
            "Text physical calls are reported separately when measured; historical native/S2 text work is not zero-cost.",
            "Download, smoke, scalar fit, and evaluation costs are separate from these full-scene workers.",
        ],
        "inputs": index.entries(),
    }
    result["identity"] = canonical_digest(result)
    write_once(root / "costs" / (scene + ".json"), result)
    return result
