"""Input-exact operation unions, separate from paid and physical work."""

from pathlib import Path

from src.static_ovmap.composition_study.io import read_json
from src.static_ovmap.module_validation.contracts import canonical_digest


def operation_key(model_identity, request):
    keys = ("image_sha256", "target_mask_sha256", "native_union_mask_sha256", "bbox_xyxy", "crop_convention")
    return canonical_digest({"model_processor": model_identity, "precision": "float32", "crop_count": 6,
                             "actual_inputs": {k: request[k] for k in keys}})


def scene_operations(binding, scene):
    config = read_json(binding["scenes"][scene]["config"])
    data = config["scenes"][scene]
    cap = read_json(data["capture"])
    requests = {r["request_id"]: r for f in cap["frames"] for r in f["requests"]}
    native_paid = [requests[r] for f in cap["frames"] for r in f["native_selected_request_ids"]]
    query = read_json(Path(config["attempt_root"]) / "query" / scene / "Q_GAIN/decisions.json")
    static = read_json(Path(data["source_directory"]) / "semantic_requests.json")
    paid = {"N0": native_paid, "Q_GAIN": [r["request"] for r in query["paid_requests"]], "S_SIGLIP2_AREA": list(static["requests"].values())}
    operations = {name: sorted({operation_key(config["models"]["siglip2" if name == "S_SIGLIP2_AREA" else "native"]["identity"], r) for r in rows}) for name, rows in paid.items()}
    return {"scene": scene, "operations": operations, "logical_attempts": {k: len(v) for k, v in paid.items()},
            "crop_inputs": {k: len(v) * 6 for k, v in paid.items()}, "shared_native_mapping_always_required": True}


def required_union(operations, selected):
    # Even A3 requires the original map, owner registry and ranks.
    return set().union(*(set(operations[name]) for name in {"N0", *selected}))
