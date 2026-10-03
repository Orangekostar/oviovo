"""Lock the fixed method outputs before opening any semantic/instance labels."""

import argparse
from pathlib import Path
import time

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read

from .costs import load_base_cost_inventory, recovery_usage
from .outputs import build_method_outputs
from .projected_views import _verified_identity
from .protocol import load_spec
from .runtime import require_frozen_execution


def lock_outputs(binding, scene, projected_receipt, fc_receipt, native_receipt, output_root, *, context=None):
    freeze = require_frozen_execution(binding, scene)
    spec, root = load_spec(binding["spec"]), Path(output_root)
    data = context or binding["scenes"][scene]
    index = ConsumptionIndex(root / "input_verifications.json")
    resolver = PathResolver(binding["path_map"])
    for receipt in (projected_receipt, fc_receipt, native_receipt):
        _verified_identity(receipt)
        if receipt["status"] != "COMPLETE" or receipt["scene"] != scene:
            raise ValueError("fixed outputs require complete same-scene visual prerequisites")
        for row in receipt["outputs"]:
            index.identity(row["path"], row)
    registry = read(projected_receipt["registry"])
    _verified_identity(registry)
    index.identity(data["config"])
    config = resolver.rewrite(read(data["config"]))
    projection_path = Path(config["scenes"][scene]["projection"])
    index.identity(projection_path)
    projection = read(projection_path)
    projection_arrays = projection_path.with_name("projection.npz")
    index.identity(projection_arrays, {"sha256": projection["sha256"]})
    sources = {}
    for name, item in data["sources"].items():
        index.identity(item["path"])
        source = read(item["path"])
        _verified_identity(source)
        if source["identity"] != item["identity"]:
            raise ValueError("fixed existing N/Q/F source changed")
        sources[name] = source
    recovery = {}
    for arm, item in fc_receipt["sources"].items():
        source = read(item["path"])
        _verified_identity(source)
        recovery[{"G1": "G1_FC", "G3": "G3_FC", "U2": "ARCHIVED_U2_FC"}[arm]] = source
    recovery["G1_NATIVE"] = read(native_receipt["source"]["path"])
    _verified_identity(recovery["G1_NATIVE"])
    cohort = "replica8" if scene in spec["cohorts"]["replica8"] or scene == spec["smoke_scene"] else "scannet_cf18"
    methods = [method for method in spec["methods"] if cohort in method["cohorts"]]
    cost_inventory = load_base_cost_inventory(binding, data, index=index)
    recovery_costs = {name: recovery_usage(source, native_receipt if name == "G1_NATIVE" else fc_receipt, name)
                      for name, source in recovery.items()}
    identity = canonical_digest({"binding": binding["identity"], "scene": scene,
        "baseline": index.identity(data["predictions"]["NATIVE_READOUT"]), "registry": registry["identity"],
        "sources": {name: source["identity"] for name, source in sources.items()},
        "recovery": {name: source["identity"] for name, source in recovery.items()},
        "temperatures": binding["final_temperatures"], "projection": index.identity(projection_arrays),
        "methods": methods, "freeze": freeze,
        "base_cost_inventory": cost_inventory["identity"],
        "recovery_cost_identities": {name: value["identity"] for name, value in recovery_costs.items()},
        "producers": {name: index.identity(Path(__file__).with_name(name + ".py"))["sha256"]
                      for name in ("outputs", "prediction_worker", "costs")}})
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        previous = read(receipt_path)
        _verified_identity(previous)
        if previous["status"] == "PREDICTIONS_LOCKED":
            if previous["input_identity"] != identity:
                raise ValueError("completed fixed outputs changed; invalidate affected descendants explicitly")
            for item in previous["inputs"] + previous["outputs"]:
                index.identity(item["path"], item)
            for path in previous["predictions"].values():
                load_prediction(path)
            return previous
        receipt_path.rename(root / ("receipt.failed_" + str(time.time_ns()) + ".json"))
    started = time.monotonic()
    receipt = {"status": "RUNNING", "scene": scene, "input_identity": identity,
               "freeze": freeze, "cohort": cohort, "GT_input": False}
    atomic_write_json(receipt_path, receipt)
    try:
        baseline = load_prediction(data["predictions"]["NATIVE_READOUT"])
        if (baseline.scene_id != scene or baseline.geometry.projection_identity != projection["identity"]
                or baseline.geometry.source_row_count != projection["source_rows"]):
            raise ValueError("official rank projection differs from the fixed Native geometry")
        index.identity(data["capture_manifest"])
        capture = read(data["capture_manifest"])
        _verified_identity(capture)
        surface_path = Path(data["capture_manifest"]).parent / capture["surface"]["path"]
        index.identity(surface_path, capture["surface"])
        with np.load(surface_path, allow_pickle=False) as arrays:
            raw = arrays["original_owner"]
        with np.load(projection_arrays, allow_pickle=False) as arrays:
            nearest, matched = arrays["nearest"], arrays["matched"]
        ids = data["models"]["native"]["valid_ids"]
        outputs = build_method_outputs(baseline, raw, registry, sources, binding["final_temperatures"],
            ids, nearest, matched, recovery, methods,
            cost_inventory=cost_inventory, recovery_costs=recovery_costs)
        a0 = outputs["CT_A0_NATIVE"]
        if (not np.array_equal(a0.owner_ids, baseline.owner_ids)
                or not np.array_equal(a0.semantic_labels, baseline.semantic_labels)
                or a0.instance_ranks != baseline.instance_ranks):
            raise RuntimeError("A0 does not reproduce exact Native arrays and official ranks")
        if scene in spec["cohorts"]["replica8"]:
            archived_d2 = load_prediction(data["predictions"]["D2"])
            a1 = outputs["CT_A1_E"]
            if (not np.array_equal(a1.owner_ids, archived_d2.owner_ids)
                    or not np.array_equal(a1.semantic_labels, archived_d2.semantic_labels)):
                raise RuntimeError("A1 decisions do not reproduce the final frozen Replica D2")
        paths, output_identities, keys = {}, [], {}
        from static_ovmap.backbone_wave1.readouts import fuse_readout
        from static_ovmap.composition_study.object_evidence import owner_labels
        d2, probability_audit = fuse_readout(sources, binding["final_temperatures"], "D2", ids, owner_labels(baseline))
        if {str(owner): label for owner, label in d2.items()} != outputs["CT_A1_E"].metadata["owner_semantic_decisions"]:
            raise RuntimeError("original D2 probability audit differs from the locked existing-owner decisions")
        audit_path = root / "existing_D2_probability_audit.json"
        atomic_write_json(audit_path, {"scene": scene, "original_operator": "backbone_wave1.readouts.fuse_readout",
                         "decision_identity": outputs["CT_A1_E"].metadata["existing_D2_decision_identity"],
                         "probabilities_are_not_official_instance_confidence": True, "owners": probability_audit})
        output_identities.append(index.identity(audit_path))
        for method, output in outputs.items():
            path = save_prediction(output, root / method)
            paths[method], keys[method] = str(path), {"record_key": output.record_key, "prediction_key": output.prediction_key}
            output_identities.append(index.identity(path))
            manifest = read(path)
            output_identities.append(index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"]))
        receipt.update(status="PREDICTIONS_LOCKED", predictions=paths, prediction_identities=keys,
            outputs=output_identities, output_count=len(paths), baseline_array_and_rank_parity=True,
            final_temperature_vector=binding["final_temperatures"], geometry_identity=baseline.geometry.to_dict(),
            common_G1_FC_success=[int(owner) for owner, row in recovery["G1_FC"]["objects"].items() if row["available"]])
        receipt.update(base_cost_inventory=cost_inventory,
                       recovery_cost_identities={name: value["identity"] for name, value in recovery_costs.items()})
    except BaseException as exc:
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
    parser.add_argument("--regions", required=True)
    parser.add_argument("--native-regions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--context")
    args = parser.parse_args()
    result = lock_outputs(read(args.binding), args.scene, read(args.views), read(args.regions),
        read(args.native_regions), args.output_root, context=read(args.context) if args.context else None)
    print(args.scene, result["status"], "outputs", result["output_count"], flush=True)
