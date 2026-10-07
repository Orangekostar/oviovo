"""Narrow binding of consumed immutable inputs, without replaying parent guards."""

from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read


REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "OVIMAP_EVIDENCE_EXPLORATION_V1"
SOURCE_FILES = (
    "runtime_parity/binding.py", "runtime_parity/runner.py", "runtime_parity/views.py",
    "runtime_parity/kernels.py", "runtime_parity/session.py", "cvpr_compact/projected_views.py",
    "cvpr_compact/area_fallback.py", "a7_evidence_upgrade/region_adapter.py",
    "cvpr_compact/region_worker.py", "cvpr_compact/outputs.py", "cvpr_compact/evaluation.py",
    "recovery_wave2/evaluation.py", "released_trace.py",
)
BASELINES = {"EV00_D2": "CT_A1_E", "EV01_G1_V2": "CT_A3_ER"}


def bind(spec_path, parent_root, output_root, *, gpu=None, path_map=None, parent_reference=None):
    spec_path, parent_root, root = Path(spec_path).resolve(), Path(parent_root).resolve(), Path(output_root).resolve()
    if root == parent_root or root.is_relative_to(parent_root):
        raise ValueError("new output cannot modify the read-only parent")
    root.mkdir(parents=True, exist_ok=True)
    resolver, index = PathResolver(path_map or {}), ConsumptionIndex(root / "input_verifications.json")
    def document(path, expected=None, sealed=True):
        path = Path(resolver.resolve(path))
        index.identity(path, expected)
        value = read(path)
        if sealed:
            _verified_identity(value)
        return value
    spec = document(spec_path, sealed=False)
    authoritative = read(REPO / "docs/paper/static_ovmap/evidence_exploration_v1/spec/PROTOCOL_SPEC.json")
    if spec != authoritative:
        raise ValueError("experimental constants/methods/cohorts differ from the supplied frozen protocol")
    refpath = Path(parent_reference or parent_root / spec["parent_reference_file"])
    reference = document(refpath)
    if reference["cohorts"] != spec["cohorts"]:
        raise ValueError("study cohorts differ from actual parent cohorts")
    # Resolve only consumed paths. Historical worktree source aliases are represented
    # by their matching published files in this exact-base checkout.
    reference = resolver.rewrite(reference)
    for row in reference["producer_aliases"]:
        index.identity(REPO / row["repository_path"], row["original"])
    v2 = Path(reference["v2_results_root"])
    experiment = document(v2 / "experiment.json")
    if experiment["identity"] != reference["v2_experiment_identity"]:
        raise ValueError("actual measured v2 experiment changed")
    imported = document(parent_root / "imported_metrics.json")
    if (len(imported["scene_metrics"]) != 172 or len(imported["pooled_metrics"]) != 14
            or imported["original_scientific_identity"] != reference["imported_scientific_identity"]):
        raise ValueError("parent does not contain the actual complete 172/14 science store")
    scenes = {}
    for cohort, names in spec["cohorts"].items():
        for scene in names:
            row = reference["contexts"][scene]
            context = document(row["common_context"]["path"], row["common_context"])
            regions = document(v2 / "regions" / scene / "receipt.json")
            predictions = document(v2 / "predictions" / scene / "receipt.json")
            manifest = document(Path(reference["parent_root"]) / "projected_views" / scene / "manifest.json")
            registry = document(Path(reference["parent_root"]) / "projected_views" / scene / "registry.json")
            if (regions["identity"] != row["v2_region_identity"]
                    or predictions["identity"] != row["v2_prediction_identity"]
                    or regions["experiment_identity"] != experiment["identity"]
                    or predictions["regions_identity"] != regions["identity"]
                    or regions["status"] != "COMPLETE" or predictions["status"] != "PREDICTIONS_LOCKED"):
                raise ValueError("actual same-scene v2 recovery/prediction lineage differs")
            source = document(regions["sources"]["G1"]["path"], regions["sources"]["G1"])
            if (source["request_manifest_identity"] != manifest["identity"]
                    or source["registry_identity"] != registry["identity"]
                    or source["valid_ids"] != row["valid_ids"]
                    or source["model_identity"] != context["FC_physical_model_identity"]):
                raise ValueError("G1 source/view/registry/model/vocabulary identity differs")
            if len(registry["candidates"]) > 128:
                raise ValueError("parent exceeds the fixed candidate cap")
            baseline_rows = {}
            for method, inherited in BASELINES.items():
                metric = next(x for x in imported["scene_metrics"] if x["scene"] == scene and x["method_id"] == inherited)
                scored_row = document(metric["receipt_path"])
                if scored_row["identity"] != metric["receipt_identity"] or scored_row["metrics"] != metric["metrics"]:
                    raise ValueError("baseline scene metrics do not match their actual receipts")
                manifest_path = predictions["predictions"][inherited]
                prediction = document(manifest_path, sealed=False)
                index.identity(Path(manifest_path).parent / prediction["arrays"]["path"], prediction["arrays"])
                baseline_rows[method] = {"metric": metric, "row": scored_row, "prediction_manifest": manifest_path}
            scenes[scene] = {"cohort": cohort, "context": row["common_context"],
                "view_manifest": str(Path(reference["parent_root"]) / "projected_views" / scene / "manifest.json"),
                "view_identity": manifest["identity"], "registry": str(Path(reference["parent_root"]) / "projected_views" / scene / "registry.json"),
                "registry_identity": registry["identity"], "G1_source": regions["sources"]["G1"],
                "region_receipt": str(v2 / "regions" / scene / "receipt.json"), "region_identity": regions["identity"],
                "baseline_rows": baseline_rows}
    baseline_pools = {}
    for cohort in spec["cohorts"]:
        baseline_pools[cohort] = {}
        for method, inherited in BASELINES.items():
            metric = next(x for x in imported["pooled_metrics"] if x["cohort"] == cohort and x["method_id"] == inherited)
            receipt = document(metric["receipt_path"])
            if receipt["identity"] != metric["receipt_identity"] or receipt["metrics"] != metric["metrics"]:
                raise ValueError("baseline pool differs from its actual full-cohort receipt")
            baseline_pools[cohort][method] = metric
    sources = {rel: index.identity(REPO / "src/static_ovmap" / rel) for rel in SOURCE_FILES}
    result = seal({"protocol": PROTOCOL, "status": "BOUND_ACTUAL_V2",
        "spec": index.identity(spec_path), "specification": spec, "parent_reference": index.identity(refpath),
        "parent_reference_identity": reference["identity"], "reference": reference,
        "cohorts": spec["cohorts"], "scenes": scenes, "baseline_pools": baseline_pools,
        "base_commit": spec["base_commit"], "sources_read": sources,
        "gpu": str(gpu if gpu is not None else reference["gpu"]), "path_map": path_map or {},
        "FC_python": reference["fc"]["python"], "imported_baseline_scene_rows": 52,
        "imported_baseline_full_pools": 4, "GT_used_by_predictor": False,
        "lineage_inputs": index.entries()})
    target = root / "source_binding.json"
    if target.exists():
        previous = document(target)
        for key in ("spec", "parent_reference", "sources_read", "scenes", "gpu", "path_map"):
            if previous[key] != result[key]:
                raise ValueError("completed narrow task binding changed: " + key)
        result = previous
    else:
        atomic_write_json(target, result)
    index.write_memo(root / "input_verifications.json")
    print("BOUND", result["identity"], "26 scenes; 52 baseline rows; 4 baseline pools", flush=True)
    return result


def load_binding(root):
    result = read(Path(root) / "source_binding.json")
    _verified_identity(result)
    index = ConsumptionIndex(Path(root) / "input_verifications.json")
    index.identity(result["spec"]["path"], result["spec"])
    index.identity(result["parent_reference"]["path"], result["parent_reference"])
    for item in result["sources_read"].values():
        index.identity(item["path"], item)
    return result
