"""Resolve real v2 lineage once; never mutate the consumed parent experiment."""

import hashlib
from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.area_fallback_experiment import document, seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read


REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "OVIMAP_RUNTIME_PARITY_V1"


def file_identity(path):
    return ConsumptionIndex().identity(path)


def validate_producer_alias(source, copied, expected):
    first, second = file_identity(source), file_identity(copied)
    if any(item["sha256"] != expected["sha256"] or item["bytes"] != expected["bytes"] for item in (first,second)):
        raise ValueError("copied source differs from the actual measured producer")
    return second


def validate_bound_reference(binding):
    _verified_identity(binding)
    index = ConsumptionIndex()
    for row in binding["producer_aliases"]:
        index.identity(row["original"]["path"],row["original"])
        index.identity(REPO/row["repository_path"],row["original"])
    for item in binding["lineage_inputs"]:
        index.identity(item["path"],item)
    return binding


def import_metrics(v2, destination, index):
    store_path = v2/"tables/result_store.json"
    store = document(store_path,index)
    if len(store["scene_metrics"]) != 172 or len(store["pooled_metrics"]) != 14:
        raise ValueError("actual measured v2 does not contain the complete 172/14 scientific grid")
    links = []
    for row in [*store["scene_metrics"],*store["pooled_metrics"]]:
        record = document(row["receipt_path"],index)
        if (record["identity"] != row["receipt_identity"] or record["status"] != "COMPLETE"
                or record["metrics"] != row["metrics"] or record["metric_unit"] != "FRACTION"):
            raise ValueError("v2 scientific cell does not match its actual scoring receipt")
        links.append({"cohort":row["cohort"],"scene":row.get("scene"),"method":row["method_id"],
            "receipt":index.identity(row["receipt_path"]),"science_identity":record["identity"],
            "evaluation_identity":record.get("evaluation_identity"),"scoring_protocol_id":row.get("scoring_protocol_id")})
    imported = seal({"status":"IMPORTED_VERIFIED","reuse_status":"AWAITING_OUTPUT_PARITY",
        "original_scientific_identity":store["identity"],"original_store":index.identity(store_path),
        "scene_records":172,"pooled_records":14,"metric_unit":"FRACTION","links":links,
        "scene_metrics":store["scene_metrics"],"pooled_metrics":store["pooled_metrics"]})
    atomic_write_json(destination,imported)
    return imported


def bind_reference(spec_path, source_worktree, output_root, *, v2_results_root=None, gpu=None, path_map=None):
    spec_path, source, root = Path(spec_path).resolve(),Path(source_worktree).resolve(),Path(output_root).resolve()
    spec = read(spec_path)
    path = root/"reference_binding.json"
    if path.exists():
        existing = validate_bound_reference(read(path))
        if (existing["spec"]["sha256"] != file_identity(spec_path)["sha256"]
                or existing["source_worktree"] != str(source)
                or (v2_results_root and existing["v2_results_root"] != str(Path(v2_results_root).resolve()))
                or (gpu is not None and existing["gpu"] != str(gpu))):
            raise ValueError("existing task binding differs; choose a new attempt")
        return existing
    resolver = PathResolver(path_map)
    if v2_results_root is None:
        clue = read(source/"artifacts/static_ovmap/area_fallback_v2/result_store.json")
        v2_results_root = Path(clue["scene_metrics"][0]["receipt_path"]).parents[3]
    v2 = Path(resolver.resolve(v2_results_root)).resolve()
    index = ConsumptionIndex(root/"binding_input_verifications.json")
    experiment = document(v2/"experiment.json",index)
    parent = Path(resolver.resolve(experiment["parent"]))
    parent_binding = document(parent/"resolved_inputs.json",index)
    if parent_binding["identity"] != experiment["parent_binding_identity"]:
        raise ValueError("v2 experiment differs from its actual parent scientific binding")
    parent_spec = read(parent_binding["spec"])
    if spec["cohorts"] != parent_spec["cohorts"] or experiment["cohorts"] != spec["cohorts"]:
        raise ValueError("runtime task must retain exactly the measured 26 sequences")
    timing = document(v2/"timing/pool.json",index)
    plan = document(v2/"timing/plan.json",index)
    if (timing["status"] != "COMPLETE" or timing["leaf_count"] != 16
            or plan["scientific_experiment_identity"] != experiment["identity"]
            or plan["pooling_producer"] != experiment["producer"]):
        raise ValueError("latest complete v2 cold series cannot be bound to its actual pooling producer")
    aliases = []
    for expected in (experiment["producer"],experiment["driver"],timing["producer"]):
        original = Path(resolver.resolve(expected["path"]))
        relative = original.relative_to(source)
        copied = validate_producer_alias(original,REPO/relative,expected)
        aliases.append({"original":index.identity(original,expected),"repository_path":str(relative),"copied":copied})
    frozen = document(source/"artifacts/static_ovmap/cvpr_compact_tables_v1/freeze/experiment.json",index)
    for item in frozen["implementation_sources"]:
        original = index.identity(source/item["path"],item)
        copied = validate_producer_alias(original["path"],REPO/item["path"],original)
        aliases.append({"original":original,"repository_path":item["path"],"copied":copied})
    contexts = {}
    for scene in [*spec["cohorts"]["replica8"],*spec["cohorts"]["scannet_cf18"]]:
        data = document(parent/"contexts"/(scene+".json"),index)
        regions = document(v2/"regions"/scene/"receipt.json",index)
        predictions = document(v2/"predictions"/scene/"receipt.json",index)
        views = document(parent/"projected_views"/scene/"receipt.json",index)
        if (regions["experiment_identity"] != experiment["identity"] or predictions["regions_identity"] != regions["identity"]
                or regions["status"] != "COMPLETE" or predictions["status"] != "PREDICTIONS_LOCKED"):
            raise ValueError("v2 same-scene recovery-to-prediction lineage is incomplete")
        sources = {arm:document(item["path"],index,item) for arm,item in regions["sources"].items()}
        if any(row["model_identity"] != data["FC_physical_model_identity"] for row in sources.values()):
            raise ValueError("measured v2 model differs from bound same-scene FC evidence")
        contexts[scene] = {"common_context":index.identity(parent/"contexts"/(scene+".json")),
            "model_identity":data["FC_physical_model_identity"],"text":data["FC_text"],
            "valid_ids":data["models"]["native"]["valid_ids"],"v2_region_identity":regions["identity"],
            "v2_prediction_identity":predictions["identity"],"original_view_identity":views["identity"],
            "parent_frame_diagnostics":str(parent/"projected_views"/scene/"frame_diagnostics.json")}
    imported = import_metrics(v2,root/"imported_metrics.json",index)
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    patch = subprocess.check_output(["git","diff","HEAD","--"],cwd=source)
    binding = seal({"protocol":PROTOCOL,"status":"BOUND_ACTUAL_MEASURED_V2","spec":index.identity(spec_path),
        "source_worktree":str(source),"source_head":head,"source_patch_sha256":hashlib.sha256(patch).hexdigest(),
        "source_snapshot":read(REPO/"artifacts/static_ovmap/runtime_parity_v1/source_snapshot.json"),
        "reference_commit":subprocess.check_output(["git","log","-1","--format=%H","--","src/static_ovmap/cvpr_compact/area_fallback.py"],cwd=REPO,text=True).strip(),
        "parent_root":str(parent),"parent_binding_file":index.identity(parent/"resolved_inputs.json"),
        "parent_binding_identity":parent_binding["identity"],"v2_results_root":str(v2),
        "v2_experiment_identity":experiment["identity"],"v2_timing_identity":timing["identity"],
        "imported_scientific_identity":imported["original_scientific_identity"],
        "producer_aliases":aliases,"contexts":contexts,"cohorts":spec["cohorts"],
        "fc":parent_binding["fc"],"final_temperatures":parent_binding["final_temperatures"],
        "gpu":str(gpu if gpu is not None else parent_binding["gpu"]),"path_map":path_map or parent_binding["path_map"],
        "fallback_operator":{"callable":"static_ovmap.cvpr_compact.area_fallback.region_vector",
            "producer":experiment["producer"],"trigger":"bilinear(signed,dense_hw,align_corners=False)>0 has zero support",
            "weights":"area((signed+1)/2,dense_hw)","pool":"einsum(bchw,bqhw->bqc,dense,weights/(sum(weights)+1e-8))",
            "head":"original pinned visual_prediction_forward_convnext", "normal_path":"original MaskPooling unchanged"},
        "arms":{"U2":"genuine archived Native-selected single observation; original signed pooling; no v2 fallback",
            "G1":"full-sequence search; first fixed Top-3 view; actual v2 fallback",
            "G3":"same fixed Top-3; original successful-view static-area aggregation; actual v2 fallback"},
        "unknowns":{"OS_page_cache_state":"UNCONTROLLED","unrecorded_historical_benchmark_exposure":"UNKNOWN"},
        "lineage_inputs":index.entries()})
    atomic_write_json(path,binding)
    index.write_memo(root/"binding_input_verifications.json")
    atomic_write_json(REPO/"artifacts/static_ovmap/runtime_parity_v1/reference_binding.json",binding)
    print("BOUND",binding["identity"],"actual v2; 172 scene rows / 14 pools",flush=True)
    return binding
