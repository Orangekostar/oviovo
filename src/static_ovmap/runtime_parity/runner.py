"""New cold return boundary; immutable old science is consumed without old guards."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.costs import load_base_cost_inventory, recovery_usage
from static_ovmap.cvpr_compact.outputs import build_method_outputs
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.recovery_run import ARM_METHOD, ARM_SOURCE
from static_ovmap.cvpr_compact.region_worker import archived_loader, archived_u2_plan, classify_regions, projected_plan
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_registry import build_registry

from .instrumentation import Stages
from .views import CallLoader, build_views


@dataclass
class CommonInputs:
    scene: str
    data: dict
    baseline: object
    xyz: np.ndarray
    faces: np.ndarray
    raw: np.ndarray
    sources: dict
    valid_ids: list
    nearest: np.ndarray
    matched: np.ndarray
    cost_inventory: dict
    identity: str


@dataclass
class ColdContext:
    output_root: Path
    implementation: str


def execution_config(reference):
    original = read(reference["parent_binding_file"]["path"])
    _verified_identity(original)
    if original["identity"] != reference["parent_binding_identity"]:
        raise ValueError("consumed immutable parent binding changed")
    return {key:original[key] for key in ("fc","assets_root","path_map","spec","final_temperatures",
        "parent_recovery_root","gpu_lock","gpu")} | {"gpu":reference["gpu"],"identity":reference["identity"]}


def load_common(reference, scene, root):
    row = reference["contexts"][scene]
    index = ConsumptionIndex(Path(root)/"common_input_verifications.json")
    index.identity(row["common_context"]["path"],row["common_context"])
    data = read(row["common_context"]["path"])
    _verified_identity(data)
    config = execution_config(reference)
    path = Path(data["predictions"]["NATIVE_READOUT"])
    index.identity(path)
    manifest = read(path)
    index.identity(path.parent/manifest["arrays"]["path"],manifest["arrays"])
    baseline = load_prediction(path)
    capture_path = Path(data["capture_manifest"])
    index.identity(capture_path)
    capture = read(capture_path)
    _verified_identity(capture)
    surface = capture_path.parent/capture["surface"]["path"]
    index.identity(surface,capture["surface"])
    with np.load(surface,allow_pickle=False) as arrays:
        xyz,faces,raw = arrays["surface_xyz"],arrays["surface_faces"],arrays["original_owner"]
    if (baseline.scene_id != scene or capture["scene_id"] != scene
            or _array_digest(xyz) != baseline.geometry.xyz_sha256
            or _array_digest(faces) != baseline.geometry.faces_sha256
            or capture["tsdf"]["sha256"] != baseline.geometry.tsdf_sha256):
        raise ValueError("common recovery geometry differs from the bound Native anchor")
    index.identity(data["config"])
    resolved = PathResolver(config["path_map"]).rewrite(read(data["config"]))
    projection_path = Path(resolved["scenes"][scene]["projection"])
    index.identity(projection_path)
    projection = read(projection_path)
    index.identity(projection_path.with_name("projection.npz"),{"sha256":projection["sha256"]})
    if projection["identity"] != baseline.geometry.projection_identity:
        raise ValueError("common export projection changed")
    with np.load(projection_path.with_name("projection.npz"),allow_pickle=False) as arrays:
        nearest,matched = arrays["nearest"],arrays["matched"]
    ids = list(map(int,data["models"]["native"]["valid_ids"]))
    sources = {}
    for name,item in data["sources"].items():
        index.identity(item["path"])
        value = read(item["path"])
        _verified_identity(value)
        if value["identity"] != item["identity"] or value["valid_ids"] != ids:
            raise ValueError("common N/Q/F evidence or ordered vocabulary changed")
        sources[name] = value
    parent = read(reference["parent_binding_file"]["path"])
    costs = load_base_cost_inventory(parent,data,index=index)
    identity = canonical_digest({"reference":reference["identity"],"scene":scene,
        "native":baseline.record_key,"capture":capture["identity"],"projection":projection["identity"],
        "sources":{name:value["identity"] for name,value in sources.items()},"costs":costs["identity"]})
    index.write_memo(Path(root)/"common_input_verifications.json")
    return CommonInputs(scene,data,baseline,xyz,faces,raw,sources,ids,nearest,matched,costs,identity)


def write_exports(config, inputs, registry, sources, stats, methods, root, index):
    named_sources = {ARM_SOURCE[arm]:source for arm,source in sources.items()}
    costs = {ARM_SOURCE[arm]:recovery_usage(source,stats,ARM_SOURCE[arm]) for arm,source in sources.items()}
    payloads = build_method_outputs(inputs.baseline,inputs.raw,registry,inputs.sources,config["final_temperatures"],
        inputs.valid_ids,inputs.nearest,inputs.matched,named_sources,methods,
        cost_inventory=inputs.cost_inventory,recovery_costs=costs)
    exports = {}
    for method,payload in payloads.items():
        path = save_prediction(payload,Path(root)/method)
        manifest = read(path)
        exports[method] = {"manifest":str(path),"record_key":payload.record_key,"prediction_key":payload.prediction_key,
            "files":[index.identity(path),index.identity(path.parent/manifest["arrays"]["path"],manifest["arrays"])]}
    return exports


def recover(inputs, *, arm, implementation, model_session, run_context, config, stages=None):
    if not isinstance(run_context,ColdContext) or run_context.implementation != implementation:
        raise ValueError("timed recovery requires the isolated cold context")
    root = Path(run_context.output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("cold output directory must be new and empty")
    if arm not in ARM_METHOD or model_session.cache is not None or model_session.model is None:
        raise ValueError("cold recovery requires one original arm and a resident cache-free model")
    stages = stages or Stages()
    index = ConsumptionIndex()
    if arm == "U2":
        with stages.span("registry_geometry"):
            registry = build_registry(inputs.xyz,inputs.raw,inputs.baseline.owner_ids)
            stages.counters["raw_candidates"] += len(registry["candidates"])
        with stages.span("view_search"):
            plan,requests,archival = archived_u2_plan(config,inputs.scene,registry,index=index)
            original_loader = archived_loader(archival,index=index)
        def loader(rid):
            with stages.span("rgb_preprocess"):
                stages.counters["RGB_decodes"] += 1
                return original_loader(rid)
        manifest_identity = archival["identity"]
    else:
        registry,manifest = build_views(inputs,root/"views",implementation,index,stages)
        plan = projected_plan(registry,manifest,(arm,))
        selected = dict.fromkeys(rid for rows in plan[arm].values() for rid in rows)
        requests = {rid:manifest["requests"][rid] for rid in selected}
        with stages.span("rgb_preprocess"):
            loader = CallLoader(root/"views/manifest.json",index,stages,reuse=implementation!="R0_REFERENCE")
        manifest_identity = manifest["identity"]
    model_session.index = index
    features,stats = model_session.encode_exact(requests,{rid:loader for rid in requests},stages,implementation=implementation,arm=arm)
    with stages.span("recognition"):
        classified = classify_regions(registry,plan,requests,features,model_session.text,model_session.ids)
        sources = {arm:seal({"source":arm,"scene":inputs.scene,"objects":classified[arm],
            "valid_ids":inputs.valid_ids,"model_identity":model_session.model_key,"text_identity":model_session.text_identity,
            "registry_identity":registry["identity"],"score_kind":"ORIGINAL_FC_COSINE",
            "request_manifest_identity":manifest_identity,"GT_input":False})}
    with stages.span("export_bookkeeping"):
        atomic_write_json(root/"registry.json",registry)
        source_path = root/"regions"/(arm+"_FC.json")
        atomic_write_json(source_path,sources[arm])
        ordered = sorted(features)
        feature_path = root/"feature_vectors.npz"
        _write_npz(feature_path,{"request_ids":np.asarray(ordered,dtype="U64"),
            "features":np.stack([features[rid] for rid in ordered]) if ordered else np.empty((0,model_session.text.shape[1]),np.float32)})
        methods = [row for row in read(config["spec"])["methods"] if row["id"]==ARM_METHOD[arm]]
        exports = write_exports(config,inputs,registry,sources,stats,methods,root/"predictions",index)
        result = seal({"status":"COMPLETE","scene":inputs.scene,"arm":arm,"implementation":implementation,
            "common_input_identity":inputs.identity,"registry_identity":registry["identity"],"plan":plan,
            "sources":{arm:index.identity(source_path)},"features":index.identity(feature_path),"exports":exports,
            "requests":requests,"stats":stats,"work_counters":dict(stages.counters),"GT_input":False,
            "cold":True,"persistent_feature_cache_enabled":False,"persistent_view_cache_enabled":False,
            "persistent_result_cache_enabled":False,"input_checks":index.entries()})
        atomic_write_json(root/"receipt.json",result)
        stages.counters["output_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return result
