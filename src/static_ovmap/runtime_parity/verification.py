"""CPU view verification and cached, non-timed regeneration of all readouts."""

from concurrent.futures import ProcessPoolExecutor
import copy
import os
from pathlib import Path
import time

import numpy as np

from static_ovmap.cvpr_compact.area_fallback_experiment import document, seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.region_worker import classify_regions, projected_plan
from static_ovmap.cvpr_compact.runtime import process_state
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .instrumentation import Stages
from .parity import VerificationContext, compare_prediction, compare_source_objects, compare_views
from .runner import execution_config, load_common, write_exports
from .views import build_views
from .workflow import code_inventory


def cached_exports(context, inputs, registry, manifest, root):
    if not isinstance(context,VerificationContext):
        raise TypeError("cached recovery handles require a VerificationContext")
    reference = context.reference
    root = Path(root)
    index = ConsumptionIndex()
    parity = compare_views(context,inputs.scene,root/"views")
    mapping = parity["mapping"]
    original = document(Path(reference["v2_results_root"])/"regions"/inputs.scene/"receipt.json",index)
    index.identity(original["features"]["path"],original["features"])
    with np.load(original["features"]["path"],allow_pickle=False) as arrays:
        features = {mapping[rid]:vector for rid,vector in zip(arrays["request_ids"].tolist(),arrays["features"],strict=True)}
    text_item = reference["contexts"][inputs.scene]["text"]
    index.identity(text_item["path"],text_item)
    with np.load(text_item["path"],allow_pickle=False) as arrays:
        text,ids = arrays["text_embeddings"],arrays["valid_ids"].tolist()
    if ids != inputs.valid_ids:
        raise ValueError("cached export changed the complete ordered text vocabulary")
    plan = projected_plan(registry,manifest,("G1","G3"))
    objects = classify_regions(registry,plan,manifest["requests"],features,text,ids)
    sources,errors = {},{}
    for arm,item in original["sources"].items():
        parent_source = document(item["path"],index,item)
        sources[arm] = seal({**parent_source,"objects":objects[arm],"request_manifest_identity":manifest["identity"]})
        errors[arm] = compare_source_objects(parent_source,sources[arm],mapping)
    stats = copy.deepcopy(original)
    stats["requests"] = {mapping[rid]:{**row,"request_id":mapping[rid]} for rid,row in original["requests"].items()}
    config = execution_config(reference)
    methods = [row for row in read(config["spec"])["methods"] if row["id"] in ("CT_A2_R","CT_A3_ER","CT_A5_FC_ONLY","CT_G3")
        and any(inputs.scene in reference["cohorts"][cohort] for cohort in row["cohorts"])]
    exports = write_exports(config,inputs,registry,sources,stats,methods,root/"predictions",index)
    parent_lock = document(Path(reference["v2_results_root"])/"predictions"/inputs.scene/"receipt.json",index)
    comparisons = {name:compare_prediction(parent_lock["predictions"][name],row["manifest"]) for name,row in exports.items()}
    if not {"CT_A2_R","CT_A3_ER","CT_A5_FC_ONLY"} <= set(comparisons):
        raise ValueError("cached verification omitted a shared G1 readout")
    return {"exports":comparisons,"max_score_absolute_error":errors,
        "shared_G1_recovery_exact":True,"new_GPU_image_encodings":0,"new_GPU_poolings":0}


def verify_scene(arguments):
    reference,scene,variant,task_root = arguments
    task_root = Path(task_root)
    root = task_root/"verification"/variant/scene
    path = root/"receipt.json"
    identity = canonical_digest({"reference":reference["identity"],"scene":scene,"variant":variant,
        "implementation":code_inventory()})
    if path.is_file():
        old = read(path)
        if old["input_identity"] != identity:
            raise ValueError("CPU projection reservation changed; select a new verification attempt")
        if old["status"]=="PASS":
            _verified_identity(old)
            compare_views(VerificationContext(reference),scene,root/"views")
            return old
        live = process_state(old["pid"])
        if live and live["live"] and live["start_ticks"]==old["start_ticks"]:
            raise RuntimeError("LIVE_CPU_VERIFICATION: observe "+str(old["pid"]))
        raise RuntimeError("CPU verification reservation incomplete; do not repeat a projection blindly")
    process = process_state(os.getpid())
    reservation = {"status":"RUNNING","scene":scene,"implementation":variant,"input_identity":identity,
        "pid":os.getpid(),"start_ticks":process["start_ticks"],"projection_passes_reserved":1}
    atomic_write_json(path,reservation)
    start = time.perf_counter()
    inputs = load_common(reference,scene,root)
    stages = Stages()
    registry,manifest = build_views(inputs,root/"views",variant,ConsumptionIndex(),stages)
    context = VerificationContext(reference)
    projection = compare_views(context,scene,root/"views")
    export = cached_exports(context,inputs,registry,manifest,root)
    record = seal({**reservation,"status":"PASS","verification_kind":"CPU_VIEW_AND_CACHED_EXPORT_PARITY",
        "reference_binding_identity":reference["identity"],"projection_passes":1,
        "projection":projection,**export,"elapsed_seconds":time.perf_counter()-start,
        "work_counters":dict(stages.counters),"GT_used":False})
    atomic_write_json(path,record)
    print("CPU parity",scene,variant,"PASS",flush=True)
    return record


def verify_all(reference, spec, root, selected, *, workers=3):
    scenes = [*spec["cohorts"]["replica8"],*spec["cohorts"]["scannet_cf18"]]
    missing = [scene for scene in scenes if not Path(reference["contexts"][scene]["parent_frame_diagnostics"]).is_file()]
    if missing:
        raise RuntimeError("missing recorded per-frame diagnostics require an explicit bounded reference pass: "+", ".join(missing))
    jobs = [(reference,scene,selected,str(root)) for scene in scenes]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(verify_scene,jobs))
    passes = sum(row["projection_passes"] for row in receipts)
    if len(receipts)!=26 or passes>52 or any(row["status"]!="PASS" for row in receipts):
        raise ValueError("the full 26-scene CPU parity/budget gate failed")
    result = seal({"status":"ALL_26_PASS","selected":selected,"reference_binding_identity":reference["identity"],
        "projection_passes":passes,"new_CF18_GPU_image_encodings":0,"scene_order":scenes,
        "receipts":receipts,"metric_reuse_status":"PREDICTION_IDENTICAL_PARENT_METRIC"})
    atomic_write_json(Path(root)/"verification/summary.json",result)
    return result
