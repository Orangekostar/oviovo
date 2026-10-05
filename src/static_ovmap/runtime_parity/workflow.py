"""Fixed repeat plans, synchronized measurements and independent cold reservations."""

import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.recovery_run import gpu_lease
from static_ovmap.cvpr_compact.runtime import process_state
from static_ovmap.cvpr_compact.timing import _hardware, assert_serial_execution
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .binding import REPO, file_identity, validate_bound_reference
from .instrumentation import Stages
from .kernels import VARIANTS
from .parity import VerificationContext, compare_cold
from .runner import ColdContext, execution_config, load_common, recover
from .session import RuntimeSession


def code_inventory():
    directory = Path(__file__).parent
    names = ("binding.py","instrumentation.py","kernels.py","views.py","session.py","runner.py","parity.py","workflow.py")
    return {name:file_identity(directory/name)["sha256"] for name in names}


def call_plan(spec, phase, selected="R0_REFERENCE"):
    if phase == "pilot":
        labels = [(name,"G1",name) for name in VARIANTS]
        scenes = spec["pilot_scenes"]
    elif phase == "final":
        labels = [("U2_CONTROL","U2","R0_REFERENCE"),("G1_REFERENCE","G1","R0_REFERENCE")]
        if selected == "R0_REFERENCE":
            labels.append(("G3_REFERENCE","G3",selected))
        else:
            labels += [("G1_SELECTED","G1",selected),("G3_SELECTED","G3",selected)]
        scenes = spec["cohorts"]["replica8"]
    else:
        raise ValueError("unknown cold plan phase")
    return [{"phase":phase,"round":repeat,"scene":scene,"label":label,"arm":arm,"implementation":implementation}
        for repeat in (1,2) for scene in (scenes if repeat==1 else scenes[::-1])
        for label,arm,implementation in (labels if repeat==1 else labels[::-1])]


def select_pilot(rows, scenes, variants=VARIANTS):
    means,per_scene,rejected = {},{},{}
    for variant in variants:
        chosen = [row for row in rows if row["implementation"]==variant]
        if any(row["status"]=="REJECTED_PARITY" for row in chosen):
            rejected[variant] = "REJECTED_PARITY"
            continue
        values = {}
        for scene in scenes:
            scene_rows = [row for row in chosen if row["scene"]==scene]
            if (len(scene_rows)!=2 or {row["round"] for row in scene_rows}!={1,2}
                    or any(row["status"]!="COMPLETE" or row["parity"]["status"]!="PASS"
                           or not np.isfinite(row["seconds"]) or row["seconds"]<=0 for row in scene_rows)):
                raise ValueError("pilot selection requires exactly two passing finite repeats per scene/variant")
            values[scene] = sum(row["seconds"] for row in scene_rows)/2
        per_scene[variant] = values
        means[variant] = sum(values.values())/len(scenes)
    if not means:
        raise ValueError("no verified reference pilot measurements")
    fastest = min(means.values())
    selected = min((name for name,value in means.items() if value<=fastest*1.03),key=VARIANTS.index)
    return seal({"status":"PILOT_SELECTED","selected":selected,"means":means,"scene_means":per_scene,
        "rejected":rejected,"rule":"equal scene means of two repeats; lower index within 3% of fastest",
        "accuracy_used_for_selection":False})


def hardware_settings(gpu):
    import open3d
    import torch
    import open_clip
    import timm
    cpu = next((line.split(":",1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    return {"GPU":_hardware(gpu),"CPU":cpu,"python":sys.version,"torch":torch.__version__,
        "open3d":open3d.__version__,"open_clip":open_clip.__version__,"timm":timm.__version__,
        "numpy":np.__version__,"CUDA":torch.version.cuda,"torch_threads":torch.get_num_threads(),
        "projector_threads":4,"encoder_batch_size":1,"torch_matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
        "torch_cudnn_tf32":torch.backends.cudnn.allow_tf32,"cudnn_benchmark":torch.backends.cudnn.benchmark,
        "cudnn_deterministic":torch.backends.cudnn.deterministic,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "thread_environment":{name:os.environ.get(name) for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        "OS_page_cache":"UNCONTROLLED_NOT_CLEARED","precision":"original_FP32","operator_settings_changed":False}


def warmup(session, root, settings):
    import torch
    from types import SimpleNamespace
    from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
    from static_ovmap.cvpr_compact.area_fallback import region_vector
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image,size = image_tensor(np.zeros((480,640,3),np.uint8),session.device)
        dense = session.operators["extract_features_convnext"](SimpleNamespace(clip_model=session.model),image)["clip_vis_dense"]
        signed,_,_ = signed_mask(np.ones((480,640),bool),size,image.shape[-2:],dense.shape[-2:],session.device)
        vector,_ = region_vector(session.model,session.operators,dense,signed)
        del image,dense,signed,vector
    torch.cuda.synchronize()
    record = seal({"status":"COMPLETE","kind":"SHAPE_ONLY_WARMUP","scene_pixels_used":False,
        "shape_hw":[480,640],"physical_image_encodings":1,"physical_region_poolings":1,
        "seconds":time.perf_counter()-start,"model_identity":session.model_key,"hardware":settings,
        "pid":os.getpid(),"outside_incremental_timer":True})
    atomic_write_json(Path(root)/"warmup"/(str(os.getpid())+".json"),record)
    return record


def run_measurements(reference, spec, root, phase, *, selected="R0_REFERENCE"):
    import torch
    reference = validate_bound_reference(reference)
    root = Path(root)
    config = execution_config(reference)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != reference["gpu"]:
        raise ValueError("worker must use exactly the bound physical GPU")
    inventory = code_inventory()
    plan = call_plan(spec,phase,selected)
    settings = hardware_settings(reference["gpu"])
    plan_identity = canonical_digest({"reference":reference["identity"],"phase":phase,"calls":plan,
        "implementation_sources":inventory,"hardware":settings})
    if phase=="final":
        freeze = read(root/"implementation_freeze.json")
        _verified_identity(freeze)
        if (freeze["selected"]!=selected or freeze["reference_binding_identity"]!=reference["identity"]
                or freeze["implementation_sources"]!=inventory or freeze["hardware"]!=settings or freeze["calls"]!=plan):
            raise ValueError("final worker differs from the committed implementation freeze")
        committed = subprocess.check_output(["git","show","HEAD:artifacts/static_ovmap/runtime_parity_v1/implementation_freeze.json"],cwd=REPO)
        if committed != (root/"implementation_freeze.json").read_bytes():
            raise ValueError("final benchmark requires the exact committed freeze")
    atomic_write_json(root/phase/"plan.json",seal({"plan_identity":plan_identity,"calls":plan,
        "implementation_sources":inventory,"hardware":settings,"reference_binding_identity":reference["identity"]}))
    phase_root = root/phase
    old_binding = read(reference["parent_binding_file"]["path"])
    session,inputs,last_scene = None,None,None
    measurements = []
    with exclusive_lock(root/".timing-controller.lock"),gpu_lease(config,phase_root):
        assert_serial_execution(old_binding)
        for planned in plan:
            leaf = phase_root/("round_"+str(planned["round"]))/planned["scene"]/planned["label"]
            path = leaf/"measurement.json"
            key = canonical_digest({"plan":plan_identity,**planned})
            if path.exists():
                row = read(path)
                if row["input_identity"] != key:
                    raise ValueError("reserved runtime leaf changed source/config/hardware; replay forbidden")
                if row["status"] in ("COMPLETE","REJECTED_PARITY"):
                    _verified_identity(row)
                    result = read(leaf/"call/receipt.json")
                    _verified_identity(result)
                    if result["identity"]!=row["recovery_identity"]:
                        raise ValueError("completed cold output differs from its measurement receipt")
                    check = ConsumptionIndex()
                    for item in [result["features"],*result["sources"].values(),
                            *(item for export in result["exports"].values() for item in export["files"])]:
                        check.identity(item["path"],item)
                    measurements.append(row)
                    continue
                if row["status"]=="MEASURED":
                    result = read(leaf/"call/receipt.json")
                    try:
                        row["parity"] = compare_cold(VerificationContext(reference),result,leaf/"call")
                        row["status"] = "COMPLETE"
                    except ValueError as exc:
                        row.update(status="REJECTED_PARITY",parity={"status":"FAIL","error":str(exc)})
                    atomic_write_json(path,seal(row))
                    measurements.append(row)
                    continue
                state = process_state(row["pid"])
                if state and state["live"] and state["start_ticks"]==row["start_ticks"]:
                    raise RuntimeError("LIVE_RUNTIME_CALL: observe existing worker "+str(row["pid"]))
                raise RuntimeError("reserved failed call requires documented external-failure review; no blind replay")
            if last_scene != planned["scene"]:
                inputs = load_common(reference,planned["scene"],root)
                last_scene = planned["scene"]
                current = RuntimeSession(config,inputs.data,ConsumptionIndex(root/"model_input_verifications.json"),cache=None)
                if session is None:
                    current.load_model()
                    for name in ("static_ovmap.cvpr_compact.area_fallback","static_ovmap.cvpr_compact.projected_views",
                                 "static_ovmap.a7_evidence_upgrade.region_adapter","static_ovmap.runtime_parity.session"):
                        module = sys.modules[name]
                        if not Path(module.__file__).resolve().is_relative_to(REPO/"src"):
                            raise ValueError("loaded producer is outside the bound task namespace: "+name)
                    atomic_write_json(phase_root/"model_load.json",seal({"seconds":current.model_load_seconds,
                        "model_identity":current.model_key,"weight_audit":current.weight_audit,"hardware":settings,
                        "actual_loaded_modules":{name:module.__file__ for name,module in sys.modules.items()
                            if module is not None and getattr(module,"__file__",None) and
                            (name.startswith('static_ovmap.runtime_parity') or name=='static_ovmap.cvpr_compact.area_fallback')}}))
                    session = current
                    warmup(session,phase_root,settings)
                else:
                    if (current.model_key,current.text_identity["sha256"],list(current.ids)) != (session.model_key,session.text_identity["sha256"],list(session.ids)):
                        raise ValueError("resident FC model/text changed between Replica scenes")
                    current.model,current.weight_audit = session.model,session.weight_audit
                    session = current
            assert_serial_execution(old_binding)
            process = process_state(os.getpid())
            row = {**planned,"status":"CALL_STARTED","input_identity":key,"plan_identity":plan_identity,
                "reference_binding_identity":reference["identity"],"implementation_sources":inventory,"hardware":settings,
                "pid":os.getpid(),"start_ticks":process["start_ticks"],"started_unix":time.time(),
                "common_input_identity":inputs.identity,"physical_calls_reserved":1}
            atomic_write_json(path,row)
            stages = Stages()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                result = recover(inputs,arm=planned["arm"],implementation=planned["implementation"],model_session=session,
                    run_context=ColdContext(leaf/"call",planned["implementation"]),config=config,stages=stages)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                row.update(status="MEASURED",seconds=seconds,stage_seconds=stages.finish(seconds),
                    CUDA_service_seconds_non_additive=stages.gpu_seconds(),work_counters=dict(stages.counters),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    stats=result["stats"],recovery_receipt=str(leaf/"call/receipt.json"),recovery_identity=result["identity"])
                atomic_write_json(path,seal(row))
            except BaseException as exc:
                row.update(status="FAILED_MEASUREMENT",observed_seconds=time.perf_counter()-start,
                    error=f"{type(exc).__name__}: {exc}",retained_costs=getattr(session,"last_stats",{}))
                atomic_write_json(path,seal(row))
                raise
            parity_start = time.perf_counter()
            try:
                row["parity"] = compare_cold(VerificationContext(reference),result,leaf/"call")
                row["status"] = "COMPLETE"
            except ValueError as exc:
                row.update(status="REJECTED_PARITY",parity={"status":"FAIL","error":str(exc)})
            row["post_call_parity_seconds"] = time.perf_counter()-parity_start
            atomic_write_json(path,seal(row))
            measurements.append(row)
            print(f"{phase} r{planned['round']} {planned['scene']}/{planned['label']}: {seconds:.3f}s, {row['status']}",flush=True)
            session.last_stats = {}
            del result
            if phase=="final" and row["status"]!="COMPLETE":
                raise RuntimeError("frozen final implementation parity failed; no accuracy reuse or speed claim")
        summary = seal({"phase":phase,"status":"COMPLETE" if all(row['status']=='COMPLETE' for row in measurements) else "PARITY_REJECTIONS",
            "plan_identity":plan_identity,"hardware":settings,"calls":len(measurements),"planned_calls":len(plan),"measurements":measurements})
        atomic_write_json(phase_root/"summary.json",summary)
        return summary
