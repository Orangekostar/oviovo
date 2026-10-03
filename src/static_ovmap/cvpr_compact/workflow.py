"""Fixed-cohort task orchestration with exact-input worker resumption."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .anchor import run_anchors
from .base_sources import run_base_sources
from .benchmark_inputs import prepare_inputs
from .binding import bind_inputs
from .projected_views import _verified_identity, build_projected_views
from .protocol import experiment_matrix, load_spec
from .recovery_run import load_recovery_inputs
from .runtime import LiveLeafError, execute_leaf, require_frozen_execution
from .timing import _process


PHASES = ("bind", "prepare", "anchors", "views", "encode", "predict", "evaluate", "time", "tables", "publish", "all")
GPU_MODULES = {"recovery_run", "native_region_worker", "timing"}
REQUIRED_MODULES = ("protocol", "binding", "benchmark_inputs", "anchor", "base_sources", "projected_views",
    "outputs", "region_worker", "native_region_worker", "prediction_worker", "evaluation", "recovery_run",
    "runtime", "timing", "diagnostics", "external", "costs", "tables", "reports", "freezing", "publication", "workflow")
WORKER_DEPENDENCIES = {
    "recovery_run": ("region_worker", "projected_views", "outputs", "costs", "runtime", "protocol"),
    "native_region_worker": ("projected_views", "runtime", "protocol"),
    "prediction_worker": ("outputs", "costs", "region_worker", "projected_views", "runtime", "protocol"),
    "evaluation": ("runtime", "protocol", "projected_views"),
    "timing": ("recovery_run", "region_worker", "projected_views", "outputs", "costs", "runtime", "protocol"),
}


def _verify_result(path, status, index):
    index.identity(path)
    result = read(path)
    _verified_identity(result)
    if result["status"] != status:
        raise RuntimeError("worker lacks its required complete result: " + str(path))
    for item in result.get("inputs", []) + result.get("outputs", []):
        index.identity(item["path"], item)
    return result


def run_worker(binding, *, module, python, arguments, result_path, expected_status, inputs):
    if module not in REQUIRED_MODULES:
        raise ValueError("worker must belong to the new compact task")
    root = Path(binding["output_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    producer = Path(__file__).with_name(module + ".py")
    arguments = list(map(str, arguments))
    argv = [str(python), "-m", "static_ovmap.cvpr_compact." + module,
            "--binding", str(root / "resolved_inputs.json"), *arguments]
    identity = canonical_digest({"binding": binding["identity"], "argv": argv,
        "producer": index.identity(producer), "inputs": [index.identity(path) for path in inputs],
        "dependency_sources": [index.identity(producer.with_name(name + ".py")) for name in WORKER_DEPENDENCIES.get(module, ())],
        "expected_result": str(result_path), "expected_status": expected_status})
    log = root / "execution" / module / ("job_" + identity[:16] + ".log")
    ledger_path = log.with_name("command_" + identity + ".json")
    if ledger_path.is_file():
        ledger = read(ledger_path)
        if ledger["input_identity"] != identity:
            raise ValueError("worker execution ledger changed its exact input identity")
        previous = ledger["attempts"][-1] if ledger["attempts"] else None
        if previous and previous["status"] == "COMPLETE":
            if previous["argv"] != argv or previous["cwd"] != binding["repository_root"] or previous["exit_code"] != 0:
                raise ValueError("successful task worker differs from its exact command")
            return _verify_result(result_path, expected_status, index)
        if previous and previous["status"] == "RUNNING" and "child_pid" in previous:
            actual = _process(previous["child_pid"])
            if actual and actual["live"] and actual["argv"] == argv:
                raise RuntimeError("the identical task worker is still live; do not restart it")
    environment = {**os.environ, "PYTHONPATH": binding["repository_root"] + "/src:" + binding["repository_root"],
        "CUDA_VISIBLE_DEVICES": str(binding["gpu"]) if module in GPU_MODULES else "",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for attempt in range(3):
        try:
            execute_leaf(argv, binding["repository_root"], log, env=environment, input_identity=identity)
            break
        except LiveLeafError:
            raise
        except RuntimeError:
            if attempt == 2:
                raise
    result = _verify_result(result_path, expected_status, index)
    index.write_memo(root / "validation/input_verifications.json")
    return result


def _context(binding, scene):
    path = Path(binding["output_root"]) / "contexts" / (scene + ".json")
    if not path.is_file():
        result = run_base_sources(binding, scene)
        if result["status"] != "NQF_READY":
            raise RuntimeError("existing N/Q/F evidence is incomplete: " + scene)
        if isinstance(result["context"], dict):
            data = {**result["context"], "binding_identity": binding["identity"],
                    "alias_kind": result["alias_kind"]}
            data.pop("identity", None)
            data["identity"] = canonical_digest(data)
            atomic_write_json(path, data)
    data = read(path)
    _verified_identity(data)
    return data, path


def _wait_existing_anchor_controller(binding):
    path = Path(binding["output_root"]) / "execution/anchors/background_controller.json"
    if not path.is_file():
        return
    expected = read(path)
    last_report = 0.
    while True:
        actual = _process(expected["pid"])
        if (not actual or not actual["live"] or actual["start_ticks"] != expected["process_start_ticks"]
                or actual["argv"] != expected["argv"]):
            return
        if time.monotonic() - last_report >= 30:
            print(f"Waiting for live label-free anchor controller pid={expected['pid']} start_ticks={actual['start_ticks']}", flush=True)
            last_report = time.monotonic()
        time.sleep(5)


def _views(binding, scenes):
    root = Path(binding["output_root"])
    results = {}
    for scene in scenes:
        data, context = _context(binding, scene)
        inputs = load_recovery_inputs(binding, scene, context=data)
        results[scene] = build_projected_views(data["capture_manifest"], inputs.baseline,
                                             root / "projected_views" / scene)
    return results


def _encode(binding, scenes):
    root = Path(binding["output_root"])
    results = {}
    for scene in scenes:
        data, context = _context(binding, scene)
        views = root / "projected_views" / scene / "receipt.json"
        fc_root = root / "recovery" / scene
        fc = run_worker(binding, module="recovery_run", python=binding["fc"]["python"],
            arguments=["--scene", scene, "--output-root", fc_root], result_path=fc_root / "receipt.json",
            expected_status="COMPLETE", inputs=[context, views])
        native_root = root / "native_recovery" / scene
        native = run_worker(binding, module="native_region_worker", python=data["runtime"]["native_perception_python"],
            arguments=["--scene", scene, "--views", views, "--context", context, "--output-root", native_root],
            result_path=native_root / "receipt.json", expected_status="COMPLETE", inputs=[context, views])
        results[scene] = {"FC": fc["identity"], "Native": native["identity"]}
    return results


def _predict(binding, scenes):
    root = Path(binding["output_root"])
    results = {}
    for scene in scenes:
        data, context = _context(binding, scene)
        views = root / "projected_views" / scene / "receipt.json"
        fc = root / "recovery" / scene / "regions/receipt.json"
        native = root / "native_recovery" / scene / "receipt.json"
        output = root / "predictions" / scene
        locked = run_worker(binding, module="prediction_worker", python=data["runtime"]["native_perception_python"],
            arguments=["--scene", scene, "--views", views, "--regions", fc, "--native-regions", native,
                "--context", context, "--output-root", output], result_path=output / "receipt.json",
            expected_status="PREDICTIONS_LOCKED", inputs=[context, views, fc, native])
        standalone = read(root / "recovery" / scene / "receipt.json")
        _verified_identity(standalone)
        for method, exported in standalone["exports"].items():
            payload = load_prediction(locked["predictions"][method])
            if payload.record_key != exported["record_key"] or payload.prediction_key != exported["prediction_key"]:
                raise RuntimeError("final scientific output differs from the timed production export: " + scene + "/" + method)
        results[scene] = locked["identity"]
    return results


def _evaluate(binding, scenes, *, pool=True):
    root = Path(binding["output_root"])
    def evaluate(scene):
        data, context = _context(binding, scene)
        lock = root / "predictions" / scene / "receipt.json"
        output = root / "evaluation" / scene
        return run_worker(binding, module="evaluation", python=data["runtime"]["native_perception_python"],
            arguments=["--scene", scene, "--lock", lock, "--output-root", output], result_path=output / "receipt.json",
            expected_status="COMPLETE", inputs=[context, lock])
    results = {}
    with ThreadPoolExecutor(max_workers=3) as workers:
        jobs = {workers.submit(evaluate, scene): scene for scene in scenes}
        for job in as_completed(jobs):
            scene = jobs[job]
            results[scene] = job.result()["identity"]
    if pool:
        for cohort, selected in binding["cohorts"].items():
            first, context = _context(binding, selected[0])
            run_worker(binding, module="evaluation", python=first["runtime"]["native_perception_python"],
                arguments=["--cohort", cohort], result_path=root / "pools" / cohort / "receipt.json",
                expected_status="COMPLETE", inputs=[root / "evaluation" / scene / "receipt.json" for scene in selected])
    return results


def run_smoke(binding):
    scene = load_spec(binding["spec"])["smoke_scene"]
    require_frozen_execution(binding, scene)
    _views(binding, [scene])
    _encode(binding, [scene])
    _predict(binding, [scene])
    _evaluate(binding, [scene], pool=False)
    root = Path(binding["output_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    locked = read(root / "predictions" / scene / "receipt.json")
    if locked["output_count"] != 8 or not locked["baseline_array_and_rank_parity"]:
        raise RuntimeError("the single existing-anchor smoke lacks all eight fixed outputs and exact A0 parity")
    result = {"status": "COMPLETE", "scene": scene, "logical_condition_count": 8,
        "new_development_map_count": 0, "prediction_lock_identity": locked["identity"],
        "outputs": [index.identity(root / directory / scene / "receipt.json")
                    for directory in ("projected_views", "recovery", "native_recovery", "predictions", "evaluation")],
        "main_performance_outcomes_seen": False}
    from .freezing import implementation_inventory
    result["implementation_sources"] = implementation_inventory(binding, index=index)
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "validation/final_smoke.json", result)
    return result


def _require_complete_implementation(repository_root):
    repo = Path(repository_root)
    missing = ["src/static_ovmap/cvpr_compact/" + name + ".py" for name in REQUIRED_MODULES
               if not (repo / "src/static_ovmap/cvpr_compact" / (name + ".py")).is_file()]
    if not (repo / "scripts/evaluation/run_ovimap_cvpr_compact.py").is_file():
        missing.append("scripts/evaluation/run_ovimap_cvpr_compact.py")
    if missing:
        raise RuntimeError("INCOMPLETE_IMPLEMENTATION: finish the full required pipeline before all/main execution: " + ", ".join(missing))


def run_pipeline(spec_path, repository_root, *, phase="all", resume=False, gpu=None, path_maps=()):
    if phase not in PHASES:
        raise ValueError("unknown compact-table phase")
    spec = load_spec(spec_path)
    binding = bind_inputs(spec_path, repository_root, path_maps=path_maps, gpu=gpu)
    root = Path(binding["output_root"])
    scenes = [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]
    if phase == "all":
        _require_complete_implementation(repository_root)
    if phase in ("views", "encode", "predict", "evaluate", "time", "tables", "publish"):
        for scene in scenes:
            require_frozen_execution(binding, scene)
    selected = list(PHASES[:-1]) if phase == "all" else [phase]
    results = {}
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "execution/.workflow_controller.lock"):
        for current in selected:
            started = time.monotonic()
            record = {"status": "RUNNING", "phase": current, "binding_identity": binding["identity"],
                "matrix_identity": experiment_matrix(spec)["identity"], "resume": resume, "controller_pid": os.getpid(),
                "controller_start_ticks": _process(os.getpid())["start_ticks"]}
            receipt_path = root / "phases" / (current + ".json")
            if receipt_path.is_file():
                previous = read(receipt_path)
                _verified_identity(previous)
                if previous["binding_identity"] != binding["identity"]:
                    raise ValueError("workflow phase belongs to a different immutable task binding")
                if not resume:
                    raise RuntimeError("existing compact task phase requires --resume; previous work is preserved")
                receipt_path.rename(receipt_path.with_name(current + ".previous_" + str(time.time_ns()) + ".json"))
            atomic_write_json(receipt_path, record)
            try:
                if current == "bind":
                    result = {"binding_identity": binding["identity"], "status": binding["status"]}
                elif current == "prepare":
                    result = prepare_inputs(binding, spec)
                    if result["complete"] != 18:
                        raise RuntimeError("fixed CF18 preparation is incomplete; no capture may be dropped")
                elif current == "anchors":
                    _wait_existing_anchor_controller(binding)
                    result = run_anchors(binding)
                    if result["status"] != "COMPLETE" or result["complete_main_contexts"] != 26:
                        raise RuntimeError("fixed native anchors are incomplete")
                elif current == "views":
                    if phase == "all":
                        from .freezing import ensure_freeze
                        ensure_freeze(binding)
                    for scene in scenes:
                        require_frozen_execution(binding, scene)
                    result = _views(binding, scenes)
                elif current == "encode":
                    result = _encode(binding, scenes)
                elif current == "predict":
                    result = _predict(binding, scenes)
                elif current == "evaluate":
                    result = _evaluate(binding, scenes)
                elif current == "time":
                    scientific = {scene: str(root / "recovery" / scene / "receipt.json") for scene in spec["cohorts"]["replica8"]}
                    inputs_path = root / "timing/scientific_parents.json"
                    atomic_write_json(inputs_path, scientific)
                    result = run_worker(binding, module="timing", python=binding["fc"]["python"],
                        arguments=["--scientific-receipts", inputs_path], result_path=root / "timing/pool.json",
                        expected_status="COMPLETE", inputs=[inputs_path, *scientific.values()])
                elif current == "tables":
                    from .diagnostics import run_diagnostics
                    from .tables import build_tables
                    from .reports import write_reports
                    diagnostic = run_diagnostics(binding)
                    result = build_tables(binding, diagnostic)
                    write_reports(binding)
                else:
                    from .publication import publish
                    result = publish(binding)
                record.update(status="COMPLETE", result=result)
            except BaseException as exc:
                record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                record["identity"] = canonical_digest({k: v for k, v in record.items() if k != "identity"})
                atomic_write_json(receipt_path, record)
            results[current] = record["identity"]
            print("Compact phase", current, "COMPLETE", flush=True)
    return {"phase": phase, "status": "COMPLETE", "binding_identity": binding["identity"], "phase_identities": results}
