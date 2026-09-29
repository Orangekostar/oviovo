"""Bound family execution with separate visual environments and GPU queues."""

import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .calibration import fit_source
from .decisions import compose, shortlist
from .e01 import generate_sources
from .evaluation import RANKS, evaluate_sources, pool_methods
from .selection import freeze_nomination, freeze_pair, lock_transfer

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("e01", "e02", "c0", "e03", "e04")


def workers(binding, sam2_python, region_python):
    config = read_json(next(iter(binding["scenes"].values()))["config"])
    result = {"coordinator": config["runtime"]["native_perception_python"],
              "semantic": config["runtime"]["semantic_python"],
              "sam2": str(sam2_python), "region": str(region_python)}
    for path in result.values():
        if not Path(path).is_file():
            raise ValueError(f"bound worker executable missing: {path}")
    if Path(sys.executable).resolve() != Path(result["coordinator"]).resolve():
        raise ValueError("run the coordinator in the original native environment")
    write_once(Path(binding["output_root"]) / "workers.json", result)
    return result


def visual_jobs(binding, jobs, gpus):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("visual worker GPU list must be nonempty and unique")
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    root = Path(binding["output_root"])
    environment = dict(os.environ)
    environment.update({k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})

    def on_gpu(gpu):
        completed = []
        while True:
            try:
                executable, module, arguments = pending.get_nowait()
            except queue.Empty:
                return completed
            command = [executable, "-u", "-m", "src.static_ovmap.a7_evidence_upgrade." + module,
                       "--binding", str(root / "binding.json"), *arguments, "--gpu", str(gpu)]
            key = canonical_digest(command)
            log_root = root / "worker_logs" / key
            log_root.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (log_root / f"attempt_{attempt:03d}.log").exists():
                attempt += 1
            log = log_root / f"attempt_{attempt:03d}.log"
            write_once(log.with_suffix(".command.json"), {"command": command, "gpu": str(gpu), "cwd": str(ROOT)})
            with log.open("x") as handle:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"visual worker failed ({result.returncode}); inspect {log}")
            print({"worker": module, "arguments": arguments, "gpu": str(gpu), "status": "PROCESS_EXIT_ZERO", "log": str(log)}, flush=True)
            completed.append(str(log))

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(on_gpu, gpu) for gpu in gpus]
        return [p for future in futures for p in future.result()]


def family(binding, name, split, environments, gpus, *, scene=None):
    if name not in FAMILIES or split not in ("cal", "replica"):
        raise ValueError("family execution requires one registered family and split")
    spec, root = read_json(binding["spec"]), Path(binding["output_root"])
    expected = spec["datasets"]["calibration" if split == "cal" else "replica"]
    if scene and scene not in expected:
        raise ValueError("scene outside requested frozen split")
    scenes = [scene] if scene else expected
    if split == "replica":
        lock_transfer(binding)
    assets = root.parent / "assets"
    variants = [v["id"] for v in spec["source_variants"] if v["family"].lower() == name]
    if name == "e01":
        for selected in scenes:
            generate_sources(binding, selected)
            print({"family": name, "scene": selected, "status": "SOURCES_COMPLETE"}, flush=True)
    elif name == "e02":
        visual_jobs(binding, [(environments["sam2"], "sam2_worker",
            ["--scene", s, "--assets", str(assets / "sam2")]) for s in scenes], gpus)
        visual_jobs(binding, [(environments["semantic"], "recognition_worker",
            ["--scene", s, "--mode", mode]) for s in scenes for mode in ("GLOBAL", "SAM2")], gpus)
    elif name == "c0":
        visual_jobs(binding, [(environments["semantic"], "capacity_worker",
            ["--scene", s, "--assets", str(assets / "so400m")]) for s in scenes], gpus)
    elif name == "e03":
        visual_jobs(binding, [(environments["region"], "region_worker",
            ["--scene", s, "--assets", str(assets), "--branch", branch])
            for s in scenes for branch in ("FC_FROZEN", "OVR")], gpus)
    if name == "e04":
        access = read_json(root / "e04/access_probe.json")
        if access["status"] != "BLOCKED_ASSET_ACCESS":
            raise ValueError("SAM3 access changed; complete its real adapter before declaring E04 measured")
        for selected in scenes:
            shortlist(binding, selected)
        methods = ["AW_E04_SHORTLIST"]
        # A blocked method has no prediction or invented zero metric.
        write_once(root / "e04" / (split + "_blocked.json"),
            {"status": "BLOCKED_ASSET_ACCESS", "methods": access["affected_methods"],
             "access_receipt": str(root / "e04/access_probe.json"), "split": split, "scene_metrics_created": False})
    else:
        if split == "cal":
            for variant in variants:
                fit_source(binding, variant)
        for selected in scenes:
            evaluate_sources(binding, selected, variants, controls=name == "e01")
            print({"family": name, "scene": selected, "status": "SCENE_RANKS_COMPLETE"}, flush=True)
        methods = [v + suffix for v in variants for suffix in ("_DIRECT", "_A7")]
        if name == "e01":
            methods = spec["references"] + methods
    complete = all((root / "rows" / s / m / (r + ".json")).exists() for s in expected for m in methods for r in RANKS)
    if complete:
        pooled = pool_methods(binding, split, methods)
        result = {"status": "FAMILY_SPLIT_MEASURED", "family": name, "split": split, "methods": methods,
                  "scene_rank_rows": len(expected) * len(methods) * 2, "pool_rows": len(pooled)}
        write_once(root / "phases" / (name + "_" + split + ".json"), result)
        return result
    return {"status": "CONTROLLED_SCENE_LEAF_COMPLETE", "family": name, "split": split, "scene": scene}


def composition(binding, split, *, scene=None):
    spec, root = read_json(binding["spec"]), Path(binding["output_root"])
    if split == "cal":
        pair = freeze_pair(binding)
    else:
        lock_transfer(binding)
        pair = read_json(root / "composition.json")
    if pair["status"] == "PAIR_FROZEN":
        expected = spec["datasets"]["calibration" if split == "cal" else "replica"]
        scenes = [scene] if scene else expected
        if not set(scenes) <= set(expected):
            raise ValueError("composition scene outside split")
        for selected in scenes:
            compose(binding, selected)
        if all((root / "rows" / s / "AW_COMBO_QR" / (rank + ".json")).exists() for s in expected for rank in RANKS):
            pool_methods(binding, split, ["AW_COMBO_QR"])
    if split == "cal" and scene is None:
        freeze_nomination(binding)
        lock_transfer(binding)
    return {"status": "COMPOSITION_PHASE_EXECUTED", "split": split, "pair_status": pair["status"]}


def evaluate(binding, split, *, scene=None):
    spec = read_json(binding["spec"])
    if split == "replica":
        lock_transfer(binding)
    expected = spec["datasets"]["calibration" if split == "cal" else "replica"]
    scenes = [scene] if scene else expected
    if not set(scenes) <= set(expected):
        raise ValueError("evaluation scene outside split")
    variants = [v["id"] for v in spec["source_variants"]]
    for selected in scenes:
        evaluate_sources(binding, selected, variants, controls=True)
        shortlist(binding, selected)
    composition(binding, split, scene=scene)
    if scene is None:
        methods = spec["references"] + [v + suffix for v in variants for suffix in ("_DIRECT", "_A7")] + ["AW_E04_SHORTLIST"]
        pool_methods(binding, split, methods)
    return {"status": "EVALUATION_PHASE_EXECUTED", "split": split}
