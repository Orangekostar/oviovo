"""Authorized phase DAG, bounded processes, and complete artifact reuse."""

import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess
import time
import hashlib

from static_ovmap.backbone_wave1.runtime import execute
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, PathResolver, bind_inputs, read
from .light import METHODS, build_scene, require_transfer_freeze


PHASES = ("bind", "cache-screen", "association-screen", "sam-screen", "compose", "freeze", "transfer", "report", "publish")
A_MAPS = ("RW_A1_NATIVE_FALLBACK", "RW_A2_MULTI_FREE", "RW_A3_MULTI_UNION")
S_MAPS = ("RW_S1_CROP_PRIORITY", "RW_S2_CONFLICT")


def _geometry_job(binding, scene, map_id):
    from .geometry import diagnose_scene
    return diagnose_scene(binding, scene, map_id, resume=True)


def _standard_eval_job(binding, scene, map_id):
    from static_ovmap.backbone_wave1.readouts import evaluate_map

    root = Path(binding["output_root"]) / "readouts" / scene / map_id
    path = root / "evaluation_rows.json"
    if path.is_file():
        receipt = read(path)
        if receipt["status"] != "COMPLETE":
            raise ValueError("standard evaluation receipt is incomplete")
        current = read(root / "receipt.json")
        keys = {method: read(value)["prediction_key"] for method, value in current["predictions"].items()}
        for row in receipt["rows"]:
            if row["prediction_identity"] != keys[row["method"]]:
                raise ValueError("completed standard evaluator used another prediction")
        return receipt["rows"]
    return evaluate_map(binding, scene, read(root / "receipt.json"))


def _light_eval_job(binding, scene, map_id, context, subdir):
    from .evaluation import evaluate_scene
    root = Path(binding["output_root"]) / "light" / scene / map_id
    if subdir:
        root = root / subdir
    if (root / "evaluation_rows.json").is_file():
        receipt, lock = read(root / "evaluation_rows.json"), read(root / "receipt.json")
        if receipt["status"] != "COMPLETE" or receipt["prediction_lock_identity"] != lock["identity"]:
            raise ValueError("completed light scorer belongs to another prediction lock")
        index = ConsumptionIndex(root / "input_verifications.json")
        for condition in lock["conditions"].values():
            path = Path(condition["prediction_manifest"])
            index.identity(path)
            manifest = read(path)
            if manifest["prediction_key"] != condition["prediction_key"]:
                raise ValueError("completed light prediction no longer matches its lock")
            index.identity(path.parent / manifest["arrays"]["path"], manifest["arrays"])
        index.write_memo(root / "input_verifications.json")
        return receipt
    return evaluate_scene(binding, scene, map_id=map_id, context=context, output_subdir=subdir)


def parallel_jobs(function, jobs, workers):
    if not 1 <= workers <= 3:
        raise ValueError("evaluation process limit exceeded")
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(function, *job): i for i, job in enumerate(jobs)}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    return results


def pool_standard(binding, cohort, map_id, *, leave_out=None):
    from static_ovmap.m2_reviewer_study.evaluation import pool
    from static_ovmap.released_loader import load_released_module

    root = Path(binding["output_root"])
    scenes = [scene for scene in binding["datasets"][cohort] if scene != leave_out]
    rows = [row for scene in scenes for row in read(root / "readouts" / scene / map_id / "evaluation_rows.json")["rows"]]
    config = read(root / "readouts" / scenes[0] / map_id / "config.json")
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Replica" if binding["scenes"][scenes[0]]["dataset"] == "Replica" else "Scannet200")
    pool_cohort = cohort if leave_out is None else "leave_out_" + leave_out
    output = root / "pools" / pool_cohort / map_id
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in ("NATIVE_READOUT", "FC_EQ", "D2"):
        for rank in ("OFFICIAL_CURRENT_CLASS", "FROZEN_N0"):
            with (output / (method + "_" + rank + ".log")).open("a") as stream, contextlib.redirect_stdout(stream):
                result = pool(output, rows, namespace, scenes, method, rank)
            result = dict(result, method=method, map_id=map_id, rank_mode=rank)
            atomic_write_json(output / method / (rank + ".json"), result)
            results[method + "/" + rank] = result
    receipt = {"status": "COMPLETE", "map_id": map_id, "cohort": pool_cohort, "scene_order": scenes, "pools": results}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(output / "receipt.json", receipt)
    return receipt


def _verified_outputs(path, statuses):
    receipt = read(path)
    if receipt["status"] not in statuses:
        raise ValueError("cannot reuse an incomplete artifact: " + str(path))
    index = ConsumptionIndex(Path(path).parent / "input_verifications.json")
    index.identity(path)
    for row in receipt.get("outputs", []):
        index.identity(row["path"], row)
    return receipt


def _map_job(binding, build, scene, map_id, threads):
    path = Path(binding["output_root"]) / "maps" / scene / map_id / "map_receipt.json"
    if path.is_file() and read(path)["status"] == "COMPLETE":
        receipt = _verified_outputs(path, {"COMPLETE"})
        if receipt["scene"] != scene or receipt["map_id"] != map_id or receipt["native_build_identity"] != build["recovery_identity"]:
            raise ValueError("completed map differs from the bound scene, recipe or native build")
        from .maps import recipe_for
        from static_ovmap.module_validation.boundary_jobs import verify_capture

        expected = recipe_for(read(binding["spec"]), map_id)
        if any(receipt["recipe"].get(key) != value for key, value in expected.items()):
            raise ValueError("completed map scientific recipe differs from the frozen specification")
        running = read(path.parent / "running.json")
        key = {"binding": binding["identity"], "scene": scene, "recipe": receipt["recipe"],
               "command": receipt["command"]["argv"], "environment": running["environment"],
               "inputs": receipt["inputs"], "build": build["recovery_identity"]}
        if canonical_digest(key) != receipt["input_identity"] or running["input_identity"] != receipt["input_identity"]:
            raise ValueError("completed map recorded input identity is inconsistent")
        capture = verify_capture(receipt["capture_manifest"], allow_skipped=True)
        data = binding["scenes"][scene]
        if (capture["scheduled_frame_ids"] != data["schedule"]
                or capture["completed_frame_ids"] != data["completed_frame_ids"]):
            raise ValueError("completed map schedule or native frame policy changed")
        index = ConsumptionIndex(path.parent / "input_verifications.json")
        repo = Path(binding["repository_root"])
        for row in receipt["inputs"]:
            source = Path(row["path"])
            if source.is_relative_to(repo) and source.suffix in {".py", ".patch"}:
                blob = subprocess.check_output(["git", "show", running["source_commit"] + ":" + str(source.relative_to(repo))], cwd=repo)
                if hashlib.sha256(blob).hexdigest() != row["sha256"]:
                    raise ValueError("historical map producer is not restorable from its recorded source commit")
            else:
                index.identity(source, row)
        return receipt
    from .maps import run_map
    return run_map(binding, build, scene, map_id, threads=threads, resume=True)


def _ensure_fc(binding, scene, gpu, *, context=None, map_id="BB00_NATIVE", arms=("U2", "U3")):
    root = Path(binding["output_root"]) / "recovery" / scene / map_id
    path = root / "fc_recovery_receipt.json"
    if path.is_file() and read(path)["status"] == "COMPLETE":
        receipt = _verified_outputs(path, {"COMPLETE"})
        if receipt["scene"] != scene or receipt["map_id"] != map_id or not set(arms) <= set(receipt["sources"]):
            raise ValueError("completed FC acquisition differs from the requested frozen scope")
        documents = {name: read(root / (name + ".json")) for name in
                     ("registry", "captured_requests", "cached_sources", "fc_request_plan")}
        data = context or binding["scenes"][scene]
        worker = next(row for row in receipt["inputs"] if Path(row["path"]).name == "recovery_fc_worker.py")
        key = {"scene": scene, "map_id": map_id, "prerequisites": {name: row["identity"] for name, row in documents.items()},
            "model": data["FC_physical_model_identity"], "text": data["FC_text"]["sha256"], "worker": worker["sha256"]}
        if "requested_arms" in receipt:
            key["requested_arms"] = receipt["requested_arms"]
        if canonical_digest(key) != receipt["identity"]:
            raise ValueError("complete FC producer identity does not match its exact prerequisite artifacts")
        return receipt
    command = [binding["fc"]["python"], "-m", "static_ovmap.recovery_wave2.recovery_fc_worker",
        "--binding", Path(binding["output_root"]) / "resolved_inputs.json", "--scene", scene, "--gpu", str(gpu), "--map-id", map_id]
    if context is not None:
        context_path = root / "map_context.json"
        atomic_write_json(context_path, context)
        command += ["--context", context_path]
    if not arms:
        command += ["--no-fc"]
    else:
        for arm in arms:
            command += ["--arm", arm]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(Path(binding["repository_root"]) / "src") + ":" + binding["repository_root"],
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    execute(command, binding["repository_root"], root / "fc_run.log", env=env)
    return _verified_outputs(path, {"COMPLETE"})


def prepare_recovery(binding, scene, gpu, *, context=None, map_id="BB00_NATIVE", arms=("U2", "U3")):
    from .recovery_registry import prepare_registry
    from .recovery_sources import prepare_cached_sources

    receipt = prepare_registry(binding, scene, context=context)
    root = Path(receipt["registry"]).parent
    if (root / "cached_source_receipt.json").is_file():
        _verified_outputs(root / "cached_source_receipt.json", {"CACHED_SOURCES_AND_FC_PLAN_LOCKED"})
        if read(root / "cached_sources.json")["registry_identity"] != read(receipt["registry"])["identity"]:
            raise ValueError("completed recovery sources belong to another registry")
    else:
        prepare_cached_sources(binding, scene, receipt, context=context)
    _ensure_fc(binding, scene, gpu, context=context, map_id=map_id, arms=arms)
    return receipt


class Workflow:
    def __init__(self, args):
        self.args, self.spec = args, read(args.spec)
        self.repo, self.root = Path(__file__).parents[3], Path(args.output_root).resolve()
        if not (1 <= args.mapping_workers <= 2 and 1 <= args.mapping_threads <= 8 and 1 <= args.evaluation_workers <= 3):
            raise ValueError("configured worker/thread count exceeds the fixed protocol")
        self.binding = None

    def bind(self):
        path = self.root / "resolved_inputs.json"
        if path.is_file() and self.args.resume:
            self.binding = read(path)
            if (self.binding["identity"] != canonical_digest({key: value for key, value in self.binding.items() if key != "identity"})
                    or self.binding["spec_sha256"] != ConsumptionIndex().identity(self.args.spec)["sha256"]
                    or Path(self.binding["parent_root"]).resolve() != Path(self.args.parent_root).resolve()
                    or Path(self.binding["repository_root"]).resolve() != self.repo.resolve()):
                raise ValueError("resolved task inputs differ from this exact resume command")
        else:
            self.binding = bind_inputs(self.args.spec, self.repo, self.args.parent_root, self.root, path_map=self.args.path_map)
        return {"status": self.binding["status"], "binding_identity": self.binding["identity"]}

    def build(self):
        path = Path(self.args.native_build) if self.args.native_build else self.root.parent / "tooling/recovery_native_v2/recovery_native_build_receipt.json"
        if not path.is_file():
            raise FileNotFoundError("restore or build the isolated recovery native extension: " + str(path))
        build = read(path)
        ConsumptionIndex().identity(build["extension"]["path"], build["extension"])
        validation = read(self.root / "validation/native/validation_receipt.json")
        if validation["status"] != "VERIFIED" or validation["new_extension"] != build["extension"]:
            raise ValueError("mapping requires the verified exact recovery native extension")
        return build

    def cache_screen(self):
        from .evaluation import pool_cohort
        from .selection import write_components

        scenes = self.binding["datasets"]["development"]
        for scene in scenes:
            prepare_recovery(self.binding, scene, self.args.gpu)
            path = self.root / "light" / scene / "BB00_NATIVE/receipt.json"
            if not path.is_file():
                build_scene(self.binding, scene)
        parallel_jobs(_light_eval_job, [(self.binding, scene, "BB00_NATIVE", None, None) for scene in scenes], self.args.evaluation_workers)
        pools = pool_cohort(self.binding, "development", scenes, METHODS)
        components = write_components(self.binding)
        return {"status": "COMPLETE", "pools": pools["identity"], "components": components["identity"]}

    def maps(self, cohort, map_ids):
        build = self.build()
        name = cohort + "_" + "_".join(map_ids)
        external = self.root / "execution" / (name + ".running.json")
        final = self.root / "execution" / (name + ".json")
        while external.is_file() and not final.is_file() and read(external)["completed"] < read(external)["total"]:
            time.sleep(2)
        jobs = [(self.binding, build, scene, map_id, self.args.mapping_threads) for map_id in map_ids
                for scene in self.binding["datasets"][cohort]]
        return parallel_jobs(_map_job, jobs, self.args.mapping_workers)

    def screen_maps(self, map_ids):
        from .geometry import gate_development
        from .readouts import build_sources

        scenes, build, results = self.binding["datasets"]["development"], self.build(), {}
        for map_id in map_ids:
            parallel_jobs(_geometry_job, [(self.binding, scene, map_id) for scene in scenes], self.args.evaluation_workers)
            screen = gate_development(self.binding, map_id)
            results[map_id] = {"geometry": screen["identity"], "status": screen["status"]}
            if screen["status"] == "GEOMETRY_PASS":
                for scene in scenes:
                    path = self.root / "readouts" / scene / map_id / "receipt.json"
                    if path.is_file():
                        _verified_outputs(path, {"PREDICTIONS_LOCKED"})
                    else:
                        build_sources(self.binding, scene, map_id, build, gpu=self.args.gpu)
                parallel_jobs(_standard_eval_job, [(self.binding, scene, map_id) for scene in scenes], self.args.evaluation_workers)
                results[map_id]["semantic_pool"] = pool_standard(self.binding, "development", map_id)["identity"]
        return results

    def association_screen(self):
        self.maps("development", A_MAPS)
        return {"status": "COMPLETE", "arms": self.screen_maps(A_MAPS)}

    def sam_screen(self):
        from .sam_completion import build_s1

        frontends = [build_s1(self.binding, scene, resume=True) for scene in self.binding["datasets"]["development"]]
        changed = sum(row["changed_canonical_frames"] for row in frontends)
        if not changed:
            for map_id in S_MAPS:
                row = {"status": "EQUIVALENT_INPUT", "map_id": map_id, "scene_order": self.binding["datasets"]["development"],
                    "frontend_identities": [item["identity"] for item in frontends], "new_full_maps": 0,
                    "semantic_status": "EXACT_PARENT_ALIAS", "S2_implemented": True}
                row["identity"] = canonical_digest(row)
                atomic_write_json(self.root / "geometry/screens" / (map_id + ".json"), row)
            return {"status": "COMPLETE", "S1_gate": "EQUIVALENT_INPUT", "new_maps": 0}
        self.maps("development", S_MAPS)
        return {"status": "COMPLETE", "changed_frontend_frames": changed, "arms": self.screen_maps(S_MAPS)}

    def selected_light(self, scene, map_id, recipe, method):
        from .readouts import recovery_context

        context = recovery_context(self.binding, scene, map_id)
        arm = recipe["recovery"]
        prepare_recovery(self.binding, scene, self.args.gpu, context=context, map_id=map_id,
                         arms=(arm,) if arm in {"U2", "U3"} else ())
        subdir = "extra_conditions/" + method
        if not (self.root / "light" / scene / map_id / subdir / "receipt.json").is_file():
            build_scene(self.binding, scene, context=context, map_id=map_id,
                        conditions_override={method: recipe}, output_subdir=subdir)
        return _light_eval_job(self.binding, scene, map_id, context, subdir)

    def compose(self):
        from .evaluation import pool_cohort
        from .selection import select_final, write_components

        components = write_components(self.binding)
        scenes, subdir = self.binding["datasets"]["development"], "extra_conditions/RW_LIGHT_COMBO"
        for scene in scenes:
            if not (self.root / "light" / scene / "BB00_NATIVE" / subdir / "receipt.json").is_file():
                build_scene(self.binding, scene, conditions_override=components["combination"], output_subdir=subdir)
        parallel_jobs(_light_eval_job, [(self.binding, scene, "BB00_NATIVE", None, subdir) for scene in scenes], self.args.evaluation_workers)
        pool_cohort(self.binding, "development", scenes, ["RW_LIGHT_COMBO"])
        from .coverage_diagnostics import common_coverage

        common_coverage(self.binding, "development", workers=self.args.evaluation_workers)
        selection = select_final(self.binding)
        from .sensitivity import development_sensitivity

        sensitivity = development_sensitivity(self.binding)
        recipe, nominee = selection["light_package"]["recipe"], selection["nominated_map"]
        checks = {"old_map_fixed_D2": read(self.root / "light/pools/development/BB00_NATIVE/RW_B_D2.json"),
                  "old_map_light": read(self.root / "light/pools/development/BB00_NATIVE" / (selection["light_package"]["id"] + ".json"))}
        if nominee:
            checks["new_map_fixed_D2"] = read(self.root / "pools/development" / nominee / "D2/OFFICIAL_CURRENT_CLASS.json")
            if recipe == {"gamma": .5, "recovery": None}:
                checks["new_map_light"] = {**checks["new_map_fixed_D2"], "reuse_kind": "FROZEN_LIGHT_IS_EXACT_D2"}
            else:
                receipts = [self.selected_light(scene, nominee, recipe, "RW_FROZEN_LIGHT") for scene in scenes]
                checks["new_map_light"] = pool_selected(self.binding, "development", nominee, receipts)
        result = {"status": "COMPLETE", "nominated_map": nominee, "light_recipe": recipe, "cells": checks,
            "map_composition": "TWO_BY_TWO" if nominee else "NOT_RUN_NO_QUALIFYING_MAP", "new_A_plus_S_sweeps": 0}
        result["identity"] = canonical_digest(result)
        atomic_write_json(self.root / "selection/composition_check.json", result)
        return {"status": "COMPLETE", "selection_identity": selection["identity"],
                "sensitivity_identity": sensitivity["identity"], "composition_check": result["identity"]}

    def freeze(self):
        from .selection import commit_freeze
        return commit_freeze(self.binding, read(self.root / "selection/final.json"), read(self.root / "selection/composition_check.json"))

    def transfer(self):
        from .evaluation import pool_cohort
        from .readouts import build_sources

        scenes = self.binding["datasets"]["replica"]
        freeze = read(self.root / "freeze/receipt.json")
        for scene in scenes:
            require_transfer_freeze(self.binding, scene)
            prepare_recovery(self.binding, scene, self.args.gpu)
            if not (self.root / "light" / scene / "BB00_NATIVE/receipt.json").is_file():
                build_scene(self.binding, scene)
        parallel_jobs(_light_eval_job, [(self.binding, scene, "BB00_NATIVE", None, None) for scene in scenes], self.args.evaluation_workers)
        methods = list(METHODS)
        package = freeze["light_package"]
        if package["id"] not in METHODS:
            subdir = "extra_conditions/RW_FROZEN_LIGHT"
            for scene in scenes:
                if not (self.root / "light" / scene / "BB00_NATIVE" / subdir / "receipt.json").is_file():
                    build_scene(self.binding, scene, conditions_override={"RW_FROZEN_LIGHT": package["recipe"]}, output_subdir=subdir)
            parallel_jobs(_light_eval_job, [(self.binding, scene, "BB00_NATIVE", None, subdir) for scene in scenes], self.args.evaluation_workers)
            methods.append("RW_FROZEN_LIGHT")
        pools = pool_cohort(self.binding, "replica", scenes, methods)
        from .coverage_diagnostics import common_coverage

        common_coverage(self.binding, "replica", workers=self.args.evaluation_workers)
        nominee, map_pools = freeze["nominated_map"], None
        if nominee:
            if nominee in S_MAPS:
                from .sam_completion import build_s1
                for scene in scenes:
                    build_s1(self.binding, scene, resume=True)
            self.maps("replica", [nominee])
            for scene in scenes:
                path = self.root / "readouts" / scene / nominee / "receipt.json"
                if not path.is_file():
                    build_sources(self.binding, scene, nominee, self.build(), gpu=self.args.gpu)
            parallel_jobs(_standard_eval_job, [(self.binding, scene, nominee) for scene in scenes], self.args.evaluation_workers)
            map_pools = pool_standard(self.binding, "replica", nominee)
            parallel_jobs(_geometry_job, [(self.binding, scene, nominee) for scene in scenes], self.args.evaluation_workers)
            if package["recipe"] != {"gamma": .5, "recovery": None}:
                receipts = [self.selected_light(scene, nominee, package["recipe"], "RW_FROZEN_LIGHT") for scene in scenes]
                pool_selected(self.binding, "replica", nominee, receipts)
        return {"status": "COMPLETE", "light_pool_identity": pools["identity"],
                "map_pool_identity": map_pools["identity"] if map_pools else None, "selection_changed": False}

    def report(self):
        from .reporting import generate
        return generate(self.binding)

    def publish(self):
        from .publication import publish
        return publish(self.binding)

    def run(self, phase):
        if phase not in (*PHASES, "all"):
            raise ValueError("unknown recovery phase")
        if phase not in {"bind", "all"}:
            self.bind()
        results = {}
        for name in PHASES if phase == "all" else (phase,):
            started = time.monotonic()
            path = self.root / "execution/phases" / (name + ".json")
            if path.is_file() and read(path)["status"] in {"FAILED_IMPLEMENTATION", "FAILED", "RUNNING"}:
                path.rename(path.with_name(name + ".previous_" + str(time.time_ns()) + ".json"))
            atomic_write_json(path, {"status": "RUNNING", "phase": name, "started_unix": time.time()})
            try:
                result = getattr(self, name.replace("-", "_"))()
                receipt = {"phase": name, **result, "elapsed_seconds": time.monotonic() - started,
                    "binding_identity": self.binding["identity"], "actual_options": vars(self.args)}
                receipt["identity"] = canonical_digest(receipt)
                atomic_write_json(path, receipt)
                results[name] = receipt
                print(name, receipt["status"], flush=True)
            except BaseException as exc:
                atomic_write_json(path, {"status": "FAILED_IMPLEMENTATION", "phase": name,
                    "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
                raise
        return results


def pool_selected(binding, cohort, map_id, receipts):
    from static_ovmap.m2_reviewer_study.evaluation import pool
    from static_ovmap.released_loader import load_released_module

    scenes = binding["datasets"][cohort]
    rows = [row for receipt in receipts for row in receipt["rows"]]
    config = read(Path(binding["output_root"]) / "readouts" / scenes[0] / map_id / "config.json")
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Replica" if binding["scenes"][scenes[0]]["dataset"] == "Replica" else "Scannet200")
    root = Path(binding["output_root"]) / "pools" / cohort / map_id
    with (root / "frozen_light.log").open("a") as stream, contextlib.redirect_stdout(stream):
        result = pool(root, rows, namespace, scenes, "RW_FROZEN_LIGHT", "OFFICIAL_CURRENT_CLASS")
    atomic_write_json(root / "RW_FROZEN_LIGHT/OFFICIAL_CURRENT_CLASS.json", result)
    return result
