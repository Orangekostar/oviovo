"""Receipt-driven exposed-development study and frozen historical transfer."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from static_ovmap.m2_reviewer_study.evaluation import pool
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.boundary_jobs import file_identity
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.released_loader import load_released_module
from .binding import bind_inputs, read
from .diagnostics import compare_maps, diagnose_map
from .maps import run_map
from .prepare import native_traces, sam_preflight
from .readouts import build_sources, ensure_frontend, evaluate_map
from .runtime import build_native, check_gpu_once
from .selection import freeze_candidates, freeze_transfer


def git_value(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def implementation_commit(repo):
    paths = ["src/static_ovmap/backbone_wave1", "scripts/evaluation/run_ovimap_backbone_wave1.py",
             "third_party_patches/ovimap/backbone_wave1_v1"]
    dirty = git_value(repo, "status", "--porcelain", "--", *paths)
    if dirty:
        raise ValueError("implementation must be committed before main measurements: " + dirty)
    return git_value(repo, "rev-parse", "HEAD")


def _evaluate_diagnose(binding, spec, scene, receipt):
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rows = evaluate_map(binding, scene, receipt)
    diagnose_map(binding, spec, scene, receipt)
    return rows


class Study:
    def __init__(self, args):
        self.args = args
        self.spec = read(args.spec)
        self.repo = Path(self.spec["worktree"])
        self.root = Path(args.output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = None
        self.build = None

    def bind(self):
        self.binding = bind_inputs(self.args.spec, self.repo, self.root)
        return self.binding

    def bound(self):
        if self.binding is None:
            self.binding = read(self.root / "resolved_inputs.json")
        if Path(self.binding["output_root"]).resolve() != self.root.resolve():
            raise ValueError("attempt root differs from the resolved input binding")
        return self.binding

    def native_build(self):
        if self.build is None:
            self.build = build_native(self.spec, self.repo, resume=self.args.resume)
        return self.build

    def available_gpu(self, phase):
        value = check_gpu_once(self.args.gpu, self.root / "resources" / (phase + ".json"))
        if value["status"] != "AVAILABLE":
            raise RuntimeError("RESOURCE_BLOCK: configured GPU is occupied; bounded check recorded")

    def prepare(self):
        binding = self.bound()
        self.available_gpu("prepare")
        trace = native_traces(binding, self.spec, self.native_build())
        sam_path = self.root / "prepare/sam2_preflight/receipt.json"
        cached_sam = read(sam_path) if sam_path.is_file() else None
        worker = file_identity(Path(__file__).with_name("frontend_sam2.py"))
        matching_worker = cached_sam and worker in cached_sam.get("inputs_and_outputs", [])
        if cached_sam and not matching_worker:
            number = 1
            old_root = sam_path.parent
            while old_root.with_name(f"{old_root.name}.previous_{number:03d}").exists():
                number += 1
            old_root.rename(old_root.with_name(f"{old_root.name}.previous_{number:03d}"))
            cached_sam = None
        sam = cached_sam if cached_sam and cached_sam["status"] == "COMPLETE" else sam_preflight(binding, gpu=self.args.gpu)
        if not trace["probe_read_only"] or len(sam["frames"]) != 3:
            raise ValueError("real native/SAM preflight evidence is incomplete")
        return {"native": trace, "SAM": sam}

    def run_jobs(self, scenes, recipes, phase):
        binding, build = self.bound(), self.native_build()
        commit = implementation_commit(self.repo)
        self.available_gpu(phase)
        successful = [read(path) for path in (self.root / "maps").glob("*/*/map_receipt.json")
                      if read(path)["status"] == "COMPLETE"]
        required = {(scene, recipe["id"]) for scene in scenes for recipe in recipes}
        existing = {(row["scene"], row["map_id"]) for row in successful}
        if len(existing | required) > 64:
            raise ValueError("authorized map-scene budget would be exceeded")
        if shutil.disk_usage(self.root).free < 8 * 1024 ** 3:
            raise RuntimeError("RESOURCE_BLOCK: fewer than 8GiB available for a fresh map")
        failures, completed, evaluation_jobs = [], [], []
        blocked_frontends = {}
        if any(recipe["frontend"] != "cropformer" for recipe in recipes):
            for scene in scenes:
                try:
                    ensure_frontend(binding, scene, gpu=self.args.gpu)
                except Exception as exc:
                    blocked_frontends[scene] = f"{type(exc).__name__}: {exc}"
                    failures.extend({"scene": scene, "map_id": recipe["id"],
                        "stage": "FRONTEND_PREREQUISITE", "error": blocked_frontends[scene]}
                        for recipe in recipes if recipe["frontend"] != "cropformer")
        with ThreadPoolExecutor(max_workers=self.args.mapping_workers) as mapper, ProcessPoolExecutor(
                max_workers=self.args.evaluation_workers, mp_context=multiprocessing.get_context("spawn")) as evaluator:
            jobs = {mapper.submit(run_map, binding, self.spec, build, scene, recipe,
                threads=self.args.mapping_threads, resume=self.args.resume): (scene, recipe)
                for scene in scenes for recipe in recipes
                if recipe["frontend"] == "cropformer" or scene not in blocked_frontends}
            for future in as_completed(jobs):
                scene, recipe = jobs[future]
                try:
                    mapping = future.result()
                    print(json.dumps({"phase": phase, "scene": scene, "map_id": recipe["id"], "stage": "MAP_COMPLETE"}), flush=True)
                    sources = build_sources(binding, self.spec, scene, mapping, gpu=self.args.gpu)
                    evaluation_jobs.append((evaluator.submit(_evaluate_diagnose, binding, self.spec, scene, sources), scene, recipe))
                    completed.append({"scene": scene, "map_id": recipe["id"], "source_identity": sources["identity"]})
                except Exception as exc:
                    failures.append({"scene": scene, "map_id": recipe["id"], "error": f"{type(exc).__name__}: {exc}"})
                    print(json.dumps(failures[-1]), flush=True)
            for future, scene, recipe in evaluation_jobs:
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"scene": scene, "map_id": recipe["id"], "stage": "EVALUATION", "error": str(exc)})
        result = {"status": "COMPLETE" if not failures else "INCOMPLETE", "phase": phase,
            "implementation_commit": commit, "completed": completed, "failures": failures,
            "mapping_workers": self.args.mapping_workers, "mapping_threads": self.args.mapping_threads,
            "evaluation_workers": self.args.evaluation_workers, "neural_workers_per_gpu": 1}
        atomic_write_json(self.root / "execution" / (phase + f"_{time.time_ns()}.json"), result)
        return result

    def bridge(self):
        scene = "scene0056_00"
        value = self.run_jobs([scene], [self.spec["map_variants"][0]], "bridge")
        if value["status"] != "COMPLETE":
            raise RuntimeError("full BB00 bridge is incomplete; inspect execution and worker logs")
        root = self.root / "readouts" / scene / "BB00_NATIVE"
        rows = read(root / "evaluation_rows.json")["rows"]
        historical = Path(read(self.spec["parent_paired_binding"])["output_root"])
        comparisons = {}
        for method, old_method in (("NATIVE_READOUT", "N0"), ("FC_EQ", "AW_E03_FC_FROZEN_A7"), ("D2", "PE_D2_GROUPED")):
            old = read(historical / "rows" / scene / old_method / "OFFICIAL_CURRENT_CLASS.json")
            new = next(row for row in rows if row["method"] == method and row["rank_mode"] == "OFFICIAL_CURRENT_CLASS")
            differences = {key: 100 * (new["metrics"][key] - old["metrics"][key])
                           for key in self.spec["evaluation"]["metrics"]}
            comparisons[method] = {"historical": old["metrics"], "new": new["metrics"], "delta_pp": differences,
                                   "within_0.05pp": all(abs(v) <= .05 for v in differences.values())}
        parent = self.bound()["scenes"][scene]
        metadata = read(self.root / "maps" / scene / "BB00_NATIVE/diagnostics/native_deferred_metadata.json")
        with Path(parent["parent_scene"]["native_features"]).open("rb") as stream:
            import pickle
            old_features = pickle.load(stream)
        with (root / "native_query/native_features.pkl").open("rb") as stream:
            new_features = pickle.load(stream)
        exact_features = list(old_features) == list(new_features) and all(
            np.array_equal(old_features[owner][key], new_features[owner][key])
            for owner in old_features for key in ("feat", "vis_area", "frame_id", "pose", "box_2d", "color"))
        nq = read(root / "native_query/native_query_receipt.json")
        result = {"status": "VERIFIED" if all(v["within_0.05pp"] for v in comparisons.values()) else "DIAGNOSIS_REQUIRED",
            "scene": scene, "comparisons": comparisons, "native_deferred_pickle_exact": exact_features,
            "deferred_request_attempts": len(nq["native_requests"]),
            "metadata_observations": sum(len(r["frame_id"]) for r in metadata["rows"]),
            "new_temperature_fit": False, "historical_import_eligible": False,
            "8_thread_fusion_parallel_variation": read(self.root / "prepare/native_trace/summary.json")["native_parallel_variation"]}
        atomic_write_json(self.root / "bridge_parity.json", result)
        if result["status"] == "DIAGNOSIS_REQUIRED":
            self.bridge_diagnosis()
        return result

    def bridge_diagnosis(self):
        path = self.root / "bridge_diagnosis.json"
        if path.is_file():
            return read(path)
        scene = "scene0056_00"
        root = self.root / "readouts" / scene / "BB00_NATIVE"
        parent_path = Path(self.bound()["scenes"][scene]["parent_capture"])
        new_path = self.root / "maps" / scene / "BB00_NATIVE/capture" / scene / "manifest.json"
        old, new = read(parent_path), read(new_path)
        frame_changes = []
        for a, b in zip(old["frames"], new["frames"]):
            frame_changes.append({"frame_id": a["frame_id"], "RGB_equal": a["rgb_sha256"] == b["rgb_sha256"],
                "depth_equal": a["depth_sha256"] == b["depth_sha256"], "pano_equal": a["panoptic_sha256"] == b["panoptic_sha256"],
                "owner_raycast_equal": a["global_owner_sha256"] == b["global_owner_sha256"],
                "native_selected_request_ids_equal": a["native_selected_request_ids"] == b["native_selected_request_ids"]})
        result = {"status": "ONE_TARGETED_DIAGNOSIS_RECORDED", "frames": frame_changes,
            "new_readout": read(root / "anchor/native_readout.json"), "bridge": read(self.root / "bridge_parity.json"),
            "refit": False, "comparative_claims_status": "BLOCKED_UNRESOLVED_BRIDGE_UNTIL_PRIMARY_REVIEW",
            "remaining_correctly_specified_arms_authorized": True}
        atomic_write_json(path, result)
        return result

    def cohort_pools(self, cohort, scenes, recipes):
        for recipe in recipes:
            rows = []
            for scene in scenes:
                path = self.root / "readouts" / scene / recipe["id"] / "evaluation_rows.json"
                if path.is_file():
                    rows.extend(read(path)["rows"])
            if set(row["scene"] for row in rows) != set(scenes):
                continue
            namespace = load_released_module(Path(self.spec["upstream_worktree"]) / "scripts/eval_utils.py")
            namespace["init"]("Replica" if cohort == "replica" else "Scannet200")
            for method in self.spec["semantics"]["readouts"]:
                for rank in ("OFFICIAL_CURRENT_CLASS", "FROZEN_N0"):
                    result = pool(self.root / "pools" / cohort / recipe["id"], rows, namespace, scenes, method, rank)
                    atomic_write_json(self.root / "pools" / cohort / recipe["id"] / method / (rank + ".json"), result)
        if any(recipe["id"] == "BB00_NATIVE" for recipe in recipes):
            for scene in scenes:
                for recipe in recipes:
                    if recipe["id"] != "BB00_NATIVE" and (self.root / "readouts" / scene / recipe["id"] / "raw_geometry_diagnostics.json").is_file():
                        compare_maps(self.bound(), scene, recipe["id"])

    def screen(self):
        value = self.run_jobs(self.spec["datasets"]["development"], self.spec["map_variants"], "screen")
        self.cohort_pools("development", self.spec["datasets"]["development"], self.spec["map_variants"])
        return value

    def compose(self):
        freeze = freeze_candidates(self.bound(), self.spec, implementation_commit(self.repo))
        if freeze["composition_recipe"] is None:
            return freeze
        value = self.run_jobs(self.spec["datasets"]["development"], [freeze["composition_recipe"]], "compose")
        self.cohort_pools("development", self.spec["datasets"]["development"], [self.spec["map_variants"][0], freeze["composition_recipe"]])
        return value

    def freeze(self):
        selection = freeze_transfer(self.bound(), self.spec, implementation_commit(self.repo))
        release = self.repo / self.spec["publication"]["repo_artifacts"]
        release.mkdir(parents=True, exist_ok=True)
        for leaf in ("candidate_freeze.json", "selection.json"):
            shutil.copyfile(self.root / leaf, release / leaf)
        relative = str(release.relative_to(self.repo))
        subprocess.run(["git", "add", "--", relative + "/candidate_freeze.json", relative + "/selection.json"], cwd=self.repo, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.repo).returncode:
            subprocess.run(["git", "commit", "-m", "research: freeze backbone development selection before Replica"], cwd=self.repo, check=True)
        commit = git_value(self.repo, "rev-parse", "HEAD")
        freeze_path = self.root / "transfer_freeze_commit.json"
        if not freeze_path.is_file():
            atomic_write_json(freeze_path, {"status": "COMMITTED_BEFORE_REPLICA", "commit": commit,
                "selection_identity": selection["identity"], "Replica_started": False})
        return selection

    def transfer(self):
        selection, freeze = read(self.root / "selection.json"), read(self.root / "transfer_freeze_commit.json")
        if freeze["selection_identity"] != selection["identity"]:
            raise ValueError("Replica selection differs from the committed freeze")
        for leaf in ("candidate_freeze.json", "selection.json"):
            committed = git_value(self.repo, "show", f'{freeze["commit"]}:{self.spec["publication"]["repo_artifacts"]}/{leaf}')
            if json.loads(committed) != read(self.root / leaf):
                raise ValueError("Replica configuration is not the committed selection")
        value = self.run_jobs(selection["replica_scene_order"], selection["replica_recipes"], "transfer")
        self.cohort_pools("replica", selection["replica_scene_order"], selection["replica_recipes"])
        return value

    def diagnose(self):
        for path in (self.root / "readouts").glob("*/*/receipt.json"):
            receipt = read(path)
            diagnose_map(self.bound(), self.spec, receipt["scene"], receipt)
        return self.bridge_diagnosis() if (self.root / "bridge_parity.json").is_file() and read(self.root / "bridge_parity.json")["status"] != "VERIFIED" else {"status": "DIAGNOSED_EXISTING_MAPS"}

    def report(self):
        from .reporting import render
        return render(self.bound(), self.spec)

    def publish(self):
        from .reporting import publish
        return publish(self.bound(), self.spec)

    def all(self):
        results = {}
        for phase in ("bind", "prepare", "bridge", "screen", "compose", "freeze", "transfer", "report", "publish"):
            try:
                results[phase] = getattr(self, phase)()
            except Exception as exc:
                results[phase] = {"status": "FAILED_OR_DEPENDENCY_BLOCKED", "phase": phase,
                    "failures": [{"stage": phase, "error": f"{type(exc).__name__}: {exc}"}],
                    "automatic_retry": False}
                atomic_write_json(self.root / "execution" / ("phase_failure_" + phase + f"_{time.time_ns()}.json"), results[phase])
            print(json.dumps({"phase": phase, "status": results[phase].get("status", "RECORDED")}), flush=True)
            if phase == "bind" and results[phase]["status"] == "FAILED_OR_DEPENDENCY_BLOCKED":
                break
        atomic_write_json(self.root / "execution" / (f"all_{time.time_ns()}.json"), results)
        return results
