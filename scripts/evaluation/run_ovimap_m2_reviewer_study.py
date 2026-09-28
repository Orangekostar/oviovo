#!/usr/bin/env python3
"""Execute the source-bound M2 reviewer evidence study."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "docs/paper/static_ovmap/m2_reviewer_study_v1/PROTOCOL_SPEC.json")
    parser.add_argument("--source-transfer", type=Path, default=Path("/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001"))
    parser.add_argument("--phase", required=True, choices=("bind", "core", "query-controls", "diagnostics", "robustness", "fresh", "report", "publish", "all"))
    parser.add_argument("--scene")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--query-policy", choices=("Q_GAIN", "Q_COMBINE", "RV_Q_RANDOM"))
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="write clearly marked draft reports only")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    if args.phase == "query-controls":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from src.static_ovmap.composition_study.io import read_json
    from src.static_ovmap.m2_reviewer_study.binding import bind

    if args.phase in ("all", "publish") and (args.scene or args.query_policy or args.seed is not None or args.allow_partial):
        parser.error("all/publish require the complete prescribed study without leaf or draft overrides")
    if args.phase == "all":
        from src.static_ovmap.m2_reviewer_study.workflow import run_all

        print(json.dumps(run_all(spec=args.spec, source_transfer=args.source_transfer,
                                 output_root=args.output_root, gpu=args.gpu)), flush=True)
    elif args.phase == "publish":
        from src.static_ovmap.m2_reviewer_study.publication import publish

        print(json.dumps(publish(read_json(args.output_root / "source_binding.json"))), flush=True)
    elif args.phase == "bind":
        result = bind(args.spec, args.source_transfer, args.output_root)
        print(json.dumps({"status": result["status"], "scenes": len(result["scenes"])}), flush=True)
    elif args.phase == "report":
        from src.static_ovmap.composition_study.io import write_once
        from src.static_ovmap.m2_reviewer_study.reporting import (
            write_documents,
            write_report_tables,
        )

        binding = read_json(args.output_root / "source_binding.json")
        if (args.output_root / "query_controls/scene_stage.json").exists() and not (args.output_root / "query_controls/stage_complete.json").exists():
            from src.static_ovmap.m2_reviewer_study.budget_summary import (
                finish_query_stage,
            )

            finish_query_stage(binding)
        result, destination = write_report_tables(binding, allow_partial=args.allow_partial)
        write_documents(binding, result, destination)
        if result["status"] == "FINAL_EVIDENCE_READY":
            write_once(args.output_root / "final_report.json", {"status": result["status"], "identity": result["identity"],
                                                               "path": str(destination), "binding": binding["identity"]})
        print(json.dumps({"status": result["status"], "path": str(destination)}), flush=True)
    elif args.phase == "fresh":
        from src.static_ovmap.m2_reviewer_study.fresh_execution import run_fresh

        binding = read_json(args.output_root / "source_binding.json")
        result = run_fresh(binding, gpu=args.gpu)
        print(json.dumps({"status": result["status"], "details": str(args.output_root / "fresh/status.json")}), flush=True)
    elif args.phase == "robustness":
        from src.static_ovmap.m2_reviewer_study.aggregates import recover_aggregates
        from src.static_ovmap.m2_reviewer_study.robustness import (
            evaluate_robustness_scene,
        )

        binding = read_json(args.output_root / "source_binding.json")
        scenes = [args.scene] if args.scene else [s for s, r in binding["scenes"].items() if r["dataset"] == "Replica"]
        if any(s not in binding["scenes"] or binding["scenes"][s]["dataset"] != "Replica" for s in scenes):
            raise ValueError("robustness accepts bound Replica scenes only")
        config = read_json(binding["scenes"][scenes[0]]["config"])
        for model, python in (("native", config["runtime"]["mapping_python"]), ("siglip2", config["runtime"]["semantic_python"])):
            subprocess.run([python, "-m", "src.static_ovmap.m2_reviewer_study.text_spaces",
                            "--binding", str(args.output_root / "source_binding.json"), "--dataset", "Replica", "--model", model],
                           cwd=ROOT, check=True)
        for scene in scenes:
            recover_aggregates(binding, scene)
            result = evaluate_robustness_scene(binding, scene)
            print(json.dumps({"status": result["status"], "scene": scene, "methods": result["methods"]}), flush=True)
    elif args.phase == "diagnostics":
        from src.static_ovmap.m2_reviewer_study.attribution import attribute_scene
        from src.static_ovmap.m2_reviewer_study.diagnostics import (
            core_macro_bootstrap,
            diagnose_scene,
        )
        from src.static_ovmap.m2_reviewer_study.mechanisms import mechanism_scene

        binding = read_json(args.output_root / "source_binding.json")
        scenes = [args.scene] if args.scene else list(binding["scenes"])
        for scene in scenes:
            if scene not in binding["scenes"]:
                raise ValueError("scene outside study binding")
            result = diagnose_scene(binding, scene)
            attribute_scene(binding, scene)
            mechanism_scene(binding, scene)
            print(json.dumps({"status": "SCENE_DIAGNOSTICS_COMPLETE", "scene": scene,
                              "owners": result["owners"], "identifiable": result["identifiable"]}), flush=True)
        if not args.scene:
            core_macro_bootstrap(binding)
    elif args.phase == "query-controls":
        from src.static_ovmap.m2_reviewer_study.query_jobs import (
            authorize_query,
            run_query_job,
        )

        binding = read_json(args.output_root / "source_binding.json")
        if not args.scene:
            from src.static_ovmap.m2_reviewer_study.budget_controls import (
                run_query_controls,
            )

            if args.query_policy or args.seed is not None or args.budget != 200:
                parser.error("policy/seed/budget overrides require a leaf --scene")
            result = run_query_controls(binding, gpu=args.gpu)
            from src.static_ovmap.m2_reviewer_study.budget_summary import (
                finish_query_stage,
            )

            result = finish_query_stage(binding)
            print(json.dumps(result), flush=True)
            return
        policies = [(args.query_policy, args.seed)] if args.query_policy else [
            ("Q_GAIN", None), ("Q_COMBINE", None),
            ("RV_Q_RANDOM", 17), ("RV_Q_RANDOM", 23), ("RV_Q_RANDOM", 41)]
        for policy, seed in policies:
            if not args.query_policy:
                command = [sys.executable, str(Path(__file__).resolve()), "--phase", "query-controls",
                           "--output-root", str(args.output_root), "--scene", args.scene,
                           "--query-policy", policy, "--budget", str(args.budget), "--gpu", args.gpu, "--resume"]
                if seed is not None:
                    command += ["--seed", str(seed)]
                subprocess.run(command, check=True)
                continue
            method = authorize_query(binding, args.scene, policy, args.budget, seed)
            lock_root = args.output_root / "query_job_locks"
            lock_root.mkdir(parents=True, exist_ok=True)
            # Policies for one scene share operation-cache writes. Keep their
            # initial publication serial while allowing different scenes on
            # different GPUs; per-job locking alone does not protect overlap.
            with (lock_root / f"scene_{args.scene}.lock").open("a") as scene_handle:
                fcntl.flock(scene_handle, fcntl.LOCK_EX)
                with (lock_root / f"{args.scene}_{method}.lock").open("a") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    receipt = run_query_job(binding, args.scene, policy, budget=args.budget, seed=seed, gpu=args.gpu)
            print(json.dumps({"status": "QUERY_ACQUISITION_COMPLETE", "scene": args.scene,
                              "method": receipt["method_id"], "logical": receipt["logical"]}), flush=True)
    else:
        from src.static_ovmap.m2_reviewer_study.core import evaluate_core_scene
        from src.static_ovmap.m2_reviewer_study.core_pipeline import run_core

        binding = read_json(args.output_root / "source_binding.json")
        if not args.scene:
            print(json.dumps(run_core(binding)), flush=True)
            return
        scenes = [args.scene] if args.scene else list(binding["scenes"])
        for scene in scenes:
            if scene not in binding["scenes"]:
                raise ValueError("scene outside study binding")
            rows = evaluate_core_scene(binding, scene)
            print(json.dumps({"status": "SCENE_CORE_COMPLETE", "scene": scene, "rank_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
