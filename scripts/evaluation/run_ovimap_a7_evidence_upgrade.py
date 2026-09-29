#!/usr/bin/env python3
"""Execute the fixed A7 evidence-upgrade wave with resumable family phases."""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "docs/paper/static_ovmap/a7_evidence_upgrade_wave1/PROTOCOL_SPEC.json")
    parser.add_argument("--reviewer-binding", type=Path, default=Path("/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001/source_binding.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001"))
    parser.add_argument("--phase", required=True, choices=("bind", "prepare", "e01", "e02", "c0", "e03", "e04", "compose", "evaluate", "report", "publish", "all"))
    parser.add_argument("--split", choices=("cal", "replica", "all"), default="all")
    parser.add_argument("--scene")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--sam2-python", type=Path, default=Path("/home/ww/miniconda3/envs/oviovo-ovo-official/bin/python"))
    parser.add_argument("--region-python", type=Path, default=Path("/home/ww/miniconda3/envs/oviovo-radseg/bin/python"))
    args = parser.parse_args()
    if args.scene and (args.split == "all" or args.phase in {"all", "bind", "prepare", "report", "publish"}):
        parser.error("--scene is only a controlled single-split leaf/resume option")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    from src.static_ovmap.a7_evidence_upgrade.binding import bind
    from src.static_ovmap.a7_evidence_upgrade.selection import lock_transfer
    from src.static_ovmap.a7_evidence_upgrade.workflow import (
        FAMILIES,
        composition,
        evaluate,
        family,
        workers,
    )
    from src.static_ovmap.composition_study.io import read_json

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".coordinator.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output_root / "binding.json"
        if path.exists() and args.phase != "bind":
            if not args.resume:
                parser.error("existing study requires --resume")
            binding = read_json(path)
            if Path(binding["spec"]).resolve() != args.spec.resolve() or Path(binding["reviewer_binding"]).resolve() != args.reviewer_binding.resolve():
                parser.error("resume spec/reviewer binding differs")
        else:
            binding = bind(args.spec, args.reviewer_binding, args.output_root)
        if args.phase == "bind":
            print(json.dumps({"status": binding["status"], "identity": binding["identity"]}), flush=True)
            return
        environments = workers(binding, args.sam2_python, args.region_python)
        gpus = args.gpus.split(",")
        splits = ("cal", "replica") if args.split == "all" else (args.split,)
        results = []
        if args.phase in {"prepare", "all"}:
            from src.static_ovmap.a7_evidence_upgrade.prerequisites import prepare

            results.append(prepare(binding, environments, gpus))
        if args.phase in FAMILIES:
            for split in splits:
                results.append(family(binding, args.phase, split, environments, gpus, scene=args.scene))
        elif args.phase in {"compose", "evaluate"}:
            for split in splits:
                results.append((composition if args.phase == "compose" else evaluate)(binding, split, scene=args.scene))
        elif args.phase == "all":
            for split in splits:
                if split == "cal" and (args.output_root / "transfer_lock.json").exists():
                    lock_transfer(binding)
                    continue
                for name in FAMILIES:
                    results.append(family(binding, name, split, environments, gpus))
                results.append(composition(binding, split))
            from src.static_ovmap.a7_evidence_upgrade.publication import publish
            from src.static_ovmap.a7_evidence_upgrade.reporting import report

            results += [report(binding), publish(binding)]
        elif args.phase == "report":
            from src.static_ovmap.a7_evidence_upgrade.reporting import report

            results.append(report(binding))
        elif args.phase == "publish":
            from src.static_ovmap.a7_evidence_upgrade.publication import publish

            results.append(publish(binding))
        # Full provenance stays in immutable receipts, not multi-megabyte logs.
        print(json.dumps([{k: v for k, v in result.items() if k not in {"inputs", "outputs"}}
                          for result in results]), flush=True)


if __name__ == "__main__":
    main()
