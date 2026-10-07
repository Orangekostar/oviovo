#!/usr/bin/env python3
"""Controller for the fixed evidence exploration study."""

import argparse
import json
import os
import sys
from pathlib import Path

# Set before importing NumPy/PyTorch/scorer modules, including forked CPU workers.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = str(min(4, max(1, int(os.environ.get(name, "4")))))
repository = Path(__file__).resolve().parents[2]
for path in (repository, repository/"src"):
    sys.path.insert(0, str(path))
os.environ["PYTHONPATH"] = os.pathsep.join([str(repository/"src"), str(repository), os.environ.get("PYTHONPATH", "")])

from static_ovmap.evidence_exploration.binding import bind, load_binding


PHASES = ("bind", "diagnose", "assets", "prepare", "encode", "predict", "evaluate", "select", "time", "tables", "publish")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=Path("configs/static_ovmap/evidence_exploration_v1.json"))
    p.add_argument("--parent-runtime-root", type=Path, default=Path("/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001"))
    p.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001"))
    p.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    p.add_argument("--gpu")
    p.add_argument("--path-map", type=Path)
    p.add_argument("--parent-reference", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    phases = PHASES if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "bind":
            bind(args.spec, args.parent_runtime_root, args.output_root, gpu=args.gpu,
                 path_map={} if args.path_map is None else json.loads(args.path_map.read_text()), parent_reference=args.parent_reference)
        else:
            binding = load_binding(args.output_root)
            if phase == "assets":
                from static_ovmap.evidence_exploration.assets import validate_assets
                validate_assets(binding, args.output_root)
            else:
                from static_ovmap.evidence_exploration.workflow import run_phase
                run_phase(binding, args.output_root, phase, resume=args.resume)


if __name__ == "__main__":
    main()
