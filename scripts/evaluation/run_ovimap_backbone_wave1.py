#!/usr/bin/env python3
"""Run the fixed backbone-wave1 exposed-development and Replica study."""

import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("bind", "prepare", "bridge", "diagnose", "screen", "compose", "freeze", "transfer", "report", "publish", "all"))
    parser.add_argument("--spec", type=Path, default=REPO / "docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json")
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--mapping-workers", type=int, default=2)
    parser.add_argument("--mapping-threads", type=int, default=8)
    parser.add_argument("--evaluation-workers", type=int, default=3)
    args = parser.parse_args()
    if not (1 <= args.mapping_workers <= 2 and 1 <= args.mapping_threads <= 8 and 1 <= args.evaluation_workers <= 3):
        parser.error("workers/threads exceed the fixed resource configuration")
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from static_ovmap.backbone_wave1.workflow import Study
    with exclusive_lock(args.output_root / ".coordinator.lock"):
        getattr(Study(args), args.phase)()


if __name__ == "__main__":
    main()
