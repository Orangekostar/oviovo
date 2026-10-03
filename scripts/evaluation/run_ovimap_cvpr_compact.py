#!/usr/bin/env python3
"""Execute the fixed CVPR compact-table protocol from its isolated worktree."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from static_ovmap.cvpr_compact.workflow import PHASES, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu")
    parser.add_argument("--path-map", action="append", default=[])
    args = parser.parse_args()
    result = run_pipeline(args.spec.resolve(), REPO_ROOT, phase=args.phase, resume=args.resume,
                          gpu=args.gpu, path_maps=args.path_map)
    print(result["phase"], result["status"], flush=True)


if __name__ == "__main__":
    main()
