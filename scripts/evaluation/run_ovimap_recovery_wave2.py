"""Execute the fixed recovery-wave2 task phases and complete authorized DAG."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from static_ovmap.recovery_wave2.workflow import PHASES, Workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=(*PHASES, "all"), required=True)
    parser.add_argument("--spec", default=str(REPO / "docs/paper/static_ovmap/recovery_wave2_v1/PROTOCOL_SPEC.json"))
    parser.add_argument("--parent-root", default="/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001")
    parser.add_argument("--output-root", default="/mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--mapping-workers", type=int, default=2)
    parser.add_argument("--mapping-threads", type=int, default=8)
    parser.add_argument("--evaluation-workers", type=int, default=3)
    parser.add_argument("--native-build")
    parser.add_argument("--path-map")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    Workflow(args).run(args.phase)


if __name__ == "__main__":
    main()
