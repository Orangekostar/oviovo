#!/usr/bin/env python3
"""Build aligned evaluator-only GT sidecars for RSCAN_T2_DEV_V1."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.rscan_gt_instances import (
    RScanGroundTruthError,
    build_ground_truth_sidecars,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_ground_truth_sidecars(args.selection_manifest, args.output)
    except (OSError, TypeError, ValueError, RScanGroundTruthError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.absolute() / "summary.json"),
                "pair_count": result["pair_count"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
