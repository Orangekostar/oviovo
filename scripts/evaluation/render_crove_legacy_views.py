#!/usr/bin/env python3
"""Render the unchanged legacy CROVE RGB, instance, and semantic views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.render_crove_fine_current_views import (  # noqa: E402
    render_legacy_views,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--maximum-points", type=int, default=200_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_legacy_views(
        {
            "rgb": args.rgb,
            "instance": args.instance,
            "semantic": args.semantic,
        },
        args.output,
        scene=args.scene,
        maximum_points=args.maximum_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
