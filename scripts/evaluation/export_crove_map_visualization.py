#!/usr/bin/env python3
"""Export an explicit RGB, instance, semantic, or dynamic CROVE PLY view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.crove_map_visualization import (
    export_labeled_ply_visualization,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("rgb", "instance", "semantic", "dynamic"), required=True
    )
    parser.add_argument("--rgb-provenance", choices=("raw_tsdf",))
    parser.add_argument("--semantic-palette", type=Path)
    parser.add_argument("--dynamic-states", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_labeled_ply_visualization(
        args.source,
        args.output,
        mode=args.mode,
        rgb_provenance=args.rgb_provenance,
        semantic_palette=args.semantic_palette,
        dynamic_states=args.dynamic_states,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
