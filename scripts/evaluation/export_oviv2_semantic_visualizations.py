#!/usr/bin/env python3
"""Export viewer-compatible semantic meshes from verified OVIV2 Replica runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.semantic_visualization import export_semantic_visualizations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--palette",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/replica41_semantic_palette.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes")
    args = parser.parse_args(argv)
    result = export_semantic_visualizations(
        args.batch_root,
        args.manifest,
        args.palette,
        args.output,
        scenes=args.scenes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
