#!/usr/bin/env python3
"""Freeze the complete, development-only 3RScan T=2 selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.rscan_t2_dev import (
    RScanT2DevelopmentError,
    build_t2_development_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation-list", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--validation-database", type=Path, required=True)
    parser.add_argument("--train-database", type=Path, required=True)
    parser.add_argument("--sequence-database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_t2_development_manifest(
            metadata_path=args.metadata,
            validation_list_path=args.validation_list,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            validation_database_path=args.validation_database,
            train_database_path=args.train_database,
            sequence_database_path=args.sequence_database,
            checkpoint_path=args.checkpoint,
            checkpoint_provenance_path=args.checkpoint_provenance,
            count=args.count,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, RScanT2DevelopmentError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.absolute()),
                "selected_pair_count": manifest["selected_pair_count"],
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
