#!/usr/bin/env python3
"""Publish hash-bound Apartment authority and ghost attribution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.crove_failure_attribution import (
    analyze_crove_tesse_failures,
    render_attribution_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composed-run-manifest", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = analyze_crove_tesse_failures(
        composed_run_manifest=args.composed_run_manifest,
        source_run_manifest=args.source_run_manifest,
        anchor_manifest=args.anchor_manifest,
        schedule=args.schedule,
        target_manifest=args.target_manifest,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        json_path = temporary / "attribution.json"
        markdown_path = temporary / "attribution.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_attribution_markdown(report), encoding="utf-8")
        for path in (json_path, markdown_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
