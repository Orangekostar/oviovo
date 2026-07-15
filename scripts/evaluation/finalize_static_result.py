#!/usr/bin/env python3
"""Finalize a tracked baseline result JSON from aggregate metrics and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.baselines.result_manifest import finalize_static_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    result = finalize_static_result(args.aggregate, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
