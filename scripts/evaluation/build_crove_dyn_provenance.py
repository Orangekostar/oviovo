#!/usr/bin/env python3
"""Build a hash-bound sidecar for official CROVE dynamic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.crove_dyn_provenance import publish_crove_dyn_provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-manifest", required=True, type=Path)
    parser.add_argument("--bridge-manifest", required=True, type=Path)
    parser.add_argument("--official-dynamic-csv", required=True, type=Path)
    parser.add_argument("--visualization-objects-csv", required=True, type=Path)
    parser.add_argument("--visualization-associations-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = publish_crove_dyn_provenance(
        composition_manifest=args.composition_manifest,
        bridge_manifest=args.bridge_manifest,
        official_dynamic_csv=args.official_dynamic_csv,
        visualization_objects_csv=args.visualization_objects_csv,
        visualization_associations_csv=args.visualization_associations_csv,
        output_root=args.output_root,
    )
    print(json.dumps({"dyn_provenance": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
