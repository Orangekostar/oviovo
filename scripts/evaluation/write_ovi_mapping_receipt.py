#!/usr/bin/env python3
"""Bind two completed native OVI visits into one D2 mapping receipt."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluation.prepare_ovi_rescene_dense_instance_pair import (
    DensePairPreparationError,
    _json_bytes,
    _mapping_receipt,
    _pair_record,
    _read_json,
    _write_or_verify,
)


def write_mapping_receipt(
    *,
    pair_id: str,
    selection_manifest: Path,
    native_manifests: tuple[Path, Path],
    materialized_manifests: tuple[Path, Path],
    output: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Validate both visit identities and publish the existing receipt schema."""

    selection_path = selection_manifest.resolve()
    _, selection = _read_json(selection_path, label="selection manifest")
    pair = _pair_record(selection, pair_id)
    receipt = _mapping_receipt(
        pair_id=pair_id,
        pair_record=pair,
        selection_manifest=selection_path,
        native_manifests=tuple(path.resolve() for path in native_manifests),
        materialized_manifests=tuple(
            path.resolve() for path in materialized_manifests
        ),
        repository_root=repository_root.resolve(),
    )
    _write_or_verify(output.resolve(), _json_bytes(receipt), label="mapping receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--native-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--materialized-manifests", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        result = write_mapping_receipt(
            pair_id=args.pair_id,
            selection_manifest=args.selection_manifest,
            native_manifests=tuple(args.native_manifests),
            materialized_manifests=tuple(args.materialized_manifests),
            output=args.output,
        )
    except (DensePairPreparationError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "pair_id": result["pair_id"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "write_mapping_receipt"]
