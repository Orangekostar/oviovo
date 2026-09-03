#!/usr/bin/env python3
"""Audit Khronos exact attribution and prove official-output non-interference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.khronos_attribution import (
    AttributionContractError,
    audit_exact_attribution,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpatched-results", type=Path, required=True)
    parser.add_argument("--patched-results", type=Path, required=True)
    parser.add_argument("--object-sidecar", type=Path, required=True)
    parser.add_argument("--dynamic-sidecar", type=Path, required=True)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--input-map-sha256", required=True)
    parser.add_argument("--input-map-byte-count", type=int, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = audit_exact_attribution(
            unpatched_results=args.unpatched_results,
            patched_results=args.patched_results,
            object_sidecar=args.object_sidecar,
            dynamic_sidecar=args.dynamic_sidecar,
            input_map=args.input_map,
            expected_input_sha256=args.input_map_sha256,
            expected_input_byte_count=args.input_map_byte_count,
            source_checkout=args.source_checkout,
            source_commit=args.source_commit,
            patch_file=args.patch,
            patch_sha256=args.patch_sha256,
        )
        _write_json_atomic(args.output, payload)
    except (AttributionContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
