#!/usr/bin/env python3
"""Evaluate the source-bound TESSE current-diagonal diagnostic."""

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

from src.evaluation.tesse_current_slice import (
    CurrentSliceError,
    audit_current_diagonal,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--map-timestamps", type=Path)
    parser.add_argument("--object-sidecar", type=Path)
    parser.add_argument("--dynamic-sidecar", type=Path)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = audit_current_diagonal(
            args.results_dir,
            map_timestamps=args.map_timestamps,
            object_sidecar=args.object_sidecar,
            dynamic_sidecar=args.dynamic_sidecar,
        )
        payload = result.to_dict()
        payload["method_id"] = args.method_id
        _atomic_json(args.output, payload)
    except (CurrentSliceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
