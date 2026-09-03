#!/usr/bin/env python3
"""Audit local TESSE aggregation against one frozen Khronos source commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.khronos_metric_parity import (
    KhronosParityError,
    audit_metric_parity,
)

UPSTREAM_UTILS = Path("khronos_eval/plotting/utils.py")


def _git(checkout: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise KhronosParityError(f"upstream Git identity check failed: {detail}") from error


def _bind_upstream(checkout: Path, commit: str) -> tuple[Path, str]:
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise KhronosParityError(f"upstream Khronos checkout is missing: {checkout}")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise KhronosParityError("upstream commit must be a full lowercase Git SHA")
    resolved = _git(checkout, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise KhronosParityError(f"upstream commit resolved unexpectedly: {resolved}")
    relative = UPSTREAM_UTILS.as_posix()
    frozen_source = _git(checkout, "show", f"{commit}:{relative}")
    source_path = checkout / UPSTREAM_UTILS
    if not source_path.is_file():
        raise KhronosParityError(f"upstream Khronos utils.py is missing: {source_path}")
    working_source = source_path.read_bytes()
    if working_source != frozen_source:
        raise KhronosParityError(
            "working Khronos utils.py differs from the declared commit"
        )
    return source_path, hashlib.sha256(frozen_source).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--map-timestamps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source_path, source_sha256 = _bind_upstream(
            args.upstream_checkout, args.upstream_commit
        )
        result = audit_metric_parity(
            args.results_dir,
            upstream_utils=source_path,
            map_timestamps=args.map_timestamps,
            expected_upstream_sha256=source_sha256,
            atol=args.atol,
        )
        payload = result.to_dict()
        payload.update(
            {
                "protocol_id": "KHRONOS_POST_RELEASE_AGGREGATION_PARITY",
                "upstream": {
                    "repository": "MIT-SPARK/Khronos",
                    "commit": args.upstream_commit,
                    "utils_path": UPSTREAM_UTILS.as_posix(),
                    "utils_sha256": source_sha256,
                },
            }
        )
        _write_json_atomic(args.output, payload)
    except (KhronosParityError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": result.status}))
    return 0 if result.status == "PARITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
