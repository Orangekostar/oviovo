#!/usr/bin/env python3
"""Freeze or evaluate the causal 3RScan temporal pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.datasets.rscan import (
    RScanDatasetError,
    build_selection_manifest,
)
from src.evaluation.json_contracts import loads_strict
from src.evaluation.rscan_temporal import (
    RScanProtocolError,
    TemporalGroundTruth,
    TemporalPrediction,
    evaluate_temporal_identity,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _load_records(path: Path, *, prediction: bool) -> tuple[object, ...]:
    source = path.resolve()
    value = loads_strict(source.read_text(encoding="utf-8"), label=source.name)
    if not isinstance(value, list):
        raise RScanProtocolError(f"{source.name} must contain a list")
    rows: list[object] = []
    for record in value:
        if not isinstance(record, Mapping):
            raise RScanProtocolError(f"{source.name} contains a malformed record")
        if prediction:
            rows.append(
                TemporalPrediction(
                    session_index=record.get("session_index"),
                    track_id=record.get("track_id"),
                    matched_instance_id=record.get("matched_instance_id"),
                )
            )
        else:
            rows.append(
                TemporalGroundTruth(
                    session_index=record.get("session_index"),
                    instance_id=record.get("instance_id"),
                    present=record.get("present"),
                    change_type=record.get("change_type"),
                )
            )
    return tuple(rows)


def _binding(path: Path) -> dict[str, object]:
    data = path.resolve().read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--metadata", type=Path, required=True)
    freeze.add_argument("--validation-list", type=Path, required=True)
    freeze.add_argument("--asset-root", type=Path, required=True)
    freeze.add_argument("--count", type=int, default=10)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--evaluation-session", type=int, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "freeze":
            payload = build_selection_manifest(
                args.metadata,
                args.validation_list,
                asset_root=args.asset_root,
                count=args.count,
                output=args.output,
            )
        else:
            gt = _load_records(args.ground_truth, prediction=False)
            predictions = _load_records(args.predictions, prediction=True)
            result = evaluate_temporal_identity(
                gt,
                predictions=predictions,
                evaluation_session=args.evaluation_session,
            )
            payload = result.to_dict()
            payload["source_bindings"] = {
                "ground_truth": _binding(args.ground_truth),
                "predictions": _binding(args.predictions),
            }
            _atomic_json(args.output, payload)
    except (RScanDatasetError, RScanProtocolError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
