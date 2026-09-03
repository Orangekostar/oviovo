#!/usr/bin/env python3
"""Inspect Flat inputs or evaluate one causal current-state checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.datasets.panoptic_flat import FlatDatasetError, load_flat_dataset
from src.evaluation.json_contracts import loads_strict
from src.evaluation.panoptic_flat_current import (
    FlatGroundTruth,
    FlatPrediction,
    FlatProtocolError,
    evaluate_flat_current,
)

_GT_FIELDS = {"frame_index", "object_id", "present", "change_type", "change_frame"}
_PREDICTION_FIELDS = {
    "frame_index",
    "track_id",
    "matched_object_id",
    "stale_geometry_fp",
    "geometry_tp",
    "geometry_fp",
    "geometry_fn",
    "free_space_tp",
    "free_space_fp",
    "free_space_fn",
}


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


def _binding(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    source = path.resolve()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "byte_count": source.stat().st_size}


def _records(path: Path, *, prediction: bool) -> tuple[object, ...]:
    source = path.resolve()
    value = loads_strict(source.read_text(encoding="utf-8"), label=source.name)
    if not isinstance(value, list):
        raise FlatProtocolError(f"{source.name} must contain a list")
    expected = _PREDICTION_FIELDS if prediction else _GT_FIELDS
    records: list[object] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise FlatProtocolError(f"{source.name} contains an invalid record schema")
        records.append(FlatPrediction(**row) if prediction else FlatGroundTruth(**row))
    return tuple(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--dataset-root", type=Path, required=True)
    inspect.add_argument("--condition", required=True)
    inspect.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--evaluation-frame", type=int, required=True)
    evaluate.add_argument("--condition", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            dataset = load_flat_dataset(args.dataset_root, condition=args.condition)
            payload: dict[str, Any] = {
                "schema_version": 1,
                "status": "PASS",
                "protocol_id": "PANOPTIC_FLAT_TWO_RUN_INPUT_V1",
                "condition": dataset.condition.condition_id,
                "evidence_class": (
                    "ORACLE_DIAGNOSTIC" if dataset.condition.oracle else "NON_ORACLE"
                ),
                "oracle": dataset.condition.oracle,
                "causal_order": ["run1", "run2"],
                "frame_count": len(dataset.frames),
                "run_frame_counts": dict(Counter(frame.run_id for frame in dataset.frames)),
                "change_counts": dict(
                    Counter(change.change_type for change in dataset.changes)
                ),
                "source_bindings": [asdict(binding) for binding in dataset.source_bindings],
            }
        else:
            ground_truth = _records(args.ground_truth, prediction=False)
            predictions = _records(args.predictions, prediction=True)
            result = evaluate_flat_current(
                ground_truth,
                predictions=predictions,
                evaluation_frame=args.evaluation_frame,
                condition=args.condition,
            )
            payload = result.to_dict()
            payload["source_bindings"] = {
                "ground_truth": _binding(args.ground_truth),
                "predictions": _binding(args.predictions),
            }
        _atomic_json(args.output, payload)
    except (FlatDatasetError, FlatProtocolError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
