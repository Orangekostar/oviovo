#!/usr/bin/env python3
"""Audit the pre-registered K0/K1/C0/C1 TESSE input matrix."""

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

from src.evaluation.benchmark_alignment import classify_frontend_gap
from src.evaluation.json_contracts import loads_strict


class FairnessAuditError(ValueError):
    """Raised when a supplied condition manifest is malformed."""


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


def _load_condition(path: Path | None, condition_id: str) -> dict[str, Any]:
    if path is None:
        return {
            "condition_id": condition_id,
            "status": "INCONCLUSIVE_MISSING_CONDITION",
            "metrics": None,
            "source_binding": None,
            "evidence_bindings": [],
        }
    source = path.resolve()
    if source.is_symlink() or not source.is_file():
        raise FairnessAuditError(f"{condition_id} manifest is missing or indirect")
    try:
        value = loads_strict(source.read_text(encoding="utf-8"), label=condition_id)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FairnessAuditError(f"{condition_id} manifest is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("condition_id") != condition_id:
        raise FairnessAuditError(f"{condition_id} manifest identity mismatch")
    status = value.get("status")
    metrics = value.get("metrics")
    if not isinstance(status, str):
        raise FairnessAuditError(f"{condition_id} status is missing")
    if status == "PASS" and not isinstance(metrics, Mapping):
        raise FairnessAuditError(f"{condition_id} PASS manifest has no metrics")
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list) or (status == "PASS" and not evidence):
        raise FairnessAuditError(f"{condition_id} has no source evidence")
    evidence_bindings: list[dict[str, object]] = []
    for index, record in enumerate(evidence):
        if not isinstance(record, Mapping):
            raise FairnessAuditError(f"{condition_id} evidence is malformed")
        evidence_path = record.get("path")
        if not isinstance(evidence_path, str) or not evidence_path:
            raise FairnessAuditError(f"{condition_id} evidence path is missing")
        source_evidence = Path(evidence_path).resolve()
        if source_evidence.is_symlink() or not source_evidence.is_file():
            raise FairnessAuditError(f"{condition_id} evidence is missing or indirect")
        digest = hashlib.sha256(source_evidence.read_bytes()).hexdigest()
        byte_count = source_evidence.stat().st_size
        if digest != record.get("sha256") or byte_count != record.get("byte_count"):
            raise FairnessAuditError(f"{condition_id} evidence binding mismatch")
        evidence_bindings.append(
            {"ordinal": index, "sha256": digest, "byte_count": byte_count}
        )
    binding = {
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "byte_count": source.stat().st_size,
    }
    return {
        "condition_id": condition_id,
        "status": status,
        "metrics": dict(metrics) if isinstance(metrics, Mapping) else None,
        "source_binding": binding,
        "evidence_bindings": evidence_bindings,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in ("k0", "k1", "c0", "c1"):
        parser.add_argument(f"--{condition}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        conditions = {
            name.upper(): _load_condition(getattr(args, name), name.upper())
            for name in ("k0", "k1", "c0", "c1")
        }
        available = lambda name: (
            conditions[name]["metrics"]
            if conditions[name]["status"] == "PASS"
            else None
        )
        diagnosis = classify_frontend_gap(
            available("K0"), available("C0"), available("C1")
        )
        payload = {
            "schema_version": 1,
            "protocol_id": "TESSE_INPUT_FAIRNESS_K0_K1_C0_C1",
            "status": diagnosis.status,
            "conditions": conditions,
            "diagnosis": diagnosis.to_dict(),
            "ranking_eligible": False,
        }
        _atomic_json(args.output, payload)
    except (FairnessAuditError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
