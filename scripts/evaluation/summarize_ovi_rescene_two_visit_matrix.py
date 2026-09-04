#!/usr/bin/env python3
"""Reopen a two-visit matrix run and publish a small Pareto summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.evaluation.run_ovi_rescene_two_visit_matrix import (
    BLOCKED_STATUS,
    MATRIX_ID,
    _validate_metric_receipt,
    load_and_validate_matrix,
)


class SummaryError(ValueError):
    """Raised when a matrix result graph cannot be verified."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.resolve(strict=True) != absolute or not absolute.is_file():
        raise SummaryError(f"artifact must be a direct regular file: {absolute}")
    data = absolute.read_bytes()
    name = str(absolute)
    if relative_to is not None:
        try:
            name = absolute.relative_to(relative_to).as_posix()
        except ValueError as error:
            raise SummaryError("artifact escapes matrix run root") from error
    return {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"{label} must be valid JSON") from error
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must contain a JSON object")
    return value


def _bound_artifact(
    record: object, *, root: Path, label: str
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise SummaryError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        raise SummaryError(f"{label} binding path is invalid")
    path = root / raw_path
    observed = _record(path, relative_to=root)
    if observed != dict(record):
        raise SummaryError(f"{label} artifact binding mismatch")
    return path, observed


def _gate(observed: float, threshold: float, *, relation: str) -> dict[str, object]:
    if relation == "at_most":
        passed = observed <= threshold
    elif relation == "at_least":
        passed = observed >= threshold
    else:
        raise AssertionError("unknown gate relation")
    return {
        "observed": observed,
        "threshold": threshold,
        "relation": relation,
        "passed": passed,
    }


def summarize_matrix(
    *, matrix_path: Path, matrix_summary_path: Path, output_path: Path
) -> Path:
    """Validate the complete artifact graph and write one compact result receipt."""

    matrix_path = Path(matrix_path).absolute()
    matrix_summary_path = Path(matrix_summary_path).absolute()
    output_path = Path(output_path).absolute()
    matrix = load_and_validate_matrix(matrix_path)
    summary = _load_json(matrix_summary_path, label="matrix summary")
    run_root = matrix_summary_path.parent
    if (
        set(summary)
        != {
            "schema_version",
            "matrix_id",
            "status",
            "scene",
            "source_commit",
            "matrix",
            "variants",
        }
        or summary.get("schema_version") != 1
        or summary.get("matrix_id") != MATRIX_ID
        or summary.get("status")
        != "MATRIX_EXECUTION_COMPLETE_WITH_BLOCKED_VARIANTS"
        or summary.get("scene") != "apartment"
        or summary.get("matrix") != _record(matrix_path)
    ):
        raise SummaryError("matrix summary identity is invalid")
    source_commit = summary.get("source_commit")
    rows = summary.get("variants")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise SummaryError("matrix source commit is invalid")
    if not isinstance(rows, list) or len(rows) != len(matrix["rows"]):
        raise SummaryError("matrix variant summary is incomplete")

    variants: list[dict[str, object]] = []
    measured: dict[str, dict[str, Any]] = {}
    matrix_sha256 = summary["matrix"]["sha256"]
    for declared, row in zip(rows, matrix["rows"], strict=True):
        variant_id = row["id"]
        if (
            not isinstance(declared, Mapping)
            or set(declared) != {"variant_id", "run_id", "status", "receipt"}
            or declared.get("variant_id") != variant_id
            or not isinstance(declared.get("run_id"), str)
        ):
            raise SummaryError(f"{variant_id} matrix summary row is invalid")
        receipt_path, receipt_record = _bound_artifact(
            declared["receipt"], root=run_root, label=f"{variant_id} receipt"
        )
        receipt = _load_json(receipt_path, label=f"{variant_id} receipt")
        expected_status = "PASS" if variant_id in {"B0", "B1", "B2", "B3", "B4"} else BLOCKED_STATUS
        if declared.get("status") != expected_status or receipt.get("status") != expected_status:
            raise SummaryError(f"{variant_id} status is invalid")
        if (
            receipt.get("variant_id") != variant_id
            or receipt.get("run_id") != declared["run_id"]
            or receipt.get("source_commit") != source_commit
            or receipt.get("matrix_sha256") != matrix_sha256
        ):
            raise SummaryError(f"{variant_id} receipt identity is invalid")
        if expected_status == BLOCKED_STATUS:
            if receipt.get("ranking_eligible") is not False or receipt.get("metrics") is not None:
                raise SummaryError(f"{variant_id} blocked receipt is invalid")
            variants.append(
                {
                    "variant_id": variant_id,
                    "name": row["name"],
                    "status": BLOCKED_STATUS,
                    "metrics": None,
                    "unavailable_reason": BLOCKED_STATUS,
                    "receipt": receipt_record,
                }
            )
            continue
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "stdout_log",
            "stderr_log",
            "output_artifact",
            "metric_receipt",
            "command",
        }:
            raise SummaryError(f"{variant_id} artifact inventory is invalid")
        artifact_records: dict[str, dict[str, object]] = {}
        artifact_paths: dict[str, Path] = {}
        for name, record in artifacts.items():
            path, observed = _bound_artifact(
                record, root=run_root, label=f"{variant_id} {name}"
            )
            artifact_paths[name] = path
            artifact_records[name] = observed
        _validate_metric_receipt(
            artifact_paths["metric_receipt"],
            variant_id=variant_id,
            run_id=declared["run_id"],
            metric_contract=matrix["metric_contract"],
        )
        metric_payload = _load_json(
            artifact_paths["metric_receipt"], label=f"{variant_id} metrics"
        )
        measured[variant_id] = metric_payload
        variants.append(
            {
                "variant_id": variant_id,
                "name": row["name"],
                "status": "PASS",
                "metrics": metric_payload["metric_groups"],
                "unavailable": metric_payload["unavailable"],
                "receipt": receipt_record,
                "artifacts": artifact_records,
            }
        )

    b2_ghost = float(measured["B2"]["metric_groups"]["current_state"]["ghost"])
    b2_recall = float(
        measured["B2"]["metric_groups"]["geometry"][
            "t1_unobserved_region_recall"
        ]
    )
    ghost_delta = float(matrix["success_gates"]["ghost_delta_over_b2_max"])
    recall_gain = float(
        matrix["success_gates"]["minimum_unobserved_recall_gain_over_b2"]
    )
    gates: dict[str, object] = {
        "b2_practical_ghost": _gate(
            b2_ghost,
            float(matrix["success_gates"]["practical_ghost_max"]),
            relation="at_most",
        )
    }
    for variant_id in ("B3", "B4"):
        key = variant_id.lower()
        ghost = float(
            measured[variant_id]["metric_groups"]["current_state"]["ghost"]
        )
        recall = float(
            measured[variant_id]["metric_groups"]["geometry"][
                "t1_unobserved_region_recall"
            ]
        )
        gates[f"{key}_ghost_relative_to_b2"] = _gate(
            ghost, b2_ghost + ghost_delta, relation="at_most"
        )
        gates[f"{key}_unobserved_recall_gain_over_b2"] = _gate(
            recall - b2_recall, recall_gain, relation="at_least"
        )
    gates["full_method"] = {
        "passed": None,
        "reason": "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
    }

    payload = {
        "schema_version": 1,
        "status": "DETERMINISTIC_INFRASTRUCTURE_COMPLETE",
        "scene": "apartment",
        "source_commit": source_commit,
        "matrix": summary["matrix"],
        "matrix_execution": _record(matrix_summary_path),
        "variants": variants,
        "gates": gates,
        "rescene_verdict": "RESCENE_BLOCKED_EXTERNAL_ASSET",
    }
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"summary output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/evaluation/ovi_rescene_two_visit_matrix.json"),
    )
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = summarize_matrix(
        matrix_path=args.matrix,
        matrix_summary_path=args.matrix_summary,
        output_path=args.output,
    )
    print(output)
    return 0


__all__ = ["SummaryError", "main", "summarize_matrix"]


if __name__ == "__main__":
    raise SystemExit(main())
