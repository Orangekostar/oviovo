#!/usr/bin/env python3
"""Run and aggregate the fixed observation-query multi-environment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_ovi_observation_query import run_evaluation


_DOMAINS = (
    "COMMON_INPUT_SUPPORT_V2",
    "FULL_GT_V2",
    "RAW_COMMON_INPUT_SUPPORT_V2",
    "RAW_FULL_GT_V2",
)
_PER_ENVIRONMENT_FIELDS = (
    "method_id",
    "checkpoint_id",
    "training_updates",
    "run_id",
    "pair_id",
    "environment_uuid",
    "split_role",
    "common_f1_50",
    "common_tp50",
    "common_fp50",
    "common_fn50",
    "common_raw_best_iou",
    "common_raw_ar50",
    "common_raw_ar25",
    "full_f1_50",
    "full_tp50",
    "full_fp50",
    "full_fn50",
    "full_raw_best_iou",
    "full_raw_ar50",
    "full_raw_ar25",
    "identity_recall",
    "status",
)
_AGGREGATE_FIELDS = (
    "method_id",
    "checkpoint_id",
    "training_updates",
    "environment_count",
    "valid_common_environment_count",
    "common_macro_f1_50",
    "common_micro_tp50",
    "common_micro_fp50",
    "common_micro_fn50",
    "common_micro_f1_50",
    "common_raw_best_iou_macro",
    "common_raw_ar50_macro",
    "common_raw_ar25_macro",
    "full_macro_f1_50",
    "full_micro_tp50",
    "full_micro_fp50",
    "full_micro_fn50",
    "full_micro_f1_50",
    "full_raw_best_iou_macro",
    "full_raw_ar50_macro",
    "full_raw_ar25_macro",
    "identity_recall_macro",
    "status",
)
_SELECTION_FIELDS = (
    "method_id",
    "checkpoint_id",
    "training_updates",
    "common_macro_f1_50",
    "common_raw_best_iou_macro",
    "selected",
    "selection_reason",
)


class MultiEnvironmentEvaluationError(ValueError):
    """Raised when a multi-environment evaluation contract is invalid."""


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MultiEnvironmentEvaluationError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, Mapping):
        raise MultiEnvironmentEvaluationError(f"{label} must be a mapping")
    return value


def _float(value: object, label: str, *, nullable: bool = False) -> float | None:
    if value in (None, ""):
        if nullable:
            return None
        raise MultiEnvironmentEvaluationError(f"{label} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MultiEnvironmentEvaluationError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise MultiEnvironmentEvaluationError(f"{label} is not finite")
    return result


def _count(value: object, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise MultiEnvironmentEvaluationError(f"{label} is not an integer") from error
    if result < 0:
        raise MultiEnvironmentEvaluationError(f"{label} is negative")
    return result


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else sum(valid) / len(valid)


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _text(row.get(field)) for field in fields})


def _manifest_runs(manifest: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if manifest.get("schema_version") != 1:
        raise MultiEnvironmentEvaluationError("manifest schema_version must be 1")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise MultiEnvironmentEvaluationError("manifest runs must be a non-empty list")
    required = {"config", "runtime", "method", "role", "pair_id", "run_id"}
    runs: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for value in raw_runs:
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise MultiEnvironmentEvaluationError("manifest run is incomplete")
        run_id = str(value["run_id"])
        if not run_id or run_id in seen:
            raise MultiEnvironmentEvaluationError("manifest run_id is empty or duplicated")
        seen.add(run_id)
        runs.append(value)
    return tuple(runs)


def _results_path(run: Mapping[str, object]) -> Path:
    runtime = _load_json(Path(str(run["runtime"])).absolute(), "runtime")
    cache_root = runtime.get("cache_root")
    if not isinstance(cache_root, str) or not cache_root:
        raise MultiEnvironmentEvaluationError("runtime cache_root is invalid")
    return Path(cache_root).absolute() / "evaluation_runs" / str(run["run_id"]) / "results.csv"


def _read_pair_result(run: Mapping[str, object]) -> dict[str, object]:
    path = _results_path(run)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise MultiEnvironmentEvaluationError(f"result is unreadable: {path}") from error
    selected = [row for row in rows if row.get("visit_id") == "1"]
    by_domain: dict[str, Mapping[str, str]] = {}
    for domain in _DOMAINS:
        matches = [row for row in selected if row.get("evaluation_domain") == domain]
        if len(matches) != 1:
            raise MultiEnvironmentEvaluationError(
                f"result requires one t1 {domain} row: {path}"
            )
        by_domain[domain] = matches[0]
    all_selected = tuple(by_domain.values())
    expected = {
        "run_id": str(run["run_id"]),
        "method_id": str(run["method"]),
        "pair_id": str(run["pair_id"]),
        "split_role": str(run["role"]),
    }
    for row in all_selected:
        if row.get("status") != "PASS":
            raise MultiEnvironmentEvaluationError(f"result row is not PASS: {path}")
        for field, value in expected.items():
            if row.get(field) != value:
                raise MultiEnvironmentEvaluationError(
                    f"result {field} differs from manifest: {path}"
                )
    identities = {
        (
            row.get("checkpoint_id"),
            row.get("training_updates"),
            row.get("environment_uuid"),
        )
        for row in all_selected
    }
    if len(identities) != 1:
        raise MultiEnvironmentEvaluationError(f"result identity varies by domain: {path}")
    common = by_domain["COMMON_INPUT_SUPPORT_V2"]
    full = by_domain["FULL_GT_V2"]
    raw_common = by_domain["RAW_COMMON_INPUT_SUPPORT_V2"]
    raw_full = by_domain["RAW_FULL_GT_V2"]
    checkpoint_id, updates, environment_uuid = next(iter(identities))
    if not checkpoint_id or not environment_uuid:
        raise MultiEnvironmentEvaluationError(f"result identity is incomplete: {path}")
    return {
        "method_id": str(run["method"]),
        "checkpoint_id": checkpoint_id,
        "training_updates": _count(updates, "training_updates"),
        "run_id": str(run["run_id"]),
        "pair_id": str(run["pair_id"]),
        "environment_uuid": environment_uuid,
        "split_role": str(run["role"]),
        "common_f1_50": _float(common.get("f1_50"), "common f1", nullable=True),
        "common_tp50": _count(common.get("tp50"), "common tp50"),
        "common_fp50": _count(common.get("fp50"), "common fp50"),
        "common_fn50": _count(common.get("fn50"), "common fn50"),
        "common_raw_best_iou": _float(raw_common.get("raw_best_iou_mean"), "common raw IoU", nullable=True),
        "common_raw_ar50": _float(raw_common.get("raw_ar50"), "common raw AR50", nullable=True),
        "common_raw_ar25": _float(raw_common.get("raw_ar25"), "common raw AR25", nullable=True),
        "full_f1_50": _float(full.get("f1_50"), "full f1", nullable=True),
        "full_tp50": _count(full.get("tp50"), "full tp50"),
        "full_fp50": _count(full.get("fp50"), "full fp50"),
        "full_fn50": _count(full.get("fn50"), "full fn50"),
        "full_raw_best_iou": _float(raw_full.get("raw_best_iou_mean"), "full raw IoU", nullable=True),
        "full_raw_ar50": _float(raw_full.get("raw_ar50"), "full raw AR50", nullable=True),
        "full_raw_ar25": _float(raw_full.get("raw_ar25"), "full raw AR25", nullable=True),
        "identity_recall": _float(common.get("identity_recall"), "identity recall", nullable=True),
        "status": "PASS",
    }


def aggregate_results(
    runs: Sequence[Mapping[str, object]], output_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_environment = [_read_pair_result(run) for run in runs]
    per_environment.sort(
        key=lambda row: (
            str(row["method_id"]),
            int(row["training_updates"]),
            str(row["environment_uuid"]),
        )
    )
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_environment:
        key = (
            str(row["method_id"]),
            str(row["checkpoint_id"]),
            int(row["training_updates"]),
        )
        groups[key].append(row)
    coverage_by_method: dict[str, set[str]] = {}
    for (method, _checkpoint, _updates), rows in sorted(groups.items()):
        environments = {str(row["environment_uuid"]) for row in rows}
        expected = coverage_by_method.setdefault(method, environments)
        if environments != expected:
            raise MultiEnvironmentEvaluationError(
                f"{method} checkpoints require identical environment coverage"
            )
    aggregate: list[dict[str, object]] = []
    for (method, checkpoint, updates), rows in sorted(groups.items()):
        environments = [str(row["environment_uuid"]) for row in rows]
        if len(environments) != len(set(environments)):
            raise MultiEnvironmentEvaluationError(
                "a checkpoint has duplicate environment results"
            )
        common_tp = sum(int(row["common_tp50"]) for row in rows)
        common_fp = sum(int(row["common_fp50"]) for row in rows)
        common_fn = sum(int(row["common_fn50"]) for row in rows)
        full_tp = sum(int(row["full_tp50"]) for row in rows)
        full_fp = sum(int(row["full_fp50"]) for row in rows)
        full_fn = sum(int(row["full_fn50"]) for row in rows)
        common_f1 = [row["common_f1_50"] for row in rows]
        aggregate.append(
            {
                "method_id": method,
                "checkpoint_id": checkpoint,
                "training_updates": updates,
                "environment_count": len(rows),
                "valid_common_environment_count": sum(value is not None for value in common_f1),
                "common_macro_f1_50": _mean(common_f1),
                "common_micro_tp50": common_tp,
                "common_micro_fp50": common_fp,
                "common_micro_fn50": common_fn,
                "common_micro_f1_50": _f1(common_tp, common_fp, common_fn),
                "common_raw_best_iou_macro": _mean([row["common_raw_best_iou"] for row in rows]),
                "common_raw_ar50_macro": _mean([row["common_raw_ar50"] for row in rows]),
                "common_raw_ar25_macro": _mean([row["common_raw_ar25"] for row in rows]),
                "full_macro_f1_50": _mean([row["full_f1_50"] for row in rows]),
                "full_micro_tp50": full_tp,
                "full_micro_fp50": full_fp,
                "full_micro_fn50": full_fn,
                "full_micro_f1_50": _f1(full_tp, full_fp, full_fn),
                "full_raw_best_iou_macro": _mean([row["full_raw_best_iou"] for row in rows]),
                "full_raw_ar50_macro": _mean([row["full_raw_ar50"] for row in rows]),
                "full_raw_ar25_macro": _mean([row["full_raw_ar25"] for row in rows]),
                "identity_recall_macro": _mean([row["identity_recall"] for row in rows]),
                "status": "PASS",
            }
        )
    selections: list[dict[str, object]] = []
    methods = sorted({str(row["method_id"]) for row in aggregate})
    for method in methods:
        candidates = [row for row in aggregate if row["method_id"] == method]
        selected = max(
            candidates,
            key=lambda row: (
                float(row["common_macro_f1_50"])
                if row["common_macro_f1_50"] is not None
                else -math.inf,
                float(row["common_raw_best_iou_macro"])
                if row["common_raw_best_iou_macro"] is not None
                else -math.inf,
                -int(row["training_updates"]),
            ),
        )
        for row in sorted(candidates, key=lambda value: int(value["training_updates"])):
            selections.append(
                {
                    "method_id": method,
                    "checkpoint_id": row["checkpoint_id"],
                    "training_updates": row["training_updates"],
                    "common_macro_f1_50": row["common_macro_f1_50"],
                    "common_raw_best_iou_macro": row["common_raw_best_iou_macro"],
                    "selected": row is selected,
                    "selection_reason": "MAX_COMMON_MACRO_THEN_RAW_THEN_EARLIER",
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "per_environment_metrics.csv", _PER_ENVIRONMENT_FIELDS, per_environment)
    _write_csv(output_root / "macro_micro_comparison.csv", _AGGREGATE_FIELDS, aggregate)
    _write_csv(output_root / "checkpoint_selection.csv", _SELECTION_FIELDS, selections)
    return aggregate, selections


def run_manifest(path: Path, output_root: Path, *, aggregate_only: bool) -> int:
    manifest = _load_json(path, "manifest")
    runs = _manifest_runs(manifest)
    if not aggregate_only:
        for run in runs:
            result = run_evaluation(
                config_path=str(run["config"]),
                runtime_path=str(run["runtime"]),
                method=str(run["method"]),
                checkpoint=(
                    None if run.get("checkpoint") in (None, "") else str(run["checkpoint"])
                ),
                role=str(run["role"]),
                run_id=str(run["run_id"]),
                pair_id=str(run["pair_id"]),
            )
            if result != 0:
                raise MultiEnvironmentEvaluationError(
                    f"evaluation failed for run_id={run['run_id']} with exit={result}"
                )
    aggregate_results(runs, output_root)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_manifest(
        Path(args.manifest).absolute(),
        Path(args.output_root).absolute(),
        aggregate_only=bool(args.aggregate_only),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MultiEnvironmentEvaluationError",
    "aggregate_results",
    "run_manifest",
]
