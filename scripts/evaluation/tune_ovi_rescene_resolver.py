#!/usr/bin/env python3
"""Tune the fragment-union resolver on the frozen 3RScan development pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_3rscan_t2_matrix import (
    _load_config as load_matrix_config,
)
from scripts.evaluation.run_3rscan_t2_matrix import (
    _validated_directory as validated_directory,
)
from scripts.evaluation.run_3rscan_t2_matrix import load_native_pair_artifact
from src.evaluation.rscan_association_metrics import (
    evaluate_pair_relations,
    relation_outcome_rows,
)
from src.evaluation.rscan_gt_instances import load_pair_ground_truth
from src.evaluation.rscan_method_views import (
    build_method_pair_view_from_manifest,
    native_query_evidence,
    resolve_native_queries_fragment_union,
)
from src.oviv2.query_instance_projection import ProjectionConfig

_RECORD_KEYS = {"path", "sha256", "byte_count"}
_CONFIG_KEYS = {
    "schema_version",
    "config_id",
    "decision_result",
    "matrix_result",
    "candidate_thresholds",
    "objective",
    "output",
}
_COUNT_KEYS = (
    "paired_prediction_edges",
    "true_positive_edges",
    "false_positive_edges",
    "ground_truth_persistent_edges",
    "rigid_true_positive_edges",
    "rigid_ground_truth_edges",
    "false_reid_count",
    "same_class_mismatch_count",
    "unmatched_endpoint_edge_count",
)
_IOU_KEYS = ("iou_0_50", "iou_0_25")
_EFFECT_FIELDS = (
    "pair_id",
    "query_id",
    "query_confidence",
    "selected_threshold",
    "retained",
    "mask_membership_unchanged",
    "before_outcome",
    "after_outcome",
    "change_source",
    "before_t0_assigned_gt_id",
    "after_t0_assigned_gt_id",
    "before_t1_assigned_gt_id",
    "after_t1_assigned_gt_id",
    "t0_best_gt_id",
    "t0_best_iou",
    "t1_best_gt_id",
    "t1_best_iou",
    "duplicate_competition_before",
    "duplicate_competition_after",
)
_FIXED_TRANSFER_COUNT_KEYS = (
    "paired_prediction_edges",
    "true_positive_edges",
    "false_positive_edges",
    "ground_truth_persistent_edges",
    "rigid_true_positive_edges",
    "rigid_ground_truth_edges",
)
_FIXED_TRANSFER_FIELDS = (
    "aggregation",
    "pair_id",
    "environment_id",
    "instance_iou_threshold",
    "resolver_confidence_threshold",
    "matching_policy",
    "count_status",
    "count_null_reason",
    *_FIXED_TRANSFER_COUNT_KEYS,
    "paired_precision",
    "paired_recall",
    "paired_f1",
    "rigid_recall",
    "rate_status",
    "rate_null_reason",
)


class ResolverTuningError(ValueError):
    """Raised when resolver tuning inputs or metrics violate the protocol."""


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def aggregate_candidate_metrics(
    pair_candidates: Sequence[Mapping[float, Mapping[str, object]]],
) -> dict[float, dict[str, object]]:
    """Pool count metrics across pairs before deriving rates and F1."""

    rows = tuple(pair_candidates)
    if not rows:
        raise ResolverTuningError("candidate aggregation requires at least one pair")
    thresholds = set(rows[0])
    if not thresholds or any(set(row) != thresholds for row in rows):
        raise ResolverTuningError("candidate thresholds differ across pairs")
    result: dict[float, dict[str, object]] = {}
    for threshold in sorted(thresholds):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ResolverTuningError("candidate threshold is invalid")
        threshold_result: dict[str, object] = {}
        for iou_key in _IOU_KEYS:
            pooled: dict[str, int | float | None] = {}
            for key in _COUNT_KEYS:
                values: list[int] = []
                for pair_row in rows:
                    metrics = pair_row[threshold].get(iou_key)
                    if not isinstance(metrics, Mapping):
                        raise ResolverTuningError("candidate metric row is invalid")
                    value = metrics.get(key)
                    if type(value) is not int or value < 0:
                        raise ResolverTuningError(f"candidate count {key} is invalid")
                    values.append(value)
                pooled[key] = sum(values)
            precision = _ratio(
                int(pooled["true_positive_edges"]),
                int(pooled["paired_prediction_edges"]),
            )
            recall = _ratio(
                int(pooled["true_positive_edges"]),
                int(pooled["ground_truth_persistent_edges"]),
            )
            pooled["paired_precision"] = precision
            pooled["paired_recall"] = recall
            pooled["paired_f1"] = (
                None
                if precision is None or recall is None
                else 0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            pooled["rigid_recall"] = _ratio(
                int(pooled["rigid_true_positive_edges"]),
                int(pooled["rigid_ground_truth_edges"]),
            )
            threshold_result[iou_key] = pooled
        result[float(threshold)] = threshold_result
    return result


def select_candidate(candidates: Mapping[float, Mapping[str, object]]) -> float:
    """Select pooled IoU@0.5 F1, then precision, recall, and lower threshold."""

    if not candidates:
        raise ResolverTuningError("candidate selection requires metrics")

    def ranking(item: tuple[float, Mapping[str, object]]) -> tuple[float, ...]:
        threshold, value = item
        primary = value.get("iou_0_50")
        if not isinstance(primary, Mapping):
            raise ResolverTuningError("primary candidate metrics are invalid")
        metrics = tuple(primary.get(name) for name in ("paired_f1", "paired_precision", "paired_recall"))
        if any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            for metric in metrics
        ):
            raise ResolverTuningError("candidate ranking metric is invalid")
        return tuple(float(metric) for metric in metrics) + (-float(threshold),)

    return float(max(candidates.items(), key=ranking)[0])


def _fixed_transfer_rates(counts: Mapping[str, int]) -> dict[str, float | None]:
    precision = _ratio(
        counts["true_positive_edges"], counts["paired_prediction_edges"]
    )
    recall = _ratio(
        counts["true_positive_edges"], counts["ground_truth_persistent_edges"]
    )
    return {
        "paired_precision": precision,
        "paired_recall": recall,
        "paired_f1": (
            None
            if precision is None or recall is None
            else 0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        ),
        "rigid_recall": _ratio(
            counts["rigid_true_positive_edges"], counts["rigid_ground_truth_edges"]
        ),
    }


def _fixed_transfer_rate_status(
    rates: Mapping[str, float | None],
) -> tuple[str, str | None]:
    missing = [key for key, value in rates.items() if value is None]
    return (
        ("PASS", None)
        if not missing
        else ("PARTIAL", "ZERO_DENOMINATOR:" + ",".join(missing))
    )


def summarize_fixed_threshold_transfer(
    pair_results: Sequence[Mapping[str, object]],
    *,
    selected_threshold: float,
) -> tuple[dict[str, object], ...]:
    """Build per-environment, micro, and environment-macro transfer rows."""

    if selected_threshold != 0.3:
        raise ResolverTuningError("transfer evaluation requires frozen threshold 0.3")
    values = tuple(pair_results)
    if not values:
        raise ResolverTuningError("fixed transfer requires at least one environment")
    pair_ids: set[str] = set()
    environment_ids: set[str] = set()
    validated: list[tuple[str, str, Mapping[str, object]]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ResolverTuningError("fixed transfer pair result is invalid")
        pair_id = value.get("pair_id")
        environment_id = value.get("environment_id")
        thresholds = value.get("thresholds")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in pair_ids
            or not isinstance(environment_id, str)
            or not environment_id
            or environment_id in environment_ids
            or not isinstance(thresholds, Mapping)
            or set(thresholds) != set(_IOU_KEYS)
        ):
            raise ResolverTuningError(
                "fixed transfer pair/environment identity is invalid"
            )
        pair_ids.add(pair_id)
        environment_ids.add(environment_id)
        validated.append((pair_id, environment_id, thresholds))

    output: list[dict[str, object]] = []
    for iou_key, iou_threshold in (("iou_0_50", 0.50), ("iou_0_25", 0.25)):
        environment_rows: list[dict[str, object]] = []
        for pair_id, environment_id, thresholds in sorted(
            validated, key=lambda item: (item[1], item[0])
        ):
            raw_metrics = thresholds.get(iou_key)
            if not isinstance(raw_metrics, Mapping):
                raise ResolverTuningError("fixed transfer threshold row is invalid")
            counts: dict[str, int] = {}
            for key in _FIXED_TRANSFER_COUNT_KEYS:
                value = raw_metrics.get(key)
                if type(value) is not int or value < 0:
                    raise ResolverTuningError(
                        f"fixed transfer count {iou_key}.{key} is invalid"
                    )
                counts[key] = value
            if (
                counts["true_positive_edges"] > counts["paired_prediction_edges"]
                or counts["false_positive_edges"]
                != counts["paired_prediction_edges"] - counts["true_positive_edges"]
                or counts["true_positive_edges"]
                > counts["ground_truth_persistent_edges"]
                or counts["rigid_true_positive_edges"]
                > counts["rigid_ground_truth_edges"]
            ):
                raise ResolverTuningError("fixed transfer counts are inconsistent")
            rates = _fixed_transfer_rates(counts)
            rate_status, rate_null_reason = _fixed_transfer_rate_status(rates)
            row = {
                "aggregation": "environment",
                "pair_id": pair_id,
                "environment_id": environment_id,
                "instance_iou_threshold": iou_threshold,
                "resolver_confidence_threshold": 0.3,
                "matching_policy": "max_valid_count_then_iou",
                "count_status": "PASS",
                "count_null_reason": None,
                **counts,
                **rates,
                "rate_status": rate_status,
                "rate_null_reason": rate_null_reason,
            }
            environment_rows.append(row)
            output.append(row)

        pooled_counts = {
            key: sum(int(row[key]) for row in environment_rows)
            for key in _FIXED_TRANSFER_COUNT_KEYS
        }
        pooled_rates = _fixed_transfer_rates(pooled_counts)
        pooled_status, pooled_null_reason = _fixed_transfer_rate_status(pooled_rates)
        output.append(
            {
                "aggregation": "micro",
                "pair_id": "__micro__",
                "environment_id": "__all__",
                "instance_iou_threshold": iou_threshold,
                "resolver_confidence_threshold": 0.3,
                "matching_policy": "max_valid_count_then_iou",
                "count_status": "PASS",
                "count_null_reason": None,
                **pooled_counts,
                **pooled_rates,
                "rate_status": pooled_status,
                "rate_null_reason": pooled_null_reason,
            }
        )
        rate_keys = (
            "paired_precision",
            "paired_recall",
            "paired_f1",
            "rigid_recall",
        )
        macro_rates: dict[str, float | None] = {}
        missing_macro: list[str] = []
        for key in rate_keys:
            finite = [float(row[key]) for row in environment_rows if row[key] is not None]
            macro_rates[key] = None if not finite else sum(finite) / len(finite)
            if not finite:
                missing_macro.append(key)
        output.append(
            {
                "aggregation": "macro",
                "pair_id": "__macro__",
                "environment_id": "__all__",
                "instance_iou_threshold": iou_threshold,
                "resolver_confidence_threshold": 0.3,
                "matching_policy": "max_valid_count_then_iou",
                "count_status": "NOT_APPLICABLE",
                "count_null_reason": "ENVIRONMENT_MACRO_HAS_NO_COUNTS",
                **{key: None for key in _FIXED_TRANSFER_COUNT_KEYS},
                **macro_rates,
                "rate_status": "PASS" if not missing_macro else "PARTIAL",
                "rate_null_reason": (
                    None
                    if not missing_macro
                    else "NO_VALID_ENVIRONMENT_RATE:" + ",".join(missing_macro)
                ),
            }
        )
    return tuple(output)


def write_fixed_threshold_transfer_csv(
    output_path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Publish canonical fixed-threshold transfer metrics without clobbering."""

    output = Path(os.path.abspath(os.fspath(output_path)))
    if output.exists() or output.is_symlink():
        raise ResolverTuningError(f"fixed transfer output already exists: {output}")
    values = tuple(rows)
    if not values or any(set(row) != set(_FIXED_TRANSFER_FIELDS) for row in values):
        raise ResolverTuningError("fixed transfer rows have an invalid schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=_FIXED_TRANSFER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in values:
                writer.writerow(
                    {
                        key: "" if row[key] is None else row[key]
                        for key in _FIXED_TRANSFER_FIELDS
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ResolverTuningError(
                f"fixed transfer output already exists: {output}"
            ) from error
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def build_resolver_effect_rows(
    *,
    pair_id: str,
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    selected_threshold: float,
) -> tuple[dict[str, object], ...]:
    """Join threshold-zero and selected diagnostics without changing query masks."""

    if not isinstance(pair_id, str) or not pair_id:
        raise ResolverTuningError("effect pair_id must be non-empty")
    if (
        isinstance(selected_threshold, bool)
        or not isinstance(selected_threshold, (int, float))
        or not math.isfinite(float(selected_threshold))
        or not 0.0 < float(selected_threshold) < 1.0
    ):
        raise ResolverTuningError("effect threshold must be in (0, 1)")

    def indexed(
        values: Sequence[Mapping[str, object]], *, label: str
    ) -> dict[str, Mapping[str, object]]:
        result: dict[str, Mapping[str, object]] = {}
        for row in values:
            if not isinstance(row, Mapping):
                raise ResolverTuningError(f"{label} diagnostic row is invalid")
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id or query_id in result:
                raise ResolverTuningError(f"{label} diagnostic query ID is invalid")
            result[query_id] = row
        return result

    baseline = indexed(before, label="before")
    selected = indexed(after, label="after")
    if not set(selected) <= set(baseline):
        raise ResolverTuningError("selected diagnostics contain a new query")
    stable_keys = (
        "query_confidence",
        "t0_entity_ids",
        "t1_entity_ids",
        "t0_best_gt_id",
        "t0_best_iou",
        "t1_best_gt_id",
        "t1_best_iou",
    )
    output: list[dict[str, object]] = []
    for query_id in sorted(baseline):
        initial = baseline[query_id]
        retained = query_id in selected
        final = selected.get(query_id)
        confidence = initial.get("query_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise ResolverTuningError("diagnostic confidence is invalid")
        if retained:
            assert final is not None
            if any(initial.get(key) != final.get(key) for key in stable_keys):
                raise ResolverTuningError(
                    "selected query changed its mask-derived endpoint evidence"
                )
        elif float(confidence) >= float(selected_threshold):
            raise ResolverTuningError("selected diagnostics omitted an eligible query")
        before_outcome = initial.get("outcome")
        after_outcome = None if final is None else final.get("outcome")
        if not retained:
            change_source = "filtered_low_confidence"
        elif before_outcome != "true_positive" and after_outcome == "true_positive":
            change_source = "newly_true_positive_after_reassignment"
        elif before_outcome == "true_positive" and after_outcome == "true_positive":
            change_source = "retained_true_positive"
        elif before_outcome != after_outcome:
            change_source = "retained_outcome_changed"
        else:
            change_source = "retained_outcome_unchanged"
        output.append(
            {
                "pair_id": pair_id,
                "query_id": query_id,
                "query_confidence": float(confidence),
                "selected_threshold": float(selected_threshold),
                "retained": retained,
                "mask_membership_unchanged": retained,
                "before_outcome": before_outcome,
                "after_outcome": after_outcome,
                "change_source": change_source,
                "before_t0_assigned_gt_id": initial.get("t0_assigned_gt_id"),
                "after_t0_assigned_gt_id": (
                    None if final is None else final.get("t0_assigned_gt_id")
                ),
                "before_t1_assigned_gt_id": initial.get("t1_assigned_gt_id"),
                "after_t1_assigned_gt_id": (
                    None if final is None else final.get("t1_assigned_gt_id")
                ),
                "t0_best_gt_id": initial.get("t0_best_gt_id"),
                "t0_best_iou": initial.get("t0_best_iou"),
                "t1_best_gt_id": initial.get("t1_best_gt_id"),
                "t1_best_iou": initial.get("t1_best_iou"),
                "duplicate_competition_before": (
                    int(initial.get("t0_best_competitor_count", 0)) > 1
                    or int(initial.get("t1_best_competitor_count", 0)) > 1
                ),
                "duplicate_competition_after": (
                    False
                    if final is None
                    else int(final.get("t0_best_competitor_count", 0)) > 1
                    or int(final.get("t1_best_competitor_count", 0)) > 1
                ),
            }
        )
    return tuple(output)


def write_resolver_effect_csv(
    output_path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Atomically write canonical resolver-attribution rows without clobbering."""

    output = Path(os.path.abspath(os.fspath(output_path)))
    if output.exists() or output.is_symlink():
        raise ResolverTuningError(f"effect output already exists: {output}")
    values = tuple(rows)
    if not values or any(set(row) != set(_EFFECT_FIELDS) for row in values):
        raise ResolverTuningError("effect rows have an invalid schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=_EFFECT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in values:
                writer.writerow(
                    {
                        key: ""
                        if row[key] is None
                        else str(row[key]).lower()
                        if isinstance(row[key], bool)
                        else row[key]
                        for key in _EFFECT_FIELDS
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def run_effect_breakdown(
    tuning_result_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, tuple[dict[str, object], ...]]:
    """Re-evaluate cached DEV6 predictions at 0 and the frozen threshold."""

    _tuning_path, content = _regular_bytes(
        tuning_result_path, label="resolver tuning result"
    )
    tuning = _json_object(content, label="resolver tuning result")
    if (
        tuning.get("artifact_id") != "OVI_RESCENE_RESOLVER_TUNING_V1"
        or tuning.get("status") != "PASS"
    ):
        raise ResolverTuningError("resolver tuning result identity is invalid")
    selected_threshold = tuning.get("selected_threshold")
    if selected_threshold != 0.3:
        raise ResolverTuningError("resolver effect requires frozen threshold 0.3")
    bindings = tuning.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ResolverTuningError("resolver tuning source bindings are invalid")
    matrix_config_path, _matrix_payload = _load_bound(
        bindings.get("matrix_config"), label="matrix config"
    )
    matrix_config, selection = load_matrix_config(matrix_config_path)
    runtime = matrix_config.get("runtime")
    parameters = matrix_config.get("method_parameters")
    selection_record = matrix_config.get("selection_manifest")
    if not all(
        isinstance(value, Mapping)
        for value in (runtime, parameters, selection_record)
    ):
        raise ResolverTuningError("matrix protocol is invalid")
    projection = parameters.get("projection")
    if not isinstance(projection, Mapping):
        raise ResolverTuningError("matrix projection config is invalid")
    projection_config = ProjectionConfig(**projection)
    native_root = validated_directory(runtime["output_root"], label="native root")
    gt_root = validated_directory(runtime["gt_root"], label="GT root")
    pair_records = selection.get("pairs")
    if not isinstance(pair_records, list) or not pair_records:
        raise ResolverTuningError("selection pairs are invalid")

    output_rows: list[dict[str, object]] = []
    for raw_pair in pair_records:
        if not isinstance(raw_pair, Mapping):
            raise ResolverTuningError("selection pair is invalid")
        pair = build_method_pair_view_from_manifest(
            pair_record=raw_pair,
            source_manifest_sha256=str(selection_record["sha256"]),
            domain_id="D0_NATIVE_PROCESSED",
        )
        native = load_native_pair_artifact(native_root / pair.pair_id)
        ground_truth = load_pair_ground_truth(gt_root / pair.pair_id / "manifest.json")
        sample = pair.geometric_sample(
            neural_voxel_size_m=float(matrix_config["neural_voxel_size_m"])
        )
        evidence = native_query_evidence(
            pair,
            sample,
            native.arrays,
            checkpoint_sha256=native.checkpoint_sha256,
        )
        before_relations = resolve_native_queries_fragment_union(
            sample,
            evidence,
            projection_config,
            minimum_query_confidence=0.0,
        )
        after_relations = resolve_native_queries_fragment_union(
            sample,
            evidence,
            projection_config,
            minimum_query_confidence=float(selected_threshold),
        )
        output_rows.extend(
            build_resolver_effect_rows(
                pair_id=pair.pair_id,
                before=relation_outcome_rows(
                    pair, before_relations, ground_truth, iou_threshold=0.50
                ),
                after=relation_outcome_rows(
                    pair, after_relations, ground_truth, iou_threshold=0.50
                ),
                selected_threshold=float(selected_threshold),
            )
        )
    rows = tuple(
        sorted(output_rows, key=lambda row: (str(row["pair_id"]), str(row["query_id"])))
    )
    return write_resolver_effect_csv(output_path, rows), rows


def run_fixed_threshold_transfer(
    matrix_config_path: str | Path,
    transfer_manifest_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, tuple[dict[str, object], ...]]:
    """Evaluate the frozen 0.3 resolver on preselected transfer environments."""

    absolute_config, _config_content = _regular_bytes(
        matrix_config_path, label="transfer matrix config"
    )
    matrix_config, source_selection = load_matrix_config(absolute_config)
    transfer_path, transfer_content = _regular_bytes(
        transfer_manifest_path, label="transfer selection manifest"
    )
    transfer = _json_object(transfer_content, label="transfer selection manifest")
    selection_record = matrix_config.get("selection_manifest")
    runtime = matrix_config.get("runtime")
    parameters = matrix_config.get("method_parameters")
    if not all(
        isinstance(value, Mapping)
        for value in (selection_record, runtime, parameters)
    ):
        raise ResolverTuningError("transfer matrix protocol is invalid")
    if (
        transfer.get("artifact_id") != "RSCAN_T2_SELECTION_SLICE_V1"
        or transfer.get("schema_version") != 1
        or transfer.get("status") != "PASS"
        or transfer.get("role") != "RESOLVER_TRANSFER"
        or transfer.get("source_manifest") != selection_record
    ):
        raise ResolverTuningError("transfer selection identity is invalid")
    pair_ids = transfer.get("pair_ids")
    pairs = transfer.get("pairs")
    if (
        not isinstance(pair_ids, list)
        or not pair_ids
        or any(not isinstance(value, str) or not value for value in pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or not isinstance(pairs, list)
        or len(pairs) != len(pair_ids)
        or transfer.get("selected_pair_count") != len(pair_ids)
    ):
        raise ResolverTuningError("transfer selection pairs are invalid")
    source_pairs = source_selection.get("pairs")
    if not isinstance(source_pairs, list):
        raise ResolverTuningError("source selection pairs are invalid")
    source_by_id = {
        value.get("pair_id"): value
        for value in source_pairs
        if isinstance(value, Mapping) and isinstance(value.get("pair_id"), str)
    }
    if len(source_by_id) != len(source_pairs):
        raise ResolverTuningError("source selection pair identities are invalid")
    for pair_id, pair in zip(pair_ids, pairs, strict=True):
        if not isinstance(pair, Mapping) or pair.get("pair_id") != pair_id:
            raise ResolverTuningError("transfer pair ordering is invalid")
        if source_by_id.get(pair_id) != pair:
            raise ResolverTuningError("transfer pair differs from frozen source")

    projection = parameters.get("projection")
    association = parameters.get("association_metrics")
    if not isinstance(projection, Mapping) or not isinstance(association, Mapping):
        raise ResolverTuningError("transfer method parameters are invalid")
    projection_config = ProjectionConfig(**projection)
    observed_fraction = float(association["minimum_gt_observed_fraction"])
    native_root = validated_directory(runtime["output_root"], label="native root")
    gt_root = validated_directory(runtime["gt_root"], label="GT root")
    measured_pairs: list[dict[str, object]] = []
    for raw_pair in pairs:
        assert isinstance(raw_pair, Mapping)
        pair = build_method_pair_view_from_manifest(
            pair_record=raw_pair,
            source_manifest_sha256=str(selection_record["sha256"]),
            domain_id="D0_NATIVE_PROCESSED",
        )
        native = load_native_pair_artifact(native_root / pair.pair_id)
        ground_truth = load_pair_ground_truth(
            gt_root / pair.pair_id / "manifest.json"
        )
        if (
            native.method_input_sha256 != pair.method_tensor_sha256()
            or native.checkpoint_sha256 != runtime["checkpoint"]["sha256"]
            or native.source_commit != runtime["rescene_commit"]
        ):
            raise ResolverTuningError(
                "transfer native pair identity differs from frozen protocol"
            )
        sample = pair.geometric_sample(
            neural_voxel_size_m=float(matrix_config["neural_voxel_size_m"])
        )
        evidence = native_query_evidence(
            pair,
            sample,
            native.arrays,
            checkpoint_sha256=native.checkpoint_sha256,
        )
        relations = resolve_native_queries_fragment_union(
            sample,
            evidence,
            projection_config,
            minimum_query_confidence=0.3,
        )
        measured = evaluate_pair_relations(
            pair,
            relations,
            ground_truth,
            minimum_observed_fraction=observed_fraction,
            matching_policy="max_valid_count_then_iou",
        )
        environment_id = raw_pair.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id:
            raise ResolverTuningError("transfer environment ID is invalid")
        measured_pairs.append(
            {
                "pair_id": pair.pair_id,
                "environment_id": environment_id,
                "thresholds": measured["thresholds"],
            }
        )
    rows = summarize_fixed_threshold_transfer(
        measured_pairs, selected_threshold=0.3
    )
    return write_fixed_threshold_transfer_csv(output_path, rows), rows


def _regular_bytes(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ResolverTuningError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise ResolverTuningError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise ResolverTuningError(f"{label} changed while being read")
    return absolute, content


def _record(path: str | Path, *, label: str) -> dict[str, object]:
    absolute, content = _regular_bytes(path, label=label)
    return {
        "path": str(absolute),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResolverTuningError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ResolverTuningError(f"{label} must contain an object")
    return value


def _load_bound(record: object, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ResolverTuningError(f"{label} binding schema is invalid")
    path = record.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ResolverTuningError(f"{label} path must be absolute")
    absolute, content = _regular_bytes(path, label=label)
    if dict(record) != {
        "path": str(absolute),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }:
        raise ResolverTuningError(f"{label} binding mismatch")
    return absolute, _json_object(content, label=label)


def _candidate_thresholds(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ResolverTuningError("candidate thresholds must be a non-empty list")
    try:
        thresholds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ResolverTuningError("candidate threshold is not numeric") from error
    if (
        any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in thresholds)
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ResolverTuningError("candidate thresholds must be unique and increasing")
    return thresholds


def _metric_subset(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (*_COUNT_KEYS, "paired_precision", "paired_recall", "rigid_recall", "geometry")
    }


def _baseline_parity(observed: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for iou_key in _IOU_KEYS:
        current = observed.get(iou_key)
        reference = expected.get(iou_key)
        if not isinstance(current, Mapping) or not isinstance(reference, Mapping):
            raise ResolverTuningError("baseline parity metric is invalid")
        for key in (*_COUNT_KEYS, "paired_precision", "paired_recall", "rigid_recall"):
            left = current.get(key)
            right = reference.get(key)
            if isinstance(left, float) or isinstance(right, float):
                if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15):
                    raise ResolverTuningError(f"threshold-zero parity failed for {iou_key}.{key}")
            elif left != right:
                raise ResolverTuningError(f"threshold-zero parity failed for {iou_key}.{key}")


def run_tuning(config_path: str | Path) -> tuple[Path, dict[str, object]]:
    """Evaluate a fixed threshold grid and atomically publish the development result."""

    started = time.perf_counter()
    config_path, config_content = _regular_bytes(config_path, label="tuning config")
    config = _json_object(config_content, label="tuning config")
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_LOCAL_ADAPTATION_CONFIG_V1"
        or config.get("objective")
        != {
            "threshold": "iou_0_50",
            "metric": "paired_f1",
            "tie_breakers": [
                "paired_precision",
                "paired_recall",
                "lower_confidence_threshold",
            ],
        }
    ):
        raise ResolverTuningError("tuning config identity or objective is invalid")
    thresholds = _candidate_thresholds(config.get("candidate_thresholds"))
    _decision_path, decision = _load_bound(
        config.get("decision_result"), label="adaptation decision"
    )
    _matrix_path, matrix = _load_bound(config.get("matrix_result"), label="matrix result")
    if decision.get("action") != "TUNE_RESOLVER" or decision.get("status") != "PASS":
        raise ResolverTuningError("adaptation decision does not authorize resolver tuning")
    if matrix.get("artifact_id") != "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1" or matrix.get("status") != "PASS":
        raise ResolverTuningError("matrix result identity is invalid")
    matrix_bindings = matrix.get("source_bindings")
    matrix_config_record = matrix_bindings.get("config") if isinstance(matrix_bindings, Mapping) else None
    matrix_config_path, _matrix_config_payload = _load_bound(
        matrix_config_record, label="matrix config"
    )
    matrix_config, selection = load_matrix_config(matrix_config_path)
    runtime = matrix_config.get("runtime")
    parameters = matrix_config.get("method_parameters")
    selection_record = matrix_config.get("selection_manifest")
    if not all(isinstance(value, Mapping) for value in (runtime, parameters, selection_record)):
        raise ResolverTuningError("matrix protocol is invalid")
    projection = parameters.get("projection")
    association = parameters.get("association_metrics")
    if not isinstance(projection, Mapping) or not isinstance(association, Mapping):
        raise ResolverTuningError("matrix method parameters are invalid")
    projection_config = ProjectionConfig(**projection)
    observed_fraction = float(association["minimum_gt_observed_fraction"])
    native_root = validated_directory(runtime["output_root"], label="native root")
    gt_root = validated_directory(runtime["gt_root"], label="GT root")
    pair_records = selection.get("pairs")
    if not isinstance(pair_records, list) or not pair_records:
        raise ResolverTuningError("selection pairs are invalid")
    pair_rows: list[dict[float, Mapping[str, object]]] = []
    per_pair: dict[str, object] = {}
    native_bindings: dict[str, object] = {}
    gt_bindings: dict[str, object] = {}
    for raw_pair in pair_records:
        if not isinstance(raw_pair, Mapping):
            raise ResolverTuningError("selection pair is invalid")
        pair = build_method_pair_view_from_manifest(
            pair_record=raw_pair,
            source_manifest_sha256=str(selection_record["sha256"]),
            domain_id="D0_NATIVE_PROCESSED",
        )
        native_manifest = native_root / pair.pair_id / "manifest.json"
        gt_manifest = gt_root / pair.pair_id / "manifest.json"
        native = load_native_pair_artifact(native_manifest.parent)
        ground_truth = load_pair_ground_truth(gt_manifest)
        if (
            native.method_input_sha256 != pair.method_tensor_sha256()
            or native.checkpoint_sha256 != runtime["checkpoint"]["sha256"]
            or native.source_commit != runtime["rescene_commit"]
        ):
            raise ResolverTuningError("native pair identity differs from matrix protocol")
        sample = pair.geometric_sample(
            neural_voxel_size_m=float(matrix_config["neural_voxel_size_m"])
        )
        evidence = native_query_evidence(
            pair,
            sample,
            native.arrays,
            checkpoint_sha256=native.checkpoint_sha256,
        )
        candidates: dict[float, Mapping[str, object]] = {}
        relation_fingerprints: dict[str, object] = {}
        for threshold in (0.0, *thresholds):
            relations = resolve_native_queries_fragment_union(
                sample,
                evidence,
                projection_config,
                minimum_query_confidence=threshold,
            )
            measured = evaluate_pair_relations(
                pair,
                relations,
                ground_truth,
                minimum_observed_fraction=observed_fraction,
            )
            candidates[threshold] = {
                iou_key: _metric_subset(measured["thresholds"][iou_key])
                for iou_key in _IOU_KEYS
            }
            relation_payload = [
                {
                    "query_id": str(relation.temporal_query_id),
                    "t0": list(relation.t0_entity_ids),
                    "t1": list(relation.t1_entity_ids),
                    "confidence": relation.query_confidence,
                }
                for relation in relations
            ]
            relation_fingerprints[f"{threshold:.2f}"] = {
                "relation_count": len(relations),
                "paired_relation_count": sum(
                    bool(relation.t0_entity_ids and relation.t1_entity_ids)
                    for relation in relations
                ),
                "sha256": hashlib.sha256(
                    json.dumps(relation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        pair_rows.append(candidates)
        per_pair[pair.pair_id] = {
            "candidate_metrics": {
                f"{threshold:.2f}": candidates[threshold] for threshold in candidates
            },
            "relation_fingerprints": relation_fingerprints,
        }
        native_bindings[pair.pair_id] = _record(native_manifest, label="native manifest")
        gt_bindings[pair.pair_id] = _record(gt_manifest, label="GT manifest")
    aggregate = aggregate_candidate_metrics(pair_rows)
    baseline = aggregate.pop(0.0)
    expected_legacy = matrix.get("methods", {}).get("R_legacy", {}).get("pooled")
    if not isinstance(expected_legacy, Mapping):
        raise ResolverTuningError("matrix R_legacy baseline is invalid")
    _baseline_parity(baseline, expected_legacy)
    selected_threshold = select_candidate(aggregate)
    selected = aggregate[selected_threshold]
    initial_supported = matrix.get("methods", {}).get("R_supported", {}).get("pooled")
    if not isinstance(initial_supported, Mapping):
        raise ResolverTuningError("matrix R_supported baseline is invalid")
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_RESOLVER_TUNING_V1",
        "status": "PASS",
        "claim_boundary": "3RScan_development_threshold_selection_not_unseen_generalization",
        "action": "TUNE_RESOLVER",
        "method": "query_consistent_fragment_union_with_confidence_gate",
        "objective": config["objective"],
        "candidate_thresholds": list(thresholds),
        "selected_threshold": selected_threshold,
        "before": {
            "R_legacy": baseline,
            "R_supported": initial_supported,
        },
        "candidates": {
            f"{threshold:.2f}": aggregate[threshold] for threshold in aggregate
        },
        "selected": selected,
        "per_pair": per_pair,
        "runtime": {
            "device": "cpu",
            "runtime_s": time.perf_counter() - started,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "source_bindings": {
            "config": {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_content).hexdigest(),
                "byte_count": len(config_content),
            },
            "decision_result": dict(config["decision_result"]),
            "matrix_result": dict(config["matrix_result"]),
            "matrix_config": dict(matrix_config_record),
            "selection_manifest": dict(selection_record),
            "native_manifests": native_bindings,
            "ground_truth_manifests": gt_bindings,
            "tuning_source": _record(Path(__file__).resolve(), label="tuning source"),
            "resolver_source": _record(
                REPO_ROOT / "src/evaluation/rscan_method_views.py",
                label="resolver source",
            ),
        },
    }
    output = config.get("output")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ResolverTuningError("tuning output path must be absolute")
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise ResolverTuningError(f"tuning output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--config", type=Path)
    action.add_argument("--effect-from", type=Path)
    action.add_argument("--fixed-transfer-config", type=Path)
    parser.add_argument("--effect-output", type=Path)
    parser.add_argument("--transfer-manifest", type=Path)
    parser.add_argument("--transfer-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.config is not None:
            if any(
                value is not None
                for value in (
                    arguments.effect_output,
                    arguments.transfer_manifest,
                    arguments.transfer_output,
                )
            ):
                raise ResolverTuningError(
                    "tuning mode does not accept effect or transfer outputs"
                )
            output, result = run_tuning(arguments.config)
            summary = {
                "output": str(output),
                "selected_threshold": result["selected_threshold"],
            }
        elif arguments.effect_from is not None:
            if (
                arguments.effect_output is None
                or arguments.transfer_manifest is not None
                or arguments.transfer_output is not None
            ):
                raise ResolverTuningError(
                    "effect mode requires only --effect-output"
                )
            output, rows = run_effect_breakdown(
                arguments.effect_from, arguments.effect_output
            )
            summary = {"output": str(output), "row_count": len(rows)}
        else:
            if (
                arguments.effect_output is not None
                or arguments.transfer_manifest is None
                or arguments.transfer_output is None
            ):
                raise ResolverTuningError(
                    "fixed transfer mode requires --transfer-manifest and --transfer-output"
                )
            output, rows = run_fixed_threshold_transfer(
                arguments.fixed_transfer_config,
                arguments.transfer_manifest,
                arguments.transfer_output,
            )
            summary = {
                "output": str(output),
                "row_count": len(rows),
                "selected_threshold": 0.3,
            }
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = [
    "ResolverTuningError",
    "aggregate_candidate_metrics",
    "build_resolver_effect_rows",
    "run_effect_breakdown",
    "run_fixed_threshold_transfer",
    "run_tuning",
    "select_candidate",
    "summarize_fixed_threshold_transfer",
    "write_resolver_effect_csv",
    "write_fixed_threshold_transfer_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
