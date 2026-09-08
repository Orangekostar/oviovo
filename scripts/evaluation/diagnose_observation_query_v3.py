#!/usr/bin/env python3
"""Diagnose cached observation-query predictions across dense readout stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_ovi_observation_query import _load_dense_inputs
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    PredictedInstance,
    evaluate_instance_geometry,
    voxelize_points,
)
from src.oviv2.rescene_dense_instance_readout import (
    _exclusive_query_winners,
    _query_scores,
    build_dense_instance_readout,
    build_dense_to_model_indices,
)

DEFAULT_RUN_IDS = (
    "20260908_scene0109_obs_base_frozen_v2_domains",
    "20260908_scene0109_obs_base_tuned_train200_dev_v2",
    "20260908_scene0109_obs_base_tuned_train500_dev_v2",
    "20260908_scene0109_obs_base_tuned_train1000_dev_v2",
    "20260908_scene0109_obs_fuse_train200_dev_v2",
    "20260908_scene0109_obs_fuse_train500_dev_v2",
    "20260908_scene0109_obs_fuse_train1000_dev_v2",
    "20260908_scene0109_obs_full_smoke200_dev_v2",
    "20260908_scene0109_obs_full_train500_dev_v2",
    "20260908_scene0109_obs_full_train1000_dev_v2",
)
DEFAULT_INTERVENTION_RUNS = (
    (200, "I_FULL", "20260908_scene0109_obs_full_train200_i_full_diag_v3"),
    (200, "I_BETA0", "20260908_scene0109_obs_full_train200_i_beta0_diag_v3"),
    (
        200,
        "I_ALPHA0_BETA0",
        "20260908_scene0109_obs_full_train200_i_alpha0_beta0_diag_v3",
    ),
    (1000, "I_FULL", "20260908_scene0109_obs_full_train1000_i_full_diag_v3"),
    (1000, "I_BETA0", "20260908_scene0109_obs_full_train1000_i_beta0_diag_v3"),
    (
        1000,
        "I_ALPHA0_BETA0",
        "20260908_scene0109_obs_full_train1000_i_alpha0_beta0_diag_v3",
    ),
)

STAGE_FIELDS = (
    "run_id",
    "method_id",
    "checkpoint_id",
    "training_updates",
    "environment",
    "pair",
    "visit",
    "evaluation_domain",
    "stage",
    "prediction_count",
    "target_count",
    "maximum_valid_pairs_50",
    "maximum_valid_pairs_25",
    "best_iou_mean",
    "ar50",
    "ar25",
    "tp50",
    "fp50",
    "fn50",
    "f1_50",
    "tp25",
    "fp25",
    "fn25",
    "f1_25",
)

OBJECT_FIELDS = (
    "run_id",
    "method_id",
    "checkpoint_id",
    "training_updates",
    "environment",
    "pair",
    "visit",
    "evaluation_domain",
    "gt_instance_id",
    "semantic_label",
    "full_surface_support",
    "common_support_fraction",
    "included_in_evaluation_domain",
    "raw_best_query_id",
    "raw_best_iou",
    "query_score",
    "eligible_best_query_id",
    "eligible_best_iou",
    "exclusive_query_best_iou",
    "final_best_iou",
    "raw_query_survives_threshold",
    "survived_point_fraction",
    "final_tp_match_id",
    "residual_involvement",
    "classification",
)

INTERVENTION_FIELDS = (
    "run_id",
    "method_id",
    "training_updates",
    "checkpoint_id",
    "environment",
    "pair",
    "visit",
    "intervention_id",
    "applied_observation_mode",
    "full_f1_50",
    "common_tp50",
    "common_fp50",
    "common_fn50",
    "common_f1_50",
    "common_f1_50_delta_vs_full",
    "raw_common_best_iou_mean",
    "raw_common_best_iou_delta_vs_full",
    "raw_common_ar50",
    "raw_common_ar25",
    "alpha_by_layer_json",
    "beta_by_layer_json",
    "observation_stats_json",
    "status",
)


class ObservationQueryDiagnosisError(ValueError):
    """Raised when cached diagnostic inputs violate their frozen contract."""


def _voxels(points: np.ndarray, voxel_size_m: float) -> frozenset[tuple[int, int, int]]:
    if not len(points):
        return frozenset()
    return voxelize_points(points, voxel_size_m=voxel_size_m)


def _iou(
    left: frozenset[tuple[int, int, int]],
    right: frozenset[tuple[int, int, int]],
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _proposal(
    prediction_id: str,
    point_indices: np.ndarray,
    points_xyz: np.ndarray,
    voxel_size_m: float,
    *,
    raw_query_id: int | None,
    owner_source: str,
) -> dict[str, object] | None:
    rows = np.asarray(point_indices, dtype=np.int64)
    voxels = _voxels(points_xyz[rows], voxel_size_m)
    if not voxels:
        return None
    return {
        "prediction_id": prediction_id,
        "point_indices": rows,
        "voxels": voxels,
        "raw_query_id": raw_query_id,
        "owner_source": owner_source,
    }


def _matching_metrics(
    proposals: Sequence[Mapping[str, object]],
    targets: Sequence[GroundTruthInstance],
    *,
    report_final_metrics: bool,
) -> tuple[dict[str, object], object]:
    predictions = tuple(
        PredictedInstance(str(row["prediction_id"]), row["voxels"])
        for row in proposals
    )
    result = evaluate_instance_geometry(
        predictions, targets, matching_policy="max_valid_count_then_iou"
    )
    best = [
        max((_iou(target.voxels, row["voxels"]) for row in proposals), default=0.0)
        for target in targets
    ]

    def threshold_values(threshold: object) -> tuple[int, int, int, float | None]:
        tp = int(threshold.matched_count)
        fp = len(predictions) - tp
        fn = len(targets) - tp
        denominator = 2 * tp + fp + fn
        return tp, fp, fn, None if denominator == 0 else 2 * tp / denominator

    tp50, fp50, fn50, f1_50 = threshold_values(result.primary)
    tp25, fp25, fn25, f1_25 = threshold_values(result.sensitivity)
    return (
        {
            "prediction_count": len(predictions),
            "target_count": len(targets),
            "maximum_valid_pairs_50": tp50,
            "maximum_valid_pairs_25": tp25,
            "best_iou_mean": None if not best else float(np.mean(best)),
            "ar50": None if not best else sum(value >= 0.50 for value in best) / len(best),
            "ar25": None if not best else sum(value >= 0.25 for value in best) / len(best),
            "tp50": tp50 if report_final_metrics else None,
            "fp50": fp50 if report_final_metrics else None,
            "fn50": fn50 if report_final_metrics else None,
            "f1_50": f1_50 if report_final_metrics else None,
            "tp25": tp25 if report_final_metrics else None,
            "fp25": fp25 if report_final_metrics else None,
            "fn25": fn25 if report_final_metrics else None,
            "f1_25": f1_25 if report_final_metrics else None,
        },
        result,
    )


def _best_proposal(
    target_voxels: frozenset[tuple[int, int, int]],
    proposals: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object] | None, float]:
    ranked = sorted(
        ((_iou(target_voxels, row["voxels"]), str(row["prediction_id"]), row) for row in proposals),
        key=lambda value: (-value[0], value[1]),
    )
    if not ranked or ranked[0][0] <= 0.0:
        return None, 0.0
    return ranked[0][2], float(ranked[0][0])


def diagnose_visit_stages(
    *,
    visit_id: int,
    points_xyz: np.ndarray,
    entity_owner_indices: np.ndarray,
    dense_to_model_indices: np.ndarray,
    ground_truth: Sequence[GroundTruthInstance],
    pred_masks_mq: np.ndarray,
    pred_logits_qc: np.ndarray,
    minimum_query_score: float,
    voxel_size_m: float,
    evaluation_domain: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Return stage metrics and per-GT trajectories for one dense visit."""

    if evaluation_domain not in {"FULL_GT_V2", "COMMON_INPUT_SUPPORT_V2"}:
        raise ObservationQueryDiagnosisError("evaluation domain is unsupported")
    points = np.asarray(points_xyz, dtype=np.float32)
    owners = np.asarray(entity_owner_indices, dtype=np.int64)
    dense_to_model = np.asarray(dense_to_model_indices, dtype=np.int64)
    masks = np.asarray(pred_masks_mq)
    logits = np.asarray(pred_logits_qc)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or owners.shape != (len(points),)
        or dense_to_model.shape != (len(points),)
        or np.any(dense_to_model >= len(masks))
    ):
        raise ObservationQueryDiagnosisError("dense visit inputs are inconsistent")

    retained, query_scores, sigmoid = _query_scores(masks, logits)
    eligible = retained[query_scores[retained] >= float(minimum_query_score)]
    winners = _exclusive_query_winners(
        dense_to_model=dense_to_model,
        eligible_queries=eligible,
        query_scores=query_scores,
        masks_mq=masks,
        sigmoid_mq=sigmoid,
        point_chunk_size=131072,
    )
    common_mask = dense_to_model >= 0
    domain_mask = (
        np.ones(len(points), dtype=np.bool_)
        if evaluation_domain == "FULL_GT_V2"
        else common_mask
    )
    common_voxels = _voxels(points[common_mask], voxel_size_m)
    domain_voxels = _voxels(points[domain_mask], voxel_size_m)

    source_targets = tuple(ground_truth)
    if evaluation_domain == "FULL_GT_V2":
        targets = source_targets
    else:
        targets = tuple(
            GroundTruthInstance(
                target.instance_id,
                target.semantic_label,
                target.voxels & domain_voxels,
            )
            for target in source_targets
            if target.voxels & domain_voxels
        )

    raw: list[dict[str, object]] = []
    eligible_rows: list[dict[str, object]] = []
    for query in retained.tolist():
        rows = np.flatnonzero(
            common_mask
            & domain_mask
            & (masks[np.maximum(dense_to_model, 0), query] > 0.0)
        )
        row = _proposal(
            f"query_{query:04d}",
            rows,
            points,
            voxel_size_m,
            raw_query_id=query,
            owner_source="query",
        )
        if row is None:
            continue
        raw.append(row)
        if query in eligible:
            eligible_rows.append(row)

    exclusive: list[dict[str, object]] = []
    for query in sorted({int(value) for value in winners if value >= 0}):
        row = _proposal(
            f"query_{query:04d}",
            np.flatnonzero(domain_mask & (winners == query)),
            points,
            voxel_size_m,
            raw_query_id=query,
            owner_source="query",
        )
        if row is not None:
            exclusive.append(row)

    final = list(exclusive)
    for owner in sorted({int(value) for value in owners if value >= 0}):
        row = _proposal(
            f"residual_{owner:04d}",
            np.flatnonzero(domain_mask & (owners == owner) & (winners < 0)),
            points,
            voxel_size_m,
            raw_query_id=None,
            owner_source="ovi_residual",
        )
        if row is not None:
            final.append(row)

    stage_definitions = (
        ("R0_RAW_ALL", raw, False),
        ("R1_ELIGIBLE", eligible_rows, False),
        ("R2_EXCLUSIVE_QUERY", exclusive, True),
        ("R3_FINAL_MAP", final, True),
    )
    stage_rows: list[dict[str, object]] = []
    stage_results: dict[str, object] = {}
    for stage, proposals, report_final in stage_definitions:
        values, result = _matching_metrics(
            proposals, targets, report_final_metrics=report_final
        )
        stage_rows.append(
            {
                "visit": visit_id,
                "evaluation_domain": evaluation_domain,
                "stage": stage,
                **values,
            }
        )
        stage_results[stage] = result

    final_matches = {
        match.gt_instance_id: match.prediction_id
        for match in stage_results["R3_FINAL_MAP"].primary.matches
    }
    final_by_id = {str(row["prediction_id"]): row for row in final}
    object_rows: list[dict[str, object]] = []
    for target in sorted(source_targets, key=lambda value: value.instance_id):
        target_domain_voxels = target.voxels & domain_voxels
        raw_best, raw_iou = _best_proposal(target_domain_voxels, raw)
        eligible_best, eligible_iou = _best_proposal(target_domain_voxels, eligible_rows)
        _exclusive_best, exclusive_iou = _best_proposal(target_domain_voxels, exclusive)
        final_best, final_iou = _best_proposal(target_domain_voxels, final)
        raw_query = None if raw_best is None else int(raw_best["raw_query_id"])
        survived = raw_query is not None and raw_query in set(eligible.tolist())
        exclusive_same = next(
            (row for row in exclusive if row["raw_query_id"] == raw_query), None
        )
        raw_points = None if raw_best is None else set(raw_best["point_indices"].tolist())
        exclusive_points = (
            set() if exclusive_same is None else set(exclusive_same["point_indices"].tolist())
        )
        survived_fraction = (
            None
            if not raw_points
            else len(raw_points & exclusive_points) / len(raw_points)
        )
        match_id = final_matches.get(target.instance_id)
        matched_final = None if match_id is None else final_by_id[match_id]
        residual_involvement = bool(
            (matched_final is not None and matched_final["owner_source"] == "ovi_residual")
            or (final_best is not None and final_best["owner_source"] == "ovi_residual")
        )
        common_fraction = len(target.voxels & common_voxels) / len(target.voxels)
        reasons: list[str] = []
        if common_fraction < 0.5:
            reasons.append("support_limited")
        if raw_iou < 0.5:
            reasons.append("raw_regression")
        elif not survived:
            reasons.append("filtered")
        if eligible_iou >= 0.5 and exclusive_iou < 0.5:
            reasons.append("competition")
        if exclusive_iou >= 0.5 and (final_iou < 0.5 or match_id is None):
            reasons.append("residual_or_assignment")
        if final_iou < 0.5:
            reasons.append("below_50")
        if not reasons:
            reasons.append("unresolved")
        object_rows.append(
            {
                "visit": visit_id,
                "evaluation_domain": evaluation_domain,
                "gt_instance_id": target.instance_id,
                "semantic_label": target.semantic_label,
                "full_surface_support": len(target.voxels),
                "common_support_fraction": common_fraction,
                "included_in_evaluation_domain": bool(target_domain_voxels),
                "raw_best_query_id": raw_query,
                "raw_best_iou": raw_iou,
                "query_score": None if raw_query is None else float(query_scores[raw_query]),
                "eligible_best_query_id": (
                    None if eligible_best is None else int(eligible_best["raw_query_id"])
                ),
                "eligible_best_iou": eligible_iou,
                "exclusive_query_best_iou": exclusive_iou,
                "final_best_iou": final_iou,
                "raw_query_survives_threshold": survived,
                "survived_point_fraction": survived_fraction,
                "final_tp_match_id": match_id,
                "residual_involvement": residual_involvement,
                "classification": "|".join(reasons),
            }
        )
    return tuple(stage_rows), tuple(object_rows)


def _file_record(
    path: Path, *, recorded_path: str | None = None
) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path) if recorded_path is None else recorded_path,
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ObservationQueryDiagnosisError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ObservationQueryDiagnosisError(f"JSON root must be an object: {path}")
    return value


def _collect_intervention_rows(
    evaluation_root: Path,
    specifications: Sequence[tuple[int, str, str]],
) -> tuple[dict[str, object], ...]:
    loaded: list[tuple[int, str, str, dict[str, object], list[dict[str, str]]]] = []
    for updates, intervention_id, run_id in specifications:
        root = evaluation_root / run_id
        summary = _read_json(root / "summary.json")
        if (
            summary.get("status") != "PASS"
            or summary.get("method_id") != "OBS_FULL"
            or summary.get("training_updates") != updates
        ):
            raise ObservationQueryDiagnosisError(
                f"intervention run identity mismatch: {run_id}"
            )
        with (root / "results.csv").open(encoding="utf-8", newline="") as stream:
            result_rows = list(csv.DictReader(stream))
        loaded.append((updates, intervention_id, run_id, summary, result_rows))

    gates_by_updates: dict[int, Mapping[str, object]] = {}
    checkpoints_by_updates: dict[int, str] = {}
    for updates, intervention_id, run_id, summary, _result_rows in loaded:
        checkpoint_id = summary.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise ObservationQueryDiagnosisError(
                f"intervention checkpoint is missing: {run_id}"
            )
        previous = checkpoints_by_updates.setdefault(updates, checkpoint_id)
        if checkpoint_id != previous:
            raise ObservationQueryDiagnosisError(
                f"interventions do not share the {updates}-update checkpoint"
            )
        metadata = summary.get("inference_intervention")
        if isinstance(metadata, Mapping):
            if metadata.get("intervention_id") != intervention_id:
                raise ObservationQueryDiagnosisError(
                    f"intervention metadata mismatch: {run_id}"
                )
            gates_by_updates.setdefault(updates, metadata)
    missing_gates = sorted(set(checkpoints_by_updates) - set(gates_by_updates))
    if missing_gates:
        raise ObservationQueryDiagnosisError(
            f"intervention gates unavailable for updates: {missing_gates}"
        )

    def domain_row(rows: Sequence[dict[str, str]], domain: str) -> dict[str, str]:
        matches = [
            row
            for row in rows
            if row.get("visit_id") == "1" and row.get("evaluation_domain") == domain
        ]
        if len(matches) != 1:
            raise ObservationQueryDiagnosisError(
                f"intervention result lacks unique t1 {domain} row"
            )
        return matches[0]

    def numeric(row: Mapping[str, str], key: str) -> float | None:
        value = row.get(key)
        return None if value in (None, "") else float(value)

    rows: list[dict[str, object]] = []
    for updates, intervention_id, run_id, summary, result_rows in loaded:
        full = domain_row(result_rows, "FULL_GT_V2")
        common = domain_row(result_rows, "COMMON_INPUT_SUPPORT_V2")
        raw = domain_row(result_rows, "RAW_COMMON_INPUT_SUPPORT_V2")
        gates = gates_by_updates[updates]
        metadata = summary.get("inference_intervention")
        applied_mode = (
            "full"
            if intervention_id == "I_FULL"
            else metadata.get("applied_observation_mode")
            if isinstance(metadata, Mapping)
            else None
        )
        rows.append(
            {
                "run_id": run_id,
                "method_id": "OBS_FULL",
                "training_updates": updates,
                "checkpoint_id": summary["checkpoint_id"],
                "environment": summary.get("eval_environment_id"),
                "pair": summary.get("eval_pair_id"),
                "visit": 1,
                "intervention_id": intervention_id,
                "applied_observation_mode": applied_mode,
                "full_f1_50": numeric(full, "f1_50"),
                "common_tp50": numeric(common, "tp50"),
                "common_fp50": numeric(common, "fp50"),
                "common_fn50": numeric(common, "fn50"),
                "common_f1_50": numeric(common, "f1_50"),
                "common_f1_50_delta_vs_full": None,
                "raw_common_best_iou_mean": numeric(raw, "raw_best_iou_mean"),
                "raw_common_best_iou_delta_vs_full": None,
                "raw_common_ar50": numeric(raw, "raw_ar50"),
                "raw_common_ar25": numeric(raw, "raw_ar25"),
                "alpha_by_layer_json": json.dumps(
                    gates.get("alpha_by_layer"), separators=(",", ":")
                ),
                "beta_by_layer_json": json.dumps(
                    gates.get("beta_by_layer"), separators=(",", ":")
                ),
                "observation_stats_json": json.dumps(
                    (
                        metadata.get("observation_stats")
                        if isinstance(metadata, Mapping)
                        else gates.get("observation_stats")
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "status": "PASS_DIAGNOSTIC_NOT_TRAINING_ABLATION",
            }
        )
    full_by_updates = {
        int(row["training_updates"]): row
        for row in rows
        if row["intervention_id"] == "I_FULL"
    }
    for row in rows:
        reference = full_by_updates[int(row["training_updates"])]
        for value_key, delta_key in (
            ("common_f1_50", "common_f1_50_delta_vs_full"),
            (
                "raw_common_best_iou_mean",
                "raw_common_best_iou_delta_vs_full",
            ),
        ):
            value = row[value_key]
            baseline = reference[value_key]
            row[delta_key] = (
                None if value is None or baseline is None else float(value) - float(baseline)
            )
    return tuple(rows)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_diagnosis(
    *,
    evaluation_root: Path,
    run_ids: Sequence[str],
    legacy_config: Path,
    output_root: Path,
    minimum_query_score: float = 0.3,
    include_default_interventions: bool = False,
) -> None:
    pair, bundle, ground_truth = _load_dense_inputs(legacy_config)
    dense_maps = build_dense_to_model_indices(
        pair, surface=bundle.surface, supported_view=bundle.supported_view
    )
    stage_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for run_id in run_ids:
        run_root = evaluation_root / run_id
        summary_path = run_root / "summary.json"
        prediction_path = run_root / "predictions.npz"
        summary = _read_json(summary_path)
        if summary.get("status") != "PASS" or summary.get("eval_pair_id") != pair.pair_id:
            raise ObservationQueryDiagnosisError(f"evaluation run is incompatible: {run_id}")
        expected = summary.get("artifacts", {}).get("predictions")
        actual = _file_record(prediction_path)
        if not isinstance(expected, Mapping) or any(
            expected.get(key) != actual[key] for key in ("sha256", "byte_count")
        ):
            raise ObservationQueryDiagnosisError(f"prediction binding mismatch: {run_id}")
        try:
            with np.load(prediction_path, allow_pickle=False) as arrays:
                masks = np.asarray(arrays["pred_masks_mq"])
                logits = np.asarray(arrays["pred_logits_qc"])
                cached_retained = np.asarray(arrays["raw_query_indices"])
                cached_scores = np.asarray(arrays["query_scores"])
                cached_owners = (
                    np.asarray(arrays["visit0_owner_instance_indices"]),
                    np.asarray(arrays["visit1_owner_instance_indices"]),
                )
        except (OSError, KeyError, ValueError) as error:
            raise ObservationQueryDiagnosisError(f"invalid predictions: {run_id}") from error
        readout = build_dense_instance_readout(
            pair,
            surface=bundle.surface,
            supported_view=bundle.supported_view,
            pred_masks_mq=masks,
            pred_logits_qc=logits,
            minimum_query_score=minimum_query_score,
        )
        if (
            not np.array_equal(readout.raw_query_indices, cached_retained)
            or not np.allclose(readout.query_scores, cached_scores, rtol=0.0, atol=1e-7)
            or any(
                not np.array_equal(readout.visits[index].owner_instance_indices, cached_owners[index])
                for index in (0, 1)
            )
        ):
            raise ObservationQueryDiagnosisError(f"cached readout mismatch: {run_id}")
        metadata = {
            "run_id": run_id,
            "method_id": summary.get("method_id"),
            "checkpoint_id": summary.get("checkpoint_id"),
            "training_updates": summary.get("training_updates"),
            "environment": summary.get("eval_environment_id"),
            "pair": pair.pair_id,
        }
        for evaluation_domain in ("FULL_GT_V2", "COMMON_INPUT_SUPPORT_V2"):
            for visit_id in (0, 1):
                stages, objects = diagnose_visit_stages(
                    visit_id=visit_id,
                    points_xyz=pair.visits[visit_id].points_xyz,
                    entity_owner_indices=pair.visits[visit_id].entity_owner_indices,
                    dense_to_model_indices=dense_maps[visit_id],
                    ground_truth=ground_truth.visits[visit_id],
                    pred_masks_mq=masks,
                    pred_logits_qc=logits,
                    minimum_query_score=minimum_query_score,
                    voxel_size_m=ground_truth.voxel_size_m,
                    evaluation_domain=evaluation_domain,
                )
                stage_rows.extend({**metadata, **row} for row in stages)
                object_rows.extend({**metadata, **row} for row in objects)
        inputs.append({"run_id": run_id, "summary": _file_record(summary_path), "predictions": actual})

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        _write_csv(staging / "readout_stages.csv", STAGE_FIELDS, stage_rows)
        _write_csv(staging / "object_trajectories.csv", OBJECT_FIELDS, object_rows)
        intervention_rows = (
            _collect_intervention_rows(evaluation_root, DEFAULT_INTERVENTION_RUNS)
            if include_default_interventions
            else ()
        )
        if intervention_rows:
            _write_csv(
                staging / "intervention_results.csv",
                INTERVENTION_FIELDS,
                intervention_rows,
            )
        summary = {
            "schema_version": 1,
            "artifact_id": "OVI_OBSERVATION_QUERY_V3_READOUT_DIAGNOSIS_V1",
            "status": "PASS",
            "pair_id": pair.pair_id,
            "run_count": len(run_ids),
            "stage_row_count": len(stage_rows),
            "object_row_count": len(object_rows),
            "minimum_query_score": minimum_query_score,
            "evaluation_domains": ["FULL_GT_V2", "COMMON_INPUT_SUPPORT_V2"],
            "inputs": inputs,
            "artifacts": {
                "readout_stages": _file_record(
                    staging / "readout_stages.csv",
                    recorded_path="readout_stages.csv",
                ),
                "object_trajectories": _file_record(
                    staging / "object_trajectories.csv",
                    recorded_path="object_trajectories.csv",
                ),
                **(
                    {
                        "intervention_results": _file_record(
                            staging / "intervention_results.csv",
                            recorded_path="intervention_results.csv",
                        )
                    }
                    if intervention_rows
                    else {}
                ),
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            raise ObservationQueryDiagnosisError(f"output already exists: {output_root}")
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument(
        "--legacy-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/ovi_rescene_dense_instance_repair_v1.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--intervention-output", type=Path)
    parser.add_argument("--minimum-query-score", type=float, default=0.3)
    parser.add_argument("--include-default-interventions", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.output_root is None) == (args.intervention_output is None):
        parser.error("choose exactly one of --output-root or --intervention-output")
    if args.intervention_output is not None:
        output = args.intervention_output.absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(
            output,
            INTERVENTION_FIELDS,
            _collect_intervention_rows(
                args.evaluation_root.absolute(), DEFAULT_INTERVENTION_RUNS
            ),
        )
        return 0
    assert args.output_root is not None
    run_diagnosis(
        evaluation_root=args.evaluation_root.absolute(),
        run_ids=tuple(args.run_ids or DEFAULT_RUN_IDS),
        legacy_config=args.legacy_config.absolute(),
        output_root=args.output_root.absolute(),
        minimum_query_score=args.minimum_query_score,
        include_default_interventions=args.include_default_interventions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
