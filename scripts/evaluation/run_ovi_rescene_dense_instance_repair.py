#!/usr/bin/env python3
"""Evaluate P0/P1/P2 dense instance readouts on a bound 3RScan OVI pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.rescene_pair_executor import NativeForwardResult
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
    D2InferenceBundle,
    build_d2_inference_bundle,
    build_temporal_query_evidence,
    load_object_transfer_config,
)
from scripts.evaluation.run_ovi_rescene_object_recovery import (
    _bound_content,
    _json_object,
    _mapping_manifest_paths,
    _pair_record,
)
from src.evaluation.dense_instance_repair_artifacts import (
    export_dense_method_artifacts,
)
from src.evaluation.dense_instance_repair_metrics import (
    DenseMethodView,
    build_dense_endpoint_bindings,
    build_p0_method_view,
    build_p1_method_view,
    build_p2_method_view,
    evaluate_dense_identity,
    evaluate_dense_instance_method,
)
from src.evaluation.ovi_endpoint_diagnosis import (
    RawDenseProposal,
    diagnose_visit_endpoints,
    endpoint_diagnosis_rows,
)
from src.evaluation.ovi_pair_artifact_loader import restore_bound_ovi_pair_artifact
from src.evaluation.ovi_pair_views import OviObjectPairView, build_ovi_object_pair_view
from src.evaluation.rscan_gt_instances import (
    GroundTruthPair,
    PredictedInstance,
    load_pair_ground_truth,
    voxelize_points,
)
from src.evaluation.temporal_object_groups import (
    GroupingConfig,
    build_temporal_object_groups,
)
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
)
from src.oviv2.rescene_dense_instance_readout import (
    DensePairReadout,
    build_dense_instance_readout,
)

_SOURCE_BINDINGS = frozenset(
    {
        "object_transfer_config",
        "selection_manifest",
        "d2_mapping_receipt",
        "d2_pair_receipt",
        "d2_forward_metadata",
        "d2_forward_arrays",
        "ground_truth_manifest",
    }
)
_METHODS = ("P0", "P1", "P2")
_INSTANCE_FIELDS = (
    "pair",
    "visit",
    "method",
    "evaluation_domain",
    "tp_at_050",
    "fp_at_050",
    "fn_at_050",
    "precision_at_050",
    "recall_at_050",
    "f1_at_050",
    "fragment_gt_count_at_050",
    "merge_prediction_count_at_050",
    "duplicate_prediction_count_at_050",
    "mean_matched_iou_at_050",
    "tp_at_025",
    "fp_at_025",
    "fn_at_025",
    "precision_at_025",
    "recall_at_025",
    "f1_at_025",
    "fragment_gt_count_at_025",
    "merge_prediction_count_at_025",
    "duplicate_prediction_count_at_025",
    "mean_matched_iou_at_025",
    "raw_candidate_count",
    "final_instance_count",
    "split_parent_count",
    "merged_candidate_count",
    "query_owned_fraction",
    "residual_fraction",
    "background_fraction",
    "unknown_fraction",
    "source_point_count",
    "final_point_count",
    "geometric_change_count",
    "method_view_sha256",
)
_IDENTITY_FIELDS = (
    "schema_version",
    "protocol_id",
    "status",
    "pair",
    "method",
    "support_domain",
    "iou_threshold",
    "method_view_sha256",
    "binding_sha256",
    "paired_prediction_count",
    "ambiguous_identity_count",
    "true_positive_count",
    "false_positive_count",
    "endpoint_failure_count",
    "false_reid_count",
    "same_class_mismatch_count",
    "duplicate_count",
    "precision",
    "persistent_gt_count",
    "end_to_end_recall",
    "conditional_gt_count",
    "conditional_recall",
    "conditional_status",
    "rigid_gt_count",
    "rigid_recall",
)
_RUN_FIELDS = (
    "pair",
    "method",
    "pair_content_sha256",
    "method_view_sha256",
    "dense_point_count",
    "raw_query_count",
    "build_runtime_s",
    "instance_evaluation_runtime_s",
    "identity_evaluation_runtime_s",
    "artifact_runtime_s",
    "cached_joint_forward_runtime_s",
    "cached_joint_forward_peak_memory_bytes",
    "status",
)


class DenseInstanceRepairRunError(ValueError):
    """Raised when a dense repair run is malformed or not source-bound."""


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def csv_bytes(
    rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]
) -> bytes:
    """Serialize schema-exact, null-preserving LF CSV bytes."""

    values = tuple(rows)
    fields = tuple(fieldnames)
    if not values or not fields or any(tuple(row) != fields for row in values):
        raise DenseInstanceRepairRunError("CSV row schema is inconsistent")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: _csv_value(row[field]) for field in fields} for row in values
    )
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _npz_arrays(content: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            expected = {
                "independent_t0",
                "independent_t1",
                "pred_masks_mq",
                "pred_logits_qc",
            }
            if set(archive.files) != expected:
                raise DenseInstanceRepairRunError(
                    "cached forward array schema is invalid"
                )
            return {name: np.asarray(archive[name]) for name in expected}
    except (OSError, ValueError) as error:
        if isinstance(error, DenseInstanceRepairRunError):
            raise
        raise DenseInstanceRepairRunError("cached forward arrays are malformed") from error


def cached_forward_outputs(
    metadata_content: bytes,
    arrays_content: bytes,
    *,
    pair_id: str,
    pair_content_sha256: str,
    sample_content_sha256: str,
    candidate_counts: tuple[int, int],
    independent_model_counts: tuple[int, int],
    supported_model_count: int,
    expected_query_count: int = 100,
    config_sha256: str,
    checkpoint_sha256: str,
) -> AssociationForwardOutputs:
    """Load cached native outputs only after validating every bound domain."""

    try:
        metadata = _json_object(metadata_content, label="cached forward metadata")
    except ValueError as error:
        raise DenseInstanceRepairRunError(str(error)) from error
    if (
        metadata.get("schema_version") != 1
        or metadata.get("artifact_id")
        != "OVI_RESCENE_OBJECT_LEVEL_NATIVE_FORWARDS_V1"
        or metadata.get("status") != "PASS"
        or metadata.get("pair_id") != pair_id
    ):
        raise DenseInstanceRepairRunError("cached forward metadata identity mismatch")
    checks = (
        ("pair_content_sha256", pair_content_sha256, "pair identity mismatch"),
        ("sample_content_sha256", sample_content_sha256, "sample identity mismatch"),
        ("candidate_counts", list(candidate_counts), "candidate counts mismatch"),
        ("config_sha256", config_sha256, "config identity mismatch"),
        ("checkpoint_sha256", checkpoint_sha256, "checkpoint identity mismatch"),
    )
    for key, expected, message in checks:
        if metadata.get(key) != expected:
            raise DenseInstanceRepairRunError(message)
    domains = metadata.get("domains")
    if not isinstance(domains, Mapping) or domains.get("supported_model") != supported_model_count:
        raise DenseInstanceRepairRunError("supported model domain mismatch")
    record = metadata.get("npz")
    if (
        not isinstance(record, Mapping)
        or record.get("sha256") != hashlib.sha256(arrays_content).hexdigest()
        or record.get("byte_count") != len(arrays_content)
    ):
        raise DenseInstanceRepairRunError("cached forward array binding mismatch")
    arrays = _npz_arrays(arrays_content)
    independent = (arrays["independent_t0"], arrays["independent_t1"])
    if any(
        array.ndim != 2
        or array.shape[0] != independent_model_counts[index]
        or not np.issubdtype(array.dtype, np.floating)
        or not np.all(np.isfinite(array))
        for index, array in enumerate(independent)
    ) or independent[0].shape[1] != independent[1].shape[1]:
        raise DenseInstanceRepairRunError("independent visit row count mismatch")
    masks = arrays["pred_masks_mq"]
    logits = arrays["pred_logits_qc"]
    if masks.ndim != 2 or masks.shape[0] != supported_model_count:
        raise DenseInstanceRepairRunError("joint mask M dimension mismatch")
    if (
        logits.ndim != 2
        or masks.shape[1] != logits.shape[0]
        or masks.shape[1] != expected_query_count
        or logits.shape[1] < 2
    ):
        raise DenseInstanceRepairRunError("joint query dimensions mismatch")
    if any(
        not np.issubdtype(array.dtype, np.floating)
        or not np.all(np.isfinite(array))
        for array in (masks, logits)
    ):
        raise DenseInstanceRepairRunError("joint outputs are not finite floats")
    forward = metadata.get("forward")
    if not isinstance(forward, Mapping):
        raise DenseInstanceRepairRunError("cached forward runtime metadata is missing")
    try:
        visit_hashes = tuple(forward["visit_forward_sha256"])
        runtimes = tuple(float(value) for value in forward["independent_runtime_s"])
        if (
            len(visit_hashes) != 2
            or any(not _digest(value) for value in visit_hashes)
            or len(runtimes) != 2
            or any(value < 0.0 or not np.isfinite(value) for value in runtimes)
        ):
            raise DenseInstanceRepairRunError(
                "cached independent forward metadata is invalid"
            )
        result = AssociationForwardOutputs(
            independent_model_features=independent,
            visit_forward_sha256=(str(visit_hashes[0]), str(visit_hashes[1])),
            independent_runtime_s=(runtimes[0], runtimes[1]),
            joint_forward=NativeForwardResult(
                pred_masks_mq=masks,
                pred_logits_qc=logits,
                runtime_s=float(forward["joint_runtime_s"]),
                peak_memory_bytes=int(forward["peak_memory_bytes"]),
                peak_reserved_memory_bytes=int(forward["peak_reserved_memory_bytes"]),
                rss_peak_bytes=int(forward["rss_peak_bytes"]),
                device_name=str(forward["device_name"]),
                model_tensor_count=int(forward["model_tensor_count"]),
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, DenseInstanceRepairRunError):
            raise
        raise DenseInstanceRepairRunError(
            "cached forward runtime metadata is malformed"
        ) from error
    return result


def _independent_model_counts(bundle: D2InferenceBundle) -> tuple[int, int]:
    visits = np.asarray(bundle.model_input.model_visit_ids)
    if visits.ndim != 1 or np.any(~np.isin(visits, (0, 1))):
        raise DenseInstanceRepairRunError("model visit domain is invalid")
    return int(np.count_nonzero(visits == 0)), int(np.count_nonzero(visits == 1))


def _build_method_views(
    pair: OviObjectPairView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
    *,
    minimum_query_score: float,
    minimum_entity_token_coverage: float,
    minimum_source_point_coverage: float,
    static_centroid_tolerance_m: float,
    backend_config_sha256: str,
    checkpoint_sha256: str,
    point_chunk_size: int,
) -> tuple[Mapping[str, DenseMethodView], DensePairReadout, Mapping[str, float]]:
    runtimes: dict[str, float] = {}
    start = time.perf_counter()
    evidence = build_temporal_query_evidence(
        bundle,
        outputs,
        backend_config_sha256=backend_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    projection = project_queries_to_instances(
        bundle.sample,
        evidence,
        ProjectionConfig(
            minimum_entity_token_coverage=minimum_entity_token_coverage,
            minimum_source_point_coverage=minimum_source_point_coverage,
            static_centroid_tolerance_m=static_centroid_tolerance_m,
        ),
    )
    p1_grouping = build_temporal_object_groups(
        pair,
        projection.relations,
        variant_id="U3",
        config=GroupingConfig(minimum_query_confidence=minimum_query_score),
    )
    runtimes["P1"] = time.perf_counter() - start
    start = time.perf_counter()
    readout = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=outputs.joint_forward.pred_masks_mq,
        pred_logits_qc=outputs.joint_forward.pred_logits_qc,
        minimum_query_score=minimum_query_score,
        point_chunk_size=point_chunk_size,
    )
    runtimes["P2"] = time.perf_counter() - start
    start = time.perf_counter()
    p0 = build_p0_method_view(pair)
    runtimes["P0"] = time.perf_counter() - start
    views = {
        "P0": p0,
        "P1": build_p1_method_view(pair, p1_grouping),
        "P2": build_p2_method_view(pair, readout),
    }
    if tuple(views) != _METHODS or len({view.source_xyz_sha256 for view in views.values()}) != 1:
        raise DenseInstanceRepairRunError("method view geometry is inconsistent")
    return MappingProxyType(views), readout, MappingProxyType(runtimes)


def _identity_rows(
    pair: OviObjectPairView,
    view: DenseMethodView,
    ground_truth: GroundTruthPair,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for domain in ("full", "supported"):
        binding = build_dense_endpoint_bindings(
            pair, view, ground_truth, support_domain=domain
        )
        for threshold in (0.50, 0.25):
            result = dict(
                evaluate_dense_identity(
                    view, ground_truth, binding, iou_threshold=threshold
                )
            )
            result.pop("outcomes")
            rows.append(result)
    return tuple(rows)


def _change_types(ground_truth: GroundTruthPair, visit_id: int) -> dict[int, str]:
    rules = ground_truth.identity_rules
    nonrigid = set(rules.rescan_to_reference.values()) - set(
        rules.rigid_transform_by_reference
    )
    result = {}
    for target in ground_truth.visits[visit_id]:
        reference = (
            target.instance_id
            if visit_id == 0
            else rules.reference_id_for_rescan(target.instance_id)
        )
        result[target.instance_id] = (
            "removed"
            if reference in rules.removed_reference_ids
            else "rigid"
            if reference in rules.rigid_transform_by_reference
            else "nonrigid"
            if reference in nonrigid
            else "unchanged_or_unknown"
        )
    return result


def _diagnosis_rows(
    pair: OviObjectPairView,
    ground_truth: GroundTruthPair,
    readout: DensePairReadout,
    p2: DenseMethodView,
    *,
    exact_union_candidate_limit: int,
) -> tuple[Mapping[str, object], ...]:
    rows = []
    score_by_query = dict(
        zip(readout.raw_query_indices.tolist(), readout.query_scores.tolist(), strict=True)
    )
    confident = frozenset(
        f"query_{query:04d}"
        for query, score in score_by_query.items()
        if score >= readout.minimum_query_score
    )
    for visit_id in (0, 1):
        raw = tuple(
            RawDenseProposal(
                query_id=row.query_id,
                query_score=row.query_score,
                point_indices=row.point_indices,
            )
            for row in readout.raw_proposals[visit_id]
        )
        predictions = tuple(
            PredictedInstance(
                prediction_id=row.candidate_id,
                voxels=voxelize_points(
                    pair.visits[visit_id].points_xyz[row.point_indices],
                    voxel_size_m=ground_truth.voxel_size_m,
                ),
            )
            for row in p2.candidates[visit_id]
        )
        rows.extend(
            diagnose_visit_endpoints(
                pair_id=pair.pair_id,
                visit=pair.visits[visit_id],
                ground_truth=ground_truth.visits[visit_id],
                raw_proposals=raw,
                confident_query_ids=confident
                & frozenset(row.query_id for row in readout.raw_proposals[visit_id]),
                final_predictions=predictions,
                change_type_by_gt=_change_types(ground_truth, visit_id),
                voxel_size_m=ground_truth.voxel_size_m,
                neural_supported_point_mask=readout.visits[visit_id].neural_valid,
                sensor_visible_gt_by_instance=None,
                exact_union_candidate_limit=exact_union_candidate_limit,
            )
        )
    return endpoint_diagnosis_rows(tuple(rows))


def _load_config(config_path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    path = Path(os.path.abspath(os.fspath(config_path)))
    try:
        content = path.read_bytes()
        config = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DenseInstanceRepairRunError("dense repair config is unavailable") from error
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_DENSE_INSTANCE_REPAIR_V1"
        or config.get("status") != "FROZEN_BEFORE_DENSE_REPAIR_RESULTS"
        or set(config.get("source_bindings", {})) != _SOURCE_BINDINGS
        or not isinstance(config.get("protocol"), dict)
        or not isinstance(config.get("outputs"), dict)
    ):
        raise DenseInstanceRepairRunError("dense repair config identity is invalid")
    return path, content, config


def _resolve_output(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DenseInstanceRepairRunError(f"{label} path is invalid")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def run_dense_instance_repair(
    config_path: str | Path, *, evaluated_commit: str
) -> dict[str, object]:
    """Rebuild all bound inputs and publish measured P0/P1/P2 outputs."""

    if len(evaluated_commit) != 40 or any(
        value not in "0123456789abcdef" for value in evaluated_commit
    ):
        raise DenseInstanceRepairRunError("evaluated commit is invalid")
    config_path, config_content, config = _load_config(config_path)
    pair_id = config.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise DenseInstanceRepairRunError("dense repair pair ID is invalid")
    protocol = config["protocol"]
    outputs_config = config["outputs"]
    try:
        minimum_query_score = float(protocol["minimum_query_score"])
        minimum_entity_token_coverage = float(
            protocol["minimum_entity_token_coverage"]
        )
        minimum_source_point_coverage = float(
            protocol["minimum_source_point_coverage"]
        )
        expected_query_count = int(protocol["expected_query_count"])
        point_chunk_size = int(protocol["point_chunk_size"])
        exact_union_limit = int(protocol["exact_union_candidate_limit"])
    except (KeyError, TypeError, ValueError) as error:
        raise DenseInstanceRepairRunError("dense repair protocol is invalid") from error
    if (
        not 0.0 <= minimum_query_score <= 1.0
        or not 0.0 <= minimum_entity_token_coverage <= 1.0
        or not 0.0 <= minimum_source_point_coverage <= 1.0
        or expected_query_count < 1
        or point_chunk_size < 1
        or exact_union_limit < 1
    ):
        raise DenseInstanceRepairRunError("dense repair protocol is invalid")
    bindings = config["source_bindings"]
    loaded = {}
    try:
        loaded = {
            name: _bound_content(bindings[name], label=name)
            for name in sorted(_SOURCE_BINDINGS)
        }
    except ValueError as error:
        raise DenseInstanceRepairRunError(str(error)) from error
    object_config = load_object_transfer_config(loaded["object_transfer_config"][0])
    if (
        object_config.source_sha256
        != hashlib.sha256(loaded["object_transfer_config"][1]).hexdigest()
        or object_config.selection_manifest != loaded["selection_manifest"][0]
        or object_config.selection_manifest_sha256
        != hashlib.sha256(loaded["selection_manifest"][1]).hexdigest()
        or object_config.d2_mapping_receipt != loaded["d2_mapping_receipt"][0]
        or object_config.d2_pair_view_receipt != loaded["d2_pair_receipt"][0]
    ):
        raise DenseInstanceRepairRunError("object-transfer parent bindings differ")
    selection = _json_object(loaded["selection_manifest"][1], label="selection")
    pair_record = _pair_record(selection, pair_id)
    mapping = _json_object(loaded["d2_mapping_receipt"][1], label="mapping")
    native_manifests, materialized_manifests = _mapping_manifest_paths(mapping)
    pair = build_ovi_object_pair_view(
        pair_record=pair_record,
        source_manifest_sha256=object_config.selection_manifest_sha256,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    pair = restore_bound_ovi_pair_artifact(
        pair,
        _json_object(loaded["d2_pair_receipt"][1], label="D2 pair receipt"),
    )
    if pair.pair_id != pair_id:
        raise DenseInstanceRepairRunError("D2 pair identity mismatch")
    ground_truth = load_pair_ground_truth(loaded["ground_truth_manifest"][0])
    if ground_truth.pair_id != pair_id:
        raise DenseInstanceRepairRunError("ground truth pair identity mismatch")
    bundle_start = time.perf_counter()
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=object_config.native_sampler_source.path,
        sampler_source_sha256=object_config.native_sampler_source.sha256,
        sampler_seed=object_config.sampler_seed,
        maximum_candidates=object_config.maximum_candidates,
    )
    bundle_runtime = time.perf_counter() - bundle_start
    cached = cached_forward_outputs(
        loaded["d2_forward_metadata"][1],
        loaded["d2_forward_arrays"][1],
        pair_id=pair_id,
        pair_content_sha256=pair.content_sha256(),
        sample_content_sha256=bundle.sample.content_sha256(),
        candidate_counts=tuple(len(value) for value in pair.candidate_ids),
        independent_model_counts=_independent_model_counts(bundle),
        supported_model_count=len(bundle.model_input.model_visit_ids),
        expected_query_count=expected_query_count,
        config_sha256=object_config.source_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
    )
    views, readout, build_runtimes = _build_method_views(
        pair,
        bundle,
        cached,
        minimum_query_score=minimum_query_score,
        minimum_entity_token_coverage=minimum_entity_token_coverage,
        minimum_source_point_coverage=minimum_source_point_coverage,
        static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        backend_config_sha256=object_config.source_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
        point_chunk_size=point_chunk_size,
    )
    instance_rows: list[Mapping[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    instance_runtimes: dict[str, float] = {}
    identity_runtimes: dict[str, float] = {}
    for method in _METHODS:
        start = time.perf_counter()
        instance_rows.extend(
            evaluate_dense_instance_method(pair, views[method], ground_truth)
        )
        instance_runtimes[method] = time.perf_counter() - start
        start = time.perf_counter()
        identity_rows.extend(_identity_rows(pair, views[method], ground_truth))
        identity_runtimes[method] = time.perf_counter() - start
    diagnosis = _diagnosis_rows(
        pair,
        ground_truth,
        readout,
        views["P2"],
        exact_union_candidate_limit=exact_union_limit,
    )
    local_root = _resolve_output(outputs_config.get("local_root"), label="local output")
    tracked_root = _resolve_output(
        outputs_config.get("tracked_root"), label="tracked output"
    )
    if any(path.exists() or path.is_symlink() for path in (local_root, tracked_root)):
        raise DenseInstanceRepairRunError("a configured output already exists")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{local_root.name}.", dir=local_root.parent)
    )
    artifact_runtimes: dict[str, float] = {}
    try:
        for method in _METHODS:
            start = time.perf_counter()
            export_dense_method_artifacts(pair, views[method], staging / method)
            artifact_runtimes[method] = time.perf_counter() - start
        run_rows = tuple(
            {
                "pair": pair_id,
                "method": method,
                "pair_content_sha256": pair.content_sha256(),
                "method_view_sha256": views[method].content_sha256(),
                "dense_point_count": sum(visit.point_count for visit in pair.visits),
                "raw_query_count": len(readout.raw_query_indices),
                "build_runtime_s": build_runtimes[method] + (
                    bundle_runtime if method == "P0" else 0.0
                ),
                "instance_evaluation_runtime_s": instance_runtimes[method],
                "identity_evaluation_runtime_s": identity_runtimes[method],
                "artifact_runtime_s": artifact_runtimes[method],
                "cached_joint_forward_runtime_s": cached.joint_forward.runtime_s,
                "cached_joint_forward_peak_memory_bytes": cached.joint_forward.peak_memory_bytes,
                "status": "PASS",
            }
            for method in _METHODS
        )
        diagnosis_fields = tuple(diagnosis[0])
        contents = {
            "endpoint_diagnosis.csv": csv_bytes(
                diagnosis, fieldnames=diagnosis_fields
            ),
            "instance_metrics.csv": csv_bytes(
                instance_rows, fieldnames=_INSTANCE_FIELDS
            ),
            "identity_metrics.csv": csv_bytes(
                identity_rows, fieldnames=_IDENTITY_FIELDS
            ),
            "runs.csv": csv_bytes(run_rows, fieldnames=_RUN_FIELDS),
        }
        for name, content in contents.items():
            _write_bytes(staging / name, content)
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_DENSE_INSTANCE_REPAIR_RESULT_V1",
            "status": "PASS",
            "pair_id": pair_id,
            "evaluated_commit": evaluated_commit,
            "config_sha256": hashlib.sha256(config_content).hexdigest(),
            "pair_content_sha256": pair.content_sha256(),
            "sample_content_sha256": bundle.sample.content_sha256(),
            "method_view_sha256": {
                method: views[method].content_sha256() for method in _METHODS
            },
            "protocol": protocol,
            "source_bindings": bindings,
            "tables": {
                name: _artifact_record(staging / name, root=staging)
                for name in sorted(contents)
            },
            "artifacts": {
                method: _artifact_record(staging / method / "manifest.json", root=staging)
                for method in _METHODS
            },
        }
        _write_bytes(staging / "manifest.json", _json_bytes(manifest))
        os.replace(staging, local_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    tracked_root.parent.mkdir(parents=True, exist_ok=True)
    tracked_staging = Path(
        tempfile.mkdtemp(prefix=f".{tracked_root.name}.", dir=tracked_root.parent)
    )
    try:
        for name in (
            "endpoint_diagnosis.csv",
            "instance_metrics.csv",
            "identity_metrics.csv",
            "runs.csv",
            "manifest.json",
        ):
            shutil.copyfile(local_root / name, tracked_staging / name)
        preview_dir = tracked_staging / "P2"
        preview_dir.mkdir()
        for name in ("instances.png", "rgb.png"):
            shutil.copyfile(local_root / "P2" / "current" / name, preview_dir / name)
        os.replace(tracked_staging, tracked_root)
    except BaseException:
        shutil.rmtree(tracked_staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluated-commit", required=True)
    arguments = parser.parse_args(argv)
    result = run_dense_instance_repair(
        arguments.config, evaluated_commit=arguments.evaluated_commit
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DenseInstanceRepairRunError",
    "cached_forward_outputs",
    "csv_bytes",
    "run_dense_instance_repair",
]
