#!/usr/bin/env python3
"""Evaluate G/F/R association on one immutable dense P2 candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_ovi_rescene_dense_instance_repair import (
    _IDENTITY_FIELDS,
    _SOURCE_BINDINGS,
    _artifact_record,
    _bound_content,
    _build_method_views,
    _independent_model_counts,
    _json_bytes,
    _json_object,
    _load_config,
    _mapping_manifest_paths,
    _pair_record,
    _resolve_output,
    _source_bound_grid_sample_factory,
    _write_bytes,
    cached_forward_outputs,
    csv_bytes,
)
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
    D2InferenceBundle,
    build_d2_inference_bundle,
    build_temporal_query_evidence,
    load_object_transfer_config,
)
from src.evaluation.dense_instance_repair_metrics import (
    DenseMethodView,
    build_dense_endpoint_bindings,
    build_dense_feature_score_matrix,
    build_dense_geometric_score_matrix,
    build_dense_rescene_score_matrix,
    evaluate_dense_association,
    solve_dense_pair_assignment,
)
from src.evaluation.ovi_pair_artifact_loader import restore_bound_ovi_pair_artifact
from src.evaluation.ovi_pair_views import OviObjectPairView, build_ovi_object_pair_view
from src.evaluation.rscan_gt_instances import GroundTruthPair, load_pair_ground_truth

_METHODS = ("G_full", "G_supported", "F_obj", "R_obj")


class DenseP2AssociationRunError(ValueError):
    """Raised when fixed-P2 association inputs or outputs are inconsistent."""


def _dense_to_surface(
    pair: OviObjectPairView, bundle: D2InferenceBundle
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    surface = bundle.surface
    for visit_id, visit in enumerate(pair.visits):
        mapping = np.full(visit.point_count, -1, dtype=np.int64)
        selected = np.flatnonzero(surface.source_visit_ids == visit_id)
        original = surface.original_vertex_indices[selected]
        if (
            np.any(original < 0)
            or np.any(original >= visit.point_count)
            or len(np.unique(original)) != len(original)
        ):
            raise DenseP2AssociationRunError("surface-to-dense mapping is invalid")
        mapping[original] = selected
        result.append(mapping)
    return result[0], result[1]


def pool_dense_candidate_features(
    pair: OviObjectPairView,
    view: DenseMethodView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Pool separate-visit model features over fixed P2 candidate points."""

    if view.method_id != "P2" or view.pair_content_sha256 != pair.content_sha256():
        raise DenseP2AssociationRunError("feature pool requires the bound P2 view")
    width = outputs.independent_model_features[0].shape[1]
    model_features = np.empty(
        (len(bundle.model_input.model_visit_ids), width), dtype=np.float32
    )
    for visit_id in (0, 1):
        model_indices = np.flatnonzero(bundle.model_input.model_visit_ids == visit_id)
        values = outputs.independent_model_features[visit_id]
        if values.shape != (len(model_indices), width):
            raise DenseP2AssociationRunError("independent feature rows are invalid")
        model_features[model_indices] = values
    contributor_counts = np.diff(bundle.geometry.source_to_adapter_offsets)
    old_model_by_contributor = np.repeat(
        bundle.sampling.adapter_to_model, contributor_counts
    )
    source_to_old_model = np.empty(bundle.geometry.source_point_count, dtype=np.int64)
    source_to_old_model[bundle.geometry.source_point_indices] = old_model_by_contributor
    source_to_model = bundle.supported_view.old_to_new_model_indices[
        source_to_old_model
    ]
    dense_to_surface = _dense_to_surface(pair, bundle)
    features = []
    valid = []
    for visit_id, candidates in enumerate(view.candidates):
        pooled = np.zeros((len(candidates), width), dtype=np.float32)
        available = np.zeros(len(candidates), dtype=np.bool_)
        for candidate_index, candidate in enumerate(candidates):
            source = dense_to_surface[visit_id][candidate.point_indices]
            source = source[source >= 0]
            models = source_to_model[source]
            models = models[models >= 0]
            if not len(models):
                continue
            pooled[candidate_index] = np.mean(
                model_features[models], axis=0, dtype=np.float64
            )
            available[candidate_index] = True
        features.append(pooled)
        valid.append(available)
    return (features[0], features[1]), (valid[0], valid[1])


def pool_dense_candidate_query_affinities(
    pair: OviObjectPairView,
    view: DenseMethodView,
    bundle: D2InferenceBundle,
    token_scores: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Pool joint-query token scores by exact supported P2 ownership."""

    scores = np.asarray(token_scores, dtype=np.float64)
    adapter_count = len(bundle.sample.coordinates_xyzt)
    if scores.ndim != 2 or scores.shape[1] != adapter_count:
        raise DenseP2AssociationRunError("query token score shape is invalid")
    offsets = bundle.sample.source_to_token_offsets
    token_by_source = np.repeat(np.arange(adapter_count), np.diff(offsets))
    old_source = bundle.supported_view.new_to_old_source_point_indices
    if len(token_by_source) != len(old_source):
        raise DenseP2AssociationRunError("supported source/token mapping is invalid")
    visits = bundle.surface.source_visit_ids[old_source]
    dense_indices = bundle.surface.original_vertex_indices[old_source]
    affinities = []
    valid = []
    for visit_id, candidates in enumerate(view.candidates):
        owner = np.full(len(old_source), -1, dtype=np.int64)
        selected = visits == visit_id
        owner[selected] = view.owner_instance_indices[visit_id][dense_indices[selected]]
        supported = selected & (owner >= 0)
        candidate_count = len(candidates)
        counts = np.bincount(
            token_by_source[supported] * candidate_count + owner[supported],
            minlength=adapter_count * candidate_count,
        ).reshape(adapter_count, candidate_count)
        denominators = counts.sum(axis=0)
        available = denominators > 0
        pooled = np.zeros((scores.shape[0], candidate_count), dtype=np.float64)
        pooled[:, available] = (
            scores @ counts[:, available]
        ) / denominators[available]
        affinities.append(pooled)
        valid.append(available)
    return (affinities[0], affinities[1]), (valid[0], valid[1])


def evaluate_fixed_p2_methods(
    pair: OviObjectPairView,
    view: DenseMethodView,
    bundle: D2InferenceBundle,
    outputs: AssociationForwardOutputs,
    ground_truth: GroundTruthPair,
    *,
    minimum_match_scores: Mapping[str, float],
    centroid_scale_m: float,
    static_centroid_tolerance_m: float,
    backend_config_sha256: str,
    checkpoint_sha256: str,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, object]]:
    """Build and evaluate all fixed-P2 G/F/R comparisons."""

    if set(minimum_match_scores) != set(_METHODS):
        raise DenseP2AssociationRunError("minimum scores must cover G/F/R")
    evidence = build_temporal_query_evidence(
        bundle,
        outputs,
        backend_config_sha256=backend_config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    assert evidence.token_scores is not None and evidence.query_scores is not None
    features, feature_valid = pool_dense_candidate_features(pair, view, bundle, outputs)
    affinities, affinity_valid = pool_dense_candidate_query_affinities(
        pair, view, bundle, evidence.token_scores
    )
    matrices = {
        "G_full": build_dense_geometric_score_matrix(
            pair, view, supported_only=False, centroid_scale_m=centroid_scale_m
        ),
        "G_supported": build_dense_geometric_score_matrix(
            pair, view, supported_only=True, centroid_scale_m=centroid_scale_m
        ),
        "F_obj": build_dense_feature_score_matrix(view, features, feature_valid),
        "R_obj": build_dense_rescene_score_matrix(
            view,
            query_scores=evidence.query_scores,
            candidate_affinities=affinities,
            candidate_valid=affinity_valid,
        ),
    }
    predictions = {
        method: solve_dense_pair_assignment(
            pair,
            view,
            matrices[method],
            minimum_match_score=float(minimum_match_scores[method]),
            static_centroid_tolerance_m=static_centroid_tolerance_m,
        )
        for method in _METHODS
    }
    bindings = {
        domain: build_dense_endpoint_bindings(
            pair, view, ground_truth, support_domain=domain
        )
        for domain in ("full", "supported")
    }
    rows = []
    for method in _METHODS:
        binding = bindings["full" if method == "G_full" else "supported"]
        for threshold in (0.50, 0.25):
            result = dict(
                evaluate_dense_association(
                    view,
                    ground_truth,
                    binding,
                    predictions[method],
                    iou_threshold=threshold,
                )
            )
            result.pop("outcomes")
            rows.append(result)
    metadata = {
        "method_view_sha256": view.content_sha256(),
        "candidate_counts": [len(visit) for visit in view.candidates],
        "candidate_pool_dependency": "joint_rescene_query_derived_P2",
        "matrix_sha256": {
            method: matrices[method].content_sha256() for method in _METHODS
        },
        "prediction_counts": {
            method: len(predictions[method]) for method in _METHODS
        },
        "binding_sha256": {
            domain: bindings[domain].content_sha256()
            for domain in ("full", "supported")
        },
    }
    return tuple(rows), metadata


def _restore_inputs(config_path: str | Path):
    path, config_content, config = _load_config(config_path)
    pair_id = config.get("pair_id")
    protocol = config["protocol"]
    if not isinstance(pair_id, str) or not pair_id:
        raise DenseP2AssociationRunError("pair ID is invalid")
    try:
        minimum_query_score = float(protocol["minimum_query_score"])
        minimum_entity_token_coverage = float(
            protocol["minimum_entity_token_coverage"]
        )
        minimum_source_point_coverage = float(
            protocol["minimum_source_point_coverage"]
        )
        point_chunk_size = int(protocol["point_chunk_size"])
        expected_query_count = int(protocol["expected_query_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise DenseP2AssociationRunError("dense protocol is invalid") from error
    loaded = {
        name: _bound_content(config["source_bindings"][name], label=name)
        for name in sorted(_SOURCE_BINDINGS)
    }
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
        raise DenseP2AssociationRunError("object-transfer parent bindings differ")
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
        pair, _json_object(loaded["d2_pair_receipt"][1], label="D2 pair receipt")
    )
    ground_truth = load_pair_ground_truth(loaded["ground_truth_manifest"][0])
    if pair.pair_id != pair_id or ground_truth.pair_id != pair_id:
        raise DenseP2AssociationRunError("pair, config, and ground truth IDs differ")
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=object_config.native_sampler_source.path,
        sampler_source_sha256=object_config.native_sampler_source.sha256,
        sampler_seed=object_config.sampler_seed,
        maximum_candidates=object_config.maximum_candidates,
        grid_sample_factory=_source_bound_grid_sample_factory(
            object_config.native_sampler_source.path
        ),
    )
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
    views, _readout, _runtimes = _build_method_views(
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
    return path, config_content, config, object_config, pair, ground_truth, bundle, cached, views["P2"]


def run_dense_p2_association(
    config_path: str | Path, *, evaluated_commit: str
) -> dict[str, object]:
    if len(evaluated_commit) != 40 or any(
        value not in "0123456789abcdef" for value in evaluated_commit
    ):
        raise DenseP2AssociationRunError("evaluated commit is invalid")
    (
        _path,
        config_content,
        config,
        object_config,
        pair,
        ground_truth,
        bundle,
        cached,
        view,
    ) = _restore_inputs(config_path)
    started = time.perf_counter()
    rows, metadata = evaluate_fixed_p2_methods(
        pair,
        view,
        bundle,
        cached,
        ground_truth,
        minimum_match_scores=object_config.minimum_match_scores,
        centroid_scale_m=object_config.centroid_scale_m,
        static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        backend_config_sha256=object_config.source_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
    )
    runtime_s = time.perf_counter() - started
    content = csv_bytes(rows, fieldnames=_IDENTITY_FIELDS)
    outputs = config["outputs"]
    local_root = _resolve_output(outputs.get("local_root"), label="local output") / "fixed_p2_association"
    tracked_root = _resolve_output(outputs.get("tracked_root"), label="tracked output") / "fixed_p2_association"
    if any(path.exists() or path.is_symlink() for path in (local_root, tracked_root)):
        raise DenseP2AssociationRunError("fixed-P2 association output already exists")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{local_root.name}.", dir=local_root.parent))
    try:
        _write_bytes(staging / "association_metrics.csv", content)
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_DENSE_FIXED_P2_ASSOCIATION_V1",
            "status": "PASS",
            "pair_id": pair.pair_id,
            "evaluated_commit": evaluated_commit,
            "config_sha256": hashlib.sha256(config_content).hexdigest(),
            "pair_content_sha256": pair.content_sha256(),
            "sample_content_sha256": bundle.sample.content_sha256(),
            "runtime_s": runtime_s,
            "protocol": {
                "support_domains": {
                    "G_full": "full",
                    "G_supported": "supported",
                    "F_obj": "supported",
                    "R_obj": "supported",
                },
                "minimum_match_scores": dict(object_config.minimum_match_scores),
                "centroid_scale_m": object_config.centroid_scale_m,
                "static_centroid_tolerance_m": object_config.static_centroid_tolerance_m,
                "ground_truth_used_by_method": False,
            },
            "fixed_p2": metadata,
            "table": _artifact_record(staging / "association_metrics.csv", root=staging),
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
        shutil.copyfile(local_root / "association_metrics.csv", tracked_staging / "association_metrics.csv")
        shutil.copyfile(local_root / "manifest.json", tracked_staging / "manifest.json")
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
    result = run_dense_p2_association(
        arguments.config, evaluated_commit=arguments.evaluated_commit
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DenseP2AssociationRunError",
    "evaluate_fixed_p2_methods",
    "pool_dense_candidate_features",
    "pool_dense_candidate_query_affinities",
    "run_dense_p2_association",
]
