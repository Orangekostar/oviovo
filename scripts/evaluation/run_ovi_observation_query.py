#!/usr/bin/env python3
"""Evaluate observation-query variants as raw, dense, and temporal outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LONG_TABLE_FIELDS = (
    "run_id",
    "method_id",
    "pair_id",
    "environment_uuid",
    "visit_id",
    "split_role",
    "previously_inspected",
    "input_domain",
    "evaluation_domain",
    "checkpoint_id",
    "seed",
    "raw_query_count",
    "retained_query_count",
    "final_instance_count",
    "raw_best_iou_mean",
    "raw_ar50",
    "raw_ar25",
    "tp50",
    "fp50",
    "fn50",
    "precision50",
    "recall50",
    "f1_50",
    "tp25",
    "fp25",
    "fn25",
    "precision25",
    "recall25",
    "f1_25",
    "fragment_count",
    "merge_count",
    "duplicate_count",
    "paired_identity_tp",
    "paired_identity_fp",
    "persistent_gt_count",
    "identity_precision",
    "identity_recall",
    "moved_identity_recall",
    "dense_point_count",
    "neural_supported_point_count",
    "residual_point_count",
    "unknown_point_count",
    "xyz_changed_count",
    "training_updates",
    "gpu_seconds",
    "inference_seconds",
    "peak_memory_bytes",
    "status",
    "unavailable_reason",
)
EVALUATION_DOMAINS = (
    "FULL_GT_LEGACY",
    "COMMON_M_SUPPORTED",
    "CAMERA_VISIBLE_DIAGNOSTIC",
    "RAW_QUERY_DIAGNOSTIC",
)


class ObservationEvaluationError(ValueError):
    """Raised when an observation evaluation is not source-bound."""


@dataclass(frozen=True, slots=True)
class _ForwardArrays:
    pred_masks_mq: np.ndarray
    pred_logits_qc: np.ndarray
    peak_memory_bytes: int


def _blank_row() -> dict[str, object]:
    return {field: None for field in LONG_TABLE_FIELDS}


def unavailable_result_rows(
    *,
    run_id: str,
    method_id: str,
    pair_id: str,
    environment_uuid: str,
    split_role: str,
    previously_inspected: bool,
    checkpoint_id: str | None,
    seed: int,
    reason: str,
) -> tuple[dict[str, object], ...]:
    rows = []
    for visit_id in (0, 1):
        for domain in EVALUATION_DOMAINS:
            row = _blank_row()
            row.update(
                {
                    "run_id": run_id,
                    "method_id": method_id,
                    "pair_id": pair_id,
                    "environment_uuid": environment_uuid,
                    "visit_id": visit_id,
                    "split_role": split_role,
                    "previously_inspected": previously_inspected,
                    "input_domain": (
                        "JOINT_NATIVE_M"
                        if method_id in {"OBS_BASE_FROZEN", "OBS_BASE_TUNED"}
                        else "JOINT_NATIVE_M_PLUS_OBSERVATION_R"
                    ),
                    "evaluation_domain": domain,
                    "checkpoint_id": checkpoint_id,
                    "seed": seed,
                    "status": "NOT_RUN_MISSING_TRAIN_ASSETS",
                    "unavailable_reason": reason,
                }
            )
            rows.append(row)
    return tuple(rows)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, object]]:
    source = Path(path).absolute()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationEvaluationError(f"{label} is unavailable") from error
    if not isinstance(value, dict):
        raise ObservationEvaluationError(f"{label} must be a JSON object")
    return source, value


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _artifact_record(path: Path) -> dict[str, object]:
    record = _file_record(path)
    record["path"] = path.name
    return record


def _split_pair(split: Mapping[str, object], role: str) -> Mapping[str, object]:
    environments = split.get("environments")
    matches = (
        [
            row
            for row in environments
            if isinstance(row, Mapping) and row.get("role") == role
        ]
        if isinstance(environments, list)
        else []
    )
    if len(matches) != 1:
        raise ObservationEvaluationError(f"split must contain exactly one {role} pair")
    return matches[0]


def _runtime_pair(runtime: Mapping[str, object], role: str) -> Mapping[str, object]:
    pairs = runtime.get("pairs")
    value = pairs.get(role.casefold()) if isinstance(pairs, Mapping) else None
    if not isinstance(value, Mapping):
        raise ObservationEvaluationError(f"runtime lacks its {role} pair")
    return value


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=LONG_TABLE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: "" if row[field] is None else row[field] for field in LONG_TABLE_FIELDS}
        for row in rows
    )
    return stream.getvalue().encode("utf-8")


def _publish(
    output: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    predictions: Mapping[str, np.ndarray] | None,
) -> None:
    if output.exists() or output.is_symlink():
        raise ObservationEvaluationError(f"evaluation run already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "results.csv").write_bytes(_csv_bytes(rows))
        if predictions is not None:
            with (staging / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **predictions)
        payload = dict(summary)
        source_bindings = dict(payload.get("source_bindings", {}))
        source_bindings.update(
            {
                "dense_adapter": _file_record(
                    REPO_ROOT / "src/oviv2/observation_query/dense_adapter.py"
                ),
                "dense_metrics": _file_record(
                    REPO_ROOT / "src/evaluation/dense_instance_repair_metrics.py"
                ),
                "evaluation_runner": _file_record(Path(__file__).absolute()),
            }
        )
        payload["source_bindings"] = source_bindings
        payload["artifacts"] = {
            "results": _artifact_record(staging / "results.csv"),
            "predictions": (
                None
                if predictions is None
                else _artifact_record(staging / "predictions.npz")
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_dense_inputs(legacy_config_path: Path):
    from scripts.evaluation import run_ovi_rescene_dense_instance_repair as legacy
    from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
        build_d2_inference_bundle,
        load_object_transfer_config,
    )
    from scripts.evaluation.run_ovi_rescene_object_recovery import (
        _bound_content,
        _json_object,
        _mapping_manifest_paths,
        _pair_record,
    )
    from src.evaluation.ovi_pair_artifact_loader import restore_bound_ovi_pair_artifact
    from src.evaluation.ovi_pair_views import build_ovi_object_pair_view
    from src.evaluation.rscan_gt_instances import load_pair_ground_truth

    _path, _content, config = legacy._load_config(legacy_config_path)
    bindings = config["source_bindings"]
    loaded = {
        name: _bound_content(bindings[name], label=name)
        for name in sorted(legacy._SOURCE_BINDINGS)
    }
    object_config = load_object_transfer_config(loaded["object_transfer_config"][0])
    selection = _json_object(loaded["selection_manifest"][1], label="selection")
    pair_record = _pair_record(selection, str(config["pair_id"]))
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
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=object_config.native_sampler_source.path,
        sampler_source_sha256=object_config.native_sampler_source.sha256,
        sampler_seed=object_config.sampler_seed,
        maximum_candidates=object_config.maximum_candidates,
        grid_sample_factory=legacy._source_bound_grid_sample_factory(
            object_config.native_sampler_source.path
        ),
    )
    ground_truth = load_pair_ground_truth(loaded["ground_truth_manifest"][0])
    return pair, bundle, ground_truth


def _voxel_membership(
    points: np.ndarray, domain_points: np.ndarray, voxel_size_m: float
) -> np.ndarray:
    dense = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m).astype(
        np.int64
    )
    domain = np.floor(
        np.asarray(domain_points, dtype=np.float64) / voxel_size_m
    ).astype(np.int64)
    packed = np.dtype((np.void, dense.dtype.itemsize * dense.shape[1]))
    dense_rows = np.ascontiguousarray(dense).view(packed).reshape(-1)
    domain_rows = np.ascontiguousarray(domain).view(packed).reshape(-1)
    return np.isin(dense_rows, np.unique(domain_rows))


def _camera_visible_masks(
    pair: object,
    pair_record: Mapping[str, object],
    artifact_root: Path,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    path = artifact_root / "d1_camera_visible" / "arrays.npz"
    try:
        with np.load(path, allow_pickle=False) as arrays:
            support = (
                np.asarray(arrays["visit0_support_mask"], dtype=np.bool_),
                np.asarray(arrays["visit1_support_mask"], dtype=np.bool_),
            )
    except (OSError, KeyError, ValueError) as error:
        raise ObservationEvaluationError(
            "camera-visible diagnostic arrays are invalid"
        ) from error
    sessions = pair_record.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 2:
        raise ObservationEvaluationError("camera-visible split sessions are invalid")
    result = []
    for visit_id in (0, 1):
        session = sessions[visit_id]
        processed = (
            session.get("processed_points") if isinstance(session, Mapping) else None
        )
        processed_path = (
            Path(str(processed.get("path"))).absolute()
            if isinstance(processed, Mapping)
            else Path()
        )
        try:
            source_points = np.load(processed_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ObservationEvaluationError(
                "camera-visible processed source points are unavailable"
            ) from error
        if (
            source_points.ndim != 2
            or source_points.shape[1] < 3
            or len(source_points) != len(support[visit_id])
        ):
            raise ObservationEvaluationError(
                "camera-visible support differs from its D0 processed source"
            )
        result.append(
            _voxel_membership(
                pair.visits[visit_id].points_xyz,
                np.asarray(source_points[support[visit_id], :3]),
                voxel_size_m,
            )
        )
    return result[0], result[1]


def _geometry_metrics(pair, view, ground_truth, domain_masks):
    from src.evaluation.rscan_gt_instances import (
        GroundTruthInstance,
        PredictedInstance,
        evaluate_instance_geometry,
        voxelize_points,
    )

    rows = []
    for visit_id in (0, 1):
        visit = pair.visits[visit_id]
        domain = domain_masks[visit_id]
        domain_voxels = voxelize_points(
            visit.points_xyz[np.flatnonzero(domain)],
            voxel_size_m=ground_truth.voxel_size_m,
        )
        targets = tuple(
            GroundTruthInstance(target.instance_id, target.semantic_label, voxels)
            for target in ground_truth.visits[visit_id]
            if (voxels := target.voxels & domain_voxels)
        )
        predictions = []
        for candidate in view.candidates[visit_id]:
            points = candidate.point_indices[domain[candidate.point_indices]]
            if len(points):
                predictions.append(
                    PredictedInstance(
                        candidate.candidate_id,
                        voxelize_points(
                            visit.points_xyz[points],
                            voxel_size_m=ground_truth.voxel_size_m,
                        ),
                    )
                )
        result = evaluate_instance_geometry(
            tuple(predictions), targets, matching_policy="max_valid_count_then_iou"
        )

        def values(
            threshold,
            prediction_count: int = len(predictions),
            target_count: int = len(targets),
        ):
            tp = threshold.matched_count
            precision = None if not prediction_count else tp / prediction_count
            recall = None if not target_count else tp / target_count
            f1 = (
                None
                if precision is None or recall is None or precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            return {
                "tp": tp,
                "fp": prediction_count - tp,
                "fn": target_count - tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        rows.append(
            {
                "primary": values(result.primary),
                "sensitivity": values(result.sensitivity),
                "final_instance_count": len(predictions),
                "fragment_count": result.primary.fragment_gt_count,
                "merge_count": result.primary.merge_prediction_count,
                "duplicate_count": result.primary.duplicate_prediction_count,
            }
        )
    return rows[0], rows[1]


def _raw_metrics(pair, readout, ground_truth, domain_masks):
    from src.evaluation.rscan_gt_instances import voxelize_points

    rows = []
    for visit_id in (0, 1):
        visit = pair.visits[visit_id]
        domain = domain_masks[visit_id]
        domain_voxels = voxelize_points(
            visit.points_xyz[np.flatnonzero(domain)],
            voxel_size_m=ground_truth.voxel_size_m,
        )
        targets = [
            target.voxels & domain_voxels for target in ground_truth.visits[visit_id]
        ]
        targets = [value for value in targets if value]
        proposals = []
        for proposal in readout.raw_proposals[visit_id]:
            points = proposal.point_indices[domain[proposal.point_indices]]
            if len(points):
                proposals.append(
                    voxelize_points(
                        visit.points_xyz[points], voxel_size_m=ground_truth.voxel_size_m
                    )
                )
        best = [
            max(
                (
                    len(target & prediction) / len(target | prediction)
                    for prediction in proposals
                ),
                default=0.0,
            )
            for target in targets
        ]
        rows.append(
            {
                "raw_best_iou_mean": None if not best else float(np.mean(best)),
                "raw_ar50": None
                if not best
                else sum(value >= 0.50 for value in best) / len(best),
                "raw_ar25": None
                if not best
                else sum(value >= 0.25 for value in best) / len(best),
            }
        )
    return rows[0], rows[1]


def _identity_metrics(pair, view, ground_truth, support_domain: str):
    from src.evaluation.dense_instance_repair_metrics import (
        build_dense_endpoint_bindings,
        evaluate_dense_identity,
    )

    binding = build_dense_endpoint_bindings(
        pair, view, ground_truth, support_domain=support_domain
    )
    return evaluate_dense_identity(view, ground_truth, binding, iou_threshold=0.50)


def _actual_rows(
    *,
    run_id: str,
    method_id: str,
    pair_record: Mapping[str, object],
    checkpoint_id: str,
    seed: int,
    pair,
    layers,
    ground_truth,
    camera_masks,
    training_updates: int,
    inference_seconds: float,
    peak_memory_bytes: int,
) -> tuple[dict[str, object], ...]:
    full_masks = tuple(
        np.ones(visit.point_count, dtype=np.bool_) for visit in pair.visits
    )
    common_masks = tuple(visit.neural_valid for visit in layers.readout.visits)
    geometry = {
        "FULL_GT_LEGACY": _geometry_metrics(
            pair, layers.method_view, ground_truth, full_masks
        ),
        "COMMON_M_SUPPORTED": _geometry_metrics(
            pair, layers.method_view, ground_truth, common_masks
        ),
        "CAMERA_VISIBLE_DIAGNOSTIC": _geometry_metrics(
            pair, layers.method_view, ground_truth, camera_masks
        ),
    }
    raw = _raw_metrics(pair, layers.readout, ground_truth, full_masks)
    identities = {
        "FULL_GT_LEGACY": _identity_metrics(
            pair, layers.method_view, ground_truth, "full"
        ),
        "COMMON_M_SUPPORTED": _identity_metrics(
            pair, layers.method_view, ground_truth, "supported"
        ),
    }
    rows = []
    for visit_id in (0, 1):
        visit = layers.readout.visits[visit_id]
        for domain in EVALUATION_DOMAINS:
            row = _blank_row()
            row.update(
                {
                    "run_id": run_id,
                    "method_id": method_id,
                    "pair_id": pair.pair_id,
                    "environment_uuid": pair_record["environment_uuid"],
                    "visit_id": visit_id,
                    "split_role": pair_record["role"],
                    "previously_inspected": pair_record["previously_inspected"],
                    "input_domain": (
                        "JOINT_NATIVE_M"
                        if method_id in {"OBS_BASE_FROZEN", "OBS_BASE_TUNED"}
                        else "JOINT_NATIVE_M_PLUS_OBSERVATION_R"
                    ),
                    "evaluation_domain": domain,
                    "checkpoint_id": checkpoint_id,
                    "seed": seed,
                    "raw_query_count": layers.raw_query_count,
                    "retained_query_count": layers.retained_query_count,
                    "dense_point_count": len(visit.owner_instance_indices),
                    "neural_supported_point_count": int(
                        np.count_nonzero(visit.neural_valid)
                    ),
                    "residual_point_count": int(
                        np.count_nonzero(visit.owner_source_codes == 2)
                    ),
                    "unknown_point_count": int(
                        np.count_nonzero(visit.owner_source_codes == 4)
                    ),
                    "xyz_changed_count": layers.xyz_changed_count,
                    "training_updates": training_updates,
                    "gpu_seconds": inference_seconds,
                    "inference_seconds": inference_seconds,
                    "peak_memory_bytes": peak_memory_bytes,
                    "status": "PASS",
                }
            )
            if domain == "RAW_QUERY_DIAGNOSTIC":
                row.update(raw[visit_id])
                row["unavailable_reason"] = (
                    "final_and_identity_fields_not_applicable_to_raw_queries"
                )
            else:
                result = geometry[domain][visit_id]
                primary = result["primary"]
                sensitivity = result["sensitivity"]
                row.update(
                    {
                        "final_instance_count": result["final_instance_count"],
                        "tp50": primary["tp"],
                        "fp50": primary["fp"],
                        "fn50": primary["fn"],
                        "precision50": primary["precision"],
                        "recall50": primary["recall"],
                        "f1_50": primary["f1"],
                        "tp25": sensitivity["tp"],
                        "fp25": sensitivity["fp"],
                        "fn25": sensitivity["fn"],
                        "precision25": sensitivity["precision"],
                        "recall25": sensitivity["recall"],
                        "f1_25": sensitivity["f1"],
                        "fragment_count": result["fragment_count"],
                        "merge_count": result["merge_count"],
                        "duplicate_count": result["duplicate_count"],
                    }
                )
                identity = identities.get(domain)
                if identity is None:
                    row["unavailable_reason"] = (
                        "temporal_identity_not_defined_for_camera_visible_diagnostic"
                    )
                else:
                    row.update(
                        {
                            "paired_identity_tp": identity["true_positive_count"],
                            "paired_identity_fp": identity["false_positive_count"],
                            "persistent_gt_count": identity["persistent_gt_count"],
                            "identity_precision": identity["precision"],
                            "identity_recall": identity["end_to_end_recall"],
                            "moved_identity_recall": identity["rigid_recall"],
                        }
                    )
            rows.append(row)
    return tuple(rows)


def _native_predictions(model_input, runtime: Mapping[str, object]):
    from scripts.evaluation.rescene_pair_executor import _native_forward

    checkouts = runtime["checkouts"]
    assets = runtime["assets"]
    os.environ["CONCERTO_CHECKPOINT"] = str(assets["concerto_checkpoint"])
    return _native_forward(
        model_input,
        Path(str(checkouts["rescene"])),
        Path(str(assets["rescene_checkpoint"])),
    )


def _checkpoint_directory(path: Path) -> Path:
    for candidate in (path, path / "checkpoint"):
        if (candidate / "manifest.json").is_file() and (
            candidate / "trainable.safetensors"
        ).is_file():
            return candidate
    raise ObservationEvaluationError(
        "checkpoint must contain manifest.json and trainable.safetensors"
    )


def _validate_checkpoint_evaluation_contract(
    *,
    metadata: object,
    config: Mapping[str, object],
    method: str,
    config_path: Path,
    split_path: Path,
    split_id: str,
    observation_sha256: str,
) -> None:
    resolved = getattr(metadata, "resolved_config", None)
    expected = {
        "method": method,
        "model": config.get("model"),
        "loss": config.get("loss"),
        "training": config.get("training"),
        "method_config": config.get("methods", {}).get(method),
    }
    if not isinstance(resolved, Mapping) or any(
        resolved.get(name) != value for name, value in expected.items()
    ):
        raise ObservationEvaluationError(
            "checkpoint resolved training config differs from evaluation"
        )
    training = config.get("training")
    if (
        getattr(metadata, "model_variant", None) != method
        or getattr(metadata, "split_id", None) != split_id
        or getattr(metadata, "observation_sha256", None) != observation_sha256
        or not isinstance(training, Mapping)
        or getattr(metadata, "seed", None) != training.get("seed")
    ):
        raise ObservationEvaluationError(
            "checkpoint data identity differs from evaluation"
        )
    bindings = resolved.get("source_bindings")
    sources = {
        "config": config_path,
        "split": split_path,
        "model": REPO_ROOT / "src/oviv2/observation_query/model.py",
        "losses": REPO_ROOT / "src/oviv2/observation_query/losses.py",
        "training_state": REPO_ROOT / "src/oviv2/observation_query/training.py",
    }
    if not isinstance(bindings, Mapping):
        raise ObservationEvaluationError("checkpoint source bindings are unavailable")
    for name, path in sources.items():
        record = bindings.get(name)
        actual = _file_record(path)
        if not isinstance(record, Mapping) or any(
            record.get(field) != actual[field] for field in ("sha256", "byte_count")
        ):
            raise ObservationEvaluationError(
                f"checkpoint {name} source binding differs from evaluation"
            )


def _trained_predictions(
    *,
    model_input: object,
    observations: object,
    runtime: Mapping[str, object],
    config: Mapping[str, object],
    config_path: Path,
    split_path: Path,
    split_id: str,
    method: str,
    checkpoint_path: Path,
) -> tuple[_ForwardArrays, float, int, str, dict[str, object]]:
    import torch

    from scripts.training.train_ovi_observation_query import (
        _build_native_model,
        _build_point,
        _configure_observation_trainables,
        _criterion_from_config,
        _metadata_from_manifest,
        training_method_contract,
    )
    from src.oviv2.observation_query.model import ObservationReScene
    from src.oviv2.observation_query.training import load_trainable_checkpoint

    contract = training_method_contract(method)
    checkpoint_root = _checkpoint_directory(checkpoint_path)
    manifest_path = checkpoint_root / "manifest.json"
    weights_path = checkpoint_root / "trainable.safetensors"
    metadata = _metadata_from_manifest(manifest_path)
    _validate_checkpoint_evaluation_contract(
        metadata=metadata,
        config=config,
        method=method,
        config_path=config_path,
        split_path=split_path,
        split_id=split_id,
        observation_sha256=observations.content_sha256(),
    )
    model_config = config.get("model")
    loss_config = config.get("loss")
    if not isinstance(model_config, Mapping) or not isinstance(loss_config, Mapping):
        raise ObservationEvaluationError("model and loss configs must be mappings")
    device = torch.device(os.environ.get("RESCENE_DEVICE", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ObservationEvaluationError("trained evaluation requires CUDA")
    torch.cuda.set_device(device)
    native, base_checkpoint, _concerto, _commit = _build_native_model(
        runtime=runtime, device=device
    )
    if metadata.base_checkpoint_sha256 != _file_record(base_checkpoint)["sha256"]:
        raise ObservationEvaluationError("checkpoint base model differs from runtime")
    model = ObservationReScene(
        native,
        observation_feature_dim=int(observations.region_features.shape[1]),
        metadata_dim=int(observations.region_metadata.shape[1]),
        alpha_initial=float(model_config.get("alpha_initial", 0.001)),
        beta_initial=float(model_config.get("beta_initial", 0.0)),
        fuse_initial=float(model_config.get("fuse_initial", 0.001)),
    ).to(device)
    _configure_observation_trainables(model, contract)
    criterion = _criterion_from_config(
        loss_config,
        method=method,
        num_queries=int(native.num_queries),
        device=device,
    )
    load_trainable_checkpoint(
        model=model,
        criterion=criterion,
        checkpoint_root=checkpoint_root,
        expected_metadata=metadata,
    )
    point, point2segment = _build_point(model_input, device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            point,
            point2segment=point2segment,
            raw_coordinates=point.raw_coordinates,
            is_eval=True,
            observations=(observations if contract.uses_region_supervision else None),
            observation_mode=contract.observation_mode,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    logits = output["pred_logits"].detach().cpu().numpy()
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ObservationEvaluationError("trained class output must have one pair batch")
    forward = _ForwardArrays(
        pred_masks_mq=np.asarray(output["pred_masks"][0].detach().cpu().numpy()),
        pred_logits_qc=np.asarray(logits[0]),
        peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
    checkpoint_id = str(_file_record(weights_path)["sha256"])
    return (
        forward,
        inference_seconds,
        metadata.optimizer_updates,
        checkpoint_id,
        {
            "checkpoint_manifest": _file_record(manifest_path),
            "checkpoint_weights": _file_record(weights_path),
        },
    )


def _historical_prediction_comparison(
    current_masks: np.ndarray,
    current_logits: np.ndarray,
    legacy_config_path: Path,
) -> dict[str, object]:
    _path, config = _load_json(legacy_config_path, "legacy dense config")
    bindings = config.get("source_bindings")
    binding = (
        bindings.get("d2_forward_arrays") if isinstance(bindings, Mapping) else None
    )
    historical_path = (
        Path(str(binding.get("path"))).absolute()
        if isinstance(binding, Mapping)
        else Path()
    )
    try:
        with np.load(historical_path, allow_pickle=False) as arrays:
            historical_masks = np.asarray(arrays["pred_masks_mq"])
            historical_logits = np.asarray(arrays["pred_logits_qc"])
    except (OSError, KeyError, ValueError) as error:
        raise ObservationEvaluationError(
            "historical raw prediction cache is unavailable"
        ) from error
    if (
        historical_masks.shape != current_masks.shape
        or historical_logits.shape != current_logits.shape
    ):
        raise ObservationEvaluationError("historical raw prediction shape changed")
    mask_difference = np.abs(current_masks - historical_masks)
    class_difference = np.abs(current_logits - historical_logits)
    exact = np.array_equal(current_masks, historical_masks) and np.array_equal(
        current_logits, historical_logits
    )
    return {
        "status": "EXACT" if exact else "NOT_EXACT_CURRENT_RERUN",
        "historical_arrays": _file_record(historical_path),
        "maximum_absolute_mask_logit_difference": float(mask_difference.max()),
        "mean_absolute_mask_logit_difference": float(mask_difference.mean()),
        "maximum_absolute_class_logit_difference": float(class_difference.max()),
        "mean_absolute_class_logit_difference": float(class_difference.mean()),
    }


def run_evaluation(
    *,
    config_path: str | Path,
    runtime_path: str | Path,
    method: str,
    checkpoint: str | None,
    role: str,
    run_id: str,
) -> int:
    config_source, config = _load_json(config_path, "pilot config")
    runtime_source, runtime = _load_json(runtime_path, "runtime config")
    if method not in config.get("methods", {}):
        raise ObservationEvaluationError("method is not declared in pilot config")
    split_source, split = _load_json(str(config["split_manifest"]), "split manifest")
    pair_record = _split_pair(split, role)
    runtime_pair = _runtime_pair(runtime, role)
    if runtime_pair.get("pair_id") != pair_record.get("pair_id"):
        raise ObservationEvaluationError("runtime and split pair differ")
    cache_root = Path(str(runtime["cache_root"])).absolute()
    output = cache_root / "evaluation_runs" / run_id
    seed = int(config["training"]["seed"])
    artifact_root_value = runtime_pair.get("artifact_root")
    artifact_root = (
        Path(str(artifact_root_value)).absolute() if artifact_root_value else None
    )
    trained = bool(config["methods"][method].get("trained"))
    checkpoint_path = None if checkpoint is None else Path(checkpoint).absolute()
    if trained and (checkpoint_path is None or not checkpoint_path.is_dir()):
        reason = (
            "trained checkpoint is unavailable because declared TRAIN RGB-D sequence.zip "
            "assets are absent; DEV was not substituted for TRAIN"
        )
        rows = unavailable_result_rows(
            run_id=run_id,
            method_id=method,
            pair_id=str(pair_record["pair_id"]),
            environment_uuid=str(pair_record["environment_uuid"]),
            split_role=role,
            previously_inspected=bool(pair_record["previously_inspected"]),
            checkpoint_id=None,
            seed=seed,
            reason=reason,
        )
        _publish(
            output,
            rows=rows,
            predictions=None,
            summary={
                "schema_version": 1,
                "artifact_id": "OVI_RESCENE_OBSERVATION_QUERY_EVALUATION_V1",
                "status": "NOT_RUN_MISSING_TRAIN_ASSETS",
                "run_id": run_id,
                "method_id": method,
                "reason": reason,
                "source_bindings": {
                    "config": _file_record(config_source),
                    "runtime": _file_record(runtime_source),
                    "split": _file_record(split_source),
                },
            },
        )
        return 2
    if artifact_root is None:
        raise ObservationEvaluationError("runtime pair lacks artifact_root")
    from scripts.evaluation.prepare_ovi_observations import (
        load_model_observation_bundle,
    )
    from src.oviv2.observation_query.contracts import load_observation_bank
    from src.oviv2.observation_query.dense_adapter import (
        build_observation_composer_inputs,
        build_observation_dense_layers,
    )

    model_input, _points, _center, model_manifest = load_model_observation_bundle(
        artifact_root / "model_bundle"
    )
    observations = load_observation_bank(artifact_root / "observation_bank" / "bank")
    legacy_path = Path(
        str(
            runtime.get("assets", {}).get(
                "legacy_dense_config",
                REPO_ROOT
                / "configs/evaluation/ovi_rescene_dense_instance_repair_v1.json",
            )
        )
    )
    pair, bundle, ground_truth = _load_dense_inputs(legacy_path)
    if bundle.model_input.content_sha256() != model_input.content_sha256():
        raise ObservationEvaluationError(
            "dense evaluator and observation M input differ"
        )
    checkpoint_bindings: dict[str, object] = {}
    if method == "OBS_BASE_FROZEN":
        started = time.perf_counter()
        forward = _native_predictions(model_input, runtime)
        inference_seconds = time.perf_counter() - started
        training_updates = 0
        checkpoint_id = str(
            _file_record(Path(str(runtime["assets"]["rescene_checkpoint"])))[
                "sha256"
            ]
        )
    else:
        assert checkpoint_path is not None
        (
            forward,
            inference_seconds,
            training_updates,
            checkpoint_id,
            checkpoint_bindings,
        ) = _trained_predictions(
            model_input=model_input,
            observations=observations,
            runtime=runtime,
            config=config,
            config_path=config_source,
            split_path=split_source,
            split_id=str(split.get("artifact_id", "splits_v1")),
            method=method,
            checkpoint_path=checkpoint_path,
        )
    layers = build_observation_dense_layers(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=forward.pred_masks_mq,
        pred_logits_qc=forward.pred_logits_qc,
        method_id=method,
        minimum_query_score=float(config["dense_readout"]["query_score_threshold"]),
    )
    composer_inputs = build_observation_composer_inputs(pair, layers)
    camera_masks = _camera_visible_masks(
        pair,
        pair_record,
        artifact_root,
        voxel_size_m=ground_truth.voxel_size_m,
    )
    historical_comparison = (
        _historical_prediction_comparison(
            forward.pred_masks_mq, forward.pred_logits_qc, legacy_path
        )
        if method == "OBS_BASE_FROZEN"
        else {"status": "NOT_APPLICABLE_TRAINED_METHOD"}
    )
    rows = _actual_rows(
        run_id=run_id,
        method_id=method,
        pair_record=pair_record,
        checkpoint_id=str(checkpoint_id),
        seed=seed,
        pair=pair,
        layers=layers,
        ground_truth=ground_truth,
        camera_masks=camera_masks,
        training_updates=training_updates,
        inference_seconds=inference_seconds,
        peak_memory_bytes=forward.peak_memory_bytes,
    )
    predictions = {
        "pred_masks_mq": forward.pred_masks_mq,
        "pred_logits_qc": forward.pred_logits_qc,
        "raw_query_indices": layers.readout.raw_query_indices,
        "query_scores": layers.readout.query_scores,
        "visit0_owner_instance_indices": layers.readout.visits[
            0
        ].owner_instance_indices,
        "visit0_owner_source_codes": layers.readout.visits[0].owner_source_codes,
        "visit1_owner_instance_indices": layers.readout.visits[
            1
        ].owner_instance_indices,
        "visit1_owner_source_codes": layers.readout.visits[1].owner_source_codes,
    }
    identity_recall = next(
        row["identity_recall"]
        for row in rows
        if row["visit_id"] == 0 and row["evaluation_domain"] == "FULL_GT_LEGACY"
    )
    current_map_status = (
        "INSTANCE_CONSTRUCTION_PASS_MAP_EFFECT_NOT_EVALUABLE"
        if not identity_recall
        else "INSTANCE_CONSTRUCTION_PASS_MAP_EFFECT_PENDING_VISIBILITY_EVALUATION"
    )
    _publish(
        output,
        rows=rows,
        predictions=predictions,
        summary={
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_QUERY_EVALUATION_V1",
            "status": "PASS",
            "run_id": run_id,
            "method_id": method,
            "pair_id": pair.pair_id,
            "model_input_sha256": model_input.content_sha256(),
            "model_bundle_manifest": model_manifest,
            "dense_layers_sha256": layers.content_sha256(),
            "current_map_adapter": {
                "status": current_map_status,
                "composer_inputs_sha256": composer_inputs.content_sha256(),
                "visit_map_sha256": [
                    value.snapshot_sha256 for value in composer_inputs.visit_maps
                ],
                "candidate_relation_count": len(composer_inputs.relations),
                "candidate_relation_state": "uncertain",
                "source_d_row_count": [
                    value.point_count for value in pair.visits
                ],
                "unknown_source_d_row_count": [
                    len(value) for value in composer_inputs.unknown_source_d_rows
                ],
                "full_domain_identity_recall": identity_recall,
                "reason": (
                    "The real DEV readout has zero end-to-end identity recall at "
                    "IoU 0.50; candidate-only shared queries were not promoted to "
                    "confirmed persistent identities or used to claim a map effect."
                ),
            },
            "historical_raw_prediction_comparison": historical_comparison,
            "claim_boundary": (
                "Frozen native checkpoint diagnostic; this is not a trained "
                "observation-query result."
                if method == "OBS_BASE_FROZEN"
                else "Trained observation-query output on the declared evaluation role."
            ),
            "source_bindings": {
                "config": _file_record(config_source),
                "runtime": _file_record(runtime_source),
                "split": _file_record(split_source),
                "legacy_dense_config": _file_record(legacy_path),
                **checkpoint_bindings,
            },
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--role", required=True, choices=("DEV", "CONFIRM", "LEGACY_REGRESSION")
    )
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_evaluation(
        config_path=args.config,
        runtime_path=args.runtime,
        method=args.method,
        checkpoint=args.checkpoint,
        role=args.role,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LONG_TABLE_FIELDS",
    "ObservationEvaluationError",
    "run_evaluation",
    "unavailable_result_rows",
]
