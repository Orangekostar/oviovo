#!/usr/bin/env python3
"""Run a real camera-visible ReScene diagnostic for one 3RScan pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.prepare_ovi_observations import (
    RealFrameAssets,
    find_pair_record,
    load_real_pair_frames,
    resolve_runtime_pair,
)
from src.core.data_structures import CameraIntrinsics
from src.oviv2.observation_query.observations import (
    DepthRelation,
    classify_depth_relations,
    neighborhood_depth_reliability,
)
from src.oviv2.ovi_surface_attributes import project_world_points


class CameraVisibleDiagnosticError(ValueError):
    """Raised when the camera-visible diagnostic loses input identity."""


@dataclass(frozen=True, slots=True)
class CameraVisibleSupport:
    support_mask: np.ndarray
    support_frame_counts: np.ndarray
    relation_counts: dict[str, int]


def resolve_diagnostic_pair_runtime(
    runtime_config: Mapping[str, object], pair_id: str
) -> Mapping[str, object]:
    """Resolve the explicit DEV pair used by the camera-visible diagnostic."""

    return resolve_runtime_pair(runtime_config, pair_id=pair_id, role="DEV")


def build_camera_visible_support_mask(
    *,
    points_reference_xyz: object,
    frames: Iterable[RealFrameAssets],
    visit_id: int,
    tolerance_m: float,
    minimum_valid_neighbours: int,
) -> CameraVisibleSupport:
    """Union positive depth-supported native points over declared real cameras."""

    points = np.asarray(points_reference_xyz, dtype=np.float64)
    frame_values = tuple(frames)
    if points.ndim != 2 or points.shape[1:] != (3,) or np.any(~np.isfinite(points)):
        raise CameraVisibleDiagnosticError("native points must have finite shape (N, 3)")
    if not len(points):
        raise CameraVisibleDiagnosticError("native point set must be non-empty")
    if visit_id not in (0, 1) or isinstance(visit_id, bool):
        raise CameraVisibleDiagnosticError("visit_id must be 0 or 1")
    if not frame_values or any(
        not isinstance(frame, RealFrameAssets) or frame.visit_id != visit_id
        for frame in frame_values
    ):
        raise CameraVisibleDiagnosticError("frames must belong to the requested visit")

    counts = np.zeros(len(points), dtype=np.int16)
    relation_counts: Counter[str] = Counter()
    for frame in frame_values:
        intrinsic = CameraIntrinsics(
            float(frame.depth_intrinsic[0, 0]),
            float(frame.depth_intrinsic[1, 1]),
            float(frame.depth_intrinsic[0, 2]),
            float(frame.depth_intrinsic[1, 2]),
            int(frame.depth_m.shape[1]),
            int(frame.depth_m.shape[0]),
        )
        projection = project_world_points(
            points, frame.camera_to_reference, intrinsic
        )
        observed = np.full(len(points), np.nan, dtype=np.float64)
        inside = projection.projectable
        observed[inside] = frame.depth_m[
            projection.rows[inside], projection.columns[inside]
        ]
        reliable = neighborhood_depth_reliability(
            frame.depth_m,
            projection.rows,
            projection.columns,
            tolerance_m=tolerance_m,
            minimum_valid_neighbours=minimum_valid_neighbours,
        )
        relations = classify_depth_relations(
            observed_depth_m=observed,
            projected_depth_m=projection.camera_xyz[:, 2],
            neighborhood_reliable=reliable,
            tolerance_m=tolerance_m,
        )
        counts += relations == DepthRelation.SUPPORTED
        relation_counts.update(
            {
                relation.name.lower(): int(np.count_nonzero(relations == relation))
                for relation in DepthRelation
            }
        )
    mask = counts > 0
    mask.setflags(write=False)
    counts.setflags(write=False)
    return CameraVisibleSupport(
        support_mask=mask,
        support_frame_counts=counts,
        relation_counts={
            relation.name.lower(): int(relation_counts[relation.name.lower()])
            for relation in DepthRelation
        },
    )


def filter_native_sample_by_visibility(
    sample: object,
    *,
    support_masks: Sequence[object],
) -> tuple[object, ...]:
    """Filter every point-aligned native sample field with the camera union."""

    if not isinstance(sample, tuple) or len(sample) != 9:
        raise CameraVisibleDiagnosticError("native dataset sample schema is invalid")
    if isinstance(support_masks, (str, bytes)) or len(support_masks) != 2:
        raise CameraVisibleDiagnosticError("support_masks must contain two visits")
    masks = tuple(np.asarray(value) for value in support_masks)
    if any(mask.ndim != 1 or mask.dtype != np.bool_ or not np.any(mask) for mask in masks):
        raise CameraVisibleDiagnosticError("each visit support mask must be non-empty boolean")
    raw_coordinates = np.asarray(sample[6])
    total = sum(len(mask) for mask in masks)
    if raw_coordinates.ndim != 2 or raw_coordinates.shape != (total, 4):
        raise CameraVisibleDiagnosticError("raw native coordinates do not match support masks")
    expected_visits = np.concatenate(
        (
            np.zeros(len(masks[0]), dtype=np.int8),
            np.ones(len(masks[1]), dtype=np.int8),
        )
    )
    if not np.array_equal(raw_coordinates[:, 3], expected_visits):
        raise CameraVisibleDiagnosticError("native sample visit order is invalid")
    selected = np.concatenate(masks)
    result = list(sample)
    for index in (0, 1, 2, 4, 5, 6):
        values = np.asarray(sample[index])
        if len(values) != total:
            raise CameraVisibleDiagnosticError(
                f"native sample field {index} is not point-aligned"
            )
        result[index] = np.array(values[selected], copy=True, order="C")
    return tuple(result)


def strip_native_ground_truth(sample: object) -> tuple[object, ...]:
    """Replace supervised columns while retaining only method-visible mesh segments."""

    if not isinstance(sample, tuple) or len(sample) != 9:
        raise CameraVisibleDiagnosticError("native dataset sample schema is invalid")
    labels = np.asarray(sample[2])
    if labels.ndim != 2 or labels.shape[1] != 4 or not np.issubdtype(
        labels.dtype, np.integer
    ):
        raise CameraVisibleDiagnosticError("native labels must contain four integer columns")
    method_labels = np.zeros(labels.shape, dtype=labels.dtype)
    method_labels[:, 0] = 255
    method_labels[:, 1] = labels[:, 3]
    method_labels[:, 2] = 0
    method_labels[:, 3] = labels[:, 3]
    result = list(sample)
    result[2] = method_labels
    return tuple(result)


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


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CameraVisibleDiagnosticError(f"{label} is unavailable") from error
    if not isinstance(value, dict):
        raise CameraVisibleDiagnosticError(f"{label} must be a JSON object")
    return value


def _file_record(path: Path, recorded_path: str | None = None) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path if recorded_path is None else recorded_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _validate_native_raw_sample(pair: object, sample: object) -> None:
    if not isinstance(sample, tuple) or len(sample) != 9:
        raise CameraVisibleDiagnosticError("native dataset sample schema is invalid")
    visits = pair.visits
    expected_points = np.concatenate([visit.points_xyz for visit in visits], axis=0)
    expected_rgb = np.concatenate([visit.rgb for visit in visits], axis=0)
    expected_normals = np.concatenate([visit.normals_xyz for visit in visits], axis=0)
    expected_visits = np.concatenate(
        [
            np.full(visit.point_count, visit.visit_id, dtype=np.int8)
            for visit in visits
        ]
    )
    second_segments = visits[1].segment_ids + int(np.max(visits[0].segment_ids)) + 1
    expected_segments = np.concatenate(
        (visits[0].segment_ids, second_segments)
    ).astype(np.int64)
    raw_coordinates = np.asarray(sample[6])
    labels = np.asarray(sample[2])
    if (
        raw_coordinates.shape != (len(expected_points), 4)
        or labels.shape != (len(expected_points), 4)
        or not np.array_equal(raw_coordinates[:, :3], expected_points)
        or not np.array_equal(raw_coordinates[:, 3], expected_visits)
        or not np.array_equal(np.asarray(sample[4]), expected_rgb)
        or not np.array_equal(np.asarray(sample[5]), expected_normals)
        or not np.array_equal(labels[:, -1], expected_segments)
    ):
        raise CameraVisibleDiagnosticError(
            "native dataset method arrays differ from the frozen GT-free pair view"
        )


def run_camera_visible_diagnostic(
    *,
    runtime_config_path: str | Path,
    pilot_config_path: str | Path,
    matrix_config_path: str | Path,
    selection_manifest_path: str | Path,
    mapping_receipt_path: str | Path,
    observation_artifact_root: str | Path,
    output_root: str | Path,
    pair_id: str | None = None,
) -> dict[str, object]:
    """Execute one uncached native ReScene forward on real camera-visible points."""

    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from scripts.evaluation.run_3rscan_t2_matrix import (
        _load_config,
        _move_tensors,
        _seed_everything,
        _validated_checkout,
        _validated_directory,
        _validated_external_record,
    )
    from src.evaluation.rscan_method_views import build_method_pair_view_from_manifest
    from src.oviv2.observation_query.contracts import load_observation_bank

    runtime_path = Path(runtime_config_path).absolute()
    pilot_path = Path(pilot_config_path).absolute()
    matrix_path = Path(matrix_config_path).absolute()
    selection_path = Path(selection_manifest_path).absolute()
    mapping_path = Path(mapping_receipt_path).absolute()
    observation_root = Path(observation_artifact_root).absolute()
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise CameraVisibleDiagnosticError(f"D1 output already exists: {output}")
    runtime_local = _load_json(runtime_path, "observation runtime config")
    pilot = _load_json(pilot_path, "observation pilot config")
    selection = _load_json(selection_path, "selection manifest")
    mapping = _load_json(mapping_path, "D2 mapping receipt")
    matrix_config, _matrix_selection = _load_config(matrix_path)
    matrix_runtime = matrix_config.get("runtime")
    observation_config = pilot.get("observation_bank")
    correspondence = pilot.get("correspondence")
    if not all(
        isinstance(value, Mapping)
        for value in (matrix_runtime, observation_config, correspondence)
    ):
        raise CameraVisibleDiagnosticError("diagnostic config sections are invalid")
    assert isinstance(matrix_runtime, Mapping)
    assert isinstance(observation_config, Mapping)
    assert isinstance(correspondence, Mapping)
    expected_python = Path(str(matrix_runtime.get("python", ""))).absolute()
    if not expected_python.is_file() or not os.path.samefile(sys.executable, expected_python):
        raise CameraVisibleDiagnosticError("D1 runner must use the frozen model Python")
    selected_pair_id = pair_id
    if selected_pair_id is None:
        pairs = runtime_local.get("pairs")
        dev = pairs.get("dev") if isinstance(pairs, Mapping) else None
        selected_pair_id = str(dev.get("pair_id", "")) if isinstance(dev, Mapping) else ""
    pair_record = find_pair_record(selection, selected_pair_id)
    pair_runtime = resolve_diagnostic_pair_runtime(runtime_local, selected_pair_id)
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    d0_pair = build_method_pair_view_from_manifest(
        pair_record=pair_record,
        source_manifest_sha256=selection_sha256,
        domain_id="D0_NATIVE_PROCESSED",
    )
    frames, _materialized, _native = load_real_pair_frames(
        pair_runtime=pair_runtime,
        observation_config=observation_config,
        pair_id=selected_pair_id,
        pair_record=pair_record,
        mapping_receipt=mapping,
    )
    tolerance = float(correspondence["positive_depth_tolerance_m"])
    minimum_neighbours = int(observation_config["minimum_valid_depth_neighbours"])
    supports = tuple(
        build_camera_visible_support_mask(
            points_reference_xyz=d0_pair.visits[visit_id].points_xyz,
            frames=tuple(frame for frame in frames if frame.visit_id == visit_id),
            visit_id=visit_id,
            tolerance_m=tolerance,
            minimum_valid_neighbours=minimum_neighbours,
        )
        for visit_id in (0, 1)
    )
    d1_pair = build_method_pair_view_from_manifest(
        pair_record=pair_record,
        source_manifest_sha256=selection_sha256,
        domain_id="D1_NATIVE_SENSOR_SUPPORT",
        support_masks=tuple(value.support_mask for value in supports),
    )
    observation_manifest = _load_json(
        observation_root / "manifest.json", "observation preparation manifest"
    )
    bank = load_observation_bank(observation_root / "bank")
    if (
        observation_manifest.get("status") != "PASS"
        or observation_manifest.get("pair_id") != selected_pair_id
        or bank.pair_id != selected_pair_id
        or observation_manifest.get("observation_bank", {}).get("content_sha256")
        != bank.content_sha256()
    ):
        raise CameraVisibleDiagnosticError("observation preparation identity mismatch")

    checkout, source_commit = _validated_checkout(matrix_runtime)
    checkpoint = _validated_external_record(
        matrix_runtime["checkpoint"], label="ReScene checkpoint"
    )
    concerto = _validated_external_record(
        matrix_runtime["concerto_checkpoint"], label="Concerto checkpoint"
    )
    processed_root = _validated_directory(
        matrix_runtime["processed_root"], label="processed root"
    )
    dataset_spec = _validated_external_record(
        matrix_runtime["dataset_spec"], label="dataset spec"
    )
    label_database = _validated_external_record(
        matrix_runtime["label_database"], label="label database"
    )
    change_database = _validated_external_record(
        matrix_runtime["change_label_database"], label="change label database"
    )
    color_mean_std = _validated_external_record(
        matrix_runtime["color_mean_std"], label="color mean/std"
    )
    _validated_external_record(
        matrix_runtime["validation_database"], label="validation database"
    )
    _validated_external_record(
        matrix_runtime["sequence_database"], label="sequence database"
    )

    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    with initialize_config_dir(version_base=None, config_dir=str(checkout / "conf")):
        native = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "data/datasets=rio",
                "general.train_mode=false",
                "general.train_on_segments=true",
                "general.eval_on_segments=true",
                "general.use_dbscan=false",
                "general.gpus=1",
                "general.seed=45",
                f"backbone.name={concerto}",
            ],
        )
    with open_dict(native):
        native.data.batch_size = 1
        native.data.test_batch_size = 1
        native.data.num_workers = 0
        native.instance_metric.dataset = str(dataset_spec)
        if hasattr(native, "aux_metric") and native.aux_metric is not None:
            native.aux_metric.dataset = str(dataset_spec)
        for split in ("train_dataset", "validation_dataset", "test_dataset"):
            dataset_config = native.data[split]
            dataset_config.data_dir = str(processed_root)
            dataset_config.label_db_filepath = str(label_database)
            dataset_config.change_label_db_filepath = str(change_database)
            dataset_config.color_mean_std = str(color_mean_std)
            dataset_config.temporal_window = 2

    device_name = os.environ.get("RESCENE_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise CameraVisibleDiagnosticError("D1 forward requires one explicit CUDA device")
    torch.cuda.set_device(device)
    runtime_workdir = _validated_directory(
        matrix_runtime["runtime_workdir"], label="runtime workdir"
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(runtime_workdir)
        from trainer.trainer import InstanceSegmentation

        system = InstanceSegmentation(native)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise CameraVisibleDiagnosticError(
                "checkpoint is missing its Lightning state_dict"
            )
        excluded = {str(key) for key in state if not str(key).startswith("model.")}
        if excluded != {"criterion.empty_weight", "criterion.change_weights"}:
            raise CameraVisibleDiagnosticError(
                "checkpoint contains unexpected non-model tensors"
            )
        model_state = {
            str(key)[len("model.") :]: value
            for key, value in state.items()
            if str(key).startswith("model.")
        }
        incompatible = system.model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CameraVisibleDiagnosticError("strict checkpoint load is incompatible")
        system = system.to(device).eval()
        dataset = hydra.utils.instantiate(native.data.validation_dataset)
        system.validation_dataset = dataset
        try:
            dataset_index = dataset.sequence_names.index(selected_pair_id)
        except ValueError as error:
            raise CameraVisibleDiagnosticError(
                "pair is absent from the native sequence database"
            ) from error
        seed = int(matrix_config["seed"])
        _seed_everything(seed)
        sample = dataset[dataset_index]
        _validate_native_raw_sample(d0_pair, sample)
        method_only = strip_native_ground_truth(sample)
        filtered = filter_native_sample_by_visibility(
            method_only,
            support_masks=tuple(value.support_mask for value in supports),
        )
        collate = hydra.utils.instantiate(native.data.validation_collation)
        _seed_everything(seed)
        data, targets, names = collate([filtered])
        if list(names) != [selected_pair_id] or len(targets) != 1:
            raise CameraVisibleDiagnosticError("native collator rejected D1 input")
        point2segment = targets[0].get("point2segment")
        if not isinstance(point2segment, torch.Tensor):
            raise CameraVisibleDiagnosticError("D1 collator lost mesh segments")
        inverse_n = np.asarray(data.inverse_maps[0].detach().cpu(), dtype=np.int64)
        lowres_segments = np.asarray(point2segment.detach().cpu(), dtype=np.int64)
        lowres_visits = np.asarray(
            data.temporal_stages[0].detach().cpu(), dtype=np.int8
        )
        full_source_indices = np.concatenate(
            [np.flatnonzero(value.support_mask) for value in supports]
        ).astype(np.int64)
        full_visits = np.concatenate(
            [
                np.full(np.count_nonzero(value.support_mask), visit_id, dtype=np.int8)
                for visit_id, value in enumerate(supports)
            ]
        )
        full_frame_counts = np.concatenate(
            [value.support_frame_counts[value.support_mask] for value in supports]
        ).astype(np.int16)
        if (
            len(inverse_n) != len(full_source_indices)
            or len(lowres_segments) != len(lowres_visits)
            or np.any(inverse_n < 0)
            or np.any(inverse_n >= len(lowres_segments))
        ):
            raise CameraVisibleDiagnosticError("D1 collator index domains are inconsistent")
        _move_tensors(data, device)
        _move_tensors(targets, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            forward = system.forward(
                data,
                point2segment=[target["point2segment"] for target in targets],
                raw_coordinates=system._process_raw_coordinates(data),
                is_eval=True,
                targets=None,
            )
        torch.cuda.synchronize(device)
        runtime_s = time.perf_counter() - started
        logits = forward.get("pred_logits")
        masks = forward.get("pred_masks")
        features = forward.get("backbone_features")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape[:2] != (1, 100)
            or not isinstance(masks, (list, tuple))
            or len(masks) != 1
            or not isinstance(masks[0], torch.Tensor)
            or not hasattr(features, "F")
        ):
            raise CameraVisibleDiagnosticError("D1 forward output schema is invalid")
        raw_logits = np.asarray(logits[0].detach().float().cpu(), dtype=np.float32)
        raw_masks = np.asarray(masks[0].detach().float().cpu(), dtype=np.float32)
        backbone_features = np.asarray(
            features.F.detach().float().cpu(), dtype=np.float32
        )
        if (
            raw_masks.shape[1] != raw_logits.shape[0]
            or backbone_features.shape[0] != len(lowres_segments)
            or np.any(~np.isfinite(raw_masks))
            or np.any(~np.isfinite(raw_logits))
            or np.any(~np.isfinite(backbone_features))
        ):
            raise CameraVisibleDiagnosticError("D1 forward arrays are invalid")
        arrays = {
            "pred_masks_sq": raw_masks,
            "pred_logits_qc": raw_logits,
            "backbone_features_mf": backbone_features,
            "lowres_point2segment_m": lowres_segments,
            "lowres_visit_ids_m": lowres_visits,
            "inverse_n": inverse_n,
            "full_visit_ids_n": full_visits,
            "full_source_point_indices_n": full_source_indices,
            "full_support_frame_counts_n": full_frame_counts,
            "visit0_support_mask": supports[0].support_mask,
            "visit1_support_mask": supports[1].support_mask,
            "visit0_support_frame_counts": supports[0].support_frame_counts,
            "visit1_support_frame_counts": supports[1].support_frame_counts,
        }
        runtime_record = {
            "forward_s": runtime_s,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "device_name": f"{torch.cuda.get_device_name(device)} ({device})",
            "model_tensor_count": len(model_state),
        }
    finally:
        os.chdir(previous_cwd)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        support_records = []
        for visit_id, value in enumerate(supports):
            support_records.append(
                {
                    "visit_id": visit_id,
                    "native_point_count": d0_pair.visits[visit_id].point_count,
                    "camera_visible_point_count": int(
                        np.count_nonzero(value.support_mask)
                    ),
                    "camera_visible_fraction": float(value.support_mask.mean()),
                    "single_view_point_count": int(
                        np.count_nonzero(value.support_frame_counts == 1)
                    ),
                    "multi_view_point_count": int(
                        np.count_nonzero(value.support_frame_counts >= 2)
                    ),
                    "maximum_support_frames": int(value.support_frame_counts.max()),
                    "depth_relation_counts": value.relation_counts,
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_D1_CAMERA_VISIBLE_FORWARD_V1",
            "status": "REAL_D1_CAMERA_VISIBLE_FORWARD_PASS",
            "pair_id": selected_pair_id,
            "definition": (
                "union of positive 3cm depth-consistent native points over the same "
                "16 selected real RGB-D cameras per visit"
            ),
            "ground_truth_used_for_input_selection": False,
            "ground_truth_columns_present_in_forward": False,
            "mesh_segment_proxy_for_collator": True,
            "forward_reused": False,
            "d0_method_input_sha256": d0_pair.method_tensor_sha256(),
            "d1_method_input_sha256": d1_pair.method_tensor_sha256(),
            "observation_bank_sha256": bank.content_sha256(),
            "support": support_records,
            "forward_shapes": {
                "full_visible_points": len(full_source_indices),
                "model_points": len(lowres_segments),
                "segments": raw_masks.shape[0],
                "queries": raw_masks.shape[1],
                "classes_with_no_object": raw_logits.shape[1],
                "backbone_feature_width": backbone_features.shape[1],
            },
            "runtime": runtime_record,
            "source_bindings": {
                "diagnostic_source": _file_record(
                    Path(__file__).absolute(), str(Path(__file__).absolute())
                ),
                "runtime_config": _file_record(runtime_path, str(runtime_path)),
                "pilot_config": _file_record(pilot_path, str(pilot_path)),
                "matrix_config": _file_record(matrix_path, str(matrix_path)),
                "selection_manifest": _file_record(
                    selection_path, str(selection_path)
                ),
                "mapping_receipt": _file_record(mapping_path, str(mapping_path)),
                "observation_manifest": _file_record(
                    observation_root / "manifest.json",
                    str(observation_root / "manifest.json"),
                ),
                "checkpoint": dict(matrix_runtime["checkpoint"]),
                "concerto_checkpoint": dict(matrix_runtime["concerto_checkpoint"]),
                "rescene_checkout": {
                    "path": str(checkout),
                    "commit": source_commit,
                },
            },
            "arrays": _file_record(arrays_path, "arrays.npz"),
            "claim_boundary": (
                "This is an input-coverage diagnostic and a real uncached base ReScene "
                "forward, not a trained observation-query result or a full-GT metric."
            ),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=REPO_ROOT / "configs/observation_query/runtime.local.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=REPO_ROOT / "configs/observation_query/pilot_v1.json",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/ovi_rescene_3rscan_t2_matrix_v1.json",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=REPO_ROOT
        / "configs/evaluation/manifests/ovi_rescene_object_level_pairs_v1.json",
    )
    parser.add_argument(
        "--mapping-receipt",
        type=Path,
        default=REPO_ROOT
        / "configs/evaluation/results/ovi_rescene_object_level_transfer/d2_ovimap_smoke_v1.json",
    )
    parser.add_argument("--observation-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-id")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    manifest = run_camera_visible_diagnostic(
        runtime_config_path=args.runtime_config,
        pilot_config_path=args.pilot_config,
        matrix_config_path=args.matrix_config,
        selection_manifest_path=args.selection_manifest,
        mapping_receipt_path=args.mapping_receipt,
        observation_artifact_root=args.observation_artifact,
        output_root=args.output_root,
        pair_id=args.pair_id,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pair_id": manifest["pair_id"],
                "forward_shapes": manifest["forward_shapes"],
                "runtime": manifest["runtime"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CameraVisibleDiagnosticError",
    "CameraVisibleSupport",
    "build_camera_visible_support_mask",
    "filter_native_sample_by_visibility",
    "resolve_diagnostic_pair_runtime",
    "run_camera_visible_diagnostic",
    "strip_native_ground_truth",
]


if __name__ == "__main__":
    raise SystemExit(main())
