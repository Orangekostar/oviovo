#!/usr/bin/env python3
"""Prepare source-bound raw OVI observations for observation-query ReScene."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.data_structures import CameraIntrinsics
from src.oviv2.observation_query.contracts import ObservationBank, save_observation_bank
from src.oviv2.observation_query.observations import (
    DepthRelation,
    RawRegionObservation,
    RegionSupportCandidate,
    build_region_metadata,
    camera_to_reference,
    classify_depth_relations,
    extract_raw_regions,
    neighborhood_depth_reliability,
    prepare_observation_bank,
    register_labels_nearest,
    select_observation_frames,
)
from src.oviv2.ovi_surface_attributes import project_world_points
from src.oviv2.rescene_input_bridge import (
    ReSceneModelInput,
    load_model_input_artifact,
    write_model_input_artifact,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ObservationPreparationError(ValueError):
    """Raised when real observation preparation loses source identity."""


@dataclass(frozen=True, slots=True)
class FrameRegionSupport:
    candidates: tuple[RegionSupportCandidate, ...]
    relation_counts: dict[str, int]
    signed_residuals_by_region: dict[int, np.ndarray]


@dataclass(frozen=True, slots=True)
class ModelObservationBundlePaths:
    root: Path
    manifest: Path
    model_input_root: Path
    geometry_arrays: Path


@dataclass(frozen=True, slots=True)
class RealFrameAssets:
    visit_id: int
    scan_uuid: str
    target_frame_id: int
    source_frame_id: int
    rgb: np.ndarray
    frontend_labels_color: np.ndarray
    registered_frontend_labels: np.ndarray
    geometric_labels_depth: np.ndarray
    depth_m: np.ndarray
    depth_intrinsic: np.ndarray
    color_to_depth_homography: np.ndarray
    camera_to_reference: np.ndarray
    paths: dict[str, Path]
    raw_regions: tuple[RawRegionObservation, ...]


@dataclass(frozen=True, slots=True)
class PreparedObservationBank:
    bank: ObservationBank
    regions: tuple[RawRegionObservation, ...]
    frame_supports: tuple[tuple[int, int, FrameRegionSupport], ...]
    relation_counts: dict[str, int]
    support_candidate_count: int


@dataclass(frozen=True, slots=True)
class ObservationPreparationArtifactPaths:
    root: Path
    manifest: Path
    bank_root: Path
    pixel_sidecar: Path
    diagnostics: Path
    overlays: tuple[Path, ...]


def _file_record(path: Path, recorded_path: str) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": recorded_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def save_model_observation_bundle(
    *,
    model_input: ReSceneModelInput,
    shared_center_xyz: object,
    pair_content_sha256: str,
    output_root: str | Path,
) -> ModelObservationBundlePaths:
    """Publish the exact M input together with its reference-frame coordinates."""

    if not isinstance(model_input, ReSceneModelInput):
        raise TypeError("model_input must be ReSceneModelInput")
    center = np.asarray(shared_center_xyz, dtype=np.float64)
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        raise ObservationPreparationError("shared_center_xyz must be a finite 3-vector")
    if not isinstance(pair_content_sha256, str) or _SHA256.fullmatch(pair_content_sha256) is None:
        raise ObservationPreparationError("pair_content_sha256 must be a lowercase SHA-256")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationPreparationError(f"model observation bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model_paths = write_model_input_artifact(model_input, staging / "model_input")
        geometry_path = staging / "model_geometry.npz"
        reference_points = model_input.features[:, :3].astype(np.float64) + center
        with geometry_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                reference_points_xyz=reference_points,
                shared_center_xyz=center,
                model_visit_ids=model_input.model_visit_ids,
            )
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_MODEL_BUNDLE_V1",
            "status": "PASS",
            "model_input_sha256": model_input.content_sha256(),
            "pair_content_sha256": pair_content_sha256,
            "model_count": len(model_input.model_visit_ids),
            "model_input_manifest": _file_record(
                model_paths.manifest, "model_input/manifest.json"
            ),
            "geometry_arrays": _file_record(geometry_path, geometry_path.name),
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
    return ModelObservationBundlePaths(
        output,
        output / "manifest.json",
        output / "model_input",
        output / "model_geometry.npz",
    )


def load_model_observation_bundle(
    output_root: str | Path,
) -> tuple[ReSceneModelInput, np.ndarray, np.ndarray, dict[str, object]]:
    """Load and bind reference points to the exact serialized ReScene input."""

    root = Path(output_root).absolute()
    manifest_path = root / "manifest.json"
    geometry_path = root / "model_geometry.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationPreparationError("model observation manifest is unavailable") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "model_input_sha256",
        "pair_content_sha256",
        "model_count",
        "model_input_manifest",
        "geometry_arrays",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_RESCENE_OBSERVATION_MODEL_BUNDLE_V1"
        or manifest.get("status") != "PASS"
    ):
        raise ObservationPreparationError("model observation manifest identity is invalid")
    if manifest.get("geometry_arrays") != _file_record(geometry_path, geometry_path.name):
        raise ObservationPreparationError("model geometry binding mismatch")
    model_manifest = root / "model_input" / "manifest.json"
    if manifest.get("model_input_manifest") != _file_record(
        model_manifest, "model_input/manifest.json"
    ):
        raise ObservationPreparationError("model input manifest binding mismatch")
    model_input = load_model_input_artifact(root / "model_input")
    if model_input.content_sha256() != manifest.get("model_input_sha256"):
        raise ObservationPreparationError("model input content identity mismatch")
    try:
        with np.load(io.BytesIO(geometry_path.read_bytes()), allow_pickle=False) as source:
            if set(source.files) != {
                "reference_points_xyz",
                "shared_center_xyz",
                "model_visit_ids",
            }:
                raise ObservationPreparationError("model geometry array schema is invalid")
            points = np.asarray(source["reference_points_xyz"], dtype=np.float64)
            center = np.asarray(source["shared_center_xyz"], dtype=np.float64)
            visits = np.asarray(source["model_visit_ids"])
    except (OSError, ValueError) as error:
        if isinstance(error, ObservationPreparationError):
            raise
        raise ObservationPreparationError("model geometry arrays are invalid") from error
    if (
        points.shape != (len(model_input.model_visit_ids), 3)
        or center.shape != (3,)
        or np.any(~np.isfinite(points))
        or np.any(~np.isfinite(center))
        or not np.array_equal(visits, model_input.model_visit_ids)
        or not np.array_equal(points, model_input.features[:, :3].astype(np.float64) + center)
        or manifest.get("model_count") != len(points)
    ):
        raise ObservationPreparationError("model geometry differs from the serialized M input")
    points = np.array(points, copy=True, order="C")
    center = np.array(center, copy=True, order="C")
    points.setflags(write=False)
    center.setflags(write=False)
    return model_input, points, center, manifest


def find_pair_record(
    selection: Mapping[str, object], pair_id: str
) -> Mapping[str, object]:
    """Return one exact pair record from a frozen selection manifest."""

    pairs = selection.get("pairs")
    if not isinstance(pairs, list) or not isinstance(pair_id, str) or not pair_id:
        raise ObservationPreparationError("selection pairs or pair_id are invalid")
    matches = [
        item
        for item in pairs
        if isinstance(item, Mapping) and item.get("pair_id") == pair_id
    ]
    if len(matches) != 1:
        raise ObservationPreparationError("selected pair must be unique")
    return matches[0]


def mapping_receipt_pair_id(mapping: Mapping[str, object]) -> str:
    """Read the selected pair from the frozen D2 receipt schema."""

    selection = mapping.get("selection")
    pair_id = selection.get("pair_id") if isinstance(selection, Mapping) else None
    if not isinstance(pair_id, str) or not pair_id:
        raise ObservationPreparationError("D2 mapping receipt selection is invalid")
    return pair_id


def pair_alignment_row(pair_record: Mapping[str, object]) -> np.ndarray:
    """Load the declared official rescan-to-reference row-vector transform."""

    common = pair_record.get("common_method_inputs")
    alignment = common.get("global_alignment") if isinstance(common, Mapping) else None
    if (
        not isinstance(alignment, Mapping)
        or alignment.get("application") != "homogeneous_row_vector_right_multiply"
        or alignment.get("direction") != "rescan_row_vector_to_reference"
        or alignment.get("storage") != "row_major_flat_4x4"
    ):
        raise ObservationPreparationError("pair alignment convention is invalid")
    matrix = np.asarray(alignment.get("matrix"), dtype=np.float64)
    if matrix.shape != (16,) or np.any(~np.isfinite(matrix)):
        raise ObservationPreparationError("pair alignment matrix is invalid")
    matrix = np.array(matrix.reshape(4, 4), copy=True, order="C")
    if not np.allclose(matrix[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ObservationPreparationError("pair alignment matrix is not row-vector affine")
    matrix.setflags(write=False)
    return matrix


def _image_array(path: Path, label: str) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ObservationPreparationError(f"{label} is missing or symlinked: {path}")
    try:
        with Image.open(path) as image:
            result = np.asarray(image)
    except (OSError, ValueError) as error:
        raise ObservationPreparationError(f"{label} cannot be decoded: {path}") from error
    return np.ascontiguousarray(result)


def _finite_matrix(path: Path, shape: tuple[int, int], label: str) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ObservationPreparationError(f"{label} is missing or symlinked: {path}")
    try:
        matrix = np.asarray(np.loadtxt(path), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ObservationPreparationError(f"{label} cannot be decoded: {path}") from error
    if matrix.shape != shape or np.any(~np.isfinite(matrix)):
        raise ObservationPreparationError(f"{label} must be finite {shape[0]}x{shape[1]}")
    return matrix


def load_real_frame_assets(
    *,
    visit_id: int,
    scan_uuid: str,
    target_frame_id: int,
    materialized_root: str | Path,
    native_attempt_root: str | Path,
    materialized_manifest: Mapping[str, object],
    rescan_to_reference_row: object,
    minimum_valid_depth_pixels: int,
) -> RealFrameAssets:
    """Load one source-bound raw OVI frame in calibrated depth coordinates."""

    if visit_id not in (0, 1) or isinstance(visit_id, bool):
        raise ObservationPreparationError("visit_id must be 0 or 1")
    if isinstance(target_frame_id, bool) or not isinstance(target_frame_id, Integral):
        raise ObservationPreparationError("target_frame_id must be an integer")
    if materialized_manifest.get("status") != "MATERIALIZED_INPUT_PASS":
        raise ObservationPreparationError("materialized manifest status is invalid")
    if materialized_manifest.get("scan_id") != scan_uuid:
        raise ObservationPreparationError("materialized manifest scan identity mismatch")
    frame_count = materialized_manifest.get("frame_count")
    frame_map = materialized_manifest.get("frame_map")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, Integral)
        or not isinstance(frame_map, list)
        or len(frame_map) != int(frame_count)
    ):
        raise ObservationPreparationError("materialized frame mapping is invalid")
    matches = [
        record
        for record in frame_map
        if isinstance(record, Mapping)
        and record.get("target_frame_id") == int(target_frame_id)
    ]
    if len(matches) != 1:
        raise ObservationPreparationError("target frame must have one source mapping")
    source_frame_id = matches[0].get("source_frame_id")
    if (
        isinstance(source_frame_id, bool)
        or not isinstance(source_frame_id, Integral)
        or source_frame_id < 0
    ):
        raise ObservationPreparationError("source frame mapping is invalid")

    dimensions = materialized_manifest.get("dimensions")
    calibration = materialized_manifest.get("calibration")
    if not isinstance(dimensions, Mapping) or not isinstance(calibration, Mapping):
        raise ObservationPreparationError("materialized dimensions or calibration are invalid")
    try:
        color_shape = (
            int(dimensions["color"]["height"]),
            int(dimensions["color"]["width"]),
        )
        depth_shape = (
            int(dimensions["depth"]["height"]),
            int(dimensions["depth"]["width"]),
        )
        color_intrinsic_4 = np.asarray(calibration["color_intrinsic"], dtype=np.float64)
        depth_intrinsic_4 = np.asarray(calibration["depth_intrinsic"], dtype=np.float64)
        color_extrinsic = np.asarray(calibration["color_extrinsic"], dtype=np.float64)
        depth_extrinsic = np.asarray(calibration["depth_extrinsic"], dtype=np.float64)
        depth_shift = float(materialized_manifest["depth_shift"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ObservationPreparationError("materialized calibration values are invalid") from error
    matrices = (color_intrinsic_4, depth_intrinsic_4, color_extrinsic, depth_extrinsic)
    if (
        any(matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)) for matrix in matrices)
        or not np.isfinite(depth_shift)
        or depth_shift <= 0.0
        or not np.allclose(color_extrinsic, np.eye(4), atol=1e-8)
        or not np.allclose(depth_extrinsic, np.eye(4), atol=1e-8)
    ):
        raise ObservationPreparationError("unsupported materialized calibration")
    color_intrinsic = color_intrinsic_4[:3, :3]
    depth_intrinsic = depth_intrinsic_4[:3, :3]
    if (
        abs(float(np.linalg.det(color_intrinsic))) <= 1e-12
        or abs(float(np.linalg.det(depth_intrinsic))) <= 1e-12
    ):
        raise ObservationPreparationError("camera intrinsics must be invertible")

    materialized = Path(materialized_root).absolute()
    attempt = Path(native_attempt_root).absolute()
    paths = {
        "rgb": materialized / "color" / f"{target_frame_id}.jpg",
        "depth": materialized / "depth" / f"{target_frame_id}.png",
        "pose": materialized / "pose" / f"{target_frame_id}.txt",
        "frontend": attempt / "frontend" / f"{target_frame_id}.png",
        "geometric": attempt
        / "geometric_segments"
        / f"{target_frame_id:05d}_mask.png",
    }
    rgb = _image_array(paths["rgb"], "RGB frame")
    raw_depth = _image_array(paths["depth"], "depth frame")
    frontend = _image_array(paths["frontend"], "frontend labels")
    geometric = _image_array(paths["geometric"], "geometric labels")
    pose = _finite_matrix(paths["pose"], (4, 4), "camera pose")
    if rgb.shape != (*color_shape, 3) or rgb.dtype != np.uint8:
        raise ObservationPreparationError("RGB frame shape or dtype mismatch")
    if raw_depth.shape != depth_shape or not np.issubdtype(raw_depth.dtype, np.integer):
        raise ObservationPreparationError("depth frame shape or dtype mismatch")
    if (
        frontend.shape != color_shape
        or geometric.shape != depth_shape
        or not np.issubdtype(frontend.dtype, np.integer)
        or not np.issubdtype(geometric.dtype, np.integer)
    ):
        raise ObservationPreparationError("raw label frame shape or dtype mismatch")

    depth_m = np.asarray(raw_depth, dtype=np.float32) / depth_shift
    homography = depth_intrinsic @ np.linalg.inv(color_intrinsic)
    registered = register_labels_nearest(frontend, homography, depth_shape)
    reference_pose = camera_to_reference(pose, visit_id, rescan_to_reference_row)
    raw_regions = extract_raw_regions(
        scan_uuid=scan_uuid,
        visit_id=visit_id,
        source_frame_id=int(source_frame_id),
        frontend_labels_color=frontend,
        geometric_labels_depth=geometric,
        depth_m=depth_m,
        color_to_depth_homography=homography,
        minimum_valid_depth_pixels=minimum_valid_depth_pixels,
    )
    for array in (rgb, frontend, registered, geometric, depth_m, depth_intrinsic, homography):
        array.setflags(write=False)
    return RealFrameAssets(
        visit_id=int(visit_id),
        scan_uuid=scan_uuid,
        target_frame_id=int(target_frame_id),
        source_frame_id=int(source_frame_id),
        rgb=rgb,
        frontend_labels_color=frontend,
        registered_frontend_labels=registered,
        geometric_labels_depth=geometric,
        depth_m=depth_m,
        depth_intrinsic=depth_intrinsic,
        color_to_depth_homography=homography,
        camera_to_reference=reference_pose,
        paths=paths,
        raw_regions=raw_regions,
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationPreparationError(f"{label} is unavailable") from error
    if not isinstance(value, dict):
        raise ObservationPreparationError(f"{label} must be a JSON object")
    return value


def build_model_bundle_from_d2(
    *,
    object_transfer_config: str | Path,
    pair_id: str,
    output_root: str | Path,
) -> ModelObservationBundlePaths:
    """Rebuild the exact D2 M input without executing a ReScene forward."""

    from scripts.evaluation.run_ovi_rescene_dense_instance_repair import (
        _source_bound_grid_sample_factory,
    )
    from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
        build_d2_inference_bundle,
        load_object_transfer_config,
    )
    from src.evaluation.ovi_pair_views import build_ovi_object_pair_view

    config = load_object_transfer_config(object_transfer_config)
    selection_bytes = config.selection_manifest.read_bytes()
    if hashlib.sha256(selection_bytes).hexdigest() != config.selection_manifest_sha256:
        raise ObservationPreparationError("selection manifest binding mismatch")
    selection = _load_json_object(config.selection_manifest, "selection manifest")
    pair_record = find_pair_record(selection, pair_id)
    mapping = _load_json_object(config.d2_mapping_receipt, "D2 mapping receipt")
    if mapping_receipt_pair_id(mapping) != pair_id or mapping.get("status") != "REAL_D2_PAIR_PASS":
        raise ObservationPreparationError("D2 mapping receipt does not bind the selected pair")
    visits = mapping.get("visits")
    if not isinstance(visits, list) or len(visits) != 2:
        raise ObservationPreparationError("D2 mapping receipt must bind two visits")
    try:
        native_manifests = [
            Path(visit["native_mapping"]["manifest"]["path"]) for visit in visits
        ]
        materialized_manifests = [
            Path(visit["materialized"]["manifest"]["path"]) for visit in visits
        ]
    except (KeyError, TypeError) as error:
        raise ObservationPreparationError("D2 mapping visit bindings are incomplete") from error
    pair = build_ovi_object_pair_view(
        pair_record=pair_record,
        source_manifest_sha256=config.selection_manifest_sha256,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=config.native_sampler_source.path,
        sampler_source_sha256=config.native_sampler_source.sha256,
        sampler_seed=config.sampler_seed,
        maximum_candidates=config.maximum_candidates,
        grid_sample_factory=_source_bound_grid_sample_factory(
            config.native_sampler_source.path
        ),
    )
    return save_model_observation_bundle(
        model_input=bundle.model_input,
        shared_center_xyz=bundle.geometry.shared_center_xyz,
        pair_content_sha256=pair.content_sha256(),
        output_root=output_root,
    )


def _balanced_parent_order(
    parents: Iterable[RawRegionObservation],
) -> list[RawRegionObservation]:
    ordered = sorted(
        parents,
        key=lambda item: (
            -len(item.mask_pixel_indices),
            item.frontend_instance_id,
            item.key,
        ),
    )
    result: list[RawRegionObservation] = []
    left, right = 0, len(ordered) - 1
    while left <= right:
        result.append(ordered[left])
        left += 1
        if left <= right:
            result.append(ordered[right])
            right -= 1
    return result


def select_region_budget(
    regions: Iterable[RawRegionObservation],
    *,
    maximum_regions: int,
    maximum_fragments_per_parent: int,
) -> tuple[RawRegionObservation, ...]:
    """Cover parent crops across frames before spending tokens on fragments."""

    for value, name in (
        (maximum_regions, "maximum_regions"),
        (maximum_fragments_per_parent, "maximum_fragments_per_parent"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ObservationPreparationError(f"{name} must be a positive integer")
    values = tuple(regions)
    if any(not isinstance(item, RawRegionObservation) for item in values):
        raise TypeError("regions must contain RawRegionObservation")
    if len({item.key for item in values}) != len(values):
        raise ObservationPreparationError("raw region keys must be unique")
    parents = {item.key: item for item in values if item.fragment_id == -1}
    if any(item.parent_region_key not in parents for item in values):
        raise ObservationPreparationError("every fragment must retain its parent region")

    frame_queues: dict[tuple[int, int], list[RawRegionObservation]] = {}
    grouped_parents: dict[tuple[int, int], list[RawRegionObservation]] = defaultdict(list)
    for parent in parents.values():
        grouped_parents[(parent.visit_id, parent.source_frame_id)].append(parent)
    for frame, frame_parents in grouped_parents.items():
        frame_queues[frame] = _balanced_parent_order(frame_parents)

    selected_parents: list[RawRegionObservation] = []
    frame_keys = sorted(frame_queues)
    while len(selected_parents) < maximum_regions:
        changed = False
        for frame in frame_keys:
            queue = frame_queues[frame]
            if queue and len(selected_parents) < maximum_regions:
                selected_parents.append(queue.pop(0))
                changed = True
        if not changed:
            break

    selected: list[RawRegionObservation] = list(selected_parents)
    fragments_by_parent: dict[str, list[RawRegionObservation]] = defaultdict(list)
    for item in values:
        if item.fragment_id >= 0 and item.parent_region_key in {
            parent.key for parent in selected_parents
        }:
            fragments_by_parent[item.parent_region_key].append(item)
    for fragments in fragments_by_parent.values():
        fragments.sort(
            key=lambda item: (-len(item.support_pixel_indices), item.fragment_id, item.key)
        )
        del fragments[maximum_fragments_per_parent:]
    while len(selected) < maximum_regions:
        changed = False
        for parent in selected_parents:
            queue = fragments_by_parent[parent.key]
            if queue and len(selected) < maximum_regions:
                selected.append(queue.pop(0))
                changed = True
        if not changed:
            break
    return tuple(selected)


def encode_parent_region_features(
    regions: Iterable[RawRegionObservation],
    *,
    load_rgb: Callable[[int, int], np.ndarray],
    load_frontend_labels: Callable[[int, int], np.ndarray],
    encode_six_crop: Callable[
        [np.ndarray, np.ndarray, tuple[int, int, int, int]], np.ndarray
    ],
) -> np.ndarray:
    """Encode each parent once, L2-normalize the mean, and share it with fragments."""

    values = tuple(regions)
    if not values:
        raise ObservationPreparationError("cannot infer feature width from no regions")
    parents: dict[str, RawRegionObservation] = {}
    for item in values:
        if not isinstance(item, RawRegionObservation):
            raise TypeError("regions must contain RawRegionObservation")
        if item.fragment_id == -1:
            parents[item.key] = item
    if any(item.parent_region_key not in parents for item in values):
        raise ObservationPreparationError("selected fragments require their selected parent")

    frame_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    features_by_parent: dict[str, np.ndarray] = {}
    feature_width: int | None = None
    for parent in parents.values():
        frame_key = (parent.visit_id, parent.source_frame_id)
        if frame_key not in frame_cache:
            rgb = np.asarray(load_rgb(*frame_key))
            labels = np.asarray(load_frontend_labels(*frame_key))
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ObservationPreparationError("RGB loader must return HxWx3 uint8")
            if (
                labels.ndim != 2
                or labels.shape != rgb.shape[:2]
                or not np.issubdtype(labels.dtype, np.integer)
            ):
                raise ObservationPreparationError("frontend labels must align with RGB")
            frame_cache[frame_key] = (rgb, labels)
        rgb, labels = frame_cache[frame_key]
        mask = labels == parent.frontend_instance_id
        if not np.any(mask):
            raise ObservationPreparationError("parent frontend mask is empty")
        feature = np.asarray(
            encode_six_crop(rgb, mask, parent.parent_color_bbox_xyxy),
            dtype=np.float64,
        )
        if feature.ndim != 1 or len(feature) == 0:
            raise ObservationPreparationError("six-crop feature must be a non-empty vector")
        if np.any(~np.isfinite(feature)):
            raise ObservationPreparationError("six-crop feature must be finite")
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ObservationPreparationError("six-crop feature must have nonzero norm")
        feature = feature / norm
        if feature_width is None:
            feature_width = len(feature)
        elif len(feature) != feature_width:
            raise ObservationPreparationError("six-crop feature dimensions differ")
        features_by_parent[parent.key] = feature

    result = np.stack(
        [features_by_parent[item.parent_region_key] for item in values], axis=0
    ).astype(np.float32)
    result.setflags(write=False)
    return result


def write_region_pixel_sidecar(
    regions: Iterable[RawRegionObservation], path: str | Path
) -> Path:
    """Store variable-length depth-domain masks without object arrays."""

    values = tuple(regions)
    if not values or any(not isinstance(item, RawRegionObservation) for item in values):
        raise ObservationPreparationError("region pixel sidecar requires raw regions")
    parent_indices = {
        item.key: index for index, item in enumerate(values) if item.fragment_id == -1
    }
    if any(item.parent_region_key not in parent_indices for item in values):
        raise ObservationPreparationError("region pixel sidecar lacks a selected parent")

    def concatenate(name: str) -> tuple[np.ndarray, np.ndarray]:
        arrays = [np.asarray(getattr(item, name), dtype=np.int64) for item in values]
        indptr = np.empty(len(arrays) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum([len(array) for array in arrays], out=indptr[1:])
        flattened = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        return indptr, flattened

    mask_indptr, mask_indices = concatenate("mask_pixel_indices")
    support_indptr, support_indices = concatenate("support_pixel_indices")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            mask_indptr=mask_indptr,
            mask_pixel_indices=mask_indices,
            support_indptr=support_indptr,
            support_pixel_indices=support_indices,
            parent_region_indices=np.asarray(
                [parent_indices[item.parent_region_key] for item in values],
                dtype=np.int64,
            ),
            fragment_ids=np.asarray([item.fragment_id for item in values], dtype=np.int64),
            bbox_xyxy=np.asarray([item.bbox_xyxy for item in values], dtype=np.int64),
            parent_color_bbox_xyxy=np.asarray(
                [item.parent_color_bbox_xyxy for item in values], dtype=np.int64
            ),
            depth_shapes=np.asarray([item.depth_shape for item in values], dtype=np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    return output


def project_frame_region_support(
    *,
    model_points_reference_xyz: object,
    model_visit_ids: object,
    frame_visit_id: int,
    source_frame_id: int,
    camera_to_reference: object,
    depth_intrinsic: object,
    depth_m: object,
    registered_frontend_labels: object,
    geometric_labels_depth: object,
    selected_region_indices: Mapping[tuple[int, int], int],
    eligible_fragment_identities: AbstractSet[tuple[int, int]],
    tolerance_m: float,
    minimum_valid_neighbours: int = 5,
) -> FrameRegionSupport:
    """Project same-visit M rows and create positive raw-region support candidates."""

    points = np.asarray(model_points_reference_xyz, dtype=np.float64)
    visits = np.asarray(model_visit_ids)
    depth = np.asarray(depth_m, dtype=np.float64)
    frontend = np.asarray(registered_frontend_labels)
    geometric = np.asarray(geometric_labels_depth)
    intrinsic = np.asarray(depth_intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_reference, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or np.any(~np.isfinite(points)):
        raise ObservationPreparationError("model points must have finite shape (M, 3)")
    if visits.shape != (len(points),) or not np.issubdtype(visits.dtype, np.integer):
        raise ObservationPreparationError("model visit IDs must align with M")
    if frame_visit_id not in (0, 1) or isinstance(frame_visit_id, bool):
        raise ObservationPreparationError("frame_visit_id must be 0 or 1")
    if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, Integral) or source_frame_id < 0:
        raise ObservationPreparationError("source_frame_id must be nonnegative")
    if depth.ndim != 2 or frontend.shape != depth.shape or geometric.shape != depth.shape:
        raise ObservationPreparationError("depth-domain frame arrays must have identical shapes")
    if not np.issubdtype(frontend.dtype, np.integer) or not np.issubdtype(geometric.dtype, np.integer):
        raise ObservationPreparationError("registered region images must be integer")
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ObservationPreparationError("camera matrices have invalid shapes")

    model_indices = np.flatnonzero(visits == frame_visit_id).astype(np.int64)
    intrinsics = CameraIntrinsics(
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        int(depth.shape[1]),
        int(depth.shape[0]),
    )
    projection = project_world_points(points[model_indices], pose, intrinsics)
    observed = np.full(len(model_indices), np.nan, dtype=np.float64)
    projected_rows = projection.rows
    projected_columns = projection.columns
    inside = projection.projectable
    observed[inside] = depth[projected_rows[inside], projected_columns[inside]]
    reliable = neighborhood_depth_reliability(
        depth,
        projected_rows,
        projected_columns,
        tolerance_m=tolerance_m,
        minimum_valid_neighbours=minimum_valid_neighbours,
    )
    relations = classify_depth_relations(
        observed_depth_m=observed,
        projected_depth_m=projection.camera_xyz[:, 2],
        neighborhood_reliable=reliable,
        tolerance_m=tolerance_m,
    )
    counts = {
        relation.name.lower(): int(np.count_nonzero(relations == relation))
        for relation in DepthRelation
    }

    candidates: list[RegionSupportCandidate] = []
    residuals: dict[int, list[float]] = defaultdict(list)
    supported_local = np.flatnonzero(relations == DepthRelation.SUPPORTED)
    for local_index in supported_local:
        row = int(projected_rows[local_index])
        column = int(projected_columns[local_index])
        frontend_id = int(frontend[row, column])
        if frontend_id <= 0:
            continue
        fragment_id = int(geometric[row, column])
        fragment_identity = (frontend_id, fragment_id)
        if fragment_id > 0 and fragment_identity in eligible_fragment_identities:
            identity = fragment_identity
            is_fragment = True
        else:
            identity = (frontend_id, -1)
            is_fragment = False
        region_index = selected_region_indices.get(identity)
        if region_index is None:
            continue
        residual = float(observed[local_index] - projection.camera_xyz[local_index, 2])
        weight = max(0.5, 1.0 - 0.5 * abs(residual) / float(tolerance_m))
        candidates.append(
            RegionSupportCandidate(
                region_index=int(region_index),
                model_index=int(model_indices[local_index]),
                visit_id=int(frame_visit_id),
                source_frame_id=int(source_frame_id),
                pixel_index=row * depth.shape[1] + column,
                weight=weight,
                is_fragment=is_fragment,
            )
        )
        residuals[int(region_index)].append(residual)
    frozen_residuals: dict[int, np.ndarray] = {}
    for region_index, values in residuals.items():
        array = np.asarray(values, dtype=np.float32)
        array.setflags(write=False)
        frozen_residuals[region_index] = array
    return FrameRegionSupport(tuple(candidates), counts, frozen_residuals)


def build_observation_bank_from_frames(
    *,
    pair_id: str,
    model_input_sha256: str,
    model_points_reference_xyz: object,
    model_visit_ids: object,
    frames: Iterable[RealFrameAssets],
    encode_six_crop: Callable[
        [np.ndarray, np.ndarray, tuple[int, int, int, int]], np.ndarray
    ],
    source_manifest: dict[str, object],
    maximum_regions_per_visit: int,
    maximum_fragments_per_parent: int,
    maximum_distinct_frames_per_model_point: int,
    tolerance_m: float,
    minimum_valid_neighbours: int = 5,
) -> PreparedObservationBank:
    """Build one real two-visit bank from already source-bound frame assets."""

    frame_values = tuple(frames)
    if not frame_values or any(
        not isinstance(frame, RealFrameAssets) for frame in frame_values
    ):
        raise ObservationPreparationError("frames must contain real frame assets")
    frame_lookup: dict[tuple[int, int], RealFrameAssets] = {}
    for frame in frame_values:
        key = (frame.visit_id, frame.source_frame_id)
        if key in frame_lookup:
            raise ObservationPreparationError("frame visit/source identities must be unique")
        frame_lookup[key] = frame
    if {frame.visit_id for frame in frame_values} != {0, 1}:
        raise ObservationPreparationError("a real pair bank requires both visits")

    selected: list[RawRegionObservation] = []
    for visit_id in (0, 1):
        available = tuple(
            region
            for frame in frame_values
            if frame.visit_id == visit_id
            for region in frame.raw_regions
        )
        selected.extend(
            select_region_budget(
                available,
                maximum_regions=maximum_regions_per_visit,
                maximum_fragments_per_parent=maximum_fragments_per_parent,
            )
        )
    regions = tuple(selected)
    if not regions:
        raise ObservationPreparationError("real selected frames contain no eligible regions")

    features = encode_parent_region_features(
        regions,
        load_rgb=lambda visit, source: frame_lookup[(visit, source)].rgb,
        load_frontend_labels=lambda visit, source: frame_lookup[
            (visit, source)
        ].frontend_labels_color,
        encode_six_crop=encode_six_crop,
    )
    global_region_index = {region.key: index for index, region in enumerate(regions)}
    all_candidates: list[RegionSupportCandidate] = []
    frame_supports: list[tuple[int, int, FrameRegionSupport]] = []
    residuals_by_region: dict[int, list[np.ndarray]] = defaultdict(list)
    relation_counts = {relation.name.lower(): 0 for relation in DepthRelation}
    for frame in sorted(
        frame_values, key=lambda value: (value.visit_id, value.source_frame_id)
    ):
        frame_regions = [
            region
            for region in regions
            if region.visit_id == frame.visit_id
            and region.source_frame_id == frame.source_frame_id
        ]
        selected_indices = {
            (region.frontend_instance_id, region.fragment_id): global_region_index[
                region.key
            ]
            for region in frame_regions
        }
        eligible_fragments = {
            (region.frontend_instance_id, region.fragment_id)
            for region in frame.raw_regions
            if region.fragment_id >= 0
        }
        support = project_frame_region_support(
            model_points_reference_xyz=model_points_reference_xyz,
            model_visit_ids=model_visit_ids,
            frame_visit_id=frame.visit_id,
            source_frame_id=frame.source_frame_id,
            camera_to_reference=frame.camera_to_reference,
            depth_intrinsic=frame.depth_intrinsic,
            depth_m=frame.depth_m,
            registered_frontend_labels=frame.registered_frontend_labels,
            geometric_labels_depth=frame.geometric_labels_depth,
            selected_region_indices=selected_indices,
            eligible_fragment_identities=eligible_fragments,
            tolerance_m=tolerance_m,
            minimum_valid_neighbours=minimum_valid_neighbours,
        )
        frame_supports.append((frame.visit_id, frame.source_frame_id, support))
        all_candidates.extend(support.candidates)
        for name, count in support.relation_counts.items():
            relation_counts[name] += count
        for region_index, residuals in support.signed_residuals_by_region.items():
            residuals_by_region[region_index].append(residuals)

    metadata_rows: list[np.ndarray] = []
    reliability: list[float] = []
    for region_index, region in enumerate(regions):
        frame = frame_lookup[(region.visit_id, region.source_frame_id)]
        chunks = residuals_by_region.get(region_index, [])
        residuals = (
            np.concatenate(chunks)
            if chunks
            else np.empty(0, dtype=np.float32)
        )
        metadata, fixed_weight = build_region_metadata(
            region,
            depth_m=frame.depth_m,
            depth_intrinsic=frame.depth_intrinsic,
            camera_to_reference=frame.camera_to_reference,
            signed_depth_residuals_m=residuals,
        )
        metadata_rows.append(metadata)
        reliability.append(fixed_weight)

    model_visits = np.asarray(model_visit_ids)
    bank = prepare_observation_bank(
        pair_id=pair_id,
        model_input_sha256=model_input_sha256,
        region_keys=tuple(region.key for region in regions),
        region_visit_ids=np.asarray([region.visit_id for region in regions], dtype=np.int8),
        region_frame_ids=np.asarray(
            [region.source_frame_id for region in regions], dtype=np.int64
        ),
        region_features=features,
        region_metadata=np.stack(metadata_rows),
        region_reliability=np.asarray(reliability, dtype=np.float32),
        model_visit_ids=model_visits,
        support_candidates=all_candidates,
        source_manifest=source_manifest,
        maximum_distinct_frames_per_model_point=maximum_distinct_frames_per_model_point,
    )
    return PreparedObservationBank(
        bank=bank,
        regions=regions,
        frame_supports=tuple(frame_supports),
        relation_counts=relation_counts,
        support_candidate_count=len(all_candidates),
    )


def build_observation_source_manifest(
    *,
    pair_id: str,
    pair_record: Mapping[str, object],
    selection_manifest_path: str | Path,
    mapping_receipt_path: str | Path,
    model_bundle_manifest_path: str | Path,
    materialized_manifest_paths: Iterable[str | Path],
    native_mapping_manifest_paths: Iterable[str | Path],
    frames: Iterable[RealFrameAssets],
    feature_encoder_manifest: dict[str, object],
    observation_config: dict[str, object],
) -> dict[str, object]:
    """Build a no-GT provenance record for raw observation preparation."""

    frame_values = tuple(frames)
    materialized_paths = tuple(Path(path).absolute() for path in materialized_manifest_paths)
    native_paths = tuple(Path(path).absolute() for path in native_mapping_manifest_paths)
    if len(materialized_paths) != 2 or len(native_paths) != 2:
        raise ObservationPreparationError("source manifest requires two visit manifests")
    if {frame.visit_id for frame in frame_values} != {0, 1}:
        raise ObservationPreparationError("source manifest requires both visit frame sets")
    try:
        encoder = json.loads(
            json.dumps(feature_encoder_manifest, allow_nan=False, ensure_ascii=True)
        )
        config = json.loads(
            json.dumps(observation_config, allow_nan=False, ensure_ascii=True)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ObservationPreparationError("source metadata must be canonical JSON") from error
    alignment = pair_alignment_row(pair_record)
    visits: list[dict[str, object]] = []
    for visit_id in (0, 1):
        visit_frames = sorted(
            (frame for frame in frame_values if frame.visit_id == visit_id),
            key=lambda frame: frame.target_frame_id,
        )
        scan_ids = {frame.scan_uuid for frame in visit_frames}
        if len(scan_ids) != 1:
            raise ObservationPreparationError("visit frames disagree on scan identity")
        frame_records = []
        for frame in visit_frames:
            required = {"rgb", "depth", "pose", "frontend", "geometric"}
            if set(frame.paths) != required:
                raise ObservationPreparationError("frame source path schema is invalid")
            frame_records.append(
                {
                    "target_frame_id": frame.target_frame_id,
                    "source_frame_id": frame.source_frame_id,
                    "files": {
                        name: _file_record(frame.paths[name], str(frame.paths[name]))
                        for name in sorted(required)
                    },
                }
            )
        first = visit_frames[0]
        visits.append(
            {
                "visit_id": visit_id,
                "scan_uuid": next(iter(scan_ids)),
                "materialized_manifest": _file_record(
                    materialized_paths[visit_id], str(materialized_paths[visit_id])
                ),
                "native_mapping_manifest": _file_record(
                    native_paths[visit_id], str(native_paths[visit_id])
                ),
                "selected_frames": [
                    {
                        "target_frame_id": frame.target_frame_id,
                        "source_frame_id": frame.source_frame_id,
                    }
                    for frame in visit_frames
                ],
                "frame_sources": frame_records,
                "depth_intrinsic": first.depth_intrinsic.tolist(),
                "color_to_depth_homography": first.color_to_depth_homography.tolist(),
                "camera_pose": "column_vector_camera_to_reference",
            }
        )
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBSERVATION_SOURCES_V1",
        "pair_id": pair_id,
        "selection_manifest": _file_record(
            Path(selection_manifest_path).absolute(),
            str(Path(selection_manifest_path).absolute()),
        ),
        "mapping_receipt": _file_record(
            Path(mapping_receipt_path).absolute(),
            str(Path(mapping_receipt_path).absolute()),
        ),
        "model_bundle_manifest": _file_record(
            Path(model_bundle_manifest_path).absolute(),
            str(Path(model_bundle_manifest_path).absolute()),
        ),
        "region_construction": "raw_region_plus_depth_fragments",
        "region_identity": [
            "scan_uuid",
            "visit_id",
            "source_frame_id",
            "frontend_instance_id",
            "fragment_id",
        ],
        "bbox_convention": "xyxy_half_open",
        "pixel_domain": "registered_depth_image",
        "frontend_registration": "calibrated_nearest_neighbor_labels",
        "rescan_to_reference_row": alignment.tolist(),
        "feature_encoder": encoder,
        "observation_config": config,
        "feature_source": (
            "one frozen six-crop parent embedding shared by eligible depth fragments; "
            "six-crop mean is L2-normalized"
        ),
        "ground_truth_used": False,
        "visits": visits,
    }


def _nested_mapping(
    value: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ObservationPreparationError(f"{label} is invalid")
    return result


def _validate_bound_file(record: object, expected_path: Path, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ObservationPreparationError(f"{label} binding is invalid")
    expected = _file_record(expected_path, str(expected_path))
    if dict(record) != expected:
        raise ObservationPreparationError(f"{label} binding mismatch")


def load_real_pair_frames(
    *,
    runtime_config: Mapping[str, object],
    observation_config: Mapping[str, object],
    pair_id: str,
    pair_record: Mapping[str, object],
    mapping_receipt: Mapping[str, object],
) -> tuple[
    tuple[RealFrameAssets, ...],
    tuple[Path, Path],
    tuple[Path, Path],
]:
    """Cross-bind the real caches and load the fixed two-visit frame subset."""

    if (
        mapping_receipt.get("status") != "REAL_D2_PAIR_PASS"
        or mapping_receipt_pair_id(mapping_receipt) != pair_id
    ):
        raise ObservationPreparationError("D2 mapping receipt identity mismatch")
    runtime_pairs = _nested_mapping(runtime_config, "pairs", "runtime pairs")
    dev = _nested_mapping(runtime_pairs, "dev", "runtime DEV pair")
    if dev.get("pair_id") != pair_id or dev.get("status") != "READY":
        raise ObservationPreparationError("runtime DEV pair is not ready")
    scan_ids = dev.get("scan_ids")
    materialized_roots = dev.get("materialized_roots")
    attempt_roots = dev.get("native_attempt_roots")
    receipt_visits = mapping_receipt.get("visits")
    if any(
        not isinstance(value, list) or len(value) != 2
        for value in (scan_ids, materialized_roots, attempt_roots, receipt_visits)
    ):
        raise ObservationPreparationError("runtime or D2 visit bindings are incomplete")
    pair_sessions = pair_record.get("sessions")
    if not isinstance(pair_sessions, list) or len(pair_sessions) != 2:
        raise ObservationPreparationError("selected pair must contain two sessions")
    selected_scans = [
        session.get("scan_id") if isinstance(session, Mapping) else None
        for session in pair_sessions
    ]
    if selected_scans != scan_ids:
        raise ObservationPreparationError("selection and runtime scan order mismatch")
    try:
        maximum_frames = int(observation_config["max_frames_per_visit"])
        minimum_pixels = int(observation_config["minimum_valid_depth_pixels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ObservationPreparationError("observation frame limits are invalid") from error

    alignment = pair_alignment_row(pair_record)
    frames: list[RealFrameAssets] = []
    materialized_manifests: list[Path] = []
    native_manifests: list[Path] = []
    for visit_id in (0, 1):
        scan_uuid = scan_ids[visit_id]
        if not isinstance(scan_uuid, str) or not scan_uuid:
            raise ObservationPreparationError("runtime scan identity is invalid")
        materialized_root = Path(materialized_roots[visit_id]).absolute()
        attempt_root = Path(attempt_roots[visit_id]).absolute()
        materialized_path = materialized_root / "materialized_manifest.json"
        native_path = attempt_root / "native_mapping_manifest.json"
        materialized = _load_json_object(materialized_path, "materialized manifest")
        native = _load_json_object(native_path, "native mapping manifest")
        receipt_visit = receipt_visits[visit_id]
        if (
            not isinstance(receipt_visit, Mapping)
            or receipt_visit.get("scan_id") != scan_uuid
            or native.get("status") != "PASS"
            or native.get("scene") != scan_uuid
        ):
            raise ObservationPreparationError("native visit identity mismatch")
        try:
            receipt_materialized = receipt_visit["materialized"]["manifest"]
            receipt_native = receipt_visit["native_mapping"]["manifest"]
        except (KeyError, TypeError) as error:
            raise ObservationPreparationError("D2 visit manifest bindings are absent") from error
        _validate_bound_file(
            receipt_materialized, materialized_path, "materialized manifest"
        )
        _validate_bound_file(receipt_native, native_path, "native mapping manifest")
        frame_count = materialized.get("frame_count")
        if isinstance(frame_count, bool) or not isinstance(frame_count, Integral):
            raise ObservationPreparationError("materialized frame count is invalid")
        selected = select_observation_frames(
            frame_count=int(frame_count), maximum_frames=maximum_frames
        )
        for target_frame_id in selected:
            frames.append(
                load_real_frame_assets(
                    visit_id=visit_id,
                    scan_uuid=scan_uuid,
                    target_frame_id=target_frame_id,
                    materialized_root=materialized_root,
                    native_attempt_root=attempt_root,
                    materialized_manifest=materialized,
                    rescan_to_reference_row=alignment,
                    minimum_valid_depth_pixels=minimum_pixels,
                )
            )
        materialized_manifests.append(materialized_path)
        native_manifests.append(native_path)
    return (
        tuple(frames),
        (materialized_manifests[0], materialized_manifests[1]),
        (native_manifests[0], native_manifests[1]),
    )


def load_ovimap_siglip_encoder(
    *,
    runtime_config: Mapping[str, object],
    mapping_receipt: Mapping[str, object],
    image_shape: tuple[int, int],
    device: str,
) -> tuple[
    Callable[[np.ndarray, np.ndarray, tuple[int, int, int, int]], np.ndarray],
    dict[str, object],
]:
    """Load the exact frozen OVI six-crop SigLIP implementation and weights."""

    checkouts = _nested_mapping(runtime_config, "checkouts", "runtime checkouts")
    assets = _nested_mapping(runtime_config, "assets", "runtime assets")
    ovimap = Path(str(checkouts.get("ovimap", ""))).absolute()
    model_root = Path(str(assets.get("siglip_model", ""))).absolute()
    source = ovimap / "scripts" / "vl_models.py"
    weights = model_root / "model.safetensors"
    external = _nested_mapping(mapping_receipt, "external_runtime", "D2 runtime")
    weight_binding = external.get("siglip_weights")
    _validate_bound_file(weight_binding, weights, "SigLIP weights")
    if source.is_symlink() or not source.is_file():
        raise ObservationPreparationError("OVI VLModel source is unavailable")
    for processor_name in ("config.json", "preprocessor_config.json"):
        processor_path = model_root / processor_name
        if processor_path.is_symlink() or not processor_path.is_file():
            raise ObservationPreparationError(
                f"SigLIP processor source is unavailable: {processor_path}"
            )
    spec = importlib.util.spec_from_file_location(
        "_observation_query_ovimap_vl_models", source
    )
    if spec is None or spec.loader is None:
        raise ObservationPreparationError("OVI VLModel source cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.VLModel(
        "siglip-l-16-384",
        image_shape,
        device,
        model_path=str(model_root),
    )

    def encode(
        rgb: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        return np.asarray(model.encode_image_with_bbox(rgb, mask, bbox))

    manifest = {
        "name": "OVI_MAP_VLModel_SigLIP",
        "model_name": "siglip-l-16-384",
        "device": device,
        "image_shape": list(image_shape),
        "six_crop_source": _file_record(source, str(source)),
        "weights": dict(weight_binding),
        "processor": [
            _file_record(model_root / name, str(model_root / name))
            for name in ("config.json", "preprocessor_config.json")
        ],
        "per_crop_normalization": "OVI_VLModel_L2",
        "mean_normalization": "final_L2",
        "frozen": True,
    }
    return encode, manifest


def prepare_real_observation_artifact(
    *,
    runtime_config_path: str | Path,
    pilot_config_path: str | Path,
    selection_manifest_path: str | Path,
    mapping_receipt_path: str | Path,
    model_bundle_root: str | Path,
    output_root: str | Path,
    pair_id: str | None,
    device: str,
) -> ObservationPreparationArtifactPaths:
    """Prepare and publish one real DEV ObservationBank from frozen raw caches."""

    runtime_path = Path(runtime_config_path).absolute()
    pilot_path = Path(pilot_config_path).absolute()
    selection_path = Path(selection_manifest_path).absolute()
    mapping_path = Path(mapping_receipt_path).absolute()
    runtime = _load_json_object(runtime_path, "runtime config")
    pilot = _load_json_object(pilot_path, "pilot config")
    selection = _load_json_object(selection_path, "selection manifest")
    mapping = _load_json_object(mapping_path, "D2 mapping receipt")
    observation = _nested_mapping(pilot, "observation_bank", "observation config")
    correspondence = _nested_mapping(pilot, "correspondence", "correspondence config")
    selected_pair_id = pair_id
    if selected_pair_id is None:
        runtime_pairs = _nested_mapping(runtime, "pairs", "runtime pairs")
        selected_pair_id = str(
            _nested_mapping(runtime_pairs, "dev", "runtime DEV pair").get(
                "pair_id", ""
            )
        )
    pair_record = find_pair_record(selection, selected_pair_id)
    model_input, points, _center, model_manifest = load_model_observation_bundle(
        model_bundle_root
    )
    frames, materialized_manifests, native_manifests = load_real_pair_frames(
        runtime_config=runtime,
        observation_config=observation,
        pair_id=selected_pair_id,
        pair_record=pair_record,
        mapping_receipt=mapping,
    )
    image_shapes = {frame.rgb.shape[:2] for frame in frames}
    if len(image_shapes) != 1:
        raise ObservationPreparationError("selected RGB frames have inconsistent shapes")
    encoder, encoder_manifest = load_ovimap_siglip_encoder(
        runtime_config=runtime,
        mapping_receipt=mapping,
        image_shape=next(iter(image_shapes)),
        device=device,
    )
    source_manifest = build_observation_source_manifest(
        pair_id=selected_pair_id,
        pair_record=pair_record,
        selection_manifest_path=selection_path,
        mapping_receipt_path=mapping_path,
        model_bundle_manifest_path=Path(model_bundle_root).absolute() / "manifest.json",
        materialized_manifest_paths=materialized_manifests,
        native_mapping_manifest_paths=native_manifests,
        frames=frames,
        feature_encoder_manifest=encoder_manifest,
        observation_config=dict(observation),
    )
    try:
        prepared = build_observation_bank_from_frames(
            pair_id=selected_pair_id,
            model_input_sha256=model_manifest["model_input_sha256"],
            model_points_reference_xyz=points,
            model_visit_ids=model_input.model_visit_ids,
            frames=frames,
            encode_six_crop=encoder,
            source_manifest=source_manifest,
            maximum_regions_per_visit=int(
                observation["max_regions_and_fragments_per_visit"]
            ),
            maximum_fragments_per_parent=int(
                observation["max_fragments_per_parent"]
            ),
            maximum_distinct_frames_per_model_point=int(
                observation["max_distinct_support_frames_per_model_point"]
            ),
            tolerance_m=float(correspondence["positive_depth_tolerance_m"]),
            minimum_valid_neighbours=int(
                observation["minimum_valid_depth_neighbours"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ObservationPreparationError):
            raise
        raise ObservationPreparationError("pilot observation limits are invalid") from error
    return publish_prepared_observation_bank(
        prepared=prepared,
        frames=frames,
        output_root=output_root,
    )


def _warp_rgb_to_depth(frame: RealFrameAssets) -> np.ndarray:
    height, width = frame.depth_m.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    target = np.stack(
        (columns.ravel(), rows.ravel(), np.ones(height * width)), axis=0
    )
    inverse = np.linalg.inv(frame.color_to_depth_homography)
    source = inverse @ target
    scale = source[2]
    finite = np.isfinite(source).all(axis=0) & (
        np.abs(scale) > np.finfo(np.float64).eps
    )
    source_columns = np.zeros(height * width, dtype=np.int64)
    source_rows = np.zeros(height * width, dtype=np.int64)
    source_columns[finite] = np.floor(
        source[0, finite] / scale[finite] + 0.5
    ).astype(np.int64)
    source_rows[finite] = np.floor(
        source[1, finite] / scale[finite] + 0.5
    ).astype(np.int64)
    inside = (
        finite
        & (source_columns >= 0)
        & (source_columns < frame.rgb.shape[1])
        & (source_rows >= 0)
        & (source_rows < frame.rgb.shape[0])
    )
    result = np.zeros((height * width, 3), dtype=np.uint8)
    result[inside] = frame.rgb[source_rows[inside], source_columns[inside]]
    return result.reshape(height, width, 3)


def _label_colors(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(np.uint64, copy=False)
    result = np.stack(
        (
            (values * 37 + 53) % 256,
            (values * 67 + 97) % 256,
            (values * 109 + 193) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    result[labels <= 0] = 0
    return result


def _blend_labels(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    active = labels > 0
    if np.any(active):
        colors = _label_colors(labels)
        result[active] = np.rint(
            0.55 * result[active].astype(np.float32)
            + 0.45 * colors[active].astype(np.float32)
        ).astype(np.uint8)
    return result


def _write_frame_overlay(
    frame: RealFrameAssets,
    support: FrameRegionSupport,
    output: Path,
) -> Path:
    rgb_depth = _warp_rgb_to_depth(frame)
    raw_panel = _blend_labels(rgb_depth, frame.registered_frontend_labels)
    fragment_panel = _blend_labels(rgb_depth, frame.geometric_labels_depth)
    support_panel = rgb_depth.copy()
    if support.candidates:
        pixels = np.unique(
            np.asarray(
                [candidate.pixel_index for candidate in support.candidates],
                dtype=np.int64,
            )
        )
        rows, columns = np.unravel_index(pixels, frame.depth_m.shape)
        support_panel[rows, columns] = [255, 32, 32]
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour_rows = np.clip(rows + row_offset, 0, frame.depth_m.shape[0] - 1)
            neighbour_columns = np.clip(
                columns + column_offset, 0, frame.depth_m.shape[1] - 1
            )
            support_panel[neighbour_rows, neighbour_columns] = [255, 224, 32]
    canvas = np.concatenate((raw_panel, fragment_panel, support_panel), axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        Image.fromarray(canvas, mode="RGB").save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    return output


def _preparation_diagnostics(
    prepared: PreparedObservationBank,
    frames: tuple[RealFrameAssets, ...],
) -> dict[str, object]:
    bank = prepared.bank
    supported_models = np.unique(bank.csr_model_indices)
    model_frames: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for _visit, _source, support in prepared.frame_supports:
        for candidate in support.candidates:
            model_frames[candidate.model_index].add(
                (candidate.visit_id, candidate.source_frame_id)
            )
    distinct_counts = np.asarray(
        [len(model_frames.get(index, ())) for index in range(len(bank.model_visit_ids))],
        dtype=np.int64,
    )
    per_visit: list[dict[str, object]] = []
    for visit_id in (0, 1):
        visit_regions = [
            region for region in prepared.regions if region.visit_id == visit_id
        ]
        visit_edge_rows = np.repeat(
            np.arange(len(prepared.regions)), np.diff(bank.csr_indptr)
        )
        edge_mask = bank.region_visit_ids[visit_edge_rows] == visit_id
        visit_model_indices = bank.csr_model_indices[edge_mask]
        per_visit.append(
            {
                "visit_id": visit_id,
                "scan_uuid": next(
                    frame.scan_uuid for frame in frames if frame.visit_id == visit_id
                ),
                "selected_target_frames": [
                    frame.target_frame_id
                    for frame in frames
                    if frame.visit_id == visit_id
                ],
                "selected_source_frames": [
                    frame.source_frame_id
                    for frame in frames
                    if frame.visit_id == visit_id
                ],
                "region_count": len(visit_regions),
                "parent_region_count": sum(
                    region.fragment_id == -1 for region in visit_regions
                ),
                "fragment_region_count": sum(
                    region.fragment_id >= 0 for region in visit_regions
                ),
                "edge_count": int(np.count_nonzero(edge_mask)),
                "model_points_with_support": len(np.unique(visit_model_indices)),
            }
        )
    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBSERVATION_SUPPORT_DIAGNOSTICS_V1",
        "status": "PASS",
        "pair_id": bank.pair_id,
        "model_input_sha256": bank.model_input_sha256,
        "observation_bank_sha256": bank.content_sha256(),
        "frame_count": len(frames),
        "region_count": len(prepared.regions),
        "parent_region_count": sum(
            region.fragment_id == -1 for region in prepared.regions
        ),
        "fragment_region_count": sum(
            region.fragment_id >= 0 for region in prepared.regions
        ),
        "model_count": len(bank.model_visit_ids),
        "support_candidate_count_before_deduplication": prepared.support_candidate_count,
        "edge_count": len(bank.csr_model_indices),
        "model_points_with_support": len(supported_models),
        "model_points_without_support": int(
            len(bank.model_visit_ids) - len(supported_models)
        ),
        "single_view_model_points": int(np.count_nonzero(distinct_counts == 1)),
        "multi_view_model_points": int(np.count_nonzero(distinct_counts >= 2)),
        "maximum_distinct_support_frames_per_model_point": int(
            distinct_counts.max(initial=0)
        ),
        "depth_relation_counts": dict(prepared.relation_counts),
        "per_visit": per_visit,
        "overlay_panel_order": [
            "registered_raw_frontend_regions",
            "raw_depth_fragments",
            "positive_model_support",
        ],
        "claim_boundary": (
            "Positive CSR edges cover only existing same-visit ReScene M rows; "
            "unknown, occluded, and visible-free rays are not positive evidence, "
            "and no GT identity or class participates in preparation."
        ),
    }


def publish_prepared_observation_bank(
    *,
    prepared: PreparedObservationBank,
    frames: Iterable[RealFrameAssets],
    output_root: str | Path,
) -> ObservationPreparationArtifactPaths:
    """Atomically publish the bank, raw pixels, diagnostics, and fixed overlays."""

    if not isinstance(prepared, PreparedObservationBank):
        raise TypeError("prepared must be PreparedObservationBank")
    frame_values = tuple(frames)
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationPreparationError(f"preparation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        bank_paths = save_observation_bank(prepared.bank, staging / "bank")
        pixel_path = write_region_pixel_sidecar(
            prepared.regions, staging / "region_pixels.npz"
        )
        diagnostics_path = staging / "diagnostics.json"
        diagnostics_path.write_bytes(_json_bytes(_preparation_diagnostics(prepared, frame_values)))

        support_lookup = {
            (visit, source): support
            for visit, source, support in prepared.frame_supports
        }
        visit_zero = sorted(
            (frame for frame in frame_values if frame.visit_id == 0),
            key=lambda frame: frame.source_frame_id,
        )
        overlay_frames: list[RealFrameAssets] = []
        overlay_keys: set[tuple[int, int]] = set()
        if visit_zero:
            for index in (0, len(visit_zero) // 2, len(visit_zero) - 1):
                frame = visit_zero[index]
                key = (frame.visit_id, frame.source_frame_id)
                if key not in overlay_keys:
                    overlay_frames.append(frame)
                    overlay_keys.add(key)
        overlays: list[Path] = []
        for frame in overlay_frames:
            overlays.append(
                _write_frame_overlay(
                    frame,
                    support_lookup[(frame.visit_id, frame.source_frame_id)],
                    staging
                    / "overlays"
                    / (
                        f"visit{frame.visit_id}_target{frame.target_frame_id:06d}_"
                        f"source{frame.source_frame_id:06d}.png"
                    ),
                )
            )
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_PREPARATION_V1",
            "status": "PASS",
            "pair_id": prepared.bank.pair_id,
            "model_input_sha256": prepared.bank.model_input_sha256,
            "observation_bank": {
                "path": "bank",
                "content_sha256": prepared.bank.content_sha256(),
                "manifest": _file_record(bank_paths.manifest, "bank/manifest.json"),
                "arrays": _file_record(bank_paths.arrays, "bank/arrays.npz"),
            },
            "region_pixels": _file_record(pixel_path, "region_pixels.npz"),
            "diagnostics": _file_record(diagnostics_path, "diagnostics.json"),
            "overlays": [
                _file_record(path, str(path.relative_to(staging))) for path in overlays
            ],
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
    return ObservationPreparationArtifactPaths(
        root=output,
        manifest=output / "manifest.json",
        bank_root=output / "bank",
        pixel_sidecar=output / "region_pixels.npz",
        diagnostics=output / "diagnostics.json",
        overlays=tuple(output / path.relative_to(staging) for path in overlays),
    )


__all__ = [
    "FrameRegionSupport",
    "ModelObservationBundlePaths",
    "ObservationPreparationArtifactPaths",
    "ObservationPreparationError",
    "PreparedObservationBank",
    "RealFrameAssets",
    "build_model_bundle_from_d2",
    "build_observation_bank_from_frames",
    "build_observation_source_manifest",
    "encode_parent_region_features",
    "find_pair_record",
    "load_model_observation_bundle",
    "load_ovimap_siglip_encoder",
    "load_real_frame_assets",
    "load_real_pair_frames",
    "mapping_receipt_pair_id",
    "pair_alignment_row",
    "prepare_real_observation_artifact",
    "project_frame_region_support",
    "publish_prepared_observation_bank",
    "save_model_observation_bundle",
    "select_region_budget",
    "write_region_pixel_sidecar",
]


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser(
        "build-model-bundle", help="rebuild and save the exact D2 ReScene M input"
    )
    model.add_argument(
        "--object-transfer-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/ovi_rescene_object_level_transfer_v1.json",
    )
    model.add_argument("--pair-id", required=True)
    model.add_argument("--output-root", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-bank", help="prepare a real source-bound ObservationBank"
    )
    prepare.add_argument(
        "--runtime-config",
        type=Path,
        default=REPO_ROOT / "configs/observation_query/runtime.local.json",
    )
    prepare.add_argument(
        "--pilot-config",
        type=Path,
        default=REPO_ROOT / "configs/observation_query/pilot_v1.json",
    )
    prepare.add_argument(
        "--selection-manifest",
        type=Path,
        default=REPO_ROOT
        / "configs/evaluation/manifests/ovi_rescene_object_level_pairs_v1.json",
    )
    prepare.add_argument(
        "--mapping-receipt",
        type=Path,
        default=REPO_ROOT
        / "configs/evaluation/results/ovi_rescene_object_level_transfer/d2_ovimap_smoke_v1.json",
    )
    prepare.add_argument("--model-bundle", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--pair-id")
    prepare.add_argument("--device", default="cuda:0")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    if args.command == "build-model-bundle":
        paths = build_model_bundle_from_d2(
            object_transfer_config=args.object_transfer_config,
            pair_id=args.pair_id,
            output_root=args.output_root,
        )
        _model_input, points, _center, manifest = load_model_observation_bundle(
            paths.root
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output_root": str(paths.root),
                    "model_input_sha256": manifest["model_input_sha256"],
                    "model_count": len(points),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-bank":
        paths = prepare_real_observation_artifact(
            runtime_config_path=args.runtime_config,
            pilot_config_path=args.pilot_config,
            selection_manifest_path=args.selection_manifest,
            mapping_receipt_path=args.mapping_receipt,
            model_bundle_root=args.model_bundle,
            output_root=args.output_root,
            pair_id=args.pair_id,
            device=args.device,
        )
        manifest = _load_json_object(paths.manifest, "preparation manifest")
        diagnostics = _load_json_object(paths.diagnostics, "support diagnostics")
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output_root": str(paths.root),
                    "observation_bank_sha256": manifest["observation_bank"][
                        "content_sha256"
                    ],
                    "region_count": diagnostics["region_count"],
                    "edge_count": diagnostics["edge_count"],
                    "model_points_with_support": diagnostics[
                        "model_points_with_support"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
