"""Evaluator-only 3RScan instance geometry and identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment

from src.evaluation.json_contracts import loads_strict

Voxel = tuple[int, int, int]
MatchingPolicy = Literal["max_total_iou", "max_valid_count_then_iou"]


class RScanGroundTruthError(ValueError):
    """Raised when 3RScan evaluator-only evidence is malformed."""


def _normalized_voxels(value: object, *, label: str) -> frozenset[Voxel]:
    if not isinstance(value, (set, frozenset)) or not value:
        raise RScanGroundTruthError(f"{label} voxels must be a non-empty set")
    normalized: set[Voxel] = set()
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or any(type(axis) is not int for axis in item)
        ):
            raise RScanGroundTruthError(f"{label} voxel must contain three integers")
        normalized.add(item)
    return frozenset(normalized)


@dataclass(frozen=True, slots=True)
class GroundTruthInstance:
    instance_id: int
    semantic_label: str
    voxels: frozenset[Voxel]

    def __post_init__(self) -> None:
        if type(self.instance_id) is not int or self.instance_id <= 0:
            raise RScanGroundTruthError("GT instance_id must be a positive integer")
        if not isinstance(self.semantic_label, str) or not self.semantic_label:
            raise RScanGroundTruthError("GT semantic_label must be non-empty")
        object.__setattr__(
            self,
            "voxels",
            _normalized_voxels(self.voxels, label="GT instance"),
        )


@dataclass(frozen=True, slots=True)
class PredictedInstance:
    prediction_id: str
    voxels: frozenset[Voxel]

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, str) or not self.prediction_id:
            raise RScanGroundTruthError("prediction_id must be non-empty")
        object.__setattr__(
            self,
            "voxels",
            _normalized_voxels(self.voxels, label="predicted instance"),
        )


@dataclass(frozen=True, slots=True)
class InstanceMatch:
    prediction_id: str
    gt_instance_id: int
    iou: float


@dataclass(frozen=True, slots=True)
class ThresholdGeometryResult:
    threshold: float
    matches: tuple[InstanceMatch, ...]
    unmatched_prediction_ids: tuple[str, ...]
    unmatched_gt_instance_ids: tuple[int, ...]
    duplicate_prediction_count: int
    fragment_gt_count: int
    merge_prediction_count: int
    mean_matched_iou: float | None

    @property
    def matched_count(self) -> int:
        return len(self.matches)


@dataclass(frozen=True, slots=True)
class InstanceGeometryResult:
    primary: ThresholdGeometryResult
    sensitivity: ThresholdGeometryResult


@dataclass(frozen=True, slots=True)
class GroundTruthPair:
    pair_id: str
    voxel_size_m: float
    visits: tuple[tuple[GroundTruthInstance, ...], tuple[GroundTruthInstance, ...]]
    identity_rules: IdentityRules


def _matrix(value: object, *, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RScanGroundTruthError(f"{label} is not numeric") from error
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RScanGroundTruthError(f"{label} must be a finite 4x4 matrix")
    return matrix


def apply_row_vector_transform(points: np.ndarray, transform: object) -> np.ndarray:
    """Apply the official rescan-to-reference row-vector transform."""

    coordinates = np.asarray(points, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1:] != (3,)
        or not np.all(np.isfinite(coordinates))
    ):
        raise RScanGroundTruthError("points must have shape (N, 3) and be finite")
    matrix = _matrix(transform, label="global alignment")
    homogeneous = np.column_stack((coordinates, np.ones(len(coordinates))))
    transformed = homogeneous @ matrix
    weights = transformed[:, 3]
    if np.any(np.isclose(weights, 0.0)):
        raise RScanGroundTruthError("global alignment produced zero homogeneous weight")
    result = transformed[:, :3] / weights[:, None]
    if not np.all(np.isfinite(result)):
        raise RScanGroundTruthError("global alignment produced non-finite coordinates")
    return result


def voxelize_points(
    points: np.ndarray, *, voxel_size_m: float = 0.05
) -> frozenset[Voxel]:
    """Map finite metric points onto a shared floor-quantized voxel grid."""

    if isinstance(voxel_size_m, bool) or not isinstance(voxel_size_m, (int, float)):
        raise RScanGroundTruthError("voxel_size_m must be numeric")
    voxel_size = float(voxel_size_m)
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise RScanGroundTruthError("voxel_size_m must be finite and positive")
    coordinates = np.asarray(points, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1:] != (3,)
        or not len(coordinates)
        or not np.all(np.isfinite(coordinates))
    ):
        raise RScanGroundTruthError("points must have non-empty shape (N, 3) and be finite")
    quantized = np.floor(coordinates / voxel_size).astype(np.int64)
    return frozenset(tuple(int(axis) for axis in row) for row in quantized)


def _iou(left: frozenset[Voxel], right: frozenset[Voxel]) -> float:
    intersection = len(left.intersection(right))
    return intersection / (len(left) + len(right) - intersection)


def _validate_instances(
    predictions: Sequence[PredictedInstance],
    ground_truth: Sequence[GroundTruthInstance],
) -> tuple[tuple[PredictedInstance, ...], tuple[GroundTruthInstance, ...]]:
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise RScanGroundTruthError("predictions must be a sequence")
    if isinstance(ground_truth, (str, bytes)) or not isinstance(ground_truth, Sequence):
        raise RScanGroundTruthError("ground_truth must be a sequence")
    predicted = tuple(predictions)
    targets = tuple(ground_truth)
    if any(not isinstance(value, PredictedInstance) for value in predicted):
        raise RScanGroundTruthError("predictions contain an invalid instance")
    if any(not isinstance(value, GroundTruthInstance) for value in targets):
        raise RScanGroundTruthError("ground_truth contains an invalid instance")
    prediction_ids = [value.prediction_id for value in predicted]
    target_ids = [value.instance_id for value in targets]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise RScanGroundTruthError("prediction IDs are duplicated")
    if len(target_ids) != len(set(target_ids)):
        raise RScanGroundTruthError("GT instance IDs are duplicated")
    return (
        tuple(sorted(predicted, key=lambda value: value.prediction_id)),
        tuple(sorted(targets, key=lambda value: value.instance_id)),
    )


def _threshold_result(
    predictions: tuple[PredictedInstance, ...],
    ground_truth: tuple[GroundTruthInstance, ...],
    ious: np.ndarray,
    *,
    threshold: float,
    matching_policy: MatchingPolicy = "max_total_iou",
) -> ThresholdGeometryResult:
    if matching_policy not in {"max_total_iou", "max_valid_count_then_iou"}:
        raise RScanGroundTruthError("instance matching policy is unsupported")
    accepted: list[InstanceMatch] = []
    if len(predictions) and len(ground_truth):
        if matching_policy == "max_total_iou":
            scores = ious
        else:
            valid = ious >= threshold
            cardinality_weight = min(ious.shape) + 1.0
            scores = valid.astype(np.float64) * cardinality_weight + np.where(
                valid, ious, 0.0
            )
        row_indices, column_indices = linear_sum_assignment(-scores)
        accepted = [
            InstanceMatch(
                predictions[row].prediction_id,
                ground_truth[column].instance_id,
                float(ious[row, column]),
            )
            for row, column in zip(row_indices, column_indices, strict=True)
            if ious[row, column] >= threshold
        ]
    accepted.sort(key=lambda value: (value.prediction_id, value.gt_instance_id))
    matched_predictions = {value.prediction_id for value in accepted}
    matched_targets = {value.gt_instance_id for value in accepted}
    adjacency = ious >= threshold
    predictions_per_target = (
        np.count_nonzero(adjacency, axis=0)
        if len(ground_truth)
        else np.zeros(0, dtype=np.int64)
    )
    targets_per_prediction = (
        np.count_nonzero(adjacency, axis=1)
        if len(predictions)
        else np.zeros(0, dtype=np.int64)
    )
    return ThresholdGeometryResult(
        threshold=threshold,
        matches=tuple(accepted),
        unmatched_prediction_ids=tuple(
            value.prediction_id
            for value in predictions
            if value.prediction_id not in matched_predictions
        ),
        unmatched_gt_instance_ids=tuple(
            value.instance_id
            for value in ground_truth
            if value.instance_id not in matched_targets
        ),
        duplicate_prediction_count=int(
            np.maximum(predictions_per_target - 1, 0).sum()
        ),
        fragment_gt_count=int(np.count_nonzero(predictions_per_target > 1)),
        merge_prediction_count=int(np.count_nonzero(targets_per_prediction > 1)),
        mean_matched_iou=(
            float(np.mean([value.iou for value in accepted])) if accepted else None
        ),
    )


def evaluate_instance_geometry(
    predictions: Sequence[PredictedInstance],
    ground_truth: Sequence[GroundTruthInstance],
    *,
    matching_policy: MatchingPolicy = "max_total_iou",
) -> InstanceGeometryResult:
    """Run deterministic maximum-IoU one-to-one assignment at 0.50 and 0.25."""

    predicted, targets = _validate_instances(predictions, ground_truth)
    ious = np.asarray(
        [[_iou(left.voxels, right.voxels) for right in targets] for left in predicted],
        dtype=np.float64,
    )
    if ious.shape != (len(predicted), len(targets)):
        ious = np.empty((len(predicted), len(targets)), dtype=np.float64)
    return InstanceGeometryResult(
        primary=_threshold_result(
            predicted,
            targets,
            ious,
            threshold=0.50,
            matching_policy=matching_policy,
        ),
        sensitivity=_threshold_result(
            predicted,
            targets,
            ious,
            threshold=0.25,
            matching_policy=matching_policy,
        ),
    )


@dataclass(frozen=True, slots=True)
class IdentityRules:
    rescan_to_reference: Mapping[int, int]
    symmetry_by_reference: Mapping[int, int]
    rigid_transform_by_reference: Mapping[int, tuple[float, ...]]
    removed_reference_ids: frozenset[int]
    ambiguity_groups: tuple[frozenset[int], ...]

    @classmethod
    def from_official_records(
        cls,
        *,
        changes: Mapping[str, object],
        ambiguity: object,
    ) -> IdentityRules:
        if not isinstance(changes, Mapping):
            raise RScanGroundTruthError("official changes must be a mapping")
        mapping: dict[int, int] = {}
        symmetries: dict[int, int] = {}
        transforms: dict[int, tuple[float, ...]] = {}

        def bind(rescan_id: int, reference_id: int) -> None:
            previous = mapping.setdefault(rescan_id, reference_id)
            if previous != reference_id:
                raise RScanGroundTruthError("official cross-time identity conflicts")

        rigid = changes.get("rigid", [])
        nonrigid = changes.get("nonrigid", [])
        removed = changes.get("removed", [])
        if any(not isinstance(value, list) for value in (rigid, nonrigid, removed)):
            raise RScanGroundTruthError("official change arrays are malformed")
        for record in rigid:
            if not isinstance(record, Mapping):
                raise RScanGroundTruthError("rigid change must be a mapping")
            reference_id = _positive_id(
                record.get("instance_reference"), label="rigid reference"
            )
            rescan_id = _positive_id(record.get("instance_rescan"), label="rigid rescan")
            symmetry = record.get("symmetry")
            if type(symmetry) is not int or symmetry < 0:
                raise RScanGroundTruthError("rigid symmetry must be non-negative")
            matrix = _matrix(record.get("transform"), label="rigid transform")
            bind(rescan_id, reference_id)
            symmetries[reference_id] = symmetry
            transforms[reference_id] = tuple(float(value) for value in matrix.reshape(-1))
        for record in nonrigid:
            if isinstance(record, Mapping):
                reference_id = _positive_id(
                    record.get("instance_reference"), label="nonrigid reference"
                )
                rescan_id = _positive_id(
                    record.get("instance_rescan"), label="nonrigid rescan"
                )
            else:
                reference_id = _positive_id(record, label="nonrigid reference")
                rescan_id = reference_id
            bind(rescan_id, reference_id)
        removed_ids = frozenset(
            _positive_id(
                record.get("instance_reference")
                if isinstance(record, Mapping)
                else record,
                label="removed reference",
            )
            for record in removed
        )
        ambiguity_groups = _ambiguity_groups(ambiguity)
        return cls(
            MappingProxyType(dict(sorted(mapping.items()))),
            MappingProxyType(dict(sorted(symmetries.items()))),
            MappingProxyType(dict(sorted(transforms.items()))),
            removed_ids,
            ambiguity_groups,
        )

    def reference_id_for_rescan(self, instance_id: int) -> int:
        value = _positive_id(instance_id, label="rescan instance")
        return self.rescan_to_reference.get(value, value)

    def is_ambiguous_equivalent(self, left: int, right: int) -> bool:
        first = _positive_id(left, label="left ambiguity instance")
        second = _positive_id(right, label="right ambiguity instance")
        return first == second or any(
            first in group and second in group for group in self.ambiguity_groups
        )


def _positive_id(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RScanGroundTruthError(f"{label} must be a positive integer")
    return value


def _ambiguity_groups(value: object) -> tuple[frozenset[int], ...]:
    if not isinstance(value, list):
        raise RScanGroundTruthError("ambiguity must be a list")
    groups: list[frozenset[int]] = []
    for raw_group in value:
        if not isinstance(raw_group, list) or not raw_group:
            raise RScanGroundTruthError("ambiguity group must be non-empty")
        identities: set[int] = set()
        for record in raw_group:
            if not isinstance(record, Mapping):
                raise RScanGroundTruthError("ambiguity record must be a mapping")
            identities.add(
                _positive_id(record.get("instance_source"), label="ambiguity source")
            )
            identities.add(
                _positive_id(record.get("instance_target"), label="ambiguity target")
            )
            _matrix(record.get("transform"), label="ambiguity transform")
        if len(identities) < 2:
            raise RScanGroundTruthError("ambiguity group must contain two identities")
        groups.append(frozenset(identities))
    return tuple(sorted(groups, key=lambda group: tuple(sorted(group))))


def _regular_path(path: str | Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        record = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise RScanGroundTruthError(f"{label} is unavailable") from error
    if not stat.S_ISREG(record.st_mode):
        raise RScanGroundTruthError(f"{label} must be a regular non-symlink file")
    return absolute


def _semantic_labels(path: Path) -> dict[int, str]:
    try:
        payload = loads_strict(path.read_text(encoding="utf-8"), label="3RScan semseg")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RScanGroundTruthError("3RScan semseg is unreadable") from error
    groups = payload.get("segGroups") if isinstance(payload, Mapping) else None
    if not isinstance(groups, list):
        raise RScanGroundTruthError("3RScan semseg lacks segGroups")
    labels: dict[int, str] = {}
    for record in groups:
        if not isinstance(record, Mapping):
            raise RScanGroundTruthError("3RScan semantic record is malformed")
        instance_id = _positive_id(record.get("objectId"), label="semantic objectId")
        label = record.get("label")
        if not isinstance(label, str) or not label:
            raise RScanGroundTruthError("3RScan semantic label is empty")
        if instance_id in labels:
            raise RScanGroundTruthError("3RScan semantic record is duplicated")
        labels[instance_id] = label
    return labels


def load_annotated_instances(
    ply_path: str | Path,
    semantic_path: str | Path,
    *,
    voxel_size_m: float = 0.05,
    transform: object | None = None,
) -> tuple[GroundTruthInstance, ...]:
    """Load positive annotated object IDs and align them onto the common grid."""

    ply_source = _regular_path(ply_path, label="annotated instance PLY")
    semantic_source = _regular_path(semantic_path, label="semantic JSON")
    labels = _semantic_labels(semantic_source)
    try:
        vertices = PlyData.read(ply_source, mmap="c")["vertex"].data
    except (OSError, ValueError, KeyError) as error:
        raise RScanGroundTruthError("annotated instance PLY is unreadable") from error
    names = set(vertices.dtype.names or ())
    required = {"x", "y", "z", "objectId"}
    if not required.issubset(names):
        raise RScanGroundTruthError("annotated instance PLY lacks required properties")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64,
        copy=False,
    )
    if transform is not None:
        points = apply_row_vector_transform(points, transform)
    elif not np.all(np.isfinite(points)):
        raise RScanGroundTruthError("annotated instance PLY contains non-finite points")
    object_ids = np.asarray(vertices["objectId"], dtype=np.int64)
    positive_ids = tuple(int(value) for value in np.unique(object_ids) if value > 0)
    missing = set(positive_ids).difference(labels)
    if missing:
        raise RScanGroundTruthError(
            f"positive instance lacks semantic record: {sorted(missing)}"
        )
    return tuple(
        GroundTruthInstance(
            instance_id,
            labels[instance_id],
            voxelize_points(points[object_ids == instance_id], voxel_size_m=voxel_size_m),
        )
        for instance_id in positive_ids
    )


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


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _file_record(
    path: str | Path, *, label: str, recorded_path: str | None = None
) -> dict[str, object]:
    absolute = _regular_path(path, label=label)
    before = absolute.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with absolute.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise RScanGroundTruthError(f"{label} cannot be read") from error
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise RScanGroundTruthError(f"{label} changed while being read")
    return {
        "path": recorded_path if recorded_path is not None else str(absolute),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _validated_record(
    record: object, *, label: str, root: Path | None = None
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise RScanGroundTruthError(f"{label} binding schema is invalid")
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise RScanGroundTruthError(f"{label} binding path is invalid")
    relative = Path(recorded_path)
    if relative.is_absolute():
        source = relative
    else:
        if root is None or ".." in relative.parts:
            raise RScanGroundTruthError(f"{label} binding path is invalid")
        source = root / relative
    observed = _file_record(source, label=label, recorded_path=recorded_path)
    if dict(record) != observed:
        raise RScanGroundTruthError(f"{label} binding mismatch")
    return source, observed


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def publish_pair_ground_truth(
    pair_record: Mapping[str, object],
    *,
    selection_manifest_record: Mapping[str, object],
    output_root: str | Path,
    voxel_size_m: float = 0.05,
) -> dict[str, object]:
    """Atomically publish an aligned, evaluator-only GT sidecar for one pair."""

    if not isinstance(pair_record, Mapping):
        raise RScanGroundTruthError("pair_record must be a mapping")
    pair_id = pair_record.get("pair_id")
    sessions = pair_record.get("sessions")
    method_inputs = pair_record.get("common_method_inputs")
    evaluator = pair_record.get("evaluator_only")
    if not isinstance(pair_id, str) or not pair_id:
        raise RScanGroundTruthError("pair_record identity is invalid")
    if not isinstance(sessions, list) or len(sessions) != 2:
        raise RScanGroundTruthError("pair_record must contain exactly two sessions")
    if not isinstance(method_inputs, Mapping) or not isinstance(evaluator, Mapping):
        raise RScanGroundTruthError("pair_record method/evaluator boundaries are invalid")
    alignment = method_inputs.get("global_alignment")
    if (
        not isinstance(alignment, Mapping)
        or alignment.get("direction") != "rescan_row_vector_to_reference"
        or alignment.get("storage") != "row_major_flat_4x4"
        or alignment.get("application") != "homogeneous_row_vector_right_multiply"
    ):
        raise RScanGroundTruthError("pair_record global alignment is invalid")
    transform = alignment.get("matrix")
    _matrix(transform, label="global alignment")
    changes = evaluator.get("changes")
    ambiguity = evaluator.get("ambiguity")
    if not isinstance(changes, Mapping):
        raise RScanGroundTruthError("pair_record changes are invalid")
    rules = IdentityRules.from_official_records(changes=changes, ambiguity=ambiguity)
    _, selection_binding = _validated_record(
        selection_manifest_record, label="selection manifest"
    )
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise RScanGroundTruthError(f"GT sidecar output already exists: {output}")

    visits: list[tuple[GroundTruthInstance, ...]] = []
    input_bindings: dict[str, dict[str, dict[str, object]]] = {}
    for visit_index, session in enumerate(sessions):
        if not isinstance(session, Mapping) or session.get("visit_index") != visit_index:
            raise RScanGroundTruthError("pair session order is invalid")
        raw_assets = session.get("raw_assets")
        if not isinstance(raw_assets, Mapping):
            raise RScanGroundTruthError("pair raw assets are invalid")
        visit_bindings: dict[str, dict[str, object]] = {}
        paths: dict[str, Path] = {}
        for member in ("labels.instances.annotated.v2.ply", "semseg.v2.json"):
            if member not in raw_assets:
                raise RScanGroundTruthError(f"pair raw assets lack {member}")
            path, binding = _validated_record(
                raw_assets[member], label=f"visit {visit_index} {member}"
            )
            paths[member] = path
            visit_bindings[member] = binding
        input_bindings[str(visit_index)] = visit_bindings
        visits.append(
            load_annotated_instances(
                paths["labels.instances.annotated.v2.ply"],
                paths["semseg.v2.json"],
                voxel_size_m=voxel_size_m,
                transform=None if visit_index == 0 else transform,
            )
        )

    metadata_instances: list[dict[str, object]] = []
    offsets = [0]
    voxel_rows: list[tuple[int, int, int]] = []
    for visit_index, instances in enumerate(visits):
        for instance in instances:
            ordered_voxels = sorted(instance.voxels)
            voxel_rows.extend(ordered_voxels)
            offsets.append(len(voxel_rows))
            metadata_instances.append(
                {
                    "visit_index": visit_index,
                    "instance_id": instance.instance_id,
                    "semantic_label": instance.semantic_label,
                }
            )
    coordinates = np.asarray(voxel_rows, dtype=np.int64)
    if coordinates.shape != (len(voxel_rows), 3):
        coordinates = np.empty((0, 3), dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "artifact_id": "RSCAN_PAIR_GT_METADATA_V1",
        "pair_id": pair_id,
        "voxel_size_m": float(voxel_size_m),
        "instances": metadata_instances,
        "official_identity": {
            "changes": dict(changes),
            "ambiguity": ambiguity,
        },
        "global_alignment": dict(alignment),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "instances.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                voxel_offsets=np.asarray(offsets, dtype=np.int64),
                voxel_coordinates=coordinates,
            )
            stream.flush()
            os.fsync(stream.fileno())
        metadata_path = staging / "metadata.json"
        _write_exclusive(metadata_path, _json_bytes(metadata))
        label_counts = Counter(
            value.semantic_label for instances in visits for value in instances
        )
        manifest = {
            "schema_version": 1,
            "artifact_id": "RSCAN_PAIR_GT_V1",
            "status": "PASS",
            "pair_id": pair_id,
            "pair_record_sha256": _canonical_sha256(pair_record),
            "voxel_size_m": float(voxel_size_m),
            "selection_manifest": selection_binding,
            "input_bindings": input_bindings,
            "arrays": _file_record(
                arrays_path, label="GT arrays", recorded_path="instances.npz"
            ),
            "metadata": _file_record(
                metadata_path, label="GT metadata", recorded_path="metadata.json"
            ),
            "instance_counts": {
                str(index): len(instances) for index, instances in enumerate(visits)
            },
            "voxel_counts": {
                str(index): sum(len(value.voxels) for value in instances)
                for index, instances in enumerate(visits)
            },
            "semantic_label_counts": dict(sorted(label_counts.items())),
            "identity_rule_counts": {
                "mapped": len(rules.rescan_to_reference),
                "removed": len(rules.removed_reference_ids),
                "symmetric": len(rules.symmetry_by_reference),
                "ambiguity_groups": len(rules.ambiguity_groups),
            },
        }
        _write_exclusive(staging / "manifest.json", _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_pair_ground_truth(path: str | Path) -> GroundTruthPair:
    """Load a GT sidecar after revalidating its selection and internal bindings."""

    manifest_path = _regular_path(path, label="GT sidecar manifest")
    try:
        manifest = loads_strict(
            manifest_path.read_text(encoding="utf-8"), label="GT sidecar manifest"
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RScanGroundTruthError("GT sidecar manifest is invalid") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "pair_id",
        "pair_record_sha256",
        "voxel_size_m",
        "selection_manifest",
        "input_bindings",
        "arrays",
        "metadata",
        "instance_counts",
        "voxel_counts",
        "semantic_label_counts",
        "identity_rule_counts",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "RSCAN_PAIR_GT_V1"
        or manifest.get("status") != "PASS"
    ):
        raise RScanGroundTruthError("GT sidecar identity is invalid")
    pair_id = manifest.get("pair_id")
    voxel_size = manifest.get("voxel_size_m")
    if (
        not isinstance(pair_id, str)
        or not pair_id
        or isinstance(voxel_size, bool)
        or not isinstance(voxel_size, (int, float))
        or not math.isfinite(float(voxel_size))
        or float(voxel_size) <= 0
    ):
        raise RScanGroundTruthError("GT sidecar values are invalid")
    _validated_record(manifest.get("selection_manifest"), label="selection manifest")
    root = manifest_path.parent
    arrays_path, _ = _validated_record(
        manifest.get("arrays"), label="GT arrays", root=root
    )
    metadata_path, _ = _validated_record(
        manifest.get("metadata"), label="GT metadata", root=root
    )
    try:
        metadata = loads_strict(
            metadata_path.read_text(encoding="utf-8"), label="GT metadata"
        )
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {"voxel_offsets", "voxel_coordinates"}:
                raise RScanGroundTruthError("GT array schema is invalid")
            offsets = archive["voxel_offsets"]
            coordinates = archive["voxel_coordinates"]
    except (OSError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, RScanGroundTruthError):
            raise
        raise RScanGroundTruthError("GT sidecar payload is unreadable") from error
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != 1
        or metadata.get("artifact_id") != "RSCAN_PAIR_GT_METADATA_V1"
        or metadata.get("pair_id") != pair_id
        or metadata.get("voxel_size_m") != voxel_size
    ):
        raise RScanGroundTruthError("GT metadata identity is invalid")
    rows = metadata.get("instances")
    official = metadata.get("official_identity")
    if not isinstance(rows, list) or not isinstance(official, Mapping):
        raise RScanGroundTruthError("GT metadata records are invalid")
    if (
        offsets.shape != (len(rows) + 1,)
        or not np.issubdtype(offsets.dtype, np.integer)
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (3,)
        or not np.issubdtype(coordinates.dtype, np.integer)
        or offsets[0] != 0
        or offsets[-1] != len(coordinates)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise RScanGroundTruthError("GT arrays are inconsistent")
    visits: list[list[GroundTruthInstance]] = [[], []]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "visit_index",
            "instance_id",
            "semantic_label",
        }:
            raise RScanGroundTruthError("GT instance metadata is invalid")
        visit_index = row.get("visit_index")
        if visit_index not in (0, 1):
            raise RScanGroundTruthError("GT instance visit is invalid")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        voxels = frozenset(
            tuple(int(axis) for axis in coordinate)
            for coordinate in coordinates[start:stop]
        )
        visits[visit_index].append(
            GroundTruthInstance(
                row.get("instance_id"), row.get("semantic_label"), voxels
            )
        )
    rules = IdentityRules.from_official_records(
        changes=official.get("changes"), ambiguity=official.get("ambiguity")
    )
    return GroundTruthPair(
        pair_id=pair_id,
        voxel_size_m=float(voxel_size),
        visits=(tuple(visits[0]), tuple(visits[1])),
        identity_rules=rules,
    )


def build_ground_truth_sidecars(
    selection_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Audit a selection and atomically publish GT sidecars for all selected pairs."""

    from src.evaluation.rscan_t2_dev import audit_t2_development_manifest

    selection_path = _regular_path(
        selection_manifest_path, label="RSCAN_T2_DEV_V1 manifest"
    )
    selection = audit_t2_development_manifest(selection_path)
    selection_record = _file_record(selection_path, label="RSCAN_T2_DEV_V1 manifest")
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise RScanGroundTruthError(f"GT output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        pair_records: list[dict[str, object]] = []
        total_instances = Counter({"0": 0, "1": 0})
        total_voxels = Counter({"0": 0, "1": 0})
        for pair in selection["pairs"]:
            pair_id = pair["pair_id"]
            pair_manifest = publish_pair_ground_truth(
                pair,
                selection_manifest_record=selection_record,
                output_root=staging / pair_id,
            )
            for visit in ("0", "1"):
                total_instances[visit] += pair_manifest["instance_counts"][visit]
                total_voxels[visit] += pair_manifest["voxel_counts"][visit]
            pair_records.append(
                {
                    "pair_id": pair_id,
                    "manifest": _file_record(
                        staging / pair_id / "manifest.json",
                        label=f"{pair_id} GT manifest",
                        recorded_path=f"{pair_id}/manifest.json",
                    ),
                    "instance_counts": pair_manifest["instance_counts"],
                    "voxel_counts": pair_manifest["voxel_counts"],
                    "identity_rule_counts": pair_manifest["identity_rule_counts"],
                }
            )
        summary = {
            "schema_version": 1,
            "artifact_id": "RSCAN_T2_DEV_GT_COLLECTION_V1",
            "status": "PASS",
            "voxel_size_m": 0.05,
            "selection_manifest": selection_record,
            "pair_count": len(pair_records),
            "instance_counts": dict(total_instances),
            "voxel_counts": dict(total_voxels),
            "pairs": pair_records,
        }
        _write_exclusive(staging / "summary.json", _json_bytes(summary))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "GroundTruthInstance",
    "GroundTruthPair",
    "IdentityRules",
    "InstanceGeometryResult",
    "InstanceMatch",
    "MatchingPolicy",
    "PredictedInstance",
    "RScanGroundTruthError",
    "ThresholdGeometryResult",
    "apply_row_vector_transform",
    "build_ground_truth_sidecars",
    "evaluate_instance_geometry",
    "load_annotated_instances",
    "load_pair_ground_truth",
    "publish_pair_ground_truth",
    "voxelize_points",
]
