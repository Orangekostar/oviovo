"""Deterministic rigid registration for source-bound two-visit OVI entities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from itertools import permutations, product
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from src.oviv2.two_visit_contracts import PairRelation

REGISTRATION_METHOD_ID = "ovi_two_visit_trimmed_icp_v1"
ORACLE_REGISTRATION_METHOD_ID = "ovi_two_visit_3rscan_transform_oracle_v1"
_REGISTRATION_METHOD_IDS = frozenset(
    {REGISTRATION_METHOD_ID, ORACLE_REGISTRATION_METHOD_ID}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ELIGIBLE_STATES = frozenset({"persistent_static", "persistent_moved"})
_DEFAULT_RECOVERABLE_LABELS = frozenset(
    {
        "Bed",
        "Bin",
        "Books",
        "Chair",
        "Couch",
        "Drawer",
        "Fridge",
        "Lamp",
        "Painting",
        "Screens",
        "Table",
        "Vase",
    }
)
_RIGID_ATOL = 1e-6


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _points(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must be numeric")
    try:
        points = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be convertible to float64") from exc
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values")
    points.setflags(write=False)
    return points


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    payload = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def point_cloud_sha256(points_xyz: object) -> str:
    """Return the canonical digest used by registration input witnesses."""

    return _array_sha256(_points(points_xyz, name="points_xyz"))


def validate_rigid_transform(value: object) -> np.ndarray:
    """Return an immutable finite proper SE(3) matrix."""

    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError("rigid transform must be numeric")
    try:
        transform = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("rigid transform must be convertible to float64") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError("rigid transform must have a homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=_RIGID_ATOL):
        raise ValueError("rigid transform rotation must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=_RIGID_ATOL):
        raise ValueError("rigid transform rotation must have determinant one")
    transform.setflags(write=False)
    return transform


def apply_rigid_transform(points_xyz: object, transform: object) -> np.ndarray:
    """Apply a proper SE(3) matrix without mutating the source points."""

    points = _points(points_xyz, name="points_xyz")
    matrix = validate_rigid_transform(transform)
    result = points @ matrix[:3, :3].T + matrix[:3, 3]
    result.setflags(write=False)
    return result


def official_row_vector_to_internal_transform(value: object) -> np.ndarray:
    """Convert one official row-vector SE(3) matrix to the internal convention."""

    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError("official row-vector transform must be numeric")
    try:
        matrix = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "official row-vector transform must be convertible to float64"
        ) from exc
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("official row-vector transform must be a finite 4x4 matrix")
    return validate_rigid_transform(matrix.T)


def apply_rigid_transform_normals(normals_xyz: object, transform: object) -> np.ndarray:
    """Rotate normals with an internal SE(3) transform, ignoring translation."""

    normals = _points(normals_xyz, name="normals_xyz")
    matrix = validate_rigid_transform(transform)
    result = normals @ matrix[:3, :3].T
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    registration_voxel_size_m: float = 0.02
    maximum_sample_points: int = 20_000
    maximum_iterations: int = 40
    trimmed_correspondence_fraction: float = 0.80
    maximum_correspondence_distance_m: float = 0.15
    inlier_distance_m: float = 0.05
    minimum_sample_points: int = 64
    minimum_inlier_count: int = 64
    minimum_directional_overlap: float = 0.35
    maximum_directional_median_m: float = 0.08
    maximum_directional_p90_m: float = 0.20
    minimum_second_singular_ratio: float = 0.02
    maximum_centroid_residual_m: float = 0.10
    minimum_principal_extent_ratio: float = 0.40
    maximum_principal_extent_ratio: float = 2.50
    maximum_static_centroid_displacement_m: float = 0.15
    maximum_moved_centroid_displacement_m: float = 2.00
    convergence_tolerance: float = 1e-5
    recoverable_semantic_labels: frozenset[str] = _DEFAULT_RECOVERABLE_LABELS

    def __post_init__(self) -> None:
        positive = (
            "registration_voxel_size_m",
            "maximum_correspondence_distance_m",
            "inlier_distance_m",
            "maximum_directional_median_m",
            "maximum_directional_p90_m",
            "maximum_centroid_residual_m",
            "minimum_principal_extent_ratio",
            "maximum_principal_extent_ratio",
            "maximum_static_centroid_displacement_m",
            "maximum_moved_centroid_displacement_m",
            "convergence_tolerance",
        )
        for name in positive:
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name, minimum=np.nextafter(0.0, 1.0)),
            )
        probabilities = (
            "trimmed_correspondence_fraction",
            "minimum_directional_overlap",
            "minimum_second_singular_ratio",
        )
        for name in probabilities:
            number = _finite(getattr(self, name), name=name, minimum=0.0)
            if not 0.0 < number <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, number)
        for name in (
            "maximum_sample_points",
            "maximum_iterations",
            "minimum_sample_points",
            "minimum_inlier_count",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        if self.maximum_sample_points < self.minimum_sample_points:
            raise ValueError(
                "maximum_sample_points cannot be below minimum_sample_points"
            )
        if self.minimum_inlier_count > self.maximum_sample_points:
            raise ValueError("minimum_inlier_count cannot exceed maximum_sample_points")
        if self.inlier_distance_m > self.maximum_correspondence_distance_m:
            raise ValueError("inlier distance cannot exceed correspondence distance")
        if self.maximum_directional_median_m > self.maximum_directional_p90_m:
            raise ValueError("median threshold cannot exceed p90 threshold")
        if self.minimum_principal_extent_ratio > self.maximum_principal_extent_ratio:
            raise ValueError("principal extent ratio bounds are reversed")
        labels = self.recoverable_semantic_labels
        if (
            not isinstance(labels, frozenset)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError(
                "recoverable_semantic_labels must be a non-empty frozenset"
            )
        normalized = frozenset(label.strip() for label in labels)
        if len({label.casefold() for label in normalized}) != len(normalized):
            raise ValueError(
                "recoverable semantic labels must be case-insensitively unique"
            )
        object.__setattr__(self, "recoverable_semantic_labels", normalized)

    def to_json_record(self) -> dict[str, object]:
        record = asdict(self)
        record["recoverable_semantic_labels"] = sorted(
            self.recoverable_semantic_labels, key=str.casefold
        )
        return record

    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_json_record(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class OracleTransformSource:
    """Evaluator-only official 3RScan object-motion source."""

    reference_instance_id: int
    official_object_reference_to_rescan_row: np.ndarray
    official_rescan_to_reference_row: np.ndarray
    source_type: str = "official_3rscan_rigid_object_row_v1"

    def __post_init__(self) -> None:
        if type(self.reference_instance_id) is not int or self.reference_instance_id <= 0:
            raise ValueError("reference_instance_id must be a positive integer")
        if self.source_type != "official_3rscan_rigid_object_row_v1":
            raise ValueError("oracle transform source type is invalid")
        object_row = official_row_vector_to_internal_transform(
            self.official_object_reference_to_rescan_row
        ).T.copy()
        global_row = official_row_vector_to_internal_transform(
            self.official_rescan_to_reference_row
        ).T.copy()
        object_row.setflags(write=False)
        global_row.setflags(write=False)
        object.__setattr__(
            self, "official_object_reference_to_rescan_row", object_row
        )
        object.__setattr__(self, "official_rescan_to_reference_row", global_row)

    @property
    def transform_world_from_t0(self) -> np.ndarray:
        combined_row = (
            self.official_object_reference_to_rescan_row
            @ self.official_rescan_to_reference_row
        )
        return official_row_vector_to_internal_transform(combined_row)

    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "reference_instance_id": self.reference_instance_id,
                    "source_type": self.source_type,
                    "official_object_reference_to_rescan_row": (
                        self.official_object_reference_to_rescan_row.tolist()
                    ),
                    "official_rescan_to_reference_row": (
                        self.official_rescan_to_reference_row.tolist()
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class RegistrationEvidence:
    method_id: str
    config_sha256: str
    relation_id: str
    relation_state: str
    identity_source: str
    t0_entity_ids: tuple[str, ...]
    t1_entity_ids: tuple[str, ...]
    semantic_label: str | None
    source_points_sha256: str
    target_points_sha256: str
    source_point_count: int
    target_point_count: int
    source_sample_count: int
    target_sample_count: int
    transform_world_from_t0: np.ndarray | None
    selected_initialization: str | None
    iteration_count: int
    source_to_target_inlier_count: int
    target_to_source_inlier_count: int
    source_to_target_overlap: float | None
    target_to_source_overlap: float | None
    source_to_target_median_m: float | None
    target_to_source_median_m: float | None
    source_to_target_p90_m: float | None
    target_to_source_p90_m: float | None
    centroid_residual_m: float | None
    centroid_displacement_m: float | None
    principal_extent_ratios: tuple[float, float, float] | None
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("method_id", "relation_id", "relation_state", "identity_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        if self.method_id not in _REGISTRATION_METHOD_IDS:
            raise ValueError("registration method identity is invalid")
        if (
            not isinstance(self.config_sha256, str)
            or _SHA256.fullmatch(self.config_sha256) is None
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256")
        for name in ("source_points_sha256", "target_points_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("t0_entity_ids", "t1_entity_ids"):
            raw_values = getattr(self, name)
            if not isinstance(raw_values, tuple):
                raise TypeError(f"{name} must be a tuple")
            values = tuple(str(item).strip() for item in raw_values)
            if any(not item for item in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty IDs")
            object.__setattr__(self, name, values)
        if self.semantic_label is not None:
            if (
                not isinstance(self.semantic_label, str)
                or not self.semantic_label.strip()
            ):
                raise ValueError("semantic_label must be non-empty when present")
            object.__setattr__(self, "semantic_label", self.semantic_label.strip())
        for name in (
            "source_point_count",
            "target_point_count",
            "source_sample_count",
            "target_sample_count",
            "iteration_count",
            "source_to_target_inlier_count",
            "target_to_source_inlier_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        for name in ("source_to_target_overlap", "target_to_source_overlap"):
            value = getattr(self, name)
            if value is not None:
                number = _finite(value, name=name, minimum=0.0)
                if number > 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
                object.__setattr__(self, name, number)
        for name in (
            "source_to_target_median_m",
            "target_to_source_median_m",
            "source_to_target_p90_m",
            "target_to_source_p90_m",
            "centroid_residual_m",
            "centroid_displacement_m",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name=name, minimum=0.0))
        if self.principal_extent_ratios is not None:
            if (
                not isinstance(self.principal_extent_ratios, tuple)
                or len(self.principal_extent_ratios) != 3
            ):
                raise ValueError("principal_extent_ratios must have three values")
            ratios = tuple(
                _finite(
                    value, name="principal_extent_ratio", minimum=np.nextafter(0.0, 1.0)
                )
                for value in self.principal_extent_ratios
            )
            object.__setattr__(self, "principal_extent_ratios", ratios)
        transform = None
        if self.transform_world_from_t0 is not None:
            transform = validate_rigid_transform(self.transform_world_from_t0)
        if not isinstance(self.rejection_reasons, tuple):
            raise TypeError("rejection_reasons must be a tuple")
        reasons = tuple(
            sorted({str(reason).strip() for reason in self.rejection_reasons})
        )
        if any(not reason for reason in reasons):
            raise ValueError("rejection reasons must be non-empty")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be boolean")
        if self.accepted:
            if reasons or transform is None or self.selected_initialization is None:
                raise ValueError(
                    "accepted registration requires a transform and no reasons"
                )
            quality = (
                self.source_to_target_overlap,
                self.target_to_source_overlap,
                self.source_to_target_median_m,
                self.target_to_source_median_m,
                self.source_to_target_p90_m,
                self.target_to_source_p90_m,
                self.centroid_residual_m,
                self.centroid_displacement_m,
                self.principal_extent_ratios,
            )
            if any(value is None for value in quality):
                raise ValueError(
                    "accepted registration requires complete quality measurements"
                )
            if len(self.t0_entity_ids) != 1 or len(self.t1_entity_ids) != 1:
                raise ValueError("accepted registration must bind one entity per visit")
        elif not reasons or transform is not None:
            raise ValueError(
                "rejected registration requires reasons and no usable transform"
            )
        if self.selected_initialization is not None and (
            not isinstance(self.selected_initialization, str)
            or not self.selected_initialization.strip()
        ):
            raise ValueError("selected_initialization must be non-empty")
        if self.selected_initialization is not None:
            object.__setattr__(
                self, "selected_initialization", self.selected_initialization.strip()
            )
        object.__setattr__(self, "transform_world_from_t0", transform)
        object.__setattr__(self, "rejection_reasons", reasons)

    def to_json_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "config_sha256": self.config_sha256,
            "relation_id": self.relation_id,
            "relation_state": self.relation_state,
            "identity_source": self.identity_source,
            "t0_entity_ids": list(self.t0_entity_ids),
            "t1_entity_ids": list(self.t1_entity_ids),
            "semantic_label": self.semantic_label,
            "source_points_sha256": self.source_points_sha256,
            "target_points_sha256": self.target_points_sha256,
            "source_point_count": self.source_point_count,
            "target_point_count": self.target_point_count,
            "source_sample_count": self.source_sample_count,
            "target_sample_count": self.target_sample_count,
            "transform_world_from_t0": (
                None
                if self.transform_world_from_t0 is None
                else self.transform_world_from_t0.tolist()
            ),
            "selected_initialization": self.selected_initialization,
            "iteration_count": self.iteration_count,
            "source_to_target_inlier_count": self.source_to_target_inlier_count,
            "target_to_source_inlier_count": self.target_to_source_inlier_count,
            "source_to_target_overlap": self.source_to_target_overlap,
            "target_to_source_overlap": self.target_to_source_overlap,
            "source_to_target_median_m": self.source_to_target_median_m,
            "target_to_source_median_m": self.target_to_source_median_m,
            "source_to_target_p90_m": self.source_to_target_p90_m,
            "target_to_source_p90_m": self.target_to_source_p90_m,
            "centroid_residual_m": self.centroid_residual_m,
            "centroid_displacement_m": self.centroid_displacement_m,
            "principal_extent_ratios": (
                None
                if self.principal_extent_ratios is None
                else list(self.principal_extent_ratios)
            ),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }

    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_json_record(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _Candidate:
    transform: np.ndarray
    initialization: str
    iterations: int
    source_to_target_distances: np.ndarray
    target_to_source_distances: np.ndarray
    source_to_target_inliers: int
    target_to_source_inliers: int
    source_to_target_overlap: float
    target_to_source_overlap: float
    source_to_target_median: float
    target_to_source_median: float
    source_to_target_p90: float
    target_to_source_p90: float
    centroid_residual: float
    centroid_displacement: float
    extent_ratios: tuple[float, float, float]


def _voxel_sample(points: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    keys = np.floor(points / config.registration_voxel_size_m).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sampled = sums / counts[:, None]
    if len(sampled) > config.maximum_sample_points:
        indices = np.linspace(
            0,
            len(sampled) - 1,
            config.maximum_sample_points,
            dtype=np.int64,
        )
        sampled = sampled[indices]
    sampled.setflags(write=False)
    return sampled


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    centered = points - points.mean(axis=0)
    _, singular, axes_t = np.linalg.svd(centered, full_matrices=False)
    axes = axes_t.T
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    valid = bool(
        len(singular) >= 2
        and singular[0] > np.finfo(np.float64).eps
        and singular[1] / singular[0] > 0.0
    )
    scales = singular / math.sqrt(max(len(points) - 1, 1))
    return axes, scales, valid


def _proper_axis_permutations() -> tuple[tuple[str, np.ndarray], ...]:
    values: list[tuple[str, np.ndarray]] = []
    for order in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for column, row in enumerate(order):
                matrix[row, column] = signs[column]
            if np.linalg.det(matrix) > 0.0:
                identifier = (
                    "pca_"
                    + "".join(str(value) for value in order)
                    + "_"
                    + "".join("p" if value > 0.0 else "n" for value in signs)
                )
                values.append((identifier, matrix))
    return tuple(values)


def _initial_transform(
    rotation: np.ndarray, source_centroid: np.ndarray, target_centroid: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_centroid - rotation @ source_centroid
    return transform


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    try:
        left, singular, right_t = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    if len(singular) < 2 or singular[1] <= np.finfo(np.float64).eps:
        return None
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    try:
        return validate_rigid_transform(transform)
    except (TypeError, ValueError):
        return None


def _rotation_delta(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _run_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, int]:
    transform = np.array(initial, dtype=np.float64, copy=True)
    tree = cKDTree(target)
    iterations = 0
    for iteration in range(config.maximum_iterations):
        moved = source @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1, workers=1)
        valid = np.flatnonzero(distances <= config.maximum_correspondence_distance_m)
        if len(valid) < config.minimum_inlier_count:
            break
        order = valid[np.argsort(distances[valid], kind="stable")]
        keep_count = max(
            config.minimum_inlier_count,
            math.ceil(len(order) * config.trimmed_correspondence_fraction),
        )
        selected = order[:keep_count]
        delta = _kabsch(moved[selected], target[indices[selected]])
        if delta is None:
            break
        transform = np.asarray(delta) @ transform
        iterations = iteration + 1
        if (
            np.linalg.norm(delta[:3, 3]) <= config.convergence_tolerance
            and _rotation_delta(delta[:3, :3]) <= config.convergence_tolerance
        ):
            break
    return validate_rigid_transform(transform), iterations


def _measure_candidate(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    initialization: str,
    iterations: int,
    source_scales: np.ndarray,
    target_scales: np.ndarray,
    config: RegistrationConfig,
) -> _Candidate:
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    source_to_target, _ = cKDTree(target).query(moved, k=1, workers=1)
    target_to_source, _ = cKDTree(moved).query(target, k=1, workers=1)
    source_to_target = np.asarray(source_to_target, dtype=np.float64)
    target_to_source = np.asarray(target_to_source, dtype=np.float64)
    source_inliers = int(np.count_nonzero(source_to_target <= config.inlier_distance_m))
    target_inliers = int(np.count_nonzero(target_to_source <= config.inlier_distance_m))
    scale_floor = max(float(source_scales[0]), float(target_scales[0])) * 1e-8
    ratios_list: list[float] = []
    for source_scale, target_scale in zip(source_scales, target_scales, strict=True):
        if source_scale <= scale_floor and target_scale <= scale_floor:
            ratios_list.append(1.0)
        elif source_scale <= scale_floor or target_scale <= scale_floor:
            ratios_list.append(config.maximum_principal_extent_ratio + 1.0)
        else:
            ratios_list.append(float(target_scale / source_scale))
    ratios = tuple(ratios_list)
    return _Candidate(
        transform=validate_rigid_transform(transform),
        initialization=initialization,
        iterations=iterations,
        source_to_target_distances=source_to_target,
        target_to_source_distances=target_to_source,
        source_to_target_inliers=source_inliers,
        target_to_source_inliers=target_inliers,
        source_to_target_overlap=source_inliers / len(source),
        target_to_source_overlap=target_inliers / len(target),
        source_to_target_median=float(np.median(source_to_target)),
        target_to_source_median=float(np.median(target_to_source)),
        source_to_target_p90=float(np.quantile(source_to_target, 0.9)),
        target_to_source_p90=float(np.quantile(target_to_source, 0.9)),
        centroid_residual=float(
            np.linalg.norm(moved.mean(axis=0) - target.mean(axis=0))
        ),
        centroid_displacement=float(
            np.linalg.norm(moved.mean(axis=0) - source.mean(axis=0))
        ),
        extent_ratios=ratios,
    )


def _candidate_key(
    candidate: _Candidate,
) -> tuple[float, float, float, float, float, str]:
    symmetric_median = 0.5 * (
        candidate.source_to_target_median + candidate.target_to_source_median
    )
    symmetric_p90 = 0.5 * (
        candidate.source_to_target_p90 + candidate.target_to_source_p90
    )
    transform_magnitude = candidate.centroid_displacement + _rotation_delta(
        candidate.transform[:3, :3]
    )
    return (
        -min(candidate.source_to_target_overlap, candidate.target_to_source_overlap),
        symmetric_median,
        symmetric_p90,
        candidate.centroid_residual,
        transform_magnitude,
        candidate.initialization,
    )


def _evidence(
    *,
    relation: PairRelation,
    config: RegistrationConfig,
    source: np.ndarray,
    target: np.ndarray,
    source_sample_count: int,
    target_sample_count: int,
    semantic_label: str | None,
    candidate: _Candidate | None,
    reasons: tuple[str, ...],
    method_id: str = REGISTRATION_METHOD_ID,
) -> RegistrationEvidence:
    accepted = not reasons and candidate is not None
    return RegistrationEvidence(
        method_id=method_id,
        config_sha256=config.content_sha256(),
        relation_id=str(relation.temporal_query_id),
        relation_state=relation.state,
        identity_source=relation.identity_source,
        t0_entity_ids=relation.t0_entity_ids,
        t1_entity_ids=relation.t1_entity_ids,
        semantic_label=semantic_label,
        source_points_sha256=_array_sha256(source),
        target_points_sha256=_array_sha256(target),
        source_point_count=len(source),
        target_point_count=len(target),
        source_sample_count=source_sample_count,
        target_sample_count=target_sample_count,
        transform_world_from_t0=(candidate.transform if accepted else None),
        selected_initialization=(
            None if candidate is None else candidate.initialization
        ),
        iteration_count=(0 if candidate is None else candidate.iterations),
        source_to_target_inlier_count=(
            0 if candidate is None else candidate.source_to_target_inliers
        ),
        target_to_source_inlier_count=(
            0 if candidate is None else candidate.target_to_source_inliers
        ),
        source_to_target_overlap=(
            None if candidate is None else candidate.source_to_target_overlap
        ),
        target_to_source_overlap=(
            None if candidate is None else candidate.target_to_source_overlap
        ),
        source_to_target_median_m=(
            None if candidate is None else candidate.source_to_target_median
        ),
        target_to_source_median_m=(
            None if candidate is None else candidate.target_to_source_median
        ),
        source_to_target_p90_m=(
            None if candidate is None else candidate.source_to_target_p90
        ),
        target_to_source_p90_m=(
            None if candidate is None else candidate.target_to_source_p90
        ),
        centroid_residual_m=(
            None if candidate is None else candidate.centroid_residual
        ),
        centroid_displacement_m=(
            None if candidate is None else candidate.centroid_displacement
        ),
        principal_extent_ratios=(
            None if candidate is None else candidate.extent_ratios
        ),
        accepted=accepted,
        rejection_reasons=reasons,
    )


def _quality_reasons(
    candidate: _Candidate,
    relation: PairRelation,
    config: RegistrationConfig,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if (
        candidate.source_to_target_inliers < config.minimum_inlier_count
        or candidate.target_to_source_inliers < config.minimum_inlier_count
    ):
        reasons.add("insufficient_inliers")
    if (
        candidate.source_to_target_overlap < config.minimum_directional_overlap
        or candidate.target_to_source_overlap < config.minimum_directional_overlap
    ):
        reasons.add("insufficient_directional_overlap")
    if (
        candidate.source_to_target_median > config.maximum_directional_median_m
        or candidate.target_to_source_median > config.maximum_directional_median_m
    ):
        reasons.add("median_residual_too_large")
    if (
        candidate.source_to_target_p90 > config.maximum_directional_p90_m
        or candidate.target_to_source_p90 > config.maximum_directional_p90_m
    ):
        reasons.add("p90_residual_too_large")
    if candidate.centroid_residual > config.maximum_centroid_residual_m:
        reasons.add("centroid_residual_too_large")
    if any(
        ratio < config.minimum_principal_extent_ratio
        or ratio > config.maximum_principal_extent_ratio
        for ratio in candidate.extent_ratios
    ):
        reasons.add("extent_ratio_out_of_range")
    maximum_displacement = (
        config.maximum_static_centroid_displacement_m
        if relation.state == "persistent_static"
        else config.maximum_moved_centroid_displacement_m
    )
    if candidate.centroid_displacement > maximum_displacement:
        reasons.add("excessive_centroid_displacement")
    return tuple(sorted(reasons))


def _register_pair_relation(
    relation: PairRelation,
    source_points_xyz: object,
    target_points_xyz: object,
    *,
    source_semantic_label: str | None,
    target_semantic_label: str | None,
    config: RegistrationConfig,
    require_semantic_eligibility: bool,
) -> RegistrationEvidence:
    if not isinstance(relation, PairRelation):
        raise TypeError("relation must be a PairRelation")
    if not isinstance(config, RegistrationConfig):
        raise TypeError("config must be a RegistrationConfig")
    source = _points(source_points_xyz, name="source_points_xyz")
    target = _points(target_points_xyz, name="target_points_xyz")
    source_label = (
        None if source_semantic_label is None else str(source_semantic_label).strip()
    )
    target_label = (
        None if target_semantic_label is None else str(target_semantic_label).strip()
    )
    early_reasons: set[str] = set()
    if relation.state not in _ELIGIBLE_STATES or (
        len(relation.t0_entity_ids) != 1 or len(relation.t1_entity_ids) != 1
    ):
        early_reasons.add("ineligible_relation_state")
    if require_semantic_eligibility:
        if not source_label or not target_label:
            early_reasons.add("missing_semantic_label")
        elif source_label.casefold() != target_label.casefold():
            early_reasons.add("semantic_label_mismatch")
        allowed = {label.casefold() for label in config.recoverable_semantic_labels}
        if source_label and source_label.casefold() not in allowed:
            early_reasons.add("ineligible_semantic_label")
    semantic_label = (
        target_label
        if source_label
        and target_label
        and source_label.casefold() == target_label.casefold()
        else None
    )
    source_sample = _voxel_sample(source, config) if len(source) else source
    target_sample = _voxel_sample(target, config) if len(target) else target
    if (
        len(source_sample) < config.minimum_sample_points
        or len(target_sample) < config.minimum_sample_points
    ):
        early_reasons.add("low_support")
    if early_reasons:
        return _evidence(
            relation=relation,
            config=config,
            source=source,
            target=target,
            source_sample_count=len(source_sample),
            target_sample_count=len(target_sample),
            semantic_label=(target_label or source_label)
            if require_semantic_eligibility
            else semantic_label,
            candidate=None,
            reasons=tuple(sorted(early_reasons)),
        )

    source_axes, source_scales, source_valid = _principal_axes(source_sample)
    target_axes, target_scales, target_valid = _principal_axes(target_sample)
    if (
        not source_valid
        or not target_valid
        or source_scales[1] / source_scales[0] < config.minimum_second_singular_ratio
        or target_scales[1] / target_scales[0] < config.minimum_second_singular_ratio
        or source_scales[1] <= np.finfo(np.float64).eps
        or target_scales[1] <= np.finfo(np.float64).eps
    ):
        return _evidence(
            relation=relation,
            config=config,
            source=source,
            target=target,
            source_sample_count=len(source_sample),
            target_sample_count=len(target_sample),
            semantic_label=semantic_label,
            candidate=None,
            reasons=("degenerate_geometry",),
        )

    source_centroid = source_sample.mean(axis=0)
    target_centroid = target_sample.mean(axis=0)
    initializations: list[tuple[str, np.ndarray]] = []
    if relation.state == "persistent_static":
        initializations.append(("static_identity", np.eye(4, dtype=np.float64)))
    else:
        initializations.append(
            (
                "centroid_identity",
                _initial_transform(np.eye(3), source_centroid, target_centroid),
            )
        )
        for identifier, axis_map in _proper_axis_permutations():
            rotation = target_axes @ axis_map @ source_axes.T
            if np.linalg.det(rotation) <= 0.0:
                continue
            initializations.append(
                (
                    identifier,
                    _initial_transform(rotation, source_centroid, target_centroid),
                )
            )

    candidates: list[_Candidate] = []
    for identifier, initial in initializations:
        if relation.state == "persistent_static":
            transform, iterations = validate_rigid_transform(initial), 0
        else:
            transform, iterations = _run_icp(
                source_sample, target_sample, initial, config
            )
        candidates.append(
            _measure_candidate(
                source_sample,
                target_sample,
                transform,
                identifier,
                iterations,
                source_scales,
                target_scales,
                config,
            )
        )
    best = min(candidates, key=_candidate_key)
    reasons = _quality_reasons(best, relation, config)
    return _evidence(
        relation=relation,
        config=config,
        source=source,
        target=target,
        source_sample_count=len(source_sample),
        target_sample_count=len(target_sample),
        semantic_label=semantic_label,
        candidate=best,
        reasons=reasons,
    )


def register_pair_relation(
    relation: PairRelation,
    source_points_xyz: object,
    target_points_xyz: object,
    *,
    source_semantic_label: str | None,
    target_semantic_label: str | None,
    config: RegistrationConfig,
) -> RegistrationEvidence:
    """Estimate one atomic relation with the legacy semantic eligibility gate."""

    return _register_pair_relation(
        relation,
        source_points_xyz,
        target_points_xyz,
        source_semantic_label=source_semantic_label,
        target_semantic_label=target_semantic_label,
        config=config,
        require_semantic_eligibility=True,
    )


def register_composite_relation(
    relation: PairRelation,
    source_points_xyz: object,
    target_points_xyz: object,
    *,
    source_semantic_label: str | None,
    target_semantic_label: str | None,
    config: RegistrationConfig,
) -> RegistrationEvidence:
    """Estimate one 1:1 composite transform with semantics recorded, not gated."""

    return _register_pair_relation(
        relation,
        source_points_xyz,
        target_points_xyz,
        source_semantic_label=source_semantic_label,
        target_semantic_label=target_semantic_label,
        config=config,
        require_semantic_eligibility=False,
    )


def register_oracle_pair_relation(
    relation: PairRelation,
    source_points_xyz: object,
    target_points_xyz: object,
    *,
    transform_source: OracleTransformSource,
    source_semantic_label: str | None,
    target_semantic_label: str | None,
    config: RegistrationConfig,
) -> RegistrationEvidence:
    """Build evaluator-only evidence from a bound official 3RScan transform."""

    if not isinstance(relation, PairRelation):
        raise TypeError("relation must be a PairRelation")
    if not isinstance(transform_source, OracleTransformSource):
        raise TypeError("transform_source must be OracleTransformSource")
    if not isinstance(config, RegistrationConfig):
        raise TypeError("config must be a RegistrationConfig")
    if relation.evidence.get("reference_instance_id") != (
        transform_source.reference_instance_id
    ):
        raise ValueError("oracle transform and relation identity differ")
    source = _points(source_points_xyz, name="source_points_xyz")
    target = _points(target_points_xyz, name="target_points_xyz")
    source_sample = _voxel_sample(source, config) if len(source) else source
    target_sample = _voxel_sample(target, config) if len(target) else target
    reasons: set[str] = set()
    if relation.state not in _ELIGIBLE_STATES or (
        len(relation.t0_entity_ids) != 1 or len(relation.t1_entity_ids) != 1
    ):
        reasons.add("ineligible_relation_state")
    if (
        len(source_sample) < config.minimum_sample_points
        or len(target_sample) < config.minimum_sample_points
    ):
        reasons.add("low_support")
    source_label = (
        None if source_semantic_label is None else str(source_semantic_label).strip()
    )
    target_label = (
        None if target_semantic_label is None else str(target_semantic_label).strip()
    )
    semantic_label = (
        target_label
        if source_label
        and target_label
        and source_label.casefold() == target_label.casefold()
        else None
    )
    if reasons:
        return _evidence(
            relation=relation,
            config=config,
            source=source,
            target=target,
            source_sample_count=len(source_sample),
            target_sample_count=len(target_sample),
            semantic_label=semantic_label,
            candidate=None,
            reasons=tuple(sorted(reasons)),
            method_id=ORACLE_REGISTRATION_METHOD_ID,
        )
    source_axes, source_scales, source_valid = _principal_axes(source_sample)
    target_axes, target_scales, target_valid = _principal_axes(target_sample)
    del source_axes, target_axes
    if (
        not source_valid
        or not target_valid
        or source_scales[1] / source_scales[0] < config.minimum_second_singular_ratio
        or target_scales[1] / target_scales[0] < config.minimum_second_singular_ratio
    ):
        reasons.add("degenerate_geometry")
    candidate = None
    if not reasons:
        candidate = _measure_candidate(
            source_sample,
            target_sample,
            transform_source.transform_world_from_t0,
            transform_source.source_type,
            0,
            source_scales,
            target_scales,
            config,
        )
    return _evidence(
        relation=relation,
        config=config,
        source=source,
        target=target,
        source_sample_count=len(source_sample),
        target_sample_count=len(target_sample),
        semantic_label=semantic_label,
        candidate=candidate,
        reasons=tuple(sorted(reasons)),
        method_id=ORACLE_REGISTRATION_METHOD_ID,
    )


__all__ = [
    "ORACLE_REGISTRATION_METHOD_ID",
    "REGISTRATION_METHOD_ID",
    "OracleTransformSource",
    "RegistrationConfig",
    "RegistrationEvidence",
    "apply_rigid_transform",
    "apply_rigid_transform_normals",
    "official_row_vector_to_internal_transform",
    "point_cloud_sha256",
    "register_composite_relation",
    "register_oracle_pair_relation",
    "register_pair_relation",
    "validate_rigid_transform",
]
