"""Causal geometry agreement for dense moved-anchor readout."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

DENSE_GEOMETRY_GATE_ID = "crove_dense_moved_geometry_gate_v1"


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _finite_tuple3(value: object, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{label} must be a length-three tuple")
    result = tuple(_finite_number(item, label=label) for item in value)
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DenseMovedReadoutGateConfig:
    distance_threshold_m: float = 0.05
    minimum_directional_coverage: float = 0.90
    maximum_directional_median_m: float = 0.05
    maximum_directional_p90_m: float = 0.10
    maximum_centroid_residual_m: float = 0.05
    maximum_extent_residual_m: float = 0.10
    minimum_extent_ratio: float = 0.5
    maximum_extent_ratio: float = 2.0
    minimum_extent_for_ratio_m: float = 0.05

    def __post_init__(self) -> None:
        values = {
            name: _finite_number(getattr(self, name), label=name)
            for name in self.__dataclass_fields__
        }
        if not 0.0 <= values["minimum_directional_coverage"] <= 1.0:
            raise ValueError("minimum directional coverage must be in [0, 1]")
        if any(
            values[name] <= 0.0
            for name in (
                "distance_threshold_m",
                "maximum_directional_median_m",
                "maximum_directional_p90_m",
                "maximum_centroid_residual_m",
                "maximum_extent_residual_m",
                "minimum_extent_ratio",
                "maximum_extent_ratio",
                "minimum_extent_for_ratio_m",
            )
        ):
            raise ValueError("geometry gate distances and ratios must be positive")
        if (
            values["maximum_directional_median_m"]
            > values["maximum_directional_p90_m"]
        ):
            raise ValueError("median threshold must not exceed p90 threshold")
        if values["minimum_extent_ratio"] > values["maximum_extent_ratio"]:
            raise ValueError("minimum extent ratio must not exceed maximum")

    def to_json_record(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_json_record(
        cls, payload: Mapping[str, object]
    ) -> DenseMovedReadoutGateConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("geometry gate config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("geometry gate config fields are invalid")
        return cls(**{name: payload[name] for name in expected})  # type: ignore[arg-type]


def dense_moved_readout_gate_config_from_json(
    payload: Mapping[str, object],
) -> DenseMovedReadoutGateConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("geometry gate policy must be a mapping")
    config_fields = set(DenseMovedReadoutGateConfig.__dataclass_fields__)
    if set(payload) != config_fields | {"schema_version", "gate_id"}:
        raise ValueError("geometry gate policy fields are invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("gate_id") != DENSE_GEOMETRY_GATE_ID
    ):
        raise ValueError("geometry gate policy identity is invalid")
    return DenseMovedReadoutGateConfig.from_json_record(
        {name: payload[name] for name in config_fields}
    )


@dataclass(frozen=True, slots=True)
class DenseMovedGeometryAgreement:
    compact_to_template_coverage: float
    template_to_compact_coverage: float
    compact_to_template_median_m: float
    template_to_compact_median_m: float
    compact_to_template_p90_m: float
    template_to_compact_p90_m: float
    centroid_residual_m: float
    compact_extent_xyz: tuple[float, float, float]
    template_extent_xyz: tuple[float, float, float]
    extent_residual_xyz: tuple[float, float, float]
    extent_ratio_xyz: tuple[float | None, float | None, float | None]
    compact_point_count: int
    template_point_count: int

    def __post_init__(self) -> None:
        for name in (
            "compact_to_template_coverage",
            "template_to_compact_coverage",
        ):
            value = _finite_number(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "compact_to_template_median_m",
            "template_to_compact_median_m",
            "compact_to_template_p90_m",
            "template_to_compact_p90_m",
            "centroid_residual_m",
        ):
            if _finite_number(getattr(self, name), label=name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "compact_extent_xyz",
            "template_extent_xyz",
            "extent_residual_xyz",
        ):
            values = _finite_tuple3(getattr(self, name), label=name)
            if any(item < 0.0 for item in values):
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.extent_ratio_xyz, tuple) or len(
            self.extent_ratio_xyz
        ) != 3:
            raise TypeError("extent_ratio_xyz must be a length-three tuple")
        for ratio in self.extent_ratio_xyz:
            if ratio is not None and _finite_number(
                ratio, label="extent_ratio_xyz"
            ) <= 0.0:
                raise ValueError("extent ratios must be positive")
        for name in ("compact_point_count", "template_point_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_json_record(self) -> dict[str, object]:
        return {
            "compact_to_template_coverage": self.compact_to_template_coverage,
            "template_to_compact_coverage": self.template_to_compact_coverage,
            "compact_to_template_median_m": self.compact_to_template_median_m,
            "template_to_compact_median_m": self.template_to_compact_median_m,
            "compact_to_template_p90_m": self.compact_to_template_p90_m,
            "template_to_compact_p90_m": self.template_to_compact_p90_m,
            "centroid_residual_m": self.centroid_residual_m,
            "compact_extent_xyz": list(self.compact_extent_xyz),
            "template_extent_xyz": list(self.template_extent_xyz),
            "extent_residual_xyz": list(self.extent_residual_xyz),
            "extent_ratio_xyz": list(self.extent_ratio_xyz),
            "compact_point_count": self.compact_point_count,
            "template_point_count": self.template_point_count,
        }


@dataclass(frozen=True, slots=True)
class DenseMovedReadoutDecision:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    measurement: DenseMovedGeometryAgreement
    config: DenseMovedReadoutGateConfig

    def __post_init__(self) -> None:
        if self.accepted is bool(self.rejection_reasons):
            raise ValueError("accepted decision and rejection reasons disagree")

    def to_json_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "measurement": self.measurement.to_json_record(),
            "config": self.config.to_json_record(),
        }


def _points(value: object, *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError(f"{label} point cloud must be nonempty Nx3")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} point cloud must be finite")
    return points


def _directional_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    values, _ = cKDTree(target).query(source, k=1, workers=1)
    return np.asarray(values, dtype=np.float64)


def measure_dense_geometry_agreement(
    compact_points_xyz: object,
    translated_template_points_xyz: object,
    config: DenseMovedReadoutGateConfig | None = None,
) -> DenseMovedGeometryAgreement:
    gate = DenseMovedReadoutGateConfig() if config is None else config
    if not isinstance(gate, DenseMovedReadoutGateConfig):
        raise TypeError("config must be DenseMovedReadoutGateConfig")
    compact = _points(compact_points_xyz, label="compact")
    template = _points(translated_template_points_xyz, label="template")
    compact_to_template = _directional_distances(compact, template)
    template_to_compact = _directional_distances(template, compact)
    tolerance = max(np.finfo(np.float32).eps, gate.distance_threshold_m * 1e-7)
    compact_extent = np.ptp(compact, axis=0)
    template_extent = np.ptp(template, axis=0)
    extent_residual = np.abs(template_extent - compact_extent)
    extent_ratio: list[float | None] = []
    for compact_value, template_value in zip(
        compact_extent, template_extent, strict=True
    ):
        if (
            compact_value >= gate.minimum_extent_for_ratio_m
            and template_value >= gate.minimum_extent_for_ratio_m
        ):
            extent_ratio.append(float(template_value / compact_value))
        else:
            extent_ratio.append(None)
    return DenseMovedGeometryAgreement(
        compact_to_template_coverage=float(
            np.mean(compact_to_template <= gate.distance_threshold_m + tolerance)
        ),
        template_to_compact_coverage=float(
            np.mean(template_to_compact <= gate.distance_threshold_m + tolerance)
        ),
        compact_to_template_median_m=float(np.median(compact_to_template)),
        template_to_compact_median_m=float(np.median(template_to_compact)),
        compact_to_template_p90_m=float(np.quantile(compact_to_template, 0.9)),
        template_to_compact_p90_m=float(np.quantile(template_to_compact, 0.9)),
        centroid_residual_m=float(
            np.linalg.norm(compact.mean(axis=0) - template.mean(axis=0))
        ),
        compact_extent_xyz=tuple(float(value) for value in compact_extent),
        template_extent_xyz=tuple(float(value) for value in template_extent),
        extent_residual_xyz=tuple(float(value) for value in extent_residual),
        extent_ratio_xyz=tuple(extent_ratio),  # type: ignore[arg-type]
        compact_point_count=len(compact),
        template_point_count=len(template),
    )


def decide_dense_geometry_agreement(
    measurement: DenseMovedGeometryAgreement,
    config: DenseMovedReadoutGateConfig | None = None,
) -> DenseMovedReadoutDecision:
    if not isinstance(measurement, DenseMovedGeometryAgreement):
        raise TypeError("measurement must be DenseMovedGeometryAgreement")
    gate = DenseMovedReadoutGateConfig() if config is None else config
    if not isinstance(gate, DenseMovedReadoutGateConfig):
        raise TypeError("config must be DenseMovedReadoutGateConfig")
    reasons: list[str] = []
    minimums = (
        ("compact_to_template_coverage", measurement.compact_to_template_coverage),
        ("template_to_compact_coverage", measurement.template_to_compact_coverage),
    )
    reasons.extend(
        name
        for name, value in minimums
        if value < gate.minimum_directional_coverage
    )
    maximums = (
        ("compact_to_template_median", measurement.compact_to_template_median_m, gate.maximum_directional_median_m),
        ("template_to_compact_median", measurement.template_to_compact_median_m, gate.maximum_directional_median_m),
        ("compact_to_template_p90", measurement.compact_to_template_p90_m, gate.maximum_directional_p90_m),
        ("template_to_compact_p90", measurement.template_to_compact_p90_m, gate.maximum_directional_p90_m),
        ("centroid_residual", measurement.centroid_residual_m, gate.maximum_centroid_residual_m),
    )
    reasons.extend(name for name, value, limit in maximums if value > limit)
    for axis, value in zip("xyz", measurement.extent_residual_xyz, strict=True):
        if value > gate.maximum_extent_residual_m:
            reasons.append(f"extent_residual_{axis}")
    for axis, value in zip("xyz", measurement.extent_ratio_xyz, strict=True):
        if value is not None and not (
            gate.minimum_extent_ratio <= value <= gate.maximum_extent_ratio
        ):
            reasons.append(f"extent_ratio_{axis}")
    return DenseMovedReadoutDecision(
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        measurement=measurement,
        config=gate,
    )


__all__ = [
    "DENSE_GEOMETRY_GATE_ID",
    "DenseMovedGeometryAgreement",
    "DenseMovedReadoutDecision",
    "DenseMovedReadoutGateConfig",
    "decide_dense_geometry_agreement",
    "dense_moved_readout_gate_config_from_json",
    "measure_dense_geometry_agreement",
]
