from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.dense_moved_readout import (
    DenseMovedGeometryAgreement,
    DenseMovedReadoutGateConfig,
    decide_dense_geometry_agreement,
    measure_dense_geometry_agreement,
)


def _boundary_measurement() -> DenseMovedGeometryAgreement:
    return DenseMovedGeometryAgreement(
        compact_to_template_coverage=0.90,
        template_to_compact_coverage=0.90,
        compact_to_template_median_m=0.05,
        template_to_compact_median_m=0.05,
        compact_to_template_p90_m=0.10,
        template_to_compact_p90_m=0.10,
        centroid_residual_m=0.05,
        compact_extent_xyz=(0.2, 0.1, 0.0),
        template_extent_xyz=(0.1, 0.2, 0.1),
        extent_residual_xyz=(0.1, 0.1, 0.1),
        extent_ratio_xyz=(0.5, 2.0, None),
        compact_point_count=2,
        template_point_count=2,
    )


def test_geometry_agreement_measures_both_directions_and_accepts_match() -> None:
    compact = np.asarray(
        ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.10, 0.0, 0.0)),
        dtype=np.float32,
    )
    template = np.array(compact, copy=True)

    measurement = measure_dense_geometry_agreement(compact, template)
    decision = decide_dense_geometry_agreement(measurement)

    assert measurement.compact_to_template_coverage == 1.0
    assert measurement.template_to_compact_coverage == 1.0
    assert measurement.compact_to_template_median_m == 0.0
    assert measurement.template_to_compact_p90_m == 0.0
    assert measurement.centroid_residual_m == 0.0
    assert measurement.extent_ratio_xyz == (1.0, None, None)
    assert decision.accepted is True
    assert decision.rejection_reasons == ()


def test_geometry_agreement_exposes_directional_coverage_failure() -> None:
    compact = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64)
    template = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
    )

    measurement = measure_dense_geometry_agreement(compact, template)
    decision = decide_dense_geometry_agreement(measurement)

    assert measurement.compact_to_template_coverage == 1.0
    assert measurement.template_to_compact_coverage == 0.5
    assert decision.accepted is False
    assert "template_to_compact_coverage" in decision.rejection_reasons


def test_geometry_gate_accepts_every_exact_preregistered_boundary() -> None:
    decision = decide_dense_geometry_agreement(_boundary_measurement())

    assert decision.accepted is True
    assert decision.rejection_reasons == ()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("compact_to_template_coverage", 0.899, "compact_to_template_coverage"),
        ("template_to_compact_coverage", 0.899, "template_to_compact_coverage"),
        ("compact_to_template_median_m", 0.051, "compact_to_template_median"),
        ("template_to_compact_median_m", 0.051, "template_to_compact_median"),
        ("compact_to_template_p90_m", 0.101, "compact_to_template_p90"),
        ("template_to_compact_p90_m", 0.101, "template_to_compact_p90"),
        ("centroid_residual_m", 0.051, "centroid_residual"),
        ("extent_residual_xyz", (0.1, 0.101, 0.0), "extent_residual_y"),
        ("extent_ratio_xyz", (0.499, 1.0, None), "extent_ratio_x"),
        ("extent_ratio_xyz", (2.001, 1.0, None), "extent_ratio_x"),
    ),
)
def test_geometry_gate_rejects_values_outside_frozen_boundaries(
    field: str, value: object, reason: str
) -> None:
    measurement = replace(_boundary_measurement(), **{field: value})

    decision = decide_dense_geometry_agreement(measurement)

    assert decision.accepted is False
    assert reason in decision.rejection_reasons


def test_geometry_gate_ignores_ratio_only_for_degenerate_axis() -> None:
    measurement = replace(
        _boundary_measurement(),
        compact_extent_xyz=(0.2, 0.0, 0.0),
        template_extent_xyz=(0.1, 0.09, 0.0),
        extent_residual_xyz=(0.1, 0.09, 0.0),
        extent_ratio_xyz=(0.5, None, None),
    )

    assert decide_dense_geometry_agreement(measurement).accepted is True


@pytest.mark.parametrize(
    "points",
    (
        np.empty((0, 3), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.asarray(((0.0, float("nan"), 0.0),), dtype=np.float64),
    ),
)
def test_geometry_agreement_rejects_invalid_point_cloud(points: np.ndarray) -> None:
    valid = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="point cloud"):
        measure_dense_geometry_agreement(points, valid)


def test_gate_config_rejects_non_preregistered_relations() -> None:
    with pytest.raises(ValueError, match="coverage"):
        DenseMovedReadoutGateConfig(minimum_directional_coverage=1.1)
    with pytest.raises(ValueError, match="extent ratio"):
        DenseMovedReadoutGateConfig(
            minimum_extent_ratio=2.0,
            maximum_extent_ratio=1.0,
        )
