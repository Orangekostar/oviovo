"""Deterministic diagnostic contracts for CROVE anchor counterfactuals."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.evaluation.contracts import MapSnapshot

COUNTERFACTUAL_METRICS = (
    "object_f1",
    "dynamic_f1",
    "change_f1",
    "current_miou",
    "ghost_rate",
)
_TRANSITION_FIELDS = {
    "frame_index",
    "timestamp_ns",
    "entity_id",
    "before",
    "after",
    "evidence",
    "readout_valid",
}
_EVIDENCE_KINDS = {
    "present",
    "visible_absent",
    "occluded",
    "depth_unknown",
}


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{label} must be nonnegative")
    return normalized


def _metric(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric or unavailable")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return normalized


def _anchor_id(value: object, *, prefixed: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("transition anchor entity ID is invalid")
    prefix = "anchor:"
    if prefixed:
        if not value.startswith(prefix) or len(value) == len(prefix):
            raise ValueError("transition must name an anchor entity")
        value = value[len(prefix) :]
    if not value.startswith("ovimap:") or len(value) == len("ovimap:"):
        raise ValueError("transition must name an anchor entity")
    return value


@dataclass(frozen=True)
class CounterfactualVariant:
    """One immutable diagnostic suppression selection."""

    variant_id: str
    family: str
    selected_anchor_ids: tuple[str, ...]
    focal_anchor_id: str | None = None
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if not self.variant_id or not self.family:
            raise ValueError("counterfactual variant identity must be nonempty")
        if self.selected_anchor_ids != tuple(sorted(set(self.selected_anchor_ids))):
            raise ValueError("selected anchor IDs must be sorted and unique")
        for value in self.selected_anchor_ids:
            _anchor_id(value, prefixed=False)
        if self.focal_anchor_id is not None:
            _anchor_id(self.focal_anchor_id, prefixed=False)
        if self.diagnostic_only is not True:
            raise ValueError("counterfactual variants must remain diagnostic-only")

    def to_json_record(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "family": self.family,
            "selected_anchor_ids": list(self.selected_anchor_ids),
            "focal_anchor_id": self.focal_anchor_id,
            "diagnostic_only": True,
        }


def _transitioned_anchor_ids(
    transitions: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise TypeError("transitions must be a sequence")
    ids: list[str] = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise TypeError(f"transition {index} must be an object")
        if set(transition) != _TRANSITION_FIELDS:
            raise ValueError("transition fields are invalid")
        if (
            transition.get("before") != "active"
            or transition.get("after") != "dormant"
            or transition.get("evidence") != "visible_absent"
            or transition.get("readout_valid") is not False
        ):
            raise ValueError("counterfactual inputs must be active-to-dormant transitions")
        _nonnegative_integer(transition.get("frame_index"), label="transition frame")
        _nonnegative_integer(
            transition.get("timestamp_ns"), label="transition timestamp"
        )
        ids.append(_anchor_id(transition.get("entity_id"), prefixed=True))
    if len(ids) != 10 or len(set(ids)) != len(ids):
        raise ValueError("P6-A requires ten unique active-to-dormant transitions")
    return tuple(sorted(ids))


def _ghost_mass_by_anchor(attribution: Mapping[str, object]) -> dict[str, int]:
    if (
        not isinstance(attribution, Mapping)
        or attribution.get("status") != "PASS"
        or attribution.get("scene") != "apartment"
    ):
        raise ValueError("P2 attribution identity is invalid")
    frames = attribution.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("P2 attribution frames are invalid")
    totals: defaultdict[str, int] = defaultdict(int)
    for frame in frames:
        if (
            not isinstance(frame, Mapping)
            or frame.get("official_cross_check") != "PASS"
        ):
            raise ValueError("P2 attribution frame is not trusted")
        rows = frame.get("entity_attribution")
        if not isinstance(rows, list):
            raise ValueError("P2 entity attribution is invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("P2 entity attribution row must be an object")
            if row.get("authority") != "ovimap_anchor_unbound":
                continue
            entity_id = _anchor_id(row.get("entity_id"), prefixed=False)
            ghost_matches = _nonnegative_integer(
                row.get("ghost_matches"), label="P2 ghost matches"
            )
            totals[entity_id] += ghost_matches
    return dict(totals)


def derive_counterfactual_variants(
    transitions: Sequence[Mapping[str, object]],
    attribution: Mapping[str, object],
) -> tuple[CounterfactualVariant, ...]:
    """Build the complete fixed P6-A matrix without producing a method rule."""

    anchor_ids = _transitioned_anchor_ids(transitions)
    ghost_mass = _ghost_mass_by_anchor(attribution)
    ranked = tuple(sorted(anchor_ids, key=lambda item: (-ghost_mass.get(item, 0), item)))
    dominant = tuple(sorted(ranked[:5]))
    additional = tuple(sorted(set(anchor_ids) - set(dominant)))
    variants = [
        CounterfactualVariant("CF0", "no_suppression", ()),
        CounterfactualVariant("CF1", "full_p5", anchor_ids),
    ]
    variants.extend(
        CounterfactualVariant(
            f"CF2_{index:02d}",
            "full_except_one",
            tuple(item for item in anchor_ids if item != focal),
            focal,
        )
        for index, focal in enumerate(anchor_ids)
    )
    variants.extend(
        CounterfactualVariant(
            f"CF3_{index:02d}", "only_one", (focal,), focal
        )
        for index, focal in enumerate(anchor_ids)
    )
    variants.extend(
        (
            CounterfactualVariant("G1", "p2_dominant_five", dominant),
            CounterfactualVariant("G2", "additional_five", additional),
        )
    )
    return tuple(variants)


@dataclass(frozen=True)
class AnchorCounterfactualFeature:
    anchor_entity_id: str
    transition_frame: int
    semantic_label: str | None
    anchor_point_count: int
    anchor_voxel_count: int
    sampled_voxel_count: int
    present_evidence_count: int
    absent_evidence_count: int
    occluded_evidence_count: int
    unknown_evidence_count: int
    absence_support_at_transition: int
    distinct_viewpoint_count_at_transition: int
    first_absence_frame: int
    suppression_latency_frames: int
    bbox_extent_x_m: float
    bbox_extent_y_m: float
    bbox_extent_z_m: float


def _diagnostic_entities(
    diagnostics: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("manifest_id")
        != "crove_ovimap_unbound_visibility_diagnostics_v1"
    ):
        raise ValueError("visibility diagnostics identity is invalid")
    rows = diagnostics.get("entities")
    if not isinstance(rows, list):
        raise ValueError("visibility diagnostic entities are invalid")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("visibility diagnostic entity must be an object")
        entity_id = _anchor_id(row.get("entity_id"), prefixed=False)
        if entity_id in result:
            raise ValueError("visibility diagnostic entity IDs must be unique")
        result[entity_id] = row
    return result


def build_anchor_counterfactual_features(
    anchor: MapSnapshot,
    diagnostics: Mapping[str, object],
    transitioned_anchor_ids: Sequence[str],
    *,
    voxel_size_m: float,
) -> tuple[AnchorCounterfactualFeature, ...]:
    """Bind P5 transition diagnostics to immutable anchor geometry."""

    if not isinstance(anchor, MapSnapshot) or anchor.scene_id != "apartment":
        raise ValueError("counterfactual anchor must be an Apartment MapSnapshot")
    voxel_size = float(voxel_size_m)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    if isinstance(transitioned_anchor_ids, (str, bytes)) or not isinstance(
        transitioned_anchor_ids, Sequence
    ):
        raise TypeError("transitioned_anchor_ids must be a sequence")
    normalized_ids = tuple(
        _anchor_id(value, prefixed=False) for value in transitioned_anchor_ids
    )
    if normalized_ids != tuple(sorted(set(normalized_ids))):
        raise ValueError("transitioned anchor IDs must be sorted and unique")
    anchor_by_id = {entity.entity_id: entity for entity in anchor.entities}
    if len(anchor_by_id) != len(anchor.entities):
        raise ValueError("anchor entity IDs must be unique")
    diagnostic_by_id = _diagnostic_entities(diagnostics)
    features = []
    for entity_id in normalized_ids:
        entity = anchor_by_id.get(entity_id)
        row = diagnostic_by_id.get(entity_id)
        if entity is None or row is None:
            raise ValueError(f"missing counterfactual anchor evidence: {entity_id}")
        transition = row.get("transition")
        if not isinstance(transition, Mapping):
            raise ValueError(f"transition support is missing for {entity_id}")
        transition_frame = _nonnegative_integer(
            transition.get("frame_index"), label="transition frame"
        )
        absence_support = _nonnegative_integer(
            transition.get("absence_observation_count"),
            label="absence support at transition",
        )
        distinct_views = _nonnegative_integer(
            transition.get("distinct_absence_viewpoint_count"),
            label="distinct viewpoints at transition",
        )
        first_absence = _nonnegative_integer(
            row.get("first_absence_frame"), label="first absence frame"
        )
        if first_absence > transition_frame:
            raise ValueError("first absence frame cannot follow transition")
        evidence = row.get("evidence_counts")
        if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_KINDS:
            raise ValueError("visibility evidence counts are invalid")
        evidence_counts = {
            name: _nonnegative_integer(
                evidence[name], label=f"{name} evidence count"
            )
            for name in _EVIDENCE_KINDS
        }
        points = np.asarray(entity.points_xyz, dtype=np.float64)
        voxels = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)
        extent = np.ptp(points, axis=0)
        features.append(
            AnchorCounterfactualFeature(
                anchor_entity_id=entity_id,
                transition_frame=transition_frame,
                semantic_label=entity.semantic_label,
                anchor_point_count=len(points),
                anchor_voxel_count=len(voxels),
                sampled_voxel_count=_nonnegative_integer(
                    row.get("sampled_voxel_count"), label="sampled voxel count"
                ),
                present_evidence_count=evidence_counts["present"],
                absent_evidence_count=evidence_counts["visible_absent"],
                occluded_evidence_count=evidence_counts["occluded"],
                unknown_evidence_count=evidence_counts["depth_unknown"],
                absence_support_at_transition=absence_support,
                distinct_viewpoint_count_at_transition=distinct_views,
                first_absence_frame=first_absence,
                suppression_latency_frames=transition_frame - first_absence,
                bbox_extent_x_m=float(extent[0]),
                bbox_extent_y_m=float(extent[1]),
                bbox_extent_z_m=float(extent[2]),
            )
        )
    return tuple(features)


def build_counterfactual_metric_rows(
    variants: Sequence[CounterfactualVariant],
    metrics_by_variant: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Any], ...]:
    """Normalize measured metrics and compute deltas without estimation."""

    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise TypeError("variants must be a sequence")
    if not isinstance(metrics_by_variant, Mapping):
        raise TypeError("metrics_by_variant must be a mapping")
    variant_ids = tuple(item.variant_id for item in variants)
    if len(set(variant_ids)) != len(variant_ids) or not {"CF0", "CF1"}.issubset(
        variant_ids
    ):
        raise ValueError("metric rows require unique CF0 and CF1 variants")
    if set(metrics_by_variant) != set(variant_ids):
        raise ValueError("metric variants do not match the counterfactual matrix")
    normalized: dict[str, dict[str, float | None]] = {}
    for variant_id in variant_ids:
        values = metrics_by_variant[variant_id]
        if not isinstance(values, Mapping) or set(values) != set(
            COUNTERFACTUAL_METRICS
        ):
            raise ValueError(f"{variant_id} metric fields are invalid")
        normalized[variant_id] = {
            name: _metric(values[name], label=name)
            for name in COUNTERFACTUAL_METRICS
        }
    rows = []
    for variant in variants:
        values = normalized[variant.variant_id]
        row: dict[str, Any] = {
            "variant_id": variant.variant_id,
            "family": variant.family,
            "selected_anchor_ids": ";".join(variant.selected_anchor_ids),
            "focal_anchor_id": variant.focal_anchor_id,
            "diagnostic_only": True,
            "metric_status": (
                "COMPLETE"
                if all(values[name] is not None for name in COUNTERFACTUAL_METRICS)
                else "PARTIAL"
            ),
            **values,
        }
        for name in COUNTERFACTUAL_METRICS:
            value = values[name]
            for baseline_id, suffix in (("CF0", "cf0"), ("CF1", "cf1")):
                baseline = normalized[baseline_id][name]
                row[f"delta_{name}_vs_{suffix}"] = (
                    None if value is None or baseline is None else value - baseline
                )
        rows.append(row)
    return tuple(rows)


__all__ = [
    "COUNTERFACTUAL_METRICS",
    "AnchorCounterfactualFeature",
    "CounterfactualVariant",
    "build_anchor_counterfactual_features",
    "build_counterfactual_metric_rows",
    "derive_counterfactual_variants",
]
