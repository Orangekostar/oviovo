"""Evaluator-only diagnostics for source-bound entity episode updates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.oviv2.current_surface import CurrentSurfaceView
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateResult,
    RelationInferenceState,
    RelationSupport,
    SurfaceActionReason,
)
from src.oviv2.fine_current_composer import EntityEpochComposition


def _boolean_mask(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean mask")
    result = np.array(raw, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MetricRate:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        for name in ("numerator", "denominator"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.numerator > self.denominator:
            raise ValueError("rate numerator cannot exceed its denominator")

    @property
    def value(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def to_json_record(self) -> dict[str, int | float | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RelationEvaluationLabel:
    t0_owner_entity_id: int
    t1_owner_entity_id: int
    is_same_identity: bool
    large_motion: bool

    def __post_init__(self) -> None:
        for name in ("t0_owner_entity_id", "t1_owner_entity_id"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.is_same_identity) is not bool
            or type(self.large_motion) is not bool
        ):
            raise TypeError("relation label flags must be boolean")
        if self.large_motion and not self.is_same_identity:
            raise ValueError("large motion requires a same-identity relation")

    @property
    def key(self) -> tuple[int, int]:
        return self.t0_owner_entity_id, self.t1_owner_entity_id


@dataclass(frozen=True, slots=True)
class EntityEpochEvaluationSupport:
    """Post-prediction support masks that are never exposed to the method."""

    current_gt_supported_mask: np.ndarray
    confirmed_free_mask: np.ndarray
    t1_surface_covered_mask: np.ndarray
    relation_labels: tuple[RelationEvaluationLabel, ...]
    voxel_size_m: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "current_gt_supported_mask",
            "confirmed_free_mask",
            "t1_surface_covered_mask",
        ):
            object.__setattr__(self, name, _boolean_mask(getattr(self, name), name))
        count = len(self.current_gt_supported_mask)
        if any(
            len(getattr(self, name)) != count
            for name in ("confirmed_free_mask", "t1_surface_covered_mask")
        ):
            raise ValueError("evaluation support masks must have equal lengths")
        labels = tuple(self.relation_labels)
        if any(not isinstance(label, RelationEvaluationLabel) for label in labels):
            raise TypeError(
                "relation_labels must contain RelationEvaluationLabel values"
            )
        labels = tuple(sorted(labels, key=lambda label: label.key))
        if len({label.key for label in labels}) != len(labels):
            raise ValueError("relation evaluation labels must be unique")
        object.__setattr__(self, "relation_labels", labels)
        if isinstance(self.voxel_size_m, bool) or not isinstance(
            self.voxel_size_m, (int, float, np.integer, np.floating)
        ):
            raise TypeError("voxel_size_m must be numeric")
        voxel_size = float(self.voxel_size_m)
        if not math.isfinite(voxel_size) or not math.isclose(
            voxel_size, 0.05, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("entity epoch diagnostics require a 0.05 m voxel size")
        object.__setattr__(self, "voxel_size_m", voxel_size)


@dataclass(frozen=True, slots=True)
class EntityEpochDiagnostics:
    deleted_supported_source_rows: int
    deleted_supported_unique_voxels: int
    bad_recovery_source_rows: int
    bad_recovery_unique_voxels: int
    correct_new_coverage_source_rows: int
    correct_new_coverage_unique_voxels: int
    free_conflict_source_rows: int
    free_conflict_unique_voxels: int
    deleted_supported_rate: MetricRate
    bad_recovery_rate: MetricRate
    correct_new_coverage_rate: MetricRate
    free_conflict_rate: MetricRate
    relation_precision: MetricRate
    relation_recall: MetricRate
    relation_rejection_rate: MetricRate
    large_motion_ambiguity_rate: MetricRate


@dataclass(frozen=True, slots=True)
class ActionAttributionRow:
    source_surface_id: str
    source_vertex_indices: tuple[int, ...]
    owner_entity_id: int
    action_reason: str
    relation_id: str | None
    used_new_measurement: bool
    source_row_count: int
    unique_voxel_count: int
    current_gt_supported_rows: int
    confirmed_free_rows: int
    t1_surface_covered_rows: int


@dataclass(frozen=True, slots=True)
class RelationDiagnosticRow:
    relation_id: str
    relation_source: str
    t0_owner_entity_id: int
    t1_owner_entity_id: int
    prediction_present: bool
    accepted: bool
    confidence: float | None
    inference_state: str | None
    rejection_reasons: tuple[str, ...]
    gt_same_identity: bool | None
    gt_large_motion: bool | None


@dataclass(frozen=True, slots=True)
class D4IdentityOnlyInvariance:
    headline_metrics_equal: bool
    current_mask_equal: bool
    point_semantics_equal: bool


def _unique_voxel_count(
    points_xyz: np.ndarray,
    mask: np.ndarray,
    voxel_size_m: float,
) -> int:
    selected = points_xyz[mask]
    if not len(selected):
        return 0
    voxels = np.floor(selected.astype(np.float64) / voxel_size_m).astype(np.int64)
    return len(np.unique(voxels, axis=0))


def _validate_action_inputs(
    composition: EntityEpochComposition,
    update: EntityEpochUpdateResult,
    evaluation_support: EntityEpochEvaluationSupport,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(composition, EntityEpochComposition):
        raise TypeError("composition must be EntityEpochComposition")
    if not isinstance(update, EntityEpochUpdateResult):
        raise TypeError("update must be EntityEpochUpdateResult")
    if not isinstance(evaluation_support, EntityEpochEvaluationSupport):
        raise TypeError("evaluation_support must be EntityEpochEvaluationSupport")
    surface = composition.surface
    count = len(surface.vertices_xyz)
    if len(evaluation_support.current_gt_supported_mask) != count:
        raise ValueError("evaluation support must align with the canonical surface")
    t0_positions = np.flatnonzero(surface.source_visit_ids == 0)
    if len(t0_positions) != len(update.current_valid) or not np.array_equal(
        surface.current_valid[t0_positions], update.current_valid
    ):
        raise ValueError("entity epoch update must align with canonical t0 rows")
    return t0_positions, surface.source_vertex_indices[t0_positions]


def action_attribution_rows(
    composition: EntityEpochComposition,
    update: EntityEpochUpdateResult,
    evaluation_support: EntityEpochEvaluationSupport,
) -> tuple[ActionAttributionRow, ...]:
    """Return one source-grounded evaluator row per state action."""

    t0_positions, t0_source_rows = _validate_action_inputs(
        composition, update, evaluation_support
    )
    surface = composition.surface
    output = []
    for delta in update.surface_deltas:
        local = np.searchsorted(t0_source_rows, delta.source_vertex_indices)
        if (
            np.any(local >= len(t0_source_rows))
            or not np.array_equal(t0_source_rows[local], delta.source_vertex_indices)
            or np.any(
                surface.owner_entity_ids[t0_positions[local]] != delta.owner_entity_id
            )
        ):
            raise ValueError("surface action rows do not match canonical source keys")
        rows = t0_positions[local]
        mask = np.zeros(len(surface.vertices_xyz), dtype=np.bool_)
        mask[rows] = True
        output.append(
            ActionAttributionRow(
                source_surface_id=delta.source_surface_id,
                source_vertex_indices=tuple(
                    int(value) for value in delta.source_vertex_indices
                ),
                owner_entity_id=delta.owner_entity_id,
                action_reason=delta.action_reason.value,
                relation_id=delta.relation_id,
                used_new_measurement=delta.used_new_measurement,
                source_row_count=len(rows),
                unique_voxel_count=_unique_voxel_count(
                    surface.vertices_xyz,
                    mask,
                    evaluation_support.voxel_size_m,
                ),
                current_gt_supported_rows=int(
                    np.count_nonzero(evaluation_support.current_gt_supported_mask[rows])
                ),
                confirmed_free_rows=int(
                    np.count_nonzero(evaluation_support.confirmed_free_mask[rows])
                ),
                t1_surface_covered_rows=int(
                    np.count_nonzero(evaluation_support.t1_surface_covered_mask[rows])
                ),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda row: (
                row.action_reason,
                row.owner_entity_id,
                row.source_vertex_indices,
            ),
        )
    )


def relation_diagnostic_rows(
    relations: tuple[RelationSupport, ...],
    labels: tuple[RelationEvaluationLabel, ...],
) -> tuple[RelationDiagnosticRow, ...]:
    """Return predictions and explicit missing-label rows for CSV export."""

    if not isinstance(relations, tuple) or any(
        not isinstance(relation, RelationSupport) for relation in relations
    ):
        raise TypeError("relations must contain RelationSupport values")
    if not isinstance(labels, tuple) or any(
        not isinstance(label, RelationEvaluationLabel) for label in labels
    ):
        raise TypeError("labels must contain RelationEvaluationLabel values")
    label_by_key = {label.key: label for label in labels}
    if len(label_by_key) != len(labels):
        raise ValueError("relation labels must be unique")
    output = []
    predicted_keys: set[tuple[int, int]] = set()
    for relation in relations:
        key = (relation.t0_owner_entity_id, relation.t1_owner_entity_id)
        predicted_keys.add(key)
        label = label_by_key.get(key)
        output.append(
            RelationDiagnosticRow(
                relation_id=relation.relation_id,
                relation_source=relation.relation_source,
                t0_owner_entity_id=key[0],
                t1_owner_entity_id=key[1],
                prediction_present=True,
                accepted=relation.accepted,
                confidence=relation.confidence,
                inference_state=relation.inference_state.value,
                rejection_reasons=relation.rejection_reasons,
                gt_same_identity=None if label is None else label.is_same_identity,
                gt_large_motion=None if label is None else label.large_motion,
            )
        )
    for label in labels:
        if label.key in predicted_keys:
            continue
        output.append(
            RelationDiagnosticRow(
                relation_id="",
                relation_source="missing-prediction",
                t0_owner_entity_id=label.t0_owner_entity_id,
                t1_owner_entity_id=label.t1_owner_entity_id,
                prediction_present=False,
                accepted=False,
                confidence=None,
                inference_state=None,
                rejection_reasons=("missing_prediction",),
                gt_same_identity=label.is_same_identity,
                gt_large_motion=label.large_motion,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda row: (
                row.t0_owner_entity_id,
                row.t1_owner_entity_id,
                row.relation_id,
            ),
        )
    )


def _relation_rates(
    relations: tuple[RelationSupport, ...],
    labels: tuple[RelationEvaluationLabel, ...],
) -> tuple[MetricRate, MetricRate, MetricRate, MetricRate]:
    if not labels:
        empty = MetricRate(0, 0)
        return empty, empty, empty, empty
    label_by_key = {label.key: label for label in labels}
    if any(
        (relation.t0_owner_entity_id, relation.t1_owner_entity_id) not in label_by_key
        for relation in relations
    ):
        raise ValueError("relation support lies outside the evaluator label universe")
    accepted = tuple(relation for relation in relations if relation.accepted)
    accepted_keys = {
        (relation.t0_owner_entity_id, relation.t1_owner_entity_id)
        for relation in accepted
    }
    true_keys = {label.key for label in labels if label.is_same_identity}
    true_accepted = len(accepted_keys & true_keys)
    rejected = len(set(label_by_key) - accepted_keys)
    large_motion_labels = tuple(
        label for label in labels if label.is_same_identity and label.large_motion
    )
    state_by_key = {
        (
            relation.t0_owner_entity_id,
            relation.t1_owner_entity_id,
        ): relation.inference_state
        for relation in accepted
    }
    large_motion_ambiguous = sum(
        state_by_key.get(label.key, RelationInferenceState.UNRESOLVED)
        is RelationInferenceState.UNRESOLVED
        for label in large_motion_labels
    )
    return (
        MetricRate(true_accepted, len(accepted_keys)),
        MetricRate(true_accepted, len(true_keys)),
        MetricRate(rejected, len(labels)),
        MetricRate(large_motion_ambiguous, len(large_motion_labels)),
    )


def evaluate_entity_epoch_actions(
    composition: EntityEpochComposition,
    update: EntityEpochUpdateResult,
    evaluation_support: EntityEpochEvaluationSupport,
    *,
    relation_support: tuple[RelationSupport, ...],
) -> EntityEpochDiagnostics:
    """Compute action attribution beside, not inside, legacy headline metrics."""

    if not isinstance(relation_support, tuple) or any(
        not isinstance(relation, RelationSupport) for relation in relation_support
    ):
        raise TypeError("relation_support must contain RelationSupport values")
    surface = composition.surface
    sidecar = composition.state_sidecar
    _validate_action_inputs(composition, update, evaluation_support)
    t0_mask = surface.source_visit_ids == 0
    recovered = (
        sidecar.action_reasons == SurfaceActionReason.LOCAL_POSITIVE_CORRECTION.value
    )
    deleted_supported = (
        t0_mask
        & sidecar.current_valid_before
        & ~surface.current_valid
        & evaluation_support.current_gt_supported_mask
    )
    bad_recovery = (
        recovered & surface.current_valid & evaluation_support.confirmed_free_mask
    )
    correct_new_coverage = (
        recovered
        & surface.current_valid
        & evaluation_support.current_gt_supported_mask
        & ~evaluation_support.t1_surface_covered_mask
    )
    free_conflict = surface.current_valid & evaluation_support.confirmed_free_mask
    supported_t0 = (
        t0_mask
        & sidecar.current_valid_before
        & evaluation_support.current_gt_supported_mask
    )
    new_coverage_opportunity = (
        t0_mask
        & evaluation_support.current_gt_supported_mask
        & ~evaluation_support.t1_surface_covered_mask
    )
    relation_precision, relation_recall, rejection_rate, large_motion_rate = (
        _relation_rates(relation_support, evaluation_support.relation_labels)
    )

    def rows(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    return EntityEpochDiagnostics(
        deleted_supported_source_rows=rows(deleted_supported),
        deleted_supported_unique_voxels=_unique_voxel_count(
            surface.vertices_xyz, deleted_supported, evaluation_support.voxel_size_m
        ),
        bad_recovery_source_rows=rows(bad_recovery),
        bad_recovery_unique_voxels=_unique_voxel_count(
            surface.vertices_xyz, bad_recovery, evaluation_support.voxel_size_m
        ),
        correct_new_coverage_source_rows=rows(correct_new_coverage),
        correct_new_coverage_unique_voxels=_unique_voxel_count(
            surface.vertices_xyz,
            correct_new_coverage,
            evaluation_support.voxel_size_m,
        ),
        free_conflict_source_rows=rows(free_conflict),
        free_conflict_unique_voxels=_unique_voxel_count(
            surface.vertices_xyz, free_conflict, evaluation_support.voxel_size_m
        ),
        deleted_supported_rate=MetricRate(rows(deleted_supported), rows(supported_t0)),
        bad_recovery_rate=MetricRate(rows(bad_recovery), rows(recovered)),
        correct_new_coverage_rate=MetricRate(
            rows(correct_new_coverage), rows(new_coverage_opportunity)
        ),
        free_conflict_rate=MetricRate(rows(free_conflict), rows(surface.current_valid)),
        relation_precision=relation_precision,
        relation_recall=relation_recall,
        relation_rejection_rate=rejection_rate,
        large_motion_ambiguity_rate=large_motion_rate,
    )


_D4_HEADLINE_KEYS = (
    "current_miou",
    "ghost",
    "background_f1_at_5cm",
    "surface_f1_at_5cm",
)


def verify_d4_identity_only_invariance(
    *,
    baseline_surface: CurrentSurfaceView,
    identity_only_surface: CurrentSurfaceView,
    baseline_headline: Mapping[str, float],
    identity_only_headline: Mapping[str, float],
) -> D4IdentityOnlyInvariance:
    """Require D4 to change identity aliases and nothing in current-map scoring."""

    if not isinstance(baseline_surface, CurrentSurfaceView) or not isinstance(
        identity_only_surface, CurrentSurfaceView
    ):
        raise TypeError("D4 invariance requires CurrentSurfaceView values")
    if not isinstance(baseline_headline, Mapping) or not isinstance(
        identity_only_headline, Mapping
    ):
        raise TypeError("D4 headline metrics must be mappings")
    for key in _D4_HEADLINE_KEYS:
        if key not in baseline_headline or key not in identity_only_headline:
            raise ValueError(f"D4 headline metrics are missing {key}")
        first = float(baseline_headline[key])
        second = float(identity_only_headline[key])
        if not math.isfinite(first) or not math.isfinite(second) or first != second:
            raise ValueError(f"D4 headline metric differs from D1: {key}")
    for name in (
        "vertices_xyz",
        "normals_xyz",
        "triangles",
        "source_surface_indices",
        "source_vertex_indices",
        "source_visit_ids",
        "geometry_epochs",
        "observed_rgb_uint8",
        "rgb_valid",
        "evidence_state_codes",
        "last_supported_frames",
        "owner_entity_ids",
        "owner_confidences",
    ):
        if not np.array_equal(
            getattr(baseline_surface, name), getattr(identity_only_surface, name)
        ):
            raise ValueError(f"D4 geometry differs from D1: {name}")
    if not np.array_equal(
        baseline_surface.current_valid, identity_only_surface.current_valid
    ):
        raise ValueError("D4 current mask differs from D1")
    for name in (
        "semantic_ids",
        "semantic_confidences",
        "semantic_support_reliabilities",
        "semantic_source_codes",
    ):
        if not np.array_equal(
            getattr(baseline_surface, name), getattr(identity_only_surface, name)
        ):
            raise ValueError(f"D4 point semantics differ from D1: {name}")
    return D4IdentityOnlyInvariance(
        headline_metrics_equal=True,
        current_mask_equal=True,
        point_semantics_equal=True,
    )


__all__ = [
    "ActionAttributionRow",
    "D4IdentityOnlyInvariance",
    "EntityEpochDiagnostics",
    "EntityEpochEvaluationSupport",
    "MetricRate",
    "RelationDiagnosticRow",
    "RelationEvaluationLabel",
    "action_attribution_rows",
    "evaluate_entity_epoch_actions",
    "relation_diagnostic_rows",
    "verify_d4_identity_only_invariance",
]
