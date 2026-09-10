from __future__ import annotations

import numpy as np
import pytest

from src.oviv2.current_surface import (
    CurrentEvidenceState,
    SemanticSource,
    select_current_surface,
)
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateResult,
    IdentityAliasRecord,
)
from src.oviv2.fine_current_composer import (
    FineVisitSurface,
    compose_entity_epoch_fine_surface,
    compose_two_visit_fine_surface,
    current_surface_from_visit,
)
from src.oviv2.fine_dynamic_policy import SurfaceRetirementReason
from src.oviv2.fine_surface_validity import FineSurfaceEvidence, FineValidityConfig


def _visit_surface(visit_id: int, source_surface_index: int) -> FineVisitSurface:
    offset = float(visit_id) * 0.1
    return FineVisitSurface(
        visit_id=visit_id,
        source_surface_index=source_surface_index,
        geometry_epoch=visit_id,
        vertices_xyz=np.asarray(
            [[offset, 0.0, 1.0], [offset + 1.0, 0.0, 1.0], [offset, 1.0, 1.0]],
            dtype=np.float32,
        ),
        normals_xyz=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        source_vertex_indices=np.asarray([7, 8, 9], dtype=np.int64),
        observed_rgb_uint8=np.asarray(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8
        ),
        rgb_valid=np.asarray([True, False, True]),
        last_supported_frames=np.asarray(
            [10 + visit_id, -1, 12 + visit_id], dtype=np.int32
        ),
        owner_entity_ids=np.asarray(
            [11 + visit_id, 12 + visit_id, 13 + visit_id], dtype=np.int64
        ),
        owner_confidences=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        semantic_ids=np.asarray([2, 3, 4], dtype=np.int32),
        semantic_confidences=np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
        semantic_support_reliabilities=np.asarray(
            [0.75, 0.5, 0.25], dtype=np.float32
        ),
        semantic_source_codes=np.asarray(
            [
                SemanticSource.LOCAL_SURFACE,
                SemanticSource.OWNER_ENTITY,
                SemanticSource.UNKNOWN,
            ],
            dtype=np.uint8,
        ),
    )


def test_two_visit_composer_preserves_sources_and_applies_only_fine_decisions() -> None:
    t0 = _visit_surface(0, 4)
    t1 = _visit_surface(1, 9)
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([1, 0, 0], dtype=np.uint16),
        visible_absent_observations=np.asarray([0, 2, 0], dtype=np.uint16),
        occluded_observations=np.asarray([0, 0, 3], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([0, 2, 0], dtype=np.uint8),
        last_supported_frames=np.asarray([30, -1, -1], dtype=np.int32),
    )

    current = compose_two_visit_fine_surface(
        t0=t0,
        t1=t1,
        t0_evidence=evidence,
        coarse_visible_free_candidates=np.asarray([True, True, True]),
        validity_config=FineValidityConfig(
            minimum_absent_observations=2,
            minimum_distinct_viewpoints=2,
        ),
        surface_id="apartment-fine",
    )

    assert current.current_valid.tolist() == [False, False, True, True, True, True]
    assert current.evidence_state_codes.tolist() == [
        CurrentEvidenceState.REPLACED_BY_CURRENT,
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.HISTORICAL_OCCLUDED,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
    ]
    assert current.source_surface_indices.tolist() == [4, 4, 4, 9, 9, 9]
    assert current.source_vertex_indices.tolist() == [7, 8, 9, 7, 8, 9]
    assert current.source_visit_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert current.last_supported_frames.tolist() == [30, -1, -1, 11, -1, 13]
    assert current.owner_entity_ids.tolist() == [11, 12, 13, 12, 13, 14]
    selection = select_current_surface(current)
    assert selection.source_row_indices.tolist() == [2, 3, 4, 5]
    assert selection.triangles.tolist() == [[1, 2, 3]]


def test_visit_surface_rejects_duplicate_source_vertex_identity() -> None:
    payload = _visit_surface(0, 4)
    with pytest.raises(ValueError, match="source vertex"):
        FineVisitSurface(
            **{
                name: getattr(payload, name)
                for name in payload.__dataclass_fields__
                if name != "source_vertex_indices"
            },
            source_vertex_indices=np.asarray([7, 7, 9], dtype=np.int64),
        )


def test_single_visit_surface_is_current_without_geometry_reordering() -> None:
    source = _visit_surface(0, 4)

    current = current_surface_from_visit(source, surface_id="room0-fine")

    np.testing.assert_array_equal(current.vertices_xyz, source.vertices_xyz)
    np.testing.assert_array_equal(current.triangles, source.triangles)
    assert current.source_surface_indices.tolist() == [4, 4, 4]
    assert current.source_vertex_indices.tolist() == [7, 8, 9]
    assert current.current_valid.tolist() == [True, True, True]
    assert current.evidence_state_codes.tolist() == [
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
    ]


def test_entity_epoch_composer_preserves_source_rows_and_triangle_locality() -> None:
    t0 = _visit_surface(0, 4)
    t1 = _visit_surface(1, 9)
    update = EntityEpochUpdateResult(
        current_valid=np.asarray([True, False, False], dtype=np.bool_),
        retirement_reason_codes=np.asarray(
            [
                SurfaceRetirementReason.NONE,
                SurfaceRetirementReason.DIRECT_FREE,
                SurfaceRetirementReason.REPLACED,
            ],
            dtype=np.uint8,
        ),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                CurrentEvidenceState.HISTORICAL_OCCLUDED,
                CurrentEvidenceState.HISTORICAL_UNCERTAIN,
            ],
            dtype=np.uint8,
        ),
    )

    composition = compose_entity_epoch_fine_surface(
        t0=t0,
        t1=t1,
        update=update,
        surface_id="crove-entity-epoch-pair",
    )

    surface = composition.surface
    np.testing.assert_array_equal(surface.vertices_xyz[:3], t0.vertices_xyz)
    np.testing.assert_array_equal(surface.vertices_xyz[3:], t1.vertices_xyz)
    np.testing.assert_array_equal(surface.current_valid[:3], update.current_valid)
    assert surface.current_valid.tolist() == [True, False, False, True, True, True]
    assert surface.evidence_state_codes.tolist() == [
        CurrentEvidenceState.HISTORICAL_UNOBSERVED,
        CurrentEvidenceState.REVOKED_VISIBLE_FREE,
        CurrentEvidenceState.REPLACED_BY_CURRENT,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
        CurrentEvidenceState.CURRENT_OBSERVED,
    ]
    assert surface.triangles.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert np.all(
        surface.source_surface_indices[surface.triangles]
        == surface.source_surface_indices[surface.triangles][:, :1]
    )
    sidecar = composition.state_sidecar
    assert sidecar.source_surface_indices.tolist() == [4, 4, 4, 9, 9, 9]
    assert sidecar.source_vertex_indices.tolist() == [7, 8, 9, 7, 8, 9]
    assert sidecar.source_existed_before.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert sidecar.current_valid_after.tolist() == surface.current_valid.tolist()


def test_identity_only_composition_keeps_baseline_mask_and_semantics() -> None:
    t0 = _visit_surface(0, 4)
    t1 = _visit_surface(1, 9)
    evidence = FineSurfaceEvidence(
        present_observations=np.asarray([1, 0, 0], dtype=np.uint16),
        visible_absent_observations=np.asarray([0, 2, 0], dtype=np.uint16),
        occluded_observations=np.asarray([0, 0, 3], dtype=np.uint16),
        distinct_absent_viewpoints=np.asarray([0, 2, 0], dtype=np.uint8),
        last_supported_frames=np.asarray([30, -1, -1], dtype=np.int32),
    )
    baseline = compose_two_visit_fine_surface(
        t0=t0,
        t1=t1,
        t0_evidence=evidence,
        coarse_visible_free_candidates=np.asarray([True, True, True]),
        validity_config=FineValidityConfig(
            minimum_absent_observations=2,
            minimum_distinct_viewpoints=2,
        ),
        surface_id="d1-b3",
    )
    update = EntityEpochUpdateResult(
        current_valid=baseline.current_valid[:3],
        retirement_reason_codes=np.asarray(
            [
                SurfaceRetirementReason.REPLACED,
                SurfaceRetirementReason.DIRECT_FREE,
                SurfaceRetirementReason.NONE,
            ],
            dtype=np.uint8,
        ),
        evidence_state_codes=baseline.evidence_state_codes[:3],
        identity_aliases=(
            IdentityAliasRecord(0, 11, "stable:rescene:11", "relation:11:12"),
            IdentityAliasRecord(1, 12, "stable:rescene:11", "relation:11:12"),
        ),
    )

    identity_only = compose_entity_epoch_fine_surface(
        t0=t0,
        t1=t1,
        update=update,
        surface_id="d4-rescene-id-only",
    )

    np.testing.assert_array_equal(identity_only.surface.current_valid, baseline.current_valid)
    np.testing.assert_array_equal(identity_only.surface.semantic_ids, baseline.semantic_ids)
    np.testing.assert_array_equal(
        identity_only.surface.semantic_confidences,
        baseline.semantic_confidences,
    )
    np.testing.assert_array_equal(
        identity_only.surface.semantic_source_codes,
        baseline.semantic_source_codes,
    )
    assert identity_only.state_sidecar.stable_entity_ids[0] == "stable:rescene:11"
    assert identity_only.surface.owner_entity_ids[0] == 11
