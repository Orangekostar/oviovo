from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.current_surface import CurrentEvidenceState, CurrentSurfaceView
from src.oviv2.entity_epoch_io import (
    load_entity_epoch_composition,
    write_entity_epoch_composition,
)
from src.oviv2.entity_epoch_update import (
    EntityEpisodeRecord,
    EntityEpochUpdateResult,
    EpisodeLifecycle,
    IdentityAliasRecord,
    SurfaceActionReason,
    SurfaceStateDelta,
)
from src.oviv2.fine_current_composer import (
    FineVisitSurface,
    compose_entity_epoch_fine_surface,
)
from src.oviv2.fine_dynamic_policy import SurfaceRetirementReason


def _surface(visit_id: int, source_index: int) -> FineVisitSurface:
    offset = 0.2 * visit_id
    return FineVisitSurface(
        visit_id=visit_id,
        source_surface_index=source_index,
        geometry_epoch=visit_id,
        vertices_xyz=np.asarray(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=np.float32
        ),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0]] * 2, dtype=np.float32),
        triangles=np.empty((0, 3), dtype=np.int64),
        source_vertex_indices=np.asarray([7, 8], dtype=np.int64),
        observed_rgb_uint8=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        rgb_valid=np.asarray([True, True], dtype=np.bool_),
        last_supported_frames=np.asarray(
            [10 + visit_id, 11 + visit_id], dtype=np.int32
        ),
        owner_entity_ids=np.asarray([11, 12], dtype=np.int64),
        owner_confidences=np.asarray([0.9, 0.8], dtype=np.float32),
        semantic_ids=np.asarray([2, 3], dtype=np.int32),
        semantic_confidences=np.asarray([0.8, 0.7], dtype=np.float32),
        semantic_support_reliabilities=np.asarray([0.7, 0.6], dtype=np.float32),
        semantic_source_codes=np.asarray([1, 2], dtype=np.uint8),
    )


def _composition():
    t0 = _surface(0, 4)
    t1 = _surface(1, 9)
    delta = SurfaceStateDelta(
        source_surface_id="ovi-t0",
        source_vertex_indices=np.asarray([8], dtype=np.int64),
        owner_entity_id=12,
        stable_entity_id_before="ovi-t0:12",
        stable_entity_id_after="stable:chair-12",
        episode_id_before="ovi-t0:12/location:t0:12",
        episode_id_after="stable:chair-12/location:t0:12",
        current_valid_before=True,
        current_valid_after=False,
        retirement_reason_before=SurfaceRetirementReason.NONE,
        retirement_reason_after=SurfaceRetirementReason.DIRECT_FREE,
        action_reason=SurfaceActionReason.RELIABLE_VISIBLE_FREE,
        evidence_frame_ids=(20, 21),
        used_new_measurement=True,
        relation_id="relation:12:12",
    )
    episode = EntityEpisodeRecord(
        stable_entity_id="stable:chair-12",
        episode_id="stable:chair-12/location:t0:12",
        source_visit=0,
        source_surface_id="ovi-t0",
        owner_entity_id=12,
        source_vertex_indices=np.asarray([8], dtype=np.int64),
        first_observation_frame=20,
        last_observation_frame=21,
        lifecycle_state=EpisodeLifecycle.LOCATION_RETIRED,
        relation_id="relation:12:12",
        relation_confidence=0.9,
        transform_world_from_t0=None,
        retirement_reasons=(SurfaceRetirementReason.DIRECT_FREE,),
        revival_reasons=(),
    )
    update = EntityEpochUpdateResult(
        current_valid=np.asarray([True, False], dtype=np.bool_),
        retirement_reason_codes=np.asarray(
            [SurfaceRetirementReason.NONE, SurfaceRetirementReason.DIRECT_FREE],
            dtype=np.uint8,
        ),
        evidence_state_codes=np.asarray(
            [
                CurrentEvidenceState.HISTORICAL_UNOBSERVED,
                CurrentEvidenceState.REVOKED_VISIBLE_FREE,
            ],
            dtype=np.uint8,
        ),
        surface_deltas=(delta,),
        identity_aliases=(
            IdentityAliasRecord(0, 12, "stable:chair-12", "relation:12:12"),
            IdentityAliasRecord(1, 12, "stable:chair-12", "relation:12:12"),
        ),
        episode_records=(episode,),
    )
    return compose_entity_epoch_fine_surface(
        t0=t0,
        t1=t1,
        update=update,
        surface_id="entity-epoch-io-test",
    )


def test_entity_epoch_sidecar_preserves_action_and_provenance_rows() -> None:
    composition = _composition()
    state = composition.state_sidecar

    assert state.current_valid_before.tolist() == [True, True, False, False]
    assert state.current_valid_after.tolist() == [True, False, True, True]
    assert state.retirement_reason_before_codes.tolist() == [0, 0, 0, 0]
    assert state.retirement_reason_after_codes.tolist() == [0, 1, 0, 0]
    assert state.action_reasons.tolist() == [
        "",
        "reliable_visible_free",
        "t1_current_observed",
        "t1_current_observed",
    ]
    assert state.used_new_measurement.tolist() == [False, True, True, True]
    assert state.stable_entity_ids.tolist() == [
        "ovi-t0:11",
        "stable:chair-12",
        "ovi-t1:11",
        "stable:chair-12",
    ]
    assert state.episode_ids[1] == "stable:chair-12/location:t0:12"
    assert state.relation_ids.tolist() == ["", "relation:12:12", "", ""]
    assert state.evidence_frames_for_row(1) == (20, 21)
    assert state.evidence_frames_for_row(2) == (11,)


def test_entity_epoch_sidecar_rejects_existing_row_validity_reason_mismatch() -> None:
    state = _composition().state_sidecar
    invalid_before = np.array(state.current_valid_before, copy=True)
    invalid_before[0] = False

    with pytest.raises(ValueError, match="before-state"):
        replace(state, current_valid_before=invalid_before)


def test_entity_epoch_state_round_trip_is_hash_bound_to_canonical_surface(
    tmp_path,
) -> None:
    composition = _composition()

    artifact = write_entity_epoch_composition(composition, tmp_path / "state")
    restored = load_entity_epoch_composition(
        artifact.output_dir,
        surface=composition.surface,
    )

    assert restored.surface is composition.surface
    np.testing.assert_array_equal(
        restored.state_sidecar.current_valid_after,
        composition.state_sidecar.current_valid_after,
    )
    assert restored.state_sidecar.stable_entity_ids.tolist() == (
        composition.state_sidecar.stable_entity_ids.tolist()
    )
    manifest = json.loads(artifact.manifest.read_text(encoding="utf-8"))
    state_npz = artifact.output_dir / manifest["state_arrays"]["path"]
    assert (
        hashlib.sha256(state_npz.read_bytes()).hexdigest()
        == (manifest["state_arrays"]["sha256"])
    )
    assert manifest["canonical_vertex_count"] == len(composition.surface.vertices_xyz)


def test_entity_epoch_loader_rejects_tampered_npz(tmp_path) -> None:
    composition = _composition()
    artifact = write_entity_epoch_composition(composition, tmp_path / "state")
    arrays_path = artifact.output_dir / "entity_epoch_state.npz"
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="binding"):
        load_entity_epoch_composition(
            artifact.output_dir,
            surface=composition.surface,
        )


def test_entity_epoch_loader_rejects_another_same_length_surface(tmp_path) -> None:
    composition = _composition()
    artifact = write_entity_epoch_composition(composition, tmp_path / "state")
    surface = composition.surface
    vertices = np.array(surface.vertices_xyz, copy=True)
    vertices[0, 0] += 0.5
    changed_surface = CurrentSurfaceView(
        **{
            field: (vertices if field == "vertices_xyz" else getattr(surface, field))
            for field in surface.__dataclass_fields__
        }
    )

    with pytest.raises(ValueError, match="canonical surface"):
        load_entity_epoch_composition(
            artifact.output_dir,
            surface=changed_surface,
        )


def test_entity_epoch_loader_rejects_resigned_internal_surface_id(tmp_path) -> None:
    composition = _composition()
    artifact = write_entity_epoch_composition(composition, tmp_path / "state")
    arrays_path = artifact.output_dir / "entity_epoch_state.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["surface_id"] = np.asarray("another-surface")
    np.savez_compressed(arrays_path, **arrays)
    manifest = json.loads(artifact.manifest.read_text(encoding="utf-8"))
    content = arrays_path.read_bytes()
    manifest["state_arrays"]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest["state_arrays"]["byte_count"] = len(content)
    artifact.manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical surface"):
        load_entity_epoch_composition(
            artifact.output_dir,
            surface=composition.surface,
        )
