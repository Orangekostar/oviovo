from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.dense_projection import DenseSemanticConfig, DenseSemanticIntegrator
from src.oviv2.dense_semantics import DenseSemanticFrame, DenseSemanticProvenance
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.meshing import derive_labeled_mesh
from src.oviv2.observations import FrameObservation, ObservationKind
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.runtime import (
    Oviv2Runtime,
    Oviv2RuntimeConfig,
    RuntimeFrameResult,
    _clone_geometry_for_frame,
)
from src.oviv2.snapshot import VoxelMapSnapshot
from src.oviv2.tracking import LocalTrackerConfig

from tests.oviv2.test_dense_projection import make_dense_frame


def frame(frame_id: int = 0) -> Frame:
    depth = np.ones((32, 32), dtype=np.float32)
    return Frame(
        frame_id=frame_id,
        rgb=np.full((32, 32, 3), 128, dtype=np.uint8),
        depth=depth,
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(30.0, 30.0, 15.5, 15.5, 32, 32),
        timestamp=float(frame_id),
    )


def observation(
    frame_id: int,
    kind: ObservationKind,
    semantic_id: int,
    keys: set[tuple[int, int, int]],
    *,
    image_feature: np.ndarray | None = None,
    text_feature: np.ndarray | None = None,
    feature_model_id: str | None = None,
    view_direction_xyz: tuple[float, float, float] | None = None,
    visible_pixel_count: int = 4096,
    border_contact_fraction: float = 0.0,
) -> FrameObservation:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    return FrameObservation(
        observation_id=frame_id * 100,
        frame_id=frame_id,
        timestamp=float(frame_id),
        kind=kind,
        label="wall" if kind is ObservationKind.STRUCTURE else "chair",
        semantic_id=semantic_id,
        confidence=0.9,
        mask=np.ones((2, 2), dtype=bool),
        bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)),
        bounds_min_xyz=tuple(points.min(axis=0)),
        bounds_max_xyz=tuple(points.max(axis=0)),
        image_feature=image_feature,
        text_feature=text_feature,
        feature_model_id=feature_model_id,
        view_direction_xyz=view_direction_xyz,
        visible_pixel_count=visible_pixel_count,
        border_contact_fraction=border_contact_fraction,
    )


def precision_runtime(*, confirm_hits: int) -> Oviv2Runtime:
    return Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=confirm_hits)),
    )


def object_observation(
    frame_id: int,
    key: tuple[int, int, int],
    semantic_id: int,
    label: str,
) -> FrameObservation:
    value = observation(
        frame_id,
        ObservationKind.OBJECT,
        semantic_id,
        {key},
        image_feature=np.asarray((1.0, 0.0)),
        feature_model_id="clip",
        visible_pixel_count=4096,
    )
    return replace(value, label=label)


def dense_provenance() -> DenseSemanticProvenance:
    return DenseSemanticProvenance(
        backend="radseg",
        source_commit="a" * 40,
        radio_commit="b" * 40,
        model_id="nvidia/C-RADIOv3-H",
        model_sha256="c" * 64,
        auxiliary_model_sha256="d" * 64,
        vocabulary_sha256="e" * 64,
        prompt_sha256="f" * 64,
        inference_config_sha256="1" * 64,
        cache_prefix_sha256="2" * 64,
        language_model_id="google/siglip2-giant-opt-patch16-384",
        language_model_revision="3" * 40,
        language_model_sha256="4" * 64,
    )


def dense_frame(frame_id: int) -> DenseSemanticFrame:
    return make_dense_frame(
        image_shape=(32, 32),
        stride=32,
        source_frame_id=frame_id,
        class_count=3,
        class_ids=[[[2]]],
        probabilities=[[[1.0]]],
    )


def dense_runtime() -> Oviv2Runtime:
    return Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(
            tracker=LocalTrackerConfig(confirm_hits=1),
            dense_semantics=DenseSemanticConfig(voxel_size_m=0.05),
        ),
        dense_semantic_provenance=dense_provenance(),
    )


def _array_block_fingerprint(blocks: dict[object, object]) -> tuple[object, ...]:
    return tuple(
        (
            key,
            tuple(
                (name, value.dtype.str, value.shape, value.tobytes())
                for name, value in sorted(vars(block).items())
            ),
        )
        for key, block in sorted(blocks.items())
    )


def _geometry_fingerprint(runtime: Oviv2Runtime) -> tuple[object, ...]:
    grid = runtime.geometry._grid
    active = grid.hashmap().active_buf_indices()
    return (
        grid.hashmap().key_tensor()[active].numpy().tobytes(),
        *(
            grid.attribute(name)[active].numpy().tobytes()
            for name in runtime.geometry._ATTRIBUTE_NAMES
        ),
    )


def _canonical_geometry_fingerprint(
    geometry: SparseTsdfVolume,
) -> tuple[bytes, ...]:
    grid = geometry._grid
    active = grid.hashmap().active_buf_indices()
    keys = grid.hashmap().key_tensor()[active].numpy()
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return (
        keys[order].tobytes(),
        *(
            grid.attribute(name)[active].numpy()[order].tobytes()
            for name in geometry._ATTRIBUTE_NAMES
        ),
    )


def _runtime_state(runtime: Oviv2Runtime, registry_path: Path) -> dict[str, object]:
    runtime.registry.save(registry_path)
    return {
        "revision": (runtime.revision, runtime.last_frame_id, runtime.last_timestamp),
        "object_ids": tuple(
            id(value)
            for value in (
                runtime.geometry,
                runtime.evidence,
                runtime.ownership,
                runtime.tracker,
                runtime.registry,
            )
        ),
        "geometry": _geometry_fingerprint(runtime),
        "evidence_blocks": _array_block_fingerprint(runtime.evidence._blocks),
        "evidence_candidates": tuple(
            (
                key,
                runtime.evidence.semantic_candidates(key),
                runtime.evidence.entity_candidates(key),
            )
            for key in ((-11, -11, 20), (0, 0, 20))
        ),
        "tracker_snapshot": pickle.dumps(runtime.tracker, protocol=5),
        "registry_json": registry_path.read_bytes(),
        "ownership_blocks": _array_block_fingerprint(runtime.ownership._blocks),
        "ownership_records": runtime.ownership.records(),
        "ownership_entities": tuple(
            (entity_id, tuple(sorted(voxel_keys)))
            for entity_id, voxel_keys in sorted(
                runtime.ownership._entity_voxels.items()
            )
        ),
    }


def test_geometry_integrates_even_when_observation_batch_is_empty() -> None:
    runtime = Oviv2Runtime("room0")

    result = runtime.process_frame(frame(), ())

    assert result.geometry_blocks_touched > 0
    assert (
        result.dense_sampled_pixel_count,
        result.dense_valid_pixel_count,
        result.dense_updated_voxel_count,
    ) == (0, 0, 0)
    assert runtime.geometry.active_block_count > 0
    assert runtime.registry.entities == {}


def test_runtime_config_and_constructor_require_paired_dense_contracts() -> None:
    with pytest.raises(TypeError, match="dense_semantics"):
        Oviv2RuntimeConfig(dense_semantics=object())  # type: ignore[arg-type]

    dense_config = Oviv2RuntimeConfig(
        dense_semantics=DenseSemanticConfig(voxel_size_m=0.05)
    )
    with pytest.raises(ValueError, match="dense.*provenance|provenance.*dense"):
        Oviv2Runtime("room0", dense_config)
    with pytest.raises(ValueError, match="dense.*provenance|provenance.*dense"):
        Oviv2Runtime(
            "room0",
            dense_semantic_provenance=dense_provenance(),
        )
    with pytest.raises(TypeError, match="dense_semantic_provenance"):
        Oviv2Runtime(
            "room0",
            dense_config,
            dense_semantic_provenance=object(),  # type: ignore[arg-type]
        )


def test_runtime_config_requires_dense_and_tsdf_voxel_sizes_to_match() -> None:
    with pytest.raises(ValueError, match="dense.*voxel_size_m|voxel_size_m.*TSDF"):
        Oviv2RuntimeConfig(
            tsdf=TsdfConfig(voxel_size_m=0.05),
            dense_semantics=DenseSemanticConfig(voxel_size_m=0.1),
        )

    config = Oviv2RuntimeConfig(
        tsdf=TsdfConfig(voxel_size_m=0.05),
        dense_semantics=DenseSemanticConfig(
            voxel_size_m=float(np.float32(0.05)),
        ),
    )
    assert config.dense_semantics is not None


def test_tsdf_trial_clone_matches_three_frame_direct_integration_with_active_capacity() -> None:
    config = TsdfConfig(block_count=100_000)
    direct = SparseTsdfVolume(config)
    transactional = SparseTsdfVolume(config)

    for frame_id, translation_x in enumerate((0.0, 0.35, 0.7)):
        current = frame(frame_id)
        current.pose[0, 3] = translation_x
        direct.integrate(
            current.depth,
            current.rgb,
            current.intrinsics.to_matrix(),
            current.pose,
        )

        trial = _clone_geometry_for_frame(transactional, current)
        trial_capacity = int(trial._grid.hashmap().capacity())
        assert trial_capacity >= max(1, transactional.active_block_count)
        assert trial_capacity < config.block_count
        trial.integrate(
            current.depth,
            current.rgb,
            current.intrinsics.to_matrix(),
            current.pose,
        )
        transactional = trial

        assert _canonical_geometry_fingerprint(transactional) == (
            _canonical_geometry_fingerprint(direct)
        )


def test_runtime_requires_exactly_one_dense_frame_when_enabled() -> None:
    disabled = Oviv2Runtime("room0")
    with pytest.raises(ValueError, match="dense.*disabled|disabled.*dense"):
        disabled.process_frame(frame(0), (), dense_semantics=dense_frame(0))
    assert disabled.revision == 0

    enabled = dense_runtime()
    with pytest.raises(ValueError, match="dense.*required|required.*dense"):
        enabled.process_frame(frame(0), ())
    assert enabled.revision == 0


def test_runtime_integrates_dense_frame_at_current_revision() -> None:
    runtime = dense_runtime()

    result = runtime.process_frame(
        frame(0),
        (),
        dense_semantics=dense_frame(0),
    )

    assert (
        result.dense_sampled_pixel_count,
        result.dense_valid_pixel_count,
        result.dense_updated_voxel_count,
    ) == (1, 1, 1)
    candidates = runtime.evidence.semantic_candidates((-11, -11, 20))
    assert [candidate.label_id for candidate in candidates] == [2]
    assert candidates[0].revision == result.revision == runtime.revision == 1


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("dense_sampled_pixel_count", -1),
        ("dense_valid_pixel_count", True),
        ("dense_updated_voxel_count", 1.5),
    ],
)
def test_runtime_frame_result_rejects_invalid_dense_counts(
    field_name: str,
    invalid: object,
) -> None:
    result = RuntimeFrameResult(0, 0, 0, 0, 0, ())
    assert (
        result.dense_sampled_pixel_count,
        result.dense_valid_pixel_count,
        result.dense_updated_voxel_count,
    ) == (0, 0, 0)

    with pytest.raises(ValueError, match=field_name):
        replace(result, **{field_name: invalid})


def test_dense_failure_rolls_back_all_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = dense_runtime()
    key = (0, 0, 20)
    runtime.process_frame(
        frame(0),
        (object_observation(0, key, 2, "chair"),),
        dense_semantics=dense_frame(0),
    )
    before = _runtime_state(runtime, tmp_path / "registry-before.jsonl")
    original = DenseSemanticIntegrator.integrate

    def fail_after_dense_update(integrator, current, dense, store, revision):
        original(integrator, current, dense, store, revision)
        assert store is not runtime.evidence
        raise RuntimeError("injected dense failure")

    monkeypatch.setattr(DenseSemanticIntegrator, "integrate", fail_after_dense_update)

    with pytest.raises(RuntimeError, match="dense failure"):
        runtime.process_frame(
            frame(1),
            (object_observation(1, key, 2, "chair"),),
            dense_semantics=dense_frame(1),
        )

    after = _runtime_state(runtime, tmp_path / "registry-after.jsonl")
    assert after == before


def test_failure_after_dense_update_rolls_back_all_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = dense_runtime()
    key = (0, 0, 20)
    runtime.process_frame(
        frame(0),
        (object_observation(0, key, 2, "chair"),),
        dense_semantics=dense_frame(0),
    )
    before = _runtime_state(runtime, tmp_path / "registry-before.jsonl")
    original = ReversibleOwnershipStore.assign

    def fail_after_ownership_update(store, *args, **kwargs):
        original(store, *args, **kwargs)
        assert store is not runtime.ownership
        raise RuntimeError("injected downstream failure")

    monkeypatch.setattr(
        ReversibleOwnershipStore,
        "assign",
        fail_after_ownership_update,
    )

    with pytest.raises(RuntimeError, match="downstream failure"):
        runtime.process_frame(
            frame(1),
            (object_observation(1, key, 2, "chair"),),
            dense_semantics=dense_frame(1),
        )

    after = _runtime_state(runtime, tmp_path / "registry-after.jsonl")
    assert after == before


@pytest.mark.parametrize("invalid_timestamp", [object(), np.nan])
def test_invalid_frame_timestamp_is_rejected_before_trial_publication(
    tmp_path: Path,
    invalid_timestamp: object,
) -> None:
    runtime = dense_runtime()
    key = (0, 0, 20)
    runtime.process_frame(
        frame(0),
        (object_observation(0, key, 2, "chair"),),
        dense_semantics=dense_frame(0),
    )
    before = _runtime_state(runtime, tmp_path / "registry-before.jsonl")
    invalid = frame(1)
    invalid.timestamp = invalid_timestamp  # type: ignore[assignment]

    with pytest.raises(ValueError, match="timestamp"):
        runtime.process_frame(
            invalid,
            (),
            dense_semantics=dense_frame(1),
        )

    after = _runtime_state(runtime, tmp_path / "registry-after.jsonl")
    assert after == before


def test_structure_updates_semantics_but_never_creates_entity() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)

    runtime.process_frame(frame(), (observation(0, ObservationKind.STRUCTURE, 1, {key}),))

    assert runtime.evidence.semantic_candidates(key)[0].label_id == 1
    assert runtime.registry.entities == {}
    assert runtime.ownership.owner_of(key) is None


def test_tentative_object_cannot_write_entity_evidence() -> None:
    runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=2)),
    )
    key = (0, 0, 20)

    runtime.process_frame(frame(), (observation(0, ObservationKind.OBJECT, 2, {key}),))

    assert runtime.evidence.entity_candidates(key) == ()
    assert runtime.registry.entities == {}


def test_confirmed_object_writes_entity_evidence_and_reversible_owner() -> None:
    runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=2)),
    )
    key = (0, 0, 20)
    runtime.process_frame(frame(0), (observation(0, ObservationKind.OBJECT, 2, {key}),))

    result = runtime.process_frame(frame(1), (observation(1, ObservationKind.OBJECT, 2, {key}),))

    assert result.accepted_entity_ids == (1,)
    assert runtime.evidence.semantic_candidates(key) == ()
    assert runtime.evidence.entity_candidates(key)[0].entity_id == 1
    assert runtime.ownership.owner_of(key).entity_id == 1
    assert (result.matched_entity_count, result.new_entity_count) == (0, 1)
    mesh = derive_labeled_mesh(
        runtime.geometry,
        runtime.evidence,
        runtime.ownership,
        entity_semantics=runtime.registry.semantic_labels(),
    )
    owned = mesh.entity_ids == 1
    assert owned.any()
    assert np.all(mesh.semantic_ids[owned] == 2)


def test_owned_voxels_follow_current_entity_label_without_stale_votes() -> None:
    runtime = precision_runtime(confirm_hits=1)
    key = (0, 0, 20)
    first = runtime.process_frame(
        frame(0),
        (object_observation(0, key, 2, "chair"),),
    )
    second = runtime.process_frame(
        frame(1),
        (object_observation(1, key, 3, "stool"),),
    )
    runtime.process_frame(frame(2), (object_observation(2, key, 3, "stool"),))
    runtime.process_frame(frame(3), (object_observation(3, key, 3, "stool"),))

    mesh = derive_labeled_mesh(
        runtime.geometry,
        runtime.evidence,
        runtime.ownership,
        entity_semantics=runtime.registry.semantic_labels(),
    )
    owned = mesh.entity_ids == 1

    assert owned.any()
    assert np.all(mesh.semantic_ids[owned] == 3)
    assert runtime.evidence.semantic_candidates(key) == ()
    assert (first.matched_entity_count, first.new_entity_count) == (0, 1)
    assert (second.matched_entity_count, second.new_entity_count) == (1, 0)
    assert (second.association_conflict_count, second.revoked_edge_count) == (0, 0)


def test_structure_semantics_remain_voxel_evidence_without_owner() -> None:
    runtime = precision_runtime(confirm_hits=1)
    key = (0, 0, 20)
    runtime.process_frame(
        frame(0),
        (observation(0, ObservationKind.STRUCTURE, 1, {key}),),
    )
    runtime.process_frame(frame(1), ())

    mesh = derive_labeled_mesh(
        runtime.geometry,
        runtime.evidence,
        runtime.ownership,
        entity_semantics=runtime.registry.semantic_labels(),
    )

    assert np.any(mesh.semantic_ids == 1)
    assert np.all(mesh.entity_ids[mesh.semantic_ids == 1] == 0)


def test_runtime_resolves_entities_once_per_frame_even_for_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = precision_runtime(confirm_hits=1)
    calls: list[tuple[tuple[int, ...], int]] = []
    original = EntityRegistry.resolve_batch

    def record_batch(registry, tracks, revision):
        calls.append((tuple(track.track_id for track in tracks), revision))
        return original(registry, tracks, revision)

    monkeypatch.setattr(EntityRegistry, "resolve_batch", record_batch)
    runtime.process_frame(frame(0), ())
    left = replace(object_observation(1, (0, 0, 20), 2, "chair"), observation_id=101)
    right = replace(object_observation(1, (20, 0, 20), 2, "chair"), observation_id=102)
    runtime.process_frame(frame(1), (right, left))

    assert calls == [((), 1), ((1, 2), 2)]


def test_runtime_reports_tracker_conflict_and_revocation_counters() -> None:
    key = (0, 0, 20)
    conflict_runtime = precision_runtime(confirm_hits=1)
    chair = replace(
        observation(0, ObservationKind.OBJECT, 2, {key}),
        label="chair",
    )
    stool = replace(
        observation(1, ObservationKind.OBJECT, 3, {key}),
        label="stool",
    )
    conflict_runtime.process_frame(frame(0), (chair,))
    conflict = conflict_runtime.process_frame(frame(1), (stool,))

    assert conflict.association_conflict_count == 1
    assert conflict.revoked_edge_count == 0
    assert (conflict.matched_entity_count, conflict.new_entity_count) == (0, 1)

    revoked_runtime = Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(
            tracker=LocalTrackerConfig(confirm_hits=1, ambiguous_edge_score=0.8)
        ),
    )
    first = observation(
        0,
        ObservationKind.OBJECT,
        2,
        {key},
        image_feature=np.asarray((1.0, 0.0)),
        feature_model_id="clip",
    )
    weak = observation(
        1,
        ObservationKind.OBJECT,
        2,
        {key},
        image_feature=np.asarray((0.0, 1.0)),
        feature_model_id="clip",
    )
    revoked_runtime.process_frame(frame(0), (first,))
    revoked = revoked_runtime.process_frame(frame(1), (weak,))

    assert revoked.association_conflict_count == 0
    assert revoked.revoked_edge_count == 1
    assert (revoked.matched_entity_count, revoked.new_entity_count) == (1, 0)


def test_registry_validation_failure_leaves_runtime_state_unchanged() -> None:
    runtime = precision_runtime(confirm_hits=1)
    tracker = runtime.tracker
    registry = runtime.registry
    invalid = replace(
        observation(0, ObservationKind.OBJECT, 2, {(0, 0, 20)}),
        label="",
    )

    with pytest.raises(ValueError, match="label"):
        runtime.process_frame(frame(0), (invalid,))

    assert (runtime.revision, runtime.last_frame_id, runtime.last_timestamp) == (0, -1, 0.0)
    assert runtime.geometry.active_block_count == 0
    assert runtime.evidence.allocated_block_count == 0
    assert runtime.ownership.allocated_block_count == 0
    assert runtime.tracker is tracker
    assert tracker.tracks == {}
    assert tracker.graph.frame_ids == ()
    assert tracker._next_track_id == 1
    assert tracker._last_frame_id is None
    assert runtime.registry is registry
    assert registry.entities == {}
    assert registry._next_entity_id == 1
    assert registry._last_revision == -1


def test_recomputing_owner_retains_competing_entity_evidence() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)
    runtime.evidence.update_entity(key, 1, 2.0, 0.0, 0.0, 1)
    runtime.evidence.update_entity(key, 2, 1.0, 0.0, 0.0, 1)
    runtime.recompute_ownership((key,), revision=1)
    runtime.evidence.update_entity(key, 1, 0.0, 2.0, 1.0, 2)
    runtime.evidence.update_entity(key, 2, 2.0, 0.0, 1.0, 2)

    runtime.recompute_ownership((key,), revision=2)

    assert runtime.ownership.owner_of(key).entity_id == 2
    assert {item.entity_id for item in runtime.evidence.entity_candidates(key)} == {1, 2}


def test_only_absent_visibility_adds_negative_entity_evidence() -> None:
    runtime = Oviv2Runtime("room0")
    key = (0, 0, 20)
    runtime.evidence.update_entity(key, 1, 2.0, 0.0, 0.0, 1)
    runtime.ownership.assign(key, 1, 1.0, 1)

    runtime.apply_visibility(frame(), revision=2, entity_voxels={1: frozenset({key})})
    present = runtime.evidence.entity_candidates(key)[0]
    assert present.negative_support == 0.0

    absent_frame = frame(1)
    absent_frame.depth[:] = 2.0
    runtime.apply_visibility(absent_frame, revision=3, entity_voxels={1: frozenset({key})})
    absent = runtime.evidence.entity_candidates(key)[0]
    assert absent.negative_support > 0.0


def test_runtime_commit_round_trip_contains_only_voxel_layers(tmp_path: Path) -> None:
    runtime = Oviv2Runtime("room0")
    runtime.process_frame(frame(), ())

    committed = runtime.commit(tmp_path / "snapshot")
    restored = VoxelMapSnapshot.load(tmp_path / "snapshot")

    assert committed.metadata.revision == 1
    assert committed.metadata.schema_version == 2
    assert committed.registry is not None
    assert committed.registry.entities == runtime.registry.entities
    assert restored.metadata.scene_id == "room0"
    assert restored.registry is not None
    assert restored.registry.entities == runtime.registry.entities
    assert restored.metadata.dense_semantic_provenance is None
    assert {path.name for path in (tmp_path / "snapshot").iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "entities.jsonl",
        "checksums.json",
    }


def test_runtime_commit_new_refuses_to_replace_checkpoint(tmp_path: Path) -> None:
    runtime = Oviv2Runtime("room0")
    runtime.process_frame(frame(), ())
    target = tmp_path / "snapshot"

    committed = runtime.commit_new(target)
    before = (target / "checksums.json").read_bytes()

    with pytest.raises(FileExistsError):
        runtime.commit_new(target)

    assert committed.path == target
    assert VoxelMapSnapshot.load(target).metadata == committed.metadata
    assert (target / "checksums.json").read_bytes() == before


def test_dense_runtime_commit_round_trip_uses_schema3_typed_provenance(
    tmp_path: Path,
) -> None:
    runtime = dense_runtime()
    runtime.process_frame(
        frame(0),
        (),
        dense_semantics=dense_frame(0),
    )

    committed = runtime.commit(tmp_path / "snapshot")
    restored = VoxelMapSnapshot.load(committed.path)

    assert committed.metadata.schema_version == 3
    assert committed.metadata.dense_semantic_provenance == dense_provenance()
    assert isinstance(
        restored.metadata.dense_semantic_provenance,
        DenseSemanticProvenance,
    )
    assert restored.metadata.dense_semantic_provenance == dense_provenance()
    assert restored.metadata.dense_semantic_provenance.language_model_id == (
        "google/siglip2-giant-opt-patch16-384"
    )
    assert set(restored.checksums) == set(VoxelMapSnapshot._DATA_FILES_V2)
