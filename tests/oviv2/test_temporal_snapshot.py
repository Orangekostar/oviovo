from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_geometry import ObjectSubmap
from src.oviv2.temporal_lifecycle import TemporalLifecycle, TemporalLifecycleState
from src.oviv2.temporal_snapshot import (
    TemporalCompactCheckpoint,
    TemporalCurrentSnapshot,
    TemporalSnapshotMetadata,
    build_temporal_map_snapshot,
)
from src.oviv2.temporal_state import TemporalEntityState


def _geometry_config(**changes: object) -> TemporalGeometryConfig:
    values = dict(
        voxel_size_m=0.1,
        depth_max_m=4.0,
        maximum_entities=4,
        maximum_object_voxels=8,
        maximum_visibility_points_per_entity=8,
        background_block_count=16,
        background_mask_dilation_px=0,
        minimum_icp_points=3,
        minimum_icp_fitness=0.5,
        maximum_icp_rmse_m=0.1,
        maximum_motion_m=2.0,
    )
    values.update(changes)
    return TemporalGeometryConfig(**values)


def _entity(entity_id: int, lifecycle: TemporalLifecycle, *, x: float = 0.0) -> TemporalEntityState:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    submap = ObjectSubmap(
        reference_centroid_xyz=(0.0, 0.0, 0.0),
        local_voxel_keys=((0, 0, 0), (1, 0, 0)),
        local_points_xyz=np.asarray([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]]),
        weights=np.ones(2),
        last_seen_frame_ids=np.asarray([2, 3], dtype=np.int64),
    )
    return TemporalEntityState(
        lifecycle=TemporalLifecycleState(
            entity_id=entity_id,
            lifecycle=lifecycle,
            existence_log_odds=1.25 - entity_id,
            last_frame_id=3,
            last_timestamp=3.0,
            absent_streak=entity_id,
            absence_view_bins=tuple(range(min(entity_id, 2))),
        ),
        semantic_probabilities=((1, 0.8), (2, 0.2)),
        image_prototype=np.asarray([1.0, 0.0]),
        feature_model_id="test",
        extent_xyz=(1.0, 1.0, 1.0),
        object_to_world=pose,
        submap=submap,
        first_seen_frame_id=0,
        last_seen_frame_id=3,
    )


def _snapshot() -> TemporalCurrentSnapshot:
    return TemporalCurrentSnapshot(
        metadata=TemporalSnapshotMetadata("scene", 3, 3.0, 4, 0.1, "a" * 64),
        entities=(
            _entity(1, TemporalLifecycle.ACTIVE, x=1.0),
            _entity(2, TemporalLifecycle.UNCERTAIN, x=2.0),
            _entity(3, TemporalLifecycle.DORMANT, x=3.0),
        ),
        background=TemporalBackgroundVolume(_geometry_config()),
    )


def test_public_contracts_are_frozen_strict_and_do_not_alias_arrays() -> None:
    assert [field.name for field in fields(TemporalSnapshotMetadata)] == [
        "scene_id", "frame_id", "timestamp", "revision", "voxel_size_m", "config_sha256"
    ]
    metadata = TemporalSnapshotMetadata(" scene ", 3, 3.0, 4, 0.1, "a" * 64)
    assert metadata.scene_id == "scene"
    with pytest.raises(FrozenInstanceError):
        metadata.frame_id = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="config_sha256"):
        TemporalSnapshotMetadata("scene", 3, 3.0, 4, 0.1, "bad")

    arrays = {
        "entity_ids": np.asarray([1], dtype=np.int64),
        "lifecycle_codes": np.asarray([0], dtype=np.uint8),
        "existence_log_odds": np.asarray([1.0]),
        "absent_streaks": np.asarray([0], dtype=np.int64),
        "distinct_view_bin_counts": np.asarray([0], dtype=np.int64),
        "object_to_world": np.eye(4)[None],
        "voxel_keys": np.asarray([[0, 0, 0]], dtype=np.int64),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
    }
    checkpoint = TemporalCompactCheckpoint(metadata=metadata, **arrays)
    arrays["entity_ids"][0] = 99
    assert checkpoint.entity_ids.tolist() == [1]
    assert not checkpoint.entity_ids.flags.writeable
    invalid_code = dict(arrays)
    invalid_code["lifecycle_codes"] = np.asarray([256], dtype=np.int64)
    with pytest.raises(ValueError, match="lifecycle_codes"):
        TemporalCompactCheckpoint(metadata=metadata, **invalid_code)
    invalid_pose = dict(arrays)
    invalid_pose["object_to_world"] = np.eye(4)[None]
    invalid_pose["object_to_world"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid"):
        TemporalCompactCheckpoint(metadata=metadata, **invalid_pose)


def test_build_temporal_map_snapshot_selects_and_sorts_causally() -> None:
    result = build_temporal_map_snapshot(_snapshot(), ("unknown", "chair", "table"))
    assert result.method == "OVIV2-temporal"
    assert result.scope == "current"
    assert [entity.entity_id for entity in result.entities] == ["temporal:1", "temporal:2"]
    assert [entity.lifecycle_state for entity in result.entities] == ["active", "uncertain"]
    np.testing.assert_allclose(result.entities[0].points_xyz[:, 0], [1.05, 1.15])
    assert all(entity.last_seen <= 3.0 for entity in result.entities)


def test_build_temporal_map_snapshot_extracts_real_nonempty_background() -> None:
    background = TemporalBackgroundVolume(_geometry_config(background_block_count=32))
    frame = Frame(
        frame_id=0,
        timestamp=0.0,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
    )
    background = background.trial_integrate(frame, frame.depth)
    snapshot = TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "e" * 64), (), background
    )
    result = build_temporal_map_snapshot(snapshot, ("unknown",))
    assert result.background_xyz is not None
    assert len(result.background_xyz) > 0
    expected = result.background_xyz[np.lexsort(
        (result.background_xyz[:, 2], result.background_xyz[:, 1], result.background_xyz[:, 0])
    )]
    np.testing.assert_array_equal(result.background_xyz, expected)


def test_current_snapshot_rejects_future_or_unsorted_entities() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="sorted"):
        TemporalCurrentSnapshot(snapshot.metadata, tuple(reversed(snapshot.entities)), snapshot.background)
    future = _entity(1, TemporalLifecycle.ACTIVE)
    future = TemporalEntityState(
        lifecycle=TemporalLifecycleState(1, TemporalLifecycle.ACTIVE, 1.0, 4, 4.0, 0, ()),
        semantic_probabilities=future.semantic_probabilities,
        image_prototype=future.image_prototype,
        feature_model_id=future.feature_model_id,
        extent_xyz=future.extent_xyz,
        object_to_world=future.object_to_world,
        submap=future.submap,
        first_seen_frame_id=0,
        last_seen_frame_id=3,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        TemporalCurrentSnapshot(snapshot.metadata, (future,), snapshot.background)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"entity_id": True}, "entity_id"),
        ({"entity_id": -1}, "entity_id"),
        ({"existence_log_odds": float("nan")}, "existence_log_odds"),
        ({"absent_streak": -1}, "absent_streak"),
        ({"last_frame_id": -1}, "last_frame_id"),
        ({"last_timestamp": float("nan")}, "last_timestamp"),
        ({"lifecycle": "active"}, "lifecycle"),
    ],
)
def test_current_snapshot_strictly_validates_nested_lifecycle(
    changes: dict[str, object], message: str
) -> None:
    snapshot = _snapshot()
    entity = snapshot.entities[0]
    lifecycle = replace(entity.lifecycle)
    for name, value in changes.items():
        object.__setattr__(lifecycle, name, value)
    object.__setattr__(entity, "lifecycle", lifecycle)
    with pytest.raises((TypeError, ValueError), match=message):
        TemporalCurrentSnapshot(snapshot.metadata, (entity,), snapshot.background)


def test_current_snapshot_does_not_alias_input_lifecycle() -> None:
    entity = _entity(1, TemporalLifecycle.ACTIVE)
    snapshot = TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 3, 3.0, 4, 0.1, "a" * 64),
        (entity,),
        TemporalBackgroundVolume(_geometry_config()),
    )
    object.__setattr__(entity.lifecycle, "entity_id", 99)
    assert snapshot.entities[0].lifecycle.entity_id == 1


def test_compact_from_snapshot_contains_only_bounded_current_state() -> None:
    compact = TemporalCompactCheckpoint.from_snapshot(
        _snapshot(), maximum_entities=4, maximum_object_voxels=8
    )
    assert compact.entity_ids.tolist() == [1, 2, 3]
    assert compact.lifecycle_codes.tolist() == [0, 1, 2]
    assert compact.voxel_offsets.tolist() == [0, 2, 4, 6]
    assert compact.voxel_keys.tolist() == [[0, 0, 0], [1, 0, 0]] * 3
    assert not hasattr(compact, "background")
    assert not hasattr(compact, "semantic_probabilities")
    with pytest.raises(ValueError, match="capacity"):
        TemporalCompactCheckpoint.from_snapshot(
            _snapshot(), maximum_entities=2, maximum_object_voxels=8
        )


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_compact_commit_load_is_deterministic_no_replace_and_witnessed(tmp_path: Path) -> None:
    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    first = compact.commit_new(tmp_path / "first", maximum_entities=4, maximum_object_voxels=8)
    second = compact.commit_new(tmp_path / "second", maximum_entities=4, maximum_object_voxels=8)
    assert _tree_hashes(first.path) == _tree_hashes(second.path)
    assert set(_tree_hashes(first.path)) == {"arrays.npz", "checksums.json", "manifest.json"}
    first.revalidate_source()
    restored = TemporalCompactCheckpoint.load(first.path, maximum_entities=4, maximum_object_voxels=8)
    np.testing.assert_array_equal(restored.voxel_keys, compact.voxel_keys)
    with pytest.raises(FileExistsError):
        compact.commit_new(first.path, maximum_entities=4, maximum_object_voxels=8)
    (first.path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        first.revalidate_source()


def test_compact_rejects_symlink_parent_and_tampered_archive(tmp_path: Path) -> None:
    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        compact.commit_new(link / "bad", maximum_entities=4, maximum_object_voxels=8)
    receipt = compact.commit_new(real / "good", maximum_entities=4, maximum_object_voxels=8)
    archive = receipt.path / "arrays.npz"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        TemporalCompactCheckpoint.load(receipt.path, maximum_entities=4, maximum_object_voxels=8)


def test_compact_concurrent_publish_has_exactly_one_winner(tmp_path: Path) -> None:
    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    target = tmp_path / "race"

    def commit() -> str:
        try:
            compact.commit_new(target, maximum_entities=4, maximum_object_voxels=8)
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _: commit(), range(2)))
    assert outcomes == ["exists", "published"]
    TemporalCompactCheckpoint.load(target, maximum_entities=4, maximum_object_voxels=8)


def test_compact_known_write_failure_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oviv2.temporal_snapshot as module

    def fail(*args: object, **kwargs: object) -> tuple[int, int]:
        raise RuntimeError("write")

    monkeypatch.setattr(module, "_write_regular_at", fail)
    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    with pytest.raises(RuntimeError, match="write"):
        compact.commit_new(tmp_path / "failed", maximum_entities=4, maximum_object_voxels=8)
    assert list(tmp_path.iterdir()) == []


def test_compact_validates_capacity_before_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as module

    empty = TemporalCurrentSnapshot(
        _snapshot().metadata, (), TemporalBackgroundVolume(_geometry_config())
    )
    compact = TemporalCompactCheckpoint.from_snapshot(empty)
    monkeypatch.setattr(
        module,
        "_canonical_npz",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("serialized")),
    )
    with pytest.raises(ValueError, match="maximum_entities"):
        compact.commit_new(
            tmp_path / "failed",
            maximum_entities=True,  # type: ignore[arg-type]
            maximum_object_voxels=8,
        )
    assert list(tmp_path.iterdir()) == []


def test_compact_publication_after_rename_failure_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as module

    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot())
    target = tmp_path / "published"
    monkeypatch.setattr(
        module.TemporalCompactCheckpoint,
        "load",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load"))),
    )
    with pytest.raises(module.TemporalCheckpointPublicationUncertainError) as caught:
        compact.commit_new(target, maximum_entities=4, maximum_object_voxels=8)
    assert caught.value.published is True
    assert target.is_dir()


def test_compact_witness_hash_rejects_same_size_tamper_and_inode_swap(tmp_path: Path) -> None:
    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot())
    loaded = compact.commit_new(tmp_path / "checkpoint", maximum_entities=4, maximum_object_voxels=8)
    member = loaded.path / "manifest.json"
    before = member.stat()
    content = member.read_bytes()
    replacement = bytes([content[0] ^ 1]) + content[1:]
    member.write_bytes(replacement)
    os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="content|hash|identity"):
        loaded.revalidate_source()

    second = compact.commit_new(tmp_path / "second", maximum_entities=4, maximum_object_voxels=8)
    original = second.path / "manifest.json"
    swapped = second.path / "replacement"
    swapped.write_bytes(original.read_bytes())
    os.utime(swapped, ns=(original.stat().st_atime_ns, original.stat().st_mtime_ns))
    os.replace(swapped, original)
    with pytest.raises(ValueError, match="identity"):
        second.revalidate_source()


def test_compact_load_explicitly_disables_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oviv2.temporal_snapshot as module

    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    target = compact.commit_new(tmp_path / "checkpoint", maximum_entities=4, maximum_object_voxels=8).path
    original = module.np.load
    calls: list[object] = []

    def capture(*args: object, **kwargs: object):
        calls.append(kwargs.get("allow_pickle"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.np, "load", capture)
    TemporalCompactCheckpoint.load(target, maximum_entities=4, maximum_object_voxels=8)
    assert calls and set(calls) == {False}


def test_compact_load_rejects_bad_npy_contract_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as module

    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot(), maximum_entities=4, maximum_object_voxels=8)
    target = compact.commit_new(tmp_path / "checkpoint", maximum_entities=4, maximum_object_voxels=8).path
    arrays = {name: getattr(compact, name) for name in module._COMPACT_ARRAY_NAMES}
    arrays["entity_ids"] = arrays["entity_ids"].astype(np.float32)
    archive = module._canonical_npz(arrays, module._COMPACT_ARRAY_NAMES)
    (target / "arrays.npz").write_bytes(archive)
    checksums = json.loads((target / "checksums.json").read_text(encoding="utf-8"))
    checksums["arrays.npz"] = hashlib.sha256(archive).hexdigest()
    (target / "checksums.json").write_bytes(module._canonical_json(checksums))

    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    with pytest.raises(ValueError, match="dtype"):
        TemporalCompactCheckpoint.load(target, maximum_entities=4, maximum_object_voxels=8)


def test_compact_load_rejects_directory_swap_during_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as module

    compact = TemporalCompactCheckpoint.from_snapshot(_snapshot())
    source = compact.commit_new(
        tmp_path / "source", maximum_entities=4, maximum_object_voxels=8
    ).path
    empty = TemporalCurrentSnapshot(
        _snapshot().metadata, (), TemporalBackgroundVolume(_geometry_config())
    )
    replacement = TemporalCompactCheckpoint.from_snapshot(empty).commit_new(
        tmp_path / "replacement", maximum_entities=4, maximum_object_voxels=8
    ).path
    original_load = module.np.load
    swapped = False

    def swap_after_open(*args: object, **kwargs: object):
        nonlocal swapped
        result = original_load(*args, **kwargs)
        if not swapped:
            swapped = True
            os.rename(source, tmp_path / "old-source")
            os.rename(replacement, source)
        return result

    monkeypatch.setattr(module.np, "load", swap_after_open)
    with pytest.raises(ValueError, match="identity|changed"):
        TemporalCompactCheckpoint.load(
            source, maximum_entities=4, maximum_object_voxels=8
        )
