from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame
from tests.oviv2.test_entities import _track


def _components():
    config = TsdfConfig()
    geometry = SparseTsdfVolume(config)
    depth, rgb, intrinsics = _plane_frame()
    _integrate_twice(geometry, depth, rgb, intrinsics, np.eye(4))
    evidence = SparseEvidenceStore(EvidenceConfig(block_resolution=config.block_resolution))
    ownership = ReversibleOwnershipStore(block_resolution=config.block_resolution)
    key = (0, 0, 20)
    evidence.update_semantic(key, 5, 2.0, 1)
    evidence.update_entity(key, 11, 2.0, 0.0, 1.0, 1)
    ownership.assign(key, 11, 0.8, 1)
    metadata = VoxelSnapshotMetadata(
        scene_id="room0",
        frame_id=10,
        timestamp=1.0,
        revision=1,
        voxel_size_m=config.voxel_size_m,
        block_resolution=config.block_resolution,
    )
    return metadata, geometry, evidence, ownership


def _registry() -> EntityRegistry:
    registry = EntityRegistry()
    original = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    registry.entities = {11: replace(original, entity_id=11)}
    registry._next_entity_id = 12
    return registry


def test_snapshot_commit_writes_and_loads_complete_contract(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"

    committed = VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    restored = VoxelMapSnapshot.load(target)

    assert {path.name for path in target.iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "checksums.json",
    }
    assert committed.metadata == metadata
    assert restored.metadata == metadata
    assert restored.registry is None
    assert restored.geometry.active_block_count == geometry.active_block_count
    assert restored.evidence.entity_candidates((0, 0, 20)) == evidence.entity_candidates((0, 0, 20))
    assert restored.ownership.owner_of((0, 0, 20)) == ownership.owner_of((0, 0, 20))


def test_v2_snapshot_embeds_registry_and_hashes_entities(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()

    snapshot = VoxelMapSnapshot.commit(
        tmp_path / "snapshot",
        metadata,
        geometry,
        evidence,
        ownership,
        registry=registry,
    )
    restored = VoxelMapSnapshot.load(snapshot.path)

    assert restored.metadata.schema_version == 2
    assert restored.registry is not None
    assert restored.registry.entities == registry.entities
    assert restored.registry._next_entity_id == 12
    assert "entities.jsonl" in restored.checksums
    assert {path.name for path in restored.path.iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "entities.jsonl",
        "checksums.json",
    }


@pytest.mark.parametrize("schema_version", [True, 0, 3, 2.0])
def test_snapshot_metadata_requires_integer_schema_one_or_two(
    schema_version: object,
) -> None:
    metadata, _, _, _ = _components()

    with pytest.raises(ValueError, match="schema_version"):
        replace(metadata, schema_version=schema_version)


def test_snapshot_commit_requires_registry_matching_schema(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    registry = _registry()

    with pytest.raises(ValueError, match="schema.*registry|registry.*schema"):
        VoxelMapSnapshot.commit(
            tmp_path / "missing",
            replace(metadata, schema_version=2),
            geometry,
            evidence,
            ownership,
        )
    with pytest.raises(ValueError, match="schema.*registry|registry.*schema"):
        VoxelMapSnapshot.commit(
            tmp_path / "unexpected",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


def test_snapshot_load_rejects_changed_checksum(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    with (target / "evidence.npz").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_corrupt_entities_checksum(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        target,
        replace(metadata, schema_version=2),
        geometry,
        evidence,
        ownership,
        registry=_registry(),
    )
    with (target / "entities.jsonl").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="checksum.*entities"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_schema_file_set_mismatch(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    metadata_path = target / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["schema_version"] = 2
    metadata_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="schema|files"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_extra_physical_file(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    (target / "extra.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="files"):
        VoxelMapSnapshot.load(target)


def test_snapshot_rejects_mismatched_voxel_configuration(tmp_path: Path) -> None:
    metadata, geometry, _, ownership = _components()
    evidence = SparseEvidenceStore(EvidenceConfig(block_resolution=4))

    with pytest.raises(ValueError, match="block_resolution"):
        VoxelMapSnapshot.commit(tmp_path / "snapshot", metadata, geometry, evidence, ownership)


def test_snapshot_rejects_owner_without_entity_evidence(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    ownership.assign((1, 0, 20), 99, 0.7, 2)

    with pytest.raises(ValueError, match="entity evidence"):
        VoxelMapSnapshot.commit(tmp_path / "snapshot", metadata, geometry, evidence, ownership)


def test_v2_snapshot_rejects_entity_references_missing_from_registry(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()
    evidence.update_entity((1, 0, 20), 99, 1.0, 0.0, 1.0, 1)

    with pytest.raises(ValueError, match="registry.*99|99.*registry"):
        VoxelMapSnapshot.commit(
            tmp_path / "evidence",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )

    evidence.update_entity((2, 0, 20), 98, 1.0, 0.0, 1.0, 1)
    ownership.assign((2, 0, 20), 98, 0.7, 1)
    with pytest.raises(ValueError, match="registry.*98|98.*registry"):
        VoxelMapSnapshot.commit(
            tmp_path / "ownership",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


@pytest.mark.parametrize("future", ["entity", "watermark"])
def test_v2_snapshot_rejects_registry_revision_after_metadata(
    tmp_path: Path,
    future: str,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()
    if future == "entity":
        registry.entities[11] = replace(registry.entities[11], last_revision=2)
    else:
        registry._last_revision = 2

    with pytest.raises(ValueError, match="revision"):
        VoxelMapSnapshot.commit(
            tmp_path / future,
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


def test_snapshot_overwrite_is_complete_and_leaves_no_temporary_directory(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = VoxelSnapshotMetadata(
        scene_id=metadata.scene_id,
        frame_id=20,
        timestamp=2.0,
        revision=2,
        voxel_size_m=metadata.voxel_size_m,
        block_resolution=metadata.block_resolution,
    )

    VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert VoxelMapSnapshot.load(target).metadata == updated
    assert not list(tmp_path.glob(".snapshot.*"))
