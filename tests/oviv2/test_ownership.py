from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.ownership import ReversibleOwnershipStore

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame


def test_assignment_replacement_release_and_epoch_progression() -> None:
    store = ReversibleOwnershipStore(block_resolution=8)
    key = (-1, 2, 3)

    store.assign(key, entity_id=4, confidence=0.8, evidence_revision=1)
    first = store.owner_of(key)
    store.assign(key, entity_id=7, confidence=0.9, evidence_revision=2)
    second = store.owner_of(key)

    assert first is not None and first.epoch == 1
    assert second is not None
    assert second.entity_id == 7
    assert second.confidence == pytest.approx(0.9)
    assert second.epoch == 2
    assert store.voxels_for_entity(4) == frozenset()
    assert store.voxels_for_entity(7) == frozenset({key})
    assert store.release(key, expected_entity_id=4, evidence_revision=3) is False
    assert store.owner_of(key) == second
    assert store.release(key, expected_entity_id=7, evidence_revision=3) is True
    assert store.owner_of(key) is None

    store.assign(key, entity_id=4, confidence=0.6, evidence_revision=4)
    reassigned = store.owner_of(key)
    assert reassigned is not None and reassigned.epoch == 4


def test_ownership_rejects_invalid_values_and_revision_rollback() -> None:
    store = ReversibleOwnershipStore()
    key = (0, 0, 0)

    with pytest.raises(ValueError, match="entity_id"):
        store.assign(key, 0, 0.5, 1)
    with pytest.raises(ValueError, match="confidence"):
        store.assign(key, 1, 1.1, 1)
    store.assign(key, 1, 0.5, 2)
    with pytest.raises(ValueError, match="revision"):
        store.assign(key, 1, 0.6, 1)
    with pytest.raises(ValueError, match="revision"):
        store.release(key, 1, 1)


def test_release_preserves_geometry_and_competing_evidence() -> None:
    depth, rgb, intrinsics = _plane_frame()
    geometry = SparseTsdfVolume()
    _integrate_twice(geometry, depth, rgb, intrinsics, np.eye(4))
    evidence = SparseEvidenceStore()
    key = (0, 0, 20)
    evidence.update_entity(key, 1, 2.0, 0.0, 1.0, 1)
    evidence.update_entity(key, 2, 1.0, 0.0, 1.0, 1)
    ownership = ReversibleOwnershipStore()
    ownership.assign(key, 1, 0.75, 1)
    vertices_before = geometry.extract_mesh().vertex.positions.shape[0]
    evidence_before = evidence.entity_candidates(key)

    assert ownership.release(key, 1, 2) is True

    assert geometry.extract_mesh().vertex.positions.shape[0] == vertices_before
    assert evidence.entity_candidates(key) == evidence_before


def test_ownership_save_load_preserves_records_and_inverted_index(tmp_path: Path) -> None:
    store = ReversibleOwnershipStore(block_resolution=4)
    store.assign((-1, 4, 8), 3, 0.7, 2)
    store.assign((7, 4, 8), 3, 0.6, 3)
    path = tmp_path / "ownership.npz"

    store.save(path)
    restored = ReversibleOwnershipStore.load(path, block_resolution=4)

    assert restored.owner_of((-1, 4, 8)) == store.owner_of((-1, 4, 8))
    assert restored.voxels_for_entity(3) == frozenset({(-1, 4, 8), (7, 4, 8)})
    with pytest.raises(ValueError, match="block_resolution"):
        ReversibleOwnershipStore.load(path, block_resolution=8)
