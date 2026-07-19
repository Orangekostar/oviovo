from __future__ import annotations

from pathlib import Path

import pytest

from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore


def test_absent_query_does_not_allocate_a_block() -> None:
    store = SparseEvidenceStore()

    assert store.semantic_candidates((1, 2, 3)) == ()
    assert store.entity_candidates((1, 2, 3)) == ()
    assert store.allocated_block_count == 0


def test_first_update_allocates_once_and_repeated_update_accumulates() -> None:
    store = SparseEvidenceStore()

    store.update_semantic((1, 2, 3), label_id=7, support_delta=0.25, revision=1)
    store.update_semantic((2, 2, 3), label_id=8, support_delta=1.0, revision=1)
    store.update_semantic((1, 2, 3), label_id=7, support_delta=0.75, revision=2)

    assert store.allocated_block_count == 1
    candidate = store.semantic_candidates((1, 2, 3))[0]
    assert candidate.label_id == 7
    assert candidate.support == pytest.approx(1.0)
    assert candidate.revision == 2


def test_semantic_top_k_evicts_by_support_then_integer_id() -> None:
    store = SparseEvidenceStore(EvidenceConfig(semantic_top_k=2))
    key = (-1, 0, 0)

    store.update_semantic(key, label_id=9, support_delta=1.0, revision=1)
    store.update_semantic(key, label_id=3, support_delta=1.0, revision=2)
    store.update_semantic(key, label_id=5, support_delta=2.0, revision=3)

    candidates = store.semantic_candidates(key)
    assert [(item.label_id, item.support) for item in candidates] == [(5, 2.0), (3, 1.0)]


def test_entity_candidates_keep_positive_and_negative_support() -> None:
    store = SparseEvidenceStore(EvidenceConfig(entity_top_k=2))
    key = (8, 0, 0)

    store.update_entity(key, 4, 3.0, 0.5, timestamp=1.0, revision=1)
    store.update_entity(key, 4, 0.0, 1.0, timestamp=2.0, revision=2)
    store.update_entity(key, 7, 1.5, 0.0, timestamp=2.0, revision=2)

    candidates = store.entity_candidates(key)
    assert [item.entity_id for item in candidates] == [4, 7]
    assert candidates[0].positive_support == pytest.approx(3.0)
    assert candidates[0].negative_support == pytest.approx(1.5)
    assert candidates[0].timestamp == pytest.approx(2.0)
    assert candidates[0].revision == 2


def test_updates_reject_invalid_ids_deltas_and_revision_rollback() -> None:
    store = SparseEvidenceStore()
    key = (0, 0, 0)

    with pytest.raises(ValueError, match="label_id"):
        store.update_semantic(key, 0, 1.0, 1)
    with pytest.raises(ValueError, match="support_delta"):
        store.update_semantic(key, 1, 0.0, 1)
    with pytest.raises(ValueError, match="entity_id"):
        store.update_entity(key, 0, 1.0, 0.0, 1.0, 1)
    with pytest.raises(ValueError, match="delta"):
        store.update_entity(key, 1, 0.0, 0.0, 1.0, 1)

    store.update_semantic(key, 1, 1.0, 2)
    with pytest.raises(ValueError, match="revision"):
        store.update_semantic(key, 1, 1.0, 1)


def test_save_load_preserves_candidates_config_and_revisions(tmp_path: Path) -> None:
    config = EvidenceConfig(block_resolution=4, semantic_top_k=2, entity_top_k=3)
    store = SparseEvidenceStore(config)
    store.update_semantic((-1, 4, 8), 2, 1.25, 4)
    store.update_semantic((-1, 4, 8), 3, 0.75, 5)
    store.update_entity((-1, 4, 8), 11, 2.0, 0.25, 3.5, 6)
    path = tmp_path / "evidence.npz"

    store.save(path)
    restored = SparseEvidenceStore.load(path, config)

    assert restored.allocated_block_count == 1
    assert restored.semantic_candidates((-1, 4, 8)) == store.semantic_candidates((-1, 4, 8))
    assert restored.entity_candidates((-1, 4, 8)) == store.entity_candidates((-1, 4, 8))
    with pytest.raises(ValueError, match="block_resolution"):
        SparseEvidenceStore.load(path, EvidenceConfig(block_resolution=8))
