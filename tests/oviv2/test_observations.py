from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import (
    CachedFrontendAdapter,
    ObservationKind,
    ReplicaVocabulary,
)


def _frame() -> Frame:
    depth = np.ones((4, 5), dtype=np.float32)
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    return Frame(
        frame_id=20,
        source_frame_id=20,
        rgb=rgb,
        depth=depth,
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(2.0, 2.0, 2.0, 1.5, 5, 4),
        timestamp=2.0,
    )


def _write_cache(path: Path, *, masks: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mask": masks,
        "xyxy": np.tile(np.asarray([[0, 0, 5, 4]], dtype=np.float32), (len(labels), 1)),
        "confidence": np.linspace(0.9, 0.8, len(labels), dtype=np.float32),
        "class_id": np.arange(len(labels), dtype=np.int64),
        "classes": labels,
    }
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream)


@pytest.fixture
def vocabulary() -> ReplicaVocabulary:
    return ReplicaVocabulary(
        classes=("wall", "chair"),
        aliases={"seat": "chair", "wall-other": "wall"},
    )


def test_vocabulary_normalizes_aliases_and_assigns_one_based_ids(vocabulary: ReplicaVocabulary) -> None:
    assert vocabulary.resolve(" Seat ") == ("chair", 2, ObservationKind.OBJECT)
    assert vocabulary.resolve("wall_other") == ("wall", 1, ObservationKind.STRUCTURE)
    assert vocabulary.resolve("not-in-vocabulary") == (
        "not-in-vocabulary",
        0,
        ObservationKind.UNKNOWN,
    )


def test_adapter_lifts_requested_frame_to_deterministic_voxel_observations(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    masks = np.zeros((2, 4, 5), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, 0:2, 0:2] = True
    _write_cache(tmp_path / "frame000002.pkl.gz", masks=masks, labels=["seat", "wall-other"])
    adapter = CachedFrontendAdapter(tmp_path, vocabulary, voxel_size_m=0.5, min_valid_points=1)

    first = adapter.observe(_frame(), cache_frame_id=2)
    second = adapter.observe(_frame(), cache_frame_id=2)

    assert [item.kind for item in first] == [ObservationKind.OBJECT, ObservationKind.STRUCTURE]
    assert [item.semantic_id for item in first] == [2, 1]
    assert first[0].voxel_keys == second[0].voxel_keys
    assert first[0].centroid_xyz == second[0].centroid_xyz
    assert first[0].mask.flags.writeable is False
    assert first[0].frame_id == 20
    assert first[0].observation_id == 20_000_000


def test_adapter_returns_empty_batch_for_empty_cache(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    _write_cache(tmp_path / "frame000000.pkl.gz", masks=np.zeros((0, 4, 5), bool), labels=[])

    assert CachedFrontendAdapter(tmp_path, vocabulary).observe(_frame(), cache_frame_id=0) == ()


def test_adapter_rejects_missing_or_shape_incompatible_cache(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    adapter = CachedFrontendAdapter(tmp_path, vocabulary)
    with pytest.raises(FileNotFoundError):
        adapter.observe(_frame(), cache_frame_id=0)

    _write_cache(tmp_path / "frame000001.pkl.gz", masks=np.zeros((1, 3, 5), bool), labels=["chair"])
    with pytest.raises(ValueError, match="mask shape"):
        adapter.observe(_frame(), cache_frame_id=1)


def test_adapter_drops_observation_without_enough_valid_depth(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    masks = np.ones((1, 4, 5), dtype=bool)
    _write_cache(tmp_path / "frame000000.pkl.gz", masks=masks, labels=["chair"])
    frame = _frame()
    frame.depth[:] = 0.0

    assert CachedFrontendAdapter(tmp_path, vocabulary, min_valid_points=1).observe(frame, 0) == ()
