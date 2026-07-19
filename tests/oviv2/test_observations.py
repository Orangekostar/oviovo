from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2.observations import (
    CachedFrontendAdapter,
    FrameObservation,
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


def _observation(**changes: object) -> FrameObservation:
    values = {
        "observation_id": 1,
        "frame_id": 2,
        "timestamp": 3.0,
        "kind": ObservationKind.OBJECT,
        "label": "chair",
        "semantic_id": 2,
        "confidence": 0.9,
        "mask": np.ones((2, 2), dtype=bool),
        "bbox_xyxy": (0.0, 0.0, 2.0, 2.0),
        "voxel_keys": frozenset({(0, 0, 1)}),
        "centroid_xyz": (0.0, 0.0, 1.0),
        "bounds_min_xyz": (0.0, 0.0, 1.0),
        "bounds_max_xyz": (0.0, 0.0, 1.0),
    }
    values.update(changes)
    return FrameObservation(**values)


def _write_cache(
    path: Path,
    *,
    masks: np.ndarray,
    labels: list[str],
    image_feats: np.ndarray | None = None,
    text_feats: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mask": masks,
        "xyxy": np.tile(np.asarray([[0, 0, 5, 4]], dtype=np.float32), (len(labels), 1)),
        "confidence": np.linspace(0.9, 0.8, len(labels), dtype=np.float32),
        "class_id": np.arange(len(labels), dtype=np.int64),
        "classes": labels,
    }
    if image_feats is not None:
        payload["image_feats"] = image_feats
    if text_feats is not None:
        payload["text_feats"] = text_feats
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
    assert first[0].image_feature is None
    assert first[0].text_feature is None
    assert first[0].feature_model_id is None


def test_adapter_preserves_normalized_cached_features_and_observation_quality(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    masks = np.ones((2, 4, 5), dtype=bool)
    image_feats = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    text_feats = np.asarray([[0.0, 5.0], [8.0, 6.0]], dtype=np.float64)
    _write_cache(
        tmp_path / "frame000002.pkl.gz",
        masks=masks,
        labels=["chair", "wall"],
        image_feats=image_feats,
        text_feats=text_feats,
    )
    adapter = CachedFrontendAdapter(
        tmp_path,
        vocabulary,
        min_valid_points=1,
        feature_model_id="clip-sha256:test",
    )

    observations = adapter.observe(_frame(), cache_frame_id=2)

    assert len(observations) == 2
    for observation in observations:
        for feature in (observation.image_feature, observation.text_feature):
            assert feature is not None
            assert feature.dtype == np.float32
            assert feature.flags.c_contiguous
            assert feature.flags.writeable is False
            assert np.linalg.norm(feature) == pytest.approx(1.0)
        assert observation.feature_model_id == "clip-sha256:test"
        assert np.linalg.norm(observation.view_direction_xyz) == pytest.approx(1.0)
        assert observation.visible_pixel_count == 20
        assert 0.0 <= observation.border_contact_fraction <= 1.0
        assert observation.border_contact_fraction == pytest.approx(14.0 / 20.0)


@pytest.mark.parametrize(
    ("image_feats", "text_feats", "message"),
    [
        (np.ones((2, 2)), None, "row count"),
        (np.ones(2), None, "two dimensional"),
        (np.zeros((1, 2)), None, "nonzero"),
        (np.asarray([[np.nan, 1.0]]), None, "finite"),
        (np.ones((1, 2)), np.ones((1, 3)), "dimensions"),
    ],
    ids=("row-count", "rank", "zero-norm", "non-finite", "dimension-mismatch"),
)
def test_adapter_rejects_invalid_cached_features(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
    image_feats: np.ndarray,
    text_feats: np.ndarray | None,
    message: str,
) -> None:
    _write_cache(
        tmp_path / "frame000000.pkl.gz",
        masks=np.ones((1, 4, 5), dtype=bool),
        labels=["chair"],
        image_feats=image_feats,
        text_feats=text_feats,
    )
    adapter = CachedFrontendAdapter(
        tmp_path,
        vocabulary,
        min_valid_points=1,
        feature_model_id="clip-sha256:test",
    )

    with pytest.raises(ValueError, match=message):
        adapter.observe(_frame(), cache_frame_id=0)


def test_adapter_requires_model_id_when_cache_contains_features(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    _write_cache(
        tmp_path / "frame000000.pkl.gz",
        masks=np.ones((1, 4, 5), dtype=bool),
        labels=["chair"],
        image_feats=np.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="feature_model_id"):
        CachedFrontendAdapter(tmp_path, vocabulary, min_valid_points=1).observe(_frame(), 0)


@pytest.mark.parametrize("feature_model_id", ["", "   "])
def test_adapter_rejects_blank_feature_model_id(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
    feature_model_id: str,
) -> None:
    with pytest.raises(ValueError, match="feature_model_id"):
        CachedFrontendAdapter(tmp_path, vocabulary, feature_model_id=feature_model_id)


def test_frame_observation_normalizes_optional_fields() -> None:
    observation = _observation(
        image_feature=np.asarray([3.0, 4.0]),
        text_feature=np.asarray([0.0, 5.0]),
        feature_model_id="  clip-sha256:test  ",
        view_direction_xyz=(0.0, 0.0, 4.0),
        visible_pixel_count=np.int64(4),
        border_contact_fraction=np.float32(0.5),
    )

    assert observation.image_feature.shape == (2,)
    assert observation.text_feature.shape == (2,)
    assert observation.feature_model_id == "clip-sha256:test"
    assert observation.view_direction_xyz == (0.0, 0.0, 1.0)
    assert type(observation.visible_pixel_count) is int
    assert type(observation.border_contact_fraction) is float


@pytest.mark.parametrize("magnitude", [1e-300, 1e300], ids=("tiny", "large"))
def test_frame_observation_normalizes_extreme_finite_features(magnitude: float) -> None:
    observation = _observation(
        image_feature=np.asarray([magnitude, 0.0], dtype=np.float64),
        text_feature=np.asarray([0.0, magnitude], dtype=np.float64),
        feature_model_id="clip-sha256:test",
    )

    for feature in (observation.image_feature, observation.text_feature):
        assert feature.dtype == np.float32
        assert feature.flags.c_contiguous
        assert feature.flags.writeable is False
        assert np.all(np.isfinite(feature))
        assert np.any(feature)
        assert np.linalg.norm(feature) == pytest.approx(1.0)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300], ids=("tiny", "large"))
def test_adapter_normalizes_extreme_finite_cached_features(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
    magnitude: float,
) -> None:
    _write_cache(
        tmp_path / "frame000000.pkl.gz",
        masks=np.ones((1, 4, 5), dtype=bool),
        labels=["chair"],
        image_feats=np.asarray([[magnitude, 0.0]], dtype=np.float64),
        text_feats=np.asarray([[0.0, magnitude]], dtype=np.float64),
    )
    observation = CachedFrontendAdapter(
        tmp_path,
        vocabulary,
        min_valid_points=1,
        feature_model_id="clip-sha256:test",
    ).observe(_frame(), 0)[0]

    assert np.array_equal(observation.image_feature, np.asarray([1.0, 0.0]))
    assert np.array_equal(observation.text_feature, np.asarray([0.0, 1.0]))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"feature_model_id": 3}, TypeError, "feature_model_id"),
        (
            {
                "image_feature": np.ones((1, 2)),
                "feature_model_id": "clip-sha256:test",
            },
            ValueError,
            "one dimensional",
        ),
        (
            {
                "text_feature": np.asarray(1.0),
                "feature_model_id": "clip-sha256:test",
            },
            ValueError,
            "one dimensional",
        ),
        ({"view_direction_xyz": (1.0, 2.0)}, ValueError, "three dimensional"),
        ({"view_direction_xyz": (np.nan, 0.0, 1.0)}, ValueError, "finite"),
        ({"view_direction_xyz": (0.0, 0.0, 0.0)}, ValueError, "nonzero"),
        ({"visible_pixel_count": True}, TypeError, "integer"),
        ({"visible_pixel_count": 1.0}, TypeError, "integer"),
        ({"visible_pixel_count": -1}, ValueError, "non-negative"),
        ({"border_contact_fraction": np.nan}, ValueError, "finite"),
        ({"border_contact_fraction": -0.1}, ValueError, r"\[0, 1\]"),
        ({"border_contact_fraction": 1.1}, ValueError, r"\[0, 1\]"),
    ],
    ids=(
        "model-id-type",
        "image-rank",
        "text-rank",
        "view-rank",
        "view-non-finite",
        "view-zero",
        "visible-bool",
        "visible-float",
        "visible-negative",
        "border-non-finite",
        "border-low",
        "border-high",
    ),
)
def test_frame_observation_rejects_invalid_optional_fields(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _observation(**changes)


def test_adapter_uses_none_for_degenerate_view_direction(
    tmp_path: Path,
    vocabulary: ReplicaVocabulary,
) -> None:
    _write_cache(
        tmp_path / "frame000000.pkl.gz",
        masks=np.ones((1, 4, 5), dtype=bool),
        labels=["chair"],
    )
    frame = _frame()
    frame.pose[:3, :3] = 0.0

    observation = CachedFrontendAdapter(
        tmp_path,
        vocabulary,
        min_valid_points=1,
    ).observe(frame, 0)[0]

    assert observation.view_direction_xyz is None


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
