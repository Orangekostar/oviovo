from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from src.oviv2.dense_semantics import DenseSemanticFrame
from src.oviv2.hybrid_frontend import (
    DenseMaskLabel,
    FrontendBatch,
    HybridFrameResult,
    HybridFrontendConfig,
    MaskOverlap,
    aggregate_dense_mask,
    mask_overlap,
    select_hybrid_proposals,
)


def _batch(**changes: object) -> FrontendBatch:
    values = {
        "masks": np.asarray(
            [
                [[1, 1, 0], [1, 0, 0]],
                [[0, 0, 1], [0, 1, 1]],
            ],
            dtype=np.uint8,
        ),
        "boxes_xyxy": np.asarray([[0, 0, 2, 2], [1, 0, 3, 2]], dtype=np.float64),
        "confidences": np.asarray([0.9, 0.7], dtype=np.float64),
        "labels": ("chair", "table"),
        "image_features": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64),
    }
    values.update(changes)
    return FrontendBatch(**values)


def _dense_frame(
    class_ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    image_shape: tuple[int, int] = (4, 4),
    sample_stride: int = 2,
    class_count: int | None = None,
) -> DenseSemanticFrame:
    class_ids = np.asarray(class_ids, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    sampled_shape = (
        math.ceil(image_shape[0] / sample_stride),
        math.ceil(image_shape[1] / sample_stride),
    )
    if class_count is None:
        class_count = max(1, int(np.max(class_ids, initial=0)))
    if class_ids.shape[:2] != sampled_shape:
        raise AssertionError("test fixture does not match sampled image shape")
    second = (
        probabilities[..., 1]
        if probabilities.shape[2] > 1
        else np.zeros(sampled_shape, dtype=np.float32)
    )
    return DenseSemanticFrame(
        cache_frame_id=4,
        source_frame_id=9,
        image_shape=image_shape,
        sample_stride=sample_stride,
        class_count=class_count,
        class_ids=class_ids,
        probabilities=probabilities,
        entropy=np.zeros(sampled_shape, dtype=np.float32),
        margin=np.asarray(probabilities[..., 0] - second, dtype=np.float32),
    )


def _dense_aggregation_fixture() -> DenseSemanticFrame:
    return _dense_frame(
        class_ids=np.asarray(
            [
                [[1, 2, 3], [2, 1, 3]],
                [[3, 2, 1], [2, 4, 1]],
            ]
        ),
        probabilities=np.asarray(
            [
                [[0.6, 0.3, 0.1], [0.8, 0.1, 0.05]],
                [[0.5, 0.4, 0.1], [0.7, 0.2, 0.1]],
            ]
        ),
        class_count=4,
    )


def _sampled_mask(
    selected: tuple[tuple[int, int], ...],
    *,
    image_shape: tuple[int, int] = (4, 4),
    sample_stride: int = 2,
    dtype: np.dtype = np.dtype(np.bool_),
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=dtype)
    for row, column in selected:
        mask[row * sample_stride, column * sample_stride] = 1
    return mask


def test_mask_overlap_reports_iou_and_directed_coverages() -> None:
    left = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=bool)
    right = np.asarray([[0, 1, 1], [0, 1, 1]], dtype=bool)

    overlap = mask_overlap(left, right)

    assert overlap.intersection == 2
    assert overlap.iou == pytest.approx(2 / 6)
    assert overlap.left_coverage == pytest.approx(0.5)
    assert overlap.right_coverage == pytest.approx(0.5)


def test_mask_overlap_returns_zero_ratios_for_empty_masks() -> None:
    overlap = mask_overlap(np.zeros((2, 3)), np.zeros((2, 3)))

    assert overlap.intersection == 0
    assert overlap.iou == 0.0
    assert overlap.left_coverage == 0.0
    assert overlap.right_coverage == 0.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.zeros(3), np.zeros(3)),
        (np.zeros((2, 3)), np.zeros((3, 2))),
        (np.zeros((2, 3)), np.zeros((1, 2, 3))),
    ],
    ids=("one-dimensional", "unequal-shape", "right-three-dimensional"),
)
def test_mask_overlap_requires_equal_two_dimensional_shapes(
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="two dimensional.*equal shape"):
        mask_overlap(left, right)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "invalid_mask",
    [
        np.asarray([[np.nan, 0.0], [0.0, 1.0]]),
        np.asarray([[np.inf, 0.0], [0.0, 1.0]]),
        np.asarray([[1j, 0j], [0j, 1 + 0j]]),
        np.asarray([[0, 1], [1, 0]], dtype=object),
        np.asarray([[0, 2], [1, 0]], dtype=np.int64),
    ],
    ids=("nan", "infinite", "complex", "object", "non-binary"),
)
def test_mask_overlap_rejects_unsafe_mask_values(
    side: str,
    invalid_mask: np.ndarray,
) -> None:
    masks = {
        "left": np.zeros((2, 2), dtype=bool),
        "right": np.ones((2, 2), dtype=bool),
    }
    masks[side] = invalid_mask

    with pytest.raises(ValueError, match="mask"):
        mask_overlap(masks["left"], masks["right"])


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("boxes_xyxy", np.zeros((1, 4))),
        ("confidences", np.ones(1)),
        ("labels", ("chair",)),
        ("image_features", np.ones((1, 3))),
    ],
)
def test_frontend_batch_rejects_misaligned_vectors(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match="batch dimensions"):
        _batch(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("masks", np.zeros((2, 3)), "masks"),
        ("boxes_xyxy", np.zeros((2, 5)), "boxes_xyxy"),
        ("confidences", np.zeros((2, 1)), "confidences"),
        ("image_features", np.zeros(2), "image_features"),
        ("image_features", np.zeros((2, 0)), "image_features"),
    ],
)
def test_frontend_batch_rejects_invalid_array_shapes(
    field_name: str,
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _batch(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("boxes_xyxy", np.full((2, 4), "1")),
        ("confidences", np.asarray([0.5, object()], dtype=object)),
        ("image_features", np.ones((2, 2), dtype=bool)),
    ],
)
def test_frontend_batch_requires_numeric_arrays(field_name: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match=field_name):
        _batch(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "shape"),
    [
        ("boxes_xyxy", (2, 4)),
        ("confidences", (2,)),
        ("image_features", (2, 2)),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype(np.bool_),
        np.dtype(np.complex64),
        np.dtype(object),
        np.dtype("datetime64[D]"),
        np.dtype("timedelta64[D]"),
    ],
    ids=("bool", "complex", "object", "datetime64", "timedelta64"),
)
def test_frontend_batch_allows_only_integer_unsigned_or_float_arrays(
    field_name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    value = np.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match=field_name):
        _batch(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("boxes_xyxy", np.asarray([[np.nan, 0, 1, 1], [0, 0, 1, 1]])),
        ("confidences", np.asarray([0.5, np.inf])),
        ("image_features", np.asarray([[1.0, 0.0], [np.nan, 1.0]])),
    ],
)
def test_frontend_batch_requires_finite_arrays(field_name: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError, match=field_name):
        _batch(**{field_name: value})


@pytest.mark.parametrize(
    "masks",
    [
        np.asarray([[[np.nan]], [[1.0]]]),
        np.asarray([[[np.inf]], [[1.0]]]),
        np.asarray([[[1j]], [[1 + 0j]]]),
        np.asarray([[[0]], [[1]]], dtype=object),
        np.asarray([[[2]], [[1]]], dtype=np.int64),
    ],
    ids=("nan", "infinite", "complex", "object", "non-binary"),
)
def test_frontend_batch_rejects_unsafe_mask_values(masks: np.ndarray) -> None:
    with pytest.raises(ValueError, match="masks"):
        _batch(masks=masks)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_frontend_batch_rejects_confidences_outside_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError, match=r"confidences.*\[0, 1\]"):
        _batch(confidences=np.asarray([0.5, confidence]))


@pytest.mark.parametrize("labels", [("chair", ""), ("chair", "   "), ("chair", 2)])
def test_frontend_batch_rejects_invalid_labels(labels: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="labels"):
        _batch(labels=labels)


def test_frontend_batch_copies_normalizes_and_locks_arrays() -> None:
    masks = np.ones((2, 2, 3), dtype=np.uint8, order="F")
    boxes = np.arange(8, dtype=np.float64).reshape(2, 4)[:, ::-1]
    confidences = np.asarray([0.9, 0.8], dtype=np.float64)[::-1]
    features = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64, order="F")
    batch = _batch(
        masks=masks,
        boxes_xyxy=boxes,
        confidences=confidences,
        labels=(" chair ", "table"),
        image_features=features,
    )

    masks[...] = 0
    boxes[...] = -1
    confidences[...] = 0
    features[...] = 0

    assert batch.masks.dtype == np.bool_
    assert batch.boxes_xyxy.dtype == np.float32
    assert batch.confidences.dtype == np.float32
    assert batch.image_features.dtype == np.float32
    assert batch.labels == ("chair", "table")
    assert np.all(batch.masks)
    assert np.all(batch.boxes_xyxy >= 0)
    assert np.all(batch.confidences > 0)
    assert np.linalg.norm(batch.image_features, axis=1) == pytest.approx([1.0, 1.0])
    for value in (
        batch.masks,
        batch.boxes_xyxy,
        batch.confidences,
        batch.image_features,
    ):
        assert value.flags.c_contiguous
        assert value.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE"):
            value.flags.writeable = True


def test_frontend_batch_uses_identity_equality_and_hashing() -> None:
    first = _batch()
    second = _batch()

    assert first == first
    assert first != second
    assert isinstance(hash(first), int)
    assert isinstance(hash(second), int)


def test_frontend_batch_accepts_an_empty_batch() -> None:
    batch = FrontendBatch(
        masks=np.empty((0, 4, 5), dtype=bool),
        boxes_xyxy=np.empty((0, 4), dtype=np.float64),
        confidences=np.empty((0,), dtype=np.float64),
        labels=(),
        image_features=np.empty((0, 3), dtype=np.float64),
    )

    assert batch.masks.shape == (0, 4, 5)
    assert batch.boxes_xyxy.shape == (0, 4)
    assert batch.confidences.shape == (0,)
    assert batch.labels == ()
    assert batch.image_features.shape == (0, 3)
    for value in (
        batch.masks,
        batch.boxes_xyxy,
        batch.confidences,
        batch.image_features,
    ):
        assert value.flags.c_contiguous
        assert value.flags.writeable is False


@pytest.mark.parametrize("magnitude", [1e-300, 1e300], ids=("tiny", "large"))
def test_frontend_batch_normalizes_extreme_finite_feature_rows(magnitude: float) -> None:
    batch = _batch(
        image_features=np.asarray([[magnitude, 0.0], [0.0, magnitude]], dtype=np.float64)
    )

    assert np.all(np.isfinite(batch.image_features))
    assert np.all(np.any(batch.image_features != 0.0, axis=1))
    assert np.linalg.norm(batch.image_features, axis=1) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "magnitude",
    [
        np.finfo(np.longdouble).max,
        np.nextafter(np.longdouble(0), np.longdouble(1)),
    ],
    ids=("longdouble-maximum", "longdouble-smallest-positive"),
)
def test_frontend_batch_normalizes_longdouble_extreme_feature_rows(
    magnitude: np.longdouble,
) -> None:
    features = np.asarray([[magnitude, 0], [0, magnitude]], dtype=np.longdouble)

    batch = _batch(image_features=features)

    assert np.all(np.isfinite(batch.image_features))
    assert np.all(np.any(batch.image_features != 0.0, axis=1))
    assert np.linalg.norm(batch.image_features, axis=1) == pytest.approx([1.0, 1.0])


def test_frontend_batch_rejects_zero_feature_rows() -> None:
    with pytest.raises(ValueError, match="image_features.*nonzero"):
        _batch(image_features=np.asarray([[1.0, 0.0], [0.0, 0.0]]))


def test_mask_overlap_contract_normalizes_numpy_scalars() -> None:
    overlap = MaskOverlap(
        intersection=np.int64(2),
        iou=np.float32(0.5),
        left_coverage=np.float32(0.75),
        right_coverage=np.float64(1.0),
    )

    assert type(overlap.intersection) is int
    assert type(overlap.iou) is float
    assert type(overlap.left_coverage) is float
    assert type(overlap.right_coverage) is float


@pytest.mark.parametrize("intersection", [True, -1, 1.5, "1"])
def test_mask_overlap_contract_rejects_invalid_intersection(intersection: object) -> None:
    with pytest.raises(ValueError, match="intersection"):
        MaskOverlap(
            intersection=intersection,
            iou=0.5,
            left_coverage=0.5,
            right_coverage=0.5,
        )


@pytest.mark.parametrize("field_name", ["iou", "left_coverage", "right_coverage"])
@pytest.mark.parametrize("value", [True, np.nan, np.inf, -0.01, 1.01, "0.5"])
def test_mask_overlap_contract_rejects_invalid_ratios(
    field_name: str,
    value: object,
) -> None:
    values = {
        "intersection": 1,
        "iou": 0.5,
        "left_coverage": 0.5,
        "right_coverage": 0.5,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        MaskOverlap(**values)


@pytest.mark.parametrize(
    "variant",
    ["sam_labeled", "yolo_novel_sam", "quota_nms_ensemble"],
)
def test_hybrid_config_accepts_exact_supported_variants(variant: str) -> None:
    assert HybridFrontendConfig(variant=variant).variant == variant


@pytest.mark.parametrize("variant", ["sam", "YOLO_NOVEL_SAM", "", None, ["sam_labeled"]])
def test_hybrid_config_rejects_unsupported_variants(variant: object) -> None:
    with pytest.raises(ValueError, match="variant"):
        HybridFrontendConfig(variant=variant)


@pytest.mark.parametrize(
    "field_name",
    [
        "yolo_match_iou",
        "yolo_match_coverage",
        "novel_iou",
        "duplicate_iou",
        "minimum_area_fraction",
        "maximum_area_fraction",
        "minimum_valid_depth_fraction",
        "minimum_dense_probability",
        "minimum_dense_margin",
        "maximum_structure_probability",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01, True, "0.5"])
def test_hybrid_config_requires_finite_real_unit_interval_thresholds(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        HybridFrontendConfig(variant="yolo_novel_sam", **{field_name: value})


class _TypeErrorFloat(float):
    def __float__(self) -> float:
        raise TypeError("cannot convert")


class _ValueErrorFloat(float):
    def __float__(self) -> float:
        raise ValueError("cannot convert")


@pytest.mark.parametrize(
    "value",
    [10**10000, _TypeErrorFloat(0.5), _ValueErrorFloat(0.5)],
    ids=("overflow", "type-error", "value-error"),
)
def test_hybrid_config_wraps_float_conversion_errors(value: object) -> None:
    with pytest.raises(ValueError, match="novel_iou"):
        HybridFrontendConfig(variant="yolo_novel_sam", novel_iou=value)


def test_hybrid_config_rejects_novel_iou_above_one() -> None:
    with pytest.raises(ValueError, match="novel_iou"):
        HybridFrontendConfig(variant="yolo_novel_sam", novel_iou=1.1)


def test_hybrid_config_rejects_inverted_area_fraction_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_area_fraction.*maximum_area_fraction"):
        HybridFrontendConfig(
            variant="sam_labeled",
            minimum_area_fraction=0.6,
            maximum_area_fraction=0.5,
        )


@pytest.mark.parametrize("field_name", ["maximum_proposals", "maximum_per_class"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_hybrid_config_requires_positive_non_bool_integer_limits(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        HybridFrontendConfig(variant="quota_nms_ensemble", **{field_name: value})


def test_hybrid_config_normalizes_numpy_scalar_values() -> None:
    config = HybridFrontendConfig(
        variant="quota_nms_ensemble",
        novel_iou=np.float32(0.75),
        maximum_proposals=np.int64(32),
    )

    assert type(config.novel_iou) is float
    assert type(config.maximum_proposals) is int
    assert replace(config, maximum_per_class=4).maximum_per_class == 4


def test_dense_mask_aggregation_returns_stride_two_nonstructural_label() -> None:
    dense = _dense_aggregation_fixture()
    mask = _sampled_mask(((0, 0), (0, 1), (1, 1)), dtype=np.dtype(np.uint8))

    result = aggregate_dense_mask(
        mask,
        np.ones((4, 4), dtype=np.float32),
        dense,
        structure_ids={1},
    )

    assert result.semantic_id == 2
    assert result.probability == pytest.approx(0.6)
    assert result.margin == pytest.approx(0.6 - (0.2 / 3.0))
    assert result.structure_probability == pytest.approx(0.8 / 3.0)
    assert result.valid_depth_fraction == 1.0


def test_dense_mask_aggregation_returns_zero_label_for_empty_mask() -> None:
    result = aggregate_dense_mask(
        np.zeros((4, 4), dtype=bool),
        np.ones((4, 4), dtype=bool),
        _dense_aggregation_fixture(),
        structure_ids={1},
    )

    assert result == DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)


def test_dense_mask_aggregation_ignores_pixels_outside_sampling_phase() -> None:
    mask = np.ones((4, 4), dtype=bool)
    mask[::2, ::2] = False

    result = aggregate_dense_mask(
        mask,
        np.ones((4, 4), dtype=bool),
        _dense_aggregation_fixture(),
        structure_ids={1},
    )

    assert result == DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)


def test_dense_mask_aggregation_returns_zero_label_without_valid_depth() -> None:
    result = aggregate_dense_mask(
        _sampled_mask(((0, 0), (0, 1))),
        np.zeros((4, 4), dtype=bool),
        _dense_aggregation_fixture(),
        structure_ids={1},
    )

    assert result == DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)


def test_dense_mask_aggregation_uses_only_valid_selected_samples() -> None:
    valid_depth = _sampled_mask(((0, 0), (1, 0)))

    result = aggregate_dense_mask(
        _sampled_mask(((0, 0), (0, 1), (1, 0), (1, 1))),
        valid_depth,
        _dense_aggregation_fixture(),
        structure_ids={1},
    )

    assert result.semantic_id == 2
    assert result.probability == pytest.approx(0.35)
    assert result.margin == pytest.approx(0.05)
    assert result.structure_probability == pytest.approx(0.35)
    assert result.valid_depth_fraction == 0.5


def test_dense_mask_aggregation_can_label_object_under_structure_dominance() -> None:
    dense = _dense_frame(
        class_ids=np.asarray([[[1, 2, 3]]]),
        probabilities=np.asarray([[[0.7, 0.2, 0.1]]]),
        image_shape=(1, 1),
        sample_stride=1,
        class_count=3,
    )

    result = aggregate_dense_mask(
        np.ones((1, 1), dtype=bool),
        np.ones((1, 1), dtype=bool),
        dense,
        structure_ids={1},
    )

    assert result.semantic_id == 2
    assert result.probability == pytest.approx(0.2)
    assert result.margin == pytest.approx(0.1)
    assert result.structure_probability == pytest.approx(0.7)


def test_dense_mask_aggregation_returns_no_object_for_all_structure_support() -> None:
    dense = _dense_frame(
        class_ids=np.asarray([[[1, 2]]]),
        probabilities=np.asarray([[[0.8, 0.2000005]]]),
        image_shape=(1, 1),
        sample_stride=1,
        class_count=2,
    )

    result = aggregate_dense_mask(
        np.ones((1, 1), dtype=bool),
        np.ones((1, 1), dtype=bool),
        dense,
        structure_ids={1, 2},
    )

    assert result.semantic_id == 0
    assert result.probability == 0.0
    assert result.margin == 0.0
    assert result.structure_probability == 1.0
    assert result.valid_depth_fraction == 1.0


def test_dense_mask_aggregation_accumulates_all_top_k_slots() -> None:
    dense = _dense_frame(
        class_ids=np.asarray([[[2, 5, 1], [3, 5, 1], [4, 5, 1]]]),
        probabilities=np.asarray(
            [[[0.4, 0.35, 0.1], [0.4, 0.35, 0.1], [0.4, 0.35, 0.1]]]
        ),
        image_shape=(1, 3),
        sample_stride=1,
        class_count=5,
    )

    result = aggregate_dense_mask(
        np.ones((1, 3), dtype=bool),
        np.ones((1, 3), dtype=bool),
        dense,
        structure_ids={1},
    )

    assert result.semantic_id == 5
    assert result.probability == pytest.approx(0.35)
    assert result.margin == pytest.approx(0.35 - (0.4 / 3.0))
    assert result.structure_probability == pytest.approx(0.1)


def test_dense_mask_aggregation_breaks_support_ties_by_smaller_class_id() -> None:
    dense = _dense_frame(
        class_ids=np.asarray([[[3, 2]]]),
        probabilities=np.asarray([[[0.4, 0.4]]]),
        image_shape=(1, 1),
        sample_stride=1,
        class_count=3,
    )

    result = aggregate_dense_mask(
        np.ones((1, 1), dtype=bool),
        np.ones((1, 1), dtype=bool),
        dense,
        structure_ids=frozenset(),
    )

    assert result.semantic_id == 2
    assert result.probability == pytest.approx(0.4)
    assert result.margin == 0.0


def test_dense_mask_aggregation_supports_ceiling_divided_sample_shape() -> None:
    class_ids = np.zeros((3, 4, 2), dtype=np.int64)
    class_ids[..., 0] = 1
    class_ids[..., 1] = 2
    probabilities = np.zeros((3, 4, 2), dtype=np.float32)
    probabilities[..., 0] = 0.6
    probabilities[..., 1] = 0.2
    dense = _dense_frame(
        class_ids,
        probabilities,
        image_shape=(5, 7),
        sample_stride=2,
        class_count=2,
    )

    result = aggregate_dense_mask(
        _sampled_mask(((2, 3),), image_shape=(5, 7)),
        np.ones((5, 7), dtype=bool),
        dense,
        structure_ids={1},
    )

    assert result.semantic_id == 2
    assert result.probability == pytest.approx(0.2)
    assert result.margin == pytest.approx(0.2)
    assert result.structure_probability == pytest.approx(0.6)
    assert result.valid_depth_fraction == 1.0


@pytest.mark.parametrize("field_name", ["mask", "valid_depth"])
def test_dense_mask_aggregation_requires_dense_image_shape(field_name: str) -> None:
    values = {
        "mask": np.ones((4, 4), dtype=bool),
        "valid_depth": np.ones((4, 4), dtype=bool),
    }
    values[field_name] = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="image_shape"):
        aggregate_dense_mask(
            values["mask"],
            values["valid_depth"],
            _dense_aggregation_fixture(),
            structure_ids={1},
        )


@pytest.mark.parametrize("field_name", ["mask", "valid_depth"])
@pytest.mark.parametrize(
    "invalid",
    [
        np.full((4, 4), np.nan),
        np.full((4, 4), np.inf),
        np.zeros((4, 4), dtype=np.complex64),
        np.zeros((4, 4), dtype=object),
        np.zeros((4, 4), dtype="datetime64[D]"),
        np.zeros((4, 4), dtype="timedelta64[D]"),
        np.full((4, 4), 2, dtype=np.int64),
    ],
    ids=("nan", "infinite", "complex", "object", "datetime", "timedelta", "nonbinary"),
)
def test_dense_mask_aggregation_rejects_unsafe_binary_arrays(
    field_name: str,
    invalid: np.ndarray,
) -> None:
    values = {
        "mask": np.ones((4, 4), dtype=bool),
        "valid_depth": np.ones((4, 4), dtype=bool),
    }
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        aggregate_dense_mask(
            values["mask"],
            values["valid_depth"],
            _dense_aggregation_fixture(),
            structure_ids={1},
        )


@pytest.mark.parametrize("structure_ids", [[1], (1,), {1: "wall"}])
def test_dense_mask_aggregation_requires_set_structure_ids(
    structure_ids: object,
) -> None:
    with pytest.raises(ValueError, match="structure_ids"):
        aggregate_dense_mask(
            np.ones((4, 4), dtype=bool),
            np.ones((4, 4), dtype=bool),
            _dense_aggregation_fixture(),
            structure_ids=structure_ids,
        )


@pytest.mark.parametrize(
    "structure_id",
    [True, np.bool_(False), 0, -1, 1.0, "1", 5],
    ids=("bool", "numpy-bool", "zero", "negative", "float", "string", "out-of-range"),
)
def test_dense_mask_aggregation_validates_structure_id_elements(
    structure_id: object,
) -> None:
    with pytest.raises(ValueError, match="structure_ids"):
        aggregate_dense_mask(
            np.ones((4, 4), dtype=bool),
            np.ones((4, 4), dtype=bool),
            _dense_aggregation_fixture(),
            structure_ids={structure_id},
        )


def test_dense_mask_aggregation_does_not_modify_input_arrays() -> None:
    dense = _dense_aggregation_fixture()
    mask = _sampled_mask(((0, 0), (1, 1)), dtype=np.dtype(np.uint8))
    valid_depth = np.ones((4, 4), dtype=np.float32)
    inputs = (mask, valid_depth, dense.class_ids, dense.probabilities)
    snapshots = tuple(value.copy() for value in inputs)

    aggregate_dense_mask(mask, valid_depth, dense, structure_ids={np.int64(1)})

    for value, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(value, snapshot)


def test_dense_mask_label_normalizes_numpy_scalars_and_is_frozen() -> None:
    label = DenseMaskLabel(
        semantic_id=np.int64(2),
        probability=np.float32(0.75),
        margin=np.float64(0.25),
        structure_probability=np.float32(0.1),
        valid_depth_fraction=np.float64(0.5),
    )

    assert type(label.semantic_id) is int
    for field_name in (
        "probability",
        "margin",
        "structure_probability",
        "valid_depth_fraction",
    ):
        assert type(getattr(label, field_name)) is float
    with pytest.raises(FrozenInstanceError):
        label.semantic_id = 3


@pytest.mark.parametrize("semantic_id", [True, np.bool_(False), -1, 1.5, "1"])
def test_dense_mask_label_rejects_invalid_semantic_id(semantic_id: object) -> None:
    with pytest.raises(ValueError, match="semantic_id"):
        DenseMaskLabel(semantic_id, 0.5, 0.25, 0.1, 1.0)


@pytest.mark.parametrize(
    "field_name",
    ["probability", "margin", "structure_probability", "valid_depth_fraction"],
)
@pytest.mark.parametrize("value", [True, np.nan, np.inf, -0.01, 1.01, "0.5"])
def test_dense_mask_label_rejects_invalid_ratios(
    field_name: str,
    value: object,
) -> None:
    values = {
        "semantic_id": 2,
        "probability": 0.5,
        "margin": 0.25,
        "structure_probability": 0.1,
        "valid_depth_fraction": 1.0,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        DenseMaskLabel(**values)


def test_dense_mask_label_rejects_margin_above_probability() -> None:
    with pytest.raises(ValueError, match="margin.*probability"):
        DenseMaskLabel(2, 0.4, 0.5, 0.1, 1.0)


def _proposal_batch(
    masks: np.ndarray,
    confidences: tuple[float, ...],
    *,
    labels: tuple[str, ...] | None = None,
    feature_dimension: int = 3,
) -> FrontendBatch:
    masks = np.asarray(masks, dtype=bool)
    proposal_count = masks.shape[0]
    boxes = np.zeros((proposal_count, 4), dtype=np.float64)
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        if rows.size:
            boxes[index] = (
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            )
    features = np.ones((proposal_count, feature_dimension), dtype=np.float64)
    if proposal_count:
        features[:, 0] = np.arange(1, proposal_count + 1, dtype=np.float64)
    return FrontendBatch(
        masks=masks,
        boxes_xyxy=boxes,
        confidences=np.asarray(confidences, dtype=np.float64),
        labels=labels or tuple(f"source-{index}" for index in range(proposal_count)),
        image_features=features,
    )


def _empty_proposal_batch(
    shape: tuple[int, int], *, feature_dimension: int = 3
) -> FrontendBatch:
    return _proposal_batch(
        np.empty((0, *shape), dtype=bool),
        (),
        feature_dimension=feature_dimension,
    )


def _selector_dense(
    primary_ids: np.ndarray,
    primary_probabilities: float | np.ndarray = 0.8,
    *,
    runner_ids: np.ndarray | None = None,
    runner_probabilities: float | np.ndarray | None = None,
    class_count: int = 3,
) -> DenseSemanticFrame:
    primary_ids = np.asarray(primary_ids, dtype=np.int64)
    primary = np.broadcast_to(
        np.asarray(primary_probabilities, dtype=np.float32), primary_ids.shape
    )
    class_ids = primary_ids[..., None]
    probabilities = primary[..., None]
    if runner_ids is not None:
        if runner_probabilities is None:
            raise AssertionError("runner probabilities are required")
        runner_ids = np.asarray(runner_ids, dtype=np.int64)
        runner = np.broadcast_to(
            np.asarray(runner_probabilities, dtype=np.float32), primary_ids.shape
        )
        class_ids = np.stack((primary_ids, runner_ids), axis=-1)
        probabilities = np.stack((primary, runner), axis=-1)
    return _dense_frame(
        class_ids,
        probabilities,
        image_shape=primary_ids.shape,
        sample_stride=1,
        class_count=class_count,
    )


def _select(
    *,
    yolo: FrontendBatch,
    sam: FrontendBatch,
    dense: DenseSemanticFrame,
    config: HybridFrontendConfig,
    valid_depth: np.ndarray | None = None,
    class_names: tuple[str, ...] = ("wall", "chair", "table"),
    structure_ids: set[int] | frozenset[int] = frozenset({1}),
) -> HybridFrameResult:
    return select_hybrid_proposals(
        yolo=yolo,
        sam=sam,
        dense=dense,
        valid_depth=(
            np.ones(dense.image_shape, dtype=bool)
            if valid_depth is None
            else valid_depth
        ),
        class_names=class_names,
        structure_ids=structure_ids,
        config=config,
    )


def test_yolo_novel_sam_keeps_yolo_and_adds_only_novel_sam() -> None:
    shape = (4, 5)
    yolo_masks = np.zeros((2, *shape), dtype=bool)
    yolo_masks[0, :2, :2] = True
    yolo_masks[1, 2:, 3:] = True
    sam_masks = np.zeros((2, *shape), dtype=bool)
    sam_masks[0] = yolo_masks[0]
    sam_masks[1, 2:, :2] = True

    result = _select(
        yolo=_proposal_batch(yolo_masks, (0.9, 0.8), labels=("chair", "table")),
        sam=_proposal_batch(sam_masks, (0.2, 0.2)),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(variant="yolo_novel_sam"),
    )

    assert result.batch.labels == ("chair", "table", "chair")
    assert result.diagnostics["accepted_yolo"] == 2
    assert result.diagnostics["accepted_novel_sam"] == 1
    assert result.diagnostics["rejected_duplicate"] == 1


def test_yolo_novel_sam_rejects_reliable_yolo_matches_below_novel_iou() -> None:
    shape = (2, 8)
    yolo_mask = np.zeros((1, *shape), dtype=bool)
    yolo_mask[0, 0, :4] = True
    sam_mask = np.zeros((1, *shape), dtype=bool)
    sam_mask[0, 0, :2] = True

    result = _select(
        yolo=_proposal_batch(yolo_mask, (0.8,), labels=("chair",)),
        sam=_proposal_batch(sam_mask, (0.1,)),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(variant="yolo_novel_sam"),
    )

    assert result.batch.labels == ("chair",)
    assert result.diagnostics["accepted_inherited_sam"] == 0
    assert result.diagnostics["accepted_novel_sam"] == 0
    assert result.diagnostics["rejected_not_novel"] == 1


def test_sam_labeled_omits_yolo_and_orders_inherited_before_dense_fallback() -> None:
    shape = (3, 5)
    yolo_mask = np.zeros((1, *shape), dtype=bool)
    yolo_mask[0, 0, :4] = True
    sam_masks = np.zeros((2, *shape), dtype=bool)
    sam_masks[0, 0, :2] = True
    sam_masks[1, 2, 2:] = True
    dense_ids = np.full(shape, 2)
    dense_ids[2, 2:] = 3

    result = _select(
        yolo=_proposal_batch(yolo_mask, (0.4,), labels=("chair",)),
        sam=_proposal_batch(sam_masks, (0.1, 0.1)),
        dense=_selector_dense(dense_ids, 0.95),
        config=HybridFrontendConfig(variant="sam_labeled"),
    )

    assert result.batch.labels == ("chair", "table")
    assert result.batch.confidences == pytest.approx((0.4, 0.95))
    assert result.diagnostics["accepted_yolo"] == 0
    assert result.diagnostics["accepted_inherited_sam"] == 1
    assert result.diagnostics["accepted_novel_sam"] == 1


def test_yolo_inheritance_accepts_iou_threshold_and_uses_confidence_formula() -> None:
    shape = (3, 4)
    yolo_mask = np.zeros((1, *shape), dtype=bool)
    sam_mask = np.zeros((1, *shape), dtype=bool)
    yolo_mask[0, 0, :3] = True
    sam_mask[0, 0, 1:4] = True

    result = _select(
        yolo=_proposal_batch(yolo_mask, (0.81,), labels=("chair",)),
        sam=_proposal_batch(sam_mask, (0.1,)),
        dense=_selector_dense(np.full(shape, 3)),
        config=HybridFrontendConfig(
            variant="sam_labeled",
            yolo_match_iou=0.5,
            yolo_match_coverage=0.75,
        ),
    )

    assert result.batch.labels == ("chair",)
    assert result.batch.confidences[0] == pytest.approx(0.81 * math.sqrt(2 / 3))


def test_yolo_inheritance_accepts_directed_coverage_and_breaks_best_tie_by_index() -> None:
    shape = (3, 5)
    yolo_masks = np.zeros((2, *shape), dtype=bool)
    yolo_masks[:, 0, :4] = True
    sam_mask = np.zeros((1, *shape), dtype=bool)
    sam_mask[0, 0, :2] = True

    result = _select(
        yolo=_proposal_batch(
            yolo_masks,
            (0.7, 0.7),
            labels=("lower-index", "higher-index"),
        ),
        sam=_proposal_batch(sam_mask, (0.1,)),
        dense=_selector_dense(np.full(shape, 3)),
        config=HybridFrontendConfig(
            variant="sam_labeled",
            yolo_match_iou=0.75,
            yolo_match_coverage=1.0,
        ),
    )

    assert result.batch.labels == ("lower-index",)
    assert result.batch.confidences[0] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("primary_probability", "runner_probability", "accepted"),
    [(0.6, 0.4, True), (0.59, 0.2, False), (0.59, 0.4, False)],
    ids=("equal-boundaries", "probability-below", "margin-below"),
)
def test_dense_fallback_probability_and_margin_boundaries(
    primary_probability: float,
    runner_probability: float,
    accepted: bool,
) -> None:
    shape = (2, 3)
    sam_mask = np.ones((1, *shape), dtype=bool)
    dense = _selector_dense(
        np.full(shape, 2),
        primary_probability,
        runner_ids=np.full(shape, 3),
        runner_probabilities=runner_probability,
    )

    result = _select(
        yolo=_empty_proposal_batch(shape),
        sam=_proposal_batch(sam_mask, (0.1,)),
        dense=dense,
        config=HybridFrontendConfig(
            variant="sam_labeled",
            minimum_dense_probability=0.6,
            minimum_dense_margin=0.2,
            maximum_area_fraction=1.0,
        ),
    )

    assert len(result.batch.labels) == int(accepted)
    assert result.diagnostics["rejected_unlabeled"] == int(not accepted)


def test_sam_filters_report_area_depth_and_structure_rejections() -> None:
    shape = (4, 5)
    sam_masks = np.zeros((3, *shape), dtype=bool)
    sam_masks[0, 0, 0] = True
    sam_masks[1, 1, :2] = True
    sam_masks[2, 3, :2] = True
    valid_depth = np.ones(shape, dtype=bool)
    valid_depth[1, :2] = False
    dense_ids = np.full(shape, 2)
    dense_ids[3, :2] = 1

    result = _select(
        yolo=_empty_proposal_batch(shape),
        sam=_proposal_batch(sam_masks, (0.1, 0.1, 0.1)),
        dense=_selector_dense(dense_ids),
        valid_depth=valid_depth,
        config=HybridFrontendConfig(
            variant="sam_labeled",
            minimum_area_fraction=0.1,
        ),
    )

    assert result.batch.labels == ()
    assert result.diagnostics["rejected_area"] == 1
    assert result.diagnostics["rejected_depth"] == 1
    assert result.diagnostics["rejected_structure"] == 1


def test_sam_duplicate_nms_rejects_at_equality() -> None:
    shape = (2, 4)
    sam_masks = np.ones((2, *shape), dtype=bool)

    result = _select(
        yolo=_empty_proposal_batch(shape),
        sam=_proposal_batch(sam_masks, (0.1, 0.1)),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(
            variant="sam_labeled",
            duplicate_iou=1.0,
            maximum_area_fraction=1.0,
        ),
    )

    assert result.batch.labels == ("chair",)
    assert result.diagnostics["rejected_duplicate"] == 1


def test_novel_iou_equality_is_not_novel() -> None:
    shape = (2, 4)
    yolo_mask = np.zeros((1, *shape), dtype=bool)
    sam_mask = np.zeros((1, *shape), dtype=bool)
    yolo_mask[0, 0, :3] = True
    sam_mask[0, 0, 1:4] = True

    result = _select(
        yolo=_proposal_batch(yolo_mask, (0.8,), labels=("chair",)),
        sam=_proposal_batch(sam_mask, (0.1,)),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(variant="yolo_novel_sam", novel_iou=0.5),
    )

    assert result.batch.labels == ("chair",)
    assert result.diagnostics["rejected_not_novel"] == 1


def test_quota_ensemble_stably_sorts_yolo_inherited_and_fallback_groups() -> None:
    shape = (5, 8)
    yolo_masks = np.zeros((3, *shape), dtype=bool)
    yolo_masks[0, 0, :4] = True
    yolo_masks[1, 1, :4] = True
    yolo_masks[2, 2, :4] = True
    sam_masks = np.zeros((4, *shape), dtype=bool)
    sam_masks[0, 0, :2] = True
    sam_masks[1, 1, :3] = True
    sam_masks[2, 4, :2] = True
    sam_masks[3, 4, 4:8] = True
    dense_ids = np.full(shape, 2)
    dense_ids[4, :] = 3

    result = _select(
        yolo=_proposal_batch(
            yolo_masks,
            (0.6, 0.9, 0.9),
            labels=("yolo-0", "yolo-1", "yolo-2"),
        ),
        sam=_proposal_batch(sam_masks, (0.1, 0.1, 0.1, 0.1)),
        dense=_selector_dense(dense_ids, 0.8),
        config=HybridFrontendConfig(
            variant="quota_nms_ensemble",
            novel_iou=0.8,
            yolo_match_iou=0.9,
            yolo_match_coverage=1.0,
        ),
    )

    assert result.batch.labels == (
        "yolo-1",
        "yolo-2",
        "yolo-0",
        "yolo-1",
        "yolo-0",
        "table",
        "table",
    )
    np.testing.assert_array_equal(result.batch.masks[3:], sam_masks[[1, 0, 3, 2]])


def test_quota_class_cap_precedes_global_cap() -> None:
    shape = (3, 5)
    yolo_masks = np.zeros((3, *shape), dtype=bool)
    yolo_masks[0, 0, :2] = True
    yolo_masks[1, 1, :2] = True
    yolo_masks[2, 2, :2] = True

    result = _select(
        yolo=_proposal_batch(
            yolo_masks,
            (0.9, 0.8, 0.7),
            labels=("chair", "chair", "table"),
        ),
        sam=_empty_proposal_batch(shape),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(
            variant="quota_nms_ensemble",
            maximum_per_class=1,
            maximum_proposals=1,
        ),
    )

    assert result.batch.labels == ("chair",)
    assert result.diagnostics["accepted_yolo"] == 1
    assert result.diagnostics["rejected_class_cap"] == 1
    assert result.diagnostics["rejected_global_cap"] == 1


def test_yolo_group_is_truncated_by_global_cap() -> None:
    shape = (3, 4)
    yolo_masks = np.zeros((3, *shape), dtype=bool)
    yolo_masks[:, :, 0] = True

    result = _select(
        yolo=_proposal_batch(
            yolo_masks,
            (0.5, 0.9, 0.7),
            labels=("low", "high", "middle"),
        ),
        sam=_empty_proposal_batch(shape),
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(
            variant="yolo_novel_sam",
            maximum_proposals=2,
        ),
    )

    assert result.batch.labels == ("high", "middle")
    assert result.diagnostics["rejected_global_cap"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"yolo": object()}, "yolo"),
        ({"sam": object()}, "sam"),
        ({"dense": object()}, "dense"),
        ({"config": object()}, "config"),
    ],
)
def test_selection_validates_input_types(change: dict[str, object], message: str) -> None:
    shape = (2, 3)
    values = {
        "yolo": _empty_proposal_batch(shape),
        "sam": _empty_proposal_batch(shape),
        "dense": _selector_dense(np.full(shape, 2)),
        "valid_depth": np.ones(shape, dtype=bool),
        "class_names": ("wall", "chair", "table"),
        "structure_ids": {1},
        "config": HybridFrontendConfig(variant="sam_labeled"),
    }
    values.update(change)

    with pytest.raises(ValueError, match=message):
        select_hybrid_proposals(**values)


def test_selection_requires_matching_image_shapes_and_feature_dimensions() -> None:
    shape = (2, 3)
    dense = _selector_dense(np.full(shape, 2))
    config = HybridFrontendConfig(variant="sam_labeled")

    with pytest.raises(ValueError, match="image shape"):
        _select(
            yolo=_empty_proposal_batch(shape),
            sam=_empty_proposal_batch((3, 3)),
            dense=dense,
            config=config,
        )
    with pytest.raises(ValueError, match="feature dimension"):
        _select(
            yolo=_empty_proposal_batch(shape, feature_dimension=2),
            sam=_empty_proposal_batch(shape, feature_dimension=3),
            dense=dense,
            config=config,
        )
    with pytest.raises(ValueError, match="valid_depth"):
        _select(
            yolo=_empty_proposal_batch(shape),
            sam=_empty_proposal_batch(shape),
            dense=dense,
            valid_depth=np.ones((1, 3), dtype=bool),
            config=config,
        )


@pytest.mark.parametrize(
    "class_names",
    [
        ["wall", "chair", "table"],
        ("wall", "chair"),
        ("wall", "", "table"),
        ("wall", "chair", " chair "),
    ],
)
def test_selection_validates_dense_class_names(class_names: object) -> None:
    shape = (2, 3)
    with pytest.raises(ValueError, match="class_names"):
        _select(
            yolo=_empty_proposal_batch(shape),
            sam=_empty_proposal_batch(shape),
            dense=_selector_dense(np.full(shape, 2)),
            config=HybridFrontendConfig(variant="sam_labeled"),
            class_names=class_names,
        )


def test_structure_class_name_cannot_be_emitted_by_dense_fallback() -> None:
    shape = (2, 3)
    result = _select(
        yolo=_empty_proposal_batch(shape),
        sam=_proposal_batch(np.ones((1, *shape), dtype=bool), (0.1,)),
        dense=_selector_dense(np.full(shape, 1)),
        config=HybridFrontendConfig(
            variant="sam_labeled",
            maximum_area_fraction=1.0,
            maximum_structure_probability=1.0,
        ),
    )

    assert result.batch.labels == ()
    assert result.diagnostics["rejected_unlabeled"] == 1


@pytest.mark.parametrize(
    "invalid_depth",
    [
        np.full((2, 3), np.nan),
        np.full((2, 3), 2, dtype=np.int64),
        np.zeros((2, 3), dtype=object),
    ],
)
def test_selection_rejects_unsafe_valid_depth(invalid_depth: np.ndarray) -> None:
    shape = (2, 3)
    with pytest.raises(ValueError, match="valid_depth"):
        _select(
            yolo=_empty_proposal_batch(shape),
            sam=_empty_proposal_batch(shape),
            dense=_selector_dense(np.full(shape, 2)),
            valid_depth=invalid_depth,
            config=HybridFrontendConfig(variant="sam_labeled"),
        )


def test_selection_is_repeatable_immutable_and_does_not_modify_inputs() -> None:
    shape = (2, 4)
    yolo_mask = np.zeros((1, *shape), dtype=bool)
    yolo_mask[0, 0, :2] = True
    sam_mask = np.zeros((1, *shape), dtype=bool)
    sam_mask[0, 1, :2] = True
    yolo = _proposal_batch(yolo_mask, (0.8,), labels=("chair",))
    sam = _proposal_batch(sam_mask, (0.1,))
    dense = _selector_dense(np.full(shape, 2))
    valid_depth = np.ones(shape, dtype=np.uint8)
    snapshots = (
        yolo.masks.copy(),
        sam.masks.copy(),
        dense.class_ids.copy(),
        valid_depth.copy(),
    )
    kwargs = {
        "yolo": yolo,
        "sam": sam,
        "dense": dense,
        "valid_depth": valid_depth,
        "class_names": ("wall", "chair", "table"),
        "structure_ids": {1},
        "config": HybridFrontendConfig(variant="yolo_novel_sam"),
    }

    first = select_hybrid_proposals(**kwargs)
    second = select_hybrid_proposals(**kwargs)

    for first_array, second_array in zip(
        (
            first.batch.masks,
            first.batch.boxes_xyxy,
            first.batch.confidences,
            first.batch.image_features,
        ),
        (
            second.batch.masks,
            second.batch.boxes_xyxy,
            second.batch.confidences,
            second.batch.image_features,
        ),
    ):
        np.testing.assert_array_equal(first_array, second_array)
        assert first_array.flags.writeable is False
    assert first.batch.labels == second.batch.labels
    assert dict(first.diagnostics) == dict(second.diagnostics)
    with pytest.raises(TypeError):
        first.diagnostics["accepted_yolo"] = 99
    for current, snapshot in zip(
        (yolo.masks, sam.masks, dense.class_ids, valid_depth), snapshots
    ):
        np.testing.assert_array_equal(current, snapshot)


@pytest.mark.parametrize(
    ("yolo_count", "sam_count", "expected_count"),
    [(0, 0, 0), (1, 0, 1), (0, 1, 1)],
)
def test_selection_supports_empty_source_batches(
    yolo_count: int, sam_count: int, expected_count: int
) -> None:
    shape = (2, 3)
    mask = np.ones((1, *shape), dtype=bool)
    yolo = (
        _proposal_batch(mask, (0.8,), labels=("chair",))
        if yolo_count
        else _empty_proposal_batch(shape)
    )
    sam = _proposal_batch(mask, (0.1,)) if sam_count else _empty_proposal_batch(shape)

    result = _select(
        yolo=yolo,
        sam=sam,
        dense=_selector_dense(np.full(shape, 2)),
        config=HybridFrontendConfig(
            variant="yolo_novel_sam",
            maximum_area_fraction=1.0,
        ),
    )

    assert result.batch.masks.shape == (expected_count, *shape)
    assert result.batch.image_features.shape == (expected_count, 3)


def test_hybrid_frame_result_copies_validates_and_locks_diagnostics() -> None:
    batch = _empty_proposal_batch((2, 3))
    diagnostics = {"accepted_yolo": np.int64(1)}
    result = HybridFrameResult(batch=batch, diagnostics=diagnostics)
    diagnostics["accepted_yolo"] = 2

    assert result.diagnostics["accepted_yolo"] == 1
    with pytest.raises(TypeError):
        result.diagnostics["accepted_yolo"] = 3

    for invalid in ({"": 0}, {"ok": True}, {"ok": -1}, {"ok": 1.5}):
        with pytest.raises(ValueError, match="diagnostics"):
            HybridFrameResult(batch=batch, diagnostics=invalid)
