from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.oviv2.hybrid_frontend import (
    FrontendBatch,
    HybridFrontendConfig,
    MaskOverlap,
    mask_overlap,
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
