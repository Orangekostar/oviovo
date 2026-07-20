from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import io
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

import src.oviv2.dense_semantics as dense_semantics
from src.oviv2.dense_semantics import (
    DenseSemanticFrame,
    DenseSemanticProvenance,
    load_dense_frame,
    sha256_file,
    write_dense_frame,
)


def _provenance_kwargs() -> dict[str, str]:
    return {
        "backend": "  openclip  ",
        "source_commit": "A" * 40,
        "radio_commit": "B" * 40,
        "model_id": "  ViT-H-14  ",
        "model_sha256": "C" * 64,
        "auxiliary_model_sha256": "",
        "vocabulary_sha256": "D" * 64,
        "prompt_sha256": "E" * 64,
        "inference_config_sha256": "F" * 64,
        "cache_prefix_sha256": "0" * 64,
    }


def _frame_inputs(
    *,
    image_shape: tuple[int, int] = (3, 5),
    sample_stride: int = 2,
    class_count: int = 3,
    k: int = 2,
) -> dict[str, object]:
    sampled_shape = (
        math.ceil(image_shape[0] / sample_stride),
        math.ceil(image_shape[1] / sample_stride),
    )
    ids = np.zeros((*sampled_shape, k), dtype=np.int64)
    probabilities = np.zeros((*sampled_shape, k), dtype=np.float32)
    if k:
        ids[..., 0] = 1
        probabilities[..., 0] = 0.7
    if k > 1:
        ids[..., 1] = min(2, class_count)
        probabilities[..., 1] = 0.2
    return {
        "cache_frame_id": 4,
        "source_frame_id": 9,
        "image_shape": image_shape,
        "sample_stride": sample_stride,
        "class_count": class_count,
        "class_ids": ids,
        "probabilities": probabilities,
        "entropy": np.full(sampled_shape, 0.5, dtype=np.float32),
        "margin": np.full(
            sampled_shape,
            0.5 if k > 1 else 0.7,
            dtype=np.float32,
        ),
    }


def _frame(**updates: object) -> DenseSemanticFrame:
    values = _frame_inputs()
    values.update(updates)
    return DenseSemanticFrame(**values)


def _serialized_payload(frame: DenseSemanticFrame) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "cache_frame_id": np.asarray(frame.cache_frame_id, dtype=np.int64),
        "source_frame_id": np.asarray(frame.source_frame_id, dtype=np.int64),
        "image_shape": np.asarray(frame.image_shape, dtype=np.int64),
        "sample_stride": np.asarray(frame.sample_stride, dtype=np.int64),
        "class_count": np.asarray(frame.class_count, dtype=np.int64),
        "class_ids": frame.class_ids,
        "probabilities": frame.probabilities,
        "entropy": frame.entropy,
        "margin": frame.margin,
    }


def _patch_zip_central_uint32(
    path: Path,
    member_name: str,
    field_offset: int,
    value: int,
) -> None:
    content = bytearray(path.read_bytes())
    cursor = 0
    while True:
        header_offset = content.find(b"PK\x01\x02", cursor)
        if header_offset < 0:
            raise AssertionError(f"missing central directory entry {member_name}")
        name_length = struct.unpack_from("<H", content, header_offset + 28)[0]
        extra_length = struct.unpack_from("<H", content, header_offset + 30)[0]
        comment_length = struct.unpack_from("<H", content, header_offset + 32)[0]
        name_start = header_offset + 46
        name = bytes(content[name_start : name_start + name_length]).decode()
        if name == member_name:
            struct.pack_into("<I", content, header_offset + field_offset, value)
            path.write_bytes(content)
            return
        cursor = name_start + name_length + extra_length + comment_length


def _replace_zip_member(path: Path, member_name: str, replacement: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
        }
    members[member_name] = replacement
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _forbid_array_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.load must not run before archive preflight")

    def forbidden_array(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.array must not run before archive preflight")

    def forbidden_read_array(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("NPY payloads must not load before archive preflight")

    monkeypatch.setattr(dense_semantics.np, "load", forbidden_load)
    monkeypatch.setattr(dense_semantics.np, "array", forbidden_array)
    monkeypatch.setattr(dense_semantics.npy_format, "read_array", forbidden_read_array)


def test_provenance_normalizes_text_commits_and_hashes() -> None:
    provenance = DenseSemanticProvenance(**_provenance_kwargs())

    assert provenance.backend == "openclip"
    assert provenance.model_id == "ViT-H-14"
    assert provenance.source_commit == "a" * 40
    assert provenance.radio_commit == "b" * 40
    assert provenance.model_sha256 == "c" * 64
    assert provenance.vocabulary_sha256 == "d" * 64
    assert provenance.prompt_sha256 == "e" * 64
    assert provenance.inference_config_sha256 == "f" * 64
    assert provenance.auxiliary_model_sha256 == ""
    assert all(field.type == "str" for field in fields(provenance))
    with pytest.raises(FrozenInstanceError):
        provenance.backend = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["backend", "model_id"])
@pytest.mark.parametrize("invalid", ["", "   ", None, 7])
def test_provenance_rejects_invalid_required_text(
    field_name: str,
    invalid: object,
) -> None:
    values: dict[str, object] = _provenance_kwargs()
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticProvenance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["source_commit", "radio_commit"])
@pytest.mark.parametrize("invalid", ["a" * 39, "a" * 41, "g" * 40, "", None, 7])
def test_provenance_rejects_invalid_commits(
    field_name: str,
    invalid: object,
) -> None:
    values: dict[str, object] = _provenance_kwargs()
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticProvenance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
        "cache_prefix_sha256",
    ],
)
@pytest.mark.parametrize("invalid", ["a" * 63, "a" * 65, "z" * 64, "", None, 7])
def test_provenance_rejects_invalid_required_hashes(
    field_name: str,
    invalid: object,
) -> None:
    values: dict[str, object] = _provenance_kwargs()
    values[field_name] = invalid

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticProvenance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["a" * 63, "a" * 65, "z" * 64, None, 7])
def test_provenance_rejects_invalid_optional_hash(invalid: object) -> None:
    values: dict[str, object] = _provenance_kwargs()
    values["auxiliary_model_sha256"] = invalid

    with pytest.raises(ValueError, match="auxiliary_model_sha256"):
        DenseSemanticProvenance(**values)  # type: ignore[arg-type]


def test_provenance_normalizes_present_optional_hash() -> None:
    values = _provenance_kwargs()
    values["auxiliary_model_sha256"] = "A" * 64

    assert DenseSemanticProvenance(**values).auxiliary_model_sha256 == "a" * 64


def test_dense_frame_accepts_valid_sampled_top_k_contract() -> None:
    frame = _frame()

    assert frame.cache_frame_id == 4
    assert frame.source_frame_id == 9
    assert frame.image_shape == (3, 5)
    assert frame.class_ids.shape == (2, 3, 2)
    assert frame.class_ids.dtype == np.int64
    assert frame.probabilities.dtype == np.float32
    assert frame.entropy.dtype == np.float32
    assert frame.margin.dtype == np.float32


@pytest.mark.parametrize("field_name", ["cache_frame_id", "source_frame_id"])
@pytest.mark.parametrize("invalid", [-1, True, 1.0, "1", None])
def test_dense_frame_rejects_invalid_non_negative_integer_ids(
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _frame(**{field_name: invalid})


@pytest.mark.parametrize("field_name", ["sample_stride", "class_count"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.0, "1", None])
def test_dense_frame_rejects_invalid_positive_integer_fields(
    field_name: str,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _frame(**{field_name: invalid})


def test_dense_frame_accepts_numpy_integer_scalar_metadata() -> None:
    frame = _frame(
        cache_frame_id=np.int64(1),
        source_frame_id=np.int32(2),
        image_shape=(np.int64(3), np.int32(5)),
        sample_stride=np.int64(2),
        class_count=np.int32(3),
    )

    assert frame.cache_frame_id == 1
    assert frame.source_frame_id == 2
    assert frame.image_shape == (3, 5)


@pytest.mark.parametrize(
    "invalid",
    [
        [3, 5],
        (3,),
        (3, 5, 7),
        (0, 5),
        (-1, 5),
        (3, 0),
        (True, 5),
        (3.0, 5),
        (3, "5"),
    ],
)
def test_dense_frame_rejects_invalid_image_shape(invalid: object) -> None:
    with pytest.raises(ValueError, match="image_shape"):
        _frame(image_shape=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "field_name"),
    [
        ({"class_ids": np.ones((1, 3, 2), dtype=np.int64)}, "class_ids"),
        ({"probabilities": np.ones((2, 2, 2), dtype=np.float32)}, "probabilities"),
        ({"entropy": np.ones((2, 2), dtype=np.float32)}, "entropy"),
        ({"margin": np.ones((3, 3), dtype=np.float32)}, "margin"),
    ],
)
def test_dense_frame_rejects_arrays_not_matching_sampled_shape(
    updates: dict[str, np.ndarray],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _frame(**updates)


@pytest.mark.parametrize("k", [0, 4])
def test_dense_frame_requires_top_k_between_one_and_class_count(k: int) -> None:
    inputs = _frame_inputs(k=max(k, 0))

    with pytest.raises(ValueError, match="class_ids|class_count|top"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize(
    ("field_name", "dtype"),
    [
        ("class_ids", np.int32),
        ("class_ids", np.uint64),
        ("class_ids", np.float64),
        ("class_ids", np.bool_),
        ("probabilities", np.float64),
        ("probabilities", np.float16),
        ("probabilities", np.int64),
        ("entropy", np.float64),
        ("entropy", np.float16),
        ("margin", np.float64),
        ("margin", np.int32),
    ],
)
def test_dense_frame_rejects_wrong_original_array_dtype(
    field_name: str,
    dtype: np.dtype,
) -> None:
    inputs = _frame_inputs()
    inputs[field_name] = np.asarray(inputs[field_name]).astype(dtype)

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize("field_name", ["class_ids", "probabilities", "entropy", "margin"])
def test_dense_frame_requires_ndarrays(field_name: str) -> None:
    inputs = _frame_inputs()
    inputs[field_name] = np.asarray(inputs[field_name]).tolist()

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("probabilities", np.nan),
        ("probabilities", np.inf),
        ("entropy", np.nan),
        ("entropy", -np.inf),
        ("margin", np.nan),
        ("margin", np.inf),
    ],
)
def test_dense_frame_rejects_non_finite_floating_arrays(
    field_name: str,
    invalid: float,
) -> None:
    inputs = _frame_inputs()
    array = np.asarray(inputs[field_name]).copy()
    array.flat[0] = invalid
    inputs[field_name] = array

    with pytest.raises(ValueError, match=field_name):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize("invalid", [-1, 4])
def test_dense_frame_rejects_class_ids_outside_class_range(invalid: int) -> None:
    inputs = _frame_inputs()
    class_ids = np.asarray(inputs["class_ids"]).copy()
    class_ids[0, 0, 0] = invalid
    inputs["class_ids"] = class_ids

    with pytest.raises(ValueError, match="class_ids"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_dense_frame_rejects_probability_outside_unit_interval(invalid: float) -> None:
    inputs = _frame_inputs()
    probabilities = np.asarray(inputs["probabilities"]).copy()
    probabilities[0, 0, 0] = invalid
    inputs["probabilities"] = probabilities

    with pytest.raises(ValueError, match="probabilities"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_rejects_probability_mass_above_one() -> None:
    inputs = _frame_inputs()
    probabilities = np.asarray(inputs["probabilities"]).copy()
    probabilities[0, 0] = (0.7, 0.4)
    inputs["probabilities"] = probabilities
    margin = np.asarray(inputs["margin"]).copy()
    margin[0, 0] = 0.3
    inputs["margin"] = margin

    with pytest.raises(ValueError, match="sum|probabilities"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_allows_probability_mass_within_tolerance() -> None:
    inputs = _frame_inputs()
    probabilities = np.asarray(inputs["probabilities"]).copy()
    probabilities[0, 0] = (0.8, 0.2000005)
    inputs["probabilities"] = probabilities
    margin = np.asarray(inputs["margin"]).copy()
    margin[0, 0] = probabilities[0, 0, 0] - probabilities[0, 0, 1]
    inputs["margin"] = margin

    DenseSemanticFrame(**inputs)


def test_dense_frame_rejects_probabilities_not_in_descending_order() -> None:
    inputs = _frame_inputs()
    probabilities = np.asarray(inputs["probabilities"]).copy()
    probabilities[0, 0] = (0.2, 0.7)
    inputs["probabilities"] = probabilities
    margin = np.asarray(inputs["margin"]).copy()
    margin[0, 0] = 0.0
    inputs["margin"] = margin

    with pytest.raises(ValueError, match="order|probabilities"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_rejects_duplicate_positive_class_ids_per_pixel() -> None:
    inputs = _frame_inputs()
    class_ids = np.asarray(inputs["class_ids"]).copy()
    class_ids[0, 0] = (1, 1)
    inputs["class_ids"] = class_ids

    with pytest.raises(ValueError, match="duplicate|class_ids"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize(
    ("ids", "probabilities"),
    [
        ((0, 2), (0.7, 0.2)),
        ((1, 2), (0.7, 0.0)),
    ],
)
def test_dense_frame_rejects_inconsistent_class_id_probability_pairs(
    ids: tuple[int, int],
    probabilities: tuple[float, float],
) -> None:
    inputs = _frame_inputs()
    class_ids_array = np.asarray(inputs["class_ids"]).copy()
    class_ids_array[0, 0] = ids
    inputs["class_ids"] = class_ids_array
    probability_array = np.asarray(inputs["probabilities"]).copy()
    probability_array[0, 0] = probabilities
    inputs["probabilities"] = probability_array

    with pytest.raises(ValueError, match="class_ids|probabilities"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_accepts_zero_id_only_for_zero_probability() -> None:
    inputs = _frame_inputs(k=3)

    frame = DenseSemanticFrame(**inputs)

    assert np.all(frame.class_ids[..., 2] == 0)
    assert np.all(frame.probabilities[..., 2] == 0.0)


@pytest.mark.parametrize("invalid", [-0.01, math.log(3) + 0.01])
def test_dense_frame_rejects_entropy_outside_class_bound(invalid: float) -> None:
    inputs = _frame_inputs()
    entropy = np.asarray(inputs["entropy"]).copy()
    entropy[0, 0] = invalid
    inputs["entropy"] = entropy

    with pytest.raises(ValueError, match="entropy"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_accepts_float32_representation_of_maximum_entropy() -> None:
    inputs = _frame_inputs()
    inputs["entropy"] = np.full(
        (2, 3),
        np.float32(math.log(3)),
        dtype=np.float32,
    )

    DenseSemanticFrame(**inputs)


def test_dense_frame_accepts_float32_uniform_entropy_accumulation() -> None:
    probabilities = np.full(7, np.float32(1.0 / 7.0), dtype=np.float32)
    uniform_entropy = -np.sum(probabilities * np.log(probabilities))
    inputs = _frame_inputs(class_count=7)
    inputs["entropy"] = np.full((2, 3), uniform_entropy, dtype=np.float32)

    DenseSemanticFrame(**inputs)


def test_dense_frame_class_count_one_requires_zero_entropy() -> None:
    inputs = _frame_inputs(class_count=1, k=1)
    inputs["entropy"] = np.zeros((2, 3), dtype=np.float32)
    DenseSemanticFrame(**inputs)
    entropy = np.zeros((2, 3), dtype=np.float32)
    entropy[0, 0] = np.finfo(np.float32).eps
    inputs["entropy"] = entropy

    with pytest.raises(ValueError, match="entropy"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize("invalid", [-0.01, 1.01, 0.25])
def test_dense_frame_rejects_invalid_or_incorrect_margin(invalid: float) -> None:
    inputs = _frame_inputs()
    margin = np.asarray(inputs["margin"]).copy()
    margin[0, 0] = invalid
    inputs["margin"] = margin

    with pytest.raises(ValueError, match="margin"):
        DenseSemanticFrame(**inputs)


def test_dense_frame_allows_float_tolerance_in_margin() -> None:
    inputs = _frame_inputs()
    margin = np.asarray(inputs["margin"]).copy()
    margin[0, 0] += 5e-7
    inputs["margin"] = margin

    DenseSemanticFrame(**inputs)


def test_dense_frame_top_one_margin_uses_zero_second_probability() -> None:
    inputs = _frame_inputs(k=1)

    frame = DenseSemanticFrame(**inputs)

    assert np.allclose(frame.margin, 0.7)


@pytest.mark.parametrize("invalid", [2**63, np.uint64(2**63)])
def test_dense_frame_rejects_metadata_outside_int64_range(invalid: object) -> None:
    with pytest.raises(ValueError, match="cache_frame_id"):
        _frame(cache_frame_id=invalid)


def test_dense_frame_copies_callers_arrays() -> None:
    inputs = _frame_inputs()
    originals = {
        name: np.asarray(inputs[name])
        for name in ("class_ids", "probabilities", "entropy", "margin")
    }
    frame = DenseSemanticFrame(**inputs)

    originals["class_ids"].fill(0)
    originals["probabilities"].fill(0.0)
    originals["entropy"].fill(0.0)
    originals["margin"].fill(0.0)

    assert np.all(frame.class_ids[..., 0] == 1)
    assert np.allclose(frame.probabilities[..., 0], 0.7)
    assert np.allclose(frame.entropy, 0.5)
    assert np.allclose(frame.margin, 0.5)


def test_dense_frame_validates_private_copy_during_caller_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs()
    caller_probabilities = np.asarray(inputs["probabilities"])
    original_isfinite = dense_semantics.np.isfinite
    mutated = False

    def mutate_caller_after_check(value: object) -> np.ndarray:
        nonlocal mutated
        result = original_isfinite(value)
        if not mutated and isinstance(value, np.ndarray) and value.ndim == 3:
            caller_probabilities.fill(0.0)
            mutated = True
        return result

    monkeypatch.setattr(dense_semantics.np, "isfinite", mutate_caller_after_check)

    frame = DenseSemanticFrame(**inputs)

    assert mutated
    assert not caller_probabilities.any()
    assert frame.probabilities[0, 0].tolist() == pytest.approx([0.7, 0.2])


def test_dense_frame_rejects_resource_budget_before_private_array_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs()
    copied = False

    def forbidden_copy(*_args: object, **_kwargs: object) -> object:
        nonlocal copied
        copied = True
        raise AssertionError("resource budget must run before np.array copy")

    monkeypatch.setattr(dense_semantics, "_MAX_MEMBER_UNCOMPRESSED_BYTES", 1)
    monkeypatch.setattr(dense_semantics.np, "array", forbidden_copy)

    with pytest.raises(ValueError, match="resource|budget|size|limit"):
        DenseSemanticFrame(**inputs)

    assert not copied


def test_dense_frame_revalidates_private_shapes_after_budget_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs()
    original_estimator = dense_semantics._estimate_archive_budget
    reshaped = False

    def reshape_after_budget(arrays: dict[str, np.ndarray]) -> object:
        nonlocal reshaped
        budget = original_estimator(arrays)
        arrays["class_ids"].shape = (6, 2)
        arrays["probabilities"].shape = (6, 2)
        arrays["entropy"].shape = (6,)
        arrays["margin"].shape = (6,)
        reshaped = True
        return budget

    monkeypatch.setattr(
        dense_semantics,
        "_estimate_archive_budget",
        reshape_after_budget,
    )

    with pytest.raises(ValueError, match="class_ids|shape|dimension|snapshot"):
        DenseSemanticFrame(**inputs)

    assert reshaped


def test_dense_frame_rejects_dtype_change_during_budget_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs()
    original_estimator = dense_semantics._estimate_archive_budget

    def change_dtype_after_budget(arrays: dict[str, np.ndarray]) -> object:
        budget = original_estimator(arrays)
        arrays["class_ids"].dtype = np.float64
        arrays["class_ids"][...] = (1.0, 2.0)
        return budget

    monkeypatch.setattr(
        dense_semantics,
        "_estimate_archive_budget",
        change_dtype_after_budget,
    )

    with pytest.raises(ValueError, match="class_ids|dtype|snapshot"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_dense_frame_wraps_private_copy_race_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    inputs = _frame_inputs()

    def fail_copy(*_args: object, **_kwargs: object) -> object:
        raise error_type("injected concurrent resize")

    monkeypatch.setattr(dense_semantics.np, "array", fail_copy)

    with pytest.raises(ValueError, match="class_ids|snapshot|copy"):
        DenseSemanticFrame(**inputs)


@pytest.mark.parametrize(
    ("limit_name", "budget_field"),
    [
        ("_MAX_MEMBER_UNCOMPRESSED_BYTES", "max_member_bytes"),
        ("_MAX_TOTAL_UNCOMPRESSED_BYTES", "total_uncompressed_bytes"),
        ("_MAX_ARCHIVE_BYTES", "archive_bytes"),
    ],
)
def test_dense_frame_rejects_one_byte_over_each_resource_budget(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    budget_field: str,
) -> None:
    inputs = _frame_inputs()
    raw_arrays = {
        name: np.asarray(inputs[name])
        for name in ("class_ids", "probabilities", "entropy", "margin")
    }
    budget = dense_semantics._estimate_archive_budget(raw_arrays)
    monkeypatch.setattr(
        dense_semantics,
        limit_name,
        getattr(budget, budget_field) - 1,
    )

    with pytest.raises(ValueError, match="resource|budget|size|limit"):
        DenseSemanticFrame(**inputs)


def test_frame_at_conservative_resource_bound_roundtrips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs()
    raw_arrays = {
        name: np.asarray(inputs[name])
        for name in ("class_ids", "probabilities", "entropy", "margin")
    }
    budget = dense_semantics._estimate_archive_budget(raw_arrays)
    monkeypatch.setattr(
        dense_semantics,
        "_MAX_MEMBER_UNCOMPRESSED_BYTES",
        budget.max_member_bytes,
    )
    monkeypatch.setattr(
        dense_semantics,
        "_MAX_TOTAL_UNCOMPRESSED_BYTES",
        budget.total_uncompressed_bytes,
    )
    monkeypatch.setattr(
        dense_semantics,
        "_MAX_ARCHIVE_BYTES",
        budget.archive_bytes,
    )
    path = tmp_path / "bounded.npz"

    frame = DenseSemanticFrame(**inputs)
    write_dense_frame(path, frame)
    restored = load_dense_frame(path)

    assert restored.source_frame_id == frame.source_frame_id


@pytest.mark.parametrize(
    "frame_inputs",
    [
        _frame_inputs(),
        _frame_inputs(image_shape=(1, 1), sample_stride=1, class_count=1, k=1),
        _frame_inputs(image_shape=(8, 9), sample_stride=4, class_count=4, k=3),
    ],
)
def test_archive_budget_conservatively_covers_all_payload_members(
    frame_inputs: dict[str, object],
) -> None:
    if frame_inputs["class_count"] == 1:
        frame_inputs["entropy"] = np.zeros((1, 1), dtype=np.float32)
    frame = DenseSemanticFrame(**frame_inputs)
    payload = dense_semantics._archive_payload(frame)
    budget = dense_semantics._estimate_archive_budget(
        {
            name: payload[name]
            for name in ("class_ids", "probabilities", "entropy", "margin")
        }
    )
    member_bounds = dict(budget.member_bytes)
    actual_member_sizes: dict[str, int] = {}
    for name, array in payload.items():
        stream = io.BytesIO()
        np.lib.format.write_array(stream, array, allow_pickle=False)
        actual_member_sizes[name] = len(stream.getvalue())
        assert actual_member_sizes[name] <= member_bounds[name]
    archive_stream = io.BytesIO()
    np.savez_compressed(archive_stream, **payload)

    assert max(actual_member_sizes.values()) <= budget.max_member_bytes
    assert sum(actual_member_sizes.values()) <= budget.total_uncompressed_bytes
    assert len(archive_stream.getvalue()) <= budget.archive_bytes


def test_generated_zero_archive_is_not_rejected_by_ratio_heuristic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs(class_count=1, k=1)
    inputs["class_ids"] = np.zeros((2, 3, 1), dtype=np.int64)
    inputs["probabilities"] = np.zeros((2, 3, 1), dtype=np.float32)
    inputs["entropy"] = np.zeros((2, 3), dtype=np.float32)
    inputs["margin"] = np.zeros((2, 3), dtype=np.float32)
    frame = DenseSemanticFrame(**inputs)
    path = tmp_path / "zeros.npz"
    monkeypatch.setattr(
        dense_semantics,
        "_MAX_COMPRESSION_RATIO",
        0.5,
        raising=False,
    )

    write_dense_frame(path, frame)

    assert load_dense_frame(path).class_ids.shape == frame.class_ids.shape


def test_load_preflights_conservative_contract_budget_before_large_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _frame_inputs(class_count=1, k=1)
    inputs["class_ids"] = np.zeros((2, 3, 1), dtype=np.int64)
    inputs["probabilities"] = np.zeros((2, 3, 1), dtype=np.float32)
    inputs["entropy"] = np.zeros((2, 3), dtype=np.float32)
    inputs["margin"] = np.zeros((2, 3), dtype=np.float32)
    frame = DenseSemanticFrame(**inputs)
    path = tmp_path / "compressed.npz"
    write_dense_frame(path, frame)
    budget = dense_semantics._estimate_archive_budget(
        {
            name: getattr(frame, name)
            for name in ("class_ids", "probabilities", "entropy", "margin")
        }
    )
    archive_size = path.stat().st_size
    constrained_limit = (archive_size + budget.archive_bytes) // 2
    assert archive_size < constrained_limit < budget.archive_bytes
    monkeypatch.setattr(
        dense_semantics,
        "_MAX_ARCHIVE_BYTES",
        constrained_limit,
    )
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("large payload arrays must not load before budget preflight")

    monkeypatch.setattr(dense_semantics.np, "load", forbidden_load)

    with pytest.raises(ValueError, match="archive|budget|resource|limit"):
        load_dense_frame(path)

    assert not called


@pytest.mark.parametrize("field_name", ["class_ids", "probabilities", "entropy", "margin"])
def test_dense_frame_arrays_have_irrecoverable_read_only_backing(field_name: str) -> None:
    array = getattr(_frame(), field_name)

    assert not array.flags.writeable
    assert isinstance(array.base, np.ndarray)
    assert isinstance(array.base.base, bytes)
    with pytest.raises(ValueError):
        array.flags.writeable = True
    with pytest.raises(ValueError):
        array.flat[0] = array.flat[0]


def test_sha256_file_streams_the_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"dense semantic cache" * 100_000)

    assert sha256_file(path) == "cdfac1722a5ecf324911e913e2471ed749870a6998aff9e73603f74d5ae0a266"


def test_dense_frame_atomic_roundtrip_and_storage_dtypes(tmp_path: Path) -> None:
    frame = _frame()
    path = tmp_path / "nested" / "frame.npz"

    write_dense_frame(path, frame)
    restored = load_dense_frame(path)

    assert restored.cache_frame_id == frame.cache_frame_id
    assert restored.source_frame_id == frame.source_frame_id
    assert restored.image_shape == frame.image_shape
    for field_name in ("class_ids", "probabilities", "entropy", "margin"):
        np.testing.assert_array_equal(getattr(restored, field_name), getattr(frame, field_name))
        assert not getattr(restored, field_name).flags.writeable
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "cache_frame_id",
            "source_frame_id",
            "image_shape",
            "sample_stride",
            "class_count",
            "class_ids",
            "probabilities",
            "entropy",
            "margin",
        }
        for field_name in (
            "schema_version",
            "cache_frame_id",
            "source_frame_id",
            "image_shape",
            "sample_stride",
            "class_count",
            "class_ids",
        ):
            assert archive[field_name].dtype == np.int64
        for field_name in ("probabilities", "entropy", "margin"):
            assert archive[field_name].dtype == np.float32
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_load_dense_frame_uses_public_numpy_header_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    original_reader = dense_semantics.npy_format._read_array_header
    original_payload_reader = dense_semantics.npy_format.read_array

    def forbidden_private_reader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("production code must not call private NumPy header APIs")

    def public_reader_v1(stream: object, max_header_size: int) -> object:
        return original_reader(stream, (1, 0), max_header_size=max_header_size)

    def public_reader_v2(stream: object, max_header_size: int) -> object:
        return original_reader(stream, (2, 0), max_header_size=max_header_size)

    def public_payload_reader(*args: object, **kwargs: object) -> object:
        monkeypatch.setattr(
            dense_semantics.npy_format,
            "_read_array_header",
            original_reader,
        )
        try:
            return original_payload_reader(*args, **kwargs)
        finally:
            monkeypatch.setattr(
                dense_semantics.npy_format,
                "_read_array_header",
                forbidden_private_reader,
            )

    monkeypatch.setattr(
        dense_semantics.npy_format,
        "_read_array_header",
        forbidden_private_reader,
    )
    monkeypatch.setattr(
        dense_semantics.npy_format,
        "read_array_header_1_0",
        public_reader_v1,
    )
    monkeypatch.setattr(
        dense_semantics.npy_format,
        "read_array_header_2_0",
        public_reader_v2,
    )
    monkeypatch.setattr(
        dense_semantics.npy_format,
        "read_array",
        public_payload_reader,
    )

    assert load_dense_frame(path).source_frame_id == 9


def test_load_dense_frame_checks_checksum_before_opening_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    expected = sha256_file(path)
    assert load_dense_frame(path, expected_sha256=expected.upper()).cache_frame_id == 4
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not be called after a checksum mismatch")

    monkeypatch.setattr(dense_semantics.np, "load", forbidden_load)

    with pytest.raises(ValueError, match="checksum"):
        load_dense_frame(path, expected_sha256="0" * 64)
    assert not called


def test_load_dense_frame_hashes_and_loads_an_immutable_content_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    replacement = tmp_path / "replacement.npz"
    write_dense_frame(path, _frame(source_frame_id=9))
    write_dense_frame(replacement, _frame(source_frame_id=99))
    expected = sha256_file(path)
    replacement_content = replacement.read_bytes()
    assert len(replacement_content) == path.stat().st_size
    original_open = Path.open
    mutated = False

    class MutatingStream:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def read(self, size: int = -1) -> bytes:
            nonlocal mutated
            data = self._stream.read(size)  # type: ignore[attr-defined]
            if data and not mutated:
                mutated = True
                with original_open(path, "wb") as replacement_stream:
                    replacement_stream.write(replacement_content)
            return data

        def __enter__(self) -> "MutatingStream":
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._stream, name)

    def open_with_mutation(
        candidate: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        stream = original_open(candidate, mode, *args, **kwargs)
        if candidate == path and mode == "rb":
            return MutatingStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_with_mutation)

    restored = load_dense_frame(path, expected_sha256=expected)

    assert mutated
    assert restored.source_frame_id == 9


@pytest.mark.parametrize("invalid", ["", "a" * 63, "a" * 65, "z" * 64, 7])
def test_load_dense_frame_rejects_invalid_expected_checksum(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())

    with pytest.raises(ValueError, match="checksum"):
        load_dense_frame(path, expected_sha256=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_load_dense_frame_rejects_non_exact_key_set(tmp_path: Path, change: str) -> None:
    path = tmp_path / "frame.npz"
    payload = _serialized_payload(_frame())
    if change == "missing":
        payload.pop("entropy")
    else:
        payload["unexpected"] = np.asarray(1, dtype=np.int64)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="keys"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    with zipfile.ZipFile(path) as archive:
        duplicate_payload = archive.read("schema_version.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr("schema_version.npy", duplicate_payload)

    with pytest.raises(ValueError, match="keys"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_many_duplicate_members_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    with zipfile.ZipFile(path) as archive:
        duplicate_payload = archive.read("schema_version.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, mode="a") as archive:
            for _ in range(64):
                archive.writestr("schema_version.npy", duplicate_payload)
    _forbid_array_loading(monkeypatch)

    def forbidden_zipfile(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ZipFile must not parse an unbounded central directory")

    monkeypatch.setattr(dense_semantics.zipfile, "ZipFile", forbidden_zipfile)

    with pytest.raises(ValueError, match="keys|duplicate"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_forged_high_compression_ratio_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    _patch_zip_central_uint32(path, "class_ids.npy", 20, 0)
    _forbid_array_loading(monkeypatch)

    with pytest.raises(ValueError, match="compression|ratio|resource"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_oversized_member_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    _patch_zip_central_uint32(
        path,
        "class_ids.npy",
        24,
        512 * 1024 * 1024 + 1,
    )
    _forbid_array_loading(monkeypatch)

    with pytest.raises(ValueError, match="class_ids|member|size|resource"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_forged_npy_shape_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    forged_header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        forged_header,
        {
            "descr": np.dtype(np.int64).str,
            "fortran_order": False,
            "shape": (500_000_000, 500_000_000, 1),
        },
    )
    _replace_zip_member(path, "class_ids.npy", forged_header.getvalue())
    _forbid_array_loading(monkeypatch)

    with pytest.raises(ValueError, match="class_ids|shape|size|resource"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_archive_over_byte_limit_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    monkeypatch.setattr(dense_semantics, "_MAX_ARCHIVE_BYTES", path.stat().st_size - 1)
    _forbid_array_loading(monkeypatch)

    with pytest.raises(ValueError, match="archive|size|limit"):
        load_dense_frame(path)


def test_dense_frame_roundtrip_allows_small_zero_filled_arrays(tmp_path: Path) -> None:
    inputs = _frame_inputs(class_count=1, k=1)
    inputs["class_ids"] = np.zeros((2, 3, 1), dtype=np.int64)
    inputs["probabilities"] = np.zeros((2, 3, 1), dtype=np.float32)
    inputs["entropy"] = np.zeros((2, 3), dtype=np.float32)
    inputs["margin"] = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "zeros.npz"

    write_dense_frame(path, DenseSemanticFrame(**inputs))
    restored = load_dense_frame(path)

    assert not restored.class_ids.any()
    assert not restored.probabilities.any()


@pytest.mark.parametrize(
    "schema",
    [
        np.asarray(2, dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray(1, dtype=np.int32),
        np.asarray(True),
    ],
)
def test_load_dense_frame_rejects_invalid_schema(
    tmp_path: Path,
    schema: np.ndarray,
) -> None:
    path = tmp_path / "frame.npz"
    payload = _serialized_payload(_frame())
    payload["schema_version"] = schema
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="schema"):
        load_dense_frame(path)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("cache_frame_id", np.asarray([4], dtype=np.int64)),
        ("source_frame_id", np.asarray(9, dtype=np.int32)),
        ("sample_stride", np.asarray(True)),
        ("class_count", np.asarray(3.0, dtype=np.float64)),
        ("image_shape", np.asarray([3], dtype=np.int64)),
        ("image_shape", np.asarray([3, 5], dtype=np.int32)),
    ],
)
def test_load_dense_frame_rejects_wrong_metadata_shape_or_dtype(
    tmp_path: Path,
    field_name: str,
    invalid: np.ndarray,
) -> None:
    path = tmp_path / "frame.npz"
    payload = _serialized_payload(_frame())
    payload[field_name] = invalid
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match=field_name):
        load_dense_frame(path)


def test_load_dense_frame_rejects_corrupt_zip_with_context(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    path.write_bytes(b"not an npz file")

    with pytest.raises(ValueError, match="NPZ|dense frame|corrupt"):
        load_dense_frame(path)


def test_load_dense_frame_rejects_bad_zip_member_crc_with_context(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame())
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("class_ids.npy")
    content = bytearray(path.read_bytes())
    name_length = int.from_bytes(
        content[member.header_offset + 26 : member.header_offset + 28],
        "little",
    )
    extra_length = int.from_bytes(
        content[member.header_offset + 28 : member.header_offset + 30],
        "little",
    )
    payload_offset = member.header_offset + 30 + name_length + extra_length
    content[payload_offset + member.compress_size // 2] ^= 0xFF
    path.write_bytes(content)

    with pytest.raises(ValueError, match="NPZ|dense frame|corrupt"):
        load_dense_frame(path)


@pytest.mark.parametrize("operation", ["write", "load"])
def test_dense_frame_io_rejects_wrong_suffix(tmp_path: Path, operation: str) -> None:
    path = tmp_path / "frame.npy"

    with pytest.raises(ValueError, match=".npz"):
        if operation == "write":
            write_dense_frame(path, _frame())
        else:
            load_dense_frame(path)


def test_load_dense_frame_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exist|file"):
        load_dense_frame(tmp_path / "missing.npz")


def test_write_dense_frame_requires_frame_instance(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="DenseSemanticFrame"):
        write_dense_frame(tmp_path / "frame.npz", object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "limit_name",
    [
        "_MAX_ARCHIVE_BYTES",
        "_MAX_MEMBER_UNCOMPRESSED_BYTES",
        "_MAX_TOTAL_UNCOMPRESSED_BYTES",
    ],
)
def test_write_dense_frame_never_publishes_archive_exceeding_loader_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    path = tmp_path / "frame.npz"
    monkeypatch.setattr(dense_semantics, limit_name, 1)

    with pytest.raises(ValueError, match="size|resource|limit"):
        write_dense_frame(path, _frame())

    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_atomic_replace_failure_preserves_target_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    old_content = b"old valid target sentinel"
    path.write_bytes(old_content)

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(dense_semantics.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        write_dense_frame(path, _frame())

    assert path.read_bytes() == old_content
    assert [candidate.name for candidate in tmp_path.iterdir()] == ["frame.npz"]


def test_write_dense_frame_fsyncs_parent_directory_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    events: list[object] = []
    original_replace = dense_semantics.os.replace

    def tracked_replace(source: object, destination: object) -> None:
        events.append("replace")
        original_replace(source, destination)

    def tracked_directory_fsync(directory: Path) -> None:
        events.append(("directory_fsync", directory))

    monkeypatch.setattr(dense_semantics.os, "replace", tracked_replace)
    monkeypatch.setattr(
        dense_semantics,
        "_fsync_directory",
        tracked_directory_fsync,
        raising=False,
    )

    write_dense_frame(path, _frame())

    assert events == ["replace", ("directory_fsync", tmp_path)]


def test_parent_fsync_failure_reports_potentially_visible_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    write_dense_frame(path, _frame(source_frame_id=9))

    def fail_directory_fsync(_directory: Path) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(
        dense_semantics,
        "_fsync_directory",
        fail_directory_fsync,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="publication.*may already be visible"):
        write_dense_frame(path, _frame(source_frame_id=99))

    assert load_dense_frame(path).source_frame_id == 99
    assert [candidate.name for candidate in tmp_path.iterdir()] == ["frame.npz"]
