from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math
from pathlib import Path
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


def test_load_dense_frame_hashes_and_loads_the_same_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame.npz"
    replacement = tmp_path / "replacement.npz"
    write_dense_frame(path, _frame(source_frame_id=9))
    write_dense_frame(replacement, _frame(source_frame_id=99))
    expected = sha256_file(path)
    original_sha256_stream = dense_semantics._sha256_stream

    def replace_after_hash(stream: object) -> str:
        checksum = original_sha256_stream(stream)  # type: ignore[arg-type]
        dense_semantics.os.replace(replacement, path)
        return checksum

    monkeypatch.setattr(dense_semantics, "_sha256_stream", replace_after_hash)

    restored = load_dense_frame(path, expected_sha256=expected)

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
