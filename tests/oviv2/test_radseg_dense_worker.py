from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.oviv2.dense_semantics import DenseSemanticFrame


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from radseg_dense_worker import (  # noqa: E402
    RADIO_COMMIT,
    RADSEG_COMMIT,
    RAYFRONTS_COMMIT,
    CheckpointTracker,
    DenseWorker,
    NARadioRuntime,
    RadsegRuntime,
    SourceSpec,
    _instantiate_naradio,
    _instantiate_radseg,
    canonical_sha256,
    decode_rgb,
    encode_array,
    load_frozen_classes,
    local_radio_hub,
    parse_args,
    reduce_probabilities,
    resolve_model_checkpoint,
    run_request,
    serve_jsonl,
    sha256_file,
    validate_cli_args,
    validate_source_checkout,
)


def _array_block(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "encoding": "base64",
        "dtype": contiguous.dtype.name,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode_array(block: dict[str, Any]) -> np.ndarray:
    data = base64.b64decode(block["data"], validate=True)
    return np.frombuffer(data, dtype=np.dtype(block["dtype"])).reshape(block["shape"])


def _classes() -> list[str]:
    return [f"class-{index:02d}" for index in range(41)]


def _provenance() -> dict[str, str]:
    return {
        "backend": "radseg",
        "source_commit": RADSEG_COMMIT,
        "radio_commit": RADIO_COMMIT,
        "model_id": f"radseg:test:test:sam=0:sha256={'a' * 64}",
        "model_sha256": "a" * 64,
        "auxiliary_model_sha256": "",
        "vocabulary_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "inference_config_sha256": "d" * 64,
    }


def test_reduce_probabilities_emits_normalized_one_based_topk_and_uncertainty() -> None:
    probabilities = np.asarray(
        [[[[0.1, 0.7], [0.4, 0.2]], [[0.9, 0.3], [0.6, 0.8]]]],
        dtype=np.float32,
    )

    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=2)

    assert reduced["class_ids"].tolist() == [
        [[2, 1], [1, 2]],
        [[2, 1], [2, 1]],
    ]
    np.testing.assert_allclose(reduced["probabilities"][..., 0], [[0.9, 0.7], [0.6, 0.8]])
    np.testing.assert_allclose(reduced["margin"], [[0.8, 0.4], [0.2, 0.6]])
    assert np.all(reduced["entropy"] >= 0.0)
    assert reduced["class_ids"].dtype == np.int64
    for name in ("probabilities", "entropy", "margin"):
        assert reduced[name].dtype == np.float32
        assert reduced[name].flags.c_contiguous


def test_reduce_probabilities_preserves_topk_and_normalizes_only_for_entropy() -> None:
    probabilities = np.asarray([[[[0.4]], [[0.2]], [[0.2]]]], dtype=np.float32)

    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=2)

    assert reduced["class_ids"].tolist() == [[[1, 2]]]
    np.testing.assert_allclose(reduced["probabilities"], [[[0.4, 0.2]]])
    np.testing.assert_allclose(reduced["margin"], [[0.2]])
    np.testing.assert_allclose(reduced["entropy"], [[1.0397208]], rtol=1e-6)


def test_reduce_probabilities_uses_stable_class_id_tie_break() -> None:
    probabilities = np.full((1, 4, 1, 1), 0.25, dtype=np.float32)
    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=4)
    assert reduced["class_ids"].tolist() == [[[1, 2, 3, 4]]]


def test_reduce_probabilities_samples_from_origin() -> None:
    probabilities = np.zeros((1, 2, 3, 5), dtype=np.float32)
    probabilities[:, 0] = 1.0
    reduced = reduce_probabilities(probabilities, sample_stride=2, top_k=1)
    assert reduced["class_ids"].shape == (2, 3, 1)
    assert reduced["entropy"].shape == (2, 3)


def test_reduce_probabilities_rejects_zero_mass_without_log_warning() -> None:
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="probability mass"):
            reduce_probabilities(
                np.zeros((1, 3, 2, 2), dtype=np.float32),
                sample_stride=1,
                top_k=2,
            )


@pytest.mark.parametrize(
    ("probabilities", "stride", "top_k", "message"),
    [
        (np.zeros((2, 3, 4), dtype=np.float32), 1, 1, "shape"),
        (np.zeros((2, 3, 4, 5), dtype=np.float32), 1, 1, "shape"),
        (np.zeros((1, 0, 4, 5), dtype=np.float32), 1, 1, "non-empty"),
        (np.full((1, 2, 1, 1), np.nan, dtype=np.float32), 1, 1, "finite"),
        (np.asarray([[[[-1.0]], [[2.0]]]], dtype=np.float32), 1, 1, "non-negative"),
        (np.zeros((1, 2, 1, 1), dtype=np.float32), 0, 1, "sample_stride"),
        (np.zeros((1, 2, 1, 1), dtype=np.float32), True, 1, "sample_stride"),
        (np.zeros((1, 2, 1, 1), dtype=np.float32), 1, 0, "top_k"),
        (np.zeros((1, 2, 1, 1), dtype=np.float32), 1, 3, "class count"),
    ],
)
def test_reduce_probabilities_rejects_invalid_inputs(
    probabilities: np.ndarray,
    stride: object,
    top_k: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reduce_probabilities(probabilities, sample_stride=stride, top_k=top_k)  # type: ignore[arg-type]


def test_decode_rgb_round_trips_c_contiguous_uint8() -> None:
    source = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    decoded = decode_rgb(_array_block(source))
    np.testing.assert_array_equal(decoded, source)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda block: block.update(dtype="float32"), "dtype"),
        (lambda block: block.update(encoding="hex"), "encoding"),
        (lambda block: block.update(shape=[2, 2]), "shape"),
        (lambda block: block.update(shape=[2, 2, 4]), "shape"),
        (lambda block: block.update(shape=[True, 2, 3]), "shape"),
        (lambda block: block.update(shape=[0, 2, 3]), "shape"),
        (lambda block: block.update(data="not base64"), "base64"),
        (lambda block: block.update(data="AA=="), "byte count"),
        (lambda block: block.update(extra="unexpected"), "keys"),
    ],
)
def test_decode_rgb_strictly_rejects_invalid_blocks(mutate: Any, message: str) -> None:
    block = _array_block(np.zeros((2, 2, 3), dtype=np.uint8))
    mutate(block)
    with pytest.raises(ValueError, match=message):
        decode_rgb(block)


def test_decode_rgb_enforces_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import radseg_dense_worker as worker_module

    monkeypatch.setattr(worker_module, "MAX_RGB_BYTES", 11)
    with pytest.raises(ValueError, match="size limit"):
        decode_rgb(_array_block(np.zeros((2, 2, 3), dtype=np.uint8)))


def test_encode_array_copies_noncontiguous_input_with_explicit_contract() -> None:
    source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    assert not source.flags.c_contiguous
    block = encode_array(source)
    assert block.keys() == {"encoding", "dtype", "shape", "data"}
    assert block["encoding"] == "base64"
    assert block["dtype"] == "float32"
    assert block["shape"] == [4, 3]
    np.testing.assert_array_equal(_decode_array(block), source)


def test_canonical_sha256_is_key_order_independent_and_value_sensitive() -> None:
    left = canonical_sha256({"b": [2, 3], "a": 1})
    right = canonical_sha256({"a": 1, "b": [2, 3]})
    changed = canonical_sha256({"a": 1, "b": [3, 2]})
    assert left == right
    assert left != changed
    assert len(left) == 64


def test_load_frozen_classes_accepts_replica_contract_and_hashes_raw_file(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    payload = {"classes": _classes(), "aliases": {"seat": "class-00"}}
    raw = json.dumps(payload, indent=2).encode("utf-8")
    path.write_bytes(raw)

    classes, digest = load_frozen_classes(path)

    assert classes == _classes()
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"classes": _classes()[:-1]}, "exactly 41"),
        ({"classes": _classes()[:-1] + [""]}, "non-empty"),
        ({"classes": _classes()[:-1] + [" padded "]}, "whitespace"),
        ({"classes": _classes()[:-1] + ["CLASS-00"]}, "unique"),
        ({"classes": "not-a-list"}, "list"),
        ({"classes": _classes(), "unexpected": 1}, "keys"),
        ({"classes": _classes(), "aliases": []}, "aliases"),
    ],
)
def test_load_frozen_classes_rejects_invalid_contract(
    tmp_path: Path,
    payload: dict[str, Any],
    message: str,
) -> None:
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_frozen_classes(path)


def test_load_frozen_classes_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    path.write_text('{"classes": [], "classes": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_frozen_classes(path)


def test_load_frozen_classes_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import radseg_dense_worker as worker_module

    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": _classes()}), encoding="utf-8")
    monkeypatch.setattr(worker_module, "MAX_CLASSES_JSON_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        load_frozen_classes(path)


class _FakeHub:
    def __init__(self, root: Path | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._root = root

    def load(self, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self.calls.append((args, kwargs))
        return args, kwargs

    def get_dir(self) -> str:
        assert self._root is not None
        return str(self._root)

    def load_state_dict_from_url(self, url: str, *args: Any, **kwargs: Any) -> object:
        self.calls.append(((url, *args), kwargs))
        return object()


def test_local_radio_hub_forces_pinned_local_source_and_preserves_kwargs(tmp_path: Path) -> None:
    hub = _FakeHub()
    torch_module = SimpleNamespace(hub=hub)
    original = hub.load

    with local_radio_hub(torch_module, tmp_path):
        result = torch_module.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="c-radio_v3-b",
            source="github",
            progress=False,
            custom=7,
        )

    args, kwargs = result
    assert args[:2] == (str(tmp_path.resolve()), "radio_model")
    assert kwargs == {
        "version": "c-radio_v3-b",
        "source": "local",
        "progress": False,
        "custom": 7,
    }
    assert torch_module.hub.load == original


def test_local_radio_hub_blocks_every_other_repository_and_restores(tmp_path: Path) -> None:
    hub = _FakeHub()
    torch_module = SimpleNamespace(hub=hub)
    original = hub.load

    with pytest.raises(RuntimeError, match="prohibited"):
        with local_radio_hub(torch_module, tmp_path):
            torch_module.hub.load("some/remote-repository", "model")

    assert torch_module.hub.load == original
    assert hub.calls == []


def test_checkpoint_tracker_captures_the_exact_url_cache_path(tmp_path: Path) -> None:
    hub = _FakeHub(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    expected = checkpoints / "weights.pt"
    expected.write_bytes(b"weights")

    with CheckpointTracker(hub) as tracker:
        hub.load_state_dict_from_url("https://weights.example/models/weights.pt?download=1")

    assert tracker.paths == (expected.resolve(),)


def test_checkpoint_tracker_restores_after_loader_failure(tmp_path: Path) -> None:
    hub = _FakeHub(tmp_path)
    original = hub.load_state_dict_from_url
    with pytest.raises(RuntimeError, match="load failed"):
        with CheckpointTracker(hub):
            def fail(*args: Any, **kwargs: Any) -> object:
                raise RuntimeError("load failed")

            hub.load_state_dict_from_url = fail
            raise RuntimeError("load failed")
    assert hub.load_state_dict_from_url == original


def test_resolve_model_checkpoint_uses_only_captured_file_and_ignores_stale(tmp_path: Path) -> None:
    stale = tmp_path / "stale.pt"
    used = tmp_path / "used.pt"
    stale.write_bytes(b"stale")
    used.write_bytes(b"used")

    resolved = resolve_model_checkpoint((used,), "published-model-name")

    assert resolved == used.resolve()
    assert sha256_file(resolved) == hashlib.sha256(b"used").hexdigest()


def test_resolve_model_checkpoint_supports_explicit_model_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "explicit.pt"
    checkpoint.write_bytes(b"explicit")
    assert resolve_model_checkpoint((), str(checkpoint)) == checkpoint.resolve()


@pytest.mark.parametrize("paths", [(), (Path("a.pt"), Path("b.pt"))])
def test_resolve_model_checkpoint_rejects_zero_or_multiple_actual_files(
    tmp_path: Path, paths: tuple[Path, ...]
) -> None:
    materialized: list[Path] = []
    for path in paths:
        item = tmp_path / path
        item.write_bytes(path.name.encode("ascii"))
        materialized.append(item)
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_model_checkpoint(tuple(materialized), "published-model-name")


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _chmod_tree(root: Path, writable: bool) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if writable:
            path.chmod(mode | stat.S_IWUSR)
        else:
            path.chmod(mode & ~0o222)


def test_validate_source_checkout_checks_commit_detached_clean_read_only_and_origin(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", "https://github.com/example/official.git")
    _git(repo, "switch", "--detach")
    _chmod_tree(repo, writable=False)
    spec = SourceSpec(
        name="Fixture",
        commit=commit,
        official_origins=("https://github.com/example/official.git",),
    )
    try:
        assert validate_source_checkout(repo, spec) == commit

        with pytest.raises(RuntimeError, match="commit"):
            validate_source_checkout(repo, replace(spec, commit="0" * 40))

        with pytest.raises(RuntimeError, match="official origin"):
            validate_source_checkout(
                repo,
                replace(spec, official_origins=("https://github.com/other/repo.git",)),
            )

        source = repo / "source.py"
        source.chmod(stat.S_IMODE(source.stat().st_mode) | stat.S_IWUSR)
        with pytest.raises(RuntimeError, match="read-only"):
            validate_source_checkout(repo, spec)
    finally:
        _chmod_tree(repo, writable=True)


def test_pinned_source_constants_are_exact() -> None:
    assert RADSEG_COMMIT == "3fe8789a3c1b11e41688f7deadc8b8db088ef1a7"
    assert RAYFRONTS_COMMIT == "031262a9ed4d0ea456a1d5605df835ba9e53b027"
    assert RADIO_COMMIT == "c0f37017930e9dda53f93424cf4bf39fc51f287e"


def _base_cli(tmp_path: Path, backend: str = "radseg") -> list[str]:
    return [
        "--backend",
        backend,
        "--source-root",
        str(tmp_path / "source"),
        "--radio-root",
        str(tmp_path / "radio"),
        "--model-version",
        "model-v1",
        "--lang-model",
        "language-v1",
        "--classes-json",
        str(tmp_path / "classes.json"),
        "--device",
        "cpu",
    ]


def test_parse_args_exposes_the_frozen_worker_controls(tmp_path: Path) -> None:
    args = parse_args(
        _base_cli(tmp_path)
        + [
            "--sample-stride",
            "3",
            "--top-k",
            "5",
            "--amp",
        ]
    )
    assert args.backend == "radseg"
    assert args.sample_stride == 3
    assert args.top_k == 5
    assert args.amp is True
    assert args.sam_refinement is False


@pytest.mark.parametrize(
    "extra",
    [["--sample-stride", "0"], ["--top-k", "0"], ["--sample-stride", "true"]],
)
def test_parse_args_rejects_nonpositive_integer_controls(
    tmp_path: Path, extra: list[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(_base_cli(tmp_path) + extra)


def test_validate_cli_args_rejects_topk_above_vocabulary(tmp_path: Path) -> None:
    args = parse_args(_base_cli(tmp_path) + ["--top-k", "42"])
    with pytest.raises(ValueError, match="top-k"):
        validate_cli_args(args, _classes())


def test_validate_cli_args_requires_radseg_and_checkpoint_for_sam(tmp_path: Path) -> None:
    naradio = parse_args(_base_cli(tmp_path, backend="naradio") + ["--sam-refinement"])
    with pytest.raises(ValueError, match="RADSeg"):
        validate_cli_args(naradio, _classes())

    radseg = parse_args(_base_cli(tmp_path) + ["--sam-refinement"])
    with pytest.raises(ValueError, match="sam-checkpoint"):
        validate_cli_args(radseg, _classes())


def test_validate_cli_args_rejects_unrequested_sam_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"sam")
    args = parse_args(_base_cli(tmp_path) + ["--sam-checkpoint", str(checkpoint)])
    with pytest.raises(ValueError, match="sam-refinement"):
        validate_cli_args(args, _classes())


class _RadsegEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], bool, bool]] = []

    def encode_image_to_feat_map(
        self,
        image: torch.Tensor,
        *,
        orig_img_size: tuple[int, int],
        return_preds: bool,
        ignore_label: bool,
    ) -> torch.Tensor:
        self.calls.append((orig_img_size, return_preds, ignore_label))
        height, width = orig_img_size
        return torch.ones((1, 41, height, width), dtype=torch.float32) / 41.0


def test_radseg_runtime_requests_exactly_41_non_ignore_probabilities() -> None:
    encoder = _RadsegEncoder()
    runtime = RadsegRuntime(encoder=encoder, device="cpu", amp=False)
    rgb = np.zeros((7, 11, 3), dtype=np.uint8)

    probabilities = runtime.infer_probabilities(rgb)

    assert probabilities.shape == (1, 41, 7, 11)
    assert encoder.calls == [((7, 11), False, False)]


def test_radseg_runtime_output_reduces_into_frozen_dense_frame() -> None:
    runtime = RadsegRuntime(encoder=_RadsegEncoder(), device="cpu", amp=False)
    reduced = reduce_probabilities(
        runtime.infer_probabilities(np.zeros((7, 11, 3), dtype=np.uint8)),
        sample_stride=2,
        top_k=4,
    )

    frame = DenseSemanticFrame(
        cache_frame_id=0,
        source_frame_id=0,
        image_shape=(7, 11),
        sample_stride=2,
        class_count=41,
        **reduced,
    )
    assert frame.class_ids.shape == (4, 6, 4)


def test_reduce_probabilities_zero_pads_denoised_topk_for_dense_frame() -> None:
    probabilities = np.zeros((1, 41, 1, 1), dtype=np.float32)
    probabilities[:, 0] = 0.8
    probabilities[:, 1] = 0.2
    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=4)

    frame = DenseSemanticFrame(
        cache_frame_id=0,
        source_frame_id=0,
        image_shape=(1, 1),
        sample_stride=1,
        class_count=41,
        **reduced,
    )
    assert frame.class_ids.tolist() == [[[1, 2, 0, 0]]]


def test_radseg_runtime_fails_closed_on_extra_ignore_channel() -> None:
    class BadEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.ones((1, 42, 2, 2), dtype=torch.float32)

    runtime = RadsegRuntime(encoder=BadEncoder(), device="cpu", amp=False)
    with pytest.raises(RuntimeError, match="41 classes"):
        runtime.infer_probabilities(np.zeros((2, 2, 3), dtype=np.uint8))


def test_radseg_runtime_preserves_prompt_denoised_probability_mass() -> None:
    class DenoisedEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            output = torch.zeros((1, 41, 1, 1), dtype=torch.float32)
            output[:, 0] = 0.4
            output[:, 1] = 0.2
            output[:, 2] = 0.2
            return output

    runtime = RadsegRuntime(encoder=DenoisedEncoder(), device="cpu", amp=False)
    probabilities = runtime.infer_probabilities(np.zeros((1, 1, 3), dtype=np.uint8))

    np.testing.assert_allclose(probabilities[0, :3, 0, 0], [0.4, 0.2, 0.2])
    np.testing.assert_allclose(probabilities.sum(axis=1), 0.8)


class _NARadioEncoder:
    def __init__(self) -> None:
        self.updated_resolution: tuple[int, int] | None = None
        self.encoded_shape: tuple[int, ...] | None = None

    def get_nearest_size(self, height: int, width: int) -> tuple[int, int]:
        assert (height, width) == (17, 29)
        return (32, 48)

    @property
    def input_resolution(self) -> tuple[int, int] | None:
        return self.updated_resolution

    @input_resolution.setter
    def input_resolution(self, value: tuple[int, int]) -> None:
        self.updated_resolution = tuple(value)

    def encode_image_to_feat_map(self, image: torch.Tensor) -> torch.Tensor:
        self.encoded_shape = tuple(image.shape)
        feature = torch.zeros((1, 3, 2, 3), dtype=torch.float32)
        feature[:, 0] = 1.0
        return feature

    def align_spatial_features_with_language(self, features: torch.Tensor) -> torch.Tensor:
        return features


def test_naradio_runtime_updates_gaussian_resolution_and_returns_original_size() -> None:
    encoder = _NARadioEncoder()
    text = torch.zeros((41, 3), dtype=torch.float32)
    text[:, 0] = 1.0
    runtime = NARadioRuntime(
        encoder=encoder,
        text_embeddings=text,
        device="cpu",
        amp=False,
    )

    probabilities = runtime.infer_probabilities(np.zeros((17, 29, 3), dtype=np.uint8))

    assert encoder.updated_resolution == (32, 48)
    assert encoder.encoded_shape == (1, 3, 32, 48)
    assert probabilities.shape == (1, 41, 17, 29)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_naradio_runtime_fails_closed_on_invalid_nearest_resolution() -> None:
    encoder = _NARadioEncoder()
    encoder.get_nearest_size = lambda height, width: (0, 48)  # type: ignore[method-assign]
    runtime = NARadioRuntime(
        encoder=encoder,
        text_embeddings=torch.ones((41, 3)),
        device="cpu",
        amp=False,
    )
    with pytest.raises(RuntimeError, match="resolution"):
        runtime.infer_probabilities(np.zeros((17, 29, 3), dtype=np.uint8))


def test_instantiate_radseg_uses_the_pinned_paper_configuration(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    args = parse_args(_base_cli(tmp_path))
    encoder = _instantiate_radseg(Encoder, args, _classes())

    assert isinstance(encoder, Encoder)
    assert calls == [
        {
            "device": "cpu",
            "model_version": "model-v1",
            "lang_model": "language-v1",
            "predict": True,
            "classes": _classes(),
            "amp": False,
            "sam_refinement": False,
            "prompt_denoising_thresh": 0.5,
            "scra_scaling": 10.0,
            "scga_scaling": 10.0,
            "slide_crop": 336,
            "slide_stride": 112,
        }
    ]


def test_instantiate_radseg_passes_sam_checkpoint_when_enabled(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"sam")
    args = parse_args(
        _base_cli(tmp_path)
        + ["--sam-refinement", "--sam-checkpoint", str(checkpoint)]
    )
    _instantiate_radseg(Encoder, args, _classes())
    assert calls[0]["sam_refinement"] is True
    assert calls[0]["sam_ckpt"] == str(checkpoint.resolve())


def test_instantiate_naradio_uses_language_aligned_public_encoder_api(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    args = parse_args(_base_cli(tmp_path, backend="naradio"))
    encoder = _instantiate_naradio(Encoder, args)
    assert isinstance(encoder, Encoder)
    assert calls == [
        {
            "device": "cpu",
            "model_version": "model-v1",
            "lang_model": "language-v1",
            "input_resolution": (224, 224),
            "return_radio_features": True,
            "compile": False,
            "amp": False,
        }
    ]


class _FakeRuntime:
    def __init__(self, probabilities: np.ndarray | None = None, error: str | None = None) -> None:
        self.probabilities = (
            probabilities
            if probabilities is not None
            else np.ones((1, 41, 2, 2), dtype=np.float32) / 41.0
        )
        self.error = error

    def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.probabilities


def _fake_worker(runtime: _FakeRuntime | None = None) -> DenseWorker:
    return DenseWorker(
        runtime=runtime or _FakeRuntime(),
        classes=tuple(_classes()),
        sample_stride=1,
        top_k=4,
        provenance=_provenance(),
    )


def test_metadata_response_contains_complete_provenance_and_contract() -> None:
    response = run_request(_fake_worker(), {"id": {"frame": 1}, "operation": "metadata"})
    assert response["id"] == {"frame": 1}
    assert response["ok"] is True
    assert response["class_count"] == 41
    assert response["sample_stride"] == 1
    assert response["top_k"] == 4
    assert response["classes"] == _classes()
    assert set(response["provenance"]) == {
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
    }


def test_infer_response_encodes_all_sampled_arrays() -> None:
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    response = run_request(
        _fake_worker(),
        {"id": "frame-7", "operation": "infer", "rgb": _array_block(rgb)},
    )
    assert response["id"] == "frame-7"
    assert response["ok"] is True
    assert response["image_shape"] == [2, 2]
    assert response["class_count"] == 41
    assert _decode_array(response["class_ids"]).shape == (2, 2, 4)
    assert _decode_array(response["probabilities"]).dtype == np.float32
    assert _decode_array(response["entropy"]).shape == (2, 2)
    assert _decode_array(response["margin"]).shape == (2, 2)


def test_infer_response_arrays_construct_the_frozen_dense_frame_contract() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    response = run_request(
        _fake_worker(),
        {"id": 7, "operation": "infer", "rgb": _array_block(rgb)},
    )

    frame = DenseSemanticFrame(
        cache_frame_id=7,
        source_frame_id=70,
        image_shape=tuple(response["image_shape"]),
        sample_stride=response["sample_stride"],
        class_count=response["class_count"],
        class_ids=_decode_array(response["class_ids"]),
        probabilities=_decode_array(response["probabilities"]),
        entropy=_decode_array(response["entropy"]),
        margin=_decode_array(response["margin"]),
    )
    assert frame.class_count == 41


def test_run_request_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        run_request(_fake_worker(), {"id": 1, "operation": "future"})


def test_serve_jsonl_returns_one_line_per_request_and_continues_after_error() -> None:
    class PrintingRuntime(_FakeRuntime):
        def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
            print("model diagnostic")
            return super().infer_probabilities(rgb)

    invalid = {"id": 1, "operation": "infer", "rgb": _array_block(np.zeros((2, 2, 3), dtype=np.uint8))}
    invalid["rgb"]["data"] = "AA=="
    valid = {"id": 2, "operation": "metadata"}
    input_stream = io.StringIO(
        "not-json\n" + json.dumps(invalid) + "\n" + json.dumps(valid) + "\n"
    )
    output_stream = io.StringIO()
    stderr = io.StringIO()

    serve_jsonl(_fake_worker(PrintingRuntime()), input_stream, output_stream, stderr)

    lines = output_stream.getvalue().splitlines()
    assert len(lines) == 3
    responses = [json.loads(line) for line in lines]
    assert responses[0]["id"] is None and responses[0]["ok"] is False
    assert responses[1]["id"] == 1 and responses[1]["ok"] is False
    assert responses[2]["id"] == 2 and responses[2]["ok"] is True
    assert "model diagnostic" not in output_stream.getvalue()


def test_serve_jsonl_redirects_model_stdout_to_stderr() -> None:
    request = {
        "id": 9,
        "operation": "infer",
        "rgb": _array_block(np.zeros((2, 2, 3), dtype=np.uint8)),
    }
    input_stream = io.StringIO(json.dumps(request) + "\n")
    output_stream = io.StringIO()
    stderr = io.StringIO()

    class PrintingRuntime(_FakeRuntime):
        def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
            print("backend-log")
            return super().infer_probabilities(rgb)

    serve_jsonl(_fake_worker(PrintingRuntime()), input_stream, output_stream, stderr)

    assert len(output_stream.getvalue().splitlines()) == 1
    assert json.loads(output_stream.getvalue())["ok"] is True
    assert "backend-log" in stderr.getvalue()


def test_worker_error_response_does_not_emit_nonfinite_json() -> None:
    worker = _fake_worker(_FakeRuntime(error="inference failed"))
    request = {
        "id": 3,
        "operation": "infer",
        "rgb": _array_block(np.zeros((2, 2, 3), dtype=np.uint8)),
    }
    input_stream = io.StringIO(json.dumps(request) + "\n")
    output_stream = io.StringIO()
    serve_jsonl(worker, input_stream, output_stream, io.StringIO())
    assert json.loads(output_stream.getvalue()) == {
        "error": "inference failed",
        "id": 3,
        "ok": False,
    }
