from __future__ import annotations

import base64
from contextlib import nullcontext
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
    LANGUAGE_MODEL_SPECS,
    MAX_JSONL_LINE_CHARS,
    RADIO_COMMIT,
    RADSEG_COMMIT,
    RAYFRONTS_COMMIT,
    CheckpointTracker,
    DenseWorker,
    LanguageModelAssets,
    NARadioRuntime,
    RadsegRuntime,
    SourceSpec,
    _detach_text_embeddings,
    _instantiate_naradio,
    _instantiate_radseg,
    _fingerprint_file,
    build_worker,
    canonical_sha256,
    decode_rgb,
    encode_array,
    load_frozen_classes,
    local_radio_hub,
    language_model_tree_sha256,
    parse_args,
    pinned_language_model_load,
    reduce_probabilities,
    resolve_model_checkpoint,
    run_request,
    serve_jsonl,
    sha256_file,
    validate_cli_args,
    validate_language_model_assets,
    validate_inference_budget,
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


def _classes(count: int = 41) -> list[str]:
    return [f"class-{index:02d}" for index in range(count)]


def _provenance() -> dict[str, str]:
    return {
        "backend": "radseg",
        "source_commit": RADSEG_COMMIT,
        "radio_commit": RADIO_COMMIT,
        "model_id": f"radseg:test:test:sam=0:sha256={'a' * 64}",
        "model_sha256": "a" * 64,
        "auxiliary_model_sha256": "",
        "language_model_id": "google/siglip2-giant-opt-patch16-384",
        "language_model_revision": "a713301b217d38485fb2204c808367d10bc3cc40",
        "language_model_sha256": "e" * 64,
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


def test_reduce_probabilities_rejects_mass_outside_dense_frame_tolerance() -> None:
    probabilities = np.asarray([[[[0.500003]], [[0.500002]]]], dtype=np.float32)
    with pytest.raises(ValueError, match="mass"):
        reduce_probabilities(probabilities, sample_stride=1, top_k=2)


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


def test_reduce_probabilities_encodes_zero_mass_as_unknown_without_log_warning() -> None:
    with np.errstate(all="raise"):
        reduced = reduce_probabilities(
            np.zeros((1, 3, 2, 2), dtype=np.float32),
            sample_stride=1,
            top_k=2,
        )

    np.testing.assert_array_equal(reduced["class_ids"], 0)
    np.testing.assert_array_equal(reduced["probabilities"], 0.0)
    np.testing.assert_array_equal(reduced["entropy"], 0.0)
    np.testing.assert_array_equal(reduced["margin"], 0.0)


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


def test_inference_budget_accepts_replica_stride4_top4() -> None:
    budget = validate_inference_budget(
        height=680,
        width=1200,
        sample_stride=4,
        top_k=4,
    )
    assert budget.probability_tensor_bytes == 680 * 1200 * 41 * 4
    assert budget.sampled_shape == (170, 300)
    assert budget.encoded_response_chars < MAX_JSONL_LINE_CHARS


def test_inference_budget_accounts_for_scannet200_class_count() -> None:
    budget = validate_inference_budget(
        height=480,
        width=640,
        sample_stride=4,
        top_k=4,
        class_count=200,
    )

    assert budget.probability_tensor_bytes == 480 * 640 * 200 * 4


def test_inference_budget_rejects_stride1_top41_before_inference() -> None:
    with pytest.raises(ValueError, match="response|budget"):
        validate_inference_budget(
            height=680,
            width=1200,
            sample_stride=1,
            top_k=41,
        )


def test_inference_budget_rejects_giant_probability_tensor() -> None:
    with pytest.raises(ValueError, match="probability tensor|budget"):
        validate_inference_budget(
            height=20_000,
            width=20_000,
            sample_stride=100,
            top_k=1,
        )


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


def test_load_frozen_classes_accepts_scannet200_contract(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": _classes(200)}), encoding="utf-8")

    classes, _ = load_frozen_classes(path)

    assert classes == _classes(200)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"classes": []}, "non-empty"),
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


def test_checkpoint_tracker_rejects_non_radio_url_before_loader_call(tmp_path: Path) -> None:
    hub = _FakeHub(tmp_path)
    expected_url = "https://huggingface.co/nvidia/RADIO/resolve/main/radio.pt"
    with pytest.raises(RuntimeError, match="RADIO checkpoint URL"):
        with CheckpointTracker(hub, expected_url=expected_url):
            hub.load_state_dict_from_url("https://example.com/unexpected.pt")
    assert hub.calls == []


def test_checkpoint_tracker_for_explicit_model_rejects_every_url(tmp_path: Path) -> None:
    hub = _FakeHub(tmp_path)
    with pytest.raises(RuntimeError, match="RADIO checkpoint URL"):
        with CheckpointTracker(hub, expected_url=None):
            hub.load_state_dict_from_url("https://example.com/unexpected.pt")
    assert hub.calls == []


def test_checkpoint_tracker_rejects_cached_weight_changed_during_load(tmp_path: Path) -> None:
    url = "https://huggingface.co/nvidia/RADIO/resolve/main/weights.pt"
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    cached = checkpoints / "weights.pt"
    cached.write_bytes(b"before")

    class MutatingHub(_FakeHub):
        def load_state_dict_from_url(
            self, requested_url: str, *args: Any, **kwargs: Any
        ) -> object:
            self.calls.append(((requested_url, *args), kwargs))
            cached.write_bytes(b"after")
            return object()

    hub = MutatingHub(tmp_path)
    with pytest.raises(RuntimeError, match="changed during checkpoint loading"):
        with CheckpointTracker(hub, expected_url=url):
            hub.load_state_dict_from_url(url)


def test_explicit_checkpoint_fingerprint_requires_read_only_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="read-only"):
        _fingerprint_file(checkpoint, require_read_only=True)
    checkpoint.chmod(stat.S_IMODE(checkpoint.stat().st_mode) & ~0o222)
    assert _fingerprint_file(checkpoint, require_read_only=True).sha256 == hashlib.sha256(
        b"weights"
    ).hexdigest()


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


def _make_language_model_root(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "language-model"
    (root / "tokenizer").mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"siglip"}\n', encoding="utf-8")
    (root / "tokenizer" / "tokenizer.json").write_bytes(b"frozen-tokenizer")
    _chmod_tree(root, writable=False)
    return root, language_model_tree_sha256(root)


def _language_assets(tmp_path: Path, backend: str = "radseg") -> LanguageModelAssets:
    root, digest = _make_language_model_root(tmp_path)
    if backend == "radseg":
        model_id = "google/siglip2-giant-opt-patch16-384"
        revision = "a713301b217d38485fb2204c808367d10bc3cc40"
    else:
        model_id = "timm/ViT-SO400M-14-SigLIP-384"
        revision = "ac16108d567c4389e6cd2b11c9b8585f7474435b"
    return LanguageModelAssets(
        backend=backend,
        root=root,
        model_id=model_id,
        revision=revision,
        sha256=digest,
    )


def test_language_model_tree_hash_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    root, digest = _make_language_model_root(tmp_path)
    assert language_model_tree_sha256(root) == digest
    try:
        _chmod_tree(root, writable=True)
        (root / "config.json").write_text('{"model_type":"changed"}\n', encoding="utf-8")
        _chmod_tree(root, writable=False)
        assert language_model_tree_sha256(root) != digest
    finally:
        _chmod_tree(root, writable=True)


def test_language_model_tree_hash_covers_empty_directories(tmp_path: Path) -> None:
    root, digest = _make_language_model_root(tmp_path)
    try:
        _chmod_tree(root, writable=True)
        (root / "empty").mkdir()
        _chmod_tree(root, writable=False)
        assert language_model_tree_sha256(root) != digest
    finally:
        _chmod_tree(root, writable=True)


def test_language_model_tree_rejects_writable_symlink_and_nonregular_content(
    tmp_path: Path,
) -> None:
    root, _ = _make_language_model_root(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    try:
        _chmod_tree(root, writable=True)
        (root / "external-link").symlink_to(outside)
        _chmod_tree(root, writable=False)
        with pytest.raises(RuntimeError, match="symlink"):
            language_model_tree_sha256(root)
        _chmod_tree(root, writable=True)
        (root / "external-link").unlink()

        writable = root / "config.json"
        writable.chmod(stat.S_IMODE(writable.stat().st_mode) | stat.S_IWUSR)
        with pytest.raises(RuntimeError, match="read-only"):
            language_model_tree_sha256(root)
        writable.chmod(stat.S_IMODE(writable.stat().st_mode) & ~0o222)

        _chmod_tree(root, writable=True)
        fifo = root / "named-pipe"
        os.mkfifo(fifo)
        _chmod_tree(root, writable=False)
        with pytest.raises(RuntimeError, match="regular files|non-regular"):
            language_model_tree_sha256(root)
    finally:
        _chmod_tree(root, writable=True)


def test_language_model_specs_are_exactly_the_reviewed_revisions() -> None:
    assert {
        model_id: (spec.revision, spec.backends)
        for model_id, spec in LANGUAGE_MODEL_SPECS.items()
    } == {
        "google/siglip2-giant-opt-patch16-384": (
            "a713301b217d38485fb2204c808367d10bc3cc40",
            frozenset({"radseg"}),
        ),
        "google/siglip2-so400m-patch16-naflex": (
            "cc24074f717b612951c2dead130904ab9b65a81e",
            frozenset({"radseg"}),
        ),
        "timm/ViT-SO400M-14-SigLIP-384": (
            "ac16108d567c4389e6cd2b11c9b8585f7474435b",
            frozenset({"naradio"}),
        ),
    }


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
    language_model_id = (
        "google/siglip2-giant-opt-patch16-384"
        if backend == "radseg"
        else "timm/ViT-SO400M-14-SigLIP-384"
    )
    language_model_revision = (
        "a713301b217d38485fb2204c808367d10bc3cc40"
        if backend == "radseg"
        else "ac16108d567c4389e6cd2b11c9b8585f7474435b"
    )
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
        "siglip2" if backend == "radseg" else "siglip",
        "--language-model-root",
        str(tmp_path / "language-model"),
        "--language-model-id",
        language_model_id,
        "--language-model-revision",
        language_model_revision,
        "--language-model-sha256",
        "e" * 64,
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
    assert args.language_model_id == "google/siglip2-giant-opt-patch16-384"
    assert args.language_model_revision == "a713301b217d38485fb2204c808367d10bc3cc40"
    assert args.language_model_sha256 == "e" * 64


@pytest.mark.parametrize(
    "extra",
    [["--sample-stride", "0"], ["--top-k", "0"], ["--sample-stride", "true"]],
)
def test_parse_args_rejects_nonpositive_integer_controls(
    tmp_path: Path, extra: list[str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(_base_cli(tmp_path) + extra)


def test_parse_args_requires_language_asset_provenance(tmp_path: Path) -> None:
    argv = _base_cli(tmp_path)
    flag_index = argv.index("--language-model-root")
    del argv[flag_index : flag_index + 2]
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_parse_args_rejects_invalid_language_model_sha256(tmp_path: Path) -> None:
    argv = _base_cli(tmp_path)
    argv[argv.index("--language-model-sha256") + 1] = "not-a-sha256"
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_validate_cli_args_rejects_topk_above_vocabulary(tmp_path: Path) -> None:
    args = parse_args(_base_cli(tmp_path) + ["--top-k", "42"])
    with pytest.raises(ValueError, match="top-k"):
        validate_cli_args(args, _classes())


def test_validate_cli_args_accepts_both_pinned_radseg_language_models(tmp_path: Path) -> None:
    args = parse_args(_base_cli(tmp_path))
    validate_cli_args(args, _classes())

    args.language_model_id = "google/siglip2-so400m-patch16-naflex"
    args.language_model_revision = "cc24074f717b612951c2dead130904ab9b65a81e"
    validate_cli_args(args, _classes())


@pytest.mark.parametrize(
    ("backend", "lang_model", "model_id", "revision", "message"),
    [
        (
            "naradio",
            "siglip",
            "google/siglip2-giant-opt-patch16-384",
            "a713301b217d38485fb2204c808367d10bc3cc40",
            "backend",
        ),
        (
            "radseg",
            "siglip2",
            "google/siglip2-giant-opt-patch16-384",
            "0" * 40,
            "revision",
        ),
        (
            "radseg",
            "siglip",
            "google/siglip2-giant-opt-patch16-384",
            "a713301b217d38485fb2204c808367d10bc3cc40",
            "lang-model",
        ),
    ],
)
def test_validate_cli_args_rejects_unpinned_language_combinations(
    tmp_path: Path,
    backend: str,
    lang_model: str,
    model_id: str,
    revision: str,
    message: str,
) -> None:
    args = parse_args(_base_cli(tmp_path, backend=backend))
    args.lang_model = lang_model
    args.language_model_id = model_id
    args.language_model_revision = revision
    with pytest.raises(ValueError, match=message):
        validate_cli_args(args, _classes())


def test_validate_language_model_assets_checks_local_tree_hash(tmp_path: Path) -> None:
    root, digest = _make_language_model_root(tmp_path)
    args = parse_args(_base_cli(tmp_path))
    args.language_model_root = root
    args.language_model_sha256 = digest
    assets = validate_language_model_assets(args)
    assert assets.root == root.resolve()
    assert assets.sha256 == digest

    args.language_model_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256|hash"):
        validate_language_model_assets(args)
    _chmod_tree(root, writable=True)


def test_transformers_language_patch_forces_local_no_remote_code_and_restores(
    tmp_path: Path,
) -> None:
    assets = _language_assets(tmp_path)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    environ = {"HF_HUB_OFFLINE": "previous"}

    def model_loader(*args: Any, **kwargs: Any) -> object:
        calls.append(("model", args, kwargs))
        return object()

    def processor_loader(*args: Any, **kwargs: Any) -> object:
        calls.append(("processor", args, kwargs))
        return object()

    auto_model = SimpleNamespace(from_pretrained=model_loader)
    auto_processor = SimpleNamespace(from_pretrained=processor_loader)
    transformers_module = SimpleNamespace(
        AutoModel=auto_model,
        AutoProcessor=auto_processor,
    )
    try:
        with pinned_language_model_load(
            assets,
            transformers_module=transformers_module,
            environ=environ,
        ):
            auto_model.from_pretrained(
                assets.model_id,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            auto_processor.from_pretrained(
                pretrained_model_name_or_path=assets.model_id,
                trust_remote_code=True,
            )
            assert environ["HF_HUB_OFFLINE"] == "1"
            assert environ["TRANSFORMERS_OFFLINE"] == "1"

        expected_root = str(assets.root)
        assert calls == [
            (
                "model",
                (expected_root,),
                {
                    "trust_remote_code": False,
                    "torch_dtype": "auto",
                    "local_files_only": True,
                },
            ),
            (
                "processor",
                (),
                {
                    "pretrained_model_name_or_path": expected_root,
                    "trust_remote_code": False,
                    "local_files_only": True,
                },
            ),
        ]
        assert auto_model.from_pretrained is model_loader
        assert auto_processor.from_pretrained is processor_loader
        assert environ == {"HF_HUB_OFFLINE": "previous"}
    finally:
        _chmod_tree(assets.root, writable=True)


def test_transformers_language_patch_rejects_wrong_id_and_restores_after_error(
    tmp_path: Path,
) -> None:
    assets = _language_assets(tmp_path)

    def network_sentinel(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("network loader must receive only the pinned local root")

    auto_model = SimpleNamespace(from_pretrained=network_sentinel)
    auto_processor = SimpleNamespace(from_pretrained=network_sentinel)
    module = SimpleNamespace(AutoModel=auto_model, AutoProcessor=auto_processor)
    try:
        with pytest.raises(RuntimeError, match="language model ID"):
            with pinned_language_model_load(assets, transformers_module=module):
                auto_model.from_pretrained("google/unpinned")
        assert auto_model.from_pretrained is network_sentinel
        assert auto_processor.from_pretrained is network_sentinel
    finally:
        _chmod_tree(assets.root, writable=True)


def test_transformers_language_patch_restores_when_local_loader_raises(
    tmp_path: Path,
) -> None:
    assets = _language_assets(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> object:
        raise RuntimeError("local load failed")

    auto_model = SimpleNamespace(from_pretrained=fail)
    auto_processor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    module = SimpleNamespace(AutoModel=auto_model, AutoProcessor=auto_processor)
    try:
        with pytest.raises(RuntimeError, match="local load failed"):
            with pinned_language_model_load(assets, transformers_module=module):
                auto_model.from_pretrained(assets.model_id)
        assert auto_model.from_pretrained is fail
    finally:
        _chmod_tree(assets.root, writable=True)


def test_language_patch_restores_partial_setup_failure(tmp_path: Path) -> None:
    assets = _language_assets(tmp_path)
    original = lambda *args, **kwargs: object()
    auto_model = SimpleNamespace(from_pretrained=original)
    incomplete_module = SimpleNamespace(AutoModel=auto_model)
    environ = {"TRANSFORMERS_OFFLINE": "previous"}
    try:
        with pytest.raises(AttributeError, match="AutoProcessor"):
            with pinned_language_model_load(
                assets,
                transformers_module=incomplete_module,
                environ=environ,
            ):
                pass
        assert auto_model.from_pretrained is original
        assert environ == {"TRANSFORMERS_OFFLINE": "previous"}
    finally:
        _chmod_tree(assets.root, writable=True)


def test_language_patch_restores_when_second_patch_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radseg_dense_worker as worker_module

    assets = _language_assets(tmp_path)
    model_loader = lambda *args, **kwargs: object()
    processor_loader = lambda *args, **kwargs: object()
    auto_model = SimpleNamespace(from_pretrained=model_loader)
    auto_processor = SimpleNamespace(from_pretrained=processor_loader)
    module = SimpleNamespace(AutoModel=auto_model, AutoProcessor=auto_processor)
    environ = {"HF_HUB_OFFLINE": "previous"}
    real_patch = worker_module._patch_attribute
    calls = 0

    def fail_second(owner: Any, name: str, replacement: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second patch failed")
        return real_patch(owner, name, replacement)

    monkeypatch.setattr(worker_module, "_patch_attribute", fail_second)
    try:
        with pytest.raises(RuntimeError, match="second patch failed"):
            with pinned_language_model_load(
                assets,
                transformers_module=module,
                environ=environ,
            ):
                pass
        assert auto_model.from_pretrained is model_loader
        assert auto_processor.from_pretrained is processor_loader
        assert environ == {"HF_HUB_OFFLINE": "previous"}
    finally:
        _chmod_tree(assets.root, writable=True)


def test_transformers_language_patch_rejects_extra_model_request(tmp_path: Path) -> None:
    assets = _language_assets(tmp_path)
    auto_model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    auto_processor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    module = SimpleNamespace(AutoModel=auto_model, AutoProcessor=auto_processor)
    try:
        with pytest.raises(RuntimeError, match="extra AutoModel"):
            with pinned_language_model_load(assets, transformers_module=module):
                auto_model.from_pretrained(assets.model_id)
                auto_model.from_pretrained(assets.model_id)
    finally:
        _chmod_tree(assets.root, writable=True)


def test_language_patch_rejects_tree_change_during_model_load(tmp_path: Path) -> None:
    assets = _language_assets(tmp_path)
    auto_model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    auto_processor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    module = SimpleNamespace(AutoModel=auto_model, AutoProcessor=auto_processor)
    try:
        with pytest.raises(RuntimeError, match="changed during model loading"):
            with pinned_language_model_load(assets, transformers_module=module):
                auto_model.from_pretrained(assets.model_id)
                auto_processor.from_pretrained(assets.model_id)
                _chmod_tree(assets.root, writable=True)
                (assets.root / "config.json").write_text("changed\n", encoding="utf-8")
                _chmod_tree(assets.root, writable=False)
        assert auto_model.from_pretrained.__name__ == "<lambda>"
        assert auto_processor.from_pretrained.__name__ == "<lambda>"
    finally:
        _chmod_tree(assets.root, writable=True)


def test_open_clip_language_patch_forces_local_dir_and_restores(tmp_path: Path) -> None:
    assets = _language_assets(tmp_path, backend="naradio")
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    environ: dict[str, str] = {}

    def create(*args: Any, **kwargs: Any) -> object:
        calls.append(("create", args, kwargs))
        return object()

    def tokenizer(*args: Any, **kwargs: Any) -> object:
        calls.append(("tokenizer", args, kwargs))
        return object()

    module = SimpleNamespace(
        create_model_from_pretrained=create,
        get_tokenizer=tokenizer,
    )
    try:
        with pinned_language_model_load(
            assets,
            open_clip_module=module,
            environ=environ,
        ):
            module.create_model_from_pretrained(
                model_name="ViT-SO400M-14-SigLIP-384",
                pretrained="webli",
                return_transform=False,
            )
            module.get_tokenizer("ViT-SO400M-14-SigLIP-384")
            assert environ["HF_HUB_OFFLINE"] == "1"

        local_name = f"local-dir:{assets.root}"
        assert calls == [
            (
                "create",
                (),
                {
                    "model_name": local_name,
                    "pretrained": "webli",
                    "return_transform": False,
                },
            ),
            ("tokenizer", (local_name,), {}),
        ]
        assert module.create_model_from_pretrained is create
        assert module.get_tokenizer is tokenizer
        assert environ == {}
    finally:
        _chmod_tree(assets.root, writable=True)


@pytest.mark.parametrize(
    ("model_name", "pretrained", "message"),
    [
        ("ViT-B-32", "webli", "model_name"),
        ("ViT-SO400M-14-SigLIP-384", "openai", "pretrained"),
    ],
)
def test_open_clip_language_patch_rejects_unpinned_requests(
    tmp_path: Path,
    model_name: str,
    pretrained: str,
    message: str,
) -> None:
    assets = _language_assets(tmp_path, backend="naradio")
    create = lambda *args, **kwargs: object()
    tokenizer = lambda *args, **kwargs: object()
    module = SimpleNamespace(
        create_model_from_pretrained=create,
        get_tokenizer=tokenizer,
    )
    try:
        with pytest.raises(RuntimeError, match=message):
            with pinned_language_model_load(assets, open_clip_module=module):
                module.create_model_from_pretrained(model_name, pretrained=pretrained)
        assert module.create_model_from_pretrained is create
        assert module.get_tokenizer is tokenizer
    finally:
        _chmod_tree(assets.root, writable=True)


def test_open_clip_language_patch_rejects_unpinned_tokenizer_request(
    tmp_path: Path,
) -> None:
    assets = _language_assets(tmp_path, backend="naradio")
    module = SimpleNamespace(
        create_model_from_pretrained=lambda *args, **kwargs: object(),
        get_tokenizer=lambda *args, **kwargs: object(),
    )
    try:
        with pytest.raises(RuntimeError, match="tokenizer model_name"):
            with pinned_language_model_load(assets, open_clip_module=module):
                module.create_model_from_pretrained(
                    "ViT-SO400M-14-SigLIP-384",
                    pretrained="webli",
                )
                module.get_tokenizer("ViT-B-32")
    finally:
        _chmod_tree(assets.root, writable=True)


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


def test_radseg_runtime_accepts_scannet200_probability_channels() -> None:
    class ScanNetEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            height, width = kwargs["orig_img_size"]
            return torch.ones((1, 200, height, width), dtype=torch.float32) / 200.0

    runtime = RadsegRuntime(
        encoder=ScanNetEncoder(),
        device="cpu",
        amp=False,
        class_count=200,
    )

    probabilities = runtime.infer_probabilities(np.zeros((3, 5, 3), dtype=np.uint8))

    assert probabilities.shape == (1, 200, 3, 5)


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


def test_radseg_runtime_renormalizes_float16_probability_mass_drift() -> None:
    class DriftedEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            output = torch.zeros((1, 41, 1, 1), dtype=torch.float32)
            output[:, 0] = 0.5002
            output[:, 1] = 0.5002
            return output

    runtime = RadsegRuntime(encoder=DriftedEncoder(), device="cpu", amp=True)
    probabilities = runtime.infer_probabilities(np.zeros((1, 1, 3), dtype=np.uint8))

    np.testing.assert_allclose(probabilities[0, :2, 0, 0], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    reduce_probabilities(probabilities, sample_stride=1, top_k=2)


def test_radseg_runtime_rejects_non_numerical_probability_mass_error() -> None:
    class InvalidEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            output = torch.zeros((1, 41, 1, 1), dtype=torch.float32)
            output[:, 0] = 0.51
            output[:, 1] = 0.51
            return output

    runtime = RadsegRuntime(encoder=InvalidEncoder(), device="cpu", amp=True)

    with pytest.raises(RuntimeError, match="probability mass above one"):
        runtime.infer_probabilities(np.zeros((1, 1, 3), dtype=np.uint8))


def test_radseg_runtime_allows_local_unknown_pixels_but_rejects_all_unknown() -> None:
    class PartiallyUnknownEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            output = torch.zeros((1, 41, 1, 2), dtype=torch.float32)
            output[:, 0, 0, 0] = 0.6
            return output

    runtime = RadsegRuntime(encoder=PartiallyUnknownEncoder(), device="cpu", amp=True)
    probabilities = runtime.infer_probabilities(np.zeros((1, 2, 3), dtype=np.uint8))
    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=2)

    assert probabilities.sum(axis=1).tolist() == [[[pytest.approx(0.6), 0.0]]]
    assert reduced["class_ids"][0, 1].tolist() == [0, 0]
    assert reduced["probabilities"][0, 1].tolist() == [0.0, 0.0]

    class AllUnknownEncoder(_RadsegEncoder):
        def encode_image_to_feat_map(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.zeros((1, 41, 1, 2), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="zero probability mass"):
        RadsegRuntime(encoder=AllUnknownEncoder(), device="cpu", amp=True).infer_probabilities(
            np.zeros((1, 2, 3), dtype=np.uint8)
        )


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


def test_naradio_runtime_accepts_scannet200_text_embeddings() -> None:
    text = torch.zeros((200, 3), dtype=torch.float32)
    text[:, 0] = 1.0
    runtime = NARadioRuntime(
        encoder=_NARadioEncoder(),
        text_embeddings=text,
        device="cpu",
        amp=False,
    )

    probabilities = runtime.infer_probabilities(np.zeros((17, 29, 3), dtype=np.uint8))

    assert probabilities.shape == (1, 200, 17, 29)


def test_detach_text_embeddings_accepts_frozen_scannet200_shape() -> None:
    embeddings = torch.ones((200, 3), requires_grad=True) * 2.0

    detached = _detach_text_embeddings(embeddings, "RADSeg", class_count=200)

    assert detached.shape == (200, 3)
    assert detached.requires_grad is False
    assert detached.grad_fn is None


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
            "lang_model": "siglip2",
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
            "lang_model": "siglip",
            "input_resolution": (224, 224),
            "return_radio_features": True,
            "compile": False,
            "amp": False,
        }
    ]


def test_build_worker_pins_language_assets_and_restores_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radseg_dense_worker as worker_module
    import transformers

    language_root, language_hash = _make_language_model_root(tmp_path)
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps({"classes": _classes()}), encoding="utf-8")
    checkpoint = tmp_path / "radio.pt"
    checkpoint.write_bytes(b"radio-checkpoint")
    checkpoint.chmod(stat.S_IMODE(checkpoint.stat().st_mode) & ~0o222)
    args = parse_args(_base_cli(tmp_path))
    args.model_version = str(checkpoint)
    args.classes_json = classes_path
    args.language_model_root = language_root
    args.language_model_sha256 = language_hash

    loader_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def model_loader(*loader_args: Any, **loader_kwargs: Any) -> object:
        assert loader_args == (str(language_root.resolve()),)
        assert loader_kwargs["local_files_only"] is True
        assert loader_kwargs["trust_remote_code"] is False
        loader_calls.append(("model", loader_args, loader_kwargs))
        return object()

    def processor_loader(*loader_args: Any, **loader_kwargs: Any) -> object:
        assert loader_args == (str(language_root.resolve()),)
        assert loader_kwargs["local_files_only"] is True
        assert loader_kwargs["trust_remote_code"] is False
        loader_calls.append(("processor", loader_args, loader_kwargs))
        return object()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(model_loader))
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(processor_loader),
    )

    graph_text_embeddings = torch.ones((41, 3), requires_grad=True) * 2.0

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            assert torch.is_grad_enabled() is False
            transformers.AutoModel.from_pretrained(
                args.language_model_id,
                trust_remote_code=True,
            )
            self.text_embeds = graph_text_embeddings
            transformers.AutoProcessor.from_pretrained(
                args.language_model_id,
                trust_remote_code=True,
            )

        def insert_labels_into_templates(self, labels: list[str]) -> list[list[str]]:
            return [[f"a photo of {label}"] for label in labels]

    monkeypatch.setattr(
        worker_module,
        "_import_pinned_module",
        lambda root, module_name: SimpleNamespace(RADSegEncoder=Encoder),
    )
    source_validation_calls: list[str] = []

    def validate_source(root: Path, spec: Any) -> str:
        source_validation_calls.append(spec.name)
        return spec.commit

    monkeypatch.setattr(worker_module, "validate_source_checkout", validate_source)
    hash_payloads: list[Any] = []
    original_canonical_sha256 = worker_module.canonical_sha256

    def record_hash(payload: Any) -> str:
        hash_payloads.append(payload)
        return original_canonical_sha256(payload)

    monkeypatch.setattr(worker_module, "canonical_sha256", record_hash)
    offline_keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    )
    environment_before = {key: os.environ.get(key) for key in offline_keys}
    try:
        worker = build_worker(args)

        assert [name for name, _args, _kwargs in loader_calls] == ["model", "processor"]
        assert transformers.AutoModel.from_pretrained is model_loader
        assert transformers.AutoProcessor.from_pretrained is processor_loader
        assert {key: os.environ.get(key) for key in offline_keys} == environment_before
        assert worker.provenance["language_model_id"] == args.language_model_id
        assert worker.provenance["language_model_revision"] == args.language_model_revision
        assert worker.provenance["language_model_sha256"] == language_hash
        assert worker.runtime.encoder.text_embeds.requires_grad is False
        assert worker.runtime.encoder.text_embeds.grad_fn is None
        assert source_validation_calls == ["RADSeg", "RADIO", "RADSeg", "RADIO"]
        config_payload = next(
            payload
            for payload in hash_payloads
            if isinstance(payload, dict) and "inference_config_version" in payload
        )
        assert config_payload["inference_config_version"] == 4
        assert config_payload["language_model_id"] == args.language_model_id
        assert config_payload["language_model_revision"] == args.language_model_revision
        assert config_payload["language_model_sha256"] == language_hash
        assert config_payload["model"]["probability_mass_policy"] == (
            "preserve-denoised-mass-normalize-numerical-overshoot"
        )
        assert config_payload["model"]["probability_mass_drift_tolerance"] == 1e-3
        assert config_payload["model"]["zero_mass_policy"] == (
            "encode-local-zero-mass-as-unknown-reject-all-zero-image"
        )
    finally:
        _chmod_tree(language_root, writable=True)


def test_build_worker_constructs_naradio_and_text_embeddings_without_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_clip = pytest.importorskip("open_clip")
    import radseg_dense_worker as worker_module

    language_root, language_hash = _make_language_model_root(tmp_path)
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps({"classes": _classes()}), encoding="utf-8")
    checkpoint = tmp_path / "radio.pt"
    checkpoint.write_bytes(b"radio-checkpoint")
    checkpoint.chmod(stat.S_IMODE(checkpoint.stat().st_mode) & ~0o222)
    args = parse_args(_base_cli(tmp_path, backend="naradio"))
    args.model_version = str(checkpoint)
    args.classes_json = classes_path
    args.language_model_root = language_root
    args.language_model_sha256 = language_hash
    graph_text_embeddings = torch.ones((41, 3), requires_grad=True) * 3.0
    loader_calls: list[str] = []

    def create_loader(*loader_args: Any, **loader_kwargs: Any) -> object:
        model_name = loader_args[0] if loader_args else loader_kwargs["model_name"]
        assert model_name == f"local-dir:{language_root.resolve()}"
        loader_calls.append("model")
        return object()

    def tokenizer_loader(*loader_args: Any, **loader_kwargs: Any) -> object:
        assert loader_args[0] == f"local-dir:{language_root.resolve()}"
        loader_calls.append("tokenizer")
        return object()

    monkeypatch.setattr(open_clip, "create_model_from_pretrained", create_loader)
    monkeypatch.setattr(open_clip, "get_tokenizer", tokenizer_loader)

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            assert torch.is_grad_enabled() is False
            open_clip.create_model_from_pretrained(
                model_name="ViT-SO400M-14-SigLIP-384",
                pretrained="webli",
                return_transform=False,
            )
            open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

        def encode_labels(self, labels: list[str]) -> torch.Tensor:
            assert torch.is_grad_enabled() is False
            return graph_text_embeddings

        def insert_labels_into_templates(self, labels: list[str]) -> list[list[str]]:
            return [[f"a photo of {label}"] for label in labels]

    monkeypatch.setattr(
        worker_module,
        "_import_pinned_module",
        lambda root, module_name: SimpleNamespace(NARadioEncoder=Encoder),
    )
    validation_calls: list[str] = []

    def validate_source(root: Path, spec: Any) -> str:
        validation_calls.append(spec.name)
        return spec.commit

    monkeypatch.setattr(worker_module, "validate_source_checkout", validate_source)
    try:
        worker = build_worker(args)
        assert loader_calls == ["model", "tokenizer"]
        assert worker.runtime.text_embeddings.requires_grad is False
        assert worker.runtime.text_embeddings.grad_fn is None
        assert validation_calls == ["RayFronts", "RADIO", "RayFronts", "RADIO"]
    finally:
        _chmod_tree(language_root, writable=True)


@pytest.mark.parametrize("mutated_asset", ["radio", "sam"])
def test_build_worker_rejects_model_asset_changed_before_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_asset: str,
) -> None:
    import radseg_dense_worker as worker_module

    language_assets = _language_assets(tmp_path)
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps({"classes": _classes()}), encoding="utf-8")
    radio_checkpoint = tmp_path / "radio.pt"
    radio_checkpoint.write_bytes(b"radio-before")
    sam_checkpoint = tmp_path / "sam.pt"
    sam_checkpoint.write_bytes(b"sam-before")
    for path in (radio_checkpoint, sam_checkpoint):
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)

    args = parse_args(_base_cli(tmp_path))
    args.model_version = str(radio_checkpoint)
    args.classes_json = classes_path
    args.language_model_root = language_assets.root
    args.language_model_sha256 = language_assets.sha256
    if mutated_asset == "sam":
        args.sam_refinement = True
        args.sam_checkpoint = sam_checkpoint

    class Encoder:
        text_embeds = torch.ones((41, 3), dtype=torch.float32)

        def insert_labels_into_templates(self, labels: list[str]) -> list[list[str]]:
            return [[label] for label in labels]

    target = radio_checkpoint if mutated_asset == "radio" else sam_checkpoint

    def instantiate(*unused: Any, **unused_kwargs: Any) -> Encoder:
        target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)
        target.write_bytes(f"{mutated_asset}-after".encode("ascii"))
        target.chmod(stat.S_IMODE(target.stat().st_mode) & ~0o222)
        return Encoder()

    monkeypatch.setattr(
        worker_module,
        "validate_language_model_assets",
        lambda unused_args: language_assets,
    )
    monkeypatch.setattr(
        worker_module,
        "pinned_language_model_load",
        lambda unused_assets: nullcontext(),
    )
    monkeypatch.setattr(
        worker_module,
        "validate_source_checkout",
        lambda root, spec: spec.commit,
    )
    monkeypatch.setattr(
        worker_module,
        "_import_pinned_module",
        lambda root, module_name: SimpleNamespace(RADSegEncoder=Encoder),
    )
    monkeypatch.setattr(worker_module, "_instantiate_radseg", instantiate)
    provenance_hash_calls: list[Any] = []
    monkeypatch.setattr(
        worker_module,
        "canonical_sha256",
        lambda payload: provenance_hash_calls.append(payload) or "0" * 64,
    )
    try:
        with pytest.raises(RuntimeError, match=f"{mutated_asset.upper()}|checkpoint"):
            build_worker(args)
        assert provenance_hash_calls == []
    finally:
        _chmod_tree(language_assets.root, writable=True)


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
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
    }


def test_metadata_response_supports_scannet200_class_count() -> None:
    worker = DenseWorker(
        runtime=_FakeRuntime(
            np.ones((1, 200, 2, 2), dtype=np.float32) / 200.0
        ),
        classes=tuple(_classes(200)),
        sample_stride=1,
        top_k=4,
        provenance=_provenance(),
    )

    response = run_request(worker, {"id": 1, "operation": "metadata"})

    assert response["class_count"] == 200
    assert response["classes"] == _classes(200)


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


def test_infer_budget_rejects_before_runtime_allocation() -> None:
    class CountingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def infer_probabilities(self, rgb: np.ndarray) -> np.ndarray:
            self.calls += 1
            return super().infer_probabilities(rgb)

    runtime = CountingRuntime()
    worker = DenseWorker(
        runtime=runtime,
        classes=tuple(_classes()),
        sample_stride=1,
        top_k=41,
        provenance=_provenance(),
    )
    rgb = np.zeros((680, 1200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="response|budget"):
        run_request(
            worker,
            {"id": 1, "operation": "infer", "rgb": _array_block(rgb)},
        )
    assert runtime.calls == 0


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


def test_serve_jsonl_bounded_read_drains_oversized_line_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radseg_dense_worker as worker_module

    limit = 80
    monkeypatch.setattr(worker_module, "MAX_JSONL_LINE_CHARS", limit)
    valid = json.dumps({"id": 2, "operation": "metadata"}) + "\n"

    class ReadlineOnly(io.StringIO):
        def __iter__(self) -> Any:
            raise AssertionError("serve_jsonl must not use unbounded iteration")

        def readline(self, size: int = -1) -> str:
            assert 0 < size <= limit + 1
            return super().readline(size)

    input_stream = ReadlineOnly("x" * (limit * 3) + "\n" + valid)
    output_stream = io.StringIO()
    serve_jsonl(_fake_worker(), input_stream, output_stream, io.StringIO())

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert len(responses) == 2
    assert responses[0]["id"] is None
    assert responses[0]["ok"] is False
    assert "line" in responses[0]["error"]
    assert responses[1]["id"] == 2
    assert responses[1]["ok"] is True


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
