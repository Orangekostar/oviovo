from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import weakref
import zipfile

import numpy as np
import pytest

import src.oviv2.compact_checkpoint as compact_module
from src.oviv2.compact_checkpoint import (
    COMPACT_OWNERSHIP_FORMAT,
    CompactOwnershipCheckpoint,
    CompactOwnershipMetadata,
)
from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig
from src.oviv2.dense_projection import DenseSemanticConfig


def _provenance() -> DenseSemanticProvenance:
    return DenseSemanticProvenance(
        backend="radseg",
        source_commit="1" * 40,
        radio_commit="2" * 40,
        model_id="radseg:test",
        model_sha256="3" * 64,
        auxiliary_model_sha256="4" * 64,
        vocabulary_sha256="5" * 64,
        prompt_sha256="6" * 64,
        inference_config_sha256="7" * 64,
        cache_prefix_sha256="8" * 64,
        language_model_id="clip:test",
        language_model_revision="9" * 40,
        language_model_sha256="a" * 64,
    )


def _metadata() -> CompactOwnershipMetadata:
    return CompactOwnershipMetadata(
        scene_id="apartment",
        frame_id=7,
        timestamp=1.25,
        revision=9,
        voxel_size_m=0.05,
        block_resolution=8,
        dense_semantic_provenance=_provenance(),
    )


def _ownership() -> ReversibleOwnershipStore:
    ownership = ReversibleOwnershipStore(block_resolution=8)
    ownership.assign((1, 2, 3), 11, 0.75, 4)
    ownership.assign((-9, 2, 17), 12, 0.5, 5)
    ownership.assign((1, 2, 3), 11, 0.8, 6)
    return ownership


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def _staging_path(target: Path) -> Path:
    return target.parent / f".{target.name}.compact-staging"


def _rewrite_ownership_archive(
    target: Path,
    replacements: dict[str, bytes],
) -> None:
    ownership_path = target / "ownership.npz"
    with zipfile.ZipFile(ownership_path, mode="r") as source:
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
        }
    members.update(replacements)
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    ownership_path.write_bytes(destination.getvalue())
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["ownership.npz"] = hashlib.sha256(destination.getvalue()).hexdigest()
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    destination = io.BytesIO()
    np.lib.format.write_array(destination, array, allow_pickle=False)
    return destination.getvalue()


def _replace_directory_with_hardlinked_members(
    parent_fd: int,
    source_name: str,
    saved_name: str,
) -> None:
    os.rename(
        source_name,
        saved_name,
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
    )
    os.mkdir(source_name, mode=0o700, dir_fd=parent_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    saved_fd = os.open(saved_name, flags, dir_fd=parent_fd)
    replacement_fd = os.open(source_name, flags, dir_fd=parent_fd)
    try:
        for name in ("metadata.json", "ownership.npz", "checksums.json"):
            os.link(
                name,
                name,
                src_dir_fd=saved_fd,
                dst_dir_fd=replacement_fd,
                follow_symlinks=False,
            )
    finally:
        os.close(replacement_fd)
        os.close(saved_fd)


def test_compact_checkpoint_round_trip_is_exact_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = CompactOwnershipCheckpoint.commit_new(
        tmp_path / "first", _metadata(), _ownership()
    )
    second = CompactOwnershipCheckpoint.commit_new(
        tmp_path / "second", _metadata(), _ownership()
    )

    assert {path.name for path in (tmp_path / "first").iterdir()} == {
        "metadata.json",
        "ownership.npz",
        "checksums.json",
    }
    assert _tree_bytes(tmp_path / "first") == _tree_bytes(tmp_path / "second")
    assert not _staging_path(tmp_path / "first").exists()
    assert not _staging_path(tmp_path / "second").exists()
    assert first.metadata.format == COMPACT_OWNERSHIP_FORMAT
    assert first.metadata.schema_version == 1
    assert asdict(first.metadata.dense_semantic_provenance) == asdict(_provenance())
    assert first.ownership.records() == _ownership().records()
    assert first.checksums == second.checksums
    first.revalidate_source()


def test_compact_checkpoint_canonicalizes_cross_block_record_order(
    tmp_path: Path,
) -> None:
    ownership = ReversibleOwnershipStore(block_resolution=8)
    ownership.assign((-16, -2, 0), 11, 0.75, 1)
    ownership.assign((-16, -1, -2), 12, 0.75, 1)
    assert [key for key, _ in ownership.records()] != sorted(
        key for key, _ in ownership.records()
    )

    checkpoint = CompactOwnershipCheckpoint.commit_new(
        tmp_path / "checkpoint", _metadata(), ownership
    )

    assert set(checkpoint.ownership.records()) == set(ownership.records())


def test_compact_checkpoint_is_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    before = _tree_bytes(target)

    with pytest.raises(FileExistsError):
        CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())

    assert _tree_bytes(target) == before


@pytest.mark.parametrize("attack", ["extra", "checksum", "symlink"])
def test_compact_checkpoint_rejects_inventory_checksum_and_symlink_attacks(
    tmp_path: Path,
    attack: str,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    if attack == "extra":
        (target / "geometry.npz").write_bytes(b"forbidden")
    elif attack == "checksum":
        (target / "ownership.npz").write_bytes(b"drift")
    else:
        original = target / "metadata.json"
        moved = tmp_path / "metadata.json"
        original.rename(moved)
        original.symlink_to(moved)

    with pytest.raises(ValueError, match="inventory|checksum|regular|symlink"):
        CompactOwnershipCheckpoint.load(target)


@pytest.mark.parametrize(
    "content, message",
    [
        (
            b'{"format":"oviv2_compact_ownership_checkpoint","format":"fake"}',
            "duplicate JSON key",
        ),
        (
            b'{"format":"oviv2_compact_ownership_checkpoint","timestamp":NaN}',
            "non-finite JSON",
        ),
    ],
)
def test_compact_checkpoint_rejects_duplicate_and_nonfinite_metadata(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    metadata_path = target / "metadata.json"
    metadata_path.write_bytes(content)
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = hashlib.sha256(content).hexdigest()
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CompactOwnershipCheckpoint.load(target)


def test_compact_checkpoint_revalidation_detects_replacement(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    loaded = CompactOwnershipCheckpoint.commit_new(
        target, _metadata(), _ownership()
    )
    replacement = tmp_path / "replacement"
    CompactOwnershipCheckpoint.commit_new(replacement, _metadata(), _ownership())
    original = tmp_path / "original"
    target.rename(original)
    replacement.rename(target)

    with pytest.raises(ValueError, match="changed|identity"):
        loaded.revalidate_source()


def test_compact_checkpoint_revalidation_rejects_hardlink_swap_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    for source in target.iterdir():
        os.link(source, replacement / source.name)
    loaded = CompactOwnershipCheckpoint.load(target)
    original_lstat = compact_module.os.lstat
    swapped = False

    def swap_after_identity_read(path: str | Path, *args: object, **kwargs: object):
        nonlocal swapped
        status = original_lstat(path, *args, **kwargs)
        if Path(path) == target and not swapped:
            target.rename(tmp_path / "original")
            replacement.rename(target)
            swapped = True
        return status

    monkeypatch.setattr(compact_module.os, "lstat", swap_after_identity_read)

    with pytest.raises(ValueError, match="changed|identity"):
        loaded.revalidate_source()


def test_compact_checkpoint_rejects_nonfinite_ownership(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    with np.load(target / "ownership.npz", allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    arrays["confidence"][0] = np.nan
    np.savez_compressed(target / "ownership.npz", **arrays)
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["ownership.npz"] = hashlib.sha256(
        (target / "ownership.npz").read_bytes()
    ).hexdigest()
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(ValueError, match="confidence.*finite"):
        CompactOwnershipCheckpoint.load(target)


def test_compact_checkpoint_rejects_evidence_revision_after_checkpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    with np.load(target / "ownership.npz", allow_pickle=False) as payload:
        revisions = np.array(payload["evidence_revisions"], copy=True)
    revisions[0] = _metadata().revision + 1
    _rewrite_ownership_archive(
        target,
        {"evidence_revisions.npy": _npy_bytes(revisions)},
    )

    with pytest.raises(ValueError, match="evidence_revisions.*checkpoint revision"):
        CompactOwnershipCheckpoint.load(target)


@pytest.mark.parametrize(
    "limit_name,limit_value",
    [
        ("_MAX_ARCHIVE_BYTES", 1),
        ("_MAX_MEMBER_COMPRESSED_BYTES", 1),
        ("_MAX_MEMBER_UNCOMPRESSED_BYTES", 1),
        ("_MAX_TOTAL_UNCOMPRESSED_BYTES", 1),
        ("_MAX_CENTRAL_DIRECTORY_BYTES", 1),
        ("_MAX_NPY_HEADER_BYTES", 1),
        ("_MAX_OWNERSHIP_RECORDS", 1),
        ("_MAX_JSON_BYTES", 1),
    ],
)
def test_compact_checkpoint_preflights_resource_budgets_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    monkeypatch.setattr(compact_module, limit_name, limit_value)
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before compact archive preflight")

    monkeypatch.setattr(compact_module.np, "load", forbidden_load)

    with pytest.raises(ValueError, match="resource|budget|size|limit|header|record"):
        CompactOwnershipCheckpoint.load(target)

    assert not called


def test_compact_checkpoint_rejects_block_budget_before_dense_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    monkeypatch.setattr(compact_module, "_MAX_OWNERSHIP_BLOCKS", 1)
    called = False

    def forbidden_new_block(self: ReversibleOwnershipStore) -> object:
        nonlocal called
        called = True
        raise AssertionError("dense blocks must not allocate before block budget check")

    monkeypatch.setattr(ReversibleOwnershipStore, "_new_block", forbidden_new_block)

    with pytest.raises(ValueError, match="block count.*resource limit"):
        CompactOwnershipCheckpoint.load(target)

    assert not called


def test_compact_checkpoint_rejects_dense_storage_budget_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    monkeypatch.setattr(compact_module, "_MAX_DENSE_OWNERSHIP_BYTES", 1)
    called = False

    def forbidden_new_block(self: ReversibleOwnershipStore) -> object:
        nonlocal called
        called = True
        raise AssertionError("dense blocks must not allocate before storage budget check")

    monkeypatch.setattr(ReversibleOwnershipStore, "_new_block", forbidden_new_block)

    with pytest.raises(ValueError, match="dense block storage.*resource limit"):
        CompactOwnershipCheckpoint.load(target)

    assert not called


def test_compact_checkpoint_rejects_huge_block_resolution_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    huge_resolution = 1_000_000
    _rewrite_ownership_archive(
        target,
        {
            "block_resolution.npy": _npy_bytes(
                np.asarray([huge_resolution], dtype=np.int64)
            )
        },
    )
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["block_resolution"] = huge_resolution
    metadata_bytes = compact_module._canonical_json(metadata)
    metadata_path.write_bytes(metadata_bytes)
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = hashlib.sha256(metadata_bytes).hexdigest()
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    called = False

    def forbidden_new_block(self: ReversibleOwnershipStore) -> object:
        nonlocal called
        called = True
        raise AssertionError("huge dense block must not allocate")

    monkeypatch.setattr(ReversibleOwnershipStore, "_new_block", forbidden_new_block)

    with pytest.raises(ValueError, match="dense block storage.*resource limit"):
        CompactOwnershipCheckpoint.load(target)

    assert not called


@pytest.mark.parametrize(
    "member_name,replacement,message",
    [
        (
            "voxel_keys.npy",
            _npy_bytes(np.zeros((2, 3), dtype=np.float32)),
            "voxel_keys.*dtype",
        ),
        (
            "entity_ids.npy",
            _npy_bytes(np.zeros((2, 1), dtype=np.int64)),
            "entity_ids.*shape",
        ),
    ],
)
def test_compact_checkpoint_preflights_npy_contract_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
    replacement: bytes,
    message: str,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    _rewrite_ownership_archive(target, {member_name: replacement})
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before NPY contract preflight")

    monkeypatch.setattr(compact_module.np, "load", forbidden_load)

    with pytest.raises(ValueError, match=message):
        CompactOwnershipCheckpoint.load(target)

    assert not called


def test_compact_checkpoint_rejects_forged_huge_shape_before_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        forged,
        {
            "descr": np.dtype(np.int64).str,
            "fortran_order": False,
            "shape": (compact_module._MAX_OWNERSHIP_RECORDS + 1, 3),
        },
    )
    _rewrite_ownership_archive(target, {"voxel_keys.npy": forged.getvalue()})
    called = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before shape budget preflight")

    monkeypatch.setattr(compact_module.np, "load", forbidden_load)

    with pytest.raises(ValueError, match="voxel_keys.*record|shape.*resource"):
        CompactOwnershipCheckpoint.load(target)

    assert not called


def test_compact_checkpoint_100k_records_is_deterministic_and_compact(
    tmp_path: Path,
) -> None:
    ownership = ReversibleOwnershipStore(block_resolution=8)
    record_count = 100_000
    for index in range(record_count):
        x = index % 50 - 25
        y = (index // 50) % 40 - 20
        z = index // 2_000 - 25
        ownership.assign(
            (x, y, z),
            index % 127 + 1,
            (index % 101) / 100.0,
            index % (_metadata().revision + 1),
        )

    CompactOwnershipCheckpoint.commit_new(
        tmp_path / "first", _metadata(), ownership
    )
    CompactOwnershipCheckpoint.commit_new(
        tmp_path / "second", _metadata(), ownership
    )

    first = (tmp_path / "first/ownership.npz").read_bytes()
    second = (tmp_path / "second/ownership.npz").read_bytes()
    assert first == second
    assert len(first) < 2 * 1024 * 1024
    assert len(CompactOwnershipCheckpoint.load(tmp_path / "first").ownership.records()) == (
        record_count
    )
    assert compact_module._OWNERSHIP_COMPRESSION_LEVEL == 6


def test_compact_checkpoint_max_writer_contract_fits_resource_budgets() -> None:
    assert compact_module._MAX_OWNERSHIP_RECORDS >= 1_000_000
    assert compact_module._MAX_DENSE_OWNERSHIP_BYTES >= 512 * 1024 * 1024

    budget = compact_module._canonical_archive_budget(
        compact_module._MAX_OWNERSHIP_RECORDS
    )

    assert (
        budget.max_member_uncompressed_bytes
        <= compact_module._MAX_MEMBER_UNCOMPRESSED_BYTES
    )
    assert (
        budget.max_member_compressed_bytes
        <= compact_module._MAX_MEMBER_COMPRESSED_BYTES
    )
    assert (
        budget.total_uncompressed_bytes
        <= compact_module._MAX_TOTAL_UNCOMPRESSED_BYTES
    )
    assert budget.archive_bytes <= compact_module._MAX_ARCHIVE_BYTES


def test_compact_checkpoint_writer_materializes_ownership_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = compact_module._materialize_ownership
    calls = 0

    def counted(*args: object, **kwargs: object) -> ReversibleOwnershipStore:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(compact_module, "_materialize_ownership", counted)

    checkpoint = CompactOwnershipCheckpoint.commit_new(
        tmp_path / "checkpoint", _metadata(), _ownership()
    )

    assert calls == 1
    assert isinstance(checkpoint.ownership, ReversibleOwnershipStore)


def test_compact_checkpoint_commit_close_failure_does_not_mask_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "checkpoint"
    parent_identity = compact_module._identity(os.stat(parent))
    real_close = compact_module.os.close
    writer_failed = False

    def fail_writer(*_args: object, **_kwargs: object) -> tuple[int, int]:
        nonlocal writer_failed
        writer_failed = True
        raise RuntimeError("primary writer failure")

    def fail_parent_close(descriptor: int) -> None:
        identity = compact_module._identity(os.fstat(descriptor))
        real_close(descriptor)
        if writer_failed and identity == parent_identity:
            raise OSError("secondary close failure")

    monkeypatch.setattr(compact_module, "_write_regular_at", fail_writer)
    monkeypatch.setattr(compact_module.os, "close", fail_parent_close)

    with pytest.raises(RuntimeError, match="primary writer failure"):
        CompactOwnershipCheckpoint.commit_new(
            target,
            _metadata(),
            _ownership(),
        )

    assert not target.exists()
    assert _staging_path(target).is_dir()


def test_compact_checkpoint_file_close_failure_does_not_mask_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_close = compact_module.os.close

    def fail_write(_descriptor: int, _content: object) -> int:
        raise OSError("primary write failure")

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("secondary close failure")

    monkeypatch.setattr(compact_module.os, "write", fail_write)
    monkeypatch.setattr(compact_module.os, "close", fail_close)
    try:
        with pytest.raises(OSError, match="primary write failure"):
            compact_module._write_regular_at(directory_fd, "member", b"content")
    finally:
        real_close(directory_fd)

    assert (tmp_path / "member").is_file()


def test_compact_checkpoint_failed_staging_is_single_slot_and_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    writer_calls = 0

    def fail_writer(*_args: object, **_kwargs: object) -> tuple[int, int]:
        nonlocal writer_calls
        writer_calls += 1
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(compact_module, "_write_regular_at", fail_writer)

    with pytest.raises(RuntimeError, match="injected writer failure"):
        CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())
    with pytest.raises(FileExistsError):
        CompactOwnershipCheckpoint.commit_new(target, _metadata(), _ownership())

    assert writer_calls == 1
    assert not target.exists()
    assert _staging_path(target).is_dir()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_compact_checkpoint_concurrent_commits_are_no_clobber_and_single_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    barrier = threading.Barrier(2)

    def synchronize(stage: str) -> None:
        if stage == "after_parent_check":
            barrier.wait(timeout=10.0)

    def attempt() -> str:
        try:
            CompactOwnershipCheckpoint.commit_new(
                target,
                _metadata(),
                _ownership(),
            )
        except FileExistsError:
            return "exists"
        return "committed"

    monkeypatch.setattr(compact_module, "_publication_test_hook", synchronize)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(2)))

    assert sorted(outcomes) == ["committed", "exists"]
    CompactOwnershipCheckpoint.load(target)
    staging = _staging_path(target)
    if staging.exists():
        assert {path.name for path in staging.iterdir()} == {
            "metadata.json",
            "ownership.npz",
            "checksums.json",
        }
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_compact_checkpoint_directory_close_failure_does_not_mask_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = compact_module.os.close

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("secondary close failure")

    monkeypatch.setattr(compact_module.os, "close", fail_close)

    with pytest.raises(FileNotFoundError, match="missing"):
        compact_module._open_directory_without_symlinks(tmp_path / "missing")


def test_compact_checkpoint_file_initial_fstat_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = compact_module.os.open
    real_close = compact_module.os.close
    real_fstat = compact_module.os.fstat
    directory_fd = real_open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    member_fd: int | None = None
    member_closed = False

    def track_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal member_fd
        descriptor = real_open(path, *args, **kwargs)
        if path == "member":
            member_fd = descriptor
        return descriptor

    def fail_member_fstat(descriptor: int) -> os.stat_result:
        if descriptor == member_fd:
            raise OSError("primary member fstat failure")
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        nonlocal member_closed
        if descriptor == member_fd:
            member_closed = True
        real_close(descriptor)

    try:
        monkeypatch.setattr(compact_module.os, "open", track_open)
        monkeypatch.setattr(compact_module.os, "fstat", fail_member_fstat)
        monkeypatch.setattr(compact_module.os, "close", track_close)
        with pytest.raises(OSError, match="primary member fstat failure"):
            compact_module._write_regular_at(directory_fd, "member", b"content")
    finally:
        if member_fd is not None and not member_closed:
            real_close(member_fd)
        real_close(directory_fd)
        (tmp_path / "member").unlink(missing_ok=True)

    assert member_closed


def test_compact_checkpoint_parent_initial_fstat_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "checkpoint"
    real_close = compact_module.os.close
    real_fstat = compact_module.os.fstat
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    parent_closed = False

    def return_parent_fd(_path: Path) -> int:
        return parent_fd

    def fail_parent_fstat(descriptor: int) -> os.stat_result:
        if descriptor == parent_fd:
            raise OSError("primary parent fstat failure")
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        nonlocal parent_closed
        if descriptor == parent_fd:
            parent_closed = True
        real_close(descriptor)

    try:
        monkeypatch.setattr(
            compact_module,
            "_open_directory_without_symlinks",
            return_parent_fd,
        )
        monkeypatch.setattr(compact_module.os, "fstat", fail_parent_fstat)
        monkeypatch.setattr(compact_module.os, "close", track_close)
        with pytest.raises(OSError, match="primary parent fstat failure"):
            CompactOwnershipCheckpoint.commit_new(
                target,
                _metadata(),
                _ownership(),
            )
    finally:
        if not parent_closed:
            real_close(parent_fd)

    assert parent_closed
    assert not target.exists()


def test_compact_checkpoint_receipt_validates_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_materializer = compact_module._materialize_ownership

    def forbidden_materializer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("receipt commit must not materialize ownership")

    monkeypatch.setattr(
        compact_module,
        "_materialize_ownership",
        forbidden_materializer,
    )

    receipt = CompactOwnershipCheckpoint.commit_receipt_new(
        tmp_path / "checkpoint", _metadata(), _ownership()
    )

    assert isinstance(receipt, compact_module.CompactOwnershipCommitReceipt)
    assert not hasattr(receipt, "ownership")
    assert receipt.source_witness.path == receipt.path
    assert all(
        value is not receipt
        for value in vars(receipt.source_witness).values()
    )
    receipt.revalidate_source()
    monkeypatch.setattr(
        compact_module,
        "_materialize_ownership",
        original_materializer,
    )
    assert CompactOwnershipCheckpoint.load(receipt.path).ownership.records() == (
        _ownership().records()
    )


def test_compact_checkpoint_receipt_does_not_retain_runtime_ownership(
    tmp_path: Path,
) -> None:
    runtime = Oviv2Runtime(
        "apartment",
        Oviv2RuntimeConfig(
            dense_semantics=DenseSemanticConfig(voxel_size_m=0.05)
        ),
        dense_semantic_provenance=_provenance(),
    )
    runtime.last_frame_id = 4
    runtime.last_timestamp = 1.5
    runtime.revision = 6
    runtime.ownership.assign((1, 2, 3), 11, 0.8, 6)
    ownership_reference = weakref.ref(runtime.ownership)

    receipt = runtime.commit_compact_ownership_new(tmp_path / "compact")
    del runtime
    gc.collect()

    assert ownership_reference() is None
    assert not hasattr(receipt, "ownership")
    receipt.source_witness.revalidate()


def test_compact_checkpoint_five_100k_receipts_are_deterministic_and_loadable(
    tmp_path: Path,
) -> None:
    ownership = ReversibleOwnershipStore(block_resolution=8)
    record_count = 100_000
    for index in range(record_count):
        ownership.assign(
            (
                index % 50 - 25,
                (index // 50) % 40 - 20,
                index // 2_000 - 25,
            ),
            index % 127 + 1,
            (index % 101) / 100.0,
            index % (_metadata().revision + 1),
        )

    receipts = [
        CompactOwnershipCheckpoint.commit_receipt_new(
            tmp_path / f"checkpoint-{frame_id}",
            replace(_metadata(), frame_id=frame_id),
            ownership,
        )
        for frame_id in range(5)
    ]

    archives = [
        (receipt.path / "ownership.npz").read_bytes()
        for receipt in receipts
    ]
    assert all(archive == archives[0] for archive in archives[1:])
    assert len(
        CompactOwnershipCheckpoint.load(receipts[-1].path).ownership.records()
    ) == record_count


def test_compact_checkpoint_writer_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        CompactOwnershipCheckpoint.commit_new(
            linked / "checkpoint", _metadata(), _ownership()
        )
    assert not os.path.lexists(real / "checkpoint")


@pytest.mark.parametrize(
    "swap_stage",
    ["after_parent_check", "after_temp_create", "before_rename"],
)
def test_compact_checkpoint_writer_rejects_parent_swap_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_stage: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    old_parent = tmp_path / "old-parent"
    swapped = False

    def swap_at_stage(stage: str) -> None:
        nonlocal swapped
        if stage != swap_stage or swapped:
            return
        parent.rename(old_parent)
        parent.mkdir()
        (parent / "foreign.txt").write_text("do not delete", encoding="utf-8")
        swapped = True

    monkeypatch.setattr(compact_module, "_publication_test_hook", swap_at_stage)

    with pytest.raises(ValueError, match="parent.*identity.*changed"):
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert swapped
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "do not delete"
    assert not (parent / "checkpoint").exists()
    assert not (old_parent / "checkpoint").exists()
    old_staging = _staging_path(old_parent / "checkpoint")
    assert old_staging.exists() is (swap_stage != "after_parent_check")


def test_compact_checkpoint_parent_swap_after_rename_is_publication_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    old_parent = tmp_path / "old-parent"

    def swap_after_rename(stage: str) -> None:
        if stage != "after_rename":
            return
        parent.rename(old_parent)
        parent.mkdir()
        foreign_target = parent / "checkpoint"
        foreign_target.mkdir()
        (foreign_target / "foreign.txt").write_text(
            "do not delete", encoding="utf-8"
        )

    monkeypatch.setattr(
        compact_module,
        "_publication_test_hook",
        swap_after_rename,
    )

    with pytest.raises(
        compact_module.CompactCheckpointPublicationUncertainError
    ) as raised:
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert raised.value.published is True
    assert (old_parent / "checkpoint/ownership.npz").is_file()
    assert (parent / "checkpoint/foreign.txt").read_text(encoding="utf-8") == (
        "do not delete"
    )


def test_compact_checkpoint_rename_race_is_no_clobber_and_preserves_foreign_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()

    def insert_foreign_target(stage: str) -> None:
        if stage != "before_rename":
            return
        target = parent / "checkpoint"
        target.mkdir()
        (target / "foreign.txt").write_text("do not delete", encoding="utf-8")

    monkeypatch.setattr(
        compact_module,
        "_publication_test_hook",
        insert_foreign_target,
    )

    with pytest.raises(FileExistsError):
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert (parent / "checkpoint/foreign.txt").read_text(encoding="utf-8") == (
        "do not delete"
    )
    assert _staging_path(parent / "checkpoint").is_dir()


def test_compact_checkpoint_rename_helper_rechecks_parent_before_syscall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    old_parent = tmp_path / "old-parent"
    original = compact_module._rename_directory_no_replace_at
    swapped = False

    def swap_before_syscall(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            parent.rename(old_parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text(
                "do not delete", encoding="utf-8"
            )
            swapped = True
        original(*args, **kwargs)

    monkeypatch.setattr(
        compact_module,
        "_rename_directory_no_replace_at",
        swap_before_syscall,
    )

    with pytest.raises(ValueError, match="parent.*identity.*changed"):
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "do not delete"
    assert not (old_parent / "checkpoint").exists()
    assert _staging_path(old_parent / "checkpoint").is_dir()


def test_compact_checkpoint_rename_helper_rejects_replaced_temp_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original = compact_module._rename_directory_no_replace_at
    saved_name = ".saved-original"

    def replace_before_helper(
        parent_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        _replace_directory_with_hardlinked_members(
            parent_fd,
            source_name,
            saved_name,
        )
        original(parent_fd, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        compact_module,
        "_rename_directory_no_replace_at",
        replace_before_helper,
    )

    with pytest.raises(ValueError, match="temporary source identity changed"):
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
    )

    assert not (parent / "checkpoint").exists()
    assert (parent / saved_name).is_dir()
    foreign_staging = _staging_path(parent / "checkpoint")
    assert foreign_staging.is_dir()
    assert {path.name for path in foreign_staging.iterdir()} == {
        "metadata.json",
        "ownership.npz",
        "checksums.json",
    }


def test_compact_checkpoint_replaced_temp_after_stat_is_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    saved_name = ".saved-original"
    replaced = False

    def replace_after_identity_check(stage: str) -> None:
        nonlocal replaced
        if stage != "after_source_identity_check" or replaced:
            return
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            source_names = [
                name
                for name in os.listdir(parent_fd)
                if name == ".checkpoint.compact-staging"
            ]
            assert len(source_names) == 1
            _replace_directory_with_hardlinked_members(
                parent_fd,
                source_names[0],
                saved_name,
            )
        finally:
            os.close(parent_fd)
        replaced = True

    monkeypatch.setattr(
        compact_module,
        "_publication_test_hook",
        replace_after_identity_check,
    )

    with pytest.raises(
        compact_module.CompactCheckpointPublicationUncertainError
    ) as raised:
        CompactOwnershipCheckpoint.commit_receipt_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert replaced
    assert raised.value.published is True
    assert (parent / "checkpoint/ownership.npz").is_file()
    assert (parent / saved_name / "ownership.npz").is_file()
    assert os.stat(parent / "checkpoint").st_ino != os.stat(
        parent / saved_name
    ).st_ino


@pytest.mark.parametrize(
    "failure_point",
    ["open", "stat_before_open", "fstat", "stat_after_open"],
)
def test_compact_checkpoint_temp_creation_failure_preserves_single_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original_open = compact_module.os.open
    original_stat = compact_module.os.stat
    original_fstat = compact_module.os.fstat
    temporary_fds: list[int] = []
    temporary_stat_calls = 0

    def is_temporary_name(path: object) -> bool:
        return (
            isinstance(path, (str, bytes))
            and os.fsdecode(path) == ".checkpoint.compact-staging"
        )

    def controlled_open(path: object, *args: object, **kwargs: object) -> int:
        if failure_point == "open" and is_temporary_name(path):
            raise OSError("injected temp open failure")
        descriptor = original_open(path, *args, **kwargs)
        if is_temporary_name(path):
            temporary_fds.append(descriptor)
        return descriptor

    def controlled_stat(path: object, *args: object, **kwargs: object):
        nonlocal temporary_stat_calls
        if is_temporary_name(path):
            temporary_stat_calls += 1
            if failure_point == "stat_before_open" and temporary_stat_calls == 1:
                raise OSError("injected temp stat failure")
            if failure_point == "stat_after_open" and temporary_stat_calls == 2:
                raise OSError("injected temp stat failure")
        return original_stat(path, *args, **kwargs)

    def controlled_fstat(descriptor: int):
        if failure_point == "fstat" and descriptor in temporary_fds:
            raise OSError("injected temp fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(compact_module.os, "open", controlled_open)
    monkeypatch.setattr(compact_module.os, "stat", controlled_stat)
    monkeypatch.setattr(compact_module.os, "fstat", controlled_fstat)

    with pytest.raises(OSError, match="injected temp"):
        CompactOwnershipCheckpoint.commit_new(
            parent / "checkpoint", _metadata(), _ownership()
        )

    assert _staging_path(parent / "checkpoint").is_dir()
    assert not list(parent.glob(".checkpoint.tmp-*"))
    for descriptor in temporary_fds:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_runtime_commits_compact_receipt_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Oviv2Runtime(
        "apartment",
        Oviv2RuntimeConfig(
            dense_semantics=DenseSemanticConfig(voxel_size_m=0.05)
        ),
        dense_semantic_provenance=_provenance(),
    )
    runtime.last_frame_id = 4
    runtime.last_timestamp = 1.5
    runtime.revision = 6
    runtime.ownership.assign((1, 2, 3), 11, 0.8, 6)
    original_materializer = compact_module._materialize_ownership

    def forbidden_materializer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime receipt must not materialize ownership")

    monkeypatch.setattr(
        compact_module,
        "_materialize_ownership",
        forbidden_materializer,
    )

    receipt = runtime.commit_compact_ownership_new(tmp_path / "compact")

    assert isinstance(receipt, compact_module.CompactOwnershipCommitReceipt)
    assert receipt.metadata.frame_id == 4
    assert receipt.metadata.revision == 6
    assert not hasattr(receipt, "ownership")
    monkeypatch.setattr(
        compact_module,
        "_materialize_ownership",
        original_materializer,
    )
    loaded = CompactOwnershipCheckpoint.load(receipt.path)
    assert loaded.ownership.records() == runtime.ownership.records()
    assert not (tmp_path / "compact/geometry.npz").exists()
    assert not (tmp_path / "compact/evidence.npz").exists()
    assert not (tmp_path / "compact/entities.jsonl").exists()
