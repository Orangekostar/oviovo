from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

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


def test_runtime_commits_compact_ownership_without_full_snapshot(tmp_path: Path) -> None:
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

    checkpoint = runtime.commit_compact_ownership_new(tmp_path / "compact")

    assert checkpoint.metadata.frame_id == 4
    assert checkpoint.metadata.revision == 6
    assert checkpoint.ownership.records() == runtime.ownership.records()
    assert not (tmp_path / "compact/geometry.npz").exists()
    assert not (tmp_path / "compact/evidence.npz").exists()
    assert not (tmp_path / "compact/entities.jsonl").exists()
