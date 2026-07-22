from __future__ import annotations

from dataclasses import asdict, replace
import errno
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import weakref

import numpy as np
import pytest

import src.oviv2.snapshot as snapshot_module
from src.oviv2.dense_semantics import DenseSemanticProvenance
from src.oviv2.evidence import EvidenceConfig, SparseEvidenceStore
from src.oviv2.entities import EntityRegistry
from src.oviv2.geometry import SparseTsdfVolume, TsdfConfig
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata

from tests.oviv2.test_geometry import _integrate_twice, _plane_frame
from tests.oviv2.test_entities import _track


def _components():
    config = TsdfConfig()
    geometry = SparseTsdfVolume(config)
    depth, rgb, intrinsics = _plane_frame()
    _integrate_twice(geometry, depth, rgb, intrinsics, np.eye(4))
    evidence = SparseEvidenceStore(EvidenceConfig(block_resolution=config.block_resolution))
    ownership = ReversibleOwnershipStore(block_resolution=config.block_resolution)
    key = (0, 0, 20)
    evidence.update_semantic(key, 5, 2.0, 1)
    evidence.update_entity(key, 11, 2.0, 0.0, 1.0, 1)
    ownership.assign(key, 11, 0.8, 1)
    metadata = VoxelSnapshotMetadata(
        scene_id="room0",
        frame_id=10,
        timestamp=1.0,
        revision=1,
        voxel_size_m=config.voxel_size_m,
        block_resolution=config.block_resolution,
    )
    return metadata, geometry, evidence, ownership


def _registry() -> EntityRegistry:
    registry = EntityRegistry()
    original = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    registry.entities = {11: replace(original, entity_id=11)}
    registry._next_entity_id = 12
    return registry


def _snapshot_state(path: Path) -> tuple[int, dict[str, tuple[bytes, int, str]]]:
    return (
        path.stat().st_mtime_ns,
        {
            item.name: (
                item.read_bytes(),
                item.stat().st_mtime_ns,
                hashlib.sha256(item.read_bytes()).hexdigest(),
            )
            for item in sorted(path.iterdir())
        },
    )


def _snapshot_staging(target: Path) -> Path:
    return target.with_name(f".{target.name}.snapshot-staging")


def _snapshot_staging_candidates(target: Path) -> list[Path]:
    return sorted(target.parent.glob(f".{target.name}.*"))


def _single_preserved_staging_directory(target: Path) -> Path:
    staged = _snapshot_staging(target)
    assert staged.is_dir()
    assert _snapshot_staging_candidates(target) == [staged]
    return staged


def _dense_provenance() -> DenseSemanticProvenance:
    return DenseSemanticProvenance(
        backend="radseg",
        source_commit="a" * 40,
        radio_commit="b" * 40,
        model_id="nvidia/C-RADIOv3-H",
        model_sha256="c" * 64,
        auxiliary_model_sha256="d" * 64,
        vocabulary_sha256="e" * 64,
        prompt_sha256="f" * 64,
        inference_config_sha256="1" * 64,
        cache_prefix_sha256="2" * 64,
        language_model_id="google/siglip2-giant-opt-patch16-384",
        language_model_revision="3" * 40,
        language_model_sha256="4" * 64,
    )


def _rewrite_metadata_with_valid_checksum(
    target: Path,
    payload: dict[str, object],
) -> None:
    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


def test_snapshot_commit_writes_and_loads_complete_contract(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"

    committed = VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    restored = VoxelMapSnapshot.load(target)

    assert {path.name for path in target.iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "checksums.json",
    }
    assert committed.metadata == metadata
    assert restored.metadata == metadata
    assert restored.registry is None
    assert restored.geometry.active_block_count == geometry.active_block_count
    assert restored.evidence.entity_candidates((0, 0, 20)) == evidence.entity_candidates((0, 0, 20))
    assert restored.ownership.owner_of((0, 0, 20)) == ownership.owner_of((0, 0, 20))


def test_snapshot_commit_new_publishes_complete_contract(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"

    committed = VoxelMapSnapshot.commit_new(
        target,
        metadata,
        geometry,
        evidence,
        ownership,
    )

    assert committed.path == target
    assert VoxelMapSnapshot.load(target).metadata == metadata
    assert {path.name for path in target.iterdir()} == {
        *VoxelMapSnapshot._DATA_FILES_V1,
        "checksums.json",
    }
    assert _snapshot_staging_candidates(target) == []


def test_commit_new_returns_independent_revalidatable_source_witness(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    committed = VoxelMapSnapshot.commit_new(
        tmp_path / "snapshot", metadata, geometry, evidence, ownership
    )
    snapshot_ref = weakref.ref(committed)
    restored_geometry_ref = weakref.ref(committed.geometry)
    witness = committed.source_witness

    del committed
    gc.collect()

    assert witness is not None
    assert snapshot_ref() is None
    assert restored_geometry_ref() is None
    witness.revalidate()


def test_commit_new_rejects_equal_content_replacement_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_publish = VoxelMapSnapshot._publish_directory_no_replace

    def replace_after_publish(
        _cls,
        source: Path,
        destination: Path,
        **kwargs: object,
    ) -> None:
        original_publish(source, destination, **kwargs)
        displaced = destination.with_name("snapshot.displaced")
        destination.rename(displaced)
        shutil.copytree(displaced, destination)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_publish_directory_no_replace",
        classmethod(replace_after_publish),
    )

    with pytest.raises(snapshot_module.SnapshotPublicationUncertainError) as raised:
        VoxelMapSnapshot.commit_new(
            target, metadata, geometry, evidence, ownership
        )
    assert raised.value.published is True
    assert raised.value.target == target
    assert "post-publication verification failed" in str(raised.value)
    assert target.is_dir()
    assert (tmp_path / "snapshot.displaced").is_dir()


def test_source_witness_rejects_equal_content_new_inode_after_return(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    committed = VoxelMapSnapshot.commit_new(
        target, metadata, geometry, evidence, ownership
    )
    replacement = tmp_path / "replacement"
    shutil.copytree(target, replacement)
    target.rename(tmp_path / "original")
    replacement.rename(target)

    with pytest.raises(ValueError, match="snapshot source.*identity|changed"):
        committed.revalidate_source()


def test_source_witness_rejects_hardlink_directory_swap_during_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    replacement: Path | None = None
    original_write = VoxelMapSnapshot._write_snapshot_files

    def write_with_hardlink_replacement(
        _cls,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, ...]:
        nonlocal replacement
        data_files = original_write(destination, *args, **kwargs)
        replacement = tmp_path / "hardlink-replacement"
        replacement.mkdir()
        for member in destination.iterdir():
            os.link(member, replacement / member.name)
        return data_files

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_write_snapshot_files",
        classmethod(write_with_hardlink_replacement),
    )
    target = tmp_path / "snapshot"
    committed = VoxelMapSnapshot.commit_new(
        target, metadata, geometry, evidence, ownership
    )
    assert replacement is not None
    original_lstat = snapshot_module.os.lstat
    swapped = False

    def swap_after_lstat(path: object, *args: object, **kwargs: object):
        nonlocal swapped
        status = original_lstat(path, *args, **kwargs)
        if not swapped and Path(path) == target:
            target.rename(tmp_path / "original")
            replacement.rename(target)
            swapped = True
        return status

    monkeypatch.setattr(snapshot_module.os, "lstat", swap_after_lstat)

    with pytest.raises(ValueError, match="snapshot source.*identity|changed"):
        committed.revalidate_source()
    assert swapped is True


@pytest.mark.parametrize("replacement_mode", ["copy", "hardlink"])
def test_commit_new_rejects_self_consistent_temp_reown_after_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_mode: str,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    alternate_geometry = SparseTsdfVolume(geometry.config)
    alternate_evidence = SparseEvidenceStore(evidence.config)
    alternate_ownership = ReversibleOwnershipStore(
        block_resolution=ownership.block_resolution
    )
    alternate = VoxelMapSnapshot.commit(
        tmp_path / "alternate",
        metadata,
        alternate_geometry,
        alternate_evidence,
        alternate_ownership,
    )
    original_load = VoxelMapSnapshot.load
    replaced = False

    def replace_temp_after_load(
        _cls,
        snapshot_dir: str | Path,
    ) -> VoxelMapSnapshot:
        nonlocal replaced
        source = Path(snapshot_dir)
        restored = original_load(source)
        if source == _snapshot_staging(target) and not replaced:
            captured = tmp_path / "captured-original"
            displaced = tmp_path / "captured-twice"
            source.rename(captured)
            if replacement_mode == "copy":
                shutil.copytree(alternate.path, source)
            else:
                source.mkdir()
                for member in alternate.path.iterdir():
                    os.link(member, source / member.name)
            (source / "foreign-sentinel.txt").write_text(
                "must survive",
                encoding="utf-8",
            )
            captured.rename(displaced)
            replaced = True
        return restored

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(replace_temp_after_load),
    )
    target = tmp_path / "snapshot"

    with pytest.raises(ValueError, match="snapshot source.*identity|changed"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert replaced is True
    assert not target.exists()
    displaced = tmp_path / "captured-twice"
    assert original_load(displaced).metadata == metadata
    assert (_snapshot_staging(target) / "foreign-sentinel.txt").read_text(
        encoding="utf-8"
    ) == "must survive"


def test_commit_new_rejects_staging_replacement_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    alternate_geometry = SparseTsdfVolume(geometry.config)
    alternate_evidence = SparseEvidenceStore(evidence.config)
    alternate_ownership = ReversibleOwnershipStore(
        block_resolution=ownership.block_resolution
    )
    alternate = VoxelMapSnapshot.commit(
        tmp_path / "alternate",
        metadata,
        alternate_geometry,
        alternate_evidence,
        alternate_ownership,
    )
    target = tmp_path / "snapshot"
    displaced = tmp_path / "displaced-before-rename"
    original_rename = VoxelMapSnapshot._rename_directory_no_replace

    def replace_then_attempt_rename(
        _cls,
        source: Path,
        destination: Path,
        **kwargs: object,
    ) -> None:
        source.rename(displaced)
        shutil.copytree(alternate.path, source)
        original_rename(source, destination, **kwargs)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_rename_directory_no_replace",
        classmethod(replace_then_attempt_rename),
    )

    with pytest.raises(ValueError, match="staging.*identity"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    assert VoxelMapSnapshot.load(displaced).metadata == metadata
    assert VoxelMapSnapshot.load(_snapshot_staging(target)).metadata == metadata


def test_commit_new_failure_preserves_staged_and_unknown_foreign_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_load = VoxelMapSnapshot.load

    def inject_foreign_member_then_fail(
        _cls,
        snapshot_dir: str | Path,
    ) -> VoxelMapSnapshot:
        source = Path(snapshot_dir)
        if source == _snapshot_staging(target):
            (source / "foreign-sentinel.txt").write_text(
                "must survive",
                encoding="utf-8",
            )
            raise RuntimeError("injected pre-publication failure")
        return original_load(source)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(inject_foreign_member_then_fail),
    )

    with pytest.raises(RuntimeError, match="pre-publication failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    staged = _single_preserved_staging_directory(target)
    assert {path.name for path in staged.iterdir()} == {
        "checksums.json",
        "evidence.npz",
        "foreign-sentinel.txt",
        "geometry.npz",
        "metadata.json",
        "ownership.npz",
    }


def test_commit_new_failure_never_removes_replacement_or_displaced_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    displaced = tmp_path / "displaced-owned-stage"
    original_load = VoxelMapSnapshot.load

    def replace_stage_then_fail(
        _cls,
        snapshot_dir: str | Path,
    ) -> VoxelMapSnapshot:
        source = Path(snapshot_dir)
        source.rename(displaced)
        source.mkdir()
        raise RuntimeError("injected stage replacement")

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(replace_stage_then_fail),
    )

    with pytest.raises(RuntimeError, match="stage replacement"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    replacement = _single_preserved_staging_directory(target)
    assert replacement.is_dir()
    assert list(replacement.iterdir()) == []
    assert original_load(displaced).metadata == metadata


def test_commit_new_writes_only_to_opened_staging_after_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    staged = _snapshot_staging(target)
    displaced = tmp_path / "displaced-open-stage"
    original_write = VoxelMapSnapshot._write_snapshot_files

    def replace_before_write(
        _cls,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, ...]:
        staged.rename(displaced)
        staged.mkdir()
        (staged / "foreign-sentinel.txt").write_text(
            "must survive unchanged",
            encoding="utf-8",
        )
        return original_write(destination, *args, **kwargs)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_write_snapshot_files",
        classmethod(replace_before_write),
    )

    with pytest.raises(ValueError, match="inventory|identity"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    assert VoxelMapSnapshot.load(displaced).metadata == metadata
    assert {path.name for path in staged.iterdir()} == {"foreign-sentinel.txt"}
    assert (staged / "foreign-sentinel.txt").read_text(encoding="utf-8") == (
        "must survive unchanged"
    )


def test_snapshot_publication_uncertain_error_is_publicly_exported() -> None:
    from src.oviv2 import SnapshotPublicationUncertainError as exported_error
    from src.oviv2.snapshot import SnapshotPublicationUncertainError

    assert exported_error is SnapshotPublicationUncertainError


def test_snapshot_commit_new_rejects_missing_parent_directory(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    parent = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="missing"):
        VoxelMapSnapshot.commit_new(
            parent / "snapshot",
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not parent.exists()


def test_snapshot_commit_new_rejects_symlink_parent(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(NotADirectoryError):
        VoxelMapSnapshot.commit_new(
            linked_parent / "snapshot",
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert list(real_parent.iterdir()) == []


def test_snapshot_commit_new_existing_staging_blocks_without_mutation(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    staged = _snapshot_staging(target)
    staged.mkdir()
    (staged / "forensic.bin").write_bytes(b"preserve exactly")
    before = _snapshot_state(staged)

    with pytest.raises(FileExistsError):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    assert _snapshot_state(staged) == before


def test_snapshot_commit_new_existing_target_is_byte_for_byte_unchanged(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    before = _snapshot_state(target)

    with pytest.raises(FileExistsError):
        VoxelMapSnapshot.commit_new(
            target,
            replace(metadata, frame_id=20, revision=2),
            geometry,
            evidence,
            ownership,
        )

    assert _snapshot_state(target) == before
    assert VoxelMapSnapshot.load(target).metadata == metadata
    assert _snapshot_staging_candidates(target) == []


def test_snapshot_commit_new_publication_failure_preserves_staged_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"

    def fail_publish(
        _cls,
        _source: Path,
        _target: Path,
        **_kwargs: object,
    ) -> None:
        raise OSError("injected no-replace publication failure")

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_rename_directory_no_replace",
        classmethod(fail_publish),
    )

    with pytest.raises(OSError, match="no-replace publication failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    assert VoxelMapSnapshot.load(staged).metadata == metadata


def test_commit_new_parent_rebind_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.snapshot import SnapshotPublicationUncertainError

    metadata, geometry, evidence, ownership = _components()
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "snapshot"
    displaced_parent = tmp_path / "displaced-parent"
    original_open_owned = snapshot_module._open_owned_directory
    parent_open_calls = 0

    def rebind_after_first_parent_open(path: Path) -> tuple[int, tuple[int, int]]:
        nonlocal parent_open_calls
        result = original_open_owned(path)
        if path == parent:
            parent_open_calls += 1
            if parent_open_calls == 1:
                parent.rename(displaced_parent)
                parent.mkdir()
                (parent / "foreign-sentinel.txt").write_text(
                    "must survive",
                    encoding="utf-8",
                )
        return result

    monkeypatch.setattr(
        snapshot_module,
        "_open_owned_directory",
        rebind_after_first_parent_open,
    )

    with pytest.raises(SnapshotPublicationUncertainError) as raised:
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert raised.value.published is True
    assert parent_open_calls == 1
    assert not target.exists()
    assert {path.name for path in parent.iterdir()} == {"foreign-sentinel.txt"}
    assert VoxelMapSnapshot.load(displaced_parent / "snapshot").metadata == metadata


def test_commit_new_publication_uses_one_parent_fd_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_open_owned = snapshot_module._open_owned_directory
    parent_open_calls = 0
    opened_parent_fd: int | None = None

    def reject_second_parent_open(path: Path) -> tuple[int, tuple[int, int]]:
        nonlocal parent_open_calls, opened_parent_fd
        if path == target.parent:
            parent_open_calls += 1
            if parent_open_calls == 2:
                raise OSError("injected second parent open failure")
        result = original_open_owned(path)
        if path == target.parent:
            opened_parent_fd = result[0]
        return result

    monkeypatch.setattr(
        snapshot_module,
        "_open_owned_directory",
        reject_second_parent_open,
    )

    committed = VoxelMapSnapshot.commit_new(
        target,
        metadata,
        geometry,
        evidence,
        ownership,
    )

    assert committed.path == target
    assert parent_open_calls == 1
    assert opened_parent_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(opened_parent_fd)


def test_snapshot_commit_new_staging_failure_preserves_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_write_json = VoxelMapSnapshot._write_json

    def fail_manifest(path: Path, payload: dict) -> None:
        if path.name == "checksums.json":
            raise OSError("injected manifest failure")
        original_write_json(path, payload)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_write_json",
        staticmethod(fail_manifest),
    )

    with pytest.raises(OSError, match="manifest failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    assert {path.name for path in staged.iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
    }


def test_snapshot_commit_new_fails_closed_without_renameat2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    monkeypatch.setattr(snapshot_module.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(NotImplementedError, match="RENAME_NOREPLACE|renameat2"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    assert VoxelMapSnapshot.load(staged).metadata == metadata


@pytest.mark.parametrize(
    "error_number",
    [errno.ENOSYS, errno.EINVAL, getattr(errno, "EOPNOTSUPP", errno.ENOSYS)],
)
def test_snapshot_commit_new_fails_closed_for_unsupported_renameat2_errno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"

    class UnsupportedRenameAt2:
        def __call__(self, *_args) -> int:
            snapshot_module.ctypes.set_errno(error_number)
            return -1

    class Libc:
        renameat2 = UnsupportedRenameAt2()

    monkeypatch.setattr(snapshot_module.ctypes, "CDLL", lambda *_args, **_kwargs: Libc())

    with pytest.raises(NotImplementedError, match="RENAME_NOREPLACE"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    assert VoxelMapSnapshot.load(staged).metadata == metadata


def test_snapshot_commit_new_fallback_preserves_existing_nonempty_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    before = _snapshot_state(target)
    monkeypatch.setattr(snapshot_module.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(FileExistsError):
        VoxelMapSnapshot.commit_new(
            target,
            replace(metadata, frame_id=20, revision=2),
            geometry,
            evidence,
            ownership,
        )

    assert _snapshot_state(target) == before
    assert VoxelMapSnapshot.load(target).metadata == metadata
    assert _snapshot_staging_candidates(target) == []


def test_snapshot_commit_new_parent_fsync_failure_reports_published_uncertain_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.snapshot import SnapshotPublicationUncertainError

    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_fsync_directory = VoxelMapSnapshot._fsync_directory
    parent_fsync_calls = 0

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal parent_fsync_calls
        if path == target.parent:
            parent_fsync_calls += 1
            if parent_fsync_calls == 1:
                raise OSError("injected immutable parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_fsync_directory",
        staticmethod(fail_first_parent_fsync),
    )

    with pytest.raises(SnapshotPublicationUncertainError) as raised:
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert "immutable parent fsync failure" in str(raised.value.__cause__)
    assert raised.value.published is True
    assert VoxelMapSnapshot.load(target).metadata == metadata
    assert _snapshot_staging_candidates(target) == []
    assert parent_fsync_calls == 1


def test_snapshot_commit_new_never_deletes_concurrently_replaced_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.oviv2.snapshot import SnapshotPublicationUncertainError

    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    winner = tmp_path / "concurrent-winner"
    displaced = tmp_path / "displaced-publish"
    winner_metadata = replace(metadata, frame_id=20, revision=2)
    VoxelMapSnapshot.commit(
        winner,
        winner_metadata,
        geometry,
        evidence,
        ownership,
    )
    original_identity = VoxelMapSnapshot._directory_identity
    original_fsync_directory = VoxelMapSnapshot._fsync_directory
    parent_fsync_calls = 0

    def replace_target_after_stat(path: Path) -> tuple[int, int]:
        identity = original_identity(path)
        if path == target:
            os.replace(target, displaced)
            os.replace(winner, target)
        return identity

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal parent_fsync_calls
        if path == target.parent:
            parent_fsync_calls += 1
            if parent_fsync_calls == 1:
                raise OSError("injected immutable parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_directory_identity",
        staticmethod(replace_target_after_stat),
    )
    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_fsync_directory",
        staticmethod(fail_first_parent_fsync),
    )

    with pytest.raises(SnapshotPublicationUncertainError):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_directory_identity",
        staticmethod(original_identity),
    )
    assert VoxelMapSnapshot.load(winner).metadata == winner_metadata
    assert VoxelMapSnapshot.load(target).metadata == metadata
    assert _snapshot_staging_candidates(target) == []


def test_snapshot_commit_new_pre_publish_failure_preserves_one_stage_and_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_load = VoxelMapSnapshot.load
    staged_load_calls = 0

    def fail_first_staged_load(
        _cls,
        snapshot_dir: str | Path,
    ) -> VoxelMapSnapshot:
        nonlocal staged_load_calls
        source = Path(snapshot_dir)
        if source == target:
            raise AssertionError("snapshot load must complete before publication")
        if source == _snapshot_staging(target):
            staged_load_calls += 1
            raise RuntimeError("injected staged snapshot load failure")
        return original_load(snapshot_dir)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(fail_first_staged_load),
    )

    with pytest.raises(RuntimeError, match="staged snapshot load failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    staged_before = _snapshot_state(staged)

    with pytest.raises(FileExistsError):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert staged_load_calls == 1
    assert _snapshot_state(staged) == staged_before


def test_snapshot_commit_new_close_failure_never_masks_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    real_close = snapshot_module.os.close

    def fail_load_and_future_close(
        _cls,
        snapshot_dir: str | Path,
    ) -> VoxelMapSnapshot:
        assert Path(snapshot_dir) == _snapshot_staging(target)

        def close_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError("injected close failure")

        monkeypatch.setattr(snapshot_module.os, "close", close_then_fail)
        raise RuntimeError("original staged load failure")

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(fail_load_and_future_close),
    )

    with pytest.raises(RuntimeError, match="original staged load failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert _snapshot_staging(target).is_dir()


def test_snapshot_commit_new_member_close_failure_never_masks_fsync_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    original_write = VoxelMapSnapshot._write_snapshot_files
    real_close = snapshot_module.os.close

    def write_then_arm_failures(
        _cls,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, ...]:
        data_files = original_write(destination, *args, **kwargs)

        def fail_fsync(_descriptor: int) -> None:
            raise OSError("original member fsync failure")

        def close_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError("secondary close failure")

        monkeypatch.setattr(snapshot_module.os, "fsync", fail_fsync)
        monkeypatch.setattr(snapshot_module.os, "close", close_then_fail)
        return data_files

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_write_snapshot_files",
        classmethod(write_then_arm_failures),
    )

    with pytest.raises(OSError, match="original member fsync failure"):
        VoxelMapSnapshot.commit_new(
            target,
            metadata,
            geometry,
            evidence,
            ownership,
        )

    assert _snapshot_staging(target).is_dir()


def test_snapshot_commit_new_race_has_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    writer_entered = threading.Event()
    allow_writer = threading.Event()
    original_write = VoxelMapSnapshot._write_snapshot_files
    successes: list[VoxelMapSnapshot] = []
    failures: list[BaseException] = []

    def hold_first_writer(
        _cls,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, ...]:
        writer_entered.set()
        assert allow_writer.wait(timeout=10.0)
        return original_write(destination, *args, **kwargs)

    def commit(revision: int) -> None:
        try:
            successes.append(
                VoxelMapSnapshot.commit_new(
                    target,
                    replace(metadata, frame_id=revision * 10, revision=revision),
                    geometry,
                    evidence,
                    ownership,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_write_snapshot_files",
        classmethod(hold_first_writer),
    )
    winner = threading.Thread(target=commit, args=(1,))
    loser = threading.Thread(target=commit, args=(2,))
    winner.start()
    assert writer_entered.wait(timeout=10.0)
    loser.start()
    loser.join(timeout=2.0)
    loser_finished_while_staging_was_held = not loser.is_alive()
    allow_writer.set()
    winner.join(timeout=10.0)
    loser.join(timeout=10.0)

    assert not winner.is_alive()
    assert not loser.is_alive()
    assert loser_finished_while_staging_was_held is True
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert VoxelMapSnapshot.load(target).metadata == successes[0].metadata
    assert _snapshot_staging_candidates(target) == []


def test_snapshot_commit_new_unsupported_no_replace_race_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []
    monkeypatch.setattr(snapshot_module.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    def commit(revision: int) -> None:
        barrier.wait()
        try:
            VoxelMapSnapshot.commit_new(
                target,
                replace(metadata, frame_id=revision * 10, revision=revision),
                geometry,
                evidence,
                ownership,
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=commit, args=(revision,)) for revision in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(failures) == 2
    assert {type(error) for error in failures} == {FileExistsError, NotImplementedError}
    assert not target.exists()
    staged = _single_preserved_staging_directory(target)
    assert VoxelMapSnapshot.load(staged).metadata.revision in {1, 2}


def test_v2_snapshot_embeds_registry_and_hashes_entities(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()

    snapshot = VoxelMapSnapshot.commit(
        tmp_path / "snapshot",
        metadata,
        geometry,
        evidence,
        ownership,
        registry=registry,
    )
    restored = VoxelMapSnapshot.load(snapshot.path)

    assert restored.metadata.schema_version == 2
    assert restored.registry is not None
    assert restored.registry.entities == registry.entities
    assert restored.registry._next_entity_id == 12
    assert "entities.jsonl" in restored.checksums
    assert {path.name for path in restored.path.iterdir()} == {
        "metadata.json",
        "geometry.npz",
        "evidence.npz",
        "ownership.npz",
        "entities.jsonl",
        "checksums.json",
    }


def test_schema3_round_trip_preserves_typed_dense_provenance_and_v2_file_set(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    provenance = _dense_provenance()
    metadata = replace(
        metadata,
        schema_version=3,
        dense_semantic_provenance=provenance,
    )

    snapshot = VoxelMapSnapshot.commit(
        tmp_path / "snapshot",
        metadata,
        geometry,
        evidence,
        ownership,
        registry=_registry(),
    )
    restored = VoxelMapSnapshot.load(snapshot.path)
    payload = json.loads((snapshot.path / "metadata.json").read_text())

    assert restored.metadata.dense_semantic_provenance == provenance
    assert isinstance(
        restored.metadata.dense_semantic_provenance,
        DenseSemanticProvenance,
    )
    assert payload["dense_semantic_provenance"] == asdict(provenance)
    assert set(restored.checksums) == set(VoxelMapSnapshot._DATA_FILES_V2)
    assert {path.name for path in restored.path.iterdir()} == {
        *VoxelMapSnapshot._DATA_FILES_V2,
        "checksums.json",
    }
    assert restored.checksums["metadata.json"] == hashlib.sha256(
        (restored.path / "metadata.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("schema_version", [1, 2])
def test_old_schema_metadata_rejects_dense_provenance(schema_version: int) -> None:
    metadata, _, _, _ = _components()

    with pytest.raises(ValueError, match="dense semantic provenance"):
        replace(
            metadata,
            schema_version=schema_version,
            dense_semantic_provenance=_dense_provenance(),
        )


def test_schema3_requires_dense_provenance_and_registry(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()

    with pytest.raises(ValueError, match="dense semantic provenance"):
        replace(metadata, schema_version=3)

    metadata = replace(
        metadata,
        schema_version=3,
        dense_semantic_provenance=_dense_provenance(),
    )
    with pytest.raises(ValueError, match="schema v3.*registry|registry.*schema v3"):
        VoxelMapSnapshot.commit(
            tmp_path / "snapshot",
            metadata,
            geometry,
            evidence,
            ownership,
        )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_old_schema_metadata_json_omits_dense_field_and_loads_none(
    tmp_path: Path,
    schema_version: int,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=schema_version)
    registry = _registry() if schema_version == 2 else None

    snapshot = VoxelMapSnapshot.commit(
        tmp_path / f"snapshot-{schema_version}",
        metadata,
        geometry,
        evidence,
        ownership,
        registry=registry,
    )
    payload = json.loads((snapshot.path / "metadata.json").read_text())

    assert "dense_semantic_provenance" not in payload
    assert VoxelSnapshotMetadata(**payload).dense_semantic_provenance is None
    assert snapshot.metadata.dense_semantic_provenance is None


@pytest.mark.parametrize("malformation", ["unknown", "partial_language"])
def test_schema3_load_rejects_invalid_dense_provenance_dictionary(
    tmp_path: Path,
    malformation: str,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        target,
        replace(
            metadata,
            schema_version=3,
            dense_semantic_provenance=_dense_provenance(),
        ),
        geometry,
        evidence,
        ownership,
        registry=_registry(),
    )
    payload = json.loads((target / "metadata.json").read_text())
    provenance_payload = payload["dense_semantic_provenance"]
    if malformation == "unknown":
        provenance_payload["unknown_field"] = "unexpected"
    else:
        provenance_payload.pop("language_model_sha256")
    _rewrite_metadata_with_valid_checksum(target, payload)

    with pytest.raises(ValueError, match="metadata|provenance|language"):
        VoxelMapSnapshot.load(target)


def test_schema3_metadata_provenance_tamper_is_checksum_protected(
    tmp_path: Path,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        target,
        replace(
            metadata,
            schema_version=3,
            dense_semantic_provenance=_dense_provenance(),
        ),
        geometry,
        evidence,
        ownership,
        registry=_registry(),
    )
    metadata_path = target / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["dense_semantic_provenance"]["model_id"] = "tampered/model"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="checksum.*metadata"):
        VoxelMapSnapshot.load(target)


@pytest.mark.parametrize("schema_version", [True, 0, 4, 2.0])
def test_snapshot_metadata_requires_integer_schema_one_two_or_three(
    schema_version: object,
) -> None:
    metadata, _, _, _ = _components()

    with pytest.raises(ValueError, match="schema_version"):
        replace(metadata, schema_version=schema_version)


def test_snapshot_commit_requires_registry_matching_schema(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    registry = _registry()

    with pytest.raises(ValueError, match="schema.*registry|registry.*schema"):
        VoxelMapSnapshot.commit(
            tmp_path / "missing",
            replace(metadata, schema_version=2),
            geometry,
            evidence,
            ownership,
        )
    with pytest.raises(ValueError, match="schema.*registry|registry.*schema"):
        VoxelMapSnapshot.commit(
            tmp_path / "unexpected",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


def test_snapshot_load_rejects_changed_checksum(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    with (target / "evidence.npz").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="checksum"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_corrupt_entities_checksum(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(
        target,
        replace(metadata, schema_version=2),
        geometry,
        evidence,
        ownership,
        registry=_registry(),
    )
    with (target / "entities.jsonl").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="checksum.*entities"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_schema_file_set_mismatch(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    metadata_path = target / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["schema_version"] = 2
    metadata_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    checksums_path = target / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    checksums_path.write_text(json.dumps(checksums, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="schema|files"):
        VoxelMapSnapshot.load(target)


def test_snapshot_load_rejects_extra_physical_file(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    (target / "extra.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="files"):
        VoxelMapSnapshot.load(target)


def test_snapshot_rejects_mismatched_voxel_configuration(tmp_path: Path) -> None:
    metadata, geometry, _, ownership = _components()
    evidence = SparseEvidenceStore(EvidenceConfig(block_resolution=4))

    with pytest.raises(ValueError, match="block_resolution"):
        VoxelMapSnapshot.commit(tmp_path / "snapshot", metadata, geometry, evidence, ownership)


def test_snapshot_rejects_owner_without_entity_evidence(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    ownership.assign((1, 0, 20), 99, 0.7, 2)

    with pytest.raises(ValueError, match="entity evidence"):
        VoxelMapSnapshot.commit(tmp_path / "snapshot", metadata, geometry, evidence, ownership)


def test_v2_snapshot_rejects_entity_references_missing_from_registry(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()
    evidence.update_entity((1, 0, 20), 99, 1.0, 0.0, 1.0, 1)

    with pytest.raises(ValueError, match="registry.*99|99.*registry"):
        VoxelMapSnapshot.commit(
            tmp_path / "evidence",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )

    evidence.update_entity((2, 0, 20), 98, 1.0, 0.0, 1.0, 1)
    ownership.assign((2, 0, 20), 98, 0.7, 1)
    with pytest.raises(ValueError, match="registry.*98|98.*registry"):
        VoxelMapSnapshot.commit(
            tmp_path / "ownership",
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


@pytest.mark.parametrize("future", ["entity", "watermark"])
def test_v2_snapshot_rejects_registry_revision_after_metadata(
    tmp_path: Path,
    future: str,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = _registry()
    if future == "entity":
        registry.entities[11] = replace(registry.entities[11], last_revision=2)
    else:
        registry._last_revision = 2

    with pytest.raises(ValueError, match="revision"):
        VoxelMapSnapshot.commit(
            tmp_path / future,
            metadata,
            geometry,
            evidence,
            ownership,
            registry=registry,
        )


def test_snapshot_overwrite_is_complete_and_leaves_no_temporary_directory(tmp_path: Path) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = VoxelSnapshotMetadata(
        scene_id=metadata.scene_id,
        frame_id=20,
        timestamp=2.0,
        revision=2,
        voxel_size_m=metadata.voxel_size_m,
        block_resolution=metadata.block_resolution,
    )

    VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert VoxelMapSnapshot.load(target).metadata == updated
    assert not list(tmp_path.glob(".snapshot.*"))


def test_snapshot_exchange_failure_preserves_valid_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)

    def fail_exchange(_cls, _left: Path, _right: Path) -> None:
        raise OSError("injected exchange failure")

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_exchange_directories",
        classmethod(fail_exchange),
        raising=False,
    )

    with pytest.raises(OSError, match="injected"):
        VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert target.is_dir()
    assert VoxelMapSnapshot.load(target).metadata == metadata


def test_snapshot_overwrite_never_exposes_missing_or_mixed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)
    original_replace = os.replace

    def widen_legacy_gap(source, destination) -> None:
        original_replace(source, destination)
        if Path(source) == target:
            time.sleep(0.05)

    monkeypatch.setattr(os, "replace", widen_legacy_gap)
    stop = threading.Event()
    started = threading.Event()
    failures: list[BaseException | str] = []
    revisions: list[int] = []

    def read_repeatedly() -> None:
        started.set()
        while not stop.is_set():
            if not target.is_dir():
                failures.append("target missing")
                continue
            try:
                revisions.append(VoxelMapSnapshot.load(target).metadata.revision)
            except BaseException as exc:
                failures.append(exc)

    reader = threading.Thread(target=read_repeatedly, daemon=True)
    reader.start()
    assert started.wait(timeout=1.0)
    try:
        VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)
    finally:
        stop.set()
        reader.join(timeout=5.0)

    assert not reader.is_alive()
    assert failures == []
    assert revisions
    assert set(revisions).issubset({1, 2})
    assert VoxelMapSnapshot.load(target).metadata == updated


def test_snapshot_load_retries_after_concurrent_directory_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    alternate = tmp_path / "alternate"
    holding = tmp_path / "holding"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    VoxelMapSnapshot.commit(
        alternate,
        replace(metadata, frame_id=20, revision=2),
        geometry,
        evidence,
        ownership,
    )
    original_sha256 = VoxelMapSnapshot._sha256
    calls = 0

    def exchange_after_first_hash(path: Path) -> str:
        nonlocal calls
        digest = original_sha256(path)
        calls += 1
        if calls == 1:
            os.replace(target, holding)
            os.replace(alternate, target)
            os.replace(holding, alternate)
        return digest

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_sha256",
        staticmethod(exchange_after_first_hash),
    )

    restored = VoxelMapSnapshot.load(target)

    assert restored.metadata.revision in {1, 2}
    assert restored.path == target


def test_snapshot_post_exchange_load_failure_restores_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)
    original_load = VoxelMapSnapshot.load

    def fail_target_load(_cls, snapshot_dir: str | Path) -> VoxelMapSnapshot:
        if Path(snapshot_dir) == target:
            raise RuntimeError("injected post-exchange load failure")
        return original_load(snapshot_dir)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "load",
        classmethod(fail_target_load),
    )

    with pytest.raises(RuntimeError, match="post-exchange load failure"):
        VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert original_load(target).metadata == metadata


def test_snapshot_parent_fsync_failure_restores_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)
    original_fsync_directory = VoxelMapSnapshot._fsync_directory
    parent_fsync_calls = 0

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal parent_fsync_calls
        if path == target.parent:
            parent_fsync_calls += 1
            if parent_fsync_calls == 1:
                raise OSError("injected parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_fsync_directory",
        staticmethod(fail_first_parent_fsync),
    )

    with pytest.raises(OSError, match="parent fsync failure"):
        VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert parent_fsync_calls == 2
    assert VoxelMapSnapshot.load(target).metadata == metadata


def test_snapshot_old_directory_cleanup_failure_does_not_fail_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)
    original_rmtree = shutil.rmtree

    def fail_old_snapshot_cleanup(path: str | Path, *args, **kwargs) -> None:
        if Path(path).name.startswith(".snapshot.tmp-"):
            raise OSError("injected old snapshot cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_old_snapshot_cleanup)

    restored = VoxelMapSnapshot.commit(
        target,
        updated,
        geometry,
        evidence,
        ownership,
    )

    assert restored.metadata == updated
    assert VoxelMapSnapshot.load(target).metadata == updated


def test_snapshot_reports_publication_and_rollback_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, geometry, evidence, ownership = _components()
    target = tmp_path / "snapshot"
    VoxelMapSnapshot.commit(target, metadata, geometry, evidence, ownership)
    updated = replace(metadata, frame_id=20, revision=2)
    original_load = VoxelMapSnapshot.load
    original_exchange = VoxelMapSnapshot._exchange_directories
    exchange_calls = 0
    publication_error = RuntimeError("injected publication failure")
    rollback_error = OSError("injected rollback failure")

    def fail_target_load(_cls, snapshot_dir: str | Path) -> VoxelMapSnapshot:
        if Path(snapshot_dir) == target:
            raise publication_error
        return original_load(snapshot_dir)

    def fail_rollback_exchange(_cls, left: Path, right: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 2:
            raise rollback_error
        original_exchange(left, right)

    monkeypatch.setattr(VoxelMapSnapshot, "load", classmethod(fail_target_load))
    monkeypatch.setattr(
        VoxelMapSnapshot,
        "_exchange_directories",
        classmethod(fail_rollback_exchange),
    )

    with pytest.raises(Exception) as raised:
        VoxelMapSnapshot.commit(target, updated, geometry, evidence, ownership)

    assert not isinstance(raised.value, NameError)
    assert isinstance(raised.value, snapshot_module._SnapshotRollbackError)
    assert raised.value.publication_error is publication_error
    assert raised.value.rollback_error is rollback_error
    assert "publication failure" in str(raised.value)
    assert "rollback failure" in str(raised.value)
    assert raised.value.__cause__ is publication_error
