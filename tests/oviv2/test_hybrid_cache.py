from __future__ import annotations

import gzip
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.oviv2 import hybrid_cache
from src.oviv2.hybrid_cache import (
    load_frontend_batch,
    publish_frontend_manifest,
    sha256,
    write_hybrid_frame,
)
from src.oviv2.hybrid_frontend import FrontendBatch
from src.oviv2.observations import CachedFrontendAdapter, ReplicaVocabulary


def _batch_fixture() -> FrontendBatch:
    masks = np.zeros((2, 4, 5), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, :2, :2] = True
    return FrontendBatch(
        masks=masks,
        boxes_xyxy=np.asarray(((1, 1, 4, 3), (0, 0, 2, 2)), dtype=np.float32),
        confidences=np.asarray((0.9, 0.8), dtype=np.float32),
        labels=("chair", "table"),
        image_features=np.asarray(((3.0, 4.0), (0.0, 2.0)), dtype=np.float32),
    )


def _frame() -> Frame:
    return Frame(
        frame_id=2,
        source_frame_id=20,
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth=np.ones((4, 5), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(2.0, 2.0, 2.0, 1.5, 5, 4),
        timestamp=2.0,
    )


def _payload_from_batch(batch: FrontendBatch | None = None) -> dict[str, object]:
    value = _batch_fixture() if batch is None else batch
    classes = ("chair", "table")
    return {
        "mask": np.array(value.masks, copy=True),
        "xyxy": np.array(value.boxes_xyxy, copy=True),
        "confidence": np.array(value.confidences, copy=True),
        "class_id": np.asarray(
            [classes.index(label) for label in value.labels], dtype=np.int64
        ),
        "classes": list(classes),
        "image_feats": np.array(value.image_features, copy=True),
    }


def _write_raw_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=path, mode="wb", mtime=0) as stream:
        pickle.dump(payload, stream, protocol=4)


def _cache_hashes(output_dir: Path, frame_count: int) -> dict[str, str]:
    return {
        f"frame{index:06d}.pkl.gz": sha256(output_dir / f"frame{index:06d}.pkl.gz")
        for index in range(frame_count)
    }


def _write_frames(output_dir: Path, frame_count: int = 2) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frame_count):
        write_hybrid_frame(
            output_dir / f"frame{index:06d}.pkl.gz",
            _batch_fixture(),
            classes=("chair", "table"),
        )
    return _cache_hashes(output_dir, frame_count)


def _publish(output_dir: Path, **changes: object) -> dict[str, object]:
    cache_hashes = _write_frames(output_dir)
    values: dict[str, object] = {
        "output_dir": output_dir,
        "scene": "room0",
        "frame_count": 2,
        "source_frame_ids": (0, 10),
        "classes": ("chair", "table"),
        "feature_model_id": "clip-vit-h-14:test",
        "cache_files_sha256": cache_hashes,
        "source_sha256": {"yolo_manifest.json": "a" * 64},
        "policy": {"variant": "yolo_novel_sam", "novel_iou": 0.8},
        "diagnostics": {"accepted_yolo": 4, "accepted_novel_sam": 2},
    }
    values.update(changes)
    return publish_frontend_manifest(**values)  # type: ignore[arg-type]


def _temporary_files(path: Path) -> list[Path]:
    return sorted(entry for entry in path.iterdir() if entry.name.startswith("."))


def test_hybrid_cache_round_trip_is_adapter_compatible(tmp_path: Path) -> None:
    output = tmp_path / "frame000002.pkl.gz"
    expected = _batch_fixture()

    published_hash = write_hybrid_frame(output, expected, classes=("chair", "table"))
    restored = load_frontend_batch(output, expected_sha256=published_hash)

    assert published_hash == sha256(output)
    assert restored.labels == expected.labels
    assert restored.masks.flags.writeable is False
    assert restored.boxes_xyxy.flags.writeable is False
    assert restored.image_features.flags.writeable is False

    adapter = CachedFrontendAdapter(
        tmp_path,
        ReplicaVocabulary(classes=("wall", "chair", "table"), aliases={}),
        min_valid_points=1,
        feature_model_id="clip-vit-h-14:test",
    )
    observations = adapter.observe(_frame(), cache_frame_id=2)
    assert [observation.label for observation in observations] == ["chair", "table"]
    assert all(observation.image_feature is not None for observation in observations)
    assert all(observation.text_feature is None for observation in observations)


def test_writer_is_deterministic_and_preserves_frame_local_class_mapping(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one.pkl.gz"
    second = tmp_path / "two.pkl.gz"
    batch = _batch_fixture()

    write_hybrid_frame(first, batch, classes=("wall", "table", "chair"))
    write_hybrid_frame(second, batch, classes=("wall", "table", "chair"))

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rb") as stream:
        payload = pickle.load(stream)
    assert payload["classes"] == ["wall", "table", "chair"]
    assert payload["class_id"].tolist() == [2, 1]
    assert set(payload) == {
        "mask",
        "xyxy",
        "confidence",
        "class_id",
        "classes",
        "image_feats",
    }


def test_writer_refuses_existing_and_dangling_symlink_destinations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"
    write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))
    prior = output.read_bytes()

    with pytest.raises(FileExistsError):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))
    assert output.read_bytes() == prior

    dangling = tmp_path / "frame000001.pkl.gz"
    dangling.symlink_to(tmp_path / "missing.pkl.gz")
    with pytest.raises(FileExistsError):
        write_hybrid_frame(dangling, _batch_fixture(), classes=("chair", "table"))


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("non-dict", ["not", "a", "dict"], "dictionary"),
        ("missing-key", {"mask": np.zeros((0, 2, 2), dtype=bool)}, "keys"),
        (
            "object-array",
            {**_payload_from_batch(), "mask": np.asarray([object()], dtype=object)},
            "unsafe|pickle",
        ),
        (
            "non-finite",
            {**_payload_from_batch(), "confidence": np.asarray([np.nan, 0.8])},
            "finite",
        ),
        (
            "class-bounds",
            {**_payload_from_batch(), "class_id": np.asarray([0, 2])},
            "class_id",
        ),
        (
            "class-float",
            {**_payload_from_batch(), "class_id": np.asarray([0.0, 1.0])},
            "class_id",
        ),
        ("shape", {**_payload_from_batch(), "xyxy": np.zeros((2, 5))}, "shape"),
        (
            "zero-feature",
            {**_payload_from_batch(), "image_feats": np.zeros((2, 2))},
            "nonzero",
        ),
        (
            "blank-class",
            {**_payload_from_batch(), "classes": ["chair", " "]},
            "classes",
        ),
        (
            "text-features",
            {**_payload_from_batch(), "text_feats": np.ones((2, 2))},
            "keys",
        ),
    ],
)
def test_loader_rejects_invalid_payloads(
    tmp_path: Path,
    name: str,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / f"{name}.pkl.gz"
    _write_raw_payload(path, payload)

    with pytest.raises(ValueError, match=message):
        load_frontend_batch(path)


def test_loader_rejects_corrupt_pickle_checksum_mismatch_and_symlink(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.pkl.gz"
    corrupt.write_bytes(b"not a gzip stream")
    with pytest.raises(ValueError, match="gzip|pickle|payload"):
        load_frontend_batch(corrupt)

    valid = tmp_path / "valid.pkl.gz"
    actual = write_hybrid_frame(valid, _batch_fixture(), classes=("chair", "table"))
    with pytest.raises(ValueError, match="checksum"):
        load_frontend_batch(valid, expected_sha256="0" * 64)
    assert load_frontend_batch(valid, expected_sha256=actual).labels == (
        "chair",
        "table",
    )

    link = tmp_path / "link.pkl.gz"
    link.symlink_to(valid)
    with pytest.raises(ValueError, match="symlink"):
        load_frontend_batch(link)
    with pytest.raises(ValueError, match="sha256"):
        load_frontend_batch(valid, expected_sha256="A" * 64)


def test_writer_removes_same_directory_temp_file_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"

    def fail_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(hybrid_cache.os, "link", fail_link)
    with pytest.raises(OSError, match="link failed"):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

    assert not output.exists()
    assert _temporary_files(tmp_path) == []


def test_writer_rolls_back_hard_link_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"
    original_fsync = hybrid_cache._fsync_directory
    calls = 0

    def fail_first_fsync(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("directory fsync failed")
        original_fsync(directory)

    monkeypatch.setattr(hybrid_cache, "_fsync_directory", fail_first_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

    assert calls == 2
    assert not output.exists()
    assert _temporary_files(tmp_path) == []


def test_writer_rolls_back_when_published_temp_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"
    original_unlink = Path.unlink
    original_fsync = hybrid_cache._fsync_directory
    temporary_path: Path | None = None
    temporary_unlink_calls = 0
    rollback_inodes: list[tuple[int, int]] = []
    fsync_calls = 0

    def fail_first_temporary_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal temporary_path, temporary_unlink_calls
        if path.name.startswith(f".{output.name}."):
            temporary_path = path
            temporary_unlink_calls += 1
            if temporary_unlink_calls == 1:
                raise OSError("temporary unlink failed")
        elif path == output:
            assert temporary_path is not None
            rollback_inodes.append(
                (os.lstat(path).st_ino, os.lstat(temporary_path).st_ino)
            )
        original_unlink(path, missing_ok=missing_ok)

    def record_fsync(directory: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(directory)

    monkeypatch.setattr(Path, "unlink", fail_first_temporary_unlink)
    monkeypatch.setattr(hybrid_cache, "_fsync_directory", record_fsync)
    with pytest.raises(OSError, match="failed to clean temporary frontend cache output"):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

    assert rollback_inodes and rollback_inodes[0][0] == rollback_inodes[0][1]
    assert temporary_unlink_calls == 2
    assert fsync_calls == 2
    assert not output.exists()
    assert _temporary_files(tmp_path) == []


def test_writer_does_not_remove_competing_destination_during_cleanup_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"
    original_unlink = Path.unlink
    temporary_unlink_calls = 0
    competing_data = b"competing cache publication"

    def replace_destination_before_cleanup_failure(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal temporary_unlink_calls
        if path.name.startswith(f".{output.name}."):
            temporary_unlink_calls += 1
            if temporary_unlink_calls == 1:
                original_unlink(output)
                output.write_bytes(competing_data)
                raise OSError("temporary unlink failed")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", replace_destination_before_cleanup_failure)
    with pytest.raises(
        OSError,
        match="failed to roll back published frontend cache output",
    ):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

    assert output.read_bytes() == competing_data
    assert temporary_unlink_calls == 2
    assert _temporary_files(tmp_path) == []


def test_writer_prioritizes_rollback_error_when_cleanup_retry_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "frame000000.pkl.gz"
    original_unlink = Path.unlink
    temporary_unlink_calls = 0
    competing_data = b"competing cache publication"

    def fail_cleanup_and_replace_destination(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        nonlocal temporary_unlink_calls
        if path.name.startswith(f".{output.name}."):
            temporary_unlink_calls += 1
            if temporary_unlink_calls == 1:
                original_unlink(output)
                output.write_bytes(competing_data)
            raise OSError("temporary unlink failed")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_cleanup_and_replace_destination)
    with pytest.raises(
        OSError,
        match=(
            "failed to roll back published frontend cache output file; "
            "temporary frontend cache output cleanup also failed"
        ),
    ):
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

    assert output.read_bytes() == competing_data
    assert temporary_unlink_calls == 2
    assert len(_temporary_files(tmp_path)) == 1


def test_manifest_is_deterministic_and_exactly_matches_published_json(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = _publish(first_dir)
    second = _publish(second_dir)

    first_path = first_dir / "frontend_manifest.json"
    second_path = second_dir / "frontend_manifest.json"
    assert first == json.loads(first_path.read_text(encoding="utf-8"))
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["schema_version"] == 1
    assert first["method"] == "OVIV2"
    assert first["cache_files_sha256"] == _cache_hashes(first_dir, 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"scene": " "}, "scene"),
        ({"frame_count": 0}, "frame_count"),
        ({"source_frame_ids": (0,)}, "source_frame_ids"),
        ({"source_frame_ids": (0, True)}, "source_frame_ids"),
        ({"classes": ("chair", "chair")}, "classes"),
        ({"feature_model_id": " "}, "feature_model_id"),
        ({"source_sha256": {"source": "A" * 64}}, "sha256"),
        ({"policy": {"threshold": float("nan")}}, "policy"),
        ({"diagnostics": {"count": -1}}, "diagnostics"),
        ({"cache_files_sha256": {"frame000000.pkl.gz": "0" * 64}}, "cover"),
        (
            {
                "cache_files_sha256": {
                    "bad.pkl.gz": "0" * 64,
                    "frame000000.pkl.gz": "0" * 64,
                }
            },
            "cover",
        ),
    ],
)
def test_manifest_rejects_invalid_inputs(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _publish(tmp_path / "cache", **change)
    assert not (tmp_path / "cache" / "frontend_manifest.json").exists()


def test_manifest_refuses_existing_or_symlink_path_and_cleans_up_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache"
    _publish(output_dir)
    with pytest.raises(FileExistsError):
        _publish(output_dir)

    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(output_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _publish(linked_dir)

    cleanup_dir = tmp_path / "cleanup"
    _write_frames(cleanup_dir)

    def fail_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *args: object,
        **kwargs: object,
    ) -> None:
        raise OSError("manifest link failed")

    monkeypatch.setattr(hybrid_cache.os, "link", fail_link)
    with pytest.raises(OSError, match="manifest link failed"):
        publish_frontend_manifest(
            cleanup_dir,
            "room0",
            2,
            (0, 10),
            ("chair", "table"),
            "clip-vit-h-14:test",
            _cache_hashes(cleanup_dir, 2),
            {"source": "a" * 64},
            {"variant": "sam_labeled"},
            {"accepted": 1},
        )
    assert not (cleanup_dir / "frontend_manifest.json").exists()
    assert _temporary_files(cleanup_dir) == []
