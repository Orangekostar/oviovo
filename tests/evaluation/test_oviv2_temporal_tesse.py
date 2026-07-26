from __future__ import annotations

import json
import io
import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import zipfile

import numpy as np
import pytest

from src.core.data_structures import CameraIntrinsics, Frame
from src.evaluation.oviv2_temporal_tesse import publish_temporal_current_checkpoint
from src.evaluation.oviv2_temporal_tesse import load_temporal_current_checkpoint
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_geometry import ObjectSubmap
from src.oviv2.temporal_lifecycle import TemporalLifecycle, TemporalLifecycleState
from src.oviv2.temporal_snapshot import TemporalCurrentSnapshot, TemporalSnapshotMetadata
from src.oviv2.temporal_state import TemporalEntityState


def _snapshot() -> TemporalCurrentSnapshot:
    config = TemporalGeometryConfig(0.1, 4.0, 2, 4, 4, 16, 0, 3, 0.5, 0.1, 2.0)
    return TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "b" * 64),
        (),
        TemporalBackgroundVolume(config),
    )


def _snapshot_with_entity(*, semantic_id: int, prototype_dimension: int) -> TemporalCurrentSnapshot:
    config = TemporalGeometryConfig(0.1, 4.0, 2, 4, 4, 16, 0, 3, 0.5, 0.1, 2.0)
    prototype = np.linspace(1.0, 2.0, prototype_dimension, dtype=np.float64)
    entity = TemporalEntityState(
        lifecycle=TemporalLifecycleState(
            entity_id=1,
            lifecycle=TemporalLifecycle.ACTIVE,
            existence_log_odds=1.0,
            last_frame_id=0,
            last_timestamp=0.0,
            absent_streak=0,
            absence_view_bins=(),
        ),
        semantic_probabilities=((semantic_id, 1.0),),
        image_prototype=prototype,
        feature_model_id="test",
        extent_xyz=(1.0, 1.0, 1.0),
        object_to_world=np.eye(4),
        submap=ObjectSubmap(
            reference_centroid_xyz=(0.0, 0.0, 0.0),
            local_voxel_keys=((0, 0, 0),),
            local_points_xyz=np.asarray([[0.05, 0.0, 0.0]]),
            weights=np.ones(1),
            last_seen_frame_ids=np.asarray([0], dtype=np.int64),
        ),
        first_seen_frame_id=0,
        last_seen_frame_id=0,
    )
    return TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "b" * 64),
        (entity,),
        TemporalBackgroundVolume(config),
    )


def _snapshot_with_dormant_entities(count: int) -> TemporalCurrentSnapshot:
    config = TemporalGeometryConfig(
        0.1, 4.0, count, 1, 1, max(16, count), 0, 3, 0.5, 0.1, 2.0
    )
    empty_submap = ObjectSubmap(
        reference_centroid_xyz=(0.0, 0.0, 0.0),
        local_voxel_keys=(),
        local_points_xyz=np.empty((0, 3), dtype=np.float64),
        weights=np.empty((0,), dtype=np.float64),
        last_seen_frame_ids=np.empty((0,), dtype=np.int64),
    )
    entities = tuple(
        TemporalEntityState(
            lifecycle=TemporalLifecycleState(
                entity_id=entity_id,
                lifecycle=TemporalLifecycle.DORMANT,
                existence_log_odds=-1.0,
                last_frame_id=0,
                last_timestamp=0.0,
                absent_streak=1,
                absence_view_bins=(0,),
            ),
            semantic_probabilities=((1, 1.0),),
            image_prototype=None,
            feature_model_id=None,
            extent_xyz=(1.0, 1.0, 1.0),
            object_to_world=np.eye(4),
            submap=empty_submap,
            first_seen_frame_id=0,
            last_seen_frame_id=0,
        )
        for entity_id in range(1, count + 1)
    )
    return TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "b" * 64),
        entities,
        TemporalBackgroundVolume(config),
    )


def _files(path: Path) -> set[str]:
    return {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}


def _rewrite_member(path: Path, name: str, content: bytes) -> None:
    (path / name).write_bytes(content)
    checksums = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    checksums[name] = hashlib.sha256(content).hexdigest()
    import src.evaluation.oviv2_temporal_tesse as module
    (path / "checksums.json").write_bytes(module._canonical_json(checksums))


def _rewrite_manifest(path: Path, **changes: object) -> None:
    import src.evaluation.oviv2_temporal_tesse as module
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(changes)
    _rewrite_member(path, "manifest.json", module._canonical_json(manifest))


def test_full_publication_is_atomic_deterministic_and_has_diagnostics(tmp_path: Path) -> None:
    first = publish_temporal_current_checkpoint(
        tmp_path / "first", _snapshot_with_entity(semantic_id=1, prototype_dimension=2), ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    second = publish_temporal_current_checkpoint(
        tmp_path / "second", _snapshot_with_entity(semantic_id=1, prototype_dimension=2), ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    assert _files(first.path) == {
        "checksums.json", "diagnostics.json", "entities.jsonl", "manifest.json", "snapshot.npz"
    }
    assert first.content_sha256 == second.content_sha256
    manifest = json.loads((first.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["consumed_through_frame"] == 0
    assert manifest["code_commit"] == "c" * 40
    assert manifest["input_sha256"] == "d" * 64
    assert manifest["schema_version"] == 2
    assert json.loads((first.path / "diagnostics.json").read_text(encoding="utf-8")) == {
        "dormant": [], "uncertain": []
    }
    first.revalidate_source()
    loaded = load_temporal_current_checkpoint(first.path)
    prediction, diagnostics = loaded
    assert prediction.scene_id == "scene"
    assert diagnostics == {"dormant": [], "uncertain": []}
    assert loaded.snapshot.entities[0].metadata["geometry_epoch"] == 0
    assert loaded.snapshot.entities[0].metadata["readout_valid"] is True
    loaded.revalidate_source()
    with pytest.raises(FileExistsError):
        publish_temporal_current_checkpoint(
            first.path, _snapshot(), ("unknown", "chair"),
            code_commit="c" * 40, input_sha256="d" * 64,
        )


def test_failed_full_publication_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    def fail(*args: object, **kwargs: object) -> bytes:
        raise RuntimeError("serialize")

    monkeypatch.setattr(module, "_serialize_map_snapshot", fail)
    with pytest.raises(RuntimeError, match="serialize"):
        publish_temporal_current_checkpoint(
            tmp_path / "target", _snapshot(), ("unknown",),
            code_commit="c" * 40, input_sha256="d" * 64,
        )
    assert list(tmp_path.iterdir()) == []


def test_full_publication_rejects_symlink_traversal(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish_temporal_current_checkpoint(
            link / "target", _snapshot(), ("unknown",),
            code_commit="c" * 40, input_sha256="d" * 64,
        )


def test_full_loader_uses_numeric_temporal_id_order(tmp_path: Path) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    entities = [
        EntityPrediction(
            entity_id=f"temporal:{entity_id}", points_xyz=np.empty((0, 3)),
            semantic_embedding=None, semantic_label="unknown", semantic_score=0.0,
            lifecycle_state="active", first_seen=0.0, last_seen=0.0,
            metadata={
                "temporal_entity_id": entity_id,
                "semantic_id": 0,
                "geometry_epoch": 0,
                "readout_valid": True,
            },
        )
        for entity_id in (2, 10)
    ]
    archive, records = module._serialize_map_snapshot(MapSnapshot(
        method="OVIV2-temporal", scene_id="scene", timestamp=0.0,
        entities=entities, background_xyz=None, scope="current",
    ))
    (receipt.path / "snapshot.npz").write_bytes(archive)
    (receipt.path / "entities.jsonl").write_bytes(records)
    checksums = json.loads((receipt.path / "checksums.json").read_text(encoding="utf-8"))
    checksums["snapshot.npz"] = hashlib.sha256(archive).hexdigest()
    checksums["entities.jsonl"] = hashlib.sha256(records).hexdigest()
    (receipt.path / "checksums.json").write_bytes(module._canonical_json(checksums))

    prediction, diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert [item.entity_id for item in prediction.entities] == ["temporal:2", "temporal:10"]
    assert diagnostics == {"dormant": [], "uncertain": []}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"consumed_through_frame": True}, "frame"),
        ({"revision": True}, "revision"),
        ({"config_sha256": "BAD"}, "config_sha256"),
        ({"code_commit": "g" * 40}, "code_commit"),
        ({"input_sha256": "0" * 63}, "input_sha256"),
        ({"scope": "history"}, "scope"),
    ],
)
def test_full_loader_strictly_validates_manifest(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    _rewrite_manifest(receipt.path, **changes)
    with pytest.raises((TypeError, ValueError), match=message):
        load_temporal_current_checkpoint(receipt.path)


def test_full_loader_strictly_separates_v1_and_v2_entity_fields(tmp_path: Path) -> None:
    snapshot = _snapshot_with_entity(semantic_id=1, prototype_dimension=2)

    legacy = publish_temporal_current_checkpoint(
        tmp_path / "legacy", snapshot, ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    _rewrite_manifest(legacy.path, schema_version=1)
    record = json.loads((legacy.path / "entities.jsonl").read_text(encoding="utf-8"))
    del record["metadata"]["geometry_epoch"]
    del record["metadata"]["readout_valid"]
    import src.evaluation.oviv2_temporal_tesse as module
    _rewrite_member(legacy.path, "entities.jsonl", module._canonical_json(record))
    loaded, _ = load_temporal_current_checkpoint(legacy.path)
    assert loaded.entities[0].metadata == {
        "temporal_entity_id": 1,
        "semantic_id": 1,
        "geometry_epoch": 0,
        "readout_valid": True,
    }

    smuggled = publish_temporal_current_checkpoint(
        tmp_path / "smuggled", snapshot, ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    _rewrite_manifest(smuggled.path, schema_version=1)
    with pytest.raises(ValueError, match="metadata schema"):
        load_temporal_current_checkpoint(smuggled.path)

    missing = publish_temporal_current_checkpoint(
        tmp_path / "missing", snapshot, ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    record = json.loads((missing.path / "entities.jsonl").read_text(encoding="utf-8"))
    del record["metadata"]["geometry_epoch"]
    _rewrite_member(missing.path, "entities.jsonl", module._canonical_json(record))
    with pytest.raises(ValueError, match="metadata schema"):
        load_temporal_current_checkpoint(missing.path)


@pytest.mark.parametrize("version", [True, 0, 3])
def test_full_loader_rejects_boolean_or_unknown_schema_version(
    tmp_path: Path, version: object
) -> None:
    receipt = publish_temporal_current_checkpoint(
        tmp_path / str(version), _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    _rewrite_manifest(receipt.path, schema_version=version)
    with pytest.raises(ValueError, match="schema_version"):
        load_temporal_current_checkpoint(receipt.path)


def test_full_loader_preflights_duplicate_zip_before_np_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    archive = io.BytesIO((receipt.path / "snapshot.npz").read_bytes())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as output:
            output.writestr("entity_points.npy", b"duplicate")
    _rewrite_member(receipt.path, "snapshot.npz", archive.getvalue())
    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    with pytest.raises(ValueError, match="inventory|duplicate"):
        load_temporal_current_checkpoint(receipt.path)


def test_full_loader_preflights_zip_bomb_and_forged_shape_before_np_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    manifest = json.loads((receipt.path / "manifest.json").read_text(encoding="utf-8"))
    huge = np.zeros((manifest["maximum_background_points"] + 1, 3), dtype=np.float32)
    arrays = {
        "entity_points": np.empty((0, 3), dtype=np.float32),
        "entity_point_offsets": np.asarray([0], dtype=np.int64),
        "background_xyz": huge,
        "background_present": np.asarray([1], dtype=np.uint8),
    }
    bomb = module._canonical_npz(arrays, module._ARRAY_NAMES)
    _rewrite_member(receipt.path, "snapshot.npz", bomb)
    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    with pytest.raises(ValueError, match="capacity|limit|shape"):
        load_temporal_current_checkpoint(receipt.path)

    second = publish_temporal_current_checkpoint(
        tmp_path / "second", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        forged, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 3)}
    )
    archive = io.BytesIO()
    valid = zipfile.ZipFile(io.BytesIO((second.path / "snapshot.npz").read_bytes()))
    with valid, zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for info in valid.infolist():
            output.writestr(info.filename, forged.getvalue() if info.filename == "background_xyz.npy" else valid.read(info))
    _rewrite_member(second.path, "snapshot.npz", archive.getvalue())
    with pytest.raises(ValueError, match="shape|size|capacity"):
        load_temporal_current_checkpoint(second.path)


def test_full_publication_after_rename_failure_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    target = tmp_path / "published"
    monkeypatch.setattr(
        module._DirectoryWitness,
        "capture",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture"))),
    )
    with pytest.raises(module.TemporalCheckpointPublicationUncertainError) as caught:
        publish_temporal_current_checkpoint(
            target, _snapshot(), ("unknown",),
            code_commit="c" * 40, input_sha256="d" * 64,
        )
    assert caught.value.published is True
    assert target.is_dir()


def test_full_concurrent_no_replace_has_one_winner(tmp_path: Path) -> None:
    target = tmp_path / "race"

    def publish() -> str:
        try:
            publish_temporal_current_checkpoint(
                target, _snapshot(), ("unknown",),
                code_commit="c" * 40, input_sha256="d" * 64,
            )
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: publish(), range(2))) == ["exists", "published"]


def test_full_loaded_witness_rejects_same_size_tamper_and_inode_swap(tmp_path: Path) -> None:
    first = publish_temporal_current_checkpoint(
        tmp_path / "first", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    loaded = load_temporal_current_checkpoint(first.path)
    member = first.path / "diagnostics.json"
    before = member.stat()
    content = member.read_bytes()
    member.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="content|hash|identity"):
        loaded.revalidate_source()

    second = publish_temporal_current_checkpoint(
        tmp_path / "second", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    loaded_second = load_temporal_current_checkpoint(second.path)
    original = second.path / "diagnostics.json"
    replacement = second.path / "replacement"
    replacement.write_bytes(original.read_bytes())
    os.replace(replacement, original)
    with pytest.raises(ValueError, match="identity"):
        loaded_second.revalidate_source()


def test_full_roundtrip_preserves_out_of_vocabulary_semantic_as_none(tmp_path: Path) -> None:
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint",
        _snapshot_with_entity(semantic_id=2, prototype_dimension=2),
        ("unknown",),
        code_commit="c" * 40,
        input_sha256="d" * 64,
    )
    loaded, _diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert loaded.entities[0].metadata["semantic_id"] == 2
    assert loaded.entities[0].semantic_label is None


def test_full_roundtrip_accepts_large_bounded_entity_record(tmp_path: Path) -> None:
    snapshot = _snapshot_with_entity(semantic_id=1, prototype_dimension=10_000)
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint",
        snapshot,
        ("unknown", "chair"),
        code_commit="c" * 40,
        input_sha256="d" * 64,
    )
    assert (receipt.path / "entities.jsonl").stat().st_size > 64 * 1024
    loaded, _diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert loaded.entities[0].semantic_embedding is not None
    assert loaded.entities[0].semantic_embedding.shape == (10_000,)


def test_full_publisher_and_loader_reject_oversized_entity_records(tmp_path: Path) -> None:
    target = tmp_path / "too-large"
    with pytest.raises(ValueError, match="entity record|entities JSON|size limit"):
        publish_temporal_current_checkpoint(
            target,
            _snapshot_with_entity(semantic_id=1, prototype_dimension=100_000),
            ("unknown", "chair"),
            code_commit="c" * 40,
            input_sha256="d" * 64,
        )
    assert not target.exists()

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "valid", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    malicious = b'{"padding":"' + b"x" * (2 * 1024 * 1024) + b'"}\n'
    _rewrite_member(receipt.path, "entities.jsonl", malicious)
    with pytest.raises(ValueError, match="entity record|entities JSON|size limit"):
        load_temporal_current_checkpoint(receipt.path)


def test_full_roundtrip_accepts_large_bounded_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as temporal_module

    monkeypatch.setattr(temporal_module, "_MAX_JSON_BYTES", 8 * 1024)
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint",
        _snapshot_with_dormant_entities(64),
        ("unknown", "chair"),
        code_commit="c" * 40,
        input_sha256="d" * 64,
    )
    assert (receipt.path / "diagnostics.json").stat().st_size > 8 * 1024
    loaded, diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert loaded.entities == []
    assert len(diagnostics["dormant"]) == 64


def test_full_diagnostics_limit_is_symmetric_and_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.evaluation.oviv2_temporal_tesse as module

    monkeypatch.setattr(module, "_MAX_FULL_DIAGNOSTICS_BYTES", 1024)
    target = tmp_path / "too-large"
    with pytest.raises(ValueError, match="diagnostics.*size limit"):
        publish_temporal_current_checkpoint(
            target,
            _snapshot_with_dormant_entities(10),
            ("unknown", "chair"),
            code_commit="c" * 40,
            input_sha256="d" * 64,
        )
    assert list(tmp_path.iterdir()) == []

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "valid", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    malicious = b'{"dormant":[],"uncertain":[],"padding":"' + b"x" * 1024 + b'"}\n'
    _rewrite_member(receipt.path, "diagnostics.json", malicious)
    with pytest.raises(ValueError, match="diagnostics.*size limit"):
        load_temporal_current_checkpoint(receipt.path)


def test_full_publisher_normalizes_class_names_once_and_rejects_nonstrings(
    tmp_path: Path,
) -> None:
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "normalized",
        _snapshot_with_entity(semantic_id=1, prototype_dimension=2),
        (" unknown ", " chair "),
        code_commit="c" * 40,
        input_sha256="d" * 64,
    )
    manifest = json.loads((receipt.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["class_names"] == ["unknown", "chair"]
    loaded, _diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert loaded.entities[0].semantic_label == "chair"

    target = tmp_path / "invalid"
    with pytest.raises(TypeError, match="class_names"):
        publish_temporal_current_checkpoint(
            target,
            _snapshot(),
            ("unknown", 7),  # type: ignore[arg-type]
            code_commit="c" * 40,
            input_sha256="d" * 64,
        )
    assert not target.exists()


def test_full_nonempty_background_roundtrip(tmp_path: Path) -> None:
    config = TemporalGeometryConfig(0.1, 4.0, 2, 4, 4, 32, 0, 3, 0.5, 0.1, 2.0)
    background = TemporalBackgroundVolume(config)
    frame = Frame(
        frame_id=0,
        timestamp=0.0,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        pose=np.eye(4),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
    )
    background = background.trial_integrate(frame, frame.depth)
    snapshot = TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "b" * 64),
        (),
        background,
    )
    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", snapshot, ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    loaded, _diagnostics = load_temporal_current_checkpoint(receipt.path)
    assert loaded.background_xyz is not None
    assert len(loaded.background_xyz) > 0


def test_full_staging_write_failure_cleans_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oviv2.temporal_snapshot as module

    original = module._write_regular_at
    calls = 0

    def fail_second(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("write")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_write_regular_at", fail_second)
    with pytest.raises(RuntimeError, match="write"):
        publish_temporal_current_checkpoint(
            tmp_path / "target", _snapshot(), ("unknown",),
            code_commit="c" * 40, input_sha256="d" * 64,
        )
    assert calls == 2
    assert list(tmp_path.iterdir()) == []
