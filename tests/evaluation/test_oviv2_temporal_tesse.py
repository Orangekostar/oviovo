from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.oviv2_temporal_tesse import publish_temporal_current_checkpoint
from src.evaluation.oviv2_temporal_tesse import load_temporal_current_checkpoint
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_config import TemporalGeometryConfig
from src.oviv2.temporal_snapshot import TemporalCurrentSnapshot, TemporalSnapshotMetadata


def _snapshot() -> TemporalCurrentSnapshot:
    config = TemporalGeometryConfig(0.1, 4.0, 2, 4, 4, 16, 0, 3, 0.5, 0.1, 2.0)
    return TemporalCurrentSnapshot(
        TemporalSnapshotMetadata("scene", 0, 0.0, 1, 0.1, "b" * 64),
        (),
        TemporalBackgroundVolume(config),
    )


def _files(path: Path) -> set[str]:
    return {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}


def test_full_publication_is_atomic_deterministic_and_has_diagnostics(tmp_path: Path) -> None:
    first = publish_temporal_current_checkpoint(
        tmp_path / "first", _snapshot(), ("unknown", "chair"),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    second = publish_temporal_current_checkpoint(
        tmp_path / "second", _snapshot(), ("unknown", "chair"),
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
    assert json.loads((first.path / "diagnostics.json").read_text(encoding="utf-8")) == {
        "dormant": [], "uncertain": []
    }
    first.revalidate_source()
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
    import hashlib
    import src.evaluation.oviv2_temporal_tesse as module

    receipt = publish_temporal_current_checkpoint(
        tmp_path / "checkpoint", _snapshot(), ("unknown",),
        code_commit="c" * 40, input_sha256="d" * 64,
    )
    entities = [
        EntityPrediction(
            entity_id=f"temporal:{entity_id}", points_xyz=np.empty((0, 3)),
            semantic_embedding=None, semantic_label=None, semantic_score=0.0,
            lifecycle_state="active", first_seen=0.0, last_seen=0.0,
            metadata={"temporal_entity_id": entity_id, "semantic_id": 0},
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
