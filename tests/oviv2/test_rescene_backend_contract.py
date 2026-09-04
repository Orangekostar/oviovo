from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.oviv2.rescene_backend import ReSceneBackend, ReSceneBackendError
from src.oviv2.two_visit_contracts import NeuralSampleMap


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "rescene"
    config = checkout / "conf/backbone/concerto.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("name: concerto_base\n", encoding="utf-8")
    (checkout / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "source")
    commit = _git(checkout, "rev-parse", "HEAD")
    files = []
    for relative in ("LICENSE", "conf/backbone/concerto.yaml"):
        content = (checkout / relative).read_bytes()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "EXTERNAL_SOURCE_PASS",
                "sources": {
                    "rescene": {
                        "commit": commit,
                        "required_files": files,
                    }
                },
                "checkpoint": {
                    "selected_backbone": "Concerto",
                    "status": "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
                },
            }
        ),
        encoding="utf-8",
    )
    return checkout, manifest


def _pair() -> NeuralSampleMap:
    return NeuralSampleMap(
        coordinates_xyzt=np.array([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 1.0]]),
        features=np.ones((2, 3), dtype=np.float32),
        visit_ids=np.array([0, 1], dtype=np.int8),
        source_visit_ids=np.array([0, 1], dtype=np.int8),
        source_entity_ids=("t0:a", "t1:a"),
        source_point_indices=np.array([0, 1], dtype=np.int64),
        source_to_token_offsets=np.array([0, 1, 2], dtype=np.int64),
        neural_voxel_size_m=0.02,
        feature_schema="rgb",
        coordinate_frame_id="world",
        source_manifest_sha256=SHA_A,
        source_visit_map_sha256=(SHA_B, SHA_C),
    )


def test_missing_checkpoint_is_explicitly_blocked_without_predictions(tmp_path: Path) -> None:
    checkout, manifest = _source(tmp_path)
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=tmp_path / "missing.ckpt",
        source_manifest=manifest,
    )

    result = backend.infer(_pair())

    assert result.status == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert result.query_masks is None
    assert result.ranking_eligible is False
    assert result.diagnostics["reason"] == "checkpoint_missing"


def test_dirty_required_source_is_blocked_before_checkpoint_access(tmp_path: Path) -> None:
    checkout, manifest = _source(tmp_path)
    (checkout / "conf/backbone/concerto.yaml").write_text(
        "name: changed\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        source_manifest=manifest,
    )

    result = backend.infer(_pair())

    assert result.status == "BLOCKED_EXTERNAL_SOURCE"
    assert result.query_masks is None
    assert result.diagnostics["reason"] == "source_identity_mismatch"


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    checkout, manifest = _source(tmp_path)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    backend = ReSceneBackend(
        checkout=checkout,
        checkpoint=checkpoint,
        checkpoint_sha256="f" * 64,
        source_manifest=manifest,
    )

    with pytest.raises(ReSceneBackendError, match="checkpoint SHA-256"):
        backend.infer(_pair())
