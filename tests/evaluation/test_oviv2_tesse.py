from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.oviv2_tesse import (
    TesseCausalCheckpoint,
    build_neutral_current_snapshot,
    load_causal_checkpoints,
)
from src.oviv2.entities import EntityRegistry
from src.oviv2.evidence import SparseEvidenceStore
from src.oviv2.geometry import SparseTsdfVolume
from src.oviv2.meshing import LabeledMesh
from src.oviv2.ownership import ReversibleOwnershipStore
from src.oviv2.semantic_fusion import SemanticFusionConfig
from src.oviv2.semantic_memory import SparseClassPosterior
from src.oviv2.snapshot import VoxelMapSnapshot, VoxelSnapshotMetadata


def _write_schedule(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_id": "tesse_cd_causal_schedule_v2",
                "dataset": "TESSE-CD",
                "method_predictions_used": False,
                "parameters": {"frame_indexing": "zero_based"},
                "scenes": {
                    "apartment": {"frame_count": 5, "entries": entries},
                    "office": {"frame_count": 8, "entries": []},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _entry(frame: int, timestamp: int) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ns": timestamp,
        "relative_timestamp_ns": timestamp - 100,
        "event_ids": ["event-1"],
        "roles": ["common_v2"],
    }


def test_loads_sorted_frozen_causal_checkpoints(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule, [_entry(4, 500), _entry(0, 100), _entry(2, 300)])

    checkpoints = load_causal_checkpoints(
        schedule,
        scene="apartment",
        frame_count=5,
    )

    assert checkpoints == (
        TesseCausalCheckpoint(0, 100, 0, ("event-1",), ("common_v2",)),
        TesseCausalCheckpoint(2, 300, 200, ("event-1",), ("common_v2",)),
        TesseCausalCheckpoint(4, 500, 400, ("event-1",), ("common_v2",)),
    )
    assert isinstance(checkpoints[0].event_ids, tuple)
    with pytest.raises((AttributeError, TypeError)):
        checkpoints[0].event_ids[0] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("scene", "frame_count", "entries", "message"),
    [
        ("warehouse", 5, [_entry(0, 100)], "unknown scene"),
        ("apartment", 6, [_entry(0, 100)], "frame count mismatch"),
        ("apartment", 5, [_entry(0, 100), _entry(0, 200)], "duplicate"),
        ("apartment", 5, [_entry(5, 600)], "outside"),
        ("apartment", 5, [_entry(0, 200), _entry(2, 100)], "timestamps"),
    ],
)
def test_rejects_invalid_causal_schedule(
    tmp_path: Path,
    scene: str,
    frame_count: int,
    entries: list[dict[str, object]],
    message: str,
) -> None:
    schedule = tmp_path / "schedule.json"
    _write_schedule(schedule, entries)

    with pytest.raises(ValueError, match=message):
        load_causal_checkpoints(schedule, scene=scene, frame_count=frame_count)


def _snapshot() -> VoxelMapSnapshot:
    registry = EntityRegistry()
    registry.entities = {
        7: SimpleNamespace(
            entity_id=7,
            lifecycle_state="active",
            semantic_posterior=SparseClassPosterior(((1, 0.0),), 1.0),
            first_frame_id=0,
            last_frame_id=1,
        ),
        8: SimpleNamespace(
            entity_id=8,
            lifecycle_state="active",
            semantic_posterior=SparseClassPosterior(((2, 0.0),), 1.0),
            first_frame_id=0,
            last_frame_id=1,
        ),
        99: SimpleNamespace(
            entity_id=99,
            lifecycle_state="dormant",
            semantic_posterior=SparseClassPosterior(((1, 0.0),), 1.0),
            first_frame_id=0,
            last_frame_id=0,
        ),
    }
    geometry = SparseTsdfVolume()
    return VoxelMapSnapshot(
        path=Path("checkpoint"),
        metadata=VoxelSnapshotMetadata(
            scene_id="apartment",
            frame_id=1,
            timestamp=123 / 1_000_000_000,
            revision=2,
            voxel_size_m=geometry.config.voxel_size_m,
            block_resolution=geometry.config.block_resolution,
            schema_version=2,
        ),
        geometry=geometry,
        evidence=SparseEvidenceStore(),
        ownership=ReversibleOwnershipStore(),
        checksums={},
        registry=registry,
    )


def test_builds_current_neutral_map_from_owned_object_vertices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = LabeledMesh(
        vertices_xyz=np.asarray(
            [
                [0.16, 0.0, 0.0],  # owner 7, later voxel
                [0.05, 0.0, 0.0],  # released object semantic
                [0.10, 0.0, 0.0],  # structure
                [0.00, 0.0, 0.0],  # unknown
                [0.20, 0.0, 0.0],  # owned non-object semantic
                [0.15, 0.0, 0.0],  # owner 7, earlier voxel
            ],
            dtype=np.float32,
        ),
        triangles=np.empty((0, 3), dtype=np.int64),
        colors_rgb=np.zeros((6, 3), dtype=np.float32),
        semantic_ids=np.asarray([1, 1, 2, 0, 2, 1], dtype=np.int64),
        entity_ids=np.asarray([7, 0, 0, 0, 8, 7], dtype=np.int64),
        semantic_confidence=np.asarray([0.8, 0.7, 0.9, 0.0, 0.6, 0.9]),
        ownership_confidence=np.asarray([0.8, 0.0, 0.0, 0.0, 0.7, 0.9]),
    )
    captured: dict[str, object] = {}

    def fake_derive(*args: object, **kwargs: object) -> LabeledMesh:
        captured.update(kwargs)
        return mesh

    monkeypatch.setattr("src.evaluation.oviv2_tesse.derive_labeled_mesh", fake_derive)

    neutral = build_neutral_current_snapshot(
        _snapshot(),
        timestamp_ns=123,
        class_names=("unknown", "chair", "wall"),
        object_semantic_ids=frozenset({1}),
        fusion=SemanticFusionConfig(entity_weight_scale=0.49),
        timestamp_ns_by_frame=(100, 123),
    )

    assert neutral.method == "OVIV2"
    assert neutral.scene_id == "apartment"
    assert neutral.timestamp == 123.0
    assert neutral.scope == "current"
    assert [entity.entity_id for entity in neutral.entities] == [
        "oviv2:7:semantic:1"
    ]
    entity = neutral.entities[0]
    assert entity.semantic_label == "chair"
    assert entity.semantic_score == pytest.approx(0.85)
    assert entity.metadata == {
        "entity_type": "object",
        "owner_entity_id": 7,
        "semantic_id": 1,
    }
    np.testing.assert_allclose(
        entity.points_xyz,
        [[0.15, 0.0, 0.0], [0.16, 0.0, 0.0]],
    )
    assert neutral.background_xyz is not None
    np.testing.assert_allclose(
        neutral.background_xyz,
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ],
    )
    assert "oviv2:99:semantic:1" not in {
        item.entity_id for item in neutral.entities
    }
    assert captured["semantic_fusion"] == SemanticFusionConfig(
        entity_weight_scale=0.49
    )
    assert captured["entity_posteriors"] == {
        7: ((1, 1.0),),
        8: ((2, 1.0),),
        99: ((1, 1.0),),
    }


def test_neutral_snapshot_rejects_timestamp_mismatch() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        build_neutral_current_snapshot(
            _snapshot(),
            timestamp_ns=124,
            class_names=("unknown", "chair", "wall"),
            object_semantic_ids=frozenset({1}),
            fusion=SemanticFusionConfig(),
            timestamp_ns_by_frame=(100, 123),
        )


def test_neutral_snapshot_rejects_subnanosecond_metadata_timestamp_drift() -> None:
    snapshot = _snapshot()
    snapshot = replace(
        snapshot,
        metadata=replace(
            snapshot.metadata,
            timestamp=123 / 1_000_000_000 + 0.5e-9,
        ),
    )

    with pytest.raises(ValueError, match="metadata timestamp"):
        build_neutral_current_snapshot(
            snapshot,
            timestamp_ns=123,
            class_names=("unknown", "chair", "wall"),
            object_semantic_ids=frozenset({1}),
            fusion=SemanticFusionConfig(),
            timestamp_ns_by_frame=(100, 123),
        )


@pytest.mark.parametrize(
    ("first_frame", "last_frame", "message"),
    [
        (0, 2, "future|current frame"),
        (1, 0, "first frame"),
        (-1, 0, "at least 0"),
    ],
)
def test_neutral_snapshot_rejects_invalid_registry_frame_range(
    first_frame: int,
    last_frame: int,
    message: str,
) -> None:
    snapshot = _snapshot()
    assert snapshot.registry is not None
    snapshot.registry.entities[99].first_frame_id = first_frame
    snapshot.registry.entities[99].last_frame_id = last_frame

    with pytest.raises(ValueError, match=message):
        build_neutral_current_snapshot(
            snapshot,
            timestamp_ns=123,
            class_names=("unknown", "chair", "wall"),
            object_semantic_ids=frozenset({1}),
            fusion=SemanticFusionConfig(),
            timestamp_ns_by_frame=(100, 123, 200),
        )
