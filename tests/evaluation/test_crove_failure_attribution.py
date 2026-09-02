from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_failure_attribution import (
    ALL_AUTHORITY_BUCKETS,
    OBJECT_AUTHORITY_BUCKETS,
    attribute_authority_buckets,
    build_overlay_funnel,
    partition_background_authorities,
    partition_object_authorities,
    validate_file_record,
)


def _entity(
    entity_id: str,
    point: tuple[float, float, float],
    *,
    authority: str,
    overlay_state: str,
    temporal_entity_id: int | None = None,
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray([point], dtype=np.float32),
        semantic_embedding=None,
        semantic_label="Chair",
        semantic_score=1.0,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=1.0,
        metadata={
            "authority": authority,
            "overlay_state": overlay_state,
            "temporal_entity_id": temporal_entity_id,
        },
    )


def _snapshot(
    entities: list[EntityPrediction], background: list[tuple[float, float, float]]
) -> MapSnapshot:
    return MapSnapshot(
        method="OVIV2",
        scene_id="apartment",
        timestamp=1.0,
        entities=entities,
        background_xyz=np.asarray(background, dtype=np.float32).reshape((-1, 3)),
        scope="current",
    )


def test_object_partition_covers_four_authorities_exactly() -> None:
    snapshot = _snapshot(
        [
            _entity(
                "ovimap:1",
                (0.01, 0.01, 0.01),
                authority="ovimap_anchor",
                overlay_state="unchanged",
                temporal_entity_id=11,
            ),
            _entity(
                "ovimap:2",
                (0.11, 0.01, 0.01),
                authority="ovimap_anchor",
                overlay_state="occluded",
            ),
            _entity(
                "ovimap:3",
                (0.21, 0.01, 0.01),
                authority="crove_temporal",
                overlay_state="moved",
                temporal_entity_id=13,
            ),
            _entity(
                "temporal:20",
                (0.31, 0.01, 0.01),
                authority="crove_temporal",
                overlay_state="new",
                temporal_entity_id=20,
            ),
        ],
        [],
    )

    buckets = partition_object_authorities(snapshot)

    assert tuple(buckets) == OBJECT_AUTHORITY_BUCKETS
    assert {name: len(points) for name, points in buckets.items()} == {
        "ovimap_anchor_bound_unchanged": 1,
        "ovimap_anchor_unbound": 1,
        "crove_temporal_moved": 1,
        "crove_temporal_new": 1,
    }
    np.testing.assert_array_equal(
        np.concatenate(tuple(buckets.values())),
        np.concatenate([entity.points_xyz for entity in snapshot.entities]),
    )


@pytest.mark.parametrize(
    ("authority", "overlay_state"),
    [("ovimap_anchor", "moved"), ("crove_temporal", "unchanged"), ("other", "new")],
)
def test_object_partition_rejects_unknown_authority_combinations(
    authority: str, overlay_state: str
) -> None:
    snapshot = _snapshot(
        [
            _entity(
                "bad",
                (0.0, 0.0, 0.0),
                authority=authority,
                overlay_state=overlay_state,
            )
        ],
        [],
    )
    with pytest.raises(ValueError, match="authority"):
        partition_object_authorities(snapshot)


def test_background_partition_replays_anchor_first_lexicographic_tie_rule() -> None:
    anchor = np.asarray([[0.01, 0.01, 0.01], [0.22, 0.01, 0.01]], dtype=np.float32)
    temporal = np.asarray(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.01, 0.01],
            [0.11, 0.01, 0.01],
            [0.21, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    composed = np.asarray(
        [[0.01, 0.01, 0.01], [0.11, 0.01, 0.01], [0.21, 0.01, 0.01]],
        dtype=np.float32,
    )

    buckets = partition_background_authorities(
        composed, anchor, temporal, voxel_size_m=0.1
    )

    np.testing.assert_allclose(buckets["anchor_background"], [[0.01, 0.01, 0.01]])
    np.testing.assert_allclose(
        buckets["temporal_background"],
        [[0.11, 0.01, 0.01], [0.21, 0.01, 0.01]],
    )


def test_background_partition_rejects_noncomposed_points() -> None:
    with pytest.raises(ValueError, match="composition"):
        partition_background_authorities(
            np.asarray([[0.04, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.01, 0.0, 0.0]], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            voxel_size_m=0.1,
        )


def test_ghost_attribution_partitions_official_objects_and_extended_background() -> (
    None
):
    buckets = {
        name: np.empty((0, 3), dtype=np.float32) for name in ALL_AUTHORITY_BUCKETS
    }
    buckets["ovimap_anchor_bound_unchanged"] = np.asarray(
        [[0.025, 0.025, 0.025], [0.075, 0.025, 0.025]], dtype=np.float32
    )
    buckets["crove_temporal_moved"] = np.asarray(
        [[0.024, 0.025, 0.025]], dtype=np.float32
    )
    buckets["anchor_background"] = np.asarray([[0.026, 0.025, 0.025]], dtype=np.float32)
    region = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    free = np.asarray([[0, 0, 0]], dtype=np.int64)

    result = attribute_authority_buckets(
        buckets,
        region_voxels=region,
        confirmed_free_voxels=free,
        voxel_size_m=0.05,
        distance_threshold_m=0.05,
    )

    assert result["buckets"]["ovimap_anchor_bound_unchanged"] == {
        "predicted_points": 2,
        "ghost_matches": 1,
        "ghost_rate": 0.5,
    }
    assert result["buckets"]["crove_temporal_moved"]["ghost_matches"] == 1
    assert result["buckets"]["anchor_background"]["ghost_matches"] == 1
    assert result["official_object_ghost_contribution_fraction"] == {
        "ovimap_anchor_bound_unchanged": 0.5,
        "ovimap_anchor_unbound": 0.0,
        "crove_temporal_moved": 0.5,
        "crove_temporal_new": 0.0,
    }
    assert result["extended_all_authority_ghost_contribution_fraction"][
        "anchor_background"
    ] == pytest.approx(1 / 3)
    assert result["official_object_partition"] == {
        "predicted_points": 3,
        "ghost_matches": 2,
        "ghost_rate": pytest.approx(2 / 3),
        "partition_exact": True,
    }
    assert result["extended_all_authority"]["predicted_points"] == 4
    assert result["extended_all_authority"]["ghost_matches"] == 3


def test_ghost_distance_rule_is_strictly_less_than_threshold() -> None:
    buckets = {
        name: np.empty((0, 3), dtype=np.float32) for name in ALL_AUTHORITY_BUCKETS
    }
    buckets["crove_temporal_new"] = np.asarray(
        [[0.075, 0.025, 0.025]], dtype=np.float32
    )
    result = attribute_authority_buckets(
        buckets,
        region_voxels=np.asarray([[1, 0, 0]], dtype=np.int64),
        confirmed_free_voxels=np.asarray([[0, 0, 0]], dtype=np.int64),
        voxel_size_m=0.05,
        distance_threshold_m=0.05,
    )
    assert result["official_object_partition"]["ghost_matches"] == 0


def test_overlay_funnel_records_every_anchor_and_gate_component() -> None:
    anchor = _snapshot(
        [
            _entity(
                "ovimap:1",
                (0.0, 0.0, 0.0),
                authority="ovimap_anchor",
                overlay_state="unchanged",
            ),
            _entity(
                "ovimap:2",
                (1.0, 0.0, 0.0),
                authority="ovimap_anchor",
                overlay_state="unchanged",
            ),
            _entity(
                "ovimap:3",
                (2.0, 0.0, 0.0),
                authority="ovimap_anchor",
                overlay_state="unchanged",
            ),
        ],
        [],
    )
    trajectories = [
        {
            "frame_index": 0,
            "entity_id": 11,
            "centroid_xyz": [0.0, 0.0, 0.0],
            "dynamic_state": "static",
            "geometry_epoch": 0,
            "readout_valid": True,
        },
        {
            "frame_index": 10,
            "entity_id": 11,
            "centroid_xyz": [0.4, 0.0, 0.0],
            "dynamic_state": "dynamic",
            "geometry_epoch": 1,
            "readout_valid": True,
        },
    ]
    diagnostics = {
        "frame_index": 10,
        "moved_anchor_ids": ["ovimap:1"],
        "removed_anchor_ids": ["ovimap:3"],
        "occluded_anchor_ids": [],
        "unchanged_anchor_ids": ["ovimap:2"],
        "new_temporal_ids": [],
    }

    records = build_overlay_funnel(
        anchor,
        bindings={"ovimap:1": 11, "ovimap:3": 13},
        trajectory_rows=trajectories,
        cutoff_frame=0,
        checkpoint_frame=10,
        diagnostics=diagnostics,
        moved_displacement_m=0.25,
    )

    assert [item["anchor_entity_id"] for item in records] == [
        "ovimap:1",
        "ovimap:2",
        "ovimap:3",
    ]
    moved, unbound, removed = records
    assert moved == {
        "anchor_entity_id": "ovimap:1",
        "bound": True,
        "temporal_entity_id": 11,
        "dynamic_state": "dynamic",
        "geometry_epoch": 1,
        "initial_geometry_epoch": 0,
        "geometry_epoch_advanced": True,
        "anchor_to_current_displacement_m": pytest.approx(0.4),
        "moved_threshold_passed": True,
        "moved": True,
        "removed": False,
        "occluded": False,
        "new": False,
        "readout_valid": True,
    }
    assert unbound["bound"] is False
    assert unbound["dynamic_state"] is None
    assert unbound["moved"] is False
    assert removed["bound"] is True
    assert removed["removed"] is True
    assert removed["geometry_epoch"] is None


def test_file_record_validation_detects_content_tampering(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}\n", encoding="utf-8")
    import hashlib

    record = {
        "path": path.name,
        "byte_count": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert validate_file_record(tmp_path, record, label="artifact") == path
    path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        validate_file_record(tmp_path, record, label="artifact")


def test_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/attribute_crove_tesse_failures.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--composed-run-manifest" in result.stdout
    assert "--target-manifest" in result.stdout
