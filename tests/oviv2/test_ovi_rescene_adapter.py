from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.ovi_rescene_adapter import (
    AdapterConfig,
    AdapterError,
    adapt_visit_pair,
    load_neural_sample_artifact,
    write_neural_sample_artifact,
)
from src.oviv2.two_visit_contracts import VisitMap, snapshot_content_sha256


SOURCE_SHA256 = "a" * 64


def _entity(
    entity_id: str,
    points: list[list[float]],
    colors: list[list[int]],
) -> EntityPrediction:
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray([0.25, 0.75], dtype=np.float32),
        semantic_label="chair",
        semantic_score=0.9,
        lifecycle_state="observed",
        first_seen=0.0,
        last_seen=1.0,
        metadata={
            "point_rgb": np.asarray(colors, dtype=np.uint8),
            "point_rgb_source": "camera_rgb",
            "point_normals": np.tile(
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                (len(points), 1),
            ),
            "point_normals_source": "source_geometry",
        },
    )


def _visit(visit_id: int, *, frame: str = "world") -> VisitMap:
    if visit_id == 0:
        entities = [
            _entity(
                "ovi:t0:chair",
                [[0.001, 0.001, 0.001], [0.019, 0.001, 0.001]],
                [[255, 0, 0], [127, 0, 0]],
            ),
            _entity(
                "ovi:t0:table",
                [[0.001, 0.001, 0.001]],
                [[0, 0, 255]],
            ),
        ]
        start = 0
    else:
        entities = [
            _entity(
                "ovi:t1:chair",
                [[0.021, 0.001, 0.001], [0.039, 0.001, 0.001]],
                [[0, 255, 0], [0, 127, 0]],
            )
        ]
        start = 10
    snapshot = MapSnapshot(
        method="OVI-MAP",
        scene_id="apartment",
        timestamp=float(start),
        entities=entities,
        background_xyz=None,
        scope="current",
    )
    return VisitMap(
        visit_id=visit_id,
        snapshot=snapshot,
        coordinate_frame_id=frame,
        source_manifest_sha256=SOURCE_SHA256,
        map_voxel_size_m=0.01,
        observed_frame_start=start,
        observed_frame_end=start + 4,
    )


def test_two_centimeter_spatial_grouping_is_deterministic_and_time_is_unscaled() -> None:
    first = adapt_visit_pair(_visit(0), _visit(1), AdapterConfig(0.02, "rgb"))
    second = adapt_visit_pair(_visit(0), _visit(1), AdapterConfig(0.02, "rgb"))

    assert first.content_sha256() == second.content_sha256()
    assert set(first.coordinates_xyzt[:, 3]) == {0.0, 1.0}
    assert first.coordinates_xyzt.shape == (3, 4)
    assert first.token_entity_ids == (
        "ovi:t0:chair",
        "ovi:t0:table",
        "ovi:t1:chair",
    )


def test_reverse_index_conserves_every_source_point_once() -> None:
    pair = adapt_visit_pair(_visit(0), _visit(1), AdapterConfig(0.02, "rgb"))

    assert sorted(pair.source_point_indices.tolist()) == list(
        range(pair.source_point_count)
    )
    assert pair.source_to_token_offsets.tolist() == [0, 2, 3, 5]
    assert pair.source_entity_ids[:2] == ("ovi:t0:chair", "ovi:t0:chair")


@pytest.mark.parametrize("voxel_size_m", [0.01, 0.02, 0.04])
def test_only_neural_voxel_size_changes(voxel_size_m: float) -> None:
    t0, t1 = _visit(0), _visit(1)
    pair = adapt_visit_pair(t0, t1, AdapterConfig(voxel_size_m, "rgb"))

    assert pair.neural_voxel_size_m == voxel_size_m
    assert t0.map_voxel_size_m == t1.map_voxel_size_m == 0.01
    assert pair.source_point_count == 5


def test_adapter_does_not_mutate_ovi_maps() -> None:
    t0, t1 = _visit(0), _visit(1)
    before = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))

    adapt_visit_pair(t0, t1, AdapterConfig(0.02, "rgb_normals"))

    after = (snapshot_content_sha256(t0.snapshot), snapshot_content_sha256(t1.snapshot))
    assert before == after == (t0.snapshot_sha256, t1.snapshot_sha256)


def test_adapter_rejects_missing_source_rgb() -> None:
    t0, t1 = _visit(0), _visit(1)
    del t0.snapshot.entities[0].metadata["point_rgb"]
    with pytest.raises((AdapterError, ValueError), match="point_rgb|changed"):
        adapt_visit_pair(t0, t1, AdapterConfig(0.02, "rgb"))


def test_adapter_rejects_instance_palette_as_rgb() -> None:
    t0, t1 = _visit(0), _visit(1)
    metadata = t0.snapshot.entities[0].metadata
    metadata["point_rgb_source"] = "instance_palette"
    t0 = VisitMap(
        visit_id=0,
        snapshot=t0.snapshot,
        coordinate_frame_id=t0.coordinate_frame_id,
        source_manifest_sha256=t0.source_manifest_sha256,
        map_voxel_size_m=0.01,
        observed_frame_start=0,
        observed_frame_end=4,
    )

    with pytest.raises(AdapterError, match="instance palette"):
        adapt_visit_pair(t0, t1, AdapterConfig(0.02, "rgb"))


def test_adapter_rejects_mixed_coordinate_frames() -> None:
    with pytest.raises(ValueError, match="coordinate frame"):
        adapt_visit_pair(
            _visit(0),
            _visit(1, frame="other"),
            AdapterConfig(0.02, "rgb"),
        )


def test_adapter_rejects_unsupported_features_and_mapping_voxel_change() -> None:
    with pytest.raises(ValueError, match="feature_schema"):
        AdapterConfig(0.02, "rgb_siglip")

    t0, t1 = _visit(0), _visit(1)
    changed = VisitMap(
        visit_id=0,
        snapshot=t0.snapshot,
        coordinate_frame_id=t0.coordinate_frame_id,
        source_manifest_sha256=t0.source_manifest_sha256,
        map_voxel_size_m=0.02,
        observed_frame_start=0,
        observed_frame_end=4,
    )
    with pytest.raises(AdapterError, match="OVI mapping voxel"):
        adapt_visit_pair(changed, t1, AdapterConfig(0.02, "rgb"))


def test_hash_bound_pair_artifact_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    pair = adapt_visit_pair(_visit(0), _visit(1), AdapterConfig(0.02, "rgb"))
    output = tmp_path / "pair"

    paths = write_neural_sample_artifact(pair, output)
    restored = load_neural_sample_artifact(output)

    assert restored.content_sha256() == pair.content_sha256()
    assert paths.manifest == output / "manifest.json"
    assert paths.arrays == output / "arrays.npz"
    with pytest.raises(ValueError, match="already exists"):
        write_neural_sample_artifact(pair, output)

    original = paths.arrays.read_bytes()
    paths.arrays.write_bytes(original + b"tamper")
    assert hashlib.sha256(paths.arrays.read_bytes()).hexdigest() != hashlib.sha256(
        original
    ).hexdigest()
    with pytest.raises(AdapterError, match="SHA-256"):
        load_neural_sample_artifact(output)
