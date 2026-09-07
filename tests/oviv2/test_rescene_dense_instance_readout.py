from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.evaluation.rescene_pair_executor import postprocess_native_predictions
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    build_d2_inference_bundle,
)
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.oviv2.rescene_dense_instance_readout import (
    OWNER_BACKGROUND,
    OWNER_OVI_RESIDUAL,
    OWNER_QUERY,
    OWNER_UNKNOWN,
    _exclusive_query_winners,
    build_dense_instance_readout,
    build_dense_to_model_indices,
)


class _IdentityGridSample:
    def __init__(self, *, grid_size: float, **_options: object) -> None:
        self._grid_size = grid_size

    def __call__(self, data: dict[str, object]) -> dict[str, np.ndarray]:
        coordinates = np.asarray(data["coord"], dtype=np.float64)
        grid = np.floor(coordinates / self._grid_size).astype(np.int64)
        grid -= grid.min(axis=0)
        return {
            "adapter_index": np.asarray(data["adapter_index"], dtype=np.int64),
            "inverse": np.arange(len(coordinates), dtype=np.int64),
            "grid_coord": grid,
        }


def _visit(
    visit_id: int,
    points: list[list[float]],
    entity_points: list[list[int]],
    supported_points: set[int],
) -> OviObjectVisitView:
    xyz = np.asarray(points, dtype=np.float32)
    count = len(xyz)
    owners = np.full(count, -1, dtype=np.int64)
    entities = []
    labels = ("chair", "table")
    for entity_index, indices in enumerate(entity_points):
        rows = np.asarray(sorted(indices), dtype=np.int64)
        owners[rows] = entity_index
        entities.append(
            OviObjectEntityView(
                visit_id=visit_id,
                entity_id=f"ovimap:{entity_index + 1}",
                source_instance_id=entity_index + 1,
                point_indices=rows,
                palette_rgb=(40 + entity_index, 80, 120),
                semantic_embedding=np.asarray(
                    [1.0 - 0.5 * entity_index, 0.5 * entity_index],
                    dtype=np.float32,
                ),
                semantic_label=labels[entity_index],
                semantic_score=0.9,
                observation_frame_ids=(0,),
                observation_boxes_xyxy=((0, 0, 1, 1),),
            )
        )
    supported = np.zeros(count, dtype=np.bool_)
    supported[sorted(supported_points)] = True
    assert not np.any(supported & (owners < 0))
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=xyz,
        palette_rgb_uint8=np.full((count, 3), 100, dtype=np.uint8),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        normal_valid=np.ones(count, dtype=np.bool_),
        source_vertex_indices=np.arange(count, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([0], dtype=np.int64),
        entity_owner_indices=owners,
        entities=tuple(entities),
        camera_rgb_uint8=np.full((count, 3), 120, dtype=np.uint8),
        appearance_valid=supported,
        appearance_frame_ids=np.where(supported, 0, -1),
        appearance_source_frame_ids=np.where(supported, 0, -1),
        appearance_rows=np.where(supported, 2, -1),
        appearance_columns=np.where(supported, np.arange(count), -1),
        appearance_camera_depth_m=np.where(supported, 1.0, np.nan).astype(np.float32),
        appearance_observed_depth_m=np.where(supported, 1.0, np.nan).astype(np.float32),
        appearance_depth_residual_m=np.where(supported, 0.0, np.nan).astype(np.float32),
        native_manifest_sha256=str(visit_id + 1) * 64,
        materialized_manifest_sha256=str(visit_id + 3) * 64,
        source_artifact_sha256={
            "instance_color_log": "5" * 64,
            "instance_mesh": "6" * 64,
            "semantic_features": "7" * 64,
        },
        global_alignment_application=(
            "identity_reference" if visit_id == 0 else "rescan_to_reference_once"
        ),
    )


def _pair() -> OviObjectPairView:
    t0 = _visit(
        0,
        [
            [1.001, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.041, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        [[4, 1, 3], [0]],
        {0, 1, 3, 4},
    )
    t1 = _visit(
        1,
        [
            [0.105, 0.0, 0.0],
            [0.145, 0.0, 0.0],
            [0.101, 0.0, 0.0],
            [0.141, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        [[2, 0], [3, 1]],
        {0, 2},
    )
    return OviObjectPairView(
        pair_id="pair",
        visits=(t0, t1),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _bundle(tmp_path: Path):
    source = tmp_path / "transform.py"
    source.write_bytes(b"pinned sampler\n")
    return build_d2_inference_bundle(
        _pair(),
        sampler_source_path=source,
        sampler_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        grid_sample_factory=lambda **options: _IdentityGridSample(**options),
    )


def _bundle_for_pair(tmp_path: Path, pair: OviObjectPairView):
    source = tmp_path / "transform.py"
    source.write_bytes(b"pinned sampler\n")
    return build_d2_inference_bundle(
        pair,
        sampler_source_path=source,
        sampler_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        grid_sample_factory=lambda **options: _IdentityGridSample(**options),
    )


def _predictions(bundle) -> tuple[np.ndarray, np.ndarray]:
    assert len(bundle.model_input.model_visit_ids) == 4
    return (
        np.asarray(
            [[2.0, -2.0], [-2.0, 2.0], [-2.0, 2.0], [2.0, 2.0]],
            dtype=np.float32,
        ),
        np.asarray([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32),
    )


def test_dense_to_model_mapping_restores_nonidentity_original_vertex_rows(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)

    mapping = build_dense_to_model_indices(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
    )

    assert mapping[0].tolist() == [2, 0, -1, 1, 0, -1]
    assert mapping[1].tolist() == [3, -1, 3, -1, -1]


def test_residual_instances_preserve_missing_ovi_semantics(tmp_path: Path) -> None:
    pair = _pair()
    visits = []
    for visit in pair.visits:
        entities = list(visit.entities)
        entities[1] = replace(
            entities[1], semantic_embedding=None, semantic_label=None
        )
        visits.append(replace(visit, entities=tuple(entities)))
    pair = replace(pair, visits=(visits[0], visits[1]))
    bundle = _bundle_for_pair(tmp_path, pair)
    masks, logits = _predictions(bundle)

    readout = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=np.full_like(masks, -1.0),
        pred_logits_qc=logits,
    )

    missing = [
        instance
        for visit in readout.visits
        for instance in visit.instances
        if instance.parent_ovi_entity_point_counts[0][0] == "ovimap:2"
    ]
    assert len(missing) == 2
    assert all(instance.semantic_embedding is None for instance in missing)
    assert all(instance.semantic_labels == () for instance in missing)
    assert all(
        instance.semantic_provenance == "single_ovi_parent_embedding_unavailable"
        for instance in missing
    )


def test_query_parent_counts_are_sorted_by_entity_id(tmp_path: Path) -> None:
    pair = _pair()
    visits = []
    for visit in pair.visits:
        entities = (
            replace(
                visit.entities[0], entity_id="ovimap:2", source_instance_id=2
            ),
            replace(
                visit.entities[1], entity_id="ovimap:10", source_instance_id=10
            ),
        )
        visits.append(replace(visit, entities=entities))
    pair = replace(pair, visits=(visits[0], visits[1]))
    bundle = _bundle_for_pair(tmp_path, pair)
    masks, logits = _predictions(bundle)
    masks[:, 0] = 2.0
    masks[:, 1] = -2.0

    readout = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        minimum_query_score=0.0,
    )

    query = next(
        instance
        for instance in readout.visits[0].instances
        if instance.owner_source == "query"
    )
    assert tuple(name for name, _count in query.parent_ovi_entity_point_counts) == (
        "ovimap:10",
        "ovimap:2",
    )


def test_exclusive_readout_splits_and_merges_entities_without_losing_points(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    unknown = (
        np.asarray([False, False, False, False, False, True]),
        np.asarray([False, False, False, False, True]),
    )

    readout = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        unknown_point_masks=unknown,
        point_chunk_size=2,
    )

    t0, t1 = readout.visits
    assert t0.owner_instance_indices.tolist() == [1, 0, -1, 1, 0, -1]
    assert t0.owner_source_codes.tolist() == [
        OWNER_QUERY,
        OWNER_QUERY,
        OWNER_BACKGROUND,
        OWNER_QUERY,
        OWNER_QUERY,
        OWNER_UNKNOWN,
    ]
    assert [row.instance_id for row in t0.instances] == [
        "P2:t0:query_0000",
        "P2:t0:query_0001",
    ]
    assert t0.instances[0].parent_ovi_entity_point_counts == (("ovimap:1", 2),)
    assert t0.instances[1].parent_ovi_entity_point_counts == (
        ("ovimap:1", 1),
        ("ovimap:2", 1),
    )
    assert t0.instances[1].multi_object_conflict is True
    assert t0.instances[1].temporal_identity_id is None
    assert t1.owner_instance_indices.tolist() == [0, 1, 0, 1, -1]
    assert t1.owner_source_codes.tolist() == [
        OWNER_QUERY,
        OWNER_OVI_RESIDUAL,
        OWNER_QUERY,
        OWNER_OVI_RESIDUAL,
        OWNER_UNKNOWN,
    ]
    assert t1.instances[1].instance_id == "P2:t1:residual:ovimap:2"
    assert t1.instances[0].temporal_identity_id == "query_0000"
    assert sum(row.point_count for row in t0.instances) + 2 == pair.visits[0].point_count
    assert sum(row.point_count for row in t1.instances) + 1 == pair.visits[1].point_count
    assert readout.geometry_unchanged is True


def test_query_scores_match_native_postprocess_and_ties_choose_lower_raw_query(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    expected = postprocess_native_predictions(
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        adapter_to_model=bundle.model_input.adapter_to_model,
    )

    readout = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        point_chunk_size=1,
    )

    np.testing.assert_allclose(readout.query_scores, expected.query_scores)
    assert readout.raw_query_indices.tolist() == expected.retained_query_indices.tolist()
    assert readout.visits[1].owner_instance_indices[[0, 2]].tolist() == [0, 0]


def test_chunk_size_does_not_change_ownership_or_mutate_inputs(tmp_path: Path) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    masks, logits = _predictions(bundle)
    masks_before = masks.copy()
    logits_before = logits.copy()

    small = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        point_chunk_size=1,
    )
    large = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=masks,
        pred_logits_qc=logits,
        point_chunk_size=1024,
    )

    assert small.content_sha256() == large.content_sha256()
    for left, right in zip(small.visits, large.visits, strict=True):
        np.testing.assert_array_equal(left.owner_instance_indices, right.owner_instance_indices)
        np.testing.assert_array_equal(left.owner_source_codes, right.owner_source_codes)
    np.testing.assert_array_equal(masks, masks_before)
    np.testing.assert_array_equal(logits, logits_before)
    assert pair.content_sha256() == small.pair_content_sha256


def test_million_dense_rows_are_competed_only_in_bounded_chunks(monkeypatch) -> None:
    dense_count = 1_000_003
    chunk_size = 4096
    dense_to_model = np.zeros(dense_count, dtype=np.int64)
    masks = np.asarray([[2.0, 2.0]], dtype=np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-masks))
    query_scores = np.asarray([0.8, 0.8], dtype=np.float32)
    raw_queries = np.asarray([0, 1], dtype=np.int64)
    original_ix = np.ix_
    largest_dense_block = 0

    def bounded_ix(*arrays):
        nonlocal largest_dense_block
        largest_dense_block = max(largest_dense_block, len(arrays[0]))
        assert len(arrays[0]) <= chunk_size
        return original_ix(*arrays)

    monkeypatch.setattr(
        "src.oviv2.rescene_dense_instance_readout.np.ix_", bounded_ix
    )

    winners = _exclusive_query_winners(
        dense_to_model=dense_to_model,
        eligible_queries=raw_queries,
        query_scores=query_scores,
        masks_mq=masks,
        sigmoid_mq=sigmoid,
        point_chunk_size=chunk_size,
    )

    assert winners.shape == (dense_count,)
    assert np.all(winners == 0)
    assert largest_dense_block == chunk_size


def test_content_hash_witnesses_raw_proposals_when_exclusive_map_is_unchanged(
    tmp_path: Path,
) -> None:
    pair = _pair()
    bundle = _bundle(tmp_path)
    logits = np.asarray([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    first_masks = np.asarray(
        [[2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    second_masks = np.asarray(
        [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]],
        dtype=np.float32,
    )

    first = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=first_masks,
        pred_logits_qc=logits,
    )
    second = build_dense_instance_readout(
        pair,
        surface=bundle.surface,
        supported_view=bundle.supported_view,
        pred_masks_mq=second_masks,
        pred_logits_qc=logits,
    )

    for left, right in zip(first.visits, second.visits, strict=True):
        np.testing.assert_array_equal(
            left.owner_instance_indices, right.owner_instance_indices
        )
    np.testing.assert_allclose(first.query_scores, second.query_scores)
    assert not np.array_equal(
        first.raw_proposals[0][1].point_indices,
        second.raw_proposals[0][1].point_indices,
    )
    assert first.content_sha256() != second.content_sha256()
