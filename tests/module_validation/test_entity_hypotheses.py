"""Native-leaf construction, complete partitions, and final ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.static_ovmap.module_validation.entity_hypotheses import (
    ConflictGroup,
    PartitionHypothesis,
    apply_selected_partitions,
    build_conflict_groups,
    build_frame_leaf_evidence,
    build_native_leaves,
    build_surface_graph,
    finalize_geometry_map,
    generate_complete_hypotheses,
    project_frame_leaf_evidence,
    select_geometry_frames,
    write_final_geometry_map,
)


def _surface():
    xyz = np.array(
        [
            [0.00, 0.00, 1.0],
            [0.01, 0.00, 1.0],
            [0.02, 0.00, 1.0],
            [0.03, 0.00, 1.0],
            [1.00, 1.00, 1.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    normals = np.tile([0.0, 0.0, 1.0], (5, 1))
    valid = np.ones(5, dtype=bool)
    owners = np.array([1, 1, 2, 2, 0], dtype=np.int64)
    segments = np.array([11, 10, 20, 20, 0], dtype=np.int64)
    return xyz, faces, normals, valid, owners, segments


def test_native_leaves_resolve_aliases_preserve_owner0_and_lose_no_rows() -> None:
    xyz, faces, normals, valid, owners, segments = _surface()
    graph = build_surface_graph(xyz, faces, normals, valid)
    leaves = build_native_leaves(
        xyz,
        owners,
        segments,
        alias_table=((11, 10),),
        surface_graph=graph,
    )

    assert graph.source == "MESH_FACES"
    assert leaves.row_leaf_ids.tolist() == [0, 0, 1, 1, 2]
    assert [leaf.owner_id for leaf in leaves.leaves] == [1, 2, 0]
    assert [leaf.segment_label for leaf in leaves.leaves] == [10, 20, 0]
    assert sorted(np.concatenate([leaf.source_rows for leaf in leaves.leaves]).tolist()) == list(range(5))
    assert sum(contact.edge_count for contact in leaves.contacts) == 3


def test_mutual_eight_neighbor_fallback_is_deterministic_and_bounded() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.005, 0.01, 0.0], [1.0, 0.0, 0.0]]
    )
    graph = build_surface_graph(
        xyz,
        np.empty((0, 3), dtype=np.int64),
        np.zeros_like(xyz),
        np.zeros(4, dtype=bool),
    )

    assert graph.source == "MUTUAL_8NN_0.03M"
    assert graph.edges.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert np.all(graph.edges[:, 0] < graph.edges[:, 1])
    assert graph.normal_valid[:3].all()


def test_frame_entities_use_pixel_denominator_and_unknown_thresholds() -> None:
    leaf_pixels = np.array([0] * 16 + [1] * 20 + [2] * 15, dtype=np.int64)
    entities = np.array(
        [7] * 10 + [0] * 6 + [7] * 12 + [8] * 8 + [9] * 15,
        dtype=np.int64,
    )
    evidence = build_frame_leaf_evidence(4, leaf_pixels, entities, leaf_count=3)

    assert evidence.observed_pixels.tolist() == [16, 20, 15]
    assert evidence.entity_by_leaf.tolist() == [7, 7, 0]
    assert evidence.dominant_pixels.tolist() == [10, 12, 15]


def test_sparse_leaf_evidence_matches_histogram_reference_with_unknowns_and_ties() -> None:
    rng = np.random.default_rng(17)
    pixels = np.concatenate((rng.integers(0, 30, 1500), np.repeat(99, 20)))
    entities = np.concatenate((rng.integers(0, 4, 1500), [8] * 10 + [7] * 10))
    result = build_frame_leaf_evidence(3, pixels, entities, leaf_count=100, dominance=.5)
    expected = np.zeros(100, np.int64)
    dominant = np.zeros(100, np.int64)
    for leaf in range(100):
        labels = entities[pixels == leaf]
        if not np.any(labels > 0):
            continue
        counts = np.bincount(labels)
        counts[0] = 0
        winner = int(np.argmax(counts))
        dominant[leaf] = counts[winner]
        if len(labels) >= 16 and counts[winner] >= .5 * len(labels):
            expected[leaf] = winner
    np.testing.assert_array_equal(result.entity_by_leaf, expected)
    np.testing.assert_array_equal(result.dominant_pixels, dominant)
    assert result.entity_by_leaf[99] == 7
    assert result.observed_pixels.sum() == len(pixels)


def test_projection_uses_depth_consistency_and_one_zbuffer_winner_per_pixel() -> None:
    xyz = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.01], [1.0, 0.0, 1.0]])
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1:] = 1.0
    entities = np.zeros((3, 3), dtype=np.int64)
    entities[1, 1:] = [4, 5]
    evidence = project_frame_leaf_evidence(
        3,
        xyz,
        np.array([0, 1, 2]),
        np.eye(4),
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        depth,
        entities,
    )

    assert evidence.pixel_leaf_ids.tolist() == [0, 2]
    assert evidence.pixel_entity_ids.tolist() == [4, 5]
    assert evidence.observed_pixels.tolist() == [1, 0, 1]


def test_geometry_frame_selection_deduplicates_poses_and_evenly_caps() -> None:
    frames = []
    for frame_id in range(40):
        pose = np.eye(4)
        pose[0, 3] = frame_id
        frames.append(SimpleNamespace(frame_id=frame_id, pose_c2w=pose))
    frames.append(SimpleNamespace(frame_id=100, pose_c2w=frames[0].pose_c2w.copy()))

    selected = select_geometry_frames(tuple(reversed(frames)))

    assert len(selected) == 32
    assert selected[0].frame_id == 0
    assert selected[-1].frame_id == 39
    assert tuple(frame.frame_id for frame in selected) == tuple(
        sorted(frame.frame_id for frame in selected)
    )


def test_conflict_groups_are_disjoint_and_hypotheses_partition_every_leaf() -> None:
    xyz, faces, normals, valid, owners, segments = _surface()
    graph = build_surface_graph(xyz, faces, normals, valid)
    leaves = build_native_leaves(xyz, owners, segments, alias_table=((11, 10),), surface_graph=graph)
    pixels = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)
    entities = np.array([5] * 20 + [5] * 20 + [0] * 20, dtype=np.int64)
    frame = build_frame_leaf_evidence(8, pixels, entities, leaf_count=3)

    groups, untouched = build_conflict_groups(leaves, (frame,))
    pair = next(group for group in groups if group.owner_ids == (1, 2))
    hypotheses = generate_complete_hypotheses("scene-a", pair, leaves, (frame,))

    assert untouched == ()
    assert pair.leaf_ids == (0, 1)
    assert hypotheses[0].kind == "ORIGINAL"
    assert any(row.kind == "MERGE" for row in hypotheses)
    assert len(hypotheses) <= 8
    for hypothesis in hypotheses:
        assert hypothesis.leaf_ids == pair.leaf_ids
        assert len(hypothesis.components) == len(pair.leaf_ids)
        assert set(hypothesis.components) == set(range(hypothesis.component_count))


def test_final_map_keeps_unchanged_ids_and_assigns_deterministic_new_ids() -> None:
    xyz, faces, normals, valid, owners, segments = _surface()
    graph = build_surface_graph(xyz, faces, normals, valid)
    leaves = build_native_leaves(xyz, owners, segments, alias_table=((11, 10),), surface_graph=graph)
    group = ConflictGroup("group:1-2", (1, 2), (0, 1), 1.0)
    original = PartitionHypothesis.create("scene-a", group, leaves, "ORIGINAL", (0, 1))
    merged = PartitionHypothesis.create("scene-a", group, leaves, "MERGE", (0, 0))
    same_partition = PartitionHypothesis.create(
        "scene-a", group, leaves, "FRAME_ENTITY", (7, 7), frame_id=3
    )

    unchanged = apply_selected_partitions("scene-a", owners, leaves, {group.group_id: original})
    changed = apply_selected_partitions("scene-a", owners, leaves, {group.group_id: merged})
    repeated = apply_selected_partitions("scene-a", owners, leaves, {group.group_id: merged})

    assert unchanged.owner_ids.tolist() == owners.tolist()
    assert merged.hypothesis_id == same_partition.hypothesis_id
    assert changed.owner_ids[4] == 0
    assert np.array_equal(changed.owner_ids, repeated.owner_ids)
    assert len(set(changed.owner_ids[:4])) == 1
    new_owner = int(changed.owner_ids[0])
    assert new_owner not in {0, 1, 2}
    assert changed.ancestry[new_owner] == (1, 2)


def test_final_geometry_export_preserves_geometry_and_uses_point_count_ranks(
    tmp_path: Path,
) -> None:
    xyz, faces, normals, valid, owners, segments = _surface()
    graph = build_surface_graph(xyz, faces, normals, valid)
    leaves = build_native_leaves(xyz, owners, segments, alias_table=((11, 10),), surface_graph=graph)
    group = ConflictGroup("group:1-2", (1, 2), (0, 1), 1.0)
    merged = PartitionHypothesis.create("scene-a", group, leaves, "MERGE", (0, 0))
    final = finalize_geometry_map(
        "scene-a",
        xyz,
        faces,
        owners,
        leaves,
        {group.group_id: merged},
        tsdf_sha256="a" * 64,
        projection_identity="native-projection-v1",
    )
    manifest = write_final_geometry_map(final, tmp_path / "final")

    np.testing.assert_array_equal(final.surface_xyz, xyz)
    np.testing.assert_array_equal(final.surface_faces, faces)
    assert final.tsdf_sha256 == "a" * 64
    assert final.projection_identity == "native-projection-v1"
    new_owner = int(final.owner_ids[0])
    assert final.rank_by_owner[new_owner] == 4
    assert final.rank_by_owner[0] == 1
    assert manifest["row_count"] == 5
    with np.load(tmp_path / "final" / "arrays.npz") as archive:
        np.testing.assert_array_equal(archive["surface_xyz"], xyz)
        np.testing.assert_array_equal(archive["surface_faces"], faces)
        np.testing.assert_array_equal(archive["owner_ids"], final.owner_ids)
