from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.evaluation.evaluate_ovi_ownership_completion import (
    build_instance_readout_rows,
)
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    IdentityRules,
    voxelize_points,
)
from src.evaluation.temporal_object_groups import (
    GroupingConfig,
    build_temporal_object_groups,
    evaluate_current_instance_grouping,
)
from src.oviv2.two_visit_contracts import PairRelation

X = "ovimap:1"
Y = "ovimap:2"
A = "ovimap:1"
B = "ovimap:2"
C = "ovimap:3"


def _visit(visit_id: int) -> OviObjectVisitView:
    names = (X, Y) if visit_id == 0 else (A, B, C)
    points = np.asarray(
        [[0.0, 0.0, float(visit_id)], [0.1, 0.0, float(visit_id)], [1.0, 0.0, float(visit_id)]][
            : len(names)
        ],
        dtype=np.float32,
    )
    entities = tuple(
        OviObjectEntityView(
            visit_id=visit_id,
            entity_id=name,
            source_instance_id=index + 1,
            point_indices=np.asarray([index], dtype=np.int64),
            palette_rgb=(10 + index, 20 + index, 30 + index),
            semantic_embedding=np.asarray([1.0, float(index)], dtype=np.float32),
            semantic_label="object",
            semantic_score=0.9,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((index, 0, index, 0),),
        )
        for index, name in enumerate(names)
    )
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray(
            [[10 + index, 20 + index, 30 + index] for index in range(len(names))],
            dtype=np.uint8,
        ),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0]] * len(names), dtype=np.float32),
        normal_valid=np.ones(len(names), dtype=bool),
        source_vertex_indices=np.arange(len(names), dtype=np.int64),
        source_frame_ids_by_target=np.asarray([10], dtype=np.int64),
        entity_owner_indices=np.arange(len(names), dtype=np.int64),
        entities=entities,
        camera_rgb_uint8=np.asarray(
            [[100 + index, 110 + index, 120 + index] for index in range(len(names))],
            dtype=np.uint8,
        ),
        appearance_valid=np.ones(len(names), dtype=bool),
        appearance_frame_ids=np.zeros(len(names), dtype=np.int64),
        appearance_source_frame_ids=np.full(len(names), 10, dtype=np.int64),
        appearance_rows=np.arange(len(names), dtype=np.int64),
        appearance_columns=np.arange(len(names), dtype=np.int64),
        appearance_camera_depth_m=np.ones(len(names), dtype=np.float32),
        appearance_observed_depth_m=np.full(len(names), 1.01, dtype=np.float32),
        appearance_depth_residual_m=np.full(len(names), 0.01, dtype=np.float32),
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


@pytest.fixture
def pair() -> OviObjectPairView:
    return OviObjectPairView(
        pair_id="pair",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def _relation(
    query_id: str,
    t0: tuple[str, ...],
    t1: tuple[str, ...],
    confidence: float,
    support: float,
) -> PairRelation:
    return PairRelation(
        temporal_query_id=query_id,
        t0_entity_ids=t0,
        t1_entity_ids=t1,
        state="split" if len(t0) == 1 and len(t1) > 1 else "uncertain",
        query_confidence=confidence,
        evidence={"query_score": confidence, "soft_mass": support},
        identity_source="rescene",
    )


def _ground_truth(pair: OviObjectPairView) -> GroundTruthPair:
    return GroundTruthPair(
        pair_id=pair.pair_id,
        voxel_size_m=0.05,
        visits=(
            (
                GroundTruthInstance(
                    1,
                    "object",
                    voxelize_points(pair.visits[0].points_xyz, voxel_size_m=0.05),
                ),
            ),
            (
                GroundTruthInstance(
                    1,
                    "object",
                    voxelize_points(
                        pair.visits[1].points_xyz[:2], voxel_size_m=0.05
                    ),
                ),
                GroundTruthInstance(
                    2,
                    "object",
                    voxelize_points(
                        pair.visits[1].points_xyz[2:], voxel_size_m=0.05
                    ),
                ),
            ),
        ),
        identity_rules=IdentityRules.from_official_records(
            changes={"rigid": [], "nonrigid": [], "removed": []}, ambiguity=[]
        ),
    )


def test_u3_resolves_conflicts_without_transitive_closure(
    pair: OviObjectPairView,
) -> None:
    stronger = _relation("q1", (X,), (A, B), 0.9, 5.0)
    weaker = _relation("q2", (Y,), (B, C), 0.8, 9.0)

    result = build_temporal_object_groups(
        pair,
        (weaker, stronger),
        variant_id="U3",
        config=GroupingConfig(minimum_query_confidence=0.3),
    )
    reversed_result = build_temporal_object_groups(
        pair,
        (stronger, weaker),
        variant_id="U3",
        config=GroupingConfig(minimum_query_confidence=0.3),
    )

    members = {value.member_entity_ids for value in result.objects[1]}
    assert members == {(A, B), (C,)}
    assert (A, B, C) not in members
    conflict = next(value for value in result.rejected_conflicts if value.entity_id == B)
    assert conflict.query_id == "q2"
    assert conflict.winner_query_id == "q1"
    assert result.content_sha256() == reversed_result.content_sha256()


def test_u1_and_u2_abstain_on_overlap_and_u2_filters_low_confidence(
    pair: OviObjectPairView,
) -> None:
    q1 = _relation("q1", (X,), (A, B), 0.9, 5.0)
    q2 = _relation("q2", (Y,), (B, C), 0.8, 4.0)
    low = _relation("q-low", (X,), (A, B), 0.2, 8.0)

    conflicted = build_temporal_object_groups(pair, (q1, q2), variant_id="U2")
    union_all = build_temporal_object_groups(pair, (low,), variant_id="U1")
    filtered = build_temporal_object_groups(pair, (low,), variant_id="U2")

    assert {value.member_entity_ids for value in conflicted.objects[1]} == {
        (A,),
        (B,),
        (C,),
    }
    assert {value.member_entity_ids for value in union_all.objects[1]} == {
        (A, B),
        (C,),
    }
    assert {value.member_entity_ids for value in filtered.objects[1]} == {
        (A,),
        (B,),
        (C,),
    }


def test_dense_readout_has_exclusive_ownership_and_identical_xyz_multiset(
    pair: OviObjectPairView,
) -> None:
    relation = _relation("q1", (X,), (A, B), 0.9, 5.0)

    baseline = build_temporal_object_groups(pair, (), variant_id="U0")
    grouped = build_temporal_object_groups(pair, (relation,), variant_id="U3")

    assert grouped.source_xyz_multiset_sha256 == grouped.output_xyz_multiset_sha256
    assert baseline.source_xyz_multiset_sha256 == grouped.source_xyz_multiset_sha256
    for visit_id, visit in enumerate(pair.visits):
        owners = grouped.owner_object_indices[visit_id]
        assert len(owners) == visit.point_count
        assert np.all(owners[visit.entity_owner_indices >= 0] >= 0)
        assert len(np.concatenate([value.source_point_indices for value in grouped.objects[visit_id]])) == len(
            np.unique(
                np.concatenate(
                    [value.source_point_indices for value in grouped.objects[visit_id]]
                )
            )
        )
    assert {entity.entity_id for entity in grouped.snapshots[1].entities} == {
        "U3:query:q1:t1",
        f"U3:atomic:t1:{C}",
    }


def test_grouping_improves_class_agnostic_instance_readout(
    pair: OviObjectPairView,
) -> None:
    ground_truth = _ground_truth(pair)
    relation = _relation("q1", (X,), (A, B), 0.9, 5.0)
    baseline = build_temporal_object_groups(pair, (), variant_id="U0")
    grouped = build_temporal_object_groups(pair, (relation,), variant_id="U3")

    baseline_metrics = evaluate_current_instance_grouping(baseline, ground_truth)
    grouped_metrics = evaluate_current_instance_grouping(grouped, ground_truth)

    assert baseline_metrics["0.50"]["fragment_gt_count"] == 1
    assert baseline_metrics["0.50"]["precision"] == pytest.approx(2 / 3)
    assert grouped_metrics["0.50"]["fragment_gt_count"] == 0
    assert grouped_metrics["0.50"]["precision"] == pytest.approx(1.0)
    assert grouped_metrics["0.50"]["recall"] == pytest.approx(1.0)
    assert grouped_metrics["0.50"]["f_score"] == pytest.approx(1.0)


def test_a_id_propagates_identity_without_changing_single_visit_masks(
    pair: OviObjectPairView,
) -> None:
    relation = PairRelation(
        temporal_query_id="identity",
        t0_entity_ids=(X,),
        t1_entity_ids=(A,),
        state="persistent_moved",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )

    identity = build_temporal_object_groups(pair, (relation,), variant_id="A_ID")
    baseline = build_temporal_object_groups(pair, (), variant_id="U0")

    assert [value.member_entity_ids for value in identity.objects[1]] == [
        value.member_entity_ids for value in baseline.objects[1]
    ]
    assert next(value for value in identity.objects[0] if value.member_entity_ids == (X,)).temporal_identity_id == next(
        value for value in identity.objects[1] if value.member_entity_ids == (A,)
    ).temporal_identity_id
    assert identity.output_xyz_multiset_sha256 == baseline.output_xyz_multiset_sha256


def test_grouping_rejects_unknown_entity(pair: OviObjectPairView) -> None:
    relation = replace(
        _relation("q1", (X,), (A, B), 0.9, 5.0),
        t1_entity_ids=(A, "ovimap:999"),
    )

    with pytest.raises(ValueError, match="unknown entity"):
        build_temporal_object_groups(pair, (relation,), variant_id="U3")


def test_builds_two_measured_instance_rows_for_every_declared_variant(
    pair: OviObjectPairView,
) -> None:
    relation = _relation("q1", (X,), (A, B), 0.9, 5.0)
    groupings = {
        variant: build_temporal_object_groups(
            pair,
            () if variant == "U0" else (relation,),
            variant_id=variant,
        )
        for variant in ("A_ID", "U0", "U1", "U2", "U3")
    }

    rows = build_instance_readout_rows(
        pair,
        groupings,
        _ground_truth(pair),
        evaluated_commit="f" * 40,
        config_sha256="d" * 64,
        checkpoint_sha256="e" * 64,
    )

    assert [(row["variant_id"], row["iou_threshold"]) for row in rows] == [
        (variant, threshold)
        for variant in ("A_ID", "U0", "U1", "U2", "U3")
        for threshold in (0.5, 0.25)
    ]
    assert all(row["geometry_unchanged"] for row in rows)
    assert all(row["current_owned_point_count"] == 3 for row in rows)
    assert all(row["appearance_support_rate"] == pytest.approx(1.0) for row in rows)
    assert next(row for row in rows if row["variant_id"] == "U3")[
        "unconfirmed_atomic_object_count"
    ] == 1
