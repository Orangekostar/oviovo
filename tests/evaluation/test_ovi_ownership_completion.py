from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ovi_ownership_completion import (
    OwnershipCompletionError,
    build_dense_ownership_readout,
    build_ownership_readout,
    evaluate_completion_surface,
)
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.rscan_method_views import build_method_pair_view
from src.oviv2.two_visit_contracts import PairRelation


def _sample():
    def processed(offset: float) -> np.ndarray:
        return np.asarray(
            [
                [0.00 + offset, 0.0, 0.0, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 4, 1, 10],
                [0.01 + offset, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 4, 1, 10],
                [1.00 + offset, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 8, 1, 20],
            ],
            dtype=np.float32,
        )

    return build_method_pair_view(
        pair_id="pair",
        scan_ids=("a", "b"),
        processed_visits=(processed(0.0), processed(0.01)),
        source_manifest_sha256="a" * 64,
        domain_id="D0_NATIVE_PROCESSED",
    ).geometric_sample(neural_voxel_size_m=0.02)


def _relation() -> PairRelation:
    return PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("segment:000004",),
        t1_entity_ids=("segment:000004",),
        state="persistent_static",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )


def _dense_ovi_pair() -> OviObjectPairView:
    visits = []
    for visit_id in (0, 1):
        entity = OviObjectEntityView(
            visit_id=visit_id,
            entity_id="ovimap:1",
            source_instance_id=1,
            point_indices=np.asarray([0], dtype=np.int64),
            palette_rgb=(10, 20, 30),
            semantic_embedding=np.asarray([1.0], dtype=np.float32),
            semantic_label="object",
            semantic_score=0.9,
            observation_frame_ids=(0,),
            observation_boxes_xyxy=((0, 0, 0, 0),),
        )
        visits.append(
            OviObjectVisitView(
                visit_id=visit_id,
                scan_id=f"scan-{visit_id}",
                frame_count=1,
                points_xyz=np.asarray([[float(visit_id), 0.0, 1.0]], dtype=np.float32),
                palette_rgb_uint8=np.asarray([[10, 20, 30]], dtype=np.uint8),
                normals_xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                normal_valid=np.asarray([True]),
                source_vertex_indices=np.asarray([0], dtype=np.int64),
                source_frame_ids_by_target=np.asarray([10], dtype=np.int64),
                entity_owner_indices=np.asarray([0], dtype=np.int64),
                entities=(entity,),
                camera_rgb_uint8=np.asarray([[100, 110, 120]], dtype=np.uint8),
                appearance_valid=np.asarray([True]),
                appearance_frame_ids=np.asarray([0], dtype=np.int64),
                appearance_source_frame_ids=np.asarray([10], dtype=np.int64),
                appearance_rows=np.asarray([0], dtype=np.int64),
                appearance_columns=np.asarray([0], dtype=np.int64),
                appearance_camera_depth_m=np.asarray([1.0], dtype=np.float32),
                appearance_observed_depth_m=np.asarray([1.0], dtype=np.float32),
                appearance_depth_residual_m=np.asarray([0.0], dtype=np.float32),
                native_manifest_sha256=str(visit_id + 1) * 64,
                materialized_manifest_sha256=str(visit_id + 3) * 64,
                source_artifact_sha256={
                    "instance_color_log": "5" * 64,
                    "instance_mesh": "6" * 64,
                    "semantic_features": "7" * 64,
                },
                global_alignment_application=(
                    "identity_reference"
                    if visit_id == 0
                    else "rescan_to_reference_once"
                ),
            )
        )
    return OviObjectPairView(
        pair_id="pair",
        visits=(visits[0], visits[1]),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def test_identity_readout_groups_entities_without_relocating_geometry() -> None:
    sample = _sample()
    baseline = build_ownership_readout(sample, (), variant_id="A0")
    temporal = build_ownership_readout(sample, (_relation(),), variant_id="A2")

    assert baseline["geometry_sha256"] == temporal["geometry_sha256"]
    assert baseline["entity_geometry_sha256"] == temporal["entity_geometry_sha256"]
    assert baseline["identity_group_count"] == 4
    assert temporal["cross_visit_identity_group_count"] == 1
    assert temporal["identity_group_count"] == 3
    assert temporal["geometry_mutation_count"] == 0


def test_identity_readout_rejects_unknown_or_multiply_owned_entities() -> None:
    sample = _sample()
    unknown = PairRelation(
        temporal_query_id="bad",
        t0_entity_ids=("missing",),
        t1_entity_ids=("segment:000004",),
        state="persistent_static",
        query_confidence=0.8,
        evidence={"query_score": 0.8},
        identity_source="rescene",
    )
    with pytest.raises(OwnershipCompletionError, match="unknown"):
        build_ownership_readout(sample, (unknown,), variant_id="A2")
    with pytest.raises(OwnershipCompletionError, match="multiple"):
        build_ownership_readout(sample, (_relation(), _relation()), variant_id="A2")


def test_dense_ownership_readout_uses_current_ovi_surface_without_xyz_mutation() -> None:
    pair = _dense_ovi_pair()
    relation = PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("ovimap:1",),
        t1_entity_ids=("ovimap:1",),
        state="persistent_moved",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )

    result = build_dense_ownership_readout(pair, (relation,), variant_id="A_ID")

    assert result.variant_id == "A_ID"
    assert result.source_xyz_multiset_sha256 == result.output_xyz_multiset_sha256
    assert result.snapshots[1].entities[0].metadata["geometry_authority"] == "ovi_t1"


def test_completion_surface_uses_only_actual_historical_and_target_shapes() -> None:
    baseline = np.asarray([[0.01, 0.0, 0.0]], dtype=np.float32)
    historical = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0], [0.21, 0.0, 0.0]],
        dtype=np.float32,
    )
    target = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32
    )
    recovered = np.asarray(
        [[0.01, 0.0, 0.0], [0.11, 0.0, 0.0], [0.31, 0.0, 0.0]],
        dtype=np.float32,
    )

    result = evaluate_completion_surface(
        baseline_xyz=baseline,
        recovered_xyz=recovered,
        target_xyz=target,
        historical_candidate_xyz=historical,
        voxel_size_m=0.05,
    )

    assert result["opportunity_voxel_count"] == 1
    assert result["recoverable_voxel_count"] == 1
    assert result["recovered_new_voxel_count"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f_score"] == pytest.approx(2 / 3)
