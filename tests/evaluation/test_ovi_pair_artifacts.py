from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from src.evaluation.ovi_pair_artifact_loader import (
    OviPairArtifactLoadError,
    restore_bound_ovi_pair_artifact,
)
from src.evaluation.ovi_pair_artifacts import export_ovi_object_pair_artifacts
from src.evaluation.ovi_pair_views import (
    OviObjectEntityView,
    OviObjectPairView,
    OviObjectVisitView,
)
from src.evaluation.temporal_grouping_artifacts import (
    export_temporal_grouping_artifacts,
)
from src.evaluation.temporal_object_groups import build_temporal_object_groups
from src.oviv2.two_visit_contracts import PairRelation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _observed_record(path: Path, *, relative: bool) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix() if relative else str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _visit(visit_id: int) -> OviObjectVisitView:
    offset = 0.25 * visit_id
    points = np.asarray(
        [
            [-0.5 + offset, -0.5, 1.0],
            [0.5 + offset, 0.5, 1.0],
            [0.0 + offset, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    entity = OviObjectEntityView(
        visit_id=visit_id,
        entity_id="ovimap:7",
        source_instance_id=7,
        point_indices=np.asarray([0, 1], dtype=np.int64),
        palette_rgb=(10, 20, 30),
        semantic_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        semantic_label=None,
        semantic_score=1.0,
        observation_frame_ids=(0,),
        observation_boxes_xyxy=((0, 0, 1, 1),),
    )
    return OviObjectVisitView(
        visit_id=visit_id,
        scan_id=f"scan-{visit_id}",
        frame_count=1,
        points_xyz=points,
        palette_rgb_uint8=np.asarray(
            [[10, 20, 30], [10, 20, 30], [0, 0, 0]], dtype=np.uint8
        ),
        normals_xyz=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
        normal_valid=np.ones(3, dtype=bool),
        source_vertex_indices=np.arange(3, dtype=np.int64),
        source_frame_ids_by_target=np.asarray([10], dtype=np.int64),
        entity_owner_indices=np.asarray([0, 0, -1], dtype=np.int64),
        entities=(entity,),
        camera_rgb_uint8=np.asarray(
            [[100, 110, 120], [130, 140, 150], [0, 0, 0]], dtype=np.uint8
        ),
        appearance_valid=np.asarray([True, True, False]),
        appearance_frame_ids=np.asarray([0, 0, -1], dtype=np.int64),
        appearance_source_frame_ids=np.asarray([10, 10, -1], dtype=np.int64),
        appearance_rows=np.asarray([0, 1, -1], dtype=np.int64),
        appearance_columns=np.asarray([0, 1, -1], dtype=np.int64),
        appearance_camera_depth_m=np.asarray(
            [1.0, 1.0, np.nan], dtype=np.float32
        ),
        appearance_observed_depth_m=np.asarray(
            [1.01, 1.02, np.nan], dtype=np.float32
        ),
        appearance_depth_residual_m=np.asarray([0.01, 0.02, np.nan], dtype=np.float32),
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
        pair_id="scene0001_00-scene0001_01",
        visits=(_visit(0), _visit(1)),
        source_manifest_sha256="a" * 64,
        global_alignment=np.eye(4),
    )


def test_exports_dense_geometry_and_shared_fixed_camera(
    tmp_path: Path, pair: OviObjectPairView
) -> None:
    result = export_ovi_object_pair_artifacts(
        pair, tmp_path / "artifacts", preview_width=64, preview_height=48
    )

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "D2_PAIR_ARTIFACT_PASS"
    assert manifest["pair_content_sha256"] == pair.content_sha256()
    assert manifest["geometry"]["xyz_changed"] is False
    assert manifest["preview_camera"]["shared_across_visits_and_modes"] is True
    assert manifest["rgb"]["pixel_provenance_properties"] == [
        "appearance_frame_id",
        "appearance_source_frame_id",
        "appearance_row",
        "appearance_column",
        "appearance_camera_depth_m",
        "appearance_observed_depth_m",
        "appearance_depth_residual_m",
    ]
    assert manifest["visits"][0]["entities"] == [
        {
            "entity_index": 1,
            "entity_id": "ovimap:7",
            "palette_rgb": [10, 20, 30],
            "point_count": 2,
            "source_instance_id": 7,
        }
    ]
    assert set(manifest["outputs"]) == {
        "t0_instance_ply",
        "t0_instance_preview",
        "t0_rgb_ply",
        "t0_rgb_preview",
        "t1_instance_ply",
        "t1_instance_preview",
        "t1_rgb_ply",
        "t1_rgb_preview",
    }

    for visit_id, visit in enumerate(pair.visits):
        rgb = PlyData.read(result.output_dir / f"t{visit_id}/rgb.ply")["vertex"].data
        instance = PlyData.read(result.output_dir / f"t{visit_id}/instance.ply")[
            "vertex"
        ].data
        rgb_xyz = np.column_stack((rgb["x"], rgb["y"], rgb["z"]))
        instance_xyz = np.column_stack((instance["x"], instance["y"], instance["z"]))
        np.testing.assert_array_equal(rgb_xyz, visit.points_xyz)
        np.testing.assert_array_equal(instance_xyz, visit.points_xyz)
        np.testing.assert_array_equal(
            np.column_stack((rgb["red"], rgb["green"], rgb["blue"]))[:2],
            visit.camera_rgb_uint8[:2],
        )
        assert (rgb["red"][2], rgb["green"][2], rgb["blue"][2]) == (
            160,
            160,
            160,
        )
        np.testing.assert_array_equal(rgb["appearance_valid"], [1, 1, 0])
        np.testing.assert_array_equal(rgb["normal_valid"], [1, 1, 1])
        np.testing.assert_array_equal(rgb["appearance_frame_id"], [0, 0, -1])
        np.testing.assert_array_equal(rgb["appearance_source_frame_id"], [10, 10, -1])
        np.testing.assert_array_equal(rgb["appearance_row"], [0, 1, -1])
        np.testing.assert_array_equal(rgb["appearance_column"], [0, 1, -1])
        np.testing.assert_allclose(
            rgb["appearance_camera_depth_m"][:2], [1.0, 1.0]
        )
        np.testing.assert_allclose(
            rgb["appearance_observed_depth_m"][:2], [1.01, 1.02]
        )
        assert np.isnan(rgb["appearance_camera_depth_m"][2])
        assert np.isnan(rgb["appearance_observed_depth_m"][2])
        np.testing.assert_array_equal(instance["entity_index"], [1, 1, 0])
        for name in ("rgb.png", "instance.png"):
            assert (result.output_dir / f"t{visit_id}/{name}").is_file()


def test_geometric_sample_uses_adapter_support_without_removing_dense_domain(
    pair: OviObjectPairView,
) -> None:
    t0 = pair.visits[0]
    restricted_t0 = replace(
        t0,
        camera_rgb_uint8=np.asarray(
            [[100, 110, 120], [0, 0, 0], [0, 0, 0]], dtype=np.uint8
        ),
        appearance_valid=np.asarray([True, False, False]),
        appearance_frame_ids=np.asarray([0, -1, -1], dtype=np.int64),
        appearance_source_frame_ids=np.asarray([10, -1, -1], dtype=np.int64),
        appearance_rows=np.asarray([0, -1, -1], dtype=np.int64),
        appearance_columns=np.asarray([0, -1, -1], dtype=np.int64),
        appearance_camera_depth_m=np.asarray(
            [1.0, np.nan, np.nan], dtype=np.float32
        ),
        appearance_observed_depth_m=np.asarray(
            [1.01, np.nan, np.nan], dtype=np.float32
        ),
        appearance_depth_residual_m=np.asarray(
            [0.01, np.nan, np.nan], dtype=np.float32
        ),
    )
    restricted = replace(pair, visits=(restricted_t0, pair.visits[1]))

    full_maps = restricted.to_visit_maps()
    supported_maps = restricted.to_visit_maps(supported_only=True)
    sample = restricted.geometric_sample(neural_voxel_size_m=0.02)

    assert (
        sum(
            len(entity.points_xyz)
            for view in full_maps
            for entity in view.snapshot.entities
        )
        == 4
    )
    assert (
        sum(
            len(entity.points_xyz)
            for view in supported_maps
            for entity in view.snapshot.entities
        )
        == 3
    )
    assert all(view.snapshot.background_xyz is None for view in supported_maps)
    assert all(len(view.snapshot.background_xyz) == 1 for view in full_maps)
    assert sample.source_point_count == 3
    assert sum(visit.point_count for visit in restricted.visits) == 6


def test_pair_contract_rejects_nonrigid_row_alignment(
    pair: OviObjectPairView,
) -> None:
    nonrigid = np.eye(4)
    nonrigid[0, 0] = 2.0

    with pytest.raises(ValueError, match="rigid row-vector"):
        replace(pair, global_alignment=nonrigid)


def test_export_refuses_to_overwrite(tmp_path: Path, pair: OviObjectPairView) -> None:
    output = tmp_path / "artifacts"
    export_ovi_object_pair_artifacts(pair, output, preview_width=64, preview_height=48)

    with pytest.raises(FileExistsError):
        export_ovi_object_pair_artifacts(
            pair, output, preview_width=64, preview_height=48
        )


def test_restores_exact_bound_camera_rgb_after_decoder_drift(
    tmp_path: Path, pair: OviObjectPairView
) -> None:
    export = export_ovi_object_pair_artifacts(
        pair, tmp_path / "artifacts", preview_width=64, preview_height=48
    )
    manifest_bytes = export.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_D2_PAIR_VIEW_RECEIPT_V1",
        "status": "REAL_D2_PAIR_VIEW_PASS",
        "pair_id": pair.pair_id,
        "domain_id": pair.domain_id,
        "pair_content_sha256": pair.content_sha256(),
        "coordinate_frame_id": pair.coordinate_frame_id,
        "local_artifact_manifest": {
            "path": str(export.manifest),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "byte_count": len(manifest_bytes),
        },
        "local_full_outputs": {
            role: {
                "path": str(export.output_dir / record["path"]),
                "sha256": record["sha256"],
                "byte_count": record["byte_count"],
            }
            for role, record in manifest["outputs"].items()
            if role.endswith("_ply")
        },
    }
    drifted_visits = tuple(
        replace(
            visit,
            camera_rgb_uint8=np.where(
                visit.appearance_valid[:, None],
                np.minimum(visit.camera_rgb_uint8.astype(np.uint16) + 1, 255),
                0,
            ).astype(np.uint8),
        )
        for visit in pair.visits
    )
    drifted = replace(pair, visits=drifted_visits)

    restored = restore_bound_ovi_pair_artifact(drifted, receipt)

    assert restored.content_sha256() == pair.content_sha256()
    for expected, actual in zip(pair.visits, restored.visits, strict=True):
        np.testing.assert_array_equal(actual.camera_rgb_uint8, expected.camera_rgb_uint8)


def test_bound_pair_restore_rejects_non_color_artifact_drift(
    tmp_path: Path, pair: OviObjectPairView
) -> None:
    export = export_ovi_object_pair_artifacts(
        pair, tmp_path / "artifacts", preview_width=64, preview_height=48
    )
    manifest_bytes = export.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_D2_PAIR_VIEW_RECEIPT_V1",
        "status": "REAL_D2_PAIR_VIEW_PASS",
        "pair_id": pair.pair_id,
        "domain_id": pair.domain_id,
        "pair_content_sha256": pair.content_sha256(),
        "coordinate_frame_id": pair.coordinate_frame_id,
        "local_artifact_manifest": {
            "path": str(export.manifest),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "byte_count": len(manifest_bytes),
        },
        "local_full_outputs": {
            role: {
                "path": str(export.output_dir / record["path"]),
                "sha256": record["sha256"],
                "byte_count": record["byte_count"],
            }
            for role, record in manifest["outputs"].items()
            if role.endswith("_ply")
        },
    }
    changed = replace(
        pair,
        visits=(
            replace(
                pair.visits[0],
                points_xyz=pair.visits[0].points_xyz
                + np.asarray([0.001, 0.0, 0.0], dtype=np.float32),
            ),
            pair.visits[1],
        ),
    )

    with pytest.raises(OviPairArtifactLoadError, match="points_xyz"):
        restore_bound_ovi_pair_artifact(changed, receipt)


def test_exports_dense_current_grouping_without_changing_xyz(
    tmp_path: Path, pair: OviObjectPairView
) -> None:
    relation = PairRelation(
        temporal_query_id="q0",
        t0_entity_ids=("ovimap:7",),
        t1_entity_ids=("ovimap:7",),
        state="persistent_moved",
        query_confidence=0.9,
        evidence={"query_score": 0.9},
        identity_source="rescene",
    )
    grouping = build_temporal_object_groups(pair, (relation,), variant_id="U3")

    result = export_temporal_grouping_artifacts(
        pair,
        grouping,
        tmp_path / "grouping",
        preview_width=64,
        preview_height=48,
    )

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "DENSE_CURRENT_GROUPING_PASS"
    assert manifest["variant_id"] == "U3"
    assert manifest["grouping_sha256"] == grouping.content_sha256()
    assert manifest["geometry"]["xyz_changed"] is False
    assert set(manifest["outputs"]) == {
        "current_instance_ply",
        "current_instance_preview",
        "current_rgb_preview",
    }
    vertices = PlyData.read(result.output_dir / "current_instance.ply")["vertex"].data
    np.testing.assert_array_equal(
        np.column_stack((vertices["x"], vertices["y"], vertices["z"])),
        pair.visits[1].points_xyz,
    )
    np.testing.assert_array_equal(vertices["entity_index"], [1, 1, 0])


def test_checked_in_real_d2_receipt_binds_sources_and_small_previews() -> None:
    receipt_path = (
        REPOSITORY_ROOT
        / "configs/evaluation/results/ovi_rescene_object_level_transfer/d2_pair_view_v1.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "REAL_D2_PAIR_VIEW_PASS"
    assert receipt["domain_id"] == "D2_OVI_RECONSTRUCTION"
    assert receipt["protocol"]["ground_truth_used"] is False
    assert receipt["protocol"]["unsupported_dense_points_retained"] is True
    assert receipt["geometry_audit"] == {
        "total_point_count": 1609563,
        "t0_exact_source_rows": True,
        "t1_exact_source_rows_after_one_alignment": True,
        "rgb_instance_xyz_equal": True,
        "source_vertex_order_conserved": True,
        "smoothing_or_hole_filling": False,
        "ply_audit_status": "PASS",
    }
    for record in (
        receipt["source_pair_manifest"],
        receipt["parent_mapping_receipt"],
        *receipt["implementation_sources"].values(),
        *receipt["tracked_previews"].values(),
    ):
        path = REPOSITORY_ROOT / record["path"]
        assert _observed_record(path, relative=True) == record
