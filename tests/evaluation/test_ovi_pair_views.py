from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.evaluation.ovi_pair_views as pair_views_module
from src.evaluation.ovi_pair_views import (
    OviPairViewError,
    build_ovi_object_pair_view,
)

PAIR_ID = "scene0001_00-scene0001_01"
SCAN_IDS = ("scan-reference", "scan-rescan")
SOURCE_MANIFEST_SHA256 = "a" * 64


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.absolute())
        ),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_visit(tmp_path: Path, *, visit_id: int) -> tuple[Path, Path]:
    scan_id = SCAN_IDS[visit_id]
    materialized = tmp_path / f"materialized-{visit_id}"
    for name in ("color", "depth", "pose", "intrinsic"):
        (materialized / name).mkdir(parents=True, exist_ok=True)
    color = np.asarray(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
        ],
        dtype=np.uint8,
    )
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    assert cv2.imwrite(str(materialized / "color/0.jpg"), color)
    assert cv2.imwrite(str(materialized / "depth/0.png"), depth)
    np.savetxt(materialized / "pose/0.txt", np.eye(4))
    intrinsic = np.asarray(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.savetxt(materialized / "intrinsic/intrinsic_color.txt", intrinsic)
    np.savetxt(materialized / "intrinsic/intrinsic_depth.txt", intrinsic)
    source_zip = tmp_path / f"sequence-{visit_id}.zip"
    source_zip.write_bytes(f"sequence-{visit_id}".encode("ascii"))
    output_paths = sorted(path for path in materialized.rglob("*") if path.is_file())
    output_records = [_record(path, relative_to=materialized) for path in output_paths]
    tree_hash = hashlib.sha256()
    for record in output_records:
        tree_hash.update(str(record["path"]).encode("utf-8"))
        tree_hash.update(b"\0")
        tree_hash.update(str(record["sha256"]).encode("ascii"))
        tree_hash.update(b"\n")
    materialized_manifest = materialized / "materialized_manifest.json"
    _write_json(
        materialized_manifest,
        {
            "schema_version": 1,
            "artifact_id": "RSCAN_OVI_VISIT_V1",
            "status": "MATERIALIZED_INPUT_PASS",
            "dataset": "3RScan",
            "dataset_adapter": "scannet_nyu",
            "scan_id": scan_id,
            "pair_id": PAIR_ID,
            "visit_index": visit_id,
            "frame_step": 1,
            "source_frame_count": 1,
            "frame_count": 1,
            "frame_map": [{"source_frame_id": 0, "target_frame_id": 0}],
            "depth_shift": 1000.0,
            "dimensions": {
                "color": {"width": 3, "height": 2},
                "depth": {"width": 3, "height": 2},
            },
            "calibration": {
                "color_intrinsic": intrinsic.tolist(),
                "color_extrinsic": np.eye(4).tolist(),
                "depth_intrinsic": intrinsic.tolist(),
                "depth_extrinsic": np.eye(4).tolist(),
            },
            "sequence_zip": _record(source_zip),
            "preflight": {
                "camera_world_round_trip_error": 0.0,
                "depth_encoding": "uint16",
                "depth_units_per_meter": 1000.0,
                "rgb_depth_registration": "K_depth @ inv(K_color)",
            },
            "output_tree": {
                "sha256": tree_hash.hexdigest(),
                "byte_count": sum(
                    int(record["byte_count"]) for record in output_records
                ),
                "file_count": len(output_records),
                "files": output_records,
            },
        },
    )

    native = tmp_path / f"native-{visit_id}"
    (native / "mapping/cropformer_inst").mkdir(parents=True)
    (native / "native_audit").mkdir()
    mesh = native / "mapping/cropformer_inst/instance_mesh_1.ply"
    mesh.write_text(
        """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float normal_x
property float normal_y
property float normal_z
property uchar red
property uchar green
property uchar blue
end_header
-0.5 -0.25 1 0 0 1 10 20 30
0 -0.25 1 0 0 1 10 20 30
0.5 0.25 1 0 0 1 40 50 60
0 0.25 1 0 0 1 0 0 0
""",
        encoding="ascii",
    )
    semantics = native / "mapping/cropformer_inst/features.pkl"
    with semantics.open("wb") as handle:
        pickle.dump(
            {
                7: {
                    "feat": np.asarray([[1.0, 0.0]], dtype=np.float32),
                    "vis_area": np.asarray([1.0], dtype=np.float32),
                    "frame_id": [0],
                    "box_2d": [(0, 0, 1, 0)],
                    "pose": [np.eye(4)],
                },
                8: {
                    "feat": np.asarray([[0.0, 1.0]], dtype=np.float32),
                    "vis_area": np.asarray([1.0], dtype=np.float32),
                    "frame_id": [0],
                    "box_2d": [(2, 1, 2, 1)],
                    "pose": [np.eye(4)],
                },
            },
            handle,
        )
    color_log = native / "native_audit/instance_colors_cpp.tsv"
    color_log.write_text(
        "Instance: 7 Color: (10,20,30)\nInstance: 8 Color: (40,50,60)\n",
        encoding="utf-8",
    )
    native_manifest = native / "native_mapping_manifest.json"
    materialized_data = materialized_manifest.read_bytes()
    _write_json(
        native_manifest,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "MAPPING_PASS",
            "stage": "mapping",
            "scene": scan_id,
            "dataset": "scannet_nyu",
            "frame_ids": [0],
            "preflight": {
                "status": "PASS",
                "scene": scan_id,
                "dataset": "scannet_nyu",
                "frame_ids": [0],
                "depth_shift": 1000.0,
                "materialized_manifest": {
                    "path": str(materialized_manifest.absolute()),
                    "sha256": hashlib.sha256(materialized_data).hexdigest(),
                    "size_bytes": len(materialized_data),
                },
            },
            "artifacts": {
                "instance_mesh": _record(mesh),
                "semantic_features": _record(semantics),
                "instance_color_log": _record(color_log),
            },
        },
    )
    return native_manifest, materialized_manifest


@pytest.fixture
def pair_inputs(tmp_path: Path) -> dict[str, object]:
    manifests = tuple(_write_visit(tmp_path, visit_id=value) for value in (0, 1))
    row_transform = np.eye(4)
    row_transform[3, 0] = 10.0
    return {
        "pair_record": {
            "pair_id": PAIR_ID,
            "environment_id": SCAN_IDS[0],
            "sessions": [
                {"visit_index": index, "scan_id": scan_id}
                for index, scan_id in enumerate(SCAN_IDS)
            ],
            "common_method_inputs": {
                "global_alignment": {
                    "application": "homogeneous_row_vector_right_multiply",
                    "direction": "rescan_row_vector_to_reference",
                    "matrix": row_transform.reshape(-1).tolist(),
                    "storage": "row_major_flat_4x4",
                }
            },
            "evaluator_only": {"changes": {"rigid": ["ignored"]}},
        },
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "native_manifests": (manifests[0][0], manifests[1][0]),
        "materialized_manifests": (manifests[0][1], manifests[1][1]),
        "minimum_valid_neighbours": 1,
    }


def test_builds_dense_pair_with_unique_ownership_and_source_conservation(
    pair_inputs: dict[str, object],
) -> None:
    pair = build_ovi_object_pair_view(**pair_inputs)

    assert pair.domain_id == "D2_OVI_RECONSTRUCTION"
    assert pair.candidate_ids == (
        ((0, "ovimap:7"), (0, "ovimap:8")),
        ((1, "ovimap:7"), (1, "ovimap:8")),
    )
    for visit in pair.visits:
        assert visit.point_count == 4
        assert visit.entity_point_count == 3
        assert visit.background_point_count == 1
        assert visit.appearance_support_count == 3
        assert visit.appearance_support_fraction == pytest.approx(0.75)
        owned = np.concatenate([entity.point_indices for entity in visit.entities])
        background = np.flatnonzero(visit.entity_owner_indices < 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((owned, background))),
            np.arange(visit.point_count),
        )
        assert len(np.unique(owned)) == len(owned)
        assert np.all(np.isfinite(visit.points_xyz))
        assert np.all(np.isfinite(visit.normals_xyz))
        assert np.all(np.isfinite(visit.camera_rgb_uint8))
        assert visit.source_vertex_indices.tolist() == [0, 1, 2, 3]
        assert visit.source_frame_ids_by_target.tolist() == [0]
        assert visit.appearance_source_frame_ids.tolist() == [0, 0, 0, -1]
        np.testing.assert_allclose(
            visit.appearance_camera_depth_m[:3], [1.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            visit.appearance_observed_depth_m[:3], [1.0, 1.0, 1.0]
        )
        assert np.isnan(visit.appearance_camera_depth_m[3])
        assert np.isnan(visit.appearance_observed_depth_m[3])
        assert set(visit.source_artifact_sha256) == {
            "instance_mesh",
            "semantic_features",
            "instance_color_log",
        }


def test_applies_global_alignment_once_and_builds_independent_maps(
    pair_inputs: dict[str, object],
) -> None:
    pair = build_ovi_object_pair_view(**pair_inputs)

    np.testing.assert_allclose(pair.visits[0].points_xyz[:, 0], [-0.5, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(pair.visits[1].points_xyz[:, 0], [9.5, 10.0, 10.5, 10.0])
    assert pair.visits[0].global_alignment_application == "identity_reference"
    assert pair.visits[1].global_alignment_application == "rescan_to_reference_once"

    t0, t1 = pair.to_visit_maps()
    assert t0.snapshot is not t1.snapshot
    assert t0.snapshot.scene_id == t1.snapshot.scene_id == PAIR_ID
    assert t0.observed_frame_end < t1.observed_frame_start
    t0.assert_unchanged()
    t1.assert_unchanged()
    sample = pair.geometric_sample(neural_voxel_size_m=0.02)
    assert sample.source_point_count == 6
    assert sample.source_manifest_sha256 == SOURCE_MANIFEST_SHA256
    assert sample.coordinate_frame_id == "3rscan_reference"


def test_pair_hash_ignores_evaluator_only_ground_truth(
    pair_inputs: dict[str, object],
) -> None:
    first = build_ovi_object_pair_view(**pair_inputs)
    changed = dict(pair_inputs)
    pair_record = dict(pair_inputs["pair_record"])
    pair_record["evaluator_only"] = {"changes": {"rigid": ["different", "GT"]}}
    changed["pair_record"] = pair_record

    second = build_ovi_object_pair_view(**changed)

    assert second.content_sha256() == first.content_sha256()


def test_decodes_each_same_visit_frame_only_once(
    pair_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pair_views_module._decode_frame
    calls: list[tuple[Path, int]] = []

    def counted(materialized, frame_id):
        calls.append((materialized.path, frame_id))
        return original(materialized, frame_id)

    monkeypatch.setattr(pair_views_module, "_decode_frame", counted)

    build_ovi_object_pair_view(**pair_inputs)

    assert calls == [
        (pair_inputs["materialized_manifests"][0], 0),
        (pair_inputs["materialized_manifests"][1], 0),
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        {"processed_visits": (np.zeros((1, 10)), np.zeros((1, 10)))},
        {"support_masks": (np.ones(1, dtype=bool), np.ones(1, dtype=bool))},
    ],
)
def test_d2_builder_rejects_processed_arrays_and_support_masks(
    pair_inputs: dict[str, object], forbidden: dict[str, object]
) -> None:
    with pytest.raises(OviPairViewError, match="native OVI artifacts"):
        build_ovi_object_pair_view(**pair_inputs, **forbidden)


def test_pair_arrays_are_immutable(pair_inputs: dict[str, object]) -> None:
    pair = build_ovi_object_pair_view(**pair_inputs)

    with pytest.raises(ValueError, match="read-only"):
        pair.visits[0].points_xyz[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        pair.visits[0].entities[0].point_indices[0] = 99


def test_rejects_tampered_native_artifact(pair_inputs: dict[str, object]) -> None:
    native_path = pair_inputs["native_manifests"][1]
    native = json.loads(native_path.read_text(encoding="utf-8"))
    mesh = Path(native["artifacts"]["instance_mesh"]["path"])
    mesh.write_bytes(mesh.read_bytes() + b"tamper")

    with pytest.raises(OviPairViewError, match="artifact binding"):
        build_ovi_object_pair_view(**pair_inputs)
