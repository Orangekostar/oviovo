from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    IdentityRules,
    PredictedInstance,
    _threshold_result,
    apply_row_vector_transform,
    evaluate_instance_geometry,
    load_annotated_instances,
    load_pair_ground_truth,
    publish_pair_ground_truth,
    voxelize_points,
)


def _matrix(*, translation: tuple[float, float, float]) -> tuple[float, ...]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, :3] = translation
    return tuple(matrix.reshape(-1))


def _voxels(*values: tuple[int, int, int]) -> frozenset[tuple[int, int, int]]:
    return frozenset(values)


def _record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _write_annotated_pair(tmp_path: Path) -> tuple[dict[str, object], Path]:
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")
    sessions: list[dict[str, object]] = []
    for visit, offset in ((0, 0.0), (1, -0.05)):
        root = tmp_path / f"visit-{visit}"
        vertices = np.array(
            [
                (0.01 + offset, 0.01, 0.01, 0),
                (0.01 + offset, 0.01, 0.01, 1),
                (0.06 + offset, 0.01, 0.01, 1),
                (1.01 + offset, 0.01, 0.01, 2),
            ],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("objectId", "u2"),
            ],
        )
        ply_path = root / "labels.instances.annotated.v2.ply"
        root.mkdir()
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(ply_path)
        semantic_path = root / "semseg.v2.json"
        semantic_path.write_text(
            json.dumps(
                {
                    "segGroups": [
                        {"objectId": 1, "label": "chair"},
                        {"objectId": 2, "label": "table"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        sessions.append(
            {
                "visit_index": visit,
                "scan_id": f"scan-{visit}",
                "raw_assets": {
                    "labels.instances.annotated.v2.ply": _record(ply_path),
                    "semseg.v2.json": _record(semantic_path),
                },
            }
        )
    pair = {
        "pair_id": "scene0001_00-scene0001_01",
        "sessions": sessions,
        "common_method_inputs": {
            "global_alignment": {
                "direction": "rescan_row_vector_to_reference",
                "storage": "row_major_flat_4x4",
                "application": "homogeneous_row_vector_right_multiply",
                "matrix": list(_matrix(translation=(0.05, 0.0, 0.0))),
            }
        },
        "evaluator_only": {
            "changes": {
                "rigid": [
                    {
                        "instance_reference": 1,
                        "instance_rescan": 1,
                        "symmetry": 2,
                        "transform": list(_matrix(translation=(0.0, 0.0, 0.0))),
                    }
                ],
                "nonrigid": [],
                "removed": [2],
            },
            "ambiguity": [],
        },
    }
    return pair, selection


def test_row_vector_direction_aligns_rescan_to_reference() -> None:
    reference = np.asarray([[1.0, 2.0, 3.0], [1.04, 2.0, 3.0]])
    rescan = reference - np.asarray([0.5, -0.25, 1.0])

    aligned = apply_row_vector_transform(
        rescan, _matrix(translation=(0.5, -0.25, 1.0))
    )

    np.testing.assert_allclose(aligned, reference)
    assert voxelize_points(aligned, voxel_size_m=0.05) == voxelize_points(
        reference, voxel_size_m=0.05
    )


def test_voxelization_uses_floor_for_negative_coordinates_and_deduplicates() -> None:
    points = np.asarray(
        [
            [-0.001, 0.049, 0.050],
            [-0.049, 0.001, 0.099],
            [0.051, 0.000, 0.100],
        ]
    )

    voxels = voxelize_points(points, voxel_size_m=0.05)

    assert voxels == _voxels((-1, 0, 1), (1, 0, 2))


def test_hungarian_matching_reports_primary_and_sensitivity_thresholds() -> None:
    ground_truth = (
        GroundTruthInstance(1, "chair", _voxels((0, 0, 0), (1, 0, 0))),
        GroundTruthInstance(2, "table", _voxels((10, 0, 0), (11, 0, 0))),
    )
    predictions = (
        PredictedInstance("p-chair", _voxels((0, 0, 0), (1, 0, 0))),
        PredictedInstance("p-table", _voxels((10, 0, 0))),
    )

    result = evaluate_instance_geometry(predictions, ground_truth)

    assert result.primary.threshold == 0.50
    assert result.primary.matched_count == 2
    assert result.primary.mean_matched_iou == pytest.approx(0.75)
    assert result.sensitivity.threshold == 0.25
    assert result.sensitivity.matched_count == 2
    assert [(match.prediction_id, match.gt_instance_id) for match in result.primary.matches] == [
        ("p-chair", 1),
        ("p-table", 2),
    ]


def test_cardinality_first_matching_maximizes_threshold_valid_edges() -> None:
    predictions = (
        PredictedInstance("p0", _voxels((0, 0, 0))),
        PredictedInstance("p1", _voxels((1, 0, 0))),
    )
    ground_truth = (
        GroundTruthInstance(1, "chair", _voxels((0, 0, 0))),
        GroundTruthInstance(2, "table", _voxels((1, 0, 0))),
    )
    ious = np.asarray([[1.0, 0.50], [0.50, 0.49]], dtype=np.float64)

    legacy = _threshold_result(predictions, ground_truth, ious, threshold=0.50)
    cardinality_first = _threshold_result(
        predictions,
        ground_truth,
        ious,
        threshold=0.50,
        matching_policy="max_valid_count_then_iou",
    )

    assert legacy.matched_count == 1
    assert [
        (match.prediction_id, match.gt_instance_id)
        for match in cardinality_first.matches
    ] == [("p0", 2), ("p1", 1)]


def test_matching_reports_unmatched_duplicate_fragment_and_merge_errors() -> None:
    ground_truth = (
        GroundTruthInstance(1, "chair", _voxels((0, 0, 0), (1, 0, 0))),
        GroundTruthInstance(2, "table", _voxels((2, 0, 0), (3, 0, 0))),
        GroundTruthInstance(3, "lamp", _voxels((20, 0, 0))),
    )
    predictions = (
        PredictedInstance("fragment-a", _voxels((0, 0, 0))),
        PredictedInstance("fragment-b", _voxels((1, 0, 0))),
        PredictedInstance("merge", _voxels((2, 0, 0), (3, 0, 0), (20, 0, 0))),
        PredictedInstance("false-positive", _voxels((99, 0, 0))),
    )

    result = evaluate_instance_geometry(predictions, ground_truth)

    assert result.primary.fragment_gt_count == 1
    assert result.primary.duplicate_prediction_count == 1
    assert result.sensitivity.merge_prediction_count == 1
    assert "false-positive" in result.primary.unmatched_prediction_ids
    assert result.primary.unmatched_gt_instance_ids


def test_identity_rules_preserve_cross_time_mapping_symmetry_and_ambiguity() -> None:
    rules = IdentityRules.from_official_records(
        changes={
            "rigid": [
                {
                    "instance_reference": 3,
                    "instance_rescan": 5,
                    "symmetry": 2,
                    "transform": list(_matrix(translation=(1.0, 0.0, 0.0))),
                }
            ],
            "nonrigid": [6],
            "removed": [9],
        },
        ambiguity=[
            [
                {
                    "instance_source": 7,
                    "instance_target": 8,
                    "transform": list(_matrix(translation=(0.0, 0.0, 0.0))),
                }
            ]
        ],
    )

    assert rules.reference_id_for_rescan(5) == 3
    assert rules.reference_id_for_rescan(6) == 6
    assert rules.symmetry_by_reference[3] == 2
    assert rules.is_ambiguous_equivalent(7, 8)
    assert not rules.is_ambiguous_equivalent(3, 8)
    assert rules.removed_reference_ids == frozenset({9})


def test_annotated_ply_loader_ignores_background_and_binds_semantic_labels(
    tmp_path: Path,
) -> None:
    vertices = np.array(
        [
            (0.01, 0.01, 0.01, 0),
            (0.01, 0.01, 0.01, 1),
            (0.06, 0.01, 0.01, 1),
            (1.01, 0.01, 0.01, 2),
        ],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("objectId", "u2")],
    )
    ply_path = tmp_path / "labels.instances.annotated.v2.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(ply_path)
    semantic_path = tmp_path / "semseg.v2.json"
    semantic_path.write_text(
        json.dumps(
            {
                "scan_id": "fixture",
                "segGroups": [
                    {"objectId": 1, "label": "chair"},
                    {"objectId": 2, "label": "table"},
                ],
            }
        ),
        encoding="utf-8",
    )

    instances = load_annotated_instances(
        ply_path,
        semantic_path,
        voxel_size_m=0.05,
        transform=_matrix(translation=(0.05, 0.0, 0.0)),
    )

    assert [(value.instance_id, value.semantic_label) for value in instances] == [
        (1, "chair"),
        (2, "table"),
    ]
    assert instances[0].voxels == _voxels((1, 0, 0), (2, 0, 0))


def test_loader_rejects_positive_instance_without_semantic_record(tmp_path: Path) -> None:
    vertices = np.array(
        [(0.0, 0.0, 0.0, 4)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("objectId", "u2")],
    )
    ply_path = tmp_path / "labels.instances.annotated.v2.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(ply_path)
    semantic_path = tmp_path / "semseg.v2.json"
    semantic_path.write_text(json.dumps({"segGroups": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic record"):
        load_annotated_instances(ply_path, semantic_path)


def test_pair_sidecar_round_trip_preserves_aligned_instances_and_rules(
    tmp_path: Path,
) -> None:
    pair, selection = _write_annotated_pair(tmp_path)
    output = tmp_path / "pair-gt"

    manifest = publish_pair_ground_truth(
        pair,
        selection_manifest_record=_record(selection),
        output_root=output,
    )
    loaded = load_pair_ground_truth(output / "manifest.json")

    assert manifest["status"] == "PASS"
    assert manifest["instance_counts"] == {"0": 2, "1": 2}
    assert loaded.pair_id == pair["pair_id"]
    assert loaded.visits[0] == loaded.visits[1]
    assert loaded.identity_rules.symmetry_by_reference[1] == 2
    assert loaded.identity_rules.removed_reference_ids == frozenset({2})


def test_pair_sidecar_loader_rejects_array_tamper(tmp_path: Path) -> None:
    pair, selection = _write_annotated_pair(tmp_path)
    output = tmp_path / "pair-gt"
    publish_pair_ground_truth(
        pair,
        selection_manifest_record=_record(selection),
        output_root=output,
    )
    with (output / "instances.npz").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="binding mismatch"):
        load_pair_ground_truth(output / "manifest.json")


def test_gt_sidecar_cli_forwards_frozen_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluation.build_3rscan_gt_sidecars as cli

    selection = tmp_path / "selection.json"
    output = tmp_path / "gt"
    observed: dict[str, Path] = {}

    def fake_build(selection_manifest_path: Path, output_root: Path) -> dict[str, object]:
        observed["selection"] = selection_manifest_path
        observed["output"] = output_root
        return {"status": "PASS", "pair_count": 3}

    monkeypatch.setattr(cli, "build_ground_truth_sidecars", fake_build)

    assert cli.main(["--selection-manifest", str(selection), "--output", str(output)]) == 0
    assert observed == {"selection": selection, "output": output}
