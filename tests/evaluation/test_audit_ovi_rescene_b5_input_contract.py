from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.evaluation.audit_ovi_rescene_b5_input_contract import (
    InputContractError,
    _load_bound_module,
    audit_input_contract,
    summarize_native_preprocessed_sample,
)


def _native_witness() -> dict[str, object]:
    raw_coordinates = np.asarray(
        [
            [-0.01, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0, 1.0],
            [0.01, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dataset_features = np.zeros((4, 7), dtype=np.float32)
    dataset_features[:, :3] = np.asarray(
        [[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]] * 2,
        dtype=np.float32,
    )
    colors = np.asarray(
        [[0.0, 0.5, 1.0], [0.25, 0.5, 0.75]] * 2,
        dtype=np.float32,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 4, dtype=np.float32)
    model_coordinates = np.concatenate(
        (
            np.zeros((4, 1), dtype=np.float32),
            raw_coordinates[:, :3],
            raw_coordinates[:, 3:],
        ),
        axis=1,
    )
    processed = {
        "coord": model_coordinates,
        "grid_coord": np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64
        ),
        "color": colors,
        "feat": np.concatenate((raw_coordinates[:, :3], colors, normals), axis=1),
        "batch_idx": np.zeros(4, dtype=np.int64),
        "t": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "offset": np.asarray([2, 4], dtype=np.int64),
    }
    return summarize_native_preprocessed_sample(
        raw_coordinates=raw_coordinates,
        dataset_features=dataset_features,
        original_colors=colors,
        original_normals=normals,
        processed=processed,
        sequence_name="scene0003_00-scene0003_03",
    )


def _valid_visits() -> list[dict[str, object]]:
    return [
        {
            "visit_id": 0,
            "points_xyz": np.asarray(
                [[0.001, 0.001, 0.001], [0.021, 0.001, 0.001]],
                dtype=np.float32,
            ),
            "entity_offsets": np.asarray([0, 1, 2], dtype=np.int64),
            "entity_metadata": [
                {
                    "point_rgb_source": "camera_rgb",
                    "point_rgb": [[255, 0, 0]],
                    "point_normals_source": "source_geometry",
                    "point_normals": [[0.0, 0.0, 1.0]],
                },
                {
                    "point_rgb_source": "camera_rgb",
                    "point_rgb": [[0, 255, 0]],
                    "point_normals_source": "source_geometry",
                    "point_normals": [[0.0, 0.0, 1.0]],
                },
            ],
        },
        {
            "visit_id": 1,
            "points_xyz": np.asarray([[0.041, 0.001, 0.001]], dtype=np.float32),
            "entity_offsets": np.asarray([0, 1], dtype=np.int64),
            "entity_metadata": [
                {
                    "point_rgb_source": "camera_rgb",
                    "point_rgb": [[0, 0, 255]],
                    "point_normals_source": "source_geometry",
                    "point_normals": [[0.0, 0.0, 1.0]],
                }
            ],
        },
    ]


def test_native_witness_proves_actual_nine_channel_model_input() -> None:
    witness = _native_witness()

    assert witness["raw_coordinate_shape"] == [4, 4]
    assert witness["dataset_feature_shape"] == [4, 7]
    assert witness["processed_coordinate_shape"] == [4, 5]
    assert witness["model_feature_shape"] == [4, 9]
    assert witness["model_feature_layout"] == [
        "shared_centered_xyz",
        "camera_rgb_0_1",
        "unit_source_geometry_normals",
    ]
    assert witness["dataset_normalized_color_discarded_by_pointcept_collator"] is True
    assert witness["temporal_stage_values"] == [0, 1]
    assert witness["per_stage_point_count"] == {"0": 2, "1": 2}


def test_collision_free_features_and_tokens_pass() -> None:
    result = audit_input_contract(_native_witness(), _valid_visits(), 0.02)

    assert result["status"] == "INPUT_CONTRACT_PASS"
    assert result["gpu_authorized"] is True
    assert result["blocking_reasons"] == []
    assert result["visits"]["t0"]["adapter_token_count"] == 2
    assert result["visits"]["t0"]["model_input_token_count"] == 2
    assert result["visits"]["t0"]["token_merge_count"] == 0


def test_palette_rgb_is_rejected_as_camera_rgb() -> None:
    visits = _valid_visits()
    visits[0]["entity_metadata"][0]["point_rgb_source"] = "instance_palette"

    result = audit_input_contract(_native_witness(), visits, 0.02)

    assert result["status"] == "BLOCKED_INPUT_FEATURE_CONTRACT"
    assert "INSTANCE_PALETTE_RGB_FORBIDDEN" in result["blocking_reasons"]
    assert result["gpu_authorized"] is False


def test_missing_camera_rgb_and_normals_fail_closed() -> None:
    visits = _valid_visits()
    visits[0]["entity_metadata"][0] = {"instance_color_rgb": [103, 105, 198]}

    result = audit_input_contract(_native_witness(), visits, 0.02)

    assert result["status"] == "BLOCKED_INPUT_FEATURE_CONTRACT"
    assert "COLOR_NORMALIZATION_MISMATCH" in result["blocking_reasons"]
    assert "MISSING_SOURCE_GEOMETRY_NORMALS" in result["blocking_reasons"]


def test_cross_entity_spatial_voxel_merge_blocks_permutation_claim() -> None:
    visits = _valid_visits()
    visits[0]["points_xyz"][1] = [0.002, 0.001, 0.001]

    result = audit_input_contract(_native_witness(), visits, 0.02)

    t0 = result["visits"]["t0"]
    assert t0["adapter_token_count"] == 2
    assert t0["model_input_token_count"] == 1
    assert t0["cross_entity_collision_voxel_count"] == 1
    assert t0["token_merge_count"] == 1
    assert t0["permutation_possible"] is False
    assert result["status"] == "BLOCKED_RESCENE_TOKEN_CONSERVATION"


def test_temporal_values_must_be_exactly_zero_and_one() -> None:
    witness = _native_witness()
    witness["temporal_stage_values"] = [0, 2]

    with pytest.raises(InputContractError, match="temporal"):
        audit_input_contract(witness, _valid_visits(), 0.02)


def test_audit_does_not_mutate_input_arrays_or_metadata() -> None:
    visits = _valid_visits()
    before = copy.deepcopy(visits)

    audit_input_contract(_native_witness(), visits, 0.02)

    for observed, expected in zip(visits, before, strict=True):
        np.testing.assert_array_equal(observed["points_xyz"], expected["points_xyz"])
        np.testing.assert_array_equal(
            observed["entity_offsets"], expected["entity_offsets"]
        )
        assert observed["entity_metadata"] == expected["entity_metadata"]


def test_bound_module_loader_supports_dataclasses_without_module_leak(
    tmp_path,
) -> None:
    source = tmp_path / "bound_source.py"
    source.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Record:\n"
        "    value: int\n",
        encoding="utf-8",
    )

    module = _load_bound_module(source, "_test_bound_source")

    assert module.Record(3).value == 3
