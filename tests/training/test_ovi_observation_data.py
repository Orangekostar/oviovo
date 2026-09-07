from __future__ import annotations

import json

import numpy as np
import pytest

from src.oviv2.observation_query.contracts import ObservationBank
from src.oviv2.rescene_input_bridge import ReSceneModelInput
from src.training.ovi_observation_data import (
    LabelTransferConfig,
    NativeVisitLabels,
    ObservationTrainingDataError,
    PairIdentityRules,
    build_observation_training_sample,
    load_native_processed_labels,
    load_observation_training_sample,
    load_official_pair_training_metadata,
    load_rescene_class_mapping,
    load_split_pair,
    save_observation_training_sample,
)


def _model_input() -> ReSceneModelInput:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    visits = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    model_count = len(points)
    return ReSceneModelInput(
        coordinates_bxyzt=np.column_stack(
            (np.zeros(model_count, dtype=np.float32), points, visits)
        ),
        grid_coordinates_xyz=np.column_stack(
            (np.arange(model_count), np.zeros((model_count, 2), dtype=np.int64))
        ),
        features=np.column_stack(
            (
                points,
                np.full((model_count, 3), 0.5, dtype=np.float32),
                np.tile([0.0, 0.0, 1.0], (model_count, 1)),
            )
        ),
        sparse_batch_offsets=np.asarray([4, 7], dtype=np.int64),
        point2segment=np.arange(model_count, dtype=np.int64),
        adapter_to_model=np.asarray([2, 0, 1, 3, 6, 4, 5, 0], dtype=np.int64),
        model_visit_ids=visits,
        representative_source_point_indices=np.asarray(
            [7, 3, 5, 1, 6, 0, 2], dtype=np.int64
        ),
        local_frame_indices=np.zeros(model_count, dtype=np.int64),
        global_frame_indices=np.zeros(model_count, dtype=np.int64),
        rows=np.arange(model_count, dtype=np.int64),
        columns=np.arange(model_count, dtype=np.int64),
        camera_depth_m=np.ones(model_count, dtype=np.float32),
        observed_depth_m=np.ones(model_count, dtype=np.float32),
        depth_residual_m=np.zeros(model_count, dtype=np.float32),
        neural_voxel_size_m=0.02,
        adapter_geometry_sha256="a" * 64,
        native_sampling_sha256="b" * 64,
        surface_attributes_sha256="c" * 64,
        model_candidates_sha256="d" * 64,
        sampler_source_sha256="e" * 64,
    )


def _observation_bank(model_input: ReSceneModelInput) -> ObservationBank:
    return ObservationBank(
        pair_id="reference-rescan",
        model_input_sha256=model_input.content_sha256(),
        region_keys=("v0-mixed", "v1-shared", "v1-ambiguous"),
        region_visit_ids=np.asarray([0, 1, 1], dtype=np.int8),
        region_frame_ids=np.asarray([0, 10, 11], dtype=np.int64),
        region_features=np.eye(3, dtype=np.float32),
        region_metadata=np.zeros((3, 15), dtype=np.float32),
        csr_indptr=np.asarray([0, 2, 3, 5], dtype=np.int64),
        csr_model_indices=np.asarray([0, 2, 4, 5, 6], dtype=np.int64),
        csr_weights=np.asarray([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
        region_reliability=np.ones(3, dtype=np.float32),
        model_visit_ids=model_input.model_visit_ids,
        source_manifest={"frontend": {"sha256": "f" * 64}},
    )


def _cluster(center: float, semantic: int, instance: int, segment: int):
    points = np.asarray(
        [[center - 0.005, 0.0, 0.0], [center + 0.005, 0.0, 0.0]],
        dtype=np.float64,
    )
    return points, [semantic, semantic], [instance, instance], [segment, segment]


def _visit_labels(visit_id: int) -> NativeVisitLabels:
    if visit_id == 0:
        records = (
            _cluster(2.0, 1, 2, 200),
            _cluster(0.0, 3, 10, 100),
            _cluster(1.0, 8, 20, 110),
        )
        extra = (np.asarray([[3.0, 0.0, 0.0]]), [3], [30], [130])
    else:
        records = (
            _cluster(12.0, 40, 99, 320),
            _cluster(10.0, 3, 77, 300),
            _cluster(11.0, 8, 88, 310),
        )
        extra = None
    points = [record[0] for record in records]
    semantics = [value for record in records for value in record[1]]
    instances = [value for record in records for value in record[2]]
    segments = [value for record in records for value in record[3]]
    if extra is not None:
        points.append(extra[0])
        semantics.extend(extra[1])
        instances.extend(extra[2])
        segments.extend(extra[3])
    return NativeVisitLabels(
        visit_id=visit_id,
        points_reference_xyz=np.concatenate(points),
        semantic_ids=np.asarray(semantics, dtype=np.int64),
        instance_ids=np.asarray(instances, dtype=np.int64),
        segment_ids=np.asarray(segments, dtype=np.int64),
        source_sha256=("1" if visit_id == 0 else "2") * 64,
    )


def _sample():
    model_input = _model_input()
    observations = _observation_bank(model_input)
    sample = build_observation_training_sample(
        model_input=model_input,
        observations=observations,
        model_points_reference_xyz=model_input.features[:, :3],
        visit_labels=(_visit_labels(0), _visit_labels(1)),
        identity_rules=PairIdentityRules(
            rescan_to_reference={77: 10},
            ambiguous_instance_ids_by_visit=(frozenset({20}), frozenset({88})),
            removed_reference_ids=frozenset(),
            source_sha256="3" * 64,
        ),
        raw_semantic_to_model_class={3: 0, 8: 5},
        config=LabelTransferConfig(
            maximum_distance_m=0.02,
            minimum_neighbors=2,
            minimum_consensus_fraction=0.75,
            maximum_neighbors=8,
            trusted_background_raw_ids=(1, 2),
            ignored_raw_semantic_ids=(0, 255),
            minimum_region_valid_fraction=0.75,
        ),
    )
    return model_input, observations, sample


def test_pair_sample_keeps_visits_unknowns_and_ambiguity_separate() -> None:
    _, _, sample = _sample()

    assert sample.temporal_identity_keys == (
        "reference:10",
        "visit:0:instance:20",
        "visit:1:instance:88",
        "visit:1:instance:99",
    )
    np.testing.assert_array_equal(
        sample.label_valid.numpy(), [True, True, True, False, True, True, True]
    )
    np.testing.assert_array_equal(
        sample.instance_masks.numpy(),
        [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
    )
    assert not sample.label_valid[3]
    assert not sample.instance_masks[:, 3].any()
    assert sample.label_valid[2] and not sample.instance_masks[:, 2].any()
    np.testing.assert_array_equal(sample.class_targets.numpy(), [0, 5, 5, -100])
    np.testing.assert_array_equal(sample.class_valid.numpy(), [True, True, True, False])
    assert sample.ambiguity_metadata["strong_identity_ignored"] == {
        "visit:0:instance:20": 20,
        "visit:1:instance:88": 88,
    }


def test_region_targets_are_soft_and_use_only_trusted_m_support() -> None:
    _, _, sample = _sample()

    np.testing.assert_allclose(
        sample.region_instance_mass.numpy(),
        [
            [0.5, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
        ],
    )
    np.testing.assert_array_equal(sample.region_label_valid.numpy(), [True, True, True])
    assert sample.target_provenance["ground_truth_role"] == "criterion_only"
    assert sample.target_provenance["model_forward_fields"] == [
        "model_input",
        "observations",
    ]


def test_same_numeric_instance_is_not_shared_without_an_explicit_official_rule() -> None:
    labels = _visit_labels(1)
    labels = NativeVisitLabels(
        visit_id=1,
        points_reference_xyz=labels.points_reference_xyz,
        semantic_ids=labels.semantic_ids,
        instance_ids=np.where(labels.instance_ids == 77, 10, labels.instance_ids),
        segment_ids=labels.segment_ids,
        source_sha256=labels.source_sha256,
    )
    model_input = _model_input()

    sample = build_observation_training_sample(
        model_input=model_input,
        observations=_observation_bank(model_input),
        model_points_reference_xyz=model_input.features[:, :3],
        visit_labels=(_visit_labels(0), labels),
        identity_rules=PairIdentityRules(
            rescan_to_reference={},
            ambiguous_instance_ids_by_visit=(frozenset(), frozenset()),
            removed_reference_ids=frozenset(),
            source_sha256="3" * 64,
        ),
        raw_semantic_to_model_class={3: 0, 8: 5},
        config=LabelTransferConfig(
            maximum_distance_m=0.02,
            minimum_neighbors=2,
            minimum_consensus_fraction=0.75,
            maximum_neighbors=8,
            trusted_background_raw_ids=(1, 2),
            ignored_raw_semantic_ids=(0, 255),
            minimum_region_valid_fraction=0.75,
        ),
    )

    assert "reference:10" in sample.temporal_identity_keys
    assert "visit:1:instance:10" in sample.temporal_identity_keys
    assert sample.instance_masks[sample.temporal_identity_keys.index("reference:10"), 4] == 0


def test_training_sample_cache_round_trips_and_binds_inputs(tmp_path) -> None:
    model_input, observations, sample = _sample()

    paths = save_observation_training_sample(sample, tmp_path / "sample")
    loaded = load_observation_training_sample(
        paths.root,
        model_input=model_input,
        observations=observations,
    )

    assert loaded.temporal_identity_keys == sample.temporal_identity_keys
    assert loaded.target_provenance == sample.target_provenance
    np.testing.assert_array_equal(loaded.instance_masks, sample.instance_masks)
    np.testing.assert_allclose(loaded.region_instance_mass, sample.region_instance_mass)

    changed = _observation_bank(model_input)
    object.__setattr__(changed, "pair_id", "another-pair")
    with pytest.raises(ObservationTrainingDataError, match="observation"):
        load_observation_training_sample(
            paths.root,
            model_input=model_input,
            observations=changed,
        )


def test_rescene_class_mapping_uses_validation_order_and_label_offset(tmp_path) -> None:
    label_db = tmp_path / "labels.yaml"
    label_db.write_text(
        """
1: {name: wall, validation: true}
2: {name: floor, validation: true}
3: {name: cabinet, validation: true}
8: {name: door, validation: true}
40: {name: otherprop, validation: false}
""".lstrip(),
        encoding="utf-8",
    )

    mapping = load_rescene_class_mapping(
        label_db,
        expected_validation_label_count=4,
        label_offset=2,
    )

    assert mapping == {3: 0, 8: 1}
    assert 40 not in mapping


def test_environment_split_loader_rejects_cross_role_leakage(tmp_path) -> None:
    manifest = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBSERVATION_QUERY_SPLITS_V1",
        "environments": [
            {
                "role": "TRAIN",
                "environment_uuid": "env-a",
                "pair_id": "pair-a",
                "sessions": [{"scan_uuid": "scan-a"}, {"scan_uuid": "scan-b"}],
            },
            {
                "role": "DEV",
                "environment_uuid": "env-b",
                "pair_id": "pair-b",
                "sessions": [{"scan_uuid": "scan-c"}, {"scan_uuid": "scan-d"}],
            },
        ],
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    selected = load_split_pair(path, pair_id="pair-a", required_role="TRAIN")
    assert selected["environment_uuid"] == "env-a"

    manifest["environments"][1]["sessions"][0]["scan_uuid"] = "scan-a"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ObservationTrainingDataError, match="split"):
        load_split_pair(path, pair_id="pair-a", required_role="TRAIN")


def test_pair_and_model_identity_must_match() -> None:
    model_input = _model_input()
    bank = _observation_bank(model_input)
    object.__setattr__(bank, "pair_id", "wrong-pair")

    with pytest.raises(ObservationTrainingDataError, match="pair"):
        build_observation_training_sample(
            model_input=model_input,
            observations=bank,
            model_points_reference_xyz=model_input.features[:, :3],
            visit_labels=(_visit_labels(0), _visit_labels(1)),
            identity_rules=PairIdentityRules(
                rescan_to_reference={77: 10},
                ambiguous_instance_ids_by_visit=(frozenset(), frozenset()),
                removed_reference_ids=frozenset(),
                source_sha256="3" * 64,
            ),
            raw_semantic_to_model_class={3: 0, 8: 5},
            config=LabelTransferConfig(),
            expected_pair_id="reference-rescan",
        )


def test_official_pair_metadata_expands_verified_unchanged_ids(tmp_path) -> None:
    metadata_path = tmp_path / "3RScan.json"
    transform = np.eye(4)
    transform[3, :3] = [1.0, 2.0, 3.0]
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "reference": "reference",
                    "ambiguity": [
                        [
                            {
                                "instance_source": 20,
                                "instance_target": 21,
                                "transform": np.eye(4).reshape(-1).tolist(),
                            }
                        ]
                    ],
                    "scans": [
                        {
                            "reference": "rescan",
                            "transform": transform.reshape(-1).tolist(),
                            "rigid": [
                                {
                                    "instance_reference": 10,
                                    "instance_rescan": 77,
                                    "symmetry": 0,
                                    "transform": np.eye(4).reshape(-1).tolist(),
                                }
                            ],
                            "nonrigid": [],
                            "removed": [30],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = load_official_pair_training_metadata(
        metadata_path,
        reference_scan_uuid="reference",
        rescan_uuid="rescan",
        reference_instance_ids={5, 10, 20, 21, 30},
        rescan_instance_ids={5, 20, 21, 77},
    )

    assert metadata.identity_rules.rescan_to_reference == {
        5: 5,
        20: 20,
        21: 21,
        77: 10,
    }
    assert metadata.identity_rules.removed_reference_ids == frozenset({30})
    assert metadata.identity_rules.ambiguous_instance_ids_by_visit == (
        frozenset({20, 21}),
        frozenset({20, 21}),
    )
    np.testing.assert_array_equal(metadata.rescan_to_reference_row, transform)
    assert metadata.identity_rules.temporal_key(1, 5) == "reference:5"
    assert metadata.identity_rules.temporal_key(1, 20) == "visit:1:instance:20"


def test_native_processed_loader_preserves_preprocessor_reference_coordinates(tmp_path) -> None:
    source = tmp_path / "scan.npy"
    points = np.zeros((2, 12), dtype=np.float32)
    points[:, :3] = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    points[:, 9:12] = [[100, 3, 10], [101, 8, 20]]
    np.save(source, points)
    labels = load_native_processed_labels(
        source,
        visit_id=1,
    )

    np.testing.assert_array_equal(
        labels.points_reference_xyz,
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
    )
    np.testing.assert_array_equal(labels.segment_ids, [100, 101])
    np.testing.assert_array_equal(labels.semantic_ids, [3, 8])
    np.testing.assert_array_equal(labels.instance_ids, [10, 20])

    points[0, 10] = 3.5
    np.save(source, points)
    with pytest.raises(ObservationTrainingDataError, match="integer"):
        load_native_processed_labels(
            source,
            visit_id=0,
        )
