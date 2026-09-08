from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.oviv2.observation_query.model import FrozenReSceneFeatures
from src.oviv2.observation_query.training import (
    BackboneCacheIdentity,
    ObservationCheckpointMetadata,
    ObservationTrainingStateError,
    capture_backbone_cache,
    load_backbone_cache,
    load_trainable_checkpoint,
    materialize_backbone_cache,
    save_backbone_cache,
    save_trainable_checkpoint,
)


class _Point(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


def _feature_level(offset: float) -> _Point:
    feat = torch.tensor([[offset, 1.0], [offset + 1.0, 2.0]])
    coord = torch.tensor([[0.0, offset, 0.0, 0.0, 0.0], [0.0, offset + 1.0, 0.0, 0.0, 1.0]])
    grid = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    batch = torch.zeros(2, dtype=torch.long)
    return _Point(feat=feat, coord=coord, grid_coord=grid, batch=batch)


def _identity() -> BackboneCacheIdentity:
    return BackboneCacheIdentity(
        pair_id="pair-a",
        base_checkpoint_sha256="a" * 64,
        model_input_sha256="b" * 64,
        serialization_id="temporal_overlay_v1",
        visit_order=(0, 1),
    )


def test_backbone_cache_round_trip_reconstructs_fresh_device_tensors(tmp_path) -> None:
    full = _feature_level(0.0)
    auxiliary = [_feature_level(10.0), _feature_level(20.0)]
    coordinates = [
        [torch.tensor([[0.0, 0.0, 0.0, 0.0]])],
        [torch.tensor([[1.0, 0.0, 0.0, 1.0]])],
    ]
    captured = capture_backbone_cache(
        FrozenReSceneFeatures(full, auxiliary, coordinates), _identity()
    )
    paths = save_backbone_cache(captured, tmp_path / "cache")
    loaded = load_backbone_cache(paths.root, expected_identity=_identity())

    class _Backbone:
        model_lib = SimpleNamespace(structure=SimpleNamespace(Point=_Point))

        @staticmethod
        def format(point: _Point) -> _Point:
            point.F = point.feat
            point.decomposed_features = [point.feat]
            point.decomposed_coordinates = [point.coord[:, 1:]]
            return point

    restored = materialize_backbone_cache(loaded, _Backbone(), device="cpu")
    torch.testing.assert_close(restored.pcd_features.F, full.feat)
    torch.testing.assert_close(
        restored.auxiliary_features[1].decomposed_features[0], auxiliary[1].feat
    )
    torch.testing.assert_close(restored.coordinates[1][0], coordinates[1][0])
    assert restored.pcd_features.F.data_ptr() != full.feat.data_ptr()
    assert not restored.pcd_features.F.requires_grad


def test_backbone_cache_rejects_identity_or_payload_tampering(tmp_path) -> None:
    full = _feature_level(0.0)
    captured = capture_backbone_cache(
        FrozenReSceneFeatures(full, [full], [[full.coord[:, 1:]]]), _identity()
    )
    paths = save_backbone_cache(captured, tmp_path / "cache")
    wrong = BackboneCacheIdentity(
        pair_id="pair-a",
        base_checkpoint_sha256="c" * 64,
        model_input_sha256="b" * 64,
        serialization_id="temporal_overlay_v1",
        visit_order=(0, 1),
    )
    with pytest.raises(ObservationTrainingStateError, match="identity"):
        load_backbone_cache(paths.root, expected_identity=wrong)

    paths.arrays.write_bytes(paths.arrays.read_bytes() + b"tamper")
    with pytest.raises(ObservationTrainingStateError, match="binding"):
        load_backbone_cache(paths.root, expected_identity=_identity())


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = nn.Module()
        self.native.backbone = nn.Linear(2, 2)
        self.native.backbone.requires_grad_(False)
        self.native.decoder = nn.Linear(2, 2)
        self.obs_branch = nn.Linear(2, 2)
        self.register_buffer("runtime_scale", torch.tensor(2.0))


def _checkpoint_metadata() -> ObservationCheckpointMetadata:
    return ObservationCheckpointMetadata(
        model_variant="OBS_FULL",
        base_checkpoint_sha256="a" * 64,
        source_commit="b" * 40,
        resolved_config={"learning_rate": 1e-4},
        split_id="splits_v1",
        observation_sha256="c" * 64,
        backbone_cache_sha256="d" * 64,
        seed=45,
        optimizer_updates=3,
    )


def test_v2_checkpoint_metadata_records_training_dataset_and_feature_schema() -> None:
    legacy = _checkpoint_metadata()
    assert "training_dataset_manifest" not in legacy.as_dict()
    metadata = ObservationCheckpointMetadata(
        **legacy.as_dict(),
        training_dataset_manifest={
            "artifact_id": "OVI_OBSERVATION_TRAINING_DATASET_V2",
            "environment_ids": ["environment-a"],
            "pairs": [{"pair_id": "pair-a", "observation_sha256": "e" * 64}],
        },
        model_architecture_version="OVI_OBSERVATION_QUERY_V1",
        input_feature_schema={
            "observation_feature_dim": 1024,
            "observation_metadata_dim": 11,
            "model_input_feature_dim": 9,
        },
    )

    restored = ObservationCheckpointMetadata(**metadata.as_dict())

    assert restored == metadata
    assert restored.training_dataset_manifest["environment_ids"] == ["environment-a"]


def test_trainable_checkpoint_restores_all_updated_state_but_not_frozen_backbone(tmp_path) -> None:
    torch.manual_seed(4)
    model = _TinyModel()
    criterion = nn.Linear(2, 1)
    expected_decoder = model.native.decoder.weight.detach().clone()
    expected_observation = model.obs_branch.bias.detach().clone()
    expected_criterion = criterion.weight.detach().clone()
    expected_buffer = model.runtime_scale.detach().clone()
    frozen_before_save = model.native.backbone.weight.detach().clone()

    paths = save_trainable_checkpoint(
        model=model,
        criterion=criterion,
        output_root=tmp_path / "checkpoint",
        metadata=_checkpoint_metadata(),
    )
    with torch.no_grad():
        model.native.decoder.weight.add_(10.0)
        model.obs_branch.bias.add_(10.0)
        criterion.weight.add_(10.0)
        model.runtime_scale.add_(10.0)
        model.native.backbone.weight.add_(10.0)
    load_trainable_checkpoint(
        model=model,
        criterion=criterion,
        checkpoint_root=paths.root,
        expected_metadata=_checkpoint_metadata(),
    )

    torch.testing.assert_close(model.native.decoder.weight, expected_decoder)
    torch.testing.assert_close(model.obs_branch.bias, expected_observation)
    torch.testing.assert_close(criterion.weight, expected_criterion)
    torch.testing.assert_close(model.runtime_scale, expected_buffer)
    assert not torch.equal(model.native.backbone.weight, frozen_before_save)


def test_trainable_checkpoint_rejects_wrong_base_or_tampered_weights(tmp_path) -> None:
    model = _TinyModel()
    criterion = nn.Linear(2, 1)
    paths = save_trainable_checkpoint(
        model=model,
        criterion=criterion,
        output_root=tmp_path / "checkpoint",
        metadata=_checkpoint_metadata(),
    )
    wrong = ObservationCheckpointMetadata(
        **{
            **_checkpoint_metadata().as_dict(),
            "base_checkpoint_sha256": "e" * 64,
        }
    )
    with pytest.raises(ObservationTrainingStateError, match="metadata"):
        load_trainable_checkpoint(
            model=model,
            criterion=criterion,
            checkpoint_root=paths.root,
            expected_metadata=wrong,
        )

    paths.weights.write_bytes(paths.weights.read_bytes() + b"tamper")
    with pytest.raises(ObservationTrainingStateError, match="binding"):
        load_trainable_checkpoint(
            model=model,
            criterion=criterion,
            checkpoint_root=paths.root,
            expected_metadata=_checkpoint_metadata(),
        )
