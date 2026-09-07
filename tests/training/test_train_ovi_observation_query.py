from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.training.train_ovi_observation_query import (
    build_optimizer,
    main,
    perform_accumulated_update,
    training_method_contract,
)


class _TrainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = nn.Module()
        self.native.backbone = nn.Linear(2, 2)
        self.native.backbone.requires_grad_(False)
        self.native.decoder = nn.Linear(2, 2)
        self.obs_branch = nn.Linear(2, 2)


def test_optimizer_uses_separate_native_and_observation_rates() -> None:
    model = _TrainableModel()
    criterion = nn.Linear(2, 1)

    optimizer, groups = build_optimizer(
        model=model,
        criterion=criterion,
        method="OBS_FULL",
        native_learning_rate=1e-5,
        observation_learning_rate=1e-4,
        weight_decay=1e-4,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert groups == {
        "native": tuple(
            sorted(
                name
                for name, _ in model.native.decoder.named_parameters(
                    prefix="native.decoder"
                )
            )
        ),
        "observation": tuple(
            sorted(
                [
                    *(
                        name
                        for name, _ in model.obs_branch.named_parameters(
                            prefix="obs_branch"
                        )
                    ),
                    *(
                        name
                        for name, _ in criterion.named_parameters(prefix="criterion")
                    ),
                ]
            )
        ),
    }
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in optimized_ids
        for parameter in model.native.backbone.parameters()
    )


@pytest.mark.parametrize(
    ("method", "mode", "uses_regions", "uses_consistency"),
    [
        ("OBS_BASE_TUNED", "base_tuned", False, False),
        ("OBS_LATE", "late", True, False),
        ("OBS_FUSE", "fuse", True, False),
        ("OBS_ATTN", "attention", True, False),
        ("OBS_FULL", "full", True, True),
        ("OBS_NO_FEEDBACK", "no_feedback", True, True),
        ("OBS_NO_CONSISTENCY", "no_consistency", True, False),
    ],
)
def test_training_method_contracts_are_explicit(
    method: str, mode: str, uses_regions: bool, uses_consistency: bool
) -> None:
    contract = training_method_contract(method)

    assert contract.observation_mode == mode
    assert contract.uses_region_supervision is uses_regions
    assert contract.uses_consistency is uses_consistency


def test_base_tuned_optimizer_excludes_observation_parameters() -> None:
    model = _TrainableModel()
    criterion = nn.Linear(2, 1)

    optimizer, groups = build_optimizer(
        model=model,
        criterion=criterion,
        method="OBS_BASE_TUNED",
        native_learning_rate=1e-5,
        observation_learning_rate=1e-4,
        weight_decay=1e-4,
    )

    assert tuple(groups) == ("native",)
    assert len(optimizer.param_groups) == 1


def test_accumulated_update_changes_both_groups_and_leaves_backbone_without_grad() -> (
    None
):
    torch.manual_seed(8)
    model = _TrainableModel()
    criterion = nn.Linear(2, 1)
    optimizer, _groups = build_optimizer(
        model=model,
        criterion=criterion,
        method="OBS_FULL",
        native_learning_rate=1e-2,
        observation_learning_rate=1e-2,
        weight_decay=0.0,
    )
    native_before = model.native.decoder.weight.detach().clone()
    observation_before = model.obs_branch.weight.detach().clone()
    calls: list[int] = []

    def loss_factory(micro_step: int) -> torch.Tensor:
        calls.append(micro_step)
        value = torch.ones(1, 2)
        decoded = model.native.decoder(value)
        observed = model.obs_branch(decoded)
        return criterion(observed).square().mean()

    result = perform_accumulated_update(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loss_factory=loss_factory,
        gradient_accumulation=2,
        gradient_clip_norm=1.0,
    )

    assert calls == [0, 1]
    assert result["all_gradients_finite"] is True
    assert result["gradient_norm_before_clip"] > 0.0
    assert not torch.equal(model.native.decoder.weight, native_before)
    assert not torch.equal(model.obs_branch.weight, observation_before)
    assert all(
        parameter.grad is None for parameter in model.native.backbone.parameters()
    )


def test_missing_train_assets_publish_asset_gated_status_without_checkpoint(
    tmp_path,
) -> None:
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "environments": [
                    {
                        "role": "TRAIN",
                        "pair_id": "train-pair",
                        "environment_uuid": "train-environment",
                        "sessions": [
                            {
                                "visit_id": 0,
                                "scan_uuid": "scan-0",
                                "sequence_zip": {
                                    "path": str(tmp_path / "scan-0" / "sequence.zip"),
                                    "status": "MISSING",
                                },
                            },
                            {
                                "visit_id": 1,
                                "scan_uuid": "scan-1",
                                "sequence_zip": {
                                    "path": str(tmp_path / "scan-1" / "sequence.zip"),
                                    "status": "MISSING",
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_manifest": str(split),
                "training": {
                    "seed": 45,
                    "optimizer": "AdamW",
                    "native_learning_rate": 1e-5,
                    "observation_learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "gradient_clip_norm": 1.0,
                    "gradient_accumulation": 2,
                    "smoke_updates": 200,
                },
                "loss": {},
                "model": {},
                "methods": {"OBS_FULL": {"trained": True}},
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": {"model": str(Path(__import__("sys").executable))},
                "cache_root": str(tmp_path / "runs"),
                "assets": {"rescene_checkpoint": str(tmp_path / "base.ckpt")},
                "pairs": {
                    "train": {
                        "pair_id": "train-pair",
                        "status": "MISSING_TRAIN_RGBD_SEQUENCE",
                        "artifact_root": str(tmp_path / "train-artifacts"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    return_code = main(
        [
            "--config",
            str(config),
            "--runtime",
            str(runtime),
            "--method",
            "OBS_FULL",
            "--run-id",
            "asset-gate-test",
            "--stage",
            "smoke",
        ]
    )

    assert return_code == 2
    status_path = (
        tmp_path / "runs" / "training_runs" / "asset-gate-test" / "status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "TRAINING_ASSET_GATED"
    assert status["pair_id"] == "train-pair"
    assert status["missing_train_sequence_zip_count"] == 2
    assert status["validation_substitution_allowed"] is False
    assert not (status_path.parent / "checkpoint").exists()
