from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.training import train_ovi_observation_query as train_runner


def _module():
    return importlib.import_module("src.training.ovi_observation_multienv")


def _split_and_runtime(tmp_path: Path):
    split = {
        "environments": [
            {
                "role": "TRAIN",
                "environment_uuid": "environment-a",
                "pair_id": "pair-a",
                "sessions": [{"scan_uuid": "a0"}, {"scan_uuid": "a1"}],
            },
            {
                "role": "DEV",
                "environment_uuid": "environment-dev",
                "pair_id": "pair-dev",
                "sessions": [{"scan_uuid": "d0"}, {"scan_uuid": "d1"}],
            },
            {
                "role": "TRAIN",
                "environment_uuid": "environment-b",
                "pair_id": "pair-b",
                "sessions": [{"scan_uuid": "b0"}, {"scan_uuid": "b1"}],
            },
        ]
    }
    runtime = {
        "pairs": {
            "second": {
                "role": "TRAIN",
                "pair_id": "pair-b",
                "artifact_root": str(tmp_path / "pair-b"),
            },
            "first": {
                "role": "TRAIN",
                "pair_id": "pair-a",
                "artifact_root": str(tmp_path / "pair-a"),
            },
            "dev": {
                "role": "DEV",
                "pair_id": "pair-dev",
                "artifact_root": str(tmp_path / "pair-dev"),
            },
        }
    }
    return split, runtime


def test_collect_training_pairs_preserves_requested_order_and_legacy(
    tmp_path: Path,
) -> None:
    module = _module()
    split, runtime = _split_and_runtime(tmp_path)

    entries = module.collect_training_pair_entries(
        split=split,
        runtime=runtime,
        requested_pair_ids=("pair-b", "pair-a"),
    )
    legacy = module.collect_training_pair_entries(
        split=split,
        runtime=runtime,
        requested_pair_ids=("pair-a",),
    )

    assert [entry.pair_id for entry in entries] == ["pair-b", "pair-a"]
    assert [entry.environment_id for entry in entries] == [
        "environment-b",
        "environment-a",
    ]
    assert [entry.artifact_root for entry in entries] == [
        (tmp_path / "pair-b").absolute(),
        (tmp_path / "pair-a").absolute(),
    ]
    assert len(legacy) == 1
    assert legacy[0].pair_id == "pair-a"


def test_pair_sampler_covers_each_environment_once_per_epoch_and_resumes_exactly() -> (
    None
):
    module = _module()
    pair_ids = ("pair-a", "pair-b", "pair-c")
    sampler = module.EnvironmentBalancedPairSampler(pair_ids=pair_ids, seed=45)

    first_five = [sampler.next_pair_id() for _ in range(5)]
    state = sampler.state_dict()
    uninterrupted_tail = [sampler.next_pair_id() for _ in range(7)]
    resumed = module.EnvironmentBalancedPairSampler(pair_ids=pair_ids, seed=45)
    resumed.load_state_dict(state)

    assert first_five == ["pair-a", "pair-c", "pair-b", "pair-a", "pair-c"]
    assert uninterrupted_tail == [
        "pair-b",
        "pair-c",
        "pair-b",
        "pair-a",
        "pair-c",
        "pair-a",
        "pair-b",
    ]
    assert [resumed.next_pair_id() for _ in range(7)] == uninterrupted_tail
    assert state["epoch"] == 1
    assert state["cursor"] == 2
    assert state["order"] == ["pair-a", "pair-c", "pair-b"]
    assert state["micro_steps_completed"] == 5
    assert state["exposure_counts"] == {
        "pair-a": 2,
        "pair-b": 1,
        "pair-c": 2,
    }


def test_pair_sampler_rejects_dataset_change_on_resume() -> None:
    module = _module()
    sampler = module.EnvironmentBalancedPairSampler(
        pair_ids=("pair-a", "pair-b"), seed=45
    )
    sampler.next_pair_id()
    state = sampler.state_dict()
    changed = module.EnvironmentBalancedPairSampler(
        pair_ids=("pair-a", "pair-c"), seed=45
    )

    with pytest.raises(module.MultiEnvironmentTrainingError, match="dataset"):
        changed.load_state_dict(state)


def test_runner_accepts_plural_pair_ids_and_preserves_legacy_default(
    tmp_path: Path,
) -> None:
    split, runtime = _split_and_runtime(tmp_path)
    plural_args = train_runner._parser().parse_args(
        [
            "--config",
            "config.json",
            "--runtime",
            "runtime.json",
            "--method",
            "OBS_FULL",
            "--run-id",
            "run",
            "--stage",
            "pilot",
            "--train-pair-ids",
            "pair-b",
            "pair-a",
        ]
    )
    legacy_runtime = {
        "pairs": {"train": runtime["pairs"]["first"], "dev": runtime["pairs"]["dev"]}
    }
    legacy_args = train_runner._parser().parse_args(
        [
            "--config",
            "config.json",
            "--runtime",
            "runtime.json",
            "--method",
            "OBS_FULL",
            "--run-id",
            "run",
            "--stage",
            "pilot",
        ]
    )

    assert train_runner._requested_train_pair_ids(plural_args, runtime) == (
        "pair-b",
        "pair-a",
    )
    assert train_runner._requested_train_pair_ids(legacy_args, legacy_runtime) == (
        "pair-a",
    )
    entries = _module().collect_training_pair_entries(
        split=split,
        runtime=runtime,
        requested_pair_ids=("pair-b", "pair-a"),
    )
    assert [entry.environment_id for entry in entries] == [
        "environment-b",
        "environment-a",
    ]


def test_asset_gate_uses_derived_dependencies_not_historical_sequence_zip(
    tmp_path: Path,
) -> None:
    split, runtime = _split_and_runtime(tmp_path)
    for pair_id in ("pair-a", "pair-b"):
        root = tmp_path / pair_id
        for relative in (
            "model_bundle/manifest.json",
            "observation_bank/bank/manifest.json",
            "training_targets/manifest.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
    for environment in split["environments"]:
        if environment["role"] == "TRAIN":
            for session in environment["sessions"]:
                session["sequence_zip"] = {
                    "path": str(
                        tmp_path / "missing" / session["scan_uuid"] / "sequence.zip"
                    )
                }
    entries = _module().collect_training_pair_entries(
        split=split,
        runtime=runtime,
        requested_pair_ids=("pair-a", "pair-b"),
    )

    gate = train_runner._asset_gate(
        config_path=tmp_path / "config.json",
        runtime_path=tmp_path / "runtime.json",
        split_path=tmp_path / "split.json",
        entries=entries,
        run_id="run",
        method="OBS_FULL",
        stage="pilot",
        output_root=tmp_path / "runs" / "run",
    )

    assert gate is None
    assert not (tmp_path / "runs" / "run").exists()


def test_main_asset_gate_covers_every_requested_train_pair(tmp_path: Path) -> None:
    split_payload, runtime_payload = _split_and_runtime(tmp_path)
    for environment in split_payload["environments"]:
        if environment["role"] == "TRAIN":
            for session in environment["sessions"]:
                session["sequence_zip"] = {
                    "path": str(
                        tmp_path / "missing" / session["scan_uuid"] / "sequence.zip"
                    )
                }
    split = tmp_path / "split.json"
    split.write_text(json.dumps(split_payload), encoding="utf-8")
    runtime_payload["python"] = {"model": str(Path(sys.executable))}
    runtime_payload["cache_root"] = str(tmp_path / "runs")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(runtime_payload), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "split_manifest": str(split),
                "methods": {"OBS_FULL": {"trained": True}},
                "training": {"optimizer": "AdamW"},
            }
        ),
        encoding="utf-8",
    )

    result = train_runner.main(
        [
            "--config",
            str(config),
            "--runtime",
            str(runtime),
            "--method",
            "OBS_FULL",
            "--run-id",
            "plural-gate",
            "--stage",
            "pilot",
            "--train-pair-ids",
            "pair-b",
            "pair-a",
        ]
    )

    assert result == 2
    status = json.loads(
        (
            tmp_path / "runs" / "training_runs" / "plural-gate" / "status.json"
        ).read_text()
    )
    assert status["pair_ids"] == ["pair-b", "pair-a"]
    assert set(status["missing_training_artifacts"]) == {
        "pair-b:model_bundle_manifest",
        "pair-b:observation_bank_manifest",
        "pair-b:training_target_manifest",
        "pair-a:model_bundle_manifest",
        "pair-a:observation_bank_manifest",
        "pair-a:training_target_manifest",
    }


def test_balanced_update_uses_the_selected_pairs_bank_targets_and_shape() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    model = TinyModel()
    criterion = nn.Identity()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sampler = _module().EnvironmentBalancedPairSampler(
        pair_ids=("pair-a", "pair-b", "pair-c"), seed=45
    )
    payloads = {
        "pair-a": {
            "pair_id": "pair-a",
            "bank_pair_id": "pair-a",
            "target_pair_id": "pair-a",
            "point_count": 7,
            "target": 2.0,
        },
        "pair-b": {
            "pair_id": "pair-b",
            "bank_pair_id": "pair-b",
            "target_pair_id": "pair-b",
            "point_count": 11,
            "target": 3.0,
        },
        "pair-c": {
            "pair_id": "pair-c",
            "bank_pair_id": "pair-c",
            "target_pair_id": "pair-c",
            "point_count": 5,
            "target": 4.0,
        },
    }

    def loss_for_pair(payload, _optimizer_update: int):
        assert payload["pair_id"] == payload["bank_pair_id"]
        assert payload["pair_id"] == payload["target_pair_id"]
        loss = (model.weight - payload["target"]).square()
        return loss, (payload["pair_id"], payload["point_count"])

    stats, pair_ids, results = train_runner.perform_environment_balanced_update(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        sampler=sampler,
        pair_payloads=payloads,
        loss_for_pair=loss_for_pair,
        optimizer_update=0,
        gradient_accumulation=2,
        gradient_clip_norm=10.0,
    )

    assert pair_ids == ("pair-a", "pair-c")
    assert results == (("pair-a", 7), ("pair-c", 5))
    assert stats["all_gradients_finite"] is True
    assert model.weight.item() != 1.0


def test_training_dataset_manifest_and_aggregate_hash_bind_every_pair() -> None:
    module = _module()
    identities = (
        module.TrainingPairIdentity(
            environment_id="environment-a",
            pair_id="pair-a",
            model_input_sha256="a" * 64,
            observation_sha256="b" * 64,
            training_target_sha256="c" * 64,
            backbone_cache_sha256="d" * 64,
        ),
        module.TrainingPairIdentity(
            environment_id="environment-b",
            pair_id="pair-b",
            model_input_sha256="e" * 64,
            observation_sha256="f" * 64,
            training_target_sha256="0" * 64,
            backbone_cache_sha256="1" * 64,
        ),
    )

    manifest = module.build_training_dataset_manifest(
        identities, data_seed=45, gradient_accumulation=2
    )

    assert manifest == {
        "schema_version": 3,
        "artifact_id": "OVI_OBSERVATION_TRAINING_DATASET_MULTIENV_V3",
        "environment_ids": ["environment-a", "environment-b"],
        "pairs": [
            {
                "environment_id": "environment-a",
                "pair_id": "pair-a",
                "model_input_sha256": "a" * 64,
                "observation_sha256": "b" * 64,
                "training_target_sha256": "c" * 64,
                "backbone_cache_sha256": "d" * 64,
            },
            {
                "environment_id": "environment-b",
                "pair_id": "pair-b",
                "model_input_sha256": "e" * 64,
                "observation_sha256": "f" * 64,
                "training_target_sha256": "0" * 64,
                "backbone_cache_sha256": "1" * 64,
            },
        ],
        "sampling": {
            "policy": "environment_balanced_epoch_shuffle",
            "data_seed": 45,
            "gradient_accumulation": 2,
        },
    }
    assert (
        module.aggregate_pair_binding_sha256(identities, "observation_sha256")
        == "2b5de0bd49a6beb2630db15a9f617ea9e2eb0ec944cdc8ad0597066e80a9db2e"
    )
    assert (
        module.aggregate_pair_binding_sha256(identities, "backbone_cache_sha256")
        == "504aa6e5929005947ee009aed95a622de3b95f2b55fc00e0e331f376595bda74"
    )
