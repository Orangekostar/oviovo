#!/usr/bin/env python3
"""Train the OVI observation-query branch on an explicitly TRAIN-role pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ObservationTrainingRunError(ValueError):
    """Raised when a training request violates its fixed experiment contract."""


@dataclass(frozen=True, slots=True)
class TrainingMethodContract:
    observation_mode: str
    uses_region_supervision: bool
    uses_consistency: bool


_TRAINING_METHOD_CONTRACTS = {
    "OBS_BASE_TUNED": TrainingMethodContract("base_tuned", False, False),
    "OBS_LATE": TrainingMethodContract("late", True, False),
    "OBS_FUSE": TrainingMethodContract("fuse", True, False),
    "OBS_ATTN": TrainingMethodContract("attention", True, False),
    "OBS_FULL": TrainingMethodContract("full", True, True),
    "OBS_NO_FEEDBACK": TrainingMethodContract("no_feedback", True, True),
    "OBS_NO_CONSISTENCY": TrainingMethodContract("no_consistency", True, False),
}


def training_method_contract(method: str) -> TrainingMethodContract:
    try:
        return _TRAINING_METHOD_CONTRACTS[method]
    except KeyError as error:
        raise ObservationTrainingRunError(
            f"method is not an implemented joint-pair training variant: {method}"
        ) from error


def _load_json(path: str | Path, name: str) -> tuple[Path, dict[str, object]]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ObservationTrainingRunError(f"{name} is unavailable")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingRunError(f"{name} cannot be decoded") from error
    if not isinstance(payload, dict):
        raise ObservationTrainingRunError(f"{name} must be a JSON object")
    return source, payload


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _artifact_record(path: Path, relative_path: str) -> dict[str, object]:
    record = _file_record(path)
    record["path"] = relative_path
    return record


def _regular_file(path: str | Path, name: str) -> Path:
    result = Path(path).absolute()
    if result.is_symlink() or not result.is_file():
        raise ObservationTrainingRunError(f"{name} is unavailable")
    return result


def _regular_directory(path: str | Path, name: str) -> Path:
    result = Path(path).absolute()
    if result.is_symlink() or not result.is_dir():
        raise ObservationTrainingRunError(f"{name} is unavailable")
    return result


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ObservationTrainingRunError(f"{name} must be positive")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ObservationTrainingRunError(f"{name} must be a positive integer")
    return value


def _write_atomic_directory(output: Path, filename: str, payload: object) -> Path:
    if output.exists() or output.is_symlink():
        raise ObservationTrainingRunError(f"training run already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target = staging / filename
        with target.open("xb") as stream:
            stream.write(
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / filename


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_optimizer(
    *,
    model: nn.Module,
    criterion: nn.Module,
    method: str,
    native_learning_rate: float,
    observation_learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, dict[str, tuple[str, ...]]]:
    """Build disjoint optimizer groups without ever including the backbone."""

    if not isinstance(model, nn.Module) or not isinstance(criterion, nn.Module):
        raise TypeError("model and criterion must be torch modules")
    contract = training_method_contract(method)
    for name, value in (
        ("native_learning_rate", native_learning_rate),
        ("observation_learning_rate", observation_learning_rate),
        ("weight_decay", weight_decay),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ObservationTrainingRunError(f"{name} must be nonnegative")
    if native_learning_rate <= 0 or observation_learning_rate <= 0:
        raise ObservationTrainingRunError("optimizer learning rates must be positive")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("native.backbone.")
    ):
        raise ObservationTrainingRunError(
            "frozen backbone contains trainable parameters"
        )
    native = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("native.")
        and not name.startswith("native.backbone.")
        and parameter.requires_grad
    ]
    observation = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("obs_branch.") and parameter.requires_grad
    ]
    observation.extend(
        (f"criterion.{name}", parameter)
        for name, parameter in criterion.named_parameters()
        if parameter.requires_grad
    )
    if not native:
        raise ObservationTrainingRunError(
            "native decoder/head optimizer group is empty"
        )
    parameter_groups: list[dict[str, object]] = [
        {
            "params": [parameter for _, parameter in native],
            "lr": float(native_learning_rate),
            "weight_decay": float(weight_decay),
        }
    ]
    names: dict[str, tuple[str, ...]] = {
        "native": tuple(sorted(name for name, _ in native))
    }
    if contract.uses_region_supervision:
        if not observation:
            raise ObservationTrainingRunError("observation optimizer group is empty")
        parameter_groups.append(
            {
                "params": [parameter for _, parameter in observation],
                "lr": float(observation_learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
        names["observation"] = tuple(sorted(name for name, _ in observation))
    return torch.optim.AdamW(parameter_groups), names


def perform_accumulated_update(
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_factory: Callable[[int], torch.Tensor],
    gradient_accumulation: int,
    gradient_clip_norm: float,
) -> dict[str, float | bool]:
    """Execute one finite optimizer update from a fixed number of micro-steps."""

    if type(gradient_accumulation) is not int or gradient_accumulation <= 0:
        raise ObservationTrainingRunError("gradient_accumulation must be positive")
    if (
        isinstance(gradient_clip_norm, bool)
        or not isinstance(gradient_clip_norm, (int, float))
        or gradient_clip_norm <= 0
    ):
        raise ObservationTrainingRunError("gradient_clip_norm must be positive")
    optimizer.zero_grad(set_to_none=True)
    raw_losses: list[float] = []
    for micro_step in range(gradient_accumulation):
        loss = loss_factory(micro_step)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not torch.isfinite(loss)
        ):
            raise ObservationTrainingRunError("training loss must be a finite scalar")
        raw_losses.append(float(loss.detach().cpu()))
        (loss / gradient_accumulation).backward()
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, torch.Tensor) and parameter.requires_grad
    ]
    gradients = [
        parameter.grad for parameter in optimized if parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise ObservationTrainingRunError(
            "optimizer gradients are missing or non-finite"
        )
    norm = torch.nn.utils.clip_grad_norm_(
        optimized, float(gradient_clip_norm), error_if_nonfinite=True
    )
    group_norms = []
    for group in optimizer.param_groups:
        squared = sum(
            float(parameter.grad.detach().float().square().sum().cpu())
            for parameter in group["params"]
            if isinstance(parameter, torch.Tensor) and parameter.grad is not None
        )
        group_norms.append(squared**0.5)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("native.backbone.")
    ):
        raise ObservationTrainingRunError(
            "frozen backbone unexpectedly received gradients"
        )
    return {
        "mean_micro_loss": sum(raw_losses) / len(raw_losses),
        "gradient_norm_before_clip": float(norm.detach().cpu()),
        **{
            f"optimizer_group_{index}_gradient_norm": value
            for index, value in enumerate(group_norms)
        },
        "all_gradients_finite": True,
    }


def _build_native_model(
    *, runtime: Mapping[str, object], device: torch.device
) -> tuple[nn.Module, Path, Path, str]:
    import hydra
    from hydra import compose, initialize_config_dir

    checkouts = runtime.get("checkouts")
    assets = runtime.get("assets")
    if not isinstance(checkouts, Mapping) or not isinstance(assets, Mapping):
        raise ObservationTrainingRunError("runtime model paths are invalid")
    checkout = _regular_directory(checkouts.get("rescene", ""), "ReScene checkout")
    checkpoint = _regular_file(
        assets.get("rescene_checkpoint", ""), "ReScene checkpoint"
    )
    concerto = _regular_file(
        assets.get("concerto_checkpoint", ""), "Concerto checkpoint"
    )
    checkout_string = str(checkout)
    if checkout_string not in sys.path:
        sys.path.insert(0, checkout_string)
    with initialize_config_dir(version_base=None, config_dir=str(checkout / "conf")):
        native_config = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "general.train_mode=false",
                "general.train_on_segments=true",
                "general.eval_on_segments=true",
                "general.use_dbscan=false",
                "general.gpus=1",
                "general.seed=45",
                f"backbone.name={concerto}",
            ],
        )
    native = hydra.utils.instantiate(native_config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ObservationTrainingRunError("base checkpoint lacks a state_dict")
    excluded = {str(key) for key in state if not str(key).startswith("model.")}
    if excluded != {"criterion.empty_weight", "criterion.change_weights"}:
        raise ObservationTrainingRunError("base checkpoint non-model keys changed")
    model_state = {
        str(key)[len("model.") :]: value
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    if len(model_state) != 796:
        raise ObservationTrainingRunError("base checkpoint model tensor count changed")
    incompatible = native.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ObservationTrainingRunError("base checkpoint strict load failed")
    native.to(device)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return native, checkpoint, concerto, commit


def _build_point(
    model_input: object, device: torch.device
) -> tuple[object, list[torch.Tensor]]:
    from sonata.structure import Point

    coordinates = torch.from_numpy(
        np.array(model_input.coordinates_bxyzt, dtype="float32", copy=True, order="C")
    ).to(device)
    point = Point(
        {
            "coord": coordinates,
            "grid_coord": torch.from_numpy(
                np.array(
                    model_input.grid_coordinates_xyz,
                    dtype="int32",
                    copy=True,
                    order="C",
                )
            ).to(device),
            "feat": torch.from_numpy(
                np.array(model_input.features, dtype="float32", copy=True, order="C")
            ).to(device),
            "offset": torch.from_numpy(
                np.array(
                    model_input.sparse_batch_offsets,
                    dtype="int64",
                    copy=True,
                    order="C",
                )
            ).to(device),
        }
    )
    point2segment = [
        torch.from_numpy(
            np.array(model_input.point2segment, dtype="int64", copy=True, order="C")
        ).to(device)
    ]
    point.raw_coordinates = coordinates[:, 1:]
    point.point2segment = point2segment
    return point, point2segment


def _criterion_from_config(
    loss: Mapping[str, object], *, method: str, num_queries: int, device: torch.device
) -> nn.Module:
    from src.oviv2.observation_query.losses import ObservationSetCriterion

    contract = training_method_contract(method)
    criterion = ObservationSetCriterion(
        foreground_class_count=18,
        num_queries=num_queries,
        cost_class=float(loss.get("native_class_cost_and_weight", 2.0)),
        cost_mask=float(loss.get("native_mask_cost_and_weight", 5.0)),
        cost_dice=float(loss.get("native_dice_cost_and_weight", 2.0)),
        no_object_coefficient=float(loss.get("no_object_coefficient", 0.2)),
        native_weight=float(loss.get("native_weight", 1.0)),
        region_weight=(
            float(loss.get("region_soft_ce_weight", 0.2))
            if contract.uses_region_supervision
            else 0.0
        ),
        consistency_weight=(
            float(loss.get("consistency_weight_end", 0.1))
            if contract.uses_consistency
            else 0.0
        ),
        consistency_ramp_updates=int(loss.get("consistency_ramp_updates", 200)),
        require_region_supervision=contract.uses_region_supervision,
    ).to(device)
    if not contract.uses_consistency:
        criterion.obs_surface_null_head.requires_grad_(False)
    return criterion


def _configure_observation_trainables(
    model: nn.Module, contract: TrainingMethodContract
) -> None:
    prefixes = {
        "late": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_raw_beta",
        ),
        "fuse": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_fuse_projection.",
            "obs_raw_fuse",
        ),
        "attention": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_attention_token_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_cross_attention.",
            "obs_raw_alpha",
        ),
        "full": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_attention_token_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_cross_attention.",
            "obs_raw_alpha",
            "obs_raw_beta",
        ),
        "no_feedback": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_attention_token_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_cross_attention.",
            "obs_raw_alpha",
        ),
        "no_consistency": (
            "obs_feature_projection.",
            "obs_metadata_projection.",
            "obs_region_norm.",
            "obs_query_norm.",
            "obs_attention_token_norm.",
            "obs_query_projection.",
            "obs_region_projection.",
            "obs_dustbin_head.",
            "obs_cross_attention.",
            "obs_raw_alpha",
            "obs_raw_beta",
        ),
    }.get(contract.observation_mode, ())
    for name, parameter in model.obs_branch.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))


def _resolved_training_config(
    *,
    config: Mapping[str, object],
    method: str,
    stage: str,
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    return {
        "method": method,
        "stage": stage,
        "model": config.get("model"),
        "loss": config.get("loss"),
        "training": config.get("training"),
        "method_config": config.get("methods", {}).get(method),
        "source_bindings": {
            name: _file_record(path) for name, path in sorted(source_paths.items())
        },
    }


def _metadata_from_manifest(path: Path):
    from src.oviv2.observation_query.training import ObservationCheckpointMetadata

    _, payload = _load_json(path, "resume checkpoint manifest")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ObservationTrainingRunError("resume checkpoint metadata is invalid")
    return ObservationCheckpointMetadata(**dict(metadata))


def _same_resume_contract(current: object, previous: object) -> bool:
    def contract(metadata: object) -> dict[str, object]:
        values = metadata.as_dict()
        resolved = values.get("resolved_config")
        if not isinstance(resolved, Mapping):
            return {}
        training = resolved.get("training")
        bindings = resolved.get("source_bindings")
        if not isinstance(training, Mapping) or not isinstance(bindings, Mapping):
            return {}
        semantic_bindings = {
            name: {
                key: record.get(key)
                for key in ("sha256", "byte_count")
                if key in record
            }
            for name in ("model", "losses", "training_state")
            if isinstance((record := bindings.get(name)), Mapping)
        }
        dataset = values.get("training_dataset_manifest")
        if dataset is None:
            dataset = {
                "split_id": values.get("split_id"),
                "observation_sha256": values.get("observation_sha256"),
                "backbone_cache_sha256": values.get("backbone_cache_sha256"),
            }
        return {
            "model_variant": values.get("model_variant"),
            "base_checkpoint_sha256": values.get("base_checkpoint_sha256"),
            "model_architecture_version": values.get("model_architecture_version"),
            "input_feature_schema": values.get("input_feature_schema"),
            "model": resolved.get("model"),
            "loss": resolved.get("loss"),
            "method_config": resolved.get("method_config"),
            "optimizer": {
                name: training.get(name)
                for name in (
                    "optimizer",
                    "native_learning_rate",
                    "observation_learning_rate",
                    "weight_decay",
                    "gradient_clip_norm",
                    "gradient_accumulation",
                )
            },
            "seed": values.get("seed"),
            "training_dataset_manifest": dataset,
            "semantic_source_bindings": semantic_bindings,
        }

    return contract(current) == contract(previous)


def _target_update_count(
    *,
    target_total_updates: int,
    start_update: int,
    maximum_total_updates: int | None = None,
) -> int:
    target = _positive_integer(target_total_updates, "target_total_updates")
    if type(start_update) is not int or start_update < 0:
        raise ObservationTrainingRunError("start_update must be nonnegative")
    if maximum_total_updates is not None:
        maximum = _positive_integer(
            maximum_total_updates, "maximum_total_updates"
        )
        if target > maximum:
            raise ObservationTrainingRunError(
                "target_total_updates exceeds the configured maximum"
            )
    if target <= start_update:
        raise ObservationTrainingRunError(
            "target_total_updates must exceed the resumed optimizer updates"
        )
    return target - start_update


def _restore_training_random_state(
    state: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    required = (
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
    )
    if any(name not in state for name in required):
        raise ObservationTrainingRunError("resume random state is incomplete")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if device.type == "cuda":
        cuda_state = state.get("cuda_random_state")
        if not isinstance(cuda_state, torch.Tensor):
            raise ObservationTrainingRunError("resume CUDA random state is incomplete")
        torch.cuda.set_rng_state(cuda_state, device)
    sampler = state.get("sampler_state", {})
    if not isinstance(sampler, Mapping):
        raise ObservationTrainingRunError("resume sampler state is invalid")
    return dict(sampler)


def _validated_resume_state_path(resume_root: Path) -> Path:
    training_state = _regular_file(
        resume_root / "training_state.pt", "resume optimizer state"
    )
    summary_path = resume_root / "summary.json"
    snapshot_path = resume_root / "snapshot.json"
    if summary_path.is_file() and not summary_path.is_symlink():
        _, evidence = _load_json(summary_path, "resume summary")
        if evidence.get("artifact_id") != "OVI_RESCENE_OBSERVATION_TRAINING_RUN_V1":
            raise ObservationTrainingRunError("resume summary identity mismatch")
    elif snapshot_path.is_file() and not snapshot_path.is_symlink():
        _, evidence = _load_json(snapshot_path, "periodic snapshot")
        if (
            evidence.get("schema_version") != 2
            or evidence.get("artifact_id")
            != "OVI_RESCENE_OBSERVATION_TRAINING_SNAPSHOT_V2"
            or evidence.get("status") != "PASS"
        ):
            raise ObservationTrainingRunError("periodic snapshot identity mismatch")
    else:
        raise ObservationTrainingRunError("resume state evidence is unavailable")
    if evidence.get("training_state") != _artifact_record(
        training_state, "training_state.pt"
    ):
        raise ObservationTrainingRunError("resume optimizer state binding mismatch")
    return training_state


def _execute_training(
    *,
    config_path: Path,
    config: Mapping[str, object],
    runtime_path: Path,
    runtime: Mapping[str, object],
    split_path: Path,
    split: Mapping[str, object],
    pair_runtime: Mapping[str, object],
    method: str,
    run_id: str,
    stage: str,
    resume: str | None,
    target_total_updates: int | None,
    output_root: Path,
) -> int:
    contract = training_method_contract(method)
    python_config = runtime.get("python")
    expected_python = (
        Path(str(python_config.get("model", ""))).absolute()
        if isinstance(python_config, Mapping)
        else Path()
    )
    if not expected_python.is_file() or not os.path.samefile(
        sys.executable, expected_python
    ):
        raise ObservationTrainingRunError("training must use the frozen model Python")
    artifact_root = _regular_directory(
        str(pair_runtime["artifact_root"]), "TRAIN artifact root"
    )
    training = config.get("training")
    loss_config = config.get("loss")
    model_config = config.get("model")
    if not all(
        isinstance(value, Mapping) for value in (training, loss_config, model_config)
    ):
        raise ObservationTrainingRunError("model/loss/training config is invalid")
    assert isinstance(training, Mapping)
    assert isinstance(loss_config, Mapping)
    assert isinstance(model_config, Mapping)
    configured_target = (
        _positive_integer(training.get("smoke_updates"), "smoke_updates")
        if stage == "smoke"
        else _positive_integer(
            training.get("pilot_updates_per_method"), "pilot_updates_per_method"
        )
    )
    accumulation = _positive_integer(
        training.get("gradient_accumulation"), "gradient_accumulation"
    )
    clip_norm = _positive_number(
        training.get("gradient_clip_norm"), "gradient_clip_norm"
    )
    seed = _positive_integer(training.get("seed"), "seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device_name = os.environ.get("RESCENE_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ObservationTrainingRunError("training requires one explicit CUDA device")
    torch.cuda.set_device(device)

    from scripts.evaluation.prepare_ovi_observations import (
        load_model_observation_bundle,
    )
    from src.oviv2.observation_query.contracts import load_observation_bank
    from src.oviv2.observation_query.model import (
        FrozenReSceneFeatures,
        ObservationReScene,
    )
    from src.oviv2.observation_query.training import (
        BackboneCacheIdentity,
        ObservationCheckpointMetadata,
        capture_backbone_cache,
        load_backbone_cache,
        load_trainable_checkpoint,
        materialize_backbone_cache,
        save_backbone_cache,
        save_trainable_checkpoint,
    )
    from src.training.ovi_observation_data import load_observation_training_sample

    model_input, _points, _center, _bundle = load_model_observation_bundle(
        artifact_root / "model_bundle"
    )
    observations = load_observation_bank(artifact_root / "observation_bank" / "bank")
    sample = load_observation_training_sample(
        artifact_root / "training_targets",
        model_input=model_input,
        observations=observations,
    )
    native, base_checkpoint, _concerto, rescene_commit = _build_native_model(
        runtime=runtime, device=device
    )
    model = ObservationReScene(
        native,
        observation_feature_dim=int(observations.region_features.shape[1]),
        metadata_dim=int(observations.region_metadata.shape[1]),
        alpha_initial=float(model_config.get("alpha_initial", 0.001)),
        beta_initial=float(model_config.get("beta_initial", 0.0)),
        fuse_initial=float(model_config.get("fuse_initial", 0.001)),
    ).to(device)
    _configure_observation_trainables(model, contract)
    criterion = _criterion_from_config(
        loss_config, method=method, num_queries=int(native.num_queries), device=device
    )
    optimizer, optimizer_names = build_optimizer(
        model=model,
        criterion=criterion,
        method=method,
        native_learning_rate=_positive_number(
            training.get("native_learning_rate"), "native_learning_rate"
        ),
        observation_learning_rate=_positive_number(
            training.get("observation_learning_rate"),
            "observation_learning_rate",
        ),
        weight_decay=_positive_number(training.get("weight_decay"), "weight_decay"),
    )
    identity = BackboneCacheIdentity(
        pair_id=observations.pair_id,
        base_checkpoint_sha256=_file_record(base_checkpoint)["sha256"],
        model_input_sha256=model_input.content_sha256(),
        serialization_id=str(
            model_config.get(
                "backbone_serialization", "mixed_standard_temporal_overlay"
            )
        ),
        visit_order=tuple(
            int(value) for value in sorted(set(model_input.model_visit_ids.tolist()))
        ),
    )
    cache_root = artifact_root / "frozen_backbone_cache"
    if cache_root.exists():
        cache = load_backbone_cache(cache_root, expected_identity=identity)
        cache_generated = False
    else:
        point, point2segment = _build_point(model_input, device)
        with torch.no_grad():
            pcd, auxiliary, coordinates = model.native.backbone(point)
        cache = capture_backbone_cache(
            FrozenReSceneFeatures(pcd, auxiliary, coordinates), identity
        )
        save_backbone_cache(cache, cache_root)
        cache_generated = True
    _point, point2segment = _build_point(model_input, device)
    del _point
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_paths = {
        "runner": Path(__file__).absolute(),
        "model": Path(__file__).resolve().parents[2]
        / "src"
        / "oviv2"
        / "observation_query"
        / "model.py",
        "losses": Path(__file__).resolve().parents[2]
        / "src"
        / "oviv2"
        / "observation_query"
        / "losses.py",
        "training_state": Path(__file__).resolve().parents[2]
        / "src"
        / "oviv2"
        / "observation_query"
        / "training.py",
        "config": config_path,
        "runtime": runtime_path,
        "split": split_path,
    }
    resolved = _resolved_training_config(
        config=config,
        method=method,
        stage=stage,
        source_paths=source_paths,
    )
    initial_metadata = ObservationCheckpointMetadata(
        model_variant=method,
        base_checkpoint_sha256=identity.base_checkpoint_sha256,
        source_commit=source_commit,
        resolved_config=resolved,
        split_id=str(split.get("artifact_id", "splits_v1")),
        observation_sha256=observations.content_sha256(),
        backbone_cache_sha256=cache.content_sha256,
        seed=seed,
        optimizer_updates=0,
        training_dataset_manifest={
            "schema_version": 2,
            "artifact_id": "OVI_OBSERVATION_TRAINING_DATASET_V2",
            "environment_ids": [
                str(_select_train_pair(split, observations.pair_id)["environment_uuid"])
            ],
            "pairs": [
                {
                    "pair_id": observations.pair_id,
                    "model_input_sha256": model_input.content_sha256(),
                    "observation_sha256": observations.content_sha256(),
                    "training_target_sha256": sample.content_sha256(),
                    "backbone_cache_sha256": cache.content_sha256,
                }
            ],
        },
        model_architecture_version="OVI_OBSERVATION_QUERY_V1",
        input_feature_schema={
            "observation_feature_dim": int(observations.region_features.shape[1]),
            "observation_metadata_dim": int(observations.region_metadata.shape[1]),
            "model_input_feature_dim": int(model_input.features.shape[1]),
        },
    )
    start_update = 0
    if resume is not None:
        resume_root = _regular_directory(resume, "resume run")
        previous_metadata = _metadata_from_manifest(
            resume_root / "checkpoint" / "manifest.json"
        )
        if not _same_resume_contract(initial_metadata, previous_metadata):
            raise ObservationTrainingRunError("resume checkpoint contract mismatch")
        load_trainable_checkpoint(
            model=model,
            criterion=criterion,
            checkpoint_root=resume_root / "checkpoint",
            expected_metadata=previous_metadata,
        )
        training_state_path = _validated_resume_state_path(resume_root)
        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping) or "optimizer" not in state:
            raise ObservationTrainingRunError("resume optimizer state is invalid")
        optimizer.load_state_dict(state["optimizer"])
        start_update = previous_metadata.optimizer_updates
        sampler_state = _restore_training_random_state(state, device)
    else:
        sampler_state = {
            "pair_order": [observations.pair_id],
            "next_pair_index": 0,
        }
    target_updates = (
        configured_target
        if target_total_updates is None
        else _positive_integer(target_total_updates, "target_total_updates")
    )
    update_count = _target_update_count(
        target_total_updates=target_updates,
        start_update=start_update,
        maximum_total_updates=_positive_integer(
            training.get("cumulative_max_optimizer_updates"),
            "cumulative_max_optimizer_updates",
        ),
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise ObservationTrainingRunError(f"training run already exists: {output_root}")
    output_root.mkdir()
    status_path = output_root / "status.json"
    _write_json_atomic(
        status_path,
        {
            "schema_version": 2,
            "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_STATUS_V2",
            "status": "RUNNING",
            "run_id": run_id,
            "method": method,
            "stage": stage,
            "pair_id": observations.pair_id,
            "start_optimizer_updates": start_update,
            "target_total_updates": target_updates,
            "resume": resume,
        },
    )
    curve_path = output_root / "curve.csv"

    def save_snapshot(total_updates: int, root: Path) -> tuple[object, Path]:
        metadata = ObservationCheckpointMetadata(
            **{
                **initial_metadata.as_dict(),
                "optimizer_updates": total_updates,
            }
        )
        checkpoint = save_trainable_checkpoint(
            model=model,
            criterion=criterion,
            output_root=root / "checkpoint",
            metadata=metadata,
        )
        training_state = root / "training_state.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state(device),
                "sampler_state": {
                    **sampler_state,
                    "completed_updates": total_updates,
                },
            },
            training_state,
        )
        _write_json_atomic(
            root / "snapshot.json",
            {
                "schema_version": 2,
                "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_SNAPSHOT_V2",
                "status": "PASS",
                "optimizer_updates": total_updates,
                "checkpoint": {
                    "manifest": _artifact_record(
                        checkpoint.manifest, "checkpoint/manifest.json"
                    ),
                    "weights": _artifact_record(
                        checkpoint.weights, "checkpoint/trainable.safetensors"
                    ),
                },
                "training_state": _artifact_record(
                    training_state, "training_state.pt"
                ),
            },
        )
        return checkpoint, training_state

    model.train()
    criterion.train()
    curve: list[dict[str, object]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for local_update in range(update_count):
        update = start_update + local_update
        latest: list[object] = []

        def loss_factory(
            _micro_step: int,
            current_update: int = update,
            sink: list[object] = latest,
        ) -> torch.Tensor:
            frozen = materialize_backbone_cache(
                cache, model.native.backbone, device=device
            )
            outputs = model.forward_from_backbone(
                frozen,
                point2segment=point2segment,
                is_eval=False,
                observations=observations if contract.uses_region_supervision else None,
                observation_mode=contract.observation_mode,
            )
            result = criterion(outputs, sample, optimizer_update=current_update)
            sink[:] = [result]
            return result.total

        update_stats = perform_accumulated_update(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loss_factory=loss_factory,
            gradient_accumulation=accumulation,
            gradient_clip_norm=clip_norm,
        )
        result = latest[0]
        curve.append(
            row := {
                "optimizer_update": update + 1,
                "total": float(result.total.detach().cpu()),
                "loss_ce": float(result.components["loss_ce"].detach().cpu()),
                "loss_mask": float(result.components["loss_mask"].detach().cpu()),
                "loss_dice": float(result.components["loss_dice"].detach().cpu()),
                "loss_region": float(result.components["loss_region"].detach().cpu()),
                "loss_consistency": float(
                    result.components["loss_consistency"].detach().cpu()
                ),
                **update_stats,
            }
        )
        with curve_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if (update + 1) % 10 == 0:
                os.fsync(stream.fileno())
        if update + 1 in {200, 500, 1000}:
            save_snapshot(
                update + 1,
                output_root / "checkpoints" / f"update-{update + 1:06d}",
            )
    torch.cuda.synchronize(device)
    training_s = time.perf_counter() - started
    if not curve or max(row["optimizer_group_0_gradient_norm"] for row in curve) <= 0:
        raise ObservationTrainingRunError(
            "native decoder/head did not receive gradients"
        )
    if (
        contract.uses_region_supervision
        and max(row.get("optimizer_group_1_gradient_norm", 0.0) for row in curve) <= 0
    ):
        raise ObservationTrainingRunError(
            "observation branch did not receive gradients"
        )

    staging = output_root
    try:
        final_metadata = ObservationCheckpointMetadata(
            **{**initial_metadata.as_dict(), "optimizer_updates": target_updates}
        )
        checkpoint_paths, training_state_path = save_snapshot(
            target_updates, staging
        )

        model.eval()
        criterion.eval()
        with torch.no_grad():
            saved_output = model.forward_from_backbone(
                materialize_backbone_cache(cache, model.native.backbone, device=device),
                point2segment=point2segment,
                is_eval=True,
                observations=observations if contract.uses_region_supervision else None,
                observation_mode=contract.observation_mode,
            )
        reloaded_native, _base, _concerto, _commit = _build_native_model(
            runtime=runtime, device=device
        )
        reloaded_model = ObservationReScene(
            reloaded_native,
            observation_feature_dim=int(observations.region_features.shape[1]),
            metadata_dim=int(observations.region_metadata.shape[1]),
            alpha_initial=float(model_config.get("alpha_initial", 0.001)),
            beta_initial=float(model_config.get("beta_initial", 0.0)),
            fuse_initial=float(model_config.get("fuse_initial", 0.001)),
        ).to(device)
        _configure_observation_trainables(reloaded_model, contract)
        reloaded_criterion = _criterion_from_config(
            loss_config,
            method=method,
            num_queries=int(reloaded_native.num_queries),
            device=device,
        )
        load_trainable_checkpoint(
            model=reloaded_model,
            criterion=reloaded_criterion,
            checkpoint_root=checkpoint_paths.root,
            expected_metadata=final_metadata,
        )
        reloaded_model.eval()
        with torch.no_grad():
            reloaded_output = reloaded_model.forward_from_backbone(
                materialize_backbone_cache(
                    cache, reloaded_model.native.backbone, device=device
                ),
                point2segment=point2segment,
                is_eval=True,
                observations=observations if contract.uses_region_supervision else None,
                observation_mode=contract.observation_mode,
            )
        replay_differences = {
            "pred_masks": float(
                (saved_output["pred_masks"][0] - reloaded_output["pred_masks"][0])
                .abs()
                .max()
                .detach()
                .cpu()
            ),
            "pred_logits": float(
                (saved_output["pred_logits"] - reloaded_output["pred_logits"])
                .abs()
                .max()
                .detach()
                .cpu()
            ),
        }
        replay_pairs = [
            (saved_output["pred_masks"][0], reloaded_output["pred_masks"][0]),
            (saved_output["pred_logits"], reloaded_output["pred_logits"]),
        ]
        if contract.uses_region_supervision:
            replay_differences["pred_region_logits"] = float(
                (
                    saved_output["pred_region_logits"]
                    - reloaded_output["pred_region_logits"]
                )
                .abs()
                .max()
                .detach()
                .cpu()
            )
            replay_pairs.append(
                (
                    saved_output["pred_region_logits"],
                    reloaded_output["pred_region_logits"],
                )
            )
        replay_rtol = 1e-5
        replay_atol = 1e-6
        if not all(
            torch.allclose(left, right, rtol=replay_rtol, atol=replay_atol)
            for left, right in replay_pairs
        ):
            raise ObservationTrainingRunError("checkpoint reload changed model outputs")
        summary = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_V1",
            "status": "REAL_TRAIN_SMOKE_PASS" if stage == "smoke" else "TRAIN_RUN_PASS",
            "run_id": run_id,
            "pair_id": observations.pair_id,
            "role": "TRAIN",
            "method": method,
            "stage": stage,
            "optimizer_updates": target_updates,
            "new_optimizer_updates": update_count,
            "gradient_accumulation": accumulation,
            "first_total_loss": curve[0]["total"],
            "final_total_loss": curve[-1]["total"],
            "first_mask_loss": curve[0]["loss_mask"],
            "final_mask_loss": curve[-1]["loss_mask"],
            "optimizer_parameter_names": optimizer_names,
            "backbone_cache": {
                "path": str(cache_root),
                "content_sha256": cache.content_sha256,
                "generated_by_this_run": cache_generated,
            },
            "checkpoint": {
                "manifest": _artifact_record(
                    checkpoint_paths.manifest, "checkpoint/manifest.json"
                ),
                "weights": _artifact_record(
                    checkpoint_paths.weights, "checkpoint/trainable.safetensors"
                ),
            },
            "training_state": _artifact_record(
                training_state_path, "training_state.pt"
            ),
            "curve": _artifact_record(curve_path, "curve.csv"),
            "checkpoint_reload_maximum_absolute_differences": replay_differences,
            "checkpoint_reload_tolerance": {
                "relative": replay_rtol,
                "absolute": replay_atol,
            },
            "base_rescene_commit": rescene_commit,
            "training_runtime_s": training_s,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
            "device": f"{torch.cuda.get_device_name(device)} {device}",
        }
        _write_json_atomic(staging / "summary.json", summary)
        _write_json_atomic(
            status_path,
            {
                "schema_version": 2,
                "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_STATUS_V2",
                "status": "PASS",
                "run_id": run_id,
                "method": method,
                "stage": stage,
                "pair_id": observations.pair_id,
                "optimizer_updates": target_updates,
            },
        )
    except BaseException as error:
        _write_json_atomic(
            status_path,
            {
                "schema_version": 2,
                "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_STATUS_V2",
                "status": "FAILED",
                "run_id": run_id,
                "method": method,
                "stage": stage,
                "optimizer_updates": target_updates,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(json.dumps(summary, sort_keys=True))
    return 0


def _select_train_pair(
    split: Mapping[str, object], pair_id: str
) -> Mapping[str, object]:
    environments = split.get("environments")
    if not isinstance(environments, list):
        raise ObservationTrainingRunError("split environments are invalid")
    matches = [
        item
        for item in environments
        if isinstance(item, Mapping)
        and item.get("role") == "TRAIN"
        and item.get("pair_id") == pair_id
    ]
    if len(matches) != 1:
        raise ObservationTrainingRunError("TRAIN pair is absent or ambiguous")
    return matches[0]


def _missing_sequence_zips(pair: Mapping[str, object]) -> list[str]:
    sessions = pair.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ObservationTrainingRunError("TRAIN pair sessions are invalid")
    missing: list[str] = []
    for session in sessions:
        if not isinstance(session, Mapping):
            raise ObservationTrainingRunError("TRAIN session is invalid")
        sequence = session.get("sequence_zip")
        if not isinstance(sequence, Mapping) or not isinstance(
            sequence.get("path"), str
        ):
            raise ObservationTrainingRunError(
                "TRAIN sequence.zip declaration is invalid"
            )
        path = Path(str(sequence["path"])).absolute()
        if path.is_symlink() or not path.is_file():
            missing.append(str(path))
    return missing


def _asset_gate(
    *,
    config_path: Path,
    runtime_path: Path,
    split_path: Path,
    pair: Mapping[str, object],
    pair_runtime: Mapping[str, object],
    run_id: str,
    method: str,
    stage: str,
    output_root: Path,
) -> int | None:
    artifact_root_value = pair_runtime.get("artifact_root")
    if not isinstance(artifact_root_value, str) or not artifact_root_value:
        raise ObservationTrainingRunError("TRAIN artifact_root is missing")
    artifact_root = Path(artifact_root_value).absolute()
    required = {
        "model_bundle_manifest": artifact_root / "model_bundle" / "manifest.json",
        "observation_bank_manifest": artifact_root
        / "observation_bank"
        / "bank"
        / "manifest.json",
        "training_target_manifest": artifact_root
        / "training_targets"
        / "manifest.json",
    }
    missing_artifacts = {
        name: str(path)
        for name, path in required.items()
        if path.is_symlink() or not path.is_file()
    }
    missing_zips = _missing_sequence_zips(pair)
    if not missing_artifacts and not missing_zips:
        return None
    status = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_STATUS_V1",
        "status": "TRAINING_ASSET_GATED",
        "run_id": run_id,
        "method": method,
        "stage": stage,
        "pair_id": pair.get("pair_id"),
        "role": "TRAIN",
        "missing_train_sequence_zip_count": len(missing_zips),
        "missing_train_sequence_zips": missing_zips,
        "missing_training_artifacts": missing_artifacts,
        "validation_substitution_allowed": False,
        "checkpoint_created": False,
        "source_bindings": {
            "config": _file_record(config_path),
            "runtime": _file_record(runtime_path),
            "split": _file_record(split_path),
        },
        "required_action": (
            "Materialize the declared official TRAIN RGB-D assets only after explicit "
            "3RScan Terms-of-Use confirmation, then build the TRAIN observation artifacts."
        ),
    }
    _write_atomic_directory(output_root, "status.json", status)
    print(json.dumps(status, sort_keys=True))
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "pilot", "mechanism", "confirmation-fit"),
    )
    parser.add_argument("--resume")
    parser.add_argument("--target-total-updates", type=int)
    parser.add_argument("--train-pair-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path, config = _load_json(args.config, "experiment config")
    runtime_path, runtime = _load_json(args.runtime, "runtime config")
    methods = config.get("methods")
    if not isinstance(methods, Mapping) or args.method not in methods:
        raise ObservationTrainingRunError(
            "method is not declared by the experiment config"
        )
    training = config.get("training")
    if not isinstance(training, Mapping) or training.get("optimizer") != "AdamW":
        raise ObservationTrainingRunError("training config must select AdamW")
    split_value = config.get("split_manifest")
    if not isinstance(split_value, str) or not split_value:
        raise ObservationTrainingRunError("split_manifest is missing")
    split_path, split = _load_json(split_value, "split manifest")
    runtime_pairs = runtime.get("pairs")
    pair_runtime = None
    if isinstance(runtime_pairs, Mapping):
        matches = [
            value
            for key, value in runtime_pairs.items()
            if isinstance(key, str)
            and isinstance(value, Mapping)
            and value.get("role", key.upper()) == "TRAIN"
            and (
                args.train_pair_id is None
                or value.get("pair_id") == args.train_pair_id
            )
        ]
        if len(matches) == 1:
            pair_runtime = matches[0]
    if not isinstance(pair_runtime, Mapping) or not isinstance(
        pair_runtime.get("pair_id"), str
    ):
        raise ObservationTrainingRunError("runtime TRAIN pair is invalid")
    pair_id = str(pair_runtime["pair_id"])
    pair = _select_train_pair(split, pair_id)
    cache_root = runtime.get("cache_root")
    if not isinstance(cache_root, str) or not cache_root:
        raise ObservationTrainingRunError("runtime cache_root is invalid")
    output_root = Path(cache_root).absolute() / "training_runs" / args.run_id
    gate = _asset_gate(
        config_path=config_path,
        runtime_path=runtime_path,
        split_path=split_path,
        pair=pair,
        pair_runtime=pair_runtime,
        run_id=args.run_id,
        method=args.method,
        stage=args.stage,
        output_root=output_root,
    )
    if gate is not None:
        return gate
    try:
        return _execute_training(
            config_path=config_path,
            config=config,
            runtime_path=runtime_path,
            runtime=runtime,
            split_path=split_path,
            split=split,
            pair_runtime=pair_runtime,
            method=args.method,
            run_id=args.run_id,
            stage=args.stage,
            resume=args.resume,
            target_total_updates=args.target_total_updates,
            output_root=output_root,
        )
    except BaseException as error:
        if output_root.is_dir():
            _write_json_atomic(
                output_root / "status.json",
                {
                    "schema_version": 2,
                    "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_RUN_STATUS_V2",
                    "status": "FAILED",
                    "run_id": args.run_id,
                    "method": args.method,
                    "stage": args.stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
