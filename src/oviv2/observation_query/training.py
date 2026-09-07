"""Hash-bound frozen-backbone caches and compact trainable checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from src.oviv2.observation_query.model import FrozenReSceneFeatures

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_LEVEL_FIELDS = ("feat", "coord", "grid_coord", "batch")


class ObservationTrainingStateError(ValueError):
    """Raised when cached features or trained state lose their exact identity."""


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObservationTrainingStateError(f"{name} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationTrainingStateError(f"{name} must be a non-empty string")
    return value


def _canonical(value: object, name: str) -> tuple[bytes, object]:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        copied = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ObservationTrainingStateError(f"{name} is not canonical JSON data") from error
    return encoded, copied


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _readonly_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or array.dtype.hasobject:
        raise ObservationTrainingStateError(f"{name} must be a non-object tensor array")
    if np.issubdtype(array.dtype, np.floating) and np.any(~np.isfinite(array)):
        raise ObservationTrainingStateError(f"{name} must be finite")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class BackboneCacheIdentity:
    pair_id: str
    base_checkpoint_sha256: str
    model_input_sha256: str
    serialization_id: str
    visit_order: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _nonempty(self.pair_id, "pair_id"))
        object.__setattr__(
            self,
            "base_checkpoint_sha256",
            _sha256(self.base_checkpoint_sha256, "base checkpoint"),
        )
        object.__setattr__(
            self,
            "model_input_sha256",
            _sha256(self.model_input_sha256, "model input"),
        )
        object.__setattr__(
            self,
            "serialization_id",
            _nonempty(self.serialization_id, "serialization_id"),
        )
        if (
            not isinstance(self.visit_order, tuple)
            or not self.visit_order
            or any(type(value) is not int or value < 0 for value in self.visit_order)
            or len(set(self.visit_order)) != len(self.visit_order)
        ):
            raise ObservationTrainingStateError("visit_order must contain unique visit IDs")

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "model_input_sha256": self.model_input_sha256,
            "serialization_id": self.serialization_id,
            "visit_order": list(self.visit_order),
        }


@dataclass(frozen=True, slots=True)
class FrozenBackboneCache:
    identity: BackboneCacheIdentity
    pcd_level: Mapping[str, np.ndarray]
    auxiliary_levels: tuple[Mapping[str, np.ndarray], ...]
    coordinates: tuple[tuple[np.ndarray, ...], ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class BackboneCachePaths:
    root: Path
    manifest: Path
    arrays: Path


def _capture_level(value: object, name: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field in _LEVEL_FIELDS:
        tensor = getattr(value, field, None)
        if not isinstance(tensor, torch.Tensor):
            raise ObservationTrainingStateError(f"{name}.{field} must be a tensor")
        result[field] = _readonly_array(
            tensor.detach().cpu().contiguous().numpy(), f"{name}.{field}"
        )
    row_count = len(result["feat"])
    if row_count == 0 or any(len(result[field]) != row_count for field in _LEVEL_FIELDS):
        raise ObservationTrainingStateError(f"{name} tensor rows do not align")
    return result


def _cache_arrays(
    pcd_level: Mapping[str, np.ndarray],
    auxiliary_levels: tuple[Mapping[str, np.ndarray], ...],
    coordinates: tuple[tuple[np.ndarray, ...], ...],
) -> dict[str, np.ndarray]:
    arrays = {f"pcd__{field}": pcd_level[field] for field in _LEVEL_FIELDS}
    for level_index, level in enumerate(auxiliary_levels):
        for field in _LEVEL_FIELDS:
            arrays[f"aux_{level_index:03d}__{field}"] = level[field]
    for level_index, batches in enumerate(coordinates):
        for batch_index, value in enumerate(batches):
            arrays[f"coord_{level_index:03d}_{batch_index:03d}"] = value
    return arrays


def _cache_content(
    identity: BackboneCacheIdentity,
    arrays: Mapping[str, np.ndarray],
    auxiliary_count: int,
    coordinate_batch_counts: list[int],
) -> str:
    payload = {
        "identity": identity.as_dict(),
        "auxiliary_count": auxiliary_count,
        "coordinate_batch_counts": coordinate_batch_counts,
        "array_records": {
            name: _array_record(value) for name, value in sorted(arrays.items())
        },
    }
    return hashlib.sha256(_canonical(payload, "backbone cache")[0]).hexdigest()


def capture_backbone_cache(
    frozen: FrozenReSceneFeatures, identity: BackboneCacheIdentity
) -> FrozenBackboneCache:
    if not isinstance(frozen, FrozenReSceneFeatures):
        raise TypeError("frozen must be FrozenReSceneFeatures")
    if not isinstance(identity, BackboneCacheIdentity):
        raise TypeError("identity must be BackboneCacheIdentity")
    pcd = _capture_level(frozen.pcd_features, "pcd_features")
    try:
        auxiliary = tuple(
            _capture_level(value, f"auxiliary_features[{index}]")
            for index, value in enumerate(frozen.auxiliary_features)
        )
        coordinates = tuple(
            tuple(
                _readonly_array(
                    value.detach().cpu().contiguous().numpy()
                    if isinstance(value, torch.Tensor)
                    else value,
                    f"coordinates[{level_index}][{batch_index}]",
                )
                for batch_index, value in enumerate(level)
            )
            for level_index, level in enumerate(frozen.coordinates)
        )
    except TypeError as error:
        raise ObservationTrainingStateError("frozen backbone levels are not iterable") from error
    if not auxiliary or not coordinates or len(auxiliary) != len(coordinates):
        raise ObservationTrainingStateError("frozen auxiliary and coordinate levels differ")
    if any(not level for level in coordinates):
        raise ObservationTrainingStateError("every coordinate level needs a batch")
    arrays = _cache_arrays(pcd, auxiliary, coordinates)
    content = _cache_content(
        identity, arrays, len(auxiliary), [len(value) for value in coordinates]
    )
    return FrozenBackboneCache(identity, pcd, auxiliary, coordinates, content)


def save_backbone_cache(
    cache: FrozenBackboneCache, output_root: str | Path
) -> BackboneCachePaths:
    if not isinstance(cache, FrozenBackboneCache):
        raise TypeError("cache must be FrozenBackboneCache")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationTrainingStateError(f"backbone cache already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays = _cache_arrays(
            cache.pcd_level, cache.auxiliary_levels, cache.coordinates
        )
        expected_content = _cache_content(
            cache.identity,
            arrays,
            len(cache.auxiliary_levels),
            [len(value) for value in cache.coordinates],
        )
        if expected_content != cache.content_sha256:
            raise ObservationTrainingStateError("backbone cache content identity changed")
        arrays_path = staging / "features.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_FROZEN_BACKBONE_CACHE_V1",
            "status": "PASS",
            "identity": cache.identity.as_dict(),
            "auxiliary_count": len(cache.auxiliary_levels),
            "coordinate_batch_counts": [len(value) for value in cache.coordinates],
            "array_records": {
                name: _array_record(value) for name, value in sorted(arrays.items())
            },
            "content_sha256": cache.content_sha256,
            "arrays": _file_record(arrays_path),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BackboneCachePaths(output, output / "manifest.json", output / "features.npz")


def load_backbone_cache(
    output_root: str | Path, *, expected_identity: BackboneCacheIdentity
) -> FrozenBackboneCache:
    if not isinstance(expected_identity, BackboneCacheIdentity):
        raise TypeError("expected_identity must be BackboneCacheIdentity")
    root = Path(output_root).absolute()
    manifest_path = root / "manifest.json"
    arrays_path = root / "features.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingStateError("backbone cache manifest is unavailable") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "identity",
        "auxiliary_count",
        "coordinate_batch_counts",
        "array_records",
        "content_sha256",
        "arrays",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_RESCENE_FROZEN_BACKBONE_CACHE_V1"
        or manifest.get("status") != "PASS"
    ):
        raise ObservationTrainingStateError("backbone cache manifest is invalid")
    if manifest.get("identity") != expected_identity.as_dict():
        raise ObservationTrainingStateError("backbone cache identity mismatch")
    if manifest.get("arrays") != _file_record(arrays_path):
        raise ObservationTrainingStateError("backbone cache array binding mismatch")
    auxiliary_count = manifest.get("auxiliary_count")
    coordinate_counts = manifest.get("coordinate_batch_counts")
    if (
        type(auxiliary_count) is not int
        or auxiliary_count <= 0
        or not isinstance(coordinate_counts, list)
        or len(coordinate_counts) != auxiliary_count
        or any(type(value) is not int or value <= 0 for value in coordinate_counts)
    ):
        raise ObservationTrainingStateError("backbone cache level schema is invalid")
    try:
        with np.load(io.BytesIO(arrays_path.read_bytes()), allow_pickle=False) as source:
            arrays = {name: _readonly_array(source[name], name) for name in source.files}
    except (OSError, ValueError, KeyError) as error:
        raise ObservationTrainingStateError("backbone cache arrays cannot be decoded") from error
    if manifest.get("array_records") != {
        name: _array_record(value) for name, value in sorted(arrays.items())
    }:
        raise ObservationTrainingStateError("backbone cache array identity mismatch")
    try:
        pcd = {field: arrays[f"pcd__{field}"] for field in _LEVEL_FIELDS}
        auxiliary = tuple(
            {
                field: arrays[f"aux_{level_index:03d}__{field}"]
                for field in _LEVEL_FIELDS
            }
            for level_index in range(auxiliary_count)
        )
        coordinates = tuple(
            tuple(
                arrays[f"coord_{level_index:03d}_{batch_index:03d}"]
                for batch_index in range(batch_count)
            )
            for level_index, batch_count in enumerate(coordinate_counts)
        )
    except KeyError as error:
        raise ObservationTrainingStateError("backbone cache required array is missing") from error
    expected_names = set(
        _cache_arrays(pcd, auxiliary, coordinates)
    )
    if set(arrays) != expected_names:
        raise ObservationTrainingStateError("backbone cache has undeclared arrays")
    content = _cache_content(
        expected_identity, arrays, auxiliary_count, coordinate_counts
    )
    if content != manifest.get("content_sha256"):
        raise ObservationTrainingStateError("backbone cache content digest mismatch")
    return FrozenBackboneCache(expected_identity, pcd, auxiliary, coordinates, content)


def materialize_backbone_cache(
    cache: FrozenBackboneCache,
    backbone: object,
    *,
    device: torch.device | str,
) -> FrozenReSceneFeatures:
    if not isinstance(cache, FrozenBackboneCache):
        raise TypeError("cache must be FrozenBackboneCache")
    target = torch.device(device)
    try:
        point_factory = backbone.model_lib.structure.Point
        formatter = backbone.format
    except AttributeError as error:
        raise ObservationTrainingStateError("backbone cannot materialize cached points") from error

    def materialize(level: Mapping[str, np.ndarray]) -> Any:
        point = point_factory(
            {
                field: torch.as_tensor(
                    np.array(level[field], copy=True, order="C"), device=target
                )
                for field in _LEVEL_FIELDS
            }
        )
        return formatter(point)

    return FrozenReSceneFeatures(
        materialize(cache.pcd_level),
        [materialize(value) for value in cache.auxiliary_levels],
        [
            [
                torch.as_tensor(np.array(value, copy=True, order="C"), device=target)
                for value in batches
            ]
            for batches in cache.coordinates
        ],
    )


@dataclass(frozen=True, slots=True)
class ObservationCheckpointMetadata:
    model_variant: str
    base_checkpoint_sha256: str
    source_commit: str
    resolved_config: Mapping[str, object]
    split_id: str
    observation_sha256: str
    backbone_cache_sha256: str
    seed: int
    optimizer_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_variant", _nonempty(self.model_variant, "model_variant")
        )
        object.__setattr__(
            self,
            "base_checkpoint_sha256",
            _sha256(self.base_checkpoint_sha256, "base checkpoint"),
        )
        if not isinstance(self.source_commit, str) or _GIT_SHA.fullmatch(self.source_commit) is None:
            raise ObservationTrainingStateError("source_commit must be a lowercase Git SHA")
        encoded, config = _canonical(self.resolved_config, "resolved_config")
        del encoded
        if not isinstance(config, dict):
            raise ObservationTrainingStateError("resolved_config must be an object")
        object.__setattr__(self, "resolved_config", config)
        object.__setattr__(self, "split_id", _nonempty(self.split_id, "split_id"))
        object.__setattr__(
            self,
            "observation_sha256",
            _sha256(self.observation_sha256, "observation"),
        )
        object.__setattr__(
            self,
            "backbone_cache_sha256",
            _sha256(self.backbone_cache_sha256, "backbone cache"),
        )
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationTrainingStateError("seed must be a nonnegative integer")
        if type(self.optimizer_updates) is not int or self.optimizer_updates < 0:
            raise ObservationTrainingStateError(
                "optimizer_updates must be a nonnegative integer"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "model_variant": self.model_variant,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "source_commit": self.source_commit,
            "resolved_config": dict(self.resolved_config),
            "split_id": self.split_id,
            "observation_sha256": self.observation_sha256,
            "backbone_cache_sha256": self.backbone_cache_sha256,
            "seed": self.seed,
            "optimizer_updates": self.optimizer_updates,
        }


@dataclass(frozen=True, slots=True)
class TrainableCheckpointPaths:
    root: Path
    manifest: Path
    weights: Path


def _trainable_targets(
    model: nn.Module, criterion: nn.Module
) -> dict[str, torch.Tensor]:
    if not isinstance(model, nn.Module) or not isinstance(criterion, nn.Module):
        raise TypeError("model and criterion must be torch modules")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("native.backbone.")
    ):
        raise ObservationTrainingStateError("frozen backbone has trainable parameters")
    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result[f"model.parameter.{name}"] = parameter
    for name, buffer in model.named_buffers():
        if not name.startswith("native.backbone."):
            result[f"model.buffer.{name}"] = buffer
    for name, parameter in criterion.named_parameters():
        if parameter.requires_grad:
            result[f"criterion.parameter.{name}"] = parameter
    for name, buffer in criterion.named_buffers():
        result[f"criterion.buffer.{name}"] = buffer
    if not result:
        raise ObservationTrainingStateError("checkpoint has no trainable state")
    return dict(sorted(result.items()))


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    content = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def save_trainable_checkpoint(
    *,
    model: nn.Module,
    criterion: nn.Module,
    output_root: str | Path,
    metadata: ObservationCheckpointMetadata,
) -> TrainableCheckpointPaths:
    if not isinstance(metadata, ObservationCheckpointMetadata):
        raise TypeError("metadata must be ObservationCheckpointMetadata")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationTrainingStateError(f"checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        targets = _trainable_targets(model, criterion)
        saved = {
            name: value.detach().cpu().contiguous() for name, value in targets.items()
        }
        weights_path = staging / "trainable.safetensors"
        save_file(saved, weights_path)
        records = {name: _tensor_record(value) for name, value in saved.items()}
        content_payload = {"metadata": metadata.as_dict(), "tensor_records": records}
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINABLE_STATE_V1",
            "status": "PASS",
            "metadata": metadata.as_dict(),
            "trainable_keys": list(saved),
            "tensor_records": records,
            "content_sha256": hashlib.sha256(
                _canonical(content_payload, "checkpoint content")[0]
            ).hexdigest(),
            "weights": _file_record(weights_path),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TrainableCheckpointPaths(
        output, output / "manifest.json", output / "trainable.safetensors"
    )


def load_trainable_checkpoint(
    *,
    model: nn.Module,
    criterion: nn.Module,
    checkpoint_root: str | Path,
    expected_metadata: ObservationCheckpointMetadata,
) -> None:
    if not isinstance(expected_metadata, ObservationCheckpointMetadata):
        raise TypeError("expected_metadata must be ObservationCheckpointMetadata")
    root = Path(checkpoint_root).absolute()
    manifest_path = root / "manifest.json"
    weights_path = root / "trainable.safetensors"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingStateError("checkpoint manifest is unavailable") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "metadata",
        "trainable_keys",
        "tensor_records",
        "content_sha256",
        "weights",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id")
        != "OVI_RESCENE_OBSERVATION_TRAINABLE_STATE_V1"
        or manifest.get("status") != "PASS"
    ):
        raise ObservationTrainingStateError("checkpoint manifest is invalid")
    if manifest.get("metadata") != expected_metadata.as_dict():
        raise ObservationTrainingStateError("checkpoint metadata mismatch")
    if manifest.get("weights") != _file_record(weights_path):
        raise ObservationTrainingStateError("checkpoint weight binding mismatch")
    targets = _trainable_targets(model, criterion)
    if manifest.get("trainable_keys") != list(targets):
        raise ObservationTrainingStateError("checkpoint trainable key contract mismatch")
    try:
        saved = load_file(weights_path, device="cpu")
    except (OSError, ValueError, RuntimeError) as error:
        raise ObservationTrainingStateError("checkpoint weights cannot be decoded") from error
    if set(saved) != set(targets):
        raise ObservationTrainingStateError("checkpoint tensor keys mismatch")
    records = {name: _tensor_record(value) for name, value in sorted(saved.items())}
    if manifest.get("tensor_records") != records:
        raise ObservationTrainingStateError("checkpoint tensor identity mismatch")
    content_payload = {"metadata": expected_metadata.as_dict(), "tensor_records": records}
    content = hashlib.sha256(
        _canonical(content_payload, "checkpoint content")[0]
    ).hexdigest()
    if content != manifest.get("content_sha256"):
        raise ObservationTrainingStateError("checkpoint content digest mismatch")
    with torch.no_grad():
        for name, target in targets.items():
            source = saved[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ObservationTrainingStateError(
                    f"checkpoint tensor contract mismatch: {name}"
                )
            target.copy_(source.to(device=target.device))


__all__ = [
    "BackboneCacheIdentity",
    "BackboneCachePaths",
    "FrozenBackboneCache",
    "ObservationCheckpointMetadata",
    "ObservationTrainingStateError",
    "TrainableCheckpointPaths",
    "capture_backbone_cache",
    "load_backbone_cache",
    "load_trainable_checkpoint",
    "materialize_backbone_cache",
    "save_backbone_cache",
    "save_trainable_checkpoint",
]
