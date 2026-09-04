"""Deterministic surface adapter from independent OVI maps to ReScene tokens."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)


FeatureSchema = Literal["rgb", "rgb_normals"]
_ALLOWED_NEURAL_VOXELS_M = (0.01, 0.02, 0.04)
_ARRAY_KEYS = frozenset(
    {
        "coordinates_xyzt",
        "features",
        "visit_ids",
        "source_visit_ids",
        "source_entity_indices",
        "source_point_indices",
        "source_to_token_offsets",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "status",
        "pair_sha256",
        "arrays",
        "neural_voxel_size_m",
        "feature_schema",
        "coordinate_frame_id",
        "source_manifest_sha256",
        "source_visit_map_sha256",
        "source_entity_id_table",
    }
)


class AdapterError(ValueError):
    """Raised when source features or a serialized pair violate the contract."""


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    neural_voxel_size_m: float = 0.02
    feature_schema: FeatureSchema = "rgb"

    def __post_init__(self) -> None:
        if isinstance(self.neural_voxel_size_m, bool) or not isinstance(
            self.neural_voxel_size_m, (int, float, np.number)
        ):
            raise ValueError("neural_voxel_size_m must be numeric")
        voxel = float(self.neural_voxel_size_m)
        if not math.isfinite(voxel) or voxel not in _ALLOWED_NEURAL_VOXELS_M:
            raise ValueError(
                "neural_voxel_size_m must be one of 0.01, 0.02, or 0.04"
            )
        if self.feature_schema not in {"rgb", "rgb_normals"}:
            raise ValueError("feature_schema must be rgb or rgb_normals")
        object.__setattr__(self, "neural_voxel_size_m", voxel)


@dataclass(frozen=True, slots=True)
class PairArtifactPaths:
    root: Path
    manifest: Path
    arrays: Path


@dataclass(frozen=True, slots=True)
class _SourceSample:
    group_key: tuple[int, int, int, int, str]
    global_index: int
    visit_id: int
    entity_id: str
    xyz: np.ndarray
    feature: np.ndarray


def _entity_features(entity: object, feature_schema: FeatureSchema) -> np.ndarray:
    points = np.asarray(getattr(entity, "points_xyz", None))
    metadata = getattr(entity, "metadata", None)
    entity_id = str(getattr(entity, "entity_id", ""))
    if not isinstance(metadata, dict):
        raise AdapterError(f"OVI entity {entity_id} metadata must be a dictionary")
    if metadata.get("point_rgb_source") == "instance_palette":
        raise AdapterError("OVI instance palette colors cannot be used as camera RGB")
    if metadata.get("point_rgb_source") != "camera_rgb":
        raise AdapterError(f"OVI entity {entity_id} point_rgb is not source-bound camera RGB")
    raw_rgb = np.asarray(metadata.get("point_rgb"))
    if (
        raw_rgb.shape != (len(points), 3)
        or not np.issubdtype(raw_rgb.dtype, np.integer)
        or np.issubdtype(raw_rgb.dtype, np.bool_)
        or np.any(raw_rgb < 0)
        or np.any(raw_rgb > 255)
    ):
        raise AdapterError(
            f"OVI entity {entity_id} point_rgb must be source-bound RGB uint8 values"
        )
    features = raw_rgb.astype(np.float32) / np.float32(255.0)
    if feature_schema == "rgb_normals":
        if metadata.get("point_normals_source") != "source_geometry":
            raise AdapterError(
                f"OVI entity {entity_id} point_normals are not source-bound"
            )
        normals = np.asarray(metadata.get("point_normals"), dtype=np.float64)
        if normals.shape != (len(points), 3) or not np.all(np.isfinite(normals)):
            raise AdapterError(
                f"OVI entity {entity_id} point_normals must have shape (N, 3)"
            )
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            raise AdapterError(f"OVI entity {entity_id} point_normals cannot be zero")
        normalized = (normals / lengths[:, None]).astype(np.float32)
        features = np.concatenate((features, normalized), axis=1)
    return features


def _quantized_xyz(points: np.ndarray, voxel_size_m: float, entity_id: str) -> np.ndarray:
    scaled = np.floor(np.asarray(points, dtype=np.float64) / voxel_size_m)
    limit = np.iinfo(np.int64).max
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise AdapterError(f"OVI entity {entity_id} points exceed quantization range")
    return scaled.astype(np.int64)


def _source_samples(
    visits: tuple[VisitMap, VisitMap], config: AdapterConfig
) -> list[_SourceSample]:
    samples: list[_SourceSample] = []
    global_index = 0
    for visit in visits:
        for entity in sorted(visit.snapshot.entities, key=lambda item: item.entity_id):
            points = np.asarray(entity.points_xyz, dtype=np.float64)
            if not len(points):
                continue
            features = _entity_features(entity, config.feature_schema)
            quantized = _quantized_xyz(
                points, config.neural_voxel_size_m, entity.entity_id
            )
            for point_index in range(len(points)):
                voxel = quantized[point_index]
                samples.append(
                    _SourceSample(
                        group_key=(
                            visit.visit_id,
                            int(voxel[0]),
                            int(voxel[1]),
                            int(voxel[2]),
                            entity.entity_id,
                        ),
                        global_index=global_index,
                        visit_id=visit.visit_id,
                        entity_id=entity.entity_id,
                        xyz=points[point_index],
                        feature=features[point_index],
                    )
                )
                global_index += 1
    if not samples:
        raise AdapterError("OVI visit pair contains no entity surface points")
    samples.sort(key=lambda item: (*item.group_key, item.global_index))
    return samples


def adapt_visit_pair(
    t0: VisitMap,
    t1: VisitMap,
    config: AdapterConfig,
) -> NeuralSampleMap:
    """Aggregate OVI surface samples spatially while preserving exact visit time."""

    if not isinstance(config, AdapterConfig):
        raise TypeError("config must be an AdapterConfig")
    if t0.map_voxel_size_m != 0.01 or t1.map_voxel_size_m != 0.01:
        raise AdapterError("OVI mapping voxel must remain frozen at 0.01 m")
    validate_visit_pair(t0, t1)
    before = (
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )
    samples = _source_samples((t0, t1), config)

    coordinates: list[np.ndarray] = []
    features: list[np.ndarray] = []
    token_visits: list[int] = []
    contributor_visits: list[int] = []
    contributor_entities: list[str] = []
    contributor_points: list[int] = []
    offsets = [0]
    index = 0
    while index < len(samples):
        group_key = samples[index].group_key
        end = index + 1
        while end < len(samples) and samples[end].group_key == group_key:
            end += 1
        group = samples[index:end]
        visit_id = group_key[0]
        xyz = np.stack([item.xyz for item in group], axis=0)
        feature = np.stack([item.feature for item in group], axis=0)
        coordinates.append(
            np.concatenate(
                (xyz.mean(axis=0, dtype=np.float64), np.asarray([visit_id]))
            )
        )
        features.append(feature.mean(axis=0, dtype=np.float64).astype(np.float32))
        token_visits.append(visit_id)
        contributor_visits.extend(item.visit_id for item in group)
        contributor_entities.extend(item.entity_id for item in group)
        contributor_points.extend(item.global_index for item in group)
        offsets.append(offsets[-1] + len(group))
        index = end

    result = NeuralSampleMap(
        coordinates_xyzt=np.stack(coordinates, axis=0),
        features=np.stack(features, axis=0),
        visit_ids=np.asarray(token_visits, dtype=np.int8),
        source_visit_ids=np.asarray(contributor_visits, dtype=np.int8),
        source_entity_ids=tuple(contributor_entities),
        source_point_indices=np.asarray(contributor_points, dtype=np.int64),
        source_to_token_offsets=np.asarray(offsets, dtype=np.int64),
        neural_voxel_size_m=config.neural_voxel_size_m,
        feature_schema=config.feature_schema,
        coordinate_frame_id=t0.coordinate_frame_id,
        source_manifest_sha256=t0.source_manifest_sha256,
        source_visit_map_sha256=(t0.snapshot_sha256, t1.snapshot_sha256),
    )
    after = (
        snapshot_content_sha256(t0.snapshot),
        snapshot_content_sha256(t1.snapshot),
    )
    if before != after or before != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise AdapterError("OVI input snapshot changed during adaptation")
    return result


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_and_fsync(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_neural_sample_artifact(
    pair: NeuralSampleMap, output_root: str | Path
) -> PairArtifactPaths:
    """Publish one hash-bound NPZ/JSON pair in an atomic directory rename."""

    if not isinstance(pair, NeuralSampleMap):
        raise TypeError("pair must be a NeuralSampleMap")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"pair artifact output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        entity_table = tuple(sorted(set(pair.source_entity_ids)))
        entity_index = {entity_id: index for index, entity_id in enumerate(entity_table)}
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                coordinates_xyzt=pair.coordinates_xyzt,
                features=pair.features,
                visit_ids=pair.visit_ids,
                source_visit_ids=pair.source_visit_ids,
                source_entity_indices=np.asarray(
                    [entity_index[item] for item in pair.source_entity_ids],
                    dtype=np.int64,
                ),
                source_point_indices=pair.source_point_indices,
                source_to_token_offsets=pair.source_to_token_offsets,
            )
            stream.flush()
            os.fsync(stream.fileno())
        arrays_content = arrays_path.read_bytes()
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_PAIR_V1",
            "status": "PASS",
            "pair_sha256": pair.content_sha256(),
            "arrays": {
                "path": "arrays.npz",
                "sha256": _sha256_bytes(arrays_content),
                "byte_count": len(arrays_content),
            },
            "neural_voxel_size_m": pair.neural_voxel_size_m,
            "feature_schema": pair.feature_schema,
            "coordinate_frame_id": pair.coordinate_frame_id,
            "source_manifest_sha256": pair.source_manifest_sha256,
            "source_visit_map_sha256": list(pair.source_visit_map_sha256),
            "source_entity_id_table": list(entity_table),
        }
        _write_and_fsync(staging / "manifest.json", _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise ValueError(f"pair artifact output already exists: {output}")
        staging.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PairArtifactPaths(
        root=output,
        manifest=output / "manifest.json",
        arrays=output / "arrays.npz",
    )


def _regular_file_bytes(path: Path, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"{name} must be a regular non-symlink file")
    before = path.stat(follow_symlinks=False)
    content = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(content) != before.st_size:
        raise AdapterError(f"{name} changed while being read")
    return content


def load_neural_sample_artifact(output_root: str | Path) -> NeuralSampleMap:
    """Verify and load a pair artifact without pickle support."""

    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise AdapterError("pair artifact root must be a regular directory")
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    try:
        manifest = json.loads(
            _regular_file_bytes(manifest_path, "pair manifest").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError("pair manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise AdapterError("pair manifest schema is invalid")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_RESCENE_PAIR_V1"
        or manifest.get("status") != "PASS"
    ):
        raise AdapterError("pair manifest identity is invalid")
    record = manifest.get("arrays")
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "byte_count"}:
        raise AdapterError("pair arrays binding is invalid")
    if record.get("path") != "arrays.npz":
        raise AdapterError("pair arrays path is invalid")
    arrays_content = _regular_file_bytes(arrays_path, "pair arrays")
    if record.get("sha256") != _sha256_bytes(arrays_content):
        raise AdapterError("pair arrays SHA-256 mismatch")
    if record.get("byte_count") != len(arrays_content):
        raise AdapterError("pair arrays byte count mismatch")
    entity_table = manifest.get("source_entity_id_table")
    if (
        not isinstance(entity_table, list)
        or not entity_table
        or any(not isinstance(item, str) or not item for item in entity_table)
        or entity_table != sorted(set(entity_table))
    ):
        raise AdapterError("source entity ID table is invalid")
    try:
        with np.load(io.BytesIO(arrays_content), allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise AdapterError("pair array keys are invalid")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, AdapterError):
            raise
        raise AdapterError("pair arrays are not a valid pickle-free NPZ") from error
    entity_indices = arrays.pop("source_entity_indices")
    if (
        entity_indices.ndim != 1
        or not np.issubdtype(entity_indices.dtype, np.integer)
        or np.any(entity_indices < 0)
        or np.any(entity_indices >= len(entity_table))
    ):
        raise AdapterError("source entity indices are invalid")
    source_entity_ids = tuple(entity_table[int(index)] for index in entity_indices)
    pair = NeuralSampleMap(
        **arrays,
        source_entity_ids=source_entity_ids,
        neural_voxel_size_m=manifest["neural_voxel_size_m"],
        feature_schema=manifest["feature_schema"],
        coordinate_frame_id=manifest["coordinate_frame_id"],
        source_manifest_sha256=manifest["source_manifest_sha256"],
        source_visit_map_sha256=tuple(manifest["source_visit_map_sha256"]),
    )
    if pair.content_sha256() != manifest.get("pair_sha256"):
        raise AdapterError("pair content SHA-256 mismatch")
    return pair
