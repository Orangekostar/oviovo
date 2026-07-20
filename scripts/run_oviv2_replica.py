#!/usr/bin/env python3
"""Build and evaluate an immutable OVIV2 sparse voxel map on Replica."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.data_structures import Frame  # noqa: E402
from src.datasets.replica import ReplicaRoom0Dataset  # noqa: E402
from src.datasets.scannet200 import ScanNet200Dataset  # noqa: E402
from src.oviv2.association import AssociationConfig  # noqa: E402
from src.oviv2.dense_projection import DenseSemanticConfig  # noqa: E402
from src.oviv2.dense_semantics import (  # noqa: E402
    DenseSemanticFrame,
    DenseSemanticProvenance,
    load_dense_frame,
)
from src.oviv2.entities import EntityRegistryConfig  # noqa: E402
from src.oviv2.evidence import EvidenceConfig  # noqa: E402
from src.oviv2.geometry import TsdfConfig  # noqa: E402
from src.oviv2.meshing import derive_labeled_mesh, write_labeled_mesh  # noqa: E402
from src.oviv2.observations import CachedFrontendAdapter, ReplicaVocabulary  # noqa: E402
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig  # noqa: E402
from src.oviv2.semantic_fusion import SemanticFusionConfig  # noqa: E402
from src.oviv2.structure import DepthStructureConfig, DepthStructureFrontend  # noqa: E402
from src.oviv2.tracking import LocalTrackerConfig  # noqa: E402


SCENE_CONFIG_FIELDS = frozenset(
    {
        "scene",
        "dataset_root",
        "frontend_cache_dir",
        "gt_mesh",
        "gt_info",
        "dense_cache_dir",
        "num_frames",
        "algorithm_hash",
    }
)
_DENSE_METHOD = "OVIV2-dense-semantic-cache"
_DENSE_MANIFEST_NAME = "dense_manifest.json"
_MAX_DENSE_MANIFEST_BYTES = 8 * 1024 * 1024
_REPLICA_EVALUATOR_SCRIPT = REPO_ROOT / "scripts/evaluation/evaluate_oviv2_replica.py"
_SCANNET_EVALUATOR_SCRIPT = REPO_ROOT / "scripts/evaluation/evaluate_oviv2_scannet200.py"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DENSE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "scene",
        "frame_count",
        "source_frame_ids",
        "image_shape",
        "sample_stride",
        "top_k",
        "class_count",
        "vocabulary_sha256",
        "provenance",
        "cache_files_sha256",
    }
)
_DENSE_PROVENANCE_KEYS = frozenset(
    {
        "backend",
        "source_commit",
        "radio_commit",
        "model_id",
        "model_sha256",
        "auxiliary_model_sha256",
        "language_model_id",
        "language_model_revision",
        "language_model_sha256",
        "vocabulary_sha256",
        "prompt_sha256",
        "inference_config_sha256",
        "cache_prefix_sha256",
    }
)


@dataclass(frozen=True)
class _DenseCachePreflight:
    cache_dir: Path
    manifest_sha256: str
    frame_count: int
    source_frame_ids: tuple[int, ...]
    image_shape: tuple[int, int]
    sample_stride: int
    top_k: int
    class_count: int
    vocabulary_sha256: str
    producer_provenance: DenseSemanticProvenance
    consumed_provenance: DenseSemanticProvenance
    cache_files_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def algorithm_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in sorted(config) if key not in SCENE_CONFIG_FIELDS}


def algorithm_hash(config: dict[str, Any]) -> str:
    return _json_hash(algorithm_config(config))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    for name in ("dataset_root", "frontend_cache_dir", "manifest", "gt_mesh", "gt_info"):
        raw = resolved.get(name)
        if raw is None:
            continue
        path = Path(raw).expanduser()
        resolved[name] = str(path if path.is_absolute() else (REPO_ROOT / path).resolve())
    return resolved


def _nonempty_feature_model_id(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} feature_model_id must be a non-empty string")
    return value.strip()


def _resolve_frontend_feature_model_id(
    config: dict[str, Any],
    frontend_manifest: dict[str, Any] | None,
) -> str | None:
    config_model_id = (
        _nonempty_feature_model_id(config["feature_model_id"], "config")
        if "feature_model_id" in config
        else None
    )
    manifest_model_id: str | None = None
    if frontend_manifest is not None:
        if "feature_model_id" in frontend_manifest:
            manifest_model_id = _nonempty_feature_model_id(
                frontend_manifest["feature_model_id"],
                "frontend manifest",
            )
        else:
            provenance = frontend_manifest.get("provenance_sha256")
            if provenance is not None and not isinstance(provenance, dict):
                raise ValueError("frontend manifest provenance_sha256 must be an object")
            if isinstance(provenance, dict) and "clip_model" in provenance:
                clip_hash = provenance["clip_model"]
                if (
                    not isinstance(clip_hash, str)
                    or len(clip_hash) != 64
                    or any(value not in "0123456789abcdefABCDEF" for value in clip_hash)
                ):
                    raise ValueError(
                        "frontend manifest provenance_sha256.clip_model must be a 64-digit hex hash"
                    )
                manifest_model_id = f"clip-sha256:{clip_hash.lower()}"
    if (
        config_model_id is not None
        and manifest_model_id is not None
        and config_model_id != manifest_model_id
    ):
        raise ValueError("config and frontend manifest feature_model_id conflict")
    return manifest_model_id if manifest_model_id is not None else config_model_id


def _verify_cached_image_features(cache_dir: Path, num_frames: int) -> None:
    for cache_index in range(num_frames):
        cache_path = cache_dir / f"frame{cache_index:06d}.pkl.gz"
        try:
            with gzip.open(cache_path, "rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise ValueError(f"{cache_path.name} payload is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{cache_path.name} payload must be a dict")
        if "mask" not in payload:
            raise ValueError(f"{cache_path.name} mask is missing")
        masks = np.asarray(payload["mask"])
        if masks.ndim < 1:
            raise ValueError(f"{cache_path.name} mask must define N observations")
        observation_count = int(masks.shape[0])
        if "image_feats" not in payload:
            raise ValueError(f"{cache_path.name} image_feats is missing")
        try:
            image_features = np.asarray(payload["image_feats"], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{cache_path.name} image_feats must be a numeric (N, D) array"
            ) from exc
        if (
            image_features.ndim != 2
            or image_features.shape[0] != observation_count
            or image_features.shape[1] <= 0
        ):
            raise ValueError(
                f"{cache_path.name} image_feats must have shape (N, D) with D > 0"
            )
        if not np.isfinite(image_features).all():
            raise ValueError(f"{cache_path.name} image_feats must be finite")
        max_abs = np.max(np.abs(image_features), axis=1)
        if np.any(max_abs == 0.0):
            raise ValueError(
                f"{cache_path.name} image_feats rows must be non-zero"
            )
        scaled = image_features / max_abs[:, None]
        if not np.isfinite(np.linalg.norm(scaled, axis=1)).all():
            raise ValueError(
                f"{cache_path.name} image_feats rows must have stable non-zero norms"
            )


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _dirty_digest() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _artifact_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def _verify_resume(output: Path, config_hash: str, requested_frames: int) -> dict[str, Any]:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("resume requires an atomically completed run_manifest.json")
    manifest = _load_json(manifest_path)
    if manifest.get("config_hash") != config_hash:
        raise ValueError("resume config hash does not match completed run")
    if manifest.get("frame_selection", {}).get("sampled_frame_count") != requested_frames:
        raise ValueError("resume frame count does not match completed run")
    for relative, expected in manifest.get("artifact_checksums", {}).items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"resume artifact checksum mismatch: {relative}")
    return manifest


def _strict_integer(value: object, name: str, *, positive: bool) -> int:
    if type(value) is not int:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    normalized = int(value)
    if (positive and normalized <= 0) or (not positive and normalized < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if normalized > int(np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit in signed int64")
    return normalized


def _dense_cache_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (REPO_ROOT / path).absolute()


def _dense_cache_prefix(cache_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    expected_names = [f"frame{index:06d}.npz" for index in range(len(cache_hashes))]
    if list(cache_hashes) != expected_names:
        raise ValueError("dense cache checksum keys are not canonical")
    for cache_index, name in enumerate(expected_names):
        checksum = cache_hashes[name]
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            raise ValueError(f"dense cache checksum is invalid: {name}")
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _load_dense_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        initial = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"dense manifest must be a regular file: {path}") from exc
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"dense manifest must be a regular non-symlink file: {path}")
    if initial.st_size > _MAX_DENSE_MANIFEST_BYTES:
        raise ValueError("dense manifest exceeds the JSON size limit")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"dense manifest could not be opened safely: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        initial_identity = (initial.st_dev, initial.st_ino)
        opened_identity = (opened.st_dev, opened.st_ino)
        initial_version = (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns)
        opened_version = (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened_identity != initial_identity
            or opened_version != initial_version
        ):
            raise ValueError("dense manifest changed before it could be opened safely")
        if opened.st_size > _MAX_DENSE_MANIFEST_BYTES:
            raise ValueError("dense manifest exceeds the JSON size limit")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(_MAX_DENSE_MANIFEST_BYTES + 1)
        after = os.fstat(descriptor)
        after_identity = (after.st_dev, after.st_ino)
        after_version = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if len(raw) > _MAX_DENSE_MANIFEST_BYTES or after.st_size > _MAX_DENSE_MANIFEST_BYTES:
            raise ValueError("dense manifest exceeds the JSON size limit")
        if (
            after_identity != opened_identity
            or after_version != opened_version
            or len(raw) != after.st_size
        ):
            raise ValueError("dense manifest changed while it was being read")
    finally:
        os.close(descriptor)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"dense manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"dense manifest contains invalid JSON constant {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dense manifest must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("dense manifest root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_and_validate_dense_frame(
    path: Path,
    expected_sha256: str,
    cache_index: int,
    source_frame_id: int,
    image_shape: tuple[int, int],
    sample_stride: int,
    class_count: int,
    top_k: int,
) -> DenseSemanticFrame:
    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"dense cache {path.name} must be a regular non-symlink file"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"dense cache {path.name} must be a regular non-symlink file")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    dense = load_dense_frame(path, expected_sha256=expected_sha256)
    try:
        after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"dense cache {path.name} changed while it was loaded") from exc
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stat.S_ISREG(after.st_mode) or after_identity != before_identity:
        raise ValueError(f"dense cache {path.name} changed while it was loaded")
    if dense.cache_frame_id != cache_index:
        raise ValueError(f"dense cache_frame_id mismatch: {path.name}")
    if dense.source_frame_id != source_frame_id:
        raise ValueError(f"dense source_frame_id mismatch: {path.name}")
    if dense.image_shape != image_shape:
        raise ValueError(f"dense image_shape mismatch: {path.name}")
    if dense.sample_stride != sample_stride:
        raise ValueError(f"dense sample_stride mismatch: {path.name}")
    if dense.class_count != class_count:
        raise ValueError(f"dense class_count mismatch: {path.name}")
    if dense.class_ids.shape[2] != top_k:
        raise ValueError(f"dense top_k mismatch: {path.name}")
    return dense


def _preflight_dense_cache(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    dataset: ReplicaRoom0Dataset | ScanNet200Dataset,
    source_ids: list[int],
    num_frames: int,
) -> _DenseCachePreflight | None:
    mode = config.get("dense_semantic_mode", "disabled")
    if mode == "disabled":
        return None
    if mode != "cached_probabilities":
        raise ValueError(
            "dense_semantic_mode must be disabled or cached_probabilities"
        )
    if "dense_cache_dir" not in config:
        raise ValueError("dense_cache_dir is required for cached_probabilities")
    cache_dir = _dense_cache_path(config["dense_cache_dir"], "dense_cache_dir")
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError("dense_cache_dir must be a regular non-symlink directory")
    manifest_path = cache_dir / _DENSE_MANIFEST_NAME
    manifest, manifest_sha256 = _load_dense_manifest(manifest_path)
    if set(manifest) != _DENSE_MANIFEST_KEYS:
        raise ValueError("dense manifest keys contain unsupported or missing entries")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("dense manifest schema_version must be 1")
    if manifest.get("method") != _DENSE_METHOD:
        raise ValueError("dense manifest method mismatch")
    if manifest.get("scene") != config["scene"]:
        raise ValueError("dense manifest scene mismatch")
    frame_count = _strict_integer(
        manifest.get("frame_count"),
        "dense manifest frame_count",
        positive=True,
    )
    if frame_count < num_frames:
        raise ValueError("dense manifest does not cover the requested frame prefix")
    if frame_count > len(dataset):
        raise ValueError("dense manifest frame_count exceeds the dataset")
    recorded_source_ids = manifest.get("source_frame_ids")
    if (
        not isinstance(recorded_source_ids, list)
        or len(recorded_source_ids) != frame_count
    ):
        raise ValueError("dense manifest source_frame_ids length mismatch")
    normalized_source_ids = tuple(
        _strict_integer(value, "dense manifest source_frame_id", positive=False)
        for value in recorded_source_ids
    )
    if benchmark.get("dataset") == "ScanNet200":
        expected_source_ids = tuple(dataset.frame_indices[:frame_count])
    else:
        expected_source_ids = tuple(
            int(config.get("source_start", 0))
            + cache_index * int(config.get("source_stride", 10))
            for cache_index in range(frame_count)
        )
    if normalized_source_ids != expected_source_ids:
        raise ValueError("dense manifest source frame IDs mismatch")
    if normalized_source_ids[:num_frames] != tuple(source_ids):
        raise ValueError("dense manifest requested source frame prefix mismatch")
    raw_shape = manifest.get("image_shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != 2:
        raise ValueError("dense manifest image_shape must be [height, width]")
    image_shape = tuple(
        _strict_integer(value, "dense manifest image_shape", positive=True)
        for value in raw_shape
    )
    expected_shape = (dataset.intrinsics.height, dataset.intrinsics.width)
    if image_shape != expected_shape:
        raise ValueError("dense manifest image_shape does not match dataset")
    vocabulary = benchmark["vocabulary"]
    classes = vocabulary["classes"]
    class_count = _strict_integer(
        manifest.get("class_count"),
        "dense manifest class_count",
        positive=True,
    )
    if class_count != len(classes):
        raise ValueError("dense manifest class_count does not match benchmark vocabulary")
    sample_stride = _strict_integer(
        manifest.get("sample_stride"),
        "dense manifest sample_stride",
        positive=True,
    )
    top_k = _strict_integer(manifest.get("top_k"), "dense manifest top_k", positive=True)
    if top_k > class_count:
        raise ValueError("dense manifest top_k exceeds class_count")

    source_path = vocabulary.get("source_path")
    vocabulary_path = _dense_cache_path(
        source_path,
        "benchmark vocabulary.source_path",
    )
    if vocabulary_path.is_symlink() or not vocabulary_path.is_file():
        raise ValueError("benchmark vocabulary source must be a regular non-symlink file")
    raw_vocabulary_sha256 = _sha256(vocabulary_path)
    benchmark_vocabulary_sha256 = vocabulary.get("source_sha256")
    if (
        not isinstance(benchmark_vocabulary_sha256, str)
        or _SHA256_PATTERN.fullmatch(benchmark_vocabulary_sha256) is None
        or benchmark_vocabulary_sha256 != raw_vocabulary_sha256
    ):
        raise ValueError("benchmark vocabulary source_sha256 mismatch")
    vocabulary_sha256 = manifest.get("vocabulary_sha256")
    if (
        not isinstance(vocabulary_sha256, str)
        or _SHA256_PATTERN.fullmatch(vocabulary_sha256) is None
        or vocabulary_sha256 != benchmark_vocabulary_sha256
    ):
        raise ValueError("dense manifest vocabulary_sha256 mismatch")
    provenance_payload = manifest.get("provenance")
    if (
        not isinstance(provenance_payload, dict)
        or set(provenance_payload) != _DENSE_PROVENANCE_KEYS
    ):
        raise ValueError("dense manifest provenance keys mismatch")
    producer_provenance = DenseSemanticProvenance(**provenance_payload)
    if producer_provenance.vocabulary_sha256 != vocabulary_sha256:
        raise ValueError("dense manifest provenance vocabulary mismatch")
    cache_hashes = manifest.get("cache_files_sha256")
    if not isinstance(cache_hashes, dict) or len(cache_hashes) != frame_count:
        raise ValueError("dense cache checksum keys are missing or extra")
    prefix_sha256 = _dense_cache_prefix(cache_hashes)
    if producer_provenance.cache_prefix_sha256 != prefix_sha256:
        raise ValueError("dense manifest provenance cache prefix mismatch")
    for cache_index in range(frame_count):
        name = f"frame{cache_index:06d}.npz"
        _load_and_validate_dense_frame(
            cache_dir / name,
            cache_hashes[name],
            cache_index,
            normalized_source_ids[cache_index],
            image_shape,
            sample_stride,
            class_count,
            top_k,
        )
    consumed_hashes = {
        f"frame{cache_index:06d}.npz": cache_hashes[f"frame{cache_index:06d}.npz"]
        for cache_index in range(num_frames)
    }
    consumed_payload = asdict(producer_provenance)
    consumed_payload["cache_prefix_sha256"] = _dense_cache_prefix(consumed_hashes)
    consumed_provenance = DenseSemanticProvenance(**consumed_payload)
    return _DenseCachePreflight(
        cache_dir=cache_dir,
        manifest_sha256=manifest_sha256,
        frame_count=frame_count,
        source_frame_ids=normalized_source_ids,
        image_shape=image_shape,
        sample_stride=sample_stride,
        top_k=top_k,
        class_count=class_count,
        vocabulary_sha256=vocabulary_sha256,
        producer_provenance=producer_provenance,
        consumed_provenance=consumed_provenance,
        cache_files_sha256=dict(cache_hashes),
    )


def _load_dense_cache_frame(
    dense_cache: _DenseCachePreflight,
    cache_index: int,
    source_frame_id: int,
) -> DenseSemanticFrame:
    name = f"frame{cache_index:06d}.npz"
    return _load_and_validate_dense_frame(
        dense_cache.cache_dir / name,
        dense_cache.cache_files_sha256[name],
        cache_index,
        source_frame_id,
        dense_cache.image_shape,
        dense_cache.sample_stride,
        dense_cache.class_count,
        dense_cache.top_k,
    )


def _preflight(
    config: dict[str, Any],
    num_frames: int,
    skip_evaluation: bool,
) -> tuple[
    ReplicaRoom0Dataset | ScanNet200Dataset,
    dict[str, Any],
    list[int],
    str,
    dict[str, Any] | None,
    _DenseCachePreflight | None,
]:
    required = ("scene", "dataset_root", "frontend_cache_dir", "manifest")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"config is missing required fields: {', '.join(missing)}")
    manifest_path = Path(config["manifest"])
    benchmark = _load_json(manifest_path)
    dataset_name = benchmark.get("dataset")
    if benchmark.get("schema_version") != 1 or dataset_name not in {
        "Replica",
        "ScanNet200",
    }:
        raise ValueError(
            "benchmark manifest must use schema_version 1 for Replica or ScanNet200"
        )
    matching_scenes = [
        item
        for item in benchmark.get("scenes", ())
        if isinstance(item, dict) and item.get("scene") == config["scene"]
    ]
    if len(matching_scenes) != 1:
        raise ValueError("configured scene is absent from benchmark manifest")
    scene_record = matching_scenes[0]
    classes = benchmark.get("vocabulary", {}).get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("benchmark manifest has no frozen vocabulary")

    if dataset_name == "Replica":
        dataset: ReplicaRoom0Dataset | ScanNet200Dataset = ReplicaRoom0Dataset(
            Path(config["dataset_root"])
        )
        source_start = int(config.get("source_start", 0))
        source_stride = int(config.get("source_stride", 10))
        if source_start < 0 or source_stride <= 0:
            raise ValueError("source_start must be non-negative and source_stride positive")
        all_source_ids = [
            source_start + index * source_stride for index in range(len(dataset))
        ]
        selection = benchmark.get("frame_selection", {})
        stop = int(selection.get("stop_exclusive", all_source_ids[-1] + 1))
        if all_source_ids[-1] >= stop:
            raise ValueError("requested source frame selection exceeds benchmark manifest")
    else:
        raw_source_ids = scene_record.get("source_frame_ids")
        if not isinstance(raw_source_ids, list):
            raise ValueError("ScanNet scene source_frame_ids must be a list")
        all_source_ids = [
            _strict_integer(value, "ScanNet source_frame_id", positive=False)
            for value in raw_source_ids
        ]
        recorded_count = _strict_integer(
            scene_record.get("frame_count"), "ScanNet scene frame_count", positive=True
        )
        if len(all_source_ids) != recorded_count or len(set(all_source_ids)) != len(
            all_source_ids
        ):
            raise ValueError("ScanNet scene source frame contract is invalid")
        configured_count = config.get("num_frames", recorded_count)
        if type(configured_count) is not int or configured_count != recorded_count:
            raise ValueError("config num_frames must match ScanNet scene frame_count")
        frame_inputs = scene_record.get("frame_inputs")
        if not isinstance(frame_inputs, dict):
            raise ValueError("ScanNet scene frame_inputs must be an object")
        input_hashes = {
            source_id: {
                role: frame_inputs[str(source_id)][f"{role}_sha256"]
                for role in ("color", "depth", "pose")
            }
            for source_id in all_source_ids
        }
        raw_shape = scene_record.get("image_shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            raise ValueError("ScanNet scene image_shape must be [height, width]")
        image_shape = tuple(
            _strict_integer(value, "ScanNet image_shape", positive=True)
            for value in raw_shape
        )
        dataset = ScanNet200Dataset(
            Path(config["dataset_root"]),
            source_frame_ids=all_source_ids,
            input_hashes=input_hashes,
            expected_image_shape=image_shape,
            depth_scale=float(scene_record.get("depth_scale", 1000.0)),
        )
    if num_frames <= 0 or num_frames > len(dataset):
        raise ValueError(f"num_frames must lie in [1, {len(dataset)}]")
    source_ids = all_source_ids[:num_frames]

    cache_dir = Path(config["frontend_cache_dir"])
    frontend_manifest_path = cache_dir / "frontend_manifest.json"
    frontend_manifest = (
        _load_json(frontend_manifest_path) if frontend_manifest_path.is_file() else None
    )
    if frontend_manifest is not None:
        if frontend_manifest.get("method") != "OVIV2":
            raise ValueError("frontend manifest method must be OVIV2")
        if frontend_manifest.get("scene") != config["scene"]:
            raise ValueError("frontend manifest scene does not match runner config")
        if dataset_name == "ScanNet200":
            recorded_ids = frontend_manifest.get("source_frame_ids")
            if (
                not isinstance(recorded_ids, list)
                or recorded_ids[:num_frames] != source_ids
            ):
                raise ValueError("frontend manifest source frame IDs do not match ScanNet")
        available_frames = frontend_manifest.get("frame_count")
        if (
            not isinstance(available_frames, int)
            or isinstance(available_frames, bool)
            or available_frames < num_frames
        ):
            raise ValueError("frontend manifest does not cover the requested run prefix")
    frontend_digest = hashlib.sha256()
    for cache_index in range(num_frames):
        cache_path = cache_dir / f"frame{cache_index:06d}.pkl.gz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache_hash = _sha256(cache_path)
        if frontend_manifest is not None:
            expected_hash = frontend_manifest.get("cache_files_sha256", {}).get(cache_path.name)
            if cache_hash != expected_hash:
                raise ValueError(f"frontend manifest checksum mismatch: {cache_path.name}")
        frontend_digest.update(cache_index.to_bytes(8, "little"))
        frontend_digest.update(bytes.fromhex(cache_hash))
    if not skip_evaluation:
        for name in ("gt_mesh", "gt_info"):
            path = Path(config.get(name, ""))
            if not path.is_file():
                raise FileNotFoundError(path)
    dense_cache = _preflight_dense_cache(
        config,
        benchmark,
        dataset,
        source_ids,
        num_frames,
    )
    return (
        dataset,
        benchmark,
        source_ids,
        frontend_digest.hexdigest(),
        frontend_manifest,
        dense_cache,
    )


def _runtime_config(config: dict[str, Any]) -> Oviv2RuntimeConfig:
    if "semantic_mode" in config and config["semantic_mode"] != "owner_authoritative":
        raise ValueError("semantic_mode must be owner_authoritative")
    if "feature_mode" in config and config["feature_mode"] != "cached_image":
        raise ValueError("feature_mode must be cached_image")
    voxel_size = float(config.get("voxel_size_m", 0.05))
    block_resolution = int(config.get("block_resolution", 8))
    source_stride = int(config.get("source_stride", 10))
    dense_mode = config.get("dense_semantic_mode", "disabled")
    if dense_mode == "disabled":
        dense_semantics = None
    elif dense_mode == "cached_probabilities":
        dense_semantics = DenseSemanticConfig(
            voxel_size_m=voxel_size,
            integration_radius_m=float(config.get("dense_integration_radius_m", 6.0)),
            minimum_probability=float(config.get("dense_minimum_probability", 0.01)),
            minimum_quality=float(config.get("dense_minimum_quality", 0.01)),
            entropy_power=float(config.get("dense_entropy_power", 1.0)),
            view_angle_power=float(config.get("dense_view_angle_power", 1.0)),
        )
    else:
        raise ValueError(
            "dense_semantic_mode must be disabled or cached_probabilities"
        )
    association_keys = {
        "min_directed_overlap": "association_min_directed_overlap",
        "bounds_expansion_m": "association_bounds_expansion_m",
        "max_centroid_distance_m": "association_max_centroid_distance_m",
        "minimum_score": "association_minimum_score",
        "geometry_weight": "association_geometry_weight",
        "overlap_weight": "association_overlap_weight",
        "visual_weight": "association_visual_weight",
        "semantic_weight": "association_semantic_weight",
        "temporal_weight": "association_temporal_weight",
        "semantic_conflict_confidence": "semantic_conflict_confidence",
        "semantic_conflict_visual_override": "semantic_conflict_visual_override",
    }
    association = None
    if any(name in config for name in association_keys.values()):
        defaults = asdict(AssociationConfig())
        association = AssociationConfig(
            **{
                field: config.get(config_name, defaults[field])
                for field, config_name in association_keys.items()
            }
        )
    return Oviv2RuntimeConfig(
        tsdf=TsdfConfig(
            voxel_size_m=voxel_size,
            block_resolution=block_resolution,
            block_count=int(config.get("block_count", 100_000)),
            depth_max_m=float(config.get("depth_max_m", 10.0)),
            trunc_voxel_multiplier=float(config.get("trunc_voxel_multiplier", 4.0)),
        ),
        evidence=EvidenceConfig(
            block_resolution=block_resolution,
            semantic_top_k=int(config.get("semantic_top_k", 4)),
            entity_top_k=int(config.get("entity_top_k", 4)),
        ),
        tracker=LocalTrackerConfig(
            window_size=int(config.get("track_window_size", 5)),
            confirm_hits=int(config.get("confirm_hits", 2)),
            max_age_frames=int(config.get("max_age_frames", 3)) * source_stride,
            min_voxel_overlap=float(config.get("track_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("track_max_centroid_distance_m", 0.5)),
            association=association,
            ambiguous_edge_score=float(config.get("ambiguous_edge_score", 0.70)),
            third_view_min_score=float(config.get("third_view_min_score", 0.75)),
        ),
        registry=EntityRegistryConfig(
            min_voxel_overlap=float(config.get("entity_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("entity_max_centroid_distance_m", 0.6)),
            association=association,
            prototype_top_k=int(config.get("prototype_top_k", 3)),
            prototype_merge_cosine=float(config.get("prototype_merge_cosine", 0.90)),
            view_top_k=int(config.get("view_top_k", 10)),
            view_minimum_novelty_cosine=float(
                config.get("view_minimum_novelty_cosine", 0.10)
            ),
        ),
        visibility_depth_tolerance_m=float(
            config.get("visibility_depth_tolerance_m", 0.1)
        ),
        absence_negative_support=float(config.get("absence_negative_support", 1.0)),
        dense_semantics=dense_semantics,
    )


def _semantic_fusion_config(
    config: dict[str, Any],
) -> SemanticFusionConfig | None:
    mode = config.get("fusion_semantic_mode", "disabled")
    if mode == "disabled":
        if "fusion_entity_weight_scale" in config:
            raise ValueError(
                "fusion_entity_weight_scale requires fusion_semantic_mode"
            )
        return None
    if mode != "uncertainty_linear":
        raise ValueError(
            "fusion_semantic_mode must be disabled or uncertainty_linear"
        )
    if config.get("dense_semantic_mode", "disabled") != "cached_probabilities":
        raise ValueError("uncertainty_linear fusion requires cached dense semantics")
    return SemanticFusionConfig(
        entity_weight_scale=config.get("fusion_entity_weight_scale", 0.5)
    )


def _structure_config(config: dict[str, Any], *, voxel_size_m: float) -> DepthStructureConfig:
    return DepthStructureConfig(
        enabled=bool(config.get("structure_enabled", True)),
        voxel_size_m=voxel_size_m,
        pixel_stride=int(config.get("structure_pixel_stride", config.get("pixel_stride", 4))),
        min_valid_points=int(
            config.get("structure_min_valid_points", config.get("min_valid_points", 10))
        ),
        horizontal_threshold=float(config.get("structure_horizontal_threshold", 0.6)),
        wall_vertical_threshold=float(config.get("structure_wall_vertical_threshold", 0.5)),
        min_component_pixels=int(config.get("structure_min_component_pixels", 500)),
        min_component_fraction=float(config.get("structure_min_component_fraction", 0.01)),
        max_components_per_class=int(config.get("structure_max_components_per_class", 5)),
        object_exclusion_dilation=int(config.get("structure_object_exclusion_dilation", 3)),
        wall_confidence=float(config.get("structure_wall_confidence", 0.75)),
        floor_confidence=float(config.get("structure_floor_confidence", 0.85)),
        ceiling_confidence=float(config.get("structure_ceiling_confidence", 0.80)),
    )


def _run_evaluation(
    config: dict[str, Any],
    snapshot: Path,
    entity_info: Path,
    output: Path,
    semantic_head: str,
) -> None:
    benchmark = _load_json(Path(config["manifest"]))
    if benchmark.get("dataset") == "ScanNet200":
        command = [
            sys.executable,
            str(_SCANNET_EVALUATOR_SCRIPT),
            "--snapshot",
            str(snapshot),
            "--entity-info",
            str(entity_info),
            "--gt-ply",
            str(config["gt_mesh"]),
            "--metadata",
            str(config["gt_info"]),
            "--manifest",
            str(config["manifest"]),
            "--scene",
            str(config["scene"]),
            "--output",
            str(output),
            "--min-instance-points",
            str(int(config.get("min_instance_vertices", 100))),
            "--semantic-head",
            semantic_head,
        ]
    else:
        command = [
            sys.executable,
            str(_REPLICA_EVALUATOR_SCRIPT),
            "--snapshot",
            str(snapshot),
            "--entity-info",
            str(entity_info),
            "--gt-mesh",
            str(config["gt_mesh"]),
            "--gt-info",
            str(config["gt_info"]),
            "--manifest",
            str(config["manifest"]),
            "--scene",
            str(config["scene"]),
            "--output",
            str(output),
            "--min-instance-vertices",
            str(int(config.get("min_instance_vertices", 100))),
            "--semantic-head",
            semantic_head,
        ]
    if semantic_head == "fused_uncertainty":
        fusion_config = _semantic_fusion_config(config)
        assert fusion_config is not None
        command.extend(
            (
                "--fusion-entity-weight-scale",
                str(fusion_config.entity_weight_scale),
            )
        )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    raw_config = _load_json(config_path)
    config_hash = _json_hash(raw_config)
    frozen_algorithm_hash = algorithm_hash(raw_config)
    if raw_config.get("algorithm_hash") not in (None, frozen_algorithm_hash):
        raise ValueError("configured algorithm_hash does not match mapping parameters")
    config = _resolve_config_paths(raw_config)
    semantic_fusion = _semantic_fusion_config(config)
    requested_frames = int(args.num_frames or config.get("num_frames", 200))
    output = args.output.resolve()
    if output.exists():
        if args.resume:
            return _verify_resume(output, config_hash, requested_frames)
        raise FileExistsError(f"output already exists: {output}")

    starting_dirty_digest = _dirty_digest()
    (
        dataset,
        benchmark,
        source_ids,
        frontend_hash,
        frontend_manifest,
        dense_cache,
    ) = _preflight(
        config,
        requested_frames,
        args.skip_evaluation,
    )
    frontend_feature_model_id = _resolve_frontend_feature_model_id(config, frontend_manifest)
    if config.get("feature_mode") == "cached_image":
        if frontend_manifest is None:
            raise ValueError("cached_image feature_mode requires a frontend manifest")
        if frontend_feature_model_id is None:
            raise ValueError("cached_image feature_mode requires a feature_model_id")
        _verify_cached_image_features(
            Path(config["frontend_cache_dir"]),
            requested_frames,
        )
    vocabulary = ReplicaVocabulary(
        classes=tuple(benchmark["vocabulary"]["classes"]),
        aliases=benchmark.get("aliases", {}),
    )
    runtime_config = _runtime_config(config)
    frontend = CachedFrontendAdapter(
        config["frontend_cache_dir"],
        vocabulary,
        voxel_size_m=runtime_config.tsdf.voxel_size_m,
        pixel_stride=int(config.get("pixel_stride", 4)),
        min_valid_points=int(config.get("min_valid_points", 10)),
        feature_model_id=frontend_feature_model_id,
    )
    structure_config = _structure_config(
        config,
        voxel_size_m=runtime_config.tsdf.voxel_size_m,
    )
    structure_frontend = DepthStructureFrontend(vocabulary, structure_config)
    runtime = Oviv2Runtime(
        str(config["scene"]),
        runtime_config,
        dense_semantic_provenance=(
            dense_cache.consumed_provenance if dense_cache is not None else None
        ),
    )
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    final = output / "final"
    checkpoints.mkdir()
    final.mkdir()

    started_wall = time.time()
    started = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 20)))
    for cache_index, source_frame_id in enumerate(source_ids):
        frame_started = time.perf_counter()
        dataset_frame = dataset[cache_index]
        frame = Frame(
            frame_id=source_frame_id,
            source_frame_id=dataset_frame.frame_id,
            rgb=dataset_frame.rgb,
            depth=dataset_frame.depth,
            pose=dataset_frame.pose,
            intrinsics=dataset_frame.intrinsics,
            timestamp=float(source_frame_id),
        )
        object_observations = frontend.observe(frame, cache_index)
        structure_observations = structure_frontend.observe(
            frame,
            object_observations=object_observations,
        )
        observations = (*object_observations, *structure_observations)
        dense_frame = (
            _load_dense_cache_frame(dense_cache, cache_index, source_frame_id)
            if dense_cache is not None
            else None
        )
        result = runtime.process_frame(frame, observations, dense_frame)
        frame_record = {
            "cache_frame_id": cache_index,
            "source_frame_id": source_frame_id,
            "revision": result.revision,
            "observation_count": result.observation_count,
            "object_observation_count": len(object_observations),
            "structure_observation_count": len(structure_observations),
            "accepted_entity_count": len(result.accepted_entity_ids),
            "matched_entity_count": result.matched_entity_count,
            "new_entity_count": result.new_entity_count,
            "association_conflict_count": result.association_conflict_count,
            "revoked_edge_count": result.revoked_edge_count,
            "elapsed_sec": time.perf_counter() - frame_started,
        }
        if dense_cache is not None:
            frame_record.update(
                {
                    "dense_sampled_pixel_count": result.dense_sampled_pixel_count,
                    "dense_valid_pixel_count": result.dense_valid_pixel_count,
                    "dense_updated_voxel_count": result.dense_updated_voxel_count,
                }
            )
        frame_records.append(frame_record)
        if (cache_index + 1) % checkpoint_interval == 0 or cache_index + 1 == requested_frames:
            runtime.commit(checkpoints / "latest_voxel_snapshot.npz")
            runtime.registry.save(checkpoints / "latest_entities.jsonl")

    snapshot = runtime.commit(final / "oviv2_voxel_snapshot.npz")
    runtime.registry.save(final / "oviv2_entities.jsonl")
    if dense_cache is None:
        mesh = derive_labeled_mesh(
            snapshot.geometry,
            snapshot.evidence,
            snapshot.ownership,
            entity_semantics=runtime.registry.semantic_labels(),
        )
        write_labeled_mesh(final / "oviv2_instance_mesh.ply", mesh)
    else:
        owner_mesh = derive_labeled_mesh(
            snapshot.geometry,
            snapshot.evidence,
            snapshot.ownership,
            entity_semantics=runtime.registry.semantic_labels(),
        )
        dense_mesh = derive_labeled_mesh(
            snapshot.geometry,
            snapshot.evidence,
            snapshot.ownership,
            entity_semantics=None,
        )
        write_labeled_mesh(final / "oviv2_owner_mesh.ply", owner_mesh)
        write_labeled_mesh(final / "oviv2_dense_mesh.ply", dense_mesh)
        if semantic_fusion is not None:
            entity_posteriors = {
                entity.entity_id: entity.semantic_posterior.probabilities
                for entity in sorted(
                    runtime.registry.entities.values(),
                    key=lambda value: value.entity_id,
                )
                if entity.lifecycle_state in {"active", "dormant"}
                and entity.semantic_id > 0
            }
            fused_mesh = derive_labeled_mesh(
                snapshot.geometry,
                snapshot.evidence,
                snapshot.ownership,
                entity_posteriors=entity_posteriors,
                semantic_fusion=semantic_fusion,
            )
            write_labeled_mesh(final / "oviv2_fused_mesh.ply", fused_mesh)
    mapping_elapsed = time.perf_counter() - started

    if not args.skip_evaluation:
        evaluation_targets = (
            (("owner_authoritative", output / "evaluation"),)
            if dense_cache is None
            else (
                ("owner_authoritative", output / "evaluation_owner"),
                ("dense_only", output / "evaluation_dense"),
                *(
                    (("fused_uncertainty", output / "evaluation_fused"),)
                    if semantic_fusion is not None
                    else ()
                ),
            )
        )
        for semantic_head, evaluation_output in evaluation_targets:
            _run_evaluation(
                config,
                final / "oviv2_voxel_snapshot.npz",
                final / "oviv2_entities.jsonl",
                evaluation_output,
                semantic_head,
            )

    timing = {
        "started_unix_sec": started_wall,
        "mapping_elapsed_sec": mapping_elapsed,
        "total_elapsed_sec": time.perf_counter() - started,
        "frame_count": requested_frames,
        "frames": frame_records,
    }
    _atomic_json(output / "timing.json", timing)
    vocabulary_payload = {
        "classes": list(vocabulary.classes),
        "aliases": dict(sorted(vocabulary.aliases.items())),
    }
    model_weights = {"frozen_frontend_cache_sha256": frontend_hash}
    dense_semantics_audit: dict[str, Any] | None = None
    if dense_cache is not None:
        model_weights.update(
            {
                "dense_model_sha256": dense_cache.producer_provenance.model_sha256,
                "dense_auxiliary_model_sha256": (
                    dense_cache.producer_provenance.auxiliary_model_sha256
                ),
                "dense_language_model_sha256": (
                    dense_cache.producer_provenance.language_model_sha256
                ),
            }
        )
        counter_names = (
            "dense_sampled_pixel_count",
            "dense_valid_pixel_count",
            "dense_updated_voxel_count",
        )
        dense_semantics_audit = {
            "mode": "cached_probabilities",
            "cache_dir": str(dense_cache.cache_dir),
            "manifest_sha256": dense_cache.manifest_sha256,
            "cache_prefix_sha256": (
                dense_cache.consumed_provenance.cache_prefix_sha256
            ),
            "producer_cache_prefix_sha256": (
                dense_cache.producer_provenance.cache_prefix_sha256
            ),
            "cache_files_sha256": {
                f"frame{cache_index:06d}.npz": dense_cache.cache_files_sha256[
                    f"frame{cache_index:06d}.npz"
                ]
                for cache_index in range(requested_frames)
            },
            "producer_cache_files_sha256": dict(dense_cache.cache_files_sha256),
            "provenance": asdict(dense_cache.consumed_provenance),
            "producer_provenance": asdict(dense_cache.producer_provenance),
            "config": asdict(runtime_config.dense_semantics),
            "counters": {
                name: sum(record[name] for record in frame_records)
                for name in counter_names
            },
        }
    run_manifest = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset_name": benchmark["dataset"],
        "scene": config["scene"],
        "repository_commit": _git_value("rev-parse", "HEAD"),
        "dirty_state_digest": starting_dirty_digest,
        "command": [str(value) for value in sys.argv],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "algorithm_hash": frozen_algorithm_hash,
        "benchmark_manifest_path": str(Path(config["manifest"]).resolve()),
        "benchmark_manifest_hash": _sha256(Path(config["manifest"])),
        "source_frame_ids_hash": _json_hash(source_ids),
        "vocabulary_hash": _json_hash(vocabulary_payload),
        "model_weights": model_weights,
        "frontend_manifest_hash": (
            _sha256(Path(config["frontend_cache_dir"]) / "frontend_manifest.json")
            if frontend_manifest is not None
            else None
        ),
        "frontend_algorithm_hash": (
            frontend_manifest.get("algorithm_hash") if frontend_manifest is not None else None
        ),
        "frontend_feature_model_id": frontend_feature_model_id,
        "semantic_mode": config.get("semantic_mode", "owner_authoritative"),
        "feature_mode": config.get("feature_mode", "cached_optional"),
        "precision_backend": {
            "association": asdict(runtime_config.tracker.association),
            "tracker": {
                "ambiguous_edge_score": runtime_config.tracker.ambiguous_edge_score,
                "third_view_min_score": runtime_config.tracker.third_view_min_score,
            },
            "memory": {
                "prototype_top_k": runtime_config.registry.prototype_top_k,
                "prototype_merge_cosine": runtime_config.registry.prototype_merge_cosine,
                "view_top_k": runtime_config.registry.view_top_k,
                "view_minimum_novelty_cosine": (
                    runtime_config.registry.view_minimum_novelty_cosine
                ),
            },
        },
        "hardware": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "open3d": o3d.__version__,
        },
        "frame_selection": {
            "source_start": source_ids[0],
            "source_stride": int(config.get("source_stride", 10)),
            "sampled_frame_count": requested_frames,
            "source_frame_ids": source_ids,
        },
        "voxel_settings": {
            "voxel_size_m": runtime_config.tsdf.voxel_size_m,
            "block_resolution": runtime_config.tsdf.block_resolution,
            "semantic_top_k": runtime_config.evidence.semantic_top_k,
            "entity_top_k": runtime_config.evidence.entity_top_k,
        },
        "structure_frontend": asdict(structure_config),
        "authoritative_state": "sparse_voxel_layers",
        "dense_point_cloud_state": False,
        "final_revision": runtime.revision,
        "entity_count": len(runtime.registry.entities),
        "geometry_block_count": runtime.geometry.active_block_count,
        "artifact_checksums": _artifact_checksums(output),
    }
    if dense_semantics_audit is not None:
        run_manifest["dense_semantics"] = dense_semantics_audit
    if semantic_fusion is not None:
        run_manifest["semantic_fusion"] = {
            "mode": "uncertainty_linear",
            **asdict(semantic_fusion),
        }
    _atomic_json(output / "run_manifest.json", run_manifest)
    return run_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": manifest["frame_selection"]["sampled_frame_count"],
                "entities": manifest["entity_count"],
                "revision": manifest["final_revision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
