from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.oviv2.dense_semantics import DenseSemanticFrame, load_dense_frame
from src.oviv2.hybrid_cache import publish_frontend_manifest, write_hybrid_frame
from src.oviv2.hybrid_frontend import (
    FrontendBatch,
    HybridFrontendConfig,
    select_hybrid_proposals,
)


REQUIRED_CLIP_SHA256 = (
    "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
)
REQUIRED_GENERATOR_SHA256 = (
    "7faf64705a80f920eb269b3c842319fc9eccb1e32eda1ed1b23dbe40ce064110"
)
REQUIRED_RUNTIME_PATCH_SHA256 = (
    "74bd99e9c0a097be802b3a30f013a8fcbbc74863bae37cb78d1e3f3cb5e9dd87"
)

_VARIANTS = ("sam_labeled", "yolo_novel_sam", "quota_nms_ensemble")
_LEGACY_KEYS = frozenset(
    {
        "xyxy",
        "confidence",
        "class_id",
        "mask",
        "classes",
        "image_crops",
        "image_feats",
        "text_feats",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")


class _LegacyUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = frozenset(
        {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("PIL.Image", "Image"),
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"unsafe pickle global is not permitted: {module}.{name}"
            )
        return super().find_class(module, name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic OVIV2 YOLO/SAM hybrid frontend cache."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=_VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int)
    return parser.parse_args(argv)


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return True
    return False


def _regular_file(path: Path, field: str) -> None:
    if _has_symlink_component(path):
        raise ValueError(f"{field} must not be a symlink or reside below one")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular file")


def _read_snapshot(path: Path, field: str) -> tuple[bytes, str]:
    _regular_file(path, field)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field} must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            snapshot = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return snapshot, hashlib.sha256(snapshot).hexdigest()


def _required_sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _json_snapshot(path: Path, field: str) -> tuple[dict[str, Any], str]:
    snapshot, digest = _read_snapshot(path, field)
    try:
        value = json.loads(snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value, digest


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return normalized


def _classes(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    normalized = tuple(_string(item, f"{field} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique names")
    return normalized


def _shape(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be [height, width]")
    return (
        _integer(value[0], f"{field}[0]", positive=True),
        _integer(value[1], f"{field}[1]", positive=True),
    )


def _hash_mapping(value: object, field: str) -> Mapping[str, Any]:
    mapping = _mapping(value, field)
    for name, digest in mapping.items():
        _string(name, f"{field} key")
        _required_sha(digest, f"{field}[{name!r}]")
    return mapping


def _cache_hash_mapping(
    value: object,
    field: str,
    *,
    frame_count: int,
    suffix: str,
) -> Mapping[str, Any]:
    mapping = _hash_mapping(value, field)
    expected_names = {
        f"frame{index:06d}{suffix}" for index in range(frame_count)
    }
    if set(mapping) != expected_names:
        raise ValueError(
            f"{field} source hash mapping must exactly cover declared frame_count"
        )
    return mapping


def _verify_snapshot(
    path: Path,
    expected_sha256: str,
    field: str,
) -> tuple[bytes, str]:
    expected = _required_sha(expected_sha256, f"{field} source hash")
    snapshot, actual = _read_snapshot(path, field)
    if actual != expected:
        raise ValueError(f"{field} checksum mismatch")
    return snapshot, actual


def _verify_materialized_depth_snapshot(
    path: Path,
    expected_sha256: str,
) -> tuple[bytes, str]:
    expected = _required_sha(expected_sha256, "depth frame source hash")
    if _has_symlink_component(path.parent):
        raise ValueError("depth frame parent must not be a symlink or reside below one")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
        raise ValueError("depth frame must be a regular file or materialized leaf symlink")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        followed = os.fstat(descriptor)
        if not stat.S_ISREG(followed.st_mode):
            raise ValueError("materialized depth target must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            snapshot = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = hashlib.sha256(snapshot).hexdigest()
    if actual != expected:
        raise ValueError("depth frame checksum mismatch")
    return snapshot, actual


def _validate_legacy_payload(
    payload: object,
    *,
    field: str,
    expected_classes: tuple[str, ...] | None,
) -> FrontendBatch:
    if not isinstance(payload, dict) or set(payload) != _LEGACY_KEYS:
        raise ValueError(f"{field} payload has invalid keys")
    classes = _classes(payload["classes"], f"{field} classes")
    if expected_classes is not None and classes != expected_classes:
        raise ValueError(f"{field} payload classes disagree with YOLO manifest classes")
    masks = payload["mask"]
    boxes = payload["xyxy"]
    confidences = payload["confidence"]
    class_ids = payload["class_id"]
    features = payload["image_feats"]
    text_features = payload["text_feats"]
    if not all(
        isinstance(value, np.ndarray)
        for value in (masks, boxes, confidences, class_ids, features, text_features)
    ):
        raise ValueError(f"{field} payload arrays must be NumPy arrays")
    if not isinstance(payload["image_crops"], list):
        raise ValueError(f"{field} image_crops must be a list")
    if masks.ndim != 3:
        raise ValueError(f"{field} mask must have shape (N, H, W)")
    count = masks.shape[0]
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError(f"{field} image_feats must be an aligned two-dimensional array")
    if len(payload["image_crops"]) != count:
        raise ValueError(f"{field} image_crops must align with mask")
    if (
        text_features.dtype.kind not in "iuf"
        or text_features.ndim != 2
        or text_features.shape != (count, features.shape[1])
        or not np.all(np.isfinite(text_features))
    ):
        raise ValueError(
            f"{field} text_feats must be a finite numeric array aligned with image_feats"
        )
    if class_ids.dtype.kind not in "iu" or class_ids.shape != (count,):
        raise ValueError(f"{field} class_id must be an aligned integer vector")
    if class_ids.size and (np.any(class_ids < 0) or np.any(class_ids >= len(classes))):
        raise ValueError(f"{field} class_id lies outside classes")
    if confidences.dtype.kind not in "iuf" or confidences.shape != (count,):
        raise ValueError(f"{field} confidence must be an aligned numeric vector")
    if not np.all(np.isfinite(confidences)) or np.any(confidences < 0.0):
        raise ValueError(f"{field} confidence must be finite and non-negative")
    # Canonical SAM cache scores have a documented, small overshoot above one.
    if np.any(confidences > 1.1):
        raise ValueError(f"{field} confidence exceeds canonical tolerance")
    normalized_confidences = np.clip(confidences, 0.0, 1.0)
    labels = tuple(classes[int(index)] for index in class_ids)
    try:
        return FrontendBatch(
            masks=masks,
            boxes_xyxy=boxes,
            confidences=normalized_confidences,
            labels=labels,
            image_features=features,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} payload is invalid: {exc}") from exc


def _load_legacy_snapshot(
    snapshot: bytes,
    *,
    field: str,
    expected_classes: tuple[str, ...] | None = None,
) -> FrontendBatch:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(snapshot), mode="rb") as compressed:
            payload = _LegacyUnpickler(compressed).load()
            if compressed.read(1):
                raise ValueError(f"{field} payload has trailing data")
    except ValueError:
        raise
    except pickle.UnpicklingError as exc:
        raise ValueError(f"{field} payload rejected: {exc}") from exc
    except (EOFError, OSError, pickle.PickleError, AttributeError, ImportError) as exc:
        raise ValueError(f"{field} payload is invalid gzip or pickle data") from exc
    return _validate_legacy_payload(
        payload,
        field=field,
        expected_classes=expected_classes,
    )


def _load_legacy(
    path: Path,
    expected_sha256: str,
    *,
    field: str,
    expected_classes: tuple[str, ...] | None = None,
) -> FrontendBatch:
    snapshot, _ = _verify_snapshot(path, expected_sha256, field)
    return _load_legacy_snapshot(
        snapshot,
        field=field,
        expected_classes=expected_classes,
    )


def _load_depth(path: Path, expected_sha256: str, image_shape: tuple[int, int]) -> np.ndarray:
    snapshot, _ = _verify_materialized_depth_snapshot(path, expected_sha256)
    try:
        with Image.open(io.BytesIO(snapshot)) as image:
            depth = np.array(image, copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError("depth frame is not a valid image") from exc
    if depth.ndim != 2 or depth.shape != image_shape:
        raise ValueError("depth image shape disagrees with frontend image shape")
    return depth != 0


def _required_frame_hash(
    hashes: Mapping[str, Any], name: str, field: str
) -> str:
    if name not in hashes:
        raise ValueError(f"missing source hash for selected {field} {name}")
    return _required_sha(hashes[name], f"{field} source hash for {name}")


def _source_ids(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(_integer(item, f"{field} item") for item in value)


def _asset_hash(path: Path, expected: str, field: str) -> str:
    _regular_file(path, field)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field} must be a regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            while block := stream.read(1024 * 1024):
                digest.update(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{field} hash does not match the required production hash")
    return actual


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    if _has_symlink_component(output.parent):
        raise ValueError("output parent must not be a symlink or reside below one")

    config_path = Path(args.config)
    config, config_hash = _json_snapshot(config_path, "runner config")
    scene = _string(config.get("scene"), "scene")
    dataset_root = Path(_string(config.get("dataset_root"), "dataset_root"))
    dense_dir = Path(_string(config.get("dense_cache_dir"), "dense_cache_dir"))
    benchmark_path = Path(_string(config.get("manifest"), "manifest"))
    source_start = _integer(config.get("source_start"), "source_start")
    source_stride = _integer(config.get("source_stride"), "source_stride", positive=True)
    hybrid = _mapping(config.get("hybrid_frontend"), "hybrid_frontend")
    configured_variant = _string(hybrid.get("variant"), "hybrid_frontend.variant")
    if configured_variant != args.variant:
        raise ValueError("CLI variant disagrees with hybrid_frontend.variant")
    for variant in _VARIANTS:
        if variant not in hybrid:
            raise ValueError(f"hybrid_frontend is missing named policy {variant}")
    raw_policy = dict(_mapping(hybrid[args.variant], f"hybrid_frontend.{args.variant}"))
    if "variant" in raw_policy and raw_policy["variant"] != args.variant:
        raise ValueError("named policy variant disagrees with CLI variant")
    raw_policy["variant"] = args.variant
    policy = HybridFrontendConfig(**raw_policy)

    benchmark, benchmark_hash = _json_snapshot(benchmark_path, "benchmark manifest")
    vocabulary = _mapping(benchmark.get("vocabulary"), "benchmark vocabulary")
    vocabulary_sha = _required_sha(
        vocabulary.get("source_sha256"), "benchmark vocabulary source_sha256"
    )
    class_names = _classes(vocabulary.get("classes"), "benchmark vocabulary classes")
    scenes = benchmark.get("scenes")
    if not isinstance(scenes, list) or not any(
        isinstance(record, Mapping) and record.get("scene") == scene for record in scenes
    ):
        raise ValueError("scene is absent from benchmark manifest")

    yolo_dir = Path(_string(hybrid.get("yolo_cache_dir"), "yolo_cache_dir"))
    sam_dir = Path(_string(hybrid.get("sam_cache_dir"), "sam_cache_dir"))
    yolo_manifest_path = yolo_dir / "frontend_manifest.json"
    dense_manifest_path = dense_dir / "dense_manifest.json"
    frame_manifest_path = dataset_root / "frame_manifest.json"
    yolo_manifest, yolo_manifest_hash = _json_snapshot(
        yolo_manifest_path, "YOLO manifest"
    )
    dense_manifest, dense_manifest_hash = _json_snapshot(
        dense_manifest_path, "dense manifest"
    )
    frame_manifest, frame_manifest_hash = _json_snapshot(
        frame_manifest_path, "frame manifest"
    )
    gate_path = Path(_string(hybrid.get("sam_gate"), "sam_gate"))
    gate, gate_hash = _json_snapshot(gate_path, "SAM gate")
    if gate.get("status") != "VERIFIED" or gate.get("headline_result_eligible") is not True:
        raise ValueError("SAM gate is not VERIFIED and headline eligible")

    checkpoint = Path(_string(hybrid.get("clip_checkpoint"), "clip_checkpoint"))
    generator = Path(
        _string(hybrid.get("sam_generator_script"), "sam_generator_script")
    )
    runtime_patch = Path(
        _string(hybrid.get("sam_runtime_patch"), "sam_runtime_patch")
    )
    checkpoint_hash = _asset_hash(
        checkpoint, REQUIRED_CLIP_SHA256, "CLIP checkpoint"
    )
    generator_hash = _asset_hash(
        generator, REQUIRED_GENERATOR_SHA256, "SAM generator script"
    )
    runtime_hash = _asset_hash(
        runtime_patch, REQUIRED_RUNTIME_PATCH_SHA256, "SAM runtime patch"
    )
    yolo_provenance = _mapping(
        yolo_manifest.get("provenance_sha256"), "YOLO provenance_sha256"
    )
    if yolo_provenance.get("clip_model") != REQUIRED_CLIP_SHA256:
        raise ValueError("YOLO manifest CLIP hash disagrees with required checkpoint")

    for name, manifest in (("YOLO", yolo_manifest), ("dense", dense_manifest)):
        if manifest.get("scene") != scene:
            raise ValueError(f"{name} manifest scene mismatch")
    yolo_shape = _shape(yolo_manifest.get("mask_shape"), "YOLO mask_shape")
    dense_shape = _shape(dense_manifest.get("image_shape"), "dense image_shape")
    if yolo_shape != dense_shape:
        raise ValueError("cross-source image shape mismatch")
    yolo_classes = _classes(yolo_manifest.get("classes"), "YOLO manifest classes")
    if dense_manifest.get("vocabulary_sha256") != vocabulary_sha:
        raise ValueError("dense vocabulary hash disagrees with benchmark vocabulary")
    if _integer(dense_manifest.get("class_count"), "dense class_count", positive=True) != len(class_names):
        raise ValueError("dense class_count disagrees with benchmark vocabulary")

    yolo_count = _integer(
        yolo_manifest.get("frame_count"), "YOLO frame_count", positive=True
    )
    dense_count = _integer(
        dense_manifest.get("frame_count"), "dense frame_count", positive=True
    )
    source_count = _integer(
        frame_manifest.get("frame_count"), "frame frame_count", positive=True
    )
    if len({yolo_count, dense_count, source_count}) != 1:
        raise ValueError("YOLO, dense, and frame manifest frame_count values disagree")
    available = yolo_count
    if args.num_frames is None:
        frame_count = available
    else:
        frame_count = _integer(args.num_frames, "num_frames", positive=True)
        if frame_count > available:
            raise ValueError("num_frames exceeds available verified frames")
    yolo_hashes = _cache_hash_mapping(
        yolo_manifest.get("cache_files_sha256"),
        "YOLO cache_files_sha256",
        frame_count=yolo_count,
        suffix=".pkl.gz",
    )
    dense_hashes = _cache_hash_mapping(
        dense_manifest.get("cache_files_sha256"),
        "dense cache_files_sha256",
        frame_count=dense_count,
        suffix=".npz",
    )
    dense_source_ids = _source_ids(
        dense_manifest.get("source_frame_ids"), "dense source_frame_ids"
    )
    frame_source_ids = _source_ids(
        frame_manifest.get("source_frame_ids"), "frame manifest source_frame_ids"
    )
    frame_records = frame_manifest.get("frames")
    if not isinstance(frame_records, list):
        raise ValueError("frame manifest frames must be a list")
    if len(dense_source_ids) != dense_count:
        raise ValueError("dense source_frame_ids must equal declared frame_count")
    if len(frame_source_ids) != source_count:
        raise ValueError(
            "frame manifest source_frame_ids must equal declared frame_count"
        )
    if len(frame_records) != source_count:
        raise ValueError("frame records must equal declared frame_count")

    source_ids: list[int] = []
    source_hashes: dict[str, str] = {
        "config/runner.json": config_hash,
        "manifests/benchmark.json": benchmark_hash,
        "manifests/dense_manifest.json": dense_manifest_hash,
        "manifests/frame_manifest.json": frame_manifest_hash,
        "manifests/yolo_frontend_manifest.json": yolo_manifest_hash,
        "assets/clip_checkpoint": checkpoint_hash,
        "assets/sam_gate.json": gate_hash,
        "assets/sam_generator_script": generator_hash,
        "assets/sam_runtime_patch": runtime_hash,
    }
    selected: list[dict[str, Any]] = []
    for cache_index in range(frame_count):
        source_id = source_start + cache_index * source_stride
        source_ids.append(source_id)
        if dense_source_ids[cache_index] != source_id:
            raise ValueError(f"dense source_frame_id mismatch at cache index {cache_index}")
        if frame_source_ids[cache_index] != source_id:
            raise ValueError(
                f"frame manifest source_frame_ids mismatch at cache index {cache_index}"
            )
        record = frame_records[cache_index]
        if not isinstance(record, Mapping):
            raise ValueError("frame manifest frame record must be an object")
        if _integer(record.get("sampled_frame_id"), "sampled_frame_id") != cache_index:
            raise ValueError("frame manifest sampled_frame_id mismatch")
        if _integer(record.get("source_frame_id"), "source_frame_id") != source_id:
            raise ValueError("frame manifest source_frame_id mismatch")

        yolo_name = f"frame{cache_index:06d}.pkl.gz"
        dense_name = f"frame{cache_index:06d}.npz"
        sam_name = f"frame{source_id:06d}.pkl.gz"
        depth_name = f"depth{cache_index:06d}.png"
        yolo_hash = _required_frame_hash(yolo_hashes, yolo_name, "YOLO")
        dense_hash = _required_frame_hash(dense_hashes, dense_name, "dense")
        depth_hash = _required_sha(
            record.get("depth_sha256"), f"depth source hash for {depth_name}"
        )
        yolo_path = yolo_dir / yolo_name
        dense_path = dense_dir / dense_name
        sam_path = sam_dir / sam_name
        depth_path = dataset_root / "results" / depth_name
        yolo_snapshot, _ = _verify_snapshot(yolo_path, yolo_hash, "YOLO frame")
        dense_snapshot, _ = _verify_snapshot(dense_path, dense_hash, "dense frame")
        _verify_materialized_depth_snapshot(depth_path, depth_hash)
        sam_snapshot, sam_hash = _read_snapshot(sam_path, "SAM frame")
        source_hashes[f"yolo/{yolo_name}"] = yolo_hash
        source_hashes[f"dense/{dense_name}"] = dense_hash
        source_hashes[f"sam/{sam_name}"] = sam_hash
        source_hashes[f"depth/{depth_name}"] = depth_hash

        yolo_batch = _load_legacy_snapshot(
            yolo_snapshot, field="YOLO frame", expected_classes=yolo_classes
        )
        sam_batch = _load_legacy_snapshot(sam_snapshot, field="SAM frame")
        dense = load_dense_frame(dense_path, expected_sha256=dense_hash)
        valid_depth = _load_depth(depth_path, depth_hash, dense_shape)
        if (
            dense.cache_frame_id != cache_index
            or dense.source_frame_id != source_id
        ):
            raise ValueError("dense frame source_frame_id or cache_frame_id mismatch")
        if dense.image_shape != dense_shape:
            raise ValueError("dense frame image shape disagrees with dense manifest")
        if yolo_batch.masks.shape[1:] != dense_shape or sam_batch.masks.shape[1:] != dense_shape:
            raise ValueError("legacy frontend image shape disagrees with manifests")
        if yolo_batch.image_features.shape[1] != sam_batch.image_features.shape[1]:
            raise ValueError("YOLO and SAM feature dimensions disagree")
        selected.append(
            {
                "yolo": yolo_path,
                "yolo_hash": yolo_hash,
                "sam": sam_path,
                "sam_hash": sam_hash,
                "dense": dense_path,
                "dense_hash": dense_hash,
                "depth": depth_path,
                "depth_hash": depth_hash,
            }
        )

    output_classes = tuple(dict.fromkeys((*yolo_classes, *class_names)))
    structure_ids = {
        index + 1
        for index, name in enumerate(class_names)
        if name in {"wall", "floor", "ceiling"}
    }
    return {
        "output": output,
        "scene": scene,
        "frame_count": frame_count,
        "source_ids": source_ids,
        "source_hashes": source_hashes,
        "selected": selected,
        "yolo_classes": yolo_classes,
        "classes": output_classes,
        "class_names": class_names,
        "structure_ids": structure_ids,
        "image_shape": dense_shape,
        "policy": policy,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args, argparse.Namespace):
        raise ValueError("args must be an argparse.Namespace")
    preflight = _preflight(args)
    output: Path = preflight["output"]
    output.mkdir(parents=True, exist_ok=False)
    cache_hashes: dict[str, str] = {}
    diagnostics: dict[str, int] = {}
    try:
        for cache_index, record in enumerate(preflight["selected"]):
            yolo = _load_legacy(
                record["yolo"],
                record["yolo_hash"],
                field="YOLO frame",
                expected_classes=preflight["yolo_classes"],
            )
            sam = _load_legacy(
                record["sam"], record["sam_hash"], field="SAM frame"
            )
            dense: DenseSemanticFrame = load_dense_frame(
                record["dense"], expected_sha256=record["dense_hash"]
            )
            valid_depth = _load_depth(
                record["depth"], record["depth_hash"], preflight["image_shape"]
            )
            result = select_hybrid_proposals(
                yolo=yolo,
                sam=sam,
                dense=dense,
                valid_depth=valid_depth,
                class_names=preflight["class_names"],
                structure_ids=preflight["structure_ids"],
                config=preflight["policy"],
            )
            name = f"frame{cache_index:06d}.pkl.gz"
            cache_hashes[name] = write_hybrid_frame(
                output / name, result.batch, preflight["classes"]
            )
            for key, value in result.diagnostics.items():
                diagnostics[key] = diagnostics.get(key, 0) + int(value)

        return publish_frontend_manifest(
            output_dir=output,
            scene=preflight["scene"],
            frame_count=preflight["frame_count"],
            source_frame_ids=preflight["source_ids"],
            classes=preflight["classes"],
            feature_model_id=f"clip-sha256:{REQUIRED_CLIP_SHA256}",
            cache_files_sha256=cache_hashes,
            source_sha256=preflight["source_hashes"],
            policy=asdict(preflight["policy"]),
            diagnostics=diagnostics,
        )
    except BaseException:
        shutil.rmtree(output)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
