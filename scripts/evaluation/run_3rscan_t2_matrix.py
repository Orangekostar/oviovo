#!/usr/bin/env python3
"""Run and aggregate the controlled G/F/R matrix on frozen 3RScan pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_ARRAY_KEYS = {
    "pred_masks_nk",
    "pred_scores_k",
    "pred_classes_k",
    "target_masks_in",
    "target_labels_i",
    "target_changes_i",
    "target_ids_i",
    "target_temporal_stages_n",
    "raw_masks_sq",
    "raw_logits_qc",
    "inverse_n",
    "lowres_point2segment_m",
    "lowres_visit_ids_m",
    "full_visit_ids_n",
    "full_original_segment_ids_n",
    "backbone_features_mf",
}
_MANIFEST_KEYS = {
    "schema_version",
    "artifact_id",
    "status",
    "pair_id",
    "method_input_sha256",
    "checkpoint_sha256",
    "source_commit",
    "arrays",
    "official_metrics",
    "target_ambiguities",
    "runtime",
    "raw_forward_reuse",
}
_CONFIG_KEYS = {
    "schema_version",
    "config_id",
    "seed",
    "neural_voxel_size_m",
    "selection_manifest",
    "runtime",
    "domains",
    "methods",
    "method_parameters",
}
_RUNTIME_KEYS = {
    "python",
    "rescene_checkout",
    "rescene_commit",
    "runtime_workdir",
    "checkpoint",
    "concerto_checkpoint",
    "processed_root",
    "dataset_spec",
    "label_database",
    "change_label_database",
    "color_mean_std",
    "validation_database",
    "sequence_database",
    "gt_root",
    "output_root",
}


class MatrixRunError(ValueError):
    """Raised when matrix inputs, outputs, or provenance are invalid."""


def _readonly(value: object, dtype: np.dtype | type, ndim: int) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise MatrixRunError(f"array must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _digest(value: object, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise MatrixRunError(f"{label} is invalid")
    return value


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixRunError(f"{label} must be a non-empty string")
    return value.strip()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _finite_metric_map(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise MatrixRunError("official metrics must be a non-empty mapping")
    metrics: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _nonempty(raw_key, label="official metric key")
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float, np.number)
        ):
            raise MatrixRunError("official metric values must be numeric")
        metric = float(raw_value)
        if not math.isfinite(metric):
            raise MatrixRunError("official metric values must be finite")
        metrics[key] = metric
    classify_official_metric_keys(metrics)
    return MappingProxyType(dict(sorted(metrics.items())))


def classify_official_metric_keys(
    metrics: Mapping[str, object],
) -> Mapping[str, tuple[str, ...]]:
    """Partition official stmetrics keys without mixing custom association values."""

    if not isinstance(metrics, Mapping) or not metrics:
        raise MatrixRunError("official metrics must be a non-empty mapping")
    groups: dict[str, list[str]] = {
        "temporal_ap": [],
        "temporal_recall": [],
        "legacy_ap": [],
        "stage_ap": [],
    }
    for raw_key in metrics:
        key = _nonempty(raw_key, label="official metric key")
        if "_t-AP" in key:
            groups["temporal_ap"].append(key)
        elif "_t-REC" in key:
            groups["temporal_recall"].append(key)
        elif re.search(r"_stage\d+(?:-\d+)?-AP(?:_|$)", key):
            groups["stage_ap"].append(key)
        elif re.search(r"_AP(?:_|$)", key):
            groups["legacy_ap"].append(key)
        else:
            raise MatrixRunError(
                f"custom or unknown metric cannot enter official namespace: {key}"
            )
    return MappingProxyType(
        {name: tuple(sorted(keys)) for name, keys in groups.items()}
    )


def _normalized_arrays(value: object) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != _ARRAY_KEYS:
        raise MatrixRunError("native array schema is invalid")
    arrays = {
        "pred_masks_nk": _readonly(value["pred_masks_nk"], np.uint8, 2),
        "pred_scores_k": _readonly(value["pred_scores_k"], np.float32, 1),
        "pred_classes_k": _readonly(value["pred_classes_k"], np.int64, 1),
        "target_masks_in": _readonly(value["target_masks_in"], np.uint8, 2),
        "target_labels_i": _readonly(value["target_labels_i"], np.int64, 1),
        "target_changes_i": _readonly(value["target_changes_i"], np.int64, 1),
        "target_ids_i": _readonly(value["target_ids_i"], np.int64, 1),
        "target_temporal_stages_n": _readonly(
            value["target_temporal_stages_n"], np.int64, 1
        ),
        "raw_masks_sq": _readonly(value["raw_masks_sq"], np.float32, 2),
        "raw_logits_qc": _readonly(value["raw_logits_qc"], np.float32, 2),
        "inverse_n": _readonly(value["inverse_n"], np.int64, 1),
        "lowres_point2segment_m": _readonly(
            value["lowres_point2segment_m"], np.int64, 1
        ),
        "lowres_visit_ids_m": _readonly(value["lowres_visit_ids_m"], np.int8, 1),
        "full_visit_ids_n": _readonly(value["full_visit_ids_n"], np.int8, 1),
        "full_original_segment_ids_n": _readonly(
            value["full_original_segment_ids_n"], np.int64, 1
        ),
        "backbone_features_mf": _readonly(
            value["backbone_features_mf"], np.float32, 2
        ),
    }
    if any(
        not np.all(np.isfinite(array))
        for name, array in arrays.items()
        if name
        in {
            "pred_scores_k",
            "raw_masks_sq",
            "raw_logits_qc",
            "backbone_features_mf",
        }
    ):
        raise MatrixRunError("native floating arrays must be finite")
    n = len(arrays["full_visit_ids_n"])
    k = len(arrays["pred_scores_k"])
    i = len(arrays["target_ids_i"])
    m = len(arrays["lowres_point2segment_m"])
    s, q = arrays["raw_masks_sq"].shape
    if n == 0 or m == 0 or s == 0 or q == 0:
        raise MatrixRunError("native point, segment, and query domains must be non-empty")
    if arrays["pred_masks_nk"].shape != (n, k) or len(
        arrays["pred_classes_k"]
    ) != k:
        raise MatrixRunError("official prediction domains are misaligned")
    if arrays["target_masks_in"].shape != (i, n) or any(
        len(arrays[name]) != i
        for name in ("target_labels_i", "target_changes_i")
    ):
        raise MatrixRunError("target instance domains are misaligned")
    if len(arrays["target_temporal_stages_n"]) != n:
        raise MatrixRunError("target temporal stage domain is misaligned")
    if arrays["raw_logits_qc"].shape[0] != q or arrays["raw_logits_qc"].shape[1] < 2:
        raise MatrixRunError("raw query domains are misaligned")
    if any(
        len(arrays[name]) != n
        for name in ("inverse_n", "full_original_segment_ids_n")
    ):
        raise MatrixRunError("full-resolution domains are misaligned")
    if arrays["backbone_features_mf"].shape[0] != m or len(
        arrays["lowres_visit_ids_m"]
    ) != m:
        raise MatrixRunError("low-resolution feature domains are misaligned")
    if np.any(arrays["inverse_n"] < 0) or np.any(arrays["inverse_n"] >= m):
        raise MatrixRunError("inverse mapping is outside the low-resolution domain")
    segments = arrays["lowres_point2segment_m"]
    if np.any(segments < 0) or np.any(segments >= s) or not np.array_equal(
        np.unique(segments), np.arange(s)
    ):
        raise MatrixRunError("raw segment domain differs from point2segment")
    if np.any(arrays["full_original_segment_ids_n"] < 0):
        raise MatrixRunError("original mesh segments must be non-negative")
    if set(arrays["full_visit_ids_n"].tolist()) != {0, 1} or set(
        arrays["lowres_visit_ids_m"].tolist()
    ) != {0, 1}:
        raise MatrixRunError("native arrays must retain both visits")
    if not np.array_equal(
        arrays["target_temporal_stages_n"],
        arrays["full_visit_ids_n"].astype(np.int64),
    ):
        raise MatrixRunError("target temporal stages differ from method visits")
    if np.any(arrays["pred_masks_nk"] > 1) or np.any(
        arrays["target_masks_in"] > 1
    ):
        raise MatrixRunError("prediction and target masks must be binary")
    return MappingProxyType(dict(sorted(arrays.items())))


def _ambiguities(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (tuple, list)):
        raise MatrixRunError("target ambiguities must be a sequence")
    groups: list[tuple[int, ...]] = []
    for raw_group in value:
        if not isinstance(raw_group, (tuple, list)):
            raise MatrixRunError("target ambiguity group must be a sequence")
        group = tuple(int(item) for item in raw_group)
        if len(group) < 2 or any(item <= 0 for item in group):
            raise MatrixRunError("target ambiguity group must contain positive IDs")
        groups.append(tuple(sorted(set(group))))
    normalized = tuple(sorted(set(groups)))
    if len(normalized) != len(groups):
        raise MatrixRunError("target ambiguity groups must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class NativePairArtifact:
    pair_id: str
    method_input_sha256: str
    checkpoint_sha256: str
    source_commit: str
    arrays: Mapping[str, np.ndarray]
    official_metrics: Mapping[str, float]
    target_ambiguities: tuple[tuple[int, ...], ...]
    runtime_s: float
    peak_gpu_bytes: int
    peak_reserved_gpu_bytes: int
    peak_rss_bytes: int
    device_name: str

    def __post_init__(self) -> None:
        pair_id = _nonempty(self.pair_id, label="pair_id")
        method_hash = _digest(
            self.method_input_sha256,
            label="method input SHA-256",
            pattern=_SHA256,
        )
        checkpoint = _digest(
            self.checkpoint_sha256, label="checkpoint SHA-256", pattern=_SHA256
        )
        source = _digest(self.source_commit, label="source commit", pattern=_GIT_SHA)
        arrays = _normalized_arrays(self.arrays)
        metrics = _finite_metric_map(self.official_metrics)
        ambiguities = _ambiguities(self.target_ambiguities)
        runtime = float(self.runtime_s)
        if not math.isfinite(runtime) or runtime < 0.0:
            raise MatrixRunError("runtime_s must be finite and non-negative")
        for name in (
            "peak_gpu_bytes",
            "peak_reserved_gpu_bytes",
            "peak_rss_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise MatrixRunError(f"{name} must be a non-negative integer")
        device = _nonempty(self.device_name, label="device_name")
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "method_input_sha256", method_hash)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(self, "source_commit", source)
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "official_metrics", metrics)
        object.__setattr__(self, "target_ambiguities", ambiguities)
        object.__setattr__(self, "runtime_s", runtime)
        object.__setattr__(self, "device_name", device)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _regular_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise MatrixRunError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise MatrixRunError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise MatrixRunError(f"{label} changed while being read")
    return content


def _file_record(path: Path, *, recorded_path: str) -> dict[str, object]:
    content = _regular_bytes(path, label=path.name)
    return {
        "path": recorded_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def publish_native_pair_artifact(
    artifact: NativePairArtifact, output_root: str | Path
) -> dict[str, object]:
    """Atomically publish one no-clobber native forward and evaluator sidecar."""

    if not isinstance(artifact, NativePairArtifact):
        raise TypeError("artifact must be a NativePairArtifact")
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise MatrixRunError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **artifact.arrays)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "RSCAN_T2_NATIVE_PAIR_V1",
            "status": "PASS",
            "pair_id": artifact.pair_id,
            "method_input_sha256": artifact.method_input_sha256,
            "checkpoint_sha256": artifact.checkpoint_sha256,
            "source_commit": artifact.source_commit,
            "arrays": _file_record(arrays_path, recorded_path="arrays.npz"),
            "official_metrics": dict(artifact.official_metrics),
            "target_ambiguities": [list(group) for group in artifact.target_ambiguities],
            "runtime": {
                "forward_s": artifact.runtime_s,
                "peak_gpu_bytes": artifact.peak_gpu_bytes,
                "peak_reserved_gpu_bytes": artifact.peak_reserved_gpu_bytes,
                "peak_rss_bytes": artifact.peak_rss_bytes,
                "device_name": artifact.device_name,
            },
            "raw_forward_reuse": ["F", "R_legacy", "R_supported"],
        }
        _write_exclusive(staging / "manifest.json", _json_bytes(manifest))
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_native_pair_artifact(output_root: str | Path) -> NativePairArtifact:
    """Load and independently revalidate one native pair artifact."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    try:
        manifest = json.loads(
            _regular_bytes(manifest_path, label="native manifest").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatrixRunError("native manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise MatrixRunError("native manifest schema is invalid")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "RSCAN_T2_NATIVE_PAIR_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("raw_forward_reuse")
        != ["F", "R_legacy", "R_supported"]
    ):
        raise MatrixRunError("native manifest identity is invalid")
    expected_record = _file_record(arrays_path, recorded_path="arrays.npz")
    if manifest.get("arrays") != expected_record:
        raise MatrixRunError("native arrays binding mismatch")
    try:
        with np.load(io.BytesIO(_regular_bytes(arrays_path, label="native arrays")), allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise MatrixRunError("native array schema is invalid")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, MatrixRunError):
            raise
        raise MatrixRunError("native arrays cannot be decoded") from error
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "forward_s",
        "peak_gpu_bytes",
        "peak_reserved_gpu_bytes",
        "peak_rss_bytes",
        "device_name",
    }:
        raise MatrixRunError("native runtime schema is invalid")
    return NativePairArtifact(
        pair_id=manifest.get("pair_id"),
        method_input_sha256=manifest.get("method_input_sha256"),
        checkpoint_sha256=manifest.get("checkpoint_sha256"),
        source_commit=manifest.get("source_commit"),
        arrays=arrays,
        official_metrics=manifest.get("official_metrics"),
        target_ambiguities=manifest.get("target_ambiguities"),
        runtime_s=runtime.get("forward_s"),
        peak_gpu_bytes=runtime.get("peak_gpu_bytes"),
        peak_reserved_gpu_bytes=runtime.get("peak_reserved_gpu_bytes"),
        peak_rss_bytes=runtime.get("peak_rss_bytes"),
        device_name=runtime.get("device_name"),
    )


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        from src.evaluation.json_contracts import loads_strict

        value = loads_strict(
            _regular_bytes(path, label=label).decode("utf-8"), label=label
        )
    except (UnicodeDecodeError, ValueError) as error:
        if isinstance(error, MatrixRunError):
            raise
        raise MatrixRunError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise MatrixRunError(f"{label} must contain a JSON object")
    return value


def _validated_external_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise MatrixRunError(f"{label} binding schema is invalid")
    path_value = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise MatrixRunError(f"{label} binding value is invalid")
    path = Path(os.path.abspath(path_value))
    content = _regular_bytes(path, label=label)
    if len(content) != byte_count or hashlib.sha256(content).hexdigest() != digest:
        raise MatrixRunError(f"{label} binding mismatch")
    return path


def _validated_directory(path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, (str, os.PathLike)) or not Path(
        path_value
    ).is_absolute():
        raise MatrixRunError(f"{label} must be an absolute directory")
    path = Path(os.path.abspath(path_value))
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise MatrixRunError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(mode):
        raise MatrixRunError(f"{label} must be a non-symlink directory")
    return path


def _load_config(path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    config = _load_json(path, label="matrix config")
    if set(config) != _CONFIG_KEYS or config.get("schema_version") != 1:
        raise MatrixRunError("matrix config schema is invalid")
    if config.get("config_id") != "OVI_RESCENE_3RSCAN_T2_MATRIX_V1":
        raise MatrixRunError("matrix config identity is invalid")
    if config.get("seed") != 45 or not math.isclose(
        float(config.get("neural_voxel_size_m", -1.0)), 0.02, abs_tol=1e-12
    ):
        raise MatrixRunError("matrix seed or voxel size differs from the frozen protocol")
    selection_path = _validated_external_record(
        config.get("selection_manifest"), label="selection manifest"
    )
    selection = _load_json(selection_path, label="selection manifest")
    if selection.get("artifact_id") != "RSCAN_T2_DEV_V1":
        raise MatrixRunError("selection manifest identity is invalid")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_KEYS:
        raise MatrixRunError("matrix runtime schema is invalid")
    domains = config.get("domains")
    if domains != {
        "D0_NATIVE_PROCESSED": "PASS",
        "D1_NATIVE_SENSOR_SUPPORT": "MISSING_ASSET",
        "D2_OVI_RECONSTRUCTION": "MISSING_ASSET",
    }:
        raise MatrixRunError("matrix domain availability differs from frozen evidence")
    methods = config.get("methods")
    if methods != ["G_full", "G_supported", "F", "R_legacy", "R_supported"]:
        raise MatrixRunError("matrix method set is invalid")
    _method_configs(config.get("method_parameters"))
    return config, selection


def _method_configs(value: object) -> tuple[object, object, float, float]:
    from src.oviv2.geometric_pair_reasoner import GeometricReasonerConfig
    from src.oviv2.query_instance_projection import ProjectionConfig

    if not isinstance(value, Mapping) or set(value) != {
        "geometric_reasoner",
        "projection",
        "supported_resolver",
        "association_metrics",
    }:
        raise MatrixRunError("matrix method parameter schema is invalid")
    geometric = value["geometric_reasoner"]
    projection = value["projection"]
    resolver = value["supported_resolver"]
    association = value["association_metrics"]
    if not isinstance(geometric, Mapping) or set(geometric) != {
        "comparison_voxel_size_m",
        "maximum_centroid_distance_m",
        "minimum_pair_score",
        "require_known_label_compatibility",
        "voxel_iou_weight",
        "symmetric_coverage_weight",
        "centroid_weight",
        "size_weight",
        "extent_weight",
        "semantic_weight",
    }:
        raise MatrixRunError("geometric reasoner parameter schema is invalid")
    if not isinstance(projection, Mapping) or set(projection) != {
        "minimum_entity_token_coverage",
        "minimum_source_point_coverage",
        "static_centroid_tolerance_m",
    }:
        raise MatrixRunError("projection parameter schema is invalid")
    if not isinstance(resolver, Mapping) or set(resolver) != {
        "minimum_candidate_score"
    }:
        raise MatrixRunError("supported resolver parameter schema is invalid")
    if not isinstance(association, Mapping) or set(association) != {
        "minimum_gt_observed_fraction"
    }:
        raise MatrixRunError("association metric parameter schema is invalid")
    try:
        geometric_config = GeometricReasonerConfig(**geometric)
        projection_config = ProjectionConfig(**projection)
        resolver_score = float(resolver["minimum_candidate_score"])
        observed_fraction = float(association["minimum_gt_observed_fraction"])
    except (TypeError, ValueError) as error:
        raise MatrixRunError("matrix method parameters are invalid") from error
    if not 0.0 <= resolver_score <= 1.0 or not 0.0 < observed_fraction <= 1.0:
        raise MatrixRunError("matrix method thresholds are outside their domains")
    return geometric_config, projection_config, resolver_score, observed_fraction


def _validated_checkout(runtime: Mapping[str, object]) -> tuple[Path, str]:
    checkout = _validated_directory(
        runtime.get("rescene_checkout"), label="ReScene checkout"
    )
    expected = _digest(
        runtime.get("rescene_commit"), label="ReScene commit", pattern=_GIT_SHA
    )
    try:
        observed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise MatrixRunError("ReScene checkout cannot be audited") from error
    if observed != expected or dirty:
        raise MatrixRunError("ReScene checkout identity mismatch")
    return checkout, observed


def _pair_record(selection: Mapping[str, object], pair_id: str) -> Mapping[str, object]:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list):
        raise MatrixRunError("selection pairs are invalid")
    matches = [value for value in pairs if isinstance(value, Mapping) and value.get("pair_id") == pair_id]
    if len(matches) != 1:
        raise MatrixRunError("requested pair is absent or duplicated")
    return matches[0]


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_tensors(value: object, device: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        for key, item in list(value.items()):
            value[key] = _move_tensors(item, device)  # type: ignore[index]
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_tensors(item, device)
        return value
    return value


def _expected_full_method_arrays(pair: object) -> tuple[np.ndarray, ...]:
    visits = pair.visits
    points = np.concatenate([visit.points_xyz for visit in visits], axis=0)
    rgb = np.concatenate([visit.rgb for visit in visits], axis=0)
    normals = np.concatenate([visit.normals_xyz for visit in visits], axis=0)
    visit_ids = np.concatenate(
        [np.full(visit.point_count, visit.visit_id, dtype=np.int8) for visit in visits]
    )
    first_segments = visits[0].segment_ids
    second_segments = visits[1].segment_ids + int(np.max(first_segments)) + 1
    segments = np.concatenate((first_segments, second_segments)).astype(np.int64)
    return points, rgb, normals, visit_ids, segments


def _audit_native_sample(pair: object, sample: object, data: object) -> np.ndarray:
    points, rgb, normals, visits, segments = _expected_full_method_arrays(pair)
    raw_coordinates = np.asarray(sample[6])
    raw_rgb = np.asarray(sample[4])
    raw_normals = np.asarray(sample[5])
    if (
        raw_coordinates.shape != (len(points), 4)
        or not np.array_equal(raw_coordinates[:, :3], points)
        or not np.array_equal(raw_coordinates[:, 3].astype(np.int8), visits)
        or not np.array_equal(raw_rgb, rgb)
        or not np.array_equal(raw_normals, normals)
    ):
        raise MatrixRunError("native collator method arrays differ from the GT-free view")
    full_segments = np.asarray(data.target_full[0]["point2segment"].detach().cpu())
    if not np.array_equal(full_segments, segments):
        raise MatrixRunError("native point2segment differs from method mesh segments")
    return segments


def _finite_official_metrics(value: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        try:
            metric = float(raw_value.detach().cpu())  # type: ignore[union-attr]
        except AttributeError:
            metric = float(raw_value)
        if math.isfinite(metric):
            result[key.replace("val_", "dev_", 1)] = metric
    required = {
        "dev_mean_t-AP",
        "dev_mean_AP",
        "dev_mean_stage1-AP",
        "dev_mean_stage2-AP",
    }
    if not required.issubset(result):
        raise MatrixRunError("official stmetrics output lacks required finite means")
    classify_official_metric_keys(result)
    return result


def _target_ambiguities(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MatrixRunError("official target ambiguities are malformed")
    return tuple(tuple(int(item) for item in group) for group in value)


def run_native_pair(
    *, config_path: str | Path, pair_id: str, output_root: str | Path
) -> dict[str, object]:
    """Execute one real D0 joint forward and publish all reusable evidence."""

    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config, selection = _load_config(Path(config_path).absolute())
    runtime = config["runtime"]
    assert isinstance(runtime, Mapping)
    checkout, source_commit = _validated_checkout(runtime)
    checkpoint = _validated_external_record(runtime["checkpoint"], label="checkpoint")
    concerto = _validated_external_record(
        runtime["concerto_checkpoint"], label="Concerto checkpoint"
    )
    processed_root = _validated_directory(
        runtime["processed_root"], label="processed root"
    )
    dataset_spec = _validated_external_record(
        runtime["dataset_spec"], label="dataset spec"
    )
    label_database = _validated_external_record(
        runtime["label_database"], label="label database"
    )
    change_database = _validated_external_record(
        runtime["change_label_database"], label="change label database"
    )
    color_mean_std = _validated_external_record(
        runtime["color_mean_std"], label="color mean/std"
    )
    _validated_external_record(
        runtime["validation_database"], label="validation database"
    )
    _validated_external_record(runtime["sequence_database"], label="sequence database")
    record = _pair_record(selection, _nonempty(pair_id, label="pair_id"))
    selection_record = config["selection_manifest"]
    assert isinstance(selection_record, Mapping)
    from src.evaluation.rscan_method_views import build_method_pair_view_from_manifest

    method_pair = build_method_pair_view_from_manifest(
        pair_record=record,
        source_manifest_sha256=str(selection_record["sha256"]),
        domain_id="D0_NATIVE_PROCESSED",
    )
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    with initialize_config_dir(version_base=None, config_dir=str(checkout / "conf")):
        native = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "data/datasets=rio",
                "general.train_mode=false",
                "general.train_on_segments=true",
                "general.eval_on_segments=true",
                "general.use_dbscan=false",
                "general.gpus=1",
                "general.seed=45",
                f"backbone.name={concerto}",
            ],
        )
    with open_dict(native):
        native.data.batch_size = 1
        native.data.test_batch_size = 1
        native.data.num_workers = 0
        native.instance_metric.dataset = str(dataset_spec)
        if hasattr(native, "aux_metric") and native.aux_metric is not None:
            native.aux_metric.dataset = str(dataset_spec)
        for split in ("train_dataset", "validation_dataset", "test_dataset"):
            dataset_config = native.data[split]
            dataset_config.data_dir = str(processed_root)
            dataset_config.label_db_filepath = str(label_database)
            dataset_config.change_label_db_filepath = str(change_database)
            dataset_config.color_mean_std = str(color_mean_std)
            dataset_config.temporal_window = 2

    device_name = os.environ.get("RESCENE_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise MatrixRunError("native pair execution requires one explicit CUDA device")
    torch.cuda.set_device(device)
    runtime_workdir = _validated_directory(
        runtime["runtime_workdir"], label="runtime workdir"
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(runtime_workdir)
        from trainer.trainer import InstanceSegmentation

        system = InstanceSegmentation(native)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise MatrixRunError("checkpoint is missing its Lightning state_dict")
        excluded = {str(key) for key in state if not str(key).startswith("model.")}
        if excluded != {"criterion.empty_weight", "criterion.change_weights"}:
            raise MatrixRunError("checkpoint contains unexpected non-model tensors")
        model_state = {
            str(key)[len("model.") :]: value
            for key, value in state.items()
            if str(key).startswith("model.")
        }
        if len(model_state) != 796:
            raise MatrixRunError("checkpoint must contain exactly 796 model tensors")
        incompatible = system.model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise MatrixRunError("strict ReScene checkpoint load was incompatible")
        system = system.to(device).eval()
        dataset = hydra.utils.instantiate(native.data.validation_dataset)
        system.validation_dataset = dataset
        try:
            dataset_index = dataset.sequence_names.index(pair_id)
        except ValueError as error:
            raise MatrixRunError("pair is absent from the native sequence database") from error
        seed = int(config["seed"])
        _seed_everything(seed)
        sample = dataset[dataset_index]
        collate = hydra.utils.instantiate(native.data.validation_collation)
        _seed_everything(seed)
        data, targets, names = collate([sample])
        if list(names) != [pair_id] or len(targets) != 1 or len(data.target_full) != 1:
            raise MatrixRunError("native collator did not return the requested supervised pair")
        original_segments = _audit_native_sample(method_pair, sample, data)
        inverse_n = np.asarray(data.inverse_maps[0].detach().cpu(), dtype=np.int64)
        lowres_segments = np.asarray(
            targets[0]["point2segment"].detach().cpu(), dtype=np.int64
        )
        lowres_visits = np.asarray(data.temporal_stages[0].detach().cpu(), dtype=np.int8)
        full_visits = np.asarray(
            data.target_full[0]["temporal_stages"].detach().cpu(), dtype=np.int8
        )
        _move_tensors(data, device)
        _move_tensors(targets, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = system.forward(
                data,
                point2segment=[target["point2segment"] for target in targets],
                raw_coordinates=system._process_raw_coordinates(data),
                is_eval=True,
                targets=None,
            )
        torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - started
        logits = output.get("pred_logits")
        masks = output.get("pred_masks")
        features = output.get("backbone_features")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape[:2] != (1, 100)
            or not isinstance(masks, (list, tuple))
            or len(masks) != 1
            or not isinstance(masks[0], torch.Tensor)
            or not hasattr(features, "F")
        ):
            raise MatrixRunError("native ReScene output schema is invalid")
        raw_logits = np.asarray(logits[0].detach().float().cpu(), dtype=np.float32)
        raw_masks = np.asarray(masks[0].detach().float().cpu(), dtype=np.float32)
        backbone_features = np.asarray(
            features.F.detach().float().cpu(), dtype=np.float32
        )
        predictions = system._process_predictions(
            output=output,
            target_low_res=targets,
            target_full_res=data.target_full,
            inverse_maps=data.inverse_maps,
            file_names=list(names),
            full_res_coords=data.original_coordinates,
            original_colors=data.original_colors,
            original_normals=data.original_normals,
            raw_coords=None,
            idx=data.idx,
        )
        system.instance_metric.reset()
        system.instance_metric.update(predictions, data.target_full)
        official = _finite_official_metrics(system.instance_metric.compute())
        prediction = predictions[0]
        target = data.target_full[0]
        arrays = {
            "pred_masks_nk": np.asarray(
                prediction["pred_masks"].detach().cpu(), dtype=np.uint8
            ),
            "pred_scores_k": np.asarray(prediction["pred_scores"], dtype=np.float32),
            "pred_classes_k": np.asarray(
                prediction["pred_classes"].detach().cpu(), dtype=np.int64
            ),
            "target_masks_in": np.asarray(
                target["masks"].detach().cpu(), dtype=np.uint8
            ),
            "target_labels_i": np.asarray(
                target["labels"].detach().cpu(), dtype=np.int64
            ),
            "target_changes_i": np.asarray(
                target["changes"].detach().cpu(), dtype=np.int64
            ),
            "target_ids_i": np.asarray(target["ids"].detach().cpu(), dtype=np.int64),
            "target_temporal_stages_n": np.asarray(
                target["temporal_stages"].detach().cpu(), dtype=np.int64
            ),
            "raw_masks_sq": raw_masks,
            "raw_logits_qc": raw_logits,
            "inverse_n": inverse_n,
            "lowres_point2segment_m": lowres_segments,
            "lowres_visit_ids_m": lowres_visits,
            "full_visit_ids_n": full_visits,
            "full_original_segment_ids_n": original_segments,
            "backbone_features_mf": backbone_features,
        }
        artifact = NativePairArtifact(
            pair_id=pair_id,
            method_input_sha256=method_pair.method_tensor_sha256(),
            checkpoint_sha256=str(runtime["checkpoint"]["sha256"]),
            source_commit=source_commit,
            arrays=arrays,
            official_metrics=official,
            target_ambiguities=_target_ambiguities(target.get("ambiguities")),
            runtime_s=forward_s,
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_gpu_bytes=int(torch.cuda.max_memory_reserved(device)),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            device_name=f"{torch.cuda.get_device_name(device)} ({device})",
        )
        return publish_native_pair_artifact(artifact, output_root)
    finally:
        os.chdir(previous_cwd)


def _relation_records(relations: object) -> list[dict[str, object]]:
    return [
        {
            "temporal_query_id": str(relation.temporal_query_id),
            "t0_entity_ids": list(relation.t0_entity_ids),
            "t1_entity_ids": list(relation.t1_entity_ids),
            "state": relation.state,
            "query_confidence": relation.query_confidence,
            "identity_source": relation.identity_source,
            "evidence": dict(relation.evidence),
        }
        for relation in relations
    ]


def compute_method_rows(
    *,
    pair: object,
    artifact: NativePairArtifact,
    ground_truth: object,
    geometric_config: object,
    projection_config: object,
    resolver_minimum_candidate_score: float,
    minimum_gt_observed_fraction: float,
) -> dict[str, object]:
    """Derive all CPU G/F/R readouts from one audited native forward."""

    from src.evaluation.rscan_association_metrics import evaluate_pair_relations
    from src.evaluation.rscan_gt_instances import GroundTruthPair
    from src.evaluation.rscan_method_views import (
        RScanMethodPairView,
        build_execution_plans,
        build_feature_geometric_sample,
        native_query_evidence,
        project_queries_fast,
        resolve_native_queries_one_to_one,
    )
    from src.oviv2.geometric_pair_reasoner import (
        GeometricPairReasoner,
        GeometricReasonerConfig,
    )
    from src.oviv2.query_instance_projection import ProjectionConfig

    if not isinstance(pair, RScanMethodPairView):
        raise TypeError("pair must be an RScanMethodPairView")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if not isinstance(geometric_config, GeometricReasonerConfig):
        raise TypeError("geometric_config must be a GeometricReasonerConfig")
    if not isinstance(projection_config, ProjectionConfig):
        raise TypeError("projection_config must be a ProjectionConfig")
    if artifact.pair_id != pair.pair_id or ground_truth.pair_id != pair.pair_id:
        raise MatrixRunError("method, native, and GT pair IDs differ")
    if artifact.method_input_sha256 != pair.method_tensor_sha256():
        raise MatrixRunError("native artifact binds the wrong method input")
    plans = {value.method_id: value for value in build_execution_plans(pair)}

    def evaluate(method_id: str, sample: object, relations: object, elapsed: float) -> dict[str, object]:
        plan = plans[method_id]
        return {
            "status": "PASS",
            "domain": pair.domain_id,
            "forward_mode": plan.forward_mode,
            "method_input_sha256": plan.method_input_sha256,
            "forward_input_sha256": list(plan.forward_input_sha256),
            "raw_forward_cache_key": plan.raw_forward_cache_key,
            "candidate_schema": plan.candidate_schema,
            "sample_sha256": sample.content_sha256(),
            "cpu_postprocess_s": elapsed,
            "relations": _relation_records(relations),
            "custom_association": evaluate_pair_relations(
                pair,
                relations,
                ground_truth,
                minimum_observed_fraction=minimum_gt_observed_fraction,
            ),
        }

    methods: dict[str, dict[str, object]] = {}
    geometric_sample = pair.geometric_sample(neural_voxel_size_m=0.02)
    started = time.perf_counter()
    geometric_evidence = GeometricPairReasoner(geometric_config).infer(
        geometric_sample
    )
    geometric_relations = project_queries_fast(
        geometric_sample, geometric_evidence, projection_config
    )
    methods["G_full"] = evaluate(
        "G_full",
        geometric_sample,
        geometric_relations,
        time.perf_counter() - started,
    )
    methods["G_supported"] = {
        "status": "MISSING_ASSET",
        "domain": "D1_NATIVE_SENSOR_SUPPORT",
        "reason": "native sensor or OVI support mask is unavailable",
    }

    started = time.perf_counter()
    feature_sample = build_feature_geometric_sample(
        pair, artifact.arrays, voxel_size_m=0.02
    )
    feature_evidence = GeometricPairReasoner(geometric_config).infer(feature_sample)
    feature_relations = project_queries_fast(
        feature_sample, feature_evidence, projection_config
    )
    methods["F"] = evaluate(
        "F", feature_sample, feature_relations, time.perf_counter() - started
    )

    native_evidence = native_query_evidence(
        pair,
        geometric_sample,
        artifact.arrays,
        checkpoint_sha256=artifact.checkpoint_sha256,
    )
    started = time.perf_counter()
    legacy_relations = project_queries_fast(
        geometric_sample, native_evidence, projection_config
    )
    methods["R_legacy"] = evaluate(
        "R_legacy",
        geometric_sample,
        legacy_relations,
        time.perf_counter() - started,
    )
    started = time.perf_counter()
    supported_relations = resolve_native_queries_one_to_one(
        pair,
        native_evidence,
        geometric_sample,
        minimum_candidate_score=resolver_minimum_candidate_score,
    )
    methods["R_supported"] = evaluate(
        "R_supported",
        geometric_sample,
        supported_relations,
        time.perf_counter() - started,
    )
    raw_nonempty = np.any(artifact.arrays["raw_masks_sq"] > 0.0, axis=0)
    return {
        "schema_version": 1,
        "artifact_id": "RSCAN_T2_PAIR_METHOD_MATRIX_V1",
        "status": "PASS",
        "pair_id": pair.pair_id,
        "claim_boundary": "development_reproduction_not_unseen_generalization",
        "native_raw_queries": {
            "total_count": int(artifact.arrays["raw_masks_sq"].shape[1]),
            "nonempty_count": int(np.count_nonzero(raw_nonempty)),
            "projected_count": len(native_evidence.temporal_query_ids),
        },
        "official_stmetrics": {
            "scope": "native_raw_shared",
            "metrics": dict(artifact.official_metrics),
        },
        "methods": methods,
        "unavailable_domains": {
            "D1_NATIVE_SENSOR_SUPPORT": "MISSING_ASSET",
            "D2_OVI_RECONSTRUCTION": "MISSING_ASSET",
        },
    }


def aggregate_custom_method_rows(
    pair_results: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Pool count metrics and retain macro means across pair-level CPU rows."""

    if not pair_results:
        raise MatrixRunError("custom aggregation requires at least one pair")
    pair_ids = [value.get("pair_id") for value in pair_results]
    if any(not isinstance(value, str) or not value for value in pair_ids) or len(
        pair_ids
    ) != len(set(pair_ids)):
        raise MatrixRunError("custom pair IDs are invalid or duplicated")
    method_ids = ("G_full", "G_supported", "F", "R_legacy", "R_supported")
    passed_ids = ("G_full", "F", "R_legacy", "R_supported")
    threshold_ids = ("iou_0_50", "iou_0_25")
    count_keys = (
        "paired_prediction_edges",
        "true_positive_edges",
        "false_positive_edges",
        "duplicate_edge_count",
        "false_reid_count",
        "same_class_mismatch_count",
        "unmatched_endpoint_edge_count",
        "conditional_true_positive_edges",
        "rigid_true_positive_edges",
        "strict_true_positive_edges",
        "strict_false_positive_edges",
        "strict_duplicate_edge_count",
        "ground_truth_persistent_edges",
        "strict_ground_truth_persistent_edges",
        "conditional_ground_truth_edges",
        "rigid_ground_truth_edges",
    )
    rate_keys = (
        "paired_precision",
        "paired_recall",
        "end_to_end_persistence_recall",
        "conditional_association_recall",
        "rigid_recall",
        "strict_paired_precision",
        "strict_paired_recall",
    )
    output: dict[str, dict[str, object]] = {
        "G_supported": {
            "status": "MISSING_ASSET",
            "reason": "D1 native sensor or OVI support mask is unavailable",
        }
    }
    for method_id in passed_ids:
        rows: list[Mapping[str, object]] = []
        runtimes: list[float] = []
        relation_count = 0
        for pair_result in pair_results:
            methods = pair_result.get("methods")
            if not isinstance(methods, Mapping) or set(methods) != set(method_ids):
                raise MatrixRunError("custom method result schema is invalid")
            row = methods.get(method_id)
            if not isinstance(row, Mapping) or row.get("status") != "PASS":
                raise MatrixRunError(f"custom method {method_id} did not pass")
            custom = row.get("custom_association")
            runtime = row.get("cpu_postprocess_s")
            if (
                not isinstance(custom, Mapping)
                or custom.get("status") != "PASS"
                or isinstance(runtime, bool)
                or not isinstance(runtime, (int, float))
                or not math.isfinite(float(runtime))
                or float(runtime) < 0.0
            ):
                raise MatrixRunError("custom method metrics or runtime are invalid")
            rows.append(custom)
            runtimes.append(float(runtime))
            raw_relation_count = custom.get("relation_count")
            if type(raw_relation_count) is not int or raw_relation_count < 0:
                raise MatrixRunError("custom relation count is invalid")
            relation_count += raw_relation_count
        thresholds: dict[str, object] = {}
        for threshold_id in threshold_ids:
            threshold_rows = []
            for row in rows:
                raw_thresholds = row.get("thresholds")
                if not isinstance(raw_thresholds, Mapping):
                    raise MatrixRunError("custom thresholds are invalid")
                threshold = raw_thresholds.get(threshold_id)
                if not isinstance(threshold, Mapping):
                    raise MatrixRunError("custom threshold row is invalid")
                threshold_rows.append(threshold)
            pooled: dict[str, object] = {}
            for key in count_keys:
                values = [row.get(key) for row in threshold_rows]
                if any(type(value) is not int or value < 0 for value in values):
                    raise MatrixRunError(f"custom count metric {key} is invalid")
                pooled[key] = sum(values)
            pooled["paired_precision"] = _ratio(
                pooled["true_positive_edges"], pooled["paired_prediction_edges"]
            )
            pooled["paired_recall"] = _ratio(
                pooled["true_positive_edges"],
                pooled["ground_truth_persistent_edges"],
            )
            pooled["end_to_end_persistence_recall"] = pooled["paired_recall"]
            pooled["conditional_association_recall"] = _ratio(
                pooled["conditional_true_positive_edges"],
                pooled["conditional_ground_truth_edges"],
            )
            pooled["rigid_recall"] = _ratio(
                pooled["rigid_true_positive_edges"],
                pooled["rigid_ground_truth_edges"],
            )
            pooled["strict_paired_precision"] = _ratio(
                pooled["strict_true_positive_edges"],
                pooled["paired_prediction_edges"],
            )
            pooled["strict_paired_recall"] = _ratio(
                pooled["strict_true_positive_edges"],
                pooled["strict_ground_truth_persistent_edges"],
            )
            macro: dict[str, float | None] = {}
            for key in rate_keys:
                values = [row.get(key) for row in threshold_rows]
                if any(
                    value is not None
                    and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    )
                    for value in values
                ):
                    raise MatrixRunError(f"custom rate metric {key} is invalid")
                finite = [float(value) for value in values if value is not None]
                macro[key] = sum(finite) / len(finite) if finite else None
            geometry_counts = {
                key: sum(
                    int(row["geometry"][key])  # type: ignore[index]
                    for row in threshold_rows
                )
                for key in (
                    "duplicate_prediction_count",
                    "fragment_gt_count",
                    "merge_prediction_count",
                )
            }
            thresholds[threshold_id] = {
                "pooled": pooled,
                "macro": macro,
                "geometry_counts": geometry_counts,
            }
        output[method_id] = {
            "status": "PASS",
            "pair_count": len(rows),
            "relation_count_sum": relation_count,
            "cpu_postprocess_s_sum": sum(runtimes),
            "cpu_postprocess_s_mean": sum(runtimes) / len(runtimes),
            "pooled": {
                threshold_id: thresholds[threshold_id]["pooled"]
                for threshold_id in threshold_ids
            },
            "macro": {
                threshold_id: thresholds[threshold_id]["macro"]
                for threshold_id in threshold_ids
            },
            "geometry_counts": {
                threshold_id: thresholds[threshold_id]["geometry_counts"]
                for threshold_id in threshold_ids
            },
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "pair_count": len(pair_results),
        "pair_ids": pair_ids,
        "methods": output,
    }


def _publish_json_file(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(os.path.abspath(os.fspath(path)))
    if output.exists() or output.is_symlink():
        raise MatrixRunError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise MatrixRunError(f"output already exists: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run_method_pair(
    *,
    config_path: str | Path,
    pair_id: str,
    native_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Evaluate all CPU method views for one pair from one native artifact."""

    absolute_config = Path(config_path).absolute()
    config, selection = _load_config(absolute_config)
    runtime = config["runtime"]
    parameters = config["method_parameters"]
    assert isinstance(runtime, Mapping) and isinstance(parameters, Mapping)
    record = _pair_record(selection, _nonempty(pair_id, label="pair_id"))
    selection_record = config["selection_manifest"]
    assert isinstance(selection_record, Mapping)
    from src.evaluation.rscan_gt_instances import load_pair_ground_truth
    from src.evaluation.rscan_method_views import build_method_pair_view_from_manifest

    pair = build_method_pair_view_from_manifest(
        pair_record=record,
        source_manifest_sha256=str(selection_record["sha256"]),
        domain_id="D0_NATIVE_PROCESSED",
    )
    native_collection = _validated_directory(
        native_root, label="native artifact collection"
    )
    native_pair_root = native_collection / pair.pair_id
    artifact = load_native_pair_artifact(native_pair_root)
    if artifact.checkpoint_sha256 != runtime["checkpoint"]["sha256"]:
        raise MatrixRunError("native artifact uses the wrong checkpoint")
    if artifact.source_commit != runtime["rescene_commit"]:
        raise MatrixRunError("native artifact uses the wrong ReScene source")
    gt_root = _validated_directory(runtime["gt_root"], label="GT collection")
    gt_manifest = gt_root / pair.pair_id / "manifest.json"
    ground_truth = load_pair_ground_truth(gt_manifest)
    geometric, projection, resolver_score, observed_fraction = _method_configs(
        parameters
    )
    result = compute_method_rows(
        pair=pair,
        artifact=artifact,
        ground_truth=ground_truth,
        geometric_config=geometric,
        projection_config=projection,
        resolver_minimum_candidate_score=resolver_score,
        minimum_gt_observed_fraction=observed_fraction,
    )
    result["method_parameters"] = dict(parameters)
    result["source_bindings"] = {
        "config": _file_record(absolute_config, recorded_path=str(absolute_config)),
        "selection_manifest": dict(selection_record),
        "native_manifest": _file_record(
            native_pair_root / "manifest.json",
            recorded_path=str(native_pair_root / "manifest.json"),
        ),
        "ground_truth_manifest": _file_record(
            gt_manifest, recorded_path=str(gt_manifest)
        ),
    }
    _publish_json_file(output_path, result)
    return result


def _load_method_pair_result(
    *,
    path: Path,
    config_path: Path,
    config: Mapping[str, object],
    expected_pair_id: str,
    native_root: Path,
    gt_root: Path,
) -> Mapping[str, object]:
    result = _load_json(path, label="pair method result")
    if set(result) != {
        "schema_version",
        "artifact_id",
        "status",
        "pair_id",
        "claim_boundary",
        "native_raw_queries",
        "official_stmetrics",
        "methods",
        "unavailable_domains",
        "method_parameters",
        "source_bindings",
    } or (
        result.get("schema_version") != 1
        or result.get("artifact_id") != "RSCAN_T2_PAIR_METHOD_MATRIX_V1"
        or result.get("status") != "PASS"
        or result.get("pair_id") != expected_pair_id
        or result.get("claim_boundary")
        != "development_reproduction_not_unseen_generalization"
    ):
        raise MatrixRunError("pair method result identity is invalid")
    if result.get("method_parameters") != config.get("method_parameters"):
        raise MatrixRunError("pair method parameters differ from the frozen config")
    if result.get("unavailable_domains") != {
        "D1_NATIVE_SENSOR_SUPPORT": "MISSING_ASSET",
        "D2_OVI_RECONSTRUCTION": "MISSING_ASSET",
    }:
        raise MatrixRunError("pair method domain coverage is invalid")
    bindings = result.get("source_bindings")
    selection_record = config.get("selection_manifest")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "config",
        "selection_manifest",
        "native_manifest",
        "ground_truth_manifest",
    } or bindings.get("selection_manifest") != selection_record:
        raise MatrixRunError("pair method source bindings are invalid")
    bound_config = _validated_external_record(bindings["config"], label="matrix config")
    bound_native = _validated_external_record(
        bindings["native_manifest"], label="native manifest"
    )
    bound_gt = _validated_external_record(
        bindings["ground_truth_manifest"], label="ground-truth manifest"
    )
    expected_native = native_root / expected_pair_id / "manifest.json"
    expected_gt = gt_root / expected_pair_id / "manifest.json"
    if (
        bound_config != config_path
        or bound_native != expected_native
        or bound_gt != expected_gt
    ):
        raise MatrixRunError("pair method source paths differ from the protocol")
    native = load_native_pair_artifact(expected_native.parent)
    official = result.get("official_stmetrics")
    if not isinstance(official, Mapping) or official != {
        "scope": "native_raw_shared",
        "metrics": dict(native.official_metrics),
    }:
        raise MatrixRunError("pair method official metrics differ from native evidence")
    methods = result.get("methods")
    expected_methods = {"G_full", "G_supported", "F", "R_legacy", "R_supported"}
    if not isinstance(methods, Mapping) or set(methods) != expected_methods:
        raise MatrixRunError("pair method rows are invalid")
    if not isinstance(methods["G_supported"], Mapping) or methods[
        "G_supported"
    ].get("status") != "MISSING_ASSET":
        raise MatrixRunError("G_supported must remain explicitly unavailable")
    for method_id in expected_methods - {"G_supported"}:
        row = methods[method_id]
        if not isinstance(row, Mapping) or row.get("status") != "PASS":
            raise MatrixRunError(f"pair method {method_id} did not pass")
        relations = row.get("relations")
        custom = row.get("custom_association")
        if (
            not isinstance(relations, list)
            or not isinstance(custom, Mapping)
            or custom.get("pair_id") != expected_pair_id
            or custom.get("method_input_sha256") != native.method_input_sha256
            or custom.get("relation_count") != len(relations)
        ):
            raise MatrixRunError("pair custom metrics do not bind their relations")
    legacy_cache = methods["R_legacy"].get("raw_forward_cache_key")
    supported_cache = methods["R_supported"].get("raw_forward_cache_key")
    if not isinstance(legacy_cache, str) or legacy_cache != supported_cache:
        raise MatrixRunError("R readouts do not share one raw forward")
    return result


def aggregate_method_artifacts(
    *,
    config_path: str | Path,
    native_root: str | Path,
    method_root: str | Path,
    official_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Audit pair-level CPU rows and publish the small complete matrix summary."""

    absolute_config = Path(config_path).absolute()
    config, selection = _load_config(absolute_config)
    runtime = config["runtime"]
    assert isinstance(runtime, Mapping)
    native_collection = _validated_directory(
        native_root, label="native artifact collection"
    )
    method_collection = _validated_directory(
        method_root, label="method artifact collection"
    )
    gt_collection = _validated_directory(runtime["gt_root"], label="GT collection")
    raw_pairs = selection.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise MatrixRunError("selection pairs are invalid")
    pair_ids = [
        _nonempty(value.get("pair_id"), label="selection pair_id")
        for value in raw_pairs
        if isinstance(value, Mapping)
    ]
    if len(pair_ids) != len(raw_pairs) or len(pair_ids) != len(set(pair_ids)):
        raise MatrixRunError("selection pair IDs are invalid")
    pair_results = tuple(
        _load_method_pair_result(
            path=method_collection / f"{pair_id}.json",
            config_path=absolute_config,
            config=config,
            expected_pair_id=pair_id,
            native_root=native_collection,
            gt_root=gt_collection,
        )
        for pair_id in pair_ids
    )
    official_path = Path(official_summary_path).absolute()
    official = _load_json(official_path, label="native official summary")
    if (
        official.get("artifact_id") != "RSCAN_T2_D0_NATIVE_AGGREGATE_V1"
        or official.get("status") != "PASS"
        or official.get("pair_ids") != pair_ids
        or official.get("selection_manifest") != config.get("selection_manifest")
        or official.get("checkpoint") != runtime.get("checkpoint")
        or official.get("rescene_commit") != runtime.get("rescene_commit")
    ):
        raise MatrixRunError("native official summary differs from the matrix protocol")
    aggregate = aggregate_custom_method_rows(pair_results)
    aggregate.update(
        {
            "artifact_id": "RSCAN_T2_METHOD_MATRIX_AGGREGATE_V1",
            "claim_boundary": "development_reproduction_not_unseen_generalization",
            "domain_coverage": {
                "D0_NATIVE_PROCESSED": "PASS",
                "D1_NATIVE_SENSOR_SUPPORT": "MISSING_ASSET",
                "D2_OVI_RECONSTRUCTION": "MISSING_ASSET",
            },
            "official_stmetrics": {
                "scope": "native_raw_shared",
                "metrics": official.get("official_stmetrics"),
                "undefined_keys": official.get("official_undefined_keys"),
            },
            "native_runtime": official.get("runtime"),
            "method_parameters": config.get("method_parameters"),
            "source_bindings": {
                "config": _file_record(
                    absolute_config, recorded_path=str(absolute_config)
                ),
                "selection_manifest": config.get("selection_manifest"),
                "native_summary": _file_record(
                    official_path, recorded_path=str(official_path)
                ),
                "pair_results": {
                    pair_id: _file_record(
                        method_collection / f"{pair_id}.json",
                        recorded_path=str(method_collection / f"{pair_id}.json"),
                    )
                    for pair_id in pair_ids
                },
            },
            "per_pair": {
                pair_id: {
                    method_id: {
                        "iou_0_50": result["methods"][method_id][
                            "custom_association"
                        ]["thresholds"]["iou_0_50"],
                        "iou_0_25": result["methods"][method_id][
                            "custom_association"
                        ]["thresholds"]["iou_0_25"],
                    }
                    for method_id in ("G_full", "F", "R_legacy", "R_supported")
                }
                for pair_id, result in zip(pair_ids, pair_results, strict=True)
            },
        }
    )
    _publish_json_file(output_path, aggregate)
    return aggregate


def compute_official_metrics(
    artifacts: tuple[NativePairArtifact, ...], *, dataset_spec: str | Path
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Pool pair predictions through the official stmetrics implementation."""

    import torch
    from stmetrics import (
        InstanceMetrics,
        LegacyAPEvaluator,
        SelectTimestepEvaluator,
        TemporalEvaluator,
    )

    if not artifacts:
        raise MatrixRunError("official aggregation requires at least one pair")
    spec_path = Path(os.path.abspath(os.fspath(dataset_spec)))
    _regular_bytes(spec_path, label="dataset spec")
    metric = InstanceMetrics(
        dataset=str(spec_path),
        heads=[
            TemporalEvaluator(recall=True, aux="changes"),
            LegacyAPEvaluator(),
            SelectTimestepEvaluator(timesteps=[0]),
            SelectTimestepEvaluator(timesteps=[1]),
        ],
        log_prefix="dev",
        timestep_key="temporal_stages",
    )
    metric.reset()
    for artifact in artifacts:
        arrays = artifact.arrays
        predictions = [
            {
                "pred_masks": torch.from_numpy(
                    np.array(arrays["pred_masks_nk"], dtype=np.uint8, copy=True)
                ),
                "pred_scores": torch.from_numpy(
                    np.array(arrays["pred_scores_k"], dtype=np.float32, copy=True)
                ),
                "pred_classes": torch.from_numpy(
                    np.array(arrays["pred_classes_k"], dtype=np.int64, copy=True)
                ),
            }
        ]
        targets = [
            {
                "masks": torch.from_numpy(
                    np.array(arrays["target_masks_in"], dtype=np.uint8, copy=True)
                ),
                "labels": torch.from_numpy(
                    np.array(arrays["target_labels_i"], dtype=np.int64, copy=True)
                ),
                "changes": torch.from_numpy(
                    np.array(arrays["target_changes_i"], dtype=np.int64, copy=True)
                ),
                "ids": torch.from_numpy(
                    np.array(arrays["target_ids_i"], dtype=np.int64, copy=True)
                ),
                "temporal_stages": torch.from_numpy(
                    np.array(
                        arrays["target_temporal_stages_n"],
                        dtype=np.int64,
                        copy=True,
                    )
                ),
                "ambiguities": [list(group) for group in artifact.target_ambiguities],
            }
        ]
        metric.update(predictions, targets)
    observed = metric.compute()
    finite: dict[str, float] = {}
    undefined: list[str] = []
    for key, raw_value in observed.items():
        value = float(raw_value.detach().cpu())
        if math.isfinite(value):
            finite[str(key)] = value
        else:
            undefined.append(str(key))
    required = {
        "dev_mean_t-AP",
        "dev_mean_t-AP_50",
        "dev_mean_t-AP_25",
        "dev_mean_t-REC",
        "dev_mean_AP",
        "dev_mean_stage1-AP",
        "dev_mean_stage2-AP",
    }
    if not required.issubset(finite):
        raise MatrixRunError("pooled stmetrics output lacks required finite means")
    classify_official_metric_keys(finite)
    metric.reset()
    return dict(sorted(finite.items())), tuple(sorted(undefined))


def aggregate_native_artifacts(
    *,
    config_path: str | Path,
    input_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Audit every selected pair and publish the pooled official D0 result."""

    config, selection = _load_config(Path(config_path).absolute())
    runtime = config["runtime"]
    assert isinstance(runtime, Mapping)
    dataset_spec = _validated_external_record(
        runtime["dataset_spec"], label="dataset spec"
    )
    root = _validated_directory(input_root, label="native artifact root")
    pairs = selection.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise MatrixRunError("selection pairs are invalid")
    selection_record = config["selection_manifest"]
    assert isinstance(selection_record, Mapping)
    from src.evaluation.rscan_method_views import build_method_pair_view_from_manifest

    artifacts: list[NativePairArtifact] = []
    pair_bindings: dict[str, dict[str, object]] = {}
    for record in pairs:
        if not isinstance(record, Mapping):
            raise MatrixRunError("selection pair is invalid")
        pair_id = _nonempty(record.get("pair_id"), label="selection pair_id")
        artifact_root = root / pair_id
        artifact = load_native_pair_artifact(artifact_root)
        expected_view = build_method_pair_view_from_manifest(
            pair_record=record,
            source_manifest_sha256=str(selection_record["sha256"]),
            domain_id="D0_NATIVE_PROCESSED",
        )
        if artifact.method_input_sha256 != expected_view.method_tensor_sha256():
            raise MatrixRunError("native artifact binds the wrong method input")
        if artifact.checkpoint_sha256 != runtime["checkpoint"]["sha256"]:
            raise MatrixRunError("native artifacts do not use the frozen checkpoint")
        if artifact.source_commit != runtime["rescene_commit"]:
            raise MatrixRunError("native artifacts do not use the frozen source")
        artifacts.append(artifact)
        pair_bindings[pair_id] = _file_record(
            artifact_root / "manifest.json",
            recorded_path=f"{pair_id}/manifest.json",
        )
    finite, undefined = compute_official_metrics(
        tuple(artifacts), dataset_spec=dataset_spec
    )
    summary = {
        "schema_version": 1,
        "artifact_id": "RSCAN_T2_D0_NATIVE_AGGREGATE_V1",
        "status": "PASS",
        "claim_boundary": "development_reproduction_not_unseen_generalization",
        "pair_count": len(artifacts),
        "pair_ids": [artifact.pair_id for artifact in artifacts],
        "pair_manifests": dict(sorted(pair_bindings.items())),
        "selection_manifest": dict(selection_record),
        "checkpoint": dict(runtime["checkpoint"]),
        "rescene_commit": runtime["rescene_commit"],
        "domain": "D0_NATIVE_PROCESSED",
        "official_stmetrics": finite,
        "official_undefined_keys": list(undefined),
        "runtime": {
            "forward_s_sum": sum(artifact.runtime_s for artifact in artifacts),
            "forward_s_mean": sum(artifact.runtime_s for artifact in artifacts)
            / len(artifacts),
            "peak_gpu_bytes_max": max(
                artifact.peak_gpu_bytes for artifact in artifacts
            ),
            "peak_reserved_gpu_bytes_max": max(
                artifact.peak_reserved_gpu_bytes for artifact in artifacts
            ),
            "peak_rss_bytes_max": max(
                artifact.peak_rss_bytes for artifact in artifacts
            ),
        },
        "unavailable_domains": {
            "D1_NATIVE_SENSOR_SUPPORT": "MISSING_ASSET",
            "D2_OVI_RECONSTRUCTION": "MISSING_ASSET",
        },
    }
    output = Path(os.path.abspath(os.fspath(output_path)))
    if output.exists() or output.is_symlink():
        raise MatrixRunError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output, _json_bytes(summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    native = subparsers.add_parser("native-pair")
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--pair-id", required=True)
    native.add_argument("--output-root", type=Path, required=True)
    audit = subparsers.add_parser("audit-pair")
    audit.add_argument("--input-root", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate-native")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    methods = subparsers.add_parser("method-pair")
    methods.add_argument("--config", type=Path, required=True)
    methods.add_argument("--pair-id", required=True)
    methods.add_argument("--native-root", type=Path, required=True)
    methods.add_argument("--output", type=Path, required=True)
    matrix = subparsers.add_parser("aggregate-matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--native-root", type=Path, required=True)
    matrix.add_argument("--method-root", type=Path, required=True)
    matrix.add_argument("--official-summary", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "native-pair":
        result = run_native_pair(
            config_path=args.config,
            pair_id=args.pair_id,
            output_root=args.output_root,
        )
    elif args.command == "audit-pair":
        artifact = load_native_pair_artifact(args.input_root)
        result = {
            "status": "PASS",
            "pair_id": artifact.pair_id,
            "method_input_sha256": artifact.method_input_sha256,
        }
    elif args.command == "aggregate-native":
        result = aggregate_native_artifacts(
            config_path=args.config,
            input_root=args.input_root,
            output_path=args.output,
        )
    elif args.command == "method-pair":
        result = run_method_pair(
            config_path=args.config,
            pair_id=args.pair_id,
            native_root=args.native_root,
            output_path=args.output,
        )
    else:
        result = aggregate_method_artifacts(
            config_path=args.config,
            native_root=args.native_root,
            method_root=args.method_root,
            official_summary_path=args.official_summary,
            output_path=args.output,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "MatrixRunError",
    "NativePairArtifact",
    "aggregate_custom_method_rows",
    "aggregate_method_artifacts",
    "aggregate_native_artifacts",
    "classify_official_metric_keys",
    "compute_method_rows",
    "compute_official_metrics",
    "load_native_pair_artifact",
    "publish_native_pair_artifact",
    "run_method_pair",
    "run_native_pair",
]
