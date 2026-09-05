#!/usr/bin/env python3
"""Execute one source-bound ReScene forward over a prepared OVI pair."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.prepare_ovi_rescene_input_v2 import audit_recovered_input
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.rescene_input_bridge import (
    ReSceneModelInput,
    expand_model_predictions,
    load_model_input_artifact,
)

EXPECTED_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
EXPECTED_CONCERTO_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
EXPECTED_MODEL_TENSOR_COUNT = 796
EXCLUDED_CHECKPOINT_KEYS = frozenset(
    {"criterion.empty_weight", "criterion.change_weights"}
)
_PAIR_ARRAY_KEYS = frozenset(
    {
        "coordinates_xyzt",
        "features",
        "visit_ids",
        "source_visit_ids",
        "source_point_indices",
        "source_to_token_offsets",
    }
)


class ExecutorError(ValueError):
    """Raised when a ReScene executor identity or tensor contract is invalid."""


@dataclass(frozen=True, slots=True)
class NativeForwardResult:
    pred_masks_mq: np.ndarray
    pred_logits_qc: np.ndarray
    runtime_s: float
    peak_memory_bytes: int
    peak_reserved_memory_bytes: int
    rss_peak_bytes: int
    device_name: str
    model_tensor_count: int


@dataclass(frozen=True, slots=True)
class ProcessedPredictions:
    raw_query_indices: np.ndarray
    retained_query_indices: np.ndarray
    temporal_query_ids: tuple[str, ...]
    pred_masks_qm: np.ndarray
    pred_logits_qc: np.ndarray
    query_masks_qa: np.ndarray
    token_scores_qa: np.ndarray
    query_scores: np.ndarray


@dataclass(frozen=True, slots=True)
class ExecutorPaths:
    output_arrays: Path
    output_manifest: Path
    raw_root: Path
    raw_manifest: Path
    raw_arrays: Path


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


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ExecutorError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise ExecutorError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise ExecutorError(f"{label} changed while being read")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_record(path: Path, *, relative_path: str | None = None) -> dict[str, object]:
    content = _regular_file_bytes(path, label=path.name)
    return {
        "path": relative_path if relative_path is not None else str(path.absolute()),
        "sha256": _sha256_bytes(content),
        "byte_count": len(content),
    }


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutorError(f"{label} must be a lowercase SHA-256")
    return value


def _validated_source_commit(checkout: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(checkout)))
    if absolute.is_symlink() or not absolute.is_dir():
        raise ExecutorError("ReScene checkout is missing or a symlink")
    try:
        commit = subprocess.run(
            ["git", "-C", str(absolute), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(absolute), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError("ReScene checkout cannot be verified") from error
    if commit != EXPECTED_SOURCE_COMMIT or status:
        raise ExecutorError("ReScene checkout identity mismatch")
    return commit


def extract_model_state_dict(checkpoint_payload: object) -> dict[str, object]:
    """Extract exactly the 796 model tensors from one Lightning checkpoint."""

    if not isinstance(checkpoint_payload, Mapping):
        raise ExecutorError("checkpoint must contain a Lightning mapping")
    state = checkpoint_payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ExecutorError("checkpoint is missing state_dict")
    excluded = {key for key in state if not str(key).startswith("model.")}
    if excluded != EXCLUDED_CHECKPOINT_KEYS:
        raise ExecutorError("checkpoint contains unexpected non-model tensors")
    model_items = {
        str(key)[len("model.") :]: value
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    if len(model_items) != EXPECTED_MODEL_TENSOR_COUNT or any(
        not key for key in model_items
    ):
        raise ExecutorError("checkpoint must contain exactly 796 model tensors")
    return model_items


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    shifted = raw - np.max(raw, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def postprocess_native_predictions(
    *,
    pred_masks_mq: np.ndarray,
    pred_logits_qc: np.ndarray,
    adapter_to_model: np.ndarray,
) -> ProcessedPredictions:
    """Preserve raw Q identity, score once, and expand Q x M through A-to-M."""

    masks_mq = np.asarray(pred_masks_mq)
    logits_qc = np.asarray(pred_logits_qc)
    mapping_raw = np.asarray(adapter_to_model)
    if masks_mq.ndim != 2 or not np.issubdtype(masks_mq.dtype, np.floating):
        raise ExecutorError("pred_masks must have floating shape M x Q")
    if logits_qc.ndim != 2 or not np.issubdtype(logits_qc.dtype, np.floating):
        raise ExecutorError("pred_logits must have floating shape Q x (C+1)")
    model_count, query_count = masks_mq.shape
    if query_count == 0 or logits_qc.shape[0] != query_count:
        raise ExecutorError("mask and logit query count mismatch")
    if logits_qc.shape[1] < 2:
        raise ExecutorError(
            "pred_logits class count must include foreground and no-object"
        )
    if not np.all(np.isfinite(masks_mq)) or not np.all(np.isfinite(logits_qc)):
        raise ExecutorError("native predictions must be finite")
    if (
        mapping_raw.ndim != 1
        or not np.issubdtype(mapping_raw.dtype, np.integer)
        or np.issubdtype(mapping_raw.dtype, np.bool_)
    ):
        raise ExecutorError("adapter-to-model mapping must be integer")
    mapping = mapping_raw.astype(np.int64, copy=False)
    if (
        model_count == 0
        or len(mapping) == 0
        or np.any(mapping < 0)
        or np.any(mapping >= model_count)
        or not np.array_equal(np.unique(mapping), np.arange(model_count))
    ):
        raise ExecutorError("adapter-to-model mapping does not cover M")

    masks_qm = np.ascontiguousarray(masks_mq.T, dtype=np.float32)
    logits = np.ascontiguousarray(logits_qc, dtype=np.float32)
    positive_qm = masks_qm > 0.0
    scores_qm = _sigmoid(masks_qm)
    counts = np.count_nonzero(positive_qm, axis=1)
    retained = np.flatnonzero(counts > 0).astype(np.int64)
    foreground_confidence = np.max(_softmax(logits)[:, :-1], axis=1)
    mean_positive = np.zeros(query_count, dtype=np.float64)
    nonempty = counts > 0
    mean_positive[nonempty] = (
        np.sum(scores_qm * positive_qm, axis=1)[nonempty] / counts[nonempty]
    )
    all_scores = np.asarray(foreground_confidence * mean_positive, dtype=np.float32)
    expanded_masks = expand_model_predictions(positive_qm[retained], mapping)
    expanded_scores = expand_model_predictions(scores_qm[retained], mapping)
    return ProcessedPredictions(
        raw_query_indices=np.arange(query_count, dtype=np.int64),
        retained_query_indices=retained,
        temporal_query_ids=tuple(f"query_{index:04d}" for index in retained),
        pred_masks_qm=masks_qm,
        pred_logits_qc=logits,
        query_masks_qa=np.ascontiguousarray(expanded_masks, dtype=np.bool_),
        token_scores_qa=np.ascontiguousarray(expanded_scores, dtype=np.float32),
        query_scores=np.ascontiguousarray(all_scores[retained], dtype=np.float32),
    )


def _load_pair_arrays(path: Path) -> dict[str, np.ndarray]:
    content = _regular_file_bytes(path, label="temporary pair arrays")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if set(archive.files) != _PAIR_ARRAY_KEYS:
                raise ExecutorError("temporary pair array schema is invalid")
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ExecutorError):
            raise
        raise ExecutorError("temporary pair arrays cannot be decoded") from error


def _load_bound_sidecar(
    input_sidecar: Path,
    pair_arrays_path: Path,
    *,
    pair_sha256: str,
    feature_schema: str,
    neural_voxel_size_m: float,
) -> ReSceneModelInput:
    root = Path(os.path.abspath(os.fspath(input_sidecar)))
    receipt_path = root / "input_contract_v2.json"
    try:
        receipt = json.loads(
            _regular_file_bytes(receipt_path, label="input contract").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutorError("input contract is not valid JSON") from error
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("artifact_id") != "OVI_RESCENE_C2_INPUT_CONTRACT_V2"
        or receipt.get("status") != "C2_V2_PASS"
    ):
        raise ExecutorError("input sidecar is not C2_V2_PASS")
    try:
        static_path = Path(receipt["static_input_manifest"]["path"]).parent
        calibration_path = Path(receipt["calibration_manifest"]["path"])
    except (KeyError, TypeError) as error:
        raise ExecutorError("input contract parent bindings are invalid") from error
    try:
        audit_recovered_input(
            output_root=root,
            static_input_root=static_path,
            calibration_manifest_path=calibration_path,
        )
        model_input = load_model_input_artifact(root / "model_input")
        pair = load_neural_sample_artifact(root / "adapter_pair")
    except (OSError, TypeError, ValueError) as error:
        raise ExecutorError("input sidecar audit failed") from error
    if pair.content_sha256() != pair_sha256:
        raise ExecutorError("input sidecar pair SHA-256 mismatch")
    if pair.feature_schema != feature_schema or feature_schema != "rgb_normals":
        raise ExecutorError("input sidecar feature schema mismatch")
    if not math.isclose(
        pair.neural_voxel_size_m,
        neural_voxel_size_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExecutorError("input sidecar neural voxel size mismatch")
    pair_arrays = _load_pair_arrays(pair_arrays_path)
    expected = {
        "coordinates_xyzt": pair.coordinates_xyzt,
        "features": pair.features,
        "visit_ids": pair.visit_ids,
        "source_visit_ids": pair.source_visit_ids,
        "source_point_indices": pair.source_point_indices,
        "source_to_token_offsets": pair.source_to_token_offsets,
    }
    if any(
        not np.array_equal(pair_arrays[name], value) for name, value in expected.items()
    ):
        raise ExecutorError("temporary pair arrays differ from the input sidecar")
    return model_input


def _native_forward(
    model_input: ReSceneModelInput,
    checkout: Path,
    checkpoint: Path,
) -> NativeForwardResult:
    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    device_name = os.environ.get("RESCENE_DEVICE", "cuda:0")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ExecutorError(
            "native ReScene execution requires one explicit CUDA device"
        )
    torch.cuda.set_device(device)
    concerto_path = Path(os.environ.get("CONCERTO_CHECKPOINT", ""))
    concerto_content = _regular_file_bytes(concerto_path, label="Concerto checkpoint")
    if _sha256_bytes(concerto_content) != EXPECTED_CONCERTO_SHA256:
        raise ExecutorError("Concerto checkpoint SHA-256 mismatch")
    del concerto_content

    checkout_string = str(Path(checkout).absolute())
    if checkout_string not in sys.path:
        sys.path.insert(0, checkout_string)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(checkout).absolute() / "conf"),
    ):
        config = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "general.train_mode=false",
                "general.train_on_segments=true",
                "general.eval_on_segments=true",
                "general.use_dbscan=false",
                "general.gpus=1",
                "general.seed=45",
                f"backbone.name={concerto_path.absolute()}",
            ],
        )
    if (
        int(config.model.D) != 4
        or not bool(config.model.train_on_segments)
        or not math.isclose(float(config.model.voxel_size), 0.02, abs_tol=1e-12)
    ):
        raise ExecutorError("composed ReScene model contract is invalid")
    model = hydra.utils.instantiate(config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = extract_model_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExecutorError("strict ReScene model load returned incompatible keys")
    model.to(device)
    model.eval()

    from sonata.structure import Point

    coordinates = torch.from_numpy(
        np.ascontiguousarray(model_input.coordinates_bxyzt, dtype=np.float32)
    ).to(device)
    point = Point(
        {
            "coord": coordinates,
            "grid_coord": torch.from_numpy(
                np.ascontiguousarray(model_input.grid_coordinates_xyz, dtype=np.int32)
            ).to(device),
            "feat": torch.from_numpy(
                np.ascontiguousarray(model_input.features, dtype=np.float32)
            ).to(device),
            "offset": torch.from_numpy(
                np.ascontiguousarray(model_input.sparse_batch_offsets, dtype=np.int64)
            ).to(device),
        }
    )
    point2segment = [
        torch.from_numpy(
            np.ascontiguousarray(model_input.point2segment, dtype=np.int64)
        ).to(device)
    ]
    raw_coordinates = coordinates[:, 1:]
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    torch.cuda.manual_seed_all(45)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            point,
            point2segment=point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=True,
        )
    torch.cuda.synchronize(device)
    runtime_s = time.perf_counter() - started
    if not isinstance(output, Mapping):
        raise ExecutorError("native ReScene output must be a mapping")
    logits = output.get("pred_logits")
    masks = output.get("pred_masks")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not isinstance(masks, (list, tuple))
        or len(masks) != 1
        or not isinstance(masks[0], torch.Tensor)
        or masks[0].ndim != 2
    ):
        raise ExecutorError("native ReScene output tensor schema is invalid")
    result = NativeForwardResult(
        pred_masks_mq=np.ascontiguousarray(
            masks[0].detach().float().cpu().numpy(), dtype=np.float32
        ),
        pred_logits_qc=np.ascontiguousarray(
            logits[0].detach().float().cpu().numpy(), dtype=np.float32
        ),
        runtime_s=runtime_s,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_memory_bytes=int(torch.cuda.max_memory_reserved(device)),
        rss_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        device_name=f"{torch.cuda.get_device_name(device)} ({device})",
        model_tensor_count=len(state),
    )
    return result


def _validate_forward_result(result: NativeForwardResult, *, model_count: int) -> None:
    if not isinstance(result, NativeForwardResult):
        raise ExecutorError("native runner returned an invalid result")
    if np.asarray(result.pred_masks_mq).shape[0] != model_count:
        raise ExecutorError("native mask model count differs from input M")
    if (
        not math.isfinite(float(result.runtime_s))
        or result.runtime_s < 0.0
        or any(
            type(value) is not int or value < 0
            for value in (
                result.peak_memory_bytes,
                result.peak_reserved_memory_bytes,
                result.rss_peak_bytes,
            )
        )
        or not isinstance(result.device_name, str)
        or not result.device_name
        or result.model_tensor_count != EXPECTED_MODEL_TENSOR_COUNT
    ):
        raise ExecutorError("native runtime evidence is invalid")


def _publish_raw_output(
    *,
    output_root: Path,
    processed: ProcessedPredictions,
    model_input: ReSceneModelInput,
    forward: NativeForwardResult,
    checkpoint_sha256: str,
    pair_sha256: str,
    source_commit: str,
) -> tuple[Path, Path]:
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise ExecutorError(f"raw output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                pred_masks_qm=processed.pred_masks_qm,
                pred_logits_qc=processed.pred_logits_qc,
                raw_query_indices=processed.raw_query_indices,
                retained_query_indices=processed.retained_query_indices,
            )
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_RAW_MODEL_OUTPUT_V1",
            "status": "PASS",
            "source_commit": source_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "pair_sha256": pair_sha256,
            "model_input_sha256": model_input.content_sha256(),
            "model_count": len(model_input.model_visit_ids),
            "raw_query_count": len(processed.raw_query_indices),
            "retained_query_count": len(processed.retained_query_indices),
            "empty_support_query_count": int(
                len(processed.raw_query_indices) - len(processed.retained_query_indices)
            ),
            "runtime": {
                "device": forward.device_name,
                "runtime_s": float(forward.runtime_s),
                "peak_allocated_bytes": forward.peak_memory_bytes,
                "peak_reserved_bytes": forward.peak_reserved_memory_bytes,
                "rss_peak_bytes": forward.rss_peak_bytes,
                "model_tensor_count": forward.model_tensor_count,
            },
            "arrays": _file_record(arrays_path, relative_path="arrays.npz"),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json", output / "arrays.npz"


def _publish_public_output(
    *,
    arrays_path: Path,
    manifest_path: Path,
    processed: ProcessedPredictions,
    pair_sha256: str,
    checkpoint_sha256: str,
    forward: NativeForwardResult,
) -> None:
    output_arrays = Path(os.path.abspath(os.fspath(arrays_path)))
    output_manifest = Path(os.path.abspath(os.fspath(manifest_path)))
    if output_arrays.parent != output_manifest.parent:
        raise ExecutorError("public output arrays and manifest must share a directory")
    if any(
        path.exists() or path.is_symlink() for path in (output_arrays, output_manifest)
    ):
        raise ExecutorError("public output already exists")
    output_arrays.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".rescene-public.", dir=output_arrays.parent)
    )
    linked: list[Path] = []
    try:
        staged_arrays = staging / output_arrays.name
        with staged_arrays.open("xb") as stream:
            np.savez_compressed(
                stream,
                token_indices=np.arange(
                    processed.query_masks_qa.shape[1], dtype=np.int64
                ),
                query_masks=processed.query_masks_qa,
                token_scores=processed.token_scores_qa,
                query_scores=processed.query_scores,
            )
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "backend_name": "concerto",
            "pair_sha256": pair_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "temporal_query_ids": list(processed.temporal_query_ids),
            "runtime_s": float(forward.runtime_s),
            "peak_memory_bytes": forward.peak_memory_bytes,
            "output_arrays": _file_record(
                staged_arrays, relative_path=output_arrays.name
            ),
        }
        staged_manifest = staging / output_manifest.name
        with staged_manifest.open("xb") as stream:
            stream.write(_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        for source, target in (
            (staged_arrays, output_arrays),
            (staged_manifest, output_manifest),
        ):
            os.link(source, target)
            linked.append(target)
        parent_fd = os.open(output_arrays.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        for target in linked:
            target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_rescene_pair_executor(
    *,
    input_sidecar: str | Path,
    pair_arrays: str | Path,
    output_arrays: str | Path,
    output_manifest: str | Path,
    checkpoint: str | Path,
    checkout: str | Path,
    pair_sha256: str,
    checkpoint_sha256: str,
    feature_schema: str,
    neural_voxel_size_m: float,
    raw_output_root: str | Path | None = None,
    native_runner: Callable[[ReSceneModelInput, Path, Path], NativeForwardResult]
    | None = None,
    source_validator: Callable[[Path], str] | None = None,
) -> ExecutorPaths:
    """Validate identities, execute once, and publish public plus raw evidence."""

    pair_digest = _validate_sha256(pair_sha256, label="pair SHA-256")
    checkpoint_digest = _validate_sha256(checkpoint_sha256, label="checkpoint SHA-256")
    try:
        voxel = float(neural_voxel_size_m)
    except (TypeError, ValueError) as error:
        raise ExecutorError("neural voxel size must be numeric") from error
    if not math.isfinite(voxel) or not math.isclose(
        voxel, 0.02, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ExecutorError("neural voxel size must equal 0.02 m")
    if feature_schema != "rgb_normals":
        raise ExecutorError("feature schema must equal rgb_normals")
    output_arrays_path = Path(os.path.abspath(os.fspath(output_arrays)))
    output_manifest_path = Path(os.path.abspath(os.fspath(output_manifest)))
    input_root = Path(os.path.abspath(os.fspath(input_sidecar)))
    raw_root = (
        Path(os.path.abspath(os.fspath(raw_output_root)))
        if raw_output_root is not None
        else input_root.parent / "backend_raw"
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (output_arrays_path, output_manifest_path, raw_root)
    ):
        raise ExecutorError("executor output already exists")
    checkpoint_path = Path(os.path.abspath(os.fspath(checkpoint)))
    checkpoint_content = _regular_file_bytes(
        checkpoint_path, label="ReScene checkpoint"
    )
    if _sha256_bytes(checkpoint_content) != checkpoint_digest:
        raise ExecutorError("ReScene checkpoint SHA-256 mismatch")
    del checkpoint_content
    checkout_path = Path(os.path.abspath(os.fspath(checkout)))
    source_commit = (source_validator or _validated_source_commit)(checkout_path)
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ExecutorError("source validator returned an invalid commit")
    model_input = _load_bound_sidecar(
        input_root,
        Path(pair_arrays),
        pair_sha256=pair_digest,
        feature_schema=feature_schema,
        neural_voxel_size_m=voxel,
    )
    forward = (native_runner or _native_forward)(
        model_input, checkout_path, checkpoint_path
    )
    _validate_forward_result(forward, model_count=len(model_input.model_visit_ids))
    processed = postprocess_native_predictions(
        pred_masks_mq=forward.pred_masks_mq,
        pred_logits_qc=forward.pred_logits_qc,
        adapter_to_model=model_input.adapter_to_model,
    )
    if not processed.temporal_query_ids:
        raise ExecutorError("native ReScene output has no positive-support query")
    raw_manifest, raw_arrays = _publish_raw_output(
        output_root=raw_root,
        processed=processed,
        model_input=model_input,
        forward=forward,
        checkpoint_sha256=checkpoint_digest,
        pair_sha256=pair_digest,
        source_commit=source_commit,
    )
    _publish_public_output(
        arrays_path=output_arrays_path,
        manifest_path=output_manifest_path,
        processed=processed,
        pair_sha256=pair_digest,
        checkpoint_sha256=checkpoint_digest,
        forward=forward,
    )
    return ExecutorPaths(
        output_arrays=output_arrays_path,
        output_manifest=output_manifest_path,
        raw_root=raw_root,
        raw_manifest=raw_manifest,
        raw_arrays=raw_arrays,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sidecar", type=Path, required=True)
    parser.add_argument("--pair-arrays", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--neural-voxel-size-m", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        run_rescene_pair_executor(
            input_sidecar=arguments.input_sidecar,
            pair_arrays=arguments.pair_arrays,
            output_arrays=arguments.output_arrays,
            output_manifest=arguments.output_manifest,
            checkpoint=arguments.checkpoint,
            checkout=arguments.checkout,
            pair_sha256=arguments.pair_sha256,
            checkpoint_sha256=arguments.checkpoint_sha256,
            feature_schema=arguments.feature_schema,
            neural_voxel_size_m=arguments.neural_voxel_size_m,
        )
    except (ExecutorError, OSError, ValueError) as error:
        _argument_parser().error(str(error))
    return 0


__all__ = [
    "ExecutorError",
    "ExecutorPaths",
    "NativeForwardResult",
    "ProcessedPredictions",
    "extract_model_state_dict",
    "main",
    "postprocess_native_predictions",
    "run_rescene_pair_executor",
]


if __name__ == "__main__":
    raise SystemExit(main())
