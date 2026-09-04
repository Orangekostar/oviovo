"""Fail-closed boundary for an optional external ReScene inference process."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import NeuralSampleMap, TemporalQueryEvidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPOSITORY_ROOT / "configs/external/ovi_rescene_sources.json"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OUTPUT_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "backend_name",
    "pair_sha256",
    "checkpoint_sha256",
    "temporal_query_ids",
    "runtime_s",
    "peak_memory_bytes",
    "output_arrays",
}
_OUTPUT_ARRAY_KEYS = {
    "token_indices",
    "query_masks",
    "token_scores",
    "query_scores",
}


class ReSceneBackendError(ValueError):
    """Raised for malformed or contradictory ReScene backend identities."""


def _digest(payload: object) -> str:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _git(checkout: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def _file_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReSceneBackendError(f"{label} is missing or a symlink")
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
        raise ReSceneBackendError(f"{label} changed while being read")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReSceneBackendError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ReSceneBackendError(f"{label} must be finite and at least {minimum}")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReSceneBackendError(f"{label} must be a non-negative integer")
    return value


def _write_pair_arrays(pair: NeuralSampleMap, path: Path) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            coordinates_xyzt=pair.coordinates_xyzt,
            features=pair.features,
            visit_ids=pair.visit_ids,
            source_visit_ids=pair.source_visit_ids,
            source_point_indices=pair.source_point_indices,
            source_to_token_offsets=pair.source_to_token_offsets,
        )
        stream.flush()
        os.fsync(stream.fileno())


def _materialize_checkpoint(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ReSceneBackendError("ReScene checkpoint is missing or a symlink")
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ReSceneBackendError("ReScene checkpoint must be a regular file")
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            writer.write(chunk)
            byte_count += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat(follow_symlinks=False)
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
    ) or byte_count != before.st_size:
        raise ReSceneBackendError("ReScene checkpoint changed while being read")
    return digest.hexdigest()


class ReSceneBackend:
    """Validate external identities before any learned dependency is imported."""

    def __init__(
        self,
        *,
        checkout: str | Path,
        checkpoint: str | Path,
        source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
        checkpoint_sha256: str | None = None,
        backend_name: str = "concerto",
        executor_command: Sequence[str] | None = None,
        expected_feature_schema: str = "rgb",
        expected_neural_voxel_size_m: float = 0.02,
        timeout_s: float = 3600.0,
    ) -> None:
        self.checkout = Path(checkout)
        self.checkpoint = Path(checkpoint)
        self.source_manifest = Path(source_manifest)
        self.checkpoint_sha256 = checkpoint_sha256
        self.backend_name = backend_name.strip().lower()
        if self.backend_name != "concerto":
            raise ReSceneBackendError("only the source-bound Concerto backend is enabled")
        if executor_command is None:
            self.executor_command: tuple[str, ...] | None = None
        elif (
            isinstance(executor_command, (str, bytes))
            or not isinstance(executor_command, Sequence)
            or not executor_command
            or any(not isinstance(item, str) or not item for item in executor_command)
        ):
            raise ReSceneBackendError("executor command must be a non-empty string sequence")
        else:
            self.executor_command = tuple(executor_command)
        if not isinstance(expected_feature_schema, str) or not expected_feature_schema:
            raise ReSceneBackendError("expected feature schema must be non-empty")
        self.expected_feature_schema = expected_feature_schema
        self.expected_neural_voxel_size_m = _finite_number(
            expected_neural_voxel_size_m,
            "expected neural voxel size",
            minimum=float(np.nextafter(0.0, 1.0)),
        )
        self.timeout_s = _finite_number(
            timeout_s,
            "executor timeout",
            minimum=float(np.nextafter(0.0, 1.0)),
        )

    def _backend_config_sha256(self) -> str:
        manifest_hash = (
            hashlib.sha256(self.source_manifest.read_bytes()).hexdigest()
            if self.source_manifest.is_file() and not self.source_manifest.is_symlink()
            else "missing"
        )
        return _digest(
            {
                "backend_name": self.backend_name,
                "source_manifest_sha256": manifest_hash,
                "checkpoint_sha256": self.checkpoint_sha256,
                "executor_command": self.executor_command,
                "expected_feature_schema": self.expected_feature_schema,
                "expected_neural_voxel_size_m": self.expected_neural_voxel_size_m,
            }
        )

    def _blocked(
        self, pair: NeuralSampleMap, status: str, reason: str
    ) -> TemporalQueryEvidence:
        result = TemporalQueryEvidence.blocked(
            status=status,
            backend_name=f"rescene:{self.backend_name}",
            backend_config_sha256=self._backend_config_sha256(),
            pair_sha256=pair.content_sha256(),
            diagnostics={"reason": reason},
        )
        validate_query_evidence(pair, result)
        return result

    def _validate_source(self) -> bool:
        try:
            if self.checkout.is_symlink() or not self.checkout.is_dir():
                return False
            manifest = json.loads(
                _file_bytes(self.source_manifest, "ReScene source manifest").decode(
                    "utf-8"
                )
            )
            declaration = manifest["sources"]["rescene"]
            commit = declaration["commit"]
            required_files = declaration["required_files"]
            if (
                manifest.get("status") != "EXTERNAL_SOURCE_PASS"
                or not isinstance(commit, str)
                or _GIT_SHA.fullmatch(commit) is None
                or not isinstance(required_files, list)
                or not required_files
            ):
                return False
            if _git(self.checkout, "rev-parse", "HEAD").decode().strip() != commit:
                return False
            if (
                _git(self.checkout, "rev-parse", f"{commit}^{{commit}}")
                .decode()
                .strip()
                != commit
            ):
                return False
            if _git(
                self.checkout,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return False
            seen: set[str] = set()
            for binding in required_files:
                if not isinstance(binding, dict):
                    return False
                relative = binding.get("path")
                sha256 = binding.get("sha256")
                byte_count = binding.get("byte_count")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or relative in seen
                    or not isinstance(sha256, str)
                    or _SHA256.fullmatch(sha256) is None
                    or type(byte_count) is not int
                    or byte_count <= 0
                ):
                    return False
                seen.add(relative)
                path = self.checkout / relative
                working = _file_bytes(path, f"ReScene source {relative}")
                frozen = _git(self.checkout, "show", f"{commit}:{relative}")
                if (
                    working != frozen
                    or len(frozen) != byte_count
                    or hashlib.sha256(frozen).hexdigest() != sha256
                ):
                    return False
            if "conf/backbone/concerto.yaml" not in seen:
                return False
            checkpoint = manifest.get("checkpoint")
            return isinstance(checkpoint, dict) and checkpoint.get(
                "selected_backbone"
            ) == "Concerto"
        except (
            KeyError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
            ReSceneBackendError,
        ):
            return False

    def infer(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        if not isinstance(pair, NeuralSampleMap):
            raise TypeError("pair must be a NeuralSampleMap")
        if not self._validate_source():
            return self._blocked(
                pair, "BLOCKED_EXTERNAL_SOURCE", "source_identity_mismatch"
            )
        if pair.feature_schema != self.expected_feature_schema:
            return self._blocked(
                pair,
                "BLOCKED_EXTERNAL_SOURCE",
                "adapter_feature_schema_mismatch",
            )
        if not math.isclose(
            pair.neural_voxel_size_m,
            self.expected_neural_voxel_size_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return self._blocked(
                pair,
                "BLOCKED_EXTERNAL_SOURCE",
                "adapter_neural_voxel_size_mismatch",
            )
        if self.checkpoint.is_symlink() or not self.checkpoint.is_file():
            return self._blocked(
                pair,
                "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
                "checkpoint_missing",
            )
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ReSceneBackendError(
                "checkpoint SHA-256 binding must be 64 lowercase hexadecimal digits"
            )
        if self.executor_command is None:
            checkpoint = _file_bytes(self.checkpoint, "ReScene checkpoint")
            if _sha256_bytes(checkpoint) != self.checkpoint_sha256:
                raise ReSceneBackendError("checkpoint SHA-256 mismatch")
            return self._blocked(
                pair, "BLOCKED_EXTERNAL_SOURCE", "backend_executor_unavailable"
            )
        return self._infer_subprocess(pair)

    def _infer_subprocess(self, pair: NeuralSampleMap) -> TemporalQueryEvidence:
        if self.executor_command is None or self.checkpoint_sha256 is None:
            raise AssertionError("validated executor and checkpoint identity required")
        pair_sha256 = pair.content_sha256()
        with tempfile.TemporaryDirectory(prefix="rescene-backend-") as temporary:
            root = Path(temporary)
            pair_arrays = root / "pair_arrays.npz"
            checkpoint = root / "checkpoint.ckpt"
            output_arrays = root / "query_evidence.npz"
            output_manifest = root / "query_evidence.json"
            _write_pair_arrays(pair, pair_arrays)
            observed_checkpoint_sha256 = _materialize_checkpoint(
                self.checkpoint, checkpoint
            )
            if observed_checkpoint_sha256 != self.checkpoint_sha256:
                raise ReSceneBackendError("checkpoint SHA-256 mismatch")
            command = [
                *self.executor_command,
                "--pair-arrays",
                str(pair_arrays),
                "--output-arrays",
                str(output_arrays),
                "--output-manifest",
                str(output_manifest),
                "--checkpoint",
                str(checkpoint),
                "--checkout",
                str(self.checkout),
                "--pair-sha256",
                pair_sha256,
                "--checkpoint-sha256",
                self.checkpoint_sha256,
                "--feature-schema",
                self.expected_feature_schema,
                "--neural-voxel-size-m",
                format(self.expected_neural_voxel_size_m, ".17g"),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    check=False,
                    capture_output=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    timeout=self.timeout_s,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ReSceneBackendError("ReScene executor failed to complete") from error
            if completed.returncode != 0:
                raise ReSceneBackendError(
                    f"ReScene executor exited with status {completed.returncode}"
                )
            if not self._validate_source():
                raise ReSceneBackendError("ReScene source changed during execution")
            manifest_content = _file_bytes(
                output_manifest, "ReScene output manifest"
            )
            try:
                manifest = json.loads(manifest_content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ReSceneBackendError(
                    "ReScene output manifest is not valid UTF-8 JSON"
                ) from error
            if not isinstance(manifest, dict) or set(manifest) != _OUTPUT_MANIFEST_KEYS:
                raise ReSceneBackendError("ReScene output manifest schema is invalid")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("status") != "PASS"
                or manifest.get("backend_name") != self.backend_name
                or manifest.get("pair_sha256") != pair_sha256
                or manifest.get("checkpoint_sha256") != self.checkpoint_sha256
            ):
                raise ReSceneBackendError("ReScene output identity is invalid")
            query_ids = manifest.get("temporal_query_ids")
            if (
                not isinstance(query_ids, list)
                or not query_ids
                or any(not isinstance(item, str) or not item for item in query_ids)
                or len(query_ids) != len(set(query_ids))
            ):
                raise ReSceneBackendError("ReScene temporal query IDs are invalid")
            runtime_s = _finite_number(manifest.get("runtime_s"), "runtime_s")
            peak_memory_bytes = _nonnegative_integer(
                manifest.get("peak_memory_bytes"), "peak_memory_bytes"
            )
            record = manifest.get("output_arrays")
            if not isinstance(record, dict) or set(record) != {
                "path",
                "sha256",
                "byte_count",
            }:
                raise ReSceneBackendError("ReScene output array binding is invalid")
            if (
                record.get("path") != output_arrays.name
                or not isinstance(record.get("sha256"), str)
                or _SHA256.fullmatch(record["sha256"]) is None
                or type(record.get("byte_count")) is not int
                or record["byte_count"] <= 0
            ):
                raise ReSceneBackendError("ReScene output array binding is invalid")
            arrays_content = _file_bytes(output_arrays, "ReScene output arrays")
            if (
                _sha256_bytes(arrays_content) != record["sha256"]
                or len(arrays_content) != record["byte_count"]
            ):
                raise ReSceneBackendError("ReScene output array binding mismatch")
            arrays = self._load_output_arrays(
                arrays_content,
                query_count=len(query_ids),
                token_count=len(pair.visit_ids),
            )
            token_indices = arrays.pop("token_indices")
            raw_masks = arrays.pop("query_masks")
            raw_scores = arrays.pop("token_scores")
            restored_masks = np.empty_like(raw_masks)
            restored_scores = np.empty_like(raw_scores)
            restored_masks[:, token_indices] = raw_masks
            restored_scores[:, token_indices] = raw_scores
            selected = np.any(restored_masks, axis=0)
            if any(
                not np.any(selected[pair.visit_ids == visit_id])
                for visit_id in (0, 1)
            ):
                raise ReSceneBackendError(
                    "ReScene queries must cover tokens from both visits"
                )
            evidence = TemporalQueryEvidence(
                status="PASS",
                backend_name=f"rescene:{self.backend_name}",
                backend_config_sha256=self._backend_config_sha256(),
                pair_sha256=pair_sha256,
                temporal_query_ids=tuple(query_ids),
                query_masks=restored_masks,
                token_scores=restored_scores,
                query_scores=arrays["query_scores"],
                checkpoint_sha256=self.checkpoint_sha256,
                ranking_eligible=True,
                runtime_s=runtime_s,
                peak_memory_bytes=peak_memory_bytes,
                diagnostics={
                    "output_arrays_sha256": record["sha256"],
                    "output_manifest_sha256": _sha256_bytes(manifest_content),
                    "subprocess_stderr_sha256": _sha256_bytes(completed.stderr),
                    "subprocess_stdout_sha256": _sha256_bytes(completed.stdout),
                    "token_order_restored": "true",
                },
            )
            validate_query_evidence(pair, evidence)
            return evidence

    @staticmethod
    def _load_output_arrays(
        content: bytes, *, query_count: int, token_count: int
    ) -> dict[str, np.ndarray]:
        try:
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                if set(archive.files) != _OUTPUT_ARRAY_KEYS:
                    raise ReSceneBackendError("ReScene output array keys are invalid")
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as error:
            if isinstance(error, ReSceneBackendError):
                raise
            raise ReSceneBackendError("ReScene output arrays are invalid") from error
        token_indices = arrays["token_indices"]
        if (
            token_indices.shape != (token_count,)
            or not np.issubdtype(token_indices.dtype, np.integer)
            or np.issubdtype(token_indices.dtype, np.bool_)
            or not np.array_equal(
                np.sort(token_indices), np.arange(token_count, dtype=np.int64)
            )
        ):
            raise ReSceneBackendError(
                "ReScene token indices must be an exact adapter-token permutation"
            )
        query_masks = arrays["query_masks"]
        token_scores = arrays["token_scores"]
        query_scores = arrays["query_scores"]
        if query_masks.dtype != np.bool_ or query_masks.shape != (
            query_count,
            token_count,
        ):
            raise ReSceneBackendError("ReScene query mask shape or dtype is invalid")
        if (
            token_scores.shape != (query_count, token_count)
            or not np.issubdtype(token_scores.dtype, np.floating)
            or query_scores.shape != (query_count,)
            or not np.issubdtype(query_scores.dtype, np.floating)
            or not np.all(np.isfinite(token_scores))
            or not np.all(np.isfinite(query_scores))
            or np.any(token_scores < 0.0)
            or np.any(token_scores > 1.0)
            or np.any(query_scores < 0.0)
            or np.any(query_scores > 1.0)
        ):
            raise ReSceneBackendError("ReScene query score arrays are invalid")
        return {
            "token_indices": token_indices.astype(np.int64, copy=False),
            "query_masks": query_masks,
            "token_scores": token_scores,
            "query_scores": query_scores,
        }
