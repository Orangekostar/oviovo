"""Strict sidecar contract for raw 2D evidence attached to ReScene M rows."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OBSERVATION_METADATA_COLUMNS = (
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "extent_x_m",
    "extent_y_m",
    "extent_z_m",
    "valid_depth_fraction",
    "view_direction_x",
    "view_direction_y",
    "view_direction_z",
    "depth_residual_mean_m",
    "depth_residual_std_m",
    "visit_embedding_input",
    "border_contact_fraction",
    "pixel_area",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARRAY_KEYS = frozenset(
    {
        "region_visit_ids",
        "region_frame_ids",
        "region_features",
        "region_metadata",
        "csr_indptr",
        "csr_model_indices",
        "csr_weights",
        "region_reliability",
        "model_visit_ids",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "status",
        "content_sha256",
        "pair_id",
        "model_input_sha256",
        "region_keys",
        "metadata_columns",
        "source_manifest",
        "arrays",
    }
)


class ObservationBankError(ValueError):
    """Raised when observation provenance or index domains disagree."""


def _readonly(values: object, dtype: np.dtype | type, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ObservationBankError(f"{name} must have {ndim} dimensions")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_vector(values: object, dtype: np.dtype | type, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ObservationBankError(f"{name} must be a one-dimensional integer array")
    return _readonly(raw, dtype, 1, name)


def _canonical_json(value: object, name: str) -> tuple[bytes, object]:
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
        raise ObservationBankError(f"{name} must be canonical JSON data") from error
    return encoded, copied


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _content_digest(bank: ObservationBank) -> str:
    payload = {
        "pair_id": bank.pair_id,
        "model_input_sha256": bank.model_input_sha256,
        "region_keys": list(bank.region_keys),
        "metadata_columns": list(OBSERVATION_METADATA_COLUMNS),
        "source_manifest": bank.source_manifest,
        "arrays": {
            name: _array_record(getattr(bank, name)) for name in sorted(_ARRAY_KEYS)
        },
    }
    encoded, _ = _canonical_json(payload, "observation content")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationBank:
    """Frozen R-to-M sparse observation evidence for one two-visit pair."""

    pair_id: str
    model_input_sha256: str
    region_keys: tuple[str, ...]
    region_visit_ids: np.ndarray
    region_frame_ids: np.ndarray
    region_features: np.ndarray
    region_metadata: np.ndarray
    csr_indptr: np.ndarray
    csr_model_indices: np.ndarray
    csr_weights: np.ndarray
    region_reliability: np.ndarray
    model_visit_ids: np.ndarray
    source_manifest: dict[str, object]
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ObservationBankError("pair_id must be a non-empty string")
        if not isinstance(self.model_input_sha256, str) or _SHA256.fullmatch(
            self.model_input_sha256
        ) is None:
            raise ObservationBankError("model_input_sha256 must be a lowercase SHA-256")
        keys = tuple(self.region_keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ObservationBankError("region_keys must contain non-empty strings")
        if len(set(keys)) != len(keys):
            raise ObservationBankError("region_keys must be unique")

        region_visits = _integer_vector(
            self.region_visit_ids, np.int8, "region_visit_ids"
        )
        region_frames = _integer_vector(
            self.region_frame_ids, np.int64, "region_frame_ids"
        )
        features = _readonly(
            self.region_features, np.float32, 2, "region_features"
        )
        metadata = _readonly(
            self.region_metadata, np.float32, 2, "region_metadata"
        )
        indptr = _integer_vector(self.csr_indptr, np.int64, "csr_indptr")
        indices = _integer_vector(
            self.csr_model_indices, np.int64, "csr_model_indices"
        )
        weights = _readonly(self.csr_weights, np.float32, 1, "csr_weights")
        reliability = _readonly(
            self.region_reliability, np.float32, 1, "region_reliability"
        )
        model_visits = _integer_vector(
            self.model_visit_ids, np.int8, "model_visit_ids"
        )

        region_count = len(keys)
        if any(
            len(value) != region_count
            for value in (region_visits, region_frames, features, metadata, reliability)
        ):
            raise ObservationBankError("region arrays must have R aligned rows")
        if features.shape[1] == 0:
            raise ObservationBankError("region_features must retain a feature dimension")
        if metadata.shape[1] != len(OBSERVATION_METADATA_COLUMNS):
            raise ObservationBankError("region_metadata columns do not match the contract")
        if np.any(region_frames < 0):
            raise ObservationBankError("region_frame_ids must be nonnegative")
        if np.any(~np.isfinite(features)) or np.any(~np.isfinite(metadata)):
            raise ObservationBankError("region feature and metadata values must be finite")
        if np.any(~np.isfinite(reliability)) or np.any(
            (reliability < 0.0) | (reliability > 1.0)
        ):
            raise ObservationBankError("region_reliability must be finite in [0, 1]")
        if np.any((region_visits != 0) & (region_visits != 1)) or np.any(
            (model_visits != 0) & (model_visits != 1)
        ):
            raise ObservationBankError("visit IDs must be 0 or 1")

        edge_count = len(indices)
        if len(weights) != edge_count:
            raise ObservationBankError("CSR indices and weights must have E aligned rows")
        if indptr.shape != (region_count + 1,) or indptr[0] != 0 or indptr[-1] != edge_count:
            raise ObservationBankError("csr_indptr must exactly span every edge")
        if np.any(np.diff(indptr) < 0):
            raise ObservationBankError("csr_indptr must be monotonic")
        if np.any(indices < 0) or np.any(indices >= len(model_visits)):
            raise ObservationBankError("CSR model index is outside the M domain")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ObservationBankError("CSR weights must be finite and nonnegative")
        for region_index in range(region_count):
            edge_slice = slice(indptr[region_index], indptr[region_index + 1])
            if np.any(model_visits[indices[edge_slice]] != region_visits[region_index]):
                raise ObservationBankError("a region-model CSR edge crosses visits")

        if not isinstance(self.source_manifest, dict):
            raise ObservationBankError("source_manifest must be a dict")
        _, source_manifest = _canonical_json(self.source_manifest, "source_manifest")
        if not isinstance(source_manifest, dict):
            raise ObservationBankError("source_manifest must be a JSON object")

        object.__setattr__(self, "pair_id", self.pair_id.strip())
        object.__setattr__(self, "region_keys", keys)
        object.__setattr__(self, "region_visit_ids", region_visits)
        object.__setattr__(self, "region_frame_ids", region_frames)
        object.__setattr__(self, "region_features", features)
        object.__setattr__(self, "region_metadata", metadata)
        object.__setattr__(self, "csr_indptr", indptr)
        object.__setattr__(self, "csr_model_indices", indices)
        object.__setattr__(self, "csr_weights", weights)
        object.__setattr__(self, "region_reliability", reliability)
        object.__setattr__(self, "model_visit_ids", model_visits)
        object.__setattr__(self, "source_manifest", source_manifest)
        object.__setattr__(self, "_content_sha256", _content_digest(self))

    def content_sha256(self) -> str:
        return self._content_sha256


@dataclass(frozen=True, slots=True)
class ObservationBankArtifactPaths:
    root: Path
    manifest: Path
    arrays: Path


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def save_observation_bank(
    bank: ObservationBank, output_root: str | Path
) -> ObservationBankArtifactPaths:
    """Atomically publish a source-bound observation bank."""

    if not isinstance(bank, ObservationBank):
        raise TypeError("bank must be ObservationBank")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationBankError(f"observation bank already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                **{name: getattr(bank, name) for name in sorted(_ARRAY_KEYS)},
            )
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_BANK_V1",
            "status": "PASS",
            "content_sha256": bank.content_sha256(),
            "pair_id": bank.pair_id,
            "model_input_sha256": bank.model_input_sha256,
            "region_keys": list(bank.region_keys),
            "metadata_columns": list(OBSERVATION_METADATA_COLUMNS),
            "source_manifest": bank.source_manifest,
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
    return ObservationBankArtifactPaths(
        output, output / "manifest.json", output / "arrays.npz"
    )


def load_observation_bank(output_root: str | Path) -> ObservationBank:
    """Load an observation bank and verify its byte and semantic identity."""

    root = Path(output_root).absolute()
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, arrays_path)):
        raise ObservationBankError("observation bank files are missing or symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationBankError("observation bank manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ObservationBankError("observation bank manifest schema is invalid")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_RESCENE_OBSERVATION_BANK_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("metadata_columns") != list(OBSERVATION_METADATA_COLUMNS)
    ):
        raise ObservationBankError("observation bank manifest identity is invalid")
    if manifest.get("arrays") != _file_record(arrays_path):
        raise ObservationBankError("observation bank arrays SHA-256 or byte count mismatch")
    try:
        with np.load(io.BytesIO(arrays_path.read_bytes()), allow_pickle=False) as source:
            if set(source.files) != _ARRAY_KEYS:
                raise ObservationBankError("observation bank array schema is invalid")
            arrays = {name: source[name] for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ObservationBankError):
            raise
        raise ObservationBankError("observation bank arrays are invalid") from error
    result = ObservationBank(
        pair_id=manifest["pair_id"],
        model_input_sha256=manifest["model_input_sha256"],
        region_keys=tuple(manifest["region_keys"]),
        source_manifest=manifest["source_manifest"],
        **arrays,
    )
    if result.content_sha256() != manifest.get("content_sha256"):
        raise ObservationBankError("observation bank content SHA-256 mismatch")
    return result
