"""Training-only targets for the OVI observation-query ReScene model."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from src.oviv2.observation_query.contracts import ObservationBank
from src.oviv2.rescene_input_bridge import ReSceneModelInput

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IGNORE_CLASS_TARGET = -100
_TARGET_ARRAY_NAMES = (
    "instance_masks",
    "label_valid",
    "class_targets",
    "class_valid",
    "region_instance_mass",
    "region_label_valid",
)


class ObservationTrainingDataError(ValueError):
    """Raised when training supervision or its provenance is inconsistent."""


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObservationTrainingDataError(f"{name} must be a lowercase SHA-256")
    return value


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
        raise ObservationTrainingDataError(f"{name} must be canonical JSON data") from error
    return encoded, copied


def _readonly(values: object, dtype: np.dtype | type, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ObservationTrainingDataError(f"{name} must have {ndim} dimensions")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_vector(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ObservationTrainingDataError(f"{name} must be a one-dimensional integer array")
    return _readonly(raw, np.int64, 1, name)


def _positive_ids(values: object, name: str) -> frozenset[int]:
    try:
        normalized = frozenset(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ObservationTrainingDataError(f"{name} must contain positive integers") from error
    if any(value <= 0 for value in normalized):
        raise ObservationTrainingDataError(f"{name} must contain positive integers")
    return normalized


@dataclass(frozen=True, slots=True)
class NativeVisitLabels:
    """High-resolution native labels already expressed in the reference frame."""

    visit_id: int
    points_reference_xyz: np.ndarray
    semantic_ids: np.ndarray
    instance_ids: np.ndarray
    segment_ids: np.ndarray
    source_sha256: str

    def __post_init__(self) -> None:
        if type(self.visit_id) is not int or self.visit_id not in (0, 1):
            raise ObservationTrainingDataError("visit_id must be 0 or 1")
        points = _readonly(
            self.points_reference_xyz, np.float64, 2, "points_reference_xyz"
        )
        if points.shape[1:] != (3,) or not len(points) or np.any(~np.isfinite(points)):
            raise ObservationTrainingDataError(
                "points_reference_xyz must have non-empty finite shape (N, 3)"
            )
        semantic = _integer_vector(self.semantic_ids, "semantic_ids")
        instance = _integer_vector(self.instance_ids, "instance_ids")
        segment = _integer_vector(self.segment_ids, "segment_ids")
        if any(len(value) != len(points) for value in (semantic, instance, segment)):
            raise ObservationTrainingDataError("native label arrays must have N aligned rows")
        object.__setattr__(self, "points_reference_xyz", points)
        object.__setattr__(self, "semantic_ids", semantic)
        object.__setattr__(self, "instance_ids", instance)
        object.__setattr__(self, "segment_ids", segment)
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "label source"))


@dataclass(frozen=True, slots=True)
class PairIdentityRules:
    """Explicit official cross-visit identities; equal numeric IDs are not implicit."""

    rescan_to_reference: Mapping[int, int]
    ambiguous_instance_ids_by_visit: tuple[frozenset[int], frozenset[int]]
    removed_reference_ids: frozenset[int]
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.rescan_to_reference, Mapping):
            raise ObservationTrainingDataError("rescan_to_reference must be a mapping")
        mapping: dict[int, int] = {}
        for raw_rescan, raw_reference in self.rescan_to_reference.items():
            if (
                isinstance(raw_rescan, bool)
                or isinstance(raw_reference, bool)
                or not isinstance(raw_rescan, Integral)
                or not isinstance(raw_reference, Integral)
                or int(raw_rescan) <= 0
                or int(raw_reference) <= 0
            ):
                raise ObservationTrainingDataError("identity mappings must use positive integers")
            mapping[int(raw_rescan)] = int(raw_reference)
        ambiguity = self.ambiguous_instance_ids_by_visit
        if not isinstance(ambiguity, tuple) or len(ambiguity) != 2:
            raise ObservationTrainingDataError("ambiguity IDs must contain both visits")
        normalized_ambiguity = (
            _positive_ids(ambiguity[0], "visit-0 ambiguity IDs"),
            _positive_ids(ambiguity[1], "visit-1 ambiguity IDs"),
        )
        removed = _positive_ids(self.removed_reference_ids, "removed reference IDs")
        object.__setattr__(self, "rescan_to_reference", dict(sorted(mapping.items())))
        object.__setattr__(self, "ambiguous_instance_ids_by_visit", normalized_ambiguity)
        object.__setattr__(self, "removed_reference_ids", removed)
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "identity source"))

    def temporal_key(self, visit_id: int, instance_id: int) -> str:
        if visit_id not in (0, 1) or instance_id <= 0:
            raise ObservationTrainingDataError("temporal identity input is invalid")
        if instance_id in self.ambiguous_instance_ids_by_visit[visit_id]:
            return f"visit:{visit_id}:instance:{instance_id}"
        if visit_id == 0:
            return f"reference:{instance_id}"
        reference_id = self.rescan_to_reference.get(instance_id)
        if (
            reference_id is None
            or reference_id in self.ambiguous_instance_ids_by_visit[0]
        ):
            return f"visit:1:instance:{instance_id}"
        return f"reference:{reference_id}"


@dataclass(frozen=True, slots=True)
class OfficialPairTrainingMetadata:
    reference_scan_uuid: str
    rescan_uuid: str
    rescan_to_reference_row: np.ndarray
    identity_rules: PairIdentityRules

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_scan_uuid, str)
            or not self.reference_scan_uuid
            or not isinstance(self.rescan_uuid, str)
            or not self.rescan_uuid
            or self.reference_scan_uuid == self.rescan_uuid
        ):
            raise ObservationTrainingDataError("official pair scan identities are invalid")
        matrix = _row_transform(
            self.rescan_to_reference_row, "rescan_to_reference_row"
        )
        if not isinstance(self.identity_rules, PairIdentityRules):
            raise TypeError("identity_rules must be PairIdentityRules")
        object.__setattr__(self, "rescan_to_reference_row", matrix)


@dataclass(frozen=True, slots=True)
class LabelTransferConfig:
    maximum_distance_m: float = 0.04
    minimum_neighbors: int = 3
    minimum_consensus_fraction: float = 0.75
    maximum_neighbors: int = 8
    trusted_background_raw_ids: tuple[int, ...] = (1, 2)
    ignored_raw_semantic_ids: tuple[int, ...] = (0, 255)
    minimum_region_valid_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "maximum_distance_m",
            "minimum_consensus_fraction",
            "minimum_region_valid_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ObservationTrainingDataError(f"{name} must be finite")
        if self.maximum_distance_m <= 0.0:
            raise ObservationTrainingDataError("maximum_distance_m must be positive")
        if not 0.5 < self.minimum_consensus_fraction <= 1.0:
            raise ObservationTrainingDataError(
                "minimum_consensus_fraction must be in (0.5, 1]"
            )
        if not 0.0 <= self.minimum_region_valid_fraction <= 1.0:
            raise ObservationTrainingDataError(
                "minimum_region_valid_fraction must be in [0, 1]"
            )
        if type(self.minimum_neighbors) is not int or self.minimum_neighbors <= 0:
            raise ObservationTrainingDataError("minimum_neighbors must be positive")
        if (
            type(self.maximum_neighbors) is not int
            or self.maximum_neighbors < self.minimum_neighbors
        ):
            raise ObservationTrainingDataError(
                "maximum_neighbors must be at least minimum_neighbors"
            )
        background = tuple(sorted(_nonnegative_unique(self.trusted_background_raw_ids, "background IDs")))
        ignored = tuple(sorted(_nonnegative_unique(self.ignored_raw_semantic_ids, "ignored IDs")))
        if set(background).intersection(ignored):
            raise ObservationTrainingDataError("background and ignored semantic IDs overlap")
        object.__setattr__(self, "maximum_distance_m", float(self.maximum_distance_m))
        object.__setattr__(self, "minimum_consensus_fraction", float(self.minimum_consensus_fraction))
        object.__setattr__(self, "minimum_region_valid_fraction", float(self.minimum_region_valid_fraction))
        object.__setattr__(self, "trusted_background_raw_ids", background)
        object.__setattr__(self, "ignored_raw_semantic_ids", ignored)


def _nonnegative_unique(values: object, name: str) -> set[int]:
    if not isinstance(values, tuple) or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in values
    ):
        raise ObservationTrainingDataError(f"{name} must be a tuple of nonnegative integers")
    if len(set(values)) != len(values):
        raise ObservationTrainingDataError(f"{name} must be unique")
    return {int(value) for value in values}


def _row_transform(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ObservationTrainingDataError(f"{name} must be numeric") from error
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if (
        matrix.shape != (4, 4)
        or np.any(~np.isfinite(matrix))
        or not np.allclose(matrix[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ObservationTrainingDataError(f"{name} must be a finite row-vector affine matrix")
    result = np.array(matrix, copy=True, order="C")
    result.setflags(write=False)
    return result


def _regular_file(path: str | Path, name: str) -> Path:
    result = Path(path).absolute()
    if result.is_symlink() or not result.is_file():
        raise ObservationTrainingDataError(f"{name} is unavailable")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_native_processed_labels(
    source_path: str | Path,
    *,
    visit_id: int,
) -> NativeVisitLabels:
    """Load RIO columns already written in the shared reference frame."""

    source = _regular_file(source_path, "native processed labels")
    try:
        values = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ObservationTrainingDataError("native processed labels cannot be decoded") from error
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] < 12
        or not len(values)
        or np.any(~np.isfinite(values[:, :12]))
    ):
        raise ObservationTrainingDataError(
            "native processed labels must have non-empty finite shape (N, >=12)"
        )
    label_columns = values[:, [9, 10, 11]]
    if not np.array_equal(label_columns, np.floor(label_columns)):
        raise ObservationTrainingDataError("native segment, semantic, and instance labels must be integer-valued")
    limits = np.iinfo(np.int64)
    if np.any(label_columns < limits.min) or np.any(label_columns > limits.max):
        raise ObservationTrainingDataError("native integer labels exceed int64")
    return NativeVisitLabels(
        visit_id=visit_id,
        points_reference_xyz=values[:, :3].astype(np.float64),
        segment_ids=label_columns[:, 0].astype(np.int64),
        semantic_ids=label_columns[:, 1].astype(np.int64),
        instance_ids=label_columns[:, 2].astype(np.int64),
        source_sha256=_file_sha256(source),
    )


def _official_positive(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ObservationTrainingDataError(f"{name} must be a positive integer")
    return value


def load_official_pair_training_metadata(
    metadata_path: str | Path,
    *,
    reference_scan_uuid: str,
    rescan_uuid: str,
    reference_instance_ids: object,
    rescan_instance_ids: object,
) -> OfficialPairTrainingMetadata:
    """Expand one official 3RScan pair into explicit, ambiguity-safe identities."""

    source = _regular_file(metadata_path, "3RScan metadata")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingDataError("3RScan metadata cannot be decoded") from error
    if not isinstance(payload, list):
        raise ObservationTrainingDataError("3RScan metadata root must be a list")
    environments = [
        record
        for record in payload
        if isinstance(record, Mapping) and record.get("reference") == reference_scan_uuid
    ]
    if len(environments) != 1:
        raise ObservationTrainingDataError("official reference environment is not unique")
    environment = environments[0]
    scans = environment.get("scans")
    if not isinstance(scans, list):
        raise ObservationTrainingDataError("official environment scans are invalid")
    matches = [
        record
        for record in scans
        if isinstance(record, Mapping) and record.get("reference") == rescan_uuid
    ]
    if len(matches) != 1:
        raise ObservationTrainingDataError("official rescan is not unique")
    rescan = matches[0]
    transform = _row_transform(rescan.get("transform"), "official global transform")
    reference_ids = _positive_ids(reference_instance_ids, "reference instance IDs")
    rescan_ids = _positive_ids(rescan_instance_ids, "rescan instance IDs")
    mapping: dict[int, int] = {}

    def bind(rescan_id: int, reference_id: int) -> None:
        if rescan_id not in rescan_ids or reference_id not in reference_ids:
            raise ObservationTrainingDataError("official identity is absent from native labels")
        previous = mapping.setdefault(rescan_id, reference_id)
        if previous != reference_id:
            raise ObservationTrainingDataError("official identity mappings conflict")

    for change_kind in ("rigid", "nonrigid"):
        records = rescan.get(change_kind, [])
        if not isinstance(records, list):
            raise ObservationTrainingDataError(f"official {change_kind} changes are invalid")
        for record in records:
            if isinstance(record, Mapping):
                reference_id = _official_positive(
                    record.get("instance_reference"), f"{change_kind} reference"
                )
                rescan_id = _official_positive(
                    record.get("instance_rescan"), f"{change_kind} rescan"
                )
                if change_kind == "rigid":
                    _row_transform(record.get("transform"), "official rigid transform")
                    symmetry = record.get("symmetry")
                    if type(symmetry) is not int or symmetry < 0:
                        raise ObservationTrainingDataError("official rigid symmetry is invalid")
            elif change_kind == "nonrigid":
                reference_id = _official_positive(record, "nonrigid reference")
                rescan_id = reference_id
            else:
                raise ObservationTrainingDataError("official rigid change is invalid")
            bind(rescan_id, reference_id)
    for shared_id in sorted(reference_ids.intersection(rescan_ids)):
        bind(shared_id, shared_id)
    removed_records = rescan.get("removed", [])
    if not isinstance(removed_records, list):
        raise ObservationTrainingDataError("official removed changes are invalid")
    removed: set[int] = set()
    for record in removed_records:
        value = record.get("instance_reference") if isinstance(record, Mapping) else record
        reference_id = _official_positive(value, "removed reference")
        if reference_id not in reference_ids:
            raise ObservationTrainingDataError("removed reference is absent from native labels")
        removed.add(reference_id)
    ambiguity = environment.get("ambiguity", [])
    if not isinstance(ambiguity, list):
        raise ObservationTrainingDataError("official ambiguity is invalid")
    ambiguous_ids: set[int] = set()
    for group in ambiguity:
        if not isinstance(group, list) or not group:
            raise ObservationTrainingDataError("official ambiguity group is invalid")
        group_ids: set[int] = set()
        for record in group:
            if not isinstance(record, Mapping):
                raise ObservationTrainingDataError("official ambiguity record is invalid")
            group_ids.add(
                _official_positive(record.get("instance_source"), "ambiguity source")
            )
            group_ids.add(
                _official_positive(record.get("instance_target"), "ambiguity target")
            )
            _row_transform(record.get("transform"), "official ambiguity transform")
        if len(group_ids) < 2:
            raise ObservationTrainingDataError("official ambiguity group is degenerate")
        ambiguous_ids.update(group_ids)
    identity = PairIdentityRules(
        rescan_to_reference=mapping,
        ambiguous_instance_ids_by_visit=(
            frozenset(ambiguous_ids.intersection(reference_ids)),
            frozenset(ambiguous_ids.intersection(rescan_ids)),
        ),
        removed_reference_ids=frozenset(removed),
        source_sha256=_file_sha256(source),
    )
    return OfficialPairTrainingMetadata(
        reference_scan_uuid=reference_scan_uuid,
        rescan_uuid=rescan_uuid,
        rescan_to_reference_row=transform,
        identity_rules=identity,
    )


def _tensor(
    values: object,
    *,
    dtype: torch.dtype,
    dimensions: int,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=dtype, device="cpu").detach().clone().contiguous()
    if result.ndim != dimensions:
        raise ObservationTrainingDataError(f"{name} must have {dimensions} dimensions")
    result.requires_grad_(False)
    return result


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    array = value.detach().cpu().contiguous().numpy()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class ObservationTrainingSample:
    model_input: ReSceneModelInput
    observations: ObservationBank
    instance_masks: torch.Tensor
    label_valid: torch.Tensor
    class_targets: torch.Tensor
    class_valid: torch.Tensor
    temporal_identity_keys: tuple[str, ...]
    ambiguity_metadata: dict[str, object]
    region_instance_mass: torch.Tensor
    region_label_valid: torch.Tensor
    target_provenance: dict[str, object]
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_input, ReSceneModelInput):
            raise TypeError("model_input must be ReSceneModelInput")
        if not isinstance(self.observations, ObservationBank):
            raise TypeError("observations must be ObservationBank")
        if self.observations.model_input_sha256 != self.model_input.content_sha256():
            raise ObservationTrainingDataError("observation and model input identities differ")
        if not np.array_equal(
            self.observations.model_visit_ids, self.model_input.model_visit_ids
        ):
            raise ObservationTrainingDataError("observation and model visit domains differ")
        masks = _tensor(
            self.instance_masks,
            dtype=torch.float32,
            dimensions=2,
            name="instance_masks",
        )
        valid = _tensor(
            self.label_valid, dtype=torch.bool, dimensions=1, name="label_valid"
        )
        classes = _tensor(
            self.class_targets,
            dtype=torch.int64,
            dimensions=1,
            name="class_targets",
        )
        class_valid = _tensor(
            self.class_valid, dtype=torch.bool, dimensions=1, name="class_valid"
        )
        region_mass = _tensor(
            self.region_instance_mass,
            dtype=torch.float32,
            dimensions=2,
            name="region_instance_mass",
        )
        region_valid = _tensor(
            self.region_label_valid,
            dtype=torch.bool,
            dimensions=1,
            name="region_label_valid",
        )
        keys = tuple(self.temporal_identity_keys)
        model_count = len(self.model_input.model_visit_ids)
        region_count = len(self.observations.region_keys)
        instance_count = len(keys)
        if (
            masks.shape != (instance_count, model_count)
            or valid.shape != (model_count,)
            or classes.shape != (instance_count,)
            or class_valid.shape != (instance_count,)
            or region_mass.shape != (region_count, instance_count + 1)
            or region_valid.shape != (region_count,)
        ):
            raise ObservationTrainingDataError("training target shapes do not align to K, M, and R")
        if (
            any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
            or tuple(sorted(keys)) != keys
        ):
            raise ObservationTrainingDataError("temporal identity keys must be unique and sorted")
        if not torch.isfinite(masks).all() or torch.any((masks < 0.0) | (masks > 1.0)):
            raise ObservationTrainingDataError("instance masks must be finite in [0, 1]")
        if torch.any(masks[:, ~valid] != 0.0):
            raise ObservationTrainingDataError("unknown M rows cannot become mask negatives")
        if instance_count and torch.any(masks.sum(dim=0) > 1.0 + 1e-6):
            raise ObservationTrainingDataError("an M row belongs to multiple strong identities")
        if torch.any(class_valid & (classes < 0)) or torch.any(
            ~class_valid & (classes != _IGNORE_CLASS_TARGET)
        ):
            raise ObservationTrainingDataError("class targets disagree with class_valid")
        if not torch.isfinite(region_mass).all() or torch.any(region_mass < 0.0):
            raise ObservationTrainingDataError("region mass must be finite and nonnegative")
        if torch.any(region_valid & ~torch.isclose(region_mass.sum(1), torch.ones(region_count))):
            raise ObservationTrainingDataError("valid region mass rows must sum to one")
        _, ambiguity = _canonical_json(self.ambiguity_metadata, "ambiguity_metadata")
        _, provenance = _canonical_json(self.target_provenance, "target_provenance")
        if not isinstance(ambiguity, dict) or not isinstance(provenance, dict):
            raise ObservationTrainingDataError("training metadata must be JSON objects")
        object.__setattr__(self, "instance_masks", masks)
        object.__setattr__(self, "label_valid", valid)
        object.__setattr__(self, "class_targets", classes)
        object.__setattr__(self, "class_valid", class_valid)
        object.__setattr__(self, "temporal_identity_keys", keys)
        object.__setattr__(self, "ambiguity_metadata", ambiguity)
        object.__setattr__(self, "region_instance_mass", region_mass)
        object.__setattr__(self, "region_label_valid", region_valid)
        object.__setattr__(self, "target_provenance", provenance)
        payload = {
            "pair_id": self.observations.pair_id,
            "model_input_sha256": self.model_input.content_sha256(),
            "observation_sha256": self.observations.content_sha256(),
            "temporal_identity_keys": keys,
            "ambiguity_metadata": ambiguity,
            "target_provenance": provenance,
            "arrays": {
                name: _tensor_record(getattr(self, name))
                for name in _TARGET_ARRAY_NAMES
            },
        }
        encoded, _ = _canonical_json(payload, "training sample")
        object.__setattr__(self, "_content_sha256", hashlib.sha256(encoded).hexdigest())

    def content_sha256(self) -> str:
        return self._content_sha256


def _transfer_visit_labels(
    model_points: np.ndarray,
    labels: NativeVisitLabels,
    config: LabelTransferConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.zeros(len(model_points), dtype=np.bool_)
    semantics = np.full(len(model_points), -1, dtype=np.int64)
    instances = np.full(len(model_points), -1, dtype=np.int64)
    tree = cKDTree(labels.points_reference_xyz)
    background = set(config.trusted_background_raw_ids)
    ignored = set(config.ignored_raw_semantic_ids)
    for model_index, point in enumerate(model_points):
        candidates = tree.query_ball_point(point, config.maximum_distance_m)
        if len(candidates) < config.minimum_neighbors:
            continue
        distances = np.linalg.norm(labels.points_reference_xyz[candidates] - point, axis=1)
        ordered = sorted(
            zip(distances.tolist(), candidates), key=lambda item: (item[0], item[1])
        )[: config.maximum_neighbors]
        tokens: list[tuple[str, int, int] | None] = []
        for _, native_index in ordered:
            semantic_id = int(labels.semantic_ids[native_index])
            instance_id = int(labels.instance_ids[native_index])
            if semantic_id in ignored:
                tokens.append(None)
            elif semantic_id in background:
                tokens.append(("background", semantic_id, -1))
            elif instance_id > 0:
                tokens.append(("object", semantic_id, instance_id))
            else:
                tokens.append(None)
        counts = Counter(token for token in tokens if token is not None)
        if not counts:
            continue
        winner, count = min(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count / len(tokens) + 1e-12 < config.minimum_consensus_fraction:
            continue
        valid[model_index] = True
        semantics[model_index] = winner[1]
        instances[model_index] = winner[2]
    return valid, semantics, instances


def _region_targets(
    observations: ObservationBank,
    label_valid: np.ndarray,
    model_slots: np.ndarray,
    instance_count: int,
    minimum_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(
        (len(observations.region_keys), instance_count + 1), dtype=np.float32
    )
    result_valid = np.zeros(len(observations.region_keys), dtype=np.bool_)
    for region_index in range(len(observations.region_keys)):
        edge_start = int(observations.csr_indptr[region_index])
        edge_end = int(observations.csr_indptr[region_index + 1])
        indices = observations.csr_model_indices[edge_start:edge_end]
        weights = observations.csr_weights[edge_start:edge_end].astype(np.float64)
        total_weight = float(weights.sum())
        if total_weight <= 0.0 or observations.region_reliability[region_index] <= 0.0:
            continue
        trusted = label_valid[indices]
        trusted_weight = float(weights[trusted].sum())
        if trusted_weight <= 0.0:
            continue
        for model_index, weight in zip(indices[trusted], weights[trusted]):
            slot = int(model_slots[model_index])
            result[region_index, slot if slot >= 0 else instance_count] += float(weight)
        result[region_index] /= trusted_weight
        result_valid[region_index] = (
            trusted_weight / total_weight + 1e-12 >= minimum_valid_fraction
        )
    return result, result_valid


def build_observation_training_sample(
    *,
    model_input: ReSceneModelInput,
    observations: ObservationBank,
    model_points_reference_xyz: object,
    visit_labels: tuple[NativeVisitLabels, NativeVisitLabels],
    identity_rules: PairIdentityRules,
    raw_semantic_to_model_class: Mapping[int, int],
    config: LabelTransferConfig,
    expected_pair_id: str | None = None,
    target_source_manifest: Mapping[str, object] | None = None,
) -> ObservationTrainingSample:
    """Build criterion-only pair supervision without changing inference inputs."""

    if not isinstance(model_input, ReSceneModelInput):
        raise TypeError("model_input must be ReSceneModelInput")
    if not isinstance(observations, ObservationBank):
        raise TypeError("observations must be ObservationBank")
    if not isinstance(identity_rules, PairIdentityRules):
        raise TypeError("identity_rules must be PairIdentityRules")
    if not isinstance(config, LabelTransferConfig):
        raise TypeError("config must be LabelTransferConfig")
    if expected_pair_id is not None and observations.pair_id != expected_pair_id:
        raise ObservationTrainingDataError("observation pair identity mismatch")
    if observations.model_input_sha256 != model_input.content_sha256():
        raise ObservationTrainingDataError("observation model identity mismatch")
    if not np.array_equal(observations.model_visit_ids, model_input.model_visit_ids):
        raise ObservationTrainingDataError("observation model visit domain mismatch")
    points = np.asarray(model_points_reference_xyz, dtype=np.float64)
    model_count = len(model_input.model_visit_ids)
    if points.shape != (model_count, 3) or np.any(~np.isfinite(points)):
        raise ObservationTrainingDataError("model reference points must have finite shape (M, 3)")
    if (
        not isinstance(visit_labels, tuple)
        or len(visit_labels) != 2
        or tuple(item.visit_id for item in visit_labels) != (0, 1)
    ):
        raise ObservationTrainingDataError("visit labels must contain ordered visits 0 and 1")
    class_mapping: dict[int, int] = {}
    if not isinstance(raw_semantic_to_model_class, Mapping):
        raise ObservationTrainingDataError("class mapping must be a mapping")
    for raw_id, class_id in raw_semantic_to_model_class.items():
        if (
            isinstance(raw_id, bool)
            or isinstance(class_id, bool)
            or not isinstance(raw_id, Integral)
            or not isinstance(class_id, Integral)
            or int(raw_id) < 0
            or int(class_id) < 0
        ):
            raise ObservationTrainingDataError("class mapping IDs must be nonnegative integers")
        class_mapping[int(raw_id)] = int(class_id)
    label_valid = np.zeros(model_count, dtype=np.bool_)
    model_semantics = np.full(model_count, -1, dtype=np.int64)
    model_instances = np.full(model_count, -1, dtype=np.int64)
    for labels in visit_labels:
        selected = model_input.model_visit_ids == labels.visit_id
        valid, semantics, instances = _transfer_visit_labels(
            points[selected], labels, config
        )
        label_valid[selected] = valid
        model_semantics[selected] = semantics
        model_instances[selected] = instances

    model_keys: list[str | None] = [None] * model_count
    semantic_by_key: dict[str, set[int]] = defaultdict(set)
    strong_identity_ignored: dict[str, int] = {}
    for model_index in np.flatnonzero(label_valid & (model_instances > 0)):
        visit_id = int(model_input.model_visit_ids[model_index])
        instance_id = int(model_instances[model_index])
        key = identity_rules.temporal_key(visit_id, instance_id)
        model_keys[int(model_index)] = key
        semantic_by_key[key].add(int(model_semantics[model_index]))
        if instance_id in identity_rules.ambiguous_instance_ids_by_visit[visit_id]:
            strong_identity_ignored[key] = instance_id
    keys = tuple(sorted(semantic_by_key))
    slot_by_key = {key: index for index, key in enumerate(keys)}
    model_slots = np.full(model_count, -1, dtype=np.int64)
    masks = np.zeros((len(keys), model_count), dtype=np.float32)
    for model_index, key in enumerate(model_keys):
        if key is None:
            continue
        slot = slot_by_key[key]
        model_slots[model_index] = slot
        masks[slot, model_index] = 1.0
    class_targets = np.full(len(keys), _IGNORE_CLASS_TARGET, dtype=np.int64)
    class_valid = np.zeros(len(keys), dtype=np.bool_)
    for key, semantics in semantic_by_key.items():
        mapped = {class_mapping[value] for value in semantics if value in class_mapping}
        if len(mapped) == 1 and len(mapped) == len(semantics):
            slot = slot_by_key[key]
            class_targets[slot] = mapped.pop()
            class_valid[slot] = True
    region_mass, region_valid = _region_targets(
        observations,
        label_valid,
        model_slots,
        len(keys),
        config.minimum_region_valid_fraction,
    )
    ambiguity_metadata = {
        "strong_identity_ignored": dict(sorted(strong_identity_ignored.items())),
        "ambiguous_instance_ids_by_visit": [
            sorted(identity_rules.ambiguous_instance_ids_by_visit[0]),
            sorted(identity_rules.ambiguous_instance_ids_by_visit[1]),
        ],
        "removed_reference_ids": sorted(identity_rules.removed_reference_ids),
    }
    provided_manifest = {} if target_source_manifest is None else dict(target_source_manifest)
    _, normalized_manifest = _canonical_json(provided_manifest, "target_source_manifest")
    target_provenance = {
        "ground_truth_role": "criterion_only",
        "model_forward_fields": ["model_input", "observations"],
        "native_label_sha256": [item.source_sha256 for item in visit_labels],
        "identity_source_sha256": identity_rules.source_sha256,
        "target_source_manifest": normalized_manifest,
        "label_transfer": {
            "maximum_distance_m": config.maximum_distance_m,
            "minimum_neighbors": config.minimum_neighbors,
            "minimum_consensus_fraction": config.minimum_consensus_fraction,
            "maximum_neighbors": config.maximum_neighbors,
            "trusted_background_raw_ids": list(config.trusted_background_raw_ids),
            "ignored_raw_semantic_ids": list(config.ignored_raw_semantic_ids),
            "minimum_region_valid_fraction": config.minimum_region_valid_fraction,
        },
    }
    return ObservationTrainingSample(
        model_input=model_input,
        observations=observations,
        instance_masks=torch.from_numpy(masks),
        label_valid=torch.from_numpy(label_valid),
        class_targets=torch.from_numpy(class_targets),
        class_valid=torch.from_numpy(class_valid),
        temporal_identity_keys=keys,
        ambiguity_metadata=ambiguity_metadata,
        region_instance_mass=torch.from_numpy(region_mass),
        region_label_valid=torch.from_numpy(region_valid),
        target_provenance=target_provenance,
    )


def load_rescene_class_mapping(
    label_database: str | Path,
    *,
    expected_validation_label_count: int,
    label_offset: int,
) -> dict[int, int]:
    """Reproduce ReScene's validation-label order and target offset."""

    if type(expected_validation_label_count) is not int or expected_validation_label_count <= 0:
        raise ObservationTrainingDataError("expected label count must be positive")
    if type(label_offset) is not int or not 0 <= label_offset < expected_validation_label_count:
        raise ObservationTrainingDataError("label_offset is invalid")
    path = Path(label_database).absolute()
    if path.is_symlink() or not path.is_file():
        raise ObservationTrainingDataError("label database is unavailable")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ObservationTrainingDataError("label database cannot be decoded") from error
    if not isinstance(payload, Mapping):
        raise ObservationTrainingDataError("label database must be a mapping")
    validation: list[int] = []
    for raw_id, metadata in payload.items():
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, Integral)
            or not isinstance(metadata, Mapping)
            or type(metadata.get("validation")) is not bool
        ):
            raise ObservationTrainingDataError("label database row is invalid")
        if metadata["validation"]:
            validation.append(int(raw_id))
    if len(validation) != expected_validation_label_count:
        raise ObservationTrainingDataError("validation label count mismatch")
    return {
        raw_id: remapped_id - label_offset
        for remapped_id, raw_id in enumerate(validation)
        if remapped_id >= label_offset
    }


def load_split_pair(
    split_manifest: str | Path,
    *,
    pair_id: str,
    required_role: str,
) -> dict[str, object]:
    """Load one pair only after enforcing environment- and scan-level separation."""

    path = Path(split_manifest).absolute()
    if path.is_symlink() or not path.is_file():
        raise ObservationTrainingDataError("split manifest is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingDataError("split manifest cannot be decoded") from error
    environments = payload.get("environments") if isinstance(payload, Mapping) else None
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_id") != "OVI_RESCENE_OBSERVATION_QUERY_SPLITS_V1"
        or not isinstance(environments, list)
    ):
        raise ObservationTrainingDataError("split manifest identity is invalid")
    seen_environments: dict[str, str] = {}
    seen_pairs: dict[str, str] = {}
    seen_scans: dict[str, str] = {}
    normalized: list[dict[str, object]] = []
    for environment in environments:
        if not isinstance(environment, Mapping):
            raise ObservationTrainingDataError("split environment is invalid")
        role = environment.get("role")
        environment_id = environment.get("environment_uuid")
        candidate_pair = environment.get("pair_id")
        sessions = environment.get("sessions")
        if (
            role not in {"TRAIN", "DEV", "CONFIRM"}
            or not isinstance(environment_id, str)
            or not environment_id
            or not isinstance(candidate_pair, str)
            or not candidate_pair
            or not isinstance(sessions, list)
            or len(sessions) != 2
        ):
            raise ObservationTrainingDataError("split environment schema is invalid")
        if environment_id in seen_environments or candidate_pair in seen_pairs:
            raise ObservationTrainingDataError("split repeats an environment or pair")
        seen_environments[environment_id] = role
        seen_pairs[candidate_pair] = role
        for session in sessions:
            scan_id = session.get("scan_uuid") if isinstance(session, Mapping) else None
            if not isinstance(scan_id, str) or not scan_id or scan_id in seen_scans:
                raise ObservationTrainingDataError("split repeats or omits a scan identity")
            seen_scans[scan_id] = role
        normalized.append(dict(environment))
    matches = [
        item
        for item in normalized
        if item.get("pair_id") == pair_id and item.get("role") == required_role
    ]
    if len(matches) != 1:
        raise ObservationTrainingDataError("requested split pair or role is not unique")
    return matches[0]


@dataclass(frozen=True, slots=True)
class ObservationTrainingArtifactPaths:
    root: Path
    manifest: Path
    arrays: Path


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def save_observation_training_sample(
    sample: ObservationTrainingSample, output_root: str | Path
) -> ObservationTrainingArtifactPaths:
    """Atomically save criterion targets while binding model and observation inputs."""

    if not isinstance(sample, ObservationTrainingSample):
        raise TypeError("sample must be ObservationTrainingSample")
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ObservationTrainingDataError(f"training sample already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        arrays_path = staging / "targets.npz"
        arrays = {
            name: getattr(sample, name).detach().cpu().contiguous().numpy()
            for name in _TARGET_ARRAY_NAMES
        }
        with arrays_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": 1,
            "artifact_id": "OVI_RESCENE_OBSERVATION_TRAINING_SAMPLE_V1",
            "status": "PASS",
            "pair_id": sample.observations.pair_id,
            "content_sha256": sample.content_sha256(),
            "model_input_sha256": sample.model_input.content_sha256(),
            "observation_sha256": sample.observations.content_sha256(),
            "temporal_identity_keys": list(sample.temporal_identity_keys),
            "ambiguity_metadata": sample.ambiguity_metadata,
            "target_provenance": sample.target_provenance,
            "array_records": {
                name: _array_record(value) for name, value in arrays.items()
            },
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
    return ObservationTrainingArtifactPaths(
        output, output / "manifest.json", output / "targets.npz"
    )


def load_observation_training_sample(
    output_root: str | Path,
    *,
    model_input: ReSceneModelInput,
    observations: ObservationBank,
) -> ObservationTrainingSample:
    """Reload a cache only when its exact inference inputs still match."""

    root = Path(output_root).absolute()
    manifest_path = root / "manifest.json"
    arrays_path = root / "targets.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationTrainingDataError("training sample manifest is unavailable") from error
    expected_keys = {
        "schema_version",
        "artifact_id",
        "status",
        "pair_id",
        "content_sha256",
        "model_input_sha256",
        "observation_sha256",
        "temporal_identity_keys",
        "ambiguity_metadata",
        "target_provenance",
        "array_records",
        "arrays",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id")
        != "OVI_RESCENE_OBSERVATION_TRAINING_SAMPLE_V1"
        or manifest.get("status") != "PASS"
    ):
        raise ObservationTrainingDataError("training sample manifest identity is invalid")
    if manifest.get("pair_id") != observations.pair_id:
        raise ObservationTrainingDataError("observation pair differs from training cache")
    if manifest.get("model_input_sha256") != model_input.content_sha256():
        raise ObservationTrainingDataError("model input differs from training cache")
    if manifest.get("observation_sha256") != observations.content_sha256():
        raise ObservationTrainingDataError("observation content differs from training cache")
    if manifest.get("arrays") != _file_record(arrays_path):
        raise ObservationTrainingDataError("training target file binding mismatch")
    try:
        with np.load(io.BytesIO(arrays_path.read_bytes()), allow_pickle=False) as source:
            if set(source.files) != set(_TARGET_ARRAY_NAMES):
                raise ObservationTrainingDataError("training target array schema is invalid")
            arrays = {name: np.array(source[name], copy=True) for name in _TARGET_ARRAY_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ObservationTrainingDataError):
            raise
        raise ObservationTrainingDataError("training target arrays are invalid") from error
    if manifest.get("array_records") != {
        name: _array_record(value) for name, value in arrays.items()
    }:
        raise ObservationTrainingDataError("training target array identity mismatch")
    sample = ObservationTrainingSample(
        model_input=model_input,
        observations=observations,
        instance_masks=torch.from_numpy(arrays["instance_masks"]),
        label_valid=torch.from_numpy(arrays["label_valid"]),
        class_targets=torch.from_numpy(arrays["class_targets"]),
        class_valid=torch.from_numpy(arrays["class_valid"]),
        temporal_identity_keys=tuple(manifest["temporal_identity_keys"]),
        ambiguity_metadata=dict(manifest["ambiguity_metadata"]),
        region_instance_mass=torch.from_numpy(arrays["region_instance_mass"]),
        region_label_valid=torch.from_numpy(arrays["region_label_valid"]),
        target_provenance=dict(manifest["target_provenance"]),
    )
    if sample.content_sha256() != manifest.get("content_sha256"):
        raise ObservationTrainingDataError("training sample content identity mismatch")
    return sample


__all__ = [
    "LabelTransferConfig",
    "NativeVisitLabels",
    "ObservationTrainingArtifactPaths",
    "ObservationTrainingDataError",
    "ObservationTrainingSample",
    "OfficialPairTrainingMetadata",
    "PairIdentityRules",
    "build_observation_training_sample",
    "load_native_processed_labels",
    "load_observation_training_sample",
    "load_official_pair_training_metadata",
    "load_rescene_class_mapping",
    "load_split_pair",
    "save_observation_training_sample",
]
