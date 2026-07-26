from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.oviv2.compact_checkpoint import (
    _close_best_effort,
    _create_temporary_directory_at,
    _fingerprint,
    _identity,
    _open_directory_without_symlinks,
    _reject_symlink_components,
    _rename_directory_no_replace_at,
    _write_regular_at,
)
from src.oviv2.reference_readout import ReferenceReadoutState
from src.oviv2.temporal_background import TemporalBackgroundVolume
from src.oviv2.temporal_geometry import ObjectSubmap
from src.oviv2.temporal_lifecycle import TemporalLifecycle, TemporalLifecycleState
from src.oviv2.temporal_state import TemporalEntityState, TemporalRuntimeState


TEMPORAL_COMPACT_FORMAT = "oviv2_temporal_compact_checkpoint"
_COMPACT_SCHEMA_VERSION = 2
_COMPACT_INVENTORY = frozenset({"manifest.json", "arrays.npz", "checksums.json"})
_COMPACT_ARRAY_NAMES_V1 = (
    "entity_ids",
    "lifecycle_codes",
    "existence_log_odds",
    "absent_streaks",
    "distinct_view_bin_counts",
    "object_to_world",
    "voxel_keys",
    "voxel_offsets",
)
_COMPACT_ARRAY_NAMES = _COMPACT_ARRAY_NAMES_V1 + (
    "geometry_epochs",
    "readout_valid",
)
_LIFECYCLE_TO_CODE = {
    TemporalLifecycle.ACTIVE: 0,
    TemporalLifecycle.UNCERTAIN: 1,
    TemporalLifecycle.DORMANT: 2,
}
_MAX_JSON_BYTES = 64 * 1024
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _validate_reference_lifecycle(
    value: TemporalLifecycleState,
    metadata: "TemporalSnapshotMetadata",
) -> None:
    if not isinstance(value, TemporalLifecycleState):
        raise TypeError("reference lifecycle state has an invalid type")
    expected_fields = {
        "entity_id",
        "lifecycle",
        "existence_log_odds",
        "last_frame_id",
        "last_timestamp",
        "absent_streak",
        "absence_view_bins",
    }
    if set(vars(value)) != expected_fields:
        raise ValueError("reference lifecycle state fields are invalid")
    if isinstance(value.entity_id, (bool, np.bool_)) or not isinstance(
        value.entity_id, Integral
    ) or int(value.entity_id) < 1:
        raise ValueError("reference lifecycle entity_id must be a positive integer")
    if not isinstance(value.lifecycle, TemporalLifecycle):
        raise TypeError("reference lifecycle must be a TemporalLifecycle")
    if isinstance(value.existence_log_odds, (bool, np.bool_)) or not isinstance(
        value.existence_log_odds, Real
    ) or not math.isfinite(float(value.existence_log_odds)):
        raise ValueError("reference lifecycle existence_log_odds must be finite")
    if isinstance(value.last_frame_id, (bool, np.bool_)) or not isinstance(
        value.last_frame_id, Integral
    ) or not 0 <= int(value.last_frame_id) <= metadata.frame_id:
        raise ValueError("reference lifecycle last_frame_id is later than checkpoint")
    if isinstance(value.last_timestamp, (bool, np.bool_)) or not isinstance(
        value.last_timestamp, Real
    ) or not math.isfinite(float(value.last_timestamp)):
        raise ValueError("reference lifecycle last_timestamp must be finite")
    if float(value.last_timestamp) > metadata.timestamp:
        raise ValueError("reference lifecycle last_timestamp is later than checkpoint")
    if isinstance(value.absent_streak, (bool, np.bool_)) or not isinstance(
        value.absent_streak, Integral
    ) or int(value.absent_streak) < 0:
        raise ValueError("reference lifecycle absent_streak must be nonnegative")
    bins = value.absence_view_bins
    if type(bins) is not tuple or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, Integral)
        or int(item) < 0
        for item in bins
    ):
        raise ValueError("reference lifecycle absence_view_bins are invalid")
    if bins != tuple(sorted(set(bins))) or len(bins) > int(value.absent_streak):
        raise ValueError("reference lifecycle absence_view_bins are inconsistent")
    if int(value.absent_streak) > 0 and not bins:
        raise ValueError("reference lifecycle absence_view_bins are required")


class TemporalCheckpointPublicationUncertainError(RuntimeError):
    def __init__(self, target: str | Path, publication_error: Exception) -> None:
        self.target = Path(target)
        self.publication_error = publication_error
        self.published = True
        super().__init__(
            f"temporal checkpoint was published at {self.target}, but durability "
            "or source binding could not be verified"
        )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(content: bytes, *, label: str) -> dict[str, Any]:
    if len(content) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds size limit")
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _readonly(value: object, *, dtype: np.dtype, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (dtype.kind in "iu" and raw.dtype.kind not in "iu"):
        raise TypeError(f"{name} has an invalid dtype")
    wide = np.asarray(raw, dtype=np.longdouble)
    if not np.isfinite(wide).all():
        raise ValueError(f"{name} must be finite")
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        if np.any(wide < limits.min) or np.any(wide > limits.max):
            raise ValueError(f"{name} values are outside the {dtype.name} range")
    elif np.any(np.abs(wide) > np.longdouble(np.finfo(dtype).max)):
        raise ValueError(f"{name} values are outside the {dtype.name} range")
    try:
        array = np.array(raw, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} has an invalid dtype") from exc
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} has an invalid shape")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class TemporalSnapshotMetadata:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    voxel_size_m: float
    config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        object.__setattr__(self, "scene_id", self.scene_id.strip())
        for name in ("frame_id", "revision"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, positive in (("timestamp", False), ("voxel_size_m", True)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric")
            if positive and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(value))
        if (
            not isinstance(self.config_sha256, str)
            or len(self.config_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.config_sha256)
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TemporalSnapshotEntity:
    entity: TemporalEntityState
    geometry_epoch: int
    readout_valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.entity, TemporalEntityState):
            raise TypeError("entity must be a TemporalEntityState")
        if type(self.geometry_epoch) is not int or self.geometry_epoch < 0:
            raise ValueError("geometry_epoch must be a non-negative integer")
        if type(self.readout_valid) is not bool:
            raise TypeError("readout_valid must be an exact bool")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.entity, name)


@dataclass(frozen=True)
class TemporalCurrentSnapshot:
    metadata: TemporalSnapshotMetadata
    entities: tuple[TemporalSnapshotEntity | TemporalEntityState, ...]
    background: TemporalBackgroundVolume

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, TemporalSnapshotMetadata):
            raise TypeError("metadata must be TemporalSnapshotMetadata")
        owned_metadata = TemporalSnapshotMetadata(
            scene_id=self.metadata.scene_id,
            frame_id=self.metadata.frame_id,
            timestamp=self.metadata.timestamp,
            revision=self.metadata.revision,
            voxel_size_m=self.metadata.voxel_size_m,
            config_sha256=self.metadata.config_sha256,
        )
        if type(self.entities) is not tuple or any(
            type(item) not in {TemporalSnapshotEntity, TemporalEntityState}
            for item in self.entities
        ):
            raise TypeError("entities must contain temporal snapshot entities")
        validated_entities: list[TemporalSnapshotEntity] = []
        for item in self.entities:
            if type(item) is TemporalSnapshotEntity:
                unexpected = set(vars(item)) - {
                    "entity", "geometry_epoch", "readout_valid", "lifecycle"
                }
                if unexpected:
                    raise ValueError(
                        f"snapshot entity has unexpected field: {sorted(unexpected)[0]}"
                    )
            entity = item.entity if type(item) is TemporalSnapshotEntity else item
            geometry_epoch = item.geometry_epoch if type(item) is TemporalSnapshotEntity else 0
            readout_valid = item.readout_valid if type(item) is TemporalSnapshotEntity else True
            lifecycle = vars(item).get("lifecycle", entity.lifecycle)
            if type(lifecycle.entity_id) is not int or not 0 <= lifecycle.entity_id <= np.iinfo(np.int64).max:
                raise ValueError("lifecycle entity_id must be a non-negative int64 integer")
            if not isinstance(lifecycle.lifecycle, TemporalLifecycle):
                raise TypeError("lifecycle lifecycle must be a TemporalLifecycle enum")
            if isinstance(lifecycle.existence_log_odds, bool) or not isinstance(lifecycle.existence_log_odds, Real) or not math.isfinite(float(lifecycle.existence_log_odds)):
                raise ValueError("lifecycle existence_log_odds must be finite numeric")
            if type(lifecycle.last_frame_id) is not int or lifecycle.last_frame_id < 0:
                raise ValueError("lifecycle last_frame_id must be a non-negative integer")
            if isinstance(lifecycle.last_timestamp, bool) or not isinstance(lifecycle.last_timestamp, Real) or not math.isfinite(float(lifecycle.last_timestamp)):
                raise ValueError("lifecycle last_timestamp must be finite numeric")
            if type(lifecycle.absent_streak) is not int or lifecycle.absent_streak < 0:
                raise ValueError("lifecycle absent_streak must be a non-negative integer")
            if type(lifecycle.absence_view_bins) is not tuple:
                raise TypeError("lifecycle absence_view_bins must be an exact tuple")
            bins = lifecycle.absence_view_bins
            if any(type(value) is not int or value < 0 for value in bins):
                raise ValueError("lifecycle absence_view_bins must contain non-negative integers")
            if bins != tuple(sorted(set(bins))) or len(bins) > lifecycle.absent_streak:
                raise ValueError("lifecycle absence_view_bins must be sorted, unique, and bounded by absent_streak")
            if lifecycle.absent_streak > 0 and not bins:
                raise ValueError("lifecycle absence_view_bins are required for an absence streak")
            owned_lifecycle = TemporalLifecycleState(
                entity_id=lifecycle.entity_id,
                lifecycle=lifecycle.lifecycle,
                existence_log_odds=lifecycle.existence_log_odds,
                last_frame_id=lifecycle.last_frame_id,
                last_timestamp=lifecycle.last_timestamp,
                absent_streak=lifecycle.absent_streak,
                absence_view_bins=tuple(lifecycle.absence_view_bins),
            )
            submap = entity.submap
            if not isinstance(submap, ObjectSubmap):
                raise TypeError("entity submap must be an ObjectSubmap")
            owned_submap = ObjectSubmap(
                reference_centroid_xyz=submap.reference_centroid_xyz,
                local_voxel_keys=submap.local_voxel_keys,
                local_points_xyz=submap.local_points_xyz,
                weights=submap.weights,
                last_seen_frame_ids=submap.last_seen_frame_ids,
            )
            validated = TemporalEntityState(
                lifecycle=owned_lifecycle,
                semantic_probabilities=entity.semantic_probabilities,
                image_prototype=entity.image_prototype,
                extent_xyz=entity.extent_xyz,
                object_to_world=entity.object_to_world,
                submap=owned_submap,
                first_seen_frame_id=entity.first_seen_frame_id,
                last_seen_frame_id=entity.last_seen_frame_id,
                feature_model_id=entity.feature_model_id,
            )
            validated_entities.append(
                TemporalSnapshotEntity(validated, geometry_epoch, readout_valid)
            )
        ids = tuple(item.lifecycle.entity_id for item in validated_entities)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("entities must be sorted by unique temporal entity ID")
        for entity in validated_entities:
            if (
                entity.lifecycle.last_frame_id > owned_metadata.frame_id
                or entity.lifecycle.last_timestamp > owned_metadata.timestamp
                or entity.last_seen_frame_id > owned_metadata.frame_id
                or entity.first_seen_frame_id > owned_metadata.frame_id
                or (entity.submap.last_seen_frame_ids.size and int(entity.submap.last_seen_frame_ids.max()) > owned_metadata.frame_id)
            ):
                raise ValueError("entity record is later than checkpoint")
        if not isinstance(self.background, TemporalBackgroundVolume):
            raise TypeError("background must be TemporalBackgroundVolume")
        if not math.isclose(self.background.config.voxel_size_m, owned_metadata.voxel_size_m, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("background voxel size does not match metadata")
        object.__setattr__(self, "metadata", owned_metadata)
        object.__setattr__(self, "entities", tuple(validated_entities))
        object.__setattr__(self, "background", self.background._clone(max(1, self.background.active_block_count)))


@dataclass(frozen=True)
class _DirectoryWitness:
    path: Path
    directory_fingerprint: tuple[int, int, int, int, int]
    file_fingerprints: tuple[tuple[str, tuple[int, int, int, int, int]], ...]
    content_sha256: tuple[tuple[str, str], ...]

    @classmethod
    def capture(cls, path: Path, inventory: frozenset[str]) -> "_DirectoryWitness":
        _reject_symlink_components(path, label="checkpoint source")
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            directory = os.fstat(descriptor)
            if set(os.listdir(descriptor)) != inventory:
                raise ValueError("checkpoint inventory changed")
            members = []
            hashes = []
            for name in sorted(inventory):
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(status.st_mode):
                    raise ValueError("checkpoint member is not regular")
                members.append((name, _fingerprint(status)))
                member_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
                try:
                    digest = hashlib.sha256()
                    while True:
                        chunk = os.read(member_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                    if _fingerprint(os.fstat(member_fd)) != _fingerprint(status):
                        raise ValueError("checkpoint member changed while hashing")
                    hashes.append((name, digest.hexdigest()))
                finally:
                    _close_best_effort(member_fd)
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(directory):
                raise ValueError("checkpoint source changed while binding")
            witness = cls(path, _fingerprint(directory), tuple(members), tuple(hashes))
        finally:
            _close_best_effort(descriptor)
        witness.revalidate()
        return witness

    def revalidate(self) -> None:
        _reject_symlink_components(self.path, label="checkpoint source")
        descriptor = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if _fingerprint(os.fstat(descriptor)) != self.directory_fingerprint:
                raise ValueError("checkpoint source identity changed")
            if set(os.listdir(descriptor)) != {name for name, _ in self.file_fingerprints}:
                raise ValueError("checkpoint inventory changed")
            hashes = dict(self.content_sha256)
            for name, expected in self.file_fingerprints:
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(current.st_mode) or _fingerprint(current) != expected:
                    raise ValueError("checkpoint file identity changed")
                member_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
                try:
                    digest = hashlib.sha256()
                    while True:
                        chunk = os.read(member_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                    if digest.hexdigest() != hashes[name]:
                        raise ValueError("checkpoint file content hash changed")
                    if _fingerprint(os.fstat(member_fd)) != expected:
                        raise ValueError("checkpoint file identity changed while hashing")
                finally:
                    _close_best_effort(member_fd)
            if _fingerprint(os.fstat(descriptor)) != self.directory_fingerprint:
                raise ValueError("checkpoint source identity changed while hashing")
        finally:
            _close_best_effort(descriptor)


def _cleanup_stage(parent_fd: int, name: str, expected_identity: tuple[int, int] | None) -> None:
    if expected_identity is None:
        return
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(status.st_mode) or _identity(status) != expected_identity:
            return
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            for member in os.listdir(descriptor):
                os.unlink(member, dir_fd=descriptor)
        finally:
            _close_best_effort(descriptor)
        os.rmdir(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _publish_new_directory(target_dir: str | Path, files: Mapping[str, bytes], inventory: frozenset[str]) -> _DirectoryWitness:
    if set(files) != inventory:
        raise ValueError("publication inventory is invalid")
    raw = Path(target_dir)
    target = Path(os.path.abspath(os.fspath(raw)))
    if not target.name or target.name in {".", ".."}:
        raise ValueError("checkpoint target name is invalid")
    _reject_symlink_components(target.parent, label="checkpoint parent")
    try:
        parent_fd = _open_directory_without_symlinks(target.parent)
    except OSError as exc:
        raise ValueError("checkpoint parent must be a real symlink-free directory") from exc
    temporary_name: str | None = None
    temporary_fd: int | None = None
    temporary_identity: tuple[int, int] | None = None
    published = False
    try:
        parent_identity = _identity(os.fstat(parent_fd))
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(target)
        temporary_name, temporary_fd, temporary_identity = _create_temporary_directory_at(parent_fd, target.name)
        for name in sorted(files):
            _write_regular_at(temporary_fd, name, files[name])
        os.fsync(temporary_fd)
        _rename_directory_no_replace_at(
            parent_fd,
            temporary_name,
            target.name,
            parent_path=target.parent,
            expected_parent_identity=parent_identity,
            expected_source_identity=temporary_identity,
        )
        published = True
        os.fsync(parent_fd)
        return _DirectoryWitness.capture(target, inventory)
    except TemporalCheckpointPublicationUncertainError:
        raise
    except Exception as exc:
        if published:
            raise TemporalCheckpointPublicationUncertainError(target, exc) from exc
        raise
    finally:
        if temporary_fd is not None:
            _close_best_effort(temporary_fd)
        if not published and temporary_name is not None:
            _cleanup_stage(parent_fd, temporary_name, temporary_identity)
        _close_best_effort(parent_fd)


def _canonical_npz(arrays: Mapping[str, np.ndarray], order: Sequence[str]) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=False) as archive:
        for name in order:
            payload = io.BytesIO()
            npy_format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return destination.getvalue()


def _compact_byte_limit(
    maximum_entities: int,
    maximum_object_voxels: int,
    *,
    schema_version: int = _COMPACT_SCHEMA_VERSION,
) -> int:
    if type(maximum_entities) is not int or maximum_entities <= 0:
        raise ValueError("maximum_entities capacity must be positive")
    if type(maximum_object_voxels) is not int or maximum_object_voxels <= 0:
        raise ValueError("maximum_object_voxels capacity must be positive")
    voxels = maximum_entities * maximum_object_voxels
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("compact schema_version must be integer 1 or 2")
    epoch_bytes = 0 if schema_version == 1 else 8 + 1
    raw = maximum_entities * (8 + 1 + 8 + 8 + 8 + 16 * 8 + 8 + epoch_bytes) + voxels * 3 * 8
    return min(_MAX_ARCHIVE_BYTES, raw * 2 + 128 * 1024)


def _read_npy_contract(archive: zipfile.ZipFile, info: zipfile.ZipInfo, name: str) -> tuple[tuple[int, ...], np.dtype]:
    try:
        with archive.open(info, "r") as member:
            version = npy_format.read_magic(member)
            if version == (1, 0):
                shape, fortran, dtype = npy_format.read_array_header_1_0(member, max_header_size=16 * 1024)
            elif version == (2, 0):
                shape, fortran, dtype = npy_format.read_array_header_2_0(member, max_header_size=16 * 1024)
            else:
                raise ValueError(f"compact {name} has unsupported NPY version")
            header_bytes = member.tell()
    except (EOFError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("compact"):
            raise
        raise ValueError(f"compact {name} has an invalid NPY header") from exc
    normalized = np.dtype(dtype)
    if normalized.hasobject:
        raise ValueError(f"compact {name} object dtype is forbidden")
    if fortran:
        raise ValueError(f"compact {name} Fortran order is forbidden")
    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"compact {name} shape is invalid")
        count *= dimension
    if header_bytes + count * normalized.itemsize != info.file_size:
        raise ValueError(f"compact {name} shape/data size is invalid")
    return tuple(shape), normalized


def _preflight_compact_archive(
    content: bytes,
    *,
    maximum_entities: int,
    maximum_object_voxels: int,
    byte_limit: int,
    array_names: tuple[str, ...],
) -> None:
    try:
        archive_context = zipfile.ZipFile(io.BytesIO(content), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("compact array archive is invalid") from exc
    with archive_context as archive:
        infos = archive.infolist()
        expected = {f"{name}.npy" for name in array_names}
        counts = Counter(info.filename for info in infos)
        if set(counts) != expected or any(count != 1 for count in counts.values()) or len(infos) != len(expected):
            raise ValueError("compact array inventory is invalid")
        total = 0
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if (
                info.is_dir()
                or "/" in info.filename
                or "\\" in info.filename
                or info.flag_bits & 0x1
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            ):
                raise ValueError("compact array member is unsafe")
            total += info.file_size
            if info.file_size > byte_limit or info.compress_size > byte_limit or total > byte_limit:
                raise ValueError("compact array member exceeds capacity-derived size limit")
            by_name[info.filename.removesuffix(".npy")] = info
        contracts = {
            name: _read_npy_contract(archive, by_name[name], name)
            for name in array_names
        }
        entity_shape, entity_dtype = contracts["entity_ids"]
        if entity_dtype != np.dtype(np.int64) or len(entity_shape) != 1:
            raise ValueError("compact entity_ids dtype or shape is invalid")
        entity_count = entity_shape[0]
        if entity_count > maximum_entities:
            raise ValueError("compact entity capacity exceeded")
        voxel_shape, voxel_dtype = contracts["voxel_keys"]
        if voxel_dtype != np.dtype(np.int64) or len(voxel_shape) != 2 or voxel_shape[1:] != (3,):
            raise ValueError("compact voxel_keys dtype or shape is invalid")
        voxel_count = voxel_shape[0]
        if voxel_count > maximum_entities * maximum_object_voxels:
            raise ValueError("compact voxel capacity exceeded")
        expected_contracts = {
            "lifecycle_codes": ((entity_count,), np.dtype(np.uint8)),
            "existence_log_odds": ((entity_count,), np.dtype(np.float64)),
            "absent_streaks": ((entity_count,), np.dtype(np.int64)),
            "distinct_view_bin_counts": ((entity_count,), np.dtype(np.int64)),
            "object_to_world": ((entity_count, 4, 4), np.dtype(np.float64)),
            "voxel_offsets": ((entity_count + 1,), np.dtype(np.int64)),
        }
        if "geometry_epochs" in contracts:
            expected_contracts["geometry_epochs"] = (
                (entity_count,), np.dtype(np.int64)
            )
            expected_contracts["readout_valid"] = (
                (entity_count,), np.dtype(np.uint8)
            )
        for name, expected_contract in expected_contracts.items():
            if contracts[name] != expected_contract:
                raise ValueError(f"compact {name} dtype or shape is invalid")


@dataclass(frozen=True)
class TemporalCompactCheckpoint:
    metadata: TemporalSnapshotMetadata
    entity_ids: np.ndarray
    lifecycle_codes: np.ndarray
    existence_log_odds: np.ndarray
    absent_streaks: np.ndarray
    distinct_view_bin_counts: np.ndarray
    object_to_world: np.ndarray
    voxel_keys: np.ndarray
    voxel_offsets: np.ndarray
    geometry_epochs: np.ndarray | None = None
    readout_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, TemporalSnapshotMetadata):
            raise TypeError("metadata must be TemporalSnapshotMetadata")
        count_hint = len(np.asarray(self.entity_ids))
        if self.geometry_epochs is None:
            object.__setattr__(self, "geometry_epochs", np.zeros(count_hint, dtype=np.int64))
        if self.readout_valid is None:
            object.__setattr__(self, "readout_valid", np.ones(count_hint, dtype=np.uint8))
        contracts = {
            "entity_ids": (np.dtype(np.int64), ()),
            "lifecycle_codes": (np.dtype(np.uint8), ()),
            "existence_log_odds": (np.dtype(np.float64), ()),
            "absent_streaks": (np.dtype(np.int64), ()),
            "distinct_view_bin_counts": (np.dtype(np.int64), ()),
            "object_to_world": (np.dtype(np.float64), (4, 4)),
            "voxel_keys": (np.dtype(np.int64), (3,)),
            "voxel_offsets": (np.dtype(np.int64), ()),
            "geometry_epochs": (np.dtype(np.int64), ()),
            "readout_valid": (np.dtype(np.uint8), ()),
        }
        for name, (dtype, tail) in contracts.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype, shape_tail=tail, name=name))
        count = len(self.entity_ids)
        if any(len(getattr(self, name)) != count for name in ("lifecycle_codes", "existence_log_odds", "absent_streaks", "distinct_view_bin_counts", "object_to_world", "geometry_epochs", "readout_valid")):
            raise ValueError("compact entity arrays must have matching lengths")
        if self.entity_ids.tolist() != sorted(set(self.entity_ids.tolist())) or np.any(self.entity_ids < 0):
            raise ValueError("entity_ids must be sorted, unique, and nonnegative")
        if np.any(self.lifecycle_codes > 2):
            raise ValueError("lifecycle_codes are invalid")
        if np.any(self.geometry_epochs < 0):
            raise ValueError("geometry_epochs must be nonnegative")
        if np.any(self.readout_valid > 1):
            raise ValueError("readout_valid values must be zero or one")
        if np.any(self.absent_streaks < 0) or np.any(self.distinct_view_bin_counts < 0):
            raise ValueError("evidence counters must be nonnegative")
        if len(self.voxel_offsets) != count + 1 or self.voxel_offsets[0] != 0 or self.voxel_offsets[-1] != len(self.voxel_keys) or np.any(np.diff(self.voxel_offsets) < 0):
            raise ValueError("voxel_offsets are invalid")
        for index in range(count):
            start, stop = int(self.voxel_offsets[index]), int(self.voxel_offsets[index + 1])
            keys = [tuple(row) for row in self.voxel_keys[start:stop].tolist()]
            if keys != sorted(set(keys)):
                raise ValueError("voxel_keys must be sorted and unique per entity")
        if not np.allclose(self.object_to_world[:, 3, :], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
            raise ValueError("object_to_world must contain homogeneous poses")
        rotations = self.object_to_world[:, :3, :3]
        products = np.transpose(rotations, (0, 2, 1)) @ rotations
        if not np.allclose(products, np.eye(3), rtol=0.0, atol=1e-6) or not np.allclose(
            np.linalg.det(rotations), 1.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError("object_to_world must contain rigid poses")

    @property
    def path(self) -> Path:
        value = getattr(self, "_path", None)
        if value is None:
            raise ValueError("checkpoint is not bound to a published source")
        return value

    def revalidate_source(self) -> None:
        witness = getattr(self, "_source_witness", None)
        if witness is None:
            raise ValueError("checkpoint has no source witness")
        witness.revalidate()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TemporalCurrentSnapshot,
        *,
        maximum_entities: int | None = None,
        maximum_object_voxels: int | None = None,
    ) -> "TemporalCompactCheckpoint":
        if not isinstance(snapshot, TemporalCurrentSnapshot):
            raise TypeError("snapshot must be TemporalCurrentSnapshot")
        config = snapshot.background.config
        maximum_entities = config.maximum_entities if maximum_entities is None else maximum_entities
        maximum_object_voxels = config.maximum_object_voxels if maximum_object_voxels is None else maximum_object_voxels
        _compact_byte_limit(maximum_entities, maximum_object_voxels)
        if len(snapshot.entities) > maximum_entities:
            raise ValueError("entity capacity exceeded")
        if any(len(entity.submap.local_voxel_keys) > maximum_object_voxels for entity in snapshot.entities):
            raise ValueError("object voxel capacity exceeded")
        offsets = [0]
        chunks = []
        for entity in snapshot.entities:
            chunks.extend(entity.submap.local_voxel_keys)
            offsets.append(len(chunks))
        return cls(
            snapshot.metadata,
            np.asarray([item.lifecycle.entity_id for item in snapshot.entities], dtype=np.int64),
            np.asarray([_LIFECYCLE_TO_CODE[item.lifecycle.lifecycle] for item in snapshot.entities], dtype=np.uint8),
            np.asarray([item.lifecycle.existence_log_odds for item in snapshot.entities], dtype=np.float64),
            np.asarray([item.lifecycle.absent_streak for item in snapshot.entities], dtype=np.int64),
            np.asarray([len(item.lifecycle.absence_view_bins) for item in snapshot.entities], dtype=np.int64),
            np.asarray([item.object_to_world for item in snapshot.entities], dtype=np.float64).reshape((-1, 4, 4)),
            np.asarray(chunks, dtype=np.int64).reshape((-1, 3)),
            np.asarray(offsets, dtype=np.int64),
            np.asarray([item.geometry_epoch for item in snapshot.entities], dtype=np.int64),
            np.asarray([item.readout_valid for item in snapshot.entities], dtype=np.uint8),
        )

    @classmethod
    def from_reference(
        cls,
        metadata: TemporalSnapshotMetadata,
        state: ReferenceReadoutState,
        *,
        maximum_entities: int,
        maximum_object_voxels: int,
    ) -> "TemporalCompactCheckpoint":
        if not isinstance(metadata, TemporalSnapshotMetadata):
            raise TypeError("metadata must be TemporalSnapshotMetadata")
        if not isinstance(state, ReferenceReadoutState):
            raise TypeError("state must be ReferenceReadoutState")
        view = state.cumulative_view
        if view is None:
            raise ValueError("reference state has no cumulative view")
        if (
            state.scene_id != view.scene_id
            or state.revision != view.revision
            or state.last_frame_id != view.last_frame_id
            or state.last_timestamp != view.last_timestamp
            or metadata.scene_id != view.scene_id
            or metadata.revision != view.revision
            or metadata.frame_id != view.last_frame_id
            or metadata.timestamp != view.last_timestamp
            or not math.isclose(
                metadata.voxel_size_m, view.voxel_size_m, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError("reference state and metadata binding mismatch")

        _compact_byte_limit(maximum_entities, maximum_object_voxels)
        if len(view.entities) > maximum_entities:
            raise ValueError("entity capacity exceeded")
        if any(len(entity.voxel_keys) > maximum_object_voxels for entity in view.entities):
            raise ValueError("object voxel capacity exceeded")

        entities_by_id = {entity.entity_id: entity for entity in view.entities}
        raw_lifecycles = state.entity_lifecycles
        if type(raw_lifecycles) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in raw_lifecycles
        ):
            raise TypeError("reference entity_lifecycles must be an exact tuple of pairs")
        if any(
            type(entity_id) is not int
            or entity_id < 1
            or type(lifecycle) is not str
            for entity_id, lifecycle in raw_lifecycles
        ):
            raise ValueError("reference entity_lifecycles contain invalid values")
        lifecycle_ids = tuple(item[0] for item in raw_lifecycles)
        if lifecycle_ids != tuple(sorted(set(lifecycle_ids))):
            raise ValueError("reference lifecycle IDs must be sorted and unique")
        lifecycle_names = dict(raw_lifecycles)
        if set(lifecycle_names) != set(entities_by_id):
            raise ValueError("reference lifecycle IDs do not match cumulative entities")
        if type(state.lifecycle_states) is not tuple:
            raise TypeError("reference lifecycle_states must contain lifecycle states")
        for item in state.lifecycle_states:
            _validate_reference_lifecycle(item, metadata)
        evidence_ids = tuple(item.entity_id for item in state.lifecycle_states)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("reference evidence IDs must be sorted and unique")
        evidence = {item.entity_id: item for item in state.lifecycle_states}
        if evidence and set(evidence) != set(entities_by_id):
            raise ValueError("reference evidence IDs do not match cumulative entities")

        lifecycle_by_name = {
            item.value: item
            for item in (
                TemporalLifecycle.ACTIVE,
                TemporalLifecycle.UNCERTAIN,
                TemporalLifecycle.DORMANT,
            )
        }
        ordered = tuple(view.entities)
        lifecycles: list[TemporalLifecycle] = []
        log_odds: list[float] = []
        absent_streaks: list[int] = []
        distinct_bins: list[int] = []
        for entity in ordered:
            name = lifecycle_names[entity.entity_id]
            if name not in lifecycle_by_name:
                raise ValueError("reference lifecycle is invalid")
            item = evidence.get(entity.entity_id)
            if item is None and name != entity.lifecycle_state:
                raise ValueError("reference lifecycle does not match cumulative lifecycle")
            lifecycle = lifecycle_by_name[name] if item is None else item.lifecycle
            if item is not None and item.lifecycle.value != name:
                raise ValueError("reference lifecycle evidence is inconsistent")
            lifecycles.append(lifecycle)
            log_odds.append(0.0 if item is None else item.existence_log_odds)
            absent_streaks.append(0 if item is None else item.absent_streak)
            distinct_bins.append(0 if item is None else len(item.absence_view_bins))

        offsets = [0]
        keys: list[tuple[int, int, int]] = []
        for entity in ordered:
            keys.extend(sorted(entity.voxel_keys))
            offsets.append(len(keys))
        identity = np.eye(4, dtype=np.float64)
        return cls(
            metadata,
            np.asarray([item.entity_id for item in ordered], dtype=np.int64),
            np.asarray([_LIFECYCLE_TO_CODE[item] for item in lifecycles], dtype=np.uint8),
            np.asarray(log_odds, dtype=np.float64),
            np.asarray(absent_streaks, dtype=np.int64),
            np.asarray(distinct_bins, dtype=np.int64),
            np.repeat(identity[None, :, :], len(ordered), axis=0),
            np.asarray(keys, dtype=np.int64).reshape((-1, 3)),
            np.asarray(offsets, dtype=np.int64),
            np.zeros(len(ordered), dtype=np.int64),
            np.asarray(
                [item is not TemporalLifecycle.DORMANT for item in lifecycles],
                dtype=np.uint8,
            ),
        )

    def commit_new(
        self,
        target_dir: str | Path,
        *,
        maximum_entities: int,
        maximum_object_voxels: int,
    ) -> "TemporalCompactCheckpoint":
        byte_limit = _compact_byte_limit(maximum_entities, maximum_object_voxels)
        if len(self.entity_ids) > maximum_entities or any(np.diff(self.voxel_offsets) > maximum_object_voxels):
            raise ValueError("compact checkpoint capacity exceeded")
        if len(self.voxel_keys) > maximum_entities * maximum_object_voxels:
            raise ValueError("compact checkpoint total voxel capacity exceeded")
        arrays = {name: getattr(self, name) for name in _COMPACT_ARRAY_NAMES}
        raw_bytes = sum(array.nbytes for array in arrays.values())
        if raw_bytes > byte_limit:
            raise ValueError("compact checkpoint raw arrays exceed capacity-derived byte limit")
        archive = _canonical_npz(arrays, _COMPACT_ARRAY_NAMES)
        if len(archive) > byte_limit:
            raise ValueError("serialized compact checkpoint exceeds capacity-derived byte limit")
        manifest = _canonical_json({
            "format": TEMPORAL_COMPACT_FORMAT,
            "schema_version": _COMPACT_SCHEMA_VERSION,
            "metadata": asdict(self.metadata),
            "maximum_entities": maximum_entities,
            "maximum_object_voxels": maximum_object_voxels,
            "serialized_byte_limit": byte_limit,
        })
        checksums = {"arrays.npz": _sha256(archive), "manifest.json": _sha256(manifest)}
        files = {"arrays.npz": archive, "manifest.json": manifest, "checksums.json": _canonical_json(checksums)}
        witness = _publish_new_directory(target_dir, files, _COMPACT_INVENTORY)
        try:
            bound = TemporalCompactCheckpoint.load(target_dir, maximum_entities=maximum_entities, maximum_object_voxels=maximum_object_voxels)
        except TemporalCheckpointPublicationUncertainError:
            raise
        except Exception as exc:
            raise TemporalCheckpointPublicationUncertainError(target_dir, exc) from exc
        object.__setattr__(bound, "_source_witness", witness)
        object.__setattr__(bound, "_path", witness.path)
        return bound

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        maximum_entities: int | None = None,
        maximum_object_voxels: int | None = None,
    ) -> "TemporalCompactCheckpoint":
        source = Path(os.path.abspath(os.fspath(checkpoint_dir)))
        _reject_symlink_components(source, label="compact checkpoint source")
        witness = _DirectoryWitness.capture(source, _COMPACT_INVENTORY)
        descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if set(os.listdir(descriptor)) != _COMPACT_INVENTORY:
                raise ValueError("compact checkpoint inventory is invalid")
            contents: dict[str, bytes] = {}
            for name in sorted(_COMPACT_INVENTORY):
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(status.st_mode):
                    raise ValueError("compact checkpoint member is not regular")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    limit = _MAX_ARCHIVE_BYTES if name == "arrays.npz" else _MAX_JSON_BYTES
                    data = os.read(fd, limit + 1)
                    if len(data) > limit or os.read(fd, 1):
                        raise ValueError("compact checkpoint member exceeds size limit")
                    contents[name] = data
                finally:
                    _close_best_effort(fd)
        finally:
            _close_best_effort(descriptor)
        checksums = _strict_json(contents["checksums.json"], label="checksums")
        if set(checksums) != {"arrays.npz", "manifest.json"}:
            raise ValueError("compact checkpoint checksum inventory is invalid")
        for name, digest in checksums.items():
            if not isinstance(digest, str) or _sha256(contents[name]) != digest:
                raise ValueError(f"compact checkpoint checksum mismatch for {name}")
        manifest = _strict_json(contents["manifest.json"], label="manifest")
        if set(manifest) != {"format", "schema_version", "metadata", "maximum_entities", "maximum_object_voxels", "serialized_byte_limit"}:
            raise ValueError("compact checkpoint manifest fields are not exact")
        if (
            manifest["format"] != TEMPORAL_COMPACT_FORMAT
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] not in {1, 2}
        ):
            raise ValueError("compact checkpoint schema is invalid")
        stored_entities = manifest["maximum_entities"]
        stored_voxels = manifest["maximum_object_voxels"]
        if maximum_entities is not None and maximum_entities != stored_entities:
            raise ValueError("maximum_entities capacity mismatch")
        if maximum_object_voxels is not None and maximum_object_voxels != stored_voxels:
            raise ValueError("maximum_object_voxels capacity mismatch")
        schema_version = manifest["schema_version"]
        expected_limit = _compact_byte_limit(
            stored_entities, stored_voxels, schema_version=schema_version
        )
        if manifest["serialized_byte_limit"] != expected_limit or len(contents["arrays.npz"]) > expected_limit:
            raise ValueError("compact checkpoint serialized capacity is invalid")
        _preflight_compact_archive(
            contents["arrays.npz"],
            maximum_entities=stored_entities,
            maximum_object_voxels=stored_voxels,
            byte_limit=expected_limit,
            array_names=(
                _COMPACT_ARRAY_NAMES_V1
                if schema_version == 1
                else _COMPACT_ARRAY_NAMES
            ),
        )
        array_names = (
            _COMPACT_ARRAY_NAMES_V1
            if schema_version == 1
            else _COMPACT_ARRAY_NAMES
        )
        with np.load(io.BytesIO(contents["arrays.npz"]), allow_pickle=False) as payload:
            if set(payload.files) != set(array_names):
                raise ValueError("compact array schema is invalid")
            arrays = {name: np.array(payload[name], copy=True) for name in array_names}
        expected_dtypes = {
            "entity_ids": np.dtype(np.int64), "lifecycle_codes": np.dtype(np.uint8),
            "existence_log_odds": np.dtype(np.float64), "absent_streaks": np.dtype(np.int64),
            "distinct_view_bin_counts": np.dtype(np.int64), "object_to_world": np.dtype(np.float64),
            "voxel_keys": np.dtype(np.int64), "voxel_offsets": np.dtype(np.int64),
        }
        if schema_version == 2:
            expected_dtypes.update(
                geometry_epochs=np.dtype(np.int64),
                readout_valid=np.dtype(np.uint8),
            )
        if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
            raise ValueError("compact array dtype is invalid")
        metadata_payload = manifest["metadata"]
        if not isinstance(metadata_payload, dict) or set(metadata_payload) != {field.name for field in fields(TemporalSnapshotMetadata)}:
            raise ValueError("temporal metadata fields are not exact")
        result = cls(metadata=TemporalSnapshotMetadata(**metadata_payload), **arrays)
        if len(result.entity_ids) > stored_entities or any(np.diff(result.voxel_offsets) > stored_voxels):
            raise ValueError("compact checkpoint capacity exceeded")
        witness.revalidate()
        object.__setattr__(result, "_source_witness", witness)
        object.__setattr__(result, "_path", source)
        return result


def _background_points(background: TemporalBackgroundVolume) -> np.ndarray:
    mesh = background._volume.extract_mesh(weight_threshold=0.0)
    if "positions" not in mesh.vertex:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(mesh.vertex["positions"].numpy(), dtype=np.float32).reshape((-1, 3))
    if len(points):
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    return points


def build_temporal_map_snapshot(snapshot: TemporalCurrentSnapshot, class_names: Sequence[str]) -> MapSnapshot:
    if not isinstance(snapshot, TemporalCurrentSnapshot):
        raise TypeError("snapshot must be TemporalCurrentSnapshot")
    if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence) or not class_names:
        raise ValueError("class_names must be a non-empty sequence")
    names = tuple(str(item).strip() for item in class_names)
    if any(not item for item in names):
        raise ValueError("class_names must contain non-empty strings")
    predictions: list[EntityPrediction] = []
    for entity in snapshot.entities:
        if entity.lifecycle.lifecycle is TemporalLifecycle.DORMANT:
            continue
        probabilities = entity.semantic_probabilities
        class_id, score = max(probabilities, key=lambda item: (item[1], -item[0])) if probabilities else (0, 0.0)
        label = names[class_id] if 0 <= class_id < len(names) else None
        points = entity.submap.world_points(entity.object_to_world).astype(np.float32, copy=False)
        predictions.append(EntityPrediction(
            entity_id=f"temporal:{entity.lifecycle.entity_id}",
            points_xyz=points,
            semantic_embedding=entity.image_prototype,
            semantic_label=label,
            semantic_score=score,
            lifecycle_state=entity.lifecycle.lifecycle.value,
            first_seen=float(entity.first_seen_frame_id),
            last_seen=float(entity.last_seen_frame_id),
            metadata={
                "temporal_entity_id": entity.lifecycle.entity_id,
                "semantic_id": class_id,
                "geometry_epoch": entity.geometry_epoch,
                "readout_valid": entity.readout_valid,
            },
        ))
    background = _background_points(snapshot.background)
    return MapSnapshot(
        method="OVIV2-temporal",
        scene_id=snapshot.metadata.scene_id,
        timestamp=snapshot.metadata.timestamp,
        entities=predictions,
        background_xyz=background if len(background) else None,
        scope="current",
        runtime={},
    )


def build_temporal_snapshot(
    state: TemporalRuntimeState,
    *,
    config_sha256: str | None = None,
) -> TemporalCurrentSnapshot:
    if not isinstance(state, TemporalRuntimeState):
        raise TypeError("state must be a TemporalRuntimeState")
    raw_background = object.__getattribute__(state, "_background_state")
    raw_ledger = object.__getattribute__(state, "_ledger_state")
    if raw_ledger is None:
        if raw_background.active_block_count or raw_background.last_blocks_touched:
            raise ValueError("background without a ledger must remain empty")
        committed_background = raw_background
    else:
        committed_background = raw_ledger._volume
        if (
            raw_background.config != committed_background.config
            or raw_background.canonical_block_state()
            != committed_background.canonical_block_state()
            or raw_background.last_blocks_touched
            != committed_background.last_blocks_touched
        ):
            raise ValueError("background must match the ledger committed volume")
    digest = config_sha256
    if digest is None:
        digest = hashlib.sha256(
            _canonical_json(asdict(committed_background.config))
        ).hexdigest()
    wrappers: list[TemporalSnapshotEntity] = []
    seen_ids: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    assert state.geometry is not None
    for wrapper in state.entities:
        entity_id = wrapper.lifecycle.entity_id
        if entity_id in seen_ids:
            raise ValueError("duplicate temporal entity ID")
        seen_ids.add(entity_id)
        try:
            epoch = state.geometry.current(entity_id)
        except KeyError as exc:
            raise ValueError("temporal entity is missing a current geometry epoch") from exc
        pair = (entity_id, epoch.epoch_id)
        if pair in seen_pairs:
            raise ValueError("duplicate entity/current-epoch pair")
        seen_pairs.add(pair)
        if epoch.entity_id != entity_id:
            raise ValueError("wrapper and geometry epoch entity IDs do not match")
        if (
            not np.array_equal(wrapper.object_to_world, epoch.object_to_world)
            or wrapper.submap != epoch.submap
        ):
            raise ValueError("wrapper geometry does not match its current epoch")
        if not epoch.readout_valid:
            continue
        authority = TemporalEntityState(
            lifecycle=wrapper.lifecycle,
            semantic_probabilities=wrapper.semantic_probabilities,
            image_prototype=wrapper.image_prototype,
            extent_xyz=wrapper.extent_xyz,
            object_to_world=epoch.object_to_world,
            submap=epoch.submap,
            first_seen_frame_id=wrapper.first_seen_frame_id,
            last_seen_frame_id=wrapper.last_seen_frame_id,
            feature_model_id=wrapper.feature_model_id,
        )
        wrappers.append(TemporalSnapshotEntity(authority, epoch.epoch_id, True))
    return TemporalCurrentSnapshot(
        TemporalSnapshotMetadata(
            state.scene_id,
            max(state.last_frame_id, 0),
            state.last_timestamp,
            state.revision,
            committed_background.config.voxel_size_m,
            digest,
        ),
        tuple(wrappers),
        committed_background,
    )


__all__ = [
    "TemporalCheckpointPublicationUncertainError",
    "TemporalSnapshotMetadata",
    "TemporalCurrentSnapshot",
    "TemporalSnapshotEntity",
    "TemporalCompactCheckpoint",
    "build_temporal_map_snapshot",
    "build_temporal_snapshot",
]
