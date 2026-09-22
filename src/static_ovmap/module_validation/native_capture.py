"""Immutable, lineage-safe records for instrumented native OVI capture."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .assets import sha256_file
from .contracts import atomic_write_json, canonical_digest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY = re.compile(
    r"(^|_)(gt|ground_truth|correct(?:ness)?|cause_ledger|future)(_|$)",
    re.IGNORECASE,
)


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _relative_artifact_path(value: str, name: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative artifact path")
    return path.as_posix()


def _freeze_metadata(value: Any, location: str = "$") -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key)
            if _FORBIDDEN_KEY.search(normalized):
                raise ValueError(f"forbidden prediction metadata key at {location}.{normalized}")
            frozen[normalized] = _freeze_metadata(item, f"{location}.{normalized}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item, f"{location}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(f"unsupported or non-finite prediction metadata at {location}")


def _json_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_metadata(item) for item in value]
    return value


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype | str | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    array.flags.writeable = False
    return array


def _array_digest(array: np.ndarray) -> str:
    return canonical_digest(
        {
            "dtype": array.dtype.str,
            "shape": array.shape,
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


@dataclass(frozen=True)
class RegionRequest:
    scene_id: str
    frame_id: int
    target_id: str
    lineage: tuple[str, ...]
    source_map_version: str
    target_mask_sha256: str
    bbox_xyxy: tuple[int, int, int, int]
    native_union_mask_sha256: str
    visible_target_pixels: int
    crop_convention: str
    requested_view_rank: int
    image_sha256: str
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.scene_id or self.frame_id < 0 or not self.target_id:
            raise ValueError("request scene, nonnegative frame, and target are required")
        lineage = tuple(str(item) for item in self.lineage)
        if not lineage or lineage[-1] != self.target_id or any(not item for item in lineage):
            raise ValueError("request lineage must be nonempty and end at target_id")
        if not self.source_map_version or not self.crop_convention:
            raise ValueError("request source map version and crop convention are required")
        for name in ("target_mask_sha256", "native_union_mask_sha256", "image_sha256"):
            _validate_sha256(getattr(self, name), name)
        bbox = tuple(int(value) for value in self.bbox_xyxy)
        if len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must be a nonempty nonnegative half-open box")
        if self.visible_target_pixels <= 0 or self.requested_view_rank < 0:
            raise ValueError("request visible pixels must be positive and rank nonnegative")
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "request_id", canonical_digest(self.identity_fields()))

    def identity_fields(self) -> dict[str, Any]:
        return {
            "schema": "ovimap-region-request-v1",
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "target_id": self.target_id,
            "lineage": self.lineage,
            "source_map_version": self.source_map_version,
            "target_mask_sha256": self.target_mask_sha256,
            "bbox_xyxy": self.bbox_xyxy,
            "native_union_mask_sha256": self.native_union_mask_sha256,
            "visible_target_pixels": self.visible_target_pixels,
            "crop_convention": self.crop_convention,
            "requested_view_rank": self.requested_view_rank,
            "image_sha256": self.image_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, **self.identity_fields()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RegionRequest:
        request = cls(
            scene_id=str(value["scene_id"]),
            frame_id=int(value["frame_id"]),
            target_id=str(value["target_id"]),
            lineage=tuple(value["lineage"]),
            source_map_version=str(value["source_map_version"]),
            target_mask_sha256=str(value["target_mask_sha256"]),
            bbox_xyxy=tuple(value["bbox_xyxy"]),
            native_union_mask_sha256=str(value["native_union_mask_sha256"]),
            visible_target_pixels=int(value["visible_target_pixels"]),
            crop_convention=str(value["crop_convention"]),
            requested_view_rank=int(value["requested_view_rank"]),
            image_sha256=str(value["image_sha256"]),
        )
        if value.get("request_id") != request.request_id:
            raise ValueError("region request identity mismatch")
        return request


@dataclass(frozen=True)
class FrameObservation:
    scene_id: str
    frame_id: int
    pose_c2w: np.ndarray
    image_size_hw: tuple[int, int]
    intrinsics: np.ndarray
    rgb_path: str
    rgb_sha256: str
    depth_path: str
    depth_sha256: str
    panoptic_path: str
    panoptic_sha256: str
    global_owner_path: str
    global_owner_sha256: str
    map_state_id: str
    refined_segment_ids: np.ndarray
    registered_labels: np.ndarray
    requests: tuple[RegionRequest, ...]
    native_selected_request_ids: tuple[str, ...]
    request_completion_boundary: int

    def __post_init__(self) -> None:
        if not self.scene_id or self.frame_id < 0 or not self.map_state_id:
            raise ValueError("frame scene, nonnegative ID, and map state are required")
        pose = _readonly_array(self.pose_c2w, name="pose_c2w", dtype=np.float64, ndim=2)
        intrinsics = _readonly_array(self.intrinsics, name="intrinsics", dtype=np.float64, ndim=2)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("pose_c2w must be a finite homogeneous 4x4 matrix")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        image_size = tuple(int(value) for value in self.image_size_hw)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size_hw must contain two positive integers")
        paths = ("rgb_path", "depth_path", "panoptic_path", "global_owner_path")
        hashes = ("rgb_sha256", "depth_sha256", "panoptic_sha256", "global_owner_sha256")
        for name in paths:
            object.__setattr__(self, name, _relative_artifact_path(getattr(self, name), name))
        for name in hashes:
            _validate_sha256(getattr(self, name), name)
        refined = _readonly_array(
            self.refined_segment_ids,
            name="refined_segment_ids",
            dtype=np.int64,
            ndim=1,
        )
        registered = _readonly_array(
            self.registered_labels,
            name="registered_labels",
            dtype=np.int64,
            ndim=1,
        )
        if len(refined) != len(registered) or np.any(refined < 0) or np.any(registered < 0):
            raise ValueError("refined segment IDs and registered labels must align and be nonnegative")
        requests = tuple(self.requests)
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("frame requests must have unique identities")
        for request in requests:
            if request.scene_id != self.scene_id or request.frame_id != self.frame_id:
                raise ValueError("request scene/frame does not match observation")
            if request.source_map_version != self.map_state_id:
                raise ValueError("request source map version does not match observation")
        selected = tuple(self.native_selected_request_ids)
        request_ids = {request.request_id for request in requests}
        if len(set(selected)) != len(selected) or not set(selected).issubset(request_ids):
            raise ValueError("native selected request IDs must be a unique request subset")
        if self.request_completion_boundary < 0:
            raise ValueError("request completion boundary must be nonnegative")
        object.__setattr__(self, "pose_c2w", pose)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "image_size_hw", image_size)
        object.__setattr__(self, "refined_segment_ids", refined)
        object.__setattr__(self, "registered_labels", registered)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "native_selected_request_ids", selected)

    def to_manifest(self, index: int) -> dict[str, Any]:
        prefix = f"frame_{index:04d}"
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "pose_c2w": self.pose_c2w.tolist(),
            "image_size_hw": list(self.image_size_hw),
            "intrinsics": self.intrinsics.tolist(),
            "rgb_path": self.rgb_path,
            "rgb_sha256": self.rgb_sha256,
            "depth_path": self.depth_path,
            "depth_sha256": self.depth_sha256,
            "panoptic_path": self.panoptic_path,
            "panoptic_sha256": self.panoptic_sha256,
            "global_owner_path": self.global_owner_path,
            "global_owner_sha256": self.global_owner_sha256,
            "map_state_id": self.map_state_id,
            "refined_segment_ids_array": f"{prefix}_refined_segment_ids",
            "registered_labels_array": f"{prefix}_registered_labels",
            "requests": [request.to_dict() for request in self.requests],
            "native_selected_request_ids": list(self.native_selected_request_ids),
            "request_completion_boundary": self.request_completion_boundary,
        }


@dataclass(frozen=True)
class NativeScenePack:
    scene_id: str
    family_id: str
    split: str
    domain: str
    source_config: Mapping[str, Any]
    source_revisions: Mapping[str, Any]
    surface_xyz: np.ndarray
    surface_normals: np.ndarray
    normal_valid: np.ndarray
    original_owner: np.ndarray
    segment_labels: np.ndarray
    alias_table: tuple[tuple[int, int], ...]
    instance_registry: Mapping[int, Mapping[str, Any]]
    tsdf_sha256: str
    projection_identity: str
    frames: tuple[FrameObservation, ...]
    class_vocabulary: tuple[str, ...]
    text_space_id: str
    surface_faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    array_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.scene_id or not self.family_id or self.split not in {
            "FIT",
            "CAL",
            "SELECT",
            "CONFIRM",
            "HISTORICAL",
        }:
            raise ValueError("scene/family and supported split are required")
        if self.domain not in {"H", "N", "R"}:
            raise ValueError("domain must be H, N, or R")
        source_config = _freeze_metadata(self.source_config, "$.source_config")
        source_revisions = _freeze_metadata(self.source_revisions, "$.source_revisions")
        xyz = _readonly_array(self.surface_xyz, name="surface_xyz", dtype=np.float64, ndim=2)
        normals = _readonly_array(
            self.surface_normals,
            name="surface_normals",
            dtype=np.float64,
            ndim=2,
        )
        normal_valid = _readonly_array(self.normal_valid, name="normal_valid", dtype=bool, ndim=1)
        owners = _readonly_array(self.original_owner, name="original_owner", dtype=np.int64, ndim=1)
        segments = _readonly_array(self.segment_labels, name="segment_labels", dtype=np.int64, ndim=1)
        faces = _readonly_array(self.surface_faces, name="surface_faces", dtype=np.int64, ndim=2)
        row_count = len(xyz)
        if (
            xyz.shape != (row_count, 3)
            or normals.shape != (row_count, 3)
            or normal_valid.shape != (row_count,)
            or owners.shape != (row_count,)
            or segments.shape != (row_count,)
        ):
            raise ValueError("all surface rows must align exactly")
        if row_count == 0 or not np.isfinite(xyz).all() or not np.isfinite(normals).all():
            raise ValueError("surface coordinates and estimated normals must be finite and nonempty")
        if faces.shape[1:] != (3,) or (
            faces.size and (np.any(faces < 0) or np.any(faces >= row_count))
        ):
            raise ValueError("surface faces must be valid triangle row indices")
        if np.any(owners < 0) or np.any(segments < 0):
            raise ValueError("surface owner and segment labels must be nonnegative")
        aliases = tuple((int(old), int(new)) for old, new in self.alias_table)
        if any(old <= 0 or new <= 0 or old == new for old, new in aliases):
            raise ValueError("alias rows must contain distinct positive old/new labels")
        if len({old for old, _ in aliases}) != len(aliases):
            raise ValueError("an old label may have only one alias target")
        registry: dict[int, Any] = {}
        for owner, metadata in self.instance_registry.items():
            owner_id = int(owner)
            if owner_id <= 0:
                raise ValueError("instance registry IDs must be positive")
            registry[owner_id] = _freeze_metadata(metadata, f"$.instance_registry.{owner_id}")
        active_owners = set(map(int, np.unique(owners))) - {0}
        if set(registry) != active_owners:
            raise ValueError("instance registry must exactly match positive original owners")
        _validate_sha256(self.tsdf_sha256, "tsdf_sha256")
        if not self.projection_identity or not self.text_space_id:
            raise ValueError("projection and text-space identities are required")
        frames = tuple(self.frames)
        if any(frame.scene_id != self.scene_id for frame in frames):
            raise ValueError("all frames must belong to the scene pack")
        if tuple(frame.frame_id for frame in frames) != tuple(sorted({frame.frame_id for frame in frames})):
            raise ValueError("frames must have unique ascending IDs")
        vocabulary = tuple(str(label) for label in self.class_vocabulary)
        if not vocabulary or any(not label for label in vocabulary) or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("class vocabulary must be nonempty and unique")
        if self.array_sha256 is not None:
            _validate_sha256(self.array_sha256, "array_sha256")
        object.__setattr__(self, "source_config", source_config)
        object.__setattr__(self, "source_revisions", source_revisions)
        object.__setattr__(self, "surface_xyz", xyz)
        object.__setattr__(self, "surface_normals", normals)
        object.__setattr__(self, "normal_valid", normal_valid)
        object.__setattr__(self, "original_owner", owners)
        object.__setattr__(self, "segment_labels", segments)
        object.__setattr__(self, "surface_faces", faces)
        object.__setattr__(self, "alias_table", aliases)
        object.__setattr__(self, "instance_registry", MappingProxyType(registry))
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "class_vocabulary", vocabulary)

    @property
    def identity(self) -> str:
        return canonical_digest(
            {
                "schema": "ovimap-native-scene-pack-v1",
                "scene_id": self.scene_id,
                "family_id": self.family_id,
                "split": self.split,
                "domain": self.domain,
                "source_config": _json_metadata(self.source_config),
                "source_revisions": _json_metadata(self.source_revisions),
                "array_digests": {
                    "surface_xyz": _array_digest(self.surface_xyz),
                    "surface_normals": _array_digest(self.surface_normals),
                    "normal_valid": _array_digest(self.normal_valid),
                    "original_owner": _array_digest(self.original_owner),
                    "segment_labels": _array_digest(self.segment_labels),
                    "surface_faces": _array_digest(self.surface_faces),
                    "frame_arrays": [
                        {
                            "refined": _array_digest(frame.refined_segment_ids),
                            "registered": _array_digest(frame.registered_labels),
                        }
                        for frame in self.frames
                    ],
                },
                "alias_table": self.alias_table,
                "instance_registry": _json_metadata(self.instance_registry),
                "tsdf_sha256": self.tsdf_sha256,
                "projection_identity": self.projection_identity,
                "frames": [frame.to_manifest(index) for index, frame in enumerate(self.frames)],
                "class_vocabulary": self.class_vocabulary,
                "text_space_id": self.text_space_id,
            }
        )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_native_scene_pack(pack: NativeScenePack, output: Path | str) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {
        "surface_xyz": pack.surface_xyz,
        "surface_normals": pack.surface_normals,
        "normal_valid": pack.normal_valid,
        "original_owner": pack.original_owner,
        "segment_labels": pack.segment_labels,
        "surface_faces": pack.surface_faces,
    }
    frame_manifests = []
    for index, frame in enumerate(pack.frames):
        arrays[f"frame_{index:04d}_refined_segment_ids"] = frame.refined_segment_ids
        arrays[f"frame_{index:04d}_registered_labels"] = frame.registered_labels
        frame_manifests.append(frame.to_manifest(index))
    arrays_path = root / "arrays.npz"
    _write_npz(arrays_path, arrays)
    manifest = {
        "schema_version": 1,
        "artifact_type": "OVIMAP_NATIVE_SCENE_PACK",
        "identity": pack.identity,
        "scene_id": pack.scene_id,
        "family_id": pack.family_id,
        "split": pack.split,
        "domain": pack.domain,
        "source_config": _json_metadata(pack.source_config),
        "source_revisions": _json_metadata(pack.source_revisions),
        "arrays": {
            "path": arrays_path.name,
            "sha256": sha256_file(arrays_path),
            "keys": sorted(arrays),
        },
        "alias_table": [list(row) for row in pack.alias_table],
        "instance_registry": {
            str(owner): _json_metadata(metadata)
            for owner, metadata in sorted(pack.instance_registry.items())
        },
        "tsdf_sha256": pack.tsdf_sha256,
        "projection_identity": pack.projection_identity,
        "frames": frame_manifests,
        "class_vocabulary": list(pack.class_vocabulary),
        "text_space_id": pack.text_space_id,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def _frame_from_manifest(value: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> FrameObservation:
    return FrameObservation(
        scene_id=str(value["scene_id"]),
        frame_id=int(value["frame_id"]),
        pose_c2w=value["pose_c2w"],
        image_size_hw=tuple(value["image_size_hw"]),
        intrinsics=value["intrinsics"],
        rgb_path=str(value["rgb_path"]),
        rgb_sha256=str(value["rgb_sha256"]),
        depth_path=str(value["depth_path"]),
        depth_sha256=str(value["depth_sha256"]),
        panoptic_path=str(value["panoptic_path"]),
        panoptic_sha256=str(value["panoptic_sha256"]),
        global_owner_path=str(value["global_owner_path"]),
        global_owner_sha256=str(value["global_owner_sha256"]),
        map_state_id=str(value["map_state_id"]),
        refined_segment_ids=arrays[str(value["refined_segment_ids_array"])],
        registered_labels=arrays[str(value["registered_labels_array"])],
        requests=tuple(RegionRequest.from_dict(item) for item in value["requests"]),
        native_selected_request_ids=tuple(value["native_selected_request_ids"]),
        request_completion_boundary=int(value["request_completion_boundary"]),
    )


def load_native_scene_pack(manifest_path: Path | str) -> NativeScenePack:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "OVIMAP_NATIVE_SCENE_PACK":
        raise ValueError("unsupported native scene pack manifest")
    arrays_relative = _relative_artifact_path(manifest["arrays"]["path"], "arrays.path")
    arrays_path = (path.parent / arrays_relative).resolve()
    if not arrays_path.is_relative_to(path.parent) or not arrays_path.is_file():
        raise ValueError("native scene array path leaves its pack")
    actual_sha256 = sha256_file(arrays_path)
    if actual_sha256 != manifest["arrays"]["sha256"]:
        raise ValueError("native scene array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if sorted(arrays) != sorted(manifest["arrays"]["keys"]):
        raise ValueError("native scene array key manifest mismatch")
    frames = tuple(_frame_from_manifest(value, arrays) for value in manifest["frames"])
    pack = NativeScenePack(
        scene_id=str(manifest["scene_id"]),
        family_id=str(manifest["family_id"]),
        split=str(manifest["split"]),
        domain=str(manifest["domain"]),
        source_config=manifest["source_config"],
        source_revisions=manifest["source_revisions"],
        surface_xyz=arrays["surface_xyz"],
        surface_normals=arrays["surface_normals"],
        normal_valid=arrays["normal_valid"],
        original_owner=arrays["original_owner"],
        segment_labels=arrays["segment_labels"],
        surface_faces=arrays["surface_faces"],
        alias_table=tuple(tuple(row) for row in manifest["alias_table"]),
        instance_registry={int(owner): value for owner, value in manifest["instance_registry"].items()},
        tsdf_sha256=str(manifest["tsdf_sha256"]),
        projection_identity=str(manifest["projection_identity"]),
        frames=frames,
        class_vocabulary=tuple(manifest["class_vocabulary"]),
        text_space_id=str(manifest["text_space_id"]),
        array_sha256=actual_sha256,
    )
    if pack.identity != manifest["identity"]:
        raise ValueError("native scene pack identity mismatch")
    return pack


def _plain_value(value: Any, location: str = "$") -> Any:
    if isinstance(value, np.generic):
        return _plain_value(value.item(), location)
    if isinstance(value, np.ndarray):
        return _plain_value(value.tolist(), location)
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_value(item, f"{location}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(f"unsupported or non-finite native value at {location}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_png(path: Path, array: np.ndarray, *, rgb: bool = False) -> None:
    from PIL import Image

    image = np.asarray(array)
    if rgb:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB capture must be an HxWx3 uint8 array")
    elif image.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        if image.size and (np.min(image) < 0 or np.max(image) > np.iinfo(np.uint16).max):
            raise ValueError("PNG label capture exceeds uint16 range")
        image = image.astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    _atomic_write_bytes(path, buffer.getvalue())


def _read_native_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertices = ply["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32,
        copy=False,
    )
    normal_names = next(
        (
            names
            for names in (("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz"))
            if all(name in vertices.dtype.names for name in names)
        ),
        None,
    )
    if normal_names is not None:
        normals = np.column_stack(tuple(vertices[name] for name in normal_names)).astype(
            np.float32,
            copy=False,
        )
        normal_valid = np.isfinite(normals).all(axis=1) & (
            np.linalg.norm(normals, axis=1) > 1e-12
        )
        normals = np.where(normal_valid[:, None], normals, 0.0).astype(np.float32)
    else:
        normals = np.zeros_like(xyz, dtype=np.float32)
        normal_valid = np.zeros(len(xyz), dtype=bool)

    if "face" not in (element.name for element in ply.elements):
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        face_rows = [np.asarray(row, dtype=np.int64) for row in ply["face"]["vertex_indices"]]
        if any(row.shape != (3,) for row in face_rows):
            raise ValueError("native mesh contains non-triangle faces")
        faces = (
            np.stack(face_rows, axis=0)
            if face_rows
            else np.empty((0, 3), dtype=np.int64)
        )
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("native mesh must contain finite surface vertices")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(xyz))):
        raise ValueError("native mesh face indices leave the vertex table")
    return xyz, normals, normal_valid, faces


def _native_tsdf_arrays(exported: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {"block_indices", "distance", "weight", "color", "voxel_size", "voxels_per_side"}
    if set(exported) != required:
        raise ValueError("native TSDF export returned an unexpected schema")
    arrays = {name: np.asarray(value) for name, value in exported.items()}
    side = arrays["voxels_per_side"]
    size = arrays["voxel_size"]
    if side.shape != () or side.dtype.kind not in "iu" or side <= 0:
        raise ValueError("native TSDF voxels_per_side must be a positive integer")
    if size.shape != () or not np.isfinite(size) or size <= 0:
        raise ValueError("native TSDF voxel_size must be finite and positive")
    blocks = arrays["block_indices"]
    if blocks.dtype != np.int32 or blocks.ndim != 2 or blocks.shape[1] != 3:
        raise ValueError("native TSDF block_indices must be int32 [B,3]")
    if not np.array_equal(blocks, np.unique(blocks, axis=0)):
        raise ValueError("native TSDF block indices must be unique and lexicographically sorted")
    count = len(blocks) * int(side) ** 3
    for name in ("distance", "weight"):
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f"native TSDF {name} must preserve finite float32 voxel rows")
    if np.any(arrays["weight"] < 0):
        raise ValueError("native TSDF weight cannot be negative")
    if arrays["color"].dtype != np.uint8 or arrays["color"].shape != (count, 4):
        raise ValueError("native TSDF color must preserve uint8 RGBA voxel rows")
    return arrays


def _verify_native_owner_colors(mesh_path: Path, owners: np.ndarray, gsm_node: Any) -> dict:
    from plyfile import PlyData

    vertices = PlyData.read(mesh_path)["vertex"].data
    if not all(name in vertices.dtype.names for name in ("red", "green", "blue")):
        raise ValueError("native instance mesh is missing its original RGB colors")
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
    unique_owners, inverse = np.unique(owners, return_inverse=True)
    lookup = np.stack([gsm_node.getInstanceColor(int(owner)) for owner in unique_owners])
    if lookup.shape != (len(unique_owners), 3) or not np.array_equal(colors, lookup[inverse]):
        raise ValueError("native numerical owner export disagrees with instance mesh colors")
    return {"exact": True, "checked_rows": len(owners), "owner_count": len(unique_owners)}


class NativeCaptureSession:
    """Transactional receiver for the three instrumented native mapper boundaries."""

    def __init__(
        self,
        *,
        capture_root: Path | str,
        scene_id: str,
        dataset: str,
        image_size_hw: Sequence[int],
        intrinsics: Any,
        scheduled_frame_ids: Sequence[int],
        source_config: Mapping[str, Any],
    ) -> None:
        if not scene_id or not dataset:
            raise ValueError("capture scene and dataset are required")
        image_size = tuple(int(value) for value in image_size_hw)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("capture image size must contain two positive values")
        camera = _readonly_array(intrinsics, name="intrinsics", dtype=np.float64, ndim=2)
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise ValueError("capture intrinsics must be a finite 3x3 matrix")
        frame_ids = tuple(int(value) for value in scheduled_frame_ids)
        if not frame_ids or frame_ids != tuple(sorted(set(frame_ids))):
            raise ValueError("scheduled frame IDs must be nonempty, unique, and ascending")

        root = Path(capture_root).resolve()
        scene_root = root / scene_id
        root.mkdir(parents=True, exist_ok=True)
        scene_root.mkdir(parents=False, exist_ok=False)
        (scene_root / "frames").mkdir()
        self.enabled = True
        self.root = root
        self.scene_root = scene_root
        self.scene_id = scene_id
        self.dataset = dataset
        self.image_size_hw = image_size
        self.intrinsics = camera
        self.scheduled_frame_ids = frame_ids
        self.source_config = _freeze_metadata(source_config, "$.source_config")
        self._frames: dict[int, dict[str, Any]] = {}
        self._request_ranks: dict[int, int] = {}
        self._finalized = False
        atomic_write_json(
            self.scene_root / "capture_state.json",
            {
                "schema_version": 1,
                "artifact_type": "OVIMAP_NATIVE_CAPTURE_STATE",
                "status": "CAPTURING",
                "scene_id": scene_id,
                "dataset": dataset,
                "image_size_hw": list(image_size),
                "intrinsics": camera.tolist(),
                "scheduled_frame_ids": list(frame_ids),
                "source_config": _json_metadata(self.source_config),
                "completed_frame_ids": [],
            },
        )

    def _require_frame(self, frame_id: int, expected_stage: str) -> dict[str, Any]:
        frame = self._frames.get(int(frame_id))
        if frame is None or frame["stage"] != expected_stage:
            raise RuntimeError(
                f"frame {frame_id} must be at {expected_stage}, found "
                f"{None if frame is None else frame['stage']}"
            )
        return frame

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.scene_root).as_posix()

    def _write_state(self) -> None:
        completed = sorted(
            frame_id
            for frame_id, frame in self._frames.items()
            if frame["stage"] == "RAYCAST"
        )
        atomic_write_json(
            self.scene_root / "capture_state.json",
            {
                "schema_version": 1,
                "artifact_type": "OVIMAP_NATIVE_CAPTURE_STATE",
                "status": "COMPLETE" if self._finalized else "CAPTURING",
                "scene_id": self.scene_id,
                "dataset": self.dataset,
                "image_size_hw": list(self.image_size_hw),
                "intrinsics": self.intrinsics.tolist(),
                "scheduled_frame_ids": list(self.scheduled_frame_ids),
                "source_config": _json_metadata(self.source_config),
                "completed_frame_ids": completed,
            },
        )

    def before_insertion(
        self,
        *,
        frame_id: int,
        rgb_image: Any,
        depth_m: Any,
        pose_c2w: Any,
        panoptic_raster: Any,
        segments: Sequence[Any],
        rgb_source_path: str,
        depth_source_path: str,
        panoptic_source_path: str,
    ) -> None:
        frame_id = int(frame_id)
        if self._finalized or frame_id in self._frames:
            raise RuntimeError(f"frame {frame_id} is duplicate or capture is finalized")
        if frame_id not in self.scheduled_frame_ids:
            raise ValueError(f"frame {frame_id} is outside the locked schedule")
        rgb = np.array(rgb_image, dtype=np.uint8, copy=True)
        depth = np.array(depth_m, dtype=np.float32, copy=True)
        panoptic = np.array(panoptic_raster, dtype=np.int64, copy=True)
        pose = np.array(pose_c2w, dtype=np.float64, copy=True)
        height, width = self.image_size_hw
        if rgb.shape != (height, width, 3) or depth.shape != (height, width):
            raise ValueError("captured RGB/depth rows do not match registered image size")
        if panoptic.shape != (height, width) or np.any(panoptic < 0):
            raise ValueError("captured panoptic raster must be aligned and nonnegative")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("captured pose must be a finite 4x4 matrix")

        frame_root = self.scene_root / "frames" / f"{frame_id:06d}"
        frame_root.mkdir()
        rgb_path = frame_root / "rgb.png"
        depth_path = frame_root / "depth.npz"
        panoptic_path = frame_root / "panoptic.png"
        _write_png(rgb_path, rgb, rgb=True)
        _write_npz(depth_path, {"depth_m": depth})
        _write_png(panoptic_path, panoptic)

        segment_rows: list[dict[str, Any]] = []
        point_chunks: list[np.ndarray] = []
        offsets = [0]
        refined_raster = np.zeros((height, width), dtype=np.uint16)
        for insertion_index, segment in enumerate(segments):
            points = np.asarray(segment.points, dtype=np.float32).reshape(-1, 3)
            if len(points) == 0 or not np.isfinite(points).all():
                raise ValueError("captured refined segments must contain finite points")
            refined_id = int(segment.index) + 1
            if refined_id <= 0 or refined_id > np.iinfo(np.uint16).max:
                raise ValueError("captured refined segment ID exceeds uint16 range")
            z = points[:, 2]
            valid = z > 0
            u = np.rint(
                self.intrinsics[0, 0] * points[:, 0] / z + self.intrinsics[0, 2]
            ).astype(np.int64)
            v = np.rint(
                self.intrinsics[1, 1] * points[:, 1] / z + self.intrinsics[1, 2]
            ).astype(np.int64)
            valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
            refined_raster[v[valid], u[valid]] = refined_id
            point_chunks.append(points)
            offsets.append(offsets[-1] + len(points))
            segment_rows.append(
                {
                    "insertion_index": insertion_index,
                    "refined_segment_id": refined_id,
                    "source_segment_index": int(segment.index),
                    "input_instance_label": int(segment.instance_label),
                    "semantic_label": int(segment.class_label),
                    "is_thing": bool(segment.is_thing),
                    "point_count": len(points),
                }
            )
        if not segment_rows:
            raise ValueError("pre-insertion capture requires at least one segment")
        preinsert_path = frame_root / "preinsert.npz"
        _write_npz(
            preinsert_path,
            {
                "segment_points": np.concatenate(point_chunks, axis=0),
                "segment_point_offsets": np.asarray(offsets, dtype=np.int64),
                "refined_segment_raster": refined_raster,
                "refined_segment_ids": np.asarray(
                    [row["refined_segment_id"] for row in segment_rows], dtype=np.int64
                ),
            },
        )
        self._frames[frame_id] = {
            "stage": "PREINSERT",
            "frame_root": frame_root,
            "pose_c2w": pose,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "panoptic_path": panoptic_path,
            "preinsert_path": preinsert_path,
            "segment_rows": segment_rows,
            "source_paths": {
                "rgb": str(rgb_source_path),
                "depth": str(depth_source_path),
                "panoptic": str(panoptic_source_path),
            },
        }

    def after_integration(self, *, frame_id: int, native_state: Mapping[str, Any]) -> None:
        frame = self._require_frame(int(frame_id), "PREINSERT")
        state = _plain_value(native_state, "$.native_state")
        rows = state.get("segments", [])
        expected = frame["segment_rows"]
        if len(rows) != len(expected):
            raise ValueError("native registered segment rows do not align with insertion rows")
        registered_labels: list[int] = []
        for index, (native_row, source_row) in enumerate(zip(rows, expected, strict=True)):
            if int(native_row["local_index"]) != index:
                raise ValueError("native segment local indices are not insertion ordered")
            if int(native_row["point_count"]) != source_row["point_count"]:
                raise ValueError("native segment point count changed before export")
            if int(native_row["input_instance_label"]) != source_row["input_instance_label"]:
                raise ValueError("native segment instance provenance does not align")
            registered_label = int(native_row["registered_label"])
            if registered_label < 0:
                raise ValueError("native registered labels must be nonnegative")
            registered_labels.append(registered_label)
        map_state_id = canonical_digest(
            {
                "schema": "ovimap-native-map-state-v1",
                "scene_id": self.scene_id,
                "frame_id": int(frame_id),
                "native_state": state,
            }
        )
        native_state_path = frame["frame_root"] / "native_state.json"
        atomic_write_json(
            native_state_path,
            {"map_state_id": map_state_id, "native_state": state},
        )
        frame.update(
            {
                "stage": "INTEGRATED",
                "native_state": state,
                "native_state_path": native_state_path,
                "map_state_id": map_state_id,
                "registered_labels": np.asarray(registered_labels, dtype=np.int64),
            }
        )

    def after_raycast(
        self,
        *,
        frame_id: int,
        global_owner_raster: Any,
        admissible_views: Sequence[Mapping[str, Any]],
        selected_views: Sequence[Mapping[str, Any]],
        request_completion_boundary: int,
    ) -> None:
        frame_id = int(frame_id)
        frame = self._require_frame(frame_id, "INTEGRATED")
        owners = np.asarray(global_owner_raster)
        if owners.shape != self.image_size_hw or not np.issubdtype(owners.dtype, np.integer):
            raise ValueError("global owner raster must be an aligned integer image")
        if owners.size and (np.min(owners) < 0 or np.max(owners) > np.iinfo(np.uint16).max):
            raise ValueError("global owner raster exceeds uint16 range")
        owners = owners.astype(np.uint16, copy=True)
        owner_path = frame["frame_root"] / "global_owner.png"
        _write_png(owner_path, owners)

        label_instances: dict[int, list[int]] = {}
        for row in frame["native_state"].get("label_instances", []):
            owner = int(row["instance_label"])
            if owner > 0:
                label_instances.setdefault(owner, []).append(int(row["segment_label"]))

        request_arrays: dict[str, np.ndarray] = {}
        requests: list[RegionRequest] = []
        requests_by_owner: dict[int, RegionRequest] = {}
        for candidate in admissible_views:
            owner = int(candidate["glo_inst_id"])
            if owner <= 0 or owner in requests_by_owner:
                raise ValueError("admissible requests require unique positive current owners")
            target_mask = np.asarray(candidate["glo_inst_mask"], dtype=bool)
            union_mask = np.asarray(candidate["union_mask"], dtype=bool)
            if target_mask.shape != self.image_size_hw or union_mask.shape != self.image_size_hw:
                raise ValueError("request masks must align with the registered image")
            if not np.any(target_mask) or not np.all(union_mask[target_mask]):
                raise ValueError("request union mask must contain a nonempty target mask")
            bbox = tuple(int(value) for value in candidate["bbox_xyxy_native"])
            y_coords, x_coords = np.nonzero(target_mask)
            expected_bbox = (
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max()),
                int(y_coords.max()),
            )
            if bbox != expected_bbox:
                raise ValueError("request bbox does not match native global-mask bounds")
            target_id = f"owner:{owner}"
            lineage = tuple(
                [f"segment:{label}" for label in sorted(set(label_instances.get(owner, [])))]
                + [target_id]
            )
            rank = self._request_ranks.get(owner, 0)
            self._request_ranks[owner] = rank + 1
            request = RegionRequest(
                scene_id=self.scene_id,
                frame_id=frame_id,
                target_id=target_id,
                lineage=lineage,
                source_map_version=frame["map_state_id"],
                target_mask_sha256=_array_digest(target_mask),
                bbox_xyxy=bbox,
                native_union_mask_sha256=_array_digest(union_mask),
                visible_target_pixels=int(candidate["overlap_area"]),
                crop_convention="native_global_bbox_union_exclusive_upper_v1",
                requested_view_rank=rank,
                image_sha256=sha256_file(frame["rgb_path"]),
            )
            request_arrays[f"{request.request_id}_target"] = target_mask
            request_arrays[f"{request.request_id}_union"] = union_mask
            requests.append(request)
            requests_by_owner[owner] = request

        selected_ids = []
        for selected in selected_views:
            owner = int(selected["glo_inst_id"])
            if owner not in requests_by_owner:
                raise ValueError("native selection is not a subset of admissible requests")
            selected_ids.append(requests_by_owner[owner].request_id)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("native selection contains duplicate owners")
        request_arrays_path = frame["frame_root"] / "requests.npz"
        _write_npz(request_arrays_path, request_arrays)

        with np.load(frame["preinsert_path"], allow_pickle=False) as preinsert:
            refined_ids = np.array(preinsert["refined_segment_ids"], copy=True)
        observation = FrameObservation(
            scene_id=self.scene_id,
            frame_id=frame_id,
            pose_c2w=frame["pose_c2w"],
            image_size_hw=self.image_size_hw,
            intrinsics=self.intrinsics,
            rgb_path=self._relative(frame["rgb_path"]),
            rgb_sha256=sha256_file(frame["rgb_path"]),
            depth_path=self._relative(frame["depth_path"]),
            depth_sha256=sha256_file(frame["depth_path"]),
            panoptic_path=self._relative(frame["panoptic_path"]),
            panoptic_sha256=sha256_file(frame["panoptic_path"]),
            global_owner_path=self._relative(owner_path),
            global_owner_sha256=sha256_file(owner_path),
            map_state_id=frame["map_state_id"],
            refined_segment_ids=refined_ids,
            registered_labels=frame["registered_labels"],
            requests=tuple(requests),
            native_selected_request_ids=tuple(selected_ids),
            request_completion_boundary=int(request_completion_boundary),
        )
        frame_index = sum(
            item["stage"] == "RAYCAST" for item in self._frames.values()
        )
        frame_manifest = observation.to_manifest(frame_index)
        frame_manifest.update(
            {
                "preinsert": {
                    "path": self._relative(frame["preinsert_path"]),
                    "sha256": sha256_file(frame["preinsert_path"]),
                    "segments": frame["segment_rows"],
                },
                "native_state": {
                    "path": self._relative(frame["native_state_path"]),
                    "sha256": sha256_file(frame["native_state_path"]),
                },
                "request_arrays": {
                    "path": self._relative(request_arrays_path),
                    "sha256": sha256_file(request_arrays_path),
                    "keys": sorted(request_arrays),
                },
                "source_paths": frame["source_paths"],
            }
        )
        frame_manifest_path = frame["frame_root"] / "manifest.json"
        atomic_write_json(frame_manifest_path, frame_manifest)
        frame.update(
            {
                "stage": "RAYCAST",
                "observation": observation,
                "manifest": frame_manifest,
                "manifest_path": frame_manifest_path,
            }
        )
        self._write_state()

    def finalize(self, *, gsm_node: Any, instance_mesh_path: Path | str) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("capture session is already finalized")
        completed = [
            self._frames[frame_id]
            for frame_id in sorted(self._frames)
            if self._frames[frame_id]["stage"] == "RAYCAST"
        ]
        if not completed:
            raise RuntimeError("cannot finalize a capture without complete frame observations")
        mesh_path = Path(instance_mesh_path).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"native instance mesh is missing: {mesh_path}")
        xyz, normals, normal_valid, faces = _read_native_mesh(mesh_path)
        exported = gsm_node.exportStudySurfaceLabels(np.ascontiguousarray(xyz, dtype=np.float32))
        required = {
            "segment_labels",
            "instance_labels",
            "semantic_labels",
            "valid",
            "label_mapping_count_threshold_factor",
        }
        if set(exported) != required:
            raise ValueError("native surface export returned an unexpected schema")
        arrays = {
            "surface_xyz": xyz,
            "surface_normals": normals,
            "normal_valid": normal_valid,
            "surface_faces": faces,
            "segment_labels": np.asarray(exported["segment_labels"], dtype=np.int64),
            "original_owner": np.asarray(exported["instance_labels"], dtype=np.int64),
            "semantic_labels": np.asarray(exported["semantic_labels"], dtype=np.int64),
            "label_valid": np.asarray(exported["valid"], dtype=bool),
        }
        for index, frame in enumerate(completed):
            observation = frame["observation"]
            arrays[f"frame_{index:04d}_refined_segment_ids"] = (
                observation.refined_segment_ids
            )
            arrays[f"frame_{index:04d}_registered_labels"] = (
                observation.registered_labels
            )
        for name in ("segment_labels", "original_owner", "semantic_labels", "label_valid"):
            if arrays[name].shape != (len(xyz),):
                raise ValueError(f"native {name} rows do not align with surface vertices")
        threshold = float(exported["label_mapping_count_threshold_factor"])
        if not np.isclose(threshold, 0.1, rtol=0.0, atol=1e-7):
            raise ValueError("native surface export did not use the mesh label threshold")
        owner_parity = _verify_native_owner_colors(mesh_path, arrays["original_owner"], gsm_node)
        tsdf_arrays = _native_tsdf_arrays(gsm_node.exportStudyTsdfState())
        tsdf_path = self.scene_root / "tsdf.npz"
        _write_npz(tsdf_path, tsdf_arrays)
        surface_path = self.scene_root / "surface.npz"
        _write_npz(surface_path, arrays)

        aliases = sorted(
            {
                (int(alias["old_label"]), int(alias["resolved_label"]))
                for frame in completed
                for alias in frame["native_state"].get("aliases", [])
            }
        )
        registry: dict[str, dict[str, Any]] = {}
        for owner in sorted(set(arrays["original_owner"].tolist()) - {0}):
            semantic = arrays["semantic_labels"][arrays["original_owner"] == owner]
            labels, counts = np.unique(semantic, return_counts=True)
            order = np.lexsort((labels, -counts))
            registry[str(owner)] = {
                "semantic_label": int(labels[order[0]]),
                "surface_point_count": len(semantic),
                "source": "native_surface_export",
            }

        module_name = type(gsm_node).__module__
        module_identity: dict[str, Any] = {"module": module_name}
        try:
            module_path_value = getattr(importlib.import_module(module_name), "__file__", None)
            if module_path_value:
                module_path = Path(module_path_value).resolve()
                module_identity.update(
                    {
                        "path": str(module_path),
                        "sha256": sha256_file(module_path),
                    }
                )
        except (ImportError, OSError):
            module_identity["path"] = None

        manifest = {
            "schema_version": 1,
            "artifact_type": "OVIMAP_NATIVE_CAPTURE",
            "scene_id": self.scene_id,
            "dataset": self.dataset,
            "source_config": _json_metadata(self.source_config),
            "scheduled_frame_ids": list(self.scheduled_frame_ids),
            "completed_frame_ids": [
                int(frame["observation"].frame_id) for frame in completed
            ],
            "frames": [frame["manifest"] for frame in completed],
            "surface": {
                "path": self._relative(surface_path),
                "sha256": sha256_file(surface_path),
                "row_count": len(xyz),
                "face_count": len(faces),
                "keys": sorted(arrays),
                "source_mesh_path": str(mesh_path),
                "source_mesh_sha256": sha256_file(mesh_path),
                "label_mapping_count_threshold_factor": threshold,
            },
            "alias_table": [list(row) for row in aliases],
            "tsdf": {
                "path": self._relative(tsdf_path),
                "sha256": sha256_file(tsdf_path),
                "block_count": len(tsdf_arrays["block_indices"]),
                "voxel_count": len(tsdf_arrays["distance"]),
                "voxel_size": float(tsdf_arrays["voxel_size"]),
                "voxels_per_side": int(tsdf_arrays["voxels_per_side"]),
                "order": "lexicographic_blocks_native_linear_voxels",
            },
            "native_owner_mesh_parity": owner_parity,
            "instance_registry": registry,
            "native_extension": module_identity,
        }
        manifest["identity"] = canonical_digest(manifest)
        atomic_write_json(self.scene_root / "manifest.json", manifest)
        self._finalized = True
        self._write_state()
        return manifest
