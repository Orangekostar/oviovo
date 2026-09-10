"""Atomic, hash-bound serialization for entity-epoch state sidecars."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from src.oviv2.current_surface import CurrentSurfaceView
from src.oviv2.fine_current_composer import (
    EntityEpochComposition,
    EntityEpochStateSidecar,
)

_ARTIFACT_ID = "CROVE_ENTITY_EPOCH_STATE_V1"
_ARRAY_FIELDS = tuple(
    field.name
    for field in fields(EntityEpochStateSidecar)
    if field.name != "surface_id"
)
_NPZ_KEYS = frozenset((*_ARRAY_FIELDS, "surface_id", "canonical_surface_sha256"))


@dataclass(frozen=True, slots=True)
class EntityEpochArtifact:
    output_dir: Path
    manifest: Path


def _canonical_json(value: object) -> bytes:
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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _surface_sha256(surface: CurrentSurfaceView) -> str:
    digest = hashlib.sha256(b"CROVE_CURRENT_SURFACE_V1\0")
    surface_id = surface.surface_id.encode("utf-8")
    digest.update(len(surface_id).to_bytes(8, "little"))
    digest.update(surface_id)
    for field in fields(surface):
        if field.name == "surface_id":
            continue
        array = np.ascontiguousarray(getattr(surface, field.name))
        name = field.name.encode("ascii")
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(len(dtype).to_bytes(4, "little"))
        digest.update(dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic no-clobber directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(number, os.strerror(number), target)


def write_entity_epoch_composition(
    composition: EntityEpochComposition,
    output_dir: str | Path,
) -> EntityEpochArtifact:
    """Atomically publish one sidecar bound to its canonical surface."""

    if not isinstance(composition, EntityEpochComposition):
        raise TypeError("composition must be EntityEpochComposition")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        surface_sha256 = _surface_sha256(composition.surface)
        arrays_path = staging / "entity_epoch_state.npz"
        with arrays_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                surface_id=np.asarray(composition.surface.surface_id),
                canonical_surface_sha256=np.asarray(surface_sha256),
                **{
                    name: getattr(composition.state_sidecar, name)
                    for name in _ARRAY_FIELDS
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        arrays_content = arrays_path.read_bytes()
        manifest = {
            "schema_version": 1,
            "artifact_id": _ARTIFACT_ID,
            "status": "PASS",
            "surface_id": composition.surface.surface_id,
            "canonical_vertex_count": len(composition.surface.vertices_xyz),
            "canonical_surface_sha256": surface_sha256,
            "state_arrays": {
                "path": arrays_path.name,
                "sha256": _sha256_bytes(arrays_content),
                "byte_count": len(arrays_content),
            },
        }
        manifest_path = staging / "entity_epoch_state_manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _publish_directory_no_replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return EntityEpochArtifact(
        output_dir=output,
        manifest=output / "entity_epoch_state_manifest.json",
    )


def _manifest(root: Path) -> dict[str, object]:
    path = root / "entity_epoch_state_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("entity epoch manifest cannot be decoded") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "artifact_id",
        "status",
        "surface_id",
        "canonical_vertex_count",
        "canonical_surface_sha256",
        "state_arrays",
    }:
        raise ValueError("entity epoch manifest schema is invalid")
    if (
        payload["schema_version"] != 1
        or payload["artifact_id"] != _ARTIFACT_ID
        or payload["status"] != "PASS"
    ):
        raise ValueError("entity epoch manifest identity is invalid")
    return payload


def load_entity_epoch_composition(
    output_dir: str | Path,
    *,
    surface: CurrentSurfaceView,
) -> EntityEpochComposition:
    """Load a sidecar only after verifying its NPZ and canonical surface."""

    if not isinstance(surface, CurrentSurfaceView):
        raise TypeError("surface must be CurrentSurfaceView")
    root = Path(os.path.abspath(os.fspath(output_dir)))
    payload = _manifest(root)
    surface_sha256 = _surface_sha256(surface)
    if (
        payload["surface_id"] != surface.surface_id
        or payload["canonical_vertex_count"] != len(surface.vertices_xyz)
        or payload["canonical_surface_sha256"] != surface_sha256
    ):
        raise ValueError("entity epoch canonical surface binding mismatch")
    binding = payload["state_arrays"]
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError("entity epoch state array binding is invalid")
    relative = binding.get("path")
    if relative != "entity_epoch_state.npz":
        raise ValueError("entity epoch state array path is invalid")
    try:
        content = (root / relative).read_bytes()
    except OSError as error:
        raise ValueError("entity epoch state arrays are unavailable") from error
    if binding.get("byte_count") != len(content) or binding.get(
        "sha256"
    ) != _sha256_bytes(content):
        raise ValueError("entity epoch state array binding mismatch")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_KEYS:
                raise ValueError("entity epoch state array schema is invalid")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("entity epoch"):
            raise
        raise ValueError("entity epoch state arrays cannot be decoded") from error
    recorded_surface_id = arrays.pop("surface_id")
    recorded_surface_sha256 = arrays.pop("canonical_surface_sha256")
    if (
        recorded_surface_id.shape != ()
        or str(recorded_surface_id.item()) != surface.surface_id
        or recorded_surface_sha256.shape != ()
        or str(recorded_surface_sha256.item()) != surface_sha256
    ):
        raise ValueError("entity epoch state arrays differ from the canonical surface")
    sidecar = EntityEpochStateSidecar(
        surface_id=surface.surface_id,
        **arrays,
    )
    if len(sidecar.source_vertex_indices) != len(surface.vertices_xyz):
        raise ValueError("entity epoch sidecar length differs from canonical surface")
    return EntityEpochComposition(surface=surface, state_sidecar=sidecar)


__all__ = [
    "EntityEpochArtifact",
    "load_entity_epoch_composition",
    "write_entity_epoch_composition",
]
