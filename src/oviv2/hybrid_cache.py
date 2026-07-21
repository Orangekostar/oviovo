from __future__ import annotations

from collections.abc import Callable, Mapping
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
import tempfile
from numbers import Integral, Real
from typing import Any, BinaryIO

import numpy as np

from .hybrid_frontend import FrontendBatch


_PAYLOAD_KEYS = frozenset(
    {"mask", "xyxy", "confidence", "class_id", "classes", "image_feats"}
)
_SHA256_LENGTH = 64
_MANIFEST_NAME = "frontend_manifest.json"


class _RestrictedUnpickler(pickle.Unpickler):
    """Permit only NumPy constructors needed by protocol-4 numeric ndarrays."""

    _ALLOWED_GLOBALS = frozenset(
        {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"unsafe pickle global is not permitted: {module}.{name}"
            )
        return super().find_class(module, name)


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            return True
    return False


def _assert_no_symlink(path: Path, *, field_name: str) -> None:
    if _path_has_symlink_component(path):
        raise ValueError(f"{field_name} must not be a symlink or reside below one")


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _assert_regular_input(path: Path, *, field_name: str) -> None:
    _assert_no_symlink(path, field_name=field_name)
    metadata = _lstat(path)
    if metadata is None:
        raise FileNotFoundError(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} must be a regular file")


def _open_regular_input(path: Path, *, field_name: str) -> BinaryIO:
    _assert_regular_input(path, field_name=field_name)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open {field_name}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field_name} must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a lowercase sha256 string")
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 string")
    return value


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return normalized


def _positive_integer(value: object, field_name: str) -> int:
    normalized = _nonnegative_integer(value, field_name)
    if normalized == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


def _classes(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("classes must be a tuple or list of non-empty strings")
    normalized = tuple(_nonempty_string(item, "classes item") for item in value)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("classes must contain unique non-empty strings")
    return normalized


def _numeric_ndarray(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a NumPy array")
    if value.dtype.kind not in "iufb":
        raise ValueError(f"{field_name} has an unsafe dtype")
    return value


def _integer_ndarray(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a NumPy array")
    if value.dtype.kind not in "iu":
        raise ValueError(f"{field_name} must use an integer dtype")
    return value


def _validate_payload(payload: object) -> FrontendBatch:
    if not isinstance(payload, dict):
        raise ValueError("frontend cache payload must be a dictionary")
    keys = set(payload)
    if keys != _PAYLOAD_KEYS:
        raise ValueError("frontend cache payload has invalid keys")

    masks = _numeric_ndarray(payload["mask"], "mask")
    boxes = _numeric_ndarray(payload["xyxy"], "xyxy")
    confidences = _numeric_ndarray(payload["confidence"], "confidence")
    class_ids = _integer_ndarray(payload["class_id"], "class_id")
    features = _numeric_ndarray(payload["image_feats"], "image_feats")
    classes = _classes(payload["classes"])

    if masks.ndim != 3:
        raise ValueError("mask must have shape (N, H, W)")
    count = masks.shape[0]
    if boxes.shape != (count, 4):
        raise ValueError("xyxy must have shape (N, 4) matching mask")
    if confidences.shape != (count,):
        raise ValueError("confidence must have shape (N,) matching mask")
    if class_ids.shape != (count,):
        raise ValueError("class_id must have shape (N,) matching mask")
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError("image_feats must have shape (N, D) with D > 0")
    if count and (np.any(class_ids < 0) or np.any(class_ids >= len(classes))):
        raise ValueError("class_id lies outside classes")

    labels = tuple(classes[int(index)] for index in class_ids)
    try:
        return FrontendBatch(
            masks=masks,
            boxes_xyxy=boxes,
            confidences=confidences,
            labels=labels,
            image_features=features,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid frontend cache payload: {exc}") from exc


def sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one regular, non-symlink file."""

    target = Path(path)
    digest = hashlib.sha256()
    with _open_regular_input(target, field_name="sha256 input") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_frontend_batch(
    path: str | Path,
    expected_sha256: str | None = None,
) -> FrontendBatch:
    """Load one validated cache frame into the immutable frontend contract."""

    target = Path(path)
    _assert_regular_input(target, field_name="frontend cache input")
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, "expected_sha256")
        if sha256(target) != expected:
            raise ValueError("frontend cache checksum mismatch")

    try:
        with _open_regular_input(target, field_name="frontend cache input") as stream:
            with gzip.GzipFile(fileobj=stream, mode="rb") as compressed:
                payload = _RestrictedUnpickler(compressed).load()
                if compressed.read(1):
                    raise ValueError("frontend cache payload has trailing data")
    except ValueError:
        raise
    except (
        EOFError,
        OSError,
        pickle.PickleError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
    ) as exc:
        raise ValueError("invalid frontend cache gzip or pickle payload") from exc
    return _validate_payload(payload)


def _prepare_destination(path: Path, *, field_name: str) -> Path:
    parent = path.parent
    _assert_no_symlink(parent, field_name=f"{field_name} parent")
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(parent, field_name=f"{field_name} parent")
    if not parent.is_dir():
        raise ValueError(f"{field_name} parent must be a directory")
    if _lstat(path) is not None:
        raise FileExistsError(path)
    _assert_no_symlink(path, field_name=field_name)
    return parent


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_published_destination(
    destination: Path,
    temporary: Path,
    parent: Path,
) -> None:
    temporary_metadata = _lstat(temporary)
    destination_metadata = _lstat(destination)
    if (
        temporary_metadata is None
        or destination_metadata is None
        or not stat.S_ISREG(temporary_metadata.st_mode)
        or not stat.S_ISREG(destination_metadata.st_mode)
        or temporary_metadata.st_dev != destination_metadata.st_dev
        or temporary_metadata.st_ino != destination_metadata.st_ino
    ):
        raise OSError("published destination cannot be safely rolled back")
    destination.unlink()
    _fsync_directory(parent)


def _clean_failed_publication(
    destination: Path,
    temporary: Path,
    parent: Path,
    *,
    published: bool,
    field_name: str,
) -> None:
    rollback_error: OSError | None = None
    cleanup_error: OSError | None = None
    if published:
        try:
            _remove_published_destination(destination, temporary, parent)
        except OSError as exc:
            rollback_error = exc
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        cleanup_error = exc

    if rollback_error is not None:
        message = f"failed to roll back published {field_name} file"
        if cleanup_error is not None:
            message += f"; temporary {field_name} cleanup also failed"
        raise OSError(message) from rollback_error
    if cleanup_error is not None:
        raise OSError(
            f"failed to clean temporary {field_name} file"
        ) from cleanup_error


def _publish_bytes(
    path: Path,
    data_writer: Callable[[BinaryIO], None],
    *,
    field_name: str,
) -> None:
    parent = _prepare_destination(path, field_name=field_name)
    temporary_path: Path | None = None
    published = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w+b") as stream:
            data_writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        published = True
        _fsync_directory(parent)
    except BaseException:
        if temporary_path is not None:
            _clean_failed_publication(
                path,
                temporary_path,
                parent,
                published=published,
                field_name=field_name,
            )
        raise
    else:
        if temporary_path is None:
            raise RuntimeError(f"failed to allocate temporary {field_name} file")
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            _clean_failed_publication(
                path,
                temporary_path,
                parent,
                published=True,
                field_name=field_name,
            )
            raise OSError(
                f"failed to clean temporary {field_name} file"
            ) from cleanup_error


def write_hybrid_frame(
    path: str | Path,
    batch: FrontendBatch,
    classes: tuple[str, ...] | list[str],
) -> str:
    """Atomically publish one deterministic, adapter-compatible frontend frame."""

    if not isinstance(batch, FrontendBatch):
        raise ValueError("batch must be a FrontendBatch")
    normalized_classes = _classes(classes)
    class_indices = {label: index for index, label in enumerate(normalized_classes)}
    try:
        class_ids = np.asarray(
            [class_indices[label] for label in batch.labels], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError("classes must cover every batch label") from exc

    payload = {
        "mask": np.asarray(batch.masks, dtype=np.bool_),
        "xyxy": np.asarray(batch.boxes_xyxy, dtype=np.float32),
        "confidence": np.asarray(batch.confidences, dtype=np.float32),
        "class_id": class_ids,
        "classes": list(normalized_classes),
        "image_feats": np.asarray(batch.image_features, dtype=np.float32),
    }
    _validate_payload(payload)

    def write_payload(stream: BinaryIO) -> None:
        with gzip.GzipFile(
            fileobj=stream, mode="wb", filename="", mtime=0
        ) as compressed:
            pickle.dump(payload, compressed, protocol=4, fix_imports=False)

    destination = Path(path)
    _publish_bytes(destination, write_payload, field_name="frontend cache output")
    return sha256(destination)


def _source_frame_ids(value: object, frame_count: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != frame_count:
        raise ValueError("source_frame_ids must match frame_count")
    normalized = tuple(
        _nonnegative_integer(item, "source_frame_ids item") for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("source_frame_ids must be unique")
    return normalized


def _string_sha256_mapping(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_hash in value.items():
        key = _nonempty_string(raw_key, f"{field_name} key")
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate keys")
        normalized[key] = _require_sha256(raw_hash, f"{field_name}[{key!r}]")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return dict(sorted(normalized.items()))


def _json_value(value: object, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError(f"{field_name} must contain only finite JSON values")
        return normalized
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _nonempty_string(raw_key, f"{field_name} key")
            if key in normalized_mapping:
                raise ValueError(f"{field_name} contains duplicate keys")
            normalized_mapping[key] = _json_value(raw_value, f"{field_name}[{key!r}]")
        return dict(sorted(normalized_mapping.items()))
    if isinstance(value, (tuple, list)):
        return [_json_value(item, field_name) for item in value]
    raise ValueError(f"{field_name} must contain only JSON-compatible values")


def _policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("policy must be a mapping")
    return _json_value(value, "policy")


def _diagnostics(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostics must be a mapping")
    normalized: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = _nonempty_string(raw_key, "diagnostics key")
        if key in normalized:
            raise ValueError("diagnostics contains duplicate keys")
        normalized[key] = _nonnegative_integer(raw_value, f"diagnostics[{key!r}]")
    return dict(sorted(normalized.items()))


def _cache_hashes(
    output_dir: Path,
    frame_count: int,
    value: object,
) -> dict[str, str]:
    supplied = _string_sha256_mapping(value, "cache_files_sha256")
    expected_names = tuple(
        f"frame{index:06d}.pkl.gz" for index in range(frame_count)
    )
    if tuple(supplied) != expected_names:
        raise ValueError("cache_files_sha256 must exactly cover frame cache files")
    verified: dict[str, str] = {}
    for name in expected_names:
        frame = output_dir / name
        _assert_regular_input(frame, field_name=f"cache frame {name}")
        actual = sha256(frame)
        if actual != supplied[name]:
            raise ValueError(f"cache_files_sha256 checksum mismatch for {name}")
        verified[name] = actual
    return verified


def publish_frontend_manifest(
    output_dir: str | Path,
    scene: str,
    frame_count: int,
    source_frame_ids: tuple[int, ...] | list[int],
    classes: tuple[str, ...] | list[str],
    feature_model_id: str,
    cache_files_sha256: Mapping[str, str],
    source_sha256: Mapping[str, str],
    policy: Mapping[str, object],
    diagnostics: Mapping[str, int],
) -> dict[str, Any]:
    """Verify frame hashes and atomically publish the deterministic cache manifest."""

    directory = Path(output_dir)
    _assert_no_symlink(directory, field_name="manifest output_dir")
    directory.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(directory, field_name="manifest output_dir")
    if not directory.is_dir():
        raise ValueError("manifest output_dir must be a directory")

    normalized_scene = _nonempty_string(scene, "scene")
    normalized_frame_count = _positive_integer(frame_count, "frame_count")
    normalized_source_ids = _source_frame_ids(source_frame_ids, normalized_frame_count)
    normalized_classes = _classes(classes)
    normalized_feature_model_id = _nonempty_string(feature_model_id, "feature_model_id")
    normalized_cache_hashes = _cache_hashes(
        directory, normalized_frame_count, cache_files_sha256
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "OVIV2",
        "scene": normalized_scene,
        "frame_count": normalized_frame_count,
        "source_frame_ids": list(normalized_source_ids),
        "classes": list(normalized_classes),
        "feature_model_id": normalized_feature_model_id,
        "cache_files_sha256": normalized_cache_hashes,
        "source_sha256": _string_sha256_mapping(source_sha256, "source_sha256"),
        "policy": _policy(policy),
        "diagnostics": _diagnostics(diagnostics),
    }
    serialized = json.dumps(
        manifest,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8") + b"\n"

    def write_manifest(stream: BinaryIO) -> None:
        stream.write(serialized)

    _publish_bytes(
        directory / _MANIFEST_NAME,
        write_manifest,
        field_name="manifest output",
    )
    return manifest
