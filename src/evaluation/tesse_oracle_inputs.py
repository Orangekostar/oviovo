"""Fail-closed manifests for TESSE semantic-oracle diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from src.evaluation.json_contracts import loads_strict


class OracleInputError(ValueError):
    """Raised when oracle streams cannot be source-bound and aligned."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_file(path: str | Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        current = Path(absolute.anchor)
        metadata = current.lstat()
        for component in absolute.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise OracleInputError(
                    f"oracle source is missing or indirect: {absolute.name}"
                )
    except FileNotFoundError as exc:
        raise OracleInputError(
            f"oracle source is missing or indirect: {absolute.name}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise OracleInputError(
            f"oracle source is missing or indirect: {absolute.name}"
        )
    return absolute


def _binding(path: Path) -> dict[str, object]:
    path = _direct_file(path)
    return {"sha256": _sha256(path), "byte_count": path.stat().st_size}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_descriptor(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, object]]:
    if isinstance(source, Mapping):
        return dict(source), {"kind": "IN_MEMORY_TEST_DESCRIPTOR"}
    path = _direct_file(source)
    try:
        payload = loads_strict(path.read_text(encoding="utf-8"), label="oracle descriptor")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OracleInputError("oracle descriptor is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise OracleInputError("oracle descriptor must be a mapping")
    return dict(payload), _binding(path)


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise OracleInputError(f"{label} must be a non-negative integer")
    return value


def build_oracle_input_manifest(
    source: str | Path | Mapping[str, Any], *, output: str | Path
) -> dict[str, Any]:
    """Validate aligned RGB-D/semantic bag streams and emit a non-ranking manifest."""

    descriptor, descriptor_binding = _load_descriptor(source)
    if descriptor.get("schema_version") != 1:
        raise OracleInputError("unsupported oracle descriptor schema")
    scene = descriptor.get("scene")
    topics = descriptor.get("topics")
    database_record = descriptor.get("source_database")
    encoding = descriptor.get("semantic_encoding")
    tolerance = _integer(
        descriptor.get("max_timestamp_delta_ns"), label="max_timestamp_delta_ns"
    )
    if not isinstance(scene, str) or not scene:
        raise OracleInputError("oracle scene is missing")
    if not isinstance(topics, Mapping) or set(topics) != {"rgb", "depth", "semantics"}:
        raise OracleInputError("oracle topics must define rgb, depth, and semantics")
    if not isinstance(database_record, Mapping):
        raise OracleInputError("source database binding is missing")
    if encoding not in {"8UC1", "16UC1"}:
        raise OracleInputError("semantic encoding must be 8UC1 or 16UC1")
    database_path = database_record.get("path")
    if not isinstance(database_path, str) or not database_path:
        raise OracleInputError("source database path is missing")
    database = _direct_file(database_path)
    database_binding = _binding(database)
    if (
        database_binding["sha256"] != database_record.get("sha256")
        or database_binding["byte_count"] != database_record.get("byte_count")
    ):
        raise OracleInputError("source database binding mismatch")
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        topic_rows = connection.execute(
            "select id, name, type, serialization_format from topics"
        ).fetchall()
        inventory = {str(name): (int(topic_id), str(msg_type), str(serialization)) for topic_id, name, msg_type, serialization in topic_rows}
        streams: dict[str, dict[str, object]] = {}
        timestamp_sets: dict[str, tuple[int, ...]] = {}
        for role in ("rgb", "depth", "semantics"):
            topic_name = topics[role]
            if not isinstance(topic_name, str) or topic_name not in inventory:
                raise OracleInputError(f"oracle {role} topic is missing")
            topic_id, msg_type, serialization = inventory[topic_name]
            if msg_type != "sensor_msgs/msg/Image" or serialization != "cdr":
                raise OracleInputError(f"oracle {role} topic has an invalid ROS type")
            messages = connection.execute(
                "select timestamp, data from messages where topic_id=? order by timestamp, id",
                (topic_id,),
            )
            timestamps: list[int] = []
            digest = hashlib.sha256()
            for timestamp, data in messages:
                stamp = _integer(timestamp, label=f"{role} timestamp")
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    raise OracleInputError(
                        f"oracle {role} stream contains an invalid frame"
                    )
                payload = bytes(data)
                if not payload:
                    raise OracleInputError(f"oracle {role} stream contains an empty frame")
                timestamps.append(stamp)
                digest.update(stamp.to_bytes(8, "big"))
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(hashlib.sha256(payload).digest())
            if not timestamps:
                raise OracleInputError(f"oracle {role} stream is empty")
            if any(right <= left for left, right in pairwise(timestamps)):
                raise OracleInputError(
                    f"oracle {role} timestamps must be strictly increasing"
                )
            timestamp_sets[role] = tuple(timestamps)
            streams[role] = {
                "topic": topic_name,
                "message_type": msg_type,
                "serialization_format": serialization,
                "frame_count": len(timestamps),
                "stream_sha256": digest.hexdigest(),
                "first_timestamp_ns": timestamps[0],
                "last_timestamp_ns": timestamps[-1],
            }
    except sqlite3.Error as exc:
        raise OracleInputError("source database schema is invalid") from exc
    finally:
        connection.close()
    counts = {len(values) for values in timestamp_sets.values()}
    if len(counts) != 1:
        raise OracleInputError("RGB-D/semantic frame counts disagree")
    for stamps in zip(
        timestamp_sets["rgb"],
        timestamp_sets["depth"],
        timestamp_sets["semantics"],
    ):
        if max(stamps) - min(stamps) > tolerance:
            raise OracleInputError("RGB-D/semantic timestamp alignment failed")
    frame_count = counts.pop()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "condition": "ORACLE_DIAGNOSTIC",
        "condition_id": "C1_CROVE_GT_SEMANTICS",
        "ranking_eligible": False,
        "scene": scene,
        "semantic_encoding": encoding,
        "frame_count": frame_count,
        "max_timestamp_delta_ns": tolerance,
        "source_bindings": {
            "descriptor": descriptor_binding,
            "database": database_binding,
        },
        "streams": streams,
    }
    _atomic_json(Path(output), manifest)
    return manifest
