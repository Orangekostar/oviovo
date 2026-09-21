"""Immutable study metadata and strict receipt serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

EXPECTED_PHASES = (
    "bind",
    "capture",
    "semantic",
    "geometry",
    "query",
    "select",
    "confirm",
    "report",
    "all",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ReceiptStatus(str, Enum):
    """Terminal and resumable phase states written by the orchestrator."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    REUSED = "REUSED"
    FAILED = "FAILED"
    BLOCKED_PREREQUISITE = "BLOCKED_PREREQUISITE"
    BLOCKED_ASSET_IDENTITY = "BLOCKED_ASSET_IDENTITY"
    BLOCKED_INDEPENDENT_SCENES = "BLOCKED_INDEPENDENT_SCENES"
    NOT_REQUIRED_BY_FROZEN_GATE = "NOT_REQUIRED_BY_FROZEN_GATE"


def _json_value(value: Any, location: str = "$") -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value, location)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{location} contains unsupported JSON type {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON and reject non-finite values."""

    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Atomically replace one JSON artifact after a durable file flush."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StudySpec:
    """Validated immutable view of the authoritative v1 protocol constants."""

    source_path: Path
    schema_version: int
    study: str
    version: str
    specification_only: bool
    project_commit: str
    official_commit: str
    task_branch: str
    output_root: Path
    phases: tuple[str, ...]
    values: Mapping[str, Any] = field(repr=False)
    digest: str

    @classmethod
    def load(cls, path: Path | str) -> StudySpec:
        source_path = Path(path).resolve()
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load study spec {source_path}: {error}") from error
        if not isinstance(raw, dict):
            raise TypeError("study spec must be a JSON object")
        required = {
            "schema_version",
            "study",
            "version",
            "specification_only",
            "project_commit",
            "official_commit",
            "task_branch",
            "output_root",
            "phases",
            "data",
            "models",
            "runtime",
            "selection",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"study spec missing required keys: {missing}")
        if raw["schema_version"] != 1 or raw["study"] != "ovimap-module-validation-v1":
            raise ValueError("unsupported study schema or identity")
        phases = tuple(raw["phases"])
        if phases != EXPECTED_PHASES:
            raise ValueError(f"phases must equal {EXPECTED_PHASES}")
        for name in ("project_commit", "official_commit"):
            if not isinstance(raw[name], str) or not _GIT_SHA_RE.fullmatch(raw[name]):
                raise ValueError(f"{name} must be a full lowercase Git SHA")
        if raw["specification_only"] is not True:
            raise ValueError("specification_only must remain true until runtime binding")
        output_root = Path(raw["output_root"])
        if not output_root.is_absolute():
            raise ValueError("output_root must be absolute")
        _json_value(raw)
        return cls(
            source_path=source_path,
            schema_version=raw["schema_version"],
            study=raw["study"],
            version=raw["version"],
            specification_only=raw["specification_only"],
            project_commit=raw["project_commit"],
            official_commit=raw["official_commit"],
            task_branch=raw["task_branch"],
            output_root=output_root,
            phases=phases,
            values=_freeze_json(raw),
            digest=canonical_digest(raw),
        )


@dataclass(frozen=True)
class PhaseReceipt:
    """Strict, JSON-safe phase completion record."""

    phase: str
    status: ReceiptStatus
    cache_key: str
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    reused_from: str | None = None
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.phase not in EXPECTED_PHASES[:-1]:
            raise ValueError(f"unsupported leaf phase: {self.phase}")
        try:
            status = ReceiptStatus(self.status)
        except ValueError as error:
            raise ValueError(f"unsupported receipt status: {self.status}") from error
        if not _SHA256_RE.fullmatch(self.cache_key):
            raise ValueError("cache_key must be a lowercase SHA-256 digest")
        outputs = _freeze_json(_json_value(self.outputs, "$.outputs"))
        metrics = _freeze_json(_json_value(self.metrics, "$.metrics"))
        blockers = tuple(str(item) for item in self.blockers)
        if status.value.startswith("BLOCKED_") and not blockers:
            blockers = (status.value,)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "blockers", blockers)
        _json_value(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "status": self.status.value,
            "cache_key": self.cache_key,
            "outputs": dict(self.outputs),
            "metrics": dict(self.metrics),
            "blockers": list(self.blockers),
            "reused_from": self.reused_from,
            "created_utc": self.created_utc,
        }

    def write(self, path: Path | str) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> PhaseReceipt:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            phase=raw["phase"],
            status=ReceiptStatus(raw["status"]),
            cache_key=raw["cache_key"],
            outputs=raw.get("outputs", {}),
            metrics=raw.get("metrics", {}),
            blockers=tuple(raw.get("blockers", ())),
            reused_from=raw.get("reused_from"),
            created_utc=raw["created_utc"],
            schema_version=raw.get("schema_version", 1),
        )
