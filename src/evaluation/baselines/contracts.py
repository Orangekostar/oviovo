"""Baseline-specific disclosures layered on the neutral evaluation contract."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from src.evaluation.contracts import MapSnapshot

BaselineMode = Literal["native", "composed", "frozen", "offline", "oracle"]
EntityIdStability = Literal["artifact-local", "persistent-within-run"]


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class RuntimeBreakdown:
    frame_count: int
    initialization_s: float = 0.0
    frontend_s: float = 0.0
    backend_s: float = 0.0
    maintenance_s: float = 0.0
    finalization_s: float = 0.0
    evaluation_io_s: float = 0.0
    query_latencies_ms: tuple[float, ...] = ()
    peak_gpu_gb: float = 0.0
    peak_ram_gb: float = 0.0
    final_map_mb: float = 0.0
    source_labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame_count = int(self.frame_count)
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        object.__setattr__(self, "frame_count", frame_count)
        for name in (
            "initialization_s",
            "frontend_s",
            "backend_s",
            "maintenance_s",
            "finalization_s",
            "evaluation_io_s",
            "peak_gpu_gb",
            "peak_ram_gb",
            "final_map_mb",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        latencies = tuple(
            _finite_nonnegative(value, "query_latencies_ms") for value in self.query_latencies_ms
        )
        object.__setattr__(self, "query_latencies_ms", latencies)
        labels = {str(key): str(value) for key, value in self.source_labels.items()}
        for label in labels.values():
            if re.search(r"\bgt\b|ground[\s_-]*truth", label, flags=re.IGNORECASE):
                raise ValueError("runtime source labels must not reference ground truth")
        object.__setattr__(self, "source_labels", labels)

    @property
    def total_s_per_frame(self) -> float:
        return (self.frontend_s + self.backend_s + self.maintenance_s) / self.frame_count

    @property
    def processed_hz(self) -> float:
        return 0.0 if self.total_s_per_frame == 0.0 else 1.0 / self.total_s_per_frame

    def to_snapshot_runtime(self) -> dict[str, float]:
        return {
            "total_s_per_frame": self.total_s_per_frame,
            "processed_hz": self.processed_hz,
            "peak_gpu_gb": self.peak_gpu_gb,
            "peak_ram_gb": self.peak_ram_gb,
            "final_map_mb": self.final_map_mb,
        }

    def to_json(self) -> dict[str, Any]:
        query_mean = (
            sum(self.query_latencies_ms) / len(self.query_latencies_ms)
            if self.query_latencies_ms
            else 0.0
        )
        metrics: dict[str, Any] = {
            "frame_count": self.frame_count,
            "initialization_s": self.initialization_s,
            "frontend_s": self.frontend_s,
            "backend_s": self.backend_s,
            "maintenance_s": self.maintenance_s,
            "finalization_s": self.finalization_s,
            "evaluation_io_s": self.evaluation_io_s,
            "total_s_per_frame": self.total_s_per_frame,
            "processed_hz": self.processed_hz,
            "query_mean_ms": query_mean,
            "query_p95_ms": _percentile(self.query_latencies_ms, 0.95),
            "peak_gpu_gb": self.peak_gpu_gb,
            "peak_ram_gb": self.peak_ram_gb,
            "final_map_mb": self.final_map_mb,
        }
        units = {
            "frame_count": "frames",
            "initialization_s": "seconds",
            "frontend_s": "seconds",
            "backend_s": "seconds",
            "maintenance_s": "seconds",
            "finalization_s": "seconds",
            "evaluation_io_s": "seconds",
            "total_s_per_frame": "seconds/frame",
            "processed_hz": "Hz",
            "query_mean_ms": "milliseconds",
            "query_p95_ms": "milliseconds",
            "peak_gpu_gb": "GB",
            "peak_ram_gb": "GB",
            "final_map_mb": "MB",
        }
        return {"metrics": metrics, "units": units, "source_labels": dict(self.source_labels)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RuntimeBreakdown":
        metrics = payload.get("metrics", {})
        return cls(
            frame_count=int(metrics["frame_count"]),
            initialization_s=float(metrics.get("initialization_s", 0.0)),
            frontend_s=float(metrics.get("frontend_s", 0.0)),
            backend_s=float(metrics.get("backend_s", 0.0)),
            maintenance_s=float(metrics.get("maintenance_s", 0.0)),
            finalization_s=float(metrics.get("finalization_s", 0.0)),
            evaluation_io_s=float(metrics.get("evaluation_io_s", 0.0)),
            query_latencies_ms=(),
            peak_gpu_gb=float(metrics.get("peak_gpu_gb", 0.0)),
            peak_ram_gb=float(metrics.get("peak_ram_gb", 0.0)),
            final_map_mb=float(metrics.get("final_map_mb", 0.0)),
            source_labels=dict(payload.get("source_labels", {})),
        )


@dataclass(frozen=True)
class BaselineMetadata:
    method_key: str
    display_label: str
    mode: BaselineMode
    upstream_commit: str
    entity_id_stability: EntityIdStability
    semantic_label_source: str = "method_output"
    snapshot_scope: str = "current"
    protocol_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.method_key, self.display_label, self.upstream_commit)):
            raise ValueError("method key, label, and upstream commit must be non-empty")
        if self.mode not in {"native", "composed", "frozen", "offline", "oracle"}:
            raise ValueError(f"invalid baseline mode: {self.mode}")
        if self.entity_id_stability not in {"artifact-local", "persistent-within-run"}:
            raise ValueError(f"invalid entity ID stability: {self.entity_id_stability}")
        label = self.display_label.lower()
        if self.mode == "composed" and " + " not in self.display_label and "composed" not in label:
            raise ValueError("composed baseline label must disclose the added component")
        if self.mode == "offline" and "offline" not in label:
            raise ValueError("offline baseline label must disclose offline execution")
        if self.mode == "frozen" and "frozen" not in label:
            raise ValueError("frozen baseline label must disclose the intervention")
        if self.mode == "oracle" and "oracle" not in label:
            raise ValueError("oracle baseline label must disclose oracle semantics")
        if self.snapshot_scope not in {"current", "history"}:
            raise ValueError("snapshot_scope must be current or history")
        object.__setattr__(self, "protocol_notes", tuple(str(note) for note in self.protocol_notes))

    @property
    def eligible_for_ranking(self) -> bool:
        return self.mode != "oracle"

    def to_json(self) -> dict[str, Any]:
        return {
            "method_key": self.method_key,
            "display_label": self.display_label,
            "mode": self.mode,
            "upstream_commit": self.upstream_commit,
            "entity_id_stability": self.entity_id_stability,
            "semantic_label_source": self.semantic_label_source,
            "snapshot_scope": self.snapshot_scope,
            "eligible_for_ranking": self.eligible_for_ranking,
            "protocol_notes": list(self.protocol_notes),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "BaselineMetadata":
        return cls(
            method_key=str(payload["method_key"]),
            display_label=str(payload["display_label"]),
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            upstream_commit=str(payload["upstream_commit"]),
            entity_id_stability=str(payload["entity_id_stability"]),  # type: ignore[arg-type]
            semantic_label_source=str(payload.get("semantic_label_source", "method_output")),
            snapshot_scope=str(payload.get("snapshot_scope", "current")),
            protocol_notes=tuple(payload.get("protocol_notes", ())),
        )


@dataclass(frozen=True)
class BaselineArtifact:
    snapshot: MapSnapshot
    runtime: RuntimeBreakdown
    metadata: BaselineMetadata

    def __post_init__(self) -> None:
        if self.snapshot.method != self.metadata.display_label:
            raise ValueError("snapshot method must match the disclosed baseline label")
        if self.snapshot.scope != self.metadata.snapshot_scope:
            raise ValueError("snapshot scope must match baseline metadata")

    @classmethod
    def from_json_files(
        cls,
        snapshot_path: str | Path,
        entities_path: str | Path,
        runtime_path: str | Path,
        metadata_path: str | Path,
    ) -> "BaselineArtifact":
        import json

        from src.evaluation.exporters.oviovo import read_map_snapshot

        snapshot = read_map_snapshot(snapshot_path, entities_path)
        runtime = RuntimeBreakdown.from_json(
            json.loads(Path(runtime_path).read_text(encoding="utf-8"))
        )
        metadata = BaselineMetadata.from_json(
            json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        )
        return cls(snapshot=snapshot, runtime=runtime, metadata=metadata)


@dataclass
class FrozenUpdateAudit:
    update_timestamps: list[float] = field(default_factory=list)
    query_timestamps: list[float] = field(default_factory=list)
    freeze_timestamp: float | None = None

    def freeze(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("freeze timestamp must be finite")
        if self.freeze_timestamp is not None:
            raise RuntimeError("baseline is already frozen")
        self.freeze_timestamp = timestamp

    def record_update(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if self.freeze_timestamp is not None and timestamp >= self.freeze_timestamp:
            raise RuntimeError("baseline is frozen; post-intervention updates are forbidden")
        self.update_timestamps.append(timestamp)

    def record_query(self, timestamp: float) -> None:
        self.query_timestamps.append(float(timestamp))

    def to_json(self) -> dict[str, Any]:
        updates_after_freeze = 0
        if self.freeze_timestamp is not None:
            updates_after_freeze = sum(
                timestamp >= self.freeze_timestamp for timestamp in self.update_timestamps
            )
        return {
            "freeze_timestamp": self.freeze_timestamp,
            "update_timestamps": list(self.update_timestamps),
            "query_timestamps": list(self.query_timestamps),
            "updates_after_freeze": updates_after_freeze,
        }


def evaluate_identity_assignments(
    assignments: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    previous: dict[str, str] = {}
    tracked: set[str] = set()
    assignment_count = 0
    switches = 0
    for frame in assignments:
        for ground_truth_id, predicted_id in frame.items():
            ground_truth_id = str(ground_truth_id)
            predicted_id = str(predicted_id)
            tracked.add(ground_truth_id)
            assignment_count += 1
            if ground_truth_id in previous and previous[ground_truth_id] != predicted_id:
                switches += 1
            previous[ground_truth_id] = predicted_id
    return {
        "id_switches": switches,
        "tracked_gt_entities": len(tracked),
        "assignment_count": assignment_count,
    }
