"""Leakage-safe prediction payloads and adapters for the released evaluator."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .assets import sha256_file
from .contracts import atomic_write_json, canonical_digest

_FORBIDDEN_METADATA_PARTS = (
    "gt",
    "ground_truth",
    "correct",
    "cause_ledger",
    "future",
)
_BRANCHES = {"N0", "S", "G", "Q", "COMBO"}
_METRIC_NAMES = (
    "uap",
    "ap50",
    "ap25",
    "miou",
    "macc",
    "canonical_ap50",
    "canonical_ap75",
)


def _readonly(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_identity(value: np.ndarray) -> Mapping[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _safe_metadata(value: Any, location: str = "metadata") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if any(part in lowered for part in _FORBIDDEN_METADATA_PARTS):
                raise ValueError(f"forbidden prediction metadata field: {location}.{name}")
            result[name] = _safe_metadata(item, f"{location}.{name}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_safe_metadata(item, f"{location}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(f"unsupported or non-finite prediction metadata at {location}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class GeometryIdentity:
    xyz_sha256: str
    faces_sha256: str
    tsdf_sha256: str
    projection_identity: str
    source_row_count: int

    def __post_init__(self) -> None:
        for name in ("xyz_sha256", "faces_sha256", "tsdf_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.projection_identity or self.source_row_count <= 0:
            raise ValueError("projection identity and positive source-row count are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyz_sha256": self.xyz_sha256,
            "faces_sha256": self.faces_sha256,
            "tsdf_sha256": self.tsdf_sha256,
            "projection_identity": self.projection_identity,
            "source_row_count": self.source_row_count,
        }


@dataclass
class PredictionPayload:
    method_id: str
    branch: str
    scene_id: str
    geometry: GeometryIdentity
    owner_ids: np.ndarray
    semantic_labels: np.ndarray
    instance_ranks: tuple[tuple[int, float], ...]
    logical_cost: Mapping[str, int | float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _locked: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("prediction payload is locked")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.locked:
            raise AttributeError("prediction payload is locked")
        object.__delattr__(self, name)

    def __post_init__(self) -> None:
        if not self.method_id or not self.scene_id or self.branch not in _BRANCHES:
            raise ValueError("prediction method, scene, and branch are invalid")
        owners = _readonly(self.owner_ids, np.int64)
        labels = _readonly(self.semantic_labels, np.int64)
        if owners.shape != (self.geometry.source_row_count,) or labels.shape != owners.shape:
            raise ValueError("prediction rows must exactly match source geometry")
        if np.any(owners < 0) or np.any(labels < 0):
            raise ValueError("prediction owner and semantic labels must be nonnegative")
        if np.any(labels[owners == 0] != 0):
            raise ValueError("unowned source rows must use semantic class 0")
        for owner in np.unique(owners):
            if owner == 0:
                continue
            if len(np.unique(labels[owners == owner])) != 1:
                raise ValueError("each predicted owner must have one semantic label")
        ranks = tuple((int(owner), float(rank)) for owner, rank in self.instance_ranks)
        rank_owners = tuple(owner for owner, _ in ranks)
        active_owners = tuple(sorted(int(owner) for owner in np.unique(owners) if owner > 0))
        if (
            tuple(sorted(rank_owners)) != active_owners
            or len(set(rank_owners)) != len(rank_owners)
            or not np.isfinite([rank for _, rank in ranks]).all()
        ):
            raise ValueError("instance ranks must uniquely cover every positive owner")
        costs: dict[str, int | float] = {}
        for key, value in self.logical_cost.items():
            numeric = float(value)
            if not str(key) or not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError("logical costs must be named finite nonnegative values")
            costs[str(key)] = value
        self.owner_ids = owners
        self.semantic_labels = labels
        self.instance_ranks = ranks
        self.logical_cost = MappingProxyType(costs)
        self.metadata = _safe_metadata(self.metadata)

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def prediction_key(self) -> str:
        return canonical_digest(
            {
                "schema": "ovimap-module-prediction-v1",
                "scene_id": self.scene_id,
                "geometry": self.geometry.to_dict(),
                "owner_ids": _array_identity(self.owner_ids),
                "semantic_labels": _array_identity(self.semantic_labels),
                "instance_ranks": self.instance_ranks,
            }
        )

    @property
    def record_key(self) -> str:
        return canonical_digest(
            {
                "prediction_key": self.prediction_key,
                "method_id": self.method_id,
                "branch": self.branch,
                "logical_cost": _plain(self.logical_cost),
                "metadata": _plain(self.metadata),
            }
        )

    def lock(self) -> str:
        if not self.locked:
            self.__post_init__()
        record_key = self.record_key
        object.__setattr__(self, "_locked", True)
        return record_key

    def manifest(self) -> dict[str, Any]:
        if not self.locked:
            raise ValueError("prediction payload must be locked before manifest export")
        return {
            "schema_version": 1,
            "artifact_type": "OVIMAP_MODULE_PREDICTION",
            "method_id": self.method_id,
            "branch": self.branch,
            "scene_id": self.scene_id,
            "geometry": self.geometry.to_dict(),
            "prediction_key": self.prediction_key,
            "record_key": self.record_key,
            "owner_ids": _array_identity(self.owner_ids),
            "semantic_labels": _array_identity(self.semantic_labels),
            "instance_ranks": [[owner, rank] for owner, rank in self.instance_ranks],
            "logical_cost": _plain(self.logical_cost),
            "metadata": _plain(self.metadata),
            "GT_input": False,
            "locked": True,
        }

    def write_manifest(self, path: Path | str) -> None:
        atomic_write_json(Path(path), self.manifest())


def validate_prediction_invariants(
    payload: PredictionPayload, native: PredictionPayload
) -> None:
    if payload.scene_id != native.scene_id or payload.geometry != native.geometry:
        raise ValueError("prediction source geometry differs from native")
    if payload.branch in {"S", "Q"}:
        if not np.array_equal(payload.owner_ids, native.owner_ids):
            raise ValueError("S/Q owner geometry must remain fixed")
        if payload.instance_ranks != native.instance_ranks:
            raise ValueError("S/Q instance ranks must remain fixed")
    if payload.branch == "G":
        if not np.array_equal(payload.owner_ids > 0, native.owner_ids > 0):
            raise ValueError("G partition must preserve native owned support")
        if len(payload.owner_ids) != native.geometry.source_row_count:
            raise ValueError("G partition is incomplete")
        active = {int(value) for value in np.unique(payload.owner_ids)} - {0}
        if active != {owner for owner, _ in payload.instance_ranks}:
            raise ValueError("G partition owners and ranks do not align")


@dataclass(frozen=True)
class EvaluationMetrics:
    uap: float | None
    ap50: float | None
    ap25: float | None
    miou: float | None
    macc: float | None
    canonical_ap50: float | None
    canonical_ap75: float | None
    classes_observed: int
    masks_observed: int
    trace_parity: bool

    def __post_init__(self) -> None:
        for name in _METRIC_NAMES:
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"evaluation metric {name} must be null or lie in [0, 1]")
        if self.classes_observed < 0 or self.masks_observed < 0:
            raise ValueError("evaluation observation counts must be nonnegative")


@dataclass(frozen=True)
class AggregateMetrics:
    status: str
    means: Mapping[str, float | None]
    denominators: Mapping[str, int]


def aggregate_scene_metrics(
    rows: Sequence[EvaluationMetrics], *, required: Sequence[str] = ()
) -> AggregateMetrics:
    if not rows:
        raise ValueError("scene metric aggregation requires at least one row")
    if not set(required).issubset(_METRIC_NAMES):
        raise ValueError("required metric name is unsupported")
    means: dict[str, float | None] = {}
    denominators: dict[str, int] = {}
    for name in _METRIC_NAMES:
        values = [getattr(row, name) for row in rows if getattr(row, name) is not None]
        denominators[name] = len(values)
        means[name] = None if not values else float(np.mean(values))
    inconclusive = any(denominators[name] != len(rows) for name in required)
    return AggregateMetrics(
        "INCONCLUSIVE_UNDEFINED_METRIC" if inconclusive else "COMPLETE",
        MappingProxyType(means),
        MappingProxyType(denominators),
    )


@dataclass(frozen=True)
class EvaluationRow:
    method_id: str
    scene_id: str
    prediction_key: str
    record_key: str
    metrics: EvaluationMetrics
    reused_evaluation: bool


class ReleasedEvaluationAdapter:
    """Call the released evaluator only after prediction manifests are locked."""

    def __init__(
        self,
        evaluator: Callable[[PredictionPayload, Mapping[str, Any]], EvaluationMetrics],
    ) -> None:
        self._evaluator = evaluator
        self._cache: dict[tuple[str, str, str], EvaluationMetrics] = {}

    def evaluate_many(
        self,
        payloads: Sequence[PredictionPayload],
        ground_truth_by_scene: Mapping[str, Mapping[str, Any]],
    ) -> tuple[EvaluationRow, ...]:
        rows: list[EvaluationRow] = []
        for payload in payloads:
            if not payload.locked:
                raise ValueError("prediction payload must be locked before GT evaluation")
            if payload.scene_id not in ground_truth_by_scene:
                raise ValueError(f"ground truth is unavailable for scene {payload.scene_id}")
            context = ground_truth_by_scene[payload.scene_id]
            evaluator_context = getattr(self._evaluator, "context_identity", None)
            context_key = canonical_digest({
                "ground_truth": _context_value(context),
                "evaluator": (
                    evaluator_context() if evaluator_context is not None
                    else _context_value(self._evaluator)
                ),
            })
            key = (payload.scene_id, payload.prediction_key, context_key)
            reused = key in self._cache
            if not reused:
                metrics = self._evaluator(
                    payload, ground_truth_by_scene[payload.scene_id]
                )
                if not isinstance(metrics, EvaluationMetrics):
                    raise TypeError("released evaluator callback must return EvaluationMetrics")
                if not metrics.trace_parity:
                    raise ValueError("released evaluator trace parity failed")
                self._cache[key] = metrics
            rows.append(
                EvaluationRow(
                    payload.method_id,
                    payload.scene_id,
                    payload.prediction_key,
                    payload.record_key,
                    self._cache[key],
                    reused,
                )
            )
        return tuple(rows)


def _context_value(value: Any) -> Any:
    """Content identity for evaluator inputs, including in-place array/file edits."""
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, Mapping):
        return {str(key): _context_value(item) for key, item in value.items()
                if key not in {"__builtins__", "__loader__", "__spec__", "__cached__"}}
    if isinstance(value, (list, tuple)):
        return [_context_value(item) for item in value]
    if isinstance(value, (Path, str)):
        path = Path(value)
        try:
            if path.is_file():
                return {"path": str(path.resolve()), "sha256": sha256_file(path)}
        except OSError:
            pass
        return str(value)
    if isinstance(value, np.generic):
        return _context_value(value.item())
    if callable(value):
        source = inspect.getsourcefile(value) if inspect.isfunction(value) else None
        return {"callable": getattr(value, "__qualname__", type(value).__qualname__),
                "source": _context_value(Path(source)) if source else None,
                "instance": id(value)}
    if inspect.ismodule(value):
        return {"module": value.__name__, "source": _context_value(
            getattr(value, "__file__", None))}
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": value.hex()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"unsupported evaluator identity input: {type(value).__name__}")


def _optional_metric(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _parity_passed(value: Any) -> bool:
    if isinstance(value, Mapping):
        if {"ap_exact", "pr_and_fn_exact"} <= set(value):
            # The unchanged trace helper includes a descriptive source reference
            # alongside its two boolean checks; that text is not another check.
            return (value["ap_exact"] is True and value["pr_and_fn_exact"] is True
                    and set(value) <= {"ap_exact", "pr_and_fn_exact", "reference"}
                    and ("reference" not in value or isinstance(value["reference"], str)))
        return bool(value) and all(_parity_passed(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_parity_passed(item) for item in value)
    return value is True


class PinnedReleasedEvaluator:
    """Scene-aware projection adapter around the unchanged released functions."""

    def __init__(
        self,
        evaluator_namespace: Mapping[str, Any],
        output_root: Path | str,
        *,
        evaluate_set_fn: Callable[..., Mapping[str, Any]] | None = None,
        semantic_metrics_fn: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        if evaluate_set_fn is None:
            from scripts.evaluation.diagnose_static_t1_attribution import evaluate_set

            evaluate_set_fn = evaluate_set
        if semantic_metrics_fn is None:
            from scripts.evaluation.evaluate_static_ovmap_readout import (
                semantic_metrics,
            )

            semantic_metrics_fn = semantic_metrics
        self.evaluator_namespace = evaluator_namespace
        self.output_root = Path(output_root)
        self.evaluate_set_fn = evaluate_set_fn
        self.semantic_metrics_fn = semantic_metrics_fn
        self._released_cache: dict[Any, Any] = {}

    def context_identity(self) -> Any:
        return _context_value({
            "namespace": self.evaluator_namespace,
            "evaluate_set": self.evaluate_set_fn,
            "semantic_metrics": self.semantic_metrics_fn,
        })

    def __call__(
        self, payload: PredictionPayload, ground_truth: Mapping[str, Any]
    ) -> EvaluationMetrics:
        if not payload.locked:
            raise ValueError("prediction payload must be locked before released evaluation")
        nearest = np.asarray(ground_truth["nearest"], dtype=np.int64)
        matched = np.asarray(ground_truth["matched"], dtype=bool)
        gt_semantic = np.asarray(ground_truth["gt_semantic"], dtype=np.int64)
        if nearest.shape != matched.shape or nearest.shape != gt_semantic.shape:
            raise ValueError("scene projection and GT semantic rows must align")
        if np.any(nearest[matched] < 0) or np.any(nearest[matched] >= len(payload.owner_ids)):
            raise ValueError("scene projection leaves prediction source rows")
        context_key = canonical_digest({
            "ground_truth": _context_value(ground_truth),
            "evaluator": self.context_identity(),
        })
        destination = self.output_root / payload.scene_id / payload.prediction_key / context_key
        mask_root = destination / "prediction"
        mask_root.mkdir(parents=True, exist_ok=True)
        projected_owners = np.zeros(nearest.shape, dtype=np.int64)
        projected_semantic = np.zeros(nearest.shape, dtype=np.int64)
        projected_owners[matched] = payload.owner_ids[nearest[matched]]
        projected_semantic[matched] = payload.semantic_labels[nearest[matched]]
        owner_labels = {
            int(owner): int(np.unique(payload.semantic_labels[payload.owner_ids == owner])[0])
            for owner in np.unique(payload.owner_ids)
            if owner > 0
        }
        rank_by_owner = dict(payload.instance_ranks)
        owners = tuple(sorted(owner_labels))
        mask_paths: dict[int, Path] = {}
        for index, owner in enumerate(owners):
            path = mask_root / f"owner_{owner:06d}.npy"
            mask = projected_owners == owner
            if path.is_file():
                if not np.array_equal(np.load(path, allow_pickle=False), mask):
                    raise ValueError("cached projected prediction mask differs")
            else:
                np.save(path, mask)
            mask_paths[index] = path
        labels = np.asarray([owner_labels[owner] for owner in owners], dtype=np.int64)
        serialized = [f"{rank_by_owner[owner]:.6f}" for owner in owners]
        kept = np.arange(len(owners), dtype=np.int64)
        gt_path = Path(ground_truth["gt_instance_path"])
        cache_key = (payload.prediction_key, context_key)
        released = self.evaluate_set_fn(
            self.evaluator_namespace,
            destination / "released",
            mask_root,
            mask_paths,
            labels,
            serialized,
            kept,
            gt_path,
            self._released_cache,
            cache_key,
        )
        semantic = self.semantic_metrics_fn(
            gt_semantic,
            projected_semantic,
            tuple(int(value) for value in ground_truth["valid_ids"]),
        )
        canonical = ground_truth.get("canonical_metrics")
        canonical_values = (
            {}
            if canonical is None
            else canonical(payload, projected_owners, ground_truth)
        )
        released_values = released["released"]
        return EvaluationMetrics(
            uap=_optional_metric(released_values.get("all_ap")),
            ap50=_optional_metric(released_values.get("all_ap_50%")),
            ap25=_optional_metric(released_values.get("all_ap_25%")),
            miou=_optional_metric(semantic.get("semantic_miou")),
            macc=_optional_metric(semantic.get("semantic_macc")),
            canonical_ap50=_optional_metric(canonical_values.get("ap50")),
            canonical_ap75=_optional_metric(canonical_values.get("ap75")),
            classes_observed=len(set(owner_labels.values()) - {0}),
            masks_observed=len(owners),
            trace_parity=_parity_passed(released.get("trace_parity")),
        )


def inspect_pinned_evaluator_interfaces(repository_root: Path | str) -> dict[str, str]:
    """Record the unchanged released function signatures used by the adapter."""

    root = Path(repository_root)
    from scripts.evaluation.diagnose_static_t1_attribution import (
        bind_protocol,
        evaluate_set,
    )
    from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
    from src.static_ovmap.attribution_objects import released_object_outcomes

    expected_root = Path(__file__).resolve().parents[3]
    if root.resolve() != expected_root:
        raise ValueError("pinned evaluator root differs from the active repository")
    return {
        "bind_protocol": str(inspect.signature(bind_protocol)),
        "evaluate_set": str(inspect.signature(evaluate_set)),
        "semantic_metrics": str(inspect.signature(semantic_metrics)),
        "released_object_outcomes": str(inspect.signature(released_object_outcomes)),
    }


def evaluation_rows_json(rows: Sequence[EvaluationRow]) -> str:
    payload = [
        {
            "method_id": row.method_id,
            "scene_id": row.scene_id,
            "prediction_key": row.prediction_key,
            "record_key": row.record_key,
            "reused_evaluation": row.reused_evaluation,
            "metrics": {
                name: getattr(row.metrics, name)
                for name in (*_METRIC_NAMES, "classes_observed", "masks_observed", "trace_parity")
            },
        }
        for row in rows
    ]
    return json.dumps(payload, sort_keys=True, allow_nan=False)


__all__ = [
    "AggregateMetrics",
    "EvaluationMetrics",
    "EvaluationRow",
    "GeometryIdentity",
    "PinnedReleasedEvaluator",
    "PredictionPayload",
    "ReleasedEvaluationAdapter",
    "aggregate_scene_metrics",
    "evaluation_rows_json",
    "inspect_pinned_evaluator_interfaces",
    "validate_prediction_invariants",
]
