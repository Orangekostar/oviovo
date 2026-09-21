"""Direct semantic readouts and the fixed paired correction selector."""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

FEATURE_COUNT = 32
HEAD_IDS = ("S_SIMPLE", "S_NO_CONTEXT", "S_PAIRED")
TEACHER_IDS = ("S_SIGLIP2_AREA", "S_SIGLIP2_VOTE", "S_WOW_VOTE")
ADOPTION_THRESHOLDS: tuple[float | str, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.5,
    "KEEP_ALL",
)
SIMPLE_FEATURE_INDICES = frozenset((0, 1, 2, 3, 8, 9, 12, 19))
NO_CONTEXT_FEATURE_INDICES = frozenset(range(24))
PAIRED_FEATURE_INDICES = frozenset(range(32))
EVENT_LABELS = ("GAIN", "HARM", "OTHER")


def _unit_rows(value: Any, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} rows must have nonzero norm")
    return rows / norms


@dataclass(frozen=True)
class DirectReadout:
    label_id: int
    scores: tuple[float, ...]
    aggregate_feature: np.ndarray | None
    successful_views: int


def area_readout(
    view_features: Any,
    visible_areas: Any,
    text_features: Any,
    *,
    valid_ids: Sequence[int],
) -> DirectReadout:
    """Visible-area feature mean, final L2 norm, and cosine argmax."""

    views = _unit_rows(view_features, name="view features")
    text = _unit_rows(text_features, name="text features")
    weights = np.asarray(visible_areas, dtype=np.float64)
    ids = tuple(int(value) for value in valid_ids)
    if views.shape[0] == 0 or weights.shape != (views.shape[0],):
        raise ValueError("area readout requires aligned nonempty views and weights")
    if text.shape[0] != len(ids) or views.shape[1] != text.shape[1]:
        raise ValueError("area readout feature spaces or class IDs do not align")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("visible areas must be finite and positive")
    aggregate = np.average(views, axis=0, weights=weights)
    norm = float(np.linalg.norm(aggregate))
    if norm <= 0.0:
        raise ValueError("area-weighted aggregate has zero norm")
    aggregate = aggregate / norm
    scores = text @ aggregate
    index = int(np.argmax(scores))
    aggregate.flags.writeable = False
    return DirectReadout(
        label_id=ids[index],
        scores=tuple(float(value) for value in scores),
        aggregate_feature=aggregate,
        successful_views=len(views),
    )


def vote_readout(
    *,
    label_ids: Sequence[int],
    request_ranks: Sequence[int],
    valid_ids: Sequence[int],
) -> DirectReadout:
    """Plurality, earliest successful request, then valid-ID order."""

    labels = tuple(int(value) for value in label_ids)
    ranks = tuple(int(value) for value in request_ranks)
    ids = tuple(int(value) for value in valid_ids)
    if not labels or len(labels) != len(ranks):
        raise ValueError("vote readout requires aligned successful labels and ranks")
    if len(set(ids)) != len(ids) or not set(labels).issubset(ids):
        raise ValueError("vote labels must belong to unique valid_ids")
    if any(rank < 0 for rank in ranks):
        raise ValueError("request ranks must be nonnegative")
    counts = Counter(labels)
    maximum = max(counts.values())
    tied = {label for label, count in counts.items() if count == maximum}
    earliest = {
        label: min(rank for label_value, rank in zip(labels, ranks, strict=True) if label_value == label)
        for label in tied
    }
    best_rank = min(earliest.values())
    finalists = {label for label, rank in earliest.items() if rank == best_rank}
    selected = next(label for label in ids if label in finalists)
    scores = tuple(float(counts.get(label, 0)) for label in ids)
    return DirectReadout(selected, scores, None, len(labels))


@dataclass(frozen=True)
class SemanticRequestStats:
    request_mask_pixels: int
    bbox_pixels: int
    depth_valid_fraction: float
    local_global_iou: float
    representation_survived: bool
    teacher_technical_failure: bool
    unavailable_before_inference: bool
    native_valid: bool

    def __post_init__(self) -> None:
        if self.request_mask_pixels <= 0 or self.bbox_pixels <= 0:
            raise ValueError("request and bbox pixels must be positive")
        if self.request_mask_pixels > self.bbox_pixels:
            raise ValueError("request mask cannot exceed its bbox")
        for name in ("depth_valid_fraction", "local_global_iou"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class SemanticFeatureInputs:
    class_count: int
    keep_index: int | None
    replace_index: int
    native_aggregate_scores: np.ndarray
    native_view_scores: np.ndarray
    native_view_features: np.ndarray
    native_requested_views: int
    teacher_aggregate_scores: np.ndarray | None
    teacher_view_labels: tuple[int, ...]
    teacher_view_strengths: np.ndarray | None
    teacher_requested_views: int
    name_mapping_gaps: tuple[float, ...]
    source_mask_point_count: int
    request_stats: tuple[SemanticRequestStats, ...]
    native_raw_scores: np.ndarray
    native_foreground_scores: np.ndarray
    native_background_scores: np.ndarray
    background_usable: tuple[bool, ...]


@dataclass(frozen=True)
class SemanticFeatures:
    values: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        available = np.array(self.available, dtype=bool, copy=True)
        if values.shape != (FEATURE_COUNT,) or available.shape != (FEATURE_COUNT,):
            raise ValueError("semantic features require 32 values and 32 availability bits")
        if not np.isfinite(values).all() or np.any(values[~available] != 0.0):
            raise ValueError("semantic feature values must be finite and missing values zero")
        values.flags.writeable = False
        available.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)

    def with_missing(self, indices: Sequence[int]) -> SemanticFeatures:
        values = np.array(self.values, copy=True)
        available = np.array(self.available, copy=True)
        for index in indices:
            if not 0 <= int(index) < FEATURE_COUNT:
                raise ValueError("missing feature index is out of range")
            values[int(index)] = 0.0
            available[int(index)] = False
        return replace(self, values=values, available=available)


def _entropy_from_probabilities(probabilities: np.ndarray, class_count: int) -> float:
    if class_count <= 1:
        return 0.0
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)) / math.log(class_count))


def _score_entropy(scores: np.ndarray, temperature: float = 0.07) -> float:
    shifted = scores / temperature
    shifted -= np.max(shifted)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return _entropy_from_probabilities(probabilities, len(scores))


def _label_entropy(labels: Sequence[int], class_count: int) -> float:
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    return _entropy_from_probabilities(counts / counts.sum(), class_count)


def _top_gap(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def build_semantic_features(inputs: SemanticFeatureInputs) -> SemanticFeatures:
    """Build the protocol-ordered 32 scalar values and availability bits."""

    class_count = int(inputs.class_count)
    replace_index = int(inputs.replace_index)
    keep_index = None if inputs.keep_index is None else int(inputs.keep_index)
    if class_count <= 1 or not 0 <= replace_index < class_count:
        raise ValueError("semantic feature class dimensions are invalid")
    if keep_index is not None and not 0 <= keep_index < class_count:
        raise ValueError("KEEP class index is outside the text vocabulary")
    native_aggregate = np.asarray(inputs.native_aggregate_scores, dtype=np.float64)
    native_scores = np.asarray(inputs.native_view_scores, dtype=np.float64)
    native_features = np.asarray(inputs.native_view_features, dtype=np.float64)
    if native_aggregate.shape != (class_count,) or native_scores.ndim != 2:
        raise ValueError("native score arrays have invalid dimensions")
    if native_scores.shape[1] != class_count or native_features.shape[0] != len(native_scores):
        raise ValueError("native view scores and features must align")
    if native_features.ndim != 2 or not np.isfinite(native_features).all():
        raise ValueError("native view features must be a finite matrix")
    if inputs.native_requested_views < len(native_scores) or inputs.native_requested_views <= 0:
        raise ValueError("native requested-view count is invalid")
    if not np.isfinite(native_aggregate).all() or not np.isfinite(native_scores).all():
        raise ValueError("native scores must be finite")

    teacher_labels = tuple(int(label) for label in inputs.teacher_view_labels)
    if any(not 0 <= label < class_count for label in teacher_labels):
        raise ValueError("teacher view labels are outside the vocabulary")
    if inputs.teacher_requested_views < len(teacher_labels) or inputs.teacher_requested_views <= 0:
        raise ValueError("teacher requested-view count is invalid")
    teacher_strengths = (
        None
        if inputs.teacher_view_strengths is None
        else np.asarray(inputs.teacher_view_strengths, dtype=np.float64)
    )
    if teacher_strengths is not None and teacher_strengths.shape != (
        len(teacher_labels),
        class_count,
    ):
        raise ValueError("teacher strengths must align with successful teacher views")
    teacher_aggregate = (
        None
        if inputs.teacher_aggregate_scores is None
        else np.asarray(inputs.teacher_aggregate_scores, dtype=np.float64)
    )
    if teacher_aggregate is not None and teacher_aggregate.shape != (class_count,):
        raise ValueError("teacher aggregate scores must align with classes")
    if teacher_aggregate is None and teacher_strengths is not None and len(teacher_strengths):
        teacher_aggregate = teacher_strengths.mean(axis=0)

    values = np.zeros(FEATURE_COUNT, dtype=np.float64)
    available = np.zeros(FEATURE_COUNT, dtype=bool)

    def put(index: int, value: float, condition: bool = True) -> None:
        if condition:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"semantic feature {index} is non-finite")
            values[index] = numeric
            available[index] = True

    has_keep = keep_index is not None
    put(0, native_aggregate[keep_index] if has_keep else 0.0, has_keep)
    put(1, native_aggregate[replace_index])
    put(
        2,
        native_aggregate[replace_index] - native_aggregate[keep_index]
        if has_keep
        else 0.0,
        has_keep,
    )
    put(3, _top_gap(native_aggregate))
    put(4, _score_entropy(native_aggregate))
    native_labels = np.argmax(native_scores, axis=1) if len(native_scores) else np.array([])
    put(5, np.mean(native_labels == keep_index) if has_keep else 0.0, has_keep and len(native_scores) > 0)
    put(6, np.mean(native_labels == replace_index), len(native_scores) > 0)
    if len(native_features) >= 2:
        unit_features = _unit_rows(native_features, name="native view features")
        disagreements = [
            1.0 - float(unit_features[left] @ unit_features[right])
            for left in range(len(unit_features))
            for right in range(left + 1, len(unit_features))
        ]
        put(7, np.mean(disagreements))
    put(8, len(native_scores) / inputs.native_requested_views)
    put(9, len(teacher_labels) / inputs.teacher_requested_views)
    put(10, np.mean(np.asarray(teacher_labels) == replace_index), bool(teacher_labels))
    put(11, _label_entropy(teacher_labels, class_count), bool(teacher_labels))
    put(12, _top_gap(teacher_aggregate), teacher_aggregate is not None)
    put(
        13,
        teacher_aggregate[keep_index]
        if teacher_aggregate is not None and has_keep
        else 0.0,
        teacher_aggregate is not None and has_keep,
    )
    put(
        14,
        teacher_aggregate[replace_index] if teacher_aggregate is not None else 0.0,
        teacher_aggregate is not None,
    )
    mapping_gaps = np.asarray(inputs.name_mapping_gaps, dtype=np.float64)
    put(15, np.mean(mapping_gaps) if len(mapping_gaps) else 0.0, len(mapping_gaps) > 0)

    if inputs.source_mask_point_count < 0:
        raise ValueError("source mask point count must be nonnegative")
    put(16, np.log1p(inputs.source_mask_point_count))
    stats = tuple(inputs.request_stats)
    if stats:
        put(17, np.mean([np.log1p(row.request_mask_pixels) for row in stats]))
        put(18, np.mean([row.request_mask_pixels / row.bbox_pixels for row in stats]))
        put(19, min(row.depth_valid_fraction for row in stats))
        put(20, np.mean([row.local_global_iou for row in stats]))
        put(21, np.mean([row.representation_survived for row in stats]))
        put(22, np.mean([row.teacher_technical_failure for row in stats]))
        put(23, np.mean([row.unavailable_before_inference for row in stats]))

    raw = np.asarray(inputs.native_raw_scores, dtype=np.float64)
    foreground = np.asarray(inputs.native_foreground_scores, dtype=np.float64)
    background = np.asarray(inputs.native_background_scores, dtype=np.float64)
    expected_shape = (len(native_scores), 3, class_count)
    if raw.shape != expected_shape or foreground.shape != expected_shape:
        raise ValueError("native raw/foreground scale scores must be Vx3xC")
    if background.shape != expected_shape:
        raise ValueError("native background scale scores must be Vx3xC")
    background_usable = np.asarray(inputs.background_usable, dtype=bool)
    if background_usable.shape != (len(native_scores),):
        raise ValueError("background availability must align with native views")
    if len(native_scores):
        put(24, np.mean(raw[:, :, keep_index] - foreground[:, :, keep_index]) if has_keep else 0.0, has_keep)
        put(25, np.mean(raw[:, :, replace_index] - foreground[:, :, replace_index]))
        foreground_margins = foreground[:, :, replace_index].mean(axis=1)
        if has_keep:
            foreground_margins -= foreground[:, :, keep_index].mean(axis=1)
        put(27, np.mean(foreground_margins), has_keep)
        put(
            28,
            np.mean(np.std(foreground[:, :, keep_index], axis=1)) if has_keep else 0.0,
            has_keep,
        )
        put(29, np.mean(np.std(foreground[:, :, replace_index], axis=1)))
        put(30, np.std(foreground_margins), has_keep and len(native_scores) >= 2)
        usable_rows = np.flatnonzero(background_usable)
        if has_keep and len(usable_rows):
            background_margins = (
                background[usable_rows, :, replace_index].mean(axis=1)
                - background[usable_rows, :, keep_index].mean(axis=1)
            )
            put(26, np.mean(background_margins))
        put(31, len(usable_rows) / inputs.native_requested_views)

    return SemanticFeatures(values=values, available=available)


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray
    fitted: np.ndarray

    @classmethod
    def fit(cls, rows: Sequence[SemanticFeatures]) -> FeatureStandardizer:
        if not rows:
            raise ValueError("FIT feature rows must not be empty")
        mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
        std = np.ones(FEATURE_COUNT, dtype=np.float64)
        fitted = np.zeros(FEATURE_COUNT, dtype=bool)
        for index in range(FEATURE_COUNT):
            observed = [row.values[index] for row in rows if row.available[index]]
            if observed:
                mean[index] = float(np.mean(observed))
                std[index] = max(float(np.std(observed)), 1e-6)
                fitted[index] = True
        for array in (mean, std, fitted):
            array.flags.writeable = False
        return cls(mean, std, fitted)

    def transform(self, row: SemanticFeatures) -> np.ndarray:
        available = row.available & self.fitted
        values = np.zeros(FEATURE_COUNT, dtype=np.float64)
        values[available] = (
            row.values[available] - self.mean[available]
        ) / self.std[available]
        values = np.clip(values, -8.0, 8.0)
        return np.concatenate((values, available.astype(np.float64)))


def _head_indices(head_id: str) -> frozenset[int]:
    if head_id == "S_SIMPLE":
        return SIMPLE_FEATURE_INDICES
    if head_id == "S_NO_CONTEXT":
        return NO_CONTEXT_FEATURE_INDICES
    if head_id == "S_PAIRED":
        return PAIRED_FEATURE_INDICES
    raise ValueError(f"unknown semantic head: {head_id}")


def mask_features_for_head(features: Any, head_id: str) -> np.ndarray:
    row = np.asarray(features, dtype=np.float64)
    if row.shape != (64,) or not np.isfinite(row).all():
        raise ValueError("selector head input must be a finite 64-vector")
    keep = _head_indices(head_id)
    result = np.zeros(64, dtype=np.float64)
    for index in keep:
        result[index] = row[index]
        result[index + FEATURE_COUNT] = row[index + FEATURE_COUNT]
    return result


def head_parameter_counts(head_id: str) -> dict[str, int]:
    active_inputs = 2 * len(_head_indices(head_id))
    full = 64 * 32 + 32 + 32 * 3 + 3
    active = active_inputs * 32 + 32 + 32 * 3 + 3
    return {"full": full, "active": active}


def event_support_status(events: Sequence[Mapping[str, Any]]) -> str:
    by_event = {
        event: [row for row in events if row.get("event") == event]
        for event in ("GAIN", "HARM")
    }
    supported = all(
        len(rows) >= 5 and len({str(row.get("scene_id")) for row in rows}) >= 2
        for rows in by_event.values()
    )
    return "SUPPORTED" if supported else "BLOCKED_EVENT_SUPPORT"


def semantic_event_label(
    incumbent_label: Any,
    suggestion_label: Any,
    ground_truth_label: Any,
    *,
    technically_available: bool,
) -> str | None:
    """Create one supervised event after independent geometric matching."""

    if not technically_available or suggestion_label == incumbent_label:
        return None
    incumbent_correct = incumbent_label == ground_truth_label
    suggestion_correct = suggestion_label == ground_truth_label
    if suggestion_correct and not incumbent_correct:
        return "GAIN"
    if incumbent_correct and not suggestion_correct:
        return "HARM"
    return "OTHER"


def inverse_scene_weights(scene_ids: Sequence[str]) -> np.ndarray:
    scenes = tuple(str(scene_id) for scene_id in scene_ids)
    if not scenes or any(not scene_id for scene_id in scenes):
        raise ValueError("FIT scene IDs must be nonempty")
    counts = Counter(scenes)
    return np.asarray([1.0 / counts[scene_id] for scene_id in scenes], dtype=np.float32)


def select_teacher(rows: Sequence[Mapping[str, Any]]) -> str:
    order = {method: index for index, method in enumerate(TEACHER_IDS)}
    normalized = []
    for row in rows:
        method = str(row.get("method"))
        if method not in order:
            raise ValueError("teacher row contains an unsupported method")
        values = tuple(float(row[name]) for name in ("mean_uap", "mean_miou", "median_cost"))
        if not np.isfinite(values).all():
            raise ValueError("teacher selection metrics must be finite")
        normalized.append((method, *values))
    if not normalized:
        raise ValueError("no alternative teacher is available")
    return min(
        normalized,
        key=lambda row: (-row[1], -row[2], row[3], order[row[0]]),
    )[0]


def _threshold_rank(value: float | str) -> float:
    if value == "KEEP_ALL":
        return math.inf
    numeric = float(value)
    if numeric not in ADOPTION_THRESHOLDS:
        raise ValueError("threshold is outside the frozen CAL grid")
    return numeric


def select_adoption_threshold(rows: Sequence[Mapping[str, Any]]) -> float | str:
    normalized: list[tuple[float | str, float, float, int, int]] = []
    for row in rows:
        threshold = row.get("threshold")
        _threshold_rank(threshold)
        uap = float(row["mean_uap"])
        miou = float(row["mean_miou"])
        harmful = int(row["harmful"])
        replacements = int(row["replacements"])
        if not np.isfinite((uap, miou)).all() or min(harmful, replacements) < 0:
            raise ValueError("CAL threshold metrics must be finite and nonnegative")
        normalized.append((threshold, uap, miou, harmful, replacements))
    if not normalized:
        raise ValueError("CAL threshold rows must not be empty")
    return max(
        normalized,
        key=lambda row: (
            row[1],
            row[2],
            -row[3],
            -row[4],
            _threshold_rank(row[0]),
        ),
    )[0]


def apply_adoption_threshold(values: Any, threshold: float | str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("adoption values must be finite")
    rank = _threshold_rank(threshold)
    if math.isinf(rank):
        return np.zeros(scores.shape, dtype=bool)
    return scores > rank


@dataclass(frozen=True)
class HeadTrainingResult:
    head_id: str
    status: str
    checkpoint_epoch: int
    fit_epochs: int
    calibration_nll: float | None
    state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]
    scaler_state_dict: Mapping[str, Any]
    parameter_counts: Mapping[str, int]


def _semantic_head(torch: Any):
    return torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 3),
    )


def train_semantic_head(
    head_id: str,
    fit_features: Any,
    fit_labels: Sequence[str | int],
    scene_weights: Any,
    *,
    cal_features: Any | None = None,
    cal_labels: Sequence[str | int] = (),
    max_epochs: int = 100,
) -> HeadTrainingResult:
    """Train the fixed 64-32-3 head with the protocol's one-seed budget."""

    import torch

    features = np.asarray(fit_features, dtype=np.float32)
    weights = np.asarray(scene_weights, dtype=np.float32)
    if not 1 <= max_epochs <= 100:
        raise ValueError("semantic head training must use 1 to 100 epochs")
    if features.ndim != 2 or features.shape[1] != 64 or len(features) == 0:
        raise ValueError("FIT head features must be a nonempty Nx64 matrix")
    if weights.shape != (len(features),) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("FIT scene weights must be finite, positive, and aligned")
    label_to_index = {label: index for index, label in enumerate(EVENT_LABELS)}

    def labels_to_array(values: Sequence[str | int], expected: int) -> np.ndarray:
        rows = np.asarray(
            [label_to_index[value] if isinstance(value, str) else int(value) for value in values],
            dtype=np.int64,
        )
        if rows.shape != (expected,) or np.any(rows < 0) or np.any(rows >= 3):
            raise ValueError("selector labels must be GAIN, HARM, or OTHER")
        return rows

    labels = labels_to_array(fit_labels, len(features))
    masked = np.stack([mask_features_for_head(row, head_id) for row in features]).astype(np.float32)
    cal = None if cal_features is None else np.asarray(cal_features, dtype=np.float32)
    if cal is not None and (cal.ndim != 2 or cal.shape[1] != 64):
        raise ValueError("CAL head features must be an Nx64 matrix")
    cal_count = 0 if cal is None else len(cal)
    cal_targets = labels_to_array(cal_labels, cal_count)
    if cal_count:
        cal = np.stack([mask_features_for_head(row, head_id) for row in cal]).astype(np.float32)

    torch.manual_seed(17)
    model = _semantic_head(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    x = torch.from_numpy(masked)
    y = torch.from_numpy(labels)
    w = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(17)
    best_state = None
    best_optimizer_state = None
    best_epoch = 0
    best_nll = math.inf
    stale_evaluations = 0
    fit_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), 128):
            batch = permutation[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.cross_entropy(
                model(x[batch]), y[batch], reduction="none"
            )
            loss = torch.sum(losses * w[batch]) / torch.sum(w[batch])
            loss.backward()
            optimizer.step()
        fit_epochs = epoch
        if cal_count and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(cal))
                nll = float(
                    torch.nn.functional.cross_entropy(
                        logits, torch.from_numpy(cal_targets)
                    ).item()
                )
            if nll < best_nll:
                best_nll = nll
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                if stale_evaluations >= 3:
                    break
    if best_state is None:
        best_epoch = fit_epochs
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        status = "UNCALIBRATED"
        calibration_nll = None
    else:
        status = "COMPLETE"
        calibration_nll = best_nll
    return HeadTrainingResult(
        head_id=head_id,
        status=status,
        checkpoint_epoch=best_epoch,
        fit_epochs=fit_epochs,
        calibration_nll=calibration_nll,
        state_dict=best_state,
        optimizer_state_dict=best_optimizer_state,
        scaler_state_dict={},
        parameter_counts=head_parameter_counts(head_id),
    )


def adoption_values(head_id: str, state_dict: Mapping[str, Any], features: Any) -> np.ndarray:
    """Return P(GAIN)-P(HARM) from a frozen selector checkpoint."""

    import torch

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 64 or not np.isfinite(rows).all():
        raise ValueError("selector inference requires a finite Nx64 matrix")
    masked = np.stack([mask_features_for_head(row, head_id) for row in rows]).astype(
        np.float32
    )
    model = _semantic_head(torch)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(masked)), dim=1)
    return (probabilities[:, 0] - probabilities[:, 1]).cpu().numpy()
