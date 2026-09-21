"""Fixed causal query features, ranking policies, labels, and utility head."""

from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .query_state import (
    AcquisitionResult,
    FeatureStore,
    NativeCombineState,
    QueryCandidate,
    QueryPolicyState,
    apply_alias_merge,
    cumulative_quota,
    dispatch_frame_batch,
    native_combine_candidates,
    observe_geometric_candidates,
    update_cached_class_scores,
)

FEATURE_COUNT = 20
POLICY_IDS = ("Q_COMBINE", "Q_AREA", "Q_UNCERTAINTY", "Q_GAIN")


def _readonly(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _softmax(scores: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    shifted = scores / temperature
    shifted -= np.max(shifted)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _normalized_entropy(scores: np.ndarray) -> float:
    if len(scores) <= 1:
        return 0.0
    probabilities = _softmax(scores)
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)) / math.log(len(scores)))


def _top_gap(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def _mean_pairwise_disagreement(features: Sequence[np.ndarray]) -> float:
    rows = np.stack(features).astype(np.float64, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(rows).all():
        raise ValueError("retained features must be finite and nonzero")
    rows = rows / norms
    values = [
        1.0 - float(rows[left] @ rows[right])
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    ]
    return float(np.mean(values))


def _pose_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(math.acos(cosine))


@dataclass(frozen=True)
class QueryFeatureRow:
    values: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        values = _readonly(self.values, np.float64)
        available = _readonly(self.available, bool)
        if values.shape != (FEATURE_COUNT,) or available.shape != (FEATURE_COUNT,):
            raise ValueError("query features require 20 values and 20 availability bits")
        if not np.isfinite(values).all() or np.any(values[~available] != 0.0):
            raise ValueError("query features must be finite with zero missing values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)

    @property
    def unstandardized(self) -> np.ndarray:
        return np.concatenate((self.values, self.available.astype(np.float64)))


def build_query_features(
    candidate: QueryCandidate,
    state: QueryPolicyState,
    *,
    frame_count: int,
    budget: int,
    spent: int,
) -> QueryFeatureRow:
    """Build the protocol-ordered 20 pre-encoding scalar features."""

    if frame_count <= 0 or budget <= 0 or spent < 0:
        raise ValueError("query feature schedule inputs are invalid")
    object_state = state.object_state(candidate.owner_id)
    values = np.zeros(FEATURE_COUNT, dtype=np.float64)
    available = np.zeros(FEATURE_COUNT, dtype=bool)

    def put(index: int, value: float, condition: bool = True) -> None:
        if condition:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"query feature {index} is non-finite")
            values[index] = numeric
            available[index] = True

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = candidate.bbox_xyxy
    bbox_pixels = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
    candidate_cells = set(candidate.spherical_cells)
    cell_count = len(candidate_cells)
    new_geometry = candidate_cells - object_state.geometric_cells
    new_evidence = candidate_cells - object_state.acquired_evidence_cells

    put(0, math.log1p(candidate.global_pixels))
    put(1, math.log1p(candidate.local_pixels))
    put(2, candidate.overlap_pixels / candidate.global_pixels)
    put(3, candidate.overlap_pixels / candidate.local_pixels)
    put(4, candidate.global_pixels / bbox_pixels)
    put(5, candidate.valid_depth_fraction)
    put(6, candidate.median_depth)
    put(7, candidate.depth_iqr)
    put(8, len(new_geometry) / cell_count if cell_count else 0.0, bool(cell_count))
    put(9, len(new_evidence) / cell_count if cell_count else 0.0, bool(cell_count))
    put(10, math.log1p(object_state.successes))
    put(11, math.log1p(object_state.attempts))
    if object_state.last_success_frame is not None:
        elapsed = max(0, candidate.frame_index - object_state.last_success_frame)
        put(12, elapsed / frame_count)
    if object_state.cached_scores is not None:
        scores = np.asarray(object_state.cached_scores, dtype=np.float64)
        if scores.shape != (state.class_count,) or not np.isfinite(scores).all():
            raise ValueError("cached class scores do not match the policy vocabulary")
        put(13, _normalized_entropy(scores))
        put(14, _top_gap(scores), len(scores) >= 2)
    if len(object_state.features) >= 2:
        put(15, _mean_pairwise_disagreement([row.feature for row in object_state.features]))
    put(16, candidate.overlap_pixels / max(1, object_state.best_paid_overlap))
    if object_state.last_paid_pose is not None:
        translation, rotation = _pose_distance(
            np.asarray(object_state.last_paid_pose), candidate.camera_pose
        )
        put(17, translation)
        put(18, rotation)
    put(19, max(0, budget - spent) / budget)
    return QueryFeatureRow(values, available)


@dataclass(frozen=True)
class QueryFeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray
    fitted: np.ndarray

    @classmethod
    def fit(cls, rows: Sequence[QueryFeatureRow]) -> QueryFeatureStandardizer:
        if not rows:
            raise ValueError("FIT query feature rows must not be empty")
        mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
        std = np.ones(FEATURE_COUNT, dtype=np.float64)
        fitted = np.zeros(FEATURE_COUNT, dtype=bool)
        for index in range(FEATURE_COUNT):
            observed = [row.values[index] for row in rows if row.available[index]]
            if observed:
                mean[index] = float(np.mean(observed))
                std[index] = max(float(np.std(observed)), 1e-6)
                fitted[index] = True
        return cls(_readonly(mean), _readonly(std), _readonly(fitted, bool))

    def transform(self, row: QueryFeatureRow) -> np.ndarray:
        available = row.available & self.fitted
        values = np.zeros(FEATURE_COUNT, dtype=np.float64)
        values[available] = (
            row.values[available] - self.mean[available]
        ) / self.std[available]
        values = np.clip(values, -8.0, 8.0)
        return np.concatenate((values, available.astype(np.float64)))


def rank_query_candidates(
    policy_id: str,
    candidates: Sequence[QueryCandidate],
    state: QueryPolicyState,
    *,
    gain_predictor: Callable[[np.ndarray], Any] | None = None,
    standardizer: QueryFeatureStandardizer | None = None,
    frame_count: int | None = None,
    budget: int = 200,
    spent: int | None = None,
) -> tuple[QueryCandidate, ...]:
    """Rank eligible, not-yet-attempted requests with frozen deterministic ties."""

    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown query policy: {policy_id}")
    eligible = tuple(
        row
        for row in candidates
        if row.request_id not in state.object_state(row.owner_id).attempted_request_ids
    )
    tie = lambda row: (-row.overlap_pixels, row.owner_id, row.request_id)
    if policy_id in ("Q_AREA", "Q_COMBINE"):
        return tuple(sorted(eligible, key=tie))
    if policy_id == "Q_UNCERTAINTY":
        def uncertainty(row: QueryCandidate) -> float:
            scores = state.object_state(row.owner_id).cached_scores
            return 1.0 if scores is None else _normalized_entropy(np.asarray(scores))

        return tuple(
            sorted(
                eligible,
                key=lambda row: (-uncertainty(row), *tie(row)),
            )
        )
    if gain_predictor is None:
        raise ValueError("Q_GAIN ranking requires a frozen gain predictor")
    if not eligible:
        return ()
    inferred_frame_count = frame_count or max(row.frame_index for row in eligible) + 1
    logical_spent = state.logical_ledger.attempts if spent is None else spent
    feature_rows = [
        build_query_features(
            row,
            state,
            frame_count=inferred_frame_count,
            budget=budget,
            spent=logical_spent,
        )
        for row in eligible
    ]
    matrix = np.stack(
        [
            standardizer.transform(row) if standardizer is not None else row.unstandardized
            for row in feature_rows
        ]
    )
    predictions = np.asarray(gain_predictor(matrix), dtype=np.float64)
    if predictions.shape != (len(eligible),) or not np.isfinite(predictions).all():
        raise ValueError("gain predictor must return one finite score per candidate")
    indices = sorted(
        range(len(eligible)),
        key=lambda index: (-float(np.clip(predictions[index], -5.0, 5.0)), *tie(eligible[index])),
    )
    return tuple(eligible[index] for index in indices)


def random_exploration_ranking(
    candidates: Sequence[QueryCandidate], *, seed: int, scene_id: str, frame_index: int
) -> tuple[QueryCandidate, ...]:
    """Return a reproducible per-frame random order without reading model outputs."""

    identity = f"{seed}\0{scene_id}\0{frame_index}".encode()
    words = np.frombuffer(hashlib.sha256(identity).digest()[:16], dtype=np.uint32)
    generator = np.random.default_rng(np.random.SeedSequence(words.tolist()))
    ordered = sorted(candidates, key=lambda row: (row.owner_id, row.request_id))
    permutation = generator.permutation(len(ordered))
    return tuple(ordered[int(index)] for index in permutation)


def identifiable_target(
    annotations: Any,
    *,
    allowed_class_ids: Sequence[int],
    minimum_pixels: int = 64,
    minimum_fraction: float = 0.8,
) -> int | None:
    labels = np.asarray(annotations).reshape(-1)
    if minimum_pixels <= 0 or not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("target thresholds are invalid")
    allowed = frozenset(int(value) for value in allowed_class_ids)
    if not allowed or 0 in allowed:
        raise ValueError("allowed target IDs must be nonempty foreground classes")
    valid = labels[labels > 0]
    if len(valid) < minimum_pixels:
        return None
    values, counts = np.unique(valid, return_counts=True)
    eligible = [
        (int(count), int(value))
        for value, count in zip(values, counts, strict=True)
        if int(value) in allowed and count / len(valid) >= minimum_fraction
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row[0], -row[1]))[1]


def nll_gain_label(
    before_scores: Any | None,
    after_scores: Any,
    *,
    target_index: int,
) -> float:
    after = np.asarray(after_scores, dtype=np.float64)
    if after.ndim != 1 or not len(after) or not np.isfinite(after).all():
        raise ValueError("after scores must be a finite nonempty vector")
    if not 0 <= target_index < len(after):
        raise ValueError("target index is outside the score vector")
    if before_scores is None:
        before_probability = 1.0 / len(after)
    else:
        before = np.asarray(before_scores, dtype=np.float64)
        if before.shape != after.shape or not np.isfinite(before).all():
            raise ValueError("before and after score vectors must align")
        before_probability = float(_softmax(before)[target_index])
    after_probability = float(_softmax(after)[target_index])
    gain = -math.log(max(before_probability, 1e-300)) + math.log(
        max(after_probability, 1e-300)
    )
    return float(np.clip(gain, -5.0, 5.0))


def query_target_support_status(events: Sequence[Mapping[str, Any]]) -> str:
    identifiable = [
        row
        for row in events
        if row.get("target") is not None and np.isfinite(float(row["target"]))
    ]
    scenes = {str(row.get("scene_id", "")) for row in identifiable}
    positives = sum(float(row["target"]) > 1e-6 for row in identifiable)
    nonpositive = sum(float(row["target"]) <= 1e-6 for row in identifiable)
    supported = (
        len(identifiable) >= 100
        and len(scenes - {""}) >= 4
        and positives >= 10
        and nonpositive >= 10
    )
    return "SUPPORTED" if supported else "BLOCKED_QUERY_TARGET_SUPPORT"


@dataclass(frozen=True)
class QueryTrainingExample:
    scene_id: str
    features: np.ndarray
    target: float

    def __post_init__(self) -> None:
        features = _readonly(self.features, np.float32)
        if not self.scene_id or features.shape != (40,) or not np.isfinite(features).all():
            raise ValueError("query training examples require a scene ID and finite 40-vector")
        if not np.isfinite(self.target) or not -5.0 <= self.target <= 5.0:
            raise ValueError("query target must lie in [-5, 5]")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class QueryGainTrainingResult:
    status: str
    checkpoint_epoch: int
    fit_epochs: int
    calibration_mse: float | None
    state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]
    parameter_count: int = 1345


def _gain_head(torch: Any):
    return torch.nn.Sequential(
        torch.nn.Linear(40, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


def _scene_weights(examples: Sequence[QueryTrainingExample]) -> np.ndarray:
    counts = Counter(row.scene_id for row in examples)
    return np.asarray([1.0 / counts[row.scene_id] for row in examples], dtype=np.float32)


def train_query_gain_head(
    fit_examples: Sequence[QueryTrainingExample],
    *,
    cal_examples: Sequence[QueryTrainingExample] = (),
    max_epochs: int = 100,
) -> QueryGainTrainingResult:
    """Fit the single prescribed 40-32-1 Huber utility head."""

    import torch

    fit = tuple(fit_examples)
    cal = tuple(cal_examples)
    if not fit or not 1 <= max_epochs <= 100:
        raise ValueError("query head needs FIT examples and 1 to 100 epochs")
    x = torch.from_numpy(np.stack([row.features for row in fit]).astype(np.float32))
    y = torch.from_numpy(np.asarray([row.target for row in fit], dtype=np.float32))
    weights = torch.from_numpy(_scene_weights(fit))
    cal_x = (
        None
        if not cal
        else torch.from_numpy(np.stack([row.features for row in cal]).astype(np.float32))
    )
    cal_y = (
        None
        if not cal
        else torch.from_numpy(np.asarray([row.target for row in cal], dtype=np.float32))
    )

    torch.manual_seed(17)
    model = _gain_head(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    generator = torch.Generator().manual_seed(17)
    best_state = None
    best_optimizer = None
    best_epoch = 0
    best_mse = math.inf
    stale = 0
    fit_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), 128):
            batch = permutation[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.huber_loss(
                model(x[batch]).squeeze(1), y[batch], delta=1.0, reduction="none"
            )
            loss = torch.sum(losses * weights[batch]) / torch.sum(weights[batch])
            loss.backward()
            optimizer.step()
        fit_epochs = epoch
        if cal_x is not None and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(cal_x).squeeze(1)
                mse = float(torch.mean((predictions - cal_y) ** 2).item())
            if mse < best_mse:
                best_mse = mse
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= 3:
                    break
    if best_state is None:
        best_epoch = fit_epochs
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        status = "UNCALIBRATED"
        calibration_mse = None
    else:
        status = "COMPLETE"
        calibration_mse = best_mse
    return QueryGainTrainingResult(
        status=status,
        checkpoint_epoch=best_epoch,
        fit_epochs=fit_epochs,
        calibration_mse=calibration_mse,
        state_dict=best_state,
        optimizer_state_dict=best_optimizer,
    )


def predict_query_gain(state_dict: Mapping[str, Any], features: Any) -> np.ndarray:
    import torch

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 40 or not np.isfinite(rows).all():
        raise ValueError("query gain inference requires a finite Nx40 matrix")
    model = _gain_head(torch)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    with torch.no_grad():
        values = model(torch.from_numpy(rows)).squeeze(1).cpu().numpy()
    return np.clip(values, -5.0, 5.0)


@dataclass(frozen=True)
class QueryFrameDecision:
    frame_index: int
    quota: int
    ranked_request_ids: tuple[str, ...]
    attempted_request_ids: tuple[str, ...]
    results: tuple[AcquisitionResult, ...]


@dataclass(frozen=True)
class QueryReplayResult:
    policy_id: str
    budget: int
    state: QueryPolicyState
    frames: tuple[QueryFrameDecision, ...]

    def __post_init__(self) -> None:
        results = [result for frame in self.frames for result in frame.results]
        if self.state.logical_ledger.attempts != len(results):
            raise ValueError("logical attempt ledger does not reconcile with frame results")
        if self.state.logical_ledger.successes != sum(row.success for row in results):
            raise ValueError("logical success ledger does not reconcile with frame results")
        if self.state.logical_ledger.failures != sum(not row.success for row in results):
            raise ValueError("logical failure ledger does not reconcile with frame results")
        if self.state.logical_ledger.crop_inputs != sum(
            row.attempted_crop_inputs for row in results
        ):
            raise ValueError("logical crop ledger does not reconcile with frame results")


def _merge_native_combine_state(
    state: NativeCombineState, old_owner: int, new_owner: int
) -> None:
    if old_owner == new_owner:
        return
    old_coverage = state.coverage_by_owner.pop(old_owner, set())
    state.coverage_by_owner.setdefault(new_owner, set()).update(old_coverage)
    old_center = state.center_by_owner.pop(old_owner, None)
    if old_center is not None and new_owner not in state.center_by_owner:
        state.center_by_owner[new_owner] = old_center
    old_history = state.successful_overlaps_by_owner.pop(old_owner, [])
    state.successful_overlaps_by_owner.setdefault(new_owner, []).extend(old_history)


def run_query_policy(
    policy_id: str,
    frame_candidates: Sequence[Sequence[QueryCandidate]],
    feature_store: FeatureStore,
    text_features: Any,
    *,
    budget: int = 200,
    gain_predictor: Callable[[np.ndarray], Any] | None = None,
    standardizer: QueryFeatureStandardizer | None = None,
    alias_merges: Mapping[int, Sequence[tuple[int, int]]] | None = None,
) -> QueryReplayResult:
    """Replay one policy with cumulative allowance and a strict per-frame barrier."""

    if policy_id not in POLICY_IDS or budget < 0 or not frame_candidates:
        raise ValueError("query replay policy, budget, and frame schedule are invalid")
    text = np.asarray(text_features, dtype=np.float64)
    if text.ndim != 2 or not len(text) or not np.isfinite(text).all():
        raise ValueError("query replay text features must be a finite nonempty matrix")
    state = QueryPolicyState(policy_id, class_count=len(text))
    combine_state = NativeCombineState()
    merge_schedule = {} if alias_merges is None else alias_merges
    frame_count = len(frame_candidates)
    decisions: list[QueryFrameDecision] = []
    for frame_index, rows in enumerate(frame_candidates):
        candidates = tuple(rows)
        if any(candidate.frame_index != frame_index for candidate in candidates):
            raise ValueError("candidate frame indices must match replay schedule positions")
        for old_owner, new_owner in merge_schedule.get(frame_index, ()):
            old_canonical = state.canonical_owner(old_owner)
            new_canonical = state.canonical_owner(new_owner)
            apply_alias_merge(state, old_owner=old_owner, new_owner=new_owner)
            _merge_native_combine_state(combine_state, old_canonical, new_canonical)
        quota = cumulative_quota(
            budget,
            frame_index=frame_index,
            frame_count=frame_count,
            spent=state.logical_ledger.attempts,
        )
        eligible = (
            native_combine_candidates(candidates, combine_state)
            if policy_id == "Q_COMBINE"
            else candidates
        )
        ranked = rank_query_candidates(
            policy_id,
            eligible,
            state,
            gain_predictor=gain_predictor,
            standardizer=standardizer,
            frame_count=frame_count,
            budget=budget,
            spent=state.logical_ledger.attempts,
        )
        results = dispatch_frame_batch(state, feature_store, ranked, quota=quota)
        for owner_id in {row.owner_id for row in results if row.success}:
            update_cached_class_scores(state, owner_id, text)
        observe_geometric_candidates(state, candidates)
        decisions.append(
            QueryFrameDecision(
                frame_index=frame_index,
                quota=quota,
                ranked_request_ids=tuple(row.request_id for row in ranked),
                attempted_request_ids=tuple(row.request_id for row in ranked[:quota]),
                results=results,
            )
        )
    return QueryReplayResult(policy_id, budget, state, tuple(decisions))


@dataclass(frozen=True)
class QueryTraceEvent:
    scene_id: str
    frame_index: int
    request_id: str
    prefix_features: np.ndarray
    success: bool
    target_index: int | None
    gain_target: float | None

    def __post_init__(self) -> None:
        features = _readonly(self.prefix_features, np.float32)
        if not self.scene_id or self.frame_index < 0 or not self.request_id:
            raise ValueError("query trace event identity is invalid")
        if features.shape != (40,) or not np.isfinite(features).all():
            raise ValueError("query trace event requires a finite 40-vector")
        if self.target_index is None:
            if self.gain_target is not None:
                raise ValueError("gain target requires an identifiable target index")
        elif self.target_index < 0 or self.gain_target is None:
            raise ValueError("identified target events require a gain target")
        object.__setattr__(self, "prefix_features", features)


def collect_random_acquisition_trace(
    scene_id: str,
    frame_candidates: Sequence[Sequence[QueryCandidate]],
    feature_store: FeatureStore,
    text_features: Any,
    *,
    seed: int,
    budget: int,
    target_index_provider: Callable[[QueryCandidate], int | None],
    standardizer: QueryFeatureStandardizer | None = None,
) -> tuple[QueryTraceEvent, ...]:
    """Collect bounded exploration events while keeping labels outside policy inputs."""

    if seed not in (17, 23) or budget not in (256, 512):
        raise ValueError("exploration is frozen to FIT seed17/B512 or CAL seed23/B256")
    text = np.asarray(text_features, dtype=np.float64)
    if text.ndim != 2 or not len(text) or not np.isfinite(text).all():
        raise ValueError("trace text features must be a finite nonempty matrix")
    state = QueryPolicyState("Q_RANDOM_TRACE", class_count=len(text))
    frame_count = len(frame_candidates)
    events: list[QueryTraceEvent] = []
    for frame_index, rows in enumerate(frame_candidates):
        candidates = tuple(rows)
        quota = cumulative_quota(
            budget,
            frame_index=frame_index,
            frame_count=frame_count,
            spent=state.logical_ledger.attempts,
        )
        ranked = random_exploration_ranking(
            candidates, seed=seed, scene_id=scene_id, frame_index=frame_index
        )
        selected = ranked[:quota]
        prefixes: dict[str, np.ndarray] = {}
        before_scores: dict[str, np.ndarray | None] = {}
        target_indices: dict[str, int | None] = {}
        for candidate in selected:
            feature_row = build_query_features(
                candidate,
                state,
                frame_count=frame_count,
                budget=budget,
                spent=state.logical_ledger.attempts,
            )
            prefixes[candidate.request_id] = (
                feature_row.unstandardized
                if standardizer is None
                else standardizer.transform(feature_row)
            )
            scores = state.object_state(candidate.owner_id).cached_scores
            before_scores[candidate.request_id] = (
                None if scores is None else np.array(scores, copy=True)
            )
            target_indices[candidate.request_id] = target_index_provider(candidate)
        results = dispatch_frame_batch(state, feature_store, ranked, quota=quota)
        for owner_id in {row.owner_id for row in results if row.success}:
            update_cached_class_scores(state, owner_id, text)
        by_request = {row.request_id: row for row in results}
        by_candidate = {row.request_id: row for row in selected}
        for request_id in (row.request_id for row in selected):
            result = by_request[request_id]
            candidate = by_candidate[request_id]
            target_index = target_indices[request_id]
            gain = None
            if target_index is not None:
                after = state.object_state(candidate.owner_id).cached_scores
                if after is not None:
                    gain = nll_gain_label(
                        before_scores[request_id], after, target_index=target_index
                    )
                else:
                    gain = 0.0
            events.append(
                QueryTraceEvent(
                    scene_id,
                    frame_index,
                    request_id,
                    prefixes[request_id],
                    result.success,
                    target_index,
                    gain,
                )
            )
        observe_geometric_candidates(state, candidates)
    return tuple(events)


__all__ = [
    "FEATURE_COUNT",
    "POLICY_IDS",
    "QueryFeatureRow",
    "QueryFeatureStandardizer",
    "QueryFrameDecision",
    "QueryGainTrainingResult",
    "QueryReplayResult",
    "QueryTraceEvent",
    "QueryTrainingExample",
    "build_query_features",
    "collect_random_acquisition_trace",
    "identifiable_target",
    "nll_gain_label",
    "predict_query_gain",
    "query_target_support_status",
    "random_exploration_ranking",
    "rank_query_candidates",
    "run_query_policy",
    "train_query_gain_head",
]
