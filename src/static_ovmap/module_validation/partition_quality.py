"""Agreement and learned quality scoring for complete leaf partitions."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .entity_hypotheses import (
    ConflictGroup,
    FrameLeafEvidence,
    NativeLeaves,
    PartitionHypothesis,
    SurfaceGraph,
)

GEOMETRY_FEATURE_COUNT = 20
GEOMETRY_MARGINS: tuple[float | str, ...] = (0.0, 0.02, 0.05, "KEEP_ALL")


def _readonly(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class AgreementScore:
    mean_frame_agreement: float | None
    visible_pixel_weighted_agreement: float | None
    defined_frames: int
    visible_pixels: int
    matched_pixels: int


def _frame_intersections(
    hypothesis: PartitionHypothesis, frame: FrameLeafEvidence
) -> tuple[np.ndarray, int]:
    component_by_leaf = {
        leaf_id: component
        for leaf_id, component in zip(
            hypothesis.leaf_ids, hypothesis.components, strict=True
        )
    }
    mask = np.isin(frame.pixel_leaf_ids, hypothesis.leaf_ids)
    pixel_leaves = frame.pixel_leaf_ids[mask]
    pixel_entities = frame.pixel_entity_ids[mask]
    denominator = len(pixel_leaves)
    positive_entities = sorted(set(map(int, pixel_entities[pixel_entities > 0])))
    matrix = np.zeros(
        (hypothesis.component_count, len(positive_entities)), dtype=np.int64
    )
    entity_column = {entity: index for index, entity in enumerate(positive_entities)}
    for leaf_id, entity in zip(pixel_leaves, pixel_entities, strict=True):
        if entity > 0:
            matrix[component_by_leaf[int(leaf_id)], entity_column[int(entity)]] += 1
    return matrix, denominator


def partition_agreement(
    hypothesis: PartitionHypothesis,
    frames: Sequence[FrameLeafEvidence],
) -> AgreementScore:
    from scipy.optimize import linear_sum_assignment

    frame_scores: list[float] = []
    total_visible = 0
    total_matched = 0
    for frame in frames:
        intersections, denominator = _frame_intersections(hypothesis, frame)
        if denominator == 0:
            continue
        if intersections.size:
            rows, columns = linear_sum_assignment(intersections, maximize=True)
            matched = int(intersections[rows, columns].sum())
        else:
            matched = 0
        frame_scores.append(matched / denominator)
        total_visible += denominator
        total_matched += matched
    if not frame_scores:
        return AgreementScore(None, None, 0, 0, 0)
    return AgreementScore(
        float(np.mean(frame_scores)),
        total_matched / total_visible,
        len(frame_scores),
        total_visible,
        total_matched,
    )


def select_agreement_hypothesis(
    hypotheses: Sequence[PartitionHypothesis],
    frames: Sequence[FrameLeafEvidence],
) -> PartitionHypothesis:
    rows = tuple(hypotheses)
    if not rows:
        raise ValueError("agreement selection requires hypotheses")
    scored = []
    for order, hypothesis in enumerate(rows):
        score = partition_agreement(hypothesis, frames).mean_frame_agreement
        scored.append(
            (
                -math.inf if score is None else score,
                int(hypothesis.kind == "ORIGINAL"),
                -order,
                hypothesis,
            )
        )
    return max(scored, key=lambda row: row[:3])[3]


@dataclass(frozen=True)
class PartitionFeatures:
    values: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        values = _readonly(self.values, np.float64)
        available = _readonly(self.available, bool)
        if values.shape != (GEOMETRY_FEATURE_COUNT,) or available.shape != (
            GEOMETRY_FEATURE_COUNT,
        ):
            raise ValueError("partition features require 20 values and 20 bits")
        if not np.isfinite(values).all() or np.any(values[~available] != 0.0):
            raise ValueError("partition feature values must be finite and missing values zero")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)

    def standardized(self, mean: Any, std: Any) -> np.ndarray:
        means = np.asarray(mean, dtype=np.float64)
        deviations = np.asarray(std, dtype=np.float64)
        if means.shape != self.values.shape or deviations.shape != self.values.shape:
            raise ValueError("partition standardizer must contain 20 values")
        if not np.isfinite(means).all() or not np.isfinite(deviations).all() or np.any(deviations <= 0.0):
            raise ValueError("partition standardizer must be finite with positive std")
        values = np.zeros(GEOMETRY_FEATURE_COUNT, dtype=np.float64)
        values[self.available] = (
            self.values[self.available] - means[self.available]
        ) / np.maximum(deviations[self.available], 1e-6)
        values = np.clip(values, -8.0, 8.0)
        return np.concatenate((values, self.available.astype(np.float64)))


@dataclass(frozen=True)
class PartitionFeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray
    fitted: np.ndarray

    @classmethod
    def fit(cls, rows: Sequence[PartitionFeatures]) -> PartitionFeatureStandardizer:
        if not rows:
            raise ValueError("FIT partition features must not be empty")
        mean = np.zeros(GEOMETRY_FEATURE_COUNT, dtype=np.float64)
        std = np.ones(GEOMETRY_FEATURE_COUNT, dtype=np.float64)
        fitted = np.zeros(GEOMETRY_FEATURE_COUNT, dtype=bool)
        for index in range(GEOMETRY_FEATURE_COUNT):
            observed = [row.values[index] for row in rows if row.available[index]]
            if observed:
                mean[index] = np.mean(observed)
                std[index] = max(float(np.std(observed)), 1e-6)
                fitted[index] = True
        return cls(_readonly(mean), _readonly(std), _readonly(fitted, bool))

    def transform(self, row: PartitionFeatures) -> np.ndarray:
        available = row.available & self.fitted
        temporary = PartitionFeatures(
            np.where(available, row.values, 0.0), available
        )
        return temporary.standardized(self.mean, self.std)


def _leaf_pair_counts(
    left_leaf: int,
    right_leaf: int,
    frames: Sequence[FrameLeafEvidence],
) -> tuple[int, int, int]:
    known = same = different = 0
    for frame in frames:
        left = int(frame.entity_by_leaf[left_leaf])
        right = int(frame.entity_by_leaf[right_leaf])
        if left > 0 and right > 0:
            known += 1
            if left == right:
                same += 1
            else:
                different += 1
    return known, same, different


def _disconnected_ratio(
    hypothesis: PartitionHypothesis, leaves: NativeLeaves
) -> float:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    local = {leaf_id: index for index, leaf_id in enumerate(hypothesis.leaf_ids)}
    total_components = 0
    for output_component in range(hypothesis.component_count):
        component_leaves = [
            leaf_id
            for leaf_id, label in zip(
                hypothesis.leaf_ids, hypothesis.components, strict=True
            )
            if label == output_component
        ]
        component_local = {leaf_id: index for index, leaf_id in enumerate(component_leaves)}
        edges = [
            (component_local[contact.left_leaf], component_local[contact.right_leaf])
            for contact in leaves.contacts
            if contact.left_leaf in component_local and contact.right_leaf in component_local
        ]
        if edges:
            edge_array = np.asarray(edges, dtype=np.int64)
            row = np.concatenate((edge_array[:, 0], edge_array[:, 1]))
            column = np.concatenate((edge_array[:, 1], edge_array[:, 0]))
            graph = coo_matrix(
                (np.ones(len(row)), (row, column)),
                shape=(len(component_leaves), len(component_leaves)),
            )
            count, _labels = connected_components(graph, directed=False)
        else:
            count = len(component_leaves)
        total_components += count
    if len(local) != len(hypothesis.leaf_ids):
        raise ValueError("hypothesis leaf IDs must be unique")
    return total_components / hypothesis.component_count


def build_partition_features(
    hypothesis: PartitionHypothesis,
    group: ConflictGroup,
    leaves: NativeLeaves,
    surface_graph: SurfaceGraph,
    frames: Sequence[FrameLeafEvidence],
    surface_xyz: Any,
) -> PartitionFeatures:
    xyz = np.asarray(surface_xyz, dtype=np.float64)
    if xyz.shape != surface_graph.normals.shape or xyz.shape[1:] != (3,):
        raise ValueError("partition feature surface rows must align")
    if hypothesis.group_id != group.group_id or hypothesis.leaf_ids != group.leaf_ids:
        raise ValueError("partition hypothesis and conflict group do not align")
    values = np.zeros(GEOMETRY_FEATURE_COUNT, dtype=np.float64)
    available = np.zeros(GEOMETRY_FEATURE_COUNT, dtype=bool)

    def put(index: int, value: float, condition: bool = True) -> None:
        if condition:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"partition feature {index} is non-finite")
            values[index] = numeric
            available[index] = True

    component_by_leaf = {
        leaf_id: component
        for leaf_id, component in zip(
            hypothesis.leaf_ids, hypothesis.components, strict=True
        )
    }
    group_rows = np.sort(
        np.concatenate([leaves.leaves[leaf_id].source_rows for leaf_id in group.leaf_ids])
    )
    component_rows = [
        np.sort(
            np.concatenate(
                [
                    leaves.leaves[leaf_id].source_rows
                    for leaf_id, label in component_by_leaf.items()
                    if label == component
                ]
            )
        )
        for component in range(hypothesis.component_count)
    ]
    fractions = np.asarray([len(rows) / len(group_rows) for rows in component_rows])
    put(0, np.log1p(len(group_rows)))
    put(1, np.log1p(len(group.leaf_ids)))
    put(2, hypothesis.component_count)
    put(3, fractions.min())
    put(4, fractions.max())
    entropy = 0.0
    if hypothesis.component_count > 1:
        entropy = float(-np.sum(fractions * np.log(fractions)) / np.log(hypothesis.component_count))
    put(5, entropy)
    put(6, _disconnected_ratio(hypothesis, leaves))

    group_contacts = [
        contact
        for contact in leaves.contacts
        if contact.left_leaf in component_by_leaf and contact.right_leaf in component_by_leaf
    ]
    total_edges = sum(contact.edge_count for contact in group_contacts)
    crossing_edges = sum(
        contact.edge_count
        for contact in group_contacts
        if component_by_leaf[contact.left_leaf] != component_by_leaf[contact.right_leaf]
    )
    put(7, crossing_edges / total_edges if total_edges else 0.0, total_edges > 0)
    for feature_index, within in ((8, True), (9, False)):
        selected = [
            contact
            for contact in group_contacts
            if (
                component_by_leaf[contact.left_leaf]
                == component_by_leaf[contact.right_leaf]
            )
            == within
            and contact.normal_available
        ]
        count = sum(contact.edge_count for contact in selected)
        put(
            feature_index,
            sum(contact.mean_abs_normal_dot * contact.edge_count for contact in selected) / count
            if count
            else 0.0,
            count > 0,
        )
    known = {True: 0, False: 0}
    same = {True: 0, False: 0}
    different = {True: 0, False: 0}
    for contact in group_contacts:
        within = component_by_leaf[contact.left_leaf] == component_by_leaf[contact.right_leaf]
        pair_known, pair_same, pair_different = _leaf_pair_counts(
            contact.left_leaf, contact.right_leaf, frames
        )
        known[within] += pair_known
        same[within] += pair_same
        different[within] += pair_different
    put(10, same[True] / known[True] if known[True] else 0.0, known[True] > 0)
    put(11, same[False] / known[False] if known[False] else 0.0, known[False] > 0)
    put(12, different[True] / known[True] if known[True] else 0.0, known[True] > 0)
    put(13, different[False] / known[False] if known[False] else 0.0, known[False] > 0)

    group_pixels = [
        frame.pixel_entity_ids[np.isin(frame.pixel_leaf_ids, group.leaf_ids)]
        for frame in frames
    ]
    pixel_count = sum(len(row) for row in group_pixels)
    unknown_count = sum(np.count_nonzero(row == 0) for row in group_pixels)
    put(14, unknown_count / pixel_count if pixel_count else 0.0, pixel_count > 0)
    known_leaves = sum(
        any(frame.entity_by_leaf[leaf_id] > 0 for frame in frames)
        for leaf_id in group.leaf_ids
    )
    put(15, known_leaves / len(group.leaf_ids))
    put(16, hypothesis.uncertain_fill_fraction)
    agreement = partition_agreement(hypothesis, frames)
    put(
        17,
        agreement.visible_pixel_weighted_agreement or 0.0,
        agreement.visible_pixel_weighted_agreement is not None,
    )
    group_diagonal = float(np.linalg.norm(xyz[group_rows].max(axis=0) - xyz[group_rows].min(axis=0)))
    if group_diagonal > 0.0:
        diagonals = [
            float(np.linalg.norm(xyz[rows].max(axis=0) - xyz[rows].min(axis=0)))
            for rows in component_rows
        ]
        put(18, np.mean(diagonals) / group_diagonal)
    dispersions = []
    for rows in component_rows:
        valid_rows = rows[surface_graph.normal_valid[rows]]
        if len(valid_rows):
            dispersions.append(
                1.0
                - float(
                    np.linalg.norm(surface_graph.normals[valid_rows].mean(axis=0))
                )
            )
    put(19, np.mean(dispersions) if dispersions else 0.0, bool(dispersions))
    return PartitionFeatures(values, available)


@dataclass(frozen=True)
class LocalPQTarget:
    quality: float
    matched_iou_sum: float
    true_positives: int
    false_positives: int
    false_negatives: int
    included_gt_count: int
    zero_denominator: bool


def local_pq_target(
    *,
    predicted_components: Sequence[set[int] | frozenset[int]],
    group_support: set[int] | frozenset[int],
    whole_gt_objects: Sequence[set[int] | frozenset[int]],
) -> LocalPQTarget:
    from scipy.optimize import linear_sum_assignment

    predictions = tuple(frozenset(component) for component in predicted_components)
    support = frozenset(group_support)
    if any(not component for component in predictions):
        raise ValueError("predicted components must be nonempty")
    union: set[int] = set()
    for component in predictions:
        if union.intersection(component):
            raise ValueError("predicted components must not overlap")
        union.update(component)
    if frozenset(union) != support:
        raise ValueError("predicted components must completely partition group support")
    gt_objects = tuple(
        frozenset(obj) for obj in whole_gt_objects if frozenset(obj).intersection(support)
    )
    matrix = np.zeros((len(predictions), len(gt_objects)), dtype=np.float64)
    for row, prediction in enumerate(predictions):
        for column, gt_object in enumerate(gt_objects):
            intersection = len(prediction.intersection(gt_object))
            union_size = len(prediction.union(gt_object))
            matrix[row, column] = intersection / union_size if union_size else 0.0
    if matrix.size:
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        accepted = matrix[rows, columns] > 0.5
        matched_sum = float(matrix[rows[accepted], columns[accepted]].sum())
        true_positives = int(np.count_nonzero(accepted))
    else:
        matched_sum = 0.0
        true_positives = 0
    false_positives = len(predictions) - true_positives
    false_negatives = len(gt_objects) - true_positives
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    return LocalPQTarget(
        quality=matched_sum / denominator if denominator else 0.0,
        matched_iou_sum=matched_sum,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        included_gt_count=len(gt_objects),
        zero_denominator=denominator == 0.0,
    )


def geometry_target_support_status(groups: Sequence[Mapping[str, Any]]) -> str:
    differing = [
        row
        for row in groups
        if len({float(value) for value in row.get("targets", ())}) > 1
    ]
    supported = len(differing) >= 20 and len(
        {str(row.get("scene_id")) for row in differing}
    ) >= 4
    return "SUPPORTED" if supported else "BLOCKED_PARTITION_TARGET_SUPPORT"


def _margin_rank(value: float | str) -> float:
    if value == "KEEP_ALL":
        return math.inf
    numeric = float(value)
    if numeric not in GEOMETRY_MARGINS:
        raise ValueError("geometry margin is outside the frozen CAL grid")
    return numeric


def select_geometry_margin(rows: Sequence[Mapping[str, Any]]) -> float | str:
    from .metric_order import metric_max
    normalized = []
    for row in rows:
        margin = row.get("margin")
        _margin_rank(margin)
        ap50 = float(row["mean_ap50"])
        ap75 = float(row["mean_ap75"])
        changed = int(row["changed_points"])
        if not np.isfinite((ap50, ap75)).all() or changed < 0:
            raise ValueError("geometry CAL metrics must be finite and nonnegative")
        normalized.append((margin, ap50, ap75, changed))
    if not normalized:
        raise ValueError("geometry CAL rows must not be empty")
    return metric_max(
        normalized,
        key=lambda row: (row[1], row[2], -row[3], _margin_rank(row[0])),
    )[0]


def apply_geometry_margin(score_differences: Any, margin: float | str) -> np.ndarray:
    values = np.asarray(score_differences, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("geometry score differences must be finite")
    threshold = _margin_rank(margin)
    if math.isinf(threshold):
        return np.zeros(values.shape, dtype=bool)
    return values > threshold


@dataclass(frozen=True)
class PartitionTrainingExample:
    scene_id: str
    group_id: str
    features: np.ndarray
    target: float

    def __post_init__(self) -> None:
        features = _readonly(self.features, np.float32)
        if not self.scene_id or not self.group_id or features.shape != (40,):
            raise ValueError("partition training example identity/features are invalid")
        if not np.isfinite(features).all() or not np.isfinite(self.target) or not 0.0 <= self.target <= 1.0:
            raise ValueError("partition training example values are invalid")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class GeometryTrainingResult:
    status: str
    checkpoint_epoch: int
    fit_epochs: int
    calibration_loss: float | None
    state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]
    scaler_state_dict: Mapping[str, Any]
    parameter_count: int


def _quality_model(torch: Any):
    return torch.nn.Sequential(
        torch.nn.Linear(40, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )


def _grouped_loss(model: Any, examples: Sequence[PartitionTrainingExample], torch: Any):
    by_group: dict[tuple[str, str], list[PartitionTrainingExample]] = {}
    for row in examples:
        by_group.setdefault((row.scene_id, row.group_id), []).append(row)
    scene_losses: dict[str, list[Any]] = {}
    for (scene_id, _group_id), rows in by_group.items():
        features = torch.from_numpy(np.stack([row.features for row in rows]))
        targets = torch.tensor([row.target for row in rows], dtype=torch.float32)
        scores = model(features).squeeze(1)
        loss = torch.mean((scores - targets) ** 2)
        pair_losses = []
        for better in range(len(rows)):
            for worse in range(len(rows)):
                if rows[better].target - rows[worse].target >= 0.05:
                    pair_losses.append(
                        torch.nn.functional.softplus(
                            -(scores[better] - scores[worse])
                        )
                    )
        if pair_losses:
            loss = loss + 0.1 * torch.mean(torch.stack(pair_losses))
        scene_losses.setdefault(scene_id, []).append(loss)
    return torch.mean(
        torch.stack(
            [torch.mean(torch.stack(losses)) for losses in scene_losses.values()]
        )
    )


def train_geometry_head(
    fit_examples: Sequence[PartitionTrainingExample],
    *,
    cal_examples: Sequence[PartitionTrainingExample] = (),
    max_epochs: int = 100,
) -> GeometryTrainingResult:
    import torch

    fit = tuple(fit_examples)
    cal = tuple(cal_examples)
    if not fit or not 1 <= max_epochs <= 100:
        raise ValueError("geometry training requires FIT examples and 1 to 100 epochs")
    torch.manual_seed(17)
    model = _quality_model(torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    best_state = None
    best_optimizer = None
    best_epoch = 0
    best_loss = math.inf
    stale = 0
    fit_epochs = 0
    group_keys = sorted({(row.scene_id, row.group_id) for row in fit})
    generator = torch.Generator().manual_seed(17)
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(group_keys), generator=generator).tolist()
        for start in range(0, len(group_keys), 64):
            batch_keys = {
                group_keys[index] for index in permutation[start : start + 64]
            }
            batch = tuple(
                row for row in fit if (row.scene_id, row.group_id) in batch_keys
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _grouped_loss(model, batch, torch)
            loss.backward()
            optimizer.step()
        fit_epochs = epoch
        if cal and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                cal_loss = float(_grouped_loss(model, cal, torch).item())
            if cal_loss < best_loss:
                best_loss = cal_loss
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
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        status = "UNCALIBRATED"
        calibration_loss = None
    else:
        status = "COMPLETE"
        calibration_loss = best_loss
    return GeometryTrainingResult(
        status,
        best_epoch,
        fit_epochs,
        calibration_loss,
        MappingProxyType(best_state),
        MappingProxyType(best_optimizer),
        MappingProxyType({}),
        40 * 32 + 32 + 32 + 1,
    )


def predict_partition_quality(state_dict: Mapping[str, Any], features: Any) -> np.ndarray:
    import torch

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 40 or not np.isfinite(rows).all():
        raise ValueError("geometry inference requires a finite Nx40 matrix")
    model = _quality_model(torch)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(rows)).squeeze(1).numpy()
