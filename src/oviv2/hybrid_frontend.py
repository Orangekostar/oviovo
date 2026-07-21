from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .dense_semantics import DenseSemanticFrame


_VARIANTS = frozenset({"sam_labeled", "yolo_novel_sam", "quota_nms_ensemble"})
_THRESHOLD_FIELDS = (
    "yolo_match_iou",
    "yolo_match_coverage",
    "novel_iou",
    "duplicate_iou",
    "minimum_area_fraction",
    "compact_rescue_minimum_area_fraction",
    "compact_rescue_minimum_confidence",
    "maximum_area_fraction",
    "minimum_valid_depth_fraction",
    "minimum_dense_probability",
    "minimum_dense_margin",
    "maximum_structure_probability",
)
_DIAGNOSTIC_KEYS = (
    "accepted_yolo",
    "accepted_inherited_sam",
    "accepted_novel_sam",
    "accepted_compact_rescue",
    "rejected_area",
    "rejected_depth",
    "rejected_structure",
    "rejected_duplicate",
    "rejected_not_novel",
    "rejected_unlabeled",
    "rejected_class_cap",
    "rejected_global_cap",
    "rejected_compact_rescue_cap",
)


def _unit_interval(value: float, field_name: str) -> float:
    message = f"{field_name} must be a finite real number in [0, 1]"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(message)
    return normalized


def _positive_integer(value: int, field_name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{field_name} must be a positive non-bool integer")
    return int(value)


def _non_negative_integer(value: int, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative non-bool integer")
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative non-bool integer") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative non-bool integer")
    return normalized


def _numeric_array(value: np.ndarray, field_name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must have an integer, unsigned, or float dtype")
    return array


def _binary_mask_array(value: np.ndarray, field_name: str) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a boolean or binary numeric array") from exc
    if mask.dtype.kind == "b":
        return mask
    if mask.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a boolean or binary numeric array")
    try:
        valid = np.isfinite(mask) & ((mask == 0) | (mask == 1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must contain only finite binary values") from exc
    if not np.all(valid):
        raise ValueError(f"{field_name} must contain only finite binary values")
    return mask


def _immutable_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _normalized_features(value: np.ndarray) -> np.ndarray:
    try:
        features = np.array(value, dtype=np.longdouble, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image_features must be a finite numeric array") from exc
    if not np.all(np.isfinite(features)):
        raise ValueError("image_features must contain only finite values")

    scales = np.max(np.abs(features), axis=1)
    if np.any(scales == 0.0):
        raise ValueError("image_features rows must be nonzero")
    scaled = features / scales[:, None]
    squared_norms = np.sum(scaled * scaled, axis=1, dtype=np.longdouble)
    norms = np.sqrt(squared_norms)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("image_features rows must be finite and nonzero")

    normalized = np.ascontiguousarray(scaled / norms[:, None], dtype=np.float32)
    output_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    if (
        not np.all(np.isfinite(normalized))
        or np.any(output_norms == 0.0)
        or not np.allclose(output_norms, 1.0, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("normalized image_features rows must be finite, nonzero, and unit length")
    return _immutable_array(normalized, np.dtype(np.float32))


@dataclass(frozen=True)
class HybridFrontendConfig:
    variant: str
    yolo_match_iou: float = 0.50
    yolo_match_coverage: float = 0.60
    novel_iou: float = 0.80
    duplicate_iou: float = 0.85
    minimum_area_fraction: float = 0.0001
    compact_rescue_minimum_area_fraction: float = 0.00002
    compact_rescue_minimum_confidence: float = 0.60
    maximum_area_fraction: float = 0.50
    minimum_valid_depth_fraction: float = 0.50
    minimum_dense_probability: float = 0.25
    minimum_dense_margin: float = 0.05
    maximum_structure_probability: float = 0.50
    maximum_proposals: int = 64
    maximum_per_class: int = 12
    maximum_compact_rescues: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        for field_name in _THRESHOLD_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )
        if self.minimum_area_fraction > self.maximum_area_fraction:
            raise ValueError(
                "minimum_area_fraction must not exceed maximum_area_fraction"
            )
        if (
            self.variant == "quota_nms_ensemble"
            and self.compact_rescue_minimum_area_fraction
            >= self.minimum_area_fraction
        ):
            raise ValueError(
                "compact_rescue_minimum_area_fraction must be less than "
                "minimum_area_fraction"
            )
        for field_name in (
            "maximum_proposals",
            "maximum_per_class",
            "maximum_compact_rescues",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, eq=False)
class FrontendBatch:
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    labels: tuple[str, ...]
    image_features: np.ndarray

    def __post_init__(self) -> None:
        masks = _binary_mask_array(self.masks, "masks")
        boxes = _numeric_array(self.boxes_xyxy, "boxes_xyxy")
        confidences = _numeric_array(self.confidences, "confidences")
        features = _numeric_array(self.image_features, "image_features")

        if masks.ndim != 3:
            raise ValueError("masks must have shape (N, H, W)")
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("boxes_xyxy must have shape (N, 4)")
        if confidences.ndim != 1:
            raise ValueError("confidences must have shape (N,)")
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("image_features must have shape (N, D) with D > 0")
        if not isinstance(self.labels, tuple):
            raise ValueError("labels must be a tuple of non-empty strings")

        batch_size = masks.shape[0]
        if not (
            boxes.shape[0]
            == confidences.shape[0]
            == len(self.labels)
            == features.shape[0]
            == batch_size
        ):
            raise ValueError("frontend batch dimensions must agree")

        if not np.all(np.isfinite(boxes)):
            raise ValueError("boxes_xyxy must contain only finite values")
        if not np.all(np.isfinite(confidences)):
            raise ValueError("confidences must contain only finite values")
        if np.any(confidences < 0.0) or np.any(confidences > 1.0):
            raise ValueError("confidences must lie in [0, 1]")
        if not np.all(np.isfinite(features)):
            raise ValueError("image_features must contain only finite values")

        labels: list[str] = []
        for label in self.labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("labels must contain only non-empty strings")
            labels.append(label.strip())

        try:
            immutable_masks = _immutable_array(masks, np.dtype(np.bool_))
            immutable_boxes = _immutable_array(boxes, np.dtype(np.float32))
            immutable_confidences = _immutable_array(confidences, np.dtype(np.float32))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frontend arrays cannot be converted to required dtypes") from exc
        if not np.all(np.isfinite(immutable_boxes)):
            raise ValueError("boxes_xyxy must remain finite as float32")
        if not np.all(np.isfinite(immutable_confidences)):
            raise ValueError("confidences must remain finite as float32")

        object.__setattr__(self, "masks", immutable_masks)
        object.__setattr__(self, "boxes_xyxy", immutable_boxes)
        object.__setattr__(self, "confidences", immutable_confidences)
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "image_features", _normalized_features(features))


@dataclass(frozen=True, eq=False)
class HybridFrameResult:
    batch: FrontendBatch
    diagnostics: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, FrontendBatch):
            raise ValueError("batch must be a FrontendBatch")
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")

        diagnostics: dict[str, int] = {}
        for key, value in self.diagnostics.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("diagnostics keys must be non-empty strings")
            diagnostics[key] = _non_negative_integer(value, f"diagnostics[{key!r}]")
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))


@dataclass(frozen=True)
class MaskOverlap:
    intersection: int
    iou: float
    left_coverage: float
    right_coverage: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intersection",
            _non_negative_integer(self.intersection, "intersection"),
        )
        for field_name in ("iou", "left_coverage", "right_coverage"):
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class DenseMaskLabel:
    semantic_id: int
    probability: float
    margin: float
    structure_probability: float
    valid_depth_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_id",
            _non_negative_integer(self.semantic_id, "semantic_id"),
        )
        for field_name in (
            "probability",
            "margin",
            "structure_probability",
            "valid_depth_fraction",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_interval(getattr(self, field_name), field_name),
            )
        if self.margin > self.probability:
            raise ValueError("margin must not exceed probability")


def mask_overlap(left: np.ndarray, right: np.ndarray) -> MaskOverlap:
    left_mask = _binary_mask_array(left, "left mask")
    right_mask = _binary_mask_array(right, "right mask")
    if left_mask.ndim != 2 or left_mask.shape != right_mask.shape:
        raise ValueError("masks must be two dimensional with equal shape")

    left_mask = np.asarray(left_mask, dtype=bool)
    right_mask = np.asarray(right_mask, dtype=bool)

    intersection = int(np.count_nonzero(left_mask & right_mask))
    left_size = int(np.count_nonzero(left_mask))
    right_size = int(np.count_nonzero(right_mask))
    union = left_size + right_size - intersection
    return MaskOverlap(
        intersection=intersection,
        iou=float(intersection / union) if union else 0.0,
        left_coverage=float(intersection / left_size) if left_size else 0.0,
        right_coverage=float(intersection / right_size) if right_size else 0.0,
    )


def aggregate_dense_mask(
    mask: np.ndarray,
    valid_depth: np.ndarray,
    dense: DenseSemanticFrame,
    *,
    structure_ids: set[int] | frozenset[int],
) -> DenseMaskLabel:
    mask_array = _binary_mask_array(mask, "mask")
    valid_depth_array = _binary_mask_array(valid_depth, "valid_depth")
    if (
        mask_array.shape != dense.image_shape
        or valid_depth_array.shape != dense.image_shape
    ):
        raise ValueError("mask and valid_depth must match dense image_shape")

    if not isinstance(structure_ids, (set, frozenset)):
        raise ValueError("structure_ids must be a set or frozenset")
    normalized_structure_ids: set[int] = set()
    for value in structure_ids:
        normalized = _positive_integer(value, "structure_ids element")
        if normalized > dense.class_count:
            raise ValueError("structure_ids elements must not exceed dense class_count")
        normalized_structure_ids.add(normalized)

    stride = dense.sample_stride
    sampled_mask = np.asarray(mask_array[::stride, ::stride], dtype=bool)
    sampled_depth = np.asarray(valid_depth_array[::stride, ::stride], dtype=bool)
    sampled_shape = dense.class_ids.shape[:2]
    if sampled_mask.shape != sampled_shape or sampled_depth.shape != sampled_shape:
        raise ValueError("sampled mask and valid_depth must match dense class_ids shape")

    selected_count = int(np.count_nonzero(sampled_mask))
    if selected_count == 0:
        return DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)

    valid_selected = sampled_mask & sampled_depth
    valid_selected_count = int(np.count_nonzero(valid_selected))
    if valid_selected_count == 0:
        return DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)

    support = np.zeros(dense.class_count + 1, dtype=np.float64)
    for slot in range(dense.class_ids.shape[2]):
        slot_ids = dense.class_ids[..., slot][valid_selected]
        positive = slot_ids > 0
        if np.any(positive):
            slot_probabilities = dense.probabilities[..., slot][valid_selected]
            np.add.at(support, slot_ids[positive], slot_probabilities[positive])
    support /= valid_selected_count

    structure_probability = float(
        np.clip(
            sum(support[class_id] for class_id in normalized_structure_ids),
            0.0,
            1.0,
        )
    )
    winning_id = 0
    winning_support = 0.0
    runner_up_support = 0.0
    for class_id in range(1, dense.class_count + 1):
        if class_id in normalized_structure_ids:
            continue
        class_support = float(support[class_id])
        if class_support > winning_support:
            runner_up_support = winning_support
            winning_id = class_id
            winning_support = class_support
        elif class_support > runner_up_support:
            runner_up_support = class_support

    if winning_support <= 0.0:
        return DenseMaskLabel(
            semantic_id=0,
            probability=0.0,
            margin=0.0,
            structure_probability=structure_probability,
            valid_depth_fraction=valid_selected_count / selected_count,
        )
    probability = float(np.clip(winning_support, 0.0, 1.0))
    margin = float(np.clip(winning_support - runner_up_support, 0.0, 1.0))
    return DenseMaskLabel(
        semantic_id=winning_id,
        probability=probability,
        margin=margin,
        structure_probability=structure_probability,
        valid_depth_fraction=valid_selected_count / selected_count,
    )


@dataclass(frozen=True)
class _SelectedProposal:
    source: FrontendBatch
    source_index: int
    label: str
    confidence: float
    area: int
    kind: str
    compact_rescue: bool = False


def _validated_class_names(
    class_names: tuple[str, ...], class_count: int
) -> tuple[str, ...]:
    if not isinstance(class_names, tuple) or len(class_names) != class_count:
        raise ValueError("class_names must be a tuple matching dense class_count")
    normalized: list[str] = []
    for name in class_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("class_names must contain non-empty strings")
        normalized.append(name.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_names must be unique after trimming")
    return tuple(normalized)


def _empty_frontend_batch(
    image_shape: tuple[int, int], feature_dimension: int
) -> FrontendBatch:
    return FrontendBatch(
        masks=np.empty((0, *image_shape), dtype=bool),
        boxes_xyxy=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        labels=(),
        image_features=np.empty((0, feature_dimension), dtype=np.float32),
    )


def _build_selected_batch(
    proposals: list[_SelectedProposal],
    image_shape: tuple[int, int],
    feature_dimension: int,
) -> FrontendBatch:
    if not proposals:
        return _empty_frontend_batch(image_shape, feature_dimension)
    return FrontendBatch(
        masks=np.stack(
            [proposal.source.masks[proposal.source_index] for proposal in proposals]
        ),
        boxes_xyxy=np.stack(
            [
                proposal.source.boxes_xyxy[proposal.source_index]
                for proposal in proposals
            ]
        ),
        confidences=np.asarray(
            [proposal.confidence for proposal in proposals], dtype=np.float32
        ),
        labels=tuple(proposal.label for proposal in proposals),
        image_features=np.stack(
            [
                proposal.source.image_features[proposal.source_index]
                for proposal in proposals
            ]
        ),
    )


def _overlap_strength(overlap: MaskOverlap) -> float:
    return max(overlap.iou, overlap.left_coverage, overlap.right_coverage)


def _sam_candidate_sort_key(proposal: _SelectedProposal) -> tuple[int, float, int, int]:
    if proposal.kind == "inherited" and not proposal.compact_rescue:
        tier = 0
    elif proposal.compact_rescue:
        tier = 1
    else:
        tier = 2
    return (
        tier,
        -proposal.confidence,
        -proposal.area,
        proposal.source_index,
    )


def select_hybrid_proposals(
    *,
    yolo: FrontendBatch,
    sam: FrontendBatch,
    dense: DenseSemanticFrame,
    valid_depth: np.ndarray,
    class_names: tuple[str, ...],
    structure_ids: set[int] | frozenset[int],
    config: HybridFrontendConfig,
) -> HybridFrameResult:
    if not isinstance(yolo, FrontendBatch):
        raise ValueError("yolo must be a FrontendBatch")
    if not isinstance(sam, FrontendBatch):
        raise ValueError("sam must be a FrontendBatch")
    if not isinstance(dense, DenseSemanticFrame):
        raise ValueError("dense must be a DenseSemanticFrame")
    if not isinstance(config, HybridFrontendConfig):
        raise ValueError("config must be a HybridFrontendConfig")

    image_shape = dense.image_shape
    if yolo.masks.shape[1:] != image_shape or sam.masks.shape[1:] != image_shape:
        raise ValueError("yolo, sam, and dense image shapes must agree")
    feature_dimension = yolo.image_features.shape[1]
    if sam.image_features.shape[1] != feature_dimension:
        raise ValueError("yolo and sam feature dimensions must agree")

    valid_depth_array = _binary_mask_array(valid_depth, "valid_depth")
    if valid_depth_array.shape != image_shape:
        raise ValueError("valid_depth must match dense image_shape")
    normalized_class_names = _validated_class_names(class_names, dense.class_count)

    # This zero-mask call centralizes structure_ids validation in the dense aggregator.
    aggregate_dense_mask(
        np.zeros(image_shape, dtype=bool),
        valid_depth_array,
        dense,
        structure_ids=structure_ids,
    )
    normalized_structure_ids = {int(value) for value in structure_ids}
    structure_names = {
        normalized_class_names[class_id - 1]
        for class_id in normalized_structure_ids
    }

    diagnostics = {key: 0 for key in _DIAGNOSTIC_KEYS}
    yolo_candidates: list[_SelectedProposal] = []
    if config.variant != "sam_labeled":
        for index in range(yolo.masks.shape[0]):
            yolo_candidates.append(
                _SelectedProposal(
                    source=yolo,
                    source_index=index,
                    label=yolo.labels[index],
                    confidence=float(yolo.confidences[index]),
                    area=int(np.count_nonzero(yolo.masks[index])),
                    kind="yolo",
                )
            )

    yolo_candidates.sort(key=lambda proposal: (-proposal.confidence, proposal.source_index))
    selected: list[_SelectedProposal] = []
    class_counts: dict[str, int] = {}
    if config.variant == "quota_nms_ensemble":
        for proposal in yolo_candidates:
            count = class_counts.get(proposal.label, 0)
            if count >= config.maximum_per_class:
                diagnostics["rejected_class_cap"] += 1
                continue
            if len(selected) >= config.maximum_proposals:
                diagnostics["rejected_global_cap"] += 1
                continue
            selected.append(proposal)
            class_counts[proposal.label] = count + 1
    else:
        selected.extend(yolo_candidates)

    sam_candidates: list[_SelectedProposal] = []
    image_area = image_shape[0] * image_shape[1]

    for sam_index, sam_mask in enumerate(sam.masks):
        area = int(np.count_nonzero(sam_mask))
        area_fraction = area / image_area
        dense_label = aggregate_dense_mask(
            sam_mask,
            valid_depth_array,
            dense,
            structure_ids=structure_ids,
        )
        overlaps = [mask_overlap(sam_mask, yolo_mask) for yolo_mask in yolo.masks]

        compact_rescue = (
            config.variant == "quota_nms_ensemble"
            and config.compact_rescue_minimum_area_fraction
            <= area_fraction
            < config.minimum_area_fraction
        )
        if (
            area_fraction > config.maximum_area_fraction
            or (area_fraction < config.minimum_area_fraction and not compact_rescue)
        ):
            diagnostics["rejected_area"] += 1
            continue
        if dense_label.valid_depth_fraction < config.minimum_valid_depth_fraction:
            diagnostics["rejected_depth"] += 1
            continue
        if dense_label.structure_probability > config.maximum_structure_probability:
            diagnostics["rejected_structure"] += 1
            continue
        maximum_yolo_iou = max((overlap.iou for overlap in overlaps), default=0.0)
        reliable_yolo_indices = [
            index
            for index, overlap in enumerate(overlaps)
            if (
                overlap.iou >= config.yolo_match_iou
                or overlap.left_coverage >= config.yolo_match_coverage
                or overlap.right_coverage >= config.yolo_match_coverage
            )
        ]

        if config.variant == "yolo_novel_sam" and maximum_yolo_iou >= config.novel_iou:
            rejection_key = (
                "rejected_duplicate"
                if maximum_yolo_iou >= config.duplicate_iou
                else "rejected_not_novel"
            )
            diagnostics[rejection_key] += 1
            continue

        if reliable_yolo_indices:
            best_yolo_index = max(
                reliable_yolo_indices,
                key=lambda index: (
                    _overlap_strength(overlaps[index]),
                    overlaps[index].iou,
                    float(yolo.confidences[index]),
                    -index,
                ),
            )
            best_overlap = overlaps[best_yolo_index]
            inherited_confidence = float(
                np.clip(
                    float(yolo.confidences[best_yolo_index])
                    * np.sqrt(_overlap_strength(best_overlap)),
                    0.0,
                    1.0,
                )
            )
            if (
                compact_rescue
                and inherited_confidence
                < config.compact_rescue_minimum_confidence
            ):
                diagnostics["rejected_area"] += 1
                continue
            sam_candidates.append(
                _SelectedProposal(
                    source=sam,
                    source_index=sam_index,
                    label=yolo.labels[best_yolo_index],
                    confidence=inherited_confidence,
                    area=area,
                    kind="inherited",
                    compact_rescue=compact_rescue,
                )
            )
            continue

        semantic_id = dense_label.semantic_id
        fallback_label = (
            normalized_class_names[semantic_id - 1] if semantic_id > 0 else None
        )
        if (
            semantic_id <= 0
            or semantic_id in normalized_structure_ids
            or fallback_label in structure_names
            or dense_label.probability < config.minimum_dense_probability
            or dense_label.margin < config.minimum_dense_margin
        ):
            diagnostics["rejected_unlabeled"] += 1
            continue
        assert fallback_label is not None
        if (
            compact_rescue
            and dense_label.probability
            < config.compact_rescue_minimum_confidence
        ):
            diagnostics["rejected_area"] += 1
            continue
        sam_candidates.append(
            _SelectedProposal(
                source=sam,
                source_index=sam_index,
                label=fallback_label,
                confidence=dense_label.probability,
                area=area,
                kind="fallback",
                compact_rescue=compact_rescue,
            )
        )

    sam_candidates.sort(key=_sam_candidate_sort_key)
    accepted_sam_masks: list[np.ndarray] = (
        [proposal.source.masks[proposal.source_index] for proposal in selected]
        if config.variant == "quota_nms_ensemble"
        else []
    )
    accepted_sam_candidates: list[_SelectedProposal] = []
    compact_rescue_count = 0
    for proposal in sam_candidates:
        if config.variant == "quota_nms_ensemble":
            count = class_counts.get(proposal.label, 0)
            if count >= config.maximum_per_class:
                diagnostics["rejected_class_cap"] += 1
                continue
            if len(selected) >= config.maximum_proposals:
                diagnostics["rejected_global_cap"] += 1
                continue
            if (
                proposal.compact_rescue
                and compact_rescue_count >= config.maximum_compact_rescues
            ):
                diagnostics["rejected_compact_rescue_cap"] += 1
                continue
        proposal_mask = proposal.source.masks[proposal.source_index]
        if any(
            mask_overlap(proposal_mask, accepted_mask).iou >= config.duplicate_iou
            for accepted_mask in accepted_sam_masks
        ):
            diagnostics["rejected_duplicate"] += 1
            continue
        accepted_sam_masks.append(proposal_mask)
        accepted_sam_candidates.append(proposal)
        if config.variant == "quota_nms_ensemble":
            selected.append(proposal)
            class_counts[proposal.label] = count + 1
            if proposal.compact_rescue:
                compact_rescue_count += 1

    if config.variant != "quota_nms_ensemble":
        selected.extend(accepted_sam_candidates)
    if config.variant != "quota_nms_ensemble" and len(selected) > config.maximum_proposals:
        diagnostics["rejected_global_cap"] = len(selected) - config.maximum_proposals
        selected = selected[: config.maximum_proposals]

    diagnostics["accepted_yolo"] = sum(
        proposal.kind == "yolo" for proposal in selected
    )
    diagnostics["accepted_inherited_sam"] = sum(
        proposal.kind == "inherited" for proposal in selected
    )
    diagnostics["accepted_novel_sam"] = sum(
        proposal.kind == "fallback" for proposal in selected
    )
    diagnostics["accepted_compact_rescue"] = sum(
        proposal.compact_rescue for proposal in selected
    )
    return HybridFrameResult(
        batch=_build_selected_batch(selected, image_shape, feature_dimension),
        diagnostics=diagnostics,
    )
