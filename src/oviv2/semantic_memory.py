from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


_MINIMUM_PROTOTYPE_RESULTANT_RATIO = float(np.sqrt(np.finfo(np.float64).eps))


def _integer(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field_name} must be {qualifier}")
    return normalized


def _number(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{field_name} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _bounded_number(value: object, field_name: str, lower: float, upper: float) -> float:
    normalized = _number(value, field_name)
    if not lower <= normalized <= upper:
        raise ValueError(f"{field_name} must lie in [{lower:g}, {upper:g}]")
    return normalized


def _unit_tuple(value: object, field_name: str = "feature") -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a numeric vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one dimensional")
    if array.size == 0:
        raise ValueError(f"{field_name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        raise ValueError(f"{field_name} must be nonzero")
    scaled = array / scale
    unit = scaled / float(np.linalg.norm(scaled))
    return tuple(float(item) for item in unit)


def _model_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model_id must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("model_id must be non-empty")
    return normalized


def _direction_tuple(value: object) -> tuple[float, float, float]:
    try:
        direction = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("view_direction_xyz must be a numeric vector") from exc
    if direction.shape != (3,):
        raise ValueError("view_direction_xyz must be three dimensional")
    return _unit_tuple(direction, "view_direction_xyz")  # type: ignore[return-value]


@dataclass(frozen=True)
class SparseClassPosterior:
    log_evidence: tuple[tuple[int, float], ...]
    effective_support: float

    def __post_init__(self) -> None:
        try:
            entries = tuple(self.log_evidence)
        except TypeError as exc:
            raise TypeError("log_evidence must be an iterable of pairs") from exc
        normalized: list[tuple[int, float]] = []
        seen: set[int] = set()
        for entry in entries:
            try:
                semantic_id, log_value = entry
            except (TypeError, ValueError) as exc:
                raise ValueError("log_evidence entries must be pairs") from exc
            semantic_id = _integer(semantic_id, "semantic_id", minimum=1)
            if semantic_id in seen:
                raise ValueError("log_evidence semantic IDs must be unique")
            seen.add(semantic_id)
            normalized.append((semantic_id, _number(log_value, "log_evidence")))
        effective_support = _number(self.effective_support, "effective_support")
        if effective_support < 0.0:
            raise ValueError("effective_support must be non-negative")
        object.__setattr__(self, "log_evidence", tuple(sorted(normalized)))
        object.__setattr__(self, "effective_support", effective_support)

    @classmethod
    def empty(cls) -> SparseClassPosterior:
        return cls((), 0.0)

    def update_label(
        self,
        semantic_id: int,
        confidence: float,
        quality: float,
    ) -> SparseClassPosterior:
        semantic_id = _integer(semantic_id, "semantic_id", minimum=1)
        confidence = _bounded_number(confidence, "confidence", 0.0, 1.0)
        quality = _bounded_number(quality, "quality", 0.0, 1.0)
        if confidence == 0.0 or quality == 0.0:
            return self
        values = dict(self.log_evidence)
        incoming_log = float(np.log(confidence) + np.log(quality))
        values[semantic_id] = float(
            np.logaddexp(values.get(semantic_id, -np.inf), incoming_log)
        )
        mass = confidence * quality
        return SparseClassPosterior(
            tuple(values.items()),
            self.effective_support + mass,
        )

    @property
    def probabilities(self) -> tuple[tuple[int, float], ...]:
        if not self.log_evidence:
            return ()
        logs = np.asarray([value for _, value in self.log_evidence], dtype=np.float64)
        shifted = np.exp(logs - float(np.max(logs)))
        probabilities = shifted / float(np.sum(shifted))
        return tuple(
            (semantic_id, float(probability))
            for (semantic_id, _), probability in zip(self.log_evidence, probabilities)
        )

    @property
    def best_semantic_id(self) -> int:
        return min(
            self.probabilities,
            key=lambda item: (-item[1], item[0]),
            default=(0, 0.0),
        )[0]

    @property
    def entropy(self) -> float:
        if len(self.log_evidence) <= 1:
            return 0.0
        probabilities = np.asarray(
            [probability for _, probability in self.probabilities],
            dtype=np.float64,
        )
        positive = probabilities[probabilities > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    @property
    def margin(self) -> float:
        ranked = sorted(
            (probability for _, probability in self.probabilities),
            reverse=True,
        )
        if not ranked:
            return 0.0
        if len(ranked) == 1:
            return ranked[0]
        return float(ranked[0] - ranked[1])


@dataclass(frozen=True)
class FeaturePrototype:
    vector: tuple[float, ...]
    model_id: str
    support: float
    observation_count: int

    def __post_init__(self) -> None:
        vector = _unit_tuple(self.vector)
        model_id = _model_id(self.model_id)
        support = _number(self.support, "support")
        if support <= 0.0:
            raise ValueError("support must be positive")
        observation_count = _integer(
            self.observation_count,
            "observation_count",
            minimum=1,
        )
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "observation_count", observation_count)


@dataclass(frozen=True)
class FeaturePrototypeBank:
    max_prototypes: int = 3
    merge_cosine: float = 0.90
    prototypes: tuple[FeaturePrototype, ...] = ()

    def __post_init__(self) -> None:
        max_prototypes = _integer(
            self.max_prototypes,
            "max_prototypes",
            minimum=1,
        )
        merge_cosine = _bounded_number(
            self.merge_cosine,
            "merge_cosine",
            -1.0,
            1.0,
        )
        try:
            prototypes = tuple(self.prototypes)
        except TypeError as exc:
            raise TypeError("prototypes must be an iterable") from exc
        if any(not isinstance(item, FeaturePrototype) for item in prototypes):
            raise TypeError("prototypes must contain FeaturePrototype values")
        if len(prototypes) > max_prototypes:
            raise ValueError("prototypes cannot exceed max_prototypes")
        prototypes = tuple(
            sorted(
                prototypes,
                key=lambda item: (-item.support, item.model_id, item.vector),
            )
        )
        object.__setattr__(self, "max_prototypes", max_prototypes)
        object.__setattr__(self, "merge_cosine", merge_cosine)
        object.__setattr__(self, "prototypes", prototypes)

    def update(
        self,
        feature: Iterable[float],
        model_id: str,
        quality: float,
    ) -> FeaturePrototypeBank:
        vector = _unit_tuple(feature)
        model_id = _model_id(model_id)
        quality = _number(quality, "quality")
        if quality <= 0.0:
            raise ValueError("quality must be positive")

        values = list(self.prototypes)
        compatible = [
            (float(np.dot(vector, item.vector)), index)
            for index, item in enumerate(values)
            if item.model_id == model_id and len(item.vector) == len(vector)
        ]
        if compatible:
            best_cosine, best_index = max(
                compatible,
                key=lambda item: (item[0], -item[1]),
            )
        else:
            best_cosine, best_index = -np.inf, -1

        should_merge = best_cosine >= self.merge_cosine
        if should_merge:
            previous = values[best_index]
            scale = max(previous.support, quality)
            previous_weight = previous.support / scale
            incoming_weight = quality / scale
            merged = (
                np.asarray(previous.vector, dtype=np.float64) * previous_weight
                + np.asarray(vector, dtype=np.float64) * incoming_weight
            )
            resultant_ratio = float(np.linalg.norm(merged)) / (
                previous_weight + incoming_weight
            )
            should_merge = resultant_ratio > _MINIMUM_PROTOTYPE_RESULTANT_RATIO
            if should_merge:
                support = previous.support + quality
                if not np.isfinite(support):
                    raise ValueError("merged prototype support must be finite")
                values[best_index] = FeaturePrototype(
                    vector=_unit_tuple(merged),
                    model_id=model_id,
                    support=support,
                    observation_count=previous.observation_count + 1,
                )
        if not should_merge:
            incoming = FeaturePrototype(vector, model_id, quality, 1)
            if len(values) < self.max_prototypes:
                values.append(incoming)
            else:
                weakest = min(
                    range(len(values)),
                    key=lambda index: (values[index].support, index),
                )
                if quality <= values[weakest].support:
                    return self
                values[weakest] = incoming
        values.sort(key=lambda item: (-item.support, item.model_id, item.vector))
        return replace(self, prototypes=tuple(values))

    def maximum_cosine(self, feature: Iterable[float], model_id: str) -> float | None:
        vector = _unit_tuple(feature)
        model_id = _model_id(model_id)
        similarities = [
            float(np.dot(vector, item.vector))
            for item in self.prototypes
            if item.model_id == model_id and len(item.vector) == len(vector)
        ]
        return max(similarities) if similarities else None


@dataclass(frozen=True)
class InformativeView:
    observation_id: int
    frame_id: int
    visible_pixel_count: int
    quality: float
    view_direction_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        observation_id = _integer(
            self.observation_id,
            "observation_id",
            minimum=0,
        )
        frame_id = _integer(self.frame_id, "frame_id", minimum=0)
        visible_pixel_count = _integer(
            self.visible_pixel_count,
            "visible_pixel_count",
            minimum=0,
        )
        quality = _bounded_number(self.quality, "quality", 0.0, 1.0)
        direction = _direction_tuple(self.view_direction_xyz)
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "visible_pixel_count", visible_pixel_count)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "view_direction_xyz", direction)


@dataclass(frozen=True)
class InformativeViewBank:
    max_views: int = 10
    minimum_novelty_cosine: float = 0.10
    views: tuple[InformativeView, ...] = ()

    def __post_init__(self) -> None:
        max_views = _integer(self.max_views, "max_views", minimum=1)
        minimum_novelty_cosine = _bounded_number(
            self.minimum_novelty_cosine,
            "minimum_novelty_cosine",
            0.0,
            2.0,
        )
        try:
            views = tuple(self.views)
        except TypeError as exc:
            raise TypeError("views must be an iterable") from exc
        if any(not isinstance(item, InformativeView) for item in views):
            raise TypeError("views must contain InformativeView values")
        if len(views) > max_views:
            raise ValueError("views cannot exceed max_views")
        observation_ids = [item.observation_id for item in views]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("view observation IDs must be unique")
        views = tuple(
            sorted(views, key=lambda item: (-item.quality, item.observation_id))
        )
        object.__setattr__(self, "max_views", max_views)
        object.__setattr__(
            self,
            "minimum_novelty_cosine",
            minimum_novelty_cosine,
        )
        object.__setattr__(self, "views", views)

    def update(self, view: InformativeView) -> InformativeViewBank:
        if not isinstance(view, InformativeView):
            raise TypeError("view must be an InformativeView")
        if any(item.observation_id == view.observation_id for item in self.views):
            return self

        values = list(self.views)
        similarities = [
            float(np.dot(view.view_direction_xyz, item.view_direction_xyz))
            for item in values
        ]
        if similarities:
            nearest = max(
                range(len(similarities)),
                key=lambda index: (similarities[index], -index),
            )
            novelty = 1.0 - similarities[nearest]
            if novelty < self.minimum_novelty_cosine:
                if view.quality <= values[nearest].quality:
                    return self
                values[nearest] = view
            elif len(values) < self.max_views:
                values.append(view)
            else:
                weakest = min(
                    range(len(values)),
                    key=lambda index: (values[index].quality, index),
                )
                if view.quality <= values[weakest].quality:
                    return self
                values[weakest] = view
        else:
            values.append(view)
        values.sort(key=lambda item: (-item.quality, item.observation_id))
        return replace(self, views=tuple(values))
