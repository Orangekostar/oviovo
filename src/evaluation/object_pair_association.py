"""Fixed-candidate object association for real two-visit OVI views."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.ovi_pair_views import OviObjectPairView
from src.oviv2.temporal_pair_reasoner import validate_query_evidence
from src.oviv2.two_visit_contracts import NeuralSampleMap, TemporalQueryEvidence

MethodId = Literal["G_full", "G_supported", "F_obj", "R_obj"]
PredictionState = Literal[
    "persistent_static", "persistent_moved", "unmatched_t0", "unmatched_t1"
]
CandidateId = tuple[int, str]
CandidatePool = tuple[tuple[CandidateId, ...], tuple[CandidateId, ...]]
EvidenceValue = str | int | float | bool

_METHOD_IDS = frozenset({"G_full", "G_supported", "F_obj", "R_obj"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INDEPENDENT_EXTRACTION_MODE = "separate_visit_forward_before_temporal_overlay"


class ObjectPairAssociationError(ValueError):
    """Raised when an association input violates the shared-candidate protocol."""


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectPairAssociationError(f"{label} must be a lowercase SHA-256")
    return value


def _method_id(value: object) -> MethodId:
    if value not in _METHOD_IDS:
        raise ObjectPairAssociationError(
            f"unsupported object association method: {value}"
        )
    return value  # type: ignore[return-value]


def _candidate_pool(value: object) -> CandidatePool:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ObjectPairAssociationError("candidate pool must contain t0 and t1")
    visits: list[tuple[CandidateId, ...]] = []
    for visit_id, raw_candidates in enumerate(value):
        if not isinstance(raw_candidates, tuple) or not raw_candidates:
            raise ObjectPairAssociationError(
                "each candidate pool visit must be non-empty"
            )
        candidates: list[CandidateId] = []
        for raw in raw_candidates:
            if (
                not isinstance(raw, tuple)
                or len(raw) != 2
                or raw[0] != visit_id
                or not isinstance(raw[1], str)
                or not raw[1]
            ):
                raise ObjectPairAssociationError(
                    "candidate ID has invalid visit or value"
                )
            candidates.append((visit_id, raw[1]))
        if len(candidates) != len(set(candidates)):
            raise ObjectPairAssociationError("candidate IDs must be unique per visit")
        visits.append(tuple(candidates))
    return visits[0], visits[1]


def _readonly_float_matrix(
    value: object, shape: tuple[int, int], label: str
) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ObjectPairAssociationError(f"{label} must have finite shape {shape}")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ObjectPairAssociationError(f"{label} must lie in [0, 1]")
    result.setflags(write=False)
    return result


def _readonly_bool_matrix(
    value: object, shape: tuple[int, int], label: str
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != shape:
        raise ObjectPairAssociationError(f"{label} must be boolean with shape {shape}")
    result = np.array(raw, dtype=np.bool_, copy=True, order="C")
    result.setflags(write=False)
    return result


def _evidence_value(value: object, label: str) -> EvidenceValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return float(value)
    raise ObjectPairAssociationError(f"{label} has an unsupported value")


def _edge_evidence(
    value: object,
    shape: tuple[int, int],
) -> tuple[tuple[Mapping[str, EvidenceValue], ...], ...]:
    if value == ():
        return tuple(
            tuple(MappingProxyType({}) for _ in range(shape[1]))
            for _ in range(shape[0])
        )
    if not isinstance(value, tuple) or len(value) != shape[0]:
        raise ObjectPairAssociationError("edge evidence rows do not match candidates")
    rows: list[tuple[Mapping[str, EvidenceValue], ...]] = []
    for raw_row in value:
        if not isinstance(raw_row, tuple) or len(raw_row) != shape[1]:
            raise ObjectPairAssociationError(
                "edge evidence columns do not match candidates"
            )
        row: list[Mapping[str, EvidenceValue]] = []
        for raw_record in raw_row:
            if not isinstance(raw_record, Mapping):
                raise ObjectPairAssociationError(
                    "edge evidence entries must be mappings"
                )
            record = {
                str(key): _evidence_value(item, f"edge evidence {key}")
                for key, item in raw_record.items()
                if isinstance(key, str) and key
            }
            if len(record) != len(raw_record):
                raise ObjectPairAssociationError(
                    "edge evidence keys must be non-empty strings"
                )
            row.append(MappingProxyType(dict(sorted(record.items()))))
        rows.append(tuple(row))
    return tuple(rows)


def _canonical_sha256(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if isinstance(item, Mapping):
            return {str(key): normalize(value) for key, value in sorted(item.items())}
        if isinstance(item, tuple):
            return [normalize(value) for value in item]
        return item

    return hashlib.sha256(
        json.dumps(
            normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    minimum_match_score: float = 0.50
    static_centroid_tolerance_m: float = 0.20

    def __post_init__(self) -> None:
        for value, label, lower, upper in (
            (self.minimum_match_score, "minimum match score", 0.0, 1.0),
            (
                self.static_centroid_tolerance_m,
                "static centroid tolerance",
                float(np.nextafter(0.0, 1.0)),
                math.inf,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ObjectPairAssociationError(f"{label} is invalid")
        object.__setattr__(self, "minimum_match_score", float(self.minimum_match_score))
        object.__setattr__(
            self,
            "static_centroid_tolerance_m",
            float(self.static_centroid_tolerance_m),
        )


@dataclass(frozen=True, slots=True)
class ObjectPairScoreMatrix:
    method_id: MethodId
    pair_content_sha256: str
    candidate_ids: CandidatePool
    scores: np.ndarray
    eligible: np.ndarray
    edge_evidence: tuple[tuple[Mapping[str, EvidenceValue], ...], ...] = ()

    def __post_init__(self) -> None:
        method = _method_id(self.method_id)
        pair_hash = _sha256(self.pair_content_sha256, "pair content SHA-256")
        candidates = _candidate_pool(self.candidate_ids)
        shape = (len(candidates[0]), len(candidates[1]))
        scores = _readonly_float_matrix(self.scores, shape, "pair scores")
        eligible = _readonly_bool_matrix(self.eligible, shape, "pair eligibility")
        evidence = _edge_evidence(self.edge_evidence, shape)
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "pair_content_sha256", pair_hash)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "edge_evidence", evidence)

    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "method_id": self.method_id,
                "pair_content_sha256": self.pair_content_sha256,
                "candidate_ids": self.candidate_ids,
                "scores": self.scores,
                "eligible": self.eligible,
                "edge_evidence": self.edge_evidence,
            }
        )


@dataclass(frozen=True, slots=True)
class IndependentFeatureBank:
    pair_content_sha256: str
    candidate_ids: CandidatePool
    features: tuple[np.ndarray, np.ndarray]
    valid: tuple[np.ndarray, np.ndarray]
    extraction_mode: str
    visit_forward_sha256: tuple[str, str]

    def __post_init__(self) -> None:
        pair_hash = _sha256(self.pair_content_sha256, "pair content SHA-256")
        candidates = _candidate_pool(self.candidate_ids)
        if self.extraction_mode != _INDEPENDENT_EXTRACTION_MODE:
            raise ObjectPairAssociationError(
                "F_obj requires separate visit forwards before temporal overlay"
            )
        if not isinstance(self.features, tuple) or len(self.features) != 2:
            raise ObjectPairAssociationError(
                "independent features must contain two visits"
            )
        arrays: list[np.ndarray] = []
        if not isinstance(self.valid, tuple) or len(self.valid) != 2:
            raise ObjectPairAssociationError(
                "independent feature validity must contain two visits"
            )
        valid_arrays: list[np.ndarray] = []
        width: int | None = None
        for visit_id, raw in enumerate(self.features):
            array = np.array(raw, dtype=np.float32, copy=True, order="C")
            raw_valid = np.asarray(self.valid[visit_id])
            if raw_valid.dtype != np.bool_ or raw_valid.shape != (
                len(candidates[visit_id]),
            ):
                raise ObjectPairAssociationError(
                    "independent feature validity shape is invalid"
                )
            valid = np.array(raw_valid, dtype=np.bool_, copy=True, order="C")
            if (
                array.ndim != 2
                or array.shape[0] != len(candidates[visit_id])
                or array.shape[1] < 1
                or not np.all(np.isfinite(array))
            ):
                raise ObjectPairAssociationError("independent feature shape is invalid")
            if width is not None and array.shape[1] != width:
                raise ObjectPairAssociationError("independent feature widths differ")
            norms = np.linalg.norm(array, axis=1)
            if np.any(norms[valid] <= 0.0) or np.any(array[~valid] != 0.0):
                raise ObjectPairAssociationError(
                    "feature rows must be nonzero exactly when valid"
                )
            width = array.shape[1]
            array.setflags(write=False)
            valid.setflags(write=False)
            arrays.append(array)
            valid_arrays.append(valid)
        if (
            not isinstance(self.visit_forward_sha256, tuple)
            or len(self.visit_forward_sha256) != 2
        ):
            raise ObjectPairAssociationError(
                "visit forward hashes must contain t0 and t1"
            )
        hashes = (
            _sha256(self.visit_forward_sha256[0], "t0 forward SHA-256"),
            _sha256(self.visit_forward_sha256[1], "t1 forward SHA-256"),
        )
        if hashes[0] == hashes[1]:
            raise ObjectPairAssociationError("separate visit forwards must be distinct")
        object.__setattr__(self, "pair_content_sha256", pair_hash)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "features", (arrays[0], arrays[1]))
        object.__setattr__(self, "valid", (valid_arrays[0], valid_arrays[1]))
        object.__setattr__(self, "visit_forward_sha256", hashes)

    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "pair_content_sha256": self.pair_content_sha256,
                "candidate_ids": self.candidate_ids,
                "features": self.features,
                "valid": self.valid,
                "extraction_mode": self.extraction_mode,
                "visit_forward_sha256": self.visit_forward_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class ObjectPairPrediction:
    prediction_id: str
    pair_id: str
    method_id: MethodId
    t0_entity_id: str | None
    t1_entity_id: str | None
    score: float | None
    state: PredictionState
    evidence: Mapping[str, EvidenceValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, str) or not self.prediction_id:
            raise ObjectPairAssociationError("prediction ID must be non-empty")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ObjectPairAssociationError("prediction pair ID must be non-empty")
        method = _method_id(self.method_id)
        for entity_id in (self.t0_entity_id, self.t1_entity_id):
            if entity_id is not None and (
                not isinstance(entity_id, str) or not entity_id
            ):
                raise ObjectPairAssociationError("prediction entity ID is invalid")
        cardinality = (self.t0_entity_id is not None, self.t1_entity_id is not None)
        expected_states = {
            (True, True): {"persistent_static", "persistent_moved"},
            (True, False): {"unmatched_t0"},
            (False, True): {"unmatched_t1"},
        }
        if (
            cardinality not in expected_states
            or self.state not in expected_states[cardinality]
        ):
            raise ObjectPairAssociationError("prediction state and endpoints disagree")
        if cardinality == (True, True):
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float, np.number))
                or not math.isfinite(float(self.score))
                or not 0.0 <= float(self.score) <= 1.0
            ):
                raise ObjectPairAssociationError("matched prediction score is invalid")
            score: float | None = float(self.score)
        elif self.score is not None:
            raise ObjectPairAssociationError("unmatched prediction score must be null")
        else:
            score = None
        if not isinstance(self.evidence, Mapping):
            raise ObjectPairAssociationError("prediction evidence must be a mapping")
        evidence = MappingProxyType(
            {
                str(key): _evidence_value(value, f"prediction evidence {key}")
                for key, value in sorted(self.evidence.items())
            }
        )
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "evidence", evidence)

    @property
    def is_matched(self) -> bool:
        return self.t0_entity_id is not None and self.t1_entity_id is not None


def _entity_points(
    pair: OviObjectPairView, *, supported_only: bool
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    result: list[tuple[np.ndarray, ...]] = []
    for visit in pair.visits:
        values: list[np.ndarray] = []
        for entity in visit.entities:
            indices = entity.point_indices
            if supported_only:
                indices = indices[visit.appearance_valid[indices]]
            values.append(visit.points_xyz[indices])
        result.append(tuple(values))
    return result[0], result[1]


def _shape_descriptor(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.zeros(6, dtype=np.float64)
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    radii = np.linalg.norm(centered, axis=1)
    quantiles = np.quantile(radii, (0.25, 0.50, 0.75))
    covariance = centered.T @ centered / max(len(centered), 1)
    spectrum = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    return np.concatenate((quantiles, spectrum))


def _semantic_similarity(left: object, right: object) -> tuple[float, bool]:
    left_embedding = getattr(left, "semantic_embedding", None)
    right_embedding = getattr(right, "semantic_embedding", None)
    if left_embedding is not None and right_embedding is not None:
        first = np.asarray(left_embedding, dtype=np.float64)
        second = np.asarray(right_embedding, dtype=np.float64)
        if first.shape == second.shape:
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator > 0.0:
                return float(
                    np.clip(np.dot(first, second) / denominator, 0.0, 1.0)
                ), True
    left_label = getattr(left, "semantic_label", None)
    right_label = getattr(right, "semantic_label", None)
    if left_label is not None and right_label is not None:
        return float(str(left_label).casefold() == str(right_label).casefold()), True
    return 0.0, False


def build_geometric_score_matrix(
    pair: OviObjectPairView,
    *,
    supported_only: bool,
    centroid_scale_m: float = 2.0,
) -> ObjectPairScoreMatrix:
    """Score all OVI candidate pairs without a hard displacement gate."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if type(supported_only) is not bool:
        raise ObjectPairAssociationError("supported_only must be boolean")
    if (
        isinstance(centroid_scale_m, bool)
        or not isinstance(centroid_scale_m, (int, float, np.number))
        or not math.isfinite(float(centroid_scale_m))
        or float(centroid_scale_m) <= 0.0
    ):
        raise ObjectPairAssociationError("centroid scale must be finite and positive")
    before = pair.content_sha256()
    points = _entity_points(pair, supported_only=supported_only)
    centroids = tuple(
        tuple(
            np.mean(value, axis=0, dtype=np.float64) if len(value) else np.zeros(3)
            for value in visit
        )
        for visit in points
    )
    shapes = tuple(
        tuple(_shape_descriptor(value) for value in visit) for visit in points
    )
    scores = np.zeros((len(points[0]), len(points[1])), dtype=np.float64)
    eligible = np.zeros_like(scores, dtype=bool)
    evidence: list[list[Mapping[str, EvidenceValue]]] = []
    for first_index, first_entity in enumerate(pair.visits[0].entities):
        row: list[Mapping[str, EvidenceValue]] = []
        for second_index, second_entity in enumerate(pair.visits[1].entities):
            available = bool(
                len(points[0][first_index]) and len(points[1][second_index])
            )
            distance = float(
                np.linalg.norm(centroids[0][first_index] - centroids[1][second_index])
            )
            shape_distance = float(
                np.linalg.norm(shapes[0][first_index] - shapes[1][second_index])
            )
            centroid_score = math.exp(-min(distance / float(centroid_scale_m), 50.0))
            shape_score = math.exp(-min(shape_distance, 50.0))
            semantic_score, semantic_available = _semantic_similarity(
                first_entity, second_entity
            )
            if semantic_available:
                score = (
                    0.40 * centroid_score + 0.40 * shape_score + 0.20 * semantic_score
                )
            else:
                score = 0.50 * centroid_score + 0.50 * shape_score
            scores[first_index, second_index] = score if available else 0.0
            eligible[first_index, second_index] = available
            row.append(
                {
                    "centroid_distance_m": distance,
                    "centroid_score": centroid_score,
                    "shape_distance": shape_distance,
                    "shape_score": shape_score,
                    "semantic_score": semantic_score,
                    "semantic_available": semantic_available,
                    "support_available": available,
                }
            )
        evidence.append(row)
    if pair.content_sha256() != before:
        raise ObjectPairAssociationError("geometric scoring mutated the pair view")
    return ObjectPairScoreMatrix(
        method_id="G_supported" if supported_only else "G_full",
        pair_content_sha256=before,
        candidate_ids=pair.candidate_ids,
        scores=scores,
        eligible=eligible,
        edge_evidence=tuple(tuple(row) for row in evidence),
    )


def build_feature_score_matrix(
    pair: OviObjectPairView, bank: IndependentFeatureBank
) -> ObjectPairScoreMatrix:
    """Cosine-score the same candidates using visit-independent Concerto features."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(bank, IndependentFeatureBank):
        raise TypeError("bank must be an IndependentFeatureBank")
    before = pair.content_sha256()
    if bank.pair_content_sha256 != before or bank.candidate_ids != pair.candidate_ids:
        raise ObjectPairAssociationError(
            "independent features bind a different candidate pool"
        )
    first = np.zeros(bank.features[0].shape, dtype=np.float64)
    second = np.zeros(bank.features[1].shape, dtype=np.float64)
    first[bank.valid[0]] = bank.features[0][bank.valid[0]].astype(
        np.float64, copy=False
    )
    second[bank.valid[1]] = bank.features[1][bank.valid[1]].astype(
        np.float64, copy=False
    )
    first[bank.valid[0]] /= np.linalg.norm(
        first[bank.valid[0]], axis=1, keepdims=True
    )
    second[bank.valid[1]] /= np.linalg.norm(
        second[bank.valid[1]], axis=1, keepdims=True
    )
    similarities = first @ second.T
    eligible = bank.valid[0][:, None] & bank.valid[1][None, :]
    scores = np.where(eligible, np.clip(similarities, 0.0, 1.0), 0.0)
    evidence = tuple(
        tuple(
            {
                "cosine_similarity": float(similarities[row, column]),
                "feature_bank_sha256": bank.content_sha256(),
                "independent_visit_forwards": True,
                "support_available": bool(eligible[row, column]),
            }
            for column in range(scores.shape[1])
        )
        for row in range(scores.shape[0])
    )
    if pair.content_sha256() != before:
        raise ObjectPairAssociationError("feature scoring mutated the pair view")
    return ObjectPairScoreMatrix(
        method_id="F_obj",
        pair_content_sha256=before,
        candidate_ids=pair.candidate_ids,
        scores=scores,
        eligible=eligible,
        edge_evidence=evidence,
    )


def build_rescene_score_matrix(
    pair: OviObjectPairView,
    sample: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
) -> ObjectPairScoreMatrix:
    """Compute max_q c_q a_qi^0 a_qj^1 on the immutable OVI pool."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(sample, NeuralSampleMap):
        raise TypeError("sample must be a NeuralSampleMap")
    if not isinstance(evidence, TemporalQueryEvidence):
        raise TypeError("evidence must be TemporalQueryEvidence")
    before = pair.content_sha256()
    supported_maps = pair.to_visit_maps(supported_only=True)
    if (
        sample.coordinate_frame_id != pair.coordinate_frame_id
        or sample.source_manifest_sha256 != pair.source_manifest_sha256
        or sample.source_visit_map_sha256
        != tuple(value.snapshot_sha256 for value in supported_maps)
    ):
        raise ObjectPairAssociationError("ReScene sample binds a different pair view")
    validate_query_evidence(sample, evidence)
    assert evidence.token_scores is not None
    assert evidence.query_scores is not None
    token_weights = np.diff(sample.source_to_token_offsets).astype(np.float64)
    token_keys = tuple(
        zip(sample.visit_ids.tolist(), sample.token_entity_ids, strict=True)
    )
    query_count = len(evidence.temporal_query_ids)
    affinities: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    for visit_candidates in pair.candidate_ids:
        visit_affinity = np.zeros(
            (query_count, len(visit_candidates)), dtype=np.float64
        )
        visit_denominator = np.zeros(len(visit_candidates), dtype=np.float64)
        for candidate_index, candidate in enumerate(visit_candidates):
            indices = np.asarray(
                [index for index, key in enumerate(token_keys) if key == candidate],
                dtype=np.int64,
            )
            if not len(indices):
                continue
            weights = token_weights[indices]
            denominator = float(weights.sum())
            visit_denominator[candidate_index] = denominator
            visit_affinity[:, candidate_index] = (
                evidence.token_scores[:, indices] @ weights / denominator
            )
        affinities.append(visit_affinity)
        denominators.append(visit_denominator)
    products = (
        evidence.query_scores[:, None, None].astype(np.float64)
        * affinities[0][:, :, None]
        * affinities[1][:, None, :]
    )
    winners = np.argmax(products, axis=0)
    scores = np.max(products, axis=0)
    eligible = (denominators[0][:, None] > 0.0) & (denominators[1][None, :] > 0.0)
    evidence_rows: list[list[Mapping[str, EvidenceValue]]] = []
    for first_index in range(scores.shape[0]):
        row: list[Mapping[str, EvidenceValue]] = []
        for second_index in range(scores.shape[1]):
            query_index = int(winners[first_index, second_index])
            total_mass = float(evidence.token_scores[query_index] @ token_weights)
            selected_mass = (
                affinities[0][query_index, first_index] * denominators[0][first_index]
                + affinities[1][query_index, second_index]
                * denominators[1][second_index]
            )
            positive_entities = int(
                np.count_nonzero(affinities[0][query_index] > 0.0)
                + np.count_nonzero(affinities[1][query_index] > 0.0)
            )
            row.append(
                {
                    "winning_query_id": evidence.temporal_query_ids[query_index],
                    "query_confidence": float(evidence.query_scores[query_index]),
                    "t0_affinity": float(affinities[0][query_index, first_index]),
                    "t1_affinity": float(affinities[1][query_index, second_index]),
                    "query_purity": 0.0
                    if total_mass == 0.0
                    else min(selected_mass / total_mass, 1.0),
                    "conflicting_entity_count": max(positive_entities - 2, 0),
                    "t0_support_point_count": int(denominators[0][first_index]),
                    "t1_support_point_count": int(denominators[1][second_index]),
                }
            )
        evidence_rows.append(row)
    if pair.content_sha256() != before:
        raise ObjectPairAssociationError("ReScene scoring mutated the pair view")
    return ObjectPairScoreMatrix(
        method_id="R_obj",
        pair_content_sha256=before,
        candidate_ids=pair.candidate_ids,
        scores=scores,
        eligible=eligible,
        edge_evidence=tuple(tuple(row) for row in evidence_rows),
    )


def _entity_centroids(
    pair: OviObjectPairView,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return tuple(
        {
            entity.entity_id: np.mean(visit.points_xyz[entity.point_indices], axis=0)
            for entity in visit.entities
        }
        for visit in pair.visits
    )  # type: ignore[return-value]


def solve_object_pair_assignment(
    pair: OviObjectPairView,
    matrix: ObjectPairScoreMatrix,
    config: AssociationConfig,
) -> tuple[ObjectPairPrediction, ...]:
    """Apply one shared cardinality-first 1:1 solver with explicit dummies."""

    if not isinstance(pair, OviObjectPairView):
        raise TypeError("pair must be an OviObjectPairView")
    if not isinstance(matrix, ObjectPairScoreMatrix):
        raise TypeError("matrix must be an ObjectPairScoreMatrix")
    if not isinstance(config, AssociationConfig):
        raise TypeError("config must be an AssociationConfig")
    before = pair.content_sha256()
    if (
        matrix.pair_content_sha256 != before
        or matrix.candidate_ids != pair.candidate_ids
    ):
        raise ObjectPairAssociationError(
            "score matrix substitutes the shared candidate pool"
        )
    first_count, second_count = matrix.scores.shape
    size = first_count + second_count
    utility = np.full((size, size), -1e9, dtype=np.float64)
    allowed = matrix.eligible & (matrix.scores >= config.minimum_match_score)
    cardinality_bonus = min(first_count, second_count) + 1.0
    utility[:first_count, :second_count][allowed] = (
        cardinality_bonus + matrix.scores[allowed]
    )
    for index in range(first_count):
        utility[index, second_count + index] = 0.0
    for index in range(second_count):
        utility[first_count + index, index] = 0.0
    utility[first_count:, second_count:] = 0.0
    rows, columns = linear_sum_assignment(-utility)
    matched_indices = sorted(
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if row < first_count and column < second_count and allowed[row, column]
    )
    used_first = {row for row, _column in matched_indices}
    used_second = {column for _row, column in matched_indices}
    centroids = _entity_centroids(pair)
    predictions: list[ObjectPairPrediction] = []
    for first_index, second_index in matched_indices:
        first_id = matrix.candidate_ids[0][first_index][1]
        second_id = matrix.candidate_ids[1][second_index][1]
        distance = float(
            np.linalg.norm(centroids[0][first_id] - centroids[1][second_id])
        )
        edge = dict(matrix.edge_evidence[first_index][second_index])
        edge.update(
            {
                "centroid_distance_m": distance,
                "minimum_match_score": config.minimum_match_score,
                "score_matrix_sha256": matrix.content_sha256(),
            }
        )
        predictions.append(
            ObjectPairPrediction(
                prediction_id=f"{matrix.method_id}:match:{first_id}:{second_id}",
                pair_id=pair.pair_id,
                method_id=matrix.method_id,
                t0_entity_id=first_id,
                t1_entity_id=second_id,
                score=float(matrix.scores[first_index, second_index]),
                state=(
                    "persistent_static"
                    if distance <= config.static_centroid_tolerance_m
                    else "persistent_moved"
                ),
                evidence=edge,
            )
        )
    for index, (_visit_id, entity_id) in enumerate(matrix.candidate_ids[0]):
        if index not in used_first:
            predictions.append(
                ObjectPairPrediction(
                    prediction_id=f"{matrix.method_id}:unmatched:t0:{entity_id}",
                    pair_id=pair.pair_id,
                    method_id=matrix.method_id,
                    t0_entity_id=entity_id,
                    t1_entity_id=None,
                    score=None,
                    state="unmatched_t0",
                    evidence={"minimum_match_score": config.minimum_match_score},
                )
            )
    for index, (_visit_id, entity_id) in enumerate(matrix.candidate_ids[1]):
        if index not in used_second:
            predictions.append(
                ObjectPairPrediction(
                    prediction_id=f"{matrix.method_id}:unmatched:t1:{entity_id}",
                    pair_id=pair.pair_id,
                    method_id=matrix.method_id,
                    t0_entity_id=None,
                    t1_entity_id=entity_id,
                    score=None,
                    state="unmatched_t1",
                    evidence={"minimum_match_score": config.minimum_match_score},
                )
            )
    if pair.content_sha256() != before:
        raise ObjectPairAssociationError("assignment mutated the pair view")
    if {
        value.t0_entity_id for value in predictions if value.t0_entity_id is not None
    } != {value[1] for value in matrix.candidate_ids[0]} or {
        value.t1_entity_id for value in predictions if value.t1_entity_id is not None
    } != {value[1] for value in matrix.candidate_ids[1]}:
        raise AssertionError("assignment did not conserve the candidate pool")
    return tuple(predictions)


__all__ = [
    "AssociationConfig",
    "IndependentFeatureBank",
    "ObjectPairAssociationError",
    "ObjectPairPrediction",
    "ObjectPairScoreMatrix",
    "build_feature_score_matrix",
    "build_geometric_score_matrix",
    "build_rescene_score_matrix",
    "solve_object_pair_assignment",
]
