"""Causal exact-identity diagnostics for temporally sparse 3RScan sessions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import pairwise

CHANGE_TYPES = ("rigid", "nonrigid", "added", "removed")


class RScanProtocolError(ValueError):
    """Raised when predictions violate the causal 3RScan protocol."""


@dataclass(frozen=True, slots=True)
class TemporalGroundTruth:
    session_index: int
    instance_id: str
    present: bool
    change_type: str | None


@dataclass(frozen=True, slots=True)
class TemporalPrediction:
    session_index: int
    track_id: str
    matched_instance_id: str | None


@dataclass(frozen=True, slots=True)
class TemporalIdentityResult:
    evaluation_session: int
    id_switches: int
    false_reid: int
    reactivation_recall: float | None
    current_object_precision: float | None
    current_object_recall: float | None
    stale_object_fp: int
    recall_by_change_type: Mapping[str, float | None]
    community_metrics_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "PASS",
            "protocol_id": "3RSCAN_CAUSAL_EXACT_ID_V1",
            "evaluation_session": self.evaluation_session,
            "custom_exact_id_diagnostics": {
                "id_switches": self.id_switches,
                "false_reid": self.false_reid,
                "reactivation_recall": self.reactivation_recall,
                "current_object_precision": self.current_object_precision,
                "current_object_recall": self.current_object_recall,
                "stale_object_fp": self.stale_object_fp,
                "recall_by_change_type": dict(self.recall_by_change_type),
            },
            "community_metrics": {
                "status": self.community_metrics_status,
                "stage_ap": None,
                "temporal_ap": None,
                "temporal_recall": None,
            },
        }


def derive_added_ids(
    reference_instance_ids: AbstractSet[str], session_instance_ids: AbstractSet[str]
) -> frozenset[str]:
    """Derive added IDs in evaluator space, never in method input space."""

    return frozenset(session_instance_ids - reference_instance_ids)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_temporal_identity(
    ground_truth: Sequence[TemporalGroundTruth],
    *,
    predictions: Sequence[TemporalPrediction],
    evaluation_session: int,
) -> TemporalIdentityResult:
    """Evaluate exact IDs using only prediction sessions up to the checkpoint."""

    if type(evaluation_session) is not int or evaluation_session < 0:
        raise RScanProtocolError("evaluation session must be non-negative")
    if any(prediction.session_index > evaluation_session for prediction in predictions):
        raise RScanProtocolError("prediction from a future session is not causal")
    gt_by_session: dict[int, dict[str, TemporalGroundTruth]] = defaultdict(dict)
    for row in ground_truth:
        if (
            type(row.session_index) is not int
            or row.session_index < 0
            or not isinstance(row.instance_id, str)
            or not row.instance_id
            or type(row.present) is not bool
        ):
            raise RScanProtocolError("invalid ground-truth identity")
        if row.change_type not in (*CHANGE_TYPES, None):
            raise RScanProtocolError("invalid ground-truth change type")
        if row.instance_id in gt_by_session[row.session_index]:
            raise RScanProtocolError("duplicate ground-truth identity in session")
        gt_by_session[row.session_index][row.instance_id] = row
    if evaluation_session not in gt_by_session or 0 not in gt_by_session:
        raise RScanProtocolError("ground truth lacks reference or evaluation session")
    predictions_by_session: dict[int, list[TemporalPrediction]] = defaultdict(list)
    for row in predictions:
        if (
            type(row.session_index) is not int
            or row.session_index < 0
            or not isinstance(row.track_id, str)
            or not row.track_id
            or (
                row.matched_instance_id is not None
                and (
                    not isinstance(row.matched_instance_id, str)
                    or not row.matched_instance_id
                )
            )
        ):
            raise RScanProtocolError("invalid prediction identity")
        if any(existing.track_id == row.track_id for existing in predictions_by_session[row.session_index]):
            raise RScanProtocolError("duplicate predicted track in session")
        predictions_by_session[row.session_index].append(row)

    gt_tracks: dict[str, list[tuple[int, str]]] = defaultdict(list)
    predicted_tracks: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for session in range(evaluation_session + 1):
        present_ids = {
            identity
            for identity, row in gt_by_session.get(session, {}).items()
            if row.present
        }
        matched_gt_in_session: set[str] = set()
        for prediction in predictions_by_session.get(session, []):
            matched = prediction.matched_instance_id
            if matched is None or matched not in present_ids:
                continue
            if matched not in matched_gt_in_session:
                gt_tracks[matched].append((session, prediction.track_id))
                matched_gt_in_session.add(matched)
            predicted_tracks[prediction.track_id].append((session, matched))
    id_switches = sum(
        left[1] != right[1]
        for observations in gt_tracks.values()
        for left, right in pairwise(observations)
    )
    false_reid = sum(
        left[1] != right[1]
        for observations in predicted_tracks.values()
        for left, right in pairwise(observations)
    )

    current_gt = gt_by_session[evaluation_session]
    current_present = {identity for identity, row in current_gt.items() if row.present}
    reference_present = {
        identity for identity, row in gt_by_session[0].items() if row.present
    }
    added_ids = derive_added_ids(reference_present, current_present)
    if any(
        row.change_type == "added" and identity not in added_ids
        for identity, row in current_gt.items()
    ):
        raise RScanProtocolError("added identity conflicts with reference annotations")
    current_predictions = predictions_by_session.get(evaluation_session, [])
    matched_once: set[str] = set()
    true_predictions = 0
    stale = 0
    for prediction in current_predictions:
        matched = prediction.matched_instance_id
        if matched in current_present and matched not in matched_once:
            matched_once.add(matched)
            true_predictions += 1
        elif matched is not None and matched in current_gt and not current_gt[matched].present:
            stale += 1

    recall_by_type: dict[str, float | None] = {}
    for change_type in CHANGE_TYPES:
        targets = [
            row
            for identity, row in current_gt.items()
            if (
                identity in added_ids
                if change_type == "added"
                else identity not in added_ids and row.change_type == change_type
            )
        ]
        if change_type == "removed":
            detected = sum(
                not any(
                    prediction.matched_instance_id == target.instance_id
                    for prediction in current_predictions
                )
                for target in targets
            )
        else:
            detected = sum(target.instance_id in matched_once for target in targets)
        recall_by_type[change_type] = _ratio(detected, len(targets))

    reactivation_targets: list[str] = []
    reactivated = 0
    for identity in current_present:
        history = [
            gt_by_session.get(session, {}).get(identity)
            for session in range(evaluation_session + 1)
        ]
        present_sessions = [
            index for index, row in enumerate(history) if row is not None and row.present
        ]
        if len(present_sessions) < 2:
            continue
        previous = present_sessions[-2]
        if not any(
            history[index] is not None and not history[index].present
            for index in range(previous + 1, evaluation_session)
        ):
            continue
        reactivation_targets.append(identity)
        before_tracks = {
            prediction.track_id
            for prediction in predictions_by_session.get(previous, [])
            if prediction.matched_instance_id == identity
        }
        current_tracks = {
            prediction.track_id
            for prediction in current_predictions
            if prediction.matched_instance_id == identity
        }
        reactivated += bool(before_tracks & current_tracks)

    return TemporalIdentityResult(
        evaluation_session,
        id_switches,
        false_reid,
        _ratio(reactivated, len(reactivation_targets)),
        _ratio(true_predictions, len(current_predictions)),
        _ratio(len(matched_once), len(current_present)),
        stale,
        recall_by_type,
        "NOT_COMPUTED_MISSING_COMMUNITY_INPUT",
    )
