"""Noninterfering causal instrumentation for CROVE temporal development runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import src.oviv2.temporal_association as _association_module
import src.oviv2.temporal_runtime as _runtime_module
from src.evaluation.json_contracts import loads_strict
from src.oviv2.association import CandidateScore
from src.oviv2.temporal_association import TemporalAssociationResult
from src.oviv2.temporal_export import DynamicEvidenceState, TemporalExportBatch
from src.oviv2.temporal_geometry import ObjectMotionEstimate
from src.oviv2.temporal_lifecycle import TemporalLifecycle

_PATCH_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative: str | None = None) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved != path.absolute() or not resolved.is_file():
        raise ValueError(f"source must be a direct regular file: {path}")
    return {
        "path": relative if relative is not None else str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _temporal_runtime(runtime: object) -> object:
    return getattr(runtime, "temporal", runtime)


def _temporal_state(runtime: object) -> object | None:
    return getattr(_temporal_runtime(runtime), "state", None)


def _dynamic_config(runtime: object) -> object | None:
    config = getattr(_temporal_runtime(runtime), "config", None)
    return None if config is None else getattr(config, "dynamic_state", None)


def _result_temporal(result: object) -> object:
    return getattr(result, "temporal", result)


def _result_export(result: object) -> TemporalExportBatch | None:
    export = getattr(_result_temporal(result), "export", None)
    return export if type(export) is TemporalExportBatch else None


class _FrameCapture:
    def __init__(self, frame_index: int, runtime: object) -> None:
        self.frame_index = frame_index
        self.runtime = runtime
        self.observations: dict[int, object] = {}
        self.targets: dict[int, object] = {}
        self.association: TemporalAssociationResult | None = None
        self.candidate_edges: dict[tuple[int, int], dict[str, Any]] = {}
        self.point_observation_ids: dict[int, int] = {}
        self.motion: list[dict[str, Any]] = []
        self.unpaired_dynamic: list[dict[str, Any]] = []
        state = _temporal_state(runtime)
        entities = getattr(state, "entities", ()) if state is not None else ()
        self.pre_entity_ids = {int(entity.lifecycle.entity_id) for entity in entities}
        export_tracker = getattr(state, "export_tracker", None)
        self.pre_export = {
            int(entry.entity_id): entry
            for entry in getattr(export_tracker, "entries", ())
        }

    def capture_candidate(self, candidate: object) -> None:
        if candidate is None:
            return
        if not isinstance(candidate, CandidateScore):
            raise TypeError("observed candidate scorer returned an invalid result")
        target = self.targets.get(int(candidate.right_id))
        if target is None:
            raise RuntimeError("candidate target is absent from association inputs")
        lifecycle = target.lifecycle
        self.candidate_edges[(candidate.left_id, candidate.right_id)] = {
            "observation_id": candidate.left_id,
            "entity_id": candidate.right_id,
            "target_lifecycle": lifecycle.value,
            "score": candidate.score,
            "size_score": candidate.directed_overlap,
            "geometry_score": candidate.bounds_iou,
            "appearance_score": candidate.visual_cosine,
        }

    def profile_return(self, execution_frame: object, value: object) -> None:
        code = execution_frame.f_code
        local = execution_frame.f_locals
        if code is _association_module.associate_temporal_observations.__code__:
            if not isinstance(value, TemporalAssociationResult):
                return
            observations = local["observations"]
            targets = local["targets"]
            self.observations = {
                int(item.observation_id): item for item in observations
            }
            self.targets = {int(item.entity_id): item for item in targets}
            for candidate in local.get("edges", {}).values():
                self.capture_candidate(candidate)
            self.association = value
            return
        if code is _runtime_module.TemporalCurrentRuntime.process_frame.__code__:
            association = local.get("association")
            if isinstance(association, TemporalAssociationResult):
                self.association = association
            return
        if code is _runtime_module.backproject_observation.__code__:
            if isinstance(value, np.ndarray):
                observation = local["observation"]
                self.point_observation_ids[id(value)] = int(observation.observation_id)
            return
        if code in {
            _runtime_module.estimate_object_motion.__code__,
            _runtime_module.estimate_object_translation.__code__,
        }:
            if isinstance(value, ObjectMotionEstimate):
                self._capture_motion_estimate(
                    value,
                    local,
                    estimator=(
                        "gated_icp"
                        if code is _runtime_module.estimate_object_motion.__code__
                        else "bounded_translation"
                    ),
                )
            return
        if code is _runtime_module.route_active_motion_evidence.__code__:
            if value is not None:
                self._capture_routed_motion(value, local)
            return
        if code is _runtime_module.advance_dynamic_state.__code__ and isinstance(
            value, DynamicEvidenceState
        ):
            self._capture_dynamic_transition(value, local)

    def _capture_motion_estimate(
        self,
        result: ObjectMotionEstimate,
        local: Mapping[str, Any],
        *,
        estimator: str,
    ) -> None:
        points = local["points_world"]
        observation_id = self.point_observation_ids.get(id(points))
        assignments = (
            {} if self.association is None else dict(self.association.assignments)
        )
        entity_id = assignments.get(observation_id)
        if observation_id is None or entity_id is None:
            raise RuntimeError("motion estimator call cannot be bound to an assignment")
        previous_translation = np.asarray(local["previous_pose"], dtype=np.float64)[
            :3, 3
        ]
        displacement = float(
            np.linalg.norm(
                np.asarray(result.object_to_world, dtype=np.float64)[:3, 3]
                - previous_translation
            )
        )
        self.motion.append(
            {
                "path": "active_continuation",
                "observation_id": observation_id,
                "entity_id": entity_id,
                "estimator": estimator,
                "motion_decision": result.decision.value,
                "fitness": result.fitness,
                "rmse_m": result.rmse_m,
                "geometry_displacement_m": displacement,
            }
        )

    def _capture_routed_motion(self, result: object, local: Mapping[str, Any]) -> None:
        pending = next(
            (row for row in reversed(self.motion) if "routed_source" not in row),
            None,
        )
        if pending is None:
            raise RuntimeError("motion router call has no estimator record")
        config = local["normalized_config"]
        geometry_admitted = bool(local["geometry_admitted"])
        identity_admitted = bool(local["identity_admitted"])
        identity_confidence = float(local["identity_confidence"])
        pending.update(
            {
                "geometry_evidence_admitted": geometry_admitted,
                "geometry_confidence": float(local["normalized_geometry_confidence"]),
                "geometry_displacement_threshold_passed": (
                    float(local["geometry_displacement"]) >= config.displacement_floor_m
                ),
                "geometry_confidence_threshold_passed": (
                    float(local["normalized_geometry_confidence"])
                    >= config.minimum_motion_confidence
                ),
                "identity_evidence_admitted": identity_admitted,
                "identity_displacement_m": float(local["identity_displacement"]),
                "identity_confidence": identity_confidence,
                "identity_displacement_threshold_passed": (
                    float(local["identity_displacement"]) >= config.displacement_floor_m
                ),
                "identity_confidence_threshold_passed": (
                    identity_confidence >= config.minimum_motion_confidence
                ),
                "routed_source": result.source.value,
                "routed_accepted": result.accepted,
                "routed_displacement_m": result.displacement_m,
                "routed_confidence": result.confidence,
                "routed_qualifies_as_motion": result.qualifies_as_motion,
                "geometry_qualifies_as_motion": result.geometry_qualifies_as_motion,
                "identity_qualifies_as_motion": result.identity_qualifies_as_motion,
                "displacement_floor_m": config.displacement_floor_m,
                "minimum_motion_confidence": config.minimum_motion_confidence,
                "minimum_consecutive_motion_frames": (
                    config.minimum_consecutive_motion_frames
                ),
            }
        )

    def _capture_dynamic_transition(
        self, result: DynamicEvidenceState, local: Mapping[str, Any]
    ) -> None:
        state = local["state"]
        pending = next(
            (
                row
                for row in reversed(self.motion)
                if "routed_source" in row and "dynamic_state_after" not in row
            ),
            None,
        )
        record = {
            "dynamic_state_before": state.dynamic_state.value,
            "motion_streak_before": state.motion_streak,
            "static_streak_before": state.static_streak,
            "dynamic_state_after": result.dynamic_state.value,
            "motion_streak_after": result.motion_streak,
            "static_streak_after": result.static_streak,
        }
        if pending is None:
            self.unpaired_dynamic.append(
                {
                    **record,
                    "accepted_motion": local["accepted"],
                    "displacement_m": local["displacement"],
                    "confidence": local["normalized_confidence"],
                }
            )
        else:
            pending.update(record)

    def association_record(self, births: Sequence[int]) -> dict[str, Any]:
        result = self.association
        if result is None:
            return {
                "observation_count": 0,
                "target_count": 0,
                "active_target_count": 0,
                "dormant_target_count": 0,
                "candidate_active_edge_count": 0,
                "candidate_dormant_edge_count": 0,
                "candidate_active_edges": [],
                "candidate_dormant_edges": [],
                "reid_qualified_count": 0,
                "reid_qualified_pairs": [],
                "assignment_count": 0,
                "assignments": [],
                "unmatched_observation_ids": [],
                "unmatched_entity_ids": [],
                "new_identity_births": list(births),
            }
        active_edges = [
            row
            for row in self.candidate_edges.values()
            if row["target_lifecycle"] in {"active", "uncertain"}
        ]
        dormant_edges = [
            row
            for row in self.candidate_edges.values()
            if row["target_lifecycle"] == "dormant"
        ]
        diagnostics = {
            (item.observation_id, item.entity_id): item
            for item in result.assignment_diagnostics
        }
        assignments = []
        for pair in result.assignments:
            item = diagnostics[pair]
            assignments.append(
                {
                    "observation_id": item.observation_id,
                    "entity_id": item.entity_id,
                    "score": item.score,
                    "target_lifecycle": item.target_lifecycle.value,
                    "appearance_similarity": item.appearance_similarity,
                    "feature_model_match": item.feature_model_match,
                    "semantic_qualified": item.semantic_qualified,
                    "high_confidence_identity_match": item.high_confidence_identity_match,
                }
            )
        return {
            "observation_count": len(self.observations),
            "target_count": len(self.targets),
            "active_target_count": sum(
                target.lifecycle
                in {TemporalLifecycle.ACTIVE, TemporalLifecycle.UNCERTAIN}
                for target in self.targets.values()
            ),
            "dormant_target_count": sum(
                target.lifecycle is TemporalLifecycle.DORMANT
                for target in self.targets.values()
            ),
            "candidate_active_edge_count": len(active_edges),
            "candidate_dormant_edge_count": len(dormant_edges),
            "candidate_active_edges": sorted(
                active_edges, key=lambda row: (row["observation_id"], row["entity_id"])
            ),
            "candidate_dormant_edges": sorted(
                dormant_edges,
                key=lambda row: (row["observation_id"], row["entity_id"]),
            ),
            "reid_qualified_count": result.reid_opportunity_count,
            "reid_qualified_pairs": [
                {"observation_id": left, "entity_id": right}
                for left, right in result.reid_opportunity_pairs
            ],
            "assignment_count": len(assignments),
            "assignments": assignments,
            "unmatched_observation_ids": list(result.unmatched_observation_ids),
            "unmatched_entity_ids": list(result.unmatched_entity_ids),
            "new_identity_births": list(births),
        }

    def finalize(self, result: object) -> dict[str, Any]:
        temporal_result = _result_temporal(result)
        births = tuple(
            int(value) for value in getattr(temporal_result, "new_entity_ids", ())
        )
        export = _result_export(result)
        samples = (
            {} if export is None else {item.entity_id: item for item in export.samples}
        )
        state = _temporal_state(self.runtime)
        tracker = getattr(state, "export_tracker", None)
        post_dynamic = {
            int(entry.entity_id): entry.dynamic_evidence
            for entry in getattr(tracker, "entries", ())
        }
        self._append_bank_reappearance_motion(samples, post_dynamic)
        for row in self.motion:
            if "dynamic_state_after" not in row:
                entity_id = row.get("entity_id")
                sample = samples.get(entity_id)
                row.update(
                    {
                        "dynamic_state_after": None
                        if sample is None
                        else sample.dynamic_state.value,
                        "motion_streak_after": None,
                        "static_streak_after": None,
                    }
                )
        return {
            "schema_version": 1,
            "frame_index": self.frame_index,
            "association": self.association_record(births),
            "motion": sorted(
                self.motion,
                key=lambda row: (
                    row.get("observation_id", -1),
                    row.get("entity_id", -1),
                    row["path"],
                ),
            ),
            "unpaired_dynamic_updates": self.unpaired_dynamic,
        }

    def _append_bank_reappearance_motion(
        self,
        samples: Mapping[int, object],
        post_dynamic: Mapping[int, DynamicEvidenceState],
    ) -> None:
        if self.association is None:
            return
        existing_motion = {
            (row.get("observation_id"), row.get("entity_id")) for row in self.motion
        }
        config = _dynamic_config(self.runtime)
        if config is None:
            return
        for diagnostic in self.association.assignment_diagnostics:
            pair = (diagnostic.observation_id, diagnostic.entity_id)
            if (
                pair in existing_motion
                or diagnostic.target_lifecycle is not TemporalLifecycle.DORMANT
                or diagnostic.entity_id in self.pre_entity_ids
            ):
                continue
            observation = self.observations[diagnostic.observation_id]
            target = self.targets[diagnostic.entity_id]
            previous = self.pre_export.get(diagnostic.entity_id)
            previous_center = (
                target.centroid_xyz
                if previous is None or previous.last_centroid_xyz is None
                else previous.last_centroid_xyz
            )
            displacement = float(
                np.linalg.norm(
                    np.asarray(observation.centroid_xyz, dtype=np.float64)
                    - np.asarray(previous_center, dtype=np.float64)
                )
            )
            confidence = float(np.clip(diagnostic.appearance_similarity, 0.0, 1.0))
            qualifies = bool(
                displacement >= config.displacement_floor_m
                and confidence >= config.minimum_motion_confidence
            )
            sample = samples.get(diagnostic.entity_id)
            previous_dynamic = (
                DynamicEvidenceState.static()
                if previous is None
                else previous.dynamic_evidence
            )
            final_dynamic = post_dynamic.get(diagnostic.entity_id)
            if sample is not None and final_dynamic is None:
                raise RuntimeError(
                    "reidentified entity lacks post-state dynamic evidence"
                )
            self.motion.append(
                {
                    "path": "dormant_bank_reappearance",
                    "observation_id": diagnostic.observation_id,
                    "entity_id": diagnostic.entity_id,
                    "estimator": None,
                    "motion_decision": None,
                    "geometry_evidence_admitted": False,
                    "geometry_displacement_m": None,
                    "geometry_confidence": None,
                    "geometry_displacement_threshold_passed": False,
                    "geometry_confidence_threshold_passed": False,
                    "identity_evidence_admitted": True,
                    "identity_displacement_m": displacement,
                    "identity_confidence": confidence,
                    "identity_displacement_threshold_passed": (
                        displacement >= config.displacement_floor_m
                    ),
                    "identity_confidence_threshold_passed": (
                        confidence >= config.minimum_motion_confidence
                    ),
                    "routed_source": "identity",
                    "routed_accepted": True,
                    "routed_qualifies_as_motion": qualifies,
                    "displacement_floor_m": config.displacement_floor_m,
                    "minimum_motion_confidence": config.minimum_motion_confidence,
                    "minimum_consecutive_motion_frames": (
                        config.minimum_consecutive_motion_frames
                    ),
                    "dynamic_state_before": previous_dynamic.dynamic_state.value,
                    "motion_streak_before": previous_dynamic.motion_streak,
                    "static_streak_before": previous_dynamic.static_streak,
                    "dynamic_state_after": None
                    if final_dynamic is None
                    else final_dynamic.dynamic_state.value,
                    "motion_streak_after": None
                    if final_dynamic is None
                    else final_dynamic.motion_streak,
                    "static_streak_after": None
                    if final_dynamic is None
                    else final_dynamic.static_streak,
                }
            )


class RuntimeAttributionCapture:
    """Own deterministic frame records produced by one sequential runtime."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._active = False

    def observe(self, runtime: object, frame: object, call: Callable[[], Any]) -> Any:
        frame_index = getattr(frame, "frame_id", None)
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError(
                "instrumented frame must have a nonnegative integer frame_id"
            )
        if self._active:
            raise RuntimeError("runtime attribution capture cannot be nested")
        if self.records and frame_index <= self.records[-1]["frame_index"]:
            raise ValueError("instrumented frame IDs must increase strictly")
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError("another runtime attribution capture is active")
        self._active = True
        try:
            frame_capture = _FrameCapture(frame_index, runtime)
            with _profiled_runtime(frame_capture):
                result = call()
            self.records.append(frame_capture.finalize(result))
            return result
        finally:
            self._active = False
            _PATCH_LOCK.release()


@contextmanager
def _profiled_runtime(frame: _FrameCapture):
    previous = sys.getprofile()

    def profile(execution_frame, event, value):
        if previous is not None:
            previous(execution_frame, event, value)
        if event == "return":
            frame.profile_return(execution_frame, value)

    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)


class CroveRuntimeAttributionProxy:
    """Forward a runtime while observing only causal inputs and outputs."""

    def __init__(self, runtime: object, capture: RuntimeAttributionCapture) -> None:
        if not isinstance(capture, RuntimeAttributionCapture):
            raise TypeError("capture must be RuntimeAttributionCapture")
        self._runtime = runtime
        self._capture = capture

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def process_frame(self, frame: object, *args: object, **kwargs: object) -> Any:
        return self._capture.observe(
            self._runtime,
            frame,
            lambda: self._runtime.process_frame(frame, *args, **kwargs),
        )

    def process_temporal_only_frame(
        self, frame: object, *args: object, **kwargs: object
    ) -> Any:
        return self._capture.observe(
            self._runtime,
            frame,
            lambda: self._runtime.process_temporal_only_frame(frame, *args, **kwargs),
        )

    def advance_temporal_only_frame(
        self, frame: object, *args: object, **kwargs: object
    ) -> Any:
        return self._capture.observe(
            self._runtime,
            frame,
            lambda: self._runtime.advance_temporal_only_frame(frame, *args, **kwargs),
        )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def publish_runtime_attribution(
    output: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    run_manifest: str | Path,
    instrumentation_sources: Mapping[str, str | Path] | None = None,
) -> Path:
    """Atomically publish deterministic diagnostics outside a formal run root."""

    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    run_path = Path(run_manifest).resolve(strict=True)
    run = loads_strict(run_path.read_text(encoding="utf-8"), label="run manifest")
    if not isinstance(run, Mapping) or run.get("scene") != "apartment":
        raise ValueError("runtime attribution is restricted to Apartment")
    run_root = run_path.parent
    if destination.is_relative_to(run_root) or run_root.is_relative_to(destination):
        raise ValueError("diagnostic output must be outside the formal run root")
    normalized = [dict(record) for record in records]
    frame_ids = [record.get("frame_index") for record in normalized]
    if any(type(value) is not int or value < 0 for value in frame_ids):
        raise ValueError("attribution records require nonnegative integer frame IDs")
    if frame_ids != sorted(set(frame_ids)):
        raise ValueError("attribution frame IDs must be strictly increasing")
    jsonl = b"".join(_canonical_json(record) for record in normalized)
    source_records = {
        name: _file_record(Path(path))
        for name, path in sorted((instrumentation_sources or {}).items())
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        diagnostics_path = temporary / "runtime_attribution.jsonl"
        diagnostics_path.write_bytes(jsonl)
        manifest = {
            "schema_version": 1,
            "manifest_id": "crove_runtime_attribution_v1",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "status": "PASS",
            "instrumentation_only": True,
            "instrumentation_sources": source_records,
            "formal_run_manifest": _file_record(run_path),
            "runtime_attribution": _file_record(
                diagnostics_path, relative="runtime_attribution.jsonl"
            ),
            "frame_count": len(normalized),
            "first_frame_index": None if not frame_ids else frame_ids[0],
            "last_frame_index": None if not frame_ids else frame_ids[-1],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        for path in (diagnostics_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
