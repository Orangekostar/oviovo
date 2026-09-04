#!/usr/bin/env python3
"""Run the source-bound OVI B7 geometric dense-recovery extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (
    _evaluation_context,
    _load_t1_frames,
    load_two_visit_ovi_inputs,
)
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.evaluation.contracts import MapSnapshot
from src.evaluation.exporters.oviovo import read_map_snapshot
from src.evaluation.two_visit_b7_attribution import (
    attribute_b7_recovery,
    write_b7_attribution,
)
from src.evaluation.two_visit_snapshot_metrics import evaluate_two_visit_snapshot
from src.oviv2.geometric_pair_reasoner import (
    GeometricPairReasoner,
    GeometricReasonerConfig,
)
from src.oviv2.ovimap_visit_loader import load_ovimap_visit
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
)
from src.oviv2.two_visit_b7_visibility import derive_signed_visibility_for_points
from src.oviv2.two_visit_contracts import (
    CurrentCompositionDecision,
    PairRelation,
    VisitMap,
    snapshot_content_sha256,
)
from src.oviv2.two_visit_current_map import (
    CompositionPointGroup,
    TwoVisitCurrentMap,
)
from src.oviv2.two_visit_dense_recovery import (
    DenseRecoveryConfig,
    DenseRecoveryResult,
    historical_output_points,
    recover_dense_history,
    write_dense_recovery,
)
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    build_geometric_pair_sample,
    derive_observed_point_mask,
)
from src.oviv2.two_visit_registration import (
    RegistrationConfig,
    apply_rigid_transform,
    register_pair_relation,
)

B7_RUN_ARTIFACT_ID = "OVI_RESCENE_B7_GEOMETRIC_DENSE_RECOVERY_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = {"path", "sha256", "byte_count"}
_BASE_COMMIT = "9db59cb3200e0bc7f48f904c4e9fa628c71c9586"
_CONFIG_ID = "OVI_RESCENE_B7_GEOMETRIC_DENSE_RECOVERY_CONFIG_V1"
_FROZEN_INPUT_RECORDS = {
    "matrix_summary",
    "two_visit_ovi_manifest",
    "protocol",
    "b0_output_manifest",
    "b2_output_manifest",
    "b3_output_manifest",
    "b3_current_map_manifest",
    "b3_metrics",
    "b4_output_manifest",
    "b4_current_map_manifest",
    "causal_schedule",
    "rgbd_export_manifest",
    "rgbd_lock",
    "common_v2_target_manifest",
    "semantic_aliases",
    "semantic_label_space",
}
_METHOD_FROZEN_INPUT_RECORDS = _FROZEN_INPUT_RECORDS - {
    "b3_metrics",
    "common_v2_target_manifest",
    "matrix_summary",
    "semantic_aliases",
    "semantic_label_space",
}
_SOURCE_BINDING_ROLES = {
    "attribution",
    "attribution_cli",
    "b7_visibility",
    "runner",
    "registration",
    "dense_recovery",
    "visibility_execution",
    "current_map",
    "two_visit_contracts",
    "geometric_reasoner",
    "query_projection",
    "visit_loader",
    "snapshot_metrics",
    "current_metrics",
    "dataset_loader",
    "evaluation_contracts",
    "map_exporter",
    "semantic_crosswalk",
    "visibility_engine",
    "matrix_executor",
}
_SUCCESS_GATE = {
    "maximum_ghost": 0.02,
    "minimum_surface_precision_at_5cm": 0.7244127782,
    "minimum_background_f1_at_5cm": 0.3563600098,
    "minimum_observed_stale_precision": 0.9259634046,
    "minimum_unobserved_recall": 0.0806464685,
    "minimum_surface_f1_at_5cm": 0.4323354117,
    "minimum_accepted_registrations": 1,
    "minimum_recovered_points": 1,
}


class B7RunError(ValueError):
    """Raised when a B7 source, contract, or artifact is invalid."""


@dataclass(frozen=True, slots=True)
class B7EvaluationPackage:
    metrics: Mapping[str, object]
    artifacts: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not isinstance(
            self.artifacts, Mapping
        ):
            raise TypeError("evaluation package fields must be mappings")


def _canonical_json(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_regular(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise B7RunError(f"{label} is unavailable: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise B7RunError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != after.st_size:
        raise B7RunError(f"{label} changed while reading")
    return content


def _normalized_record(
    record: object,
    *,
    label: str,
    base: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise B7RunError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    digest = record.get("sha256")
    byte_count = record.get("byte_count")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise B7RunError(f"{label} binding values are invalid")
    declared = Path(raw_path)
    if base is not None:
        if declared.is_absolute() or ".." in declared.parts:
            raise B7RunError(f"{label} artifact path must be local and relative")
        path = base / declared
    else:
        path = declared if declared.is_absolute() else REPO_ROOT / declared
    content = _read_regular(path, label=label)
    observed = {
        "path": raw_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    normalized = dict(record)
    if observed != normalized:
        raise B7RunError(f"{label} binding mismatch")
    return path.absolute(), normalized


def _json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B7RunError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise B7RunError(f"{label} must contain a JSON object")
    return value


def load_bound_json(record: object, *, label: str) -> dict[str, Any]:
    """Load a hash- and byte-count-bound JSON object."""

    path, _normalized = _normalized_record(record, label=label)
    return _json_object(_read_regular(path, label=label), label=label)


def _validate_record_shape(record: object, *, label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise B7RunError(f"{label} binding schema is invalid")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("byte_count")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise B7RunError(f"{label} binding values are invalid")


def _validate_b7_config_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "config_id",
        "status",
        "base_commit",
        "method",
        "success_gate",
        "scene_policy",
        "frozen_inputs",
        "source_bindings",
    }:
        raise B7RunError("B7 configuration schema is invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("config_id") != _CONFIG_ID
        or payload.get("status") != "FROZEN_BEFORE_FIRST_B7_SCORE"
        or payload.get("base_commit") != _BASE_COMMIT
    ):
        raise B7RunError("B7 configuration identity is invalid")
    method = payload.get("method")
    if not isinstance(method, dict) or set(method) != {
        "geometry_authority",
        "semantic_authority",
        "identity_authority",
        "removal_authority",
        "registration",
        "dense_recovery",
        "signed_visibility",
        "geometric_reasoner",
        "projection",
        "evaluator_distance_threshold_m",
    }:
        raise B7RunError("B7 method schema is invalid")
    authorities = {
        "geometry_authority": "ovi",
        "semantic_authority": "ovi_vlm",
        "identity_authority": "b4_geometric_baseline",
        "removal_authority": "signed_t1_visibility",
        "evaluator_distance_threshold_m": 0.05,
    }
    if any(method.get(name) != value for name, value in authorities.items()):
        raise B7RunError("B7 method authority is invalid")
    if method.get("registration") != RegistrationConfig().to_json_record():
        raise B7RunError("registration configuration differs from the frozen design")
    if method.get("dense_recovery") != DenseRecoveryConfig().to_json_record():
        raise B7RunError("dense recovery configuration differs from the frozen design")
    if method.get("signed_visibility") != asdict(SignedVisibilityConfig()):
        raise B7RunError("signed visibility configuration differs from B3")
    if method.get("geometric_reasoner") != asdict(GeometricReasonerConfig()):
        raise B7RunError("geometric reasoner configuration differs from B4")
    if method.get("projection") != asdict(ProjectionConfig()):
        raise B7RunError("query projection configuration differs from B4")
    if payload.get("success_gate") != _SUCCESS_GATE:
        raise B7RunError("B7 success gate differs from preregistration")
    scene_policy = payload.get("scene_policy")
    if scene_policy != {
        "apartment": {"status": "ENABLED", "attempt_count": 0},
        "office": {"status": "OFFICE_NOT_RUN_HELD_OUT", "attempt_count": 0},
    }:
        raise B7RunError("B7 scene policy is invalid")
    frozen = payload.get("frozen_inputs")
    if not isinstance(frozen, dict) or set(frozen) != {"apartment", "office"}:
        raise B7RunError("B7 frozen input schema is invalid")
    apartment = frozen.get("apartment")
    if not isinstance(apartment, dict) or set(apartment) != _FROZEN_INPUT_RECORDS | {
        "rgbd_root"
    }:
        raise B7RunError("Apartment frozen input inventory is invalid")
    root = apartment.get("rgbd_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise B7RunError("Apartment RGB-D root must be absolute")
    for role in sorted(_FROZEN_INPUT_RECORDS):
        _validate_record_shape(apartment[role], label=f"frozen input {role}")
    if frozen.get("office") != {
        "status": "OFFICE_BLOCKED_ASSET",
        "attempt_count": 0,
        "missing_bindings": [
            "rgbd_export_manifest",
            "camera",
            "trajectory",
            "timestamps",
            "ground_truth",
        ],
    }:
        raise B7RunError("Office frozen input policy is invalid")
    sources = payload.get("source_bindings")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_BINDING_ROLES:
        raise B7RunError("B7 source binding inventory is invalid")
    for role in sorted(sources):
        _validate_record_shape(sources[role], label=f"source {role}")
    return payload


def load_and_validate_b7_config(
    path: str | Path,
    *,
    verify_source_bindings: bool = True,
    verify_frozen_inputs: bool = True,
) -> dict[str, Any]:
    """Load the preregistered B7 config and optionally rehash all sources."""

    config_path = Path(path).absolute()
    payload = _validate_b7_config_payload(
        _json_object(
            _read_regular(config_path, label="B7 configuration"),
            label="B7 configuration",
        )
    )
    if verify_source_bindings:
        for role, record in sorted(payload["source_bindings"].items()):
            _normalized_record(record, label=f"source {role}")
    if verify_frozen_inputs:
        for role in sorted(_FROZEN_INPUT_RECORDS):
            _normalized_record(
                payload["frozen_inputs"]["apartment"][role],
                label=f"frozen input {role}",
            )
    return payload


def _load_local_artifact(
    record: object,
    *,
    root: Path,
    label: str,
) -> Path:
    path, _normalized = _normalized_record(record, label=label, base=root)
    return path


def load_two_visit_current_map(manifest_record: object) -> TwoVisitCurrentMap:
    """Reconstruct and verify a frozen B3 current-map artifact."""

    manifest_path, _record = _normalized_record(
        manifest_record, label="B3 current-map manifest"
    )
    manifest = _json_object(
        _read_regular(manifest_path, label="B3 current-map manifest"),
        label="B3 current-map manifest",
    )
    required = {
        "schema_version",
        "artifact_id",
        "status",
        "method",
        "current_map_sha256",
        "source_visit_map_sha256",
        "source_manifest_sha256",
        "visibility_source_sha256",
        "relation_ids",
        "geometry_sources",
        "artifacts",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != "OVI_TWO_VISIT_CURRENT_MAP_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("geometry_sources") != ["ovi_t0", "ovi_t1"]
    ):
        raise B7RunError("B3 current-map manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "snapshot",
        "entities",
        "provenance",
    }:
        raise B7RunError("B3 current-map artifact inventory is invalid")
    root = manifest_path.parent
    snapshot_path = _load_local_artifact(
        artifacts["snapshot"], root=root, label="B3 snapshot artifact"
    )
    entities_path = _load_local_artifact(
        artifacts["entities"], root=root, label="B3 entities artifact"
    )
    provenance_path = _load_local_artifact(
        artifacts["provenance"], root=root, label="B3 provenance artifact"
    )
    try:
        snapshot = read_map_snapshot(snapshot_path, entities_path)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise B7RunError("B3 snapshot artifact is invalid") from error
    groups: list[CompositionPointGroup] = []
    try:
        with provenance_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict) or set(payload) != {
                    "decision",
                    "source_point_indices",
                    "source_snapshot_sha256",
                    "output_entity_id",
                    "output_point_start",
                    "output_point_count",
                }:
                    raise B7RunError(
                        f"B3 provenance line {line_number} schema is invalid"
                    )
                decision = payload["decision"]
                if not isinstance(decision, dict):
                    raise B7RunError(
                        f"B3 provenance line {line_number} decision is invalid"
                    )
                groups.append(
                    CompositionPointGroup(
                        decision=CurrentCompositionDecision(**decision),
                        source_point_indices=np.asarray(
                            payload["source_point_indices"], dtype=np.int64
                        ),
                        source_snapshot_sha256=payload["source_snapshot_sha256"],
                        output_entity_id=payload["output_entity_id"],
                        output_point_start=payload["output_point_start"],
                        output_point_count=payload["output_point_count"],
                    )
                )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, B7RunError):
            raise
        raise B7RunError("B3 provenance artifact is invalid") from error
    current = TwoVisitCurrentMap(
        snapshot=snapshot,
        provenance=tuple(groups),
        source_visit_map_sha256=tuple(manifest["source_visit_map_sha256"]),
        source_manifest_sha256=manifest["source_manifest_sha256"],
        visibility_source_sha256=manifest["visibility_source_sha256"],
        relation_ids=tuple(manifest["relation_ids"]),
    )
    if current.content_sha256() != manifest.get("current_map_sha256"):
        raise B7RunError("B3 current-map content hash mismatch")
    return current


def _snapshot_labeler(
    snapshot: MapSnapshot,
) -> Callable[[tuple[Any, ...]], Mapping[str, tuple[str, float]]]:
    entity_ids = {entity.entity_id for entity in snapshot.entities}
    labels: dict[str, tuple[str, float]] = {}
    for entity in snapshot.entities:
        if entity.entity_id in labels:
            raise B7RunError("frozen semantic receipt entity IDs are not unique")
        if entity.semantic_label is None:
            continue
        labels[entity.entity_id] = (entity.semantic_label, entity.semantic_score)

    def labeler(evidence: tuple[Any, ...]) -> Mapping[str, tuple[str, float]]:
        output: dict[str, tuple[str, float]] = {}
        for item in evidence:
            entity_id = getattr(item, "entity_id", None)
            if not isinstance(entity_id, str) or entity_id not in entity_ids:
                raise B7RunError(f"frozen semantic receipt lacks entity {entity_id!r}")
            if entity_id in labels:
                output[entity_id] = labels[entity_id]
        return output

    return labeler


def semantic_labelers_from_frozen_snapshots(
    t0_snapshot: MapSnapshot,
    t1_snapshot: MapSnapshot,
) -> tuple[
    Callable[[tuple[Any, ...]], Mapping[str, tuple[str, float]]],
    Callable[[tuple[Any, ...]], Mapping[str, tuple[str, float]]],
]:
    """Build visit-scoped labelers from frozen B0/B2 semantic receipts."""

    if not isinstance(t0_snapshot, MapSnapshot) or not isinstance(
        t1_snapshot, MapSnapshot
    ):
        raise TypeError("semantic receipts must be MapSnapshot values")
    return _snapshot_labeler(t0_snapshot), _snapshot_labeler(t1_snapshot)


def _load_snapshot_output(
    manifest_record: object,
    *,
    variant_id: str,
) -> tuple[MapSnapshot, dict[str, Any], Path]:
    manifest_path, _normalized = _normalized_record(
        manifest_record, label=f"{variant_id} output manifest"
    )
    manifest = _json_object(
        _read_regular(manifest_path, label=f"{variant_id} output manifest"),
        label=f"{variant_id} output manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("variant_id") != variant_id
        or manifest.get("scene") != "apartment"
        or manifest.get("geometry_sources")
        != {
            "B0": ["ovi_t0"],
            "B2": ["ovi_t1"],
            "B3": ["ovi_t0", "ovi_t1"],
            "B4": ["ovi_t0", "ovi_t1"],
        }[variant_id]
    ):
        raise B7RunError(f"{variant_id} output manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not {"snapshot", "entities"} <= set(
        artifacts
    ):
        raise B7RunError(f"{variant_id} output artifact inventory is invalid")
    snapshot_path = _load_local_artifact(
        artifacts["snapshot"],
        root=manifest_path.parent,
        label=f"{variant_id} snapshot",
    )
    entities_path = _load_local_artifact(
        artifacts["entities"],
        root=manifest_path.parent,
        label=f"{variant_id} entities",
    )
    try:
        snapshot = read_map_snapshot(snapshot_path, entities_path)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise B7RunError(f"{variant_id} snapshot is invalid") from error
    if snapshot_content_sha256(snapshot) != manifest.get("snapshot_sha256"):
        raise B7RunError(f"{variant_id} snapshot content hash mismatch")
    return snapshot, manifest, manifest_path


@dataclass(frozen=True, slots=True)
class FrozenB7Inputs:
    t0: VisitMap
    t1: VisitMap
    baseline: TwoVisitCurrentMap
    relations: tuple[PairRelation, ...]
    frames: tuple[Any, ...]
    protocol: Mapping[str, Any]
    method_input_sha256: str
    frozen_records: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if not isinstance(self.t0, VisitMap) or not isinstance(self.t1, VisitMap):
            raise TypeError("frozen inputs require two VisitMap values")
        if not isinstance(self.baseline, TwoVisitCurrentMap):
            raise TypeError("frozen inputs require a B3 current map")
        if not self.frames:
            raise B7RunError("frozen inputs require t1 RGB-D frames")
        if (
            not isinstance(self.method_input_sha256, str)
            or _SHA256.fullmatch(self.method_input_sha256) is None
        ):
            raise B7RunError("frozen method input digest is invalid")


def _same_artifact(
    first_record: object,
    second_record: object,
    *,
    first_root: Path | None,
    second_root: Path | None,
    label: str,
) -> None:
    first_path, first = _normalized_record(
        first_record, label=f"{label} first", base=first_root
    )
    second_path, second = _normalized_record(
        second_record, label=f"{label} second", base=second_root
    )
    if (
        first_path != second_path
        or first["sha256"] != second["sha256"]
        or first["byte_count"] != second["byte_count"]
    ):
        raise B7RunError(f"{label} bindings disagree")


def load_frozen_b7_inputs(config: Mapping[str, Any]) -> FrozenB7Inputs:
    """Open method-only Apartment inputs and reproduce frozen B4 identity."""

    _validate_b7_config_payload(dict(config))
    records = config["frozen_inputs"]["apartment"]
    snapshots: dict[str, MapSnapshot] = {}
    manifests: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for variant_id, role in (
        ("B0", "b0_output_manifest"),
        ("B2", "b2_output_manifest"),
        ("B3", "b3_output_manifest"),
        ("B4", "b4_output_manifest"),
    ):
        snapshot, manifest, path = _load_snapshot_output(
            records[role], variant_id=variant_id
        )
        snapshots[variant_id] = snapshot
        manifests[variant_id] = manifest
        paths[variant_id] = path
    common_bindings = manifests["B0"].get("input_bindings")
    if not isinstance(common_bindings, Mapping) or any(
        manifest.get("input_bindings") != common_bindings
        for manifest in manifests.values()
    ):
        raise B7RunError("frozen B0-B4 input bindings disagree")
    b3_artifacts = manifests["B3"].get("artifacts")
    b4_artifacts = manifests["B4"].get("artifacts")
    assert isinstance(b3_artifacts, Mapping) and isinstance(b4_artifacts, Mapping)
    _same_artifact(
        b3_artifacts.get("provenance_manifest"),
        records["b3_current_map_manifest"],
        first_root=paths["B3"].parent,
        second_root=None,
        label="B3 current-map manifest",
    )
    _same_artifact(
        b4_artifacts.get("provenance_manifest"),
        records["b4_current_map_manifest"],
        first_root=paths["B4"].parent,
        second_root=None,
        label="B4 current-map manifest",
    )
    baseline = load_two_visit_current_map(records["b3_current_map_manifest"])
    if snapshot_content_sha256(baseline.snapshot) != snapshot_content_sha256(
        snapshots["B3"]
    ):
        raise B7RunError("B3 outer and current-map snapshots differ")
    protocol_path, _protocol_record = _normalized_record(
        records["protocol"], label="two-visit protocol"
    )
    ovi_manifest_path, _ovi_record = _normalized_record(
        records["two_visit_ovi_manifest"], label="two-visit OVI manifest"
    )
    loaded = load_two_visit_ovi_inputs(
        protocol_path=protocol_path,
        two_visit_ovi_manifest=ovi_manifest_path,
        scene="apartment",
    )
    t0_labeler, t1_labeler = semantic_labelers_from_frozen_snapshots(
        snapshots["B0"], snapshots["B2"]
    )
    visits: list[VisitMap] = []
    for visit_id, (visit_name, labeler) in enumerate(
        (("t0", t0_labeler), ("t1", t1_labeler))
    ):
        visit = loaded["visits"][visit_name]
        visits.append(
            load_ovimap_visit(
                native_manifest=visit["native_manifest_path"],
                materialized_manifest=visit["materialized_manifest_path"],
                visit_id=visit_id,
                scene="apartment",
                coordinate_frame_id="tesse_cd_world",
                source_manifest_sha256=loaded["protocol_content_sha256"],
                observed_frame_start=int(visit["start_frame"]),
                observed_frame_end=int(visit["end_frame"]),
                semantic_labeler=labeler,
            )
        )
    t0, t1 = visits
    validate_visit_sha_bindings(
        t0,
        t1,
        *(manifests[variant_id] for variant_id in ("B0", "B2", "B3", "B4")),
        baseline=baseline,
    )
    method = config["method"]
    relations = materialize_geometric_relations(
        t0,
        t1,
        geometric_config=GeometricReasonerConfig(**method["geometric_reasoner"]),
        projection_config=ProjectionConfig(**method["projection"]),
    )
    b4_current = load_bound_json(
        records["b4_current_map_manifest"], label="B4 current-map manifest"
    )
    if [str(value.temporal_query_id) for value in relations] != b4_current.get(
        "relation_ids"
    ):
        raise B7RunError("materialized B4 relation IDs differ from frozen B4")
    schedule_path, _schedule_record = _normalized_record(
        records["causal_schedule"], label="causal schedule"
    )
    rgbd_root = Path(records["rgbd_root"])
    export_path, _export_record = _normalized_record(
        records["rgbd_export_manifest"], label="Apartment RGB-D export manifest"
    )
    _normalized_record(records["rgbd_lock"], label="official RGB-D lock")
    if export_path != (rgbd_root / "apartment" / "export_manifest.json").absolute():
        raise B7RunError("Apartment RGB-D export binding disagrees with RGB-D root")
    frames = _load_t1_frames(
        rgbd_root=rgbd_root,
        scene="apartment",
        schedule_path=schedule_path,
        start_frame=t1.observed_frame_start,
        end_frame=t1.observed_frame_end,
    )
    relation_bytes = canonical_relations_bytes(relations)
    method_digest = canonical_method_input_sha256(
        records=records,
        visit_map_sha256=(t0.snapshot_sha256, t1.snapshot_sha256),
        baseline_current_map_sha256=baseline.content_sha256(),
        relations_sha256=hashlib.sha256(relation_bytes).hexdigest(),
        method=method,
    )
    return FrozenB7Inputs(
        t0=t0,
        t1=t1,
        baseline=baseline,
        relations=relations,
        frames=frames,
        protocol=loaded["protocol"],
        method_input_sha256=method_digest,
        frozen_records={
            role: dict(records[role]) for role in sorted(_METHOD_FROZEN_INPUT_RECORDS)
        },
    )


def validate_visit_sha_bindings(
    t0: VisitMap,
    t1: VisitMap,
    *variant_manifests: Mapping[str, object],
    baseline: TwoVisitCurrentMap,
) -> None:
    """Require every frozen B0/B2/B3/B4 receipt to name the same visits."""

    expected = [t0.snapshot_sha256, t1.snapshot_sha256]
    if baseline.source_visit_map_sha256 != tuple(expected):
        raise B7RunError("VisitMap SHA binding mismatch in B3 baseline")
    for manifest in variant_manifests:
        if manifest.get("source_visit_map_sha256") != expected:
            raise B7RunError("VisitMap SHA binding mismatch in frozen variant")


def _relation_record(relation: PairRelation) -> dict[str, object]:
    return {
        "temporal_query_id": str(relation.temporal_query_id),
        "t0_entity_ids": list(relation.t0_entity_ids),
        "t1_entity_ids": list(relation.t1_entity_ids),
        "state": relation.state,
        "query_confidence": relation.query_confidence,
        "evidence": dict(relation.evidence),
        "identity_source": relation.identity_source,
    }


def canonical_relations_bytes(relations: tuple[PairRelation, ...]) -> bytes:
    """Serialize B4 relation materialization independently of input order."""

    if not isinstance(relations, tuple) or any(
        not isinstance(relation, PairRelation) for relation in relations
    ):
        raise TypeError("relations must be a tuple of PairRelation values")
    ordered = sorted(relations, key=lambda value: str(value.temporal_query_id))
    identifiers = [str(value.temporal_query_id) for value in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise B7RunError("relation IDs must be unique")
    return _canonical_json(
        {
            "schema_version": 1,
            "identity_source": "geometric_baseline",
            "relations": [_relation_record(relation) for relation in ordered],
        }
    )


def canonical_method_input_sha256(
    *,
    records: Mapping[str, object],
    visit_map_sha256: tuple[str, str],
    baseline_current_map_sha256: str,
    relations_sha256: str,
    method: Mapping[str, object],
) -> str:
    """Hash only inputs visible to the B7 method, never evaluator-only records."""

    missing = _METHOD_FROZEN_INPUT_RECORDS - set(records)
    if missing:
        raise B7RunError(f"method frozen input records are missing: {sorted(missing)}")
    return hashlib.sha256(
        _canonical_json(
            {
                "frozen_records": {
                    role: records[role] for role in sorted(_METHOD_FROZEN_INPUT_RECORDS)
                },
                "visit_map_sha256": list(visit_map_sha256),
                "b3_current_map_sha256": baseline_current_map_sha256,
                "relations_sha256": relations_sha256,
                "method": method,
            }
        )
    ).hexdigest()


def materialize_geometric_relations(
    t0: VisitMap,
    t1: VisitMap,
    *,
    geometric_config: GeometricReasonerConfig | None = None,
    projection_config: ProjectionConfig | None = None,
) -> tuple[PairRelation, ...]:
    """Rebuild the frozen B4 relation artifact without composing a map."""

    geometric = geometric_config or GeometricReasonerConfig()
    projection = projection_config or ProjectionConfig()
    pair = build_geometric_pair_sample(t0, t1, neural_voxel_size_m=0.02)
    evidence = GeometricPairReasoner(geometric).infer(pair)
    relations = project_queries_to_instances(pair, evidence, projection).relations
    canonical_relations_bytes(relations)
    return relations


def execute_geometric_dense_recovery(
    *,
    t0: VisitMap,
    t1: VisitMap,
    baseline: TwoVisitCurrentMap,
    relations: tuple[PairRelation, ...],
    frames: tuple[Any, ...],
    method_input_sha256: str,
    registration_config: RegistrationConfig | None = None,
    visibility_config: SignedVisibilityConfig | None = None,
    recovery_config: DenseRecoveryConfig | None = None,
) -> DenseRecoveryResult:
    """Execute B7-G over already validated method-only inputs."""

    if (
        not isinstance(method_input_sha256, str)
        or _SHA256.fullmatch(method_input_sha256) is None
    ):
        raise B7RunError("method_input_sha256 must be a lowercase SHA-256")
    registration = registration_config or RegistrationConfig()
    visibility_settings = visibility_config or SignedVisibilityConfig()
    recovery = recovery_config or DenseRecoveryConfig()
    if baseline.source_visit_map_sha256 != (t0.snapshot_sha256, t1.snapshot_sha256):
        raise B7RunError("B3 and VisitMap source hashes differ")
    t0_entities = {entity.entity_id: entity for entity in t0.snapshot.entities}
    t1_entities = {entity.entity_id: entity for entity in t1.snapshot.entities}
    source_before = (t0.snapshot_sha256, t1.snapshot_sha256)
    registrations = []
    candidate_chunks: list[np.ndarray] = []
    for relation in sorted(relations, key=lambda value: str(value.temporal_query_id)):
        if (
            relation.state not in {"persistent_static", "persistent_moved"}
            or len(relation.t0_entity_ids) != 1
            or len(relation.t1_entity_ids) != 1
        ):
            continue
        source = t0_entities.get(relation.t0_entity_ids[0])
        target = t1_entities.get(relation.t1_entity_ids[0])
        if source is None or target is None:
            raise B7RunError("B4 relation references a missing OVI entity")
        evidence = register_pair_relation(
            relation,
            source.points_xyz,
            target.points_xyz,
            source_semantic_label=source.semantic_label,
            target_semantic_label=target.semantic_label,
            config=registration,
        )
        registrations.append(evidence)
        if evidence.accepted:
            assert evidence.transform_world_from_t0 is not None
            candidate_chunks.append(
                apply_rigid_transform(
                    source.points_xyz,
                    evidence.transform_world_from_t0,
                )
            )
    candidates = (
        np.concatenate(candidate_chunks, axis=0)
        if candidate_chunks
        else np.empty((0, 3), dtype=np.float64)
    )
    visibility_source = hashlib.sha256(
        _canonical_json(
            {
                "method_input_sha256": method_input_sha256,
                "signed_visibility": asdict(visibility_settings),
                "registration_sha256": [
                    value.content_sha256() for value in registrations
                ],
                "candidate_point_count": len(candidates),
            }
        )
    ).hexdigest()
    candidate_visibility = derive_signed_visibility_for_points(
        candidates,
        frames,
        visibility_settings,
        source_sha256=visibility_source,
    )
    result = recover_dense_history(
        baseline,
        t0,
        t1,
        relations,
        tuple(registrations),
        candidate_visibility,
        recovery,
    )
    source_after = (
        t0.snapshot_sha256,
        t1.snapshot_sha256,
    )
    t0.assert_unchanged()
    t1.assert_unchanged()
    if source_after != source_before:
        raise B7RunError("OVI VisitMap hashes changed during B7 execution")
    return result


def evaluate_b7_gate(
    metrics: Mapping[str, object],
    *,
    accepted_registrations: int,
    recovered_points: int,
) -> dict[str, object]:
    """Evaluate every preregistered B7-G safety and material-gain predicate."""

    required = {
        "ghost",
        "surface_precision_at_5cm",
        "background_f1_at_5cm",
        "t1_observed_region_stale_precision",
        "t1_unobserved_region_recall",
        "surface_f1_at_5cm",
    }
    if not required <= set(metrics):
        raise B7RunError("B7 metric receipt is incomplete")
    values: dict[str, float] = {}
    for name in required:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise B7RunError(f"B7 metric {name} is not numeric")
        number = float(value)
        if not np.isfinite(number):
            raise B7RunError(f"B7 metric {name} is not finite")
        values[name] = number
    if type(accepted_registrations) is not int or accepted_registrations < 0:
        raise B7RunError("accepted_registrations must be non-negative")
    if type(recovered_points) is not int or recovered_points < 0:
        raise B7RunError("recovered_points must be non-negative")
    predicates = {
        "ghost": values["ghost"] <= _SUCCESS_GATE["maximum_ghost"],
        "surface_precision_at_5cm": values["surface_precision_at_5cm"]
        >= _SUCCESS_GATE["minimum_surface_precision_at_5cm"],
        "background_f1_at_5cm": values["background_f1_at_5cm"]
        >= _SUCCESS_GATE["minimum_background_f1_at_5cm"],
        "t1_observed_region_stale_precision": values[
            "t1_observed_region_stale_precision"
        ]
        >= _SUCCESS_GATE["minimum_observed_stale_precision"],
        "t1_unobserved_region_recall": values["t1_unobserved_region_recall"]
        >= _SUCCESS_GATE["minimum_unobserved_recall"],
        "surface_f1_at_5cm": values["surface_f1_at_5cm"]
        >= _SUCCESS_GATE["minimum_surface_f1_at_5cm"],
        "accepted_registrations": accepted_registrations
        >= _SUCCESS_GATE["minimum_accepted_registrations"],
        "recovered_points": recovered_points
        >= _SUCCESS_GATE["minimum_recovered_points"],
    }
    return {
        "decision": (
            "GO_B7_MAPPING_EXTENSION"
            if all(predicates.values())
            else "STOP_B7_MAPPING_EXTENSION"
        ),
        "predicates": predicates,
        "thresholds": dict(_SUCCESS_GATE),
    }


def _registration_config(record: Mapping[str, Any]) -> RegistrationConfig:
    values = dict(record)
    labels = values.get("recoverable_semantic_labels")
    if not isinstance(labels, list):
        raise B7RunError("registration semantic label set is invalid")
    values["recoverable_semantic_labels"] = frozenset(labels)
    return RegistrationConfig(**values)


def _point_count(snapshot: MapSnapshot) -> int:
    return sum(len(entity.points_xyz) for entity in snapshot.entities) + (
        0 if snapshot.background_xyz is None else len(snapshot.background_xyz)
    )


def _method_diagnostics(
    result: DenseRecoveryResult,
    relations: tuple[PairRelation, ...],
) -> dict[str, object]:
    relation_states: dict[str, int] = {}
    for relation in relations:
        relation_states[relation.state] = relation_states.get(relation.state, 0) + 1
    recovery_decisions: dict[str, int] = {}
    for group in result.provenance:
        recovery_decisions[group.decision] = recovery_decisions.get(
            group.decision, 0
        ) + len(group.source_point_indices)
    rejection_reasons: dict[str, int] = {}
    for evidence in result.registrations:
        for reason in evidence.rejection_reasons:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    return {
        "relation_count": len(relations),
        "relation_state_counts": dict(sorted(relation_states.items())),
        "registration_attempt_count": len(result.registrations),
        "registration_accepted_count": sum(
            evidence.accepted for evidence in result.registrations
        ),
        "registration_rejected_count": sum(
            not evidence.accepted for evidence in result.registrations
        ),
        "registration_rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "recovery_decision_point_counts": dict(sorted(recovery_decisions.items())),
        "recovered_historical_point_count": result.recovered_point_count,
        "removed_unwarped_b3_point_count": result.removed_baseline_point_count,
        "rejected_visible_free_point_count": recovery_decisions.get(
            "reject_visible_free", 0
        ),
        "rejected_occupied_point_count": recovery_decisions.get("reject_occupied", 0),
        "rejected_incompatible_point_count": recovery_decisions.get(
            "reject_incompatible", 0
        ),
    }


def _evaluate_apartment_result(
    *,
    inputs: FrozenB7Inputs,
    result: DenseRecoveryResult,
    config: Mapping[str, Any],
    method_runtime_s: float,
    peak_rss_bytes: int,
    attribution_output: Path,
) -> B7EvaluationPackage:
    records = config["frozen_inputs"]["apartment"]
    for role in (
        "b3_metrics",
        "common_v2_target_manifest",
        "semantic_aliases",
        "semantic_label_space",
    ):
        _normalized_record(records[role], label=f"evaluator input {role}")
    schedule_path, _schedule = _normalized_record(
        records["causal_schedule"], label="causal schedule"
    )
    target_path, _target = _normalized_record(
        records["common_v2_target_manifest"], label="common-v2 target manifest"
    )
    aliases_path, _aliases = _normalized_record(
        records["semantic_aliases"], label="semantic aliases"
    )
    label_space_path, _labels = _normalized_record(
        records["semantic_label_space"], label="semantic label space"
    )
    context, evaluation_bindings = _evaluation_context(
        protocol=inputs.protocol,
        scene="apartment",
        final_frame=inputs.t1.observed_frame_end,
        schedule_path=schedule_path,
        target_manifest_path=target_path,
        frames=inputs.frames,
        visibility_config=SignedVisibilityConfig(
            **config["method"]["signed_visibility"]
        ),
    )
    crosswalk = load_tesse_semantic_crosswalk(
        aliases_path,
        "apartment",
        label_space_path,
    )
    historical = historical_output_points(result, inputs.baseline)
    historical_observed = derive_observed_point_mask(
        historical,
        inputs.frames,
        SignedVisibilityConfig(**config["method"]["signed_visibility"]),
    )
    measured = evaluate_two_visit_snapshot(
        result.snapshot,
        context,
        crosswalk,
        retained_t0_xyz=historical,
        retained_t0_t1_observed_mask=historical_observed,
    )
    diagnostics = _method_diagnostics(result, inputs.relations)
    gate = evaluate_b7_gate(
        measured,
        accepted_registrations=int(diagnostics["registration_accepted_count"]),
        recovered_points=result.recovered_point_count,
    )
    b3_metrics = load_bound_json(records["b3_metrics"], label="B3 metric receipt")
    b3_current = b3_metrics["metric_groups"]["current_state"]
    b3_geometry = b3_metrics["metric_groups"]["geometry"]
    evaluation_source_sha256 = hashlib.sha256(
        _canonical_json(
            {
                role: dict(binding)
                for role, binding in sorted(evaluation_bindings.items())
            }
        )
    ).hexdigest()
    attribution = attribute_b7_recovery(
        result=result,
        baseline=inputs.baseline,
        t0=inputs.t0,
        t1=inputs.t1,
        relations=inputs.relations,
        evaluation=context,
        evaluation_source_sha256=evaluation_source_sha256,
    )
    attribution_manifest = write_b7_attribution(attribution, attribution_output)
    attribution_payload = load_bound_json(
        _absolute_record(attribution_manifest), label="B7 attribution manifest"
    )
    attribution_artifacts = attribution_payload.get("artifacts")
    if not isinstance(attribution_artifacts, Mapping):
        raise B7RunError("B7 attribution artifact inventory is invalid")
    artifacts: dict[str, object] = {
        "attribution_manifest": _absolute_record(attribution_manifest)
    }
    for role, record in attribution_artifacts.items():
        artifact = _load_local_artifact(
            record,
            root=attribution_manifest.parent,
            label=f"B7 attribution {role}",
        )
        artifacts[f"attribution_{role}"] = _absolute_record(artifact)
    metrics = {
        "schema_version": 1,
        "status": "PASS",
        "variant_id": "B7-G",
        "scene": "apartment",
        "method_input_sha256": inputs.method_input_sha256,
        "dense_recovery_sha256": result.content_sha256(),
        "metrics": measured,
        "b3_reference": {
            "ghost": b3_current["ghost"],
            "current_miou": b3_current["current_miou"],
            **{
                name: b3_geometry[name]
                for name in (
                    "background_f1_at_5cm",
                    "surface_precision_at_5cm",
                    "surface_recall_at_5cm",
                    "surface_f1_at_5cm",
                    "total_current_surface_coverage",
                    "t1_unobserved_region_recall",
                    "t1_observed_region_stale_precision",
                )
            },
            "final_map_point_count": b3_metrics["metric_groups"]["systems"][
                "final_map_point_count"
            ],
        },
        "diagnostics": {
            **diagnostics,
            "final_map_point_count": _point_count(result.snapshot),
            "method_runtime_s": method_runtime_s,
            "peak_resident_memory_bytes": peak_rss_bytes,
            "peak_gpu_memory_bytes": 0,
        },
        "gate": gate,
        "attribution": {
            "content_sha256": attribution.content_sha256(),
            "summary": dict(attribution.summary),
            "evaluation_source_sha256": evaluation_source_sha256,
        },
        "object_change_identity": {
            "object_f1": None,
            "change_f1": None,
            "identity": None,
            "reason": "tesse_two_visit_protocol_compatible_instance_identity_gt_unavailable",
        },
        "evaluation_only_bindings": {
            role: dict(binding) for role, binding in sorted(evaluation_bindings.items())
        },
    }
    return B7EvaluationPackage(metrics=metrics, artifacts=artifacts)


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_b7_apartment(
    *,
    config_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Run one source-bound Apartment B7-G extension and evaluate it once."""

    path = Path(config_path).absolute()
    config_bytes = _read_regular(path, label="B7 configuration")
    config = load_and_validate_b7_config(
        path,
        verify_source_bindings=True,
        verify_frozen_inputs=False,
    )
    authorize_b7_scene(config, "apartment")
    inputs = load_frozen_b7_inputs(config)
    method = config["method"]
    started = time.perf_counter()
    result = execute_geometric_dense_recovery(
        t0=inputs.t0,
        t1=inputs.t1,
        baseline=inputs.baseline,
        relations=inputs.relations,
        frames=inputs.frames,
        method_input_sha256=inputs.method_input_sha256,
        registration_config=_registration_config(method["registration"]),
        visibility_config=SignedVisibilityConfig(**method["signed_visibility"]),
        recovery_config=DenseRecoveryConfig(**method["dense_recovery"]),
    )
    method_runtime_s = time.perf_counter() - started
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    relations_bytes = canonical_relations_bytes(inputs.relations)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    def write_method(stage: Path) -> Mapping[str, object]:
        relation_path = stage / "relations.json"
        _write_fsynced(relation_path, relations_bytes)
        command_path = stage / "command.json"
        _write_fsynced(
            command_path,
            _canonical_json(
                {
                    "argv": [
                        "python",
                        "scripts/evaluation/run_ovi_rescene_b7.py",
                        "--config",
                        str(path),
                        "--scene",
                        "apartment",
                        "--output",
                        str(Path(output_root).absolute()),
                    ],
                    "config_sha256": config_sha256,
                    "method_input_sha256": inputs.method_input_sha256,
                },
                indent=2,
            ),
        )
        dense_manifest = write_dense_recovery(result, stage / "method")
        dense = _json_object(
            _read_regular(dense_manifest, label="dense recovery manifest"),
            label="dense recovery manifest",
        )
        inventory: dict[str, object] = {
            "command": _absolute_record(command_path),
            "relations": _absolute_record(relation_path),
            "dense_recovery_manifest": _absolute_record(dense_manifest),
        }
        artifacts = dense.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise B7RunError("dense recovery artifact inventory is invalid")
        for role, record in artifacts.items():
            artifact = _load_local_artifact(
                record,
                root=dense_manifest.parent,
                label=f"dense recovery {role}",
            )
            inventory[f"dense_{role}"] = _absolute_record(artifact)
        return inventory

    def evaluate(
        stage: Path,
        _inventory: dict[str, dict[str, object]],
    ) -> B7EvaluationPackage:
        return _evaluate_apartment_result(
            inputs=inputs,
            result=result,
            config=config,
            method_runtime_s=method_runtime_s,
            peak_rss_bytes=peak_rss_bytes,
            attribution_output=stage / "attribution",
        )

    return atomic_publish_b7_package(
        output_root,
        method_writer=write_method,
        evaluator=evaluate,
        manifest_identity={
            "scene": "apartment",
            "variant_id": "B7-G",
            "source_commit": _git_head(),
            "config": {
                "path": str(path),
                "sha256": config_sha256,
                "byte_count": len(config_bytes),
            },
            "method_input_sha256": inputs.method_input_sha256,
            "dense_recovery_sha256": result.content_sha256(),
            "source_visit_map_sha256": [
                inputs.t0.snapshot_sha256,
                inputs.t1.snapshot_sha256,
            ],
            "baseline_current_map_sha256": inputs.baseline.content_sha256(),
        },
    )


def authorize_b7_scene(config: Mapping[str, object], scene: str) -> None:
    """Enforce the frozen Apartment-first and zero-attempt Office policy."""

    if scene not in {"apartment", "office"}:
        raise B7RunError("scene must be apartment or office")
    policy = config.get("scene_policy")
    record = policy.get(scene) if isinstance(policy, Mapping) else None
    if not isinstance(record, Mapping):
        raise B7RunError("scene policy is incomplete")
    if scene == "apartment" and record != {"status": "ENABLED", "attempt_count": 0}:
        raise B7RunError("Apartment scene policy is invalid")
    if scene == "office":
        status = record.get("status")
        if status != "ENABLED":
            raise B7RunError(str(status or "OFFICE_NOT_RUN_HELD_OUT"))


def _absolute_record(path: Path) -> dict[str, object]:
    content = _read_regular(path, label="published artifact")
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _validate_method_inventory(
    stage: Path,
    inventory: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(inventory, Mapping) or not inventory:
        raise B7RunError("method inventory must be a non-empty mapping")
    normalized: dict[str, dict[str, object]] = {}
    root = stage.resolve()
    for raw_role, raw_record in sorted(inventory.items()):
        role = str(raw_role)
        if not role or not isinstance(raw_record, Mapping):
            raise B7RunError("method inventory is invalid")
        raw_path = raw_record.get("path")
        if not isinstance(raw_path, str):
            raise B7RunError("method inventory path is invalid")
        declared = Path(raw_path)
        path = (
            declared.absolute()
            if declared.is_absolute()
            else (stage / declared).absolute()
        )
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise B7RunError("method artifact escapes staging root") from error
        observed = _absolute_record(path)
        comparable = dict(raw_record)
        comparable["path"] = str(path)
        if observed != comparable:
            raise B7RunError("method artifact inventory mismatch")
        normalized[role] = {**observed, "path": relative}
    return normalized


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_publish_b7_package(
    output_root: str | Path,
    *,
    method_writer: Callable[[Path], Mapping[str, object]],
    evaluator: Callable[
        [Path, dict[str, dict[str, object]]],
        Mapping[str, object] | B7EvaluationPackage,
    ],
    manifest_identity: Mapping[str, object],
) -> Path:
    """Publish method bytes before evaluation and reject any evaluator mutation."""

    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise B7RunError(f"B7 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        inventory = _validate_method_inventory(stage, method_writer(stage))
        frozen_inventory = {role: dict(record) for role, record in inventory.items()}
        evaluated = evaluator(stage, frozen_inventory)
        if isinstance(evaluated, B7EvaluationPackage):
            metrics = evaluated.metrics
            evaluation_inventory = _validate_method_inventory(
                stage, evaluated.artifacts
            )
        else:
            metrics = evaluated
            evaluation_inventory = {}
        if not isinstance(metrics, Mapping):
            raise B7RunError("evaluator must return a metric mapping")
        if _validate_method_inventory(stage, frozen_inventory) != inventory:
            raise B7RunError("method inventory changed during evaluation")
        collisions = set(inventory) & set(evaluation_inventory)
        if collisions or "metrics" in inventory or "metrics" in evaluation_inventory:
            raise B7RunError("published artifact roles must be unique")
        metrics_path = stage / "metrics.json"
        _write_fsynced(metrics_path, _canonical_json(dict(metrics), indent=2))
        reserved = {"schema_version", "artifact_id", "status", "artifacts"}
        if reserved & set(manifest_identity):
            raise B7RunError("manifest identity uses reserved fields")
        manifest = {
            "schema_version": 1,
            "artifact_id": B7_RUN_ARTIFACT_ID,
            "status": "PASS",
            **dict(manifest_identity),
            "scientific_decision": (
                metrics.get("gate", {}).get("decision")
                if isinstance(metrics.get("gate"), Mapping)
                else None
            ),
            "artifacts": {
                **inventory,
                **evaluation_inventory,
                "metrics": {
                    **_absolute_record(metrics_path),
                    "path": metrics_path.relative_to(stage).as_posix(),
                },
            },
        }
        manifest_path = stage / "manifest.json"
        _write_fsynced(manifest_path, _canonical_json(manifest, indent=2))
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        stage.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return output / "manifest.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", choices=("apartment", "office"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_and_validate_b7_config(
            args.config,
            verify_source_bindings=True,
            verify_frozen_inputs=False,
        )
        authorize_b7_scene(config, args.scene)
        if args.scene != "apartment":
            raise B7RunError("Office execution is not released")
        manifest = run_b7_apartment(
            config_path=args.config,
            output_root=args.output,
        )
    except (B7RunError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(str(manifest))
    return 0


__all__ = [
    "B7EvaluationPackage",
    "B7RunError",
    "FrozenB7Inputs",
    "atomic_publish_b7_package",
    "authorize_b7_scene",
    "canonical_method_input_sha256",
    "canonical_relations_bytes",
    "evaluate_b7_gate",
    "execute_geometric_dense_recovery",
    "load_and_validate_b7_config",
    "load_bound_json",
    "load_frozen_b7_inputs",
    "load_two_visit_current_map",
    "main",
    "materialize_geometric_relations",
    "run_b7_apartment",
    "semantic_labelers_from_frozen_snapshots",
    "validate_visit_sha_bindings",
]


if __name__ == "__main__":
    raise SystemExit(main())
