#!/usr/bin/env python3
"""Run source-bound object-level historical recovery on real 3RScan OVI maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.rescene_pair_executor import NativeForwardResult
from scripts.evaluation.run_ovi_rescene_object_level_transfer import (
    AssociationForwardOutputs,
    build_d2_inference_bundle,
    build_temporal_query_evidence,
    evaluate_shared_candidate_methods,
    load_object_transfer_config,
)
from src.core.data_structures import Frame
from src.datasets.scannet200 import ScanNet200Dataset
from src.evaluation.object_pair_association import ObjectPairPrediction
from src.evaluation.ovi_ownership_completion import evaluate_completion_surface_v2
from src.evaluation.ovi_pair_artifact_loader import restore_bound_ovi_pair_artifact
from src.evaluation.ovi_pair_views import build_ovi_object_pair_view
from src.evaluation.rscan_gt_instances import (
    GroundTruthInstance,
    GroundTruthPair,
    PredictedInstance,
    evaluate_instance_geometry,
    load_pair_ground_truth,
    voxelize_points,
)
from src.evaluation.temporal_object_groups import (
    GroupingConfig,
    TemporalObjectGrouping,
    build_temporal_object_groups,
)
from src.oviv2.query_instance_projection import (
    ProjectionConfig,
    project_queries_to_instances,
)
from src.oviv2.two_visit_b7_visibility import derive_signed_visibility_for_points
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    TwoVisitCurrentMap,
    compose_current_map,
)
from src.oviv2.two_visit_dense_recovery import (
    DenseRecoveryConfig,
    DenseRecoveryResult,
    recover_dense_history,
    write_dense_recovery,
)
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    derive_signed_visibility,
)
from src.oviv2.two_visit_registration import (
    OracleTransformSource,
    RegistrationConfig,
    RegistrationEvidence,
    apply_rigid_transform,
    official_row_vector_to_internal_transform,
    register_composite_relation,
    register_oracle_pair_relation,
    validate_rigid_transform,
)

_SHA256 = frozenset("0123456789abcdef")
_GIT_SHA = frozenset("0123456789abcdef")
_VARIANTS = ("C0", "C1", "C2", "O1", "O2")
_SOURCE_BINDINGS = {
    "adaptation_decision",
    "d2_forward_arrays",
    "d2_forward_metadata",
    "d2_mapping_receipt",
    "d2_pair_receipt",
    "ground_truth_manifest",
    "object_transfer_config",
    "selection_manifest",
}


class ObjectRecoveryError(ValueError):
    """Raised when recovery inputs or evidence violate the frozen protocol."""


@dataclass(frozen=True, slots=True)
class VariantRecovery:
    variant_id: str
    baseline: TwoVisitCurrentMap
    registrations: tuple[RegistrationEvidence, ...]
    candidate_visibility: SignedVisibilityGrid
    recovery: DenseRecoveryResult


def _is_digest(value: object, *, width: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == width
        and set(value).issubset(_SHA256 if width == 64 else _GIT_SHA)
    )


def relations_from_object_predictions(
    predictions: Sequence[ObjectPairPrediction],
) -> tuple[PairRelation, ...]:
    """Convert matched G-object predictions to auditable 1:1 relations."""

    if isinstance(predictions, (str, bytes)):
        raise TypeError("predictions must be a sequence of object predictions")
    rows = tuple(predictions)
    if any(not isinstance(value, ObjectPairPrediction) for value in rows):
        raise TypeError("predictions contain an invalid value")
    pair_ids = {value.pair_id for value in rows}
    if len(pair_ids) > 1:
        raise ObjectRecoveryError("predictions span multiple pairs")
    relations: list[PairRelation] = []
    for prediction in rows:
        if not prediction.is_matched:
            continue
        if prediction.method_id not in {"G_full", "G_supported"}:
            raise ObjectRecoveryError("C1 relations must come from a G-object method")
        assert prediction.t0_entity_id is not None
        assert prediction.t1_entity_id is not None
        assert prediction.score is not None
        relations.append(
            PairRelation(
                temporal_query_id=prediction.prediction_id,
                t0_entity_ids=(prediction.t0_entity_id,),
                t1_entity_ids=(prediction.t1_entity_id,),
                state=prediction.state,
                query_confidence=prediction.score,
                evidence={"association_score": prediction.score},
                identity_source="geometric_baseline",
            )
        )
    return tuple(sorted(relations, key=lambda value: str(value.temporal_query_id)))


def relations_from_temporal_grouping(
    grouping: TemporalObjectGrouping,
    *,
    identity_source: str,
    static_centroid_tolerance_m: float,
) -> tuple[PairRelation, ...]:
    """Pair exactly one composite per visit under each shared temporal identity."""

    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be a TemporalObjectGrouping")
    if identity_source not in {"rescene", "geometric_baseline"}:
        raise ObjectRecoveryError("grouping identity source is invalid")
    if (
        isinstance(static_centroid_tolerance_m, bool)
        or not isinstance(static_centroid_tolerance_m, (int, float))
        or not math.isfinite(float(static_centroid_tolerance_m))
        or float(static_centroid_tolerance_m) < 0.0
    ):
        raise ObjectRecoveryError("static centroid tolerance is invalid")
    by_identity: dict[str, dict[int, object]] = {}
    for visit_id in (0, 1):
        for composite in grouping.objects[visit_id]:
            identity_id = composite.temporal_identity_id
            if identity_id is None:
                continue
            visits = by_identity.setdefault(identity_id, {})
            if visit_id in visits:
                raise ObjectRecoveryError(
                    "a temporal identity names multiple composites in one visit"
                )
            visits[visit_id] = composite
    entities = tuple(
        {entity.entity_id: entity for entity in grouping.snapshots[visit_id].entities}
        for visit_id in (0, 1)
    )
    relations: list[PairRelation] = []
    for identity_id, visits in sorted(by_identity.items()):
        if set(visits) != {0, 1}:
            continue
        t0 = visits[0]
        t1 = visits[1]
        t0_id = t0.object_id
        t1_id = t1.object_id
        distance = float(
            np.linalg.norm(
                np.mean(entities[0][t0_id].points_xyz, axis=0)
                - np.mean(entities[1][t1_id].points_xyz, axis=0)
            )
        )
        confidence = min(float(t0.query_confidence), float(t1.query_confidence))
        relations.append(
            PairRelation(
                temporal_query_id=f"composite:{identity_id}",
                t0_entity_ids=(t0_id,),
                t1_entity_ids=(t1_id,),
                state=(
                    "persistent_static"
                    if distance <= float(static_centroid_tolerance_m)
                    else "persistent_moved"
                ),
                query_confidence=confidence,
                evidence={
                    "centroid_distance_m": distance,
                    "query_support": min(t0.query_support, t1.query_support),
                },
                identity_source=identity_source,
            )
        )
    return tuple(relations)


def visit_maps_from_grouping(
    grouping: TemporalObjectGrouping,
    *,
    source_manifest_sha256: str,
) -> tuple[VisitMap, VisitMap]:
    """Wrap geometry-preserving grouping snapshots for two-visit composition."""

    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be a TemporalObjectGrouping")
    visit_lengths = tuple(
        max(int(grouping.snapshots[visit_id].timestamp), 0) + 1
        for visit_id in (0, 1)
    )
    visit_starts = (0, visit_lengths[0])
    maps = tuple(
        VisitMap(
            visit_id=visit_id,
            snapshot=grouping.snapshots[visit_id],
            coordinate_frame_id="3rscan_reference",
            source_manifest_sha256=source_manifest_sha256,
            map_voxel_size_m=0.01,
            observed_frame_start=visit_starts[visit_id],
            observed_frame_end=visit_starts[visit_id] + visit_lengths[visit_id] - 1,
        )
        for visit_id in (0, 1)
    )
    return maps  # type: ignore[return-value]


def _candidate_points(
    t0: VisitMap,
    relations: tuple[PairRelation, ...],
    registrations: tuple[RegistrationEvidence, ...],
) -> np.ndarray:
    entities = {entity.entity_id: entity for entity in t0.snapshot.entities}
    relation_by_id = {
        str(relation.temporal_query_id): relation for relation in relations
    }
    chunks: list[np.ndarray] = []
    for evidence in registrations:
        if not evidence.accepted:
            continue
        relation = relation_by_id[evidence.relation_id]
        source = entities[relation.t0_entity_ids[0]]
        chunks.append(
            apply_rigid_transform(
                source.points_xyz,
                evidence.transform_world_from_t0,
            )
        )
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _visibility_sha256(
    *, variant_id: str, baseline: TwoVisitCurrentMap, registrations: Sequence[RegistrationEvidence]
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "variant_id": variant_id,
                "baseline_sha256": baseline.content_sha256(),
                "registration_sha256": [value.content_sha256() for value in registrations],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def execute_recovery_variant(
    variant_id: str,
    grouping: TemporalObjectGrouping,
    relations: Sequence[PairRelation],
    *,
    frames: tuple[Frame, ...],
    baseline_visibility: SignedVisibilityGrid,
    source_manifest_sha256: str,
    visibility_config: SignedVisibilityConfig,
    registration_config: RegistrationConfig,
    composition_config: CompositionConfig,
    recovery_config: DenseRecoveryConfig,
    oracle_sources: Mapping[str, OracleTransformSource],
) -> VariantRecovery:
    """Compose and recover one C/O variant under shared visibility authority."""

    if variant_id not in {"C0", "C1", "C2", "O1", "O2"}:
        raise ObjectRecoveryError("recovery variant is invalid")
    if any(not isinstance(value, PairRelation) for value in relations):
        raise TypeError("relations contain an invalid value")
    relation_rows = tuple(relations)
    if variant_id == "C0" and relation_rows:
        raise ObjectRecoveryError("C0 cannot consume temporal relations")
    if variant_id == "O2":
        expected = {str(value.temporal_query_id) for value in relation_rows}
        if set(oracle_sources) != expected:
            raise ObjectRecoveryError("O2 oracle transform sources are incomplete")
    elif oracle_sources:
        raise ObjectRecoveryError("GT object transforms are reserved for O2")
    t0, t1 = visit_maps_from_grouping(
        grouping,
        source_manifest_sha256=source_manifest_sha256,
    )
    baseline = compose_current_map(
        t0,
        t1,
        relation_rows,
        baseline_visibility,
        composition_config,
    )
    t0_entities = {entity.entity_id: entity for entity in t0.snapshot.entities}
    t1_entities = {entity.entity_id: entity for entity in t1.snapshot.entities}
    registrations: list[RegistrationEvidence] = []
    for relation in relation_rows:
        source = t0_entities[relation.t0_entity_ids[0]]
        target = t1_entities[relation.t1_entity_ids[0]]
        arguments = {
            "source_semantic_label": source.semantic_label,
            "target_semantic_label": target.semantic_label,
            "config": registration_config,
        }
        if variant_id == "O2":
            evidence = register_oracle_pair_relation(
                relation,
                source.points_xyz,
                target.points_xyz,
                transform_source=oracle_sources[str(relation.temporal_query_id)],
                **arguments,
            )
        else:
            evidence = register_composite_relation(
                relation,
                source.points_xyz,
                target.points_xyz,
                **arguments,
            )
        registrations.append(evidence)
    registration_rows = tuple(registrations)
    candidates = _candidate_points(t0, relation_rows, registration_rows)
    candidate_visibility = derive_signed_visibility_for_points(
        candidates,
        frames,
        visibility_config,
        source_sha256=_visibility_sha256(
            variant_id=variant_id,
            baseline=baseline,
            registrations=registration_rows,
        ),
    )
    recovery = recover_dense_history(
        baseline,
        t0,
        t1,
        relation_rows,
        registration_rows,
        candidate_visibility,
        recovery_config,
    )
    return VariantRecovery(
        variant_id=variant_id,
        baseline=baseline,
        registrations=registration_rows,
        candidate_visibility=candidate_visibility,
        recovery=recovery,
    )


def _predicted_instances(
    grouping: TemporalObjectGrouping, visit_id: int, voxel_size_m: float
) -> tuple[PredictedInstance, ...]:
    return tuple(
        PredictedInstance(
            prediction_id=entity.entity_id,
            voxels=voxelize_points(entity.points_xyz, voxel_size_m=voxel_size_m),
        )
        for entity in grouping.snapshots[visit_id].entities
    )


def build_gt_oracle_relations(
    grouping: TemporalObjectGrouping,
    ground_truth: GroundTruthPair,
    *,
    iou_threshold: float,
) -> tuple[PairRelation, ...]:
    """Use evaluator-only GT identity to pair existing predicted OVI objects."""

    if not isinstance(grouping, TemporalObjectGrouping):
        raise TypeError("grouping must be a TemporalObjectGrouping")
    if not isinstance(ground_truth, GroundTruthPair):
        raise TypeError("ground_truth must be a GroundTruthPair")
    if ground_truth.pair_id != grouping.snapshots[0].scene_id:
        raise ObjectRecoveryError("grouping and ground truth name different pairs")
    if iou_threshold not in {0.25, 0.5}:
        raise ObjectRecoveryError("oracle IoU threshold must be 0.25 or 0.50")
    matches_by_visit: list[dict[int, tuple[str, float]]] = []
    for visit_id in (0, 1):
        geometry = evaluate_instance_geometry(
            _predicted_instances(grouping, visit_id, ground_truth.voxel_size_m),
            ground_truth.visits[visit_id],
            matching_policy="max_valid_count_then_iou",
        )
        result = geometry.primary if iou_threshold == 0.5 else geometry.sensitivity
        matches_by_visit.append(
            {
                match.gt_instance_id: (match.prediction_id, match.iou)
                for match in result.matches
            }
        )
    rescan_by_reference = {
        reference_id: rescan_id
        for rescan_id, reference_id in ground_truth.identity_rules.rescan_to_reference.items()
    }
    relations: list[PairRelation] = []
    for reference_id in sorted(
        ground_truth.identity_rules.rigid_transform_by_reference
    ):
        rescan_id = rescan_by_reference.get(reference_id, reference_id)
        t0_match = matches_by_visit[0].get(reference_id)
        t1_match = matches_by_visit[1].get(rescan_id)
        if t0_match is None or t1_match is None:
            continue
        relations.append(
            PairRelation(
                temporal_query_id=f"oracle:rigid:{reference_id}",
                t0_entity_ids=(t0_match[0],),
                t1_entity_ids=(t1_match[0],),
                state="persistent_moved",
                query_confidence=1.0,
                evidence={
                    "reference_instance_id": float(reference_id),
                    "t0_endpoint_iou": t0_match[1],
                    "t1_endpoint_iou": t1_match[1],
                    "ground_truth_identity_oracle": 1.0,
                },
                identity_source="geometric_baseline",
            )
        )
    return tuple(relations)


def _regular_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ObjectRecoveryError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise ObjectRecoveryError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if len(content) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ObjectRecoveryError(f"{label} changed while being read")
    return content


def _configured_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ObjectRecoveryError(f"{label} path is invalid")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _bound_content(record: object, *, label: str) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ObjectRecoveryError(f"{label} binding schema is invalid")
    path = _configured_path(record.get("path"), label=label)
    content = _regular_bytes(path, label=label)
    if (
        record.get("sha256") != hashlib.sha256(content).hexdigest()
        or record.get("byte_count") != len(content)
    ):
        raise ObjectRecoveryError(f"{label} binding mismatch")
    return path, content


def _json_object(content: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObjectRecoveryError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ObjectRecoveryError(f"{label} must contain a JSON object")
    return value


def _file_record(path: str | Path, *, recorded_path: str | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    content = _regular_bytes(absolute, label="output artifact")
    return {
        "path": str(absolute) if recorded_path is None else recorded_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _pair_record(selection: Mapping[str, object], pair_id: str) -> Mapping[str, object]:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list):
        raise ObjectRecoveryError("selection manifest pair list is invalid")
    matches = [
        value
        for value in pairs
        if isinstance(value, Mapping) and value.get("pair_id") == pair_id
    ]
    if len(matches) != 1:
        raise ObjectRecoveryError("selected recovery pair is not unique")
    return matches[0]


def _mapping_manifest_paths(
    mapping: Mapping[str, object],
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    visits = mapping.get("visits")
    if not isinstance(visits, list) or len(visits) != 2:
        raise ObjectRecoveryError("D2 mapping receipt visit schema is invalid")
    native: list[Path] = []
    materialized: list[Path] = []
    try:
        for visit_id, visit in enumerate(visits):
            if not isinstance(visit, Mapping) or visit.get("visit_index") != visit_id:
                raise ObjectRecoveryError("D2 mapping visits are not ordered")
            native.append(Path(visit["native_mapping"]["manifest"]["path"]))
            materialized.append(Path(visit["materialized"]["manifest"]["path"]))
    except (KeyError, TypeError) as error:
        raise ObjectRecoveryError("D2 mapping receipt paths are invalid") from error
    return (native[0], native[1]), (materialized[0], materialized[1])


def _cached_forward_outputs(
    metadata_content: bytes, arrays_content: bytes
) -> AssociationForwardOutputs:
    metadata = _json_object(metadata_content, label="cached forward metadata")
    forward = metadata.get("forward")
    if not isinstance(forward, Mapping):
        raise ObjectRecoveryError("cached forward metadata is incomplete")
    try:
        with np.load(io.BytesIO(arrays_content), allow_pickle=False) as archive:
            expected = {
                "independent_t0",
                "independent_t1",
                "pred_masks_mq",
                "pred_logits_qc",
            }
            if set(archive.files) != expected:
                raise ObjectRecoveryError("cached forward array schema is invalid")
            arrays = {name: archive[name] for name in expected}
        return AssociationForwardOutputs(
            independent_model_features=(
                arrays["independent_t0"],
                arrays["independent_t1"],
            ),
            visit_forward_sha256=tuple(forward["visit_forward_sha256"]),
            independent_runtime_s=tuple(forward["independent_runtime_s"]),
            joint_forward=NativeForwardResult(
                pred_masks_mq=arrays["pred_masks_mq"],
                pred_logits_qc=arrays["pred_logits_qc"],
                runtime_s=float(forward["joint_runtime_s"]),
                peak_memory_bytes=int(forward["peak_memory_bytes"]),
                peak_reserved_memory_bytes=int(forward["peak_reserved_memory_bytes"]),
                rss_peak_bytes=int(forward["rss_peak_bytes"]),
                device_name=str(forward["device_name"]),
                model_tensor_count=int(forward["model_tensor_count"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ObjectRecoveryError):
            raise
        raise ObjectRecoveryError("cached forward evidence is malformed") from error


def _relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ObjectRecoveryError(f"{label} path is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ObjectRecoveryError(f"{label} path must remain below the visit root")
    return Path(*pure.parts)


def _verified_tree_records(
    root: Path, payload: Mapping[str, object]
) -> dict[str, str]:
    tree = payload.get("output_tree")
    if not isinstance(tree, Mapping) or not isinstance(tree.get("files"), list):
        raise ObjectRecoveryError("materialized output tree is invalid")
    records: dict[str, Mapping[str, object]] = {}
    for raw in tree["files"]:
        if not isinstance(raw, Mapping):
            raise ObjectRecoveryError("materialized output record is invalid")
        relative = _relative_path(raw.get("path"), label="materialized output")
        key = relative.as_posix()
        if key in records:
            raise ObjectRecoveryError("materialized output paths are duplicated")
        records[key] = raw
    hashes: dict[str, str] = {}
    for relative, record in sorted(records.items()):
        if relative != "intrinsic/intrinsic_depth.txt" and not relative.startswith(
            ("color/", "depth/", "pose/")
        ):
            continue
        content = _regular_bytes(root / relative, label=f"materialized {relative}")
        observed = hashlib.sha256(content).hexdigest()
        if record.get("sha256") != observed or record.get("byte_count") != len(content):
            raise ObjectRecoveryError(f"materialized {relative} binding mismatch")
        hashes[relative] = observed
    return hashes


def load_materialized_visibility_frames(
    manifest_path: str | Path,
    *,
    world_from_visit: object,
) -> tuple[Frame, ...]:
    """Load bound depth frames and express camera poses in the reference frame."""

    path = Path(os.path.abspath(os.fspath(manifest_path)))
    content = _regular_bytes(path, label="materialized manifest")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObjectRecoveryError("materialized manifest is invalid JSON") from error
    if not isinstance(payload, Mapping) or payload.get("artifact_id") != "RSCAN_OVI_VISIT_V1":
        raise ObjectRecoveryError("materialized manifest identity is invalid")
    frame_count = payload.get("frame_count")
    frame_map = payload.get("frame_map")
    dimensions = payload.get("dimensions")
    if (
        type(frame_count) is not int
        or frame_count <= 0
        or not isinstance(frame_map, list)
        or len(frame_map) != frame_count
        or not isinstance(dimensions, Mapping)
        or not isinstance(dimensions.get("depth"), Mapping)
    ):
        raise ObjectRecoveryError("materialized frame contract is invalid")
    target_to_source: dict[int, int] = {}
    for record in frame_map:
        if not isinstance(record, Mapping):
            raise ObjectRecoveryError("materialized frame map is invalid")
        source = record.get("source_frame_id")
        target = record.get("target_frame_id")
        if type(source) is not int or source < 0 or type(target) is not int or target < 0:
            raise ObjectRecoveryError("materialized frame IDs are invalid")
        if target in target_to_source:
            raise ObjectRecoveryError("materialized target frame IDs are duplicated")
        target_to_source[target] = source
    if tuple(sorted(target_to_source)) != tuple(range(frame_count)):
        raise ObjectRecoveryError("materialized target frame IDs must be contiguous")
    hashes = _verified_tree_records(path.parent, payload)
    required = {"intrinsic/intrinsic_depth.txt"}
    input_hashes: dict[int, dict[str, str]] = {}
    for target in range(frame_count):
        records = {
            "color": f"color/{target}.jpg",
            "depth": f"depth/{target}.png",
            "pose": f"pose/{target}.txt",
        }
        required.update(records.values())
        input_hashes[target] = {role: hashes[relative] for role, relative in records.items()}
    if not required.issubset(hashes):
        raise ObjectRecoveryError("materialized visibility inputs are incomplete")
    depth = dimensions["depth"]
    try:
        expected_shape = (int(depth["height"]), int(depth["width"]))
        depth_scale = float(payload["depth_shift"])
    except (KeyError, TypeError, ValueError) as error:
        raise ObjectRecoveryError("materialized camera contract is invalid") from error
    transform = validate_rigid_transform(world_from_visit)
    dataset = ScanNet200Dataset(
        path.parent,
        source_frame_ids=tuple(range(frame_count)),
        input_hashes=input_hashes,
        expected_image_shape=expected_shape,
        depth_scale=depth_scale,
    )
    frames: list[Frame] = []
    for target in range(frame_count):
        source = dataset[target]
        frames.append(
            Frame(
                frame_id=target,
                source_frame_id=target_to_source[target],
                rgb=source.rgb,
                depth=source.depth,
                pose=transform @ np.asarray(source.pose, dtype=np.float64),
                intrinsics=source.intrinsics,
                timestamp=float(target),
            )
        )
    return tuple(frames)


def _snapshot_points(snapshot: object, *, entities_only: bool = False) -> np.ndarray:
    chunks = [
        np.asarray(entity.points_xyz, dtype=np.float32)
        for entity in snapshot.entities
        if len(entity.points_xyz)
    ]
    if not entities_only and snapshot.background_xyz is not None and len(
        snapshot.background_xyz
    ):
        chunks.append(np.asarray(snapshot.background_xyz, dtype=np.float32))
    return (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, 3), dtype=np.float32)
    )


def _gt_points(instances: Sequence[GroundTruthInstance], voxel_size_m: float) -> np.ndarray:
    voxels = sorted({voxel for instance in instances for voxel in instance.voxels})
    if not voxels:
        return np.empty((0, 3), dtype=np.float32)
    return (
        (np.asarray(voxels, dtype=np.float64) + 0.5) * voxel_size_m
    ).astype(np.float32)


def _visibility_points(
    grids: Sequence[SignedVisibilityGrid],
    *,
    status: str,
    target_xyz: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    keys = {
        tuple(int(axis) for axis in key)
        for grid in grids
        for key, value in zip(grid.voxel_keys, grid.statuses, strict=True)
        if value == status
    }
    target_keys = {
        tuple(int(axis) for axis in key)
        for key in np.floor(
            np.asarray(target_xyz, dtype=np.float64) / voxel_size_m
        ).astype(np.int64)
    }
    retained = sorted(keys - target_keys)
    if not retained:
        return np.empty((0, 3), dtype=np.float32)
    return (
        (np.asarray(retained, dtype=np.float64) + 0.5) * voxel_size_m
    ).astype(np.float32)


def _oracle_sources(
    relations: Sequence[PairRelation],
    ground_truth: GroundTruthPair,
    global_alignment_row: np.ndarray,
) -> dict[str, OracleTransformSource]:
    result: dict[str, OracleTransformSource] = {}
    for relation in relations:
        reference_id = int(relation.evidence["reference_instance_id"])
        raw = ground_truth.identity_rules.rigid_transform_by_reference.get(
            reference_id
        )
        if raw is None:
            raise ObjectRecoveryError("oracle relation lacks a rigid GT transform")
        result[str(relation.temporal_query_id)] = OracleTransformSource(
            reference_instance_id=reference_id,
            official_object_reference_to_rescan_row=np.asarray(raw).reshape(4, 4),
            official_rescan_to_reference_row=global_alignment_row,
        )
    return result


def _oracle_candidate_points(
    grouping: TemporalObjectGrouping,
    relations: Sequence[PairRelation],
    sources: Mapping[str, OracleTransformSource],
) -> np.ndarray:
    entities = {
        entity.entity_id: entity for entity in grouping.snapshots[0].entities
    }
    chunks = [
        apply_rigid_transform(
            entities[relation.t0_entity_ids[0]].points_xyz,
            sources[str(relation.temporal_query_id)].transform_world_from_t0,
        )
        for relation in relations
    ]
    return (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, 3), dtype=np.float64)
    )


def _voxel_set(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    return {
        tuple(int(axis) for axis in row)
        for row in np.floor(
            np.asarray(points, dtype=np.float64) / voxel_size_m
        ).astype(np.int64)
    }


def _old_location_residue(
    prediction_xyz: np.ndarray,
    ground_truth: GroundTruthPair,
) -> int:
    rigid_ids = set(ground_truth.identity_rules.rigid_transform_by_reference)
    old = _gt_points(
        tuple(
            value
            for value in ground_truth.visits[0]
            if value.instance_id in rigid_ids
        ),
        ground_truth.voxel_size_m,
    )
    current_ids = {
        rescan_id
        for rescan_id, reference_id in ground_truth.identity_rules.rescan_to_reference.items()
        if reference_id in rigid_ids
    }
    current = _gt_points(
        tuple(
            value
            for value in ground_truth.visits[1]
            if value.instance_id in current_ids
        ),
        ground_truth.voxel_size_m,
    )
    predicted = _voxel_set(prediction_xyz, ground_truth.voxel_size_m)
    return len(
        predicted
        & (
            _voxel_set(old, ground_truth.voxel_size_m)
            - _voxel_set(current, ground_truth.voxel_size_m)
        )
    )


def _relation_rigid_reference_count(
    grouping: TemporalObjectGrouping,
    ground_truth: GroundTruthPair,
    relations: Sequence[PairRelation],
) -> tuple[int, int]:
    bindings: list[dict[str, int]] = []
    for visit_id in (0, 1):
        geometry = evaluate_instance_geometry(
            _predicted_instances(
                grouping, visit_id, ground_truth.voxel_size_m
            ),
            ground_truth.visits[visit_id],
            matching_policy="max_valid_count_then_iou",
        )
        bindings.append(
            {
                match.prediction_id: match.gt_instance_id
                for match in geometry.sensitivity.matches
            }
        )
    rigid = set(ground_truth.identity_rules.rigid_transform_by_reference)
    correct = 0
    false_reid = 0
    for relation in relations:
        t0_id = bindings[0].get(relation.t0_entity_ids[0])
        t1_id = bindings[1].get(relation.t1_entity_ids[0])
        if t0_id is None or t1_id is None or t0_id not in rigid:
            continue
        t1_reference = ground_truth.identity_rules.reference_id_for_rescan(t1_id)
        if t1_reference == t0_id:
            correct += 1
        else:
            false_reid += 1
    return correct, false_reid


def _metric_row(
    *,
    variant: VariantRecovery,
    grouping: TemporalObjectGrouping,
    relations: tuple[PairRelation, ...],
    baseline_visibility: SignedVisibilityGrid,
    oracle_candidates: np.ndarray,
    oracle_relation_count: int,
    ground_truth: GroundTruthPair,
    pair_id: str,
    evaluated_commit: str,
    config_sha256: str,
) -> dict[str, object]:
    target = _gt_points(ground_truth.visits[1], ground_truth.voxel_size_m)
    baseline_xyz = _snapshot_points(variant.baseline.snapshot)
    recovered_xyz = _snapshot_points(variant.recovery.snapshot)
    t1_only = _snapshot_points(grouping.snapshots[1])
    known_negative = _visibility_points(
        (baseline_visibility, variant.candidate_visibility),
        status="visible_free",
        target_xyz=target,
        voxel_size_m=ground_truth.voxel_size_m,
    )
    total = evaluate_completion_surface_v2(
        baseline_xyz=baseline_xyz,
        recovered_xyz=recovered_xyz,
        target_xyz=target,
        historical_candidate_xyz=oracle_candidates,
        opportunity_baseline_xyz=t1_only,
        known_negative_xyz=known_negative,
        voxel_size_m=ground_truth.voxel_size_m,
    )
    object_metrics = evaluate_completion_surface_v2(
        baseline_xyz=_snapshot_points(variant.baseline.snapshot, entities_only=True),
        recovered_xyz=_snapshot_points(variant.recovery.snapshot, entities_only=True),
        target_xyz=target,
        historical_candidate_xyz=oracle_candidates,
        opportunity_baseline_xyz=_snapshot_points(
            grouping.snapshots[1], entities_only=True
        ),
        known_negative_xyz=known_negative,
        voxel_size_m=ground_truth.voxel_size_m,
    )
    correct_rigid, false_reid = _relation_rigid_reference_count(
        grouping, ground_truth, relations
    )
    decisions = {
        name: sum(
            group.output_point_count
            if name.startswith("recover_")
            else len(group.source_point_indices)
            for group in variant.recovery.provenance
            if group.decision == name
        )
        for name in (
            "recover_occluded",
            "recover_unobserved",
            "reject_incompatible",
            "reject_occupied",
            "reject_visible_free",
        )
    }
    recovered_total = total["recovered_total"]
    baseline_total = total["baseline_total"]
    current_object = object_metrics["recovered_total"]
    return {
        "pair_id": pair_id,
        "variant_id": variant.variant_id,
        "mode": "oracle" if variant.variant_id.startswith("O") else "method",
        "evaluated_commit": evaluated_commit,
        "config_sha256": config_sha256,
        "grouping_variant": grouping.variant_id,
        "paired_object_count": len(relations),
        "rigid_gt_count": len(
            ground_truth.identity_rules.rigid_transform_by_reference
        ),
        "rigid_proposal_pair_count": oracle_relation_count,
        "rigid_correct_relation_count": correct_rigid,
        "rigid_false_reid_count": false_reid,
        "accepted_registration_count": sum(
            value.accepted for value in variant.registrations
        ),
        "rejected_registration_count": sum(
            not value.accepted for value in variant.registrations
        ),
        "opportunity_voxel_count": total["opportunity_voxel_count"],
        "recovered_opportunity_voxel_count": total[
            "recovered_opportunity_voxel_count"
        ],
        "historical_unevaluable_voxel_count": total[
            "historical_unevaluable_voxel_count"
        ],
        "new_surface_voxel_count": total["new_surface_voxel_count"],
        "new_correct_surface_voxel_count": total[
            "new_correct_surface_voxel_count"
        ],
        "new_wrong_surface_voxel_count": total[
            "new_wrong_surface_voxel_count"
        ],
        "new_unevaluable_surface_voxel_count": total[
            "new_unevaluable_surface_voxel_count"
        ],
        "deleted_baseline_voxel_count": total["deleted_baseline_voxel_count"],
        "deleted_correct_baseline_voxel_count": total[
            "deleted_correct_baseline_voxel_count"
        ],
        "baseline_surface_precision": baseline_total["precision"],
        "baseline_surface_recall": baseline_total["recall"],
        "baseline_surface_f_score": baseline_total["f_score"],
        "surface_precision": recovered_total["precision"],
        "surface_recall": recovered_total["recall"],
        "surface_f_score": recovered_total["f_score"],
        "current_object_precision": current_object["precision"],
        "current_object_recall": current_object["recall"],
        "old_location_residue_voxel_count": _old_location_residue(
            recovered_xyz, ground_truth
        ),
        "background_conflict_voxel_count": total[
            "new_wrong_surface_voxel_count"
        ],
        "recover_occluded_point_count": decisions["recover_occluded"],
        "recover_unobserved_point_count": decisions["recover_unobserved"],
        "reject_incompatible_point_count": decisions["reject_incompatible"],
        "reject_occupied_point_count": decisions["reject_occupied"],
        "reject_visible_free_point_count": decisions["reject_visible_free"],
        "recovered_point_count": variant.recovery.recovered_point_count,
        "removed_baseline_point_count": variant.recovery.removed_baseline_point_count,
        "status": "PASS_NULL_PRESERVING",
    }


def _rigid_object_details(
    *,
    variant: VariantRecovery,
    grouping: TemporalObjectGrouping,
    relations: tuple[PairRelation, ...],
    oracle_relations: tuple[PairRelation, ...],
    oracle_sources: Mapping[str, OracleTransformSource],
    ground_truth: GroundTruthPair,
    known_negative: np.ndarray,
) -> list[dict[str, object]]:
    relations_by_t0 = {value.t0_entity_ids[0]: value for value in relations}
    registrations = {value.relation_id: value for value in variant.registrations}
    t0_entities = {
        value.entity_id: value for value in grouping.snapshots[0].entities
    }
    target_by_id = {
        value.instance_id: value for value in ground_truth.visits[1]
    }
    rescan_by_reference = {
        reference_id: rescan_id
        for rescan_id, reference_id in ground_truth.identity_rules.rescan_to_reference.items()
    }
    t1_only = _snapshot_points(grouping.snapshots[1])
    details: list[dict[str, object]] = []
    oracle_by_reference = {
        int(value.evidence["reference_instance_id"]): value
        for value in oracle_relations
    }
    for reference_id in sorted(
        ground_truth.identity_rules.rigid_transform_by_reference
    ):
        oracle_relation = oracle_by_reference.get(reference_id)
        if oracle_relation is None:
            details.append(
                {
                    "reference_instance_id": reference_id,
                    "status": "NULL_MISSING_PREDICTED_ENDPOINT",
                    "relation_status": None,
                    "registration_accepted": None,
                    "opportunity_voxel_count": None,
                    "new_correct_surface_voxel_count": None,
                    "new_wrong_surface_voxel_count": None,
                    "new_unevaluable_surface_voxel_count": None,
                }
            )
            continue
        source_id = oracle_relation.t0_entity_ids[0]
        expected_target_id = oracle_relation.t1_entity_ids[0]
        method_relation = relations_by_t0.get(source_id)
        relation_status = (
            "MISSING"
            if method_relation is None
            else "MATCHED_CORRECT"
            if method_relation.t1_entity_ids[0] == expected_target_id
            else "FALSE_REID"
        )
        registration = (
            None
            if method_relation is None
            else registrations.get(str(method_relation.temporal_query_id))
        )
        source = t0_entities[source_id]
        oracle_source = oracle_sources[str(oracle_relation.temporal_query_id)]
        historical = apply_rigid_transform(
            source.points_xyz, oracle_source.transform_world_from_t0
        )
        rescan_id = rescan_by_reference.get(reference_id, reference_id)
        target_instance = target_by_id.get(rescan_id)
        if target_instance is None:
            raise ObjectRecoveryError("rigid target instance is absent from GT")
        target = _gt_points((target_instance,), ground_truth.voxel_size_m)
        recovered_chunks: list[np.ndarray] = []
        if method_relation is not None and registration is not None and registration.accepted:
            transformed = apply_rigid_transform(
                source.points_xyz, registration.transform_world_from_t0
            )
            for group in variant.recovery.provenance:
                if (
                    group.relation_id == str(method_relation.temporal_query_id)
                    and group.decision.startswith("recover_")
                ):
                    recovered_chunks.append(transformed[group.source_point_indices])
        recovered = (
            np.concatenate(recovered_chunks, axis=0)
            if recovered_chunks
            else np.empty((0, 3), dtype=np.float64)
        )
        metrics = evaluate_completion_surface_v2(
            baseline_xyz=np.empty((0, 3), dtype=np.float64),
            recovered_xyz=recovered,
            target_xyz=target,
            historical_candidate_xyz=historical,
            opportunity_baseline_xyz=t1_only,
            known_negative_xyz=known_negative,
            voxel_size_m=ground_truth.voxel_size_m,
        )
        details.append(
            {
                "reference_instance_id": reference_id,
                "semantic_label": target_instance.semantic_label,
                "source_object_id": source_id,
                "expected_target_object_id": expected_target_id,
                "selected_target_object_id": (
                    None if method_relation is None else method_relation.t1_entity_ids[0]
                ),
                "status": "PASS",
                "relation_status": relation_status,
                "registration_accepted": (
                    None if registration is None else registration.accepted
                ),
                "registration_rejection_reasons": (
                    None
                    if registration is None
                    else list(registration.rejection_reasons)
                ),
                "opportunity_voxel_count": metrics["opportunity_voxel_count"],
                "new_correct_surface_voxel_count": metrics[
                    "new_correct_surface_voxel_count"
                ],
                "new_wrong_surface_voxel_count": metrics[
                    "new_wrong_surface_voxel_count"
                ],
                "new_unevaluable_surface_voxel_count": metrics[
                    "new_unevaluable_surface_voxel_count"
                ],
            }
        )
    return details


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    values = tuple(rows)
    if not values or any(tuple(value) != tuple(values[0]) for value in values):
        raise ObjectRecoveryError("completion CSV rows are inconsistent")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=tuple(values[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(values)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def run_object_recovery(
    config_path: str | Path,
    *,
    evaluated_commit: str,
) -> dict[str, object]:
    """Run C0/C1/C2/O1/O2 from frozen D2 inputs and cached forward outputs."""

    if not _is_digest(evaluated_commit, width=40):
        raise ObjectRecoveryError("evaluated commit is invalid")
    config_absolute = Path(os.path.abspath(os.fspath(config_path)))
    config_content = _regular_bytes(config_absolute, label="recovery config")
    config = _json_object(config_content, label="recovery config")
    if (
        config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_OBJECT_RECOVERY_V1"
        or config.get("status") != "FROZEN_BEFORE_COMPLETION_RESULTS"
        or set(config.get("source_bindings", {})) != _SOURCE_BINDINGS
    ):
        raise ObjectRecoveryError("recovery config identity is invalid")
    config_sha256 = hashlib.sha256(config_content).hexdigest()
    pair_id = config.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ObjectRecoveryError("recovery pair ID is invalid")
    bindings = config["source_bindings"]
    assert isinstance(bindings, Mapping)
    loaded = {
        name: _bound_content(bindings[name], label=name)
        for name in sorted(_SOURCE_BINDINGS)
    }
    decision = _json_object(
        loaded["adaptation_decision"][1], label="adaptation decision"
    )
    if decision.get("action") != "NO_ADAPTATION" or decision.get("pair_id") != pair_id:
        raise ObjectRecoveryError("recovery violates the frozen adaptation gate")
    object_config = load_object_transfer_config(
        loaded["object_transfer_config"][0]
    )
    selection = _json_object(
        loaded["selection_manifest"][1], label="selection manifest"
    )
    pair_record = _pair_record(selection, pair_id)
    mapping = _json_object(
        loaded["d2_mapping_receipt"][1], label="D2 mapping receipt"
    )
    native_manifests, materialized_manifests = _mapping_manifest_paths(mapping)
    selection_sha256 = hashlib.sha256(loaded["selection_manifest"][1]).hexdigest()
    pair = build_ovi_object_pair_view(
        pair_record=pair_record,
        source_manifest_sha256=selection_sha256,
        native_manifests=native_manifests,
        materialized_manifests=materialized_manifests,
    )
    pair_receipt = _json_object(
        loaded["d2_pair_receipt"][1], label="D2 pair receipt"
    )
    pair = restore_bound_ovi_pair_artifact(pair, pair_receipt)
    if pair.pair_id != pair_id:
        raise ObjectRecoveryError("D2 pair identity mismatch")
    ground_truth = load_pair_ground_truth(loaded["ground_truth_manifest"][0])
    if ground_truth.pair_id != pair_id:
        raise ObjectRecoveryError("ground truth names a different pair")
    outputs = _cached_forward_outputs(
        loaded["d2_forward_metadata"][1], loaded["d2_forward_arrays"][1]
    )
    bundle = build_d2_inference_bundle(
        pair,
        sampler_source_path=object_config.native_sampler_source.path,
        sampler_source_sha256=object_config.native_sampler_source.sha256,
        sampler_seed=object_config.sampler_seed,
        maximum_candidates=object_config.maximum_candidates,
    )
    shared = evaluate_shared_candidate_methods(
        pair,
        bundle,
        outputs,
        ground_truth,
        minimum_match_scores=object_config.minimum_match_scores,
        centroid_scale_m=object_config.centroid_scale_m,
        static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        backend_config_sha256=object_config.source_sha256,
        checkpoint_sha256=object_config.rescene_checkpoint.sha256,
    )
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ObjectRecoveryError("recovery protocol is invalid")
    grouping_config = GroupingConfig(
        minimum_query_confidence=float(protocol["minimum_query_confidence"])
    )
    projection = project_queries_to_instances(
        bundle.sample,
        build_temporal_query_evidence(
            bundle,
            outputs,
            backend_config_sha256=object_config.source_sha256,
            checkpoint_sha256=object_config.rescene_checkpoint.sha256,
        ),
        ProjectionConfig(
            minimum_entity_token_coverage=float(
                protocol["minimum_entity_token_coverage"]
            ),
            minimum_source_point_coverage=float(
                protocol["minimum_source_point_coverage"]
            ),
            static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        ),
    )
    g_input_relations = relations_from_object_predictions(
        shared.predictions[str(protocol["geometric_method_id"])]
    )
    groupings = {
        "C0": build_temporal_object_groups(
            pair, (), variant_id="U0", config=grouping_config
        ),
        "C1": build_temporal_object_groups(
            pair, g_input_relations, variant_id="U3", config=grouping_config
        ),
        "C2": build_temporal_object_groups(
            pair, projection.relations, variant_id="U3", config=grouping_config
        ),
    }
    groupings["O1"] = groupings["C2"]
    groupings["O2"] = groupings["C2"]
    relations = {
        "C0": (),
        "C1": relations_from_temporal_grouping(
            groupings["C1"],
            identity_source="geometric_baseline",
            static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        ),
        "C2": relations_from_temporal_grouping(
            groupings["C2"],
            identity_source="rescene",
            static_centroid_tolerance_m=object_config.static_centroid_tolerance_m,
        ),
        "O1": build_gt_oracle_relations(
            groupings["O1"],
            ground_truth,
            iou_threshold=float(protocol["oracle_endpoint_iou_threshold"]),
        ),
    }
    relations["O2"] = relations["O1"]
    frames = load_materialized_visibility_frames(
        materialized_manifests[1],
        world_from_visit=official_row_vector_to_internal_transform(
            pair.global_alignment
        ),
    )
    visibility_config = SignedVisibilityConfig(**config["visibility"])
    registration_payload = dict(config["registration"])
    registration_payload["recoverable_semantic_labels"] = frozenset(
        registration_payload["recoverable_semantic_labels"]
    )
    registration_config = RegistrationConfig(**registration_payload)
    composition_payload = dict(config["composition"])
    composition_payload["object_semantic_labels"] = frozenset(
        composition_payload["object_semantic_labels"]
    )
    composition_config = CompositionConfig(**composition_payload)
    recovery_config = DenseRecoveryConfig(**config["recovery"])
    c0_t0, _ = visit_maps_from_grouping(
        groupings["C0"], source_manifest_sha256=pair.source_manifest_sha256
    )
    baseline_visibility = derive_signed_visibility(
        c0_t0,
        frames,
        visibility_config,
        source_sha256=hashlib.sha256(
            b"OVI_RESCENE_SHARED_T1_VISIBILITY_V1\0"
            + config_content
            + loaded["d2_mapping_receipt"][1]
        ).hexdigest(),
    )
    oracle_sources_by_variant: dict[str, dict[str, OracleTransformSource]] = {}
    oracle_relations_by_variant: dict[str, tuple[PairRelation, ...]] = {}
    oracle_candidates_by_variant: dict[str, np.ndarray] = {}
    for variant_id in _VARIANTS:
        oracle_relations = build_gt_oracle_relations(
            groupings[variant_id],
            ground_truth,
            iou_threshold=float(protocol["oracle_endpoint_iou_threshold"]),
        )
        sources = _oracle_sources(
            oracle_relations,
            ground_truth,
            pair.global_alignment,
        )
        oracle_relations_by_variant[variant_id] = oracle_relations
        oracle_sources_by_variant[variant_id] = sources
        oracle_candidates_by_variant[variant_id] = _oracle_candidate_points(
            groupings[variant_id], oracle_relations, sources
        )
    variants = {
        variant_id: execute_recovery_variant(
            variant_id,
            groupings[variant_id],
            relations[variant_id],
            frames=frames,
            baseline_visibility=baseline_visibility,
            source_manifest_sha256=pair.source_manifest_sha256,
            visibility_config=visibility_config,
            registration_config=registration_config,
            composition_config=composition_config,
            recovery_config=recovery_config,
            oracle_sources=(
                oracle_sources_by_variant[variant_id]
                if variant_id == "O2"
                else {}
            ),
        )
        for variant_id in _VARIANTS
    }
    rows = tuple(
        _metric_row(
            variant=variants[variant_id],
            grouping=groupings[variant_id],
            relations=relations[variant_id],
            baseline_visibility=baseline_visibility,
            oracle_candidates=oracle_candidates_by_variant[variant_id],
            oracle_relation_count=len(oracle_relations_by_variant[variant_id]),
            ground_truth=ground_truth,
            pair_id=pair_id,
            evaluated_commit=evaluated_commit,
            config_sha256=config_sha256,
        )
        for variant_id in _VARIANTS
    )
    outputs_config = config.get("outputs")
    if not isinstance(outputs_config, Mapping):
        raise ObjectRecoveryError("recovery output configuration is invalid")
    csv_path = _configured_path(outputs_config.get("completion_csv"), label="CSV")
    receipt_path = _configured_path(outputs_config.get("receipt"), label="receipt")
    local_root = _configured_path(outputs_config.get("local_root"), label="local root")
    if any(path.exists() or path.is_symlink() for path in (csv_path, receipt_path, local_root)):
        raise ObjectRecoveryError("a configured recovery output already exists")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{local_root.name}.", dir=local_root.parent)
    )
    local_records: dict[str, dict[str, object]] = {}
    try:
        for variant_id in _VARIANTS:
            manifest_path = write_dense_recovery(
                variants[variant_id].recovery,
                temporary_root / variant_id,
            )
            local_records[variant_id] = _file_record(manifest_path)
        os.replace(temporary_root, local_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    for record in local_records.values():
        old = Path(str(record["path"]))
        record["path"] = str(local_root / old.relative_to(temporary_root))
    target = _gt_points(ground_truth.visits[1], ground_truth.voxel_size_m)
    rigid_details = {}
    for variant_id in _VARIANTS:
        known_negative = _visibility_points(
            (baseline_visibility, variants[variant_id].candidate_visibility),
            status="visible_free",
            target_xyz=target,
            voxel_size_m=ground_truth.voxel_size_m,
        )
        rigid_details[variant_id] = _rigid_object_details(
            variant=variants[variant_id],
            grouping=groupings[variant_id],
            relations=relations[variant_id],
            oracle_relations=oracle_relations_by_variant[variant_id],
            oracle_sources=oracle_sources_by_variant[variant_id],
            ground_truth=ground_truth,
            known_negative=known_negative,
        )
    csv_content = _csv_bytes(rows)
    receipt = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_OBJECT_RECOVERY_RESULT_V1",
        "status": "PASS_NULL_PRESERVING",
        "pair_id": pair_id,
        "domain_id": pair.domain_id,
        "evaluated_commit": evaluated_commit,
        "config_sha256": config_sha256,
        "pair_content_sha256": pair.content_sha256(),
        "protocol": dict(protocol),
        "variants": {
            row["variant_id"]: {
                **row,
                "registration_rejection_reasons": sorted(
                    {
                        reason
                        for value in variants[str(row["variant_id"])].registrations
                        for reason in value.rejection_reasons
                    }
                ),
                "rigid_objects": rigid_details[str(row["variant_id"])],
                "local_manifest": local_records[str(row["variant_id"])],
            }
            for row in rows
        },
        "source_bindings": {
            "config": _file_record(
                config_absolute,
                recorded_path=str(config_absolute.relative_to(REPO_ROOT)),
            ),
            "completion_csv": {
                "path": str(csv_path.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(csv_content).hexdigest(),
                "byte_count": len(csv_content),
            },
            **{name: dict(bindings[name]) for name in sorted(bindings)},
        },
        "implementation_sources": {
            "runner": _file_record(
                Path(__file__), recorded_path=str(Path(__file__).relative_to(REPO_ROOT))
            ),
            "registration": _file_record(
                REPO_ROOT / "src/oviv2/two_visit_registration.py",
                recorded_path="src/oviv2/two_visit_registration.py",
            ),
            "recovery": _file_record(
                REPO_ROOT / "src/oviv2/two_visit_dense_recovery.py",
                recorded_path="src/oviv2/two_visit_dense_recovery.py",
            ),
            "metrics": _file_record(
                REPO_ROOT / "src/evaluation/ovi_ownership_completion.py",
                recorded_path="src/evaluation/ovi_ownership_completion.py",
            ),
        },
        "claim_boundary": {
            "establishes": (
                "measured C0/C1/C2 recovery and O1/O2 oracle diagnosis on the frozen "
                "D2_DEV pair with transformed-point t1 visibility"
            ),
            "does_not_establish": (
                "held-out D2_EVAL performance, continuous-motion mapping, or recovery "
                "quality outside explicitly observed GT and visibility domains"
            ),
        },
    }
    try:
        _write_exclusive(csv_path, csv_content)
        _write_exclusive(receipt_path, _json_bytes(receipt))
    except BaseException:
        csv_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
    return {
        "status": receipt["status"],
        "pair_id": pair_id,
        "completion_csv": str(csv_path),
        "receipt": str(receipt_path),
        "local_root": str(local_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluated-commit", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_object_recovery(
            arguments.config,
            evaluated_commit=arguments.evaluated_commit,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "ObjectRecoveryError",
    "VariantRecovery",
    "build_gt_oracle_relations",
    "execute_recovery_variant",
    "load_materialized_visibility_frames",
    "relations_from_object_predictions",
    "relations_from_temporal_grouping",
    "run_object_recovery",
    "visit_maps_from_grouping",
]


if __name__ == "__main__":
    raise SystemExit(main())
