#!/usr/bin/env python3
"""Build and evaluate source-preserving CROVE fine current maps."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import (  # noqa: E402
    NON_INSTANCE_CLASSES,
    load_replica_ground_truth,
)
from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (  # noqa: E402
    _evaluation_context,
)
from src.datasets.replica import ReplicaRoom0Dataset  # noqa: E402
from src.datasets.tesse_cd import TesseCdRgbdDataset  # noqa: E402
from src.evaluation.baselines.ovimap import parse_instance_color_log  # noqa: E402
from src.evaluation.baselines.tesse_semantics import (  # noqa: E402
    TesseSemanticCrosswalk,
    load_tesse_semantic_crosswalk,
)
from src.evaluation.contracts import EntityPrediction, MapSnapshot  # noqa: E402
from src.evaluation.oviv2_replica import (  # noqa: E402
    EntityEvaluationInfo,
    evaluate_replica_voxel_map,
)
from src.evaluation.two_visit_snapshot_metrics import (  # noqa: E402
    evaluate_two_visit_snapshot,
)
from src.oviv2.current_surface import (  # noqa: E402
    CurrentEvidenceState,
    SemanticSource,
    export_current_surface,
)
from src.oviv2.fine_current_composer import (  # noqa: E402
    FineVisitSurface,
    compose_two_visit_fine_surface,
    current_surface_from_visit,
)
from src.oviv2.fine_dynamic_policy import (  # noqa: E402
    CoarseSurfaceState,
    assemble_fine_surface_evidence,
    load_b3_surface_policy,
)
from src.oviv2.fine_surface_io import (  # noqa: E402
    bind_ovi_owner_ids,
    load_cached_native_ovi_surface,
    load_native_ovi_surface,
)
from src.oviv2.fine_surface_validity import (  # noqa: E402
    FineEvidenceProjectionConfig,
    FineSurfaceEvidence,
    FineSurfaceObservation,
    FineValidityConfig,
    observe_fine_surface,
    resolve_fine_current_validity,
)
from src.oviv2.meshing import LabeledMesh  # noqa: E402
from src.oviv2.surface_semantics import (  # noqa: E402
    SurfaceSemanticConfig,
    SurfaceSemanticStrategy,
    transfer_surface_semantics,
)
from src.oviv2.two_visit_execution import (  # noqa: E402
    SignedVisibilityConfig,
    derive_observed_point_mask,
)

_MAPPING_RE = re.compile(
    r"inst-(?P<instance>\d+)_label-(?P<file_label>\d+)\.npy\s+"
    r"(?P<label>\d+)\s+(?P<confidence>[0-9.eE+-]+)"
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _configured_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ovimap_semantic_mapping(
    path: str | Path,
    *,
    source_vocabulary: tuple[str, ...],
    target_vocabulary: tuple[str, ...],
) -> dict[int, tuple[int, float]]:
    """Translate OVI semantic IDs through class names into the frozen target IDs."""

    if not source_vocabulary or not target_vocabulary:
        raise ValueError("source and target vocabularies must be non-empty")
    target_ids = {name: index + 1 for index, name in enumerate(target_vocabulary)}
    result: dict[int, tuple[int, float]] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = _MAPPING_RE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"invalid OVI semantic mapping line {line_number}")
        instance_id = int(match.group("instance"))
        file_label = int(match.group("file_label"))
        source_id = int(match.group("label"))
        confidence = float(match.group("confidence"))
        if file_label != source_id or not 1 <= source_id <= len(source_vocabulary):
            raise ValueError(f"invalid OVI semantic source ID at line {line_number}")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid OVI semantic confidence at line {line_number}")
        if instance_id in result:
            raise ValueError(f"duplicate OVI instance semantic mapping: {instance_id}")
        target_id = target_ids.get(source_vocabulary[source_id - 1], 0)
        result[instance_id] = (
            target_id,
            confidence if target_id > 0 else 0.0,
        )
    if not result:
        raise ValueError("OVI semantic mapping is empty")
    return result


def select_static_semantic_strategy(
    metrics: dict[SurfaceSemanticStrategy, dict[str, Any]],
) -> SurfaceSemanticStrategy:
    """Apply the frozen DEV rule: mIoU, then f-mIoU, then simpler readout."""

    expected = set(SurfaceSemanticStrategy)
    if set(metrics) != expected:
        raise ValueError("static selection requires exactly S0, S1, and S2")
    complexity = {
        SurfaceSemanticStrategy.S0: 0,
        SurfaceSemanticStrategy.S1: 1,
        SurfaceSemanticStrategy.S2: 2,
    }
    return max(
        SurfaceSemanticStrategy,
        key=lambda strategy: (
            float(metrics[strategy]["miou"]),
            float(metrics[strategy]["f_miou"]),
            -complexity[strategy],
        ),
    )


def select_dynamic_trial(
    metrics: dict[str, dict[str, float]],
    *,
    baseline_trial_id: str,
    gates: dict[str, float],
) -> tuple[str, dict[str, bool]]:
    """Select the strongest current-map trial after frozen quality gates."""

    if baseline_trial_id not in metrics:
        raise ValueError("dynamic metrics do not contain the baseline trial")
    required_gates = {
        "maximum_ghost",
        "minimum_background_f1_at_5cm",
        "minimum_surface_precision_at_5cm",
        "minimum_observed_stale_precision",
    }
    if set(gates) != required_gates:
        raise ValueError("dynamic selection gates are incomplete")
    eligibility = {
        trial_id: (
            float(values["ghost"]) <= float(gates["maximum_ghost"])
            and float(values["background_f1_at_5cm"])
            >= float(gates["minimum_background_f1_at_5cm"])
            and float(values["surface_precision_at_5cm"])
            >= float(gates["minimum_surface_precision_at_5cm"])
            and float(values["t1_observed_region_stale_precision"])
            >= float(gates["minimum_observed_stale_precision"])
        )
        for trial_id, values in metrics.items()
    }
    eligible = [trial_id for trial_id, accepted in eligibility.items() if accepted]
    if not eligible:
        return baseline_trial_id, eligibility
    selected = max(
        eligible,
        key=lambda trial_id: (
            float(metrics[trial_id]["current_miou"]),
            float(metrics[trial_id]["background_f1_at_5cm"]),
            float(metrics[trial_id].get("surface_f1_at_5cm", 0.0)),
            trial_id == baseline_trial_id,
        ),
    )
    return selected, eligibility


def _semantic_configs(payload: dict[str, Any]) -> tuple[SurfaceSemanticConfig, ...]:
    shared = payload["semantic_transfer"]
    return tuple(
        SurfaceSemanticConfig(
            strategy=strategy,
            maximum_correspondence_distance_m=float(
                shared["maximum_correspondence_distance_m"]
            ),
            distance_sigma_m=float(shared["distance_sigma_m"]),
            minimum_local_support=float(shared["minimum_local_support"]),
            entity_weight_scale=float(shared["entity_weight_scale"]),
            minimum_entity_confidence=float(shared["minimum_entity_confidence"]),
        )
        for strategy in SurfaceSemanticStrategy
    )


def _load_crove_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = PlyData.read(path, mmap="c")["vertex"]
    names = set(vertices.data.dtype.names or ())
    required = {"x", "y", "z", "semantic_id", "semantic_confidence"}
    if not required.issubset(names):
        raise ValueError(f"CROVE reference lacks properties: {sorted(required - names)}")
    return (
        np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
            np.float32
        ),
        np.asarray(vertices["semantic_id"], dtype=np.int32),
        np.asarray(vertices["semantic_confidence"], dtype=np.float32),
    )


def _owner_semantic_arrays(
    owner_ids: np.ndarray,
    mapping: dict[int, tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(int(owner_ids.max(initial=0)), max(mapping, default=0))
    semantic_lut = np.zeros(maximum + 1, dtype=np.int32)
    confidence_lut = np.zeros(maximum + 1, dtype=np.float32)
    for owner_id, (semantic_id, confidence) in mapping.items():
        semantic_lut[owner_id] = semantic_id
        confidence_lut[owner_id] = confidence
    return semantic_lut[owner_ids], confidence_lut[owner_ids]


def _static_rgb_observation(
    vertices_xyz: np.ndarray,
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rgb = case["rgb_projection"]
    dataset = ReplicaRoom0Dataset(rgb["dataset_root"])
    start = int(rgb["frame_start"])
    stop = int(rgb["frame_stop_exclusive"])
    source_stride = int(rgb["source_stride"])
    subsample = int(rgb["projection_subsample"])
    frame_indices = tuple(range(start, stop, source_stride * subsample))
    if any(dataset.frame_indices[index] != index for index in frame_indices):
        raise ValueError("Replica dataset indices do not match source frame IDs")
    frames = tuple(dataset[index] for index in frame_indices)

    unique_vertices, inverse = np.unique(
        vertices_xyz, axis=0, return_inverse=True
    )
    projection_config = FineEvidenceProjectionConfig(
        depth_tolerance_m=float(rgb["depth_tolerance_m"]),
        depth_max_m=float(rgb["depth_max_m"]),
        minimum_viewpoint_baseline_m=float(rgb["minimum_viewpoint_baseline_m"]),
        maximum_distinct_viewpoints=int(rgb["maximum_distinct_viewpoints"]),
        minimum_rgb_neighbours=int(rgb["minimum_rgb_neighbours"]),
        rgb_edge_range_factor=float(rgb["edge_range_factor"]),
    )
    observation = observe_fine_surface(
        unique_vertices,
        frames,
        projection_config,
        point_batch_size=int(rgb["point_batch_size"]),
    )
    return (
        observation.observed_rgb_uint8[inverse],
        observation.rgb_valid[inverse],
        observation.evidence.last_supported_frames[inverse],
        {
            "source": "same_visit_rgbd_reprojected",
            "frame_ids": list(frame_indices),
            "source_vertex_count": len(vertices_xyz),
            "unique_projected_vertex_count": len(unique_vertices),
            "rgb_valid_vertex_count": int(np.sum(observation.rgb_valid[inverse])),
            "rgb_valid_ratio": float(np.mean(observation.rgb_valid[inverse])),
            "config": rgb,
        },
    )


def _palette(path: Path, vocabulary: tuple[str, ...]) -> dict[int, tuple[int, int, int]]:
    payload = _json(path)
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != len(vocabulary):
        raise ValueError("semantic palette does not match the target vocabulary")
    result = {0: tuple(int(value) for value in payload["background"]["rgb"])}
    for index, (expected_name, record) in enumerate(zip(vocabulary, classes), start=1):
        if record.get("name") != expected_name:
            raise ValueError("semantic palette class order is invalid")
        result[index] = tuple(int(value) for value in record["rgb"])
    return result


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(metrics[name])
        for name in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")
    }


def _static_table_metric(token: str) -> str:
    prefix = "CROVE_FINE_ROOM0_DEV_"
    if not token.startswith(prefix) or len(token) == len(prefix):
        raise ValueError("invalid static table token")
    return token.removeprefix(prefix).lower()


def _entity_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid entity JSON at line {line_number}: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"invalid entity record at line {line_number}: {path}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"entity metadata is missing at line {line_number}: {path}")
        owner_id = int(metadata.get("source_instance_id", -1))
        if payload.get("entity_id") != f"ovimap:{owner_id}" or owner_id <= 0:
            raise ValueError(f"entity identity mismatch at line {line_number}: {path}")
        label = payload.get("semantic_label")
        score = float(payload.get("semantic_score", 0.0))
        last_seen = float(payload.get("last_seen", -1.0))
        first_seen = float(payload.get("first_seen", 0.0))
        if (
            owner_id in records
            or (label is not None and (not isinstance(label, str) or not label.strip()))
            or not np.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not np.isfinite(first_seen)
            or not np.isfinite(last_seen)
            or first_seen < 0.0
            or first_seen > last_seen
        ):
            raise ValueError(f"invalid entity semantics at line {line_number}: {path}")
        records[owner_id] = {
            "entity_id": payload["entity_id"],
            "semantic_label": label.strip() if label is not None else None,
            "semantic_score": score,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }
    if not records:
        raise ValueError(f"entity file is empty: {path}")
    return records


def _native_artifact_paths(
    native_manifest: Path,
) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    payload = _json(native_manifest)
    artifacts = payload.get("artifacts")
    if (
        payload.get("status") != "PASS"
        or not isinstance(artifacts, dict)
        or not {"instance_mesh", "instance_color_log"} <= set(artifacts)
    ):
        raise ValueError(f"invalid native OVI manifest: {native_manifest}")
    normalized: dict[str, dict[str, object]] = {}
    for role, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"invalid native OVI artifact binding: {role}")
        path = _configured_path(str(record["path"]))
        expected_sha256 = str(record["sha256"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(
            record["byte_count"]
        ):
            raise ValueError(f"native OVI artifact is unavailable or changed: {role}")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"native OVI artifact hash changed: {role}")
        normalized[role] = {
            "path": str(path),
            "sha256": expected_sha256,
            "byte_count": int(record["byte_count"]),
        }
    return (
        Path(normalized["instance_mesh"]["path"]),
        Path(normalized["instance_color_log"]["path"]),
        normalized,
    )


def _visit_owner_attributes(
    raw_owner_ids: np.ndarray,
    records: dict[int, dict[str, Any]],
    crosswalk: TesseSemanticCrosswalk,
    *,
    owner_offset: int,
    default_last_supported_frame: int,
    output_prefix: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, str],
]:
    maximum = max(int(raw_owner_ids.max(initial=0)), max(records, default=0))
    semantic_lut = np.zeros(maximum + 1, dtype=np.int32)
    confidence_lut = np.zeros(maximum + 1, dtype=np.float32)
    last_lut = np.full(maximum + 1, default_last_supported_frame, dtype=np.int32)
    owner_names: dict[int, str] = {}
    for raw_owner_id, record in records.items():
        label = record["semantic_label"]
        lookup = crosswalk.lookup(label) if label is not None else crosswalk.unknown
        if lookup.matched:
            semantic_lut[raw_owner_id] = lookup.semantic_id
            confidence_lut[raw_owner_id] = record["semantic_score"]
        last_lut[raw_owner_id] = int(round(record["last_seen"]))
        owner_names[owner_offset + raw_owner_id] = (
            f"{output_prefix}{record['entity_id']}"
        )
    global_owner_ids = np.where(
        raw_owner_ids > 0,
        raw_owner_ids.astype(np.int64) + owner_offset,
        0,
    )
    semantic_ids = semantic_lut[raw_owner_ids]
    semantic_confidences = confidence_lut[raw_owner_ids]
    return (
        global_owner_ids,
        (raw_owner_ids > 0).astype(np.float32),
        semantic_ids,
        semantic_confidences,
        semantic_confidences.copy(),
        last_lut[raw_owner_ids],
        owner_names,
    )


def _observe_rows(
    vertices_xyz: np.ndarray,
    rows: np.ndarray,
    frames: tuple[Any, ...],
    config: FineEvidenceProjectionConfig,
    *,
    point_batch_size: int,
) -> tuple[FineSurfaceObservation, dict[str, Any]]:
    selected = np.asarray(rows)
    if selected.ndim != 1 or selected.dtype.kind not in "iu":
        raise ValueError("projection rows must be an integer vector")
    if not len(selected):
        return (
            FineSurfaceObservation(
                evidence=FineSurfaceEvidence.empty(0),
                observed_rgb_uint8=np.empty((0, 3), dtype=np.uint8),
                rgb_valid=np.empty((0,), dtype=np.bool_),
                best_rgb_depth_residual_m=np.empty((0,), dtype=np.float32),
            ),
            {"source_row_count": 0, "unique_point_count": 0, "rgb_valid_ratio": 0.0},
        )
    points = vertices_xyz[selected]
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    unique_observation = observe_fine_surface(
        unique_points,
        frames,
        config,
        point_batch_size=point_batch_size,
    )
    evidence = unique_observation.evidence
    result = FineSurfaceObservation(
        evidence=FineSurfaceEvidence(
            present_observations=evidence.present_observations[inverse],
            visible_absent_observations=evidence.visible_absent_observations[inverse],
            occluded_observations=evidence.occluded_observations[inverse],
            distinct_absent_viewpoints=evidence.distinct_absent_viewpoints[inverse],
            last_supported_frames=evidence.last_supported_frames[inverse],
            last_absent_frames=evidence.last_absent_frames[inverse],
            last_occluded_frames=evidence.last_occluded_frames[inverse],
        ),
        observed_rgb_uint8=unique_observation.observed_rgb_uint8[inverse],
        rgb_valid=unique_observation.rgb_valid[inverse],
        best_rgb_depth_residual_m=unique_observation.best_rgb_depth_residual_m[inverse],
    )
    return result, {
        "source_row_count": len(selected),
        "unique_point_count": len(unique_points),
        "frame_ids": [int(frame.frame_id) for frame in frames],
        "rgb_valid_count": int(np.count_nonzero(result.rgb_valid)),
        "rgb_valid_ratio": float(np.mean(result.rgb_valid)),
    }


def _owner_row_groups(owner_ids: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(owner_ids, kind="stable")
    values, starts, counts = np.unique(
        owner_ids[order], return_index=True, return_counts=True
    )
    return {
        int(value): order[int(start) : int(start + count)]
        for value, start, count in zip(values, starts, counts, strict=True)
    }


def _dynamic_snapshot(
    *,
    t0_xyz: np.ndarray,
    t1_xyz: np.ndarray,
    t0_groups: dict[int, np.ndarray],
    t1_groups: dict[int, np.ndarray],
    t0_records: dict[int, dict[str, Any]],
    t1_records: dict[int, dict[str, Any]],
    t0_current: np.ndarray,
    final_frame: int,
) -> MapSnapshot:
    entities: list[EntityPrediction] = []
    background_chunks: list[np.ndarray] = []
    for visit_id, (points, groups, records, current) in enumerate(
        (
            (t0_xyz, t0_groups, t0_records, t0_current),
            (t1_xyz, t1_groups, t1_records, np.ones(len(t1_xyz), dtype=np.bool_)),
        )
    ):
        background_rows = groups.get(0, np.empty((0,), dtype=np.int64))
        background_rows = background_rows[current[background_rows]]
        if len(background_rows):
            background_chunks.append(points[background_rows])
        for owner_id in sorted(records):
            owner_rows = groups.get(owner_id)
            if owner_rows is None:
                continue
            selected_rows = owner_rows[current[owner_rows]]
            if not len(selected_rows):
                continue
            record = records[owner_id]
            entities.append(
                EntityPrediction(
                    entity_id=(
                        f"t0-fallback:{record['entity_id']}"
                        if visit_id == 0
                        else str(record["entity_id"])
                    ),
                    points_xyz=points[selected_rows],
                    semantic_embedding=None,
                    semantic_label=record["semantic_label"],
                    semantic_score=float(record["semantic_score"]),
                    lifecycle_state="current",
                    first_seen=float(record["first_seen"]),
                    last_seen=float(record["last_seen"]),
                    metadata={
                        "visit_id": visit_id,
                        "source_instance_id": owner_id,
                        "geometry_authority": f"ovi_t{visit_id}",
                        "semantic_authority": f"ovi_t{visit_id}",
                    },
                )
            )
    return MapSnapshot(
        method="CROVE_FINE_LOCAL_CURRENT",
        scene_id="apartment",
        timestamp=float(final_frame),
        entities=entities,
        background_xyz=(
            np.concatenate(background_chunks, axis=0) if background_chunks else None
        ),
        scope="current",
    )


def _legacy_b3_evidence(
    states: np.ndarray, historical_last_supported_frames: np.ndarray
) -> FineSurfaceEvidence:
    present = np.zeros(len(states), dtype=np.uint16)
    absent = np.zeros(len(states), dtype=np.uint16)
    occluded = np.zeros(len(states), dtype=np.uint16)
    distinct = np.zeros(len(states), dtype=np.uint8)
    present[states == int(CoarseSurfaceState.REPLACED_OCCUPIED)] = 1
    candidate = states == int(CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
    absent[candidate] = 1
    distinct[candidate] = 1
    occluded[states == int(CoarseSurfaceState.RETAINED_OCCLUDED)] = 1
    return FineSurfaceEvidence(
        present_observations=present,
        visible_absent_observations=absent,
        occluded_observations=occluded,
        distinct_absent_viewpoints=distinct,
        last_supported_frames=historical_last_supported_frames,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _compact_index(directory: Path) -> dict[str, Any]:
    records = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name != "compact_artifact_index.json":
            records.append(
                {
                    "path": path.name,
                    "byte_count": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {"schema_version": 1, "status": "PASS", "artifacts": records}


def run_replica_static(config: dict[str, Any], output: Path) -> None:
    case = config["cases"]["replica_room0_static"]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        fine_path = _configured_path(case["fine_surface"])
        color_log = _configured_path(case["instance_color_log"])
        semantic_mapping_path = _configured_path(case["ovimap_semantic_mapping"])
        coarse_path = _configured_path(case["crove_reference_surface"])
        manifest_path = _configured_path(case["benchmark_manifest"])
        palette_path = _configured_path(case["semantic_palette"])
        source_paths = {
            "fine_surface": fine_path,
            "instance_color_log": color_log,
            "ovimap_semantic_mapping": semantic_mapping_path,
            "crove_reference_surface": coarse_path,
            "benchmark_manifest": manifest_path,
            "semantic_palette": palette_path,
        }
        for name, path in source_paths.items():
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")

        print("loading native OVI fine surface", flush=True)
        fine = load_native_ovi_surface(fine_path)
        colors_by_instance = parse_instance_color_log(color_log)
        owner_ids = bind_ovi_owner_ids(fine.palette_rgb, colors_by_instance)
        benchmark = _json(manifest_path)
        vocabulary = tuple(str(value) for value in benchmark["vocabulary"]["classes"])
        source_vocabulary = tuple(str(value) for value in case["ovimap_source_vocabulary"])
        semantic_mapping = load_ovimap_semantic_mapping(
            semantic_mapping_path,
            source_vocabulary=source_vocabulary,
            target_vocabulary=vocabulary,
        )
        owner_semantic_ids, owner_semantic_confidences = _owner_semantic_arrays(
            owner_ids, semantic_mapping
        )
        coarse_xyz, coarse_semantic_ids, coarse_supports = _load_crove_reference(
            coarse_path
        )

        print("transferring S0/S1/S2 semantics", flush=True)
        semantic_results = transfer_surface_semantics(
            fine_vertices_xyz=fine.vertices_xyz,
            reference_vertices_xyz=coarse_xyz,
            reference_semantic_ids=coarse_semantic_ids,
            reference_supports=coarse_supports,
            entity_semantic_ids=owner_semantic_ids,
            entity_confidences=owner_semantic_confidences,
            configs=_semantic_configs(config),
            neighbor_count=int(config["semantic_transfer"]["neighbor_count"]),
            point_batch_size=int(config["semantic_transfer"]["point_batch_size"]),
        )

        scene = next(
            value for value in benchmark["scenes"] if value["scene"] == case["scene"]
        )
        gt_root = (
            Path(benchmark["ground_truth_root"])
            / scene["ground_truth_scene"]
            / "habitat"
        )
        class_to_id = {name: index + 1 for index, name in enumerate(vocabulary)}
        ground_truth = load_replica_ground_truth(
            gt_root / "mesh_semantic.ply",
            gt_root / "info_semantic.json",
            class_to_id=class_to_id,
            aliases={str(key): str(value) for key, value in benchmark["aliases"].items()},
        )
        valid_semantic_ids = set(class_to_id.values())
        instance_semantic_ids = {
            semantic_id
            for name, semantic_id in class_to_id.items()
            if name not in NON_INSTANCE_CLASSES
        }
        unique_owner_ids = np.unique(owner_ids)
        entity_info = [
            EntityEvaluationInfo(
                entity_id=owner_id,
                semantic_id=semantic_mapping[owner_id][0],
                accepted_view_count=2,
                semantic_confidence=semantic_mapping[owner_id][1],
            )
            for owner_id in unique_owner_ids
            if semantic_mapping.get(owner_id, (0, 0.0))[0] > 0
        ]
        blank_colors = np.zeros((len(fine.vertices_xyz), 3), dtype=np.float32)
        ownership_confidence = (owner_ids > 0).astype(np.float32)
        metrics: dict[SurfaceSemanticStrategy, dict[str, Any]] = {}
        for strategy, semantics in semantic_results.items():
            print(f"evaluating {strategy.value}", flush=True)
            mesh = LabeledMesh(
                vertices_xyz=fine.vertices_xyz,
                triangles=np.empty((0, 3), dtype=np.int64),
                colors_rgb=blank_colors,
                semantic_ids=semantics.semantic_ids,
                entity_ids=owner_ids,
                semantic_confidence=semantics.semantic_confidences,
                ownership_confidence=ownership_confidence,
            )
            metrics[strategy] = evaluate_replica_voxel_map(
                mesh,
                ground_truth,
                entity_info,
                valid_semantic_ids=valid_semantic_ids,
                instance_semantic_ids=instance_semantic_ids,
                min_instance_vertices=int(case["min_instance_vertices"]),
                distance_threshold_m=float(
                    benchmark["protocol"]["geometry_primary_threshold_m"]
                ),
            )
        selected = select_static_semantic_strategy(metrics)
        print(f"selected {selected.value}; projecting camera RGB", flush=True)
        observed_rgb, rgb_valid, last_supported, rgb_diagnostics = (
            _static_rgb_observation(fine.vertices_xyz, case)
        )
        selected_semantics = semantic_results[selected]
        visit_surface = FineVisitSurface(
            visit_id=0,
            source_surface_index=0,
            geometry_epoch=0,
            vertices_xyz=fine.vertices_xyz,
            normals_xyz=fine.normals_xyz,
            triangles=fine.triangles,
            source_vertex_indices=fine.source_vertex_indices,
            observed_rgb_uint8=observed_rgb,
            rgb_valid=rgb_valid,
            last_supported_frames=last_supported,
            owner_entity_ids=owner_ids,
            owner_confidences=ownership_confidence,
            semantic_ids=selected_semantics.semantic_ids,
            semantic_confidences=selected_semantics.semantic_confidences,
            semantic_support_reliabilities=selected_semantics.support_reliabilities,
            semantic_source_codes=selected_semantics.semantic_source_codes,
        )
        current = current_surface_from_visit(
            visit_surface,
            surface_id=f"crove-fine-{case['scene']}-{selected.value}",
        )
        source_hashes = {
            name: {
                "path": str(path),
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        }
        export_current_surface(
            current,
            staging / "current_map",
            semantic_palette=_palette(palette_path, vocabulary),
            owner_id_table={
                0: "background",
                **{
                    int(owner_id): f"ovimap:{int(owner_id)}"
                    for owner_id in unique_owner_ids
                    if owner_id > 0
                },
            },
            source_surfaces=(
                {
                    "surface_index": 0,
                    "surface_id": "ovimap-room0-native-1cm",
                    **source_hashes["fine_surface"],
                    "surface_voxel_size_m": 0.01,
                    "geometry_role": "prediction_native_surface",
                },
            ),
        )

        headline = {strategy.value: _headline(value) for strategy, value in metrics.items()}
        elapsed = time.perf_counter() - started
        _write_json(
            staging / "input_selection.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "selection_status": "USER_PLY_NOT_IDENTIFIED/EVIDENCE_BOUND_REPRESENTATIVE",
                "case": "replica_room0_static",
                "scene": case["scene"],
                "split_role": scene["split_role"],
                "ground_truth_role": "evaluator_only",
                "sources": source_hashes,
            },
        )
        _write_json(
            staging / "metrics_summary.json",
            {
                "schema_version": 1,
                "status": "DEV",
                "protocol_id": config["protocol_id"],
                "scene": case["scene"],
                "selected_strategy": selected.value,
                "selection_rule": "max_miou_then_f_miou_then_simpler_strategy",
                "trials": headline,
                "full_metrics": {key.value: value for key, value in metrics.items()},
            },
        )
        _write_csv(
            staging / "metrics_per_scene.csv",
            ["case", "scene", "strategy", "selected", "miou", "macc", "f_miou", "ap25", "ap50", "f5"],
            [
                {
                    "case": "replica_room0_static",
                    "scene": case["scene"],
                    "strategy": strategy.value,
                    "selected": strategy is selected,
                    **_headline(value),
                }
                for strategy, value in metrics.items()
            ],
        )
        _write_csv(
            staging / "trial_log.csv",
            ["trial_id", "hypothesis", "surface_id", "strategy", "selected", "miou", "macc", "f_miou", "ap25", "ap50", "f5", "status"],
            [
                {
                    "trial_id": f"D2-{strategy.name}",
                    "hypothesis": {
                        SurfaceSemanticStrategy.S0: "bounded nearest CROVE surface semantics",
                        SurfaceSemanticStrategy.S1: "bounded surface vote reduces lookup noise",
                        SurfaceSemanticStrategy.S2: "absolute support reliability enables OVI owner fallback",
                    }[strategy],
                    "surface_id": f"ovimap-room0-native-1cm-{strategy.value}",
                    "strategy": strategy.value,
                    "selected": strategy is selected,
                    **_headline(value),
                    "status": "PASS",
                }
                for strategy, value in metrics.items()
            ],
        )
        _write_json(
            staging / "effective_geometry_config.json",
            {
                "surface_backend": "OVI-MAP native",
                "surface_voxel_size_m": 0.01,
                "ownership_surface_resolution_m": 0.01,
                "temporal_state_voxel_size_m": 0.05,
                "evaluation_distance_m": float(
                    benchmark["protocol"]["geometry_primary_threshold_m"]
                ),
                "surface_vertex_count": len(fine.vertices_xyz),
                "surface_face_count": len(fine.triangles),
                "crove_reference_vertex_count": len(coarse_xyz),
                "gt_geometry_used_for_prediction": False,
            },
        )
        _write_json(
            staging / "selected_config.json",
            {
                "status": "DEV_SELECTED",
                "strategy": selected.value,
                "selection_rule": "max_miou_then_f_miou_then_simpler_strategy",
                "semantic_transfer": config["semantic_transfer"],
                "rgb_projection": case["rgb_projection"],
            },
        )
        s0 = semantic_results[SurfaceSemanticStrategy.S0]
        _write_json(
            staging / "export_diagnostics.json",
            {
                "canonical_vertex_count": len(current.vertices_xyz),
                "current_vertex_count": int(np.sum(current.current_valid)),
                "semantic_changed_from_s0_count": int(
                    np.sum(selected_semantics.semantic_ids != s0.semantic_ids)
                ),
                "semantic_changed_from_s0_ratio": float(
                    np.mean(selected_semantics.semantic_ids != s0.semantic_ids)
                ),
                "unknown_semantic_ratio": float(
                    np.mean(selected_semantics.semantic_ids == 0)
                ),
                "background_owner_ratio": float(np.mean(owner_ids == 0)),
                "owner_known_semantic_ratio": float(np.mean(owner_semantic_ids > 0)),
                "rgb": rgb_diagnostics,
                "elapsed_seconds": elapsed,
            },
        )
        selected_metrics = headline[selected.value]
        table_values = {
            f"CROVE_FINE_ROOM0_DEV_{name.upper()}": {
                "value": value,
                "status": "RECOMPUTED",
                "protocol_id": config["protocol_id"],
                "surface_id": current.surface_id,
            }
            for name, value in selected_metrics.items()
        }
        _write_json(staging / "table_values.json", table_values)
        _write_csv(
            staging / "table_lineage.csv",
            ["token", "table", "scene", "metric", "value", "status", "protocol_id", "surface_id", "source_json", "json_pointer"],
            [
                {
                    "token": token,
                    "table": "T1_DEV",
                    "scene": case["scene"],
                    "metric": _static_table_metric(token),
                    "value": record["value"],
                    "status": record["status"],
                    "protocol_id": record["protocol_id"],
                    "surface_id": record["surface_id"],
                    "source_json": "metrics_summary.json",
                    "json_pointer": f"/trials/{selected.value}/{_static_table_metric(token)}",
                }
                for token, record in table_values.items()
            ],
        )
        _write_json(
            staging / "run_receipt.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "case": "replica_room0_static",
                "protocol_id": config["protocol_id"],
                "selected_strategy": selected.value,
                "elapsed_seconds": elapsed,
                "model_upload_status": "NOT_APPLICABLE_NO_NEW_TRAINING",
                "large_map_policy": "LOCAL_ONLY_POLICY",
            },
        )
        _write_json(staging / "compact_artifact_index.json", _compact_index(staging))
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_apartment_two_visit(config: dict[str, Any], output: Path) -> None:
    case = config["cases"]["apartment_two_visit_dev"]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        configured_paths = {
            name: _configured_path(case[name])
            for name in (
                "t0_native_manifest",
                "t1_native_manifest",
                "b0_entities",
                "b2_entities",
                "b3_provenance",
                "b3_metrics",
                "two_visit_protocol",
                "causal_schedule",
                "common_v2_target_manifest",
                "semantic_aliases",
                "semantic_label_space",
                "rgbd_export_manifest",
            )
        }
        for name, path in configured_paths.items():
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")
        rgbd_root = _configured_path(case["rgbd_root"])
        if rgbd_root.is_symlink() or not rgbd_root.is_dir():
            raise FileNotFoundError(f"rgbd_root: {rgbd_root}")

        t0_records = _entity_records(configured_paths["b0_entities"])
        t1_records = _entity_records(configured_paths["b2_entities"])
        t0_mesh_path, t0_log_path, t0_artifacts = _native_artifact_paths(
            configured_paths["t0_native_manifest"]
        )
        t1_mesh_path, t1_log_path, t1_artifacts = _native_artifact_paths(
            configured_paths["t1_native_manifest"]
        )
        cache_root = _configured_path(case["native_surface_cache_root"])
        print("loading Apartment t0 native OVI fine surface", flush=True)
        t0_fine = load_cached_native_ovi_surface(
            t0_mesh_path,
            cache_root / "apartment_t0_native_surface.npz",
            source_sha256=str(t0_artifacts["instance_mesh"]["sha256"]),
        )
        t0_colors = parse_instance_color_log(t0_log_path)
        t0_owner_ids = bind_ovi_owner_ids(
            t0_fine.palette_rgb,
            {owner_id: t0_colors[owner_id] for owner_id in t0_records},
        )
        print("mapping frozen B3 provenance to native t0 rows", flush=True)
        policy = load_b3_surface_policy(
            configured_paths["b3_provenance"], t0_owner_ids
        )
        print("loading Apartment t1 native OVI fine surface", flush=True)
        t1_fine = load_cached_native_ovi_surface(
            t1_mesh_path,
            cache_root / "apartment_t1_native_surface.npz",
            source_sha256=str(t1_artifacts["instance_mesh"]["sha256"]),
        )
        t1_colors = parse_instance_color_log(t1_log_path)
        t1_owner_ids = bind_ovi_owner_ids(
            t1_fine.palette_rgb,
            {owner_id: t1_colors[owner_id] for owner_id in t1_records},
        )

        crosswalk = load_tesse_semantic_crosswalk(
            configured_paths["semantic_aliases"],
            case["scene"],
            configured_paths["semantic_label_space"],
        )
        visit_ranges = case["visit_frame_ranges"]
        t0_start, t0_end = (int(value) for value in visit_ranges["t0"])
        t1_start, t1_end = (int(value) for value in visit_ranges["t1"])
        print("validating TESSE-CD RGB-D and loading t1 evaluation frames", flush=True)
        dataset = TesseCdRgbdDataset(
            rgbd_root,
            case["scene"],
            configured_paths["rgbd_export_manifest"],
            configured_paths["causal_schedule"],
        )
        t1_frames = tuple(dataset[index] for index in range(t1_start, t1_end + 1))
        projection_stride = int(case["fine_projection"]["frame_stride"])
        evidence_frames = t1_frames[::projection_stride]
        projection_config = FineEvidenceProjectionConfig(
            **{
                key: value
                for key, value in case["fine_projection"].items()
                if key not in {"frame_stride", "point_batch_size"}
            }
        )
        candidate_rows = np.flatnonzero(
            policy.states == int(CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
        )
        print(
            f"projecting fine evidence for {len(candidate_rows)} B3 free candidates",
            flush=True,
        )
        candidate_observation, candidate_diagnostics = _observe_rows(
            t0_fine.vertices_xyz,
            candidate_rows,
            evidence_frames,
            projection_config,
            point_batch_size=int(case["fine_projection"]["point_batch_size"]),
        )

        (
            t0_global_owners,
            t0_owner_confidences,
            t0_semantic_ids,
            t0_semantic_confidences,
            t0_semantic_reliabilities,
            t0_last_supported,
            t0_owner_names,
        ) = _visit_owner_attributes(
            t0_owner_ids,
            t0_records,
            crosswalk,
            owner_offset=0,
            default_last_supported_frame=t0_end,
            output_prefix="t0-fallback:",
        )
        (
            t1_global_owners,
            t1_owner_confidences,
            t1_semantic_ids,
            t1_semantic_confidences,
            t1_semantic_reliabilities,
            t1_last_supported,
            t1_owner_names,
        ) = _visit_owner_attributes(
            t1_owner_ids,
            t1_records,
            crosswalk,
            owner_offset=int(case["t1_owner_id_offset"]),
            default_last_supported_frame=t1_end,
            output_prefix="",
        )
        t0_evidence = assemble_fine_surface_evidence(
            states=policy.states,
            candidate_rows=candidate_rows,
            candidate_observation=candidate_observation,
            historical_last_supported_frames=t0_last_supported,
        )
        coarse_candidates = (
            policy.states == int(CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
        )
        legacy_current = np.isin(
            policy.states,
            [
                int(CoarseSurfaceState.RETAINED_OCCLUDED),
                int(CoarseSurfaceState.RETAINED_UNOBSERVED),
            ],
        )

        visibility_config = SignedVisibilityConfig(**case["evaluation_visibility"])
        context, evaluation_bindings = _evaluation_context(
            protocol=_json(configured_paths["two_visit_protocol"]),
            scene=case["scene"],
            final_frame=t1_end,
            schedule_path=configured_paths["causal_schedule"],
            target_manifest_path=configured_paths["common_v2_target_manifest"],
            frames=t1_frames,
            visibility_config=visibility_config,
        )
        print("deriving frozen full-frame t1 observation support", flush=True)
        t0_observed = derive_observed_point_mask(
            t0_fine.vertices_xyz, t1_frames, visibility_config
        )
        t0_groups = _owner_row_groups(t0_owner_ids)
        t1_groups = _owner_row_groups(t1_owner_ids)
        masks: dict[str, np.ndarray] = {"B3_LEGACY": legacy_current}
        trial_configs: dict[str, FineValidityConfig] = {}
        for raw in case["validity_trials"]:
            trial_id = str(raw["trial_id"])
            if trial_id in masks:
                raise ValueError(f"duplicate dynamic trial ID: {trial_id}")
            validity = FineValidityConfig(
                minimum_absent_observations=int(raw["minimum_absent_observations"]),
                minimum_distinct_viewpoints=int(raw["minimum_distinct_viewpoints"]),
            )
            trial_configs[trial_id] = validity
            masks[trial_id] = resolve_fine_current_validity(
                source_visit_ids=np.zeros(len(t0_fine.vertices_xyz), dtype=np.int16),
                latest_visit_id=1,
                coarse_visible_free_candidates=coarse_candidates,
                evidence=t0_evidence,
                config=validity,
            ).current_valid

        metrics: dict[str, dict[str, float]] = {}
        for trial_id, current_mask in masks.items():
            print(
                f"evaluating {trial_id} with {int(np.count_nonzero(current_mask))} retained t0 rows",
                flush=True,
            )
            snapshot = _dynamic_snapshot(
                t0_xyz=t0_fine.vertices_xyz,
                t1_xyz=t1_fine.vertices_xyz,
                t0_groups=t0_groups,
                t1_groups=t1_groups,
                t0_records=t0_records,
                t1_records=t1_records,
                t0_current=current_mask,
                final_frame=t1_end,
            )
            measured = evaluate_two_visit_snapshot(
                snapshot,
                context,
                crosswalk,
                retained_t0_xyz=t0_fine.vertices_xyz[current_mask],
                retained_t0_t1_observed_mask=t0_observed[current_mask],
            )
            metrics[trial_id] = {
                key: float(value) for key, value in measured.items()
            }
            del snapshot
            gc.collect()

        historical_b3 = _json(configured_paths["b3_metrics"])["metric_groups"]
        expected_b3 = {
            **historical_b3["current_state"],
            **historical_b3["geometry"],
        }
        tolerance = float(case["baseline_reproduction_tolerance"])
        for name, value in metrics["B3_LEGACY"].items():
            if abs(value - float(expected_b3[name])) > tolerance:
                raise ValueError(
                    f"fine surface failed to reproduce B3 {name}: {value} vs {expected_b3[name]}"
                )
        selected, eligibility = select_dynamic_trial(
            metrics,
            baseline_trial_id="B3_LEGACY",
            gates={key: float(value) for key, value in case["selection_gates"].items()},
        )
        print(f"selected {selected}; projecting source-visit camera RGB", flush=True)

        t0_rgb_frames = tuple(
            dataset[index]
            for index in range(
                t0_start,
                t0_end + 1,
                int(case["rgb_projection"]["frame_stride"]),
            )
        )
        t1_rgb_frames = t1_frames[:: int(case["rgb_projection"]["frame_stride"])]
        rgb_config = FineEvidenceProjectionConfig(
            **{
                key: value
                for key, value in case["rgb_projection"].items()
                if key not in {"frame_stride", "point_batch_size"}
            }
        )
        t0_rgb, t0_rgb_diagnostics = _observe_rows(
            t0_fine.vertices_xyz,
            np.arange(len(t0_fine.vertices_xyz), dtype=np.int64),
            t0_rgb_frames,
            rgb_config,
            point_batch_size=int(case["rgb_projection"]["point_batch_size"]),
        )
        t1_rgb, t1_rgb_diagnostics = _observe_rows(
            t1_fine.vertices_xyz,
            np.arange(len(t1_fine.vertices_xyz), dtype=np.int64),
            t1_rgb_frames,
            rgb_config,
            point_batch_size=int(case["rgb_projection"]["point_batch_size"]),
        )
        t0_visit = FineVisitSurface(
            visit_id=0,
            source_surface_index=0,
            geometry_epoch=0,
            vertices_xyz=t0_fine.vertices_xyz,
            normals_xyz=t0_fine.normals_xyz,
            triangles=t0_fine.triangles,
            source_vertex_indices=t0_fine.source_vertex_indices,
            observed_rgb_uint8=t0_rgb.observed_rgb_uint8,
            rgb_valid=t0_rgb.rgb_valid,
            last_supported_frames=t0_last_supported,
            owner_entity_ids=t0_global_owners,
            owner_confidences=t0_owner_confidences,
            semantic_ids=t0_semantic_ids,
            semantic_confidences=t0_semantic_confidences,
            semantic_support_reliabilities=t0_semantic_reliabilities,
            semantic_source_codes=np.where(
                t0_semantic_ids > 0,
                int(SemanticSource.OWNER_ENTITY),
                int(SemanticSource.UNKNOWN),
            ),
        )
        t1_visit = FineVisitSurface(
            visit_id=1,
            source_surface_index=1,
            geometry_epoch=1,
            vertices_xyz=t1_fine.vertices_xyz,
            normals_xyz=t1_fine.normals_xyz,
            triangles=t1_fine.triangles,
            source_vertex_indices=t1_fine.source_vertex_indices,
            observed_rgb_uint8=t1_rgb.observed_rgb_uint8,
            rgb_valid=t1_rgb.rgb_valid,
            last_supported_frames=t1_last_supported,
            owner_entity_ids=t1_global_owners,
            owner_confidences=t1_owner_confidences,
            semantic_ids=t1_semantic_ids,
            semantic_confidences=t1_semantic_confidences,
            semantic_support_reliabilities=t1_semantic_reliabilities,
            semantic_source_codes=np.where(
                t1_semantic_ids > 0,
                int(SemanticSource.OWNER_ENTITY),
                int(SemanticSource.UNKNOWN),
            ),
        )
        if selected == "B3_LEGACY":
            selected_evidence = _legacy_b3_evidence(
                policy.states, t0_last_supported
            )
            selected_validity = FineValidityConfig(
                minimum_absent_observations=1,
                minimum_distinct_viewpoints=1,
            )
        else:
            selected_evidence = t0_evidence
            selected_validity = trial_configs[selected]
        current = compose_two_visit_fine_surface(
            t0=t0_visit,
            t1=t1_visit,
            t0_evidence=selected_evidence,
            coarse_visible_free_candidates=coarse_candidates,
            validity_config=selected_validity,
            surface_id=f"crove-fine-apartment-{selected.lower()}",
        )
        if not np.array_equal(
            current.current_valid[: len(t0_fine.vertices_xyz)], masks[selected]
        ):
            raise ValueError("selected evaluation mask and exported current map differ")

        print("exporting four source-bound current views", flush=True)
        prediction_source_paths = {
            name: configured_paths[name]
            for name in (
                "t0_native_manifest",
                "t1_native_manifest",
                "b0_entities",
                "b2_entities",
                "b3_provenance",
                "semantic_aliases",
                "semantic_label_space",
                "rgbd_export_manifest",
            )
        }
        prediction_source_paths.update(
            {
            "t0_instance_mesh": t0_mesh_path,
            "t1_instance_mesh": t1_mesh_path,
            "t0_instance_color_log": t0_log_path,
            "t1_instance_color_log": t1_log_path,
            }
        )
        source_hashes = {
            name: {
                "path": str(path),
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in prediction_source_paths.items()
        }
        development_selection_inputs = {
            "b3_metrics": {
                "path": str(configured_paths["b3_metrics"]),
                "byte_count": configured_paths["b3_metrics"].stat().st_size,
                "sha256": _sha256(configured_paths["b3_metrics"]),
            }
        }
        semantic_palette = {
            int(key): tuple(int(channel) for channel in value)
            for key, value in case["semantic_palette"].items()
        }
        export_current_surface(
            current,
            staging / "current_map",
            semantic_palette=semantic_palette,
            owner_id_table={0: "background", **t0_owner_names, **t1_owner_names},
            source_surfaces=(
                {
                    "surface_index": 0,
                    "surface_id": "ovimap-apartment-t0-native-1cm",
                    **source_hashes["t0_instance_mesh"],
                    "surface_voxel_size_m": 0.01,
                    "geometry_role": "prediction_native_surface",
                },
                {
                    "surface_index": 1,
                    "surface_id": "ovimap-apartment-t1-native-1cm",
                    **source_hashes["t1_instance_mesh"],
                    "surface_voxel_size_m": 0.01,
                    "geometry_role": "prediction_native_surface",
                },
            ),
        )

        elapsed = time.perf_counter() - started
        _write_json(
            staging / "input_selection.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "case": "apartment_two_visit_dev",
                "scene": case["scene"],
                "split_role": "DEV_EXPOSED",
                "prediction_inputs": source_hashes,
                "development_selection_inputs": development_selection_inputs,
                "evaluation_only_inputs": {
                    **evaluation_bindings,
                    "two_visit_protocol": {
                        "path": str(configured_paths["two_visit_protocol"]),
                        "byte_count": configured_paths[
                            "two_visit_protocol"
                        ].stat().st_size,
                        "sha256": _sha256(configured_paths["two_visit_protocol"]),
                    },
                    "causal_schedule": {
                        "path": str(configured_paths["causal_schedule"]),
                        "byte_count": configured_paths[
                            "causal_schedule"
                        ].stat().st_size,
                        "sha256": _sha256(configured_paths["causal_schedule"]),
                    },
                    "common_v2_target_manifest": {
                        "path": str(configured_paths["common_v2_target_manifest"]),
                        "byte_count": configured_paths[
                            "common_v2_target_manifest"
                        ].stat().st_size,
                        "sha256": _sha256(
                            configured_paths["common_v2_target_manifest"]
                        ),
                    },
                },
                "ground_truth_used_for_prediction": False,
            },
        )
        _write_json(
            staging / "metrics_summary.json",
            {
                "schema_version": 1,
                "status": "DEV",
                "protocol_id": case["protocol_id"],
                "scene": case["scene"],
                "baseline_trial_id": "B3_LEGACY",
                "selected_trial_id": selected,
                "selection_gates": case["selection_gates"],
                "eligibility": eligibility,
                "historical_b3": expected_b3,
                "trials": metrics,
            },
        )
        metric_names = list(metrics["B3_LEGACY"])
        _write_csv(
            staging / "metrics_per_scene.csv",
            ["case", "scene", "trial_id", "selected", "eligible", *metric_names],
            [
                {
                    "case": "apartment_two_visit_dev",
                    "scene": case["scene"],
                    "trial_id": trial_id,
                    "selected": trial_id == selected,
                    "eligible": eligibility[trial_id],
                    **values,
                }
                for trial_id, values in metrics.items()
            ],
        )
        _write_csv(
            staging / "trial_log.csv",
            [
                "trial_id",
                "hypothesis",
                "minimum_absent_observations",
                "minimum_distinct_viewpoints",
                "retained_t0_rows",
                "selected",
                "eligible",
                *metric_names,
            ],
            [
                {
                    "trial_id": trial_id,
                    "hypothesis": (
                        "frozen B3 coarse current policy"
                        if trial_id == "B3_LEGACY"
                        else "fine multiview evidence prevents coarse false revocation"
                    ),
                    "minimum_absent_observations": (
                        1
                        if trial_id == "B3_LEGACY"
                        else trial_configs[trial_id].minimum_absent_observations
                    ),
                    "minimum_distinct_viewpoints": (
                        1
                        if trial_id == "B3_LEGACY"
                        else trial_configs[trial_id].minimum_distinct_viewpoints
                    ),
                    "retained_t0_rows": int(np.count_nonzero(masks[trial_id])),
                    "selected": trial_id == selected,
                    "eligible": eligibility[trial_id],
                    **values,
                }
                for trial_id, values in metrics.items()
            ],
        )
        _write_json(
            staging / "effective_geometry_config.json",
            {
                "surface_backend": "OVI-MAP native",
                "surface_voxel_size_m": 0.01,
                "ownership_surface_resolution_m": 0.01,
                "temporal_state_voxel_size_m": 0.05,
                "evaluation_grid_size_m": 0.05,
                "t0_vertex_count": len(t0_fine.vertices_xyz),
                "t0_face_count": len(t0_fine.triangles),
                "t1_vertex_count": len(t1_fine.vertices_xyz),
                "t1_face_count": len(t1_fine.triangles),
                "gt_geometry_used_for_prediction": False,
            },
        )
        _write_json(
            staging / "selected_config.json",
            {
                "status": "DEV_SELECTED",
                "trial_id": selected,
                "validity": (
                    {"legacy_b3": True}
                    if selected == "B3_LEGACY"
                    else {
                        "minimum_absent_observations": trial_configs[
                            selected
                        ].minimum_absent_observations,
                        "minimum_distinct_viewpoints": trial_configs[
                            selected
                        ].minimum_distinct_viewpoints,
                    }
                ),
                "selection_gates": case["selection_gates"],
                "fine_projection": case["fine_projection"],
                "rgb_projection": case["rgb_projection"],
            },
        )
        state_counts = {
            state.name.lower(): int(
                np.count_nonzero(current.evidence_state_codes == int(state))
            )
            for state in CurrentEvidenceState
        }
        _write_json(
            staging / "export_diagnostics.json",
            {
                "canonical_vertex_count": len(current.vertices_xyz),
                "current_vertex_count": int(np.count_nonzero(current.current_valid)),
                "canonical_face_count": len(current.triangles),
                "policy_counts": dict(policy.counts),
                "evidence_state_counts": state_counts,
                "fine_candidate_projection": candidate_diagnostics,
                "t0_rgb_projection": t0_rgb_diagnostics,
                "t1_rgb_projection": t1_rgb_diagnostics,
                "elapsed_seconds": elapsed,
            },
        )
        table_values = {
            f"CROVE_FINE_APARTMENT_TWO_VISIT_DEV_{name.upper()}": {
                "value": value,
                "status": "RECOMPUTED",
                "protocol_id": case["protocol_id"],
                "surface_id": current.surface_id,
            }
            for name, value in metrics[selected].items()
        }
        for name in ("object_f1", "dynamic_surface_f1", "change_f1"):
            table_values[f"CROVE_FINE_APARTMENT_TWO_VISIT_DEV_{name.upper()}"] = {
                "value": None,
                "status": "N/A",
                "protocol_id": case["protocol_id"],
                "surface_id": current.surface_id,
                "reason": "two_visit_final_snapshot_has_no_continuous_sequence_metric",
            }
        _write_json(staging / "table_values.json", table_values)
        _write_csv(
            staging / "table_lineage.csv",
            [
                "token",
                "table",
                "scene",
                "metric",
                "value",
                "status",
                "protocol_id",
                "surface_id",
                "source_json",
                "json_pointer",
            ],
            [
                {
                    "token": token,
                    "table": "T2_TWO_VISIT_DEV",
                    "scene": case["scene"],
                    "metric": token.removeprefix(
                        "CROVE_FINE_APARTMENT_TWO_VISIT_DEV_"
                    ).lower(),
                    "value": record["value"],
                    "status": record["status"],
                    "protocol_id": record["protocol_id"],
                    "surface_id": record["surface_id"],
                    "source_json": (
                        "table_values.json"
                        if record["status"] == "N/A"
                        else "metrics_summary.json"
                    ),
                    "json_pointer": (
                        f"/{token}/value"
                        if record["status"] == "N/A"
                        else f"/trials/{selected}/{token.removeprefix('CROVE_FINE_APARTMENT_TWO_VISIT_DEV_').lower()}"
                    ),
                }
                for token, record in table_values.items()
            ],
        )
        _write_json(
            staging / "run_receipt.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "case": "apartment_two_visit_dev",
                "protocol_id": case["protocol_id"],
                "selected_trial_id": selected,
                "elapsed_seconds": elapsed,
                "model_upload_status": "NOT_APPLICABLE_NO_NEW_TRAINING",
                "large_map_policy": "LOCAL_ONLY_POLICY",
            },
        )
        _write_json(staging / "compact_artifact_index.json", _compact_index(staging))
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("replica_room0_static", "apartment_two_visit_dev"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _json(args.config)
    if args.case == "replica_room0_static":
        run_replica_static(config, args.output)
    elif args.case == "apartment_two_visit_dev":
        run_apartment_two_visit(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
