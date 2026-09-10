#!/usr/bin/env python3
# ruff: noqa: I001, BLE001
"""Prepare, execute, and summarize CROVE entity-episode experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_ovi_rescene_apartment_v3 import (
    load_query_evidence_artifact,
)
from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import (
    _evaluation_context,
)
from scripts.evaluation.run_crove_fine_current_map import (
    _dynamic_snapshot,
    _entity_records,
    _observe_rows,
    _owner_row_groups,
)
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import (
    TesseSemanticCrosswalk,
    load_tesse_semantic_crosswalk,
)
from src.evaluation.entity_epoch_metrics import (
    EntityEpochEvaluationSupport,
    action_attribution_rows,
    evaluate_entity_epoch_actions,
    relation_diagnostic_rows,
    verify_d4_identity_only_invariance,
)
from src.evaluation.two_visit_snapshot_metrics import (
    TwoVisitEvaluationContext,
    evaluate_two_visit_snapshot,
)
from src.oviv2.current_surface import (
    CurrentEvidenceState,
    CurrentSurfaceView,
)
from src.oviv2.entity_epoch_io import write_entity_epoch_composition
from src.oviv2.entity_epoch_memory import (
    OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT,
    CroveMemoryIndex,
    MemoryCandidateObservation,
    MemoryEntityObservation,
    MemoryRetrievalMode,
    build_memory_relation_support,
)
from src.oviv2.entity_epoch_relations import (
    StrongGeometricRelationConfig,
    build_g1_relation_support,
)
from src.oviv2.entity_epoch_update import (
    EntityEpochUpdateConfig,
    EntityEpochUpdateResult,
    EpisodeLifecycle,
    RelationSupport,
    resolve_entity_epoch_update,
)
from src.oviv2.fine_current_composer import (
    EntityEpochComposition,
    FineVisitSurface,
    compose_entity_epoch_fine_surface,
)
from src.oviv2.fine_dynamic_policy import (
    CoarseSurfaceState,
    PriorSurfaceState,
    SurfaceRetirementReason,
    assemble_fine_surface_evidence,
    load_b3_prior_surface_state,
    load_b3_surface_policy,
)
from src.oviv2.fine_surface_validity import (
    FineEvidenceProjectionConfig,
    FineSurfaceEvidence,
)
from src.oviv2.ovi_rescene_adapter import load_neural_sample_artifact
from src.oviv2.query_instance_projection import (
    project_queries_to_relation_support,
)
from src.oviv2.two_visit_contracts import (
    NeuralSampleMap,
    TemporalQueryEvidence,
)
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
)
from src.oviv2.two_visit_registration import RegistrationConfig


ORDERED_VARIANTS = (
    "D0_T1",
    "D1_B3",
    "D2_INHERIT",
    "D3_GEOM",
    "D4_RESCENE_ID_ONLY",
    "D5_RESCENE_EPOCH",
    "D6_MEMORY_EPOCH",
    "DX_ORACLE_REL",
)
_RANKING_VARIANTS = frozenset(ORDERED_VARIANTS[:-1])
_READY_STATES = frozenset({"READY"})
_CONFIRM_STATES = frozenset({"RAW_MISSING", "DERIVED_NOT_BUILT", "READY"})


def _canonical_json(value: object) -> bytes:
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("configured paths must be non-empty strings")
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _portableize_paths(value: object) -> object:
    """Replace machine-specific absolute paths in public result payloads."""

    if isinstance(value, Mapping):
        return {str(key): _portableize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portableize_paths(item) for item in value]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    path = Path(value)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return f"$HOME/{path.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return f"local:{path.name}"


def _file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def load_backbone_visits_from_surface(
    surface: CurrentSurfaceView,
) -> tuple[FineVisitSurface, FineVisitSurface, np.ndarray]:
    """Split one canonical V1 export back into its immutable native visits."""

    if not isinstance(surface, CurrentSurfaceView):
        raise TypeError("surface must be CurrentSurfaceView")
    visits = surface.source_visit_ids
    t0_count = int(np.count_nonzero(visits == 0))
    if (
        t0_count == 0
        or t0_count == len(visits)
        or not np.array_equal(visits[:t0_count], np.zeros(t0_count, dtype=np.int16))
        or not np.array_equal(
            visits[t0_count:], np.ones(len(visits) - t0_count, dtype=np.int16)
        )
    ):
        raise ValueError("backbone rows must contain contiguous visits zero then one")
    t0_triangles = np.all(surface.triangles < t0_count, axis=1)
    t1_triangles = np.all(surface.triangles >= t0_count, axis=1)
    if np.any(~(t0_triangles | t1_triangles)):
        raise ValueError("backbone contains a cross-visit triangle")
    if not np.array_equal(
        surface.source_surface_indices,
        np.concatenate(
            (
                np.zeros(t0_count, dtype=np.uint16),
                np.ones(len(visits) - t0_count, dtype=np.uint16),
            )
        ),
    ):
        raise ValueError("backbone source surface indices differ from visit order")

    array_fields = (
        "vertices_xyz",
        "normals_xyz",
        "source_vertex_indices",
        "observed_rgb_uint8",
        "rgb_valid",
        "last_supported_frames",
        "owner_entity_ids",
        "owner_confidences",
        "semantic_ids",
        "semantic_confidences",
        "semantic_support_reliabilities",
        "semantic_source_codes",
    )

    def visit(
        start: int, stop: int, visit_id: int, triangles: np.ndarray
    ) -> FineVisitSurface:
        return FineVisitSurface(
            visit_id=visit_id,
            source_surface_index=visit_id,
            geometry_epoch=visit_id,
            triangles=triangles,
            **{name: getattr(surface, name)[start:stop] for name in array_fields},
        )

    return (
        visit(0, t0_count, 0, surface.triangles[t0_triangles]),
        visit(
            t0_count,
            len(visits),
            1,
            surface.triangles[t1_triangles] - t0_count,
        ),
        np.array(surface.current_valid[:t0_count], dtype=np.bool_, copy=True),
    )


def load_backbone_visits(
    current_map_root: str | Path,
) -> tuple[FineVisitSurface, FineVisitSurface, np.ndarray]:
    """Verify and load an exported current surface used only as OVI backbone data."""

    root = Path(current_map_root)
    manifest_path = root / "current_surface_manifest.json"
    arrays_path = root / "current_surface.npz"
    manifest = _load_json(manifest_path)
    record = manifest.get("artifacts", {}).get("current_surface.npz")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or not isinstance(record, Mapping)
        or set(record) != {"sha256", "byte_count"}
        or record.get("byte_count") != arrays_path.stat().st_size
        or record.get("sha256") != _sha256(arrays_path)
    ):
        raise ValueError("backbone current surface binding mismatch")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            expected = {field.name for field in fields(CurrentSurfaceView)}
            if set(archive.files) != expected:
                raise ValueError("backbone current surface schema mismatch")
            values = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if str(error).startswith("backbone current surface"):
            raise
        raise ValueError("backbone current surface cannot be decoded") from error
    surface_id = values.pop("surface_id")
    if np.asarray(surface_id).shape != ():
        raise ValueError("backbone current surface ID is invalid")
    surface = CurrentSurfaceView(surface_id=str(surface_id.item()), **values)
    if (
        manifest.get("surface_id") != surface.surface_id
        or manifest.get("canonical_vertex_count") != len(surface.vertices_xyz)
        or manifest.get("canonical_face_count") != len(surface.triangles)
    ):
        raise ValueError("backbone current surface manifest differs from its arrays")
    return load_backbone_visits_from_surface(surface)


def adapt_relations_to_backbone(
    relations: tuple[RelationSupport, ...],
    *,
    t0: FineVisitSurface,
    t1: FineVisitSurface,
    t0_surface_id: str,
    t1_surface_id: str,
    t1_owner_offset: int,
) -> tuple[RelationSupport, ...]:
    """Rebind coarse relation identities to exact OVI-native fine row keys."""

    if not isinstance(relations, tuple) or any(
        not isinstance(relation, RelationSupport) for relation in relations
    ):
        raise TypeError("relations must contain RelationSupport values")
    if type(t1_owner_offset) is not int or t1_owner_offset < 0:
        raise ValueError("t1_owner_offset must be a nonnegative integer")
    output = []
    for relation in relations:
        t0_rows = np.flatnonzero(t0.owner_entity_ids == relation.t0_owner_entity_id)
        t1_owner_id = relation.t1_owner_entity_id + t1_owner_offset
        t1_rows = np.flatnonzero(t1.owner_entity_ids == t1_owner_id)
        if not len(t0_rows) or not len(t1_rows):
            raise ValueError(
                f"relation {relation.relation_id} lacks a fine OVI owner support"
            )
        output.append(
            replace(
                relation,
                t1_owner_entity_id=t1_owner_id,
                t0_source_surface_id=t0_surface_id,
                t1_source_surface_id=t1_surface_id,
                t0_source_vertex_indices=t0.source_vertex_indices[t0_rows],
                t1_source_vertex_indices=t1.source_vertex_indices[t1_rows],
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PreparedEntityEpochPair:
    pair_id: str
    scene: str
    t0: FineVisitSurface
    t1: FineVisitSurface
    t0_records: Mapping[int, Mapping[str, object]]
    t1_records: Mapping[int, Mapping[str, object]]
    prior: PriorSurfaceState
    evidence: FineSurfaceEvidence
    t1_replacement_mask: np.ndarray
    t0_t1_observed_mask: np.ndarray
    evaluation: TwoVisitEvaluationContext
    evaluation_support: EntityEpochEvaluationSupport
    crosswalk: TesseSemanticCrosswalk
    t0_surface_id: str
    t1_surface_id: str
    geometric_relations: tuple[RelationSupport, ...]
    rescene_relations: tuple[RelationSupport, ...]
    memory_relations: Mapping[MemoryRetrievalMode, tuple[RelationSupport, ...]]
    pair_sha256: str
    checkpoint_sha256: str
    query_evidence: TemporalQueryEvidence


def _load_cached_evidence(
    path: Path, expected_identity: str
) -> FineSurfaceEvidence | None:
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("cache_identity") != expected_identity
        or manifest.get("sha256") != _sha256(path)
        or manifest.get("byte_count") != path.stat().st_size
    ):
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return FineSurfaceEvidence(
                **{name: archive[name] for name in archive.files}
            )
    except (OSError, TypeError, ValueError):
        return None


def _write_cached_evidence(
    path: Path, evidence: FineSurfaceEvidence, cache_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                field.name: getattr(evidence, field.name)
                for field in fields(FineSurfaceEvidence)
            },
        )
    os.replace(temporary, path)
    _write_json(
        path.with_suffix(".json"),
        {
            "schema_version": 1,
            "status": "PASS",
            "cache_identity": cache_id,
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
        },
    )


def _evidence_cache_identity(
    *, pair: Mapping[str, object], config: Mapping[str, object]
) -> str:
    bindings = {
        role: _file_record(_configured_path(pair[role]))
        for role in ("b3_provenance", "rgbd_export_manifest")
    }
    payload = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "visit_frame_ranges": pair["visit_frame_ranges"],
        "state_evidence": config["state_evidence"],
        "bindings": bindings,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _within_radius(
    points_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    radius_m: float,
    batch_size: int = 500_000,
) -> np.ndarray:
    output = np.zeros(len(points_xyz), dtype=np.bool_)
    if not len(points_xyz) or not len(target_xyz):
        return output
    tree = cKDTree(np.asarray(target_xyz, dtype=np.float64))
    for start in range(0, len(points_xyz), batch_size):
        stop = min(start + batch_size, len(points_xyz))
        distances, _ = tree.query(
            np.asarray(points_xyz[start:stop], dtype=np.float64),
            k=1,
            distance_upper_bound=radius_m,
            workers=-1,
        )
        output[start:stop] = np.isfinite(distances)
    return output


def _voxel_centers(voxels: np.ndarray, voxel_size_m: float = 0.05) -> np.ndarray:
    return (np.asarray(voxels, dtype=np.float64) + 0.5) * voxel_size_m


def _entity_token_indices(pair: NeuralSampleMap) -> dict[tuple[int, str], np.ndarray]:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for token_index, visit_id in enumerate(pair.visit_ids):
        contributor = int(pair.source_to_token_offsets[token_index])
        groups[(int(visit_id), pair.source_entity_ids[contributor])].append(token_index)
    return {key: np.asarray(values, dtype=np.int64) for key, values in groups.items()}


def _memory_relations(
    *,
    pair: NeuralSampleMap,
    evidence: TemporalQueryEvidence,
    t0: FineVisitSurface,
    t1: FineVisitSurface,
    prior: PriorSurfaceState,
    t0_surface_id: str,
    t1_surface_id: str,
    t1_owner_offset: int,
    config: Mapping[str, object],
) -> dict[MemoryRetrievalMode, tuple[RelationSupport, ...]]:
    if evidence.token_scores is None:
        return {mode: () for mode in MemoryRetrievalMode}
    groups = _entity_token_indices(pair)
    feature_space = str(config["feature_space_id"])
    projection_version = str(config["projection_version"])
    index = CroveMemoryIndex()
    observation_id = 0
    for (visit_id, entity_id), token_rows in sorted(groups.items()):
        if visit_id != 0:
            continue
        try:
            owner_id = int(entity_id.removeprefix("ovimap:"))
        except ValueError:
            continue
        fine_rows = np.flatnonzero(t0.owner_entity_ids == owner_id)
        descriptor = np.mean(evidence.token_scores[:, token_rows], axis=1)
        if not len(fine_rows) or not np.any(descriptor):
            continue
        frame = int(t0.last_supported_frames[fine_rows].max(initial=0))
        active = bool(np.any(prior.current_valid[fine_rows]))
        index = index.observe(
            MemoryEntityObservation(
                stable_entity_id=f"stable:memory:{owner_id}",
                owner_entity_id=owner_id,
                source_surface_id=t0_surface_id,
                source_vertex_indices=t0.source_vertex_indices[fine_rows],
                descriptor=descriptor,
                feature_space_id=feature_space,
                projection_version=projection_version,
                observation_id=observation_id,
                frame_id=max(frame, 0),
                visible_pixel_count=len(token_rows),
                quality=max(0.01, min(1.0, float(np.mean(evidence.query_scores)))),
                view_direction_xyz=(0.0, 0.0, 1.0),
                lifecycle_state=(
                    EpisodeLifecycle.ACTIVE if active else EpisodeLifecycle.DORMANT
                ),
                observation_provenance=OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT,
            )
        )
        observation_id += 1
    candidates = []
    for (visit_id, entity_id), token_rows in sorted(groups.items()):
        if visit_id != 1:
            continue
        try:
            owner_id = int(entity_id.removeprefix("ovimap:"))
        except ValueError:
            continue
        fine_owner_id = owner_id + t1_owner_offset
        fine_rows = np.flatnonzero(t1.owner_entity_ids == fine_owner_id)
        descriptor = np.mean(evidence.token_scores[:, token_rows], axis=1)
        if not len(fine_rows) or not np.any(descriptor):
            continue
        frame = int(t1.last_supported_frames[fine_rows].max(initial=0))
        candidates.append(
            MemoryCandidateObservation(
                entity_id=entity_id,
                source_surface_id=t1_surface_id,
                source_vertex_indices=t1.source_vertex_indices[fine_rows],
                descriptor=descriptor,
                feature_space_id=feature_space,
                projection_version=projection_version,
                observation_id=observation_id,
                frame_id=max(frame, 0),
                quality=max(0.01, min(1.0, float(np.mean(evidence.query_scores)))),
            )
        )
        observation_id += 1
    output = {}
    for mode in MemoryRetrievalMode:
        raw = build_memory_relation_support(
            index,
            tuple(candidates),
            retrieval_mode=mode,
            minimum_cosine=float(config["minimum_cosine"]),
            minimum_margin=float(config["minimum_margin"]),
        )
        output[mode] = adapt_relations_to_backbone(
            raw,
            t0=t0,
            t1=t1,
            t0_surface_id=t0_surface_id,
            t1_surface_id=t1_surface_id,
            t1_owner_offset=t1_owner_offset,
        )
    return output


def _geometric_config(raw: Mapping[str, object]) -> StrongGeometricRelationConfig:
    values = dict(raw)
    registration = values.pop("registration_config", {})
    if not isinstance(registration, Mapping):
        raise TypeError("g1 registration_config must be a mapping")
    return StrongGeometricRelationConfig(
        registration_config=RegistrationConfig(**registration), **values
    )


def prepare_real_pair(
    pair_config: Mapping[str, object],
    config: Mapping[str, object],
    *,
    shared_root: Path,
) -> PreparedEntityEpochPair:
    """Load expensive source-bound pair state exactly once per run process."""

    pair_id = str(pair_config["pair_id"])
    scene = str(pair_config["scene"])
    t0, t1, exported_b3_mask = load_backbone_visits(
        _configured_path(pair_config["current_map_root"])
    )
    prior = load_b3_prior_surface_state(
        _configured_path(pair_config["b3_provenance"]), t0.owner_entity_ids
    )
    if not np.array_equal(prior.current_valid, exported_b3_mask):
        raise ValueError("frozen B3 prior differs from the OVI backbone export")
    policy = load_b3_surface_policy(
        _configured_path(pair_config["b3_provenance"]), t0.owner_entity_ids
    )
    ranges = pair_config["visit_frame_ranges"]
    t1_start, t1_end = (int(value) for value in ranges["t1"])
    dataset = TesseCdRgbdDataset(
        _configured_path(pair_config["rgbd_root"]),
        scene,
        _configured_path(pair_config["rgbd_export_manifest"]),
        _configured_path(pair_config["causal_schedule"]),
    )
    t1_frames = tuple(dataset[index] for index in range(t1_start, t1_end + 1))
    evidence_config = config["state_evidence"]
    cache_id = _evidence_cache_identity(pair=pair_config, config=config)
    cache_path = shared_root / "fine_state_evidence.npz"
    evidence = _load_cached_evidence(cache_path, cache_id)
    if evidence is None:
        candidate_rows = np.flatnonzero(
            policy.states == int(CoarseSurfaceState.VISIBLE_FREE_CANDIDATE)
        )
        observation, _ = _observe_rows(
            t0.vertices_xyz,
            candidate_rows,
            t1_frames[:: int(evidence_config["frame_stride"])],
            FineEvidenceProjectionConfig(
                **{
                    key: value
                    for key, value in evidence_config.items()
                    if key not in {"frame_stride", "point_batch_size"}
                }
            ),
            point_batch_size=int(evidence_config["point_batch_size"]),
        )
        evidence = assemble_fine_surface_evidence(
            states=policy.states,
            candidate_rows=candidate_rows,
            candidate_observation=observation,
            historical_last_supported_frames=t0.last_supported_frames,
        )
        _write_cached_evidence(cache_path, evidence, cache_id)
    replacement = policy.states == int(CoarseSurfaceState.REPLACED_OCCUPIED)
    visibility_config = SignedVisibilityConfig(**config["evaluation_visibility"])
    t0_observed = policy.states != int(CoarseSurfaceState.RETAINED_UNOBSERVED)
    evaluation, _ = _evaluation_context(
        protocol=_load_json(_configured_path(pair_config["two_visit_protocol"])),
        scene=scene,
        final_frame=t1_end,
        schedule_path=_configured_path(pair_config["causal_schedule"]),
        target_manifest_path=_configured_path(pair_config["common_v2_target_manifest"]),
        frames=t1_frames,
        visibility_config=visibility_config,
    )
    current_targets = _voxel_centers(evaluation.current_semantic_voxels[:, :3])
    free_targets = _voxel_centers(evaluation.confirmed_free_voxels)
    current_support = np.concatenate(
        (
            _within_radius(t0.vertices_xyz, current_targets, radius_m=0.05),
            _within_radius(t1.vertices_xyz, current_targets, radius_m=0.05),
        )
    )
    confirmed_free = np.concatenate(
        (
            _within_radius(t0.vertices_xyz, free_targets, radius_m=0.05),
            _within_radius(t1.vertices_xyz, free_targets, radius_m=0.05),
        )
    )
    evaluation_support = EntityEpochEvaluationSupport(
        current_gt_supported_mask=current_support,
        confirmed_free_mask=confirmed_free,
        t1_surface_covered_mask=np.concatenate(
            (replacement, np.ones(len(t1.vertices_xyz), dtype=np.bool_))
        ),
        relation_labels=(),
    )
    pair = load_neural_sample_artifact(
        _configured_path(pair_config["adapter_pair_root"])
    )
    evidence_query = load_query_evidence_artifact(
        _configured_path(pair_config["query_evidence_manifest"]), pair
    )
    t0_surface_id = f"ovi-map:{pair.source_visit_map_sha256[0]}"
    t1_surface_id = f"ovi-map:{pair.source_visit_map_sha256[1]}"
    t1_owner_offset = int(pair_config["t1_owner_offset"])
    geometry = adapt_relations_to_backbone(
        build_g1_relation_support(pair, _geometric_config(config["g1"])),
        t0=t0,
        t1=t1,
        t0_surface_id=t0_surface_id,
        t1_surface_id=t1_surface_id,
        t1_owner_offset=t1_owner_offset,
    )
    projection = config["rescene_projection"]
    rescene = adapt_relations_to_backbone(
        project_queries_to_relation_support(
            pair,
            evidence_query,
            minimum_source_coverage=float(projection["minimum_source_coverage"]),
            minimum_competition_margin=float(projection["minimum_competition_margin"]),
            minimum_query_confidence=float(projection["minimum_query_confidence"]),
        ),
        t0=t0,
        t1=t1,
        t0_surface_id=t0_surface_id,
        t1_surface_id=t1_surface_id,
        t1_owner_offset=t1_owner_offset,
    )
    memories = _memory_relations(
        pair=pair,
        evidence=evidence_query,
        t0=t0,
        t1=t1,
        prior=prior,
        t0_surface_id=t0_surface_id,
        t1_surface_id=t1_surface_id,
        t1_owner_offset=t1_owner_offset,
        config=config["memory"],
    )
    crosswalk = load_tesse_semantic_crosswalk(
        _configured_path(pair_config["semantic_aliases"]),
        scene,
        _configured_path(pair_config["semantic_label_space"]),
    )
    raw_t1_records = _entity_records(_configured_path(pair_config["b2_entities"]))
    return PreparedEntityEpochPair(
        pair_id=pair_id,
        scene=scene,
        t0=t0,
        t1=t1,
        t0_records=_entity_records(_configured_path(pair_config["b0_entities"])),
        t1_records={
            owner_id + t1_owner_offset: record
            for owner_id, record in raw_t1_records.items()
        },
        prior=prior,
        evidence=evidence,
        t1_replacement_mask=replacement,
        t0_t1_observed_mask=t0_observed,
        evaluation=evaluation,
        evaluation_support=evaluation_support,
        crosswalk=crosswalk,
        t0_surface_id=t0_surface_id,
        t1_surface_id=t1_surface_id,
        geometric_relations=geometry,
        rescene_relations=rescene,
        memory_relations=memories,
        pair_sha256=pair.content_sha256(),
        checkpoint_sha256=_sha256(_configured_path(pair_config["checkpoint"])),
        query_evidence=evidence_query,
    )


def _empty_evidence(count: int) -> FineSurfaceEvidence:
    return FineSurfaceEvidence.empty(count)


def _variant_update(
    prepared: PreparedEntityEpochPair,
    *,
    variant_id: str,
    hypothesis: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[EntityEpochUpdateResult, tuple[RelationSupport, ...]]:
    count = len(prepared.t0.vertices_xyz)
    if variant_id == "D0_T1":
        prior = PriorSurfaceState(
            current_valid=np.zeros(count, dtype=np.bool_),
            retirement_reason_codes=np.full(
                count, int(SurfaceRetirementReason.DIRECT_FREE), dtype=np.uint8
            ),
            evidence_state_codes=np.full(
                count, int(CurrentEvidenceState.REVOKED_VISIBLE_FREE), dtype=np.uint8
            ),
        )
        evidence = _empty_evidence(count)
        replacement = np.zeros(count, dtype=np.bool_)
        relations: tuple[RelationSupport, ...] = ()
        update_config = EntityEpochUpdateConfig(
            **{
                **config["state_update"],
                "evidence_is_new_measurement": False,
            }
        )
    elif variant_id in {"D1_B3", "D4_RESCENE_ID_ONLY"}:
        prior = prepared.prior
        evidence = _empty_evidence(count)
        replacement = np.zeros(count, dtype=np.bool_)
        relations = (
            prepared.rescene_relations if variant_id == "D4_RESCENE_ID_ONLY" else ()
        )
        update_config = EntityEpochUpdateConfig(
            **{
                **config["state_update"],
                "evidence_is_new_measurement": False,
            }
        )
    else:
        prior = prepared.prior
        evidence = prepared.evidence
        replacement = prepared.t1_replacement_mask
        if variant_id == "D2_INHERIT":
            relations = ()
        elif variant_id == "D3_GEOM":
            relations = prepared.geometric_relations
        elif variant_id == "D5_RESCENE_EPOCH":
            relations = prepared.rescene_relations
        elif variant_id == "D6_MEMORY_EPOCH":
            mode = MemoryRetrievalMode(str(hypothesis["memory_mode"]))
            relations = prepared.memory_relations[mode]
        else:
            raise ValueError("oracle relations are evaluator-only and not executable")
        update_config = EntityEpochUpdateConfig(**config["state_update"])
    return (
        resolve_entity_epoch_update(
            prior=prior,
            source_surface_id=prepared.t0_surface_id,
            source_vertex_indices=prepared.t0.source_vertex_indices,
            t0_owner_entity_ids=prepared.t0.owner_entity_ids,
            evidence=evidence,
            t1_replacement_mask=replacement,
            relation_support=relations,
            config=update_config,
        ),
        relations,
    )


def _evaluate_composition(
    prepared: PreparedEntityEpochPair,
    composition: EntityEpochComposition,
) -> dict[str, float]:
    t0_current = composition.surface.current_valid[: len(prepared.t0.vertices_xyz)]
    snapshot = _dynamic_snapshot(
        t0_xyz=prepared.t0.vertices_xyz,
        t1_xyz=prepared.t1.vertices_xyz,
        t0_groups=_owner_row_groups(prepared.t0.owner_entity_ids),
        t1_groups=_owner_row_groups(prepared.t1.owner_entity_ids),
        t0_records=dict(prepared.t0_records),
        t1_records=dict(prepared.t1_records),
        t0_current=t0_current,
        final_frame=prepared.evaluation.frame_id,
    )
    return {
        key: float(value)
        for key, value in evaluate_two_visit_snapshot(
            snapshot,
            prepared.evaluation,
            prepared.crosswalk,
            retained_t0_xyz=prepared.t0.vertices_xyz[t0_current],
            retained_t0_t1_observed_mask=prepared.t0_t1_observed_mask[t0_current],
        ).items()
    }


def _action_csv_rows(
    prepared: PreparedEntityEpochPair,
    composition: EntityEpochComposition,
    update: EntityEpochUpdateResult,
    *,
    config_id: str,
    variant_id: str,
) -> list[dict[str, object]]:
    output = []
    for row in action_attribution_rows(
        composition, update, prepared.evaluation_support
    ):
        indices = np.asarray(row.source_vertex_indices, dtype=np.int64)
        payload = asdict(row)
        payload.pop("source_vertex_indices")
        payload.update(
            {
                "pair_id": prepared.pair_id,
                "config_id": config_id,
                "variant_id": variant_id,
                "source_vertex_indices_sha256": hashlib.sha256(
                    indices.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "source_vertex_index_min": int(indices.min()),
                "source_vertex_index_max": int(indices.max()),
            }
        )
        output.append(payload)
    return output


def _relation_csv_rows(
    prepared: PreparedEntityEpochPair,
    relations: tuple[RelationSupport, ...],
    *,
    config_id: str,
    variant_id: str,
) -> list[dict[str, object]]:
    output = []
    for row in relation_diagnostic_rows(
        relations, prepared.evaluation_support.relation_labels
    ):
        payload = asdict(row)
        payload["rejection_reasons"] = ",".join(row.rejection_reasons)
        payload.update(
            {
                "pair_id": prepared.pair_id,
                "config_id": config_id,
                "variant_id": variant_id,
            }
        )
        output.append(payload)
    return output


def _rate_value(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def execute_real_variant(
    prepared: PreparedEntityEpochPair,
    hypothesis: Mapping[str, object],
    config: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    variant_id = str(hypothesis["variant_id"])
    config_id = str(hypothesis["config_id"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    update, relations = _variant_update(
        prepared, variant_id=variant_id, hypothesis=hypothesis, config=config
    )
    composition = compose_entity_epoch_fine_surface(
        t0=prepared.t0,
        t1=prepared.t1,
        update=update,
        surface_id=f"crove-entity-epoch:{prepared.pair_id}:{config_id}:{variant_id}",
    )
    headline = _evaluate_composition(prepared, composition)
    diagnostics = evaluate_entity_epoch_actions(
        composition,
        update,
        prepared.evaluation_support,
        relation_support=relations,
    )
    invariance: dict[str, object] | None = None
    if variant_id == "D4_RESCENE_ID_ONLY":
        baseline_update, _ = _variant_update(
            prepared,
            variant_id="D1_B3",
            hypothesis={"variant_id": "D1_B3", "config_id": "H1_B3"},
            config=config,
        )
        baseline = compose_entity_epoch_fine_surface(
            t0=prepared.t0,
            t1=prepared.t1,
            update=baseline_update,
            surface_id="crove-entity-epoch:d4-invariance-baseline",
        )
        baseline_headline = _evaluate_composition(prepared, baseline)
        invariance = asdict(
            verify_d4_identity_only_invariance(
                baseline_surface=baseline.surface,
                identity_only_surface=composition.surface,
                baseline_headline=baseline_headline,
                identity_only_headline=headline,
            )
        )
    write_entity_epoch_composition(composition, output / "entity_epoch_state")
    action_rows = _action_csv_rows(
        prepared,
        composition,
        update,
        config_id=config_id,
        variant_id=variant_id,
    )
    relation_rows = _relation_csv_rows(
        prepared,
        relations,
        config_id=config_id,
        variant_id=variant_id,
    )
    _write_csv(output / "action_attribution.csv", action_rows)
    _write_csv(output / "relation_diagnostics.csv", relation_rows)
    action_counts = {
        str(reason): int(count)
        for reason, count in zip(
            *np.unique(composition.state_sidecar.action_reasons, return_counts=True),
            strict=True,
        )
    }
    transition = {
        "schema_version": 1,
        "status": "PASS",
        "action_row_counts": action_counts,
        "current_t0_rows": int(np.count_nonzero(update.current_valid)),
        "retired_t0_rows": int(np.count_nonzero(~update.current_valid)),
        "accepted_relation_count": sum(relation.accepted for relation in relations),
        "motion_verified_relation_count": sum(
            relation.motion_verified for relation in relations
        ),
    }
    _write_json(output / "state_transition_summary.json", transition)
    supported_t0 = prepared.evaluation_support.current_gt_supported_mask[
        : len(prepared.t0.vertices_xyz)
    ]
    denominator = int(np.count_nonzero(prepared.prior.current_valid & supported_t0))
    retained = int(
        np.count_nonzero(
            update.current_valid & prepared.prior.current_valid & supported_t0
        )
    )
    metrics: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "pair_id": prepared.pair_id,
        "scene": prepared.scene,
        "config_id": config_id,
        "variant_id": variant_id,
        **headline,
        "deleted_supported_rate": _rate_value(diagnostics.deleted_supported_rate.value),
        "bad_recovery_rate": _rate_value(diagnostics.bad_recovery_rate.value),
        "correct_new_coverage_rate": _rate_value(
            diagnostics.correct_new_coverage_rate.value
        ),
        "free_conflict_rate": _rate_value(diagnostics.free_conflict_rate.value),
        "retained_history_recall": 0.0 if denominator == 0 else retained / denominator,
        "diagnostics": {
            field.name: (
                getattr(diagnostics, field.name).to_json_record()
                if hasattr(getattr(diagnostics, field.name), "to_json_record")
                else getattr(diagnostics, field.name)
            )
            for field in fields(diagnostics)
        },
        "current_t0_row_count": int(np.count_nonzero(update.current_valid)),
        "relation_count": len(relations),
        "accepted_relation_count": sum(relation.accepted for relation in relations),
        "d4_invariance": invariance,
        "prediction_uses_ground_truth": False,
        "entity_epoch_state_manifest": _file_record(
            output / "entity_epoch_state" / "entity_epoch_state_manifest.json"
        ),
    }
    _write_json(output / "metrics.json", metrics)
    return metrics


def _confirmation_hypotheses(
    config: Mapping[str, object], selection: Mapping[str, object]
) -> list[Mapping[str, object]]:
    if (
        selection.get("schema_version") != 1
        or selection.get("status") != "DEV_SELECTED"
        or not isinstance(selection.get("config_id"), str)
        or not isinstance(selection.get("variant_id"), str)
    ):
        raise ValueError("confirmation selection is not a frozen DEV selection")
    hypotheses = config.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise TypeError("configured hypotheses must be a list")
    by_config: dict[str, Mapping[str, object]] = {}
    for raw in hypotheses:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("config_id"), str):
            raise TypeError("configured hypothesis is invalid")
        config_id = str(raw["config_id"])
        if config_id in by_config:
            raise ValueError("configured hypothesis IDs must be unique")
        by_config[config_id] = raw
    winner = by_config.get(str(selection["config_id"]))
    if winner is None or winner.get("variant_id") != selection["variant_id"]:
        raise ValueError("frozen DEV selection does not match a configured hypothesis")
    required_ids = ["H0_T1", "H1_B3", "H3_GEOM"]
    missing = [config_id for config_id in required_ids if config_id not in by_config]
    if missing:
        raise ValueError(f"confirmation controls are missing: {missing}")
    if str(selection["config_id"]) not in required_ids:
        required_ids.append(str(selection["config_id"]))
    return [by_config[config_id] for config_id in required_ids]


def run(
    config: Mapping[str, object],
    *,
    split: str,
    variants: Sequence[str] | None,
    frozen_selection: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    run_root, _ = _configured_roots(config)
    split_root = run_root / split
    selection = _load_json(split_root / "input_selection.json")
    if split == "confirm":
        if frozen_selection is None:
            raise ValueError("confirmation execution requires a frozen DEV selection")
        confirmation_hypotheses = _confirmation_hypotheses(config, frozen_selection)
        attempts = selection.get("pairs")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("confirmation input selection has no asset attempts")
        statuses = {str(attempt.get("asset_status")) for attempt in attempts}
        if statuses != {"READY"}:
            if len(statuses) != 1 or not statuses.issubset(_CONFIRM_STATES):
                raise ValueError("confirmation assets have inconsistent status")
            payload = {
                "schema_version": 1,
                "status": next(iter(statuses)),
                "runs": [],
                "asset_attempts": attempts,
            }
            _write_json(split_root / "run_index.json", payload)
            return []
    selected_ids = tuple(str(value) for value in selection["selected_pair_ids"])
    raw_pairs = config["splits"][split]["pairs"]
    pair_by_id = {str(pair["pair_id"]): pair for pair in raw_pairs}
    if split == "confirm":
        hypotheses = confirmation_hypotheses
        expected_variants = {str(row["variant_id"]) for row in hypotheses}
        if variants is not None and set(variants) != expected_variants:
            raise ValueError("confirmation variants must match the frozen controls")
    else:
        requested = set(ORDERED_VARIANTS[:-1] if variants is None else variants)
        hypotheses = [
            row for row in config["hypotheses"] if str(row["variant_id"]) in requested
        ]
    if not hypotheses:
        raise ValueError("no hypothesis matches the requested variants")
    records = []
    run_index = []
    for pair_id in selected_ids:
        pair_config = pair_by_id[pair_id]
        pair_root = split_root / "pairs" / pair_id
        prepared = prepare_real_pair(
            pair_config, config, shared_root=pair_root / "shared"
        )
        for hypothesis in hypotheses:
            config_id = str(hypothesis["config_id"])
            variant_id = str(hypothesis["variant_id"])

            def execute(
                _variant: str,
                output: Path,
                *,
                prepared_pair: PreparedEntityEpochPair = prepared,
                hypothesis_config: Mapping[str, object] = hypothesis,
            ) -> Mapping[str, object]:
                return execute_real_variant(
                    prepared_pair, hypothesis_config, config, output
                )

            result = run_requested_variants(
                variants=(variant_id,),
                pair_id=config_id,
                run_root=pair_root / "hypotheses",
                execute=execute,
            )[0]
            records.append(result)
            run_index.append(
                {
                    "pair_id": pair_id,
                    "config_id": config_id,
                    "variant_id": variant_id,
                    "status": result["status"],
                    "output": str(pair_root / "hypotheses" / config_id / variant_id),
                }
            )
        model_manifest = {
            "schema_version": 1,
            "status": "PASS",
            "pair_id": pair_id,
            "pair_sha256": prepared.pair_sha256,
            "checkpoint_sha256": prepared.checkpoint_sha256,
            "checkpoint_status": "NOT_APPLICABLE_REUSED",
            "forward_status": "REUSED_EXACT_PAIR_BOUND_FORWARD",
            "forward_count": 1,
            "training_count": 0,
            "runtime_s": prepared.query_evidence.runtime_s,
            "peak_memory_bytes": prepared.query_evidence.peak_memory_bytes,
            "memory_observation_provenance": OFFLINE_REPLAY_ON_FINAL_OVI_SUPPORT,
            "multiview_status": "UNAVAILABLE_SINGLE_FINAL_OVI_OBSERVATION",
            "cache_identity": cache_identity(
                pair_sha256=prepared.pair_sha256,
                method_config=config["rescene_projection"],
                checkpoint_sha256=prepared.checkpoint_sha256,
            ),
        }
        _write_json(pair_root / "model_manifest.json", model_manifest)
    _write_json(
        split_root / "run_index.json",
        {"schema_version": 1, "status": "PASS", "runs": run_index},
    )
    return records


def cache_identity(
    *,
    pair_sha256: str,
    method_config: Mapping[str, object],
    checkpoint_sha256: str | None,
) -> str:
    """Bind a reusable forward to its pair, method configuration, and weights."""

    for value, label in ((pair_sha256, "pair_sha256"),):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be a SHA-256 hex digest")
        int(value, 16)
    if checkpoint_sha256 is not None:
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
        int(checkpoint_sha256, 16)
    if not isinstance(method_config, Mapping):
        raise TypeError("method_config must be a mapping")
    payload = {
        "schema_version": 1,
        "pair_sha256": pair_sha256,
        "method_config": dict(method_config),
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def select_development_pairs(
    candidates: Sequence[Mapping[str, object]],
    *,
    maximum_additional_pairs: int,
) -> tuple[str, ...]:
    """Freeze DEV pairs using readiness, event type, and visibility metadata only."""

    if (
        type(maximum_additional_pairs) is not int
        or not 0 <= maximum_additional_pairs <= 2
    ):
        raise ValueError("maximum_additional_pairs must be in [0, 2]")
    normalized = []
    for raw in candidates:
        pair_id = raw.get("pair_id")
        event_type = raw.get("event_type")
        coverage = raw.get("visibility_coverage")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or not isinstance(event_type, str)
            or not event_type
            or isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not 0.0 <= float(coverage) <= 1.0
            or type(raw.get("fixed")) is not bool
        ):
            raise ValueError("development pair metadata is invalid")
        normalized.append(
            (
                pair_id,
                bool(raw["fixed"]),
                str(raw.get("asset_status")),
                event_type,
                float(coverage),
            )
        )
    if len({row[0] for row in normalized}) != len(normalized):
        raise ValueError("development pair IDs must be unique")
    fixed = [row for row in normalized if row[1]]
    if len(fixed) != 1 or fixed[0][2] not in _READY_STATES:
        raise ValueError("exactly one ready fixed development pair is required")
    eligible = [row for row in normalized if not row[1] and row[2] in _READY_STATES]
    eligible.sort(key=lambda row: (-row[4], row[3], row[0]))
    return (fixed[0][0], *(row[0] for row in eligible[:maximum_additional_pairs]))


def _mean(rows: Sequence[Mapping[str, object]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return sum(values) / len(values)


def select_global_development_config(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_pair_ids: Sequence[str],
    gates: Mapping[str, float],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Select one global DEV configuration after all-pair safety gates."""

    required_gates = {
        "maximum_ghost",
        "minimum_surface_precision_at_5cm",
        "maximum_deleted_supported_rate",
    }
    if set(gates) != required_gates:
        raise ValueError("development selection gates are incomplete")
    expected = tuple(expected_pair_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_pair_ids must be unique and non-empty")
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        config_id = row.get("config_id")
        variant_id = row.get("variant_id")
        if not isinstance(config_id, str) or variant_id not in _RANKING_VARIANTS:
            continue
        groups[(config_id, str(variant_id))].append(row)
    aggregates: list[dict[str, object]] = []
    for (config_id, variant_id), values in sorted(groups.items()):
        pair_ids = tuple(str(row.get("pair_id")) for row in values)
        complete = sorted(pair_ids) == sorted(expected) and len(pair_ids) == len(
            set(pair_ids)
        )
        passed = complete and all(row.get("status") == "PASS" for row in values)
        eligible = bool(
            passed
            and all(float(row["ghost"]) <= gates["maximum_ghost"] for row in values)
            and all(
                float(row["surface_precision_at_5cm"])
                >= gates["minimum_surface_precision_at_5cm"]
                for row in values
            )
            and all(
                float(row["deleted_supported_rate"])
                <= gates["maximum_deleted_supported_rate"]
                for row in values
            )
        )
        record: dict[str, object] = {
            "config_id": config_id,
            "variant_id": variant_id,
            "pair_count": len(values),
            "complete": complete,
            "eligible": eligible,
        }
        if passed:
            record.update(
                {
                    "mean_current_miou": _mean(values, "current_miou"),
                    "mean_ghost": _mean(values, "ghost"),
                    "mean_background_f1_at_5cm": _mean(values, "background_f1_at_5cm"),
                    "mean_surface_precision_at_5cm": _mean(
                        values, "surface_precision_at_5cm"
                    ),
                    "mean_deleted_supported_rate": _mean(
                        values, "deleted_supported_rate"
                    ),
                    "mean_retained_history_recall": _mean(
                        values, "retained_history_recall"
                    ),
                }
            )
        aggregates.append(record)
    eligible_rows = [row for row in aggregates if row["eligible"]]
    if not eligible_rows:
        raise ValueError("no complete development configuration satisfies the gates")
    winner = max(
        eligible_rows,
        key=lambda row: (
            float(row["mean_current_miou"]),
            float(row["mean_background_f1_at_5cm"]),
            float(row["mean_retained_history_recall"]),
            -ORDERED_VARIANTS.index(str(row["variant_id"])),
            str(row["config_id"]),
        ),
    )
    return {
        "config_id": str(winner["config_id"]),
        "variant_id": str(winner["variant_id"]),
    }, aggregates


def run_requested_variants(
    *,
    variants: Sequence[str],
    pair_id: str,
    run_root: Path,
    execute: Callable[[str, Path], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Run ordered variants independently so one failure cannot erase later rows."""

    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be non-empty")
    requested = tuple(variants)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("variants must be unique and non-empty")
    if any(variant not in ORDERED_VARIANTS for variant in requested):
        raise ValueError("unknown entity-episode variant")
    records: list[dict[str, object]] = []
    for variant in requested:
        output = run_root / pair_id / variant
        metrics_path = output / "metrics.json"
        if metrics_path.is_file():
            cached = _load_json(metrics_path)
            if cached.get("status") == "PASS":
                records.append({"pair_id": pair_id, "variant_id": variant, **cached})
                continue
        try:
            measured = dict(execute(variant, output))
            records.append(
                {
                    "pair_id": pair_id,
                    "variant_id": variant,
                    "status": "PASS",
                    **measured,
                }
            )
        except Exception as error:
            output.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": 1,
                "status": "FAILED",
                "pair_id": pair_id,
                "variant_id": variant,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _write_json(output / "failure.json", failure)
            records.append(failure)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "run", "summarize"), required=True
    )
    parser.add_argument("--split", choices=("dev", "confirm"), required=True)
    parser.add_argument("--variants", nargs="+", choices=ORDERED_VARIANTS)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args(argv)
    if args.split == "confirm" and args.selection is None:
        parser.error("--selection is required for --split confirm")
    if args.variants is not None and len(set(args.variants)) != len(args.variants):
        parser.error("--variants cannot contain duplicates")
    return args


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _configured_roots(config: Mapping[str, object]) -> tuple[Path, Path]:
    return (
        _configured_path(config["run_root"]),
        _configured_path(config["compact_output_root"]),
    )


def _probe_pair(raw: Mapping[str, object], *, confirmation: bool) -> dict[str, object]:
    required = raw.get("required_inputs")
    if not isinstance(required, Mapping) or not required:
        raise ValueError("pair required_inputs must be a non-empty mapping")
    bindings: dict[str, object] = {}
    missing: list[str] = []
    for role, value in sorted(required.items()):
        path = _configured_path(value)
        if path.is_file() and not path.is_symlink():
            bindings[str(role)] = _file_record(path)
        elif path.is_dir() and not path.is_symlink():
            bindings[str(role)] = {"path": str(path), "status": "DIRECTORY_PRESENT"}
        else:
            missing.append(str(path))
    if confirmation:
        raw_paths = tuple(
            _configured_path(value) for value in raw.get("raw_inputs", ())
        )
        native_paths = tuple(
            _configured_path(value) for value in raw.get("native_inputs", ())
        )
        if raw_paths and any(not path.exists() for path in raw_paths):
            status = "RAW_MISSING"
        elif native_paths and any(not path.exists() for path in native_paths):
            status = "DERIVED_NOT_BUILT"
        else:
            status = "READY" if not missing else "DERIVED_NOT_BUILT"
        if status not in _CONFIRM_STATES:
            raise AssertionError("invalid confirmation status")
    else:
        status = "READY" if not missing else "INPUT_MISSING"
    return {
        "pair_id": raw["pair_id"],
        "scene": raw["scene"],
        "fixed": bool(raw.get("fixed", False)),
        "event_type": raw["event_type"],
        "visibility_coverage": float(raw["visibility_coverage"]),
        "asset_status": status,
        "input_bindings": bindings,
        "missing_paths": missing,
    }


def prepare(config: Mapping[str, object], *, split: str) -> dict[str, object]:
    run_root, compact_root = _configured_roots(config)
    raw_pairs = config["splits"][split]["pairs"]  # type: ignore[index]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("configured split must contain at least one pair")
    pairs = [_probe_pair(raw, confirmation=split == "confirm") for raw in raw_pairs]
    if split == "dev":
        selected = select_development_pairs(
            pairs,
            maximum_additional_pairs=int(
                config["splits"][split]["maximum_additional_pairs"]  # type: ignore[index]
            ),
        )
    else:
        selected = tuple(str(pair["pair_id"]) for pair in pairs)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "split": split,
        "selection_basis": ["asset_status", "event_type", "visibility_coverage"],
        "selected_pair_ids": list(selected),
        "pairs": pairs,
        "score_driven_replacement_allowed": False,
    }
    destination = run_root / split / "input_selection.json"
    _write_json(destination, payload)
    compact_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        compact_root
        / (
            "input_selection.json"
            if split == "dev"
            else "confirmation_input_selection.json"
        ),
        _portableize_paths(payload),
    )
    return payload


def _read_metric_rows(
    run_root: Path, pair_ids: Sequence[str]
) -> list[dict[str, object]]:
    rows = []
    for pair_id in pair_ids:
        pair_root = run_root / pair_id / "hypotheses"
        for path in sorted(pair_root.rglob("metrics.json")):
            payload = _load_json(path)
            raw_retained = payload.get("retained_history_recall")
            raw_deleted = payload.get("deleted_supported_rate")
            variant_id = str(payload.get("variant_id"))
            payload["raw_retained_history_value"] = raw_retained
            payload["raw_deleted_supported_rate"] = raw_deleted
            payload["deleted_supported_rate"] = (
                1.0 if variant_id == "D0_T1" else float(raw_deleted)
            )
            payload["retained_history_recall"] = 1.0 - float(
                payload["deleted_supported_rate"]
            )
            payload["retained_history_normalization"] = (
                "prior_valid_supported_rows_retained_over_prior_valid_supported_rows"
            )
            portable = _portableize_paths(payload)
            if not isinstance(portable, dict):
                raise TypeError("portable metric payload must remain a mapping")
            rows.append({"pair_id": pair_id, **portable})
        for path in sorted(pair_root.rglob("failure.json")):
            portable = _portableize_paths(_load_json(path))
            if not isinstance(portable, dict):
                raise TypeError("portable failure payload must remain a mapping")
            rows.append({"pair_id": pair_id, **portable})
    return rows


def summarize(
    config: Mapping[str, object],
    *,
    split: str,
    frozen_selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    run_root, compact_root = _configured_roots(config)
    selection = _load_json(run_root / split / "input_selection.json")
    pair_ids = tuple(str(value) for value in selection["selected_pair_ids"])
    rows = _read_metric_rows(run_root / split / "pairs", pair_ids)
    compact_root.mkdir(parents=True, exist_ok=True)
    if split == "confirm":
        if frozen_selection is None:
            raise ValueError("confirmation summary requires a frozen DEV selection")
        _confirmation_hypotheses(config, frozen_selection)
        confirmation = {
            "schema_version": 1,
            "status": (
                "PASS"
                if all(pair["asset_status"] == "READY" for pair in selection["pairs"])
                else str(selection["pairs"][0]["asset_status"])
            ),
            "pair_ids": list(pair_ids),
            "rows": rows,
            "asset_attempts": selection["pairs"],
            "office_threshold_retuning_allowed": False,
            "frozen_development_selection": dict(frozen_selection),
            "attempted_command": (
                "python scripts/evaluation/run_crove_entity_epoch.py --config "
                "configs/evaluation/crove_entity_epoch_v2.json --phase run "
                "--split confirm --selection "
                "configs/evaluation/results/crove_entity_epoch_v2/selected_config.json"
            ),
        }
        _write_json(
            compact_root / "confirmation_results.json",
            _portableize_paths(confirmation),
        )
        return confirmation
    _write_csv(compact_root / "metrics_per_pair.csv", rows)
    action_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for pair_id in pair_ids:
        pair_root = run_root / split / "pairs" / pair_id
        hypotheses_root = pair_root / "hypotheses"
        for path in sorted(hypotheses_root.rglob("action_attribution.csv")):
            with path.open("r", encoding="utf-8", newline="") as stream:
                action_rows.extend(dict(row) for row in csv.DictReader(stream))
        for path in sorted(hypotheses_root.rglob("relation_diagnostics.csv")):
            with path.open("r", encoding="utf-8", newline="") as stream:
                relation_rows.extend(dict(row) for row in csv.DictReader(stream))
        for path in sorted(hypotheses_root.rglob("state_transition_summary.json")):
            relative = path.relative_to(hypotheses_root).parts
            transition_rows.append(
                {
                    "pair_id": pair_id,
                    "config_id": relative[0],
                    "variant_id": relative[1],
                    **_load_json(path),
                }
            )
        model_path = pair_root / "model_manifest.json"
        if model_path.is_file():
            model_rows.append(_load_json(model_path))
    _write_csv(compact_root / "action_attribution.csv", action_rows)
    _write_csv(compact_root / "relation_diagnostics.csv", relation_rows)
    _write_json(
        compact_root / "state_transition_summary.json",
        {"schema_version": 1, "status": "PASS", "rows": transition_rows},
    )
    _write_json(
        compact_root / "model_manifest.json",
        {"schema_version": 1, "status": "PASS", "pairs": model_rows},
    )
    run_index_path = run_root / split / "run_index.json"
    if run_index_path.is_file():
        _write_json(
            compact_root / "run_index.json",
            _portableize_paths(_load_json(run_index_path)),
        )
    selected, aggregates = select_global_development_config(
        rows,
        expected_pair_ids=pair_ids,
        gates=config["selection_gates"],  # type: ignore[arg-type]
    )
    selected_payload = {
        "schema_version": 1,
        "status": "DEV_SELECTED",
        **selected,
        "pair_ids": list(pair_ids),
        "selection_gates": config["selection_gates"],
    }
    _write_json(compact_root / "selected_config.json", selected_payload)
    _write_json(
        compact_root / "aggregate_metrics.json",
        {"schema_version": 1, "status": "PASS", "rows": aggregates},
    )
    return selected_payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_json(args.config)
    if config.get("schema_version") != 2:
        raise ValueError("entity-episode config schema_version must be 2")
    frozen_selection = _load_json(args.selection) if args.split == "confirm" else None
    if args.phase == "prepare":
        if frozen_selection is not None:
            _confirmation_hypotheses(config, frozen_selection)
        prepare(config, split=args.split)
    elif args.phase == "summarize":
        summarize(
            config,
            split=args.split,
            frozen_selection=frozen_selection,
        )
    else:
        run(
            config,
            split=args.split,
            variants=args.variants,
            frozen_selection=frozen_selection,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ORDERED_VARIANTS",
    "cache_identity",
    "parse_args",
    "prepare",
    "prepare_real_pair",
    "run",
    "run_requested_variants",
    "select_development_pairs",
    "select_global_development_config",
    "summarize",
]
