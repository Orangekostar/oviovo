"""Strict conversion from one native OVI-MAP run to a two-visit map."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.build_tesse_ovimap_static_anchor import PINNED_OVIMAP_COMMIT
from src.evaluation.baselines.adapters import adapt_ovimap
from src.evaluation.baselines.contracts import RuntimeBreakdown
from src.evaluation.baselines.ovimap import (
    bind_mesh_instances,
    load_instance_mesh,
    parse_instance_color_log,
)
from src.oviv2.two_visit_contracts import VisitMap


@dataclass(frozen=True, slots=True)
class SemanticLabelInput:
    """OVI-owned evidence exposed to an optional open-vocabulary labeler."""

    entity_id: str
    semantic_embedding: np.ndarray
    observation_count: int

    def __post_init__(self) -> None:
        embedding = np.array(self.semantic_embedding, dtype=np.float32, copy=True)
        if embedding.ndim != 1 or not len(embedding) or not np.all(np.isfinite(embedding)):
            raise ValueError("semantic_embedding must be a finite non-empty vector")
        embedding.setflags(write=False)
        object.__setattr__(self, "semantic_embedding", embedding)


SemanticLabeler = Callable[
    [tuple[SemanticLabelInput, ...]], Mapping[str, tuple[str, float]]
]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _read_regular(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = absolute.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    data = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise ValueError(f"{label} changed while reading")
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _file_record(path: Path, *, label: str) -> dict[str, object]:
    data = _read_regular(path, label=label)
    return {
        "path": str(Path(os.path.abspath(os.fspath(path)))),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _bound_artifacts(
    native: Mapping[str, Any],
) -> dict[str, tuple[Path, dict[str, object]]]:
    raw = native.get("artifacts")
    required = {"instance_mesh", "semantic_features", "instance_color_log"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("native manifest artifacts are incomplete")
    result: dict[str, tuple[Path, dict[str, object]]] = {}
    for role in sorted(required):
        declared = raw[role]
        if not isinstance(declared, Mapping) or set(declared) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"native artifact binding is invalid: {role}")
        path_value = declared.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError(f"native artifact path must be absolute: {role}")
        path = Path(path_value)
        observed = _file_record(path, label=f"native {role}")
        if dict(declared) != observed:
            raise ValueError(f"native artifact binding mismatch: {role}")
        result[role] = (path, observed)
    return result


def _validate_materialized(
    payload: Mapping[str, Any],
    *,
    visit_name: str,
    scene: str,
    start: int,
    end: int,
) -> None:
    expected_count = end - start + 1
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "MATERIALIZED_INPUT_PASS"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("scene") != scene
        or payload.get("visit_id") != visit_name
        or payload.get("frame_count") != expected_count
        or payload.get("source_frame_interval") != [start, end]
    ):
        raise ValueError("materialized visit manifest identity mismatch")
    source_hash = payload.get("source_input_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("materialized visit source hash is invalid")


def _validate_native(
    payload: Mapping[str, Any], *, scene: str, frame_count: int
) -> None:
    expected_frames = list(range(frame_count))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "PASS"
        or payload.get("state") != "MAPPING_PASS"
        or payload.get("scene") != scene
        or payload.get("frame_ids") != expected_frames
    ):
        raise ValueError("native OVI manifest identity mismatch")
    preflight = payload.get("preflight")
    postflight = payload.get("postflight")
    if not isinstance(preflight, Mapping) or dict(preflight) != postflight:
        raise ValueError("native OVI preflight and postflight differ")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("scene") != scene
        or preflight.get("frame_ids") != expected_frames
        or preflight.get("ovimap_commit") != PINNED_OVIMAP_COMMIT
    ):
        raise ValueError("native OVI preflight identity mismatch")


def _semantic_instances(data: bytes) -> dict[int, Mapping[str, Any]]:
    try:
        raw = pickle.loads(data)
    except Exception as error:
        raise ValueError("OVI semantic features are not a readable pickle") from error
    if not isinstance(raw, Mapping):
        raise TypeError("OVI semantic features must contain a mapping")
    result: dict[int, Mapping[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("OVI semantic feature keys must be integer instance IDs")
        if not isinstance(value, Mapping):
            raise TypeError("OVI semantic feature records must be mappings")
        instance_id = int(key)
        if instance_id in result:
            raise ValueError("OVI semantic feature instance IDs must be unique")
        result[instance_id] = value
    return result


def _apply_semantic_labels(snapshot, labeler: SemanticLabeler | None) -> None:
    if labeler is None:
        return
    evidence = tuple(
        SemanticLabelInput(
            entity_id=entity.entity_id,
            semantic_embedding=entity.semantic_embedding,
            observation_count=int(entity.metadata.get("observation_count", 0)),
        )
        for entity in snapshot.entities
        if entity.semantic_embedding is not None
    )
    labels = labeler(evidence)
    if not isinstance(labels, Mapping):
        raise TypeError("semantic labeler must return a mapping")
    entities = {entity.entity_id: entity for entity in snapshot.entities}
    evidence_ids = {item.entity_id for item in evidence}
    for entity_id, value in labels.items():
        if entity_id not in evidence_ids or entity_id not in entities:
            raise ValueError(f"semantic labeler returned unknown OVI entity: {entity_id}")
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError("semantic label values must be (label, score) pairs")
        label, score = value
        if not isinstance(label, str) or not label.strip():
            raise ValueError("semantic label must be a non-empty string")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float, np.number))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("semantic label score must be finite in [0, 1]")
        entity = entities[entity_id]
        entity.semantic_label = label.strip()
        entity.semantic_score = float(score)
        entity.metadata["semantic_match_score"] = float(score)


def load_ovimap_visit(
    *,
    native_manifest: Path,
    materialized_manifest: Path,
    visit_id: int,
    scene: str,
    coordinate_frame_id: str,
    source_manifest_sha256: str,
    observed_frame_start: int,
    observed_frame_end: int,
    semantic_labeler: SemanticLabeler | None = None,
) -> VisitMap:
    """Load one independently mapped visit after verifying all native artifacts."""

    if type(visit_id) is not int or visit_id not in {0, 1}:
        raise ValueError("visit_id must be exactly 0 or 1")
    if (
        type(observed_frame_start) is not int
        or type(observed_frame_end) is not int
        or observed_frame_start < 0
        or observed_frame_end < observed_frame_start
    ):
        raise ValueError("observed frame interval is invalid")
    visit_name = f"t{visit_id}"
    native_path = Path(native_manifest)
    materialized_path = Path(materialized_manifest)
    native_bytes = _read_regular(native_path, label="native mapping manifest")
    materialized_bytes = _read_regular(
        materialized_path, label="materialized visit manifest"
    )
    native = _json_object(native_bytes, label="native mapping manifest")
    materialized = _json_object(
        materialized_bytes, label="materialized visit manifest"
    )
    frame_count = observed_frame_end - observed_frame_start + 1
    _validate_materialized(
        materialized,
        visit_name=visit_name,
        scene=scene,
        start=observed_frame_start,
        end=observed_frame_end,
    )
    _validate_native(native, scene=scene, frame_count=frame_count)
    artifacts = _bound_artifacts(native)
    semantic_bytes = _read_regular(
        artifacts["semantic_features"][0], label="native semantic_features"
    )
    semantic_instances = _semantic_instances(semantic_bytes)
    colors_by_instance = parse_instance_color_log(
        artifacts["instance_color_log"][0]
    )
    points_by_color, background = load_instance_mesh(
        artifacts["instance_mesh"][0], colors_by_instance.values()
    )
    instances = bind_mesh_instances(
        semantic_instances,
        colors_by_instance,
        points_by_color,
    )
    if not instances:
        raise ValueError("OVI visit contains no mesh-backed instances")
    artifact = adapt_ovimap(
        instances,
        points_by_color=points_by_color,
        scene_id=scene,
        timestamp=float(observed_frame_end),
        upstream_commit=PINNED_OVIMAP_COMMIT,
        runtime=RuntimeBreakdown(frame_count=frame_count),
        background_xyz=background,
        semantic_label_source=(
            "method_output.feat SigLIP-L/16-384 canonical-relative zero-shot"
        ),
        protocol_notes=(
            "visit mapped independently from zero-based RGB-D materialization",
            "geometry remains native OVI-MAP surface output",
        ),
    )
    snapshot = artifact.snapshot
    snapshot.method = f"OVI-MAP two-visit {visit_name}"
    for entity in snapshot.entities:
        local_first = entity.first_seen
        local_last = entity.last_seen
        entity.first_seen = float(observed_frame_start) + local_first
        entity.last_seen = float(observed_frame_start) + local_last
        source_instance_id = int(entity.entity_id.split(":", 1)[1])
        entity.metadata.update(
            {
                "visit_id": visit_id,
                "source_instance_id": source_instance_id,
                "geometry_authority": f"ovi_{visit_name}",
                "semantic_authority": f"ovi_{visit_name}",
                "native_manifest_sha256": hashlib.sha256(native_bytes).hexdigest(),
            }
        )
    _apply_semantic_labels(snapshot, semantic_labeler)

    if _read_regular(native_path, label="native mapping manifest") != native_bytes:
        raise ValueError("native mapping manifest changed during conversion")
    if (
        _read_regular(materialized_path, label="materialized visit manifest")
        != materialized_bytes
    ):
        raise ValueError("materialized visit manifest changed during conversion")
    for role, (path, record) in artifacts.items():
        if _file_record(path, label=f"native {role}") != record:
            raise ValueError(f"native artifact changed during conversion: {role}")
    return VisitMap(
        visit_id=visit_id,
        snapshot=snapshot,
        coordinate_frame_id=coordinate_frame_id,
        source_manifest_sha256=source_manifest_sha256,
        map_voxel_size_m=0.01,
        observed_frame_start=observed_frame_start,
        observed_frame_end=observed_frame_end,
    )


__all__ = ["SemanticLabelInput", "SemanticLabeler", "load_ovimap_visit"]
