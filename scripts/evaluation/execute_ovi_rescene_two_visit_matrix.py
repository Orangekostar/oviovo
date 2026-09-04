#!/usr/bin/env python3
"""Execute real OVI B0-B4 variants for the frozen two-visit protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_tesse_ovimap_static_anchor import (
    _directory_record,
)
from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    _file_record as _common_file_record,
)
from scripts.evaluation.evaluate_tesse_cd_common_v2 import (
    _load_schedule,
    _load_targets,
    _voxel_centers,
)
from scripts.evaluation.freeze_tesse_two_visit_protocol import (
    PROTOCOL_ID,
    protocol_content_sha256,
)
from scripts.evaluation.run_ovi_rescene_two_visit_matrix import (
    MatrixError,
    VariantExecutionArtifacts,
    execute_required_matrix,
    load_and_validate_matrix,
)
from src.datasets.tesse_cd import TesseCdRgbdDataset
from src.evaluation.baselines.tesse_semantics import (
    TesseSemanticCrosswalk,
    load_tesse_semantic_crosswalk,
)
from src.evaluation.contracts import MapSnapshot
from src.evaluation.exporters.oviovo import write_map_snapshot
from src.evaluation.two_visit_snapshot_metrics import (
    TwoVisitEvaluationContext,
    evaluate_two_visit_snapshot,
)
from src.oviv2.geometric_pair_reasoner import (
    GeometricReasonerConfig,
)
from src.oviv2.ovimap_visit_loader import (
    SemanticLabeler,
    load_ovimap_visit,
    make_relative_semantic_labeler,
)
from src.oviv2.two_visit_contracts import (
    VisitMap,
    snapshot_content_sha256,
    validate_visit_pair,
)
from src.oviv2.two_visit_current_map import (
    SignedVisibilityGrid,
    TwoVisitCurrentMap,
    write_two_visit_current_map,
)
from src.oviv2.two_visit_execution import (
    SignedVisibilityConfig,
    build_static_baseline_snapshot,
    build_visibility_baseline,
    derive_observed_point_mask,
    derive_signed_visibility,
)


@dataclass(frozen=True, slots=True)
class PreparedTwoVisitExecution:
    """Source-bound inputs shared by all deterministic Apartment variants."""

    t0: VisitMap
    t1: VisitMap
    visibility: SignedVisibilityGrid
    evaluation: TwoVisitEvaluationContext
    crosswalk: TesseSemanticCrosswalk
    object_semantic_labels: frozenset[str]
    t0_entity_observed_masks: Mapping[str, np.ndarray]
    t0_background_observed_mask: np.ndarray
    ovi_t0_artifact_bytes: int
    ovi_t1_artifact_bytes: int
    input_bindings: Mapping[str, Mapping[str, object]]
    evaluation_bindings: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        validate_visit_pair(self.t0, self.t1)
        if not isinstance(self.visibility, SignedVisibilityGrid):
            raise TypeError("visibility must be SignedVisibilityGrid")
        if not isinstance(self.evaluation, TwoVisitEvaluationContext):
            raise TypeError("evaluation must be TwoVisitEvaluationContext")
        if not isinstance(self.crosswalk, TesseSemanticCrosswalk):
            raise TypeError("crosswalk must be TesseSemanticCrosswalk")
        if (
            not isinstance(self.object_semantic_labels, frozenset)
            or not self.object_semantic_labels
            or any(
                not isinstance(label, str) or not label.strip()
                for label in self.object_semantic_labels
            )
        ):
            raise ValueError("object_semantic_labels must be a non-empty frozenset")
        if self.evaluation.frame_id != int(self.t1.snapshot.timestamp):
            raise ValueError("evaluation frame must equal the t1 snapshot timestamp")
        if self.crosswalk.scene != self.t1.snapshot.scene_id:
            raise ValueError("crosswalk scene must equal the OVI visit scene")
        expected_ids = {entity.entity_id for entity in self.t0.snapshot.entities}
        if set(self.t0_entity_observed_masks) != expected_ids:
            raise ValueError("t0 observed masks must cover every OVI entity")
        masks: dict[str, np.ndarray] = {}
        for entity in self.t0.snapshot.entities:
            raw = np.asarray(self.t0_entity_observed_masks[entity.entity_id])
            if raw.dtype != np.bool_ or raw.shape != (len(entity.points_xyz),):
                raise ValueError("t0 entity observed mask shape is invalid")
            mask = np.array(raw, dtype=np.bool_, copy=True)
            mask.setflags(write=False)
            masks[entity.entity_id] = mask
        background_count = (
            0
            if self.t0.snapshot.background_xyz is None
            else len(self.t0.snapshot.background_xyz)
        )
        raw_background = np.asarray(self.t0_background_observed_mask)
        if raw_background.dtype != np.bool_ or raw_background.shape != (
            background_count,
        ):
            raise ValueError("t0 background observed mask shape is invalid")
        background = np.array(raw_background, dtype=np.bool_, copy=True)
        background.setflags(write=False)
        for name in ("ovi_t0_artifact_bytes", "ovi_t1_artifact_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.input_bindings, Mapping)
            or not self.input_bindings
            or not isinstance(self.evaluation_bindings, Mapping)
            or not self.evaluation_bindings
            or set(self.input_bindings) & set(self.evaluation_bindings)
        ):
            raise ValueError(
                "method and evaluation bindings must be non-empty and disjoint"
            )
        object.__setattr__(self, "t0_entity_observed_masks", MappingProxyType(masks))
        object.__setattr__(self, "t0_background_observed_mask", background)
        object.__setattr__(
            self,
            "input_bindings",
            MappingProxyType(
                {
                    str(role): MappingProxyType(dict(record))
                    for role, record in self.input_bindings.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "evaluation_bindings",
            MappingProxyType(
                {
                    str(role): MappingProxyType(dict(record))
                    for role, record in self.evaluation_bindings.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticVocabulary:
    """Complete native scene label space with an explicit object subset."""

    scene: str
    classes: tuple[str, ...]
    class_semantic_ids: tuple[int, ...]
    object_semantic_ids: frozenset[int]

    @property
    def object_classes(self) -> frozenset[str]:
        return frozenset(
            label
            for label, semantic_id in zip(
                self.classes,
                self.class_semantic_ids,
                strict=True,
            )
            if semantic_id in self.object_semantic_ids
        )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, value: object) -> None:
    data = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path, *, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _absolute_record(path: Path) -> dict[str, object]:
    absolute = path.absolute()
    if absolute.resolve(strict=True) != absolute or not absolute.is_file():
        raise ValueError(f"input must be a direct regular file: {path}")
    data = absolute.read_bytes()
    return {
        "path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _bound_path(record: object, *, label: str) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} binding path must be absolute")
    path = Path(raw_path)
    observed = _absolute_record(path)
    if observed != dict(record):
        raise ValueError(f"{label} binding mismatch")
    return path, observed


def load_two_visit_ovi_inputs(
    *,
    protocol_path: Path,
    two_visit_ovi_manifest: Path,
    scene: str,
) -> dict[str, Any]:
    """Reopen the complete OVI run graph and return verified visit paths."""

    protocol_path = Path(protocol_path).absolute()
    ovi_manifest_path = Path(two_visit_ovi_manifest).absolute()
    protocol = _load_json(protocol_path, label="two-visit protocol")
    ovi_manifest = _load_json(ovi_manifest_path, label="two-visit OVI manifest")
    if scene not in {"apartment", "office"}:
        raise ValueError("scene must be apartment or office")
    if (
        ovi_manifest.get("schema_version") != 1
        or ovi_manifest.get("status") != "TWO_VISIT_OVI_PASS"
        or ovi_manifest.get("protocol_id") != PROTOCOL_ID
        or ovi_manifest.get("protocol_content_sha256")
        != protocol_content_sha256(protocol)
        or ovi_manifest.get("scene") != scene
    ):
        raise ValueError("two-visit OVI manifest identity mismatch")
    declared_protocol_path, declared_protocol = _bound_path(
        ovi_manifest.get("protocol"), label="protocol"
    )
    if declared_protocol_path != protocol_path or declared_protocol != _absolute_record(
        protocol_path
    ):
        raise ValueError("two-visit OVI protocol binding mismatch")
    scene_record = protocol.get("scenes", {}).get(scene)
    if not isinstance(scene_record, Mapping):
        raise TypeError("protocol scene is unavailable")
    expected_method_input = scene_record.get("method_input_manifest")
    if not isinstance(expected_method_input, Mapping):
        raise TypeError("protocol method input is invalid")
    method_input_path, method_input_record = _bound_path(
        ovi_manifest.get("method_input_manifest"), label="method input manifest"
    )
    method_input = _load_json(method_input_path, label="method input manifest")
    if method_input != dict(expected_method_input):
        raise ValueError("method input manifest differs from frozen protocol")
    raw_visits = ovi_manifest.get("visits")
    expected_visits = expected_method_input.get("visits")
    if (
        not isinstance(raw_visits, Mapping)
        or set(raw_visits) != {"t0", "t1"}
        or not isinstance(expected_visits, Mapping)
    ):
        raise ValueError("two-visit OVI visit records are invalid")
    visits: dict[str, dict[str, object]] = {}
    bindings: dict[str, Mapping[str, object]] = {
        "protocol": declared_protocol,
        "method_input_manifest": method_input_record,
        "two_visit_ovi_manifest": _absolute_record(ovi_manifest_path),
    }
    for visit_name in ("t0", "t1"):
        raw = raw_visits[visit_name]
        expected = expected_visits.get(visit_name)
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {
                "run_id",
                "source_frame_interval",
                "source_input_sha256",
                "materialized_manifest",
                "native_manifest",
            }
            or not isinstance(expected, Mapping)
            or raw.get("source_frame_interval")
            != [expected.get("start_frame"), expected.get("end_frame")]
            or raw.get("source_input_sha256") != expected.get("input_sha256")
            or not isinstance(raw.get("run_id"), str)
            or not raw["run_id"]
        ):
            raise ValueError(f"two-visit OVI {visit_name} identity mismatch")
        materialized_path, materialized_record = _bound_path(
            raw.get("materialized_manifest"),
            label=f"{visit_name} materialized manifest",
        )
        native_path, native_record = _bound_path(
            raw.get("native_manifest"), label=f"{visit_name} native manifest"
        )
        visits[visit_name] = {
            "run_id": raw["run_id"],
            "start_frame": expected["start_frame"],
            "end_frame": expected["end_frame"],
            "source_input_sha256": expected["input_sha256"],
            "materialized_manifest_path": materialized_path,
            "native_manifest_path": native_path,
        }
        bindings[f"{visit_name}_materialized_manifest"] = materialized_record
        bindings[f"{visit_name}_native_manifest"] = native_record
    return {
        "protocol": protocol,
        "method_input": method_input,
        "protocol_content_sha256": protocol_content_sha256(protocol),
        "visits": visits,
        "input_bindings": bindings,
    }


def _protocol_bound_path(
    record: object, *, label: str
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError(f"{label} binding schema is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path or ".." in Path(raw_path).parts:
        raise ValueError(f"{label} binding path is invalid")
    path = Path(raw_path)
    absolute = path if path.is_absolute() else REPO_ROOT / path
    observed_absolute = _absolute_record(absolute)
    observed = {**observed_absolute, "path": raw_path}
    if observed != dict(record):
        raise ValueError(f"{label} binding mismatch")
    return absolute, observed


def _load_vocabulary(path: Path, *, scene: str) -> SemanticVocabulary:
    payload = _load_json(path, label="OVI semantic vocabulary")
    classes = payload.get("classes")
    class_semantic_ids = payload.get("class_semantic_ids")
    object_semantic_ids = payload.get("object_semantic_ids")
    if (
        payload.get("schema_version") != 2
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("scene") != scene
        or not isinstance(classes, list)
        or not classes
        or any(not isinstance(value, str) or not value.strip() for value in classes)
        or not isinstance(class_semantic_ids, list)
        or len(class_semantic_ids) != len(classes)
        or any(type(value) is not int for value in class_semantic_ids)
        or class_semantic_ids != list(range(1, len(classes) + 1))
        or not isinstance(object_semantic_ids, list)
        or not object_semantic_ids
        or any(type(value) is not int for value in object_semantic_ids)
        or len(object_semantic_ids) != len(set(object_semantic_ids))
        or not set(object_semantic_ids) <= set(class_semantic_ids)
        or payload.get("unknown_semantic_id") != 0
    ):
        raise ValueError("OVI semantic vocabulary identity is invalid")
    normalized = tuple(value.strip() for value in classes)
    if len(normalized) != len(set(normalized)):
        raise ValueError("OVI semantic vocabulary classes must be unique")
    return SemanticVocabulary(
        scene=scene,
        classes=normalized,
        class_semantic_ids=tuple(class_semantic_ids),
        object_semantic_ids=frozenset(object_semantic_ids),
    )


def _siglip_semantic_labeler(
    *,
    classes: tuple[str, ...],
    model_path: Path,
    device: str,
    canonical_prompts: tuple[str, ...],
    maximum_text_length: int,
    minimum_observation_count: int,
) -> SemanticLabeler:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not device:
        raise ValueError("semantic device must be non-empty")
    model = (
        AutoModel.from_pretrained(str(model_path), local_files_only=True)
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prompts = (*classes, *canonical_prompts)
    tokens = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=maximum_text_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        features = model.get_text_features(**tokens).float().cpu().numpy()
    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    split = len(classes)
    return make_relative_semantic_labeler(
        classes=classes,
        text_features=features[:split],
        canonical_features=features[split:],
        minimum_observation_count=minimum_observation_count,
    )


def _semantic_config(
    matrix: Mapping[str, Any], vocabulary_path: Path, *, scene: str
) -> tuple[tuple[str, ...], int, int]:
    raw = matrix.get("method_config", {}).get("ovi_semantics")
    if not isinstance(raw, Mapping):
        raise TypeError("matrix lacks frozen OVI semantic settings")
    declared_paths = raw.get("vocabulary_paths")
    if (
        scene not in {"apartment", "office"}
        or not isinstance(declared_paths, Mapping)
        or set(declared_paths) != {"apartment", "office"}
        or not isinstance(declared_paths.get(scene), str)
    ):
        raise TypeError("OVI semantic vocabulary path is invalid")
    declared_path = declared_paths[scene]
    expected_path = Path(os.path.abspath(os.fspath(REPO_ROOT / declared_path)))
    observed_path = Path(os.path.abspath(os.fspath(vocabulary_path)))
    if observed_path != expected_path:
        raise ValueError("OVI semantic vocabulary path differs from frozen matrix")
    prompts = raw.get("canonical_prompts")
    maximum_text_length = raw.get("maximum_text_length")
    minimum_observation_count = raw.get("minimum_observation_count")
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(value, str) or not value for value in prompts)
        or type(maximum_text_length) is not int
        or maximum_text_length < 1
        or type(minimum_observation_count) is not int
        or minimum_observation_count < 1
    ):
        raise ValueError("OVI semantic settings are invalid")
    return tuple(prompts), maximum_text_length, minimum_observation_count


def _resolve_scene_semantic_paths(
    scene: str,
    *,
    vocabulary_path: Path | None,
    label_space_path: Path | None,
) -> tuple[Path, Path]:
    if scene not in {"apartment", "office"}:
        raise ValueError("scene must be apartment or office")
    vocabulary = vocabulary_path or (
        REPO_ROOT
        / "configs/evaluation/vocabularies"
        / f"tesse_cd_{scene}_ovi_full.json"
    )
    label_space = label_space_path or Path(
        "/home/ww/oviovo_benchmark_assets/tesse_cd/derived/"
        f"common_v2_label_spaces/tesse_cd_{scene}_label_space.yaml"
    )
    return vocabulary, label_space


def _visibility_config(matrix: Mapping[str, Any]) -> SignedVisibilityConfig:
    raw = matrix.get("method_config", {}).get("signed_visibility")
    if not isinstance(raw, Mapping):
        raise TypeError("matrix lacks frozen signed-visibility settings")
    return SignedVisibilityConfig(
        voxel_size_m=raw["voxel_size_m"],
        depth_tolerance_m=raw["depth_tolerance_m"],
        depth_max_m=raw["depth_max_m"],
        minimum_absent_fraction=raw["minimum_absent_fraction"],
        minimum_absent_observations=raw["minimum_absent_observations"],
        minimum_distinct_viewpoints=raw["minimum_distinct_viewpoints"],
        minimum_viewpoint_baseline_m=raw["minimum_viewpoint_baseline_m"],
    )


def _load_t1_frames(
    *,
    rgbd_root: Path,
    scene: str,
    schedule_path: Path,
    start_frame: int,
    end_frame: int,
) -> tuple[Any, ...]:
    dataset = TesseCdRgbdDataset(
        rgbd_root,
        scene,
        rgbd_root / scene / "export_manifest.json",
        schedule_path,
    )
    return tuple(dataset[index] for index in range(start_frame, end_frame + 1))


def _split_t0_observation_masks(
    t0: VisitMap,
    frames: tuple[Any, ...],
    config: SignedVisibilityConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    chunks = [
        np.asarray(entity.points_xyz, dtype=np.float32)
        for entity in t0.snapshot.entities
    ]
    if t0.snapshot.background_xyz is not None:
        chunks.append(np.asarray(t0.snapshot.background_xyz, dtype=np.float32))
    all_points = (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)
    )
    observed = derive_observed_point_mask(all_points, frames, config)
    offset = 0
    entity_masks: dict[str, np.ndarray] = {}
    for entity in t0.snapshot.entities:
        stop = offset + len(entity.points_xyz)
        entity_masks[entity.entity_id] = observed[offset:stop]
        offset = stop
    background_count = (
        0 if t0.snapshot.background_xyz is None else len(t0.snapshot.background_xyz)
    )
    background = observed[offset : offset + background_count]
    if offset + background_count != len(observed):
        raise ValueError("t0 observation mask split did not conserve points")
    return entity_masks, background


def _artifact_bytes(native_manifest: Path) -> int:
    payload = _load_json(native_manifest, label="native OVI manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("native OVI manifest lacks artifacts")
    total = native_manifest.stat().st_size
    for record in artifacts.values():
        if not isinstance(record, Mapping) or type(record.get("byte_count")) is not int:
            raise ValueError("native OVI artifact byte count is invalid")
        total += int(record["byte_count"])
    return total


def _evaluation_context(
    *,
    protocol: Mapping[str, Any],
    scene: str,
    final_frame: int,
    schedule_path: Path,
    target_manifest_path: Path,
    frames: tuple[Any, ...],
    visibility_config: SignedVisibilityConfig,
) -> tuple[TwoVisitEvaluationContext, dict[str, Mapping[str, object]]]:
    schedule_record = _common_file_record(schedule_path)
    events, common = _load_schedule(schedule_path, scene=scene)
    arrays, arrays_record, _metadata = _load_targets(
        target_manifest_path,
        schedule_record=schedule_record,
    )
    scene_record = protocol["scenes"][scene]
    selected_id = scene_record["selected_candidate_id"]
    selected = next(
        candidate
        for candidate in scene_record["candidates"]
        if candidate["candidate_id"] == selected_id
    )
    event_ids = selected["evaluator_only"]["event_ids"]
    if not isinstance(event_ids, list) or len(event_ids) != 1:
        raise ValueError(
            "two-visit final checkpoint must bind exactly one change event"
        )
    event_id = event_ids[0]
    event = next(item for item in events if item["event_id"] == event_id)
    checkpoint_frames = event["common_checkpoint_frame_indices"]
    if final_frame not in common or final_frame not in checkpoint_frames:
        raise ValueError("t1 final frame is not a common-v2 event checkpoint")
    current_name = f"{scene}.current_semantic.{final_frame:06d}"
    region_name = f"{event_id}.region"
    free_name = f"{event_id}.confirmed_free.{final_frame:06d}"
    background_name = f"{event_id}.revealed_background"
    required = {current_name, region_name, free_name, background_name}
    if not required <= set(arrays):
        raise ValueError("common-v2 target package lacks two-visit final arrays")
    target_points = _voxel_centers(arrays[current_name][:, :3])
    target_observed = derive_observed_point_mask(
        target_points,
        frames,
        visibility_config,
    )
    context = TwoVisitEvaluationContext(
        event_id=event_id,
        frame_id=final_frame,
        intervention_frame_id=int(event["intervention_frame_index"]),
        current_semantic_voxels=arrays[current_name],
        changed_region_voxels=arrays[region_name],
        confirmed_free_voxels=arrays[free_name],
        revealed_background_voxels=arrays[background_name],
        current_target_t1_unobserved_mask=~target_observed,
    )
    return context, {
        "causal_schedule": schedule_record,
        "common_v2_target_manifest": _common_file_record(target_manifest_path),
        "common_v2_targets": arrays_record,
    }


def prepare_two_visit_execution(
    *,
    matrix: Mapping[str, Any],
    matrix_path: Path,
    protocol_path: Path,
    two_visit_ovi_manifest: Path,
    rgbd_root: Path,
    vocabulary_path: Path,
    siglip_model: Path,
    semantic_device: str,
    aliases_path: Path,
    label_space_path: Path,
    scene: str = "apartment",
) -> PreparedTwoVisitExecution:
    """Prepare method inputs first, then open evaluator-only target evidence."""

    loaded = load_two_visit_ovi_inputs(
        protocol_path=protocol_path,
        two_visit_ovi_manifest=two_visit_ovi_manifest,
        scene=scene,
    )
    protocol = loaded["protocol"]
    if (
        matrix["execution_contract"]["protocol_content_sha256"]
        != loaded["protocol_content_sha256"]
    ):
        raise ValueError("matrix and OVI inputs bind different two-visit protocols")
    source_bindings = protocol.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise TypeError("protocol source bindings are invalid")
    schedule_path, schedule_record = _protocol_bound_path(
        source_bindings.get("causal_schedule"), label="causal schedule"
    )
    target_manifest_path, target_manifest_record = _protocol_bound_path(
        source_bindings.get("common_v2_target_manifest"),
        label="common-v2 target manifest",
    )
    target_arrays_path, target_arrays_record = _protocol_bound_path(
        source_bindings.get("common_v2_targets"),
        label="common-v2 target arrays",
    )
    vocabulary = _load_vocabulary(vocabulary_path, scene=scene)
    canonical_prompts, maximum_text_length, minimum_observation_count = (
        _semantic_config(matrix, vocabulary_path, scene=scene)
    )
    directory_record = _directory_record(siglip_model, label="SigLIP model")
    for visit_name in ("t0", "t1"):
        native = _load_json(
            loaded["visits"][visit_name]["native_manifest_path"],
            label=f"{visit_name} native OVI manifest",
        )
        preflight = native.get("preflight")
        if (
            not isinstance(preflight, Mapping)
            or preflight.get("sources", {}).get("siglip_model") != directory_record
        ):
            raise ValueError("semantic model differs from native OVI source binding")
    semantic_labeler = _siglip_semantic_labeler(
        classes=vocabulary.classes,
        model_path=siglip_model,
        device=semantic_device,
        canonical_prompts=canonical_prompts,
        maximum_text_length=maximum_text_length,
        minimum_observation_count=minimum_observation_count,
    )
    visits: dict[str, VisitMap] = {}
    for visit_id, visit_name in enumerate(("t0", "t1")):
        record = loaded["visits"][visit_name]
        visits[visit_name] = load_ovimap_visit(
            native_manifest=record["native_manifest_path"],
            materialized_manifest=record["materialized_manifest_path"],
            visit_id=visit_id,
            scene=scene,
            coordinate_frame_id="tesse_cd_world",
            source_manifest_sha256=loaded["protocol_content_sha256"],
            observed_frame_start=int(record["start_frame"]),
            observed_frame_end=int(record["end_frame"]),
            semantic_labeler=semantic_labeler,
        )
    t0 = visits["t0"]
    t1 = visits["t1"]
    visibility_config = _visibility_config(matrix)
    t1_frames = _load_t1_frames(
        rgbd_root=rgbd_root,
        scene=scene,
        schedule_path=schedule_path,
        start_frame=t1.observed_frame_start,
        end_frame=t1.observed_frame_end,
    )
    visibility_source_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "t1_source_input_sha256": loaded["visits"]["t1"]["source_input_sha256"],
                "signed_visibility": asdict(visibility_config),
            }
        )
    ).hexdigest()
    visibility = derive_signed_visibility(
        t0,
        t1_frames,
        visibility_config,
        source_sha256=visibility_source_sha256,
    )
    t0_entity_masks, t0_background_mask = _split_t0_observation_masks(
        t0,
        t1_frames,
        visibility_config,
    )

    evaluation, evaluation_bindings = _evaluation_context(
        protocol=protocol,
        scene=scene,
        final_frame=t1.observed_frame_end,
        schedule_path=schedule_path,
        target_manifest_path=target_manifest_path,
        frames=t1_frames,
        visibility_config=visibility_config,
    )
    if (
        evaluation_bindings["causal_schedule"]
        != {**schedule_record, "path": str(schedule_path)}
        or evaluation_bindings["common_v2_target_manifest"]
        != {**target_manifest_record, "path": str(target_manifest_path)}
        or evaluation_bindings["common_v2_targets"]
        != {**target_arrays_record, "path": str(target_arrays_path)}
    ):
        raise ValueError("evaluator target bindings differ from frozen protocol")
    crosswalk = load_tesse_semantic_crosswalk(
        aliases_path,
        scene,
        label_space_path,
    )
    method_bindings = {
        role: dict(record) for role, record in loaded["input_bindings"].items()
    }
    method_bindings.update(
        {
            "matrix": _absolute_record(matrix_path),
            "semantic_vocabulary": _absolute_record(vocabulary_path),
            "siglip_model": directory_record,
        }
    )
    evaluation_bindings.update(
        {
            "semantic_aliases": _absolute_record(aliases_path),
            "semantic_label_space": _absolute_record(label_space_path),
        }
    )
    return PreparedTwoVisitExecution(
        t0=t0,
        t1=t1,
        visibility=visibility,
        evaluation=evaluation,
        crosswalk=crosswalk,
        object_semantic_labels=vocabulary.object_classes,
        t0_entity_observed_masks=t0_entity_masks,
        t0_background_observed_mask=t0_background_mask,
        ovi_t0_artifact_bytes=_artifact_bytes(
            loaded["visits"]["t0"]["native_manifest_path"]
        ),
        ovi_t1_artifact_bytes=_artifact_bytes(
            loaded["visits"]["t1"]["native_manifest_path"]
        ),
        input_bindings=method_bindings,
        evaluation_bindings=evaluation_bindings,
    )


def _geometric_config(matrix: Mapping[str, Any]) -> GeometricReasonerConfig:
    raw = matrix["method_config"]["geometric_reasoner"]
    weights = raw["weights"]
    return GeometricReasonerConfig(
        comparison_voxel_size_m=float(raw["comparison_voxel_size_m"]),
        maximum_centroid_distance_m=float(raw["maximum_centroid_distance_m"]),
        minimum_pair_score=float(raw["minimum_pair_score"]),
        require_known_label_compatibility=bool(
            raw["require_known_label_compatibility"]
        ),
        voxel_iou_weight=float(weights["voxel_iou"]),
        symmetric_coverage_weight=float(weights["symmetric_coverage"]),
        centroid_weight=float(weights["centroid"]),
        size_weight=float(weights["size"]),
        extent_weight=float(weights["extent"]),
        semantic_weight=float(weights["semantic"]),
    )


def _retained_t0(
    prepared: PreparedTwoVisitExecution,
    *,
    variant_id: str,
    current: TwoVisitCurrentMap | None,
) -> tuple[np.ndarray, np.ndarray]:
    if variant_id == "B2":
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.bool_),
        )
    points: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    if current is None:
        for entity in prepared.t0.snapshot.entities:
            points.append(np.asarray(entity.points_xyz, dtype=np.float32))
            masks.append(prepared.t0_entity_observed_masks[entity.entity_id])
        if prepared.t0.snapshot.background_xyz is not None:
            points.append(
                np.asarray(prepared.t0.snapshot.background_xyz, dtype=np.float32)
            )
            masks.append(prepared.t0_background_observed_mask)
    else:
        entities = {
            entity.entity_id: entity for entity in prepared.t0.snapshot.entities
        }
        for group in current.provenance:
            if group.decision.source_visit != 0 or group.output_point_count == 0:
                continue
            source_id = group.decision.source_entity_id
            if source_id == "__background__":
                source_points = prepared.t0.snapshot.background_xyz
                source_mask = prepared.t0_background_observed_mask
                if source_points is None:
                    raise ValueError("provenance references absent t0 background")
            else:
                if source_id not in entities:
                    raise ValueError("provenance references unknown t0 entity")
                source_points = entities[source_id].points_xyz
                source_mask = prepared.t0_entity_observed_masks[source_id]
            indices = group.source_point_indices
            points.append(np.asarray(source_points[indices], dtype=np.float32))
            masks.append(np.asarray(source_mask[indices], dtype=np.bool_))
    return (
        np.concatenate(points, axis=0)
        if points
        else np.empty((0, 3), dtype=np.float32),
        np.concatenate(masks) if masks else np.empty((0,), dtype=np.bool_),
    )


def _write_snapshot_artifact(
    *,
    prepared: PreparedTwoVisitExecution,
    variant_id: str,
    row: Mapping[str, Any],
    snapshot: MapSnapshot,
    current: TwoVisitCurrentMap | None,
    variant_root: Path,
) -> Path:
    bundle = variant_root / "current-map"
    if current is None:
        paths = write_map_snapshot(snapshot, bundle)
        artifact_records: dict[str, object] = {
            name: _record(path, root=variant_root) for name, path in paths.items()
        }
        artifact_records["provenance_manifest"] = None
    else:
        inner_manifest = write_two_visit_current_map(current, bundle)
        inner = json.loads(inner_manifest.read_text(encoding="utf-8"))
        inner_artifacts = inner["artifacts"]
        artifact_records = {
            name: _record(bundle / record["path"], root=variant_root)
            for name, record in inner_artifacts.items()
        }
        artifact_records["provenance_manifest"] = _record(
            inner_manifest, root=variant_root
        )
    output = variant_root / "current-map-manifest.json"
    _write_atomic(
        output,
        {
            "schema_version": 1,
            "status": "PASS",
            "variant_id": variant_id,
            "method": snapshot.method,
            "scene": snapshot.scene_id,
            "timestamp": snapshot.timestamp,
            "snapshot_sha256": snapshot_content_sha256(snapshot),
            "source_visit_map_sha256": [
                prepared.t0.snapshot_sha256,
                prepared.t1.snapshot_sha256,
            ],
            "visibility_source_sha256": (
                prepared.visibility.source_sha256
                if bool(row["signed_visibility"])
                else None
            ),
            "geometry_sources": list(row["final_geometry_sources"]),
            "input_bindings": {
                role: dict(record) for role, record in prepared.input_bindings.items()
            },
            "artifacts": artifact_records,
        },
    )
    return output


def _snapshot_point_count(snapshot: MapSnapshot) -> int:
    return sum(len(entity.points_xyz) for entity in snapshot.entities) + (
        0 if snapshot.background_xyz is None else len(snapshot.background_xyz)
    )


def _visit_point_count(visit: VisitMap) -> int:
    return _snapshot_point_count(visit.snapshot)


def _metric_payload(
    *,
    prepared: PreparedTwoVisitExecution,
    matrix: Mapping[str, Any],
    variant_id: str,
    run_id: str,
    snapshot: MapSnapshot,
    retained_t0_xyz: np.ndarray,
    retained_t0_mask: np.ndarray,
    output_artifact: Path,
    variant_root: Path,
    adapter_runtime_s: float,
) -> dict[str, object]:
    groups = {
        group: {name: None for name in names}
        for group, names in matrix["metric_contract"]["groups"].items()
    }
    measured = evaluate_two_visit_snapshot(
        snapshot,
        prepared.evaluation,
        prepared.crosswalk,
        retained_t0_xyz=retained_t0_xyz,
        retained_t0_t1_observed_mask=retained_t0_mask,
    )
    groups["current_state"].update(
        {name: measured[name] for name in ("ghost", "current_miou")}
    )
    groups["geometry"].update({name: measured[name] for name in groups["geometry"]})
    groups["systems"].update(
        {
            "ovi_t0_point_count": _visit_point_count(prepared.t0),
            "ovi_t1_point_count": _visit_point_count(prepared.t1),
            "ovi_t0_entity_count": len(prepared.t0.snapshot.entities),
            "ovi_t1_entity_count": len(prepared.t1.snapshot.entities),
            "adapter_runtime_s": adapter_runtime_s,
            "final_map_point_count": _snapshot_point_count(snapshot),
        }
    )
    groups["artifacts"].update(
        {
            "ovi_t0_artifact_bytes": prepared.ovi_t0_artifact_bytes,
            "ovi_t1_artifact_bytes": prepared.ovi_t1_artifact_bytes,
            "final_map_artifact_bytes": sum(
                path.stat().st_size
                for path in variant_root.rglob("*")
                if path.is_file()
            ),
        }
    )
    reasons = {
        "object_f1": "no_protocol_compatible_two_visit_object_tracking_csv",
        "change_f1": "no_protocol_compatible_two_visit_change_tracking_csv",
        "dynamic_surface_f1": "two_visit_final_snapshot_has_no_dynamic_surface_sequence",
        "persistent_identity_precision": "tesse_two_visit_identity_ground_truth_unavailable",
        "persistent_identity_recall": "tesse_two_visit_identity_ground_truth_unavailable",
        "moved_association_accuracy": "tesse_two_visit_identity_ground_truth_unavailable",
        "appeared_precision": "tesse_two_visit_identity_ground_truth_unavailable",
        "appeared_recall": "tesse_two_visit_identity_ground_truth_unavailable",
        "removed_precision": "tesse_two_visit_identity_ground_truth_unavailable",
        "removed_recall": "tesse_two_visit_identity_ground_truth_unavailable",
        "split_accuracy": "tesse_two_visit_identity_ground_truth_unavailable",
        "merge_accuracy": "tesse_two_visit_identity_ground_truth_unavailable",
        "persistent_semantic_stability": "identity_ground_truth_unavailable",
        "semantic_error_rate": "metric_not_defined_by_frozen_common_v2_protocol",
        "rescene_t0_token_count": "rescene_checkpoint_unavailable_and_backend_not_run",
        "rescene_t1_token_count": "rescene_checkpoint_unavailable_and_backend_not_run",
        "rescene_runtime_s": "rescene_checkpoint_unavailable_and_backend_not_run",
        "peak_gpu_memory_bytes": "postprocessing_peak_gpu_memory_not_instrumented",
        "adapter_artifact_bytes": (
            "geometric_pair_tokens_not_serialized"
            if variant_id == "B4"
            else "variant_does_not_produce_adapter_artifact"
        ),
    }
    unavailable = {
        name: reasons[name]
        for values in groups.values()
        for name, value in values.items()
        if value is None
    }
    if set(unavailable) != {
        name
        for values in groups.values()
        for name, value in values.items()
        if value is None
    }:
        raise ValueError("metric N/A reasons are incomplete")
    if output_artifact != variant_root / "current-map-manifest.json":
        raise ValueError("output artifact path differs from frozen contract")
    return {
        "schema_version": 1,
        "status": "PASS",
        "variant_id": variant_id,
        "run_id": run_id,
        "metric_groups": groups,
        "unavailable": unavailable,
        "method_input_bindings": {
            role: dict(record) for role, record in prepared.input_bindings.items()
        },
        "evaluation_only_bindings": {
            role: dict(record) for role, record in prepared.evaluation_bindings.items()
        },
    }


def _enforce_b2_floor(
    metric_payload: Mapping[str, Any], matrix: Mapping[str, Any]
) -> None:
    try:
        ghost = metric_payload["metric_groups"]["current_state"]["ghost"]
        maximum = matrix["success_gates"]["practical_ghost_max"]
    except (KeyError, TypeError) as error:
        raise ValueError("B2 Ghost floor receipt is incomplete") from error
    if (
        isinstance(ghost, bool)
        or not isinstance(ghost, (int, float))
        or not np.isfinite(float(ghost))
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not np.isfinite(float(maximum))
    ):
        raise ValueError("B2 Ghost floor receipt is invalid")
    if float(ghost) > float(maximum):
        raise ValueError(
            f"B2 Ghost floor {float(ghost):.6f} exceeds frozen practical "
            f"maximum {float(maximum):.6f}; diagnose OVI t1/evaluator alignment"
        )


def execute_prepared_variant(
    prepared: PreparedTwoVisitExecution,
    matrix: Mapping[str, Any],
    row: Mapping[str, Any],
    run_id: str,
    variant_root: Path,
    *,
    command_line: tuple[str, ...],
) -> VariantExecutionArtifacts:
    """Execute one B0-B4 row from immutable shared inputs."""

    if not isinstance(prepared, PreparedTwoVisitExecution):
        raise TypeError("prepared must be PreparedTwoVisitExecution")
    variant_id = str(row.get("id"))
    if variant_id not in {"B0", "B1", "B2", "B3", "B4"}:
        raise ValueError("prepared executor supports only required B0-B4 variants")
    if Path(variant_root).resolve() != Path(variant_root).absolute():
        raise ValueError("variant_root must not traverse symlinks")
    before = (prepared.t0.snapshot_sha256, prepared.t1.snapshot_sha256)
    current: TwoVisitCurrentMap | None = None
    adapter_runtime_s = 0.0
    if variant_id in {"B0", "B1", "B2"}:
        snapshot = build_static_baseline_snapshot(variant_id, prepared.t0, prepared.t1)
    else:
        started = time.perf_counter()
        composer = matrix["method_config"]["composer"]
        current, _relations = build_visibility_baseline(
            prepared.t0,
            prepared.t1,
            prepared.visibility,
            use_geometric_pairing=variant_id == "B4",
            geometric_config=_geometric_config(matrix),
            object_semantic_labels=prepared.object_semantic_labels,
            minimum_entity_visible_free_fraction=float(
                composer["minimum_entity_visible_free_fraction"]
            ),
            minimum_entity_visible_free_voxels=int(
                composer["minimum_entity_visible_free_voxels"]
            ),
        )
        adapter_runtime_s = time.perf_counter() - started if variant_id == "B4" else 0.0
        snapshot = current.snapshot
    retained, retained_mask = _retained_t0(
        prepared,
        variant_id=variant_id,
        current=current,
    )
    output = _write_snapshot_artifact(
        prepared=prepared,
        variant_id=variant_id,
        row=row,
        snapshot=snapshot,
        current=current,
        variant_root=variant_root,
    )
    metrics = _metric_payload(
        prepared=prepared,
        matrix=matrix,
        variant_id=variant_id,
        run_id=run_id,
        snapshot=snapshot,
        retained_t0_xyz=retained,
        retained_t0_mask=retained_mask,
        output_artifact=output,
        variant_root=variant_root,
        adapter_runtime_s=adapter_runtime_s,
    )
    metric_path = variant_root / "metrics.json"
    _write_atomic(metric_path, metrics)
    stdout = variant_root / "stdout.log"
    stderr = variant_root / "stderr.log"
    stdout.write_text(
        json.dumps(
            {
                "status": "PASS",
                "variant_id": variant_id,
                "snapshot_sha256": snapshot_content_sha256(snapshot),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")
    after = (
        snapshot_content_sha256(prepared.t0.snapshot),
        snapshot_content_sha256(prepared.t1.snapshot),
    )
    if after != before:
        raise ValueError("source OVI visits changed during variant execution")
    return VariantExecutionArtifacts(
        command_line=command_line,
        stdout_log=stdout,
        stderr_log=stderr,
        output_artifact=output,
        metric_receipt=metric_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/ovi_rescene_two_visit_matrix.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/tesse_two_visit_current_v1.json",
    )
    parser.add_argument("--two-visit-ovi-manifest", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=("apartment", "office"),
        default="apartment",
    )
    parser.add_argument(
        "--rgbd-root",
        type=Path,
        default=Path("/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1"),
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--siglip-model",
        type=Path,
        default=Path(
            "/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/"
            "siglip-large-patch16-384"
        ),
    )
    parser.add_argument("--semantic-device", default="cuda:1")
    parser.add_argument(
        "--aliases",
        type=Path,
        default=REPO_ROOT
        / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml",
    )
    parser.add_argument(
        "--label-space",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    vocabulary_path, label_space_path = _resolve_scene_semantic_paths(
        args.scene,
        vocabulary_path=args.vocabulary,
        label_space_path=args.label_space,
    )
    matrix = load_and_validate_matrix(args.matrix)
    if args.scene != "apartment":
        raise MatrixError(str(matrix["office_policy"]["status"]))
    prepared = prepare_two_visit_execution(
        matrix=matrix,
        matrix_path=args.matrix,
        protocol_path=args.protocol,
        two_visit_ovi_manifest=args.two_visit_ovi_manifest,
        rgbd_root=args.rgbd_root,
        vocabulary_path=vocabulary_path,
        siglip_model=args.siglip_model,
        semantic_device=args.semantic_device,
        aliases_path=args.aliases,
        label_space_path=label_space_path,
        scene=args.scene,
    )
    source_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    invocation = (
        sys.executable,
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
    )

    def execute(
        row: Mapping[str, Any], run_id: str, variant_root: Path
    ) -> VariantExecutionArtifacts:
        artifacts = execute_prepared_variant(
            prepared,
            matrix,
            row,
            run_id,
            variant_root,
            command_line=tuple(invocation),
        )
        if row["id"] == "B2":
            _enforce_b2_floor(
                _load_json(artifacts.metric_receipt, label="B2 metric receipt"),
                matrix,
            )
        return artifacts

    summary = execute_required_matrix(
        matrix_path=args.matrix,
        scene=args.scene,
        output_root=args.output_root,
        source_commit=source_commit,
        executor=execute,
    )
    print(summary)
    return 0


__all__ = [
    "PreparedTwoVisitExecution",
    "execute_prepared_variant",
    "load_two_visit_ovi_inputs",
    "main",
    "prepare_two_visit_execution",
]


if __name__ == "__main__":
    raise SystemExit(main())
