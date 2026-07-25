#!/usr/bin/env python3
"""Run the OVIV2 dual cumulative/current readout on one TESSE-CD scene."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_tesse_occlusion import (  # noqa: E402
    RUNNER_SCENE_CONFIG_FIELDS,
    canonical_algorithm_config,
    canonical_algorithm_hash,
)
from scripts.evaluation.run_oviv2_tesse_cd import (  # noqa: E402
    TesseCausalCheckpoint,
    _absolute_lexical,
    _byte_record,
    _checkpoint_plan_from_target_manifest,
    _is_sha256,
    _json_hash,
    _load_causal_checkpoints_bytes,
    _load_json_bytes,
    _production_cache_loader_factory,
    _production_dataset_factory,
    _production_provenance,
    _publish_run,
    _revalidate_frozen_bindings,
    _reject_symlink_components,
    _repository_provenance,
    _require_regular_file,
    _resolve_path,
    _run_execution,
    _sha256,
    _sha256_bytes,
    _tree_record,
    _validate_repository_state,
    _verify_frozen_file_binding,
    apply_frozen_visibility_policy,
)
from src.evaluation.oviv2_temporal_tesse import (  # noqa: E402
    TEMPORAL_CURRENT_FORMAT,
    publish_temporal_current_checkpoint,
)
from src.oviv2.temporal_snapshot import (  # noqa: E402
    TEMPORAL_COMPACT_FORMAT,
    TemporalCompactCheckpoint,
    TemporalCurrentSnapshot,
    TemporalSnapshotMetadata,
)


PROTOCOL_ID = "oviv2-tessecd-v2"
SCENE_CONFIG_FIELDS = RUNNER_SCENE_CONFIG_FIELDS


@dataclass(frozen=True)
class RunnerDependencies:
    dataset_factory: Callable[[Mapping[str, Any]], Any]
    cache_loader_factory: Callable[[Mapping[str, Any], Any], Any]
    runtime_factory: Callable[[Mapping[str, Any], Any], Any]
    provenance_factory: Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class FrozenRunContext:
    manifest_path: Path
    manifest_bytes: bytes
    repository_state: Mapping[str, str]
    input_bindings: Mapping[str, Any]
    frozen_run_identity: Mapping[str, Any]
    run_slot: str


def algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_algorithm_config(config)


def algorithm_hash(config: Mapping[str, Any]) -> str:
    return canonical_algorithm_hash(config)


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[str, int, Path, tuple[int, ...], Any]:
    for key, value in config.items():
        if key == "structure_enabled":
            if type(value) is not bool:
                raise TypeError("structure_enabled must be an exact boolean")
        elif type(value) is bool:
            raise TypeError(f"{key} must not use a boolean as a numeric value")
    if config.get("schema_version") != 2:
        raise ValueError("v2 runner schema_version must be 2")
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or config.get("dataset") != "TESSE-CD"
        or config.get("method_id") != "OVIV2"
    ):
        raise ValueError("runner identity must be oviv2-tessecd-v2 on TESSE-CD")
    scene = config.get("scene")
    if scene not in {"apartment", "office"}:
        raise ValueError("scene must be apartment or office")
    frame_count = _positive_integer(config.get("frame_count"), "frame_count")
    if config.get("missing_observation_policy") not in {
        "signed_depth",
        "missing_as_absence",
    }:
        raise ValueError("missing_observation_policy is invalid")

    from src.oviv2.temporal_config import temporal_config_from_json

    temporal_config = temporal_config_from_json(
        {"temporal_readout": config.get("temporal_readout")}
    )
    if temporal_config.geometry.voxel_size_m != config.get("voxel_size_m"):
        raise ValueError("temporal voxel_size_m must match cumulative geometry")
    if temporal_config.geometry.depth_max_m != config.get("depth_max_m"):
        raise ValueError("temporal depth_max_m must match cumulative geometry")
    configured_hash = config.get("algorithm_hash")
    if not _is_sha256(configured_hash) or configured_hash != algorithm_hash(config):
        raise ValueError("configured algorithm_hash does not match mapping parameters")
    schedule = _resolve_path(config.get("schedule_manifest", ""))
    raw_frames = config.get("evaluation_checkpoint_frames")
    if not isinstance(raw_frames, list) or any(
        type(frame) is not int or not 0 <= frame < frame_count for frame in raw_frames
    ):
        raise ValueError("evaluation_checkpoint_frames must contain in-range integers")
    evaluation_frames = tuple(raw_frames)
    if list(evaluation_frames) != sorted(set(evaluation_frames)):
        raise ValueError("evaluation_checkpoint_frames must be sorted and unique")
    return str(scene), frame_count, schedule, evaluation_frames, temporal_config


def _load_frozen_run_context(
    freeze_manifest: str | Path,
    *,
    run_slot: str,
    source_config: Path,
    source_config_bytes: bytes,
    config: Mapping[str, Any],
    destination: Path,
) -> FrozenRunContext:
    manifest_path = _absolute_lexical(freeze_manifest)
    _require_regular_file(manifest_path, "freeze manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_bytes(manifest_bytes, manifest_path)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("freeze_id") == PROTOCOL_ID
        and manifest.get("status") == "FROZEN"
        and manifest.get("method") == "OVIV2"
        and manifest.get("dataset") == "TESSE-CD"
    ):
        raise ValueError(f"formal runner requires the FROZEN {PROTOCOL_ID} manifest")
    repository = manifest.get("repository")
    if not isinstance(repository, Mapping) or repository.get("clean") is not True:
        raise ValueError("freeze repository identity is invalid")
    current = _validate_repository_state(_repository_provenance(), repository)

    scene = str(config["scene"])
    scenes = manifest.get("scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != {"apartment", "office"}:
        raise ValueError("freeze scene bindings are incomplete")
    selected = scenes.get(scene)
    if not isinstance(selected, Mapping):
        raise ValueError("freeze scene binding is invalid")
    config_record = _verify_frozen_file_binding(
        selected.get("frozen_config"),
        base=manifest_path.parent,
        role=f"{scene} frozen config",
        expected_path=source_config,
    )
    if config_record != _byte_record(source_config_bytes):
        raise ValueError("frozen config content does not match runner config")
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, Mapping) or not (
        algorithm.get("sha256") == config["algorithm_hash"]
        and algorithm.get("normalized_config") == algorithm_config(config)
    ):
        raise ValueError("runner config algorithm differs from the freeze")

    output_roots = manifest.get("output_roots")
    expected_slots = {
        f"{slot_scene}_run{repeat}"
        for slot_scene in ("apartment", "office")
        for repeat in (1, 2)
    }
    if not isinstance(output_roots, Mapping) or set(output_roots) != expected_slots:
        raise ValueError("freeze output slots are incomplete")
    if run_slot not in expected_slots or not run_slot.startswith(f"{scene}_run"):
        raise ValueError("run slot does not match the frozen scene")
    raw_output = output_roots.get(run_slot)
    if not isinstance(raw_output, str) or not Path(raw_output).is_absolute():
        raise ValueError("frozen run slot output root is invalid")
    if raw_output != os.fspath(_absolute_lexical(raw_output)) or _absolute_lexical(raw_output) != destination:
        raise ValueError("run slot output does not match the frozen output root")
    normalized_roots = [_absolute_lexical(str(value)) for value in output_roots.values()]
    if len(set(normalized_roots)) != len(normalized_roots):
        raise ValueError("frozen output roots must be distinct")

    shared = manifest.get("shared_bindings")
    if not isinstance(shared, Mapping):
        raise ValueError("freeze shared input bindings are missing")
    _verify_frozen_file_binding(
        shared.get("schedule"),
        base=manifest_path.parent,
        role="shared schedule",
        expected_path=_resolve_path(config["schedule_manifest"]),
    )
    _verify_frozen_file_binding(
        shared.get("occlusion_target_manifest"),
        base=manifest_path.parent,
        role="shared occlusion target manifest",
        expected_path=_resolve_path(config["occlusion_target_manifest"]),
    )
    input_bindings = {"shared_bindings": dict(shared), "scene": dict(selected)}
    _revalidate_frozen_bindings(input_bindings, base=manifest_path.parent)
    identity = {
        "schema_version": 1,
        "freeze_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset": "TESSE-CD",
        "method_id": "OVIV2",
        "scene": scene,
        "freeze_manifest": _byte_record(manifest_bytes),
        "repository": {
            "commit": current["repository_commit"],
            "tree": current["repository_tree"],
        },
        "config": config_record,
        "algorithm_hash": algorithm["sha256"],
        "input_bindings_sha256": _json_hash(input_bindings),
    }
    return FrozenRunContext(
        manifest_path,
        manifest_bytes,
        current,
        input_bindings,
        identity,
        run_slot,
    )


def _production_runtime_factory(config: Mapping[str, Any], caches: Any) -> Any:
    from src.oviv2.dual_readout import DualReadoutRuntime
    from src.oviv2.runner_config import runtime_config_from_json
    from src.oviv2.runtime import Oviv2Runtime
    from src.oviv2.temporal_config import temporal_config_from_json
    from src.oviv2.temporal_runtime import TemporalCurrentRuntime

    runtime_config = apply_frozen_visibility_policy(
        runtime_config_from_json(dict(config)),
        str(config["missing_observation_policy"]),
    )
    temporal_config = temporal_config_from_json(
        {"temporal_readout": config["temporal_readout"]}
    )
    cumulative = Oviv2Runtime(
        str(config["scene"]),
        runtime_config,
        dense_semantic_provenance=caches.dense_provenance,
    )
    temporal = TemporalCurrentRuntime(
        str(config["scene"]),
        temporal_config,
        tracker_config=runtime_config.tracker,
    )
    return DualReadoutRuntime(cumulative, temporal)


def _production_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        dataset_factory=_production_dataset_factory,
        cache_loader_factory=_production_cache_loader_factory,
        runtime_factory=_production_runtime_factory,
        provenance_factory=_production_provenance,
    )


def _snapshot_from_runtime(
    runtime: Any,
    *,
    checkpoint: TesseCausalCheckpoint,
    scene: str,
    config_sha256: str,
    voxel_size_m: float,
) -> TemporalCurrentSnapshot:
    temporal = getattr(runtime, "temporal", None)
    state = getattr(temporal, "state", None)
    if state is None:
        raise ValueError("dual runtime does not expose temporal state")
    expected_timestamp = checkpoint.timestamp_ns / 1_000_000_000
    if not (
        state.scene_id == scene
        and state.last_frame_id == checkpoint.frame_index
        and state.revision == checkpoint.frame_index + 1
        and float(state.last_timestamp) == expected_timestamp
    ):
        raise ValueError("temporal state does not match checkpoint progress")
    metadata = TemporalSnapshotMetadata(
        scene_id=state.scene_id,
        frame_id=state.last_frame_id,
        timestamp=state.last_timestamp,
        revision=state.revision,
        voxel_size_m=voxel_size_m,
        config_sha256=config_sha256,
    )
    return TemporalCurrentSnapshot(metadata, state.entities, state.background)


def _input_sha256(
    source_config_bytes: bytes,
    schedule_bytes: bytes,
    target_bytes: bytes,
    cache_bindings: Mapping[str, Any],
) -> str:
    return _json_hash(
        {
            "config": _byte_record(source_config_bytes),
            "schedule": _byte_record(schedule_bytes),
            "target": _byte_record(target_bytes),
            "cache_bindings": dict(cache_bindings),
        }
    )


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    freeze_manifest: str | Path | None = None,
    run_slot: str | None = None,
    dependencies: RunnerDependencies | None = None,
) -> dict[str, Any]:
    if (freeze_manifest is None) != (run_slot is None):
        raise ValueError("freeze_manifest and run_slot must be provided together")
    source_config = Path(config_path).absolute()
    _require_regular_file(source_config, "runner config")
    source_config_bytes = source_config.read_bytes()
    config = _load_json_bytes(source_config_bytes, source_config)
    scene, frame_count, schedule_path, evaluation_frames, temporal_config = _validate_config(config)
    destination = _absolute_lexical(output)
    frozen = (
        _load_frozen_run_context(
            freeze_manifest,
            run_slot=str(run_slot),
            source_config=source_config,
            source_config_bytes=source_config_bytes,
            config=config,
            destination=destination,
        )
        if freeze_manifest is not None
        else None
    )

    _require_regular_file(schedule_path, "causal schedule")
    schedule_bytes = schedule_path.read_bytes()
    official = _load_causal_checkpoints_bytes(
        schedule_bytes, schedule_path, scene=scene, frame_count=frame_count
    )
    target_path = _resolve_path(config.get("occlusion_target_manifest", ""))
    target_hash = config.get("occlusion_target_manifest_sha256")
    plan_hash = config.get("evaluation_checkpoint_frames_sha256")
    if not _is_sha256(target_hash) or not _is_sha256(plan_hash):
        raise ValueError("occlusion target bindings are invalid")
    _require_regular_file(target_path, "occlusion target manifest")
    target_bytes = target_path.read_bytes()
    if _sha256_bytes(target_bytes) != target_hash:
        raise ValueError("occlusion target manifest checksum binding mismatch")
    plan = _checkpoint_plan_from_target_manifest(target_bytes, target_path)
    if (
        list(evaluation_frames) != plan["evaluation_checkpoint_frames"][scene]
        or plan_hash != plan["evaluation_checkpoint_frames_sha256"]
    ):
        raise ValueError("evaluation checkpoint frame binding mismatch")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, "output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    execution = (
        _run_execution(run_slot=frozen.run_slot, output=destination, staging=staging)
        if frozen is not None
        else None
    )
    try:
        dependencies = _production_dependencies() if dependencies is None else dependencies
        provenance = dict(dependencies.provenance_factory())
        code_commit = provenance.get("repository_commit")
        if not isinstance(code_commit, str) or len(code_commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in code_commit
        ):
            raise ValueError("provenance repository_commit is invalid")
        dataset = dependencies.dataset_factory(config)
        if len(dataset) != frame_count:
            raise ValueError("dataset frame count does not match runner config")
        caches = dependencies.cache_loader_factory(config, dataset)
        cache_bindings = getattr(caches, "bindings", {})
        if not isinstance(cache_bindings, Mapping):
            raise ValueError("cache bindings must be a mapping")
        input_sha256 = _input_sha256(
            source_config_bytes, schedule_bytes, target_bytes, cache_bindings
        )
        runtime_config = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "evaluation_checkpoint_frames",
                "evaluation_checkpoint_frames_sha256",
                "occlusion_target_manifest",
                "occlusion_target_manifest_sha256",
            }
        }
        runtime = dependencies.runtime_factory(runtime_config, caches)

        by_frame = {item.frame_index: item for item in official}
        first_timestamp_ns = int(dataset.timestamp_ns(0))
        for frame_index in evaluation_frames:
            timestamp_ns = int(dataset.timestamp_ns(frame_index))
            existing = by_frame.get(frame_index)
            if existing is None:
                by_frame[frame_index] = TesseCausalCheckpoint(
                    frame_index,
                    timestamp_ns,
                    timestamp_ns - first_timestamp_ns,
                    (),
                    ("occlusion_v1",),
                )
            elif "occlusion_v1" not in existing.roles:
                by_frame[frame_index] = TesseCausalCheckpoint(
                    existing.frame_index,
                    existing.timestamp_ns,
                    existing.relative_timestamp_ns,
                    existing.event_ids,
                    (*existing.roles, "occlusion_v1"),
                )
        checkpoints = tuple(by_frame[index] for index in sorted(by_frame))
        captured: list[int] = []
        records: list[dict[str, Any]] = []
        witnesses: list[Any] = []
        for frame_index in range(frame_count):
            frame = dataset[frame_index]
            if int(frame.frame_id) != frame_index:
                raise ValueError("dataset frame IDs must equal zero-based frame indices")
            observations, dense_semantics = caches.load(frame_index, frame)
            runtime.process_frame(
                frame,
                observations=observations,
                dense_semantics=dense_semantics,
            )
            checkpoint = by_frame.get(frame_index)
            if checkpoint is None:
                continue
            if int(dataset.timestamp_ns(frame_index)) != checkpoint.timestamp_ns:
                raise ValueError("checkpoint timestamp does not match dataset timestamp")
            snapshot = _snapshot_from_runtime(
                runtime,
                checkpoint=checkpoint,
                scene=scene,
                config_sha256=str(config["algorithm_hash"]),
                voxel_size_m=temporal_config.geometry.voxel_size_m,
            )
            root = staging / "checkpoints" / f"{frame_index:08d}-{checkpoint.timestamp_ns}"
            root.mkdir(parents=True)
            is_full = bool({"official", "common_v2"} & set(checkpoint.roles))
            if is_full:
                receipt = publish_temporal_current_checkpoint(
                    root / "temporal_current",
                    snapshot,
                    caches.class_names,
                    code_commit=code_commit,
                    input_sha256=input_sha256,
                )
                artifact_root = receipt.path
                witness = receipt.source_witness
                checkpoint_format = TEMPORAL_CURRENT_FORMAT
            else:
                if set(checkpoint.roles) != {"occlusion_v1"}:
                    raise ValueError("checkpoint role combination is invalid")
                compact = TemporalCompactCheckpoint.from_snapshot(
                    snapshot,
                    maximum_entities=temporal_config.geometry.maximum_entities,
                    maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
                ).commit_new(
                    root / "temporal_compact",
                    maximum_entities=temporal_config.geometry.maximum_entities,
                    maximum_object_voxels=temporal_config.geometry.maximum_object_voxels,
                )
                artifact_root = compact.path
                witness = compact
                checkpoint_format = TEMPORAL_COMPACT_FORMAT
            witness.revalidate_source() if hasattr(witness, "revalidate_source") else witness.revalidate()
            witnesses.append(witness)
            records.append(
                {
                    "scene": scene,
                    "frame_index": checkpoint.frame_index,
                    "timestamp_ns": checkpoint.timestamp_ns,
                    "relative_timestamp_ns": checkpoint.relative_timestamp_ns,
                    "consumed_through_frame": checkpoint.frame_index,
                    "consumed_through_frame_exclusive": checkpoint.frame_index + 1,
                    "event_ids": list(checkpoint.event_ids),
                    "roles": list(checkpoint.roles),
                    "format": checkpoint_format,
                    "artifact": _tree_record(artifact_root, relative_to=staging),
                    "checksums_sha256": _sha256(artifact_root / "checksums.json"),
                }
            )
            captured.append(frame_index)

        scheduled = [item.frame_index for item in checkpoints]
        if captured != scheduled:
            raise ValueError("captured checkpoints do not exactly match the schedule")
        expected_dirs = {
            f"{item.frame_index:08d}-{item.timestamp_ns}" for item in checkpoints
        }
        checkpoint_parent = staging / "checkpoints"
        actual_dirs = {item.name for item in checkpoint_parent.iterdir()}
        if actual_dirs != expected_dirs:
            raise ValueError("checkpoint directories do not exactly match the schedule")

        _require_regular_file(source_config, "runner config")
        _require_regular_file(schedule_path, "causal schedule")
        _require_regular_file(target_path, "occlusion target manifest")
        if source_config.read_bytes() != source_config_bytes:
            raise ValueError("runner config changed during run")
        if schedule_path.read_bytes() != schedule_bytes:
            raise ValueError("causal schedule changed during run")
        if target_path.read_bytes() != target_bytes:
            raise ValueError("occlusion target manifest changed during run")
        assert_unchanged = getattr(caches, "assert_inputs_unchanged", None)
        if callable(assert_unchanged):
            assert_unchanged()
        for witness in witnesses:
            witness.revalidate_source() if hasattr(witness, "revalidate_source") else witness.revalidate()

        normalized_config = staging / "normalized_run_config.json"
        normalized_config.write_bytes(
            (json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        formal_fields = (
            {
                "frozen_run_identity": dict(frozen.frozen_run_identity),
                "run_execution": dict(execution),
            }
            if frozen is not None and execution is not None
            else {}
        )
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "protocol_id": PROTOCOL_ID,
            "dataset": "TESSE-CD",
            "method_id": "OVIV2",
            "scene": scene,
            "mode": "dual_readout_causal_checkpoints",
            "algorithm_hash": config["algorithm_hash"],
            "processed_frame_count": frame_count,
            "scheduled_frame_indices": scheduled,
            "captured_frame_indices": captured,
            "config": _byte_record(source_config_bytes),
            "schedule": _byte_record(schedule_bytes),
            "target_manifest": _byte_record(target_bytes),
            "source_bindings": dict(cache_bindings),
            "input_sha256": input_sha256,
            "code_commit": code_commit,
            "checkpoints": records,
            **formal_fields,
        }
        (staging / "run_manifest.json").write_bytes(
            (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        (staging / "run_provenance.json").write_bytes(
            (json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        if frozen is not None:
            if frozen.manifest_path.read_bytes() != frozen.manifest_bytes:
                raise ValueError("freeze manifest changed during run")
            _revalidate_frozen_bindings(
                frozen.input_bindings, base=frozen.manifest_path.parent
            )
            if _validate_repository_state(
                _repository_provenance(),
                {
                    "commit": frozen.repository_state["repository_commit"],
                    "tree": frozen.repository_state["repository_tree"],
                },
            ) != frozen.repository_state:
                raise ValueError("repository identity changed during run")
        _publish_run(staging, destination)
        published = True
        return manifest
    except BaseException:
        if not published and staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument(
        "--run-slot",
        choices=(
            "apartment_run1",
            "apartment_run2",
            "office_run1",
            "office_run2",
        ),
    )
    args = parser.parse_args(argv)
    if (args.freeze_manifest is None) != (args.run_slot is None):
        parser.error("--freeze-manifest and --run-slot must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(
        args.config,
        args.output,
        freeze_manifest=args.freeze_manifest,
        run_slot=args.run_slot,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "processed_frame_count": manifest["processed_frame_count"],
                "scene": manifest["scene"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
