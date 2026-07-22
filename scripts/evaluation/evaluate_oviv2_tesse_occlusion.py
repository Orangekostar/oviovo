#!/usr/bin/env python3
"""Evaluate OVIV2 ownership retention on frozen TESSE-CD occlusion targets."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import numpy as np

from scripts.evaluation.derive_tesse_cd_occlusion_v1 import (
    REPO_ROOT,
    validate_generated_target,
)
from src.oviv2.snapshot import VoxelMapSnapshot


SnapshotKey = tuple[str, int]


class _LazyCheckpointSnapshots(Mapping[SnapshotKey, VoxelMapSnapshot]):
    def __init__(self, paths: Mapping[SnapshotKey, Path], *, cache_size: int = 1) -> None:
        self._paths = dict(paths)
        self._cache_size = cache_size
        self._cache: OrderedDict[SnapshotKey, VoxelMapSnapshot] = OrderedDict()

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[SnapshotKey]:
        return iter(self._paths)

    def __getitem__(self, key: SnapshotKey) -> VoxelMapSnapshot:
        if key not in self._paths:
            raise KeyError(key)
        cached = self._cache.pop(key, None)
        if cached is None:
            cached = VoxelMapSnapshot.load(self._paths[key])
        self._cache[key] = cached
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return cached


@dataclass(frozen=True)
class _FileWitness:
    path: Path
    fingerprint: tuple[int, int, int, int, int]
    sha256: str
    byte_count: int


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _fingerprint(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_bound_file(
    path: Path,
    *,
    label: str,
    capture_content: bool = True,
) -> tuple[bytes, _FileWitness]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is not a readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if capture_content:
                chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise ValueError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    return content, _FileWitness(
        path=path,
        fingerprint=_fingerprint(after),
        sha256=digest.hexdigest(),
        byte_count=byte_count,
    )


def _revalidate_witness(witness: _FileWitness, *, label: str) -> None:
    try:
        status = os.lstat(witness.path)
    except OSError as error:
        raise ValueError(f"{label} changed after validation") from error
    if not stat.S_ISREG(status.st_mode) or _fingerprint(status) != witness.fingerprint:
        raise ValueError(f"{label} changed after validation")


def _source_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("target source path is invalid")
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_target_package(
    target_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], _FileWitness, list[_FileWitness]]:
    manifest_content, manifest_witness = _read_bound_file(
        target_dir / "manifest.json", label="target manifest"
    )
    manifest = _load_json_bytes(manifest_content, label="target manifest")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("manifest_id") == "tesse_cd_occlusion_v1_targets"
        and manifest.get("dataset") == "TESSE-CD"
        and manifest.get("status") in {"FIXTURE", "SMOKE", "GENERATED"}
        and manifest.get("targets_generated") is True
        and manifest.get("prediction_inputs_used") is False
    ):
        raise ValueError("target manifest identity mismatch")

    array_record = manifest.get("target_arrays")
    if not isinstance(array_record, Mapping) or array_record.get("path") != "targets.npz":
        raise ValueError("target array binding is invalid")
    target_content, target_witness = _read_bound_file(
        target_dir / "targets.npz", label="target arrays"
    )
    if not (
        type(array_record.get("byte_count")) is int
        and array_record["byte_count"] == target_witness.byte_count
        and array_record.get("sha256") == target_witness.sha256
    ):
        raise ValueError("target array hash mismatch")
    try:
        with np.load(io.BytesIO(target_content), allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, ValueError) as error:
        raise ValueError("target arrays are not a valid safe NPZ") from error
    declared_arrays = array_record.get("arrays")
    if not isinstance(declared_arrays, Mapping) or set(declared_arrays) != set(arrays):
        raise ValueError("target array inventory mismatch")
    if array_record.get("count") != len(arrays):
        raise ValueError("target array count mismatch")
    for name, values in arrays.items():
        declaration = declared_arrays[name]
        if not isinstance(declaration, Mapping) or declaration != {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "element_count": int(values.size),
        }:
            raise ValueError(f"target array declaration mismatch: {name}")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("target metadata is missing")
    validate_generated_target(arrays, metadata)

    source_witnesses: list[_FileWitness] = []
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("target source bindings are missing")
    for role, raw_record in sorted(sources.items()):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise ValueError(f"target source binding is invalid: {role}")
        _, witness = _read_bound_file(
            _source_path(raw_record["path"]),
            label=f"target source {role}",
            capture_content=False,
        )
        if not (
            raw_record["sha256"] == witness.sha256
            and raw_record["byte_count"] == witness.byte_count
        ):
            raise ValueError(f"target source hash mismatch: {role}")
        source_witnesses.append(witness)
    return arrays, dict(metadata), manifest_witness, [target_witness, *source_witnesses]


def _canonical_relative_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError("snapshot path must be canonical and relative")
    path = Path(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("snapshot path must be canonical and relative")
    if path.as_posix() != raw_path:
        raise ValueError("snapshot path must be canonical and relative")
    return path


def _required_checkpoint_times(metadata: Mapping[str, Any]) -> dict[SnapshotKey, int]:
    required: dict[SnapshotKey, int] = {}
    for episode in metadata["episodes"]:
        scene = str(episode["scene"])
        records = [episode["anchor"], *episode["checkpoints"]]
        for record in records:
            key = (scene, int(record["frame_index"]))
            timestamp = int(record["relative_timestamp_ns"])
            previous = required.setdefault(key, timestamp)
            if previous != timestamp:
                raise ValueError(f"conflicting checkpoint timestamps for {key}")
    return required


def build_evaluation_checkpoint_plan(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen, target-derived snapshot-frame plan consumed by the runner."""
    validate_generated_target(arrays, metadata)
    required = _required_checkpoint_times(metadata)
    frames = {
        scene: sorted(
            frame_index
            for candidate_scene, frame_index in required
            if candidate_scene == scene
        )
        for scene in ("apartment", "office")
    }
    binding = {
        "schema_version": 1,
        "manifest_id": "tesse_cd_occlusion_v1_checkpoint_frames",
        "evaluation_checkpoint_frames": frames,
    }
    binding_bytes = json.dumps(
        binding,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **binding,
        "evaluation_checkpoint_frames_sha256": hashlib.sha256(
            binding_bytes
        ).hexdigest(),
    }


def _load_checkpoint_snapshots(
    checkpoint_index: Path,
    *,
    metadata: Mapping[str, Any],
    target_manifest_witness: _FileWitness,
    checkpoint_plan: Mapping[str, Any],
) -> tuple[
    Mapping[SnapshotKey, VoxelMapSnapshot],
    dict[str, Any],
    _FileWitness,
]:
    content, index_witness = _read_bound_file(
        checkpoint_index, label="checkpoint index"
    )
    payload = _load_json_bytes(content, label="checkpoint index")
    if set(payload) != {
        "schema_version",
        "manifest_id",
        "dataset",
        "method_id",
        "missing_observation_policy",
        "target_manifest",
        "evaluation_checkpoint_frames_sha256",
        "snapshots",
    } or not (
        payload["schema_version"] == 1
        and payload["manifest_id"] == "oviv2_tesse_cd_occlusion_checkpoints_v1"
        and payload["dataset"] == "TESSE-CD"
        and payload["method_id"] == "OVIV2"
        and payload["missing_observation_policy"]
        in {"signed_depth", "missing_as_absence"}
    ):
        raise ValueError("checkpoint index identity mismatch")
    if payload["target_manifest"] != {
        "sha256": target_manifest_witness.sha256,
        "byte_count": target_manifest_witness.byte_count,
    }:
        raise ValueError("checkpoint target manifest binding mismatch")
    if payload["evaluation_checkpoint_frames_sha256"] != checkpoint_plan[
        "evaluation_checkpoint_frames_sha256"
    ]:
        raise ValueError("evaluation checkpoint frame binding mismatch")
    records = payload["snapshots"]
    if not isinstance(records, list):
        raise ValueError("checkpoint snapshot inventory must be a list")
    required = _required_checkpoint_times(metadata)
    snapshot_paths: dict[SnapshotKey, Path] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "scene",
            "frame_index",
            "relative_timestamp_ns",
            "consumed_through_frame",
            "consumed_through_frame_exclusive",
            "path",
            "checksums_sha256",
        }:
            raise ValueError("checkpoint record fields are not exact")
        scene = record["scene"]
        frame_index = record["frame_index"]
        if (
            scene not in {"apartment", "office"}
            or type(frame_index) is not int
            or frame_index < 0
        ):
            raise ValueError("checkpoint scene or frame is invalid")
        key = (scene, frame_index)
        if key in snapshot_paths:
            raise ValueError(f"duplicate checkpoint: {key}")
        if key not in required:
            raise ValueError(f"unexpected checkpoint: {key}")
        if not (
            record["relative_timestamp_ns"] == required[key]
            and record["consumed_through_frame"] == frame_index
            and record["consumed_through_frame_exclusive"] == frame_index + 1
        ):
            raise ValueError(f"future or invalid causal boundary for checkpoint {key}")
        relative = _canonical_relative_path(record["path"])
        snapshot_path = checkpoint_index.parent / relative
        current = checkpoint_index.parent
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"snapshot path contains a symlink: {relative}")
        _, checksums_witness = _read_bound_file(
            snapshot_path / "checksums.json",
            label="snapshot checksum manifest",
            capture_content=False,
        )
        if record["checksums_sha256"] != checksums_witness.sha256:
            raise ValueError(f"snapshot checksum binding mismatch: {key}")
        snapshot_paths[key] = snapshot_path
    missing = sorted(set(required).difference(snapshot_paths))
    if missing:
        scene, frame_index = missing[0]
        raise ValueError(f"missing checkpoint for {scene} frame {frame_index}")
    return _LazyCheckpointSnapshots(snapshot_paths), payload, index_witness


def _render_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("occlusion result is not finite canonical JSON") from error


def _publish_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise ValueError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"output already exists: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _required_snapshot(
    snapshots: Mapping[SnapshotKey, Any], scene: str, frame_index: int
) -> Any:
    key = (scene, frame_index)
    if key not in snapshots:
        raise ValueError(f"missing checkpoint for {scene} frame {frame_index}")
    snapshot = snapshots[key]
    metadata = snapshot.metadata
    if metadata.scene_id != scene:
        raise ValueError(f"checkpoint scene mismatch for {scene} frame {frame_index}")
    if metadata.frame_id > frame_index:
        raise ValueError(f"future snapshot for {scene} frame {frame_index}")
    if metadata.frame_id < frame_index:
        raise ValueError(f"missing exact checkpoint for {scene} frame {frame_index}")
    return snapshot


def _voxel_keys(values: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    array = np.asarray(values)
    if array.dtype != np.int64 or array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("occlusion target arrays must have shape (N, 3) and dtype int64")
    return tuple(tuple(int(value) for value in row) for row in array)


def _majority_owner(
    snapshot: Any,
    voxel_keys: tuple[tuple[int, int, int], ...],
) -> tuple[int | None, int, bool]:
    counts = Counter(
        record.entity_id
        for key in voxel_keys
        if (record := snapshot.ownership.owner_of(key)) is not None
    )
    if not counts:
        return None, 0, False
    majority_count = max(counts.values())
    tied = sorted(entity_id for entity_id, count in counts.items() if count == majority_count)
    return tied[0], majority_count, len(tied) > 1


def _empty_counts() -> dict[str, int]:
    return {
        "all_gt_occluded_target_voxels": 0,
        "anchor_owned_target_voxels": 0,
        "retained_owner_count": 0,
        "false_release_count": 0,
        "false_reassignment_count": 0,
        "object_checkpoint_count": 0,
        "anchor_mapped_object_checkpoint_count": 0,
        "retained_object_checkpoint_count": 0,
        "episode_count": 0,
        "anchor_mapped_episode_count": 0,
        "zero_release_episode_count": 0,
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finalize_metrics(counts: Mapping[str, int]) -> dict[str, int | float]:
    result: dict[str, int | float] = dict(counts)
    anchor_count = counts["anchor_owned_target_voxels"]
    gt_count = counts["all_gt_occluded_target_voxels"]
    mapped_objects = counts["anchor_mapped_object_checkpoint_count"]
    result.update(
        {
            "false_release_rate": _safe_rate(
                counts["false_release_count"], anchor_count
            ),
            "false_reassignment_rate": _safe_rate(
                counts["false_reassignment_count"], anchor_count
            ),
            "retained_ownership_recall": _safe_rate(
                counts["retained_owner_count"], anchor_count
            ),
            "gt_retained_object_recall": _safe_rate(
                counts["retained_owner_count"], gt_count
            ),
            "retained_object_recall": _safe_rate(
                counts["retained_object_checkpoint_count"], mapped_objects
            ),
            "zero_release_episode_rate": _safe_rate(
                counts["zero_release_episode_count"], counts["episode_count"]
            ),
        }
    )
    return result


def evaluate_fixed_anchor_ownership(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    snapshots: Mapping[SnapshotKey, Any],
) -> dict[str, Any]:
    """Map at each anchor once, then score every later checkpoint without rematching."""
    raw_episodes = metadata.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("occlusion metadata has no episodes")
    episodes = {str(item["episode_id"]): item for item in raw_episodes}
    if len(episodes) != len(raw_episodes):
        raise ValueError("occlusion episode IDs must be unique")

    mappings: list[dict[str, Any]] = []
    episode_counts: dict[str, dict[str, int]] = {}
    for episode_id, episode in episodes.items():
        scene = str(episode["scene"])
        anchor = episode["anchor"]
        anchor_snapshot = _required_snapshot(
            snapshots, scene, int(anchor["frame_index"])
        )
        anchor_keys = _voxel_keys(arrays[str(anchor["array"])])
        mapped_owner, majority_count, tie = _majority_owner(
            anchor_snapshot, anchor_keys
        )
        mappings.append(
            {
                "episode_id": episode_id,
                "gt_object_id": episode["object_id"],
                "predicted_owner_id": mapped_owner,
                "majority_count": majority_count,
                "tie": tie,
            }
        )

        counts = _empty_counts()
        counts["episode_count"] = 1
        episode_anchor_voxels = 0
        episode_releases = 0
        for checkpoint in episode["checkpoints"]:
            counts["object_checkpoint_count"] += 1
            checkpoint_snapshot = _required_snapshot(
                snapshots, scene, int(checkpoint["frame_index"])
            )
            target_keys = _voxel_keys(arrays[str(checkpoint["array"])])
            counts["all_gt_occluded_target_voxels"] += len(target_keys)
            retained_at_checkpoint = 0
            eligible_at_checkpoint = 0
            if mapped_owner is not None:
                for key in target_keys:
                    anchor_record = anchor_snapshot.ownership.owner_of(key)
                    if (
                        anchor_record is None
                        or anchor_record.entity_id != mapped_owner
                    ):
                        continue
                    eligible_at_checkpoint += 1
                    current_record = checkpoint_snapshot.ownership.owner_of(key)
                    if current_record is None:
                        counts["false_release_count"] += 1
                        episode_releases += 1
                    elif current_record.entity_id == mapped_owner:
                        counts["retained_owner_count"] += 1
                        retained_at_checkpoint += 1
                    else:
                        counts["false_reassignment_count"] += 1
            counts["anchor_owned_target_voxels"] += eligible_at_checkpoint
            episode_anchor_voxels += eligible_at_checkpoint
            if eligible_at_checkpoint:
                counts["anchor_mapped_object_checkpoint_count"] += 1
                if retained_at_checkpoint:
                    counts["retained_object_checkpoint_count"] += 1
        if episode_anchor_voxels:
            counts["anchor_mapped_episode_count"] = 1
            if episode_releases == 0:
                counts["zero_release_episode_count"] = 1
        episode_counts[episode_id] = counts

    stress_layers: dict[str, dict[str, int | float]] = {}
    layers = metadata.get("stress_layers")
    if not isinstance(layers, Mapping):
        raise ValueError("occlusion stress layers are missing")
    for label in ("all", "0.50", "0.75", "0.90"):
        layer = layers.get(label)
        if not isinstance(layer, Mapping) or not isinstance(
            layer.get("episode_ids"), list
        ):
            raise ValueError(f"occlusion stress layer {label} is invalid")
        counts = _empty_counts()
        for episode_id in layer["episode_ids"]:
            if episode_id not in episode_counts:
                raise ValueError(f"unknown episode in stress layer {label}")
            for name, value in episode_counts[episode_id].items():
                counts[name] += value
        stress_layers[label] = _finalize_metrics(counts)

    headline = stress_layers["0.90"]
    return {
        "fixed_anchor_mapping_rule": "majority_owner_then_lowest_entity_id",
        "fixed_anchor_mappings": mappings,
        "stress_layers": stress_layers,
        "headline_gate": {
            "stress_layer": "0.90",
            "episode_count": headline["episode_count"],
            "anchor_mapped_episode_count": headline[
                "anchor_mapped_episode_count"
            ],
            "anchor_owned_target_voxels": headline[
                "anchor_owned_target_voxels"
            ],
            "false_release_count": headline["false_release_count"],
            "passed": bool(
                headline["anchor_owned_target_voxels"] > 0
                and headline["anchor_mapped_episode_count"]
                == headline["episode_count"]
                and headline["false_release_count"] == 0
            ),
        },
    }


def evaluate_occlusion_package(
    *,
    target_dir: str | Path,
    checkpoint_index: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate frozen inputs, evaluate fixed anchors, and optionally publish JSON."""
    output = Path(output_path) if output_path is not None else None
    if output is not None and os.path.lexists(output):
        raise ValueError(f"output already exists: {output}")
    arrays, metadata, target_manifest_witness, target_witnesses = _load_target_package(
        Path(target_dir)
    )
    checkpoint_plan = build_evaluation_checkpoint_plan(
        arrays=arrays,
        metadata=metadata,
    )
    snapshots, checkpoint_payload, checkpoint_witness = _load_checkpoint_snapshots(
        Path(checkpoint_index),
        metadata=metadata,
        target_manifest_witness=target_manifest_witness,
        checkpoint_plan=checkpoint_plan,
    )
    metrics = evaluate_fixed_anchor_ownership(
        arrays=arrays,
        metadata=metadata,
        snapshots=snapshots,
    )
    result = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_occlusion_evaluation_v1",
        "dataset": "TESSE-CD",
        "method_id": checkpoint_payload["method_id"],
        "missing_observation_policy": checkpoint_payload[
            "missing_observation_policy"
        ],
        "target_manifest": {
            "sha256": target_manifest_witness.sha256,
            "byte_count": target_manifest_witness.byte_count,
        },
        "checkpoint_count": len(snapshots),
        "evaluation_checkpoint_frames_sha256": checkpoint_plan[
            "evaluation_checkpoint_frames_sha256"
        ],
        **metrics,
    }
    encoded = _render_json(result)
    _revalidate_witness(target_manifest_witness, label="target manifest")
    for witness in target_witnesses:
        _revalidate_witness(witness, label="target input")
    _revalidate_witness(checkpoint_witness, label="checkpoint index")
    if output is not None:
        _publish_no_replace(output, encoded)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_occlusion_package(
        target_dir=args.targets,
        checkpoint_index=args.checkpoints,
        output_path=args.output,
    )
    print(json.dumps(result["headline_gate"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
