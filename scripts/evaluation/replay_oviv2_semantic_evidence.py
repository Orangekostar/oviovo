#!/usr/bin/env python3
"""Replay dense and structure semantics over an immutable OVIV2 snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterator

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_instance_head import (  # noqa: E402
    _publish_fresh_output,
)
from scripts.run_oviv2_replica import (  # noqa: E402
    _dirty_digest,
    _json_hash,
    _load_dense_cache_frame,
    _load_json,
    _preflight,
    _resolve_config_paths,
    _resolve_frontend_feature_model_id,
    _runtime_config,
    _sha256,
    _structure_config,
    _verify_cached_image_features,
    algorithm_config,
)
from src.core.data_structures import Frame  # noqa: E402
from src.evaluation.oviv2_semantic_replay import (  # noqa: E402
    SEMANTIC_REPLAY_SOURCE_RELATIVE_PATHS,
    SemanticReplayInput,
    replay_semantic_evidence,
    semantic_evidence_equal,
    semantic_replay_algorithm_hash,
)
from src.oviv2.dense_projection import DenseSemanticConfig  # noqa: E402
from src.oviv2.observations import (  # noqa: E402
    CachedFrontendAdapter,
    ReplicaVocabulary,
)
from src.oviv2.snapshot import VoxelMapSnapshot  # noqa: E402
from src.oviv2.structure import DepthStructureFrontend  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_class_entropy_powers(values: list[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        components = value.split("=", 1)
        if len(components) != 2:
            raise ValueError("--class-entropy-power must use CLASS_ID=POWER")
        try:
            class_id = int(components[0])
            power = float(components[1])
        except ValueError as error:
            raise ValueError(
                "--class-entropy-power must use numeric CLASS_ID=POWER"
            ) from error
        if class_id <= 0 or not np.isfinite(power) or power < 0.0:
            raise ValueError("class entropy IDs must be positive and powers non-negative")
        if class_id in result:
            raise ValueError(f"duplicate class entropy power for class {class_id}")
        result[class_id] = power
    return result


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256_file(REPO_ROOT / relative)
        for name, relative in SEMANTIC_REPLAY_SOURCE_RELATIVE_PATHS.items()
    }


def _replica_dataset_binding(dataset, source_frame_ids: list[int]) -> dict[str, Any]:
    records = getattr(dataset, "_records", None)
    if not isinstance(records, list) or len(records) < len(source_frame_ids):
        raise ValueError("Replica dataset records are unavailable for provenance binding")
    frames = []
    for cache_index, source_frame_id in enumerate(source_frame_ids):
        record = records[cache_index]
        frames.append(
            {
                "cache_index": cache_index,
                "source_frame_id": source_frame_id,
                "dataset_frame_id": int(record.frame_index),
                "rgb_path": str(record.rgb_path.resolve()),
                "rgb_sha256": _sha256_file(record.rgb_path),
                "depth_path": str(record.depth_path.resolve()),
                "depth_sha256": _sha256_file(record.depth_path),
                "pose_sha256": hashlib.sha256(
                    np.ascontiguousarray(record.pose, dtype=np.float64).tobytes()
                ).hexdigest(),
            }
        )
    frame_manifest = dataset.root / "frame_manifest.json"
    return {
        "root": str(dataset.root.resolve()),
        "trajectory_sha256": _sha256_file(dataset.traj_path),
        "frame_manifest_sha256": (
            _sha256_file(frame_manifest) if frame_manifest.is_file() else None
        ),
        "frames": frames,
    }


def _base_source_frames_match(
    manifest: dict[str, Any],
    source_frame_ids: list[int],
) -> bool:
    matched_binding = False
    if "source_frame_ids_hash" in manifest:
        matched_binding = True
        if manifest["source_frame_ids_hash"] != _json_hash(source_frame_ids):
            return False
    frame_selection = manifest.get("frame_selection")
    if isinstance(frame_selection, dict) and "source_frame_ids" in frame_selection:
        matched_binding = True
        if (
            frame_selection["source_frame_ids"] != source_frame_ids
            or frame_selection.get("sampled_frame_count") != len(source_frame_ids)
        ):
            return False
    return matched_binding


def replay(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config_path = args.config.resolve()
    raw_config = _load_json(config_path)
    config = _resolve_config_paths(raw_config)
    initial_source_hashes = _source_hashes()
    requested_frames = int(args.num_frames or config.get("num_frames", 200))
    snapshot = VoxelMapSnapshot.load(args.base_snapshot.resolve())
    if snapshot.metadata.scene_id != config["scene"]:
        raise ValueError("base snapshot scene does not match replay config")
    if requested_frames != snapshot.metadata.revision:
        raise ValueError("replay frame count must equal the base snapshot revision")
    base_run_manifest_path = args.base_snapshot.resolve().parent.parent / "run_manifest.json"
    base_run_manifest = _load_json(base_run_manifest_path)
    if base_run_manifest.get("scene") != config["scene"]:
        raise ValueError("base run manifest scene does not match replay config")
    base_config_path = Path(str(base_run_manifest.get("config_path", "")))
    base_config = _load_json(base_config_path)
    if _json_hash(base_config) != base_run_manifest.get("config_hash"):
        raise ValueError("base run config hash does not match its manifest")
    if algorithm_config(base_config) != algorithm_config(raw_config):
        raise ValueError("replay config algorithm does not match the base run manifest")

    (
        dataset,
        benchmark,
        source_ids,
        frontend_hash,
        frontend_manifest,
        dense_cache,
    ) = _preflight(config, requested_frames, True)
    if benchmark.get("dataset") != "Replica":
        raise ValueError("semantic evidence replay currently supports Replica only")
    if dense_cache is None:
        raise ValueError("semantic replay requires cached dense probabilities")
    if snapshot.metadata.dense_semantic_provenance != dense_cache.consumed_provenance:
        raise ValueError("base snapshot dense provenance does not match replay cache")
    initial_dataset_binding = _replica_dataset_binding(dataset, source_ids)

    runtime_config = _runtime_config(config)
    if runtime_config.dense_semantics is None:
        raise ValueError("semantic replay requires dense semantics in the base config")
    base_dense_manifest = base_run_manifest.get("dense_semantics")
    if (
        not isinstance(base_dense_manifest, dict)
        or base_dense_manifest.get("config") != asdict(runtime_config.dense_semantics)
    ):
        raise ValueError("base run dense config does not match replay config")
    dense_config = DenseSemanticConfig(
        voxel_size_m=runtime_config.dense_semantics.voxel_size_m,
        integration_radius_m=runtime_config.dense_semantics.integration_radius_m,
        minimum_probability=(
            runtime_config.dense_semantics.minimum_probability
            if args.dense_minimum_probability is None
            else args.dense_minimum_probability
        ),
        minimum_quality=(
            runtime_config.dense_semantics.minimum_quality
            if args.dense_minimum_quality is None
            else args.dense_minimum_quality
        ),
        entropy_power=(
            runtime_config.dense_semantics.entropy_power
            if args.dense_entropy_power is None
            else args.dense_entropy_power
        ),
        view_angle_power=(
            runtime_config.dense_semantics.view_angle_power
            if args.dense_view_angle_power is None
            else args.dense_view_angle_power
        ),
    )
    class_entropy_powers = _parse_class_entropy_powers(args.class_entropy_power)
    if any(class_id > dense_cache.class_count for class_id in class_entropy_powers):
        raise ValueError("class entropy power ID exceeds dense vocabulary")
    if snapshot.evidence.config != runtime_config.evidence:
        raise ValueError("base snapshot evidence config does not match replay config")

    vocabulary = ReplicaVocabulary(
        classes=tuple(benchmark["vocabulary"]["classes"]),
        aliases=benchmark.get("aliases", {}),
    )
    vocabulary_hash = _json_hash(
        {
            "classes": list(vocabulary.classes),
            "aliases": dict(sorted(vocabulary.aliases.items())),
        }
    )
    if not _base_source_frames_match(base_run_manifest, source_ids):
        raise ValueError("base run manifest source frames do not match replay")
    feature_model_id = _resolve_frontend_feature_model_id(config, frontend_manifest)
    if config.get("feature_mode") == "cached_image":
        if frontend_manifest is None or feature_model_id is None:
            raise ValueError("cached_image replay requires frontend feature provenance")
        _verify_cached_image_features(Path(config["frontend_cache_dir"]), requested_frames)
    frontend = CachedFrontendAdapter(
        config["frontend_cache_dir"],
        vocabulary,
        voxel_size_m=runtime_config.tsdf.voxel_size_m,
        pixel_stride=int(config.get("pixel_stride", 4)),
        min_valid_points=int(config.get("min_valid_points", 10)),
        feature_model_id=feature_model_id,
    )
    structure_config = _structure_config(
        config,
        voxel_size_m=runtime_config.tsdf.voxel_size_m,
    )
    structure_frontend = DepthStructureFrontend(vocabulary, structure_config)

    def replay_inputs() -> Iterator[SemanticReplayInput]:
        for cache_index, source_frame_id in enumerate(source_ids):
            dataset_frame = dataset[cache_index]
            current = Frame(
                frame_id=source_frame_id,
                source_frame_id=dataset_frame.frame_id,
                rgb=dataset_frame.rgb,
                depth=dataset_frame.depth,
                pose=dataset_frame.pose,
                intrinsics=dataset_frame.intrinsics,
                timestamp=float(source_frame_id),
            )
            object_observations = frontend.observe(current, cache_index)
            structure_observations = structure_frontend.observe(
                current,
                object_observations=object_observations,
            )
            yield SemanticReplayInput(
                frame=current,
                dense_semantics=_load_dense_cache_frame(
                    dense_cache,
                    cache_index,
                    source_frame_id,
                ),
                structure_observations=structure_observations,
            )

    result = replay_semantic_evidence(
        replay_inputs(),
        evidence_config=snapshot.evidence.config,
        dense_config=dense_config,
        semantic_support_scale=runtime_config.semantic_support_scale,
        entropy_power_by_class=class_entropy_powers,
    )
    frame_results = [asdict(item) for item in result.frame_results]
    (
        verified_dataset,
        _verified_benchmark,
        verified_source_ids,
        verified_frontend_hash,
        verified_frontend_manifest,
        verified_dense_cache,
    ) = _preflight(config, requested_frames, True)
    if (
        _source_hashes() != initial_source_hashes
        or verified_source_ids != source_ids
        or verified_frontend_hash != frontend_hash
        or verified_frontend_manifest != frontend_manifest
        or verified_dense_cache != dense_cache
        or _replica_dataset_binding(verified_dataset, verified_source_ids)
        != initial_dataset_binding
    ):
        raise ValueError("semantic replay inputs or source code changed during execution")
    source_hashes = initial_source_hashes
    base_dense_config = asdict(runtime_config.dense_semantics)
    replay_config = asdict(dense_config)
    class_entropy_payload = {
        str(class_id): power
        for class_id, power in sorted(class_entropy_powers.items())
    }
    control_parameters_match_base = (
        replay_config == base_dense_config and not class_entropy_powers
    )
    control_semantic_evidence_matches_base = (
        semantic_evidence_equal(snapshot.evidence, result.evidence)
        if control_parameters_match_base
        else None
    )
    if control_semantic_evidence_matches_base is False:
        raise ValueError("control replay semantic evidence does not match base snapshot")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "OVIV2-semantic-evidence-replay",
        "scene": config["scene"],
        "frame_count": requested_frames,
        "source_frame_ids": source_ids,
        "source_frame_ids_hash": _json_hash(source_ids),
        "structure_replayed": result.structure_replayed,
        "structure_config": asdict(structure_config),
        "semantic_support_scale": runtime_config.semantic_support_scale,
        "dense_config": replay_config,
        "entropy_power_by_class": class_entropy_payload,
        "base_dense_config": base_dense_config,
        "control_parameters_match_base": control_parameters_match_base,
        "control_semantic_evidence_matches_base": (
            control_semantic_evidence_matches_base
        ),
        "base_snapshot": str(args.base_snapshot.resolve()),
        "base_snapshot_checksums": dict(sorted(snapshot.checksums.items())),
        "base_snapshot_revision": snapshot.metadata.revision,
        "base_run_manifest": str(base_run_manifest_path),
        "base_run_manifest_sha256": _sha256(base_run_manifest_path),
        "config_path": str(config_path),
        "config_hash": _json_hash(raw_config),
        "benchmark_manifest": str(Path(config["manifest"]).resolve()),
        "benchmark_manifest_sha256": _sha256(Path(config["manifest"])),
        "vocabulary_hash": vocabulary_hash,
        "frontend_cache_hash": frontend_hash,
        "frontend_manifest": frontend_manifest,
        "dataset": initial_dataset_binding,
        "dense_cache": {
            "path": str(dense_cache.cache_dir),
            "manifest_sha256": dense_cache.manifest_sha256,
            "cache_files_sha256": dense_cache.cache_files_sha256,
            "producer_provenance": asdict(dense_cache.producer_provenance),
            "consumed_provenance": asdict(dense_cache.consumed_provenance),
        },
        "source_hashes": source_hashes,
        "algorithm_hash": semantic_replay_algorithm_hash(
            dense_config=replay_config,
            entropy_power_by_class=class_entropy_payload,
            semantic_support_scale=runtime_config.semantic_support_scale,
            source_hashes=source_hashes,
            structure_config=asdict(structure_config),
        ),
        "dirty_state_digest": _dirty_digest(),
        "command": [str(value) for value in sys.argv],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "counters": {
            name: sum(item[name] for item in frame_results)
            for name in (
                "dense_sampled_pixel_count",
                "dense_valid_pixel_count",
                "dense_updated_voxel_count",
                "structure_observation_count",
                "structure_updated_voxel_count",
            )
        },
    }

    def write_output(directory: Path) -> None:
        evidence_path = directory / "semantic_evidence.npz"
        result.evidence.save(evidence_path)
        manifest["overlay_sha256"] = _sha256_file(evidence_path)
        _write_json(directory / "frame_results.json", frame_results)
        _write_json(directory / "replay_manifest.json", manifest)

    _publish_fresh_output(output, write_output)
    return {
        "frame_count": requested_frames,
        "overlay_sha256": manifest["overlay_sha256"],
        "control_parameters_match_base": manifest["control_parameters_match_base"],
        "control_semantic_evidence_matches_base": manifest[
            "control_semantic_evidence_matches_base"
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--dense-entropy-power", type=float)
    parser.add_argument("--dense-minimum-quality", type=float)
    parser.add_argument("--dense-minimum-probability", type=float)
    parser.add_argument("--dense-view-angle-power", type=float)
    parser.add_argument(
        "--class-entropy-power",
        action="append",
        default=[],
        metavar="CLASS_ID=POWER",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(replay(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
