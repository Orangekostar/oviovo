#!/usr/bin/env python3
"""Build and evaluate an immutable OVIV2 sparse voxel map on Replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import evaluate as evaluate_snapshot  # noqa: E402
from src.core.data_structures import Frame  # noqa: E402
from src.datasets.replica import ReplicaRoom0Dataset  # noqa: E402
from src.oviv2.entities import EntityRegistryConfig  # noqa: E402
from src.oviv2.evidence import EvidenceConfig  # noqa: E402
from src.oviv2.geometry import TsdfConfig  # noqa: E402
from src.oviv2.meshing import derive_labeled_mesh, write_labeled_mesh  # noqa: E402
from src.oviv2.observations import CachedFrontendAdapter, ReplicaVocabulary  # noqa: E402
from src.oviv2.runtime import Oviv2Runtime, Oviv2RuntimeConfig  # noqa: E402
from src.oviv2.tracking import LocalTrackerConfig  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    for name in ("dataset_root", "frontend_cache_dir", "manifest", "gt_mesh", "gt_info"):
        raw = resolved.get(name)
        if raw is None:
            continue
        path = Path(raw).expanduser()
        resolved[name] = str(path if path.is_absolute() else (REPO_ROOT / path).resolve())
    return resolved


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _dirty_digest() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _artifact_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def _verify_resume(output: Path, config_hash: str, requested_frames: int) -> dict[str, Any]:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("resume requires an atomically completed run_manifest.json")
    manifest = _load_json(manifest_path)
    if manifest.get("config_hash") != config_hash:
        raise ValueError("resume config hash does not match completed run")
    if manifest.get("frame_selection", {}).get("sampled_frame_count") != requested_frames:
        raise ValueError("resume frame count does not match completed run")
    for relative, expected in manifest.get("artifact_checksums", {}).items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"resume artifact checksum mismatch: {relative}")
    return manifest


def _preflight(
    config: dict[str, Any],
    num_frames: int,
    skip_evaluation: bool,
) -> tuple[ReplicaRoom0Dataset, dict[str, Any], list[int], str]:
    required = ("scene", "dataset_root", "frontend_cache_dir", "manifest")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"config is missing required fields: {', '.join(missing)}")
    manifest_path = Path(config["manifest"])
    benchmark = _load_json(manifest_path)
    if benchmark.get("schema_version") != 1 or benchmark.get("dataset") != "Replica":
        raise ValueError("benchmark manifest must be Replica schema_version 1")
    if config["scene"] not in {item.get("scene") for item in benchmark.get("scenes", ())}:
        raise ValueError("configured scene is absent from benchmark manifest")
    classes = benchmark.get("vocabulary", {}).get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("benchmark manifest has no frozen vocabulary")

    dataset = ReplicaRoom0Dataset(Path(config["dataset_root"]))
    if num_frames <= 0 or num_frames > len(dataset):
        raise ValueError(f"num_frames must lie in [1, {len(dataset)}]")
    source_start = int(config.get("source_start", 0))
    source_stride = int(config.get("source_stride", 10))
    if source_start < 0 or source_stride <= 0:
        raise ValueError("source_start must be non-negative and source_stride positive")
    source_ids = [source_start + index * source_stride for index in range(num_frames)]
    selection = benchmark.get("frame_selection", {})
    stop = int(selection.get("stop_exclusive", source_ids[-1] + 1))
    if source_ids[-1] >= stop:
        raise ValueError("requested source frame selection exceeds benchmark manifest")

    cache_dir = Path(config["frontend_cache_dir"])
    frontend_digest = hashlib.sha256()
    for cache_index in range(num_frames):
        cache_path = cache_dir / f"frame{cache_index:06d}.pkl.gz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        frontend_digest.update(cache_index.to_bytes(8, "little"))
        frontend_digest.update(bytes.fromhex(_sha256(cache_path)))
    if not skip_evaluation:
        for name in ("gt_mesh", "gt_info"):
            path = Path(config.get(name, ""))
            if not path.is_file():
                raise FileNotFoundError(path)
    return dataset, benchmark, source_ids, frontend_digest.hexdigest()


def _runtime_config(config: dict[str, Any]) -> Oviv2RuntimeConfig:
    voxel_size = float(config.get("voxel_size_m", 0.05))
    block_resolution = int(config.get("block_resolution", 8))
    source_stride = int(config.get("source_stride", 10))
    return Oviv2RuntimeConfig(
        tsdf=TsdfConfig(
            voxel_size_m=voxel_size,
            block_resolution=block_resolution,
            block_count=int(config.get("block_count", 100_000)),
            depth_max_m=float(config.get("depth_max_m", 10.0)),
            trunc_voxel_multiplier=float(config.get("trunc_voxel_multiplier", 4.0)),
        ),
        evidence=EvidenceConfig(
            block_resolution=block_resolution,
            semantic_top_k=int(config.get("semantic_top_k", 4)),
            entity_top_k=int(config.get("entity_top_k", 4)),
        ),
        tracker=LocalTrackerConfig(
            window_size=int(config.get("track_window_size", 5)),
            confirm_hits=int(config.get("confirm_hits", 2)),
            max_age_frames=int(config.get("max_age_frames", 3)) * source_stride,
            min_voxel_overlap=float(config.get("track_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("track_max_centroid_distance_m", 0.5)),
        ),
        registry=EntityRegistryConfig(
            min_voxel_overlap=float(config.get("entity_min_voxel_overlap", 0.1)),
            max_centroid_distance_m=float(config.get("entity_max_centroid_distance_m", 0.6)),
        ),
        visibility_depth_tolerance_m=float(
            config.get("visibility_depth_tolerance_m", 0.1)
        ),
        absence_negative_support=float(config.get("absence_negative_support", 1.0)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    raw_config = _load_json(config_path)
    config_hash = _json_hash(raw_config)
    config = _resolve_config_paths(raw_config)
    requested_frames = int(args.num_frames or config.get("num_frames", 200))
    output = args.output.resolve()
    if output.exists():
        if args.resume:
            return _verify_resume(output, config_hash, requested_frames)
        raise FileExistsError(f"output already exists: {output}")

    starting_dirty_digest = _dirty_digest()
    dataset, benchmark, source_ids, frontend_hash = _preflight(
        config,
        requested_frames,
        args.skip_evaluation,
    )
    vocabulary = ReplicaVocabulary(
        classes=tuple(benchmark["vocabulary"]["classes"]),
        aliases=benchmark.get("aliases", {}),
    )
    runtime_config = _runtime_config(config)
    frontend = CachedFrontendAdapter(
        config["frontend_cache_dir"],
        vocabulary,
        voxel_size_m=runtime_config.tsdf.voxel_size_m,
        pixel_stride=int(config.get("pixel_stride", 4)),
        min_valid_points=int(config.get("min_valid_points", 10)),
    )
    runtime = Oviv2Runtime(str(config["scene"]), runtime_config)
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    final = output / "final"
    checkpoints.mkdir()
    final.mkdir()

    started_wall = time.time()
    started = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 20)))
    for cache_index, source_frame_id in enumerate(source_ids):
        frame_started = time.perf_counter()
        dataset_frame = dataset[cache_index]
        frame = Frame(
            frame_id=source_frame_id,
            source_frame_id=dataset_frame.frame_id,
            rgb=dataset_frame.rgb,
            depth=dataset_frame.depth,
            pose=dataset_frame.pose,
            intrinsics=dataset_frame.intrinsics,
            timestamp=float(source_frame_id),
        )
        observations = frontend.observe(frame, cache_index)
        result = runtime.process_frame(frame, observations)
        frame_records.append(
            {
                "cache_frame_id": cache_index,
                "source_frame_id": source_frame_id,
                "revision": result.revision,
                "observation_count": result.observation_count,
                "accepted_entity_count": len(result.accepted_entity_ids),
                "elapsed_sec": time.perf_counter() - frame_started,
            }
        )
        if (cache_index + 1) % checkpoint_interval == 0 or cache_index + 1 == requested_frames:
            runtime.commit(checkpoints / "latest_voxel_snapshot.npz")
            runtime.registry.save(checkpoints / "latest_entities.jsonl")

    snapshot = runtime.commit(final / "oviv2_voxel_snapshot.npz")
    runtime.registry.save(final / "oviv2_entities.jsonl")
    mesh = derive_labeled_mesh(snapshot.geometry, snapshot.evidence, snapshot.ownership)
    write_labeled_mesh(final / "oviv2_instance_mesh.ply", mesh)
    mapping_elapsed = time.perf_counter() - started

    if not args.skip_evaluation:
        evaluation_args = argparse.Namespace(
            snapshot=final / "oviv2_voxel_snapshot.npz",
            entity_info=final / "oviv2_entities.jsonl",
            gt_mesh=Path(config["gt_mesh"]),
            gt_info=Path(config["gt_info"]),
            manifest=Path(config["manifest"]),
            scene=str(config["scene"]),
            output=output / "evaluation",
            min_instance_vertices=int(config.get("min_instance_vertices", 100)),
        )
        evaluate_snapshot(evaluation_args)

    timing = {
        "started_unix_sec": started_wall,
        "mapping_elapsed_sec": mapping_elapsed,
        "total_elapsed_sec": time.perf_counter() - started,
        "frame_count": requested_frames,
        "frames": frame_records,
    }
    _atomic_json(output / "timing.json", timing)
    vocabulary_payload = {
        "classes": list(vocabulary.classes),
        "aliases": dict(sorted(vocabulary.aliases.items())),
    }
    run_manifest = {
        "schema_version": 1,
        "method": "OVIV2",
        "scene": config["scene"],
        "repository_commit": _git_value("rev-parse", "HEAD"),
        "dirty_state_digest": starting_dirty_digest,
        "command": [str(value) for value in sys.argv],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "benchmark_manifest_path": str(Path(config["manifest"]).resolve()),
        "benchmark_manifest_hash": _sha256(Path(config["manifest"])),
        "vocabulary_hash": _json_hash(vocabulary_payload),
        "model_weights": {"frozen_frontend_cache_sha256": frontend_hash},
        "hardware": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "open3d": o3d.__version__,
        },
        "frame_selection": {
            "source_start": source_ids[0],
            "source_stride": int(config.get("source_stride", 10)),
            "sampled_frame_count": requested_frames,
            "source_frame_ids": source_ids,
        },
        "voxel_settings": {
            "voxel_size_m": runtime_config.tsdf.voxel_size_m,
            "block_resolution": runtime_config.tsdf.block_resolution,
            "semantic_top_k": runtime_config.evidence.semantic_top_k,
            "entity_top_k": runtime_config.evidence.entity_top_k,
        },
        "authoritative_state": "sparse_voxel_layers",
        "dense_point_cloud_state": False,
        "final_revision": runtime.revision,
        "entity_count": len(runtime.registry.entities),
        "geometry_block_count": runtime.geometry.active_block_count,
        "artifact_checksums": _artifact_checksums(output),
    }
    _atomic_json(output / "run_manifest.json", run_manifest)
    return run_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": manifest["frame_selection"]["sampled_frame_count"],
                "entities": manifest["entity_count"],
                "revision": manifest["final_revision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
