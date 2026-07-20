#!/usr/bin/env python3
"""Run one frozen OVIV2 configuration across ScanNet200-5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_oviv2_replica import (  # noqa: E402
    algorithm_hash,
    run as run_scene,
)
from scripts.run_oviv2_replica8 import (  # noqa: E402
    SceneSpec,
    _atomic_json,
    _json_hash,
    _load_json,
    _resolve_repo_path,
    _sha256,
    _verify_frozen_config_files,
)


SCANNET200_5_SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def build_scene_specs(config_path: str | Path) -> tuple[SceneSpec, ...]:
    config = _load_json(Path(config_path).resolve())
    manifest_path = _resolve_repo_path(config["manifest"])
    manifest = _load_json(manifest_path)
    scene_records = list(manifest.get("scenes", ()))
    if manifest.get("dataset") != "ScanNet200" or tuple(
        value.get("scene") for value in scene_records
    ) != SCANNET200_5_SCENES:
        raise ValueError("ScanNet manifest must contain the frozen five scenes in order")
    base = _load_json(_resolve_repo_path(config["base_runner_config"]))
    if config.get("semantic_head") == "fused_uncertainty":
        scale = config.get("fusion_entity_weight_scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("fused batch requires a numeric fusion scale")
        if base.get("fusion_entity_weight_scale") != float(scale):
            raise ValueError("base runner config does not use the frozen fusion scale")
    view_root = Path(config["view_root"]).expanduser().resolve()
    dense_root = Path(config["dense_cache_root"]).expanduser().resolve()
    variant_template = str(config["frontend"]["gsa_variant_template"])
    specs: list[SceneSpec] = []
    for record in scene_records:
        scene = str(record["scene"])
        frame_count = int(record["frame_count"])
        if frame_count != len(record.get("source_frame_ids", ())):
            raise ValueError(f"scene {scene} frame count does not match source IDs")
        view = view_root / scene
        scene_config = dict(base)
        scene_config.update(
            {
                "scene": scene,
                "dataset_root": str(Path(record["dataset_root"]).resolve()),
                "frontend_cache_dir": str(
                    view
                    / f"gsa_detections_{variant_template.format(scene=scene)}"
                ),
                "manifest": str(manifest_path),
                "gt_mesh": str(Path(record["official_gt_path"]).resolve()),
                "gt_info": str(Path(record["metadata_path"]).resolve()),
                "num_frames": frame_count,
                "dense_cache_dir": str(dense_root / scene),
            }
        )
        frozen_hash = algorithm_hash(scene_config)
        scene_config["algorithm_hash"] = frozen_hash
        specs.append(SceneSpec(scene, scene_config, frozen_hash))
    if len({value.algorithm_hash for value in specs}) != 1:
        raise ValueError("scene expansion changed the frozen OVIV2 algorithm hash")
    return tuple(specs)


def materialize_scene_configs(
    config_path: str | Path,
    output: str | Path,
) -> tuple[SceneSpec, ...]:
    specs = build_scene_specs(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "scene_configs"
    config_dir.mkdir(exist_ok=True)
    for spec in specs:
        _atomic_json(config_dir / f"{spec.scene}.json", spec.config)
    return specs


def run_scannet200_5(
    config_path: str | Path,
    output: str | Path,
    *,
    scene_runner: Callable[[argparse.Namespace], dict[str, Any]] = run_scene,
    skip_evaluation: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config_sha256 = _sha256(config_path)
    batch_config = _load_json(config_path)
    base_path = _resolve_repo_path(batch_config["base_runner_config"])
    base_sha256 = _sha256(base_path)
    specs = materialize_scene_configs(config_path, output)
    _verify_frozen_config_files(
        config_path,
        config_sha256,
        base_path,
        base_sha256,
    )
    semantic_head = batch_config.get("semantic_head", "owner_authoritative")
    if semantic_head not in {
        "owner_authoritative",
        "dense_only",
        "fused_uncertainty",
    }:
        raise ValueError("semantic_head is unsupported")
    config_dir = output / "scene_configs"
    records: list[dict[str, Any]] = []
    frame_counts: dict[str, int] = {}
    for spec in specs:
        scene_config_path = config_dir / f"{spec.scene}.json"
        scene_output = output / spec.scene
        expected_count = int(spec.config["num_frames"])
        manifest = scene_runner(
            argparse.Namespace(
                config=scene_config_path,
                output=scene_output,
                num_frames=None,
                skip_evaluation=skip_evaluation,
                resume=scene_output.exists(),
            )
        )
        if (
            manifest.get("method") != "OVIV2"
            or manifest.get("dataset_name") != "ScanNet200"
            or manifest.get("scene") != spec.scene
        ):
            raise ValueError(f"invalid OVIV2 ScanNet scene manifest: {spec.scene}")
        if manifest.get("algorithm_hash") != spec.algorithm_hash:
            raise ValueError(f"algorithm hash mismatch after scene run: {spec.scene}")
        scene_config = _load_json(scene_config_path)
        canonical_hash = _json_hash(scene_config)
        if manifest.get("config_hash") != canonical_hash:
            raise ValueError(f"config hash mismatch after scene run: {spec.scene}")
        selection = manifest.get("frame_selection", {})
        if (
            selection.get("sampled_frame_count") != expected_count
            or len(selection.get("source_frame_ids", ())) != expected_count
            or manifest.get("final_revision") != expected_count
        ):
            raise ValueError(f"scene run did not process all frozen frames: {spec.scene}")
        run_manifest_path = scene_output / "run_manifest.json"
        records.append(
            {
                "scene": spec.scene,
                "frame_count": expected_count,
                "algorithm_hash": spec.algorithm_hash,
                "run_manifest": str(run_manifest_path),
                "run_manifest_sha256": _sha256(run_manifest_path),
                "scene_config_sha256": _sha256(scene_config_path),
                "config_hash": canonical_hash,
                "frontend_algorithm_hash": manifest.get("frontend_algorithm_hash"),
            }
        )
        frame_counts[spec.scene] = expected_count
        _verify_frozen_config_files(
            config_path,
            config_sha256,
            base_path,
            base_sha256,
        )
    frontend_hashes = {
        record["frontend_algorithm_hash"]
        for record in records
        if record["frontend_algorithm_hash"] is not None
    }
    if len(frontend_hashes) > 1:
        raise ValueError("ScanNet scenes used different frontend algorithm hashes")
    batch = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "ScanNet200",
        "split": "scannet200_5_heldout",
        "scene_ids": list(SCANNET200_5_SCENES),
        "frame_counts": frame_counts,
        "algorithm_hash": specs[0].algorithm_hash,
        "frontend_algorithm_hash": next(iter(frontend_hashes), None),
        "semantic_head": semantic_head,
        "fusion_entity_weight_scale": batch_config.get(
            "fusion_entity_weight_scale"
        ),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "base_runner_config_path": str(base_path),
        "base_runner_config_sha256": base_sha256,
        "scenes": records,
    }
    _atomic_json(output / "batch_manifest.json", batch)
    return batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/oviv2_scannet200_stage4.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.materialize_only:
        specs = materialize_scene_configs(args.config, args.output)
        print(json.dumps({"output": str(args.output), "scene_configs": len(specs)}))
        return 0
    result = run_scannet200_5(
        args.config,
        args.output,
        skip_evaluation=args.skip_evaluation,
    )
    print(json.dumps({"output": str(args.output), "scenes": len(result["scenes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
