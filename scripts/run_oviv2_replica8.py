#!/usr/bin/env python3
"""Run one frozen OVIV2 configuration across the exact Replica-8 scene set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_oviv2_replica import algorithm_hash, run as run_scene  # noqa: E402


REPLICA8_SCENES = (
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
)


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    config: dict[str, Any]
    algorithm_hash: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _verify_frozen_config_files(
    config_path: Path,
    config_sha256: str,
    base_path: Path,
    base_sha256: str,
) -> None:
    if (
        _sha256(config_path) != config_sha256
        or _sha256(base_path) != base_sha256
    ):
        raise ValueError("frozen config changed during batch")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_scene_specs(config_path: str | Path) -> tuple[SceneSpec, ...]:
    config = _load_json(Path(config_path).resolve())
    manifest_path = _resolve_repo_path(config["manifest"])
    benchmark = _load_json(manifest_path)
    scene_items = benchmark.get("scenes", ())
    if tuple(item.get("scene") for item in scene_items) != REPLICA8_SCENES:
        raise ValueError("Replica manifest must contain the frozen eight scenes in order")
    base = _load_json(_resolve_repo_path(config["base_runner_config"]))
    if config.get("semantic_head") == "fused_uncertainty":
        frozen_scale = config.get("fusion_entity_weight_scale")
        if (
            isinstance(frozen_scale, bool)
            or not isinstance(frozen_scale, (int, float))
            or not 0.0 <= float(frozen_scale) <= 1.0
        ):
            raise ValueError(
                "fused batch requires fusion_entity_weight_scale in [0, 1]"
            )
        if base.get("fusion_entity_weight_scale") != float(frozen_scale):
            raise ValueError(
                "base runner config does not use frozen fusion_entity_weight_scale"
            )
    view_root = Path(config["view_root"]).expanduser().resolve()
    gt_root = Path(
        config.get("ground_truth_root", benchmark.get("ground_truth_root", ""))
    ).expanduser().resolve()
    if not str(gt_root):
        raise ValueError("ground_truth_root is required")
    variant_template = str(config["frontend"]["gsa_variant_template"])
    dense_cache_root_value = config.get("dense_cache_root")
    dense_cache_template_value = config.get("dense_cache_template")
    if (dense_cache_root_value is None) != (dense_cache_template_value is None):
        raise ValueError("dense_cache_root and dense_cache_template must be specified together")
    dense_cache_root = (
        Path(str(dense_cache_root_value)).expanduser().resolve()
        if dense_cache_root_value is not None
        else None
    )
    dense_cache_template = (
        str(dense_cache_template_value)
        if dense_cache_template_value is not None
        else None
    )

    specs: list[SceneSpec] = []
    for item in scene_items:
        scene = str(item["scene"])
        gt_scene = str(item["ground_truth_scene"])
        view = view_root / f"{scene}{config['view_suffix']}"
        scene_config = dict(base)
        scene_config.update(
            {
                "scene": scene,
                "dataset_root": str(view),
                "frontend_cache_dir": str(
                    view / f"gsa_detections_{variant_template.format(scene=scene)}"
                ),
                "manifest": base.get("manifest", str(manifest_path)),
                "gt_mesh": str(gt_root / gt_scene / "habitat" / "mesh_semantic.ply"),
                "gt_info": str(gt_root / gt_scene / "habitat" / "info_semantic.json"),
            }
        )
        if dense_cache_root is not None:
            assert dense_cache_template is not None
            cache_relative = Path(dense_cache_template.format(scene=scene))
            if cache_relative.is_absolute() or ".." in cache_relative.parts:
                raise ValueError("dense_cache_template must produce a relative child path")
            scene_config["dense_cache_dir"] = str(dense_cache_root / cache_relative)
        frozen_hash = algorithm_hash(scene_config)
        scene_config["algorithm_hash"] = frozen_hash
        specs.append(SceneSpec(scene, scene_config, frozen_hash))
    if len({spec.algorithm_hash for spec in specs}) != 1:
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


def run_replica8(
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
    scene_records: list[dict[str, Any]] = []
    for spec in specs:
        scene_config_path = config_dir / f"{spec.scene}.json"
        scene_output = output / spec.scene
        manifest = scene_runner(
            argparse.Namespace(
                config=scene_config_path,
                output=scene_output,
                num_frames=None,
                skip_evaluation=skip_evaluation,
                resume=scene_output.exists(),
            )
        )
        if manifest.get("method") != "OVIV2" or manifest.get("scene") != spec.scene:
            raise ValueError(f"invalid OVIV2 scene manifest: {spec.scene}")
        if manifest.get("algorithm_hash") != spec.algorithm_hash:
            raise ValueError(f"algorithm hash mismatch after scene run: {spec.scene}")
        scene_config = _load_json(scene_config_path)
        canonical_config_hash = _json_hash(scene_config)
        if manifest.get("config_hash") != canonical_config_hash:
            raise ValueError(f"config hash mismatch after scene run: {spec.scene}")
        if manifest.get("frame_selection", {}).get("sampled_frame_count") != 200:
            raise ValueError(f"scene run did not process 200 frames: {spec.scene}")
        if manifest.get("final_revision") != 200:
            raise ValueError(f"scene run did not reach revision 200: {spec.scene}")
        run_manifest_path = scene_output / "run_manifest.json"
        scene_records.append(
            {
                "scene": spec.scene,
                "run_manifest": str(run_manifest_path),
                "run_manifest_sha256": _sha256(run_manifest_path),
                "scene_config_sha256": _sha256(scene_config_path),
                "config_hash": canonical_config_hash,
                "frontend_algorithm_hash": manifest.get("frontend_algorithm_hash"),
            }
        )
        _verify_frozen_config_files(
            config_path,
            config_sha256,
            base_path,
            base_sha256,
        )
    frontend_hashes = {
        item["frontend_algorithm_hash"]
        for item in scene_records
        if item["frontend_algorithm_hash"] is not None
    }
    if len(frontend_hashes) > 1:
        raise ValueError("Replica scene runs used different frontend algorithm hashes")
    batch = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "Replica",
        "scene_ids": list(REPLICA8_SCENES),
        "frame_count_per_scene": 200,
        "algorithm_hash": specs[0].algorithm_hash,
        "frontend_algorithm_hash": next(iter(frontend_hashes), None),
        "semantic_head": semantic_head,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "base_runner_config_path": str(base_path),
        "base_runner_config_sha256": base_sha256,
        "scenes": scene_records,
    }
    _atomic_json(output / "batch_manifest.json", batch)
    return batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/oviv2_replica8.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.materialize_only:
        specs = materialize_scene_configs(args.config, args.output)
        print(json.dumps({"output": str(args.output), "scene_configs": len(specs)}))
        return 0
    result = run_replica8(
        args.config,
        args.output,
        skip_evaluation=args.skip_evaluation,
    )
    print(json.dumps({"output": str(args.output), "scenes": len(result["scenes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
