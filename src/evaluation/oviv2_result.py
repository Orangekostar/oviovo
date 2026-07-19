"""Strict Replica-8/7 aggregation and provenance checks for OVIV2."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from statistics import fmean
from typing import Any


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
REPLICA7_SCENES = REPLICA8_SCENES[1:]
METRIC_FAMILIES = {
    "semantic": ("miou", "macc", "f_miou"),
    "instance": ("ap25", "ap50"),
    "geometry": ("f5",),
}
PROVENANCE_HASH_FIELDS = (
    "algorithm_hash",
    "frontend_algorithm_hash",
    "vocabulary_hash",
    "benchmark_manifest_hash",
)
EVALUATION_FILES = (
    "metrics.json",
    "per_class_semantic.json",
    "per_class_instance_ap.json",
    "gt_aligned_semantic_ids.npy",
    "gt_aligned_instance_ids.npy",
    "oviv2_instance_mesh.ply",
    "semantic_map_gt.ply",
    "instance_map_gt.ply",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_file_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    actual = {item.name for item in root.iterdir() if item.is_file()} if root.is_dir() else set()
    if actual != set(EVALUATION_FILES):
        raise ValueError(f"evaluation directory has unexpected file contract: {root}")
    return {name: _sha256(root / name) for name in EVALUATION_FILES}


def verify_byte_identical_evaluation_dirs(
    original: str | Path,
    repeated: str | Path,
) -> dict[str, str]:
    original_hashes = evaluation_file_hashes(original)
    repeated_hashes = evaluation_file_hashes(repeated)
    if original_hashes != repeated_hashes:
        changed = sorted(
            name for name in EVALUATION_FILES if original_hashes[name] != repeated_hashes[name]
        )
        raise ValueError(f"repeated evaluation is not byte-identical: {', '.join(changed)}")
    return original_hashes


def _require_exact_scenes(values: Mapping[str, Any]) -> None:
    if set(values) != set(REPLICA8_SCENES):
        raise ValueError("Replica scene set must exactly match the frozen Replica-8 manifest")


def _macro(
    scene_metrics: Mapping[str, Mapping[str, Any]],
    scene_ids: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {"scene_ids": list(scene_ids), "scene_count": len(scene_ids)}
    for family, names in METRIC_FAMILIES.items():
        family_result: dict[str, float] = {}
        for name in names:
            values: list[float] = []
            for scene in scene_ids:
                value = scene_metrics[scene].get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"OVIV2 metric {scene}/{name} must be numeric")
                normalized = float(value)
                if not math.isfinite(normalized):
                    raise ValueError(f"OVIV2 metric {scene}/{name} must be finite")
                values.append(normalized)
            family_result[name] = fmean(values)
        result[family] = family_result
    return result


def aggregate_oviv2_replica(
    scene_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require_exact_scenes(scene_metrics)
    return {
        "replica_8_compat": _macro(scene_metrics, REPLICA8_SCENES),
        "replica_7_heldout": _macro(scene_metrics, REPLICA7_SCENES),
    }


def validate_scene_run_manifests(
    scene_runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    _require_exact_scenes(scene_runs)
    for scene in REPLICA8_SCENES:
        run = scene_runs[scene]
        if run.get("method") != "OVIV2" or run.get("scene") != scene:
            raise ValueError(f"invalid OVIV2 run manifest for {scene}")
        if run.get("final_revision") != 200:
            raise ValueError(f"OVIV2 run {scene} must reach revision 200")
        if run.get("frame_selection", {}).get("sampled_frame_count") != 200:
            raise ValueError(f"OVIV2 run {scene} must contain exactly 200 frames")
    contract: dict[str, str] = {}
    for field in PROVENANCE_HASH_FIELDS:
        values = {str(scene_runs[scene].get(field) or "") for scene in REPLICA8_SCENES}
        if "" in values or len(values) != 1:
            label = field.replace("_", " ")
            raise ValueError(f"OVIV2 scene runs must share one non-empty {label}")
        contract[field] = values.pop()
    return contract
