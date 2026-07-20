"""Strict ScanNet200-5 aggregation and evaluation artifact contracts for OVIV2."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from statistics import fmean
from typing import Any


SCANNET200_5_SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)
EVALUATION_FILES = (
    "metrics.json",
    "per_class_semantic.json",
    "class_agnostic_instance_ap.json",
    "gt_aligned_semantic_ids.npy",
    "gt_aligned_instance_ids.npy",
    "oviv2_instance_mesh_aligned.ply",
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_file_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    actual = {value.name for value in root.iterdir() if value.is_file()} if root.is_dir() else set()
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
        changed = [
            name
            for name in EVALUATION_FILES
            if original_hashes[name] != repeated_hashes[name]
        ]
        raise ValueError(f"repeated evaluation is not byte-identical: {', '.join(changed)}")
    return original_hashes


def aggregate_oviv2_scannet200(
    scene_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(scene_metrics) != set(SCANNET200_5_SCENES):
        raise ValueError("ScanNet scene set must exactly match the frozen five scenes")
    aggregate: dict[str, Any] = {
        "scene_ids": list(SCANNET200_5_SCENES),
        "scene_count": len(SCANNET200_5_SCENES),
    }
    for family, names in METRIC_FAMILIES.items():
        family_metrics: dict[str, float] = {}
        for name in names:
            values = []
            for scene in SCANNET200_5_SCENES:
                value = scene_metrics[scene].get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"OVIV2 metric {scene}/{name} must be numeric")
                normalized = float(value)
                if not math.isfinite(normalized):
                    raise ValueError(f"OVIV2 metric {scene}/{name} must be finite")
                values.append(normalized)
            family_metrics[name] = fmean(values)
        aggregate[family] = family_metrics
    return {"scannet200_5_heldout": aggregate}


def validate_scene_run_manifests(
    scene_runs: Mapping[str, Mapping[str, Any]],
    frame_counts: Mapping[str, int],
) -> dict[str, str]:
    if set(scene_runs) != set(SCANNET200_5_SCENES) or set(frame_counts) != set(
        SCANNET200_5_SCENES
    ):
        raise ValueError("ScanNet run and frame-count sets must exactly match five scenes")
    for scene in SCANNET200_5_SCENES:
        run = scene_runs[scene]
        expected_count = frame_counts[scene]
        if (
            run.get("method") != "OVIV2"
            or run.get("dataset_name") != "ScanNet200"
            or run.get("scene") != scene
        ):
            raise ValueError(f"invalid OVIV2 ScanNet run manifest: {scene}")
        if (
            run.get("final_revision") != expected_count
            or run.get("frame_selection", {}).get("sampled_frame_count")
            != expected_count
        ):
            raise ValueError(f"OVIV2 ScanNet run revision mismatch: {scene}")
    contract: dict[str, str] = {}
    for field in PROVENANCE_HASH_FIELDS:
        values = {str(scene_runs[scene].get(field) or "") for scene in SCANNET200_5_SCENES}
        if "" in values or len(values) != 1:
            raise ValueError(f"OVIV2 ScanNet runs must share one non-empty {field}")
        contract[field] = values.pop()
    return contract
