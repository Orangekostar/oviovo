#!/usr/bin/env python3
"""Freeze explicit RGB-D execution inputs for OVIV2 ScanNet200 Stage4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
import tempfile
from typing import Any

import numpy as np


SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, name: str) -> dict[str, Any]:
    _require_regular_file(path, name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_regular_file(path: Path, name: str) -> None:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{name} must be a regular non-symlink file: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")


def _indexed(directory: Path, suffix: str, role: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        _require_regular_file(path, role)
        if not path.stem.isdigit():
            raise ValueError(f"{role} frame inventory contains an invalid name: {path.name}")
        frame_id = int(path.stem)
        if frame_id in indexed:
            raise ValueError(f"{role} frame inventory contains a duplicate ID")
        indexed[frame_id] = path
    return indexed


def _valid_pose(path: Path) -> bool:
    pose = np.loadtxt(path, dtype=np.float64)
    return bool(
        pose.shape == (4, 4)
        and np.all(np.isfinite(pose))
        and np.isfinite(np.linalg.det(pose))
        and abs(np.linalg.det(pose)) > 1e-12
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
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


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    raw_manifest_path = args.raw_manifest.absolute()
    raw_manifest = _load_object(raw_manifest_path, "raw manifest")
    raw_scenes = {
        str(value.get("scene_id")): value for value in raw_manifest.get("scenes", ())
    }
    if tuple(raw_scenes) != SCENES:
        raise ValueError("raw manifest must contain the frozen five-scene order")

    classes_path = args.classes_json.absolute()
    classes_payload = _load_object(classes_path, "classes JSON")
    classes = classes_payload.get("classes")
    if not isinstance(classes, list) or len(classes) != 200:
        raise ValueError("classes JSON must contain exactly 200 classes")
    constants_path = args.official_constants.absolute()
    _require_regular_file(constants_path, "official constants")
    constants = runpy.run_path(str(constants_path))
    class_ids = [int(value) for value in constants["VALID_CLASS_IDS_200"]]
    official_classes = [str(value) for value in constants["CLASS_LABELS_200"]]
    if len(class_ids) != 200 or official_classes != classes:
        raise ValueError("classes JSON does not match official ScanNet200 constants")
    classes_txt = classes_path.with_suffix(".txt")
    _require_regular_file(classes_txt, "classes TXT")
    if classes_txt.read_text(encoding="utf-8").splitlines() != classes:
        raise ValueError("classes TXT order does not match classes JSON")

    scenes: list[dict[str, Any]] = []
    for scene in SCENES:
        exported = args.exported_root.resolve() / scene
        color = _indexed(exported / "color", ".jpg", "color")
        depth = _indexed(exported / "depth", ".png", "depth")
        pose = _indexed(exported / "pose", ".txt", "pose")
        if set(color) != set(depth) or set(color) != set(pose):
            raise ValueError(f"scene {scene} frame inventory does not match across RGB-D and pose")
        candidates = sorted(color)
        source_ids = [
            value
            for value in candidates
            if value % 10 == 0 and _valid_pose(pose[value])
        ]
        if not source_ids:
            raise ValueError(f"scene {scene} has no valid stride-10 frames")
        frame_inputs = {
            str(value): {
                "color_sha256": _sha256(color[value]),
                "depth_sha256": _sha256(depth[value]),
                "pose_sha256": _sha256(pose[value]),
            }
            for value in source_ids
        }
        raw_scene = raw_scenes[scene]
        metadata_record = raw_scene["files"]["metadata"]
        metadata_path = Path(metadata_record["path"]).absolute()
        _require_regular_file(metadata_path, "metadata")
        if _sha256(metadata_path) != metadata_record["sha256"]:
            raise ValueError(f"raw metadata hash mismatch for {scene}")
        gt_path = args.official_gt_root.resolve() / f"{scene}.ply"
        depth_intrinsic = exported / "intrinsic" / "intrinsic_depth.txt"
        color_intrinsic = exported / "intrinsic" / "intrinsic_color.txt"
        _require_regular_file(gt_path, "official GT")
        _require_regular_file(depth_intrinsic, "depth intrinsic")
        _require_regular_file(color_intrinsic, "color intrinsic")
        scenes.append(
            {
                "scene": scene,
                "dataset_root": str(exported),
                "source_frame_ids": source_ids,
                "frame_count": len(source_ids),
                "frame_inputs": frame_inputs,
                "depth_scale": 1000.0,
                "image_shape": [480, 640],
                "color_resize": "bilinear_1296x968_to_640x480",
                "depth_intrinsic_path": str(depth_intrinsic.resolve()),
                "depth_intrinsic_sha256": _sha256(depth_intrinsic),
                "color_intrinsic_path": str(color_intrinsic.resolve()),
                "color_intrinsic_sha256": _sha256(color_intrinsic),
                "metadata_path": str(metadata_path),
                "metadata_sha256": metadata_record["sha256"],
                "official_gt_path": str(gt_path),
                "official_gt_sha256": _sha256(gt_path),
                "split_role": "scannet200_5_heldout",
            }
        )

    payload = {
        "schema_version": 1,
        "manifest_id": "oviv2_scannet200_5_stage4_v1",
        "dataset": "ScanNet200",
        "split": "scannet200_5_heldout",
        "raw_manifest": {
            "path": str(raw_manifest_path),
            "sha256": _sha256(raw_manifest_path),
        },
        "frame_selection": {"start": 0, "stride": 10, "invalid_pose_policy": "exclude"},
        "vocabulary": {
            "name": "scannet200_official",
            "class_count": 200,
            "class_ids": class_ids,
            "classes": classes,
            "aliases": dict(classes_payload.get("aliases", {})),
            "source_path": str(classes_path),
            "source_sha256": _sha256(classes_path),
            "text_path": str(classes_txt.resolve()),
            "text_sha256": _sha256(classes_txt),
            "official_constants_path": str(constants_path),
            "official_constants_sha256": _sha256(constants_path),
        },
        "scenes": scenes,
    }
    _atomic_json(args.output.resolve(), payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--exported-root", type=Path, required=True)
    parser.add_argument("--official-gt-root", type=Path, required=True)
    parser.add_argument("--official-constants", type=Path, required=True)
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.replace:
        print(f"output already exists: {args.output}", file=sys.stderr)
        return 1
    try:
        freeze(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
