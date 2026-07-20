#!/usr/bin/env python3
"""Materialize deterministic contiguous RGB-D views for OVIV2 ScanNet frontend runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

from src.datasets.scannet200 import ScanNet200Dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def materialize_scene(
    manifest_path: str | Path,
    scene_id: str,
    output: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).absolute()
    manifest = _load_manifest(manifest_path)
    if manifest.get("dataset") != "ScanNet200":
        raise ValueError("manifest dataset must be ScanNet200")
    matches = [value for value in manifest.get("scenes", ()) if value.get("scene") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"scene {scene_id!r} is not uniquely defined")
    scene = matches[0]
    source_ids = tuple(int(value) for value in scene["source_frame_ids"])
    if type(scene.get("frame_count")) is not int or scene["frame_count"] != len(source_ids):
        raise ValueError("scene frame_count does not match source_frame_ids")
    input_hashes = {
        int(source_id): {
            role: scene["frame_inputs"][str(source_id)][f"{role}_sha256"]
            for role in ("color", "depth", "pose")
        }
        for source_id in source_ids
    }
    dataset = ScanNet200Dataset(
        scene["dataset_root"],
        source_frame_ids=source_ids,
        input_hashes=input_hashes,
        expected_image_shape=tuple(scene["image_shape"]),
        depth_scale=float(scene["depth_scale"]),
    )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        results = temporary / "results"
        results.mkdir()
        trajectory_lines: list[str] = []
        for cache_index, frame in enumerate(dataset):
            color_output = results / f"frame{cache_index:06d}.jpg"
            Image.fromarray(frame.rgb).save(
                color_output,
                format="JPEG",
                quality=95,
                subsampling=0,
            )
            shutil.copyfile(
                Path(scene["dataset_root"]) / "depth" / f"{frame.source_frame_id}.png",
                results / f"depth{cache_index:06d}.png",
            )
            trajectory_lines.append(
                " ".join(format(float(value), ".17g") for value in frame.pose.reshape(-1))
            )
        (temporary / "traj.txt").write_text(
            "\n".join(trajectory_lines) + "\n",
            encoding="utf-8",
        )
        output_files = {
            str(path.relative_to(temporary)): _sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        published = {
            "schema_version": 1,
            "method": "OVIV2",
            "dataset": "ScanNet200",
            "scene": scene_id,
            "frame_count": len(dataset),
            "source_frame_ids": list(source_ids),
            "image_shape": list(scene["image_shape"]),
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "output_files_sha256": output_files,
        }
        (temporary / "view_manifest.json").write_text(
            json.dumps(published, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return published
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    materialize_scene(args.manifest, args.scene, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
