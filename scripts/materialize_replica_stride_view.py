#!/usr/bin/env python3
"""Materialize deterministic, symlink-only Replica stride views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _build_summary(
    source: Path,
    *,
    start: int,
    stop: int,
    stride: int,
) -> tuple[dict[str, Any], list[str]]:
    if start < 0 or stop <= start or stride <= 0:
        raise ValueError("stride selection requires 0 <= start < stop and stride > 0")
    results = source / "results"
    trajectory = source / "traj.txt"
    if not results.is_dir() or not trajectory.is_file():
        raise FileNotFoundError(f"Replica source is missing results/ or traj.txt: {source}")
    trajectory_lines = trajectory.read_text(encoding="utf-8").splitlines()
    source_ids = list(range(start, stop, stride))
    if source_ids[-1] >= len(trajectory_lines):
        raise ValueError("stride selection exceeds source trajectory")

    frames: list[dict[str, Any]] = []
    for sampled_id, source_id in enumerate(source_ids):
        rgb = results / f"frame{source_id:06d}.jpg"
        depth = results / f"depth{source_id:06d}.png"
        if not rgb.is_file() or not depth.is_file():
            raise FileNotFoundError(f"missing Replica source frame {source_id}: {source}")
        frames.append(
            {
                "sampled_frame_id": sampled_id,
                "source_frame_id": source_id,
                "rgb_source": str(rgb.resolve()),
                "rgb_sha256": _sha256(rgb),
                "depth_source": str(depth.resolve()),
                "depth_sha256": _sha256(depth),
            }
        )
    summary = {
        "schema_version": 1,
        "dataset": "Replica",
        "source_scene_root": str(source.resolve()),
        "selection": {"start": start, "stop_exclusive": stop, "stride": stride},
        "frame_count": len(source_ids),
        "source_frame_ids": source_ids,
        "source_trajectory_sha256": _sha256(trajectory),
        "frames": frames,
    }
    sampled_poses = [trajectory_lines[source_id] for source_id in source_ids]
    return summary, sampled_poses


def _validate_existing(target: Path, summary: dict[str, Any], sampled_poses: list[str]) -> None:
    results = target / "results"
    if not results.is_dir() or not (target / "traj.txt").is_file():
        raise ValueError(f"existing stride view is incomplete: {target}")
    for frame in summary["frames"]:
        sampled_id = int(frame["sampled_frame_id"])
        expected = {
            results / f"frame{sampled_id:06d}.jpg": Path(frame["rgb_source"]),
            results / f"depth{sampled_id:06d}.png": Path(frame["depth_source"]),
        }
        for link, source in expected.items():
            if not link.is_symlink() or link.resolve() != source.resolve():
                raise ValueError(f"existing stride view link mismatch: {link}")
    if (target / "traj.txt").read_text(encoding="utf-8").splitlines() != sampled_poses:
        raise ValueError(f"existing stride view trajectory mismatch: {target / 'traj.txt'}")


def materialize(
    source: str | Path,
    target: str | Path,
    *,
    start: int,
    stop: int,
    stride: int,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    summary, sampled_poses = _build_summary(
        source_path,
        start=start,
        stop=stop,
        stride=stride,
    )
    if target_path.exists():
        _validate_existing(target_path, summary, sampled_poses)
        _atomic_json(target_path / "frame_manifest.json", summary)
        return summary

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.tmp-", dir=target_path.parent))
    try:
        results = temporary / "results"
        results.mkdir()
        for frame in summary["frames"]:
            sampled_id = int(frame["sampled_frame_id"])
            (results / f"frame{sampled_id:06d}.jpg").symlink_to(frame["rgb_source"])
            (results / f"depth{sampled_id:06d}.png").symlink_to(frame["depth_source"])
        (temporary / "traj.txt").write_text("\n".join(sampled_poses) + "\n", encoding="utf-8")
        _atomic_json(temporary / "frame_manifest.json", summary)
        os.replace(temporary, target_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/oviv2_replica8.json")
    parser.add_argument("--scene", action="append")
    args = parser.parse_args(argv)
    config = _load_json(args.config.resolve())
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    benchmark = _load_json(manifest_path)
    selected = set(args.scene or [item["scene"] for item in benchmark["scenes"]])
    known = {item["scene"] for item in benchmark["scenes"]}
    if not selected <= known:
        raise ValueError(f"unknown Replica scenes: {', '.join(sorted(selected - known))}")
    selection = benchmark["frame_selection"]
    source_root = Path(config["source_root"]).expanduser().resolve()
    view_root = Path(config["view_root"]).expanduser().resolve()
    summaries = []
    for item in benchmark["scenes"]:
        scene = item["scene"]
        if scene not in selected:
            continue
        summaries.append(
            materialize(
                source_root / scene,
                view_root / f"{scene}{config['view_suffix']}",
                start=int(selection["start"]),
                stop=int(selection["stop_exclusive"]),
                stride=int(selection["stride"]),
            )
        )
    print(json.dumps({"scenes": len(summaries), "frames": sum(x["frame_count"] for x in summaries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
