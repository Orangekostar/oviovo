#!/usr/bin/env python3
"""Export trusted local baseline development outputs as neutral fixtures."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.baselines.adapters import adapt_conceptgraphs, adapt_dualmap, adapt_ovimap
from src.evaluation.baselines.artifacts import write_baseline_artifact
from src.evaluation.baselines.contracts import RuntimeBreakdown


def _load_pickle(path: Path) -> Any:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    with handle:
        return pickle.load(handle)


def _runtime(path: Path, *, source_label: str) -> RuntimeBreakdown:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_count = int(payload.get("frames_processed", payload.get("frames", 0)))
    total = float(payload.get("wall_time_total_sec", payload.get("total_s", 0.0)))
    return RuntimeBreakdown(
        frame_count=frame_count,
        frontend_s=total,
        source_labels={"frontend_s": source_label},
    )


def _dualmap(args: argparse.Namespace) -> None:
    objects = [_load_pickle(path) for path in sorted(args.map_dir.glob("*.pkl"))]
    class_names = {
        int(key): str(value)
        for key, value in json.loads(args.class_names.read_text(encoding="utf-8")).items()
    }
    artifact = adapt_dualmap(
        objects,
        class_id_names=class_names,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=_runtime(args.runtime, source_label="DualMap dataset runner wall time"),
    )
    write_baseline_artifact(artifact, args.output)


def _conceptgraphs(args: argparse.Namespace) -> None:
    payload = _load_pickle(args.input)
    artifact = adapt_conceptgraphs(
        payload,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=_runtime(args.runtime, source_label="ConceptGraphs mapping wall time"),
    )
    write_baseline_artifact(artifact, args.output)


def _sample_ovimap_vertices(
    path: Path,
    colors: set[tuple[int, int, int]],
    *,
    stride: int,
) -> dict[tuple[int, int, int], np.ndarray]:
    sampled: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {
        color: [] for color in colors
    }
    vertex_count: int | None = None
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError("OVI-MAP mesh has no vertex count")
        for index in range(vertex_count):
            fields = handle.readline().split()
            if len(fields) < 9:
                raise ValueError(f"invalid OVI-MAP vertex record at index {index}")
            if index % stride:
                continue
            color = (int(fields[6]), int(fields[7]), int(fields[8]))
            if color in sampled:
                sampled[color].append((float(fields[0]), float(fields[1]), float(fields[2])))
    return {
        color: np.asarray(points, dtype=np.float32).reshape(-1, 3)
        for color, points in sampled.items()
    }


def _ovimap(args: argparse.Namespace) -> None:
    instances = _load_pickle(args.instances)
    colors = {
        tuple(int(value) for value in np.asarray(instance["color"]).reshape(-1))
        for instance in instances.values()
    }
    points_by_color = _sample_ovimap_vertices(args.mesh, colors, stride=args.point_stride)
    artifact = adapt_ovimap(
        instances,
        points_by_color=points_by_color,
        scene_id=args.scene_id,
        timestamp=args.timestamp,
        upstream_commit=args.upstream_commit,
        runtime=RuntimeBreakdown(
            frame_count=args.frame_count,
            source_labels={"frontend_s": "not captured in historical development output"},
        ),
        protocol_notes=(f"Development fixture mesh vertices sampled at stride {args.point_stride}.",),
    )
    write_baseline_artifact(artifact, args.output)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-id", default="room0")
    parser.add_argument("--timestamp", type=float, default=1990.0)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="baseline", required=True)

    dualmap = subparsers.add_parser("dualmap")
    _common(dualmap)
    dualmap.add_argument("--map-dir", type=Path, required=True)
    dualmap.add_argument("--class-names", type=Path, required=True)
    dualmap.add_argument("--upstream-commit", default="157235ec49e6a1f439babbc571c4c02ad1f06aa9")
    dualmap.set_defaults(run=_dualmap)

    conceptgraphs = subparsers.add_parser("conceptgraphs")
    _common(conceptgraphs)
    conceptgraphs.add_argument("--input", type=Path, required=True)
    conceptgraphs.add_argument("--upstream-commit", default="93277a02bd89171f8121e84203121cf7af9ebb5d")
    conceptgraphs.set_defaults(run=_conceptgraphs)

    ovimap = subparsers.add_parser("ovimap")
    ovimap.add_argument("--scene-id", default="room0")
    ovimap.add_argument("--timestamp", type=float, default=1990.0)
    ovimap.add_argument("--frame-count", type=int, default=200)
    ovimap.add_argument("--mesh", type=Path, required=True)
    ovimap.add_argument("--instances", type=Path, required=True)
    ovimap.add_argument("--point-stride", type=int, default=20)
    ovimap.add_argument("--output", type=Path, required=True)
    ovimap.add_argument("--upstream-commit", default="58a804e2d7c82ba05a489eb071aba3367301fed8")
    ovimap.set_defaults(run=_ovimap)

    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
