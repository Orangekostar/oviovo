#!/usr/bin/env python3
"""Generate a T4 runtime summary from process collectors and neutral artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def _elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text)
    if match is None:
        raise ValueError(f"elapsed wall-clock time missing from {path}")
    values = [float(value) for value in match.group(1).split(":")]
    if len(values) == 2:
        return values[0] * 60.0 + values[1]
    if len(values) == 3:
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    raise ValueError(f"invalid elapsed wall-clock time in {path}")


def _peak_rss_gb(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match is None:
        raise ValueError(f"peak RSS missing from {path}")
    return int(match.group(1)) / 1e6


def _peak_gpu_gb(paths: list[Path]) -> float:
    samples = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields and not fields[0].startswith("#") and len(fields) >= 4:
                samples.append(float(fields[3]) / 1000.0)
    if not samples:
        raise ValueError("GPU dmon logs contain no samples")
    return max(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scene-metrics", type=Path, required=True)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--process-log", type=Path, required=True)
    parser.add_argument("--initialization-finished-at", required=True)
    parser.add_argument("--runtime-includes-initialization", action="store_true")
    parser.add_argument("--finalization-log", type=Path)
    parser.add_argument("--runtime-includes-finalization", action="store_true")
    parser.add_argument("--evaluation-log", type=Path, required=True)
    parser.add_argument("--gpu-dmon", type=Path, action="append", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = json.loads(args.scene_metrics.read_text(encoding="utf-8"))
    runtime = scene["runtime"]["metrics"]
    status = json.loads(args.run_status.read_text(encoding="utf-8"))
    query = json.loads(args.query.read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(status["started_at"])
    initialization_finished_at = datetime.fromisoformat(args.initialization_finished_at)
    initialization_s = (initialization_finished_at - started_at).total_seconds()
    if initialization_s < 0.0:
        raise ValueError("initialization finish precedes run start")

    frontend_s = float(runtime["frontend_s"])
    backend_s = float(runtime["backend_s"])
    maintenance_s = float(runtime["maintenance_s"])
    if args.runtime_includes_initialization:
        frontend_s = max(frontend_s - initialization_s, 0.0)
    if args.finalization_log is None:
        finalization_s = max(
            _elapsed_seconds(args.process_log)
            - initialization_s
            - frontend_s
            - backend_s
            - maintenance_s,
            0.0,
        )
    else:
        finalization_s = _elapsed_seconds(args.finalization_log)
        if args.runtime_includes_finalization:
            backend_s = max(backend_s - finalization_s, 0.0)

    frame_count = int(runtime["frame_count"])
    total_s_per_frame = (frontend_s + backend_s + maintenance_s) / frame_count
    payload = {
        "method": args.method,
        "scene_id": scene["scene_id"],
        "status": "VERIFIED",
        "metrics": {
            "initialization_s": initialization_s,
            "frontend_spf": frontend_s / frame_count,
            "backend_spf": backend_s / frame_count,
            "maintenance_spf": maintenance_s / frame_count,
            "total_spf": total_s_per_frame,
            "hz": 0.0 if total_s_per_frame == 0.0 else 1.0 / total_s_per_frame,
            "finalization_s": finalization_s,
            "query_p50_ms": float(query["query_p50_ms"]),
            "query_p95_ms": float(query["query_p95_ms"]),
            "peak_gpu_gb": _peak_gpu_gb(args.gpu_dmon),
            "peak_ram_gb": _peak_rss_gb(args.process_log),
            "final_map_mb": float(runtime["final_map_mb"]),
            "evaluation_io_s": _elapsed_seconds(args.evaluation_log),
        },
        "units": {
            "initialization_s": "seconds",
            "frontend_spf": "seconds/frame",
            "backend_spf": "seconds/frame",
            "maintenance_spf": "seconds/frame",
            "total_spf": "seconds/frame",
            "hz": "Hz",
            "finalization_s": "seconds",
            "query_p50_ms": "milliseconds",
            "query_p95_ms": "milliseconds",
            "peak_gpu_gb": "GB",
            "peak_ram_gb": "GB",
            "final_map_mb": "MB",
            "evaluation_io_s": "seconds",
        },
        "sources": {
            "scene_metrics": str(args.scene_metrics.resolve()),
            "run_status": str(args.run_status.resolve()),
            "process_log": str(args.process_log.resolve()),
            "evaluation_log": str(args.evaluation_log.resolve()),
            "gpu_dmon": [str(path.resolve()) for path in args.gpu_dmon],
            "query": str(args.query.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
