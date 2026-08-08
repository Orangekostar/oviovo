#!/usr/bin/env python3
"""Summarize fast-eval outputs for a Replica all-scenes batch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_replica_all_scenes_fast_eval import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_ROOT,
    experiment_name_for_scene,
    selected_scene_names,
)

METRIC_KEYS = ("miou", "fmiou", "macc", "fmacc")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_number(payload: dict[str, Any], key: str) -> int | None:
    value = _number(payload, key)
    return None if value is None else int(value)


def _required_number(payload: dict[str, Any], key: str, path: Path) -> float:
    value = _number(payload, key)
    if value is None:
        raise ValueError(f"missing numeric field '{key}' in {path}")
    return value


def _find_run_report(run_root: Path, scene: str) -> Path | None:
    for path in (run_root / scene / "run_report.json", run_root / "room0" / "run_report.json"):
        if path.exists():
            return path
    return None


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def validate_scene_completeness(
    run_root: Path,
    scene: str,
    *,
    expected_frames: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_root = Path(run_root)
    status_path = run_root / "status.json"
    mapping_path = run_root / "mapping_timer_result.json"
    export_path = run_root / "export_eval_timer_result.json"
    results_path = run_root / "replica" / "results.json"
    report_path = _find_run_report(run_root, scene)
    required_paths = {
        "status": status_path,
        "mapping timer": mapping_path,
        "export/eval timer": export_path,
        "Replica results": results_path,
    }
    missing = [label for label, path in required_paths.items() if not path.exists()]
    if report_path is None:
        missing.append("run report")
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)}")

    status = _load_json(status_path)
    mapping = _load_json(mapping_path)
    export = _load_json(export_path)
    results = _load_json(results_path)
    report = _load_json(report_path)
    if status.get("status") != "complete":
        raise ValueError(f"status is not complete: {status.get('status')!r}")

    processed_frames = _int_number(mapping, "processed_frames")
    if processed_frames is not None and processed_frames != int(expected_frames):
        raise ValueError(f"processed_frames {processed_frames} != expected {int(expected_frames)}")

    frame_metrics_path = run_root / scene / "frame_metrics.jsonl"
    if frame_metrics_path.exists():
        frame_metrics_count = _line_count(frame_metrics_path)
        if frame_metrics_count != int(expected_frames):
            raise ValueError(
                f"frame_metrics.jsonl line count {frame_metrics_count} != expected {int(expected_frames)}"
            )

    return mapping, export, results, report


def _prefetch_summary(mapping: dict[str, Any]) -> str:
    hits = int(_number(mapping, "prefetch_hit_count") or 0)
    submitted = int(_number(mapping, "prefetch_submitted_count") or 0)
    misses = int(_number(mapping, "prefetch_miss_count") or 0)
    return f"{hits}/{submitted} hits, {misses} misses"


def load_scene_summary(run_root: Path, scene: str, *, expected_frames: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    mapping_path = run_root / "mapping_timer_result.json"
    export_path = run_root / "export_eval_timer_result.json"
    results_path = run_root / "replica" / "results.json"
    if expected_frames is None:
        mapping = _load_json(mapping_path)
        export = _load_json(export_path)
        results = _load_json(results_path)
        report_path = _find_run_report(run_root, scene)
        if report_path is None:
            raise FileNotFoundError("missing run report")
        report = _load_json(report_path)
    else:
        mapping, export, results, report = validate_scene_completeness(
            run_root,
            scene,
            expected_frames=expected_frames,
        )

    objects = _int_number(export, "final_object_count")
    if objects is None:
        objects = _int_number(report, "final_object_count")
    dense_points = _int_number(export, "dense_geometry_point_count")
    if dense_points is None:
        dense_points = _int_number(report, "dense_geometry_point_count")
    if objects is None:
        raise ValueError("missing final_object_count in export timer and run report")
    if dense_points is None:
        raise ValueError("missing dense_geometry_point_count in export timer and run report")

    summary: dict[str, Any] = {
        "scene": scene,
        "run_root": str(run_root),
        "miou": _required_number(results, "miou", results_path),
        "fmiou": _required_number(results, "fmiou", results_path),
        "macc": _required_number(results, "macc", results_path),
        "fmacc": _required_number(results, "fmacc", results_path),
        "tpf": _required_number(mapping, "sec_per_frame", mapping_path),
        "mapping_sec": _required_number(mapping, "mapping_loop_sec_excluding_init_and_final_outputs", mapping_path),
        "export_sec": _required_number(export, "export_eval_sec", export_path),
        "objects": objects,
        "dense_points": dense_points,
        "prefetch": _prefetch_summary(mapping),
        "prefetch_hit_count": int(_number(mapping, "prefetch_hit_count") or 0),
        "prefetch_submitted_count": int(_number(mapping, "prefetch_submitted_count") or 0),
        "prefetch_miss_count": int(_number(mapping, "prefetch_miss_count") or 0),
        "processed_frames": _int_number(mapping, "processed_frames"),
        "frame_stride": _int_number(mapping, "frame_stride"),
        "requested_frames": _int_number(mapping, "requested_frames"),
    }
    return summary


def _mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _object_stats(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(scene["objects"]) for scene in scenes if scene.get("objects") is not None]
    if not counts:
        return {"min": None, "max": None, "mean": None, "total": 0}
    return {"min": min(counts), "max": max(counts), "mean": float(mean(counts)), "total": sum(counts)}


def summarize_batch(
    *,
    batch_name: str,
    scenes: Sequence[str],
    output_root: Path,
    frame_stride: int,
    num_frames: int,
) -> dict[str, Any]:
    scene_summaries: list[dict[str, Any]] = []
    missing_scenes: list[dict[str, str]] = []
    failed_scenes: list[dict[str, str]] = []
    output_root = Path(output_root)

    for scene in scenes:
        run_root = output_root / experiment_name_for_scene(
            batch_name,
            scene,
            frame_stride=frame_stride,
            num_frames=num_frames,
        )
        if not run_root.exists():
            missing_scenes.append({"scene": scene, "reason": "missing run root"})
            continue
        try:
            scene_summaries.append(load_scene_summary(run_root, scene, expected_frames=num_frames))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            failed_scenes.append({"scene": scene, "reason": str(exc)})

    if not scene_summaries:
        raise RuntimeError(f"No summarizable scenes found for batch '{batch_name}' under {output_root}")

    aggregate = {
        "macro_average": {key: _mean([float(scene[key]) for scene in scene_summaries]) for key in METRIC_KEYS},
        "mean_tpf": _mean([float(scene["tpf"]) for scene in scene_summaries]),
        "total_mapping_sec": sum(float(scene["mapping_sec"]) for scene in scene_summaries),
        "total_export_eval_sec": sum(float(scene["export_sec"]) for scene in scene_summaries),
        "object_count_stats": _object_stats(scene_summaries),
        "scene_count": len(scene_summaries),
        "requested_scene_count": len(scenes),
    }
    return {
        "batch_name": batch_name,
        "output_root": str(output_root),
        "frame_stride": int(frame_stride),
        "num_frames": int(num_frames),
        "scenes": scene_summaries,
        "aggregate": aggregate,
        "missing_scenes": missing_scenes,
        "failed_scenes": failed_scenes,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# Replica Batch Summary: {summary['batch_name']}",
        "",
        "| scene | mIoU | f-mIoU | mAcc | f-mAcc | TPF | mapping_sec | export_sec | objects | dense_points | prefetch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scene in summary["scenes"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(scene["scene"]),
                    _fmt(scene.get("miou")),
                    _fmt(scene.get("fmiou")),
                    _fmt(scene.get("macc")),
                    _fmt(scene.get("fmacc")),
                    _fmt(scene.get("tpf")),
                    _fmt(scene.get("mapping_sec")),
                    _fmt(scene.get("export_sec")),
                    _fmt(scene.get("objects")),
                    _fmt(scene.get("dense_points")),
                    str(scene.get("prefetch", "")),
                ]
            )
            + " |"
        )

    aggregate = summary["aggregate"]
    macro = aggregate["macro_average"]
    object_stats = aggregate["object_count_stats"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Scenes summarized: {aggregate['scene_count']} / {aggregate['requested_scene_count']}",
            f"- Macro mIoU/f-mIoU/mAcc/f-mAcc: {_fmt(macro['miou'])} / {_fmt(macro['fmiou'])} / {_fmt(macro['macc'])} / {_fmt(macro['fmacc'])}",
            f"- Mean TPF: {_fmt(aggregate['mean_tpf'])}",
            f"- Total mapping sec: {_fmt(aggregate['total_mapping_sec'])}",
            f"- Total export/eval sec: {_fmt(aggregate['total_export_eval_sec'])}",
            f"- Object counts min/mean/max/total: {_fmt(object_stats['min'])} / {_fmt(object_stats['mean'])} / {_fmt(object_stats['max'])} / {_fmt(object_stats['total'])}",
        ]
    )
    if summary["missing_scenes"]:
        lines.extend(["", "## Missing"])
        lines.extend(f"- {item['scene']}: {item['reason']}" for item in summary["missing_scenes"])
    if summary["failed_scenes"]:
        lines.extend(["", "## Incomplete"])
        lines.extend(f"- {item['scene']}: {item['reason']}" for item in summary["failed_scenes"])
    return "\n".join(lines) + "\n"


def write_summary(summary: dict[str, Any], summary_root: Path) -> tuple[Path, Path]:
    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    json_path = summary_root / "summary.json"
    markdown_path = summary_root / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(summary), encoding="utf-8")
    return json_path, markdown_path


def _repo_relative(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names to summarize. Defaults to all scenes.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.scenes is not None:
        try:
            args.scenes = selected_scene_names(args.scenes)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenes = selected_scene_names(args.scenes)
    output_root = _repo_relative(args.output_root)
    summary_root = _repo_relative(args.summary_root) if args.summary_root is not None else output_root / args.batch_name
    try:
        summary = summarize_batch(
            batch_name=args.batch_name,
            scenes=scenes,
            output_root=output_root,
            frame_stride=args.frame_stride,
            num_frames=args.num_frames,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json_path, markdown_path = write_summary(summary, summary_root)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    incomplete_count = len(summary["missing_scenes"]) + len(summary["failed_scenes"])
    if incomplete_count:
        print(f"Missing/incomplete scenes: {incomplete_count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
