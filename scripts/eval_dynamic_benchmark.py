#!/usr/bin/env python3
"""Compare dynamic benchmark event_log against pipeline frame_metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_event_log(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    events = data.get("events", [])
    # Filter to benchmark events (those with detection_window)
    return [e for e in events if "detection_window" in e]


def load_frame_metrics(path: Path) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def evaluate(event_log: list[dict], metrics: list[dict]) -> dict:
    results = []
    hits = 0
    total = len(event_log)
    total_delay = 0

    for event in event_log:
        ghost_baseline = 0
        event_frame = int(event["frame"])
        expected = event.get("expected_judgment", "")
        window = event.get("detection_window", [event_frame + 50, event_frame + 200])
        ev_type = event.get("type", "")

        detected = False
        detect_frame = None

        for m in metrics:
            f = int(m.get("frame_id", 0))
            if f < window[0]:
                ghost_baseline = m.get("ghost_object_count", 0)
                continue
            if f > window[1]:
                break

            if ev_type == "remove":
                ghost_now = m.get("ghost_object_count", 0)
                if ghost_now > ghost_baseline:
                    detected = True
                    detect_frame = f
                    break
            elif ev_type == "move":
                moved_ids = m.get("moved_this_frame", [])
                if len(moved_ids) > 0:
                    detected = True
                    detect_frame = f
                    break
            elif ev_type == "add":
                new_ids = m.get("new_candidate_this_frame", [])
                if len(new_ids) > 0:
                    detected = True
                    detect_frame = f
                    break

        delay = (detect_frame - event_frame) if detected else None
        if detected:
            hits += 1
            if delay is not None:
                total_delay += delay

        results.append({
            "event": event,
            "detected": detected,
            "detect_frame": detect_frame,
            "delay_frames": delay,
        })

    # False positive check: count frames with ghost > 0 outside any detection window
    all_windows = [(e["detection_window"][0], e["detection_window"][1]) for e in event_log]
    false_positives = 0
    for m in metrics:
        f = int(m.get("frame_id", 0))
        ghost = m.get("ghost_object_count", 0)
        if ghost > 0:
            in_any_window = any(lo <= f <= hi for lo, hi in all_windows)
            if not in_any_window:
                false_positives += 1

    return {
        "total_events": total,
        "detected": hits,
        "missed": total - hits,
        "recall": hits / total if total > 0 else 0.0,
        "avg_delay_frames": total_delay / hits if hits > 0 else 0,
        "false_positive_frames": false_positives,
        "per_event": results,
    }


def format_report(report: dict) -> str:
    lines = []
    lines.append("Dynamic Benchmark Report — apt_0 2400f 12 events")
    lines.append("=" * 54)
    lines.append(f"{'Event':<6} {'Type':<8} {'Frame':<7} {'Result':<8} {'Delay'}")
    lines.append("-" * 54)

    for i, r in enumerate(report["per_event"]):
        ev = r["event"]
        status = "PASS" if r["detected"] else "MISS"
        delay = f"{r['delay_frames']}f" if r["detected"] else "--"
        lines.append(
            f"{i+1:<6} {ev.get('type',''):<8} {ev.get('frame',''):<7} {status:<8} {delay}"
        )

    lines.append("-" * 54)
    lines.append(f"Recall:    {report['detected']}/{report['total_events']} ({report['recall']:.1%})")
    lines.append(f"Avg delay: {report['avg_delay_frames']:.0f} frames")
    lines.append(f"False +:   {report['false_positive_frames']} frames")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic benchmark")
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--frame-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"))
    args = parser.parse_args()

    events = load_event_log(args.event_log)
    metrics = load_frame_metrics(args.frame_metrics)
    report = evaluate(events, metrics)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(format_report(report))


if __name__ == "__main__":
    main()
