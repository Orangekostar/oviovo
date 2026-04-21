#!/usr/bin/env python3
"""Render visualization assets for a completed OVIOVO analysis run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.datasets import ReplicaRoom0Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Path to an output directory created by run_replica_room0_analysis.py",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "input" / "Datasets" / "Replica" / "room0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    vis_dir = analysis_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(analysis_dir / "summary.json")
    frame_metrics = load_json(analysis_dir / "frame_metrics.json")
    frame_dir = analysis_dir / "frames"
    frame_payloads = {
        int(path.stem.replace("frame", "")): load_json(path)
        for path in sorted(frame_dir.glob("frame*.json"))
    }
    lifecycle = load_json(analysis_dir / "object_lifecycle_summary.json")["objects"]
    dataset = ReplicaRoom0Dataset(args.dataset_root)

    counts_path = vis_dir / "counts_over_time.png"
    births_path = vis_dir / "instance_births_over_time.png"
    lifecycle_path = vis_dir / "object_lifecycle_hist.png"
    keyframes_path = vis_dir / "keyframes_contact_sheet.png"

    plot_counts_over_time(frame_metrics, frame_payloads, counts_path)
    plot_births_over_time(frame_metrics, births_path)
    plot_lifecycle_hist(lifecycle, lifecycle_path)
    render_keyframe_sheet(dataset, frame_metrics, frame_payloads, keyframes_path)

    manifest = {
        "counts_over_time_png": str(counts_path),
        "instance_births_over_time_png": str(births_path),
        "object_lifecycle_hist_png": str(lifecycle_path),
        "keyframes_contact_sheet_png": str(keyframes_path),
        "summary_json": str(analysis_dir / "summary.json"),
        "analysis_dir": str(analysis_dir),
        "proposal_backend": summary["proposal_backend"]["active_backend"],
    }
    manifest_path = vis_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Saved visualization assets:")
    print(f"- {counts_path}")
    print(f"- {births_path}")
    print(f"- {lifecycle_path}")
    print(f"- {keyframes_path}")
    print(f"- {manifest_path}")


def plot_counts_over_time(
    frame_metrics: list[dict[str, Any]],
    frame_payloads: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    frame_ids = [item["frame_id"] for item in frame_metrics]
    total_objects = [item["total_object_count"] for item in frame_metrics]
    active_objects = [item["active_object_count"] for item in frame_metrics]
    dormant_objects = [
        frame_payloads[item["frame_id"]]["objects"]["state_counts"].get("dormant", 0)
        for item in frame_metrics
    ]
    inactive_objects = [
        frame_payloads[item["frame_id"]]["objects"]["state_counts"].get("inactive", 0)
        for item in frame_metrics
    ]
    raw_proposals = [item["raw_proposal_count"] for item in frame_metrics]
    matched_patches = [item["matched_patch_count"] for item in frame_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(frame_ids, total_objects, label="total objects", linewidth=2.2, color="#0d3b66")
    axes[0].plot(frame_ids, active_objects, label="active", linewidth=1.8, color="#2a9d8f")
    axes[0].plot(frame_ids, dormant_objects, label="dormant", linewidth=1.8, color="#e9c46a")
    axes[0].plot(frame_ids, inactive_objects, label="inactive", linewidth=1.8, color="#e76f51")
    axes[0].set_ylabel("object count")
    axes[0].set_title("Object Counts and States Over Time")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(frame_ids, raw_proposals, label="raw proposals", linewidth=1.8, color="#6d597a")
    axes[1].plot(frame_ids, matched_patches, label="matched patches", linewidth=1.8, color="#355070")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("count")
    axes[1].set_title("Frontend Load vs Successful Associations")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_births_over_time(frame_metrics: list[dict[str, Any]], output_path: Path) -> None:
    frame_ids = [item["frame_id"] for item in frame_metrics]
    new_objects = np.asarray([item["new_object_count"] for item in frame_metrics], dtype=np.int32)
    cumulative_births = np.cumsum(new_objects)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].bar(frame_ids, new_objects, width=1.0, color="#d62828")
    axes[0].set_ylabel("new objects")
    axes[0].set_title("Instance Birth Events Per Frame")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].plot(frame_ids, cumulative_births, linewidth=2.2, color="#003049")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("cumulative births")
    axes[1].set_title("Cumulative Instance Creation")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lifecycle_hist(lifecycle: list[dict[str, Any]], output_path: Path) -> None:
    lifespans = [len(item["frames_present"]) for item in lifecycle]
    update_spans = [len(item["frames_updated"]) for item in lifecycle]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lifespans, bins=20, color="#588157", edgecolor="white")
    axes[0].set_title("Object Lifespan Histogram")
    axes[0].set_xlabel("frames present")
    axes[0].set_ylabel("object count")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].hist(update_spans, bins=20, color="#bc6c25", edgecolor="white")
    axes[1].set_title("Update Frequency Histogram")
    axes[1].set_xlabel("frames updated")
    axes[1].set_ylabel("object count")
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_keyframe_sheet(
    dataset: ReplicaRoom0Dataset,
    frame_metrics: list[dict[str, Any]],
    frame_payloads: dict[int, dict[str, Any]],
    output_path: Path,
) -> None:
    selected = select_keyframes(frame_metrics)
    cards: list[Image.Image] = []
    font = ImageFont.load_default()

    for frame_id in selected:
        frame = dataset[frame_id]
        payload = frame_payloads[frame_id]
        metric = next(item for item in frame_metrics if item["frame_id"] == frame_id)

        rgb = Image.fromarray(frame.rgb).convert("RGB").resize((480, 272))
        card = Image.new("RGB", (480, 360), color=(250, 247, 242))
        card.paste(rgb, (0, 0))

        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 272, 479, 359), fill=(245, 239, 230))
        lines = [
            f"frame {frame_id}",
            f"raw={metric['raw_proposal_count']} merged={metric['merged_proposal_count']} refined={metric['refined_proposal_count']}",
            f"matched={metric['matched_patch_count']} new_obj={metric['new_object_count']} total={metric['total_object_count']}",
            f"active_set={metric['active_set_candidate_count']} semantic={metric['semantic_updated_count']}",
            "states: "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(payload["objects"]["state_counts"].items())
            ),
        ]
        y = 280
        for line in lines:
            draw.text((12, y), line, fill=(25, 25, 25), font=font)
            y += 16
        cards.append(card)

    cols = 2
    rows = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 360), color=(255, 255, 255))
    for index, card in enumerate(cards):
        x = (index % cols) * 480
        y = (index // cols) * 360
        sheet.paste(card, (x, y))
    sheet.save(output_path)


def select_keyframes(frame_metrics: list[dict[str, Any]]) -> list[int]:
    top_birth_frames = sorted(
        frame_metrics,
        key=lambda item: (-item["new_object_count"], item["frame_id"]),
    )[:4]
    evenly_spaced = [frame_metrics[index]["frame_id"] for index in [0, 49, 99, 149, 199]]
    selected = {item["frame_id"] for item in top_birth_frames}
    selected.update(evenly_spaced)
    return sorted(selected)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
