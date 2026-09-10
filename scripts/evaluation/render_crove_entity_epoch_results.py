#!/usr/bin/env python3
# ruff: noqa: I001
"""Render source-bound CROVE entity-episode development evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "H0_T1": "T1 only",
    "H1_B3": "B3",
    "H2_INHERIT": "Inherit",
    "H3_GEOM": "+G1",
    "H4_RESCENE_ID": "R ID-only",
    "H5_RESCENE_EPOCH": "+ReScene",
    "H6_MEMORY_LAST": "Memory-last",
    "H7_MEMORY_BANK": "Memory-bank",
}
COLORS = {
    "H0_T1": "#A6A6A6",
    "H1_B3": "#222222",
    "H2_INHERIT": "#0072B2",
    "H3_GEOM": "#56B4E9",
    "H4_RESCENE_ID": "#E69F00",
    "H5_RESCENE_EPOCH": "#D55E00",
    "H6_MEMORY_LAST": "#009E73",
    "H7_MEMORY_BANK": "#009E73",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_compact_artifact_index(root: Path) -> dict[str, object]:
    """Bind every compact evidence file without making the index recursive."""

    destination = root / "compact_artifact_index.json"
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != destination
    ]
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(destination, payload)
    return payload


def _densest_recovery_crop(
    xyz: np.ndarray,
    recovered: np.ndarray,
    *,
    voxel_size_m: float = 0.10,
    crop_radius_m: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    recovered_xyz = xyz[recovered]
    if not len(recovered_xyz):
        center = np.mean(xyz, axis=0)
    else:
        voxels = np.floor(recovered_xyz / voxel_size_m).astype(np.int64)
        keys, counts = np.unique(voxels, axis=0, return_counts=True)
        densest = keys[int(np.argmax(counts))]
        center = (densest.astype(np.float64) + 0.5) * voxel_size_m
    crop = np.all(np.abs(xyz - center) <= crop_radius_m, axis=1)
    return np.asarray(center, dtype=np.float64), crop


def render_publication_figure(
    *,
    rows: Sequence[Mapping[str, object]],
    xyz: np.ndarray,
    baseline_current_valid: np.ndarray,
    updated_current_valid: np.ndarray,
    output_root: Path,
    source_bindings: Mapping[str, object],
    correct_recovery_numerator: int,
    correct_recovery_denominator: int,
) -> tuple[Path, ...]:
    """Render the DEV comparison and a real canonical-map recovery crop."""

    points = np.asarray(xyz, dtype=np.float32)
    baseline = np.asarray(baseline_current_valid, dtype=np.bool_)
    updated = np.asarray(updated_current_valid, dtype=np.bool_)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if baseline.shape != (len(points),) or updated.shape != (len(points),):
        raise ValueError("validity masks must align with xyz")
    if not rows:
        raise ValueError("at least one aggregate metric row is required")
    row_by_id = {str(row["config_id"]): row for row in rows}
    if "H1_B3" not in row_by_id:
        raise ValueError("H1_B3 is required as the comparison anchor")
    ordered = [row for row in rows if str(row["config_id"]) in LABELS]
    ordered.sort(key=lambda row: tuple(LABELS).index(str(row["config_id"])))

    recovered = ~baseline & updated
    center, crop = _densest_recovery_crop(points, recovered)
    context = crop & baseline
    context_indices = np.flatnonzero(context)
    if len(context_indices) > 50_000:
        stride = math.ceil(len(context_indices) / 50_000)
        context_indices = context_indices[::stride]
    recovered_indices = np.flatnonzero(crop & recovered)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "svg.hashsalt": "crove-entity-epoch-dynamics-v2",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(10.8, 3.25), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.35, 1.35, 1.0))

    axis_a = figure.add_subplot(grid[0, 0])
    names = [str(row["config_id"]) for row in ordered]
    miou = [float(row["mean_current_miou"]) for row in ordered]
    positions = np.arange(len(ordered))
    bars = axis_a.bar(
        positions,
        miou,
        color=[COLORS[name] for name in names],
        edgecolor="#222222",
        linewidth=0.45,
    )
    for bar, value in zip(bars, miou, strict=True):
        axis_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            rotation=90,
        )
    axis_a.set_ylim(0.0, max(miou) * 1.22)
    axis_a.set_xticks(
        positions, [LABELS[name] for name in names], rotation=35, ha="right"
    )
    axis_a.set_ylabel("Current mIoU")
    axis_a.set_title("(a) B3 remains the DEV winner", loc="left", fontweight="bold")
    axis_a.spines[["top", "right"]].set_visible(False)
    axis_a.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    axis_a.set_axisbelow(True)

    axis_b = figure.add_subplot(grid[0, 1])
    baseline_row = row_by_id["H1_B3"]
    comparison = [
        row for row in ordered if str(row["config_id"]) not in {"H0_T1", "H1_B3"}
    ]
    delta_precision = [
        100.0
        * (
            float(row["mean_surface_precision_at_5cm"])
            - float(baseline_row["mean_surface_precision_at_5cm"])
        )
        for row in comparison
    ]
    delta_recall = [
        100.0
        * (
            float(row["mean_surface_recall_at_5cm"])
            - float(baseline_row["mean_surface_recall_at_5cm"])
        )
        for row in comparison
    ]
    comparison_positions = np.arange(len(comparison))
    width = 0.36
    axis_b.bar(
        comparison_positions - width / 2,
        delta_precision,
        width,
        color="#0072B2",
        label="Precision",
    )
    axis_b.bar(
        comparison_positions + width / 2,
        delta_recall,
        width,
        color="#E69F00",
        hatch="//",
        label="Recall",
    )
    axis_b.axhline(0.0, color="#222222", linewidth=0.7)
    axis_b.set_xticks(
        comparison_positions,
        [LABELS[str(row["config_id"])] for row in comparison],
        rotation=35,
        ha="right",
    )
    axis_b.set_ylabel("Surface metric change vs B3 (pp)")
    axis_b.set_title("(b) Local recovery raises recall", loc="left", fontweight="bold")
    axis_b.legend(frameon=False, ncols=2, loc="upper center")
    axis_b.spines[["top", "right"]].set_visible(False)
    axis_b.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    axis_b.set_axisbelow(True)

    axis_c = figure.add_subplot(grid[0, 2], projection="3d")
    if len(context_indices):
        axis_c.scatter(
            points[context_indices, 0],
            points[context_indices, 1],
            points[context_indices, 2],
            s=0.25,
            c="#A6A6A6",
            alpha=0.18,
            rasterized=True,
            depthshade=False,
        )
    if len(recovered_indices):
        axis_c.scatter(
            points[recovered_indices, 0],
            points[recovered_indices, 1],
            points[recovered_indices, 2],
            s=4.0,
            c="#0072B2",
            marker="o",
            alpha=0.95,
            depthshade=False,
            label="D2 restored",
        )
    axis_c.set_xlabel("x (m)", labelpad=-4)
    axis_c.set_ylabel("y (m)", labelpad=-4)
    axis_c.set_zlabel("z (m)", labelpad=-4)
    axis_c.tick_params(pad=-2)
    axis_c.view_init(elev=22, azim=-58)
    axis_c.set_proj_type("ortho")
    axis_c.set_box_aspect((1, 1, 1))
    axis_c.set_title("(c) Real source-row correction", loc="left", fontweight="bold")
    if len(recovered_indices):
        axis_c.legend(frameon=False, loc="upper left", fontsize=7)

    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / "entity_epoch_dev_summary"
    source_path = stem.with_suffix(".json")
    source_payload = {
        "schema_version": 1,
        "status": "PASS",
        "claim_boundary": "state_recovery_without_current_miou_gain",
        "rows": [dict(row) for row in ordered],
        "recovered_source_row_count": int(np.count_nonzero(recovered)),
        "displayed_recovered_source_row_count": len(recovered_indices),
        "crop_center_xyz_m": center.tolist(),
        "crop_half_extent_m": 0.45,
        "correct_recovery": {
            "numerator": int(correct_recovery_numerator),
            "denominator": int(correct_recovery_denominator),
        },
        "source_bindings": dict(source_bindings),
        "prediction_uses_ground_truth": False,
    }
    _write_json(source_path, source_payload)
    outputs = [source_path]
    for suffix, options in (
        (
            ".png",
            {"dpi": 300, "metadata": {"Software": "CROVE figure renderer"}},
        ),
        (
            ".pdf",
            {
                "metadata": {
                    "Creator": "CROVE figure renderer",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        (
            ".svg",
            {"metadata": {"Creator": "CROVE figure renderer", "Date": None}},
        ),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        if suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
        outputs.append(path)
    plt.close(figure)
    return tuple(outputs)


def _configured_path(value: object, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("configured path must be a non-empty string")
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    compact_root = _configured_path(config["compact_output_root"], repo_root=repo_root)
    pair = config["splits"]["dev"]["pairs"][0]
    run_root = _configured_path(config["run_root"], repo_root=repo_root)
    pair_id = str(pair["pair_id"])
    with (compact_root / "metrics_per_pair.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        metric_rows = list(csv.DictReader(stream))
    rows = [
        {
            "config_id": row["config_id"],
            "variant_id": row["variant_id"],
            "mean_current_miou": float(row["current_miou"]),
            "mean_surface_precision_at_5cm": float(row["surface_precision_at_5cm"]),
            "mean_surface_recall_at_5cm": float(row["surface_recall_at_5cm"]),
            "mean_ghost": float(row["ghost"]),
        }
        for row in metric_rows
        if row.get("status") == "PASS"
    ]
    current_map_root = _configured_path(pair["current_map_root"], repo_root=repo_root)
    with np.load(
        current_map_root / "current_surface.npz", allow_pickle=False
    ) as arrays:
        xyz = arrays["vertices_xyz"]
        baseline = arrays["current_valid"]
    d2_root = (
        run_root
        / "dev"
        / "pairs"
        / pair_id
        / "hypotheses"
        / "H2_INHERIT"
        / "D2_INHERIT"
    )
    with np.load(
        d2_root / "entity_epoch_state" / "entity_epoch_state.npz",
        allow_pickle=False,
    ) as arrays:
        updated = arrays["current_valid_after"]
    d2_metrics = json.loads((d2_root / "metrics.json").read_text(encoding="utf-8"))
    correct = d2_metrics["diagnostics"]["correct_new_coverage_rate"]
    surface_manifest = json.loads(
        (current_map_root / "current_surface_manifest.json").read_text(encoding="utf-8")
    )
    state_manifest = json.loads(
        (d2_root / "entity_epoch_state" / "entity_epoch_state_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    render_publication_figure(
        rows=rows,
        xyz=xyz,
        baseline_current_valid=baseline,
        updated_current_valid=updated,
        output_root=compact_root / "figures",
        source_bindings={
            "aggregate_metrics": "../aggregate_metrics.json",
            "metrics_per_pair": "../metrics_per_pair.csv",
            "current_surface_sha256": surface_manifest["artifacts"][
                "current_surface.npz"
            ]["sha256"],
            "entity_epoch_state_sha256": state_manifest["state_arrays"]["sha256"],
        },
        correct_recovery_numerator=int(correct["numerator"]),
        correct_recovery_denominator=int(correct["denominator"]),
    )
    write_compact_artifact_index(compact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_publication_figure", "write_compact_artifact_index"]
