"""Plot every scored room0 readout relative to S2, without confirmation data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    compact = path(config["compact_output_root"])
    source = compact / "all_method_results.json"
    report = json.loads(source.read_text())
    rows = sorted(
        [r for r in report["rows"] if r["case"] == "room0" and r["split"] == "dev"],
        key=lambda r: (r["family"], r["exact_implementation"]),
    )
    baseline = next(
        r["static_miou"] for r in rows if r["exact_implementation"] == "B_SEM_CROVE_S2"
    )
    values = 100 * np.array([r["static_miou"] - baseline for r in rows])
    assert np.isfinite(values).all()
    assert np.allclose(
        values / 100, [r["delta_vs_S2"] for r in rows], atol=1e-12, rtol=0
    )
    out = compact / "figures"
    out.mkdir(exist_ok=True)
    stem = out / "room0_semantic_delta"
    with stem.with_suffix(".source.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "family",
                "method",
                "miou",
                "S2_miou",
                "delta_percentage_points",
                "owner_changed",
                "result_sha256",
            ]
        )
        for row, delta in zip(rows, values):
            writer.writerow(
                [
                    row["family"],
                    row["exact_implementation"],
                    row["static_miou"],
                    baseline,
                    delta,
                    row["owner_changed"],
                    row["result_sha256"],
                ]
            )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    width_mm, height_mm = 183, 180
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.51, right=0.97, bottom=0.11, top=0.91)
    y = np.arange(len(rows))
    colors = ["#969696" if r["family"] == "BASELINE" else "#527D9C" for r in rows]
    ax.barh(y, values, height=0.56, color=colors, linewidth=0)
    ax.set_yticks(
        y, [f"{r['family']}  {r['exact_implementation']}" for r in rows], fontsize=7
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=7)
    ax.axvline(0, color="#3B3B3B", linewidth=0.8)
    ax.set_xlim(min(float(values.min()) - 2, -2), max(float(values.max()) + 2, 2))
    ax.set_xlabel("mIoU change versus S2 (percentage points)", labelpad=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.5)
    fig.suptitle("Room0 semantic readouts relative to S2", fontsize=10, y=0.98)
    fig.text(
        0.51,
        0.935,
        "DEV progress · one scene · no uncertainty estimate",
        fontsize=7,
        ha="left",
    )
    fig.text(
        0.04,
        0.03,
        "Gray: baseline controls. Blue: evaluated readouts. All scored room0 rows shown.",
        fontsize=7,
    )
    fig.canvas.draw()
    _atomic_json(
        stem.with_suffix(".alignment.json"),
        {
            "status": "NOT APPLICABLE",
            "reason": "single plot area",
            "plot_area_rectangle_figure_fraction": list(ax.get_position().bounds),
        },
    )
    fig.savefig(stem.with_suffix(".svg"))
    svg = stem.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    _atomic_json(
        stem.with_suffix(".manifest.json"),
        {
            "source_summary_sha256": _sha256(source),
            "source_total_rows": len(report["rows"]),
            "plotted_rows": len(rows),
            "exclusion_rule": "dynamic rows excluded from this static-scene question; no room0 exclusion",
            "S2_miou": baseline,
            "positive_delta_rows": int(np.count_nonzero(values > 1e-4)),
            "width_mm": 183,
            "height_mm": 180,
            "backend": "Python/matplotlib",
            "uncertainty": "none: one fixed DEV scene, no independent repetitions supplied",
            "status": "INTERMEDIATE_DEV_FIGURE",
            "confirmation_data_used": False,
        },
    )
    print(len(rows), "room0 readouts plotted", flush=True)


if __name__ == "__main__":
    main()
