"""Display every restored source row, with diagnostic outcomes and no ROI picking."""

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
from scripts.evaluation.run_ovimap_native import _sha256, _atomic_json


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    folder = path(config["compact_output_root"]) / "figures"
    source = folder / "recovery_examples.source.csv"
    manifest = json.loads((folder / "recovery_examples.manifest.json").read_text())
    if _sha256(source) != manifest["source_csv_sha256"]:
        raise ValueError("recovery source CSV changed")
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != manifest["source_rows"]:
        raise ValueError("recovery display population changed")
    xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
    categories = np.array([r["category"] for r in rows])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
                        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                        "svg.fonttype": "none", "pdf.fonttype": 42,
                        "svg.hashsalt": "crove-recovery-examples-v1"})
    width_mm, height_mm = 183, 120
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.11, right=0.68, bottom=0.17, top=0.85)
    styles = [("GT_unmatched", "No nearby GT", "#b8b8b8", "."),
              ("retained_correct", "Retained correct", "#527d9c", "o"),
              ("improved", "Improved", "#218c5c", "^"),
              ("regressed", "Regressed", "#b77820", "v"),
              ("remaining_error", "Remaining error", "#ad4848", "x")]
    for key, label, color, marker in styles:
        mask = categories == key
        count = int(mask.sum())
        if count != manifest["categories"].get(key, 0):
            raise ValueError(f"recovery category count differs: {key}")
        ax.scatter(xyz[mask, 0], xyz[mask, 2], s=8, color=color, marker=marker,
                   linewidths=0.5, label=f"{label} ({count:,})")
    ax.set(xlabel="Source X (m)", ylabel="Source Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=7)
    fig.suptitle("All restored rows: native to frozen final readout", fontsize=10, y=0.96)
    fig.text(0.11, 0.89, "Apartment H2 − B3 · fixed X–Z projection · 2,939 source rows", fontsize=8)
    fig.text(0.11, 0.07, "Source-to-GT diagnostic only; not the headline mIoU projection.", fontsize=7)
    fig.text(0.11, 0.035, "Duplicate coordinates overlap. Counts are source rows, not independent samples.", fontsize=7)
    stem = folder / "recovery_examples"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    _atomic_json(stem.with_suffix(".alignment.json"), {
        "status": "NOT_APPLICABLE_SINGLE_PANEL", "axes_bounds": list(ax.get_position().bounds),
        "source_csv_sha256": _sha256(source), "plotted_source_rows": len(rows),
        "projection": "all original X,Z coordinates; no ROI cropping or jitter",
    })
    plt.close(fig)
    print(len(rows), "restored rows plotted", flush=True)


if __name__ == "__main__":
    main()
