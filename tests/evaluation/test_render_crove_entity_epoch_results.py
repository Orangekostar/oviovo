from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.render_crove_entity_epoch_results import (
    render_publication_figure,
    write_compact_artifact_index,
)


def test_render_publication_figure_writes_vector_preview_and_source_data(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "config_id": "H1_B3",
            "variant_id": "D1_B3",
            "mean_current_miou": 0.20,
            "mean_surface_precision_at_5cm": 0.70,
            "mean_surface_recall_at_5cm": 0.30,
            "mean_ghost": 0.0,
        },
        {
            "config_id": "H2_INHERIT",
            "variant_id": "D2_INHERIT",
            "mean_current_miou": 0.19,
            "mean_surface_precision_at_5cm": 0.71,
            "mean_surface_recall_at_5cm": 0.32,
            "mean_ghost": 0.0,
        },
    ]
    xyz = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    baseline = np.asarray([True, False, True, True])
    updated = np.asarray([True, True, True, True])

    outputs = render_publication_figure(
        rows=rows,
        xyz=xyz,
        baseline_current_valid=baseline,
        updated_current_valid=updated,
        output_root=tmp_path,
        source_bindings={"aggregate_metrics": "aggregate_metrics.json"},
        correct_recovery_numerator=1,
        correct_recovery_denominator=2,
    )

    assert {path.suffix for path in outputs} == {".json", ".pdf", ".png", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    svg_lines = (
        (tmp_path / "entity_epoch_dev_summary.svg")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert all(line == line.rstrip() for line in svg_lines)
    source = json.loads((tmp_path / "entity_epoch_dev_summary.json").read_text())
    assert source["recovered_source_row_count"] == 1
    assert source["correct_recovery"] == {"denominator": 2, "numerator": 1}
    assert source["source_bindings"] == {"aggregate_metrics": "aggregate_metrics.json"}

    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs
    }
    rerendered = render_publication_figure(
        rows=rows,
        xyz=xyz,
        baseline_current_valid=baseline,
        updated_current_valid=updated,
        output_root=tmp_path,
        source_bindings={"aggregate_metrics": "aggregate_metrics.json"},
        correct_recovery_numerator=1,
        correct_recovery_denominator=2,
    )
    assert first_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in rerendered
    }

    index = write_compact_artifact_index(tmp_path)
    assert index["artifact_count"] == 4
    assert {row["path"] for row in index["artifacts"]} == {
        "entity_epoch_dev_summary.json",
        "entity_epoch_dev_summary.pdf",
        "entity_epoch_dev_summary.png",
        "entity_epoch_dev_summary.svg",
    }
