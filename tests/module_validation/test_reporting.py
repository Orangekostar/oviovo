"""Machine matrices and exactly-five-table Markdown delivery."""

from __future__ import annotations

import json

from src.static_ovmap.module_validation.reporting import (
    ReleaseStatuses,
    build_method_matrix,
    render_handoff,
    render_results,
)


def test_results_have_exactly_five_principal_tables_and_json_safe_nulls() -> None:
    matrix = build_method_matrix(
        required_methods=("N0", "S_PAIRED", "G_QUALITY", "Q_GAIN"),
        measured={"N0": {"uap": 0.4, "miou": 0.5}},
        blocked={
            "S_PAIRED": "BLOCKED_INDEPENDENT_SCENES",
            "G_QUALITY": "BLOCKED_INDEPENDENT_SCENES",
            "Q_GAIN": "BLOCKED_QUERY_TARGET_SUPPORT",
        },
    )
    text = render_results(
        matrix,
        final_candidate="N0",
        science_status="INCONCLUSIVE_PREREQUISITES",
        confirmation_status="NOT_REQUIRED_NO_RETAINED_CANDIDATE",
        supporting_evidence=("Historical Room0 smoke only.",),
    )
    assert sum(text.count(f"## Table {letter}.") for letter in "ABCDE") == 5
    assert "NaN" not in text and "Infinity" not in text
    assert "INCONCLUSIVE_PREREQUISITES" in text
    assert "Historical Room0 smoke only." in text
    encoded = json.dumps([row.to_dict() for row in matrix], allow_nan=False)
    assert '"uap": null' in encoded


def test_handoff_keeps_five_status_dimensions_distinct() -> None:
    statuses = ReleaseStatuses(
        implementation="COMPLETE",
        experiment="BLOCKED_INDEPENDENT_SCENES",
        science="INCONCLUSIVE",
        confirmation="NOT_REQUIRED_NO_RETAINED_CANDIDATE",
        publication="LOCAL_ONLY",
    )
    text = render_handoff(
        statuses,
        branch="research/ovimap-module-validation-v1",
        commit="a" * 40,
        evidence_paths=("/mnt/shared/evidence.json",),
        next_action="Provide independent ScanNet scene assets.",
        reproduction_commands=("python run.py --phase bind",),
    )
    for value in vars(statuses).values():
        assert value in text
    assert "research/ovimap-module-validation-v1" in text
    assert "python run.py --phase bind" in text
