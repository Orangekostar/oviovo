from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "docs/superpowers/reports"
DECISION = REPORTS / "2026-09-03-benchmark-decision.md"
HANDOFF = REPORTS / "2026-09-03-benchmark-alignment-handoff.md"
LEDGER = (
    ROOT
    / "docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md"
)
SCORECARD = ROOT / "configs/evaluation/benchmark_suitability_pre_result.json"

ALLOWED_DECISIONS = {
    "KEEP_TESSE_HEADLINE",
    "KEEP_TESSE_HEADLINE_WITH_SPLIT_PROTOCOL",
    "DEMOTE_TESSE_TO_STRESS_TEST",
    "REPLACE_HEADLINE_DYNAMIC_SUITE",
    "BENCHMARK_OK_METHOD_WEAK",
}
REQUIRED_REPORTS = {
    "2026-09-03-khronos-protocol-reproduction.md",
    "2026-09-03-khronos-metric-parity.md",
    "2026-09-03-tesse-input-fairness.md",
    "2026-09-03-tesse-current-vs-full4d.md",
    "2026-09-03-3rscan-pilot.md",
    "2026-09-03-panoptic-flat-pilot.md",
    "2026-09-03-dynamic-mapping-literature-benchmark-audit.md",
    DECISION.name,
    HANDOFF.name,
}


def test_final_report_set_and_pre_result_scorecard_are_frozen() -> None:
    assert all((REPORTS / name).is_file() for name in REQUIRED_REPORTS)
    assert hashlib.sha256(SCORECARD.read_bytes()).hexdigest() == (
        "f4de8e629dacc3a8e13a1123636179fe3c20ec0a145bbcdf1aff87c64a607694"
    )


def test_handoff_answers_all_fifteen_questions_and_one_decision() -> None:
    handoff = HANDOFF.read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d+)\. ", handoff, flags=re.MULTILINE)
    assert headings == [str(index) for index in range(1, 16)]
    decision = DECISION.read_text(encoding="utf-8")
    chosen = {
        value
        for value in ALLOWED_DECISIONS
        if re.search(rf"\b{re.escape(value)}\b", decision)
    }
    assert chosen == {"KEEP_TESSE_HEADLINE_WITH_SPLIT_PROTOCOL"}


def test_final_reports_preserve_unavailable_evidence_and_pareto_axes() -> None:
    decision = DECISION.read_text(encoding="utf-8")
    handoff = HANDOFF.read_text(encoding="utf-8")
    for value in (
        "current object quality",
        "current-state semantic quality",
        "stale-state suppression",
        "dynamic identity",
        "background/geometry recovery",
    ):
        assert value in decision
    for status in (
        "BLOCKED_PAPER_ASSET",
        "INCONCLUSIVE_MISSING_CONDITION",
        "BLOCKED_DATASET_ACCESS",
        "NOT_RUN_HELD_OUT",
    ):
        assert status in handoff
    assert "unavailable, not zero" in decision
    assert not re.search(r"\b(?:TBD|TODO)\b|待补充", decision + handoff)


def test_decision_ledger_closes_b0_through_b9_with_complete_fields() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## (B\d+) ", text, flags=re.MULTILINE))
    assert [match.group(1) for match in matches] == [f"B{index}" for index in range(10)]
    required = (
        "**Question:**",
        "**Evidence before:**",
        "**Frozen source:**",
        "**Frozen protocol:**",
        "**Command:**",
        "**Result:**",
        "**Deviation from paper:**",
        "**Decision:**",
        "**What this rules in/out:**",
        "**Commit:**",
        "**Artifacts:**",
    )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start() : end]
        assert all(field in section for field in required), match.group(1)
