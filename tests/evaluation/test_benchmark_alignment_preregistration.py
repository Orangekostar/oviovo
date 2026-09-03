from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REGISTRY = ROOT / "configs/evaluation/external_benchmark_sources.json"
SCORECARD = ROOT / "configs/evaluation/benchmark_suitability_pre_result.json"
LITERATURE_REPORT = (
    ROOT
    / "docs/superpowers/reports/2026-09-03-dynamic-mapping-literature-benchmark-audit.md"
)
DECISION_LEDGER = (
    ROOT
    / "docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md"
)

EXPECTED_BENCHMARKS = {
    "tesse_official",
    "tesse_current_diagonal",
    "3rscan",
    "panoptic_flat",
}
EXPECTED_DIMENSIONS = {
    "matches_current_state_claim",
    "online_causal_compatibility",
    "open_vocab_input_fairness",
    "short_term_dynamics_coverage",
    "long_term_change_coverage",
    "exact_cross_time_identity_gt",
    "change_type_gt",
    "geometry_current_surface_gt",
    "evaluator_error_provenance",
    "official_code_reproducibility",
    "baseline_ecosystem",
    "real_world_relevance",
}
REQUIRED_REPOSITORIES = {
    "MIT-SPARK/Khronos",
    "WaldJohannaU/3RScan",
    "GradientSpaces/rescene4d",
    "GradientSpaces/stmetrics",
    "ethz-asl/panoptic_mapping",
    "VladimirYugay/GaME",
    "katadam/ObjectsCanMove",
    "BJHYZJ/DovSG",
    "MIT-SPARK/DAAAM",
}
REQUIRED_LEDGER_FIELDS = {
    "Question",
    "Evidence before",
    "Frozen source",
    "Frozen protocol",
    "Command",
    "Result",
    "Deviation from paper",
    "Decision",
    "What this rules in/out",
    "Commit",
    "Artifacts",
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_external_source_registry_freezes_required_repositories() -> None:
    payload = _load_json(SOURCE_REGISTRY)

    assert payload["status"] == "FROZEN"
    assert payload["retrieval_date"] == "2026-09-03"
    sources = payload["sources"]
    assert REQUIRED_REPOSITORIES <= {source["repo"] for source in sources.values()}
    for source_id, source in sources.items():
        assert re.fullmatch(r"SRC-[A-Z0-9-]+", source_id)
        assert set(source) >= {
            "repo",
            "commit",
            "license",
            "role",
            "paper",
            "retrieval_date",
        }
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["retrieval_date"] == payload["retrieval_date"]

    khronos = sources["SRC-KHRONOS"]
    identities = khronos["evaluator_identities"]
    assert identities["rss_2024_release"]["commit"] == (
        "742227a88de8b2ac23ac54d719b321c3af88dc75"
    )
    assert identities["rss_2024_release"]["khronos_eval_file_count"] == 0
    assert identities["latest_public"]["commit"] == khronos["commit"]
    assert identities["current_tesse_artifact"]["status"] == (
        "UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY"
    )


def test_suitability_scorecard_is_complete_and_source_bound() -> None:
    payload = _load_json(SCORECARD)
    registry = _load_json(SOURCE_REGISTRY)
    report = LITERATURE_REPORT.read_text(encoding="utf-8")
    literature_source_ids = set(
        re.findall(r"^### (LIT-[A-Z0-9-]+)\b", report, flags=re.MULTILINE)
    )
    known_source_ids = set(registry["sources"]) | literature_source_ids

    assert payload["status"] == "FROZEN_PRE_RESULT"
    assert set(payload["dimensions"]) == EXPECTED_DIMENSIONS
    assert set(payload["benchmarks"]) == EXPECTED_BENCHMARKS
    for benchmark in payload["benchmarks"].values():
        assert set(benchmark["scores"]) == EXPECTED_DIMENSIONS
        for item in benchmark["scores"].values():
            assert item["score"] in {0, 1, 2}
            assert item["rationale"].strip()
            assert item["source_ids"]
            assert set(item["source_ids"]) <= known_source_ids


def test_literature_audit_covers_required_work_and_evidence_fields() -> None:
    report = LITERATURE_REPORT.read_text(encoding="utf-8")

    assert len(re.findall(r"^### LIT-[A-Z0-9-]+\b", report, re.MULTILINE)) >= 12
    for name in (
        "RIO / 3RScan",
        "Panoptic Multi-TSDFs",
        "Objects Can Move",
        "Khronos",
        "DovSG",
        "Gaussian Mapping for Evolving Scenes",
        "ReScene4D",
        "OASIS-Map",
        "Describe Anything Anywhere At Any Moment",
        "Consistent Instance Field",
        "SuperMap",
        "DynamicTHOR",
    ):
        assert name in report
    for field in (
        "Paper:",
        "Official code:",
        "Dataset:",
        "Sensor/input:",
        "Temporal regime:",
        "Online:",
        "Open-vocabulary:",
        "GT semantics:",
        "GT instances:",
        "Cross-time identity GT:",
        "Change-type GT:",
        "Geometry GT:",
        "Metrics:",
        "Error provenance:",
        "Baseline reproducibility:",
        "Match to CROVE claim:",
    ):
        assert report.count(field) >= 12


def test_b0_b1_ledger_entries_are_complete() -> None:
    ledger = DECISION_LEDGER.read_text(encoding="utf-8")

    for stage in ("B0", "B1"):
        match = re.search(
            rf"^## {stage}\b(?P<body>.*?)(?=^## B\d\b|\Z)",
            ledger,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None
        body = match.group("body")
        for field in REQUIRED_LEDGER_FIELDS:
            assert f"**{field}:**" in body


def test_preregistration_documents_do_not_publish_local_absolute_paths() -> None:
    for path in (SOURCE_REGISTRY, SCORECARD, LITERATURE_REPORT, DECISION_LEDGER):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "file://" not in text
