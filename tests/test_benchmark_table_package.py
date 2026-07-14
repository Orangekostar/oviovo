from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
MARKDOWN = PAPER_DIR / "benchmark_tables.md"
LATEX = PAPER_DIR / "benchmark_tables.tex"
REGISTRY = PAPER_DIR / "benchmark_tokens.tsv"
GENERATOR = ROOT / "tools" / "benchmark_tables.py"
LABELS = {
    "tab:static_mapping", "tab:dynamic_current_map", "tab:causal_ablation",
    "tab:online_efficiency", "tab:supp_current_query",
    "tab:supp_temporal_identity", "tab:supp_reliability",
}
FIELDS = [
    "token", "table", "method", "dataset", "split", "metric",
    "direction", "precision", "source_json", "json_pointer", "status", "note",
]
TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def rows():
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == FIELDS
        return list(reader)


def tokens(path):
    return set(TOKEN_RE.findall(path.read_text(encoding="utf-8")))


def test_files_exist():
    assert all(path.is_file() for path in (GENERATOR, MARKDOWN, LATEX, REGISTRY))


def test_four_main_and_three_supplementary_tables():
    md = MARKDOWN.read_text(encoding="utf-8")
    tex = LATEX.read_text(encoding="utf-8")
    assert len(re.findall(r"^## Table [1-4]:", md, re.MULTILINE)) == 4
    assert len(re.findall(r"^## Table S[1-3]:", md, re.MULTILINE)) == 3
    assert set(re.findall(r"\\label\{([^}]+)\}", tex)) == LABELS


def test_registry_schema_states_and_unique_tokens():
    registry = rows()
    names = [row["token"] for row in registry]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Z0-9_]+", name) for name in names)
    assert {row["status"] for row in registry} <= {"UNFILLED", "VERIFIED", "N/A"}
    assert {row["direction"] for row in registry} <= {"higher", "lower"}
    assert {row["precision"] for row in registry} <= {"2", "3"}


def test_non_na_tokens_match_both_outputs():
    expected = {row["token"] for row in rows() if row["status"] != "N/A"}
    assert tokens(MARKDOWN) == expected
    assert tokens(LATEX) == expected


def test_provenance_requirements():
    for row in rows():
        if row["status"] == "VERIFIED":
            assert row["source_json"] and row["json_pointer"]
        elif row["status"] == "N/A":
            assert row["note"]


def test_latex_style():
    tex = LATEX.read_text(encoding="utf-8")
    assert all(command in tex for command in ("\\toprule", "\\midrule", "\\bottomrule"))
    assert "\\resizebox" not in tex
    assert all("|" not in spec for spec in re.findall(r"\\begin\{tabular\}\{([^}]+)\}", tex))


def test_protocol_disclosures_and_room0_claim_guard():
    for path in (MARKDOWN, LATEX):
        rendered = path.read_text(encoding="utf-8").lower()
        assert all(term in rendered for term in ("frozen", "composed", "offline", "oracle"))
        assert "room0" not in rendered


def test_no_manual_decimal_results_or_unfinished_markers():
    for path in (MARKDOWN, LATEX):
        text = path.read_text(encoding="utf-8")
        assert not re.search("TO" + "DO|T" + "BD", text)
    for line in MARKDOWN.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "---" not in line:
            for cell in (part.strip() for part in line.strip("|").split("|")):
                assert not re.fullmatch(r"-?\d+\.\d+", cell)


def test_generator_check_mode():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "check", "--output-dir", str(PAPER_DIR)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "benchmark table package: PASS" in result.stdout
