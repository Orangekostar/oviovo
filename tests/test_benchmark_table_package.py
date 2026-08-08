from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
MARKDOWN = PAPER_DIR / "benchmark_tables.md"
LATEX = PAPER_DIR / "benchmark_tables.tex"
REGISTRY = PAPER_DIR / "benchmark_tokens.tsv"
GENERATOR = ROOT / "tools" / "benchmark_tables.py"
BASELINE_MARKDOWN = PAPER_DIR / "benchmark_tables_baselines.md"
BASELINE_LATEX = PAPER_DIR / "benchmark_tables_baselines.tex"
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
EXPECTED_METHODS = {
    "T1": ("OPENFUSION", "OVIMAP", "CONCEPTGRAPHS", "DUALMAP", "OVIV2"),
    "T2": (
        "OVIMAP_FROZEN", "CONCEPTGRAPHS_FROZEN", "DUALMAP",
        "PANOPTIC_SHARED", "KHRONOS_OPEN", "KHRONOS_ORACLE", "OVIV2",
    ),
    "T3": ("BASE", "VIS", "OWNER", "RECLAIM", "REID", "FULL"),
    "T4": ("OVIMAP", "CONCEPTGRAPHS", "DUALMAP", "KHRONOS", "OVIV2_STATIC", "OVIV2"),
    "S1": ("OVIMAP_FROZEN", "CONCEPTGRAPHS_FROZEN", "DUALMAP", "KHRONOS_SHARED", "OVIV2"),
    "S2": ("ESAM_VISIT", "KHRONOS_ADAPTED", "RESCENE4D", "OVIV2_NO_REID", "OVIV2"),
    "S3": ("DUALMAP_CAL", "CONCEPTGRAPHS_CAL", "KHRONOS_SHARED_CAL", "OVIV2_UNCAL", "OVIV2"),
}
EXPECTED_METRICS = {
    "T1": (
        "REPLICA8_MIOU", "REPLICA8_MACC", "REPLICA8_FMIOU", "REPLICA7_MIOU",
        "SCANNET5_MIOU", "REPLICA8_AP25", "REPLICA8_AP50", "REPLICA8_F5",
        "REPLICA7_AP50", "SCANNET5_AP25", "SCANNET5_AP50", "SCANNET5_F5",
    ),
    "T2": (
        "APARTMENT_OBJECT_F1", "APARTMENT_DYNAMIC_F1", "APARTMENT_CHANGE_F1",
        "OFFICE_OBJECT_F1", "OFFICE_DYNAMIC_F1", "OFFICE_CHANGE_F1",
        "CURRENT_MIOU", "GHOST_RATE", "BG_F5", "RECOVERY_FRAMES",
    ),
    "T3": (
        "STATIC_MIOU", "CHANGE_F1", "STALE_FP", "GHOST_RATE", "BG_F5",
        "IDSW", "REACT_R1", "NOT_FOUND_F1",
    ),
    "T4": (
        "FRONTEND_SPF", "BACKEND_SPF", "MAINT_SPF", "TOTAL_SPF", "HZ", "FINAL_S",
        "QUERY_P50_MS", "QUERY_P95_MS", "GPU_GB", "RAM_GB", "MAP_MB", "EVAL_IO_S",
    ),
    "S1": tuple(
        f"{split}_{metric}"
        for split in ("VALIDATION", "TEST")
        for metric in (
            "PRESENT_R1", "MOVED_R1", "NEW_R1", "NOT_FOUND_F1",
            "STALE_FP", "LOC_ERROR_M", "RECOVERY_FRAMES",
        )
    ),
    "S2": ("STAGE_AP50", "T_AP", "T_REC", "IDSW", "REACT_R1", "FALSE_REID"),
    "S3": (
        "PRESENCE_AUROC", "NOT_FOUND_PREC", "NOT_FOUND_REC", "NOT_FOUND_F1",
        "BINARY_ECE", "RISK_COVERAGE_AUC",
    ),
}
NA_TOKENS = {
    f"T1_OPENFUSION_{metric}"
    for metric in (
        "REPLICA8_AP25", "REPLICA8_AP50", "REPLICA7_AP50",
        "SCANNET5_AP25", "SCANNET5_AP50",
    )
}
LOWER_METRICS = {
    ("T2", "GHOST_RATE"), ("T2", "RECOVERY_FRAMES"),
    ("T3", "STALE_FP"), ("T3", "GHOST_RATE"), ("T3", "IDSW"),
    ("S1", "VALIDATION_STALE_FP"), ("S1", "VALIDATION_LOC_ERROR_M"),
    ("S1", "VALIDATION_RECOVERY_FRAMES"), ("S1", "TEST_STALE_FP"),
    ("S1", "TEST_LOC_ERROR_M"), ("S1", "TEST_RECOVERY_FRAMES"),
    ("S2", "IDSW"), ("S2", "FALSE_REID"),
    ("S3", "BINARY_ECE"), ("S3", "RISK_COVERAGE_AUC"),
}


def rows():
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == FIELDS
        return list(reader)


def token_counts(path):
    return Counter(TOKEN_RE.findall(path.read_text(encoding="utf-8")))


def metric_location(table, metric):
    if table == "T1":
        if metric.startswith("REPLICA8_"):
            return "Replica", "replica_8_compat"
        if metric.startswith("REPLICA7_"):
            return "Replica", "replica_7_heldout"
        return "ScanNet200", "scannet200_5_heldout"
    if table == "T2":
        if metric.startswith("APARTMENT_"):
            return "TESSE-CD", "apartment_test"
        if metric.startswith("OFFICE_"):
            return "TESSE-CD", "office_test"
        return "TESSE-CD", "macro_test"
    if table == "T3":
        if metric == "STATIC_MIOU":
            return "Replica", "replica_7_heldout"
        if metric in {"CHANGE_F1", "STALE_FP", "GHOST_RATE", "BG_F5"}:
            return "TESSE-CD", "macro_test"
        if metric in {"IDSW", "REACT_R1"}:
            return "3RScan", "test"
        return "current_state_queries", "test"
    if table == "T4":
        return "local_same_hardware", "test"
    if table == "S1":
        return "current_state_queries", "validation" if metric.startswith("VALIDATION_") else "test"
    if table == "S2":
        return "3RScan", "test"
    return "current_state_queries", "test"


def expected_registry():
    expected = {}
    for table, methods in EXPECTED_METHODS.items():
        for method in methods:
            for metric in EXPECTED_METRICS[table]:
                token = f"{table}_{method}_{metric}"
                dataset, split = metric_location(table, metric)
                expected[token] = {
                    "table": table,
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "direction": "lower" if table == "T4" or (table, metric) in LOWER_METRICS else "higher",
                    "precision": "2" if table == "T4" or metric.endswith("LOC_ERROR_M") else "3",
                }
    for method in EXPECTED_METHODS["T4"]:
        expected[f"T4_{method}_HZ"]["direction"] = "higher"
    return expected


def test_files_exist():
    assert all(path.is_file() for path in (GENERATOR, MARKDOWN, LATEX, REGISTRY))


def test_four_main_and_three_supplementary_tables():
    md = MARKDOWN.read_text(encoding="utf-8")
    tex = LATEX.read_text(encoding="utf-8")
    assert len(re.findall(r"^## Table [1-4]:", md, re.MULTILINE)) == 4
    assert len(re.findall(r"^## Table S[1-3]:", md, re.MULTILINE)) == 3
    assert set(re.findall(r"\\label\{([^}]+)\}", tex)) == LABELS
    environments = re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}", tex, re.DOTALL)
    assert len(environments) == 7
    for environment in environments:
        assert all(command in environment for command in ("\\toprule", "\\midrule", "\\bottomrule"))


def test_registry_schema_states_and_unique_tokens():
    registry = rows()
    expected = expected_registry()
    names = [row["token"] for row in registry]
    assert len(registry) == 380
    assert len(names) == len(set(names))
    assert set(names) == set(expected)
    assert {row["status"] for row in registry} <= {"UNFILLED", "VERIFIED", "N/A"}
    for row in registry:
        assert {key: row[key] for key in expected[row["token"]]} == expected[row["token"]]
    na_rows = [row for row in registry if row["status"] == "N/A"]
    assert NA_TOKENS <= {row["token"] for row in na_rows}
    for row in na_rows:
        if row["token"] not in NA_TOKENS:
            assert row["source_json"] and row["json_pointer"]
            assert row["note"].startswith("N/A from verified run ")


def test_non_na_tokens_match_both_outputs():
    expected = Counter({row["token"]: 1 for row in rows() if row["token"] not in NA_TOKENS})
    assert token_counts(MARKDOWN) == expected
    assert token_counts(LATEX) == expected


def test_table1_uses_only_oviv2_tokens_for_our_static_method():
    registry_tokens = {row["token"] for row in rows() if row["table"] == "T1"}
    assert len({token for token in registry_tokens if token.startswith("T1_OVIV2_")}) == 12
    assert not any(token.startswith("T1_OVIOVO_") for token in registry_tokens)
    assert "| OVIV2 | online |" in MARKDOWN.read_text(encoding="utf-8")
    assert "OVIV2 & online &" in LATEX.read_text(encoding="utf-8")


def test_all_active_user_facing_method_tokens_use_oviv2() -> None:
    active_tables = {"T2", "T4", "S1", "S2", "S3"}
    active_rows = [row for row in rows() if row["table"] in active_tables]
    oviv2_rows = [row for row in active_rows if "OVIV2" in row["method"]]

    assert Counter(row["table"] for row in oviv2_rows) == {
        "T2": 10,
        "T4": 24,
        "S1": 14,
        "S2": 12,
        "S3": 12,
    }
    assert len(oviv2_rows) == 72
    assert not any("OVIOVO" in row["token"] or "OVIOVO" in row["method"] for row in active_rows)


def test_all_active_table_views_display_expected_method_name_with_online_t2_mode() -> None:
    for path, method in (
        (MARKDOWN, "OVIV2"),
        (LATEX, "OVIV2"),
        (BASELINE_MARKDOWN, "CROVE"),
        (BASELINE_LATEX, "CROVE"),
    ):
        rendered = path.read_text(encoding="utf-8")
        assert method in rendered
        assert "OVIOVO" not in rendered

    for path, method in ((MARKDOWN, "OVIV2"), (BASELINE_MARKDOWN, "CROVE")):
        rendered = path.read_text(encoding="utf-8")
        table2 = rendered.split("## Table 2:", 1)[1].split("## Table 3:", 1)[0]
        assert f"| {method} | online |" in table2
    for path, method in ((LATEX, "OVIV2"), (BASELINE_LATEX, "CROVE")):
        rendered = path.read_text(encoding="utf-8")
        table2 = next(
            table
            for table in re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}", rendered, re.DOTALL)
            if r"\label{tab:dynamic_current_map}" in table
        )
        assert f"{method} & online &" in table2


def test_ovimap_table1_stays_unfilled_until_paper_parity_passes():
    ovimap = [
        row for row in rows() if row["table"] == "T1" and row["method"] == "OVIMAP"
    ]
    assert len(ovimap) == 12
    assert {row["status"] for row in ovimap} == {"UNFILLED"}
    assert all(not row["source_json"] and not row["json_pointer"] for row in ovimap)
    rendered = (
        (PAPER_DIR / "benchmark_tables_baselines.md").read_text(encoding="utf-8")
        + (PAPER_DIR / "benchmark_tables_baselines.tex").read_text(encoding="utf-8")
    )
    assert all("{{" + row["token"] + "}}" in rendered for row in ovimap)


def test_provenance_requirements():
    for row in rows():
        if row["status"] == "VERIFIED":
            assert row["source_json"] and row["json_pointer"]
        elif row["status"] == "N/A":
            assert row["note"]
            if row["token"] not in NA_TOKENS:
                assert row["source_json"] and row["json_pointer"]
                assert row["note"].startswith("N/A from verified run ")


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


def test_generator_check_detects_tampering(tmp_path):
    output_dir = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, output_dir)
    (output_dir / MARKDOWN.name).write_text("tampered\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "check", "--output-dir", str(output_dir)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "benchmark table package: FAIL" in result.stderr


def test_generator_check_detects_frozen_registry_field_tampering(tmp_path):
    output_dir = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, output_dir)
    registry_path = output_dir / REGISTRY.name
    text = registry_path.read_text(encoding="utf-8")
    registry_path.write_text(text.replace("\tOPENFUSION\t", "\tWRONG_METHOD\t", 1), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(GENERATOR), "check", "--output-dir", str(output_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "benchmark_tokens.tsv" in result.stderr


@pytest.mark.parametrize("provenance", ["none", "missing", "unverified", "no_evidence"])
def test_generator_check_rejects_unverified_dynamic_na(tmp_path, provenance):
    output_dir = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, output_dir)
    registry_path = output_dir / REGISTRY.name
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle, delimiter="\t"))
    row = next(item for item in registry if item["token"] == "T2_OVIV2_OFFICE_OBJECT_F1")
    row["status"] = "N/A"
    row["note"] = "Unavailable without a verified import."
    if provenance != "none":
        row["source_json"] = "missing-result.json"
        row["json_pointer"] = "/unavailable/office/OBJECT_F1"
    if provenance in {"unverified", "no_evidence"}:
        row["source_json"] = "result.json"
        row["note"] = "N/A from verified run fake-run: unavailable"
        result = {
            "status": "BLOCKED" if provenance == "unverified" else "VERIFIED",
            "run_id": "fake-run",
            "unavailable_bindings": [
                {
                    "token": row["token"],
                    "reason_pointer": row["json_pointer"],
                }
            ],
        }
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry)

    result = subprocess.run(
        [sys.executable, str(GENERATOR), "check", "--output-dir", str(output_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "benchmark_tokens.tsv" in result.stderr


def test_write_refuses_to_overwrite_verified_registry(tmp_path):
    output_dir = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, output_dir)
    registry_path = output_dir / REGISTRY.name
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle, delimiter="\t"))
    verified = next(row for row in registry if row["status"] == "UNFILLED")
    verified.update(status="VERIFIED", source_json="result.json", json_pointer="/metrics/value")
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry)
    protected = registry_path.read_bytes()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "write", "--output-dir", str(output_dir)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert registry_path.read_bytes() == protected
    assert "--force" in result.stderr
    forced = subprocess.run(
        [
            sys.executable, str(GENERATOR), "write", "--output-dir", str(output_dir),
            "--force",
        ],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert forced.returncode == 0, forced.stdout + forced.stderr
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reset = list(csv.DictReader(handle, delimiter="\t"))
    assert all(row["status"] != "VERIFIED" for row in reset)
