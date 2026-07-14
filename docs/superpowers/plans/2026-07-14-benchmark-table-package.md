# AAAI Benchmark Table Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and verify four main-paper and three supplementary benchmark table templates with synchronized Markdown, LaTeX, and token-registry outputs.

**Architecture:** A dependency-free Python generator owns the frozen table blueprint and renders all committed artifacts. The TSV is the numerical provenance registry; Markdown and LaTeX show `{{TOKEN}}` for non-`N/A` cells. Focused tests enforce table count, token parity, provenance states, paper formatting, and deterministic regeneration.

**Tech Stack:** Python standard library, pytest, Markdown, LaTeX `booktabs`, TSV.

---

## File Structure

- Create `tools/__init__.py`: import marker for repository-local tooling.
- Create `tools/benchmark_tables.py`: immutable schema, seven table definitions, renderers, and CLI.
- Create `tests/test_benchmark_table_package.py`: artifact and regeneration contract.
- Generate `docs/paper/benchmark_tables.md`: readable seven-table preview.
- Generate `docs/paper/benchmark_tables.tex`: AAAI-ready table environments.
- Generate `docs/paper/benchmark_tokens.tsv`: one provenance row per numerical cell.

No existing pipeline, evaluator, configuration, result, or user-owned untracked file is modified.

### Task 1: Write the artifact contract tests

**Files:**
- Create: `tests/test_benchmark_table_package.py`

- [ ] **Step 1: Create the failing test file**

Use this complete test contract:

```python
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
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_benchmark_table_package.py
```

Expected: failure because the generator and three `docs/paper` artifacts do not exist.

### Task 2: Implement the schema and seven frozen blueprints

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/benchmark_tables.py`

- [ ] **Step 1: Add the import marker**

```python
"""Repository-local generation and verification tools."""
```

- [ ] **Step 2: Define immutable schema types**

Implement these types in `tools/benchmark_tables.py`:

```python
from dataclasses import dataclass, field
from typing import Literal, Mapping

@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    dataset: str
    split: str
    metric: str
    direction: Literal["higher", "lower"] = "higher"
    precision: Literal[2, 3] = 3

@dataclass(frozen=True)
class Method:
    key: str
    label: str
    mode: str
    unavailable: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class Panel:
    title: str
    metrics: tuple[Metric, ...]

@dataclass(frozen=True)
class Table:
    key: str
    number: str
    title: str
    label: str
    caption: str
    message: str
    methods: tuple[Method, ...]
    panels: tuple[Panel, ...]
```

Use `token = f"{table.key}_{method.key}_{metric.key}"`. Reject non-ASCII keys, duplicate tokens, invalid directions, invalid precision, missing `N/A` notes, or a table without rows/panels.

- [ ] **Step 3: Encode the exact table blueprint**

Use the following authoritative rows and metric keys. Display labels, datasets, splits, directions, and precision come directly from the approved design specification.

| Table | Rows in order | Panels and metric keys in order |
| --- | --- | --- |
| T1 | `OPENFUSION`, `OVIMAP`, `CONCEPTGRAPHS`, `DUALMAP`, `OVIOVO` | Semantic: `REPLICA8_MIOU`, `REPLICA8_MACC`, `REPLICA8_FMIOU`, `REPLICA7_MIOU`, `SCANNET5_MIOU`; Instance/geometry: `REPLICA8_AP25`, `REPLICA8_AP50`, `REPLICA8_F5`, `REPLICA7_AP50`, `SCANNET5_AP25`, `SCANNET5_AP50`, `SCANNET5_F5` |
| T2 | `OVIMAP_FROZEN`, `CONCEPTGRAPHS_FROZEN`, `DUALMAP`, `PANOPTIC_SHARED`, `KHRONOS_OPEN`, `KHRONOS_ORACLE`, `OVIOVO` | TESSE: `APARTMENT_OBJECT_F1`, `APARTMENT_DYNAMIC_F1`, `APARTMENT_CHANGE_F1`, `OFFICE_OBJECT_F1`, `OFFICE_DYNAMIC_F1`, `OFFICE_CHANGE_F1`; Current map: `CURRENT_MIOU`, `GHOST_RATE`, `BG_F5`, `RECOVERY_FRAMES` |
| T3 | `BASE`, `VIS`, `OWNER`, `RECLAIM`, `REID`, `FULL` | `STATIC_MIOU`, `CHANGE_F1`, `STALE_FP`, `GHOST_RATE`, `BG_F5`, `IDSW`, `REACT_R1`, `NOT_FOUND_F1` |
| T4 | `OVIMAP`, `CONCEPTGRAPHS`, `DUALMAP`, `KHRONOS`, `OVIOVO_STATIC`, `OVIOVO` | Latency: `FRONTEND_SPF`, `BACKEND_SPF`, `MAINT_SPF`, `TOTAL_SPF`, `HZ`, `FINAL_S`, `QUERY_P50_MS`, `QUERY_P95_MS`; Resources: `GPU_GB`, `RAM_GB`, `MAP_MB`, `EVAL_IO_S` |
| S1 | `OVIMAP_FROZEN`, `CONCEPTGRAPHS_FROZEN`, `DUALMAP`, `KHRONOS_SHARED`, `OVIOVO` | Validation: `VALIDATION_PRESENT_R1`, `VALIDATION_MOVED_R1`, `VALIDATION_NEW_R1`, `VALIDATION_NOT_FOUND_F1`, `VALIDATION_STALE_FP`, `VALIDATION_LOC_ERROR_M`, `VALIDATION_RECOVERY_FRAMES`; repeat with the `TEST_` prefix for held-out test |
| S2 | `ESAM_VISIT`, `KHRONOS_ADAPTED`, `RESCENE4D`, `OVIOVO_NO_REID`, `OVIOVO` | `STAGE_AP50`, `T_AP`, `T_REC`, `IDSW`, `REACT_R1`, `FALSE_REID` |
| S3 | `DUALMAP_CAL`, `CONCEPTGRAPHS_CAL`, `KHRONOS_SHARED_CAL`, `OVIOVO_UNCAL`, `OVIOVO` | `PRESENCE_AUROC`, `NOT_FOUND_PREC`, `NOT_FOUND_REC`, `NOT_FOUND_F1`, `BINARY_ECE`, `RISK_COVERAGE_AUC` |

Labels must be exactly:

```python
LABELS = {
    "T1": "tab:static_mapping",
    "T2": "tab:dynamic_current_map",
    "T3": "tab:causal_ablation",
    "T4": "tab:online_efficiency",
    "S1": "tab:supp_current_query",
    "S2": "tab:supp_temporal_identity",
    "S3": "tab:supp_reliability",
}
```

Known `N/A` cells are OpenFusion entity AP cells in T1. Mark its `REPLICA8_AP25`, `REPLICA8_AP50`, `REPLICA7_AP50`, `SCANNET5_AP25`, and `SCANNET5_AP50` as unavailable with note `No native entity AP output.` All other initial cells are `UNFILLED`.

Metric metadata rules are deterministic: quality, recall, AP, F-score, AUROC, and Hz are `higher`; error, false-positive, ID-switch, recovery-frame, calibration, latency, and resource metrics are `lower`. Seconds/frame, seconds, milliseconds, GB, MB, Hz, and localization error use precision `2`; all other metrics use precision `3`. T1 uses `Replica/replica_8_compat`, `Replica/replica_7_heldout`, and `ScanNet200/scannet200_5_heldout`; T2 uses `TESSE-CD/apartment_test`, `TESSE-CD/office_test`, and `TESSE-CD/macro_test`; T3 records each metric's corresponding T1/T2/S1/S2 held-out split rather than a synthetic aggregate dataset; T4 uses `local_same_hardware/test`; S1 uses `current_state_queries/validation` and `/test`; S2 uses `3RScan/test`; S3 uses `current_state_queries/test` with validation-only calibration.

### Task 3: Implement renderers and generate artifacts

**Files:**
- Modify: `tools/benchmark_tables.py`
- Generate: `docs/paper/benchmark_tables.md`
- Generate: `docs/paper/benchmark_tables.tex`
- Generate: `docs/paper/benchmark_tokens.tsv`

- [ ] **Step 1: Implement registry rendering**

Iterate table → method → panel → metric in declared order. Write fields:

```python
{
    "token": token,
    "table": table.key,
    "method": method.key,
    "dataset": metric.dataset,
    "split": metric.split,
    "metric": metric.metric,
    "direction": metric.direction,
    "precision": str(metric.precision),
    "source_json": "",
    "json_pointer": "",
    "status": "N/A" if metric.key in method.unavailable else "UNFILLED",
    "note": method.unavailable.get(metric.key, ""),
}
```

Use `csv.DictWriter` with tab delimiter and newline terminator. Refuse duplicate token names.

- [ ] **Step 2: Implement Markdown rendering**

Render one `## Table ...` heading, caption, message, and one Markdown table per panel. Each table starts with `Method | Mode`, uses right-aligned metric columns, renders `--` for `N/A`, and renders `{{TOKEN}}` otherwise. Protocol notes must use the literal classifications `frozen`, `composed`, `offline`, and `oracle`; include the `Bring me the new book` normalization under S1 and validation-only calibration rule under S3. Do not render a `room0`-only headline result.

- [ ] **Step 3: Implement LaTeX rendering**

Render one `table*` environment per logical table. Captions precede labels and carry the same literal `frozen`, `composed`, `offline`, and `oracle` protocol disclosures as Markdown. Multi-panel tables use `\multicolumn` panel headings and repeat the header after `\midrule`. Use `booktabs`, no vertical rules, and no `\resizebox`. Escape `_`, `%`, `&`, `#`, and braces in display text while leaving token braces intact.

- [ ] **Step 4: Implement CLI modes**

Commands:

```bash
python tools/benchmark_tables.py write --output-dir docs/paper
python tools/benchmark_tables.py check --output-dir docs/paper
```

`write` creates the directory and three artifacts atomically. It refuses to overwrite a registry containing any `VERIFIED` row unless `--force` is present. `check` renders into a temporary directory, byte-compares all three artifacts, and prints `benchmark table package: PASS` on equality.

- [ ] **Step 5: Generate the initial package**

```bash
python tools/benchmark_tables.py write --output-dir docs/paper
```

Expected summary: `4 main tables`, `3 supplementary tables`, and a nonzero unique token count.

- [ ] **Step 6: Verify GREEN**

```bash
pytest -q tests/test_benchmark_table_package.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit implementation**

```bash
git add tools/__init__.py tools/benchmark_tables.py \
  tests/test_benchmark_table_package.py docs/paper
git commit -m "docs: add AAAI benchmark table templates"
```

### Task 4: Run final package verification

**Files:**
- Verify: `docs/paper/benchmark_tables.md`
- Verify: `docs/paper/benchmark_tables.tex`
- Verify: `docs/paper/benchmark_tokens.tsv`

- [ ] **Step 1: Run deterministic check**

```bash
python tools/benchmark_tables.py check --output-dir docs/paper
```

Expected: `benchmark table package: PASS`.

- [ ] **Step 2: Run tests and whitespace validation**

```bash
pytest -q tests/test_benchmark_table_package.py
git diff --check HEAD~1..HEAD
```

Expected: all tests pass and the whitespace check emits no output.

- [ ] **Step 3: Verify commit scope**

```bash
git show --name-only --format= HEAD | sed '/^$/d'
```

Expected: only `tools/`, `tests/test_benchmark_table_package.py`, and `docs/paper/` files. Existing untracked `build/`, `data/`, `devel/`, `scripts/`, `weights/`, `step.md`, and `render_analysis_vis.py` remain uncommitted.
