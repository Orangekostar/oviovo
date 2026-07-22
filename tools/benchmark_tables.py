#!/usr/bin/env python3
"""Generate and verify the frozen AAAI benchmark table package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

Direction = Literal["higher", "lower"]
Precision = Literal[2, 3]

FIELDS = (
    "token",
    "table",
    "method",
    "dataset",
    "split",
    "metric",
    "direction",
    "precision",
    "source_json",
    "json_pointer",
    "status",
    "note",
)
ARTIFACT_NAMES = (
    "benchmark_tables.md",
    "benchmark_tables.tex",
    "benchmark_tokens.tsv",
)
KEY_RE = re.compile(r"^[A-Z0-9_]+$")
DYNAMIC_NA_NOTE_RE = re.compile(
    r"\AN/A from verified run (?P<run_id>.+) "
    r"\[source_sha256=(?P<source_sha256>[0-9a-f]{64})\]: (?P<reason>.+)\Z"
)
NA_NOTE = "No native entity AP output."
MISSING = object()


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    dataset: str
    split: str
    metric: str
    direction: Direction = "higher"
    precision: Precision = 3


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    mode: str
    unavailable: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


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


def _location(table: str, metric: str) -> tuple[str, str]:
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
        split = "validation" if metric.startswith("VALIDATION_") else "test"
        return "current_state_queries", split
    if table == "S2":
        return "3RScan", "test"
    return "current_state_queries", "test"


LOWER_METRICS = {
    ("T2", "GHOST_RATE"),
    ("T2", "RECOVERY_FRAMES"),
    ("T3", "STALE_FP"),
    ("T3", "GHOST_RATE"),
    ("T3", "IDSW"),
    ("S1", "VALIDATION_STALE_FP"),
    ("S1", "VALIDATION_LOC_ERROR_M"),
    ("S1", "VALIDATION_RECOVERY_FRAMES"),
    ("S1", "TEST_STALE_FP"),
    ("S1", "TEST_LOC_ERROR_M"),
    ("S1", "TEST_RECOVERY_FRAMES"),
    ("S2", "IDSW"),
    ("S2", "FALSE_REID"),
    ("S3", "BINARY_ECE"),
    ("S3", "RISK_COVERAGE_AUC"),
}


def _metric(table: str, key: str, label: str) -> Metric:
    dataset, split = _location(table, key)
    direction: Direction = "lower" if table == "T4" or (table, key) in LOWER_METRICS else "higher"
    if table == "T4" and key == "HZ":
        direction = "higher"
    precision: Precision = 2 if table == "T4" or key.endswith("LOC_ERROR_M") else 3
    return Metric(
        key=key,
        label=label,
        dataset=dataset,
        split=split,
        metric=key,
        direction=direction,
        precision=precision,
    )


def _panel(table: str, title: str, metrics: Sequence[tuple[str, str]]) -> Panel:
    return Panel(title=title, metrics=tuple(_metric(table, key, label) for key, label in metrics))


def _methods(specs: Sequence[tuple[str, str, str]]) -> tuple[Method, ...]:
    return tuple(Method(key=key, label=label, mode=mode) for key, label, mode in specs)


def _build_tables() -> tuple[Table, ...]:
    openfusion_na = MappingProxyType(
        {
            "REPLICA8_AP25": NA_NOTE,
            "REPLICA8_AP50": NA_NOTE,
            "REPLICA7_AP50": NA_NOTE,
            "SCANNET5_AP25": NA_NOTE,
            "SCANNET5_AP50": NA_NOTE,
        }
    )
    t1_methods = (
        Method("OPENFUSION", "OpenFusion", "native", openfusion_na),
        Method("OVIMAP", "OVI-MAP", "native"),
        Method("CONCEPTGRAPHS", "ConceptGraphs", "native"),
        Method("DUALMAP", "DualMap", "native"),
        Method("OVIV2", "OVIV2", "online"),
    )
    tables = (
        Table(
            key="T1",
            number="1",
            title="Static Mapping Quality",
            label="tab:static_mapping",
            caption=(
                "Static open-vocabulary mapping under native predictions. Replica-8 is a compatibility "
                "split; Replica-7 and ScanNet200-5 are held out and carry the generalization claim."
            ),
            message="Lifecycle maintenance must not trade away the static mapping foundation.",
            methods=t1_methods,
            panels=(
                _panel(
                    "T1",
                    "Semantic quality",
                    (
                        ("REPLICA8_MIOU", "Replica-8 mIoU"),
                        ("REPLICA8_MACC", "Replica-8 mAcc"),
                        ("REPLICA8_FMIOU", "Replica-8 f-mIoU"),
                        ("REPLICA7_MIOU", "Replica-7 mIoU"),
                        ("SCANNET5_MIOU", "ScanNet200-5 mIoU"),
                    ),
                ),
                _panel(
                    "T1",
                    "Instance and geometry quality",
                    (
                        ("REPLICA8_AP25", "Replica-8 AP25"),
                        ("REPLICA8_AP50", "Replica-8 AP50"),
                        ("REPLICA8_F5", "Replica-8 F@5cm"),
                        ("REPLICA7_AP50", "Replica-7 AP50"),
                        ("SCANNET5_AP25", "ScanNet200-5 AP25"),
                        ("SCANNET5_AP50", "ScanNet200-5 AP50"),
                        ("SCANNET5_F5", "ScanNet200-5 F@5cm"),
                    ),
                ),
            ),
        ),
        Table(
            key="T2",
            number="2",
            title="Dynamic Current-Map Quality",
            label="tab:dynamic_current_map",
            caption=(
                "Current-map quality on TESSE-CD. Frozen rows cannot update after intervention; "
                "Panoptic Mapping is composed with shared masks; Khronos GT semantics is an oracle "
                "and is excluded from ranking."
            ),
            message="Signed visibility and reversible maintenance remove stale geometry and recover revealed space.",
            methods=_methods(
                (
                    ("OVIMAP_FROZEN", "OVI-MAP (frozen)", "frozen"),
                    ("CONCEPTGRAPHS_FROZEN", "ConceptGraphs (frozen)", "frozen"),
                    ("DUALMAP", "DualMap", "native"),
                    ("PANOPTIC_SHARED", "Panoptic Mapping + shared masks", "composed"),
                    ("KHRONOS_OPEN", "Khronos (open-set)", "online"),
                    ("KHRONOS_ORACLE", "Khronos (GT semantics)", "oracle"),
                    ("OVIV2", "OVIV2", "online"),
                )
            ),
            panels=(
                _panel(
                    "T2",
                    "Official TESSE-CD metrics",
                    (
                        ("APARTMENT_OBJECT_F1", "Apartment object F1"),
                        ("APARTMENT_DYNAMIC_F1", "Apartment dynamic F1"),
                        ("APARTMENT_CHANGE_F1", "Apartment change F1"),
                        ("OFFICE_OBJECT_F1", "Office object F1"),
                        ("OFFICE_DYNAMIC_F1", "Office dynamic F1"),
                        ("OFFICE_CHANGE_F1", "Office change F1"),
                    ),
                ),
                _panel(
                    "T2",
                    "Common current-map metrics",
                    (
                        ("CURRENT_MIOU", "Current mIoU"),
                        ("GHOST_RATE", "Ghost rate"),
                        ("BG_F5", "Background F@5cm"),
                        ("RECOVERY_FRAMES", "Recovery frames"),
                    ),
                ),
            ),
        ),
        Table(
            key="T3",
            number="3",
            title="Causal Component Ablation",
            label="tab:causal_ablation",
            caption=(
                "Cumulative ablation across the held-out static, dynamic, identity, and current-query splits. "
                "Each row adds exactly one mechanism."
            ),
            message="Visibility, ownership, reclaim, re-identification, and rejection target distinct failures.",
            methods=_methods(
                (
                    ("BASE", "Positive-only base", "geometry-first"),
                    ("VIS", "+ Signed visibility", "visibility"),
                    ("OWNER", "+ Reversible ownership", "ownership"),
                    ("RECLAIM", "+ Background reclaim", "reclaim"),
                    ("REID", "+ Dormant re-ID", "re-identification"),
                    ("FULL", "+ Calibrated NOT_FOUND", "calibration"),
                )
            ),
            panels=(
                _panel(
                    "T3",
                    "Cumulative mechanisms",
                    (
                        ("STATIC_MIOU", "Static mIoU"),
                        ("CHANGE_F1", "Change F1"),
                        ("STALE_FP", "Stale FP"),
                        ("GHOST_RATE", "Ghost rate"),
                        ("BG_F5", "Background F@5cm"),
                        ("IDSW", "ID switches"),
                        ("REACT_R1", "Reactivation R@1"),
                        ("NOT_FOUND_F1", "NOT_FOUND F1"),
                    ),
                ),
            ),
        ),
        Table(
            key="T4",
            number="4",
            title="Online Efficiency and Memory",
            label="tab:online_efficiency",
            caption=(
                "Same-hardware measurements with initialization, online processing, finalization, and evaluation I/O "
                "reported separately. Paper-reported hardware numbers are not mixed into ranked columns."
            ),
            message="Current-state maintenance must have bounded online latency and memory cost.",
            methods=_methods(
                (
                    ("OVIMAP", "OVI-MAP", "native"),
                    ("CONCEPTGRAPHS", "ConceptGraphs", "native"),
                    ("DUALMAP", "DualMap", "native"),
                    ("KHRONOS", "Khronos", "online"),
                    ("OVIV2_STATIC", "OVIV2 (maintenance off)", "maintenance-off"),
                    ("OVIV2", "OVIV2", "online"),
                )
            ),
            panels=(
                _panel(
                    "T4",
                    "Latency",
                    (
                        ("FRONTEND_SPF", "Frontend s/frame"),
                        ("BACKEND_SPF", "Backend s/frame"),
                        ("MAINT_SPF", "Maintenance s/frame"),
                        ("TOTAL_SPF", "Total s/frame"),
                        ("HZ", "Processed Hz"),
                        ("FINAL_S", "Finalization s"),
                        ("QUERY_P50_MS", "Query p50 ms"),
                        ("QUERY_P95_MS", "Query p95 ms"),
                    ),
                ),
                _panel(
                    "T4",
                    "Resources",
                    (
                        ("GPU_GB", "Peak GPU GB"),
                        ("RAM_GB", "Peak RAM GB"),
                        ("MAP_MB", "Final map MB"),
                        ("EVAL_IO_S", "Evaluation/I/O s"),
                    ),
                ),
            ),
        ),
        Table(
            key="S1",
            number="S1",
            title="Open-Vocabulary Current-State Localization",
            label="tab:supp_current_query",
            caption=(
                "Validation and held-out current-state query macro averages. Frozen maps and the composed Khronos "
                "text head use the same query normalization and cannot exploit manipulation or navigation signals."
            ),
            message=(
                "Bring me the new book is normalized to action=locate, category=book, "
                "temporal_predicate=added_since_previous_visit, scope=current."
            ),
            methods=_methods(
                (
                    ("OVIMAP_FROZEN", "OVI-MAP (frozen)", "frozen"),
                    ("CONCEPTGRAPHS_FROZEN", "ConceptGraphs (frozen)", "frozen"),
                    ("DUALMAP", "DualMap", "native"),
                    ("KHRONOS_SHARED", "Khronos + shared text head", "composed"),
                    ("OVIV2", "OVIV2", "online"),
                )
            ),
            panels=tuple(
                _panel(
                    "S1",
                    f"{display} split",
                    tuple(
                        (f"{prefix}_{key}", label)
                        for key, label in (
                            ("PRESENT_R1", "Present R@1"),
                            ("MOVED_R1", "Moved R@1"),
                            ("NEW_R1", "New R@1"),
                            ("NOT_FOUND_F1", "NOT_FOUND F1"),
                            ("STALE_FP", "Stale FP"),
                            ("LOC_ERROR_M", "Median error m"),
                            ("RECOVERY_FRAMES", "Recovery frames"),
                        )
                    ),
                )
                for prefix, display in (("VALIDATION", "Validation"), ("TEST", "Held-out test"))
            ),
        ),
        Table(
            key="S2",
            number="S2",
            title="Temporal Identity on 3RScan",
            label="tab:supp_temporal_identity",
            caption=(
                "Streaming 3RScan identity evaluation without future scans. ReScene4D is an offline upper bound "
                "and is not ranked against online methods."
            ),
            message="Dormancy and re-identification should preserve identity without false reactivation.",
            methods=_methods(
                (
                    ("ESAM_VISIT", "ESAM (per-visit)", "per-visit"),
                    ("KHRONOS_ADAPTED", "Khronos (adapted stream)", "online-adapted"),
                    ("RESCENE4D", "ReScene4D", "offline"),
                    ("OVIV2_NO_REID", "OVIV2 (no re-ID)", "online"),
                    ("OVIV2", "OVIV2", "online"),
                )
            ),
            panels=(
                _panel(
                    "S2",
                    "Cross-visit identity",
                    (
                        ("STAGE_AP50", "Per-stage AP50"),
                        ("T_AP", "t-AP"),
                        ("T_REC", "t-REC"),
                        ("IDSW", "ID switches"),
                        ("REACT_R1", "Reactivation R@1"),
                        ("FALSE_REID", "False ReID"),
                    ),
                ),
            ),
        ),
        Table(
            key="S3",
            number="S3",
            title="Absence Reliability and Calibration",
            label="tab:supp_reliability",
            caption=(
                "Presence and absence reliability on held-out queries. Composed baseline rejectors and OVIV2 use "
                "one validation-only calibration policy serialized before test execution."
            ),
            message="A current-state map must reject absent targets instead of always returning an embedding match.",
            methods=_methods(
                (
                    ("DUALMAP_CAL", "DualMap + calibrated rejector", "composed"),
                    ("CONCEPTGRAPHS_CAL", "ConceptGraphs + calibrated rejector", "composed"),
                    ("KHRONOS_SHARED_CAL", "Khronos shared head + rejector", "composed"),
                    ("OVIV2_UNCAL", "OVIV2 (uncalibrated)", "online"),
                    ("OVIV2", "OVIV2", "calibrated"),
                )
            ),
            panels=(
                _panel(
                    "S3",
                    "Presence and rejection reliability",
                    (
                        ("PRESENCE_AUROC", "Presence AUROC"),
                        ("NOT_FOUND_PREC", "NOT_FOUND precision"),
                        ("NOT_FOUND_REC", "NOT_FOUND recall"),
                        ("NOT_FOUND_F1", "NOT_FOUND F1"),
                        ("BINARY_ECE", "Binary ECE"),
                        ("RISK_COVERAGE_AUC", "Risk-coverage AUC"),
                    ),
                ),
            ),
        ),
    )
    _validate_tables(tables)
    return tables


def _validate_tables(tables: Sequence[Table]) -> None:
    seen_tables: set[str] = set()
    seen_labels: set[str] = set()
    seen_tokens: set[str] = set()
    for table in tables:
        if not table.methods or not table.panels:
            raise ValueError(f"{table.key}: table requires methods and panels")
        if not KEY_RE.fullmatch(table.key):
            raise ValueError(f"invalid table key: {table.key}")
        if table.key in seen_tables or table.label in seen_labels:
            raise ValueError(f"duplicate table key or label: {table.key}")
        seen_tables.add(table.key)
        seen_labels.add(table.label)
        metric_keys = {metric.key for panel in table.panels for metric in panel.metrics}
        if not metric_keys:
            raise ValueError(f"{table.key}: table requires metrics")
        for panel in table.panels:
            if not panel.metrics:
                raise ValueError(f"{table.key}: panel requires metrics")
            for metric in panel.metrics:
                if not KEY_RE.fullmatch(metric.key):
                    raise ValueError(f"invalid metric key: {metric.key}")
                if metric.direction not in {"higher", "lower"} or metric.precision not in {2, 3}:
                    raise ValueError(f"invalid metric metadata: {table.key}/{metric.key}")
        for method in table.methods:
            if not KEY_RE.fullmatch(method.key):
                raise ValueError(f"invalid method key: {method.key}")
            unknown = set(method.unavailable) - metric_keys
            if unknown:
                raise ValueError(f"{table.key}/{method.key}: unknown unavailable metrics {sorted(unknown)}")
            if any(not note for note in method.unavailable.values()):
                raise ValueError(f"{table.key}/{method.key}: N/A requires a note")
            for metric in (metric for panel in table.panels for metric in panel.metrics):
                token = f"{table.key}_{method.key}_{metric.key}"
                if token in seen_tokens:
                    raise ValueError(f"duplicate token: {token}")
                seen_tokens.add(token)


TABLES = _build_tables()


def _token(table: Table, method: Method, metric: Metric) -> str:
    return f"{table.key}_{method.key}_{metric.key}"


def _cell(table: Table, method: Method, metric: Metric, latex: bool = False) -> str:
    if metric.key in method.unavailable:
        return "--"
    token = _token(table, method, metric)
    placeholder = f"{{{{{token}}}}}"
    return f"\\verb|{placeholder}|" if latex else placeholder


def _arrow(metric: Metric, latex: bool = False) -> str:
    if latex:
        return "$\\uparrow$" if metric.direction == "higher" else "$\\downarrow$"
    return "$\\uparrow$" if metric.direction == "higher" else "$\\downarrow$"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def registry_rows() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for table in TABLES:
        for method in table.methods:
            for panel in table.panels:
                for metric in panel.metrics:
                    token = _token(table, method, metric)
                    if token in seen:
                        raise ValueError(f"duplicate token: {token}")
                    seen.add(token)
                    note = method.unavailable.get(metric.key, "")
                    result.append(
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
                            "status": "N/A" if note else "UNFILLED",
                            "note": note or "Pending benchmark run.",
                        }
                    )
    return result


def render_registry() -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(registry_rows())
    return output.getvalue()


def render_markdown() -> str:
    lines = [
        "# OVIV2 AAAI Benchmark Tables",
        "",
        "Numerical cells are provenance tokens. `--` denotes a protocol-level N/A.",
        "",
    ]
    for table in TABLES:
        lines.extend(
            (
                f"## Table {table.number}: {table.title}",
                "",
                f"**Caption.** {table.caption}",
                "",
                f"**Claim.** {table.message}",
                "",
            )
        )
        for panel in table.panels:
            headers = ["Method", "Mode"] + [
                f"{metric.label} {_arrow(metric)}" for metric in panel.metrics
            ]
            alignments = ["---", "---"] + ["---:"] * len(panel.metrics)
            lines.extend(
                (
                    f"### {panel.title}",
                    "",
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(alignments) + " |",
                )
            )
            for method in table.methods:
                cells = [method.label, method.mode] + [
                    _cell(table, method, metric) for metric in panel.metrics
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_latex() -> str:
    lines = [
        "% Generated by tools/benchmark_tables.py. Requires \\usepackage{booktabs}.",
        "",
    ]
    for table in TABLES:
        lines.extend(
            (
                r"\begin{table*}[t]",
                r"\centering",
                f"\\caption{{{_latex_escape(table.caption)}}}",
                f"\\label{{{table.label}}}",
                r"\small",
            )
        )
        for index, panel in enumerate(table.panels):
            if index:
                lines.append(r"\vspace{4pt}")
            column_spec = "ll" + "r" * len(panel.metrics)
            headers = ["Method", "Mode"] + [
                f"{_latex_escape(metric.label)} {_arrow(metric, latex=True)}"
                for metric in panel.metrics
            ]
            lines.extend(
                (
                    f"\\textbf{{{_latex_escape(panel.title)}}}\\\\",
                    f"\\begin{{tabular}}{{{column_spec}}}",
                    r"\toprule",
                    " & ".join(headers) + r" \\",
                    r"\midrule",
                )
            )
            for method in table.methods:
                cells = [_latex_escape(method.label), _latex_escape(method.mode)] + [
                    _cell(table, method, metric, latex=True) for metric in panel.metrics
                ]
                lines.append(" & ".join(cells) + r" \\")
            lines.extend((r"\bottomrule", r"\end{tabular}", r"\par"))
        lines.extend(
            (
                r"\smallskip",
                f"\\noindent\\textit{{Claim: {_latex_escape(table.message)}}}",
                r"\end{table*}",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_artifacts() -> dict[str, str]:
    return {
        "benchmark_tables.md": render_markdown(),
        "benchmark_tables.tex": render_latex(),
        "benchmark_tokens.tsv": render_registry(),
    }


def _contains_verified(registry_path: Path) -> bool:
    if not registry_path.exists():
        return False
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"invalid registry schema: {registry_path}")
        return any(row["status"] == "VERIFIED" for row in reader)


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_package(output_dir: Path, force: bool = False) -> int:
    registry_path = output_dir / "benchmark_tokens.tsv"
    if not force and _contains_verified(registry_path):
        print(
            "benchmark table package: refusing to overwrite VERIFIED rows; use --force to reset",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = render_artifacts()
    for name in ARTIFACT_NAMES:
        _atomic_write(output_dir / name, artifacts[name])
    print(
        f"benchmark table package: wrote 4 main tables, 3 supplementary tables, "
        f"{len(registry_rows())} tokens"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_json_pointer(document: object, pointer: object) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return MISSING
    value = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(value, list):
                value = value[int(part)]
            elif isinstance(value, Mapping):
                value = value[part]
            else:
                return MISSING
        except (KeyError, IndexError, ValueError):
            return MISSING
    return value


def _matches_hashed_file(record: object) -> bool:
    if not isinstance(record, Mapping):
        return False
    path = Path(str(record.get("path", "")))
    byte_count = record.get("byte_count")
    return (
        path.is_file()
        and isinstance(record.get("sha256"), str)
        and record.get("sha256") == _sha256(path)
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count == path.stat().st_size
    )


def _has_verified_dynamic_na_provenance(
    registry_path: Path, row: Mapping[str, str]
) -> bool:
    if row["note"].count("source_sha256=") != 1:
        return False
    note_match = DYNAMIC_NA_NOTE_RE.fullmatch(row["note"])
    if note_match is None:
        return False
    source_path = Path(row["source_json"])
    if not source_path.is_absolute():
        source_path = registry_path.parent / source_path
    if not source_path.is_file() or _sha256(source_path) != note_match["source_sha256"]:
        return False
    try:
        result = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(result, Mapping) or result.get("status") != "VERIFIED":
        return False
    run_id = result.get("run_id")
    if run_id != note_match["run_id"]:
        return False
    bindings = result.get("unavailable_bindings")
    if not isinstance(bindings, list):
        return False
    matching_bindings = [
        binding
        for binding in bindings
        if isinstance(binding, Mapping)
        and binding.get("token") == row["token"]
    ]
    if len(matching_bindings) != 1:
        return False
    binding = matching_bindings[0]
    if binding.get("reason_pointer") != row["json_pointer"]:
        return False
    reason = _resolve_json_pointer(result, binding.get("reason_pointer"))
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or reason.strip() != note_match["reason"]
    ):
        return False
    evidence = _resolve_json_pointer(result, binding.get("evidence_pointer"))
    if not isinstance(evidence, Mapping) or evidence.get("reason") != reason:
        return False
    return _matches_hashed_file(evidence.get("source"))


def _registry_matches_template(target: Path, expected_text: str) -> bool:
    if not target.is_file():
        return False
    expected_reader = csv.DictReader(io.StringIO(expected_text), delimiter="\t")
    with target.open(newline="", encoding="utf-8") as handle:
        actual_reader = csv.DictReader(handle, delimiter="\t")
        if tuple(actual_reader.fieldnames or ()) != FIELDS:
            return False
        expected_rows = list(expected_reader)
        actual_rows = list(actual_reader)
    if len(actual_rows) != len(expected_rows):
        return False
    frozen_fields = FIELDS[:8]
    for expected, actual in zip(expected_rows, actual_rows, strict=True):
        if any(actual[field] != expected[field] for field in frozen_fields):
            return False
        if expected["status"] == "N/A":
            if actual != expected:
                return False
        elif actual["status"] == "UNFILLED":
            if actual["source_json"] or actual["json_pointer"]:
                return False
        elif actual["status"] == "VERIFIED":
            if not actual["source_json"] or not actual["json_pointer"]:
                return False
            if "OVIOVO" in actual["token"]:
                return False
        elif actual["status"] == "N/A":
            if not _has_verified_dynamic_na_provenance(target, actual):
                return False
        else:
            return False
    return True


def check_package(output_dir: Path) -> int:
    artifacts = render_artifacts()
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory(prefix="benchmark-tables-") as temporary_dir:
        expected_dir = Path(temporary_dir)
        for name in ARTIFACT_NAMES:
            (expected_dir / name).write_text(artifacts[name], encoding="utf-8", newline="")
            target = output_dir / name
            if name == "benchmark_tokens.tsv":
                matches = _registry_matches_template(target, artifacts[name])
            else:
                matches = target.is_file() and target.read_bytes() == (expected_dir / name).read_bytes()
            if not matches:
                mismatches.append(name)
    if mismatches:
        print(
            "benchmark table package: FAIL (mismatch: " + ", ".join(mismatches) + ")",
            file=sys.stderr,
        )
        return 1
    print("benchmark table package: PASS")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write", help="write the frozen table package")
    write_parser.add_argument("--output-dir", type=Path, required=True)
    write_parser.add_argument("--force", action="store_true", help="reset a registry with VERIFIED rows")
    check_parser = subparsers.add_parser("check", help="verify committed artifacts")
    check_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "write":
            return write_package(args.output_dir, force=args.force)
        return check_package(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"benchmark table package: FAIL ({error})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
