"""Machine-readable method matrix and fixed five-table study reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_write_json


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("report records cannot contain NaN or Infinity")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported report value: {type(value).__name__}")


def _branch(method_id: str) -> str:
    if method_id == "N0":
        return "N0"
    if method_id.startswith("S_"):
        return "S"
    if method_id.startswith("G_"):
        return "G"
    if method_id.startswith("Q_"):
        return "Q"
    if method_id.startswith("COMBO_"):
        return "COMBO"
    return "CONTROL"


@dataclass(frozen=True)
class MethodMatrixRow:
    method_id: str
    branch: str
    required: bool
    disposition: str
    reason: str | None
    metrics: Mapping[str, Any]
    evidence_path: str | None = None
    reused_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        metrics = {"uap": None, "miou": None, **dict(self.metrics)}
        return _json_value(
            {
                "method_id": self.method_id,
                "branch": self.branch,
                "required": self.required,
                "disposition": self.disposition,
                "reason": self.reason,
                "metrics": metrics,
                "evidence_path": self.evidence_path,
                "reused_from": self.reused_from,
            }
        )


def build_method_matrix(
    *,
    required_methods: Sequence[str],
    measured: Mapping[str, Mapping[str, Any]],
    blocked: Mapping[str, str] | None = None,
    not_required: Mapping[str, str] | None = None,
    reused: Mapping[str, str] | None = None,
    evidence_paths: Mapping[str, str] | None = None,
) -> tuple[MethodMatrixRow, ...]:
    blocked = {} if blocked is None else blocked
    not_required = {} if not_required is None else not_required
    reused = {} if reused is None else reused
    evidence_paths = {} if evidence_paths is None else evidence_paths
    methods = tuple(str(value) for value in required_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("required method IDs must be nonempty and unique")
    unknown = (set(measured) | set(blocked) | set(not_required) | set(reused)) - set(methods)
    if unknown:
        raise ValueError(f"method matrix contains undeclared methods: {sorted(unknown)}")
    rows: list[MethodMatrixRow] = []
    for method in methods:
        dispositions = sum(
            method in collection
            for collection in (measured, blocked, not_required, reused)
        )
        if dispositions != 1:
            raise ValueError(f"method {method} must have exactly one disposition")
        if method in measured:
            disposition, reason, metrics, reused_from = (
                "MEASURED",
                None,
                measured[method],
                None,
            )
        elif method in blocked:
            disposition, reason, metrics, reused_from = (
                "BLOCKED",
                blocked[method],
                {},
                None,
            )
        elif method in not_required:
            disposition, reason, metrics, reused_from = (
                "NOT_REQUIRED",
                not_required[method],
                {},
                None,
            )
        else:
            disposition, reason, metrics, reused_from = (
                "REUSED",
                None,
                {},
                reused[method],
            )
        rows.append(
            MethodMatrixRow(
                method,
                _branch(method),
                True,
                disposition,
                reason,
                _json_value(metrics),
                evidence_paths.get(method),
                reused_from,
            )
        )
    return tuple(rows)


def _format(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str:
    output = [title, "", "| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    output.extend("| " + " | ".join(_format(value) for value in row) + " |" for row in rows)
    if not rows:
        output.append("| " + " | ".join(["none", *(["null"] * (len(headers) - 1))]) + " |")
    return "\n".join(output)


def render_results(
    matrix: Sequence[MethodMatrixRow],
    *,
    final_candidate: str,
    science_status: str | None = None,
    confirmation_status: str | None = None,
    supporting_evidence: Sequence[str] = (),
) -> str:
    rows = tuple(matrix)
    if not final_candidate:
        raise ValueError("final candidate must be explicit")
    table_a = _table(
        "## Table A. Branch metrics and cost",
        ("Method", "Branch", "Status", "uAP", "mIoU", "Cost", "Evidence"),
        [
            (
                row.method_id,
                row.branch,
                row.disposition,
                row.metrics.get("uap"),
                row.metrics.get("miou"),
                row.metrics.get("logical_attempts", row.metrics.get("added_cost")),
                row.evidence_path,
            )
            for row in rows
        ],
    )
    semantic = [row for row in rows if row.branch == "S"]
    table_b = _table(
        "## Table B. Semantic suggestion, adoption, and effect",
        ("Method", "Suggestions", "Adoptions", "Corrections", "Damage", "Status"),
        [
            (
                row.method_id,
                row.metrics.get("suggestions"),
                row.metrics.get("adoptions"),
                row.metrics.get("corrections"),
                row.metrics.get("damage"),
                row.disposition,
            )
            for row in semantic
        ],
    )
    geometry = [row for row in rows if row.branch == "G"]
    table_c = _table(
        "## Table C. Geometry partitions and preservation",
        ("Method", "Hypotheses", "Changed partitions", "Preserved objects", "Status"),
        [
            (
                row.method_id,
                row.metrics.get("hypotheses"),
                row.metrics.get("changed_partitions"),
                row.metrics.get("preserved_objects"),
                row.disposition,
            )
            for row in geometry
        ],
    )
    query = [row for row in rows if row.branch == "Q"]
    table_d = _table(
        "## Table D. Query acquisition and readout",
        ("Method", "Available", "Attempts", "Successes", "Crops", "uAP", "Status"),
        [
            (
                row.method_id,
                row.metrics.get("available_requests"),
                row.metrics.get("logical_attempts"),
                row.metrics.get("successful_requests"),
                row.metrics.get("crop_inputs"),
                row.metrics.get("uap"),
                row.disposition,
            )
            for row in query
        ],
    )
    table_e = _table(
        "## Table E. Selection, combinations, and confirmation",
        ("Final candidate", "Science", "Confirmation", "Publication evidence"),
        [
            (
                final_candidate,
                science_status
                or ("NO_NET_GAIN" if final_candidate == "N0" else "RETAINED"),
                confirmation_status
                or (
                    "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
                    if final_candidate == "N0"
                    else "PENDING"
                ),
                next((row.evidence_path for row in rows if row.method_id == final_candidate), None),
            )
        ],
    )
    rendered = (
        f"# OVI-MAP Module Validation Results\n\n{table_a}\n\n{table_b}\n\n"
        f"{table_c}\n\n{table_d}\n\n{table_e}\n"
    )
    if supporting_evidence:
        rendered += "\nSupporting execution evidence:\n\n"
        rendered += "\n".join(f"- {line}" for line in supporting_evidence) + "\n"
    return rendered


@dataclass(frozen=True)
class ReleaseStatuses:
    implementation: str
    experiment: str
    science: str
    confirmation: str
    publication: str

    def __post_init__(self) -> None:
        if any(not value for value in asdict(self).values()):
            raise ValueError("all five release statuses must be explicit")


def render_handoff(
    statuses: ReleaseStatuses,
    *,
    branch: str,
    commit: str,
    evidence_paths: Sequence[str],
    next_action: str,
    execution_details: Sequence[str] = (),
    reproduction_commands: Sequence[str] = (),
) -> str:
    if not branch or len(commit) != 40 or not next_action:
        raise ValueError("handoff branch, full commit, and next action are required")
    lines = [
        "# OVI-MAP Module Validation Handoff",
        "",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Implementation: `{statuses.implementation}`",
        f"- Experiment: `{statuses.experiment}`",
        f"- Science: `{statuses.science}`",
        f"- Confirmation: `{statuses.confirmation}`",
        f"- Publication: `{statuses.publication}`",
        "",
        "Evidence:",
        "",
    ]
    lines.extend(f"- `{path}`" for path in evidence_paths)
    if execution_details:
        lines.extend(("", "Executed scope:", ""))
        lines.extend(f"- {detail}" for detail in execution_details)
    if reproduction_commands:
        lines.extend(("", "Reproduction commands:", ""))
        lines.extend(f"- `{command}`" for command in reproduction_commands)
    lines.extend(("", f"Next evidence-based action: {next_action}", ""))
    return "\n".join(lines)


def write_method_matrix(path: Path | str, rows: Sequence[MethodMatrixRow]) -> None:
    atomic_write_json(
        Path(path),
        {
            "schema_version": 1,
            "artifact_type": "OVIMAP_MODULE_METHOD_MATRIX",
            "rows": [row.to_dict() for row in rows],
        },
    )


def validate_rendered_json(rows: Sequence[MethodMatrixRow]) -> None:
    json.dumps([row.to_dict() for row in rows], allow_nan=False)


__all__ = [
    "MethodMatrixRow",
    "ReleaseStatuses",
    "build_method_matrix",
    "render_handoff",
    "render_results",
    "validate_rendered_json",
    "write_method_matrix",
]
