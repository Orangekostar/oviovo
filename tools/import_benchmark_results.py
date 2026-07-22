#!/usr/bin/env python3
"""Import verified result JSON into the benchmark registry and tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.ovimap_paper_audit import (
    PAPER_PROTOCOL,
    SUMMARY_FIELDS,
    audit_paper_parity,
    summarize_feature_file,
)
from src.evaluation.json_contracts import loads_strict
from tools.benchmark_result_contract import (
    ResultContractError,
    binding_list,
    parse_result_identity,
    require_result_document,
    resolve_evidence_pointer,
    resolve_json_pointer,
    validate_registry_identity,
)


class ImportFailure(RuntimeError):
    """A result cannot be bound to the benchmark registry without ambiguity."""


ALLOWED_METHODS = {
    "T1": {"OPENFUSION", "OVIMAP", "CONCEPTGRAPHS", "DUALMAP", "OVIV2"},
    "T2": {
        "OVIMAP_FROZEN",
        "CONCEPTGRAPHS_FROZEN",
        "DUALMAP",
        "PANOPTIC_SHARED",
        "KHRONOS_OPEN",
        "KHRONOS_ORACLE",
        "OVIV2",
    },
    "T4": {"OVIMAP", "CONCEPTGRAPHS", "DUALMAP", "KHRONOS", "OVIV2_STATIC", "OVIV2"},
}

OVIMAP_REPLICA8_PAPER_METRICS = {
    "REPLICA8_MIOU": "semantic_miou",
    "REPLICA8_MACC": "semantic_macc",
    "REPLICA8_AP25": "class_agnostic_ap25",
    "REPLICA8_AP50": "class_agnostic_ap50",
}


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ImportFailure(f"required file does not exist: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_ovimap_paper_audit(
    result: Mapping[str, Any], result_path: Path
) -> Mapping[str, float]:
    audit = result.get("protocol_audit", {})
    if not isinstance(audit, Mapping):
        audit = {}
    if (
        audit.get("status") != "PASS"
        or audit.get("protocol_name") != "ovimap_cvpr2026_replica"
    ):
        raise ImportFailure("OVI-MAP T1 result requires a passing paper-parity audit")
    audit_path = Path(str(audit.get("path", "")))
    _require_file(audit_path)
    try:
        audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ImportFailure("cannot read OVI-MAP paper-parity audit file") from error
    if not isinstance(audit_document, Mapping):
        raise ImportFailure("OVI-MAP paper-parity audit file must contain an object")
    if audit_document.get("schema_version") != 1:
        raise ImportFailure("unsupported OVI-MAP paper-parity audit schema")
    if audit_document.get("status") != "PASS" or audit_document.get("failures") != []:
        raise ImportFailure("OVI-MAP paper-parity audit file must PASS without failures")
    if audit_document.get("protocol_name") != "ovimap_cvpr2026_replica":
        raise ImportFailure("OVI-MAP paper-parity audit protocol name mismatch")
    result_source = audit_document.get("result_source", {})
    if not isinstance(result_source, Mapping) or result_source.get("sha256") != _sha256(
        result_path
    ):
        raise ImportFailure("OVI-MAP paper-parity audit result hash mismatch")
    scene_artifacts = audit_document.get("scene_artifacts", {})
    feature_sources = audit_document.get("feature_sources", {})
    expected_scenes = set(PAPER_PROTOCOL["scene_ids"])
    if not isinstance(feature_sources, Mapping) or set(feature_sources) != expected_scenes:
        raise ImportFailure("OVI-MAP paper-parity audit feature sources must match Replica-8")
    if not isinstance(scene_artifacts, Mapping) or set(scene_artifacts) != expected_scenes:
        raise ImportFailure("OVI-MAP paper-parity audit scene artifacts must be an object")
    recomputed_summaries: dict[str, Mapping[str, Any]] = {}
    for scene in PAPER_PROTOCOL["scene_ids"]:
        source = feature_sources[scene]
        if not isinstance(source, Mapping):
            raise ImportFailure(f"OVI-MAP paper-parity feature source is invalid: {scene}")
        feature_path = Path(str(source.get("path", "")))
        _require_file(feature_path)
        if source.get("sha256") != _sha256(feature_path):
            raise ImportFailure(f"OVI-MAP paper-parity feature hash mismatch: {scene}")
        try:
            observed_summary = summarize_feature_file(feature_path)
        except (OSError, ValueError) as error:
            raise ImportFailure(f"cannot audit OVI-MAP feature source: {scene}") from error
        expected_summary = scene_artifacts[scene]
        if not isinstance(expected_summary, Mapping) or any(
            expected_summary.get(field) != observed_summary[field] for field in SUMMARY_FIELDS
        ):
            raise ImportFailure(f"OVI-MAP paper-parity feature summary mismatch: {scene}")
        recomputed_summaries[scene] = observed_summary
    recomputed = audit_paper_parity(result, recomputed_summaries)
    if recomputed["status"] != "PASS":
        raise ImportFailure("OVI-MAP result does not satisfy the paper protocol audit")
    return recomputed["validated_evidence_metrics"]


def _read_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    _require_file(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ImportFailure("registry has no header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    try:
        return resolve_json_pointer(document, pointer)
    except ResultContractError as error:
        raise ImportFailure(str(error)) from error


def _canonical_relative_path(raw: object, *, label: str) -> Path:
    if type(raw) is not str or not raw or "\\" in raw:
        raise ImportFailure(f"{label} must be a canonical relative path")
    path = Path(raw)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != raw
    ):
        raise ImportFailure(f"{label} must be a canonical relative path")
    return path


def _result_artifact_root(
    result: Mapping[str, Any], result_path: Path
) -> Path | None:
    record = result.get("artifact_root")
    if record is None:
        return None
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "resolution"}
        or record.get("path") != "."
        or record.get("resolution") != "result_parent"
    ):
        raise ImportFailure("result artifact root must be result_parent at canonical path .")
    return Path(os.path.abspath(result_path.parent))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ImportFailure(f"{label} contains a symbolic link component")
    except OSError as error:
        raise ImportFailure(f"{label} cannot be inspected safely") from error


def _stable_file_snapshot(
    path: Path, *, label: str, capture: bool
) -> tuple[str, int, bytes | None]:
    _reject_symlink_components(path, label=label)
    try:
        initial = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(initial.st_mode):
            raise ImportFailure(f"{label} must be a regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ImportFailure(f"{label} cannot be opened safely") from error
    chunks: list[bytes] | None = [] if capture else None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (initial.st_dev, initial.st_ino) != identity[:2]:
            raise ImportFailure(f"{label} changed before it was opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ImportFailure(f"{label} changed while it was read") from error
    finally:
        os.close(descriptor)
    _reject_symlink_components(path, label=label)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ImportFailure(f"{label} changed while it was read") from error
    if (
        byte_count != opened.st_size
        or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        != identity
        or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        != identity
    ):
        raise ImportFailure(f"{label} changed while it was read")
    return digest.hexdigest(), byte_count, None if chunks is None else b"".join(chunks)


def _require_hashed_file(
    record: Any,
    *,
    label: str,
    relative_to: Path | None = None,
    capture: bool = False,
) -> tuple[Path, bytes | None]:
    if not isinstance(record, Mapping):
        raise ImportFailure(f"{label} is invalid")
    raw_path = record.get("path")
    if relative_to is None:
        path = Path(str(raw_path))
    else:
        relative = _canonical_relative_path(raw_path, label=f"{label} path")
        path = Path(os.path.abspath(relative_to)) / relative
    observed, byte_count, captured = _stable_file_snapshot(
        path, label=label, capture=capture
    )
    if record.get("sha256") != observed:
        raise ImportFailure(f"{label} hash mismatch")
    declared_bytes = record.get("byte_count")
    if type(declared_bytes) is not int or declared_bytes != byte_count:
        raise ImportFailure(f"{label} hash or byte count mismatch")
    return path, captured


def _require_unavailable_evidence(
    result: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    token: str,
    reason: str,
    result_path: Path,
) -> None:
    try:
        evidence = resolve_evidence_pointer(result, binding, token=token)
    except ResultContractError as error:
        raise ImportFailure(str(error)) from error
    if not isinstance(evidence, Mapping) or evidence.get("reason") != reason:
        raise ImportFailure(f"unavailable evidence reason mismatch for {token}")
    source = evidence.get("source")
    artifact_root = _result_artifact_root(result, result_path)
    if artifact_root is None:
        _require_hashed_file(source, label="unavailable evidence source")
        return

    source_base = evidence.get("source_base")
    source_base_path, source_base_bytes = _require_hashed_file(
        source_base,
        label="unavailable evidence source base",
        relative_to=artifact_root,
        capture=True,
    )
    missing = evidence.get("missing_source")
    if missing is None:
        _require_hashed_file(
            source,
            label="unavailable evidence source",
            relative_to=source_base_path.parent,
        )
        return
    if (
        not isinstance(missing, Mapping)
        or set(missing) != {"path", "status"}
        or missing.get("status") != "MISSING"
        or source != source_base
    ):
        raise ImportFailure(f"unavailable MISSING evidence is invalid for {token}")
    relative_missing = _canonical_relative_path(
        missing.get("path"), label="unavailable MISSING source path"
    )
    missing_path = source_base_path.parent / relative_missing
    _reject_symlink_components(missing_path.parent, label="unavailable MISSING source")
    if os.path.lexists(missing_path):
        raise ImportFailure(f"unavailable MISSING source unexpectedly exists for {token}")
    try:
        declaration = loads_strict(
            (source_base_bytes or b"").decode("utf-8"),
            label="unavailable metrics declaration",
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ImportFailure("unavailable metrics declaration is invalid") from error
    if not isinstance(declaration, Mapping):
        raise ImportFailure("unavailable metrics declaration must contain an object")
    declarations = declaration.get("sources")
    if (
        declaration.get("status") != "PARTIAL"
        or not isinstance(declarations, list)
        or sum(item == dict(missing) for item in declarations) != 1
    ):
        raise ImportFailure(f"MISSING source is not bound by its metrics JSON for {token}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _registry_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _source_label(result_path: Path, registry_path: Path) -> str:
    try:
        return result_path.resolve().relative_to(registry_path.parent.resolve()).as_posix()
    except ValueError:
        return str(result_path.resolve())


def import_results(
    registry_path: str | Path,
    result_paths: Sequence[str | Path],
    markdown_template: str | Path,
    latex_template: str | Path,
    markdown_output: str | Path,
    latex_output: str | Path,
) -> dict[str, Path]:
    registry_path = Path(registry_path)
    result_paths = [Path(path) for path in result_paths]
    markdown_template = Path(markdown_template)
    latex_template = Path(latex_template)
    markdown_output = Path(markdown_output)
    latex_output = Path(latex_output)
    for path in (markdown_template, latex_template, *result_paths):
        _require_file(path)

    fieldnames, rows = _read_registry(registry_path)
    by_token = {row.get("token", ""): row for row in rows}
    if len(by_token) != len(rows):
        raise ImportFailure("registry contains duplicate tokens")

    replacements: dict[str, str] = {}
    bound_tokens: set[str] = set()
    seen_result_paths: set[Path] = set()
    for result_path in result_paths:
        canonical_path = result_path.resolve()
        if canonical_path in seen_result_paths:
            raise ImportFailure(f"duplicate result file: {result_path}")
        seen_result_paths.add(canonical_path)
        try:
            raw_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeError) as error:
            raise ImportFailure(f"cannot read result JSON: {result_path}") from error
        try:
            result = require_result_document(raw_result)
            identity = parse_result_identity(result)
            bindings = binding_list(result, "token_bindings", required=True)
            unavailable_bindings = binding_list(result, "unavailable_bindings")
        except ResultContractError as error:
            raise ImportFailure(str(error)) from error
        if result.get("status") != "VERIFIED":
            raise ImportFailure(f"result status must be VERIFIED: {result_path}")
        result_method = identity.method
        if "OVIOVO" in result_method.upper():
            raise ImportFailure("OVIOVO results are outside this importer scope")
        if not bindings and not unavailable_bindings:
            raise ImportFailure("VERIFIED result must provide result bindings")
        ovimap_replica8_bindings = [
            binding
            for binding in bindings
            if (
                (row := by_token.get(binding["token"])) is not None
                and row.get("table") == "T1"
                and row.get("method") == "OVIMAP"
                and row.get("dataset") == "Replica"
                and row.get("split") == "replica_8_compat"
            )
        ]
        if result_method == "OVIMAP" and ovimap_replica8_bindings:
            evidence_metrics = _require_ovimap_paper_audit(result, result_path)
            for binding in ovimap_replica8_bindings:
                row = by_token[binding["token"]]
                evidence_name = OVIMAP_REPLICA8_PAPER_METRICS.get(row.get("metric", ""))
                if evidence_name is None:
                    raise ImportFailure(
                        f"OVI-MAP {row.get('metric')} is not defined by the paper evaluators"
                    )
                observed = _resolve_json_pointer(result, str(binding.get("json_pointer", "")))
                expected = evidence_metrics[evidence_name]
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, (int, float))
                    or not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
                ):
                    raise ImportFailure(
                        f"OVI-MAP paper evaluator metric mismatch for {row.get('metric')}"
                    )

        for binding in bindings:
            token = binding["token"]
            if "OVIOVO" in token.upper():
                raise ImportFailure(f"OVIOVO token is outside this importer scope: {token}")
            if token in bound_tokens:
                raise ImportFailure(f"duplicate token binding: {token}")
            bound_tokens.add(token)
            row = by_token.get(token)
            if row is None:
                raise ImportFailure(f"token is not present in registry: {token}")
            if row.get("method") not in ALLOWED_METHODS.get(row.get("table", ""), set()):
                raise ImportFailure(f"token is outside the allowed T1/T2/T4 baseline scope: {token}")
            try:
                validate_registry_identity(identity, row, token=token)
            except ResultContractError as error:
                raise ImportFailure(str(error)) from error
            try:
                registry_precision = int(row.get("precision", ""))
                binding_precision = int(binding.get("precision"))
            except (TypeError, ValueError) as error:
                raise ImportFailure(f"invalid precision for {token}") from error
            if registry_precision != binding_precision:
                raise ImportFailure(f"precision mismatch for {token}")
            pointer = str(binding.get("json_pointer", ""))
            value = _resolve_json_pointer(result, pointer)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ImportFailure(f"JSON pointer for {token} must resolve to a number")
            number = float(value)
            if not math.isfinite(number):
                raise ImportFailure(f"metric for {token} must be finite")
            formatted = f"{number:.{registry_precision}f}"
            replacements[token] = formatted
            row["source_json"] = _source_label(result_path, registry_path)
            row["json_pointer"] = pointer
            row["status"] = "VERIFIED"
            row["note"] = f"Imported from verified run {identity.run_id}."

        for binding in unavailable_bindings:
            token = binding["token"]
            if "OVIOVO" in token.upper():
                raise ImportFailure(f"OVIOVO token is outside this importer scope: {token}")
            if token in bound_tokens:
                raise ImportFailure(f"duplicate token binding: {token}")
            bound_tokens.add(token)
            row = by_token.get(token)
            if row is None:
                raise ImportFailure(f"token is not present in registry: {token}")
            if row.get("method") not in ALLOWED_METHODS.get(row.get("table", ""), set()):
                raise ImportFailure(f"token is outside the allowed T1/T2/T4 baseline scope: {token}")
            try:
                validate_registry_identity(identity, row, token=token)
            except ResultContractError as error:
                raise ImportFailure(str(error)) from error
            if row.get("status") != "UNFILLED":
                raise ImportFailure(f"source-bound N/A requires an UNFILLED registry row: {token}")
            pointer = str(binding.get("reason_pointer", ""))
            reason = _resolve_json_pointer(result, pointer)
            if not isinstance(reason, str) or not reason.strip():
                raise ImportFailure(f"unavailable reason for {token} must be non-empty text")
            _require_unavailable_evidence(
                result,
                binding,
                token=token,
                reason=reason,
                result_path=result_path,
            )
            replacements[token] = "--"
            row["source_json"] = _source_label(result_path, registry_path)
            row["json_pointer"] = pointer
            row["status"] = "N/A"
            row["note"] = (
                f"N/A from verified run {identity.run_id} "
                f"[source_sha256={_sha256(result_path)}]: {reason.strip()}"
            )

    markdown = markdown_template.read_text(encoding="utf-8")
    latex = latex_template.read_text(encoding="utf-8")
    for token, formatted in replacements.items():
        placeholder = "{{" + token + "}}"
        markdown = markdown.replace(placeholder, formatted)
        latex = latex.replace(r"\verb|" + placeholder + "|", formatted)
        latex = latex.replace(placeholder, formatted)

    _atomic_text(registry_path, _registry_text(fieldnames, rows))
    _atomic_text(markdown_output, markdown)
    _atomic_text(latex_output, latex)
    return {"markdown": markdown_output, "latex": latex_output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--markdown-template", type=Path, required=True)
    parser.add_argument("--latex-template", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, required=True)
    args = parser.parse_args()
    import_results(
        args.registry,
        args.result,
        args.markdown_template,
        args.latex_template,
        args.markdown_output,
        args.latex_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
