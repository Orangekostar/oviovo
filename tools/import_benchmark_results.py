#!/usr/bin/env python3
"""Import verified result JSON into the benchmark registry and tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
    },
    "T4": {"OVIMAP", "CONCEPTGRAPHS", "DUALMAP", "KHRONOS"},
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


def _require_ovimap_paper_audit(result: Mapping[str, Any]) -> None:
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
    if _sha256(audit_path) != str(audit.get("sha256", "")):
        raise ImportFailure("OVI-MAP paper-parity audit hash mismatch")


def _read_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    _require_file(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ImportFailure("registry has no header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ImportFailure(f"invalid JSON pointer: {pointer}")
    value = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(value, list):
                value = value[int(part)]
            elif isinstance(value, Mapping):
                value = value[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as error:
            raise ImportFailure(f"JSON pointer does not resolve: {pointer}") from error
    return value


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
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ImportFailure(f"cannot read result JSON: {result_path}") from error
        if result.get("status") != "VERIFIED":
            raise ImportFailure(f"result status must be VERIFIED: {result_path}")
        result_method = str(result.get("method", {}).get("key", ""))
        if "OVIOVO" in result_method.upper():
            raise ImportFailure("OVIOVO results are outside this importer scope")
        dataset = result.get("dataset", {})
        result_dataset = str(dataset.get("name", ""))
        result_splits = {str(value) for value in dataset.get("splits", ())}
        bindings = result.get("token_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ImportFailure("VERIFIED result must provide token_bindings")
        if result_method == "OVIMAP" and any(
            (row := by_token.get(str(binding.get("token", "")))) is not None
            and row.get("table") == "T1"
            and row.get("method") == "OVIMAP"
            for binding in bindings
        ):
            _require_ovimap_paper_audit(result)

        for binding in bindings:
            token = str(binding.get("token", ""))
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
            if row.get("method") != result_method:
                raise ImportFailure(
                    f"method mismatch for {token}: registry={row.get('method')} result={result_method}"
                )
            if row.get("dataset") != result_dataset or row.get("split") not in result_splits:
                raise ImportFailure(f"dataset or split mismatch for {token}")
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
            row["note"] = f"Imported from verified run {result.get('run_id', '')}.".strip()

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
