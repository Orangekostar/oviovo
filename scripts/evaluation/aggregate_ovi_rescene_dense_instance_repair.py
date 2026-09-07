#!/usr/bin/env python3
"""Aggregate pair-scoped dense instance repair results without mutation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SYSTEM_TABLES = (
    "endpoint_diagnosis.csv",
    "instance_metrics.csv",
    "identity_metrics.csv",
    "runs.csv",
)
_ASSOCIATION_SOURCE = "fixed_p2_association/association_metrics.csv"


class DenseResultAggregationError(ValueError):
    """Raised when pair-scoped result tables cannot be aggregated exactly."""


def _read_table(
    path: Path, *, pair_id: str, pair_column: str
) -> tuple[tuple[str, ...], list[dict[str, str]], bytes]:
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DenseResultAggregationError(f"result table is unavailable: {path}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = tuple(reader.fieldnames or ())
    if not header or len(header) != len(set(header)) or pair_column not in header:
        raise DenseResultAggregationError(f"result table header is invalid: {path}")
    rows = list(reader)
    if not rows or any(
        tuple(row) != header or row[pair_column] != pair_id for row in rows
    ):
        raise DenseResultAggregationError(f"result table pair identity is invalid: {path}")
    return header, rows, content


def _csv_bytes(header: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _record(path: Path, content: bytes) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def aggregate_dense_results(
    sources: Sequence[tuple[str, Path]], *, output_root: Path
) -> dict[str, object]:
    """Aggregate explicit source roots in caller-provided pair order."""

    source_rows = tuple((pair_id, Path(root)) for pair_id, root in sources)
    if (
        not source_rows
        or len({pair_id for pair_id, _root in source_rows}) != len(source_rows)
        or any(not pair_id or not root.is_dir() for pair_id, root in source_rows)
    ):
        raise DenseResultAggregationError("aggregate sources are invalid")
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise DenseResultAggregationError("aggregate output already exists")
    table_sources = {
        **{
            name: (name, "pair_id" if name == "endpoint_diagnosis.csv" else "pair")
            for name in _SYSTEM_TABLES
        },
        "association_metrics.csv": (_ASSOCIATION_SOURCE, "pair"),
    }
    aggregate_content: dict[str, bytes] = {}
    source_records: dict[str, dict[str, dict[str, object]]] = {}
    for output_name, (relative_name, pair_column) in table_sources.items():
        expected_header = None
        combined: list[dict[str, str]] = []
        for pair_id, root in source_rows:
            path = root / relative_name
            header, rows, content = _read_table(
                path, pair_id=pair_id, pair_column=pair_column
            )
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise DenseResultAggregationError(
                    f"result table headers differ: {output_name}"
                )
            combined.extend(rows)
            source_records.setdefault(pair_id, {})[output_name] = _record(path, content)
        assert expected_header is not None
        aggregate_content[output_name] = _csv_bytes(expected_header, combined)
    manifest = {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_DENSE_INSTANCE_REPAIR_AGGREGATE_V1",
        "status": "PASS",
        "pair_ids": [pair_id for pair_id, _root in source_rows],
        "sources": source_records,
        "tables": {
            name: {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
            for name, content in sorted(aggregate_content.items())
        },
    }
    manifest_content = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, content in aggregate_content.items():
            _write(staging / name, content)
        _write(staging / "manifest.json", manifest_content)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _source(value: str) -> tuple[str, Path]:
    pair_id, separator, raw_path = value.partition("=")
    if not separator or not pair_id or not raw_path:
        raise argparse.ArgumentTypeError("source must use PAIR_ID=PATH")
    return pair_id, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = aggregate_dense_results(
        arguments.source, output_root=arguments.output_root
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DenseResultAggregationError", "aggregate_dense_results"]
