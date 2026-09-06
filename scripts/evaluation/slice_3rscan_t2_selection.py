#!/usr/bin/env python3
"""Slice selected pairs from the frozen 3RScan T=2 selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.json_contracts import loads_strict


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: str | Path, *, label: str) -> Path:
    absolute = _absolute(path)
    try:
        record = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {absolute}") from error
    if not stat.S_ISREG(record.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {absolute}")
    return absolute


def _read_source(path: str | Path) -> tuple[dict[str, Any], bytes, Path]:
    source = _regular_file(path, label="source manifest")
    try:
        content = source.read_bytes()
        payload = loads_strict(content.decode("utf-8"), label="source manifest")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"source manifest is not valid strict UTF-8 JSON: {source}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"source manifest must be a JSON object: {source}")
    return payload, content, source


def _validate_pair_ids(pair_ids: Sequence[str]) -> list[str]:
    if isinstance(pair_ids, (str, bytes)) or not isinstance(pair_ids, Sequence):
        raise TypeError("pair_ids must be a non-empty sequence of strings")
    values = list(pair_ids)
    if not values:
        raise ValueError("pair_ids must be non-empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("pair_ids must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("pair_ids must be unique")
    return values


def _atomic_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"output already exists: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def slice_selection_manifest(
    source_path: str | Path,
    pair_ids: Sequence[str],
    role: str,
    output_path: str | Path,
) -> dict[str, object]:
    """Write a transfer-only slice of a frozen 3RScan T=2 selection."""

    requested_ids = _validate_pair_ids(pair_ids)
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")

    source_payload, source_bytes, source = _read_source(source_path)
    if source_payload.get("artifact_id") != "RSCAN_T2_DEV_V1":
        raise ValueError("source manifest artifact_id must be RSCAN_T2_DEV_V1")
    if source_payload.get("status") != "PASS":
        raise ValueError("source manifest status must be PASS")
    pairs = source_payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("source manifest pairs must be a non-empty list")

    selected_pairs: list[object] = []
    for pair_id in requested_ids:
        matches = [
            pair
            for pair in pairs
            if isinstance(pair, Mapping) and pair.get("pair_id") == pair_id
        ]
        if not matches:
            raise ValueError(f"pair ID not found in source manifest: {pair_id}")
        if len(matches) != 1:
            raise ValueError(f"pair ID must occur exactly once in source manifest: {pair_id}")
        selected_pairs.append(matches[0])

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": "RSCAN_T2_SELECTION_SLICE_V1",
        "status": "PASS",
        "role": role,
        "source_manifest": {
            "path": str(source),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "byte_count": len(source_bytes),
        },
        "selected_pair_count": len(selected_pairs),
        "pair_ids": requested_ids,
        "pairs": selected_pairs,
    }
    _atomic_json_no_clobber(_absolute(output_path), manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pair-id", action="append", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = slice_selection_manifest(
            args.source,
            args.pair_id,
            args.role,
            args.output,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(_absolute(args.output)),
                "selected_pair_count": manifest["selected_pair_count"],
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
