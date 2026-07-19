#!/usr/bin/env python3
"""Capture released OVI-MAP evaluator output as non-headline diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.baselines.ovimap_diagnostics import (  # noqa: E402
    parse_official_instance_output,
    parse_official_semantic_output,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--instance-stdout", type=Path, required=True)
    parser.add_argument("--semantic-stdout", type=Path, required=True)
    parser.add_argument("--neutral-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_paths = {
        "instance_stdout": args.instance_stdout,
        "semantic_stdout": args.semantic_stdout,
        "neutral_metrics": args.neutral_metrics,
    }
    sources = {name: _source(path) for name, path in source_paths.items()}
    neutral_metrics = json.loads(args.neutral_metrics.read_text(encoding="utf-8"))
    if not isinstance(neutral_metrics, dict):
        raise ValueError("neutral metrics must be a JSON object")

    payload = {
        "schema_version": 1,
        "scene_id": args.scene_id,
        "source_protocol": "released_ovimap_diagnostic",
        "headline_token_eligible": False,
        "instance": parse_official_instance_output(
            args.instance_stdout.read_text(encoding="utf-8"),
            scene_id=args.scene_id,
        ),
        "semantic": parse_official_semantic_output(
            args.semantic_stdout.read_text(encoding="utf-8"),
            scene_id=args.scene_id,
        ),
        "sources": sources,
    }
    _write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
