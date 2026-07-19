#!/usr/bin/env python3
"""Audit an OVI-MAP Replica result against the CVPR 2026 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.ovimap_paper_audit import (
    audit_paper_parity,
    summarize_feature_file,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _scene_features(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --scene-feature value: {value}")
        scene_id, raw_path = value.split("=", 1)
        if not scene_id or scene_id in parsed:
            raise ValueError(f"duplicate or empty scene ID: {scene_id}")
        parsed[scene_id] = Path(raw_path)
    return parsed


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--scene-feature", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        scene_features = _scene_features(args.scene_feature)
        result_source = _source(args.result)
        result = json.loads(args.result.read_text(encoding="utf-8"))
        feature_sources = {
            scene_id: _source(path) for scene_id, path in sorted(scene_features.items())
        }
        summaries = {
            scene_id: summarize_feature_file(path)
            for scene_id, path in sorted(scene_features.items())
        }
        audit = audit_paper_parity(result, summaries)
        audit["result_source"] = result_source
        audit["feature_sources"] = feature_sources
        _atomic_json(args.output, audit)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
