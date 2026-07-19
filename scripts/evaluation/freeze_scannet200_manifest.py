#!/usr/bin/env python3
"""Freeze a ScanNet200 five-scene manifest from validated official data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.datasets.scannet200 import (  # noqa: E402
    SCANNET200_5_SCENES,
    validate_scannet200_split,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_at(args: argparse.Namespace) -> str:
    if args.frozen_at is not None:
        value = args.frozen_at.strip()
        if not value:
            raise ValueError("--frozen-at must be non-empty")
        return value
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None:
        raise ValueError("--frozen-at or SOURCE_DATE_EPOCH is required")
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and not args.replace:
        raise ValueError(f"output already exists: {output}")
    vocabulary = args.vocabulary.expanduser().resolve()
    if not vocabulary.is_file() or vocabulary.stat().st_size <= 0:
        raise ValueError(f"vocabulary file is missing or empty: {vocabulary}")
    actual_vocabulary_hash = _sha256(vocabulary)
    if actual_vocabulary_hash != args.vocabulary_sha256.lower():
        raise ValueError("vocabulary hash does not match --vocabulary-sha256")
    inventories = validate_scannet200_split(
        args.dataset_root,
        SCANNET200_5_SCENES,
    )
    scenes = []
    for inventory in inventories:
        scenes.append(
            {
                "scene_id": inventory.scene_id,
                "split_role": "static_candidate",
                "pose_source": "sensor",
                "depth_source": "sensor",
                "files": {
                    role: asdict(record)
                    for role, record in sorted(inventory.files.items())
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_id": "scannet200_5_static_v1",
        "dataset": "ScanNet200",
        "split": "scannet200_5",
        "frozen_at": _frozen_at(args),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "vocabulary": {
            "name": "scannet200",
            "source_path": str(vocabulary),
            "source_sha256": actual_vocabulary_hash,
        },
        "scenes": scenes,
    }
    _write_json_atomic(output, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--vocabulary-sha256", required=True)
    parser.add_argument("--frozen-at")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = freeze(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"manifest_id": manifest["manifest_id"], "scenes": 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
