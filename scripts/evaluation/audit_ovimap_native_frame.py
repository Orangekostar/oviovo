#!/usr/bin/env python3
"""Audit one native OVI-MAP frame and write hash-bound JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.baselines.ovimap_native import audit_native_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--raycast-npy", required=True, type=Path)
    parser.add_argument("--colors-json", required=True, type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"cannot read instance mask: {args.mask}")
    if mask.shape != (args.height, args.width):
        raise ValueError("instance mask shape does not match declared dimensions")
    raycast = np.load(args.raycast_npy, allow_pickle=False)
    colors = json.loads(args.colors_json.read_text(encoding="utf-8"))
    if not isinstance(colors, list):
        raise ValueError("colors JSON must contain a list")

    payload = audit_native_frame(mask, raycast, colors)
    payload["schema_version"] = 1
    payload["inputs"] = {
        "mask": _input_record(args.mask),
        "raycast_npy": _input_record(args.raycast_npy),
        "colors_json": _input_record(args.colors_json),
    }
    _atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
