#!/usr/bin/env python3
"""Publish an immutable diagnostic audit for projected OVIV2 Replica labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_oviv2_replica import (  # noqa: E402
    _load_manifest,
    _normalize_label,
    _normalized_aliases,
    _verify_scene_inputs,
    load_replica_ground_truth,
)
from src.evaluation.oviv2_diagnostics import diagnose_projected_labels  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        published = True
    finally:
        try:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        finally:
            if published:
                _fsync_directory(path.parent)


def _input_sha256(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pred_semantic": _sha256(args.pred_semantic),
        "pred_entity": _sha256(args.pred_entity),
        "gt_mesh": _sha256(args.gt_mesh),
        "gt_info": _sha256(args.gt_info),
        "manifest": _sha256(args.manifest),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-semantic", type=Path, required=True)
    parser.add_argument("--pred-entity", type=Path, required=True)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--gt-info", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-overlap-vertices", type=int, default=100)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    hashes_before = _input_sha256(args)
    manifest, scene = _load_manifest(args.manifest, args.scene)
    _verify_scene_inputs(scene, args.gt_mesh, args.gt_info)
    aliases = _normalized_aliases(manifest.get("aliases", {}))
    classes = [
        _normalize_label(label, aliases)
        for label in manifest["vocabulary"]["classes"]
    ]
    class_to_id = {label: index + 1 for index, label in enumerate(classes)}
    ground_truth = load_replica_ground_truth(
        args.gt_mesh,
        args.gt_info,
        class_to_id=class_to_id,
        aliases=aliases,
    )
    predicted_semantic = np.load(args.pred_semantic, allow_pickle=False)
    predicted_entity = np.load(args.pred_entity, allow_pickle=False)
    report = diagnose_projected_labels(
        predicted_semantic,
        predicted_entity,
        ground_truth.semantic_ids,
        ground_truth.instance_ids,
        valid_semantic_ids=set(class_to_id.values()),
        minimum_overlap_vertices=args.minimum_overlap_vertices,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )
    hashes_after = _input_sha256(args)
    if hashes_before != hashes_after:
        raise RuntimeError("diagnostic inputs changed during computation")
    report.update(
        {
            "scene": args.scene,
            "protocol": {
                "manifest_id": manifest.get("manifest_id"),
                "minimum_overlap_vertices": args.minimum_overlap_vertices,
                "minimum_overlap_fraction": args.minimum_overlap_fraction,
            },
            "input_sha256": hashes_before,
        }
    )
    _atomic_json(args.output, report, overwrite=args.overwrite)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "current_miou": report["semantic"]["current_miou"],
                "oracle_miou": report["semantic"]["oracle_miou"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
