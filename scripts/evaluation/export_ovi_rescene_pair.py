#!/usr/bin/env python3
"""Export two source-bound OVI snapshots as a ReScene neural sample pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.exporters.oviovo import read_map_snapshot
from src.oviv2.ovi_rescene_adapter import (
    AdapterConfig,
    AdapterError,
    adapt_visit_pair,
    write_neural_sample_artifact,
)
from src.oviv2.two_visit_contracts import VisitMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for visit in ("t0", "t1"):
        parser.add_argument(f"--{visit}-snapshot", type=Path, required=True)
        parser.add_argument(f"--{visit}-entities", type=Path, required=True)
        parser.add_argument(f"--{visit}-frame-start", type=int, required=True)
        parser.add_argument(f"--{visit}-frame-end", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--coordinate-frame-id", required=True)
    parser.add_argument("--map-voxel-size-m", type=float, default=0.01)
    parser.add_argument("--neural-voxel-size-m", type=float, default=0.02)
    parser.add_argument(
        "--feature-schema", choices=("rgb", "rgb_normals"), default="rgb"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source_bytes = args.source_manifest.read_bytes()
        source_payload = json.loads(source_bytes.decode("utf-8"))
        if source_payload.get("status") != "EXTERNAL_SOURCE_PASS":
            raise AdapterError("source manifest is not EXTERNAL_SOURCE_PASS")
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        snapshots = (
            read_map_snapshot(args.t0_snapshot, args.t0_entities),
            read_map_snapshot(args.t1_snapshot, args.t1_entities),
        )
        visits = tuple(
            VisitMap(
                visit_id=visit_id,
                snapshot=snapshots[visit_id],
                coordinate_frame_id=args.coordinate_frame_id,
                source_manifest_sha256=source_sha256,
                map_voxel_size_m=args.map_voxel_size_m,
                observed_frame_start=getattr(args, f"t{visit_id}_frame_start"),
                observed_frame_end=getattr(args, f"t{visit_id}_frame_end"),
            )
            for visit_id in (0, 1)
        )
        pair = adapt_visit_pair(
            visits[0],
            visits[1],
            AdapterConfig(args.neural_voxel_size_m, args.feature_schema),
        )
        paths = write_neural_sample_artifact(pair, args.output)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "pair_sha256": pair.content_sha256(),
                "manifest": str(paths.manifest),
                "arrays": str(paths.arrays),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
