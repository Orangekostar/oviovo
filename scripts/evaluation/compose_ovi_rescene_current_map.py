#!/usr/bin/env python3
"""Compose two neutral OVI snapshots using relations and t1 signed visibility."""

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
from src.oviv2.two_visit_contracts import PairRelation, VisitMap
from src.oviv2.two_visit_current_map import (
    CompositionConfig,
    SignedVisibilityGrid,
    compose_current_map,
    write_two_visit_current_map,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for visit in ("t0", "t1"):
        parser.add_argument(f"--{visit}-snapshot", type=Path, required=True)
        parser.add_argument(f"--{visit}-entities", type=Path, required=True)
        parser.add_argument(f"--{visit}-frame-start", type=int, required=True)
        parser.add_argument(f"--{visit}-frame-end", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--signed-visibility", type=Path, required=True)
    parser.add_argument("--coordinate-frame-id", required=True)
    parser.add_argument("--map-voxel-size-m", type=float, default=0.01)
    parser.add_argument("--composition-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _relations(payload: object) -> tuple[PairRelation, ...]:
    values = payload.get("relations") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("relations input must contain a list")
    return tuple(PairRelation(**item) for item in values)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source_bytes = args.source_manifest.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if json.loads(source_bytes).get("status") != "EXTERNAL_SOURCE_PASS":
            raise ValueError("source manifest is not EXTERNAL_SOURCE_PASS")
        visibility_bytes = args.signed_visibility.read_bytes()
        visibility = SignedVisibilityGrid.from_payload(
            json.loads(visibility_bytes), hashlib.sha256(visibility_bytes).hexdigest()
        )
        relations = _relations(json.loads(args.relations.read_bytes()))
        snapshots = (
            read_map_snapshot(args.t0_snapshot, args.t0_entities),
            read_map_snapshot(args.t1_snapshot, args.t1_entities),
        )
        visits = tuple(
            VisitMap(
                visit_id=index,
                snapshot=snapshots[index],
                coordinate_frame_id=args.coordinate_frame_id,
                source_manifest_sha256=source_sha256,
                map_voxel_size_m=args.map_voxel_size_m,
                observed_frame_start=getattr(args, f"t{index}_frame_start"),
                observed_frame_end=getattr(args, f"t{index}_frame_end"),
            )
            for index in (0, 1)
        )
        current = compose_current_map(
            visits[0],
            visits[1],
            relations,
            visibility,
            CompositionConfig(args.composition_voxel_size_m),
        )
        manifest = write_two_visit_current_map(current, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "current_map_sha256": current.content_sha256(),
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
