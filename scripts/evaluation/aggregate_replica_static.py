#!/usr/bin/env python3
"""Aggregate frozen Replica baseline scene metrics without manual transcription."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.baselines.aggregation import aggregate_replica_static


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-result",
        action="append",
        required=True,
        metavar="SCENE=PATH",
        help="Per-scene metrics JSON; provide each frozen Replica-8 scene exactly once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene_results = {}
    source_paths = {}
    for item in args.scene_result:
        try:
            scene_id, raw_path = item.split("=", 1)
        except ValueError as error:
            raise SystemExit(f"invalid --scene-result value: {item}") from error
        if scene_id in scene_results:
            raise SystemExit(f"duplicate scene result: {scene_id}")
        path = Path(raw_path)
        scene_results[scene_id] = json.loads(path.read_text(encoding="utf-8"))
        source_paths[scene_id] = str(path.resolve())

    payload = {
        "metrics": aggregate_replica_static(scene_results),
        "scene_result_paths": source_paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
