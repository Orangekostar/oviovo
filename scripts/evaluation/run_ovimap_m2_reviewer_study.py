#!/usr/bin/env python3
"""Execute the source-bound M2 reviewer evidence study."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "docs/paper/static_ovmap/m2_reviewer_study_v1/PROTOCOL_SPEC.json")
    parser.add_argument("--source-transfer", type=Path, default=Path("/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001"))
    parser.add_argument("--phase", required=True, choices=("bind", "core"))
    parser.add_argument("--scene")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    from src.static_ovmap.composition_study.io import read_json
    from src.static_ovmap.m2_reviewer_study.binding import bind

    if args.phase == "bind":
        result = bind(args.spec, args.source_transfer, args.output_root)
        print(json.dumps({"status": result["status"], "scenes": len(result["scenes"])}), flush=True)
    else:
        from src.static_ovmap.m2_reviewer_study.core import evaluate_core_scene

        binding = read_json(args.output_root / "source_binding.json")
        scenes = [args.scene] if args.scene else list(binding["scenes"])
        for scene in scenes:
            if scene not in binding["scenes"]:
                raise ValueError("scene outside study binding")
            rows = evaluate_core_scene(binding, scene)
            print(json.dumps({"status": "SCENE_CORE_COMPLETE", "scene": scene, "rank_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
