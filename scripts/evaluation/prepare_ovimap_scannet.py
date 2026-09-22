"""Prepare only the independent ScanNet captures required by the frozen study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.static_ovmap.module_validation.assets import exposed_scannet_families
from src.static_ovmap.module_validation.contracts import atomic_write_json
from src.static_ovmap.module_validation.scannet_download import (
    DownloaderLayout,
    download_selected,
    fetch_public_splits,
    prepare_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloader", type=Path, default=Path("/home/ww/getscannet.py"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "download", "export"), default="plan")
    parser.add_argument("--authorized-scannet-download", action="store_true",
                        help="Use only after the user confirms authorized access and prior terms agreement")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.phase == "plan":
        layout = DownloaderLayout.from_script(args.downloader)
        train, val = fetch_public_splits(root)
        plan = prepare_plan(layout, train, val, exposed_scannet_families(REPOSITORY_ROOT))
        atomic_write_json(root / "acquisition_plan.json", plan)
        print(json.dumps({"status": plan["status"], "plan": str(root / "acquisition_plan.json"),
                          "candidates": [{"scene_id": r["scene_id"], "role": r["role"]} for r in plan["candidates"]]}, indent=2))
    elif args.phase == "download":
        plan = json.loads((root / "acquisition_plan.json").read_text())
        print(json.dumps(download_selected(plan, root, authorized=args.authorized_scannet_download), indent=2))
    else:
        from src.static_ovmap.module_validation.scannet_frames import export_development
        print(json.dumps(export_development(root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
