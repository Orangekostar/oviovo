"""Execute the locked development captures with resumable, hashed receipts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.static_ovmap.module_validation.scannet_runtime import capture_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--resolved-config", dest="config", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config = config.get("resolved_config", config)
    config = config.get("scannet_runtime", config)
    result = capture_development(config, scenes=args.scenes)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
