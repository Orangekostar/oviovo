#!/usr/bin/env python3
"""Probe whether the configured Python environment can run SAM3."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sam3_repo_root is not None:
        sys.path.insert(0, str(args.sam3_repo_root.resolve()))
    result: dict[str, object] = {
        "python": sys.version,
        "ok": False,
        "imports": {},
        "errors": [],
    }
    for module_name in ["torch", "sam3"]:
        try:
            module = importlib.import_module(module_name)
            result["imports"][module_name] = str(getattr(module, "__file__", "built-in"))
            if module_name == "torch":
                result["torch_version"] = str(getattr(module, "__version__", "unknown"))
                result["cuda_available"] = bool(module.cuda.is_available())
                result["cuda_version"] = str(getattr(module.version, "cuda", ""))
        except Exception as exc:
            result["imports"][module_name] = None
            result["errors"].append(f"{module_name}: {type(exc).__name__}: {exc}")
    result["ok"] = not result["errors"]
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
