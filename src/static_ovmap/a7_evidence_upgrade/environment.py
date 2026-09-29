"""Record the actual existing worker environment after a verified real smoke."""

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex


def capture(smoke_path, output):
    import torch

    index = InputIndex()
    smoke = index.identity(smoke_path)
    if read_json(smoke_path)["status"] != "SMOKE_COMPLETE":
        raise ValueError("environment snapshot requires completed real adapter smoke")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "open_clip_torch", "timm", "numpy", "Pillow", "huggingface_hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python_executable": sys.executable, "python_version": sys.version, "torch_cuda_build": torch.version.cuda,
              "packages": packages, "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
              "real_smoke_receipt": smoke, "environment_upgraded_by_wave1": False}
    write_once(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.smoke, args.output)
    print({"status": "ENVIRONMENT_CAPTURED", "python": result["python_executable"], "output": str(args.output)}, flush=True)
