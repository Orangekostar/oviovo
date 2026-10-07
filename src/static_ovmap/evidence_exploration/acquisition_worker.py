"""Process entrypoint, kept separate from content-keyed scientific operators."""

import argparse

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.runtime_parity.runner import execution_config
from .binding import load_binding
from .acquisition import encode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pilots", action="store_true")
    args = parser.parse_args()
    binding = load_binding(args.output_root)
    with exclusive_lock(execution_config(binding["reference"])["gpu_lock"]):
        encode(binding, args.output_root, binding["specification"]["pilot_scenes"] if args.pilots else None)
