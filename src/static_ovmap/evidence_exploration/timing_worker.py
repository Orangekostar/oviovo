"""One GPU process for resident-model paired cold timing."""

import argparse

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.runtime_parity.runner import execution_config
from .binding import load_binding
from .timing import time_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    binding = load_binding(args.output_root)
    with exclusive_lock(execution_config(binding["reference"])["gpu_lock"]):
        time_study(binding, args.output_root)
