"""Run the pinned mapper with a scene-specific shared-memory namespace."""

import re
import runpy
import sys
from pathlib import Path


def isolate(module, prefix):
    if not re.fullmatch(r"replica_[a-zA-Z0-9_]+", prefix):
        raise ValueError("invalid Replica IPC namespace")
    module.SHM_RGB_PREFIX = prefix


if __name__ == "__main__":
    prefix, script, *arguments = sys.argv[1:]
    sys.path.insert(0, str(Path(script).parent))
    import ipc_utils

    isolate(ipc_utils, prefix)
    sys.argv = [script, *arguments]
    runpy.run_path(script, run_name="__main__")
