"""Execute independent study stages despite a failed sibling prerequisite."""

import subprocess
import sys
from pathlib import Path

from src.static_ovmap.module_validation.contracts import atomic_write_json

from .binding import ROOT

DEPENDENCIES = {
    "bind": (), "core": ("bind",), "query-controls": ("bind", "core"),
    "diagnostics": ("core",), "robustness": ("core",), "fresh": ("core",),
    "report": ("core", "query-controls", "diagnostics", "robustness", "fresh"),
    "publish": ("report",),
}


def run_all(*, spec, source_transfer, output_root, gpu):
    states = {}
    for phase, dependencies in DEPENDENCIES.items():
        failed = [p for p in dependencies if states[p]["status"] != "COMPLETE"]
        if failed:
            states[phase] = {"status": "PREREQUISITE_FAILED", "prerequisites": failed}
        else:
            command = [sys.executable, str(ROOT / "scripts/evaluation/run_ovimap_m2_reviewer_study.py"),
                       "--phase", phase, "--spec", str(spec), "--source-transfer", str(source_transfer),
                       "--output-root", str(output_root), "--gpu", str(gpu), "--resume"]
            result = subprocess.run(command, cwd=ROOT, check=False)
            states[phase] = {"status": "COMPLETE" if result.returncode == 0 else "FAILED",
                             "returncode": result.returncode, "command": command}
        atomic_write_json(Path(output_root) / "workflow_status.json", states)
    if states["publish"]["status"] != "COMPLETE":
        raise RuntimeError(f"study workflow incomplete; see {output_root}/workflow_status.json")
    return states
