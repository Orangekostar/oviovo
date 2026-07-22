from __future__ import annotations

import subprocess
import sys


def test_baseline_metric_module_does_not_import_oviv2_dependencies() -> None:
    script = """
import importlib.abc
import sys

class BlockOviv2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('src.oviv2'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOviv2())
from src.evaluation.baselines.dynamic_metrics import aggregate_dynamic_metrics
assert callable(aggregate_dynamic_metrics)
assert not any(name.startswith('src.oviv2') for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
