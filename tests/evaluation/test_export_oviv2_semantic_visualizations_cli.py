from __future__ import annotations

import subprocess
import sys


def test_semantic_visualization_cli_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/export_oviv2_semantic_visualizations.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--batch-root" in result.stdout
    assert "--scene" in result.stdout
