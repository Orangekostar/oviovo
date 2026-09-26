"""Content-bound inputs and immutable new-study outputs."""

import json
from pathlib import Path

from src.static_ovmap.module_validation import assets
from src.static_ovmap.module_validation.contracts import (
    atomic_write_json,
    canonical_digest,
)

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if canonical_digest(read_json(path)) != canonical_digest(value):
            raise ValueError(f"immutable output differs: {path}; use a new attempt")
    else:
        atomic_write_json(path, value)
    return path


class SourceIndex:
    """Verify actual bytes once per stable identity, not every dependency DAG."""

    def __init__(self):
        self.entries = {}

    def identity(self, path, expected=None):
        path = Path(path).resolve()
        result = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": assets.sha256_file(path),
        }
        if expected is not None and any(
            result[key] != expected[key]
            for key in ("bytes", "sha256")
            if key in expected
        ):
            raise ValueError(f"source identity differs: {path}")
        self.entries[str(path)] = result
        return result

    def expected_output(self, path, receipt):
        path = Path(path).resolve()
        expected = next(
            (row for row in receipt["outputs"] if Path(row["path"]).resolve() == path),
            None,
        )
        if expected is None:
            raise ValueError(f"output missing from source receipt: {path}")
        return self.identity(path, expected)

    def manifest(self):
        return {"entries": [self.entries[path] for path in sorted(self.entries)]}


def code_identity():
    index = SourceIndex()
    files = sorted((ROOT / "src/static_ovmap/composition_study").glob("*.py"))
    files += sorted((ROOT / "src/static_ovmap/module_validation").glob("*.py"))
    files += [ROOT / "scripts/evaluation/run_ovimap_composition_study.py"]
    rows = [index.identity(path) for path in files if path.is_file()]
    return {
        "files": rows,
        "digest": canonical_digest(
            [
                {
                    "path": str(Path(row["path"]).relative_to(ROOT)),
                    "sha256": row["sha256"],
                }
                for row in rows
            ]
        ),
    }
