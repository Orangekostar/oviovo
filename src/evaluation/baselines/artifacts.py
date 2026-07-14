"""Atomic persistence for baseline artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.evaluation.baselines.contracts import BaselineArtifact
from src.evaluation.exporters.oviovo import write_map_snapshot


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_baseline_artifact(artifact: BaselineArtifact, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = write_map_snapshot(artifact.snapshot, output_dir)
    runtime_path = output_dir / "runtime.json"
    metadata_path = output_dir / "metadata.json"
    _atomic_json(runtime_path, artifact.runtime.to_json())
    _atomic_json(metadata_path, artifact.metadata.to_json())
    return {
        "snapshot": paths["snapshot"],
        "entities": paths["entities"],
        "runtime": runtime_path,
        "metadata": metadata_path,
    }
