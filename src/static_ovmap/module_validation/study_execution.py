"""Artifact and split contracts shared by the remaining scientific drivers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json
from .scannet_runtime import reusable_job
from .scannet_study import load_prediction, save_prediction

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def config_inputs(config, config_path):
    runtime = Path(config["runtime_config"])
    return [file_identity(path) for path in (config_path, runtime if runtime.is_absolute() else ROOT / runtime)]


def roles(runtime):
    lock_path = Path(runtime["data_root"]) / "acquisition_lock.json"
    lock = read_json(lock_path)
    split = {role: tuple(row["scene_id"] for row in lock["selected"] if row["role"] == role)
             for role in ("fit", "cal", "select", "confirm")}
    if tuple(map(len, split.values())) != (8, 2, 2, 2):
        raise ValueError("study requires the frozen 8/2/2/2 physical-family split")
    return split, lock_path


def reuse(path, identity):
    if reusable_job(Path(path), identity):
        return read_json(path)
    if Path(path).exists():
        raise ValueError(f"completed study inputs/outputs changed; choose a new output root: {path}")
    return None


def verify_receipt(path, seen=None):
    path = Path(path)
    value = read_json(path)
    seen = set() if seen is None else seen
    if path.resolve() in seen:
        return value
    if not reusable_job(path, value["input_identity"]):
        raise ValueError(f"study artifact changed: {path}")
    seen.add(path.resolve())
    for row in value.get("inputs", []) + value.get("sources", []):
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"study input changed: {row['path']}")
        source = Path(row["path"])
        if source.suffix == ".json" and source.stem.endswith("receipt"):
            verify_receipt(source, seen)
    for row in value["outputs"]:
        source = Path(row["path"])
        if source.suffix == ".json" and source.stem.endswith("receipt"):
            verify_receipt(source, seen)
    return value


def receipt(path, identity, inputs, outputs, sources, **values):
    result = {"status": "COMPLETE", "input_identity": identity, "inputs": inputs, "sources": sources,
        "outputs": [file_identity(p) for p in sorted(set(map(Path, outputs)))],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.argv, **values}
    atomic_write_json(path, result)
    return result


def evaluation_outputs(path):
    return sorted(file for file in Path(path).rglob("*") if file.is_file())


def store_prediction(payload, output):
    path = Path(output) / "manifest.json"
    if path.exists():
        previous = load_prediction(path)

        def stable_costs(row):
            return {key: value for key, value in row.logical_cost.items() if not key.endswith("seconds")}

        if (previous.prediction_key != payload.prediction_key or previous.method_id != payload.method_id
                or previous.metadata != payload.metadata or stable_costs(previous) != stable_costs(payload)):
            raise ValueError("frozen study prediction changed on resume")
        return path
    return save_prediction(payload, output)
