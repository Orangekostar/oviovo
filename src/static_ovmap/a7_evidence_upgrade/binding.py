"""Freeze wave1 definitions and actual parent evidence without recapturing maps."""

import subprocess
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

ROOT = Path(__file__).resolve().parents[3]


def bind(spec_path, reviewer_path, output_root):
    index = InputIndex()
    spec, reviewer = read_json(spec_path), read_json(reviewer_path)
    index.identity(spec_path)
    index.identity(reviewer_path)
    if canonical_digest({k: v for k, v in reviewer.items() if k != "identity"}) != reviewer["identity"]:
        raise ValueError("reviewer binding identity changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", spec["base_commit"], "HEAD"], cwd=ROOT, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != spec["branch"]:
        raise ValueError("wave1 requires the prescribed isolated task branch")
    scenes = spec["datasets"]["calibration"] + spec["datasets"]["replica"]
    if set(scenes) != set(reviewer["scenes"]):
        raise ValueError("wave1 scenes differ from bound reviewer scenes")
    records = {}
    expected = {row["path"]: row for row in reviewer["inputs"]}
    for scene in scenes:
        entry = reviewer["scenes"][scene]
        index.identity(entry["config"], expected[entry["config"]])
        config = read_json(entry["config"])
        data = config["scenes"][scene]
        manifest_path = Path(data["source_directory"]) / "semantic_requests.json"
        manifest_identity = index.identity(manifest_path)
        for source_path in entry["sources"].values():
            index.identity(source_path, expected[source_path])
        records[scene] = {**entry, "static_manifest": manifest_identity,
                          "role": "CAL" if scene in spec["datasets"]["calibration"] else "EXPOSED_REPLICA"}
    parent_root = Path(reviewer["output_root"])
    for path in (parent_root / "calibration/new_folds.json", parent_root / "calibration/new_final.json",
                 Path(reviewer["composition_root"]) / "calibration/folds.json",
                 Path(reviewer["composition_root"]) / "calibration/final_temperatures.json"):
        index.identity(path)
    path = Path(output_root) / "binding.json"
    definition = {"schema": 1, "status": "INPUTS_AND_DEFINITIONS_BOUND",
                  "spec": str(Path(spec_path).resolve()), "reviewer_binding": str(Path(reviewer_path).resolve()),
                  "reviewer_identity": reviewer["identity"], "output_root": str(Path(output_root).resolve()),
                  "base": spec["base_commit"], "branch": branch, "scenes": records,
                  "methods": spec["references"] + spec["mandatory_new_methods"],
                  "optional_method": spec["composition"]["method"], "inputs": index.entries(),
                  "assets": "NEW_MODEL_PREREQUISITES_NOT_YET_RESOLVED", "deployment": "N0_UNCHANGED"}
    definition["identity"] = canonical_digest(definition)
    write_once(path, definition)
    if not (Path(output_root) / "origin.json").exists():
        write_once(Path(output_root) / "origin.json", {"source_HEAD": head, "base": spec["base_commit"],
                                                     "base_is_ancestor": True, "branch": branch})
    return definition
