"""Index committed/staged compact evidence, excluding local diagnostic files."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256


def main():
    config = json.loads((ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text())
    compact = path(config["compact_output_root"])
    relative = compact.relative_to(ROOT)
    target = compact / "compact_artifact_index.json"
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(relative)], cwd=ROOT
    ).decode().split("\0")
    artifacts = []
    for name in sorted(filter(None, tracked)):
        source = ROOT / name
        if source == target:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        artifacts.append({"path": str(source.relative_to(compact)),
                          "byte_count": source.stat().st_size, "sha256": _sha256(source)})
    _atomic_json(target, {
        "schema_version": 1,
        "scope": "Git-tracked compact evidence; excludes self and ignored local diagnostic overlays",
        "artifact_count": len(artifacts), "artifacts": artifacts,
        "map_storage": "LOCAL_ONLY_POLICY; selected_apartment_map_exports.json binds local PLY paths and hashes",
        "model_storage": "REUSED_EXTERNAL; model_manifest.json binds official checkpoint source and hash",
        "completion_claim": "Index completeness is not task completion or proof of remote upload",
    })
    print(len(artifacts), "tracked compact artifacts indexed", flush=True)


if __name__ == "__main__":
    main()
