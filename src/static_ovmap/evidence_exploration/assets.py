"""Validate pinned AnyUp without installing into or upgrading the FC environment."""

import json
from pathlib import Path
import subprocess
import time

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex


def validate_assets(binding, output_root):
    root = Path(output_root)
    assets = root.parent / "assets"
    checkout, weight = assets / "anyup", assets / "anyup_paper.pth"
    spec = binding["specification"]["anyup"]
    index = ConsumptionIndex()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() != spec["commit"]:
        raise ValueError("AnyUp checkout differs from the pinned source")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=checkout):
        raise ValueError("AnyUp upstream checkout was modified; publish a separate wrapper")
    weight_identity = index.identity(weight, {"sha256": spec["checkpoint_sha256"], "bytes": spec["checkpoint_bytes"]})
    sources = {p: index.identity(checkout / p) for p in ("hubconf.py", "anyup/model.py",
        "anyup/layers/attention/chunked_attention.py", "anyup/layers/attention/attention_masking.py", "LICENSE")}
    target = root / "assets.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["checkpoint"] != weight_identity or previous["sources"] != sources:
            raise ValueError("validated asset identity changed")
        return previous
    code = """
import json,sys,torch
sys.path.insert(0,sys.argv[1])
from anyup.model import AnyUp
model=AnyUp(use_natten=False)
state=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True)
model.eval().requires_grad_(False)
print(json.dumps({'strict':True,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'python':sys.executable,'actual_inference':0}))
"""
    start = time.perf_counter()
    run = subprocess.run([binding["FC_python"], "-c", code, str(checkout), str(weight)], capture_output=True, text=True)
    (root / "asset_validation.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("pinned AnyUp strict load failed; see asset_validation.log")
    receipt = seal({"status": "ASSETS_AVAILABLE_STRICT_LOAD_VERIFIED", "checkpoint": weight_identity,
        "sources": sources, "repository": spec["repository"], "commit": spec["commit"],
        "load": json.loads(run.stdout.splitlines()[-1]), "elapsed_seconds": time.perf_counter()-start,
        "checkout": str(checkout), "use_natten": False, "license": "CC-BY-4.0",
        "weight_publication": "EXCLUDED", "environment_modified": False})
    atomic_write_json(target, receipt)
    print(receipt["status"], flush=True)
    return receipt
