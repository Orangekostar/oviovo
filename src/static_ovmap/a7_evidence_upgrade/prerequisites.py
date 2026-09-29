"""Prepare only authorized pinned official assets and deterministic CAL smokes."""

import subprocess
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_SHA = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
ROOT = Path(__file__).resolve().parents[3]


def pinned_code(output, url, commit):
    output = Path(output)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(output)], check=True)
        subprocess.run(["git", "checkout", "--detach", commit], cwd=output, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output, text=True).strip()
    if actual != commit or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=output, text=True).strip():
        raise ValueError("upstream checkout is not the clean prescribed pin; existing edits are preserved")


def sam2_asset(output, commit):
    output, index = Path(output), InputIndex()
    receipt = output / "download_receipt.json"
    if receipt.exists():
        value = read_json(receipt)
        item = index.identity(value["checkpoint"]["path"], value["checkpoint"])
        if item["sha256"] != SAM2_SHA or value["code_commit"] != commit:
            raise ValueError("SAM2 checkpoint/code differs from the fixed asset")
        return value
    checkpoint = output / "sam2.1_hiera_large.pt"
    started = time.monotonic()
    if not checkpoint.exists():
        partial = checkpoint.with_suffix(".pt.partial")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                        "--continue-at", "-", "--output", str(partial), SAM2_URL], check=True)
        identity = index.identity(partial)
        if identity["bytes"] != 898083611 or identity["sha256"] != SAM2_SHA:
            raise ValueError("official SAM2 bytes differ; partial retained for inspection")
        partial.rename(checkpoint)
    identity = index.identity(checkpoint)
    if identity["sha256"] != SAM2_SHA:
        raise ValueError("local unreceipted SAM2 checkpoint differs")
    result = {"status": "DOWNLOADED_NOT_SMOKE_TESTED", "checkpoint": identity, "code_commit": commit,
              "official_url": SAM2_URL, "download_time_seconds": time.monotonic() - started,
              "timing_status": "PREPARATION_ELAPSED_INCLUDING_LOCAL_HASH_VERIFICATION",
              "downloaded_bytes": identity["bytes"], "new_model_budget_bytes": 25 * 1024**3}
    write_once(receipt, result)
    return result


def prepare(binding, environments, gpus):
    from .ovr_assets import COMMIT
    from .workflow import visual_jobs

    root, index = Path(binding["output_root"]), InputIndex()
    assets = root.parent / "assets"
    spec = read_json(binding["spec"])
    # The prescribed four checkpoints and tokenizer/config files total < 11 GiB.
    # This routine never installs environments or fetches dataset assets.
    pinned_code(assets / "sam2/code", "https://github.com/facebookresearch/sam2.git", spec["sam2"]["commit"])
    pinned_code(assets / "ovrcoat/code", "https://github.com/nickormushev/OVRCOAT.git", COMMIT)
    sam2_asset(assets / "sam2", spec["sam2"]["commit"])
    for environment, module, folder in (("semantic", "capacity_assets", "so400m"),
                                         ("region", "fc_assets", "fc_frozen"), ("region", "ovr_assets", "ovrcoat")):
        subprocess.run([environments[environment], "-m", "src.static_ovmap.a7_evidence_upgrade." + module,
                        "--output", str(assets / folder)], cwd=ROOT, check=True)
    total = 0
    for name in ("sam2", "so400m", "fc_frozen", "ovrcoat"):
        receipt = assets / name / "download_receipt.json"
        index.identity(receipt)
        value = read_json(receipt)
        total += sum(v["bytes"] for v in value.get("files", [value.get("checkpoint")]))
    if total > 25 * 1024**3:
        raise ValueError("authorized asset budget exceeded")
    access = root / "e04/access_probe.json"
    if not access.exists():
        subprocess.run([environments["semantic"], "-m", "src.static_ovmap.a7_evidence_upgrade.sam3_access",
                        "--output", str(access)], cwd=ROOT, check=True)
    index.identity(access)
    scene = spec["datasets"]["calibration"][0]
    jobs = [
        (environments["sam2"], "sam2_worker", ["--scene", scene, "--assets", str(assets / "sam2"), "--smoke"]),
        (environments["semantic"], "capacity_worker", ["--scene", scene, "--assets", str(assets / "so400m"), "--smoke"]),
        *[(environments["region"], "region_smoke", ["--scene", scene, "--assets", str(assets), "--branch", branch])
          for branch in ("FC_FROZEN", "OVR")],
    ]
    logs = visual_jobs(binding, jobs, gpus)
    for environment, smoke in (("semantic", root / "c0/smoke" / scene / "receipt.json"),
                               ("sam2", root / "e02/sam2_smoke" / scene / "receipt.json"),
                               ("region", root / "e03/smoke" / scene / "OVR/receipt.json")):
        output = root / "environments" / (environment + ".json")
        subprocess.run([environments[environment], "-m", "src.static_ovmap.a7_evidence_upgrade.environment",
                        "--smoke", str(smoke), "--output", str(output)], cwd=ROOT, check=True)
        index.identity(output)
    parameters = root / "environments/region_parameters.json"
    if not parameters.exists():
        subprocess.run([environments["region"], "-m", "src.static_ovmap.a7_evidence_upgrade.model_inventory",
                        "--root", str(root)], cwd=ROOT, check=True)
    for item in read_json(parameters)["inputs"]:
        index.identity(item["path"], item)
    index.identity(parameters)
    subprocess.run([environments["semantic"], "-m", "src.static_ovmap.a7_evidence_upgrade.model_cards",
                    "--output", str(assets / "model_cards")], cwd=ROOT, check=True)
    result = {"status": "PINNED_ASSETS_AND_REAL_SMOKES_READY", "new_model_bytes": total,
              "SAM3_access_status": read_json(access)["status"], "worker_logs": logs, "inputs": index.entries()}
    # Each invocation has distinct cache-validation logs, so receipts are append-only.
    path = root / "prerequisites" / f"invocation_{time.time_ns()}.json"
    write_once(path, result)
    return result
