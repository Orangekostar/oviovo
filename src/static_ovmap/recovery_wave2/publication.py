"""Normal named-branch publication after a primary, evidence-bound final review."""

from pathlib import Path
import subprocess
import time

from static_ovmap.module_validation.contracts import atomic_write_json

from .binding import ConsumptionIndex, read
from .light import require_transfer_freeze


BRANCH = "research/ovimap-recovery-wave2-v1"


def verify_ready(binding):
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    compact = repo / "artifacts/static_ovmap/recovery_wave2_v1"
    completion = read(compact / "completion.json")
    if completion["implementation_status"] != "IMPLEMENTATION_COMPLETE":
        raise ValueError("publication requires complete implementation and all required measured leaves")
    review = read(root / "validation/primary_review.json")
    if review["status"] != "PASS" or set(review["contracts"]) != {"C" + str(number) for number in range(12)}:
        raise ValueError("publication requires the primary complete-spec C0-C11 review")
    if not review["reviewed_files"] or review["targeted_validation"]["status"] != "PASS":
        raise ValueError("primary review lacks file-level evidence or passing targeted validation")
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    for row in review["reviewed_files"]:
        path = Path(row["path"])
        if not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError("reviewed publication file is outside this repository")
        index.identity(path, row)
    for scene in binding["datasets"]["replica"]:
        require_transfer_freeze(binding, scene)
    return review


def publish(binding):
    review = verify_ready(binding)
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != BRANCH:
        raise ValueError("normal publication must use the specified research branch")
    paths = sorted({str(Path(row["path"]).resolve().relative_to(repo.resolve())) for row in review["reviewed_files"]})
    review_copy = "artifacts/static_ovmap/recovery_wave2_v1/validation/primary_review.json"
    if (repo / review_copy).is_file():
        if read(repo / review_copy) != review:
            raise ValueError("compact final review differs from the primary external receipt")
        paths.append(review_copy)
    subprocess.run(["git", "diff", "--check", "--", *paths], cwd=repo, check=True)
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    dirty = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo).returncode
    if dirty not in {0, 1}:
        raise RuntimeError("cannot inspect the reviewed staged publication")
    if dirty:
        subprocess.run(["git", "commit", "--only", "-m", "research: publish measured recovery-wave2 evidence and primary audit",
                        "--", *paths], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    destination = root / "publication/final.json"
    started = time.monotonic()
    commands = [["git", "push", "origin", "HEAD:refs/heads/" + BRANCH],
                ["git", "ls-remote", "origin", "refs/heads/" + BRANCH]]
    receipt = {"status": "PUSH_PENDING", "branch": BRANCH, "local_sha": sha,
        "remote_sha": None, "binding_identity": binding["identity"], "review_identity": review["identity"],
        "commands": commands, "normal_push_without_force": True, "post_push_receipt_is_external": True}
    atomic_write_json(destination, receipt)
    try:
        pushed = subprocess.run(commands[0], cwd=repo, text=True, capture_output=True, check=True)
        remote = subprocess.run(commands[1], cwd=repo, text=True, capture_output=True, check=True)
        lines = [line.split() for line in remote.stdout.splitlines() if line.strip()]
        if len(lines) != 1 or lines[0][1] != "refs/heads/" + BRANCH or lines[0][0] != sha:
            raise ValueError("full remote branch SHA differs from the reviewed local HEAD")
        receipt.update(status="PUSH_VERIFIED", remote_sha=lines[0][0], push_output=pushed.stderr.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        receipt.update(status="FAILED_PUSH", error=str(exc),
                       command_stderr=getattr(exc, "stderr", None))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(destination, receipt)
    return receipt
