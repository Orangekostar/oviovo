"""Retrieve the author-linked OVR checkpoint with byte/hash provenance."""

import argparse
import subprocess
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

URL = "https://drive.usercontent.google.com/download?id=1raLz9TybKGxMCsWVjjdoBnbraUXDgmQ0&export=download&confirm=t"
COMMIT = "9fd9450d22852d269d426b521663a127f3983a4b"


def fetch(output):
    output = Path(output)
    index = InputIndex()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output / "code", text=True).strip()
    if actual != COMMIT:
        raise ValueError("OVR code pin differs")
    receipt_path = output / "download_receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        index.identity(previous["checkpoint"]["path"], previous["checkpoint"])
        return previous
    partial, final = output / "ovrcoat.pth.partial", output / "ovrcoat.pth"
    started = time.monotonic()
    if final.exists():
        raise ValueError("unreceipted final OVR checkpoint; inspect before reuse")
    result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
        "--continue-at", "-", "--output", str(partial), URL], check=False)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"official OVR download failed: curl={result.returncode}, seconds={elapsed}")
    if partial.stat().st_size != 4633915328:
        raise ValueError("OVR file is not the author checkpoint advertised by HEAD")
    partial.rename(final)
    receipt = {"status": "DOWNLOADED_NOT_SMOKE_TESTED", "url": URL,
               "author_link": "https://drive.google.com/file/d/1raLz9TybKGxMCsWVjjdoBnbraUXDgmQ0/view",
               "code_commit": COMMIT, "checkpoint": index.identity(final), "download_seconds": elapsed}
    write_once(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(fetch(args.output), flush=True)
