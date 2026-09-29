"""Resolve original SAM3 revision and record official authenticated access status."""

import argparse
import datetime
import json
import urllib.error
import urllib.request
from pathlib import Path

from src.static_ovmap.composition_study.io import write_once


def probe(output):
    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": "Bearer " + token} if token else {}
    request = urllib.request.Request("https://huggingface.co/api/models/facebook/sam3", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    if "sam3.pt" not in {s["rfilename"] for s in metadata["siblings"]}:
        raise ValueError("official original SAM3 checkpoint absent")
    url = f"https://huggingface.co/facebook/sam3/resolve/{revision}/sam3.pt"
    error_code = None
    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            length = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        status, length = error.code, None
        error_code = error.headers.get("X-Error-Code")
    result = {"checked_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "repository": "facebook/sam3", "checkpoint": "sam3.pt", "revision": revision,
        "url": url, "gated": metadata.get("gated"), "existing_token_present": bool(token),
        "http_status": status, "error_code": error_code, "advertised_bytes": int(length) if length else None,
        "status": "ACCESSIBLE_NOT_DOWNLOADED" if status == 200 else "BLOCKED_ASSET_ACCESS" if status in (401, 403) else "ACCESS_INCONCLUSIVE",
        "model_executed": False, "mirrors_used": False, "affected_methods": ["AW_E04_SPATIAL", "AW_E04_MIX50"],
        "shortlist_control_requires_model": False}
    write_once(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(probe(args.output), flush=True)
