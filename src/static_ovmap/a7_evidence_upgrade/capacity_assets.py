"""Download only the protocol-pinned official SO400M recognition checkpoint."""

import argparse
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

REPO = "google/siglip2-so400m-patch14-384"
REVISION = "f3b7a187cd133857ff43c0dccfe88f8268372549"


def fetch(output):
    from huggingface_hub import snapshot_download

    output = Path(output)
    receipt_path = output / "download_receipt.json"
    index = InputIndex()
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["repo"] != REPO or receipt["revision"] != REVISION:
            raise ValueError("capacity asset identity changed")
        for item in receipt["files"]:
            index.identity(item["path"], item)
        return receipt
    started = time.monotonic()
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=output / "model",
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json",
                        "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"],
        max_workers=2)
    elapsed = time.monotonic() - started
    files = [index.identity(p) for p in sorted((output / "model").iterdir()) if p.is_file()]
    checkpoint = next(v for v in files if v["path"].endswith("model.safetensors"))
    if checkpoint["bytes"] != 4544143072:
        raise ValueError("pinned SO400M checkpoint length differs from official probe")
    receipt = {"status": "DOWNLOADED_NOT_SMOKE_TESTED", "repo": REPO, "revision": REVISION,
               "files": files, "download_seconds": elapsed, "total_bytes": sum(v["bytes"] for v in files)}
    write_once(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.output)
    print({k: result[k] for k in ("status", "repo", "total_bytes", "download_seconds")}, flush=True)
