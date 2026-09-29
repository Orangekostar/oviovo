"""Bind the original OpenCLIP weights for the matched FC_FROZEN region control."""

import argparse
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

REPO = "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"
REVISION = "654d0f80ff73c58e7281a3ca7dc425589049e2e1"


def fetch(output):
    from huggingface_hub import snapshot_download

    output, index = Path(output), InputIndex()
    receipt_path = output / "download_receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if previous["repo"] != REPO or previous["revision"] != REVISION:
            raise ValueError("matched control revision changed")
        for item in previous["files"]:
            index.identity(item["path"], item)
        return previous
    started = time.monotonic()
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=output / "model", max_workers=2,
        allow_patterns=["open_clip_model.safetensors", "open_clip_config.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"])
    elapsed = time.monotonic() - started
    files = [index.identity(p) for p in sorted((output / "model").iterdir()) if p.is_file()]
    checkpoint = next(v for v in files if v["path"].endswith("open_clip_model.safetensors"))
    if checkpoint["bytes"] != 1407151868:
        raise ValueError("matched control checkpoint length differs from official metadata")
    receipt = {"status": "DOWNLOADED_NOT_SMOKE_TESTED", "repo": REPO, "revision": REVISION,
               "resolution_source": "open_clip3.3.0 get_pretrained_cfg plus official HF model_info before inference",
               "model": "convnext_large_d_320", "pretraining": "laion2b_s29b_b131k_ft_soup",
               "files": files, "download_seconds": elapsed, "total_bytes": sum(v["bytes"] for v in files)}
    write_once(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.output)
    print({k: result[k] for k in ("status", "repo", "total_bytes", "download_seconds")}, flush=True)
