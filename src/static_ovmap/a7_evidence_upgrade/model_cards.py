"""Save small official pinned model cards, never datasets or alternate weights."""

import argparse
import urllib.request
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

CARDS = {
    "so400m": "https://huggingface.co/google/siglip2-so400m-patch14-384/resolve/f3b7a187cd133857ff43c0dccfe88f8268372549/README.md",
    "fc_frozen": "https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/654d0f80ff73c58e7281a3ca7dc425589049e2e1/README.md",
}


def fetch(output):
    output, index = Path(output), InputIndex()
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        previous = read_json(receipt_path)
        for row in previous["cards"]:
            index.identity(row["file"]["path"], row["file"])
        return previous
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in CARDS.items():
        path = output / (name + ".md")
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read(1024 * 1024)
            if not body.startswith(b"---") or len(body) >= 1024 * 1024:
                raise ValueError("official model card response is not bounded markdown")
            path.write_bytes(body)
        rows.append({"model": name, "url": url, "file": index.identity(path)})
    result = {"status": "OFFICIAL_MODEL_CARDS_CAPTURED", "cards": rows}
    write_once(receipt_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(fetch(args.output), flush=True)
