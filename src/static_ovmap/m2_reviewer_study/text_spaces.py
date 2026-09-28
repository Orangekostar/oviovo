"""Model-bound CPU text encodings; no image encoder calls or new fitting."""

import argparse
import os
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .binding import InputIndex


def prepare_text(binding, dataset, model_name):
    import numpy as np
    import torch
    import transformers

    from src.static_ovmap.module_validation.native_capture import _write_npz
    from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend

    scene = next(s for s, row in binding["scenes"].items() if row["dataset"] == dataset)
    config = read_json(binding["scenes"][scene]["config"])
    model = config["models"][model_name]
    if (torch.__version__, transformers.__version__) != (model["torch_version"], model["transformers_version"]):
        raise ValueError("robust text worker must use the bound model environment")
    index = InputIndex()
    for row in model["files"]:
        index.identity(row["path"], row)
    index.identity(model["text"]["path"], model["text"])
    index.identity(binding["spec"])
    index.identity(__file__)
    with np.load(model["text"]["path"], allow_pickle=False) as old:
        original = np.array(old["text_embeddings"], copy=True)
        canonical = np.array(old["canonical_embeddings"], copy=True)
        canonical_phrases = old["canonical_phrases"].tolist() if "canonical_phrases" in old else ["object", "things", "stuff", "texture"]
    names = model["class_names"]
    spec = read_json(binding["spec"])["robustness"]
    distractors = spec["distractors"]
    texts = {"original": names, "photo": [f"a photo of {name}" for name in names],
             "closeup": [f"a close-up photo of {name}" for name in names], "expanded": [*names, *distractors]}
    identity = canonical_digest({"binding": binding["identity"], "dataset": dataset, "model": model["identity"],
                                 "inputs": index.entries(), "texts": texts, "dtype": "float32", "device": "cpu", "batch_size": 16})
    output = Path(binding["output_root"]) / "robustness/text" / dataset / model_name
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["input_identity"] != identity:
            raise ValueError("robust text resume inputs changed")
        for row in receipt["outputs"]:
            index.identity(row["path"], row)
        return receipt
    started = time.monotonic()
    backend = FrozenSiglipBackend.from_local(model["path"], device="cpu")
    loaded = time.monotonic()
    if any(p.dtype != torch.float32 for p in backend.model.parameters() if p.is_floating_point()):
        raise ValueError("robust text model must use FP32")

    def encode(values):
        return np.concatenate([backend.encode_texts(values[i:i + 16]) for i in range(0, len(values), 16)])

    arrays = {"original": original, "canonical": canonical,
              "photo": encode(texts["photo"]), "closeup": encode(texts["closeup"]),
              "expanded": np.concatenate([original, encode(distractors)])}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite robust text vectors")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "text.npz"
    _write_npz(path, arrays)
    receipt = {"status": "COMPLETE", "input_identity": identity, "model_identity": model["identity"],
               "model": model_name, "dataset": dataset, "texts": texts, "valid_ids": model["valid_ids"],
               "canonical_phrases": canonical_phrases, "canonical_identity": model["text"],
               "distractor_column_indices": list(range(len(names), len(names) + len(distractors))),
               "dtype": "float32", "device": "cpu", "model_loads": 1,
               "text_inputs_encoded": 2 * len(names) + len(distractors), "reused_original_columns": len(names),
               "model_load_seconds": loaded - started, "elapsed_seconds": time.monotonic() - started,
               "physical_image_forwards": 0, "peak_gpu_memory": "NOT_APPLICABLE_CPU_TEXT_ONLY",
               "inputs": index.entries(), "outputs": [index.identity(path)]}
    write_once(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ScanNet", "Replica"), required=True)
    parser.add_argument("--model", choices=("native", "siglip2"), required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "8"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = prepare_text(read_json(args.binding), args.dataset, args.model)
    print(result["status"], args.dataset, args.model, result["text_inputs_encoded"], flush=True)


if __name__ == "__main__":
    main()
