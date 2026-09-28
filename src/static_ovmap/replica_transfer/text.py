"""FP32 Replica text spaces in the two already frozen visual models."""

import argparse
import os
import time
from pathlib import Path


def prepare(config, model_name):
    import numpy as np
    import torch
    import transformers

    from src.static_ovmap.composition_study.io import SourceIndex, read_json, write_once
    from src.static_ovmap.module_validation.contracts import canonical_digest
    from src.static_ovmap.module_validation.native_capture import _write_npz
    from src.static_ovmap.module_validation.rgb_siglip import FrozenSiglipBackend
    from src.static_ovmap.module_validation.scannet_runtime import reusable_job

    original = config["source_models"][model_name]
    if (torch.__version__, transformers.__version__) != (original["torch_version"], original["transformers_version"]):
        raise ValueError("text worker must use the original model environment")
    index = SourceIndex()
    for entry in original["files"]:
        index.identity(entry["path"], entry)
    index.identity(Path(config["runtime"]["upstream"]) / "scripts/utils/semantic_const.py")
    index.identity(Path(__file__))
    names = tuple(config["class_names"])
    canonical = ("object", "things", "stuff", "texture")
    identity = canonical_digest({"inputs": index.manifest(), "names": names, "canonical": canonical,
        "valid_ids": config["valid_ids"], "device": "cpu", "dtype": "float32",
        "torch": torch.__version__, "transformers": transformers.__version__, "batch_size": 16})
    output = Path(config["attempt_root"]) / "text" / model_name
    receipt = output / "receipt.json"
    if reusable_job(receipt, identity):
        return read_json(output / "model_binding.json")
    if receipt.exists():
        raise ValueError("changed text inputs require a new transfer attempt")
    started = time.monotonic()
    backend = FrozenSiglipBackend.from_local(original["path"], device="cpu")
    if any(p.dtype != torch.float32 for p in backend.model.parameters() if p.is_floating_point()):
        raise ValueError("text model is not FP32")
    text = np.concatenate([backend.encode_texts(names[i:i + 16]) for i in range(0, len(names), 16)])
    canon = backend.encode_texts(canonical)
    if not np.isfinite(text).all() or not np.isfinite(canon).all():
        raise ValueError("nonfinite text features")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "text.npz"
    _write_npz(path, {"text_embeddings": text, "canonical_embeddings": canon,
        "class_names": np.asarray(names), "canonical_phrases": np.asarray(canonical),
        "valid_ids": np.asarray(config["valid_ids"])})
    write_once(receipt, {"status": "COMPLETE", "input_identity": identity, "inputs": index.manifest()["entries"],
        "outputs": [index.identity(path)], "dtype": "float32", "device": "cpu", "model_loads": 1,
        "model_path": original["path"], "text_inputs": len(names) + len(canonical),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "elapsed_seconds": time.monotonic() - started})
    binding = {**original, "text": index.identity(path), "text_receipt": str(receipt),
        "valid_ids": config["valid_ids"], "class_names": config["class_names"]}
    write_once(output / "model_binding.json", binding)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("native", "siglip2"))
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "8"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .protocol import require_transfer

    config = require_transfer(args.config)
    os.environ["HF_MODULES_CACHE"] = config["runtime"]["hf_modules_cache"]
    result = prepare(config, args.model)
    print(f"COMPLETE {args.model}: {result['text']['path']}", flush=True)


if __name__ == "__main__":
    main()
