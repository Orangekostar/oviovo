"""Real two-mask CAL smoke for the pinned FC/OVR region-only adapter."""

import argparse
import fcntl
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.native_capture import _write_npz
from src.static_ovmap.module_validation.scannet_runtime import require_idle_gpu
from src.static_ovmap.module_validation.semantic_models import _load_request

from .region_adapter import (
    image_tensor,
    load_model,
    operator_nodes,
    region_vector,
    signed_mask,
)


def smoke(binding, scene, assets, branch, gpu):
    import open_clip
    import timm
    import torch
    from torch.nn import functional as F

    started, index = time.monotonic(), InputIndex()
    spec = read_json(binding["spec"])
    if scene not in spec["datasets"]["calibration"]:
        raise ValueError("region smoke requires CAL")
    bound = binding["scenes"][scene]
    config = read_json(bound["config"])
    assets = Path(assets)
    for leaf in ("ovrcoat", "fc_frozen"):
        index.identity(assets / leaf / "download_receipt.json")
        receipt = read_json(assets / leaf / "download_receipt.json")
        for item in receipt.get("files", [receipt.get("checkpoint")]):
            index.identity(item["path"], item)
    code = assets / "ovrcoat/code"
    operators, prompts = operator_nodes(code)
    for relative in ("ovrcoat/modeling/backbone/clip.py", "ovrcoat/ovrcoat.py",
                     "ovrcoat/modeling/transformer_decoder/fcclip_transformer_decoder.py", "LICENSE"):
        index.identity(code / relative)
    for path in (Path(__file__), Path(__file__).with_name("region_adapter.py")):
        index.identity(path)
    index.identity(bound["static_manifest"]["path"], bound["static_manifest"])
    manifest = read_json(bound["static_manifest"]["path"])
    index.identity(manifest["capture"]["path"], manifest["capture"])
    root = Path(manifest["capture"]["path"]).parent
    frames = {f["frame_id"]: f for f in read_json(manifest["capture"]["path"])["frames"]}
    requests = {k: v.get("request", v) for k, v in manifest["requests"].items()}
    selected = None
    for frame_id in sorted({r["frame_id"] for r in requests.values()}):
        candidates = sorted(k for k, r in requests.items() if r["frame_id"] == frame_id)
        if len(candidates) > 1:
            pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                     if requests[a]["target_mask_sha256"] != requests[b]["target_mask_sha256"]]
            if pairs:
                selected = pairs[0]
                break
    if selected is None:
        raise ValueError("no deterministic pair of distinct captured masks on one CAL frame")
    values = []
    for request_id in selected:
        request = requests[request_id]
        frame = frames[request["frame_id"]]
        index.identity(root / frame["rgb_path"], {"sha256": request["image_sha256"]})
        values.append(_load_request(root, frame, request, fresh_masks=manifest["requests"][request_id].get("masks")))
    if not np.array_equal(values[0]["rgb"], values[1]["rgb"]):
        raise ValueError("same-frame smoke RGB mismatch")
    identity = canonical_digest({"binding": binding["identity"], "scene": scene, "branch": branch,
        "requests": selected, "inputs": index.entries(), "torch": torch.__version__,
        "open_clip": open_clip.__version__, "timm": timm.__version__, "precision": "float32"})
    output = Path(binding["output_root"]) / "e03/smoke" / scene / branch
    if (output / "receipt.json").exists():
        previous = read_json(output / "receipt.json")
        if previous["identity"] != identity:
            raise ValueError("region smoke dependencies changed")
        for item in previous["inputs"] + previous["outputs"]:
            index.identity(item["path"], item)
        return previous
    lock = Path(config["gpu_lock"]).with_name(f".visual-gpu-{gpu}.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        require_idle_gpu(str(gpu), output)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        model, tokenizer, audit = load_model(assets / "ovrcoat", assets / "fc_frozen", branch, "cuda")
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - begin
        names = config["models"]["siglip2"]["class_names"]
        text_calls = 0
        inference_start = time.monotonic()
        with torch.inference_mode():
            prototypes = []
            for name in names:
                tokens = tokenizer([p.format(name) for p in prompts]).to("cuda")
                text_calls += 1
                encoded = model.encode_text(tokens, normalize=True)
                prototypes.append(F.normalize(encoded.mean(0), dim=0))
            text = torch.stack(prototypes)
            image, shape = image_tensor(values[0]["rgb"], "cuda")
            dense = operators["extract_features_convnext"](SimpleNamespace(clip_model=model), image)["clip_vis_dense"]
            vectors, diagnostics, arrays = [], [], {"text_prototypes": text.cpu().numpy()}
            for i, value in enumerate(values):
                signed, support, resized = signed_mask(value["target"], shape, image.shape[-2:], dense.shape[-2:], "cuda")
                vector, count = region_vector(model, operators, dense, signed)
                vectors.append(vector)
                diagnostics.append({"request_id": selected[i], "original_pixels": int(value["target"].sum()),
                                    "resized_pixels": int(resized.sum()), "dense_pixels": count})
                for label, mask in (("original", value["target"]), ("resized", resized), ("dense", support[0, 0].cpu().numpy())):
                    arrays[f"{i}_{label}_bits"] = np.packbits(mask)
                    arrays[f"{i}_{label}_shape"] = np.asarray(mask.shape)
            vectors = torch.stack(vectors)
            cosine = vectors @ text.T
            # The author helper expects a final void slot; append a zero dummy
            # solely for this operator comparison and discard its output.
            dummy = torch.zeros_like(text[:1])
            official = operators["get_classification_logits"](vectors[None], torch.cat([text, dummy]),
                                                               model.logit_scale, [1] * len(names))[0, :, :-1]
            scale = torch.clamp(model.logit_scale.exp(), max=100)
            error = float((official / scale - cosine).abs().max().item())
            difference = float((vectors[0] - vectors[1]).abs().max().item())
            if error > 1e-6 or not np.isfinite(difference) or difference <= 1e-8:
                raise ValueError("real region smoke failed mask sensitivity or pre-scale cosine parity")
            arrays.update(region_features=vectors.cpu().numpy(), cosines=cosine.cpu().numpy())
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - inference_start
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    audit_path, arrays_path = output / "weight_audit.json", output / "features.npz"
    write_once(audit_path, audit)
    _write_npz(arrays_path, arrays)
    receipt = {"status": "SMOKE_COMPLETE", "identity": identity, "scene": scene, "branch": branch,
        "requests": selected, "diagnostics": diagnostics, "feature_dimension": int(vectors.shape[1]),
        "two_mask_max_feature_difference": difference, "official_prescale_cosine_max_error": error,
        "image_resize": "PIL bilinear, short800/max1333, nearest mask, signed negative padding",
        "physical_image_encodings": 1, "physical_region_poolings": 2, "physical_text_calls": text_calls,
        "physical_text_inputs": len(names) * len(prompts), "precision": "float32",
        "model_load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "wall_seconds": time.monotonic() - started, "peak_gpu_allocated_bytes": allocated,
        "peak_gpu_reserved_bytes": reserved, "inputs": index.entries(),
        "outputs": [index.identity(audit_path), index.identity(arrays_path)]}
    write_once(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--branch", choices=["FC_FROZEN", "OVR"], required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result = smoke(read_json(args.binding), args.scene, args.assets, args.branch, args.gpu)
    print({k: result[k] for k in ("status", "branch", "feature_dimension", "two_mask_max_feature_difference",
                                 "official_prescale_cosine_max_error")}, flush=True)
