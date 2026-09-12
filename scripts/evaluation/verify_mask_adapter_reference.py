"""Compare CROVE wrapper to expressions loaded from the pinned official source."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.checkpoint as cp
from einops import rearrange, repeat
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oviv2.mask_adapter_readout import MaskAdapterReadout, learned_pool


def source_function(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    function = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    function.decorator_list = []
    namespace = {"torch": torch, "F": F, "cp": cp, "rearrange": rearrange, "repeat": repeat}
    exec(  # noqa: S102 -- execute pinned official reference, not the wrapper under test
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method_name], function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = MaskAdapterReadout(args.checkpoint)
    official_head_fn, _ = source_function(
        args.source / "fcclip/modeling/meta_arch/mask_adapter_head.py", "MASKAdapterHead", "forward"
    )
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    labels = np.asarray(Image.open(args.regions))
    values, counts = np.unique(labels, return_counts=True)
    # Fixed geometry-only choice for numerical testing; never uses semantic GT.
    eligible = values[(values != 0) & (counts >= 500)]
    selected = eligible[:3]
    if len(selected) < 3:
        raise ValueError("three real supported regions required")
    masks = np.stack([labels == value for value in selected])
    class_names = json.loads(
        (ROOT / "configs/evaluation/manifests/replica8.json").read_text()
    )["vocabulary"]["classes"]
    head_fn, _ = source_function(
        args.source / "fcclip/fcclip.py",
        "FCCLIP",
        "visual_prediction_forward_convnext_2d",
    )
    region_fn, _ = source_function(
        args.source / "fcclip/modeling/backbone/clip.py",
        "CLIP",
        "visual_prediction_forward_convnext",
    )
    raw_fn, _ = source_function(
        args.source / "fcclip/modeling/backbone/clip.py",
        "CLIP",
        "extract_features_convnext",
    )
    _, forward = source_function(args.source / "fcclip/fcclip.py", "FCCLIP", "forward")
    # Read the exact first (training/external-mask) pooling block from fixed source.
    assignments = [
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "maps_for_pooling" for t in n.targets
        )
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "interpolate"
    ]
    resize = min(assignments, key=lambda n: n.lineno)
    blocks = [
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.If)
        and "convnext" in ast.unparse(n.test)
        and n.lineno > resize.lineno
    ]
    block = min(blocks, key=lambda n: n.lineno)
    reference_code = compile(
        ast.Module(body=[resize, *block.body], type_ignores=[]),
        "official_pooling",
        "exec",
    )
    backbone = SimpleNamespace(clip_model=model.clip)
    backbone.visual_prediction_forward = lambda x: region_fn(backbone, x, None)
    reference_self = SimpleNamespace(backbone=backbone, num_output_maps=16)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        image, aligned = model.prepare(rgb, masks)
        raw, head = model.features(image)
        raw_reference = raw_fn(backbone, image)["clip_vis_dense"]
        head_reference = head_fn(reference_self, raw_reference)
        activations = model.head(head, aligned)
        reference_activations = official_head_fn(model.head, head_reference, aligned)
        embeddings = learned_pool(raw, activations, 16, model.region_projection)
        namespace = {
            "torch": torch,
            "F": F,
            "self": reference_self,
            "clip_feature": raw_reference,
            "outputs": reference_activations,
        }
        exec(reference_code, namespace)  # noqa: S102 -- exact pinned official pooling statements
        reference_embedding = namespace["pooled_clip_feature"]
        text = model.text_features(class_names)
        logits = (
            model.clip.logit_scale.exp().clamp(max=100)
            * F.normalize(embeddings, dim=-1)
            @ text.T
        )
        reference_logits = (
            model.clip.logit_scale.exp().clamp(max=100)
            * F.normalize(reference_embedding, dim=-1)
            @ text.T
        )
        paired = model.classify_external_masks(rgb, masks, text, mask_batch_size=2)
        errors = {}
        for name, actual, expected in [
            ("F_raw", raw, raw_reference),
            ("F_head", head, head_reference),
            ("head_activations", activations, reference_activations),
            ("embedding", embeddings, reference_embedding),
            ("logits", logits, reference_logits),
            ("chunked_logits", paired["learned_logits"], logits[0]),
        ]:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            errors[name] = float((actual - expected).abs().max())
        record = {
            "status": "PASS",
            "scope": "three_real_masks_numerical_reference_not_full_map",
            "source_commit": "c0516d8a548d90055c3dca7f2a9b4281a4da842f",
            "scene": "room0",
            "frame": 0,
            "mask_source": "cached_OVI_geometric_segments",
            "region_ids": selected.tolist(),
            "input_shape": list(image.shape),
            "F_raw_shape": list(raw.shape),
            "F_head_shape": list(head.shape),
            "maximum_absolute_errors": errors,
            "mean_top1_names": [
                class_names[i] for i in paired["mean_logits"].argmax(-1).tolist()
            ],
            "learned_top1_names": [
                class_names[i] for i in paired["learned_logits"].argmax(-1).tolist()
            ],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "classification_excludes_void_and_decoder": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
