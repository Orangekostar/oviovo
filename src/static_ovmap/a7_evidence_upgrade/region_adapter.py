"""Strict OVR/FC region operators, using selected pinned Apache-2.0 routines.

Operator source: nickormushev/OVRCOAT@9fd9450d22852d269d426b521663a127f3983a4b.
The original Bytedance/ODISE notices and LICENSE remain in the bound checkout.
Only named operator AST nodes are executed; the segmentation package is not
imported. This is operator-level reuse, not a complete upstream wrapper run.
"""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def operator_nodes(code):
    import torch
    from torch import nn
    from torch.nn import functional as F

    code = Path(code)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if commit != "9fd9450d22852d269d426b521663a127f3983a4b":
        raise ValueError("operator repository revision differs")

    def pinned_text(path):
        original = subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(code)}"], cwd=code)
        if path.read_bytes() != original:
            raise ValueError("pinned operator source was locally modified")
        return original.decode()

    namespace = {"torch": torch, "nn": nn, "F": F}
    backbone = code / "ovrcoat/modeling/backbone/clip.py"
    tree = ast.parse(pinned_text(backbone))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLIP")
    names = {"extract_features_convnext", "visual_prediction_forward_convnext"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 2:
        raise ValueError("pinned region operator boundary missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(backbone), "exec"), namespace)  # noqa: S102 -- verified pinned operators
    decoder = code / "ovrcoat/modeling/transformer_decoder/fcclip_transformer_decoder.py"
    tree = ast.parse(pinned_text(decoder))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {"MaskPooling", "get_classification_logits"}]
    if len(nodes) != 2:
        raise ValueError("pinned pooling/classification boundary missing")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(decoder), "exec"), namespace)  # noqa: S102 -- verified pinned operators
    tree = ast.parse(pinned_text(code / "ovrcoat/ovrcoat.py"))
    prompts = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "VILD_PROMPT" for t in n.targets))
    if len(prompts) != 14 or any(p.count("{}") != 1 for p in prompts):
        raise ValueError("pinned VILD prompt list changed")
    return namespace, prompts


def resized_shape(shape):
    height, width = shape
    scale = min(800 / min(height, width), 1333 / max(height, width))
    return int(height * scale + .5), int(width * scale + .5)


def image_tensor(rgb, device):
    import torch
    from torch.nn import functional as F

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("region image requires RGB HWC uint8")
    height, width = resized_shape(rgb.shape[:2])
    resized = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device)
    mean = tensor.new_tensor([122.7709383, 116.7460125, 104.09373615])[:, None, None]
    std = tensor.new_tensor([68.5005327, 66.6321579, 70.32316305])[:, None, None]
    tensor = (tensor - mean) / std
    tensor = F.pad(tensor, (0, -width % 32, 0, -height % 32), value=0)
    return tensor[None], (height, width)


def signed_mask(mask, size, padded_size, dense_size, device):
    import torch
    from torch.nn import functional as F

    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("region mask requires full-resolution boolean input")
    height, width = size
    resized = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)).copy()
    signed = torch.from_numpy(resized.astype(np.float32) * 2 - 1).to(device)[None, None]
    signed = F.pad(signed, (0, padded_size[1] - width, 0, padded_size[0] - height), value=-1)
    dense = F.interpolate(signed, size=dense_size, mode="bilinear", align_corners=False)
    support = dense > 0
    return signed, support, resized


def region_vector(model, operators, dense, signed):
    import torch
    from torch.nn import functional as F

    down = F.interpolate(signed, size=dense.shape[-2:], mode="bilinear", align_corners=False)
    support = int((down > 0).sum().item())
    if not support:
        raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
    pooled = operators["MaskPooling"]()(dense, signed)
    projected = operators["visual_prediction_forward_convnext"](SimpleNamespace(clip_model=model), pooled, signed)
    if not torch.isfinite(projected).all() or torch.linalg.vector_norm(projected).item() <= 1e-12:
        raise ValueError("INVALID_REGION_FEATURE")
    return F.normalize(projected, dim=-1)[0, 0], support


def load_model(ovr_assets, fc_assets, branch, device):
    import open_clip
    import torch
    from safetensors.torch import load_file

    if branch not in {"FC_FROZEN", "OVR"}:
        raise ValueError("unknown matched branch")
    model = open_clip.create_model("convnext_large_d_320", pretrained=None, device="cpu")
    checkpoint = torch.load(Path(ovr_assets) / "ovrcoat.pth", map_location="cpu", weights_only=False)["model"]
    prefix = "backbone.clip_model."
    learned = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    frozen = load_file(str(Path(fc_assets) / "model/open_clip_model.safetensors"), device="cpu")
    expected = model.state_dict()
    for name, state in (("OVR", learned), ("FC_FROZEN", frozen)):
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError(f"{name} active key/shape coverage is not exact")
    changed = {k: int(torch.count_nonzero(learned[k] != frozen[k]).item()) for k in expected}
    active_changed = sum(v for k, v in changed.items() if k.startswith("visual.trunk."))
    if not active_changed:
        raise ValueError("OVR active visual trunk is identical to matched original weights")
    state = learned if branch == "OVR" else frozen
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    mapping = [{"checkpoint_key": prefix + k if branch == "OVR" else k, "module_key": k,
                "shape": list(v.shape), "elements": v.numel()} for k, v in state.items()]
    audit = {"branch": branch, "loaded_key_count": len(state), "strict": True,
             "mapping": mapping, "excluded_OVR_keys": sorted(k for k in checkpoint if not k.startswith(prefix)),
             "OVR_vs_original_changed_elements": changed, "active_trunk_changed_elements": active_changed,
             "random_active_parameters": False, "complete_upstream_wrapper_executed": False,
             "operator_source": "pinned AST extraction", "open_clip_version": open_clip.__version__}
    return model, open_clip.get_tokenizer("convnext_large_d_320"), audit
