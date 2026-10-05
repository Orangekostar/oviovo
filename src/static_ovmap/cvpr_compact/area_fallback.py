"""Experimental v2 pooling for empty projected FC masks; v1 stays frozen."""

from types import SimpleNamespace

import torch
from torch.nn import functional as F


PROTOCOL = "PROJECTED_FC_EMPTY_AREA_FALLBACK_V2"


def pool_region(dense, signed, original_pool):
    if (dense.ndim != 4 or signed.ndim != 4 or dense.shape[0] != 1
            or signed.shape[:2] != (1, 1) or dense.dtype != torch.float32
            or not torch.isfinite(dense).all()):
        raise ValueError("pooling requires finite FP32 single-image dense features")
    if (signed.dtype != torch.float32 or signed.device != dense.device
            or not torch.isfinite(signed).all()
            or not ((signed == -1) | (signed == 1)).all()):
        raise ValueError("pooling requires the original finite +/-1 signed mask")
    hard = F.interpolate(signed, size=dense.shape[-2:], mode="bilinear", align_corners=False) > 0
    support = int(hard.sum().item())
    audit = {"protocol": PROTOCOL, "original_support": support,
             "dense_hw": list(dense.shape[-2:]), "fallback": not bool(support)}
    if support:
        return original_pool(dense, signed), audit
    occupancy = (signed + 1) / 2
    weights = F.interpolate(occupancy, size=dense.shape[-2:], mode="area")
    mass = weights.sum((-1, -2), keepdim=True)
    if not float(mass.item()) > 0:
        raise ValueError("EMPTY_AREA_MASK_SUPPORT")
    pooled = torch.einsum("bchw,bqhw->bqc", dense, weights / (mass + 1e-8))
    audit.update(area_support=int((weights > 0).sum().item()), area_mass=float(mass.item()),
                 resized_foreground_pixels=int(occupancy.sum().item()))
    return pooled, audit


def region_vector(model, operators, dense, signed):
    pooled, audit = pool_region(dense, signed, operators["MaskPooling"]())
    projected = operators["visual_prediction_forward_convnext"](
        SimpleNamespace(clip_model=model), pooled, signed
    )
    if not torch.isfinite(projected).all() or torch.linalg.vector_norm(projected).item() <= 1e-12:
        raise ValueError("INVALID_REGION_FEATURE")
    return F.normalize(projected, dim=-1)[0, 0], audit
