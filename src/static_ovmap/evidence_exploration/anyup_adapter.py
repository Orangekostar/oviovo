"""Owner/depth adaptation after original averaged-head attention, before raw V.

AnyUp: Thomas Wimmer et al., wimmerth/anyup@351807a9, CC-BY-4.0.
New geometry factors and streaming wrapper are our modifications; upstream stays intact.
"""

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image


def aligned_inputs(rgb, owners, depth, resized_size, padded_size, *, device):
    """Same PIL RGB resize as FC; categorical/depth nearest; normalized-zero pad."""
    h, w = map(int, resized_size)
    ph, pw = map(int, padded_size)
    if owners.shape != rgb.shape[:2] or depth.shape != owners.shape or ph < h or pw < w:
        raise ValueError("unaligned original evidence")
    image = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)).copy()
    guidance = torch.from_numpy(image).permute(2, 0, 1).to(device=device, dtype=torch.float32)/255
    guidance = (guidance-guidance.new_tensor([.485, .456, .406])[:, None, None])/guidance.new_tensor([.229, .224, .225])[:, None, None]
    guidance = F.pad(guidance[None], (0, pw-w, 0, ph-h), value=0)
    if np.max(owners, initial=0) > np.iinfo(np.int32).max:
        raise ValueError("categorical owner does not fit exact PIL integer resize")
    own = np.asarray(Image.fromarray(owners.astype(np.int32)).resize((w, h), Image.Resampling.NEAREST)).copy()
    dep = np.asarray(Image.fromarray(depth).resize((w, h), Image.Resampling.NEAREST)).copy()
    own = F.pad(torch.from_numpy(own.astype(np.int64)).to(device)[None, None], (0, pw-w, 0, ph-h), value=0)
    dep = F.pad(torch.from_numpy(dep).to(device)[None, None], (0, pw-w, 0, ph-h), value=0)
    valid = torch.isfinite(dep) & (dep > 1e-6)
    return guidance, own, torch.where(valid, dep, 0), valid


def owner_grids(owners, depth, valid, owner, key_hw, query_hw):
    present = (owners == int(owner)).float()
    usable = present * valid.float()
    def moments(hw):
        mass = F.interpolate(usable, size=hw, mode="area")
        total = F.interpolate(depth*usable, size=hw, mode="area")
        return torch.where(mass > 0, total/mass.clamp_min(1e-30), torch.nan).flatten()
    return {"occupancy": F.interpolate(present, size=key_hw, mode="area").flatten(),
            "key_depth": moments(key_hw), "query_depth": moments(query_hw)}


@torch.inference_mode()
def encode_qkv(model, guidance, dense, output_hw):
    """Pinned original encoders, positions, Q convolution, and raw FC V."""
    from anyup.utils.img import create_coordinate
    enc = model.image_encoder(guidance)
    b, c, h, w = enc.shape
    coordinates = create_coordinate(h, w, device=enc.device, dtype=enc.dtype)
    enc = model.rope(enc.permute(0, 2, 3, 1).reshape(b, h*w, c), coordinates)
    enc = enc.reshape(b, h, w, c).permute(0, 3, 1, 2)
    query = F.adaptive_avg_pool2d(model.query_encoder(enc), output_hw)
    key = F.adaptive_avg_pool2d(model.key_encoder(enc), dense.shape[-2:])
    key = model.aggregation(torch.cat([key, model.key_features_encoder(F.normalize(dense, dim=1))], dim=1))
    query = model.cross_decode.conv2d(query)
    query = query.permute(0, 2, 3, 1).reshape(b, -1, query.shape[1])
    key = key.permute(0, 2, 3, 1).reshape(b, -1, key.shape[1])
    value = dense.permute(0, 2, 3, 1).reshape(b, -1, dense.shape[1])
    return query, key, value


@torch.inference_mode()
def stream_regions(model, qkv, key_hw, output_hw, regions, unknown_key, *, chunk_size=256, return_full=False):
    """Share original attention across three readouts; keep only chunk x K weights.

    regions maps a stable region key to fine-grid weights and full owner moments.
    Only positive pooling-mass query locations run attention. Full Q convolutions
    and pinned window bounds were already computed on their unmodified grids.
    return_full is restricted to the single full-frame neutrality validation.
    """
    from anyup.layers.attention.attention_masking import window2d
    query, key, value = qkv
    cross = model.cross_decode.cross_attn
    if query.shape[0] != 1 or chunk_size not in (256, 128, 64):
        raise ValueError("AnyUp requires batch one and a prespecified query chunk")
    keys = list(regions)
    weights = torch.stack([regions[r]["weights"].flatten() for r in keys])
    if (not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(1) <= 0).any()):
        raise ValueError("nonempty bound region has invalid fine-grid area mass")
    active = torch.arange(query.shape[1], device=query.device) if return_full else torch.nonzero(weights.sum(0) > 0).flatten()
    windows = window2d(tuple(key_hw), tuple(output_hw), model.cross_decode.window_ratio).reshape(-1, 4).to(query.device)
    ky = torch.arange(key_hw[0], device=query.device).repeat_interleave(key_hw[1])
    kx = torch.arange(key_hw[1], device=query.device).repeat(key_hw[0])
    qnorm, knorm = cross.norm_q(query), cross.norm_k(key)
    sums = {r: {variant: value.new_zeros(value.shape[-1]) for variant in ("anyup", "owner_depth", "owner_only")} for r in keys}
    full = value.new_empty((query.shape[1], value.shape[-1])) if return_full else None
    counters = {"query_chunks": 0, "query_locations": len(active), "original_attention_applications": 0,
                "geometry_attention_applications": 0, "all_masked_queries": 0}
    for start in range(0, len(active), chunk_size):
        ids = active[start:start+chunk_size]
        bounds = windows[ids]
        mask = ~((ky[None] >= bounds[:, :1]) & (ky[None] < bounds[:, 1:2])
                 & (kx[None] >= bounds[:, 2:3]) & (kx[None] < bounds[:, 3:4])) if model.cross_decode.window_ratio > 0 else None
        if mask is not None and mask.all(1).any():
            raise ValueError("original AnyUp all-masked query: representation unavailable")
        # Upstream MHA value is the unnormalized K; its projected output is ignored.
        _, attn = cross.attention(qnorm[:, ids], knorm, key, average_attn_weights=True, attn_mask=mask)
        attn = attn[0]
        if not torch.isfinite(attn).all() or (attn < 0).any() or (attn.sum(1) <= 0).any():
            raise ValueError("invalid original averaged-head attention")
        counters["query_chunks"] += 1
        counters["original_attention_applications"] += 1
        if return_full:
            full[ids] = attn @ value[0]
        for r in keys:
            mass = regions[r]["weights"].flatten()[ids]
            positive = mass > 0
            if not positive.any():
                continue
            own = regions[r]
            a, wq = attn[positive], mass[positive]
            sums[r]["anyup"] += (wq @ a) @ value[0]
            for variant, use_depth in (("owner_depth", True), ("owner_only", False)):
                factor = geometry_factor(own["occupancy"], unknown_key, own["query_depth"][ids[positive]],
                                         own["key_depth"], use_depth=use_depth)
                adapted = reweight_attention(a, factor)
                sums[r][variant] += (wq @ adapted) @ value[0]
                counters["geometry_attention_applications"] += 1
    pooled = {r: {variant: tensor/(regions[r]["weights"].sum()+1e-8) for variant, tensor in values.items()}
              for r, values in sums.items()}
    if full is not None:
        full = full.reshape(*output_hw, value.shape[-1]).permute(2, 0, 1)[None]
    return pooled, counters, full


def geometry_factor(owner_key, unknown_key, query_depth, key_owner_depth, *, use_depth=True):
    own, unknown, dq, dk = owner_key, unknown_key, query_depth, key_owner_depth
    if own.ndim != 1 or unknown.shape != own.shape or dk.shape != own.shape or dq.ndim != 1:
        raise ValueError("owner/key and query shapes differ")
    if (not torch.isfinite(own).all() or not torch.isfinite(unknown).all()
            or (own < 0).any() or (unknown < 0).any() or (own + unknown > 1 + 1e-6).any()):
        raise ValueError("invalid categorical owner occupancies")
    factor = (.05 + .95 * (own + unknown).clamp(0, 1)).expand(len(dq), -1)
    if use_depth:
        valid_q, valid_k = torch.isfinite(dq) & (dq > 0), torch.isfinite(dk) & (dk > 0)
        safe_q, safe_k = torch.where(valid_q, dq, 1.), torch.where(valid_k, dk, 1.)
        z = ((safe_q[:, None] - safe_k[None, :]).abs() / (.02 + .02 * safe_q[:, None])).clamp(max=3)
        factor = factor * torch.where(valid_q[:, None] & valid_k[None, :], torch.exp(-z), 1.)
    return factor


def reweight_attention(attention, factor):
    if (attention.ndim != 2 or factor.shape != attention.shape or not torch.isfinite(attention).all()
            or not torch.isfinite(factor).all() or (attention < 0).any() or (factor <= 0).any()):
        raise ValueError("attention requires finite matching positive factors")
    weighted = attention * factor
    mass = weighted.sum(1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("original AnyUp all-masked query: representation unavailable")
    return weighted / mass
