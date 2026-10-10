"""Matched MA, VIEW and SURFACE readouts; frozen phi stays in autograd."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .mask_adapter import MASKAdapterHead


def unit(value):
    if not torch.isfinite(value).all() or torch.any(torch.linalg.vector_norm(value, dim=-1) <= 1e-12):
        raise ValueError('Nonfinite or zero learned representation')
    return F.normalize(value, dim=-1)


class ReadoutHead(nn.Module):
    """Only trainable readout weights; no encoder/text state is registered here."""
    def __init__(self, grouping='NONE', projected_channels=768, use_checkpoint=False):
        super().__init__()
        if grouping not in ('NONE', 'VIEW', 'SURFACE') or projected_channels != 768:
            raise ValueError('Fixed grouping and measured large-head width768 required')
        self.grouping = grouping
        self.ma = MASKAdapterHead('convnext_large_d_320', 16, 256, use_checkpoint, 4)
        if grouping != 'NONE':
            self.embed = nn.Sequential(nn.Linear(projected_channels + 8, 128), nn.GELU(),
                                       nn.Linear(128, 128), nn.LayerNorm(128))
            self.quality = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            self.group_ff = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
            self.group_norm = nn.LayerNorm(128)
            self.global_key = nn.Sequential(nn.Linear(projected_channels, 128), nn.LayerNorm(128))
            self.queries = nn.Parameter(torch.empty(4, 128))
            nn.init.normal_(self.queries, mean=0, std=.02)

    def forward(self, inputs, phi, *, completed_steps=0, training_forward=False):
        views = torch.nonzero(inputs['view_valid'], as_tuple=False).flatten()
        if not len(views):
            raise ValueError('No real input views')
        raw = inputs['raw'][views]
        projected = inputs['projected'][views]
        maps = self.ma(projected, inputs['masks'][views])
        if maps.shape[-2:] != raw.shape[-2:]:
            maps = F.interpolate(maps, size=raw.shape[-2:], mode='bilinear', align_corners=False)
        weights = torch.softmax(F.logsigmoid(maps).flatten(2), dim=-1)
        global_raw = torch.einsum('vmk,vck->vmc', weights, raw.flatten(2))
        # Do not average raw vectors before this nonlinear frozen projection.
        global_projected = phi(global_raw)
        ma_views = unit(global_projected.mean(dim=1))
        ma_embedding = unit(ma_views.mean(dim=0))
        frame_ids = inputs.get('frame_ids', torch.arange(len(inputs['view_valid']), device=views.device))[views]
        result = dict(embedding=ma_embedding, ma_embedding=ma_embedding, ma_view_embeddings=ma_views,
                      ma_maps=weights, fallback=None, view_indices=views, frame_ids=frame_ids,
                      eta=0.0, grouping=self.grouping)
        if self.grouping == 'NONE':
            return result
        local_raw = inputs['local_raw'][views]
        local_unit = inputs['local_unit'][views]
        metadata = inputs['metadata'][views]
        valid = inputs['local_valid'][views].bool()
        if metadata.shape[-1] != 8 or local_raw.shape[:2] != valid.shape or local_unit.shape[:2] != valid.shape:
            raise ValueError('Shared physical sites and exactly8 nonsemantic metadata scalars required')
        # Missing entries cannot inject arbitrary padded values or NaNs.
        local_unit = torch.where(valid[..., None], local_unit, 0)
        metadata = torch.where(valid[..., None], metadata, 0)
        local_raw = torch.where(valid[..., None], local_raw, 0)
        hidden = self.embed(torch.cat((local_unit, metadata), dim=-1))
        reliability = self.quality(hidden).squeeze(-1)
        result.update(reliability=reliability, local_hidden=hidden, local_valid=valid)
        if not bool(valid.any()):
            result['fallback'] = 'NO_LOCAL_SUPPORT_MA_FALLBACK'
            return result
        nv, ng = valid.shape
        if self.grouping == 'VIEW':
            group_ids = torch.arange(nv, device=raw.device)[:, None].expand(nv, ng)
        else:
            group_ids = torch.arange(ng, device=raw.device)[None, :].expand(nv, ng)
        flat_ids, flat_valid = group_ids.flatten(), valid.flatten()
        present = torch.unique(flat_ids[flat_valid], sorted=True)
        members = (present[:, None] == flat_ids[None, :]) & flat_valid[None, :]
        logits = reliability.flatten()[None, :].expand(len(present), -1).masked_fill(~members, -torch.inf)
        local_weights = torch.softmax(logits, dim=-1)
        mean = local_weights @ hidden.reshape(-1, 128)
        group_keys = self.group_norm(mean + self.group_ff(mean))
        group_values = local_weights @ local_raw.reshape(-1, local_raw.shape[-1])
        global_keys = self.global_key(unit(global_projected)).reshape(-1, 128)
        global_values = global_raw.reshape(-1, global_raw.shape[-1])
        local_attention = torch.softmax(self.queries @ group_keys.T / math.sqrt(128), dim=-1)
        global_attention = torch.softmax(self.queries @ global_keys.T / math.sqrt(128), dim=-1)
        fused_raw = .5 * (local_attention @ group_values) + .5 * (global_attention @ global_values)
        fused_embedding = unit(phi(fused_raw).mean(dim=0))
        eta = .5 * min(1., completed_steps / 200) if training_forward else .5
        result.update(embedding=unit((1-eta)*ma_embedding + eta*fused_embedding), eta=eta,
                      group_count=len(present), local_attention=local_attention,
                      global_attention=global_attention)
        return result
