"""External-mask FC-CLIP/Mask-Adapter readout on fixed CROVE observations.

Adapted inference equations from hustvl/MaskAdapter@c0516d8 (Apache-2.0),
fcclip/fcclip.py and fcclip/modeling/backbone/clip.py. No segmentation decoder,
void embedding or in/out-vocabulary ensemble participates in this interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def learned_pool(
    raw: torch.Tensor,
    activations: torch.Tensor,
    num_maps: int,
    region_projection: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pool F_raw, project each activation map, then average maps per mask."""
    b, _c, h, w = raw.shape
    if num_maps < 1 or activations.shape[1] % num_maps:
        raise ValueError("activation channels must be masks times num_maps")
    maps = F.interpolate(activations, size=(h, w), mode="bilinear", align_corners=False)
    weights = F.softmax(F.logsigmoid(maps).reshape(b, -1, h * w), dim=-1)
    pooled = torch.bmm(weights, raw.flatten(2).transpose(1, 2))
    projected = region_projection(pooled)
    return projected.reshape(b, -1, num_maps, projected.shape[-1]).mean(dim=2)


def mean_pool(
    raw: torch.Tensor,
    masks: torch.Tensor,
    region_projection: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched F_raw mask mean with an explicit validity mask, never fake support."""
    weights = F.interpolate(
        masks.float(), size=raw.shape[-2:], mode="bilinear", align_corners=False
    )
    weights = weights.flatten(2)
    mass = weights.sum(dim=-1, keepdim=True)
    valid = mass[..., 0] > 0
    pooled = torch.bmm(weights / mass.clamp_min(1e-12), raw.flatten(2).transpose(1, 2))
    return region_projection(pooled), valid


class MaskAdapterReadout(nn.Module):
    """One official large checkpoint, strict core loading, paired mask readouts."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0"):
        super().__init__()
        import open_clip

        from src.oviv2.mask_adapter_official import MASKAdapterHead

        self.clip = open_clip.create_model("convnext_large_d_320", pretrained=None)
        self.head = MASKAdapterHead(
            clip_model_name="convnext_large_d_320",
            mask_in_chans=16,
            num_channels=768,
            use_checkpoint=False,
            num_output_maps=16,
        )
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = checkpoint["model"]
        # Only explicitly unused decoder/criterion/void entries are excluded.
        used_prefixes = ("backbone.clip_model.", "mask_adapter.")
        allowed_unused = ("sem_seg_head.", "criterion.", "void_embedding.")
        unexpected = [
            k for k in weights if not k.startswith(used_prefixes + allowed_unused)
        ]
        if unexpected:
            raise ValueError(f"unrecognized checkpoint components: {unexpected[:5]}")
        self.clip.load_state_dict(
            {
                k.removeprefix(used_prefixes[0]): v
                for k, v in weights.items()
                if k.startswith(used_prefixes[0])
            },
            strict=True,
        )
        self.head.load_state_dict(
            {
                k.removeprefix(used_prefixes[1]): v
                for k, v in weights.items()
                if k.startswith(used_prefixes[1])
            },
            strict=True,
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_large_d_320")
        self.register_buffer(
            "pixel_mean",
            torch.tensor([122.7709383, 116.7460125, 104.09373615]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([68.5005327, 66.6321579, 70.32316305]).view(1, 3, 1, 1),
        )
        self.to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return self.pixel_mean.device

    def prepare(
        self, rgb: np.ndarray, masks: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Official RGB longest-side 896 resize and normalized zero padding to 32."""
        rgb = np.asarray(rgb)
        masks = np.asarray(masks)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB input must be uint8 H,W,3")
        if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
            raise ValueError("external masks must align with the original RGB grid")
        if not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1)):
            raise ValueError("external masks must be finite in [0,1]")
        h, w = rgb.shape[:2]
        scale = 896 / max(h, w)
        nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
        resized = np.asarray(
            Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        ).copy()
        image = (
            torch.as_tensor(resized, device=self.device).permute(2, 0, 1)[None].float()
        )
        image = (image - self.pixel_mean) / self.pixel_std
        mask = torch.as_tensor(masks.copy(), device=self.device).float()[None]
        if len(masks):
            mask = F.interpolate(mask, size=(nh, nw), mode="nearest")
        else:
            mask = image.new_empty(1, 0, nh, nw)
        pad = (0, (-nw) % 32, 0, (-nh) % 32)
        return F.pad(image, pad), F.pad(mask, pad)

    def features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.clip.visual.trunk
        raw = trunk.stem(image)
        for stage in trunk.stages:
            raw = stage(raw)
        raw = trunk.norm_pre(raw).contiguous()
        projected = trunk.head.norm(raw)
        projected = trunk.head.drop(projected.permute(0, 2, 3, 1))
        projected = self.clip.visual.head(projected).permute(0, 3, 1, 2)
        return raw, projected

    def region_projection(self, features: torch.Tensor) -> torch.Tensor:
        b, n, c = features.shape
        projected = self.clip.visual.trunk.head(features.reshape(b * n, c, 1, 1))
        projected = self.clip.visual.head(projected)
        return projected.reshape(b, n, -1)

    @torch.inference_mode()
    def text_features(self, class_names: Sequence[str]) -> torch.Tensor:
        if not class_names:
            raise ValueError("full fixed class vocabulary is required")
        tokens = self.tokenizer([f"a photo of a {name}." for name in class_names]).to(
            self.device
        )
        return F.normalize(self.clip.encode_text(tokens), dim=-1)

    @torch.inference_mode()
    def classify_external_masks(
        self,
        rgb: np.ndarray,
        masks: np.ndarray,
        text_features: torch.Tensor,
        *,
        mask_batch_size: int = 8,
    ) -> dict[str, torch.Tensor]:
        if mask_batch_size < 1:
            raise ValueError("mask_batch_size must be positive")
        image, masks_tensor = self.prepare(rgb, masks)
        raw, projected = self.features(image)
        mean_embeddings, adapter_embeddings, kept = [], [], []
        for start in range(0, masks_tensor.shape[1], mask_batch_size):
            subset = masks_tensor[:, start : start + mask_batch_size]
            mean, valid = mean_pool(raw, subset, self.region_projection)
            indices = valid[0].nonzero().flatten()
            if not len(indices):
                continue
            activation = self.head(projected, subset[:, indices])
            learned = learned_pool(raw, activation, 16, self.region_projection)
            mean_embeddings.append(mean[:, indices])
            adapter_embeddings.append(learned)
            kept.append(indices + start)
        dim = text_features.shape[-1]
        mean = torch.cat(mean_embeddings, 1)[0] if kept else raw.new_empty(0, dim)
        learned = torch.cat(adapter_embeddings, 1)[0] if kept else raw.new_empty(0, dim)
        mean = F.normalize(mean, dim=-1)
        learned = F.normalize(learned, dim=-1)
        scale = self.clip.logit_scale.exp().clamp(max=100)
        result = {
            "mask_indices": torch.cat(kept)
            if kept
            else torch.empty(0, dtype=torch.long, device=self.device),
            "mean_embeddings": mean,
            "learned_embeddings": learned,
            "mean_logits": scale * mean @ text_features.T,
            "learned_logits": scale * learned @ text_features.T,
        }
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise ValueError("nonfinite mask readout")
        return result
