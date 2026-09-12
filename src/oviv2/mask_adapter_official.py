# Vendored from hustvl/MaskAdapter at commit c0516d8a548d90055c3dca7f2a9b4281a4da842f.
# Licensed under Apache-2.0.
# Source files:
# - fcclip/modeling/meta_arch/convnext.py
# - fcclip/modeling/meta_arch/mask_adapter_head.py

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch import nn


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MASKAdapterHead(nn.Module):
    def __init__(
        self,
        clip_model_name,
        mask_in_chans: int,
        num_channels: int,
        use_checkpoint: bool,
        num_output_maps: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if "_base" in clip_model_name:
            clip_dim = 640
        elif "_large" in clip_model_name:
            clip_dim = 768

        self.fuse = nn.Conv2d(clip_dim, num_channels, 1)

        self.cnext1 = ConvNextBlock(num_channels)

        self.cnext2 = ConvNextBlock(num_channels)

        self.cnext3 = ConvNextBlock(num_channels)

        self.norm = nn.LayerNorm(num_channels)
        self.final = nn.Conv2d(num_channels, num_output_maps, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(
                mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1
            ),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, clip_dim, kernel_size=1),
        )

    def forward(self, clip_feature, masks):

        N = masks.size(1)
        masks = rearrange(masks, "B N H W -> (B N) H W").unsqueeze(dim=1)

        clip_feature = repeat(clip_feature, "B C H W -> (B N) C H W", N=N)

        H, W = clip_feature.shape[-2:]
        masks = F.interpolate(
            masks.float(), size=(H * 4, W * 4), mode="bilinear", align_corners=False
        )
        masks = self.mask_downscaling(masks)

        outputs = clip_feature + masks

        def _inner_forward(outputs):
            outputs = self.fuse(outputs)

            outputs = self.cnext1(outputs)

            outputs = self.cnext2(outputs)

            outputs = self.cnext3(outputs)

            outputs = outputs.permute(0, 2, 3, 1)
            outputs = self.norm(outputs.contiguous())
            outputs = outputs.permute(0, 3, 1, 2)

            outputs = self.final(outputs.contiguous())

            outputs = rearrange(outputs, "(B N) C H W -> B (N C) H W", N=N)

            return outputs

        if self.use_checkpoint and self.training:
            outputs = cp.checkpoint(_inner_forward, outputs, use_reentrant=False)
        else:
            outputs = _inner_forward(outputs)
        return outputs
