"""METER encoder + decoder + LeJEPA projector wrapper.

Architecture ported from Papa et al. (2024) METER paper.
See docs/architecture.md for detailed explanation.
"""

import torch
import torch.nn as nn
from torchvision.ops import MLP
from einops import rearrange

from src.config import EMB_DIM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Building blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, depth=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels * depth,
                                   kernel_size=kernel_size, groups=depth,
                                   padding=1, stride=stride, bias=bias)
        self.pointwise = nn.Conv2d(out_channels * depth, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU())


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        SeparableConv2d(inp, oup, kernel_size, stride=stride, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU())


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim),
                                     nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b p n (h d) -> b p h n d",
                                           h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = torch.matmul(self.attend(dots), v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MobileNetV2 Inverted Residual Block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expansion)
        self.use_res_connect = (stride == 1 and inp == oup)
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MobileViT Block (METER version: single transformer, depth=1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim,
                 dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv2(self.conv1(x))
        _, _, h, w = x.shape
        x = rearrange(x, "b d (h ph) (w pw) -> b (ph pw) (h w) d",
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, "b (ph pw) (h w) d -> b d (h ph) (w pw)",
                      h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = self.conv4(torch.cat((x, y), dim=1))
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METER Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class METEREncoder(nn.Module):
    """METER encoder backbone — all three variants (xxs / xs / s).

    Returns:
        feat: (B, C_final, H/32, W/32) — final feature map
        skips: [y0, y1, y2, y3] — intermediate features for decoder
    """

    def __init__(self, image_size, dims, channels, expansion=4,
                 kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ph, pw = patch_size
        assert image_size[0] % ph == 0 and image_size[1] % pw == 0
        # METER: single transformer per block (depth=1)
        L = [1, 1, 1]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        self.mv2 = nn.ModuleList([
            MV2Block(channels[0], channels[1], 1, expansion),
            MV2Block(channels[1], channels[2], 2, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
            MV2Block(channels[3], channels[4], 2, expansion),
            MV2Block(channels[5], channels[6], 2, expansion),
            MV2Block(channels[7], channels[8], 2, expansion),
        ])
        self.mvit = nn.ModuleList([
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size,
                           patch_size, int(dims[0] * 2)),
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size,
                           patch_size, int(dims[1] * 4)),
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size,
                           patch_size, int(dims[2] * 4)),
        ])
        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

    def forward(self, x):
        y0 = self.conv1(x)
        x = self.mv2[0](y0)

        y1 = self.mv2[1](x)
        x = self.mv2[3](self.mv2[2](y1))

        y2 = self.mv2[4](x)
        x = self.mvit[0](y2)

        y3 = self.mv2[5](x)
        x = self.mvit[1](y3)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        return x, [y0, y1, y2, y3]


# ── Variant factory functions ─────────────────────────────────────────


def meter_xxs(image_size):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 160]
    return METEREncoder(image_size, dims, channels, expansion=2)


def meter_xs(image_size):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 192]
    return METEREncoder(image_size, dims, channels)


def meter_s(image_size):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 320]
    return METEREncoder(image_size, dims, channels)


_BACKBONE_FN = {"xxs": meter_xxs, "xs": meter_xs, "s": meter_s}

# Channel configs for decoder (derived from official METER source)
_DECODER_CFG = {
    "xxs": {"in_ch": 160, "reduce_ch": 64,
             "up_chs": [(64, 32, 96), (32, 16, 64), (16, 8, 32)], "out_ch": 8},
    "xs":  {"in_ch": 192, "reduce_ch": 128,
             "up_chs": [(128, 64, 144), (64, 32, 96), (32, 16, 64)], "out_ch": 16},
    "s":   {"in_ch": 320, "reduce_ch": 128,
             "up_chs": [(128, 64, 192), (64, 32, 128), (32, 16, 80)], "out_ch": 16},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METER Decoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class UpSampleBlock(nn.Module):
    """Decoder upsampling block: ConvTranspose2d → concat skip → SeparableConv + ReLU."""

    def __init__(self, inp: int, oup: int, sep_conv_filters: int):
        super().__init__()
        self.conv2d_transpose = nn.ConvTranspose2d(
            inp, oup, kernel_size=3, stride=2, padding=1,
            output_padding=1, bias=False)
        self.end_up_layer = nn.Sequential(
            SeparableConv2d(sep_conv_filters, oup, kernel_size=3),
            nn.ReLU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_transpose(x)
        # Pad skip if spatial dims don't match (can happen with odd resolutions)
        if x.shape[-1] != skip.shape[-1]:
            skip = nn.functional.pad(skip, (0, x.shape[-1] - skip.shape[-1]))
        if x.shape[-2] != skip.shape[-2]:
            skip = nn.functional.pad(skip, (0, 0, 0, x.shape[-2] - skip.shape[-2]))
        x = torch.cat([x, skip], dim=1)
        return self.end_up_layer(x)


class METERDecoder(nn.Module):
    """METER fully-convolutional decoder with skip connections.

    Takes encoder output + 4 skip connections, produces dense depth map.
    """

    def __init__(self, variant: str = "xxs", depth_bias: float = 3.0):
        super().__init__()
        cfg = _DECODER_CFG[variant]

        self.conv_in = nn.Conv2d(cfg["in_ch"], cfg["reduce_ch"],
                                 kernel_size=1, bias=False)

        up_chs = cfg["up_chs"]
        self.up1 = UpSampleBlock(up_chs[0][0], up_chs[0][1], up_chs[0][2])
        self.up2 = UpSampleBlock(up_chs[1][0], up_chs[1][1], up_chs[1][2])
        self.up3 = UpSampleBlock(up_chs[2][0], up_chs[2][1], up_chs[2][2])

        self.conv_out = nn.Conv2d(cfg["out_ch"], 1, kernel_size=3, padding=1,
                                  bias=True)
        # Initialize bias to dataset mean depth so output starts positive
        # (prevents stuck-at-zero predictions due to ReLU at output)
        nn.init.constant_(self.conv_out.bias, depth_bias)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor],
                target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        Args:
            x: encoder output (B, C_final, H/32, W/32)
            skips: [y0, y1, y2, y3] from encoder
            target_size: (H, W) for final bilinear interpolation
        Returns:
            depth: (B, 1, H, W)
        """
        x = self.conv_in(x)
        x = self.up1(x, skips[3])   # H/32 → H/16
        x = self.up2(x, skips[2])   # H/16 → H/8
        x = self.up3(x, skips[1])   # H/8  → H/4
        x = self.conv_out(x)        # → (B, 1, H/4, W/4)

        if target_size is not None:
            x = nn.functional.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False)

        return nn.functional.relu(x)  # Depth must be ≥ 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full METER Model (encoder + decoder)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class METERModel(nn.Module):
    """Full METER model: encoder + convolutional decoder.

    Input:  (B, 3, H, W) — single RGB image
    Output: (B, 1, H, W) — dense depth map
    """

    def __init__(self, variant: str = "xxs",
                 resolution: tuple[int, int] = (256, 192),
                 depth_bias: float = 3.0):
        super().__init__()
        self.resolution = resolution  # (H, W) target output
        self.encoder = _BACKBONE_FN[variant](resolution)
        self.decoder = METERDecoder(variant, depth_bias=depth_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, skips = self.encoder(x)
        return self.decoder(feat, skips, target_size=self.resolution)

    def load_pretrained_encoder(self, checkpoint_path: str):
        """Load pretrained encoder weights from a LeJEPA checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=True)
        self.encoder.load_state_dict(state_dict, strict=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LeJEPA wrapper — backbone + GAP + projector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class METERLeJEPA(nn.Module):
    """METER encoder + projector for LeJEPA self-supervised pre-training.

    Input:  (B, V, 3, H, W) — V augmented views per image
    Output:
        emb:  (B*V, emb_dim) — backbone embeddings (for PCA probing)
        proj: (V, B, proj_dim) — projected embeddings (for LeJEPA loss)
    """

    def __init__(self, variant: str = "xxs", proj_dim: int = 16,
                 resolution: int = 128):
        super().__init__()
        image_size = (resolution, resolution)
        emb_dim = EMB_DIM[variant]

        self.backbone = _BACKBONE_FN[variant](image_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = MLP(emb_dim,
                        hidden_channels=[2048, 2048, proj_dim],
                        norm_layer=nn.BatchNorm1d)
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor):
        B, V = x.shape[:2]
        flat = x.flatten(0, 1)              # (B*V, 3, H, W)
        feat, _ = self.backbone(flat)       # (B*V, C, h, w)
        emb = self.pool(feat).flatten(1)    # (B*V, emb_dim)
        proj = self.proj(emb).reshape(B, V, -1)  # (B, V, proj_dim)
        proj = proj.transpose(0, 1)         # (V, B, proj_dim)
        return emb, proj
