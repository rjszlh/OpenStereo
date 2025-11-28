# BiFPN version of the rlightstereo backbone. Only supports MobileNetV2.
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d


class _ChannelAlign(nn.Module):
    """1x1 projection to match channel counts before fusion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                act_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True),
            )

    def forward(self, x):
        return self.proj(x)


class _WeightedFusion(nn.Module):
    """BiFPN fast normalized fusion."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.align = nn.ModuleList(
            [_ChannelAlign(in_ch, out_channels) for in_ch in in_channels_list]
        )
        self.weights = nn.Parameter(torch.ones(len(in_channels_list), dtype=torch.float32))
        self.out_conv = BasicConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        )
        self.eps = 1e-4

    def forward(self, feats):
        aligned = [align(f) for align, f in zip(self.align, feats)]
        weights = F.relu(self.weights)
        fused = sum(w * f for w, f in zip(weights, aligned)) / (weights.sum() + self.eps)
        return self.out_conv(fused)


class BiFPNBlock(nn.Module):
    """Single BiFPN block (top-down then bottom-up) over four pyramid levels."""

    def __init__(self, channels):
        """
        Args:
            channels: list/tuple like [c5, c4, c3, c2].
        """
        super().__init__()
        c5_ch, c4_ch, c3_ch, c2_ch = channels

        self.p5_in = _ChannelAlign(c5_ch, c5_ch)

        # Top-down path
        self.p4_td = _WeightedFusion([c4_ch, c5_ch], c4_ch)
        self.p3_td = _WeightedFusion([c3_ch, c4_ch], c3_ch)
        self.p2_td = _WeightedFusion([c2_ch, c3_ch], c2_ch)

        # Bottom-up path
        self.p3_out = _WeightedFusion([c3_ch, c2_ch], c3_ch)
        self.p4_out = _WeightedFusion([c4_ch, c3_ch], c4_ch)
        self.p5_out = _WeightedFusion([c5_ch, c4_ch], c5_ch)

        self.upsample = partial(F.interpolate, scale_factor=2.0, mode='bilinear', align_corners=False)
        self.downsample = partial(F.max_pool2d, kernel_size=2, stride=2)

    def forward(self, c2, c3, c4, c5):
        # Top-down pathway
        p5_td = self.p5_in(c5)
        p4_td = self.p4_td([c4, self.upsample(p5_td)])
        p3_td = self.p3_td([c3, self.upsample(p4_td)])
        p2_td = self.p2_td([c2, self.upsample(p3_td)])

        # Bottom-up pathway
        p3_out = self.p3_out([p3_td, self.downsample(p2_td)])
        p4_out = self.p4_out([p4_td, self.downsample(p3_out)])
        p5_out = self.p5_out([p5_td, self.downsample(p4_out)])

        return p2_td, p3_out, p4_out, p5_out


class BiFPNBackbone(nn.Module):
    """
    MobileNetV2 backbone with a BiFPN neck.

    Output interface matches the original FPN backbone: [p2, p3, p4, p5].
    """

    def __init__(self, bifpn_layers=1):
        super().__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        channels = [160, 96, 32, 24]  # c5, c4, c3, c2

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]

        self.bifpn = nn.Sequential(*[BiFPNBlock(channels) for _ in range(bifpn_layers)])
        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 16, H/2, W/2]
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        p2, p3, p4, p5 = c2, c3, c4, c5
        for block in self.bifpn:
            p2, p3, p4, p5 = block(p2, p3, p4, p5)

        p2 = self.out_conv(p2)
        return [p2, p3, p4, p5]
