import torch.nn as nn
import torch.nn.functional as F

from .aggregation import MobileV2Residual, AttentionModule
from .acir_block import ACIRBlockECA


class AggregationACIR(nn.Module):
    """
    Aggregation module enhanced with ACIRBlockECA (ACNet + MobileV2Residual + ECA).
    The original Aggregation in aggregation.py is kept unchanged.
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels,
                 use_eca=False):
        super().__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio
        self.use_eca = use_eca

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        conv2_blocks = max(blocks[1] - 1, 0)
        conv2_layers = []
        if conv2_blocks >= 1:
            conv2_layers.append(MobileV2Residual(in_channels * 2, in_channels * 2, stride=1,
                                                 expanse_ratio=self.expanse_ratio))
        if conv2_blocks >= 2:
            conv2_layers.extend([
                ACIRBlockECA(in_channels * 2, in_channels * 2, stride=1,
                             expanse_ratio=self.expanse_ratio,
                             use_eca=self.use_eca)
                for _ in range(conv2_blocks - 1)
            ])
        self.conv2 = nn.Sequential(*conv2_layers)

        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        conv4_blocks = max(blocks[2] - 1, 0)
        conv4_layers = [
            ACIRBlockECA(in_channels * 4, in_channels * 4, stride=1,
                         expanse_ratio=self.expanse_ratio,
                         use_eca=self.use_eca)
            for _ in range(conv4_blocks)
        ]
        self.conv4 = nn.Sequential(*conv4_layers)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )
        self.conv5_refine = ACIRBlockECA(in_channels * 2, in_channels * 2, stride=1,
                                         expanse_ratio=self.expanse_ratio,
                                         use_eca=self.use_eca)

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])

        up2 = self.conv5(conv4)
        up2 = self.conv5_refine(up2)
        conv5 = F.relu(up2 + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return [conv6]
