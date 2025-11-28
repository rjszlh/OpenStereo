import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import MobileV2Residual, AttentionModule


class DualBranchAggregation(nn.Module):
    """
    Dual-branch cost aggregation guided by backbone attention.

    Design (following user idea):
    - Use att0 (from features_left[0]) to build a soft mask that splits
      the initial cost into a detail branch and a smooth branch.
    - First apply a shared stem conv on the input cost volume (2D conv
      on [B, C, H, W]), then split into two branches:
        cv1_det = cv1 * mask
        cv1_smooth = cv1 * (1 - mask)
    - Each branch performs a shallow encoder-decoder style aggregation.
      At intermediate stages, att2 / att4 are used to enhance features.
    - The two branch outputs are merged using the same mask:
        out = mask * out_det + (1 - mask) * out_smooth
      followed by ReLU.

    Notes:
    - This module is drop-in compatible with the original Aggregation:
      forward(x, features_left) -> [cost_encoding]
      where x has shape [B, C, H, W] and features_left is a list of
      backbone feature maps.
    - Channel sizes, strides and overall spatial scales follow the
      original Aggregation to keep interfaces compatible.
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(DualBranchAggregation, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio
        self.blocks = blocks

        # Shared conv0 on cost volume before branching.
        # Keep channels as in_channels and spatial size H/4 x W/4.
        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        # Two parallel branches constructed via shared helpers (structure相同、参数独立).
        self.branch_det = self._build_branch(in_channels)
        self.branch_smooth = self._build_branch(in_channels)

        # Redirection from shared conv0 features into decoders (final stage).
        self.redir_conv0 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            # Reuse existing attention modules for att0/att2/att4.
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4, backbone_channels[2])

            # Small heads to convert att0 feature into a soft spatial mask.
            # The mask is in [0, 1] and is broadcast along channels to split
            # the shared stem feature into detail and smooth branches.
            self.mask_head = nn.Sequential(
                nn.Conv2d(in_channels, 1, 1),
                nn.Sigmoid()
            )
        else:
            # 常量mask放入buffer，避免反复分配且自动匹配设备/精度。
            self.register_buffer("default_mask", torch.tensor(0.5, dtype=torch.float32))

    def _build_branch(self, in_channels):
        branch = nn.ModuleDict()
        branch["conv1"] = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2 = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(self.blocks[1] - 1)]
        branch["conv2"] = nn.Sequential(*conv2)
        branch["conv3"] = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4 = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(self.blocks[2] - 1)]
        branch["conv4"] = nn.Sequential(*conv4)
        branch["deconv5"] = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )
        branch["deconv6"] = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        return branch

    def _forward_branch(self, branch, b0, features_left):
        b1 = branch["conv1"](b0)
        b2 = branch["conv2"](b1)
        if self.left_att:
            b2 = self.att2(b2, features_left[1])
        b3 = branch["conv3"](b2)
        b4 = branch["conv4"](b3)
        if self.left_att:
            b4 = self.att4(b4, features_left[2])

        # decoder with intra-branch skip connections
        d5 = branch["deconv5"](b4)
        d5 = F.relu(d5 + b2, inplace=True)
        d6 = branch["deconv6"](d5)
        d6 = F.relu(d6 + b0, inplace=True)
        return d6

    def forward(self, x, features_left):
        """
        Args:
            x: cost volume encoding, shape [B, C, H, W].
            features_left: list of backbone features with three scales.
        Returns:
            A single-element list [conv_out] with shape [B, C, H, W].
        """
        # Shared conv0 on cost.
        conv0 = self.conv0(x)  # [B, C, H/4, W/4]

        if self.left_att:
            # att0-guided spatial mask for detail/smooth splitting.
            att0_feat = self.att0(conv0, features_left[0])  # [B, C, H/4, W/4]
            mask = self.mask_head(att0_feat)  # [B, 1, H/4, W/4] in [0, 1]
        else:
            # If no left attention, fall back to a uniform 0.5 mask.
            mask = self.default_mask.expand(conv0.size(0), 1, conv0.size(2), conv0.size(3))

        # Split into detail and smooth branches.
        conv0_det = conv0 * mask
        conv0_smooth = conv0 * (1.0 - mask)

        # Detail / smooth branches share执行逻辑。
        d6_det = self._forward_branch(self.branch_det, conv0_det, features_left)
        d6_smooth = self._forward_branch(self.branch_smooth, conv0_smooth, features_left)

        # Merge two branches guided again by the att0-derived mask,
        # then add a global skip from the shared conv0 features.
        out = mask * d6_det + (1.0 - mask) * d6_smooth
        out = F.relu(out + self.redir_conv0(conv0), inplace=True)

        return [out]
