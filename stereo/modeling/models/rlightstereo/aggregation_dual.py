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

        # Shared conv0 on cost volume before branching.
        # Keep channels as in_channels and spatial size H/4 x W/4.
        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        # Encoder for detail branch
        self.conv1_det = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_det = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                     for _ in range(blocks[1] - 1)]
        self.conv2_det = nn.Sequential(*conv2_det)
        self.conv3_det = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_det = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                     for _ in range(blocks[2] - 1)]
        self.conv4_det = nn.Sequential(*conv4_det)

        # Decoder for detail branch
        self.deconv5_det = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )
        self.deconv6_det = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Encoder for smooth branch (same structure, independent params)
        self.conv1_smooth = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_smooth = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                        for _ in range(blocks[1] - 1)]
        self.conv2_smooth = nn.Sequential(*conv2_smooth)
        self.conv3_smooth = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_smooth = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                        for _ in range(blocks[2] - 1)]
        self.conv4_smooth = nn.Sequential(*conv4_smooth)

        # Decoder for smooth branch
        self.deconv5_smooth = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )
        self.deconv6_smooth = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

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
            mask = conv0.new_full((conv0.size(0), 1, conv0.size(2), conv0.size(3)), 0.5)

        # Split into detail and smooth branches.
        conv0_det = conv0 * mask
        conv0_smooth = conv0 * (1.0 - mask)

        # Detail branch encoder.
        # Use conv0_det directly as the first encoder stage to avoid
        # redundant conv blocks; the branch starts from conv1_det.
        b0_det = conv0_det
        b1_det = self.conv1_det(b0_det)
        b2_det = self.conv2_det(b1_det)
        if self.left_att:
            b2_det = self.att2(b2_det, features_left[1])
        b3_det = self.conv3_det(b2_det)
        b4_det = self.conv4_det(b3_det)
        if self.left_att:
            b4_det = self.att4(b4_det, features_left[2])

        # Smooth branch encoder.
        # Same for the smooth branch: start encoding from conv1_smooth.
        b0_smooth = conv0_smooth
        b1_smooth = self.conv1_smooth(b0_smooth)
        b2_smooth = self.conv2_smooth(b1_smooth)
        if self.left_att:
            b2_smooth = self.att2(b2_smooth, features_left[1])
        b3_smooth = self.conv3_smooth(b2_smooth)
        b4_smooth = self.conv4_smooth(b3_smooth)
        if self.left_att:
            b4_smooth = self.att4(b4_smooth, features_left[2])

        # Detail branch decoder (no per-branch skip).
        d5_det = F.relu(self.deconv5_det(b4_det), inplace=True)
        d6_det = F.relu(self.deconv6_det(d5_det), inplace=True)

        # Smooth branch decoder (no per-branch skip).
        d5_smooth = F.relu(self.deconv5_smooth(b4_smooth), inplace=True)
        d6_smooth = F.relu(self.deconv6_smooth(d5_smooth), inplace=True)

        # Merge two branches guided again by the att0-derived mask,
        # then add a global skip from the shared conv0 features.
        out = mask * d6_det + (1.0 - mask) * d6_smooth
        out = F.relu(out + self.redir_conv0(conv0), inplace=True)

        return [out]
