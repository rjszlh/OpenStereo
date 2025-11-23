import torch.nn as nn
import torch.nn.functional as F

from .aggregation import MobileV2Residual


class AggregationMultiAtt(nn.Module):
    """
    Aggregation with lightweight, stage-specific attention:
      - shallow: simple edge / texture spatial attention
      - middle: simple local-structure spatial attention
      - deep: lightweight semantic channel attention
    Overall设计目标：在保留分阶段注意力思想的前提下，相比原 AttentionModule
    明显降低注意力部分的参数量和 FLOPs。
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(AggregationMultiAtt, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for _ in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                     for _ in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                     for _ in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            self.att0 = ShallowEdgeAttention(in_channels, backbone_channels[0])
            self.att2 = LocalStructureAttention(in_channels * 2, backbone_channels[1])
            self.att4 = SemanticChannelAttention(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        # shallow stage
        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])

        # middle stage
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])

        # deep stage
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])

        # upsampling and skip connections
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return [conv6]


class ShallowEdgeAttention(nn.Module):
    """
    Shallow stage: focus on edges and textures.
    极简设计：从浅层 backbone 特征生成单通道空间注意力图，直接对 cost 做门控。
    """

    def __init__(self, dim, img_feat_dim):
        super(ShallowEdgeAttention, self).__init__()
        # 单层 3x3 卷积生成 [B, 1, H, W] 的边缘/纹理注意力
        self.edge_att = nn.Sequential(
            nn.Conv2d(img_feat_dim, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, cost, img_feat):
        attn = self.edge_att(img_feat)  # [B, 1, H, W]
        # 轻量门控：1 + attn ∈ [1, 2]，既能增强，又不会抑制为 0
        return cost * (1.0 + attn)


class LocalStructureAttention(nn.Module):
    """
    Middle stage: local structural attention.
    使用一个带 dilation 的 3x3 生成单通道结构注意力，同样直接门控 cost。
    """

    def __init__(self, dim, img_feat_dim):
        super(LocalStructureAttention, self).__init__()
        self.struct_att = nn.Sequential(
            nn.Conv2d(img_feat_dim, 1, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, cost, img_feat):
        attn = self.struct_att(img_feat)  # [B, 1, H, W]
        return cost * (1.0 + attn)


class SemanticChannelAttention(nn.Module):
    """
    Deep stage: semantic channel attention.
    只做轻量级 SE-style 通道注意力，避免复杂的空间卷积。
    """

    def __init__(self, dim, img_feat_dim, reduction=4):
        super(SemanticChannelAttention, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(dim // reduction, 4)
        # 直接在 cost 的通道维度上做 SE，比从 img_feat_dim 投影到 dim 更省参数
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, cost, img_feat):  # img_feat 目前未显式使用，保留接口便于后续扩展
        b, c, _, _ = cost.size()
        pooled = self.global_pool(cost).view(b, c)
        ch_attn = self.mlp(pooled).view(b, c, 1, 1)  # [B, C, 1, 1]
        return cost * ch_attn

