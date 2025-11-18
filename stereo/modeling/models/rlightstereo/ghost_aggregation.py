import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostAggregation(nn.Module):
    """
    GhostNet-based cost aggregation.
    Structure, channels and strides strictly follow Aggregation in aggregation.py,
    but MobileV2Residual blocks are replaced by GhostV2Residual.
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super().__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [
            GhostV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = GhostV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [
            GhostV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = GhostV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [
            GhostV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[2] - 1)
        ]
        self.conv4 = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.redir1 = GhostV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = GhostV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

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

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return [conv6]

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super().__init__()
        assert ratio >= 1
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        padding = (kernel_size - 1) // 2
        dw_padding = (dw_kernel_size - 1) // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_kernel_size,
                1,
                dw_padding,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x_primary = self.primary_conv(x)
        x_cheap = self.cheap_operation(x_primary)
        out = torch.cat([x_primary, x_cheap], dim=1)
        return out[:, :self.oup, :, :]


class GhostV2Residual(nn.Module):
    """
    GhostNet-style variant of MobileNetV2 inverted residual block.
    Channel sizes, stride and residual connection follow MobileV2Residual.
    """

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1, ghost_ratio=2):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # Expansion with GhostModule (keeps output channels = hidden_dim)
        self.ghost_exp = GhostModule(
            inp,
            hidden_dim,
            kernel_size=1,
            ratio=ghost_ratio,
            dw_kernel_size=3,
            stride=1,
            relu=True,
        )

        # Depthwise conv: keep the same DW/downsample behavior as MobileV2Residual
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                pad,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        # Projection with GhostModule (linear, no activation, output channels = oup)
        self.ghost_proj = GhostModule(
            hidden_dim,
            oup,
            kernel_size=1,
            ratio=ghost_ratio,
            dw_kernel_size=3,
            stride=1,
            relu=False,
        )

    def forward(self, x):
        feat = self.ghost_exp(x)
        feat = self.dwconv(feat)
        feat = self.ghost_proj(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat

class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost

