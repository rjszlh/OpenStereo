# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import AttentionModule


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class GlobalResponseNorm(nn.Module):
    """GRN from ConvNeXt V2: normalizes channel responses before projection."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expanse_ratio,
        dilation=1,
        layer_scale_init_value=1e-6,
        use_grn=True,
    ):
        super().__init__()
        assert stride in [1, 2]

        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(inp * expanse_ratio)

        self.dwconv = nn.Conv2d(
            inp,
            inp,
            kernel_size=7,
            stride=stride,
            padding=3 * dilation,
            dilation=dilation,
            groups=inp,
            bias=True,
        )
        self.norm = LayerNorm2d(inp)

        self.pwconv1 = nn.Conv2d(inp, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(hidden_dim) if use_grn else nn.Identity()
        self.pwconv2 = nn.Conv2d(hidden_dim, oup, kernel_size=1)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(oup))
        else:
            self.gamma = None

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        if self.use_res_connect:
            return shortcut + x
        return x


class ConvNeXtAggregation(nn.Module):
    """
    ConvNeXt-style cost aggregation.
    Channel sizes, strides and outputs mirror Aggregation in aggregation.py.
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels, use_grn=False):
        super().__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio
        self.use_grn = use_grn

        conv0 = [
            ConvNeXtResidual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn)
            for _ in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = ConvNeXtResidual(
            in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
        )
        conv2_add = [
            ConvNeXtResidual(
                in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
            )
            for _ in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = ConvNeXtResidual(
            in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
        )
        conv4_add = [
            ConvNeXtResidual(
                in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
            )
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

        self.redir1 = ConvNeXtResidual(
            in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
        )
        self.redir2 = ConvNeXtResidual(
            in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio, use_grn=self.use_grn
        )

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
