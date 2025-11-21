import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import MobileV2Residual

class ShuffleAggregation(nn.Module):
    """
    ShuffleNetV2-based cost aggregation.
    Structure, channels and strides strictly follow Aggregation in aggregation.py,
    but MobileV2Residual blocks are replaced by ShuffleV2Residual.
    """
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super().__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [
            ShuffleV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)

        # keep downsampling identical to Aggregation (MobileV2Residual)
        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [
            ShuffleV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)

        # keep downsampling identical to Aggregation (MobileV2Residual)
        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [
            ShuffleV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
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

        # keep skip connections identical to Aggregation (MobileV2Residual)
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

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return [conv6]

def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleV2Residual(nn.Module):
    """
    ShuffleNetV2-style block adapted to match the interface of MobileV2Residual:
    (inp, oup, stride, expanse_ratio, dilation).
    For stride=1 we use the standard split/transform/shuffle design (inp == oup required).
    For stride=2 we use the two-branch downsampling design and allow arbitrary oup
    as long as it is divisible by 2.
    """

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dilation = dilation

        if self.stride == 1:
            assert inp == oup, "ShuffleV2Residual stride=1 requires inp == oup"
            assert inp % 2 == 0, "ShuffleV2Residual stride=1 requires even number of channels"
            branch_channels = inp // 2

            hidden_dim = int(branch_channels * expanse_ratio)

            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, branch_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        else:
            assert oup % 2 == 0, "ShuffleV2Residual stride=2 requires oup to be divisible by 2"
            branch_out = oup // 2
            hidden_dim = int(inp * expanse_ratio)

            # branch1: depthwise downsample then pointwise projection
            self.branch1 = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    3,
                    stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=inp,
                    bias=False,
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_out),
                nn.ReLU(inplace=True),
            )

            # branch2: pw -> dw (downsample) -> pw
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, branch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out2 = self.branch2(x2)
            out = torch.cat((x1, out2), dim=1)
            out = channel_shuffle(out, 2)
            return out
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out = torch.cat((out1, out2), dim=1)
            out = channel_shuffle(out, 2)
            return out


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
