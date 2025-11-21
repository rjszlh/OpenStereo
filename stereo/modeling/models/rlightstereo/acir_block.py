import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECALayer(nn.Module):
    """Efficient Channel Attention module implemented with 1D convolution."""

    def __init__(self, channels, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = self._get_default_kernel_size(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * y

    @staticmethod
    def _get_default_kernel_size(channels, gamma=2, b=1):
        t = int(abs((math.log2(channels + 1) + b) / gamma))
        k = t if t % 2 else t + 1
        return max(k, 3)


class ACIRBlockECA(nn.Module):
    """
    MobileNetV2 style bottleneck augmented with ACNet-style depthwise branches and ECA attention.
    The depthwise stage can be re-parameterized into a single 3x3 depthwise convolution for inference.
    """

    def __init__(self, in_channels, out_channels, stride=1, expanse_ratio=4,
                 deploy=False, eca_kernel_size=None, use_eca=True):
        super().__init__()
        self.stride = stride
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = int(in_channels * expanse_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_eca = use_eca

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.dw_act = nn.ReLU6(inplace=True)
        if deploy:
            self.dw_reparam = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                3,
                stride=stride,
                padding=1,
                groups=self.hidden_dim,
                bias=True
            )
        else:
            self.dw_3x3 = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                3,
                stride=stride,
                padding=1,
                groups=self.hidden_dim,
                bias=False
            )
            self.dw_bn3x3 = nn.BatchNorm2d(self.hidden_dim)

            self.dw_3x1 = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                (3, 1),
                stride=stride,
                padding=(1, 0),
                groups=self.hidden_dim,
                bias=False
            )
            self.dw_bn3x1 = nn.BatchNorm2d(self.hidden_dim)

            self.dw_1x3 = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                (1, 3),
                stride=stride,
                padding=(0, 1),
                groups=self.hidden_dim,
                bias=False
            )
            self.dw_bn1x3 = nn.BatchNorm2d(self.hidden_dim)

        self.project_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # When use_eca is False, ECA is replaced by identity to keep
        # behavior and tensor shapes unchanged without extra conditionals.
        self.eca = ECALayer(out_channels, eca_kernel_size) if self.use_eca else nn.Identity()

    def forward(self, x):
        shortcut = x
        out = self.expand_conv(x)
        out = self._depthwise_forward(out)
        out = self.project_conv(out)
        out = self.eca(out)
        if self.use_residual:
            out = out + shortcut
        return out

    def _depthwise_forward(self, x):
        if self.deploy:
            out = self.dw_reparam(x)
        else:
            out = self.dw_bn3x3(self.dw_3x3(x))
            out = out + self.dw_bn3x1(self.dw_3x1(x))
            out = out + self.dw_bn1x3(self.dw_1x3(x))
        return self.dw_act(out)

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.dw_reparam.weight, self.dw_reparam.bias

        kernel3x3, bias3x3 = self._fuse_conv_bn(self.dw_3x3, self.dw_bn3x3)
        kernel3x1, bias3x1 = self._fuse_conv_bn(self.dw_3x1, self.dw_bn3x1)
        kernel1x3, bias1x3 = self._fuse_conv_bn(self.dw_1x3, self.dw_bn1x3)

        kernel3x1 = F.pad(kernel3x1, [1, 1, 0, 0])
        kernel1x3 = F.pad(kernel1x3, [0, 0, 1, 1])

        kernel = kernel3x3 + kernel3x1 + kernel1x3
        bias = bias3x3 + bias3x1 + bias1x3
        return kernel, bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dw_reparam = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            3,
            stride=self.stride,
            padding=1,
            groups=self.hidden_dim,
            bias=True
        )
        self.dw_reparam.weight.data = kernel
        self.dw_reparam.bias.data = bias

        del self.dw_3x3
        del self.dw_bn3x3
        del self.dw_3x1
        del self.dw_bn3x1
        del self.dw_1x3
        del self.dw_bn1x3

        self.deploy = True

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        if conv is None or bn is None:
            return 0, 0
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias
