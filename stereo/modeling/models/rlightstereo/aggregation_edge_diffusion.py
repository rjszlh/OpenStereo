import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import MobileV2Residual


class EdgeGuidance(nn.Module):
    """
    Predict edge strength and directional weights from image features.
    edge:   [B, 1, H, W] in [0, 1], controls how much diffusion is allowed.
    dirs:   [B, 4, H, W], softmax-normalized, weights of 4-neighbor directions.
    """

    def __init__(self, in_channels, hidden_channels=32, n_dirs=4):
        super().__init__()
        assert n_dirs == 4, "Current implementation assumes 4 directions (up, down, left, right)."
        self.n_dirs = n_dirs

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=True),
        )

        self.dir_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_dirs, 3, padding=1, bias=True),
        )

    def forward(self, guidance_feat):
        edge = torch.sigmoid(self.edge_conv(guidance_feat))
        dir_logits = self.dir_conv(guidance_feat)
        dir_weights = torch.softmax(dir_logits, dim=1)
        return edge, dir_weights


class DirectionalDiffusionStep(nn.Module):
    """
    Single anisotropic diffusion step over 4-neighborhood.
    cost_new = cost + step * sum_dir k_dir * (neighbor_dir - cost)
    """

    def __init__(self, init_step=0.5):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(float(init_step)))

    def forward(self, cost, edge, dir_weights):
        # cost: [B, C, H, W]
        # edge: [B, 1, H, W]
        # dir_weights: [B, 4, H, W]
        b, c, h, w = cost.shape

        cost_up = F.pad(cost, (0, 0, 1, 0))[:, :, :-1, :]
        cost_down = F.pad(cost, (0, 0, 0, 1))[:, :, 1:, :]
        cost_left = F.pad(cost, (1, 0, 0, 0))[:, :, :, :-1]
        cost_right = F.pad(cost, (0, 1, 0, 0))[:, :, :, 1:]

        neighbors = torch.stack([cost_up, cost_down, cost_left, cost_right], dim=1)  # [B, 4, C, H, W]
        center = cost.unsqueeze(1)  # [B, 1, C, H, W]

        k = edge * dir_weights  # [B, 4, H, W]
        k = k.unsqueeze(2)  # [B, 4, 1, H, W]

        diff = neighbors - center  # [B, 4, C, H, W]
        diffusion = (k * diff).sum(dim=1)  # [B, C, H, W]

        return cost + self.step_size * diffusion


class EdgeAwareMultiStepDiffusion(nn.Module):
    """
    Unrolled multi-step anisotropic diffusion with residual connection.
    """

    def __init__(self, cost_channels, guidance_channels, n_steps=3):
        super().__init__()
        self.guidance = EdgeGuidance(guidance_channels, hidden_channels=32, n_dirs=4)
        self.step = DirectionalDiffusionStep(init_step=0.5)
        self.n_steps = n_steps
        self.fuse = nn.Conv2d(cost_channels, cost_channels, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, cost, guidance_feat):
        # cost: [B, C, H, W]
        # guidance_feat: [B, G, H, W]
        shortcut = cost
        edge, dir_weights = self.guidance(guidance_feat)

        for _ in range(self.n_steps):
            cost = self.step(cost, edge, dir_weights)

        cost = self.fuse(cost)
        return shortcut + self.gamma * (cost - shortcut)


class EdgeDiffusionAggregation(nn.Module):
    """
    Cost aggregation with edge-aware anisotropic diffusion guided by image features.

    The backbone structure (downsample / upsample pattern) follows Aggregation
    in aggregation.py, but AttentionModule is replaced by EdgeAwareMultiStepDiffusion.
    """

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super().__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [
            MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [
            MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [
            MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
            for _ in range(blocks[2] - 1)
        ]
        self.conv4 = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 4,
                in_channels * 2,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 2,
                in_channels,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            # Use diffusion modules as edge-aware attention at three scales.
            self.diff0 = EdgeAwareMultiStepDiffusion(
                cost_channels=in_channels, guidance_channels=backbone_channels[0], n_steps=3
            )
            self.diff2 = EdgeAwareMultiStepDiffusion(
                cost_channels=in_channels * 2, guidance_channels=backbone_channels[1], n_steps=3
            )
            self.diff4 = EdgeAwareMultiStepDiffusion(
                cost_channels=in_channels * 4, guidance_channels=backbone_channels[2], n_steps=3
            )

    def forward(self, x, features_left):
        # x: correlation / cost volume features [B, C, H, W]
        # features_left: list of backbone features at multiple scales
        x = self.conv0(x)
        if self.left_att:
            x = self.diff0(x, features_left[0])

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.diff2(conv2, features_left[1])

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.diff4(conv4, features_left[2])

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return [conv6]

