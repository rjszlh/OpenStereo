import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .backbone_bifpn import BiFPNBackbone
from .aggregation import Aggregation
from .convnext_aggregation import ConvNeXtAggregation


class HAGVCostVolume(nn.Module):
    """
    Two-stage cost volume: coarse aggregation builds an attention prior over disparity,
    which modulates the fine cost volume before the main aggregation.
    """

    def __init__(self, max_disp, in_channels, agg_cls, agg_kwargs,
                 coarse_scale=2, gamma=1.0, sigma=2.0):
        super().__init__()
        self.max_disp = max_disp
        self.D = max_disp // 4
        self.D_coarse = max(self.D // coarse_scale, 1)
        self.coarse_scale = coarse_scale
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))

        self.pool = nn.AvgPool2d(kernel_size=coarse_scale, stride=coarse_scale, padding=0)
        self.coarse_agg = agg_cls(in_channels=self.D_coarse, **agg_kwargs)

    def _build_attention_from_disp(self, disp_coarse, up_size):
        # disp_coarse: [B, 1, Hc, Wc]
        disp_values = torch.arange(self.D, device=disp_coarse.device, dtype=disp_coarse.dtype)
        disp_values = disp_values.view(1, self.D, 1, 1)

        sigma = F.softplus(self.sigma) + 1e-3  # keep positive/avoid collapse
        prob = torch.exp(-0.5 * (disp_values - disp_coarse) ** 2 / (sigma ** 2))
        prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-6)

        prob_up = F.interpolate(prob, size=up_size, mode='bilinear', align_corners=False)
        prob_mean = prob_up.mean(dim=1, keepdim=True)
        gamma = F.softplus(self.gamma)
        attn = 1.0 + gamma * (prob_up - prob_mean)
        attn = torch.clamp(attn, min=0.5, max=1.5)
        return attn

    def forward(self, fine_feat_left, fine_feat_right, coarse_feat_left=None, coarse_feat_right=None,
                features_left=None):
        # coarse stage
        coarse_left = coarse_feat_left if coarse_feat_left is not None else self.pool(fine_feat_left)
        coarse_right = coarse_feat_right if coarse_feat_right is not None else self.pool(fine_feat_right)

        coarse_cost = correlation_volume(coarse_left, coarse_right, self.D_coarse)
        # reuse aggregation structure for coarse disparity estimation
        coarse_encoding = self.coarse_agg(coarse_cost, features_left or [])
        coarse_logits = coarse_encoding[0].reshape(
            coarse_encoding[0].size(0), self.D_coarse, coarse_cost.size(2), coarse_cost.size(3))
        coarse_prob = F.softmax(coarse_logits, dim=1)
        disp_coarse = disparity_regression(coarse_prob, self.D_coarse)  # [B, 1, Hc, Wc]

        attn = self._build_attention_from_disp(disp_coarse.detach(), up_size=fine_feat_left.shape[2:])

        # fine stage
        fine_cost = correlation_volume(fine_feat_left, fine_feat_right, self.D)
        cost_fine_hagv = fine_cost * attn + fine_cost  # residual gating to keep original cost

        return cost_fine_hagv, disp_coarse


class HAGVstereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        self.use_hagv = cfgs.get('USE_HAGV', True)
        self.lambda_coarse: float = float(cfgs.get('LAMBDA_COARSE', 0.3))
        self.hagv_gamma: float = float(cfgs.get('HAGV_GAMMA', 1.0))
        self.hagv_sigma: float = float(cfgs.get('HAGV_SIGMA', 2.0))
        self.hagv_coarse_scale: int = int(cfgs.get('HAGV_COARSE_SCALE', 2))
        self.D = self.max_disp // 4

        # backbobe
        backbone_name = cfgs.get('BACKCONE', 'MobileNetv2')
        if backbone_name == 'BiFPN':
            bifpn_layers = int(cfgs.get('BIFPN_LAYERS', 1))
            self.backbone = BiFPNBackbone(bifpn_layers=bifpn_layers)
        else:
            self.backbone = Backbone(backbone_name)

        # aggregation
        agg_type = cfgs.get('AGGREGATION_TYPE', None)
        if agg_type == 'ConvNeXt':
            agg_cls = ConvNeXtAggregation
        else:
            agg_cls = Aggregation
        self.cost_agg = agg_cls(in_channels=self.D,
                                left_att=self.left_att,
                                blocks=cfgs.AGGREGATION_BLOCKS,
                                expanse_ratio=cfgs.EXPANSE_RATIO,
                                backbone_channels=self.backbone.output_channels)

        if self.use_hagv:
            coarse_agg_kwargs = dict(
                left_att=False,
                blocks=cfgs.AGGREGATION_BLOCKS,
                expanse_ratio=cfgs.EXPANSE_RATIO,
                backbone_channels=self.backbone.output_channels,
            )
            self.cost_volume = HAGVCostVolume(
                max_disp=self.max_disp,
                in_channels=self.D,
                agg_cls=agg_cls,
                agg_kwargs=coarse_agg_kwargs,
                coarse_scale=self.hagv_coarse_scale,
                gamma=self.hagv_gamma,
                sigma=self.hagv_sigma,
            )

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        if self.use_hagv:
            gwc_volume, disp_coarse = self.cost_volume(
                fine_feat_left=features_left[0],
                fine_feat_right=features_right[0],
                coarse_feat_left=features_left[1],
                coarse_feat_right=features_right[1],
                features_left=[])
        else:
            gwc_volume = correlation_volume(features_left[0], features_right[0], self.D)
            disp_coarse = None

        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.D)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
            if self.use_hagv and disp_coarse is not None:
                result['disp_coarse'] = disp_coarse

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss_fine = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss_disp_4 = F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss = loss_fine + 0.3 * loss_disp_4

        loss_coarse = None
        if self.use_hagv and ('disp_coarse' in model_pred):
            disp_coarse = model_pred['disp_coarse']
            disp_coarse_up = F.interpolate(disp_coarse, disp_gt.shape[2:], mode='bilinear', align_corners=False)
            disp_coarse_up = disp_coarse_up * (4 * self.hagv_coarse_scale)

            loss_coarse = F.smooth_l1_loss(disp_coarse_up[mask], disp_gt[mask], reduction='mean')
            loss = loss + self.lambda_coarse * loss_coarse

        loss_info = {
            'scalar/train/loss_disp': loss.item(),
            'scalar/train/loss_disp_fine': loss_fine.item(),
            'scalar/train/loss_disp_4': loss_disp_4.item()
        }
        if loss_coarse is not None:
            loss_info['scalar/train/loss_disp_coarse'] = loss_coarse.item()

        return loss, loss_info
