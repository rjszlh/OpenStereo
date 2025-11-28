import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .backbone_bifpn import BiFPNBackbone
from .aggregation import Aggregation
from .aggregation_acir import AggregationACIR
from .ghost_aggregation import GhostAggregation
from .shuffle_aggregation import ShuffleAggregation
from .aggregation_multi_att import AggregationMultiAtt
from .aggregation_dual import DualBranchAggregation
from .aggregation_dual_att import DualAttentionAggregation
from .aggregation_edge_diffusion import EdgeDiffusionAggregation
from .disparity_sav import DisparitySpectralSelfAttentionVolume
from .convnext_aggregation import ConvNeXtAggregation



class RLightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbone
        backbone_type = cfgs.get('BACKCONE', 'GhostNet')
        bifpn_layers = cfgs.get('BIFPN_LAYERS', 1)
        if backbone_type == 'MobileNetv2_BiFPN':
            self.backbone = BiFPNBackbone(bifpn_layers=bifpn_layers)
        else:
            self.backbone = Backbone(backbone_type)

        # aggregation
        agg_type = cfgs.get('AGGREGATION_TYPE', 'Ghost')
        if agg_type == 'Ghost':
            agg_cls = GhostAggregation
        elif agg_type == 'Shuffle':
            agg_cls = ShuffleAggregation
        elif agg_type == 'ACIR':
            agg_cls = AggregationACIR
        elif agg_type == 'MultiAtt':
            agg_cls = AggregationMultiAtt
        elif agg_type == 'Dual':
            agg_cls = DualBranchAggregation
        elif agg_type == 'DualAtt':
            agg_cls = DualAttentionAggregation
        elif agg_type == 'EdgeDiffusion':
            agg_cls = EdgeDiffusionAggregation
        elif agg_type == 'ConvNeXt':
            agg_cls = ConvNeXtAggregation
        else:
            agg_cls = Aggregation

        ds_sav_cfg = cfgs.get('DS_SAV', None)
        self.ds_sav = None
        if ds_sav_cfg not in (None, False):
            ds_sav_cfg = ds_sav_cfg if hasattr(ds_sav_cfg, "get") else {}
            self.ds_sav = DisparitySpectralSelfAttentionVolume(
                disp_channels=self.max_disp // 4,
                embed_dim=ds_sav_cfg.get('EMBED_DIM', 32),
                num_heads=ds_sav_cfg.get('NUM_HEADS', 4),
                ff_hidden_dim=ds_sav_cfg.get('FF_HIDDEN_DIM', None),
                dropout=ds_sav_cfg.get('DROPOUT', 0.0),
                use_pos_encoding=ds_sav_cfg.get('USE_POS_ENCODING', True),
            )

        self.cost_agg = agg_cls(
            # correlation_volume 输出通道 = max_disp//4，避免 MAX_DISP 改变后与聚合模块不匹配
            in_channels=self.max_disp // 4,
            left_att=self.left_att,
            blocks=cfgs.AGGREGATION_BLOCKS,
            expanse_ratio=cfgs.EXPANSE_RATIO,
            backbone_channels=self.backbone.output_channels,
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

        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        if self.ds_sav is not None:
            gwc_volume = self.ds_sav(gwc_volume)  # disparity-wise self-attention refinement
        if getattr(self.cost_agg, "requires_right_features", False):
            encoding_volume = self.cost_agg(gwc_volume, features_left, features_right)
        else:
            encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

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

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
