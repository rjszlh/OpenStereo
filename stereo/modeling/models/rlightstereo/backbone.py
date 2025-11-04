# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming
import timm
import torch
import torch.nn as nn

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Backbone(nn.Module):
    def __init__(self, backbone='MobileNetv2'):
        super().__init__()
        self.adapter_1_4 = None
        self.adapter_1_8 = None
        self.adapter_1_16 = None

        if backbone == 'MobileNetv2':
            model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
            channels = [160, 96, 32, 24]
            self.block0 = model.blocks[0]
            self.block1 = model.blocks[1]
            self.block2 = model.blocks[2]
            self.block3 = model.blocks[3:5]
            self.block4 = model.blocks[5]
        elif backbone == 'EfficientNetv2':
            model = timm.create_model('efficientnetv2_rw_s', pretrained=True, features_only=True)
            channels = [272, 160, 64, 48]
            self.block0 = model.blocks[0]
            self.block1 = model.blocks[1]
            self.block2 = model.blocks[2]
            self.block3 = model.blocks[3:5]
            self.block4 = model.blocks[5]
        elif backbone == 'GhostNet':
            model = timm.create_model('ghostnet_100', pretrained=True, features_only=True)
            channels = [160, 96, 32, 24]
            self.block0 = model.blocks_0
            self.block1 = model.blocks_1
            # blocks_2 keeps the 1/4 resolution while blocks_3 performs the stride-2
            # transition to 1/8. Group them so the stage boundaries stay aligned with
            # the MobileNet/EfficientNet variants.
            self.block2 = nn.Sequential(model.blocks_2, model.blocks_3)
            # Likewise, blocks_4 preserves 1/8 and blocks_5 reduces to 1/16.
            self.block3 = nn.Sequential(model.blocks_4, model.blocks_5)
            # Final downsampling happens across blocks_6 and blocks_7 to reach 1/32.
            self.block4 = nn.Sequential(model.blocks_6, model.blocks_7)

            # GhostNet exposes 40/80-channel tensors at 1/8 and 1/16 resolution, while
            # the rest of the pipeline expects the MobileNet-sized 32/96 tensors.
            # Use 1x1 adapters to realign channel widths without changing spatial sizes.
            self.adapter_1_8 = BasicConv2d(40, channels[2], kernel_size=1,
                                           norm_layer=nn.BatchNorm2d)
            self.adapter_1_16 = BasicConv2d(80, channels[1], kernel_size=1,
                                            norm_layer=nn.BatchNorm2d)
        else:
            raise NotImplementedError

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])

        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)

        self.output_channels = channels[::-1]

    def forward(self, images):
        c1 = self.act1(self.bn1(self.conv_stem(images)))  # [bz, 32, H/2, W/2]
        c1 = self.block0(c1)  # [bz, 16, H/2, W/2]
        c2 = self.block1(c1)  # [bz, 24, H/4, W/4]
        c3 = self.block2(c2)  # [bz, 32, H/8, W/8]
        c4 = self.block3(c3)  # [bz, 96, H/16, W/16]
        c5 = self.block4(c4)  # [bz, 160, H/32, W/32]

        if self.adapter_1_4 is not None:
            c2 = self.adapter_1_4(c2)
        if self.adapter_1_8 is not None:
            c3 = self.adapter_1_8(c3)
        if self.adapter_1_16 is not None:
            c4 = self.adapter_1_16(c4)

        p4 = self.fpn_layer4(c5, c4)  # [bz, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3)  # [bz, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2)  # [bz, 24, H/4, W/4]
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
