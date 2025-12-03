import torch
import torch.nn as nn
import torch.nn.functional as F


class DisparitySpectralSelfAttentionVolume(nn.Module):
    """
    Disparity-Spectral Self-Attention Volume (DS-SAV).

    轻量实现：在原始代价体 [B, D, H, W] 上，不直接对完整分辨率做 D×D 自注意力，
    而是先在视差和空间上各下采样一半，得到 [B, D/2, H/2, W/2]，在该尺度上做
    多头自注意力，然后再通过插值映射回原始尺度，保证最终输出仍为 [B, D, H, W]，
    且通道索引与视差 0～D-1 的对应关系保持不变。
    """

    def __init__(self, disp_channels, embed_dim=32, num_heads=4, ff_hidden_dim=None,
                 dropout=0.0, use_pos_encoding=True):
        super().__init__()
        # 原始视差通道数 D_full
        self.disp_channels = disp_channels
        # 注意力实际作用的视差通道数（在 coarse 尺度上为 D_full / 2）
        assert disp_channels % 2 == 0, "DS-SAV expects even number of disparity channels."
        self.coarse_disp_channels = disp_channels // 2

        self.use_pos_encoding = use_pos_encoding
        ff_hidden_dim = ff_hidden_dim or embed_dim * 2

        self.input_proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

        if self.use_pos_encoding:
            # 只为 coarse 视差通道 (D/2) 学习位置编码，保持视差索引顺序。
            scale = embed_dim ** -0.5
            self.disp_embed = nn.Parameter(
                torch.randn(self.coarse_disp_channels, embed_dim) * scale
            )
        else:
            self.register_parameter('disp_embed', None)

    def _attend_along_disparity(self, x):
        """
        在给定尺度上沿视差维做自注意力。

        Args:
            x: Tensor [B, Dc, Hc, Wc]
        Returns:
            Tensor [B, Dc, Hc, Wc]
        """
        b, d, h, w = x.shape

        # (D, B*H*W, 1)：每个像素的视差向量作为长度为 D 的序列。
        tokens = x.permute(1, 0, 2, 3).reshape(d, b * h * w, 1)
        tokens = self.input_proj(tokens)

        if self.use_pos_encoding and self.disp_embed is not None:
            tokens = tokens + self.disp_embed[:d].unsqueeze(1)

        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        tokens = tokens + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(tokens))
        tokens = tokens + self.dropout(ffn_out)

        updated = self.output_proj(tokens).squeeze(-1)  # (D, B*H*W)
        updated = updated.reshape(d, b, h, w).permute(1, 0, 2, 3).contiguous()
        return updated

    def forward(self, cost_volume):
        """
        Args:
            cost_volume: Tensor [B, D, H, W]，来自相关或其他代价构造。

        Returns:
            Tensor [B, D, H, W]，在 coarse 尺度经过自注意力强化后，再映射回原尺度。
            通道 i 仍对应视差 i。
        """
        b, d, h, w = cost_volume.shape
        assert d == self.disp_channels, f"Expected {self.disp_channels} disparity channels, got {d}"
        assert h % 2 == 0 and w % 2 == 0, "DS-SAV expects even spatial size for coarse downsampling."

        # 1) 视差通道下采样一半: [B, D, H, W] -> [B, D/2, H, W]
        #    这里简单用相邻两个视差 bin 的均值作为 coarse 视差 bin。
        x = cost_volume.view(b, self.coarse_disp_channels, 2, h, w).mean(dim=2)

        # 2) 空间下采样一半: [B, D/2, H, W] -> [B, D/2, H/2, W/2]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 3) 在 coarse 尺度 [B, D/2, H/2, W/2] 上做视差自注意力
        x_refined = self._attend_along_disparity(x)

        # 4) 空间上采样回原 H, W: [B, D/2, H/2, W/2] -> [B, D/2, H, W]
        x_up_spatial = F.interpolate(
            x_refined, size=(h, w), mode='bilinear', align_corners=False
        )

        # 5) 视差维上线性插值回 D: [B, D/2, H, W] -> [B, D, H, W]
        #    对每个 (h, w) 位置，视差向量是长度为 D/2 的 1D 序列：
        #    先 reshape 为 [B*H*W, 1, D/2]，再用 1D linear interpolate 到 D。
        x_1d = x_up_spatial.permute(0, 2, 3, 1).reshape(-1, 1, self.coarse_disp_channels)
        x_1d_up = F.interpolate(
            x_1d,
            size=d,
            mode='linear',
            align_corners=True
        )  # [B*H*W, 1, D]
        x_up_disp = x_1d_up.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        # 6) 与原始代价体做残差融合：既保留基础匹配，又注入粗尺度视差关系建模。
        updated_cost = cost_volume + x_up_disp
        return updated_cost
