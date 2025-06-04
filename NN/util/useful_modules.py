# -*- coding:utf-8 -*-
# @Time      :2025/2/21 21:58
# @Author    :Chen

import torch
import torch.nn as nn
import torch.nn.functional as F


# 多尺度浅层融合模块（MSFF）
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.branches = nn.ModuleDict({
            'conv1x1': nn.Conv2d(in_channels, in_channels // 4, 1),
            'conv3x3': nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            'conv5x5': nn.Conv2d(in_channels, in_channels // 4, 5, padding=2),
            'conv7x7': nn.Conv2d(in_channels, in_channels // 4, 7, padding=3)
        })

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        # 特征融合后的1x1卷积
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 多尺度特征提取
        features = [branch(x) for branch in self.branches.values()]
        concated = torch.cat(features, dim=1)  # [B, in_channels, H, W]

        # 通道注意力加权
        channel_weights = self.channel_att(concated)
        weighted = concated * channel_weights

        # 特征融合与残差连接
        fused = self.fusion_conv(weighted)
        return x + fused  # 残差连接保持信息流


# 多尺度特征融合模块,多尺度通道空间注意力 CMSA (Channel-Multiscale Spatial Attention) 模块
class ChannelMultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        super().__init__()
        self.in_channels = in_channels
        # 多尺度卷积核
        # group=4，是分四组进行，可以减少参数量，也是正则化的一种形式
        # 四组不同尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, k, padding=k // 2, groups=4),
                nn.BatchNorm2d(in_channels // 4)
            ) for k in [1, 3, 5, 7]  # 卷积核大小分别为1,3,5,7
        ])

        # 通道重校准（借鉴SENet思想）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力生成
        self.spatial_att = nn.Sequential(
            nn.Conv2d(4, 1, 7, padding=3),  # 4个尺度
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        features = [conv(x) for conv in self.conv_layers]

        # 通道注意力重加权
        channel_weights = self.channel_att(x)  # [B,C,1,1]
        weighted_features = [f * channel_weights[:, i * self.in_channels // 4:(i + 1) * self.in_channels // 4]
                             for i, f in enumerate(features)]

        # 多尺度响应图生成
        scale_maps = [torch.mean(f, dim=1, keepdim=True) for f in weighted_features]
        combined = torch.cat(scale_maps, dim=1)  # [B,4,H,W]

        # 空间注意力生成
        spatial_mask = self.spatial_att(combined)  # [B,1,H,W]

        return x * spatial_mask

class ChannelMultiScaleSpatialAttention2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 多尺度卷积核
        # group=4，是分四组进行，可以减少参数量，也是正则化的一种形式
        # 四组不同尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, k, padding=k // 2, groups=4),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU()
            ) for k in [1, 3, 5, 7]  # 卷积核大小分别为1,3,5,7
        ])

        # 通道重校准（借鉴SENet思想）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力生成
        self.spatial_att = nn.Sequential(
            nn.Conv2d(4, 1, 7, padding=3),  # 4个尺度
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        features = [conv(x) for conv in self.conv_layers]

        # 通道注意力重加权
        channel_weights = self.channel_att(torch.cat(features, dim=1))  # [B,C,1,1]
        weighted_features = [f * channel_weights[:, i * self.in_channels // 4:(i + 1) * self.in_channels // 4]
                             for i, f in enumerate(features)]

        # 多尺度响应图生成
        scale_maps = [torch.mean(f, dim=1, keepdim=True) for f in weighted_features]
        combined = torch.cat(scale_maps, dim=1)  # [B,4,H,W]

        # 空间注意力生成
        spatial_mask = self.spatial_att(combined)  # [B,1,H,W]

        return x + x * spatial_mask



# 2. 跨模态通道注意力（Cross-Modal Channel Attention, CMCA）用于分支结合处
# 通过通道注意力实现SAR与光学特征的动态校准
class CrossModalChannelAttention1(nn.Module):
    def __init__(self, opt_channels, sar_channels):
        super().__init__()
        self.opt_gap = nn.AdaptiveAvgPool2d(1)
        self.sar_gap = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(opt_channels + sar_channels, (opt_channels + sar_channels) // 4),
            nn.ReLU(),
            nn.Linear((opt_channels + sar_channels) // 4, opt_channels + sar_channels),
            nn.Sigmoid()
        )

    def forward(self, opt_feat, sar_feat):
        # 通道统计量
        opt_stat = self.opt_gap(opt_feat).squeeze(-1).squeeze(-1)  # [B, C1]
        sar_stat = self.sar_gap(sar_feat).squeeze(-1).squeeze(-1)  # [B, C2]

        # 联合通道权重
        combined = torch.cat([opt_stat, sar_stat], dim=1)  # [B, C1+C2]
        channel_weights = self.channel_fc(combined)  # [B, C1+C2]
        opt_weight, sar_weight = torch.split(channel_weights,
                                             [opt_feat.size(1), sar_feat.size(1)],
                                             dim=1)

        # 应用权重
        opt_feat = opt_feat * opt_weight.unsqueeze(-1).unsqueeze(-1)
        sar_feat = sar_feat * sar_weight.unsqueeze(-1).unsqueeze(-1)
        return opt_feat, sar_feat

class CrossModalChannelAttention(nn.Module):
    def __init__(self, opt_channels, sar_channels):
        super().__init__()
        self.opt_gap = nn.AdaptiveAvgPool2d(1)
        self.sar_gap = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(opt_channels + sar_channels, (opt_channels + sar_channels) // 4),
            nn.ReLU(),
            nn.Linear((opt_channels + sar_channels) // 4, opt_channels + sar_channels),
            nn.Sigmoid()
        )

        # ----------------- 特征融合 -----------------
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(opt_channels + sar_channels, opt_channels, 3, padding=1),
            nn.BatchNorm2d(opt_channels),
            nn.ReLU(),
        )

    def forward(self, opt_feat, sar_feat):
        # 通道统计量
        opt_stat = self.opt_gap(opt_feat).squeeze(-1).squeeze(-1)  # [B, C1]
        sar_stat = self.sar_gap(sar_feat).squeeze(-1).squeeze(-1)  # [B, C2]

        # 联合通道权重
        combined = torch.cat([opt_stat, sar_stat], dim=1)  # [B, C1+C2]
        channel_weights = self.channel_fc(combined)  # [B, C1+C2]
        opt_weight, sar_weight = torch.split(channel_weights,
                                             [opt_feat.size(1), sar_feat.size(1)],
                                             dim=1)

        # 应用权重
        opt_feat = opt_feat * opt_weight.unsqueeze(-1).unsqueeze(-1)
        sar_feat = sar_feat * sar_weight.unsqueeze(-1).unsqueeze(-1)

        fused = self.fusion_conv(torch.cat([opt_feat, sar_feat], dim=1))

        return opt_feat+fused, sar_feat+fused


# 5. 自适应特征融合模块（Adaptive Feature Fusion, AFF）用于多尺度特征融合。
# 动态融合不同层次特征
class AdaptiveFeatureFusion1(nn.Module):
    def __init__(self, low_dim, high_dim):
        super().__init__()
        self.low_conv = nn.Conv2d(low_dim, high_dim, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(high_dim * 2, high_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(high_dim // 2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, low_feat, high_feat):
        # 统一维度
        low_feat = self.low_conv(low_feat)

        # 生成注意力图
        combined = torch.cat([low_feat, high_feat], dim=1)
        attn = self.attention(combined)  # [B,2,H,W]

        # 加权融合
        return torch.cat((attn[:, 0:1] * low_feat,attn[:, 1:2] * high_feat),dim=1)

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, low_dim, high_dim):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_dim, high_dim, 1),
            nn.BatchNorm2d(high_dim)
        )
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        # 通道交互增强模块
        self.channel_interact = nn.Sequential(
            nn.Linear(2 * high_dim, 4 * high_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * high_dim, 2 * high_dim),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        # 特征对齐
        low = self.low_proj(low_feat)

        # 通道统计量提取
        b, c, _, _ = low.size()
        gap_low = self.gap1(low).view(b, c)  # [B, C]
        gap_high = self.gap1(high_feat).view(b, c)

        # 通道交互权重生成
        combined = torch.cat([gap_low, gap_high], dim=-1)  # [B, 2C]
        attn = self.channel_interact(combined).view(b, 2, c)  # [B, 2, C]
        attn_low, attn_high = attn[:, 0], attn[:, 1]

        # 通道级特征调制
        return (torch.cat((attn_low[:, :, None, None] * low,attn_high[:, :, None, None] * high_feat),dim=1))


if __name__ == "__main__":
    data1 = torch.randint(0, 31, size=(8, 16, 128, 128), dtype=torch.float32)
    data2 = torch.randint(0, 31, size=(8, 32, 128, 128), dtype=torch.float32)
    # cmsa = MultiScaleSpatialAttention(in_channels=16)  # 以光学数据通道数为例
    aff = AdaptiveFeatureFusion(16,32)
    # msff = MultiScaleFusion(16,reduction_ratio=4)
    # cmca = CrossModalChannelAttention(32,32)
    # haf = LightHAF(32,32)
    output = aff(data1,data2)
    print(output.shape)
