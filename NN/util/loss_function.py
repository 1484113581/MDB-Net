# -*- coding:utf-8 -*-
# @Time      :2025/2/21 15:09
# @Author    :Chen

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# 足迹的损失
class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-6,is_boun=False):
        super().__init__()
        self.alpha = alpha  # Dice权重
        self.gamma = gamma  # Focal的难样本系数
        self.smooth = smooth
        self.is_boun = is_boun

    def forward(self, y_pred, y_true,boundary):
        y_pred = torch.sigmoid(y_pred)  # 二分类需sigmoid
        # --- Dice Loss ---
        intersection = (y_pred * y_true).sum(dim=(1, 2, 3))  # 按batch求和
        union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        # --- Focal Loss ---
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce)  # 概率调整项
        focal_loss = ((1 - pt) ** self.gamma * bce)

        # 组合损失
        if self.is_boun:
            boundary_mask = (boundary * 0.1) + 1  # 边界权重=3，非边界=1
            focal_loss = focal_loss * boundary_mask
            # 边界对齐损失
            boundary_loss = self._boundary_align_loss(y_pred, y_true)
            return 0.7*dice_loss + 0.2*focal_loss.mean() + 0.1*boundary_loss
            # return 0.5*dice_loss + 0.5*focal_loss.mean()
        else:
            # return self.alpha * dice_loss + (1 - self.alpha) * focal_loss.mean()
            return 0.5 * dice_loss + 0.5 * focal_loss.mean()

    def _boundary_align_loss(self, pred, gt):
        """基于距离变换的边界对齐损失"""
        gt_dist = self._compute_distance_transform(gt)
        pred_dist = self._compute_distance_transform(pred)
        return F.l1_loss(pred_dist, gt_dist)

    def _compute_distance_transform(self, x):
        """二值图距离变换"""
        x = (x > 0.5).float()
        return 1 - F.max_pool2d(1 - x, kernel_size=3, stride=1, padding=1)

# 高度的损失
class HeightLoss(nn.Module):
    def __init__(self, beta=1.0,is_height_weight = False):
        super().__init__()
        self.beta = beta  # Smooth L1的阈值参数
        self.is_height_weight = is_height_weight

    def forward(self, y_pred, y_true):
        loss_regression = F.smooth_l1_loss(y_pred, y_true, beta=self.beta,reduction='none')
        if self.is_height_weight:
            # 创建权重掩膜
            weight_map = torch.ones_like(y_true)
            weight_map = torch.where(y_true >= 0, 1, weight_map)
            weight_map = torch.where(y_true >= 10, 2, weight_map)
            weight_map = torch.where(y_true >= 20, 3, weight_map)
            weight_map = torch.where(y_true >= 30, 4, weight_map)
            loss_regression = loss_regression * weight_map
        return loss_regression.mean()


class MultiTaskWrapper(nn.Module):
    def __init__(self, seg_loss, height_loss):
        super().__init__()
        self.seg_loss = seg_loss
        self.height_loss = height_loss
        self.log_var_seg = nn.Parameter(torch.tensor(1.0))
        self.log_var_reg = nn.Parameter(torch.tensor(1.0))

    def forward(self, seg_pred, seg_true, height_pred, height_true,boundary):
        # 分割损失
        loss_seg = self.seg_loss(seg_pred, seg_true,boundary)
        # 高度损失
        loss_height = self.height_loss(height_pred, height_true)

        # # 自动学习权重（参考：https://arxiv.org/abs/1705.07115）
        # total_loss = (torch.exp(-self.log_var_seg) * loss_seg +
        #               torch.exp(-self.log_var_reg) * loss_height +
        #               0.5 * (self.log_var_seg + self.log_var_reg))
        total_loss = (1 / (2 * self.log_var_seg ** 2)) * loss_seg + (1 / (2 * self.log_var_reg ** 2)) * loss_height
        total_loss += torch.log(self.log_var_seg) + torch.log(self.log_var_reg)  # 正则化项
        return loss_seg,loss_height,total_loss

"""
is_height_weight：是否使用不同高度不同权重
is_boun：是否使用边界权重
"""
def building_loss_multi(bh_true, pf_true, bh_pred, pf_pred,is_height_weight=False,is_boun=False,boundary=None):
    # 初始化
    seg_criterion = SegmentationLoss(alpha=0.7, gamma=2.0,is_boun = is_boun)
    height_criterion = HeightLoss(beta=3,is_height_weight = is_height_weight)
    multi_criterion = MultiTaskWrapper(seg_criterion, height_criterion)
    loss = multi_criterion(pf_pred, pf_true, bh_pred, bh_true,boundary)
    return loss


if __name__ == "__main__":
    predicted_heights = torch.randint(0, 31, size=(4, 1, 128, 128), dtype=torch.float32)
    true_heights = torch.randint(0, 31, size=(4, 1, 128, 128), dtype=torch.float32)
    predicted_footprint = torch.tensor(np.random.rand(4, 1, 128, 128), dtype=torch.float32)
    true_footprint = torch.tensor(np.random.randint(0, 2, size=(4, 1, 128, 128)), dtype=torch.float32)
    boundary = torch.tensor(np.random.randint(0, 2, size=(4, 1, 128, 128)), dtype=torch.float32)
    print(building_loss_multi(true_heights,true_footprint,predicted_heights,predicted_footprint,True,True,boundary))