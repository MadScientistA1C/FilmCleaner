# 损失函数模块

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Dice损失函数
        :param smooth: 平滑参数，避免除零
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        计算Dice损失
        :param inputs: 模型输出（未经过sigmoid）
        :param targets: 真实标签
        :return: Dice损失值
        """
        # 应用sigmoid激活
        inputs = torch.sigmoid(inputs)
        
        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回1 - Dice系数作为损失
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        """
        组合损失函数（BCEWithLogitsLoss + DiceLoss）
        :param bce_weight: BCE损失权重
        :param dice_weight: Dice损失权重
        """
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        """
        计算组合损失
        :param inputs: 模型输出（未经过sigmoid）
        :param targets: 真实标签
        :return: 组合损失值
        """
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice