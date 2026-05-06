# 评估指标模块

import torch

class Metrics:
    def __init__(self, threshold=0.5):
        """
        评估指标计算类
        :param threshold: 二值化阈值
        """
        self.threshold = threshold
    
    def iou(self, inputs, targets):
        """
        计算IoU（交并比）
        :param inputs: 模型输出（未经过sigmoid）
        :param targets: 真实标签
        :return: IoU值
        """
        # 应用sigmoid激活并二值化
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        
        # 避免除零
        if union == 0:
            return 1.0
        
        # 计算IoU
        iou = intersection / union
        return iou.item()
    
    def dice(self, inputs, targets):
        """
        计算Dice系数
        :param inputs: 模型输出（未经过sigmoid）
        :param targets: 真实标签
        :return: Dice系数值
        """
        # 应用sigmoid激活并二值化
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # 避免除零
        if union == 0:
            return 1.0
        
        # 计算Dice系数
        dice = (2. * intersection) / union
        return dice.item()
    
    def calculate_metrics(self, inputs, targets):
        """
        计算所有指标
        :param inputs: 模型输出（未经过sigmoid）
        :param targets: 真实标签
        :return: 包含IoU和Dice的字典
        """
        iou = self.iou(inputs, targets)
        dice = self.dice(inputs, targets)
        return {
            'IoU': iou,
            'Dice': dice
        }