# 工具函数模块

import torch
import os
from lamalocal.config import BEST_MODEL_PATH, LATEST_MODEL_PATH

def save_model(model, optimizer, epoch, best_metric, is_best=False):
    """
    保存模型
    :param model: 模型
    :param optimizer: 优化器
    :param epoch: 当前 epoch
    :param best_metric: 最佳指标值
    :param is_best: 是否为最佳模型
    """
    # 保存最新模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }, LATEST_MODEL_PATH)
    
    # 如果是最佳模型，保存到最佳模型路径
    if is_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric
        }, BEST_MODEL_PATH)
        print(f"最佳模型已保存到: {BEST_MODEL_PATH}")

def load_model(model, optimizer):
    """
    加载模型
    :param model: 模型
    :param optimizer: 优化器
    :return: 加载的模型、优化器、起始epoch、最佳指标
    """
    if os.path.exists(LATEST_MODEL_PATH):
        checkpoint = torch.load(LATEST_MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        print(f"从 {LATEST_MODEL_PATH} 加载模型，起始epoch: {start_epoch}")
        return model, optimizer, start_epoch, best_metric
    else:
        print("未找到模型文件，从初始状态开始训练")
        return model, optimizer, 0, 0.0

def get_lr_scheduler(optimizer, T_max=100):
    """
    获取学习率调度器
    :param optimizer: 优化器
    :param T_max: 最大迭代次数
    :return: 学习率调度器
    """
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    return scheduler
