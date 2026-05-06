# 训练主程序

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

from lamalocal.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    USE_GPU, RESUME_TRAINING, NUM_WORKERS,
    BCE_WEIGHT, DICE_WEIGHT, THRESHOLD
)
from lamalocal.dataset import get_dataloaders
from lamalocal.losses import CombinedLoss
from lamalocal.metrics import Metrics
from lamalocal.utils import save_model, load_model, get_lr_scheduler

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    :param model: 模型
    :param dataloader: 训练数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 平均损失、平均IoU、平均Dice
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    metrics = Metrics(threshold=THRESHOLD)
    
    progress_bar = tqdm(dataloader, desc="Training")
    for images, masks in progress_bar:
        # 移动数据到设备
        images = images.to(device)
        masks = masks.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, masks)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 计算指标
        batch_metrics = metrics.calculate_metrics(outputs, masks)
        
        # 累计损失和指标
        running_loss += loss.item() * images.size(0)
        running_iou += batch_metrics['IoU'] * images.size(0)
        running_dice += batch_metrics['Dice'] * images.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'IoU': batch_metrics['IoU'],
            'Dice': batch_metrics['Dice']
        })
    
    # 计算平均值
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice

def val_epoch(model, dataloader, criterion, device):
    """
    验证一个epoch
    :param model: 模型
    :param dataloader: 验证数据加载器
    :param criterion: 损失函数
    :param device: 设备
    :return: 平均损失、平均IoU、平均Dice
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    metrics = Metrics(threshold=THRESHOLD)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for images, masks in progress_bar:
            # 移动数据到设备
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 计算指标
            batch_metrics = metrics.calculate_metrics(outputs, masks)
            
            # 累计损失和指标
            running_loss += loss.item() * images.size(0)
            running_iou += batch_metrics['IoU'] * images.size(0)
            running_dice += batch_metrics['Dice'] * images.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'IoU': batch_metrics['IoU'],
                'Dice': batch_metrics['Dice']
            })
    
    # 计算平均值
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = running_iou / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice

def main():
    """
    主函数
    """
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    # 初始化损失函数
    criterion = CombinedLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 初始化学习率调度器
    scheduler = get_lr_scheduler(optimizer, T_max=EPOCHS)
    
    # 加载模型（如果需要）
    start_epoch = 0
    best_metric = 0.0
    if RESUME_TRAINING:
        model, optimizer, start_epoch, best_metric = load_model(model, optimizer)
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)
    
    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # 训练
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_iou, val_dice = val_epoch(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f"训练集 - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"验证集 - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # 保存模型
        current_metric = val_dice  # 使用Dice作为评估指标
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
        
        save_model(model, optimizer, epoch, best_metric, is_best)
    
    print(f"\n训练完成！最佳验证Dice: {best_metric:.4f}")

if __name__ == "__main__":
    main()
