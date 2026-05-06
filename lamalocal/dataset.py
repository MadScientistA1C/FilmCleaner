# 数据加载和处理模块

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from lamalocal.config import IMAGES_DIR, MASKS_DIR, IMAGE_SIZE, TRAIN_VAL_SPLIT

class FilmDustDataset(Dataset):
    def __init__(self, image_files, mask_files, augment=False):
        """
        初始化数据集
        :param image_files: 图像文件路径列表
        :param mask_files: 掩码文件路径列表
        :param augment: 是否进行数据增强
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.augment = augment
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),  # 转换为[0,1]范围的tensor
        ])
        
        # 掩码变换
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # 转换为[0,1]范围的tensor
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 读取掩码
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图
        
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # 随机90度旋转
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
            
            # 随机放大缩小
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                new_size = (int(IMAGE_SIZE[0] * scale), int(IMAGE_SIZE[1] * scale))
                # 确保新尺寸至少等于目标尺寸
                new_size = (max(new_size[0], IMAGE_SIZE[0]), max(new_size[1], IMAGE_SIZE[1]))
                image = transforms.functional.resize(image, new_size)
                mask = transforms.functional.resize(mask, new_size, interpolation=transforms.InterpolationMode.NEAREST)
                # 随机裁剪回原始大小
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=IMAGE_SIZE)
                image = transforms.functional.crop(image, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)
        
        # 应用变换
        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        # 确保掩码是二值的（0或1）
        mask = (mask > 0.5).float()
        
        return image, mask

def get_dataloaders(batch_size, num_workers):
    """
    获取训练和验证数据加载器
    :param batch_size: 批次大小
    :param num_workers: 数据加载线程数
    :return: 训练数据加载器和验证数据加载器
    """
    # 获取所有图像和掩码文件
    image_files = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
    mask_files = sorted([os.path.join(MASKS_DIR, f) for f in os.listdir(MASKS_DIR) if f.endswith('_mask.png')])
    
    # 确保图像和掩码数量一致
    assert len(image_files) == len(mask_files), "图像和掩码数量不一致"
    
    # 随机打乱数据
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files, mask_files = zip(*combined)
    
    # 划分训练集和验证集
    split_idx = int(len(image_files) * TRAIN_VAL_SPLIT)
    train_image_files = image_files[:split_idx]
    train_mask_files = mask_files[:split_idx]
    val_image_files = image_files[split_idx:]
    val_mask_files = mask_files[split_idx:]
    
    # 创建数据集
    train_dataset = FilmDustDataset(train_image_files, train_mask_files, augment=True)
    val_dataset = FilmDustDataset(val_image_files, val_mask_files, augment=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader
