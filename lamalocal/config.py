# 配置文件，管理所有参数

import os

# 数据路径
DATA_ROOT = r"G:\project\LAMALocal\Dataset\FakeFilm\patches"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
MASKS_DIR = os.path.join(DATA_ROOT, "masks")

# 模型配置
MODEL_NAME = "DeepLabV3Plus"
BACKBONE = "resnet18"

# 训练参数
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 数据处理参数
IMAGE_SIZE = (512, 512)  # 图像大小
TRAIN_VAL_SPLIT = 0.8  # 训练集比例

# 增强参数
TRAIN_AUGMENTATION = True
VAL_AUGMENTATION = False

# 损失函数权重
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# 保存路径
CHECKPOINT_DIR = os.path.join("checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latest_model.pth")

# 其他参数
USE_GPU = True
RESUME_TRAINING = False
THRESHOLD = 0.5  # 二值化阈值
NUM_WORKERS = 4  # DataLoader workers

# 确保检查点目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)