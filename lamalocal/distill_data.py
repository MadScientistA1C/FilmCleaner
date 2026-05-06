import os
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def mask_name_for(image_path):
    return f"{image_path.stem}_mask.png"


def default_target_candidates(image_path):
    return [
        image_path.name,
        f"{image_path.stem}_lama{image_path.suffix}",
        f"{image_path.stem}_target{image_path.suffix}",
        f"{image_path.stem}_clean{image_path.suffix}",
        f"{image_path.stem}.png",
    ]


class FilmSegmentationPairs(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.image_files = []
        self.mask_files = []

        for image_path in list_images(self.image_dir):
            mask_path = self.mask_dir / mask_name_for(image_path)
            if mask_path.exists():
                self.image_files.append(image_path)
                self.mask_files.append(mask_path)

        if not self.image_files:
            raise FileNotFoundError(f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}")

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")
        image, mask = paired_augment(image, mask, self.augment)
        return self.image_transform(image), (self.mask_transform(mask) > 0.5).float()


class InpaintTeacherPairs(Dataset):
    def __init__(self, image_dir, mask_dir, target_dir, image_size=(512, 512), augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.target_dir = Path(target_dir)
        self.augment = augment
        self.samples = []

        for image_path in list_images(self.image_dir):
            mask_path = self.mask_dir / mask_name_for(image_path)
            if not mask_path.exists():
                continue
            target_path = None
            for candidate in default_target_candidates(image_path):
                path = self.target_dir / candidate
                if path.exists():
                    target_path = path
                    break
            if target_path is not None:
                self.samples.append((image_path, mask_path, target_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask/target triplets found in {self.image_dir}, {self.mask_dir}, {self.target_dir}"
            )

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, target_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        target = Image.open(target_path).convert("RGB")
        image, mask, target = paired_augment(image, mask, self.augment, target)
        image = self.image_transform(image)
        mask = (self.mask_transform(mask) > 0.5).float()
        target = self.image_transform(target)
        return torch.cat([image, mask], dim=0), target, image, mask


def paired_augment(image, mask, augment=False, target=None):
    if not augment:
        return (image, mask, target) if target is not None else (image, mask)

    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
        if target is not None:
            target = transforms.functional.hflip(target)
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
        if target is not None:
            target = transforms.functional.vflip(target)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        if target is not None:
            target = transforms.functional.rotate(target, angle)

    return (image, mask, target) if target is not None else (image, mask)


def split_dataset(dataset, train_ratio=0.8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    return torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
