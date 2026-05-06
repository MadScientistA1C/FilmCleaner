import argparse
import os

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lamalocal.config import IMAGES_DIR, MASKS_DIR
from lamalocal.distill_data import FilmSegmentationPairs
from lamalocal.losses import CombinedLoss
from lamalocal.metrics import Metrics
from lamalocal.mobile_models import MobileUNetV3


def load_teacher(path, device):
    teacher = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    teacher.load_state_dict(state)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def run_epoch(model, teacher, loader, optimizer, criterion, args, device, train=True):
    model.train(train)
    metrics = Metrics(threshold=args.threshold)
    bce_soft = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    desc = "Train" if train else "Val"

    for images, masks in tqdm(loader, desc=desc):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            student_logits = model(images)
            loss = criterion(student_logits, masks)

            if teacher is not None and args.distill_weight > 0:
                with torch.no_grad():
                    teacher_probs = torch.sigmoid(teacher(images))
                soft_loss = bce_soft(student_logits, teacher_probs)
                loss = loss + args.distill_weight * soft_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch = images.size(0)
        batch_metrics = metrics.calculate_metrics(student_logits.detach(), masks)
        total_loss += loss.item() * batch
        total_iou += batch_metrics["IoU"] * batch
        total_dice += batch_metrics["Dice"] * batch

    count = len(loader.dataset)
    return total_loss / count, total_iou / count, total_dice / count


def save_checkpoint(path, model, optimizer, epoch, best_dice, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Distill DeepLabV3+ dust masks into MobileNetV3-UNet.")
    parser.add_argument("--image-dir", default=IMAGES_DIR)
    parser.add_argument("--mask-dir", default=MASKS_DIR)
    parser.add_argument("--teacher", default="checkpoints/best_model.pth")
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--encoder", choices=["small", "large"], default="small")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FilmSegmentationPairs(
        args.image_dir,
        args.mask_dir,
        image_size=(args.image_size, args.image_size),
        augment=True,
    )
    val_dataset = FilmSegmentationPairs(
        args.image_dir,
        args.mask_dir,
        image_size=(args.image_size, args.image_size),
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    split = int(len(indices) * args.train_ratio)
    train_set = Subset(train_dataset, indices[:split])
    val_set = Subset(val_dataset, indices[split:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = MobileUNetV3(encoder_name=args.encoder, in_channels=3, classes=1).to(device)
    teacher = None if args.no_teacher else load_teacher(args.teacher, device)
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_dice = 0.0
    latest_path = os.path.join(args.out_dir, "mobile_seg_latest.pth")
    best_path = os.path.join(args.out_dir, "mobile_seg_best.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_iou, train_dice = run_epoch(model, teacher, train_loader, optimizer, criterion, args, device, train=True)
        val_loss, val_iou, val_dice = run_epoch(model, teacher, val_loader, optimizer, criterion, args, device, train=False)
        scheduler.step()

        print(f"train loss={train_loss:.4f} iou={train_iou:.4f} dice={train_dice:.4f}")
        print(f"val   loss={val_loss:.4f} iou={val_iou:.4f} dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(best_path, model, optimizer, epoch, best_dice, args)
            print(f"saved best: {best_path}")
        save_checkpoint(latest_path, model, optimizer, epoch, best_dice, args)


if __name__ == "__main__":
    main()
