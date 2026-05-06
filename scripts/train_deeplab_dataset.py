import argparse
import os

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lamalocal.distill_data import FilmSegmentationPairs
from lamalocal.losses import CombinedLoss
from lamalocal.metrics import Metrics
from lamalocal.utils import get_lr_scheduler


def run_epoch(model, loader, criterion, optimizer, metrics, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    desc = "Training" if train else "Validation"

    for images, masks in tqdm(loader, desc=desc):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        batch_metrics = metrics.calculate_metrics(outputs.detach(), masks)
        total_loss += loss.item() * batch_size
        total_iou += batch_metrics["IoU"] * batch_size
        total_dice += batch_metrics["Dice"] * batch_size

    count = len(loader.dataset)
    return total_loss / count, total_iou / count, total_dice / count


def save_checkpoint(path, model, optimizer, epoch, best_metric, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on a selected fake-film dataset.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, T_max=args.epochs)
    metrics = Metrics(threshold=args.threshold)

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", checkpoint.get("best_dice", 0.0))
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_metric={best_metric:.4f}")

    latest_path = f"{args.output_prefix}_latest.pth"
    best_path = f"{args.output_prefix}_best.pth"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion, optimizer, metrics, device, train=True)
        val_loss, val_iou, val_dice = run_epoch(model, val_loader, criterion, optimizer, metrics, device, train=False)
        scheduler.step()

        print(f"train loss={train_loss:.4f} iou={train_iou:.4f} dice={train_dice:.4f}")
        print(f"val   loss={val_loss:.4f} iou={val_iou:.4f} dice={val_dice:.4f}")

        if val_dice > best_metric:
            best_metric = val_dice
            save_checkpoint(best_path, model, optimizer, epoch, best_metric, args)
            print(f"saved best: {best_path}")
        save_checkpoint(latest_path, model, optimizer, epoch, best_metric, args)

    print(f"Done. Best Dice: {best_metric:.4f}")


if __name__ == "__main__":
    main()
