import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import IMAGES_DIR, MASKS_DIR
from distill_data import InpaintTeacherPairs
from mobile_models import InpaintStudentNet


def total_variation(x):
    loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def inpaint_loss(pred, target, image, mask, valid_weight=0.1, tv_weight=0.01):
    hole_loss = (F.l1_loss(pred * mask, target * mask, reduction="sum") / mask.sum().clamp_min(1.0))
    valid_loss = F.l1_loss(pred * (1.0 - mask), target * (1.0 - mask))
    tv_loss = total_variation(pred * mask)
    return hole_loss + valid_weight * valid_loss + tv_weight * tv_loss


def run_epoch(model, loader, optimizer, args, device, train=True):
    model.train(train)
    total = 0.0
    desc = "Train" if train else "Val"

    for x, target, image, mask in tqdm(loader, desc=desc):
        x = x.to(device)
        target = target.to(device)
        image = image.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = inpaint_loss(pred, target, image, mask, args.valid_weight, args.tv_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


def save_checkpoint(path, model, optimizer, epoch, best_loss, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Distill LAMA inpainting outputs into a light mobile inpainting network.")
    parser.add_argument("--image-dir", default=IMAGES_DIR)
    parser.add_argument("--mask-dir", default=MASKS_DIR)
    parser.add_argument("--target-dir", required=True, help="Directory containing LAMA-cleaned teacher images.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = InpaintTeacherPairs(
        args.image_dir,
        args.mask_dir,
        args.target_dir,
        image_size=(args.image_size, args.image_size),
        augment=True,
    )
    val_dataset = InpaintTeacherPairs(
        args.image_dir,
        args.mask_dir,
        args.target_dir,
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

    model = InpaintStudentNet(in_channels=4, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_loss = float("inf")
    latest_path = os.path.join(args.out_dir, "mobile_inpaint_latest.pth")
    best_path = os.path.join(args.out_dir, "mobile_inpaint_best.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = run_epoch(model, train_loader, optimizer, args, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, args, device, train=False)
        scheduler.step()
        print(f"train loss={train_loss:.5f}")
        print(f"val   loss={val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_loss, args)
            print(f"saved best: {best_path}")
        save_checkpoint(latest_path, model, optimizer, epoch, best_loss, args)


if __name__ == "__main__":
    main()
