import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LOSSLESS_EXTS = {".png", ".tif", ".tiff", ".bmp"}


def list_inputs(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_deeplab(checkpoint_path, device):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def enhance_patch_for_mask(patch, mode):
    if mode == "none":
        return patch

    rgb = np.array(patch.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mode in {"clahe", "clahe_sharpen", "highpass"}:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if mode in {"sharpen", "clahe_sharpen"}:
        blur = cv2.GaussianBlur(bgr, (0, 0), 0.8)
        bgr = cv2.addWeighted(bgr, 1.7, blur, -0.7, 0)

    if mode == "highpass":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
        detail = cv2.absdiff(gray, blur)
        detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
        detail_bgr = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)
        bgr = cv2.addWeighted(bgr, 0.75, detail_bgr, 0.25, 0)

    enhanced = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)


def predict_tensor_prob(model, tensor, device):
    x = tensor.unsqueeze(0).to(device)
    with torch.inference_mode():
        return torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()


def predict_patch_prob(model, patch, image_size, device, mask_enhance, tta):
    orig_w, orig_h = patch.size
    if orig_w != image_size or orig_h != image_size:
        patch = patch.resize((image_size, image_size), Image.BILINEAR)
    patch = enhance_patch_for_mask(patch, mask_enhance)
    tensor = TF.to_tensor(patch)

    if not tta:
        prob = predict_tensor_prob(model, tensor, device)
    else:
        probs = [predict_tensor_prob(model, tensor, device)]

        prob_h = predict_tensor_prob(model, torch.flip(tensor, dims=[2]), device)
        probs.append(np.flip(prob_h, axis=1).copy())

        prob_v = predict_tensor_prob(model, torch.flip(tensor, dims=[1]), device)
        probs.append(np.flip(prob_v, axis=0).copy())

        prob_hv = predict_tensor_prob(model, torch.flip(tensor, dims=[1, 2]), device)
        probs.append(np.flip(np.flip(prob_hv, axis=0), axis=1).copy())

        prob = np.mean(probs, axis=0)

    if prob.shape != (orig_h, orig_w):
        prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_mask_tiled(
    model,
    image,
    image_size,
    tile_size,
    tile_overlap,
    threshold,
    close,
    component_expand,
    dilate,
    mask_enhance,
    tta,
    device,
):
    width, height = image.size
    if tile_size <= 0:
        prob = predict_patch_prob(model, image, image_size, device, mask_enhance, tta)
        return finalize_mask(prob, threshold, close, component_expand, dilate)

    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("--tile-overlap must be >= 0 and < --tile-size")

    stride = tile_size - tile_overlap
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    xs = tile_starts(width, tile_size, stride)
    ys = tile_starts(height, tile_size, stride)

    for y in ys:
        for x in xs:
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            patch = image.crop((x, y, right, bottom))
            patch_prob = predict_patch_prob(model, patch, image_size, device, mask_enhance, tta)
            prob_sum[y:bottom, x:right] += patch_prob
            weight_sum[y:bottom, x:right] += 1.0

    prob = prob_sum / np.maximum(weight_sum, 1e-6)
    return finalize_mask(prob, threshold, close, component_expand, dilate)


def finalize_mask(prob, threshold, close, component_expand, dilate):
    mask = (prob >= threshold).astype(np.uint8) * 255

    if close > 0:
        kernel_size = close * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if component_expand > 0:
        mask = expand_connected_components(mask, component_expand)

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def expand_connected_components(mask, radius):
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    expanded = np.zeros_like(mask, dtype=np.uint8)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        component = cv2.dilate(component, kernel, iterations=1)
        expanded = np.maximum(expanded, component)

    return expanded


def load_lama(lama_checkpoint, device):
    lama_checkpoint = str(Path(lama_checkpoint).resolve())
    if not os.path.exists(lama_checkpoint):
        raise FileNotFoundError(f"LaMa checkpoint not found: {lama_checkpoint}")

    os.environ["LAMA_MODEL_URL"] = lama_checkpoint
    os.environ.setdefault("LAMA_MODEL_MD5", "")

    from iopaint.model.lama import LaMa

    return LaMa(device)


def run_lama(lama, image, mask, hd_strategy, crop_trigger_size, crop_margin, resize_limit):
    from iopaint.schema import HDStrategy, InpaintRequest

    np_rgb = np.array(image.convert("RGB"))
    strategies = {
        "original": HDStrategy.ORIGINAL,
        "crop": HDStrategy.CROP,
        "resize": HDStrategy.RESIZE,
    }
    config = InpaintRequest(
        hd_strategy=strategies[hd_strategy],
        hd_strategy_crop_trigger_size=crop_trigger_size,
        hd_strategy_crop_margin=crop_margin,
        hd_strategy_resize_limit=resize_limit,
        sd_keep_unmasked_area=True,
    )
    result_bgr = lama(np_rgb, mask, config)
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def output_image_suffix(input_path, output_format):
    if output_format == "png":
        return ".png"
    if output_format == "tiff":
        return ".tif"
    if output_format == "same-lossless":
        suffix = input_path.suffix.lower()
        return suffix if suffix in LOSSLESS_EXTS else ".png"
    if output_format == "same":
        return input_path.suffix.lower()
    raise ValueError(f"Unsupported output format: {output_format}")


def save_rgb_image_loss_aware(image, path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=100, subsampling=0)
    elif suffix in {".tif", ".tiff"}:
        image.save(path, compression="tiff_lzw")
    else:
        image.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DeepLab dust masks with local LaMa inpainting.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default="outputs/deeplab_lama")
    parser.add_argument("--deeplab-checkpoint", default="checkpoints/fakefilmcolor_deeplab_best.pth")
    parser.add_argument("--lama-checkpoint", default="model_cache/hub/checkpoints/big-lama.pt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=512, help="Sliding-window tile size for large images. Use 0 for whole-image resize.")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Overlap between neighboring tiles.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--close", type=int, default=3, help="Morphological close radius to fill small gaps in dust masks.")
    parser.add_argument("--component-expand", type=int, default=10, help="Expand each connected dust component before LaMa.")
    parser.add_argument("--dilate", type=int, default=0, help="Extra global mask dilation radius after component expansion.")
    parser.add_argument("--mask-enhance", choices=["none", "clahe", "sharpen", "clahe_sharpen", "highpass"], default="none", help="Enhance only the mask predictor input.")
    parser.add_argument("--tta", action="store_true", help="Average mask predictions over flip test-time augmentation.")
    parser.add_argument("--lama-hd-strategy", choices=["original", "crop", "resize"], default="crop")
    parser.add_argument("--lama-crop-trigger-size", type=int, default=800)
    parser.add_argument("--lama-crop-margin", type=int, default=128)
    parser.add_argument("--lama-resize-limit", type=int, default=1280)
    parser.add_argument(
        "--output-format",
        choices=["png", "tiff", "same-lossless", "same"],
        default="same",
        help="Inpainted output format. Default png avoids extra JPEG loss.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all images.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    mask_dir = out_dir / "masks"
    result_dir = out_dir / "inpainted"
    mask_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    images = list_inputs(args.input)
    if args.max_images > 0:
        images = images[: args.max_images]

    deeplab = load_deeplab(args.deeplab_checkpoint, device)
    lama = load_lama(args.lama_checkpoint, device)

    for image_path in tqdm(images, desc="DeepLab + LaMa"):
        image = Image.open(image_path).convert("RGB")
        mask = predict_mask_tiled(
            deeplab,
            image,
            args.image_size,
            args.tile_size,
            args.tile_overlap,
            args.threshold,
            args.close,
            args.component_expand,
            args.dilate,
            args.mask_enhance,
            args.tta,
            device,
        )
        result = run_lama(
            lama,
            image,
            mask,
            args.lama_hd_strategy,
            args.lama_crop_trigger_size,
            args.lama_crop_margin,
            args.lama_resize_limit,
        )

        stem = image_path.stem
        suffix = output_image_suffix(image_path, args.output_format)
        Image.fromarray(mask).save(mask_dir / f"{stem}_mask.png")
        save_rgb_image_loss_aware(result, result_dir / f"{stem}_lama{suffix}")

    print(f"Saved masks to: {mask_dir}")
    print(f"Saved LaMa results to: {result_dir}")


if __name__ == "__main__":
    main()
