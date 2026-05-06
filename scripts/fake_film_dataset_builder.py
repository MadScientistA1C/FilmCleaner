import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_image_preserve_tif(path: str) -> np.ndarray:
    """Load image while handling common TIFF bit depths and channel layouts.

    Returns BGR uint8 image.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    # Handle channel count / alpha.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # Drop alpha for style synthesis.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] != 3:
        raise ValueError(f"Unsupported channel layout for {path}: {img.shape}")

    # Normalize to uint8 for downstream processing.
    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        # Robust normalization for 16-bit TIFFs.
        img_f = img.astype(np.float32)
        # Percentile clipping preserves contrast better than direct divide.
        lo = np.percentile(img_f, 0.2)
        hi = np.percentile(img_f, 99.8)
        if hi <= lo:
            hi = img_f.max() if img_f.max() > lo else lo + 1.0
        img_f = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)
        return (img_f * 255.0).astype(np.uint8)

    # Fallback for float / other integer types.
    img_f = img.astype(np.float32)
    lo = np.percentile(img_f, 0.2)
    hi = np.percentile(img_f, 99.8)
    if hi <= lo:
        hi = img_f.max() if img_f.max() > lo else lo + 1.0
    img_f = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8)


def _film_grain(gray_01: np.ndarray, grain_strength: float) -> np.ndarray:
    h, w = gray_01.shape[:2]
    noise_small = np.random.randn(h, w).astype(np.float32)
    noise_large = np.random.randn(h, w).astype(np.float32)
    noise_small = cv2.GaussianBlur(noise_small, (0, 0), 0.4)
    noise_large = cv2.GaussianBlur(noise_large, (0, 0), 1.2)
    grain = 0.7 * noise_small + 0.3 * noise_large

    luminance_weight = 1.0 - gray_01
    luminance_weight = 0.4 + 0.6 * luminance_weight
    grain = grain * luminance_weight
    return grain_strength * grain[:, :, np.newaxis]


def _apply_vignette(img_01: np.ndarray, vignette_strength: float) -> np.ndarray:
    if vignette_strength <= 0:
        return img_01
    h, w = img_01.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    dist = dist / max(dist.max(), 1e-6)
    vignette = 1.0 - vignette_strength * (dist ** 1.8)
    vignette = np.clip(vignette, 0.0, 1.0)[:, :, np.newaxis]
    return np.clip(img_01 * vignette, 0.0, 1.0)


def apply_color_film_look(
    image: np.ndarray,
    grain_strength: float = 0.12,
    fade: float = 0.07,
    contrast: float = 0.96,
    warmth: float = 0.05,
    vignette_strength: float = 0.12,
) -> np.ndarray:
    """Color film-like rendering on BGR uint8."""
    img = image.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)
    img = img * (1.0 - fade) + fade * 0.85
    img = np.clip(img, 0.0, 1.0)

    # Warm shift in BGR.
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + warmth), 0.0, 1.0)
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - warmth * 0.5), 0.0, 1.0)

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img = np.clip(img + _film_grain(gray, grain_strength), 0.0, 1.0)
    img = _apply_vignette(img, vignette_strength)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def apply_bw_film_look(
    image: np.ndarray,
    grain_strength: float = 0.14,
    fade: float = 0.05,
    contrast: float = 1.02,
    vignette_strength: float = 0.12,
) -> np.ndarray:
    """Black-and-white film-like rendering on BGR uint8."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Slight S-curve for BW film contrast.
    gray = np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)
    gray = gray * (1.0 - fade) + fade * 0.82
    gray = np.clip(gray, 0.0, 1.0)

    grain = _film_grain(gray, grain_strength)[:, :, 0]
    gray = np.clip(gray + grain, 0.0, 1.0)

    bw = np.stack([gray, gray, gray], axis=2)
    bw = _apply_vignette(bw, vignette_strength)
    return (bw * 255.0).clip(0, 255).astype(np.uint8)


def compute_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def extract_patches(image: np.ndarray, patch_size: int = 512, stride: int = 512) -> List[Tuple[np.ndarray, int, int]]:
    h, w = image.shape[:2]
    xs = compute_positions(w, patch_size, stride)
    ys = compute_positions(h, patch_size, stride)
    patches = []
    for y in ys:
        for x in xs:
            patch = image[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                canvas = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
                canvas[:patch.shape[0], :patch.shape[1]] = patch
                patch = canvas
            patches.append((patch, x, y))
    return patches


def iter_images(input_dir: str):
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if Path(name).suffix.lower() in exts:
                yield os.path.join(root, name)


def process_folder(
    input_dir: str,
    output_dir: str,
    bw_ratio: float = 0.4,
    patch_size: int = 512,
    stride: int = 512,
    seed: int = 42,
    save_full_images: bool = True,
) -> None:
    image_paths = list(iter_images(input_dir))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    random.seed(seed)
    np.random.seed(seed)
    n_bw = int(round(len(image_paths) * bw_ratio))
    bw_set = set(random.sample(image_paths, n_bw))

    output_dir = Path(output_dir)
    stylized_dir = output_dir / "stylized"
    patches_dir = output_dir / "patches"
    stylized_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_path",
            "stylized_path",
            "patch_path",
            "style",
            "x",
            "y",
            "patch_size",
        ])

        for idx, src in enumerate(image_paths, start=1):
            img = load_image_preserve_tif(src)
            style = "bw" if src in bw_set else "color"
            if style == "bw":
                stylized = apply_bw_film_look(img)
            else:
                stylized = apply_color_film_look(img)

            src_stem = Path(src).stem
            full_name = f"{src_stem}_{style}.png"
            stylized_path = stylized_dir / full_name
            if save_full_images:
                ok = cv2.imwrite(str(stylized_path), stylized)
                if not ok:
                    raise IOError(f"Failed to save stylized image: {stylized_path}")

            patches = extract_patches(stylized, patch_size=patch_size, stride=stride)
            for patch_id, (patch, x, y) in enumerate(patches):
                patch_name = f"{src_stem}_{style}_x{x}_y{y}_p{patch_id:04d}.png"
                patch_path = patches_dir / patch_name
                ok = cv2.imwrite(str(patch_path), patch)
                if not ok:
                    raise IOError(f"Failed to save patch: {patch_path}")
                writer.writerow([
                    src,
                    str(stylized_path) if save_full_images else "",
                    str(patch_path),
                    style,
                    x,
                    y,
                    patch_size,
                ])

            print(f"[{idx}/{len(image_paths)}] done: {src} -> {style}, patches={len(patches)}")

    print(f"Finished. Manifest saved to: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a fake-film dataset with 40% BW images and 512x512 patches.")
    parser.add_argument("--input_dir", required=True, help="Folder containing input TIFF/JPG/PNG images.")
    parser.add_argument("--output_dir", required=True, help="Folder to save stylized images and patches.")
    parser.add_argument("--bw_ratio", type=float, default=0.4, help="Ratio of source images converted to BW film style.")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size.")
    parser.add_argument("--stride", type=int, default=512, help="Patch stride.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_save_full_images", action="store_true", help="Do not save full stylized images.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bw_ratio=args.bw_ratio,
        patch_size=args.patch_size,
        stride=args.stride,
        seed=args.seed,
        save_full_images=not args.no_save_full_images,
    )
