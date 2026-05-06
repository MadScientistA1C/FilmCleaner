import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------
# Basic image I/O
# ----------------------------

def load_image_preserve_tif(path: str) -> np.ndarray:
    """Load image while handling common TIFF bit depths and channel layouts.

    Returns BGR uint8 image.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] != 3:
        raise ValueError(f"Unsupported channel layout for {path}: {img.shape}")

    if img.dtype == np.uint8:
        return img

    img_f = img.astype(np.float32)
    lo = np.percentile(img_f, 0.2)
    hi = np.percentile(img_f, 99.8)
    if hi <= lo:
        hi = img_f.max() if img_f.max() > lo else lo + 1.0
    img_f = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8)


# ----------------------------
# Film look
# ----------------------------

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
    img = image.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)
    img = img * (1.0 - fade) + fade * 0.85
    img = np.clip(img, 0.0, 1.0)

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)
    gray = gray * (1.0 - fade) + fade * 0.82
    gray = np.clip(gray, 0.0, 1.0)

    grain = _film_grain(gray, grain_strength)[:, :, 0]
    gray = np.clip(gray + grain, 0.0, 1.0)

    bw = np.stack([gray, gray, gray], axis=2)
    bw = _apply_vignette(bw, vignette_strength)
    return (bw * 255.0).clip(0, 255).astype(np.uint8)


# ----------------------------
# Patch extraction
# ----------------------------

def compute_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions



def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 512,
    stride: int = 512,
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    h, w = image.shape[:2]
    xs = compute_positions(w, patch_size, stride)
    ys = compute_positions(h, patch_size, stride)
    patches = []
    for y in ys:
        for x in xs:
            patch = image[y:y + patch_size, x:x + patch_size]
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                canvas = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
                mask_canvas = np.zeros((patch_size, patch_size), dtype=mask.dtype)
                canvas[:patch.shape[0], :patch.shape[1]] = patch
                mask_canvas[:patch_mask.shape[0], :patch_mask.shape[1]] = patch_mask
                patch = canvas
                patch_mask = mask_canvas
            patches.append((patch, patch_mask, x, y))
    return patches


# ----------------------------
# FILM-AA prototype extraction
# ----------------------------

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Prototype:
    alpha: np.ndarray  # float32 [0, 1]
    texture: np.ndarray  # float32 [0, 1]
    mode: str  # 'light' or 'dark'
    label: str
    source_image: str
    source_json: str
    ann_index: int


@dataclass
class OverlayParams:
    alpha_gain: float
    alpha_gamma: float
    detail_mix: float
    dark_strength: float
    light_strength: float
    binary_threshold: float


@dataclass
class PlacementContext:
    hotspots: List[Tuple[int, int]]
    hotspot_radius_range: Tuple[int, int]
    hotspot_prob: float



def iter_images(input_dir: str) -> Generator[str, None, None]:
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if Path(name).suffix.lower() in SUPPORTED_EXTS:
                yield os.path.join(root, name)



def pair_filmaa_files(filmaa_dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    p = Path(filmaa_dir)
    json_files = sorted(p.rglob("*.json"))
    for json_path in json_files:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is not None:
            pairs.append((str(img_path), str(json_path)))
    return pairs



def _is_point_like(x: object) -> bool:
    return isinstance(x, dict) and "x" in x and "y" in x



def _iter_annotation_nodes(obj: object) -> Generator[Dict, None, None]:
    if isinstance(obj, dict):
        points = obj.get("points")
        if isinstance(points, list) and len(points) >= 3 and all(_is_point_like(p) for p in points):
            yield obj
        for v in obj.values():
            yield from _iter_annotation_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_annotation_nodes(item)



def _safe_percentile(vals: np.ndarray, q: float, default: float = 0.0) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)
    return float(np.percentile(vals, q))



def _extract_single_prototype(
    scan_img: np.ndarray,
    points: Sequence[Tuple[int, int]],
    label: str,
    source_image: str,
    source_json: str,
    ann_index: int,
    pad: int = 3,
) -> Optional[Prototype]:
    poly = np.array(points, dtype=np.int32)
    h, w = scan_img.shape[:2]

    x, y, bw, bh = cv2.boundingRect(poly)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)

    crop = scan_img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    local_poly = poly.copy()
    local_poly[:, 0] -= x0
    local_poly[:, 1] -= y0

    poly_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [local_poly], 255)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(poly_mask, kernel, iterations=2)
    ring = cv2.subtract(dilated, poly_mask)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = gray[poly_mask > 0]
    ring_vals = gray[ring > 0]

    if inside.size < 4:
        return None

    if ring_vals.size >= 8:
        bg = float(np.median(ring_vals))
    else:
        bg = _safe_percentile(inside, 10, default=float(np.median(inside)))

    pos = np.clip(gray - bg, 0.0, None)
    neg = np.clip(bg - gray, 0.0, None)
    pos[poly_mask == 0] = 0.0
    neg[poly_mask == 0] = 0.0

    pos_hi = _safe_percentile(pos[poly_mask > 0], 99, default=0.0)
    neg_hi = _safe_percentile(neg[poly_mask > 0], 99, default=0.0)

    if max(pos_hi, neg_hi) < 2.0:
        return None

    if pos_hi >= neg_hi:
        mode = "light"
        signal = pos
        scale = max(pos_hi, 1.0)
    else:
        mode = "dark"
        signal = neg
        scale = max(neg_hi, 1.0)

    alpha = np.clip(signal / scale, 0.0, 1.0)
    # Keep prototype edges crisp. Only apply a minimal smoothing to avoid jagged polygon artifacts.
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.20)
    alpha[poly_mask == 0] = 0.0
    alpha[alpha < 0.03] = 0.0

    if np.count_nonzero(alpha > 0.05) < 4:
        return None

    texture = np.clip(signal / scale, 0.0, 1.0)
    texture = cv2.GaussianBlur(texture, (0, 0), 0.12)
    texture[poly_mask == 0] = 0.0

    ys, xs = np.where(alpha > 0.02)
    if len(xs) == 0 or len(ys) == 0:
        return None
    tx0, tx1 = xs.min(), xs.max() + 1
    ty0, ty1 = ys.min(), ys.max() + 1

    alpha = alpha[ty0:ty1, tx0:tx1].astype(np.float32)
    texture = texture[ty0:ty1, tx0:tx1].astype(np.float32)

    if alpha.shape[0] < 2 or alpha.shape[1] < 2:
        return None

    return Prototype(
        alpha=alpha,
        texture=texture,
        mode=mode,
        label=label,
        source_image=source_image,
        source_json=source_json,
        ann_index=ann_index,
    )



def build_prototypes_from_filmaa(filmaa_dir: str) -> List[Prototype]:
    pairs = pair_filmaa_files(filmaa_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No FILM-AA image/json pairs found in: {filmaa_dir}. "
            "Expected names like 'Scan (1).jpg' + 'Scan (1).json'."
        )

    prototypes: List[Prototype] = []
    total_anns = 0

    for img_path, json_path in pairs:
        scan_img = load_image_preserve_tif(img_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ann_index, ann in enumerate(_iter_annotation_nodes(data)):
            points = ann.get("points", [])
            label_obj = ann.get("label", {})
            if isinstance(label_obj, dict):
                label = str(label_obj.get("name", "unknown"))
            else:
                label = str(label_obj)

            pts = []
            for p in points:
                try:
                    pts.append((int(round(float(p["x"]))), int(round(float(p["y"])))))
                except Exception:
                    pts = []
                    break

            if len(pts) < 3:
                continue

            total_anns += 1
            proto = _extract_single_prototype(
                scan_img=scan_img,
                points=pts,
                label=label,
                source_image=img_path,
                source_json=json_path,
                ann_index=ann_index,
            )
            if proto is not None:
                prototypes.append(proto)

    if not prototypes:
        raise RuntimeError(
            "FILM-AA annotations were found, but no valid prototypes were extracted. "
            "Check whether the JSON schema or scan brightness differs from the expected format."
        )

    print(f"FILM-AA pairs: {len(pairs)}, annotations visited: {total_anns}, prototypes extracted: {len(prototypes)}")
    return prototypes


# ----------------------------
# Prototype augmentation / overlay
# ----------------------------

def unsharp_mask_single_channel(img: np.ndarray, amount: float = 0.8, sigma: float = 0.6) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = img + amount * (img - blur)
    return np.clip(sharp, 0.0, 1.0)


def transform_prototype(proto: Prototype, rng: random.Random) -> Prototype:
    return transform_prototype_with_sharpness(proto, rng)


def transform_prototype_with_sharpness(
    proto: Prototype,
    rng: random.Random,
    soft_artifact_ratio: float = 0.15,
) -> Prototype:
    alpha = proto.alpha
    texture = proto.texture

    if rng.random() < 0.5:
        alpha = np.flip(alpha, axis=1).copy()
        texture = np.flip(texture, axis=1).copy()
    if rng.random() < 0.5:
        alpha = np.flip(alpha, axis=0).copy()
        texture = np.flip(texture, axis=0).copy()

    label_l = proto.label.lower()
    if "hair" in label_l or "scratch" in label_l:
        scale = rng.uniform(0.85, 2.40)
    elif "dust" in label_l or "dirt" in label_l:
        # Point-like dust dominates real scans; keep most of these prototypes compact.
        scale = rng.uniform(0.28, 1.35)
    else:
        scale = rng.uniform(0.60, 1.90)
    new_w = max(2, int(round(alpha.shape[1] * scale)))
    new_h = max(2, int(round(alpha.shape[0] * scale)))
    alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    texture = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    angle = rng.uniform(0.0, 360.0)
    h, w = alpha.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    bound_w = int(math.ceil((h * sin) + (w * cos)))
    bound_h = int(math.ceil((h * cos) + (w * sin)))
    M[0, 2] += (bound_w / 2) - cx
    M[1, 2] += (bound_h / 2) - cy

    alpha = cv2.warpAffine(alpha, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    texture = cv2.warpAffine(texture, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    is_soft = rng.random() < soft_artifact_ratio
    if is_soft:
        blur_sigma = rng.uniform(0.35, 0.95)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
        texture = cv2.GaussianBlur(texture, (0, 0), max(0.10, blur_sigma * 0.45))
        alpha = unsharp_mask_single_channel(alpha, amount=rng.uniform(0.15, 0.40), sigma=rng.uniform(0.45, 0.90))
        texture = unsharp_mask_single_channel(texture, amount=rng.uniform(0.10, 0.35), sigma=rng.uniform(0.45, 0.90))
    else:
        # Scanner dust is usually very crisp. Keep the dominant mode extremely sharp.
        alpha = unsharp_mask_single_channel(alpha, amount=rng.uniform(1.20, 2.40), sigma=rng.uniform(0.25, 0.55))
        texture = unsharp_mask_single_channel(texture, amount=rng.uniform(0.90, 1.80), sigma=rng.uniform(0.25, 0.50))

    # Clamp tiny halos that can appear after sharpening.
    alpha[alpha < 0.02] = 0.0
    texture[texture < 0.01] = 0.0

    return Prototype(
        alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
        texture=np.clip(texture, 0.0, 1.0).astype(np.float32),
        mode=proto.mode,
        label=proto.label,
        source_image=proto.source_image,
        source_json=proto.source_json,
        ann_index=proto.ann_index,
    )



def sample_overlay_params(rng: random.Random) -> OverlayParams:
    """Create mixed weak / medium / strong dust visibility.

    The distribution intentionally keeps a noticeable fraction of strong artifacts,
    while still leaving many subtle ones.
    """
    p = rng.random()
    if p < 0.10:
        # weak dust
        return OverlayParams(
            alpha_gain=rng.uniform(0.70, 1.05),
            alpha_gamma=rng.uniform(1.05, 1.35),
            detail_mix=rng.uniform(0.25, 0.45),
            dark_strength=rng.uniform(0.35, 0.65),
            light_strength=rng.uniform(0.25, 0.50),
            binary_threshold=rng.uniform(0.035, 0.055),
        )
    if p < 0.45:
        # medium dust
        return OverlayParams(
            alpha_gain=rng.uniform(1.00, 1.50),
            alpha_gamma=rng.uniform(0.85, 1.10),
            detail_mix=rng.uniform(0.38, 0.62),
            dark_strength=rng.uniform(0.60, 1.00),
            light_strength=rng.uniform(0.45, 0.80),
            binary_threshold=rng.uniform(0.040, 0.065),
        )
    # strong dust
    return OverlayParams(
        alpha_gain=rng.uniform(1.40, 2.30),
        alpha_gamma=rng.uniform(0.60, 0.95),
        detail_mix=rng.uniform(0.55, 0.82),
        dark_strength=rng.uniform(0.95, 1.45),
        light_strength=rng.uniform(0.75, 1.15),
        binary_threshold=rng.uniform(0.045, 0.075),
    )



def sample_artifact_count(rng: random.Random, megapixels: float, min_density: float, max_density: float) -> int:
    """Sample image-level artifact count with dense scenes occurring regularly."""
    p = rng.random()
    if p < 0.20:
        density = rng.uniform(max(min_density, 0.8 * max_density), 1.35 * max_density)
    elif p < 0.65:
        density = rng.uniform(0.80 * min_density, max_density)
    else:
        density = rng.uniform(min_density, 0.85 * max_density)

    count = max(1, int(round(megapixels * density)))
    count += rng.randint(0, max(1, count // 4))
    return count


def sample_layered_artifact_count(rng: random.Random, megapixels: float) -> Tuple[int, str]:
    """Sample realistic dirtyness tiers instead of making every image very dirty."""
    p = rng.random()
    if p < 0.08:
        return 0, "clean"
    if p < 0.20:
        density = rng.uniform(0.15, 0.70)
        tier = "near_clean"
    elif p < 0.68:
        density = rng.uniform(1.8, 5.5)
        tier = "normal"
    elif p < 0.92:
        density = rng.uniform(5.5, 11.0)
        tier = "medium_dirty"
    else:
        density = rng.uniform(11.0, 20.0)
        tier = "heavy_dirty"

    count = int(round(megapixels * density))
    if tier != "near_clean":
        count += rng.randint(0, max(1, count // 5))
    return max(1, count), tier


def sample_point_dust_count(rng: random.Random, megapixels: float, dirtiness_tier: str, density_scale: float) -> int:
    if density_scale <= 0:
        return 0

    if dirtiness_tier == "clean":
        density = rng.uniform(0.0, 0.25)
    elif dirtiness_tier == "near_clean":
        density = rng.uniform(0.5, 1.6)
    elif dirtiness_tier == "normal":
        density = rng.uniform(4.0, 8.5)
    elif dirtiness_tier == "medium_dirty":
        density = rng.uniform(8.5, 16.0)
    elif dirtiness_tier == "heavy_dirty":
        density = rng.uniform(16.0, 30.0)
    else:
        density = rng.uniform(4.0, 12.0)

    return int(round(megapixels * density * density_scale))


def overlay_point_dust(
    image: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    n_points: int,
    soft_artifact_ratio: float,
) -> int:
    h, w = image.shape[:2]
    successful = 0

    for _ in range(n_points):
        cx = rng.randint(0, max(0, w - 1))
        cy = rng.randint(0, max(0, h - 1))
        radius = rng.choices([1, 2, 3, 4, 5], weights=[0.30, 0.36, 0.20, 0.10, 0.04], k=1)[0]
        rx = max(1, int(round(radius * rng.uniform(0.75, 1.65))))
        ry = max(1, int(round(radius * rng.uniform(0.75, 1.65))))

        pad = max(rx, ry) + 4
        x0 = max(0, cx - pad)
        y0 = max(0, cy - pad)
        x1 = min(w, cx + pad + 1)
        y1 = min(h, cy + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        local_w = x1 - x0
        local_h = y1 - y0
        lx = cx - x0
        ly = cy - y0
        alpha = np.zeros((local_h, local_w), dtype=np.float32)
        angle = rng.uniform(0.0, 180.0)
        cv2.ellipse(alpha, (lx, ly), (rx, ry), angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)

        is_soft = rng.random() < soft_artifact_ratio
        if is_soft:
            alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.45, 1.15))
            strength = rng.uniform(0.35, 0.75)
        else:
            alpha = unsharp_mask_single_channel(alpha, amount=rng.uniform(1.2, 2.2), sigma=rng.uniform(0.25, 0.45))
            alpha = (alpha > rng.uniform(0.12, 0.28)).astype(np.float32)
            strength = rng.uniform(0.65, 1.25)

        if np.count_nonzero(alpha > 0.05) == 0:
            continue

        region = image[y0:y1, x0:x1].astype(np.float32)
        eff = np.clip(alpha * strength, 0.0, 1.0)[:, :, np.newaxis]

        if rng.random() < 0.3:
            tint = np.array([
                rng.uniform(0.82, 1.00),
                rng.uniform(0.82, 1.00),
                rng.uniform(0.78, 0.98),
            ], dtype=np.float32)
            region = region * (1.0 - eff * tint)
        else:
            light_color = np.array([
                rng.uniform(220, 255),
                rng.uniform(220, 255),
                rng.uniform(220, 255),
            ], dtype=np.float32)
            region = region + (light_color - region) * eff

        image[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        point_mask = (alpha > 0.05).astype(np.uint8) * 255
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], point_mask)
        successful += 1

    return successful


def is_point_like_filmaa_prototype(proto: Prototype, max_size: int = 32, max_area: int = 512) -> bool:
    if proto.label not in {"Dust", "Dirt"}:
        return False
    h, w = proto.alpha.shape[:2]
    return max(h, w) <= max_size and h * w <= max_area


def collect_point_like_filmaa_prototypes(prototypes: Sequence[Prototype]) -> List[Prototype]:
    point_prototypes = [p for p in prototypes if is_point_like_filmaa_prototype(p)]
    if point_prototypes:
        return point_prototypes
    return [p for p in prototypes if p.label in {"Dust", "Dirt"}]


def overlay_extra_filmaa_point_dust(
    image: np.ndarray,
    mask: np.ndarray,
    prototypes: Sequence[Prototype],
    rng: random.Random,
    n_points: int,
    soft_artifact_ratio: float,
) -> int:
    if n_points <= 0:
        return 0

    point_prototypes = collect_point_like_filmaa_prototypes(prototypes)
    if not point_prototypes:
        return 0

    placement = make_placement_context(image.shape, rng, n_points)
    successful = 0
    max_trials = max(n_points * 4, n_points + 12)
    for _ in range(max_trials):
        if successful >= n_points:
            break
        proto = rng.choice(point_prototypes)
        proto_t = transform_prototype_with_sharpness(proto, rng, soft_artifact_ratio=soft_artifact_ratio)
        if overlay_single_prototype(image, mask, proto_t, rng, placement):
            successful += 1

    return successful


def group_prototypes_by_label(prototypes: Sequence[Prototype]) -> Dict[str, List[Prototype]]:
    groups: Dict[str, List[Prototype]] = {}
    for proto in prototypes:
        groups.setdefault(proto.label, []).append(proto)
    return groups


def choose_prototype(
    prototypes: Sequence[Prototype],
    grouped: Dict[str, List[Prototype]],
    rng: random.Random,
    category_sampling: str,
) -> Prototype:
    if category_sampling == "balanced" and grouped:
        label = rng.choice(list(grouped.keys()))
        return rng.choice(grouped[label])
    if category_sampling == "point_biased" and grouped:
        label_weights = [
            ("Dust", 0.58),
            ("Dirt", 0.24),
            ("Short hair", 0.08),
            ("Long hair", 0.06),
            ("Scratch", 0.04),
        ]
        available = [(label, weight) for label, weight in label_weights if label in grouped]
        if available:
            labels = [item[0] for item in available]
            weights = [item[1] for item in available]
            label = rng.choices(labels, weights=weights, k=1)[0]
            return rng.choice(grouped[label])
    return rng.choice(prototypes)



def make_placement_context(image_shape: Tuple[int, int, int], rng: random.Random, n_artifacts: int) -> PlacementContext:
    h, w = image_shape[:2]
    n_hotspots = 0
    if n_artifacts >= 12:
        n_hotspots = rng.randint(1, 3)
    elif n_artifacts >= 6 and rng.random() < 0.6:
        n_hotspots = 1

    hotspots: List[Tuple[int, int]] = []
    for _ in range(n_hotspots):
        hx = rng.randint(0, max(0, w - 1))
        hy = rng.randint(0, max(0, h - 1))
        hotspots.append((hx, hy))

    return PlacementContext(
        hotspots=hotspots,
        hotspot_radius_range=(80, 360),
        hotspot_prob=0.45 if hotspots else 0.0,
    )



def choose_top_left(
    image_h: int,
    image_w: int,
    proto_h: int,
    proto_w: int,
    placement: PlacementContext,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    if proto_h >= image_h or proto_w >= image_w:
        return None

    if placement.hotspots and rng.random() < placement.hotspot_prob:
        cx, cy = rng.choice(placement.hotspots)
        radius = rng.randint(*placement.hotspot_radius_range)
        for _ in range(20):
            dx = int(round(rng.gauss(0, radius / 2.5)))
            dy = int(round(rng.gauss(0, radius / 2.5)))
            x0 = int(np.clip(cx + dx - proto_w // 2, 0, image_w - proto_w))
            y0 = int(np.clip(cy + dy - proto_h // 2, 0, image_h - proto_h))
            return x0, y0

    x0 = rng.randint(0, image_w - proto_w)
    y0 = rng.randint(0, image_h - proto_h)
    return x0, y0



def overlay_single_prototype(
    image: np.ndarray,
    mask: np.ndarray,
    proto: Prototype,
    rng: random.Random,
    placement: PlacementContext,
) -> bool:
    h_img, w_img = image.shape[:2]
    ph, pw = proto.alpha.shape[:2]

    if ph < 2 or pw < 2:
        return False
    if ph >= h_img or pw >= w_img:
        return False

    top_left = choose_top_left(h_img, w_img, ph, pw, placement, rng)
    if top_left is None:
        return False
    x0, y0 = top_left
    x1 = x0 + pw
    y1 = y0 + ph

    region = image[y0:y1, x0:x1].astype(np.float32)

    alpha = proto.alpha.copy()
    texture = proto.texture.copy()
    params = sample_overlay_params(rng)

    alpha = np.clip(alpha * params.alpha_gain, 0.0, 1.0)
    alpha = np.power(alpha, params.alpha_gamma)

    detail = (1.0 - params.detail_mix) + params.detail_mix * texture
    eff = np.clip(alpha * detail, 0.0, 1.0)

    # Sharp-only mode: no extra blur during compositing.
    eff_3 = eff[:, :, np.newaxis]

    if proto.mode == "dark":
        tint = np.array([
            rng.uniform(0.86, 1.00),
            rng.uniform(0.84, 0.98),
            rng.uniform(0.78, 0.95),
        ], dtype=np.float32)
        region = region * (1.0 - params.dark_strength * eff_3 * tint)
    else:
        light_color = np.array([
            rng.uniform(215, 252),
            rng.uniform(215, 252),
            rng.uniform(220, 255),
        ], dtype=np.float32)
        region = region + (light_color - region) * (params.light_strength * eff_3)

    image[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)

    binary = (alpha > params.binary_threshold).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return False
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], binary)
    return True



def overlay_random_artifacts(
    image: np.ndarray,
    prototypes: Sequence[Prototype],
    rng: random.Random,
    artifacts_per_megapixel_min: float,
    artifacts_per_megapixel_max: float,
    clean_image_ratio: float,
    category_sampling: str = "point_biased",
    soft_artifact_ratio: float = 0.15,
    dirtiness_profile: str = "layered",
    point_dust_density_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    out = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if rng.random() < clean_image_ratio:
        return out, mask, 0, "clean"

    megapixels = (image.shape[0] * image.shape[1]) / 1_000_000.0
    if dirtiness_profile == "layered":
        n, dirtiness_tier = sample_layered_artifact_count(rng, megapixels)
    else:
        n = sample_artifact_count(rng, megapixels, artifacts_per_megapixel_min, artifacts_per_megapixel_max)
        dirtiness_tier = "continuous"

    point_count = sample_point_dust_count(rng, megapixels, dirtiness_tier, point_dust_density_scale)

    placement = make_placement_context(out.shape, rng, n)
    grouped = group_prototypes_by_label(prototypes)

    successful = 0
    max_trials = max(n * 3, n + 8)
    for _ in range(max_trials):
        if successful >= n:
            break
        proto = choose_prototype(prototypes, grouped, rng, category_sampling)
        proto_t = transform_prototype_with_sharpness(proto, rng, soft_artifact_ratio=soft_artifact_ratio)
        if overlay_single_prototype(out, mask, proto_t, rng, placement):
            successful += 1

    point_successful = overlay_extra_filmaa_point_dust(
        out,
        mask,
        prototypes,
        rng,
        point_count,
        soft_artifact_ratio,
    )
    return out, mask, successful + point_successful, dirtiness_tier


# ----------------------------
# Dataset build
# ----------------------------

def process_folder(
    input_dir: str,
    filmaa_dir: str,
    output_dir: str,
    bw_ratio: float = 0.4,
    style: str = "mixed",
    max_images: int = 0,
    patch_size: int = 512,
    stride: int = 512,
    seed: int = 42,
    save_full_images: bool = True,
    artifacts_per_megapixel_min: float = 14.0,
    artifacts_per_megapixel_max: float = 30.0,
    clean_image_ratio: float = 0.0,
    category_sampling: str = "point_biased",
    soft_artifact_ratio: float = 0.15,
    dirtiness_profile: str = "layered",
    point_dust_density_scale: float = 1.0,
    save_prototype_previews: bool = False,
) -> None:
    image_paths = list(iter_images(input_dir))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    prototypes = build_prototypes_from_filmaa(filmaa_dir)
    label_counts: Dict[str, int] = {}
    for proto in prototypes:
        label_counts[proto.label] = label_counts.get(proto.label, 0) + 1
    print(f"Prototype labels: {label_counts}")

    if max_images > 0:
        if max_images > len(image_paths):
            raise ValueError(f"--max_images={max_images} exceeds available input images: {len(image_paths)}")
        image_paths = rng.sample(image_paths, max_images)

    n_bw = int(round(len(image_paths) * bw_ratio))
    bw_set = set(rng.sample(image_paths, n_bw)) if n_bw > 0 else set()

    output_dir = Path(output_dir)
    stylized_dir = output_dir / "stylized"
    masks_full_dir = output_dir / "masks_full"
    patch_img_dir = output_dir / "patches" / "images"
    patch_mask_dir = output_dir / "patches" / "masks"

    stylized_dir.mkdir(parents=True, exist_ok=True)
    masks_full_dir.mkdir(parents=True, exist_ok=True)
    patch_img_dir.mkdir(parents=True, exist_ok=True)
    patch_mask_dir.mkdir(parents=True, exist_ok=True)

    if save_prototype_previews:
        proto_dir = output_dir / "prototype_previews"
        proto_dir.mkdir(parents=True, exist_ok=True)
        for i, proto in enumerate(prototypes[: min(500, len(prototypes))]):
            alpha_u8 = (proto.alpha * 255).clip(0, 255).astype(np.uint8)
            tex_u8 = (proto.texture * 255).clip(0, 255).astype(np.uint8)
            preview = np.dstack([tex_u8, tex_u8, tex_u8])
            preview[:, :, 2] = np.maximum(preview[:, :, 2], alpha_u8)
            cv2.imwrite(str(proto_dir / f"proto_{i:05d}_{proto.mode}.png"), preview)

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_path",
            "stylized_path",
            "full_mask_path",
            "patch_path",
            "patch_mask_path",
            "style",
            "num_artifacts",
            "dirtiness_tier",
            "x",
            "y",
            "patch_size",
        ])

        for idx, src in enumerate(image_paths, start=1):
            img = load_image_preserve_tif(src)
            dusty_img, full_mask, num_artifacts, dirtiness_tier = overlay_random_artifacts(
                image=img,
                prototypes=prototypes,
                rng=rng,
                artifacts_per_megapixel_min=artifacts_per_megapixel_min,
                artifacts_per_megapixel_max=artifacts_per_megapixel_max,
                clean_image_ratio=clean_image_ratio,
                category_sampling=category_sampling,
                soft_artifact_ratio=soft_artifact_ratio,
                dirtiness_profile=dirtiness_profile,
                point_dust_density_scale=point_dust_density_scale,
            )

            if style == "mixed":
                current_style = "bw" if src in bw_set else "color"
            else:
                current_style = style

            if current_style == "bw":
                stylized = apply_bw_film_look(dusty_img)
            else:
                stylized = apply_color_film_look(dusty_img)

            src_stem = Path(src).stem
            stylized_name = f"{src_stem}_{current_style}.png"
            mask_name = f"{src_stem}_{current_style}_mask.png"
            stylized_path = stylized_dir / stylized_name
            mask_path = masks_full_dir / mask_name

            if save_full_images:
                ok1 = cv2.imwrite(str(stylized_path), stylized)
                ok2 = cv2.imwrite(str(mask_path), full_mask)
                if not ok1:
                    raise IOError(f"Failed to save stylized image: {stylized_path}")
                if not ok2:
                    raise IOError(f"Failed to save full mask: {mask_path}")

            patches = extract_patches(stylized, full_mask, patch_size=patch_size, stride=stride)
            for patch_id, (patch, patch_mask, x, y) in enumerate(patches):
                patch_name = f"{src_stem}_{current_style}_x{x}_y{y}_p{patch_id:04d}.png"
                patch_mask_name = f"{src_stem}_{current_style}_x{x}_y{y}_p{patch_id:04d}_mask.png"
                patch_path = patch_img_dir / patch_name
                patch_mask_path = patch_mask_dir / patch_mask_name
                ok1 = cv2.imwrite(str(patch_path), patch)
                ok2 = cv2.imwrite(str(patch_mask_path), patch_mask)
                if not ok1:
                    raise IOError(f"Failed to save patch: {patch_path}")
                if not ok2:
                    raise IOError(f"Failed to save patch mask: {patch_mask_path}")

                writer.writerow([
                    src,
                    str(stylized_path) if save_full_images else "",
                    str(mask_path) if save_full_images else "",
                    str(patch_path),
                    str(patch_mask_path),
                    current_style,
                    num_artifacts,
                    dirtiness_tier,
                    x,
                    y,
                    patch_size,
                ])

            print(
                f"[{idx}/{len(image_paths)}] done: {src} -> {current_style}, "
                f"tier={dirtiness_tier}, artifacts={num_artifacts}, patches={len(patches)}"
            )

    print(f"Finished. Manifest saved to: {manifest_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a fake-film dataset by: extracting FILM-AA artifact prototypes from Scan (*.jpg + *.json), "
            "randomly overlaying sharp-only FILM-AA dust onto clean images, applying color/BW film look, and exporting 512x512 patches + masks."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing clean input TIFF/JPG/PNG images.")
    parser.add_argument("--filmaa_dir", required=True, help="Folder containing FILM-AA pairs like 'Scan (1).jpg' and 'Scan (1).json'.")
    parser.add_argument("--output_dir", required=True, help="Folder to save stylized images, masks, and patches.")
    parser.add_argument("--bw_ratio", type=float, default=0.4, help="Ratio of source images converted to BW film style.")
    parser.add_argument("--style", choices=["mixed", "color", "bw"], default="mixed", help="Generate mixed, all-color, or all-BW film style.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of clean source images. 0 means all images.")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size.")
    parser.add_argument("--stride", type=int, default=512, help="Patch stride.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--artifacts_per_megapixel_min", type=float, default=3.0, help="Minimum overlaid artifacts per megapixel when --dirtiness_profile continuous.")
    parser.add_argument("--artifacts_per_megapixel_max", type=float, default=12.0, help="Maximum overlaid artifacts per megapixel when --dirtiness_profile continuous.")
    parser.add_argument("--clean_image_ratio", type=float, default=0.0, help="Probability that an image receives no synthetic artifacts.")
    parser.add_argument("--category_sampling", choices=["point_biased", "balanced", "proportional"], default="point_biased", help="How to sample FILM-AA labels.")
    parser.add_argument("--soft_artifact_ratio", type=float, default=0.15, help="Fraction of slightly blurred artifacts; the rest stay very sharp.")
    parser.add_argument("--dirtiness_profile", choices=["layered", "continuous"], default="layered", help="Layered keeps clean/normal/medium/heavy samples in the same dataset.")
    parser.add_argument("--point_dust_density_scale", type=float, default=1.0, help="Extra FILM-AA point-like Dust/Dirt prototype density multiplier.")
    parser.add_argument("--no_save_full_images", action="store_true", help="Do not save full stylized images and masks.")
    parser.add_argument("--save_prototype_previews", action="store_true", help="Save a subset of extracted prototype previews for inspection.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(
        input_dir=args.input_dir,
        filmaa_dir=args.filmaa_dir,
        output_dir=args.output_dir,
        bw_ratio=args.bw_ratio,
        style=args.style,
        max_images=args.max_images,
        patch_size=args.patch_size,
        stride=args.stride,
        seed=args.seed,
        save_full_images=not args.no_save_full_images,
        artifacts_per_megapixel_min=args.artifacts_per_megapixel_min,
        artifacts_per_megapixel_max=args.artifacts_per_megapixel_max,
        clean_image_ratio=args.clean_image_ratio,
        category_sampling=args.category_sampling,
        soft_artifact_ratio=args.soft_artifact_ratio,
        dirtiness_profile=args.dirtiness_profile,
        point_dust_density_scale=args.point_dust_density_scale,
        save_prototype_previews=args.save_prototype_previews,
    )
