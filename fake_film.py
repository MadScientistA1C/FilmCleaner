import os
import cv2
import numpy as np


def apply_film_look(
    image: np.ndarray,
    grain_strength: float = 0.10,
    fade: float = 0.08,
    contrast: float = 0.95,
    warmth: float = 0.04,
    vignette_strength: float = 0.15
) -> np.ndarray:
    """
    给图像添加胶片质感

    参数:
        image: BGR 格式图像, uint8
        grain_strength: 颗粒强度，建议 0.05 ~ 0.18
        fade: 褪色程度，建议 0.03 ~ 0.15
        contrast: 对比度，<1 会更柔和，建议 0.9 ~ 1.05
        warmth: 暖色调强度，建议 0.0 ~ 0.08
        vignette_strength: 晕影强度，建议 0.0 ~ 0.25

    返回:
        uint8 BGR 图像
    """
    if image is None:
        raise ValueError("输入图像为空，无法处理。")

    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # ----------------------------
    # 1. 轻微柔化对比度
    # ----------------------------
    img = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)

    # ----------------------------
    # 2. 轻微褪色（黑位抬升）
    # ----------------------------
    img = img * (1.0 - fade) + fade * 0.85
    img = np.clip(img, 0.0, 1.0)

    # ----------------------------
    # 3. 轻微暖色偏移
    # OpenCV是BGR顺序
    # ----------------------------
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + warmth), 0.0, 1.0)      # R
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - warmth * 0.5), 0.0, 1.0) # B

    # ----------------------------
    # 4. 胶片颗粒
    # 关键点：
    # - 使用灰度颗粒，而不是RGB独立噪声
    # - 做模糊，让颗粒更像胶片而不是电子噪声
    # - 暗部颗粒更明显
    # ----------------------------
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    noise_small = np.random.randn(h, w).astype(np.float32)
    noise_large = np.random.randn(h, w).astype(np.float32)

    noise_small = cv2.GaussianBlur(noise_small, (0, 0), 0.4)
    noise_large = cv2.GaussianBlur(noise_large, (0, 0), 1.2)

    grain = 0.7 * noise_small + 0.3 * noise_large

    # 暗部权重大，高光弱一些
    luminance_weight = 1.0 - gray
    luminance_weight = 0.4 + 0.6 * luminance_weight

    grain = grain * luminance_weight
    grain = grain[:, :, np.newaxis]

    img = img + grain_strength * grain
    img = np.clip(img, 0.0, 1.0)

    # ----------------------------
    # 5. 轻微晕影
    # ----------------------------
    if vignette_strength > 0:
        y, x = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dist = dist / dist.max()

        vignette = 1.0 - vignette_strength * (dist ** 1.8)
        vignette = np.clip(vignette, 0.0, 1.0)
        vignette = vignette[:, :, np.newaxis]

        img = img * vignette
        img = np.clip(img, 0.0, 1.0)

    # ----------------------------
    # 6. 转回 uint8
    # ----------------------------
    out = (img * 255.0).clip(0, 255).astype(np.uint8)
    return out


def process_image(input_path: str, output_path: str = None) -> str:
    """
    读取图片并输出胶片风格图片
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入图像: {input_path}")

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图像读取失败，请检查路径或文件格式。")

    result = apply_film_look(
        image,
        grain_strength=0.12,     # 颗粒强一点可调大
        fade=0.07,               # 褪色程度
        contrast=0.96,           # 柔和一点
        warmth=0.05,             # 暖色
        vignette_strength=0.12   # 轻微晕影
    )

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_film{ext}"

    success = cv2.imwrite(output_path, result)
    if not success:
        raise IOError(f"保存失败: {output_path}")

    return output_path


if __name__ == "__main__":
    # 直接把这里改成你的图片路径
    input_image_path = "Dataset/Origin/r00c0995ft.TIF"

    # 可选，不写就自动生成 input_film.jpg
    output_image_path = "output_film.jpg"

    saved_path = process_image(input_image_path, output_image_path)
    print(f"处理完成，结果已保存到: {saved_path}")