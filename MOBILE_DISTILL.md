# Mobile dust removal distillation

This project now keeps the original DeepLabV3+ training code intact and adds a
mobile pipeline around it:

1. Train or reuse the existing teacher segmentation checkpoint.
2. Distill teacher masks into `MobileUNetV3`, a UNet-style decoder with a
   MobileNetV3 encoder.
3. Generate LAMA teacher outputs offline with the image and dust mask pairs.
4. Distill those LAMA outputs into `InpaintStudentNet`, a light 4-channel
   inpainting network that takes `RGB + mask`.
5. Export the two student models as TorchScript or ONNX.

## Segmentation student

```powershell
.\.venv\Scripts\python.exe train_mobile_seg_distill.py `
  --teacher checkpoints\best_model.pth `
  --encoder small `
  --image-size 512 `
  --batch-size 8 `
  --epochs 80
```

Outputs:

- `checkpoints/mobile_seg_best.pth`
- `checkpoints/mobile_seg_latest.pth`

Use `--no-teacher` to train only from hard masks.

## LAMA teacher targets

The inpainting distillation script expects LAMA-cleaned images in one directory.
For each source image like:

```text
Dataset/FakeFilm/patches/images/sample.png
Dataset/FakeFilm/patches/masks/sample_mask.png
```

the target directory may contain any one of:

```text
sample.png
sample_lama.png
sample_target.png
sample_clean.png
```

## Inpainting student

```powershell
.\.venv\Scripts\python.exe train_inpaint_distill.py `
  --target-dir Dataset\FakeFilm\lama_outputs `
  --image-size 512 `
  --batch-size 8 `
  --epochs 80
```

Outputs:

- `checkpoints/mobile_inpaint_best.pth`
- `checkpoints/mobile_inpaint_latest.pth`

`InpaintStudentNet` preserves pixels outside the mask and only replaces masked
regions, which is useful for film dust removal on mobile.

## Export

TorchScript:

```powershell
.\.venv\Scripts\python.exe export_mobile.py `
  --model seg `
  --checkpoint checkpoints\mobile_seg_best.pth `
  --format torchscript `
  --output exports\mobile_seg.pt

.\.venv\Scripts\python.exe export_mobile.py `
  --model inpaint `
  --checkpoint checkpoints\mobile_inpaint_best.pth `
  --format torchscript `
  --output exports\mobile_inpaint.pt
```

ONNX:

```powershell
.\.venv\Scripts\python.exe export_mobile.py `
  --model seg `
  --checkpoint checkpoints\mobile_seg_best.pth `
  --format onnx `
  --output exports\mobile_seg.onnx
```

For mobile inference, run the segmentation model first, threshold its logits
with sigmoid at `0.5`, then concatenate the original RGB tensor and binary mask
as the 4-channel input to the inpainting model.
