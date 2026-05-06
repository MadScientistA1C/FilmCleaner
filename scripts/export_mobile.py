import argparse
from pathlib import Path

import torch

from lamalocal.mobile_models import InpaintStudentNet, MobileUNetV3


def load_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def build_model(args):
    if args.model == "seg":
        return MobileUNetV3(encoder_name=args.encoder, in_channels=3, classes=1), 3
    return InpaintStudentNet(in_channels=4, base_channels=args.base_channels), 4


def export_torchscript(model, dummy, output):
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output))


def export_onnx(model, dummy, output, input_name):
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=[input_name],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            input_name: {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export mobile dust removal models.")
    parser.add_argument("--model", choices=["seg", "inpaint"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", choices=["small", "large"], default="small")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    model, in_channels = build_model(args)
    model = load_state(model, args.checkpoint, device)
    dummy = torch.zeros(1, in_channels, args.image_size, args.image_size, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "torchscript":
        export_torchscript(model, dummy, output)
    else:
        input_name = "image" if args.model == "seg" else "image_mask"
        export_onnx(model, dummy, output, input_name)

    print(f"exported {args.model} {args.format}: {output}")


if __name__ == "__main__":
    main()
