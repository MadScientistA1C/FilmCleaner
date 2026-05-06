import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            ConvBNAct(in_ch, in_ch, kernel_size=3, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel_size=1),
        )


def _mobilenet_v3(name):
    if name == "small":
        try:
            return models.mobilenet_v3_small(weights=None)
        except TypeError:
            return models.mobilenet_v3_small(pretrained=False)
    if name == "large":
        try:
            return models.mobilenet_v3_large(weights=None)
        except TypeError:
            return models.mobilenet_v3_large(pretrained=False)
    raise ValueError("name must be 'small' or 'large'")


class MobileNetV3Encoder(nn.Module):
    def __init__(self, name="small", in_channels=3):
        super().__init__()
        base = _mobilenet_v3(name)
        self.features = base.features

        if in_channels != 3:
            first = self.features[0][0]
            replacement = nn.Conv2d(
                in_channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )
            with torch.no_grad():
                replacement.weight[:, :3].copy_(first.weight)
                if in_channels > 3:
                    mean_weight = first.weight.mean(dim=1, keepdim=True)
                    replacement.weight[:, 3:].copy_(mean_weight.repeat(1, in_channels - 3, 1, 1))
            self.features[0][0] = replacement

    def forward(self, x):
        skips = []
        last_hw = x.shape[-2:]
        for layer in self.features:
            x = layer(x)
            hw = x.shape[-2:]
            if hw != last_hw:
                skips.append(x)
                last_hw = hw
        if not skips or skips[-1] is not x:
            skips.append(x)
        return skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(in_ch + skip_ch, out_ch),
            SeparableConv(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return self.conv(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileUNetV3(nn.Module):
    def __init__(self, encoder_name="small", in_channels=3, classes=1, decoder_channels=(160, 96, 64, 32, 24)):
        super().__init__()
        self.encoder = MobileNetV3Encoder(encoder_name, in_channels=in_channels)
        enc_channels = self._infer_encoder_channels(in_channels)

        decoder_channels = list(decoder_channels)
        needed = len(enc_channels)
        if len(decoder_channels) < needed:
            decoder_channels.extend([decoder_channels[-1]] * (needed - len(decoder_channels)))

        self.blocks = nn.ModuleList()
        current_ch = enc_channels[-1]
        for idx, skip_ch in enumerate(reversed(enc_channels[:-1])):
            out_ch = decoder_channels[idx]
            self.blocks.append(UpBlock(current_ch, skip_ch, out_ch))
            current_ch = out_ch

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SeparableConv(current_ch, decoder_channels[-1]),
        )
        self.head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)

    def _infer_encoder_channels(self, in_channels):
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            channels = [feat.shape[1] for feat in self.encoder(dummy)]
        self.encoder.train(was_training)
        return channels

    def forward(self, x):
        input_hw = x.shape[-2:]
        skips = self.encoder(x)
        y = skips[-1]
        for block, skip in zip(self.blocks, reversed(skips[:-1])):
            y = block(y, skip)
        y = self.final_up(y)
        y = self.head(y)
        if y.shape[-2:] != input_hw:
            y = F.interpolate(y, size=input_hw, mode="bilinear", align_corners=False)
        return y


class InpaintStudentNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(ConvBNAct(in_channels, c), SeparableConv(c, c))
        self.enc2 = nn.Sequential(ConvBNAct(c, c * 2, stride=2), SeparableConv(c * 2, c * 2))
        self.enc3 = nn.Sequential(ConvBNAct(c * 2, c * 4, stride=2), SeparableConv(c * 4, c * 4))
        self.enc4 = nn.Sequential(ConvBNAct(c * 4, c * 8, stride=2), SeparableConv(c * 8, c * 8))
        self.bottleneck = nn.Sequential(
            SeparableConv(c * 8, c * 8),
            SeparableConv(c * 8, c * 8),
        )
        self.dec3 = nn.Sequential(SeparableConv(c * 8 + c * 4, c * 4), SeparableConv(c * 4, c * 4))
        self.dec2 = nn.Sequential(SeparableConv(c * 4 + c * 2, c * 2), SeparableConv(c * 2, c * 2))
        self.dec1 = nn.Sequential(SeparableConv(c * 2 + c, c), SeparableConv(c, c))
        self.head = nn.Sequential(nn.Conv2d(c, 3, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        y = self.bottleneck(e4)
        y = F.interpolate(y, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, e3], dim=1))
        y = F.interpolate(y, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, e2], dim=1))
        y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, e1], dim=1))
        pred = self.head(y)
        image = x[:, :3]
        mask = x[:, 3:4].clamp(0.0, 1.0)
        return image * (1.0 - mask) + pred * mask
