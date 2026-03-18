#!/usr/bin/env python3
"""Convert RealESRGAN x4plus PyTorch (.pth) weights to ONNX format.

Self-contained script — does NOT require basicsr. The RRDBNet architecture
is defined inline so the only dependencies are torch and onnx.

Usage:
    python Tools/Convert-RealESRGAN-to-ONNX.py \
        --input  Models/RealESRGAN/RealESRGAN_x4plus.pth \
        --output Models/RealESRGAN/RealESRGAN_x4.onnx

The exported ONNX graph uses dynamic H/W axes so it accepts arbitrary
input resolutions at inference time. Opset 17 is used for broad
compatibility with ONNX Runtime >= 1.14.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture (from basicsr.archs.rrdbnet_arch, inlined to avoid dep)
# ---------------------------------------------------------------------------

def default_init_weights(module_list: list[nn.Module], scale: float = 1.0) -> None:
    """Kaiming-normal init scaled by *scale* (standard ESRGAN practice)."""
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used inside each RRDB."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block — the core building block of ESRGAN."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Generator network for Real-ESRGAN (x4 variant).

    Architecture: conv_first -> 23 RRDB blocks -> conv_body (skip) ->
    2x nearest upsample + conv -> conv_hr -> conv_last.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, output_path: str, opset: int = 17) -> None:
    print(f"Loading weights from {input_path} ...")
    state = torch.load(input_path, map_location="cpu", weights_only=True)

    # The official .pth wraps weights under 'params_ema' or 'params'.
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    # else: assume bare state_dict

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters ({param_count * 4 / 1e6:.1f} MB fp32)")

    # Dummy input — 64x64 is small enough to be fast.
    dummy = torch.randn(1, 3, 64, 64)

    print(f"Exporting ONNX (opset {opset}, dynamic H/W) -> {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            opset_version=opset,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "out_height", 3: "out_width"},
            },
        )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. ONNX model saved: {output_path} ({size_mb:.1f} MB)")

    # Quick sanity check with ONNX Runtime if available
    try:
        import onnxruntime as ort  # noqa: F811

        sess = ort.InferenceSession(output_path)
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"Verification OK — input: {inp.name} {inp.shape}, output: {out.name} {out.shape}")
    except ImportError:
        print("(onnxruntime not installed — skipping verification)")
    except Exception as e:
        print(f"Warning: verification failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RealESRGAN x4plus .pth to ONNX with dynamic input axes."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Models/RealESRGAN/RealESRGAN_x4plus.pth",
        help="Path to the PyTorch .pth weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Models/RealESRGAN/RealESRGAN_x4.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    convert(args.input, args.output, args.opset)


if __name__ == "__main__":
    main()
