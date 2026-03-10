#!/usr/bin/env python3
"""Convert a PyTorch model checkpoint to safetensors format.

Usage:
    python Convert-PytorchToSafetensors.py Models/Janus-Pro-1B/pytorch_model.bin

Output:
    Models/Janus-Pro-1B/model.safetensors
"""

import sys
import os
from pathlib import Path


def convert(src_path: str) -> None:
    import torch
    from safetensors.torch import save_file

    src = Path(src_path)
    if not src.exists():
        print(f"ERROR: {src} does not exist")
        sys.exit(1)

    dst = src.parent / "model.safetensors"

    print(f"Loading {src} ({src.stat().st_size / 1e9:.2f} GB)...")
    state_dict = torch.load(str(src), map_location="cpu", weights_only=True)

    # Some checkpoints wrap the state dict in a 'model' or 'state_dict' key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # Filter out non-tensor entries
    tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    print(f"Converting {len(tensors)} tensors to safetensors...")
    save_file(tensors, str(dst))

    print(f"Saved: {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
    print(f"Tensors: {len(tensors)}")

    # Show sample tensor names
    sample = list(tensors.keys())[:5]
    for name in sample:
        shape = list(tensors[name].shape)
        dtype = str(tensors[name].dtype)
        print(f"  {name}: {shape} ({dtype})")
    if len(tensors) > 5:
        print(f"  ... and {len(tensors) - 5} more")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/pytorch_model.bin>")
        sys.exit(1)
    convert(sys.argv[1])
