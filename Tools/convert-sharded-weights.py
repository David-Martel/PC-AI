#!/usr/bin/env python3
"""Convert sharded PyTorch model weights to safetensors format.

Handles both single and sharded (pytorch_model-0000X-of-0000N.bin) checkpoints.
For the Rust candle pipeline which requires safetensors format.

Usage:
    python convert-sharded-weights.py [model_dir] [--output model.safetensors]
"""

import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def find_shards(model_dir: Path) -> list[Path]:
    """Find PyTorch weight shards in order."""
    # Check for single file first
    single = model_dir / "pytorch_model.bin"
    if single.exists() and single.stat().st_size > 0:
        return [single]

    # Find sharded files
    shards = sorted(model_dir.glob("pytorch_model-*-of-*.bin"))
    shards = [s for s in shards if s.stat().st_size > 0]
    return shards


def load_sharded_weights(shards: list[Path]) -> dict[str, torch.Tensor]:
    """Load all shards and merge into a single state dict."""
    merged = {}
    for i, shard in enumerate(shards):
        print(f"  Loading shard {i+1}/{len(shards)}: {shard.name} ({shard.stat().st_size / 1e9:.2f} GB)")
        state = torch.load(shard, map_location="cpu", weights_only=True)
        merged.update(state)
        del state
    return merged


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Models/Janus-Pro-7B")
    output_name = "model.safetensors"

    # Parse --output flag
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_name = sys.argv[i + 1]

    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    output_path = model_dir / output_name
    if output_path.exists():
        print(f"Output already exists: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
        sys.exit(0)

    shards = find_shards(model_dir)
    if not shards:
        print(f"No PyTorch weight files found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(shards)} shard(s) in {model_dir}")
    total_size = sum(s.stat().st_size for s in shards)
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print()

    t0 = time.time()
    state_dict = load_sharded_weights(shards)
    load_time = time.time() - t0
    print(f"\nLoaded {len(state_dict)} tensors in {load_time:.1f}s")

    # Show tensor stats
    param_count = sum(t.numel() for t in state_dict.values())
    print(f"Total parameters: {param_count:,} ({param_count / 1e9:.2f}B)")

    # Check for bfloat16 tensors (candle supports bf16)
    dtypes = set(str(t.dtype) for t in state_dict.values())
    print(f"Tensor dtypes: {dtypes}")

    # Convert to contiguous tensors (safetensors requirement)
    print("\nConverting to contiguous tensors...")
    for key in state_dict:
        if not state_dict[key].is_contiguous():
            state_dict[key] = state_dict[key].contiguous()

    # Save as safetensors
    print(f"\nSaving to {output_path}...")
    t1 = time.time()
    save_file(state_dict, str(output_path))
    save_time = time.time() - t1

    final_size = output_path.stat().st_size
    print(f"\n{'='*50}")
    print(f"  CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"  Output:     {output_path}")
    print(f"  Size:       {final_size / 1e9:.2f} GB")
    print(f"  Tensors:    {len(state_dict)}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Load time:  {load_time:.1f}s")
    print(f"  Save time:  {save_time:.1f}s")
    print(f"  Total:      {time.time() - t0:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
