#!/usr/bin/env python3
"""Convert Janus-Pro-1B safetensors weights to GGUF format.

Only the LLaMA backbone (language_model.*) is written to GGUF.
The remaining components (gen_head, gen_embed, gen_aligner, VQ codebook,
vision tower) are left in the original safetensors file.

Usage:
    # From repo root, using the AI-Media venv (has safetensors + gguf):
    AI-Media\\.venv\\Scripts\\python.exe Tools\\Convert-JanusToGGUF.py

    # Or with uv from AI-Media directory:
    cd AI-Media && uv run python ../Tools/Convert-JanusToGGUF.py

Output:
    Models/Janus-Pro-1B/janus-pro-1b-llama-f16.gguf   (BF16 -> F16 converted)
    Models/Janus-Pro-1B/janus-pro-1b-llama-q4_k_m.gguf  (if llama-quantize found)
    Models/Janus-Pro-1B/janus-pro-1b-llama-q8_0.gguf    (if llama-quantize found)
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 stdout on Windows (avoid cp1252 codec errors for non-ASCII chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

_MISSING: list[str] = []

# numpy — required for tensor dtype handling
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]
    _MISSING.append("numpy")

# torch — required for dtype comparison when converting BF16 tensors
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    _MISSING.append("torch")

# safetensors — required for reading model shards
try:
    from safetensors import safe_open as _safe_open_impl
    safe_open = _safe_open_impl
except ImportError:
    safe_open = None  # type: ignore[assignment]
    _MISSING.append("safetensors")

# gguf — required for writing GGUF output
try:
    import gguf as _gguf_module
    from gguf import GGUFWriter as _GGUFWriterImpl, GGUFValueType
    from gguf.constants import GGMLQuantizationType as _GGMLQuantizationTypeImpl
    gguf = _gguf_module
    GGUFWriter = _GGUFWriterImpl
    GGMLQuantizationType = _GGMLQuantizationTypeImpl
except ImportError:
    gguf = None  # type: ignore[assignment]
    GGUFWriter = None  # type: ignore[assignment,misc]
    GGUFValueType = None  # type: ignore[assignment]
    GGMLQuantizationType = None  # type: ignore[assignment,misc]
    _MISSING.append("gguf")

if _MISSING:
    print(
        "ERROR: Missing packages: "
        + ", ".join(_MISSING)
        + "\nInstall with: uv pip install "
        + " ".join(_MISSING),
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Models" / "Janus-Pro-1B"

# Defaults from the LLaMA architecture (not in Janus config.json)
DEFAULT_ROPE_THETA = 10000.0
DEFAULT_RMS_NORM_EPS = 1e-5

# LLaMA GGUF name prefix for each layer
_LAYER_PREFIX = "language_model.model.layers."
_ATTN_PREFIX = "self_attn."
_MLP_PREFIX = "mlp."

# ---------------------------------------------------------------------------
# Weight name mapping: safetensors → GGUF tensor name
# ---------------------------------------------------------------------------

def _gguf_name(st_name: str) -> str | None:
    """Map a safetensors tensor name to its GGUF equivalent.

    Returns None if the tensor should be skipped (not part of LLaMA backbone).

    GGUF naming convention follows llama.cpp gguf-py conventions:
      token_embd.weight, blk.N.{attn_q,attn_k,attn_v,attn_output}.weight,
      blk.N.{ffn_gate,ffn_up,ffn_down}.weight, blk.N.{attn_norm,ffn_norm}.weight,
      output_norm.weight, output.weight
    """
    if not st_name.startswith("language_model."):
        return None

    tail = st_name[len("language_model."):]

    # lm_head (unembedding)
    if tail == "lm_head.weight":
        return "output.weight"

    # model.* namespace
    if not tail.startswith("model."):
        return None

    body = tail[len("model."):]

    if body == "embed_tokens.weight":
        return "token_embd.weight"

    if body == "norm.weight":
        return "output_norm.weight"

    if not body.startswith("layers."):
        return None

    # layers.N.* — parse layer index
    rest = body[len("layers."):]
    dot = rest.index(".")
    layer_idx = int(rest[:dot])
    inner = rest[dot + 1:]

    # Attention projections
    _attn_map = {
        "self_attn.q_proj.weight": f"blk.{layer_idx}.attn_q.weight",
        "self_attn.k_proj.weight": f"blk.{layer_idx}.attn_k.weight",
        "self_attn.v_proj.weight": f"blk.{layer_idx}.attn_v.weight",
        "self_attn.o_proj.weight": f"blk.{layer_idx}.attn_output.weight",
    }
    if inner in _attn_map:
        return _attn_map[inner]

    # MLP projections
    _mlp_map = {
        "mlp.gate_proj.weight": f"blk.{layer_idx}.ffn_gate.weight",
        "mlp.up_proj.weight":   f"blk.{layer_idx}.ffn_up.weight",
        "mlp.down_proj.weight": f"blk.{layer_idx}.ffn_down.weight",
    }
    if inner in _mlp_map:
        return _mlp_map[inner]

    # Layer norms
    _norm_map = {
        "input_layernorm.weight":         f"blk.{layer_idx}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blk.{layer_idx}.ffn_norm.weight",
    }
    if inner in _norm_map:
        return _norm_map[inner]

    return None


# ---------------------------------------------------------------------------
# Safetensors shard discovery
# ---------------------------------------------------------------------------

def _find_shards(model_dir: Path) -> list[Path]:
    """Return ordered list of safetensors shard paths.

    Handles both single-file (model.safetensors) and sharded
    (model-00001-of-00002.safetensors, ...) layouts.
    """
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    # Multi-shard: model-NNNNN-of-NNNNN.safetensors
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if shards:
        return shards

    raise FileNotFoundError(
        f"No safetensors file(s) found in {model_dir}. "
        "Expected 'model.safetensors' or 'model-NNNNN-of-NNNNN.safetensors'."
    )


def _iter_lm_tensors(
    shards: list[Path],
) -> Iterator[tuple[str, str, "np.ndarray"]]:
    """Yield (safetensors_name, gguf_name, numpy_array) for LLaMA backbone tensors.

    BF16 tensors are converted to F16 because llama.cpp GGUF does not have
    a native BF16 quantization type — F16 is the standard full-precision type
    used as the source for subsequent quantization.
    """
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            lm_keys = [k for k in all_keys if k.startswith("language_model.")]
            for st_name in lm_keys:
                gguf_name = _gguf_name(st_name)
                if gguf_name is None:
                    continue
                tensor = f.get_tensor(st_name)
                # Convert bfloat16 → float16 (GGUF F16).
                # torch.dtype has no .name; compare directly with torch.bfloat16.
                if tensor.dtype == torch.bfloat16:
                    arr = tensor.to(dtype=torch.float16).numpy()
                else:
                    arr = tensor.numpy()
                yield st_name, gguf_name, arr


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_arch_params(model_dir: Path) -> dict:
    """Load architecture parameters from config.json."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found at {cfg_path}")

    with cfg_path.open() as fh:
        cfg = json.load(fh)

    lc = cfg.get("language_config", {})
    if not lc:
        raise ValueError("config.json missing 'language_config' section")

    # Some Janus configs (e.g., 7B) omit detailed architecture params in
    # language_config — infer them from safetensors tensor shapes if missing.
    params = {
        "num_hidden_layers":       lc["num_hidden_layers"],
        "vocab_size":              lc["vocab_size"],
        "max_position_embeddings": lc["max_position_embeddings"],
        "rope_theta":              lc.get("rope_theta", DEFAULT_ROPE_THETA),
        "rms_norm_eps":            lc.get("rms_norm_eps", DEFAULT_RMS_NORM_EPS),
    }

    if "hidden_size" in lc:
        params["hidden_size"] = lc["hidden_size"]
        params["num_attention_heads"] = lc["num_attention_heads"]
        params["num_key_value_heads"] = lc.get("num_key_value_heads", lc["num_attention_heads"])
        params["intermediate_size"] = lc["intermediate_size"]
    else:
        # Infer from tensor shapes in safetensors
        shard_paths = sorted(model_dir.glob("*.safetensors"))
        if not shard_paths:
            raise ValueError("No safetensors files found to infer architecture from")
        from safetensors import safe_open  # type: ignore
        with safe_open(str(shard_paths[0]), framework="pt") as st:
            embed_shape = st.get_tensor("language_model.model.embed_tokens.weight").shape
            q_shape = st.get_tensor("language_model.model.layers.0.self_attn.q_proj.weight").shape
            k_shape = st.get_tensor("language_model.model.layers.0.self_attn.k_proj.weight").shape
            gate_shape = st.get_tensor("language_model.model.layers.0.mlp.gate_proj.weight").shape
        hidden_size = embed_shape[1]  # [vocab, hidden]
        params["hidden_size"] = hidden_size
        params["num_attention_heads"] = q_shape[0] // (hidden_size // (q_shape[0] // (q_shape[0] // 128 if q_shape[0] > 128 else 1)))
        # Simpler: head_dim is typically 128, so num_heads = q_proj_out / head_dim
        head_dim = 128
        params["num_attention_heads"] = q_shape[0] // head_dim
        params["num_key_value_heads"] = k_shape[0] // head_dim
        params["intermediate_size"] = gate_shape[0]
        print(f"Inferred from tensors: hidden={hidden_size}, heads={params['num_attention_heads']}, "
              f"kv_heads={params['num_key_value_heads']}, intermediate={params['intermediate_size']}")

    return params


# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

def _write_gguf(
    output_path: Path,
    params: dict,
    shards: list[Path],
    dry_run: bool = False,
) -> None:
    """Write the GGUF file with metadata and F16 tensors."""

    print(f"\n--- Writing GGUF: {output_path} ---")
    print(f"Architecture parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    if dry_run:
        print("[dry-run] Skipping actual write.")
        return

    writer = GGUFWriter(str(output_path), arch="llama")

    # --- Metadata ---
    # NOTE: GGUFWriter.__init__ already writes general.architecture via the arch=
    # constructor argument. Do NOT call add_architecture() — it would duplicate
    # the key and trigger a "Duplicated key name" warning.
    writer.add_name("Janus-Pro-1B-LLaMA")
    writer.add_description(
        "LLaMA backbone extracted from deepseek-ai/Janus-Pro-1B multimodal model. "
        "Non-LLaMA components (vision tower, VQ codebook, gen_head, aligners) "
        "are retained in the original safetensors file."
    )
    writer.add_context_length(params["max_position_embeddings"])
    writer.add_embedding_length(params["hidden_size"])
    writer.add_block_count(params["num_hidden_layers"])
    writer.add_feed_forward_length(params["intermediate_size"])
    writer.add_head_count(params["num_attention_heads"])
    writer.add_head_count_kv(params["num_key_value_heads"])
    writer.add_rope_freq_base(params["rope_theta"])
    writer.add_layer_norm_rms_eps(params["rms_norm_eps"])
    writer.add_vocab_size(params["vocab_size"])
    # Head dimension = hidden_size / num_attention_heads
    head_dim = params["hidden_size"] // params["num_attention_heads"]
    writer.add_rope_dimension_count(head_dim)
    # File type: F16 (pass integer value; GGMLQuantizationType is an IntEnum)
    writer.add_file_type(GGMLQuantizationType.F16.value)
    # Base model provenance — source_id is 0-based index into add_base_model_count entries
    writer.add_base_model_count(1)
    writer.add_base_model_name(0, "deepseek-ai/Janus-Pro-1B")
    writer.add_base_model_repo_url(0, "https://huggingface.co/deepseek-ai/Janus-Pro-1B")

    # --- Tensors ---
    total_params = 0
    tensor_count = 0
    t_start = time.perf_counter()

    for st_name, gguf_name, arr in _iter_lm_tensors(shards):
        writer.add_tensor(gguf_name, arr)
        total_params += arr.size
        tensor_count += 1
        print(
            f"  [{tensor_count:3d}] {gguf_name:<50s}  "
            f"shape={list(arr.shape)}  dtype={arr.dtype}"
        )

    elapsed = time.perf_counter() - t_start
    print(
        f"\nTensors loaded: {tensor_count}  "
        f"({total_params/1e6:.1f} M params)  "
        f"in {elapsed:.1f}s"
    )

    print("Writing GGUF header + tensor data...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = output_path.stat().st_size / 1e9
    print(f"Written: {output_path}  ({size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# Optional quantization via llama-quantize
# ---------------------------------------------------------------------------

_QUANT_TYPES = [
    ("Q4_K_M", "janus-pro-1b-llama-q4_k_m.gguf"),
    ("Q8_0",   "janus-pro-1b-llama-q8_0.gguf"),
]


def _quantize(f16_path: Path, model_dir: Path) -> None:
    """Run llama-quantize to produce Q4_K_M and Q8_0 variants."""
    quantize_exe = shutil.which("llama-quantize") or shutil.which("llama.cpp-quantize")
    if quantize_exe is None:
        # Search common build locations
        candidates = [
            REPO_ROOT / "bin" / "llama-quantize.exe",
            REPO_ROOT / "bin" / "llama-quantize",
            Path("C:/tools/llama.cpp/llama-quantize.exe"),
        ]
        for c in candidates:
            if c.exists():
                quantize_exe = str(c)
                break

    if quantize_exe is None:
        print(
            "\nllama-quantize not found in PATH or known locations. "
            "Skipping quantization.\n"
            "To quantize manually:\n"
            f"  llama-quantize {f16_path} <output.gguf> Q4_K_M"
        )
        return

    print(f"\nUsing quantizer: {quantize_exe}")

    for quant_type, out_name in _QUANT_TYPES:
        out_path = model_dir / out_name
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists (delete to re-quantize)")
            continue
        cmd = [quantize_exe, str(f16_path), str(out_path), quant_type]
        print(f"\nQuantizing {quant_type}: {' '.join(cmd)}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"ERROR: llama-quantize exited {result.returncode} for {quant_type}")
        else:
            size_gb = out_path.stat().st_size / 1e9
            print(
                f"Done ({elapsed:.0f}s): {out_path.name}  ({size_gb:.2f} GB)"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing Janus-Pro-1B model files "
             f"(default: {MODEL_DIR})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GGUF file path. Defaults to <model-dir>/janus-pro-1b-llama-f16.gguf",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip post-conversion quantization step",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without writing any files",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    model_dir: Path = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    # Derive output filename from model directory name
    model_name = model_dir.name.lower().replace(" ", "-")
    output_path: Path = args.output or (model_dir / f"{model_name}-llama-f16.gguf")
    output_path = output_path.resolve()

    print("=" * 72)
    print("Janus-Pro-1B -> GGUF Converter")
    print("=" * 72)
    print(f"Model dir : {model_dir}")
    print(f"Output    : {output_path}")

    # --- Idempotency check ---
    if output_path.exists() and not args.dry_run:
        size_gb = output_path.stat().st_size / 1e9
        print(
            f"\n[skip] {output_path.name} already exists ({size_gb:.2f} GB). "
            "Delete it to re-convert."
        )
        if not args.no_quantize:
            _quantize(output_path, model_dir)
        return 0

    # --- Load config ---
    params = _load_arch_params(model_dir)

    # --- Discover shards ---
    shards = _find_shards(model_dir)
    total_shard_size = sum(s.stat().st_size for s in shards) / 1e9
    print(
        f"\nInput shards: {len(shards)} file(s), "
        f"{total_shard_size:.2f} GB total"
    )
    for s in shards:
        print(f"  {s.name}")

    # --- Write GGUF ---
    _write_gguf(output_path, params, shards, dry_run=args.dry_run)

    # --- Quantize ---
    if not args.no_quantize and not args.dry_run:
        _quantize(output_path, model_dir)

    print("\nConversion complete.")
    print(f"F16 GGUF : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
