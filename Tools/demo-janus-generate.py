#!/usr/bin/env python3
"""Janus-Pro image generation demo using the DeepSeek Janus pipeline.

Optimized for GPU with:
  - Batched CFG (cond+uncond in single forward pass)
  - bfloat16 inference
  - torch.compile() for kernel fusion

Usage:
    python demo-janus-generate.py [model_path] [prompt] [output_path]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "Models/Janus-Pro-1B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "A glowing blue circuit board floating in space, digital art, 8k"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "Reports/media/demo_output.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print()

    # Load model
    t0 = time.time()
    print("Loading model...")

    from janus.models import VLChatProcessor, MultiModalityCausalLM

    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        use_safetensors=False,  # Our safetensors lacks metadata; use pytorch_model.bin
    ).to(device).eval()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    if device == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    # Generate image
    t1 = time.time()
    print(f"\nGenerating image for: '{prompt}'")

    # Janus generation format
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt_text = sft_format + processor.image_start_tag

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    uncond_ids = tokenizer.encode("", return_tensors="pt").to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    # Autoregressive generation of image tokens with batched CFG
    cfg_weight = 5.0
    temperature = 1.0
    image_token_num = 576  # 24x24 grid for 384px image

    # Use inner LlamaModel (returns last_hidden_state, not logits)
    lm = model.language_model.model

    # torch.compile requires Triton (Linux-only). Use eager mode on Windows.
    lm_compiled = lm
    compiled = False
    if device == "cuda":
        torch.set_float32_matmul_precision('high')

    with torch.no_grad():
        # Initial embeddings
        cond_embeds = lm.embed_tokens(input_ids)     # [1, cond_len, D]
        uncond_embeds = lm.embed_tokens(uncond_ids)   # [1, uncond_len, D]

        # Pad uncond to match cond length for batched KV cache
        cond_len = cond_embeds.shape[1]
        uncond_len = uncond_embeds.shape[1]
        if uncond_len < cond_len:
            pad = torch.zeros(1, cond_len - uncond_len, cond_embeds.shape[2],
                              device=device, dtype=dtype)
            uncond_embeds = torch.cat([pad, uncond_embeds], dim=1)

        # Batch both branches: [2, seq_len, D]
        batched_embeds = torch.cat([cond_embeds, uncond_embeds], dim=0)

        # Single forward pass for prefill
        prefill_out = lm_compiled(inputs_embeds=batched_embeds, use_cache=True, return_dict=True)

        # Extract hidden states from batched output
        hidden = prefill_out.last_hidden_state[:, -1:, :]  # [2, 1, D]
        cond_hidden = hidden[0:1]
        uncond_hidden = hidden[1:2]
        batched_past = prefill_out.past_key_values

        # First token from combined logits
        cond_logits = model.gen_head(cond_hidden)
        uncond_logits = model.gen_head(uncond_hidden)
        logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
        probs = torch.softmax(logits.squeeze() / temperature, dim=-1)
        token = torch.multinomial(probs, 1)
        generated_tokens = [token.squeeze()]

        if device == "cuda":
            torch.cuda.synchronize()
        first_token_time = time.time() - t1
        print(f"  First token: {first_token_time:.1f}s (prompt processing)")

        # Remaining tokens: fully batched cond+uncond in single forward pass
        for i in range(1, image_token_num):
            # Get embedding for last generated token
            last_id = generated_tokens[-1].unsqueeze(0).unsqueeze(0)  # [1, 1]
            token_embed = model.prepare_gen_img_embeds(last_id)       # [1, 1, D]

            # Duplicate for both branches: [2, 1, D]
            batched_embed = token_embed.expand(2, -1, -1)

            # Single forward pass for both branches (KV cache already batched)
            out = lm_compiled(
                inputs_embeds=batched_embed,
                past_key_values=batched_past,
                use_cache=True,
                return_dict=True,
            )

            # Extract hidden states and update cache
            hidden = out.last_hidden_state[:, -1:, :]  # [2, 1, D]
            batched_past = out.past_key_values

            # CFG: combine cond and uncond logits
            cond_logits = model.gen_head(hidden[0:1])
            uncond_logits = model.gen_head(hidden[1:2])
            logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)

            # Sample
            probs = torch.softmax(logits.squeeze() / temperature, dim=-1)
            token = torch.multinomial(probs, 1)
            generated_tokens.append(token.squeeze())

            if (i + 1) % 100 == 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t1
                tps = (i + 1) / elapsed
                print(f"  Token {i+1}/{image_token_num} ({tps:.1f} tok/s)")

    if device == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - t1
    tps = image_token_num / gen_time
    print(f"\nGenerated {image_token_num} tokens in {gen_time:.1f}s ({tps:.1f} tok/s)")

    # Decode tokens to image
    print("Decoding tokens to image...")
    all_tokens = torch.stack(generated_tokens).unsqueeze(0)  # [1, 576]
    decoded = model.gen_vision_model.decode_code(
        all_tokens,
        shape=[1, 8, 24, 24]  # batch, channels, h, w
    )

    # Convert to PIL image
    decoded = decoded.detach().float().cpu().clamp(0, 1)
    img_array = (decoded[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_array)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    file_size = Path(output_path).stat().st_size

    print(f"\n{'='*50}")
    print(f"  IMAGE GENERATED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"  Output:      {output_path}")
    print(f"  Size:        {file_size:,} bytes")
    print(f"  Dimensions:  {image.size[0]}x{image.size[1]}")
    print(f"  Model load:  {load_time:.1f}s")
    print(f"  Generation:  {gen_time:.1f}s")
    print(f"  Throughput:  {tps:.1f} tok/s")
    print(f"  Compiled:    {compiled}")
    print(f"  Total time:  {time.time() - t0:.1f}s")
    if device == "cuda":
        print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
