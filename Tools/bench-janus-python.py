#!/usr/bin/env python3
"""Comprehensive Janus-Pro benchmark: 1B vs 7B, GPU vs CPU, with detailed metrics.

Usage:
    python bench-janus-python.py [--model-1b PATH] [--model-7b PATH] [--output-dir PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
        "vram_free_gb": round(torch.cuda.mem_get_info()[0] / 1e9, 1),
        "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        "cuda_version": torch.version.cuda,
    }


def benchmark_model(model_path: str, prompt: str, output_path: str, device: str = "cuda"):
    """Run a single benchmark and return metrics."""
    import numpy as np
    from PIL import Image

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load model
    t0 = time.time()
    from janus.models import VLChatProcessor, MultiModalityCausalLM

    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = MultiModalityCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=False,
        use_safetensors=False,
    ).to(device).eval()
    load_time = time.time() - t0

    vram_after_load = 0
    if device == "cuda":
        torch.cuda.synchronize()
        vram_after_load = torch.cuda.memory_allocated() / 1e6

    # Generate
    t1 = time.time()
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt="",
    )
    prompt_text = sft_format + processor.image_start_tag
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    uncond_ids = tokenizer.encode("", return_tensors="pt").to(device)

    cfg_weight = 5.0
    temperature = 1.0
    image_token_num = 576
    lm = model.language_model.model

    first_token_time = 0
    with torch.no_grad():
        cond_embeds = lm.embed_tokens(input_ids)
        uncond_embeds = lm.embed_tokens(uncond_ids)
        cond_len = cond_embeds.shape[1]
        uncond_len = uncond_embeds.shape[1]
        if uncond_len < cond_len:
            pad = torch.zeros(1, cond_len - uncond_len, cond_embeds.shape[2],
                              device=device, dtype=dtype)
            uncond_embeds = torch.cat([pad, uncond_embeds], dim=1)

        batched_embeds = torch.cat([cond_embeds, uncond_embeds], dim=0)
        prefill_out = lm(inputs_embeds=batched_embeds, use_cache=True, return_dict=True)
        hidden = prefill_out.last_hidden_state[:, -1:, :]
        batched_past = prefill_out.past_key_values

        cond_logits = model.gen_head(hidden[0:1])
        uncond_logits = model.gen_head(hidden[1:2])
        logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
        probs = torch.softmax(logits.squeeze() / temperature, dim=-1)
        token = torch.multinomial(probs, 1)
        generated_tokens = [token.squeeze()]

        if device == "cuda":
            torch.cuda.synchronize()
        first_token_time = time.time() - t1

        for i in range(1, image_token_num):
            last_id = generated_tokens[-1].unsqueeze(0).unsqueeze(0)
            token_embed = model.prepare_gen_img_embeds(last_id)
            batched_embed = token_embed.expand(2, -1, -1)
            out = lm(inputs_embeds=batched_embed, past_key_values=batched_past,
                     use_cache=True, return_dict=True)
            hidden = out.last_hidden_state[:, -1:, :]
            batched_past = out.past_key_values
            cond_logits = model.gen_head(hidden[0:1])
            uncond_logits = model.gen_head(hidden[1:2])
            logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
            probs = torch.softmax(logits.squeeze() / temperature, dim=-1)
            token = torch.multinomial(probs, 1)
            generated_tokens.append(token.squeeze())

    if device == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - t1
    tps = image_token_num / gen_time

    # Decode
    all_tokens = torch.stack(generated_tokens).unsqueeze(0)
    decoded = model.gen_vision_model.decode_code(all_tokens, shape=[1, 8, 24, 24])
    decoded = decoded.detach().float().cpu().clamp(0, 1)
    img_array = (decoded[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_array)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    peak_vram = 0
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e6

    # Cleanup
    del model, processor, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return {
        "model_path": model_path,
        "device": device,
        "dtype": str(dtype),
        "load_time_s": round(load_time, 1),
        "gen_time_s": round(gen_time, 1),
        "first_token_s": round(first_token_time, 1),
        "tok_per_sec": round(tps, 1),
        "image_tokens": image_token_num,
        "output_path": output_path,
        "output_size_bytes": Path(output_path).stat().st_size,
        "vram_after_load_mb": round(vram_after_load),
        "peak_vram_mb": round(peak_vram),
        "prompt": prompt,
        "total_time_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Janus-Pro benchmark")
    parser.add_argument("--model-1b", default="Models/Janus-Pro-1B")
    parser.add_argument("--model-7b", default="Models/Janus-Pro-7B")
    parser.add_argument("--output-dir", default="Reports/media/bench")
    parser.add_argument("--skip-1b", action="store_true")
    parser.add_argument("--skip-7b", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'N/A')}")
    print(f"VRAM: {gpu_info.get('vram_total_gb', 0)} GB")
    print(f"CUDA: {gpu_info.get('cuda_version', 'N/A')}")
    print(f"PyTorch: {torch.__version__}")
    print()

    prompts = [
        "A glowing blue circuit board floating in space, digital art, 8k",
        "A majestic wolf standing on a mountain peak at sunset, oil painting",
    ]

    results = {"gpu_info": gpu_info, "pytorch_version": torch.__version__, "benchmarks": []}

    # 1B benchmark
    if not args.skip_1b and Path(args.model_1b).exists():
        print("=" * 60)
        print("  JANUS-PRO-1B BENCHMARK")
        print("=" * 60)
        for i, prompt in enumerate(prompts):
            out = str(output_dir / f"python_1b_gpu_{i+1}.png")
            print(f"\n  [{i+1}/{len(prompts)}] {prompt[:50]}...")
            r = benchmark_model(args.model_1b, prompt, out)
            results["benchmarks"].append(r)
            print(f"    Load: {r['load_time_s']}s | Gen: {r['gen_time_s']}s | {r['tok_per_sec']} tok/s | VRAM: {r['peak_vram_mb']}MB")

    # 7B benchmark
    if not args.skip_7b and Path(args.model_7b).exists():
        # Check if weights exist (not 0-byte placeholders)
        has_weights = False
        pt_files = list(Path(args.model_7b).glob("pytorch_model*.bin"))
        for f in pt_files:
            if f.stat().st_size > 1000:
                has_weights = True
                break
        if not has_weights:
            print(f"\n  Skipping 7B: weights not downloaded yet")
        else:
            print("\n" + "=" * 60)
            print("  JANUS-PRO-7B BENCHMARK")
            print("=" * 60)
            for i, prompt in enumerate(prompts):
                out = str(output_dir / f"python_7b_gpu_{i+1}.png")
                print(f"\n  [{i+1}/{len(prompts)}] {prompt[:50]}...")
                try:
                    r = benchmark_model(args.model_7b, prompt, out)
                    results["benchmarks"].append(r)
                    print(f"    Load: {r['load_time_s']}s | Gen: {r['gen_time_s']}s | {r['tok_per_sec']} tok/s | VRAM: {r['peak_vram_mb']}MB")
                except Exception as e:
                    print(f"    FAILED: {e}")
                    results["benchmarks"].append({"model_path": args.model_7b, "error": str(e), "prompt": prompt})

    # Summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)

    ok_results = [r for r in results["benchmarks"] if "tok_per_sec" in r]
    for r in ok_results:
        model_name = "1B" if "1B" in r["model_path"] else "7B"
        print(f"  {model_name} | {r['tok_per_sec']:>6.1f} tok/s | {r['gen_time_s']:>5.1f}s gen | {r['peak_vram_mb']:>5}MB VRAM | {r['prompt'][:40]}...")

    # Save report
    report_path = str(output_dir / "python-benchmark-report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
