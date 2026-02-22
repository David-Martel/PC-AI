import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, VLChatProcessor


# --- Configuration & Constants ---
@dataclass
class PipelineConfig:
    model_path: str = "deepseek-ai/Janus-Pro-7B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    parallel_size: int = 1  # Number of images to generate per batch
    img_size: int = 384  # Native Janus resolution
    patch_size: int = 16  # Fixed patch size for the tokenizer


# --- Utility: Metadata Logging ---
class MetadataManager:
    """Handles saving of generation artifacts and traceable documentation."""

    def __init__(self, base_output_dir: str = "janus_outputs"):
        self.base_dir = base_output_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_run(self, image: Image.Image, prompt: str, params: Dict[str, Any]) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename_base = f"{timestamp}_janus"

        # Save Image
        img_path = os.path.join(self.base_dir, f"{filename_base}.png")
        image.save(img_path)

        # Save Metadata (Exhaustive)
        meta_path = os.path.join(self.base_dir, f"{filename_base}.json")
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "file_path": img_path,
            "parameters": params,
            "system_info": {
                "device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU",
                "torch_version": torch.__version__,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        return img_path


# --- Utility: Upscaling (Optional) ---
class ImageUpscaler:
    """Wrapper for RealESRGAN to upscale the native 384px images."""

    def __init__(self, device: str):
        self.model = None
        self.device = device
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Initialize 4x upscale model
            model_arch = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            self.model = RealESRGANer(
                scale=4,
                model_path=None,  # Downloads automatically if None
                model=model_arch,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if "cuda" in device else False,
                device=device,
            )
            print("[Info] RealESRGAN initialized successfully.")
        except ImportError:
            print(
                "[Warning] RealESRGAN or BasicSR not installed. Upscaling will strictly use BICUBIC resize."
            )

    def upscale(self, image: Image.Image, scale: int = 4) -> Image.Image:
        if self.model:
            # Convert PIL to CV2 format (RGB -> BGR)
            img_np = np.array(image)
            img_cv2 = img_np[:, :, ::-1]  # RGB to BGR

            output, _ = self.model.enhance(img_cv2, outscale=scale)

            # Convert back to PIL (BGR -> RGB)
            output = output[:, :, ::-1]
            return Image.fromarray(output)
        else:
            # Fallback
            new_size = (image.width * scale, image.height * scale)
            return image.resize(new_size, Image.Resampling.BICUBIC)


# --- Core Pipeline ---
class JanusPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        print(f"[Init] Loading Janus-Pro from {self.cfg.model_path}...")

        try:
            self.processor = VLChatProcessor.from_pretrained(
                self.cfg.model_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_path,
                    trust_remote_code=True,
                    torch_dtype=self.cfg.dtype,
                )
                .to(self.cfg.device)
                .eval()
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Janus model. Ensure you have internet access for HuggingFace.\nError: {e}"
            )

    @torch.inference_mode()
    def generate(
        self, prompt: str, temperature: float = 1.0, cfg_weight: float = 5.0
    ) -> Image.Image:
        """
        Executes the Janus generation loop.
        Janus generates images by predicting 'image tokens' after a text prompt.
        """

        # 1. Prepare Conversation Context
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 2. Format Inputs
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt_ids = self.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(prompt_ids).to(self.cfg.device)

        # 3. Prepare Image Token Placeholders
        # Janus requires us to inject specific tokens to trigger the image decoder
        tokens = torch.zeros(
            (self.cfg.parallel_size * 2, len(prompt_ids)), dtype=torch.int
        ).to(self.cfg.device)
        for i in range(self.cfg.parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:  # Negative/Unconditional samples for CFG
                tokens[i, 1:-1] = self.processor.pad_id

        # 4. Get Embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        # 5. Generation Loop (Autoregressive)
        # We generate 576 tokens (384/16 = 24, 24*24 = 576 patches)
        generated_tokens = torch.zeros(
            (self.cfg.parallel_size, 576), dtype=torch.int
        ).to(self.cfg.device)

        past_key_values = None

        print(f"[Gen] Generating image tokens for: '{prompt}'")
        for i in range(576):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            hidden_states = outputs.last_hidden_state

            # Project to image vocabulary size
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            # Classifier Free Guidance (CFG)
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            # Sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Prepare next input
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            past_key_values = outputs.past_key_values

        # 6. Decode Tokens to Pixels
        print("[Decode] Decoding visual tokens...")
        dec = self.model.gen_vision_model.decode(generated_tokens)
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        # Normalize and Clip
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = dec.astype(np.uint8)
        return Image.fromarray(visual_img[0])


# --- Main Execution ---
if __name__ == "__main__":
    # Settings
    PROMPT = "A schematic diagram of a high-voltage embedded system circuit board, technical drawing style, blueprint blue background, white lines, highly detailed components."
    UPSCALING = True

    # Initialize
    config = PipelineConfig()
    pipeline = JanusPipeline(config)
    meta_mgr = MetadataManager()
    upscaler = ImageUpscaler(config.device) if UPSCALING else None

    try:
        # Run Generation
        start_time = time.time()
        raw_image = pipeline.generate(PROMPT, temperature=1.0, cfg_weight=5.0)

        # Post-Processing
        final_image = raw_image
        if upscaler:
            print("[Post] Upscaling image...")
            final_image = upscaler.upscale(raw_image, scale=4)

        # Save
        params = {"model": config.model_path, "upscaled": UPSCALING, "cfg": 5.0}
        saved_path = meta_mgr.save_run(final_image, PROMPT, params)

        elapsed = time.time() - start_time
        print(f"\n[Done] Image saved to: {saved_path}")
        print(f"       Total time: {elapsed:.2f}s")

    except KeyboardInterrupt:
        print("\n[Stop] Generation interrupted by user.")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
