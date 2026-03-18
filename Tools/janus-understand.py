#!/usr/bin/env python3
"""Janus-Pro image understanding helper for the Rust media fallback path."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path


JANUS_REPO_URL = "https://github.com/deepseek-ai/Janus.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_ROOT = PROJECT_ROOT / ".pcai" / "janus-python"
SITE_ROOT = BOOTSTRAP_ROOT / "site-packages"
SOURCE_ROOT = BOOTSTRAP_ROOT / "Janus"


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_site_root() -> None:
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    site.addsitedir(str(SITE_ROOT))


def remove_matching(prefixes: list[str]) -> None:
    if not SITE_ROOT.exists():
        return

    for path in SITE_ROOT.iterdir():
        name = path.name
        if any(
            name == prefix
            or name == f"{prefix}.py"
            or name.startswith(f"{prefix}-")
            for prefix in prefixes
        ):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def pip_install(packages: list[str], *, no_deps: bool = False) -> None:
    if not packages:
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--target",
        str(SITE_ROOT),
    ]
    if no_deps:
        command.append("--no-deps")
    command.extend(packages)
    subprocess.run(command, check=True)


def write_attrdict_shim() -> None:
    shim_path = SITE_ROOT / "attrdict.py"
    shim_path.write_text(
        "import copy\n"
        "\n"
        "class AttrDict(dict):\n"
        "    def __getattr__(self, name):\n"
        "        try:\n"
        "            return self[name]\n"
        "        except KeyError as exc:\n"
        "            raise AttributeError(name) from exc\n"
        "\n"
        "    def __setattr__(self, name, value):\n"
        "        self[name] = value\n"
        "\n"
        "    def __delattr__(self, name):\n"
        "        try:\n"
        "            del self[name]\n"
        "        except KeyError as exc:\n"
        "            raise AttributeError(name) from exc\n"
        "\n"
        "    def copy(self):\n"
        "        return AttrDict(self)\n"
        "\n"
        "    def __deepcopy__(self, memo):\n"
        "        return AttrDict({\n"
        "            copy.deepcopy(key, memo): copy.deepcopy(value, memo)\n"
        "            for key, value in self.items()\n"
        "        })\n",
        encoding="utf-8",
    )


def compatible_torchvision_requirement() -> str:
    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    major_minor = ".".join(torch_version.split(".")[:2])
    mapping = {
        "2.8": "torchvision==0.23.0",
        "2.7": "torchvision==0.22.1",
        "2.6": "torchvision==0.21.0",
    }
    requirement = mapping.get(major_minor)
    if requirement is None:
        fail(f"Unsupported torch version for Janus bootstrap: {torch.__version__}")
    return requirement


def ensure_python_dependencies() -> None:
    write_attrdict_shim()

    # Reuse the working global torch/transformers install from Python 3.12 and
    # keep the vendor path limited to the small Janus-specific gaps.
    remove_matching(["torch", "huggingface_hub", "fsspec", "torchvision"])

    missing = []
    if not module_available("einops"):
        missing.append("einops")
    if not module_available("timm"):
        missing.append("timm")
    if not module_available("torchvision"):
        missing.append(compatible_torchvision_requirement())

    if missing:
        pip_install(missing, no_deps=True)


def ensure_janus_source() -> None:
    override = os.environ.get("PCAI_MEDIA_JANUS_SOURCE")
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.append(SOURCE_ROOT)
    candidates.append(PROJECT_ROOT / ".codex-tmp" / "Janus-upstream")

    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            return

    SOURCE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", JANUS_REPO_URL, str(SOURCE_ROOT)],
        check=True,
    )
    sys.path.insert(0, str(SOURCE_ROOT))


def resolve_device(requested_device: str, torch_module) -> str:
    requested = (requested_device or "cpu").lower()
    if requested.startswith("cuda") and torch_module.cuda.is_available():
        return requested
    return "cpu"


def resolve_dtype(device: str, torch_module):
    if device.startswith("cuda"):
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    if hasattr(torch_module, "bfloat16"):
        return torch_module.bfloat16
    return torch_module.float32


def clean_answer(raw_text: str, prompt_prefix: str | None) -> str:
    text = raw_text.strip()
    if prompt_prefix and text.startswith(prompt_prefix):
        return text[len(prompt_prefix) :].strip()
    return text


def main() -> None:
    ensure_site_root()
    ensure_python_dependencies()
    ensure_janus_source()

    request = json.load(sys.stdin)
    model_path = str(request["model"])
    image_path = str(request["image_path"])
    prompt = str(request["prompt"])
    max_tokens = int(request["max_tokens"])
    temperature = float(request["temperature"])
    requested_device = str(request["device"])

    if not Path(image_path).is_file():
        fail(f"Input image not found: {image_path}")

    with redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModelForCausalLM
        from janus.models import VLChatProcessor
        from janus.utils.io import load_pil_images

        device = resolve_device(requested_device, torch)
        dtype = resolve_dtype(device, torch)

        processor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model = model.to(device).eval()

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        with torch.inference_mode():
            prepare_inputs = processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
            ).to(device=device, dtype=dtype)
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            generation_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": prepare_inputs.attention_mask,
                "pad_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0.01,
                "use_cache": True,
            }
            if temperature > 0.01:
                generation_kwargs["temperature"] = temperature

            outputs = model.language_model.generate(**generation_kwargs)

    prompt_prefix = None
    sft_format = getattr(prepare_inputs, "sft_format", None)
    if sft_format and len(sft_format) > 0:
        prompt_prefix = str(sft_format[0])

    full_output_tokens = outputs[0].cpu().tolist()
    candidate_tokens = full_output_tokens
    input_ids = getattr(prepare_inputs, "input_ids", None)
    if input_ids is not None:
        prompt_len = int(input_ids.shape[-1])
        candidate_tokens = full_output_tokens[prompt_len:]

    answer = clean_answer(
        tokenizer.decode(candidate_tokens, skip_special_tokens=True),
        prompt_prefix,
    )
    if not answer:
        answer = clean_answer(
            tokenizer.decode(full_output_tokens, skip_special_tokens=True),
            prompt_prefix,
        )

    print(json.dumps({"text": answer}))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        fail(f"Janus Python fallback dependency bootstrap failed: {exc}")
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Janus Python fallback crashed: {exc}")
