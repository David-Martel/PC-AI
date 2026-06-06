# Python Dependency Modernization Plan

**Repo:** `C:\codedev\PC_AI`
**Date:** 2026-06-06
**Scope:** Python tooling only (Rust/Cargo Dependabot alerts tracked separately)
**Status:** READ-ONLY AUDIT — no installs or lockfile modifications included

---

## Executive Summary

The repo has **one project-owned Python manifest** (`AI-Media/requirements.txt`) and **no** existing `.python-version`, `pyproject.toml`, or `uv.lock` at the AI-Media or repo root. The critical gap is `torch>=2.1.0` with no CUDA index specified, meaning pip silently resolves the CPU-only wheel; the RTX 5060 Ti (SM_120 / Blackwell) requires `cu128` wheels from the PyTorch nightly/stable index. PyTorch 2.7.0 is the first release with prototype Blackwell (SM_120) native support and ships `torch-2.7.0+cu128-cp313-cp313-win_amd64.whl` — confirmed against `download.pytorch.org/whl/cu128/`. The upscaler path (`basicsr`/`realesrgan`) is a dead upstream (last release August 2022, CVE-2024-27763 unfixed) and will not install cleanly under Python 3.13; it must be replaced or isolated before the Python version bump.

---

## Manifest Inventory

### Owned Manifests (actionable)

| File | Role | Has torch pin | requires-python |
|------|------|--------------|-----------------|
| `AI-Media/requirements.txt` | Janus-Pro CUDA agent, main.py + Tools/\*.py | `torch>=2.1.0` (no index) | none — no pyproject.toml |

### Vendored / Transient (DO NOT EDIT — excluded from this plan)

| File | Why excluded |
|------|-------------|
| `.codex-tmp/Janus-upstream/requirements.txt` | Upstream DeepSeek Janus clone; overwritten on re-clone |
| `.codex-tmp/Janus-upstream/pyproject.toml` | Same upstream clone; `requires-python = ">=3.8"` is DeepSeek's declaration |
| `.codex_tmp/Janus/requirements.txt` | Duplicate local clone |
| `.codex_tmp/Janus/pyproject.toml` | Duplicate local clone |
| `.tmp/ollm/pyproject.toml` | Third-party `ollm` library clone; `requires-python = ">=3.10"` |
| `Native/pcai_core/third_party/ggml-sys/llama-cpp/requirements.txt` | Vendored llama.cpp Python tooling (`numpy==1.24`, `sentencepiece==0.1.98`); updated only when llama.cpp vendor is bumped |

### Files encoding version policy (not manifests, but change with manifests)

| File | What to update |
|------|---------------|
| `Tools/janus-understand.py` (lines 114–121) | `compatible_torchvision_requirement()` torchvision mapping — add `"2.7": "torchvision==0.22.0"` entry (already present in code), verify `"2.8"` entry; update the `fail()` guard range |
| (new) `AI-Media/.python-version` | Create this file — does not exist yet |
| (new) `AI-Media/pyproject.toml` | Create for uv index config — does not exist yet |

---

## Change Table

### Priority 1 — torch >=2.7 + cu128 (Blackwell / SM_120 fix)

**Background:** PyTorch 2.7.0 is the first release with prototype Blackwell support (SM_120 / compute capability 12.0). The `cu128` wheel index is `https://download.pytorch.org/whl/cu128`. Wheel `torch-2.7.0+cu128-cp313-cp313-win_amd64.whl` and `torchvision-0.22.0+cu128-cp313-cp313-win_amd64.whl` confirmed present at that index (verified 2026-06-06).

#### Edit 1A — `AI-Media/requirements.txt`

**Current:**
```
torch>=2.1.0
transformers>=4.38.0
accelerate>=0.27.0
sentencepiece
numpy
pillow
# Optional: For upscaling
basicsr>=1.4.2
realesrgan>=0.3.0
```

**Proposed:**
```
# PyTorch with CUDA 12.8 wheels (RTX 5060 Ti SM_120 / Blackwell support)
# Install via: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# Or use pyproject.toml + uv (recommended — see Create 1B)
torch>=2.7.0
torchvision>=0.22.0   # must match torch major.minor; 0.22.x pairs with torch 2.7.x
transformers>=4.38.2  # keep Janus upstream floor; test-bump to 4.47+ only if needed — see Priority 4
accelerate>=1.2.0     # 0.27 is from early 2024; 1.x is current stable API
sentencepiece
numpy>=2.0            # numpy 2.x is required for Python 3.13; 2.4.3 already installed in janus-python env
pillow>=11.0          # 11.x is current; 12.1.1 already in env
# Optional: For upscaling — SEE RISK NOTE R-4 below before enabling
# basicsr>=1.4.2      # BLOCKED: abandoned upstream, CVE-2024-27763, no Python 3.13 wheels
# realesrgan>=0.3.0   # BLOCKED: depends on basicsr; same issues
```

**Risk:** NEEDS-TEST (see R-1, R-2, R-4 in Risks section)

#### Create 1B — `AI-Media/pyproject.toml` (NEW FILE — uv recommended path)

This is the preferred mechanism for uv to resolve the cu128 index without requiring `--extra-index-url` flags on every `pip install`:

```toml
[project]
name = "pcai-janus-agent"
version = "0.1.0"
description = "Janus-Pro CUDA media agent for PC_AI"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "transformers>=4.38.2",
    "accelerate>=1.2.0",
    "sentencepiece",
    "numpy>=2.0",
    "pillow>=11.0",
    # "basicsr>=1.4.2",   # BLOCKED — see R-4
    # "realesrgan>=0.3.0", # BLOCKED — see R-4
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

> **FOOTGUN WARNING:** Bumping `torch>=2.7.0` in `requirements.txt` alone does NOT fix SM_120. Without the cu128 index, `pip install -r requirements.txt` silently resolves the CPU-only `torch-2.7.0-cp313-cp313-win_amd64.whl` from PyPI. The RTX 5060 Ti requires the `+cu128` build. The uv `pyproject.toml` path with `explicit = true` is the safe fix — it forces `torch` and `torchvision` to resolve exclusively from the cu128 index. Using `pip` directly requires `--extra-index-url https://download.pytorch.org/whl/cu128` on every install invocation.

**Why uv path over `--extra-index-url`:** `explicit = true` tells uv to resolve `torch`/`torchvision` *only* from the cu128 index, eliminating the multi-index resolution ambiguity where pip can silently pick the CPU wheel from PyPI. `torchaudio` is not listed in `dependencies` because it is not used by the Janus pipeline; it can be added to both sections later if needed.

**Risk:** SAFE to create (no lock file generated until `uv lock` is run, which is explicitly excluded from this plan).

---

### Priority 2 — Python >=3.13 pin

**Background:** uv 0.8.22 is installed. CPython 3.13.13 is available at `C:\Python313\python.exe`. No `.python-version` file exists anywhere in PC_AI. `torch-2.7.0+cu128-cp313-cp313-win_amd64.whl` confirmed available.

#### Create 2A — `AI-Media/.python-version` (NEW FILE)

```
3.13
```

This file instructs `uv python` and tooling to use Python 3.13 when running from the `AI-Media/` directory. The project-root level can have its own or can inherit this.

**Command to apply (after creating the file):**
```powershell
cd AI-Media
uv python pin 3.13
```
This updates `.python-version` in place and symlinks to the managed interpreter.

**Risk:** NEEDS-TEST (see R-1, R-2)

#### Note on `python_fallback.rs` (line 67)

`python_command()` in `Native/pcai_core/pcai_media/src/python_fallback.rs` invokes bare `"python"` on Windows (not `python3.13`). The `.python-version` pin and `uv python pin` do **not** redirect the OS-level `python` command — they only affect `uv run` invocations. To enforce 3.13 for the Rust fallback path, set the environment variable `PCAI_MEDIA_PYTHON_EXECUTABLE` to the full path of the 3.13 interpreter (e.g., `C:\Python313\python.exe`) in whichever service or shell launches pcai_media. This is a runtime configuration step, not a file edit.

---

### Priority 3 — `Tools/janus-understand.py` torchvision mapping (line 114–121)

**Current code (lines 114–121):**
```python
mapping = {
    "2.8": "torchvision==0.23.0",
    "2.7": "torchvision==0.22.1",
    "2.6": "torchvision==0.21.0",
}
requirement = mapping.get(major_minor)
if requirement is None:
    fail(f"Unsupported torch version for Janus bootstrap: {torch.__version__}")
```

**Assessment:** The mapping already contains a `"2.7"` entry (`torchvision==0.22.1`). This entry is correct — `torchvision==0.22.1+cu128-cp313` is confirmed available. No edit needed to the mapping itself.

**One defensive change recommended:** The `fail()` guard (line 121) will fire for any torch version not in the mapping (e.g., 2.7.1 if PyPI ever ships a patch). Consider replacing the hard fail with a warning + latest-known fallback, but this is a code improvement, not a blocking change.

**Risk:** SAFE (no edit needed for 2.7.0 compatibility; mapping already covers it)

---

### Priority 4 — Other Outdated / Insecure Python Deps

| Package in `AI-Media/requirements.txt` | Current pin | Proposed | Reason | Risk |
|----------------------------------------|-------------|----------|--------|------|
| `transformers>=4.38.0` | ~18 months old | `>=4.38.2` (no change for now) | transformers ships pure-Python wheels; Python 3.13 compat is not the gate. The binding constraint is Janus-Pro `trust_remote_code` compatibility — `VLChatProcessor.apply_sft_template_for_multi_turn_prompts` has a history of breaking on newer transformers. Minimum-risk approach: keep the upstream Janus floor (`>=4.38.2`) and only test-bump (toward 4.47+) if the Python 3.13 install fails in practice. Latest stable is 5.10.2 but do not jump there without Janus regression testing. | NEEDS-TEST before any bump beyond 4.38.2 |
| `accelerate>=0.27.0` | ~2 years old (0.27 = Feb 2024) | `>=1.2.0` | Major version bump; 1.x is current stable API | NEEDS-TEST (HF Accelerate changed distributed-training defaults in 1.0) |
| `numpy` (unpinned) | Resolves to whatever | `>=2.0` | numpy 2.x is required for Python 3.13; 2.4.3 already observed in `janus-python` env | SAFE (numpy 2.x is backwards-compatible for consumption; may break old extension modules) |
| `pillow` (unpinned) | Resolves to whatever | `>=11.0` | CVE exposure in older Pillow; 12.1.1 already observed in env; explicit floor prevents backslide | SAFE |
| `basicsr>=1.4.2` | 1.4.2 (Aug 2022) | **REMOVE / ISOLATE** | Abandoned upstream; CVE-2024-27763 (no fixed version); no Python 3.13 wheels; `torchvision.transforms.functional_tensor` removed in torchvision ≥0.17 (used internally by basicsr) — will fail on import with torch 2.7 | NEEDED for upscaler |
| `realesrgan>=0.3.0` | 0.3.0 (Sep 2022) | **REMOVE / ISOLATE** | Hard-depends on `basicsr`; same issues; `ImageUpscaler` in `main.py` gracefully falls back to `BICUBIC` when not available | SAFE to comment out |

**Upscaler replacement path (not scoped for this plan, but flagged):** The `ImageUpscaler` class in `AI-Media/main.py` already has a `try/except ImportError` guard that falls back to `BICUBIC` resize. Commenting out `basicsr` and `realesrgan` in requirements immediately stops the broken import without functional regression for generation. If real upscaling is needed, `facexlib` + `gfpgan` maintain active Python 3.x support, or the ONNX export at `Tools/Convert-RealESRGAN-to-ONNX.py` can replace the Python dependency with an ONNX Runtime inference path.

---

## Risks

| ID | Risk | Severity | Affected files | Mitigation |
|----|------|----------|----------------|------------|
| R-1 | **Python 3.13 + torch 2.7 full pipeline untested** — the Janus-Pro generation loop uses `VLChatProcessor`, `AutoModelForCausalLM`, `bfloat16`, and CUDA synchronize; must run end-to-end on 3.13/cu128 before production | HIGH | `AI-Media/main.py`, `Tools/bench-janus-python.py`, `Tools/demo-janus-generate.py`, `Tools/janus-understand.py` | Run `AI-Media/main.py` + `Tools/bench-janus-python.py --model-7b` after install; confirm `torch.cuda.get_device_properties(0).major == 12` for RTX 5060 Ti |
| R-2 | **Rust fallback path (`python_fallback.rs`) does not respect `.python-version`** — invokes bare `python` on Windows (PATH-dependent); if system default is 3.12, the 3.13 env is bypassed | MEDIUM | `Native/pcai_core/pcai_media/src/python_fallback.rs` (line 67) | Set `PCAI_MEDIA_PYTHON_EXECUTABLE=C:\Python313\python.exe` in service/shell environment; no code change required |
| R-3 | **`janus-understand.py` reuses global torch from system Python 3.12** (comment at line 128–130: "Reuse the working global torch/transformers install from Python 3.12") — after moving to 3.13, torch cu128 must be installed for the 3.13 interpreter, not just 3.12 | MEDIUM | `Tools/janus-understand.py` | After `uv python pin 3.13`, run `uv run pip install torch>=2.7.0 --extra-index-url https://download.pytorch.org/whl/cu128` from `AI-Media/` to populate 3.13 site-packages |
| R-4 | **`basicsr` / `realesrgan` will not install on Python 3.13** — both are abandoned (last release 2022), no 3.13 wheels, `functional_tensor` removed from torchvision ≥0.17 causing import error; `main.py` `ImageUpscaler` already has graceful fallback | HIGH for upscaler, SAFE for core generation | `AI-Media/requirements.txt`, `AI-Media/main.py` | Comment out both lines; `ImageUpscaler.__init__` catches `ImportError` and falls back to BICUBIC; no action needed for generation pipeline |
| R-5 | **transformers 4.47+ API changes** — `VLChatProcessor.apply_sft_template_for_multi_turn_prompts` and `processor.sft_format` used in `main.py` are Janus-specific `trust_remote_code` additions; not part of core transformers API; should survive version bumps but needs testing | MEDIUM | `AI-Media/main.py`, `Tools/demo-janus-generate.py` | Test with `transformers==4.47.0` before bumping to 5.x; 4.47 LTS range is safer than jumping to 5.x immediately |
| R-6 | **candle-core 0.9.2 + cudarc 0.19.0 Rust SM_120 gap** — the Rust media path (`AI-Media/cargo.toml`, `Deploy/rust-functiongemma-core/Cargo.toml`) uses candle 0.9.2 with cudarc 0.19.0; the vendored `candle-kernels-0.9.2/build.rs` handles CUDA 13 MSVC flags but SM_120 PTX support depends on the CUDA Toolkit version present, not the Rust crate version; `CUDA_COMPUTE_CAPS=89,120` must be set at build time | MEDIUM | `AI-Media/cargo.toml`, `Deploy/rust-functiongemma-core/Cargo.toml`, `Deploy/vendor/candle-kernels-0.9.2/build.rs` | This is a build-time env var issue already documented in CLAUDE.md; set `$env:CUDA_COMPUTE_CAPS = "89,120"` before `cargo build`. Python torch 2.7+cu128 path is independent and is the primary route for Blackwell. |

---

## Safe-to-edit-now vs. Needs-test-first

### Safe to edit immediately (no build/test cycle needed)

| Action | File | Why safe |
|--------|------|----------|
| Create `AI-Media/.python-version` with `3.13` | NEW | Read-only pin file; no install triggered by creation |
| Create `AI-Media/pyproject.toml` with uv index config | NEW | uv does not install or lock until `uv lock`/`uv sync` is run explicitly |
| Comment out `basicsr>=1.4.2` and `realesrgan>=0.3.0` | `AI-Media/requirements.txt` | `ImageUpscaler` already handles `ImportError` gracefully; core generation unaffected |
| Add `numpy>=2.0` and `pillow>=11.0` floors | `AI-Media/requirements.txt` | Floors only; env already has numpy 2.4.3 and pillow 12.1.1 |

### Needs a build/test cycle before committing

| Action | File | What to test |
|--------|------|-------------|
| `torch>=2.7.0` with cu128 index | `AI-Media/requirements.txt` + new pyproject.toml | `import torch; torch.cuda.get_device_properties(0).major` should return 12 on RTX 5060 Ti; run `Tools/bench-janus-python.py` |
| `requires-python = ">=3.13"` in pyproject.toml | NEW `AI-Media/pyproject.toml` | Full pipeline: `AI-Media/main.py` generation + `Tools/janus-understand.py` fallback via `PCAI_MEDIA_PYTHON_EXECUTABLE` |
| `transformers>=4.47.0` | `AI-Media/requirements.txt` | `VLChatProcessor.from_pretrained`, `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)` in `main.py` |
| `accelerate>=1.2.0` | `AI-Media/requirements.txt` | Import-level; test `from accelerate import Accelerator` + `init_process_group` path if used |
| `torchvision>=0.22.0` | `AI-Media/requirements.txt` | Only if upscaler is being replaced; not needed for generation path |

---

## Recommended Execution Order

1. **Comment out `basicsr`/`realesrgan`** in `AI-Media/requirements.txt` — zero risk, unblocks Python 3.13 install.
2. **Create `AI-Media/.python-version`** (contents: `3.13`) — no install side effect.
3. **Create `AI-Media/pyproject.toml`** with `requires-python = ">=3.13"`, `torch>=2.7.0`, and `[[tool.uv.index]]` cu128 block.
4. **Set `PCAI_MEDIA_PYTHON_EXECUTABLE`** in the shell/service that invokes `pcai_media` → `C:\Python313\python.exe`.
5. **Run `uv sync`** from `AI-Media/` (explicitly — not done in this plan) → installs torch 2.7.0+cu128 for Python 3.13.
6. **Validate**: `python -c "import torch; print(torch.__version__, torch.cuda.get_device_properties(0).major)"` → expect `2.7.x+cu128` and `12`.
7. **Test generation pipeline**: `python AI-Media/main.py` + `python Tools/bench-janus-python.py`.
8. **Test fallback path**: Invoke through Rust `pcai_media` with `cuda:1` device, confirm `janus-understand.py` runs on the 3.13 interpreter.
9. Bump `transformers` and `accelerate` after torch/Python are confirmed stable.

---

## Out of Scope

- **Rust Dependabot alerts (3 × rust-openssl):** tracked separately as noted in the brief; not addressed here.
- **Candle-core / cudarc Rust crate version bumps:** candle 0.9.2 and cudarc 0.19.0 are mildly behind latest but functional with the patched vendored kernels; bumping them requires Rust build + CUDA validation cycle separate from Python work.
- **Vendored upstream manifests** (Janus-upstream, ollm, llama-cpp): not owned by this project; any edits are overwritten on re-clone.
- **Notebooks (`Notebooks/pcai_eval_*.ipynb`):** use ad-hoc `pip install lm-eval[openai]>=0.4.4` inline; no manifest to modernize; acceptable for now.
- **`mistralrs 0.7` / `llm` (rustformers) in pcai_core workspace:** backend-specific Rust deps; Rust scope.

---

## Reference: Confirmed Wheel Availability (2026-06-06)

| Package | Version | Index | Python | Platform | Status |
|---------|---------|-------|--------|----------|--------|
| torch | 2.7.0+cu128 | download.pytorch.org/whl/cu128 | cp313 | win_amd64 | CONFIRMED |
| torch | 2.7.0+cu128 | download.pytorch.org/whl/cu128 | cp313t (free-threaded) | win_amd64 | CONFIRMED |
| torchvision | 0.22.0+cu128 | download.pytorch.org/whl/cu128 | cp313 | win_amd64 | CONFIRMED |
| torchvision | 0.22.1+cu128 | download.pytorch.org/whl/cu128 | cp313 | win_amd64 | CONFIRMED |

PyTorch 2.7.0 release blog confirmed: "prototype support for NVIDIA Blackwell GPU architecture (SM_120, compute capability 12.0); cuDNN, NCCL, and CUTLASS upgraded for Blackwell."
