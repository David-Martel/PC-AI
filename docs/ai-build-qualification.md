# AI build and workload qualification

Run the [machine capability inventory](../Tools/Get-PcaiMachineCapabilities.ps1) on Windows and record adapter and driver evidence before selecting an inference target. Pair it with a build-toolchain probe and an actual model smoke test. A display device, a compiler, an artifact's Cargo features, and a successful workload are separate observations. Keep them separate in build and deployment receipts. See the [test guide](../Tests/README.md) and [benchmark admission rules](benchmark-admission.md).

## Build and runtime policy

The current [build entry point](../Build.ps1) exposes `-EnableCuda`, and the [inference builder](../Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1) selects features separately for llama-rs and mistral.rs. The inference child can fall back independently of the parent. The parent derives manifest metadata and package names from the requested switch, so a child fallback can leave packaging metadata inconsistent with the artifact. Fix this before using automatic selection for release builds.

| Target | Build admission | Deployment and runtime admission |
| --- | --- | --- |
| CPU, Windows/Linux | Compile a portable CPU artifact and record the minimum instruction set. Build from a declared target profile so local AVX features are never an implicit release requirement. | Require a bounded CPU inference run using a pinned model. Retain this as a separately launchable artifact. |
| NVIDIA CUDA, including eGPU | Verify the toolkit, linker, compatible host compiler and explicit target SM list. A release builder can produce CUDA for declared targets with no GPU attached. An explicit CUDA request with missing prerequisites should fail before build and packaging. | Probe the selected runtime device, driver, available memory and model load. Rerun after attach, detach or restart. In auto mode choose CPU when CUDA initialization fails, including via a separate process if the CUDA artifact cannot start. |
| Intel iGPU, Windows/Linux | Qualify a maintained llama.cpp Vulkan build first. Evaluate SYCL or OpenVINO independently for models and operators that benefit. Intel MKL is a CPU path. | Run inference on the selected Intel device, report offloaded layers, and compare available shared-memory budget to measured peak usage. Device enumeration alone is insufficient. |

Generate a build plan from target OS/architecture, configured features, discovered compilers and deployment profiles. Record which artifacts were actually built, the backend, feature flags, toolchain/version, target SMs, and each artifact hash. Runtime receipts should independently record selected device, observed offload, elapsed time and fallback reason. The [mistral.rs server CLI](../Native/pcai_core/pcai_inference/src/bin/mistralrs.rs) accepts `backend.device` values `cpu` and `auto`, with auto as the default. The existing [FFI](../Native/pcai_core/pcai_inference/src/ffi/mod.rs) `gpu_layers=0` requests CPU. The [pinned mistral.rs implementation](../Native/pcai_core/pcai_inference/src/backends/mistralrs.rs) forwards CPU mode to both its GGUF and SafeTensors text builders. In auto mode it chooses CPU when the CUDA device probe returns an initialization error. A GPU heartbeat or hot unplug during an in-progress request needs separate qualification.

The November 22 audit observed an Intel iGPU and an RTX PRO 2000 Blackwell Laptop GPU with 8,151 MiB on Gen9. Both enumerate in Vulkan and the NVIDIA driver responds, while `nvcc` was absent. The NUC reports Iris Xe and a Vulkan probe, with no CUDA compiler. An eGPU destination remains a valid declared build target independent of what is attached today. No machine name is needed by the capability rules. [Upstream llama.cpp build options](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) · [Intel SYCL device support](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)

The [Windows CUDA bootstrap](../Tools/Initialize-CudaEnvironment.ps1) accepts
explicit compiler targets such as `-TargetComputeCaps '89,120'`. It checks each
complete toolkit's `nvcc --list-gpu-code` for each numeric target. For `120a` or
`120f`, nvcc must advertise `sm_120` and the same toolkit's `ptxas --help` must
list the exact suffixed code. Plain numeric targets use only the nvcc list. The
[native inference build](../Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1)
validates its requested targets before invoking Cargo. The legacy llama-rs
backend builds for numeric targets. NVFP4 execution belongs to the modern
llama.cpp qualification path. [NVIDIA nvcc target listing](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#list-gpu-code-code-ls)

## Model admission

The [inference workspace](../Native/pcai_core/Cargo.toml) pins archived rustformers llama-rs. Its [legacy loader](../Native/pcai_core/pcai_inference/src/backends/llamacpp.rs) infers architectures from filenames. The Qwen2.5 GGUF paired with `llama_cpp` in the [server example](../Config/pcai-inference-server.json) therefore needs a compatible backend before promotion. Start with a pinned modern llama.cpp process behind the existing provider interface. Require a port-bound health check, an explicit device, and actual GGUF architecture and projector checks. Reuse the [native GGUF header parser](../Native/pcai_core/pcai_core_lib/src/preflight/gguf.rs) in [model discovery](../Tools/Invoke-ModelDiscovery.ps1) before advertising compatibility. [Archived llama-rs project](https://github.com/rustformers/llm) · [Current llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)

| Slice | First candidate | Admission required |
| --- | --- | --- |
| Pinned mistral.rs text | [Qwen3-4B Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main) weighs 2.50 GB. The pinned GGUF architecture list includes Qwen3. | One load and text-generation request on CPU, then on CUDA with explicit offload evidence. |
| Low-memory CPU or Intel iGPU | [LFM2.5-1.2B Q4_K_M](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/tree/main) weighs about 731 MB. [Intel's Qwen3.5-4B INT4 IR](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) gives a separate OpenVINO comparison. | Use an upstream runner that lists the exact architecture, then validate task quality and device memory before changing defaults. |
| Vision, 8–16 GB CUDA or Intel shared memory | [Gemma 4 E2B QAT Q4_0](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/tree/main) with its published projector. [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main) is a larger comparison. | Fixed image and expected fields, exact model and projector hashes, and a real multimodal request path. The existing [native text endpoint](../Native/pcai_core/pcai_inference/src/http/mod.rs) uses string chat content. |
| 24 GB CUDA comparison | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) has an [18 GB Ollama package](https://ollama.com/library/qwen3.8). | Begin with short context and measure remaining memory after loading, including KV cache, scratch and any projector. |

Keep the current [Janus image generation path](../Native/pcai_core/pcai_media_model/src/lib.rs) and [FunctionGemma router](../Config/pcai-functiongemma.json) as separate tasks. New understanding models do not establish Janus generation parity. The existing [FunctionGemma runtime gate](functiongemma-runtime-validation.md) exercises heuristic HTTP behavior. Evaluate the pinned checkpoint and LoRA against the same expected tool names and typed arguments before changing the router model.

## Functional slices and failures

| Slice | Entry point and required evidence |
| --- | --- |
| Unit | Run Rust unit tests and Pester with fixture providers. Cover capability classification, invalid configuration, model header compatibility, requested vs effective device, and exact parity for the Rust and PowerShell fallbacks. |
| Integration | Verify native artifact provenance and ABI before exercising PowerShell → C# → Rust. Run model-free FFI/error fixtures and actual bounded HTTP contracts. A configured checkpoint load error fails the selected slice. |
| Functional | Give the runner a specific artifact and checkpoint digest. Require one successful action, one meaningful failure and teardown. Run [`Invoke-PcaiMediaCudaFunctionalTests.ps1`](../Tests/Hardware/Invoke-PcaiMediaCudaFunctionalTests.ps1) explicitly for Janus CUDA. Selecting Pester's Functional tier only discovers Pester tests. |
| End to end | Exercise the installed package outside a checkout. Check package version, chosen native library, model load, health, generation/stream completion, cancellation and no residual child after failure. Execute CPU first, then supported GPU profiles and the absent-eGPU recovery case. |
| Hardware | Pair Intel/NVIDIA enumeration with a controlled device-enabled workload. Run eGPU boot/attach/detach and reconnect tests in isolated runs. Report unavailable optional hardware explicitly. Required release hardware needs a real passing receipt. |

Keep runner verdict work in [PR #384](https://github.com/David-Martel/PC-AI/pull/384). The [current inference functional tests](../Tests/Functional/Inference.Functional.Tests.ps1) and [media E2E tests](../Tests/E2E/Media.E2E.Tests.ps1) still skip some attempted execution failures. The media E2E detector also treats Intel presence as CUDA. Fix these before admitting their results as release evidence. Replace ambient model discovery in required slices with an explicit path and digest. The [inference evaluator](../Tests/Evaluation/Invoke-InferenceEvaluation.ps1) already exposes controls for model, seed, context and case count.

## Optimization order

- Isolate child process launch, PowerShell import, and first native call. The [acceleration loader](../Modules/PC-AI.Acceleration/PC-AI.Acceleration.psm1) dot-sources both command trees. Measure an empty child and a warm operation before changing exports or loading strategy.
- Fix route attribution and output comparison in the [tooling benchmark](../Tests/Benchmarks/Invoke-PcaiToolingBenchmarks.ps1) first. Its search wrappers can select the same native implementation as a direct-native row. Require exact outputs on a controlled corpus and record the actual route and binary hash. Measure the full manifest independently of `StatsOnly`.
- Reuse [content search's bounded result reservation](../Native/pcai_core/pcai_core_lib/src/search/content.rs) to fix the separate count-and-push paths in [file manifests](../Native/pcai_core/pcai_core_lib/src/search/files.rs). Add exact-cap tests for full and stats paths before measuring wide and deep trees. Keep the existing C# bridge for complex ownership and measure typed vs JSON on identical inputs before extending direct bindings.
- Fix [Rust profiling](../Tools/Invoke-RustProfile.ps1) to use the real `pcai_core_lib` package name and the child executable from Cargo JSON. Record child peak memory directly. Compare dedicated target directories, sccache hit rate and linker time. When adding nextest, keep the doctest command and fail on empty required selections.
- Work through [FunctionGemma CPU recovery](https://github.com/David-Martel/PC-AI/pull/405), the CUDA Candle/QLoRA compatibility gap, and the supported .NET package matrix independently. Version and cache keys need the exact model and toolchain inputs. Consolidate repeated device, readiness and child-process contracts before retiring the [prototype media stack](../AI-Media).

## November 22 benchmark observations

The Windows tooling run at `2c8504d3` completed nine cases. Fresh child import took 2.50 seconds median, including process startup. The direct manifest row took 40.33 ms median and the PowerShell row took 242.00 ms. The runner withheld comparative admission because it could not verify all output parity and actual backend routes. The installed C# bridge's separate transport sweep found essentially equal disk timings and noisy process results. That sweep compares two calls against changing process snapshots. Use a fixed result fixture before replacing the JSON path.

The staged [llama.cpp b12633 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12633) at `9c10c94e1` ran the same [LFM2.5-1.2B Q4_K_M](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/tree/8ed288026e23958ad9dfa92d53ed773a8eee7125) artifact on all three devices. The Vulkan package, CUDA 13.5 package and runtime, and model all matched their published SHA-256 digests. Five repetitions used four host threads, 256 prompt tokens and 64 generated tokens. The numbers are mean tokens per second from those external runners.

| Route | Prefill | Decode |
| --- | ---: | ---: |
| CPU | 183.0 | 37.6 |
| Intel Vulkan | 3,426.3 | 77.6 |
| NVIDIA Vulkan | 11,621.1 | 265.6 |
| NVIDIA CUDA | 13,806.5 | 261.1 |

Reproduce each route with `llama-bench -m <model> -p 256 -n 64 -r 5 -t 4 -ngl <layers> -dev <device> -o json`. Use `0` and `none` for CPU. Resolve current device names using `--list-devices` before selecting a GPU and use a high offload-layer request. A 15M smoke checkpoint decoded faster on CPU than on either Vulkan GPU in the same audit. Qualify actual model classes and report first-token latency, prefill, decode and peak memory separately. These results qualify the staged upstream binaries on one host. Native PC-AI CUDA builds and reinstall/reattach behavior require their own acceptance slices.

The [PC-AI evaluator](../Tests/Evaluation/Invoke-InferenceEvaluation.ps1) completed one `general` case over a temporary loopback llama.cpp server running the 1.2B model on Intel Vulkan. The server produced 48 tokens. The case failed with a score of 0.4 and a zero pass rate. The sample establishes request and response transport.

For automated acceptance, pass `-MinimumPassRate <0–100>` with one backend.
The evaluator requires a nonempty, complete run with no case errors and writes
the `Acceptance` decision before returning failure for an unavailable backend,
cancellation, incomplete run, case error, or pass rate below the threshold.
The [mock smoke workflow](../.github/workflows/evaluation-smoke.yml) uses
`-MinimumPassRate 0` to verify transport and completion. Its negative control
requires a nonzero evaluator exit and a saved `CaseErrors` report after a
healthy endpoint rejects a completion. Model promotion uses
a threshold chosen for the dataset and recorded with the staged artifact.
The pass rate uses the suite's composite score, which includes performance
metrics. Verify expected answers independently when qualifying model quality.

With the same staged release and model SHA-256
`b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`,
the selected NVIDIA CUDA, NVIDIA Vulkan, and Intel Vulkan devices each
reported 17 of 17 layers offloaded and returned a nonempty completion.
The single-case strict evaluation produced accepted and rejected reports
across those runs. This checks device selection, generated output, exit
status, and report preservation. A single case cannot establish model quality.
