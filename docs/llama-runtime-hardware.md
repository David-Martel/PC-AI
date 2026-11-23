# Staged llama.cpp hardware qualification

Install PowerShell 7.4 or later and use a dedicated output directory. The
[staging script](../Tools/Install-PcaiLlamaRuntime.ps1) installs the official
Windows x64 `b12635` archives into versioned CPU, Intel Vulkan, or NVIDIA CUDA
directories. The CUDA profile includes the matching CUDA 13.5 runtime. It
checks each archive's size and SHA-256 against the [release pins](../Config/llama-cpp-runtime.json).
Use `-DryRun` to inspect the selected platform and files. [Upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b12635)

```powershell
$stage = ./Tools/Install-PcaiLlamaRuntime.ps1 -Route IntelVulkan -OutputRoot ./.pcai/llama-runtime
./Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1 `
    -BenchPath $stage.BenchmarkPath -ReceiptPath $stage.ReceiptPath `
    -ModelPath $modelPath -ModelSha256 $modelSha256 -Route IntelVulkan -Repetitions 5
```

Use the expected model digest from the model publisher for `$modelSha256`.
Use the receipt produced during digest-verified installation. The benchmark
verifies the complete recorded executable and DLL set and holds open read
handles during the run. Keep the receipt and runtime together under your
account's controlled staging directory. An actor who can replace the account's
receipt and staged files crosses this per-user trust boundary.
The [benchmark runner](../Tests/Hardware/Invoke-PcaiLlamaDeviceBenchmark.ps1)
discovers device names on each run, requires a unique match, and requires the
selected GPU to report full layer offload. CPU uses explicit no-device mode.
`-MmqPrecision q4` and `q8` control the CUDA comparison. Each request sets the
precision only for its own benchmark child. The q4 route checks the
current selected device's compute capability before timing. Kernel instruction
qualification requires a profiler. [Upstream MMQ selection](https://github.com/ggml-org/llama.cpp/blob/b12635/ggml/src/ggml-cuda/mmq.cu)

Run the portable staging and validation fixtures from the repository root.

```powershell
Invoke-Pester -Path Tests/Unit/Install-PcaiLlamaRuntime.Tests.ps1,Tests/Unit/LlamaDeviceBenchmark.Tests.ps1
```

## Gen9 measurements

These are single-host measurements of device placement and workload throughput.
All rows used build `b12635` (`ad50351f5`), four host threads, 64 decode tokens,
and five repetitions. The Intel comparison used the same LFM2.5 Q4_K_M weights
and 256 prefill tokens on both routes. The Blackwell comparison used the same
Spark X2.6 4B Small NVFP4 weights and 512 prefill tokens for both precision
requests. [LFM2.5 checkpoint](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/tree/8ed288026e23958ad9dfa92d53ed773a8eee7125) · [Spark checkpoint](https://huggingface.co/WhiskyAKM/Spark-X2.6-4B-NVFP4-GGUF/tree/88b72c5799fb4c1e48e51bdf06421e4384b7d200)

| Model | Route | Prefill tokens/s | Decode tokens/s | GPU layers |
|---|---|---:|---:|---:|
| LFM2.5 Q4_K_M | CPU | 120.1 | 22.9 | 0 |
| LFM2.5 Q4_K_M | Intel Vulkan | 2,951.0 | 72.7 | 17/17 |
| Spark NVFP4 | NVIDIA CUDA, q4 requested | 7,516.9 | 91.2 | 37/37 |
| Spark NVFP4 | NVIDIA CUDA, q8 requested | 2,975.5 | 93.8 | 37/37 |

The loader rejected a separate 4B NVFP4 file because its recorded tensor
offset disagreed with the format expected by the loader. The selected model
digest and export revision belong in any deployment preset. These measurements
apply to the upstream Windows runtime. See [PC-AI native build qualification](ai-build-qualification.md)
for its separate compiler, model, and release gates.
