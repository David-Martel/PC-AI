# Rust FunctionGemma Workspace

This workspace groups the Rust runtime and training crates for FunctionGemma.
It is intentionally modular so the same base model can be retrained for
multiple tool sets or future router tasks.

## Members
- rust-functiongemma-runtime (Deploy/rust-functiongemma-runtime)
- rust-functiongemma-train (Deploy/rust-functiongemma-train)

## Build (Primary: Build.ps1)
From repo root:

  .\Build.ps1 -Component functiongemma
  .\Build.ps1 -Component functiongemma -RunTests
  .\Build.ps1 -Component functiongemma-router-data
  .\Build.ps1 -Component functiongemma-token-cache
  .\Build.ps1 -Component functiongemma-train
  .\Build.ps1 -Component functiongemma-eval

## Build/Test scripts
From Deploy/rust-functiongemma:

  .\build.ps1
  .\test.ps1

Legacy CargoTools wrappers still exist for advanced crate-level control, but
`Build.ps1` is the canonical build/test entrypoint for this repository.

## Fast CI mode
Skip dataset/doc generation for quicker CI runs:

  .\test.ps1 -Fast

## Eval report
Generate a metrics report (skips if the model is missing):

  .\Build.ps1 -Component functiongemma-eval -FunctionGemmaArgs @('--fast-eval')

Or integrate it into the test runner:

  .\test.ps1 -EvalReport

## C# integration (PcaiNative)
The native DLL exposes a router dataset generator for C#/PowerShell:

  PcaiNative.FunctionGemmaModule.BuildRouterDataset(...)

## Tool documentation
Generate up-to-date tool docs (including NO_TOOL negatives):

  .\Tools\generate-functiongemma-tool-docs.ps1

## Native router dataset (optional)
If PcaiNative.dll is built and available, you can generate the router dataset via FFI:

  .\Tools\prepare-functiongemma-router-data.ps1 -UseNative

## Documentation automation
Run the unified docs + FunctionGemma dataset pipeline:

  .\Tools\Invoke-DocPipeline.ps1 -Mode Full

Use the native C# router dataset generator when available:

  .\Tools\Invoke-DocPipeline.ps1 -Mode TrainingOnly -UseNativeRouter

## Future
A shared core crate will be added for:
- Prompt formatting
- Tool schema parsing
- Chat template rendering
- Shared config and utilities
