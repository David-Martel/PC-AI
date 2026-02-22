# TODO

## Now (stability + parity)
- [x] Replace external script references in prompts with module cmdlets
- [x] Add module fallback when external scripts are missing
- [x] Align tool schema with module cmdlets (SearchDocs/GetSystemInfo/SearchLogs vs pcai-tools.json)
- [x] Enforce JSON-only diagnose output in routed/diagnose flows
- [x] Enforce JSON-only diagnose output in UI/TUI rendering paths
- [x] Rust FunctionGemma runtime: add /health + /v1/models metadata (version/model/tools)
- [x] Rust FunctionGemma runtime: deterministic tool routing defaults for tool_choice=required
- [x] Rust FunctionGemma runtime: optional GPU selection + LoRA adapter load when available
- [ ] oLLM parity: explore large-context offload ideas for pcai_inference (llama.cpp/mistralrs) including KV cache offload and streaming prefill

## Architecture alignments (C#/Rust backend as primary engine)
- [ ] Replace PowerShell-only diagnostics with Rust/C# native backends where feasible (logs, inventory, health checks)
  - Native-first log/content search now wired in Acceleration module
  - Native disk usage + top process listing wired in Acceleration module
- [ ] Define a versioned C ABI contract for all Rust DLL exports (error codes, result structs, memory ownership)
- [ ] Create a unified C# service layer that exposes Rust capabilities to PowerShell and TUI
- [x] Consolidate pcai_fs exports into pcai_core_lib and route FsModule to the core DLL
- [ ] Standardize JSON schemas for native outputs (shared schema folder + version pinning)
- [x] Add a capability registry (DLL presence, feature flags, CPU/GPU support) surfaced to PowerShell UI
- [ ] Centralize error translation (Rust error -> C# status -> PowerShell error record)
- [ ] Provide cancellation/timeouts across PowerShell -> C# -> Rust (Ctrl+C propagation)
- [ ] Add structured logging + metrics from native layer (ETW or JSON log file)

## Agentic interface robustness
- [x] Ensure tool outputs are deterministic and bounded in size (truncate, summarize, or compress)
- [x] Add retry/timeout policy per tool invocation (with per-tool max runtime)
- [x] Add health gates for LLM providers (ollama/vllm/lmstudio) before tool routing
- [x] Normalize tool result envelopes (Success/ExitCode/Warnings/Evidence)
- [x] Expand diagnostic coverage to match DIAGNOSE_LOGIC.md expectations

## Memory + RAG integrations
- [ ] Integrate rag-redis (W:\dropbox-local\rag-redis) with Redis endpoints 6379/6380 for tool memory + retrieval.
- [ ] Evaluate simsimd (or similar SIMD distance kernels) for faster vector similarity in local RAG flows.
- [ ] Add optional Postgres 18/MS SQL backed memory store for long-term tool history.

## UI/TUI usability
- [x] Add a unified status command (LLM/Native DLL status)
- [x] Extend unified status command to include GPU checks
- [ ] Provide progress + streaming updates for long native operations
- [x] Implement a one-command "doctor" flow for common runtime failures
- [x] Add a concise summary view for JSON diagnose output (human-readable view)
- [x] Document common workflows in README (diagnose, repair, analyze, optimize)

## LLM endpoint integration
- [x] Ensure router base URL and hvsock mappings are consistent with actual services
- [x] Add automatic detection of local endpoints (host vs runtime)

## Testing + QA
- [x] Add tests for module fallbacks when external scripts are missing
- [ ] Add tests for native DLL availability and graceful fallbacks
  - [x] DLL availability tests
  - [x] Graceful fallback tests
- [x] Add integration tests for router tool schema coverage and JSON output compliance
- [x] Add unit tests for routed JSON enforcement and TUI wrapper failures
- [x] Add integration tests for router tool execution with mock tool outputs
