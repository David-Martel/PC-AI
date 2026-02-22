# FunctionGemma Tool Catalog

Source: `Config/pcai-tools.json`

## Tools

### SearchDocs
Search vendor documentation for error codes and device guidance.

Parameters:
- query (string, required)
- source (string, optional) enum=Microsoft, Intel, AMD, Dell, HP, Lenovo, Generic

### GetSystemInfo
Query system diagnostics for storage, network, USB, BIOS, or OS details.

Parameters:
- category (string, required) enum=Storage, Network, USB, BIOS, OS, GPU, DiskDrive, Net, Display, Media, HIDClass
- detail (string, optional)

### SearchLogs
Search local log files for a regex pattern (native-first).

Parameters:
- pattern (string, required)
- rootPath (string, optional)
- filePattern (string, optional)
- caseSensitive (boolean, optional)
- contextLines (integer, optional)
- maxMatches (integer, optional)

### pcai_get_device_errors
Get devices reporting ConfigManager errors (Device Manager).

Parameters:

### pcai_get_disk_status
Get disk SMART/status details for physical drives.

Parameters:

### pcai_get_system_events
Get recent system events related to disk/USB/storage issues.

Parameters:
- days (integer, optional)

### pcai_get_usb_status
Get USB device and controller status (optionally errors only).

Parameters:
- errors_only (boolean, optional)

### pcai_get_network_status
Get physical network adapter status.

Parameters:

### pcai_get_llm_status
Get status of local LLM endpoints/providers.

Parameters:

### pcai_get_service_health
Check pcai-inference, FunctionGemma, WSL, Docker, and GPU health.

Parameters:
- distribution (string, optional)
- pcai_inference_url (string, optional)
- functiongemma_url (string, optional)
- check_legacy_providers (boolean, optional)

### pcai_get_wsl_status
Get WSL status, versions, and distributions.

Parameters:
- detailed (boolean, optional)

### pcai_run_wsl_docker_health_check
Run WSL/Docker health check (with fallback if script is missing).

Parameters:
- auto_recover (boolean, optional)
- quick (boolean, optional)
- verbose (boolean, optional)

### pcai_get_disk_usage
Get high-performance disk usage summary for a path.

Parameters:
- path (string, optional)
- top (integer, optional)

### pcai_get_top_processes
Get top resource-consuming processes.

Parameters:
- sort_by (string, optional) enum=memory, cpu
- top (integer, optional)

### pcai_get_memory_stats
Get system memory statistics (native).

Parameters:

### pcai_run_wsl_network_tool
Run the WSL network toolkit with a mode: check, diagnose, repair, full.

Parameters:
- mode (string, required) enum=check, diagnose, repair, full

### pcai_get_wsl_health
Collect WSL and Docker environment health summary.

Parameters:

### pcai_optimize_model_host
Tune WSL and GPU resources for performance/safety.

Parameters:
- gpu_limit (number, optional)

### pcai_restart_wsl
Restart WSL to reinitialize networking and services.

Parameters:

### pcai_get_docker_status
Return Docker Desktop health and runtime status.

Parameters:

### pcai_set_provider_order
Update LLM provider fallback order (comma-separated list).

Parameters:
- order (string, required)

### pcai_start_service
Start a PC_AI service (e.g., PC_AI-VLLM, PC_AI-HVSockProxy).

Parameters:
- service (string, required)

### pcai_stop_service
Stop a PC_AI service (e.g., PC_AI-VLLM, PC_AI-HVSockProxy).

Parameters:
- service (string, required)

### pcai_restart_service
Restart a PC_AI service.

Parameters:
- service (string, required)

### pcai_native_inference_status
Get the status of native inference backend (PcaiInference FFI).

Parameters:

### pcai_switch_inference_backend
Switch the active native inference backend (llamacpp or mistralrs).

Parameters:
- backend (string, required) enum=llamacpp, mistralrs, auto

### pcai_load_model
Load a GGUF or SafeTensors model into the native inference backend.

Parameters:
- model_path (string, required)
- gpu_layers (integer, optional)

### pcai_run_evaluation
Run an evaluation suite against the active inference backend.

Parameters:
- suite_name (string, required)
- dataset (string, optional)
- metrics (string, optional)

## Negative examples (NO_TOOL)
- Hello, how are you today?
- What is the capital of France?
- Tell me a joke.
- How do I cook pasta?
- Write a poem about the sea.

## Notes
- Negative examples map to `NO_TOOL` responses.
- Keep this file in sync with `Deploy/rust-functiongemma-train/src/schema_utils.rs`.
