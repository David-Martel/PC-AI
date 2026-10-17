# Temporary native library ownership controls

The [two controls](test_native_library_lifetime.py) copy the production managed bridge unchanged into a private build directory and invoke its actual helper in fresh .NET8 processes. This isolates build outputs from shared agent work. The [register](native-library-lifetime-review.json) records independent review by Codex benchmark_fix_review and the integrating root reviewer.

The rejection control compiles a real partial Rust shared library, confirms it starts unloaded, calls PcaiCore.VerifyLibraryExports, requires rejection and missing exports, then checks it is no longer loaded before child exit. Ubuntu uses libdl RTLD_NOLOAD; Windows uses exact kernel32 GetModuleHandleW. The control observes loader state without keeping its own reference; Linux closes its temporary observation handle if one exists. Original54b5af0 fails with loadedAfter=true. The repaired implementation passes.

The real-core control supplies an explicit host artifact, requires IsAvailable and a UTF-8 string-copy round trip. It protects against accidentally releasing the legitimate P/Invoke-owned handle while cleaning temporary admission references. Missing artifact yields explicit SKIP; CI must supply its host-built artifact to count this as qualification. Two controls passed on Ubuntu/private .NET8.0.32 without skips. Windows execution is pending.

All process calls use subprocess.run with captured stdout/stderr and explicit20/120/180-second bounds. Its communicate behavior drains both output pipes and handles child timeout, unlike read-before-wait C# controls. These are disposable validation limits, not production network settings. Compilation may launch toolchain descendants; this slice does not claim a general process-tree supervisor.

This exercises VerifyLibraryExports rejection and valid real-core admission/operation paths. It does not execute EvaluateAvailabilityCore's unconfigured defaultHandle branch directly, prove every resolver lifetime, characterize steady-state performance, validate all C ABI signatures, or qualify physical USB4 networking.

Root corrected an escaped-ASCII fixture to actual Unicode (café 你好) before final approval. Independent current-source controls pass twice; the original source fails the rejection/unload assertion. The admission workflow now runs these controls with its real artifact and requires the full managed suite to execute at least its current 49-test inventory with all tests passing and none skipped. This count is an inventory floor, not proof of complete ABI coverage.
