# FFI Implementation for PowerShell Integration

## Overview

The `pcai-inference` library now provides a complete C FFI layer enabling direct integration with PowerShell via P/Invoke, bypassing the need for HTTP server overhead.

## Architecture

### Global State Management

```rust
struct GlobalState {
    runtime: Result<Runtime, String>,            // Tokio runtime (error reported per call)
    backend: RwLock<Option<Box<dyn InferenceBackend>>>,
    generation_gate: Mutex<()>,                  // one generation at a time
    has_backend: AtomicBool,                     // lock-free status mirrors
    model_loaded: AtomicBool,
    async_engine: AsyncEngine,                   // request table + single worker
}

static GLOBAL_STATE: OnceLock<GlobalState>;      // Thread-safe singleton
```

**Design Decisions:**
- **OnceLock**: Ensures runtime is created exactly once
- **RwLock + generation gate**: generating takes the gate, then a read lock on the
  backend; init / load / shutdown take the write lock. Generations stay serialised.
- **Separate async request table**: `pcai_poll_result` and `pcai_cancel` only touch a
  short-lived mutex on the table, never the backend or the gate, so they return in
  microseconds while a generation runs. See `src/ffi/async_engine.rs` for the request
  life cycle, cancellation semantics (llama.cpp stops per token; other backends
  discard the late result), the bounded queue (16) and the eviction rules (terminal
  entries are freed on poll, evicted after 10 minutes, table capped at 256).
- **Tokio Runtime**: Bridges async backend methods to synchronous FFI calls

### Error Handling

```rust
thread_local! {
    static LAST_ERROR: RefCell<Option<String>>;
}
```

**Features:**
- Thread-local storage prevents error cross-contamination
- Errors never panic across FFI boundary
- `pcai_last_error()` retrieves error without copying

## FFI Functions

### 1. pcai_init

```c
int pcai_init(const char* backend_name);
```

**Purpose:** Initialize the inference backend

**Parameters:**
- `backend_name`: "llamacpp" or "mistralrs"

**Returns:**
- `0`: Success
- `-1`: Error (check `pcai_last_error()`)

**Example:**
```powershell
[PCAIInference]::pcai_init("mistralrs")
```

### 2. pcai_load_model

```c
int pcai_load_model(const char* model_path, int gpu_layers);
```

**Purpose:** Load a model into the backend

**Parameters:**
- `model_path`: Path to GGUF or SafeTensors model
- `gpu_layers`: Number of layers to offload (-1 = all, 0 = CPU only)

**Returns:**
- `0`: Success
- `-1`: Error

**Implementation Notes:**
- For llamacpp backend, recreates backend with specified GPU layers
- Calls async `load_model()` via `runtime.block_on()`
- Uses destructuring to satisfy borrow checker: `let GlobalState { runtime, backend } = &mut *guard;`

**Example:**
```powershell
[PCAIInference]::pcai_load_model("C:\models\phi-3.gguf", -1)
```

### 3. pcai_generate

```c
char* pcai_generate(const char* prompt, uint32_t max_tokens, float temperature);
```

**Purpose:** Generate text synchronously

**Parameters:**
- `prompt`: Input text
- `max_tokens`: Maximum tokens to generate (0 = default 512)
- `temperature`: Sampling temperature (0.0 = greedy)

**Returns:**
- Pointer to generated text (caller must free with `pcai_free_string`)
- `null`: Error

**Example:**
```powershell
$ptr = [PCAIInference]::pcai_generate("Explain Rust:", 100, 0.7)
$text = [Marshal]::PtrToStringUTF8($ptr)
[PCAIInference]::pcai_free_string($ptr)
```

### 4. pcai_generate_streaming

```c
typedef void (*TokenCallback)(const char* token, void* user_data);

int pcai_generate_streaming(
    const char* prompt,
    uint32_t max_tokens,
    float temperature,
    TokenCallback callback,
    void* user_data
);
```

**Purpose:** Generate text with token-by-token streaming

**Parameters:**
- `prompt`: Input text
- `max_tokens`: Maximum tokens
- `temperature`: Sampling temperature
- `callback`: Function called for each token
- `user_data`: Pointer passed to callback

**Returns:**
- `0`: Success
- `-1`: Error

**Limitations:**
- Only supported for llamacpp backend
- Requires unsafe downcast from trait object

**Implementation:**
```rust
let backend_ptr = backend.as_ref() as *const dyn InferenceBackend as *const LlamaCppBackend;
let llamacpp_backend = unsafe { &*backend_ptr };
```

### 5. pcai_free_string

```c
void pcai_free_string(char* str);
```

**Purpose:** Free a string returned by `pcai_generate`

**Safety:**
- Must only be called on pointers from `pcai_generate`
- Do NOT call on `pcai_last_error()` results (those are leaked intentionally)

### 6. pcai_shutdown

```c
void pcai_shutdown();
```

**Purpose:** Unload model and free backend resources

**Behavior:**
- Calls `backend.unload_model()` asynchronously
- Drops backend, but runtime persists for reuse

### 7. pcai_last_error

```c
const char* pcai_last_error();
```

**Purpose:** Retrieve thread-local error message

**Returns:**
- Pointer to error string (valid until next FFI call)
- `null`: No error

**Implementation:**
```rust
// Leak CString to return stable pointer
let ptr = c_str.as_ptr();
std::mem::forget(c_str);
ptr
```

## PowerShell Integration

### P/Invoke Declarations

```powershell
Add-Type @"
using System;
using System.Runtime.InteropServices;

public class PCAIInference {
    private const string DLL_PATH = @"target\release\pcai_inference.dll";

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_init(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string backend_name
    );

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern int pcai_load_model(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string model_path,
        int gpu_layers
    );

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_generate(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
        uint max_tokens,
        float temperature
    );

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr str);

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_shutdown();

    [DllImport(DLL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_last_error();
}
"@
```

### Helper Functions

```powershell
function Invoke-PCAIInference {
    param(
        [string]$Prompt,
        [int]$MaxTokens = 512,
        [float]$Temperature = 0.7
    )

    $resultPtr = [PCAIInference]::pcai_generate($Prompt, $MaxTokens, $Temperature)

    if ($resultPtr -eq [IntPtr]::Zero) {
        $error = [Marshal]::PtrToStringUTF8([PCAIInference]::pcai_last_error())
        throw "Generation failed: $error"
    }

    $result = [Marshal]::PtrToStringUTF8($resultPtr)
    [PCAIInference]::pcai_free_string($resultPtr)
    return $result
}
```

## Build Instructions

### With mistralrs backend (recommended for Windows)

```bash
cargo build --no-default-features --features ffi,mistralrs-backend --release
```

**Output:** `target\release\pcai_inference.dll`

### With llamacpp backend

```bash
# Requires MSVC compiler properly configured
cargo build --features ffi,llamacpp --release
```

**Known Issues:**
- llama-cpp-sys-2 has Windows build issues with GNU toolchain
- Requires Visual Studio Build Tools with C++ workload

## Thread Safety

### FFI Layer
- ✅ All functions are thread-safe
- ✅ Backend behind an RwLock; async request table behind its own short-lived Mutex
- ✅ Poll and cancel never wait for a running generation
- ✅ Errors are thread-local

### Backend Implementations
- ✅ `LlamaCppBackend`: Uses Arc<Mutex<Context>>
- ✅ `MistralRsBackend`: Uses Arc<Model>
- ⚠️  Concurrent calls to same backend will serialize

## Performance Considerations

### Overhead
- **Runtime creation:** One-time ~10ms cost
- **Request table lock (poll/cancel):** sub-microsecond uncontended
- **FFI call:** ~50ns overhead
- **String marshaling:** O(n) for prompt/response

### Optimization Tips
1. **Keep model loaded:** Avoid repeated load/unload cycles
2. **Batch prompts:** Reuse loaded model across multiple inferences
3. **Use streaming:** For real-time UX with llamacpp backend

## Testing

### Unit Tests

```rust
#[test]
fn test_init_null_backend() {
    let result = pcai_init(std::ptr::null());
    assert_eq!(result, -1);
    assert!(!pcai_last_error().is_null());
}

#[test]
fn test_generate_before_init() {
    pcai_shutdown();
    let prompt = CString::new("test").unwrap();
    let result = pcai_generate(prompt.as_ptr(), 10, 0.7);
    assert!(result.is_null());
}
```

### Integration Test (PowerShell)

See `examples/powershell_ffi.ps1` for complete example.

## Limitations

1. **Global State:** Only one backend active at a time
2. **Streaming:** Only supported for llamacpp backend
3. **GPU Configuration:** Must set before loading model (llamacpp only)
4. **Error Context:** Limited to string messages (no structured errors)
5. **Model Switching:** Requires shutdown + reinit to change backends

## Future Enhancements

- [ ] Multi-backend support (multiple concurrent backends)
- [ ] Structured error codes (not just strings)
- [ ] Progress callbacks for model loading
- [ ] Streaming support for mistralrs backend
- [ ] Model metadata queries (vocab size, context length, etc.)
- [ ] Fine-grained GPU layer control for mistralrs

## Security Considerations

- **Input Validation:** All string pointers checked for null
- **UTF-8 Validation:** Invalid UTF-8 returns error, not panic
- **Memory Safety:** No buffer overruns (Rust guarantees)
- **Resource Cleanup:** RAII ensures cleanup even on panic

## Troubleshooting

### "Backend not initialized"
```powershell
# Call pcai_init before other functions
[PCAIInference]::pcai_init("mistralrs")
```

### "Model not loaded"
```powershell
# Call pcai_load_model after init
[PCAIInference]::pcai_load_model("path/to/model.gguf", -1)
```

### "DllNotFoundException"
```powershell
# Ensure DLL path is correct and file exists
Test-Path "target\release\pcai_inference.dll"
```

### "Failed to load model: ..."
```powershell
# Check error details
$error = [Marshal]::PtrToStringUTF8([PCAIInference]::pcai_last_error())
Write-Host "Error: $error"
```

## References

- **Rust FFI:** https://doc.rust-lang.org/nomicon/ffi.html
- **PowerShell P/Invoke:** https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.dllimportattribute
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **mistral.rs:** https://github.com/EricLBuehler/mistral.rs
