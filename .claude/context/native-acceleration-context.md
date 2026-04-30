# PC_AI Native Acceleration Framework Context

> Last Updated: 2026-01-23
> Context Version: 1.0.0
> Phase: 3 COMPLETE (Performance Module)
> Latest Commit: 9404eb9 "feat(native): add Phase 3 Performance Module with FFI integration"

## 1. Project Overview

**PC_AI** is a Windows PC diagnostics and optimization agent with **native Rust acceleration**.

### Architecture Pattern: Rust DLL + C# Hybrid Framework

This follows the **NukeNul reference implementation** documented at:
`~/.claude/context/rust-csharp-hybrid-framework.md`

```
Rust DLL (FFI) --> C# P/Invoke Wrapper --> PowerShell Cmdlets
                                      |
                                      +--> Pester Integration Tests
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Native | Rust | High-performance FFI libraries |
| Interop | C# (.NET 8) | P/Invoke wrappers |
| Testing | Pester 5.x | FFI integration tests |
| CLI | PowerShell 7+ | User-facing cmdlets |

### Key Dependencies (Rust)

| Crate | Version | Purpose |
|-------|---------|---------|
| sysinfo | 0.32 | System/process monitoring |
| rayon | 1.10 | Parallel processing |
| ignore | 0.4 | Fast file walking (ripgrep engine) |
| walkdir | 2.5 | Directory traversal |
| sha2 | 0.10 | SHA-256 hashing |
| regex | 1.11 | Pattern matching |
| parking_lot | 0.12 | Efficient concurrency primitives |
| serde/serde_json | 1.0 | JSON serialization |
| windows-sys | 0.59 | Windows API bindings |
| widestring | 1.1 | UTF-16 for Win32 |

---

## 2. Current State - Phase 3 COMPLETE

### Implementation Progress

| Phase | Module | Status | Tests |
|-------|--------|--------|-------|
| 1 | pcai_core_lib | COMPLETE | 16 FFI, 26 Unit |
| 2 | pcai_search | COMPLETE | 21 FFI, 14 Unit |
| 3 | pcai_performance | COMPLETE | 25 FFI, 19 Unit |
| 4 | pcai_system | PLANNED | - |

### Test Summary

- **FFI Integration Tests**: 62 passing (Core: 16, Search: 21, Performance: 25)
- **Rust Unit Tests**: 59 passing (Core: 26, Search: 14, Performance: 19)
- **Total**: 121 tests passing

### Module Capabilities

#### pcai_core_lib (Foundation)
- PcaiStatus enum for error codes
- PcaiStringBuffer for cross-FFI string handling
- PcaiResult for operation results
- CPU count and version APIs
- Memory allocation/deallocation helpers

#### pcai_search (File Operations)
- `pcai_find_duplicates()` - Parallel duplicate file detection by hash
- `pcai_find_files()` - Glob pattern file search
- `pcai_search_content()` - Regex content search with context lines

#### pcai_performance (System Monitoring)
- `pcai_get_disk_usage()` - Directory size analysis
- `pcai_get_process_list()` - Process enumeration with CPU/memory
- `pcai_get_memory_stats()` - System memory statistics

---

## 3. Design Decisions

### FFI Contract

**Rust side**: Export `#[repr(C)]` structs and `extern "C"` functions

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaiStatus {
    Success = 0,
    InvalidArgument = 1,
    NullPointer = 2,
    // ...
}

#[repr(C)]
pub struct PcaiStringBuffer {
    pub status: PcaiStatus,
    pub data: *mut c_char,
    pub length: usize,
}

#[no_mangle]
pub extern "C" fn pcai_xxx() -> PcaiStringBuffer { ... }
```

**C# side**: Use `[DllImport]` with `[StructLayout(LayoutKind.Sequential)]`

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct PcaiStringBuffer {
    public PcaiStatus Status;
    public IntPtr Data;
    public UIntPtr Length;
}

[DllImport("pcai_xxx.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern PcaiStringBuffer pcai_xxx(
    [MarshalAs(UnmanagedType.LPUTF8Str)] string path);
```

### String Handling Pattern

**Critical**: Memory ownership transfers from Rust to C#. Always free in finally block.

```csharp
public static string? GetResult(string path)
{
    var buffer = NativeXxx.pcai_xxx(path);
    try
    {
        return buffer.IsValid ? buffer.ToManagedString() : null;
    }
    finally
    {
        if (buffer.Data != IntPtr.Zero)
        {
            NativeCore.pcai_free_string_buffer(ref buffer);
        }
    }
}
```

### Error Handling

All operations return `PcaiStatus` or include it in result structs:

| Code | Name | Meaning |
|------|------|---------|
| 0 | Success | Operation completed |
| 1 | InvalidArgument | Bad parameter |
| 2 | NullPointer | Null input |
| 3 | InvalidUtf8 | Bad encoding |
| 4 | PathNotFound | Path doesn't exist |
| 5 | PermissionDenied | Access denied |
| 6 | IoError | I/O failure |
| 12 | JsonError | Serialization failed |

### JSON Output

All complex results serialize to JSON for LLM consumption:

```rust
pub fn json_to_buffer<T: serde::Serialize>(value: &T) -> PcaiStringBuffer {
    match serde_json::to_string(value) {
        Ok(json) => PcaiStringBuffer::from_string(&json),
        Err(_) => PcaiStringBuffer::error(PcaiStatus::JsonError),
    }
}
```

---

## 4. Code Patterns

### Rust FFI Function Template

```rust
/// Returns JSON array of items.
///
/// # Safety
/// - `path` must be valid null-terminated UTF-8 or null
/// - Returned buffer must be freed with `pcai_free_string_buffer`
#[no_mangle]
pub extern "C" fn pcai_xxx(
    path: *const c_char,
    options: u32,
) -> PcaiStringBuffer {
    // Validate inputs
    let path = match pcai_core_lib::path::path_from_c_str(path) {
        Some(p) => p,
        None => return PcaiStringBuffer::error(PcaiStatus::InvalidArgument),
    };

    // Do work
    let results = match internal_function(&path, options) {
        Ok(r) => r,
        Err(e) => return PcaiStringBuffer::error(e.into()),
    };

    // Return JSON
    pcai_core_lib::string::json_to_buffer(&results)
}
```

### C# P/Invoke Template

```csharp
internal static partial class NativeXxx
{
    private const string DllName = "pcai_xxx.dll";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_xxx(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? path,
        uint options);
}

public static class XxxModule
{
    public static List<Item>? GetItems(string path, uint options = 0)
    {
        var buffer = NativeXxx.pcai_xxx(path, options);
        try
        {
            if (!buffer.IsValid) return null;
            var json = buffer.ToManagedString();
            return JsonSerializer.Deserialize<List<Item>>(json!);
        }
        finally
        {
            if (buffer.Data != IntPtr.Zero)
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }
    }
}
```

### Pester Test Template

```powershell
BeforeDiscovery {
    # Check DLL availability for Skip conditions
    $script:ModuleAvailable = $false
    $dll = Join-Path $PSScriptRoot "..\..\bin\pcai_xxx.dll"
    if (Test-Path $dll) {
        try {
            Add-Type -Path (Join-Path $PSScriptRoot "..\..\bin\PcaiNative.dll")
            $script:ModuleAvailable = [PcaiNative.XxxModule]::Test()
        } catch { }
    }
}

BeforeAll {
    Add-Type -Path (Join-Path $PSScriptRoot "..\..\bin\PcaiNative.dll")
}

Describe "Xxx Module" -Tag "FFI", "Unit" {
    It "Should return valid results" -Skip:(-not $script:ModuleAvailable) {
        $result = [PcaiNative.XxxModule]::GetItems($TestDrive)
        $result | Should -Not -BeNullOrEmpty
    }
}
```

---

## 5. Key Files

### Rust Source (Native/pcai_core/)

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace configuration |
| `pcai_core_lib/src/lib.rs` | Core FFI utilities entry |
| `pcai_core_lib/src/error.rs` | PcaiStatus enum |
| `pcai_core_lib/src/string.rs` | PcaiStringBuffer |
| `pcai_core_lib/src/result.rs` | PcaiResult struct |
| `pcai_core_lib/src/path.rs` | Path utilities |
| `pcai_search/src/lib.rs` | Search module entry |
| `pcai_search/src/duplicates.rs` | Duplicate detection |
| `pcai_search/src/files.rs` | File search |
| `pcai_search/src/content.rs` | Content search |
| `pcai_performance/src/lib.rs` | Performance module entry |
| `pcai_performance/src/disk.rs` | Disk usage analysis |
| `pcai_performance/src/process.rs` | Process monitoring |
| `pcai_performance/src/memory.rs` | Memory statistics |

### C# Wrappers (Native/PcaiNative/)

| File | Purpose |
|------|---------|
| `PcaiNative.csproj` | .NET 8 project |
| `PcaiNative.cs` | Core types and P/Invoke |
| `SearchModule.cs` | Search wrapper |
| `PerformanceModule.cs` | Performance wrapper |

### Pester Tests (Tests/Integration/)

| File | Tests | Purpose |
|------|-------|---------|
| `FFI.Core.Tests.ps1` | 16 | Core library |
| `FFI.Search.Tests.ps1` | 21 | Search module |
| `FFI.Performance.Tests.ps1` | 25 | Performance module |

### Build Output (bin/)

| File | Source |
|------|--------|
| `pcai_core_lib.dll` | Rust |
| `pcai_search.dll` | Rust |
| `pcai_performance.dll` | Rust |
| `PcaiNative.dll` | C# |

---

## 6. Build Commands

### Rust Build

```bash
# Navigate to workspace
cd C:\Users\david\PC_AI\Native\pcai_core

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run Rust unit tests
cargo test --all

# Output location
# T:\RustCache\cargo-target\release\*.dll
```

### C# Build

```bash
cd C:\Users\david\PC_AI\Native\PcaiNative

# Debug build
dotnet build

# Release build
dotnet build -c Release

# Output: Native\PcaiNative\bin\Release\net8.0\win-x64\
```

### Copy DLLs to bin/

```powershell
$RustTarget = "T:\RustCache\cargo-target\release"
$CSharpOut = "C:\Users\david\PC_AI\Native\PcaiNative\bin\Release\net8.0\win-x64"
$Bin = "C:\Users\david\PC_AI\bin"

Copy-Item "$RustTarget\pcai_core_lib.dll" $Bin
Copy-Item "$RustTarget\pcai_search.dll" $Bin
Copy-Item "$RustTarget\pcai_performance.dll" $Bin
Copy-Item "$CSharpOut\PcaiNative.dll" $Bin
```

### Run Tests

```powershell
# All FFI tests
Invoke-Pester -Path 'C:\Users\david\PC_AI\Tests\Integration\FFI.*.Tests.ps1'

# Specific module
Invoke-Pester -Path 'C:\Users\david\PC_AI\Tests\Integration\FFI.Performance.Tests.ps1' -Output Detailed

# With coverage
Invoke-Pester -Path 'Tests\Integration' -CodeCoverage 'Native\PcaiNative\*.cs'
```

---

## 7. Future Roadmap

### Phase 4: System Module (Next)

Create `pcai_system` crate with:
- PATH analysis (duplicates, invalid entries)
- Event log search (System, Application, Security)
- Registry query helpers
- Service status monitoring

### Phase 5: Integration

Update PowerShell cmdlets to use native modules:
- Add `-UseNative` switch to existing cmdlets
- Graceful fallback when DLLs unavailable
- Performance benchmarks (native vs PowerShell)

### Phase 6: CI/CD

- GitHub Actions workflow for Rust + C# build
- Automatic DLL artifact publishing
- Cross-platform build matrix (future: Linux)

---

## 8. Troubleshooting

### DLL Not Found

```powershell
# Check bin directory
Get-ChildItem C:\Users\david\PC_AI\bin\*.dll

# Verify Rust target directory
Get-ChildItem T:\RustCache\cargo-target\release\pcai*.dll
```

### Version Mismatch

```powershell
# Check DLL versions match
[PcaiNative.PcaiCore]::Version
[PcaiNative.SearchModule]::GetVersion()
[PcaiNative.PerformanceModule]::GetVersion()
```

### Memory Leaks

Always use try/finally pattern for string buffers:

```csharp
var buffer = NativeXxx.pcai_xxx();
try {
    // Use buffer
} finally {
    NativeCore.pcai_free_string_buffer(ref buffer);
}
```

---

## 9. Session Restoration

### Quick Start

```powershell
cd C:\Users\david\PC_AI

# Verify DLLs present
Get-ChildItem bin\*.dll | Select Name

# Run FFI tests
Invoke-Pester -Path 'Tests\Integration\FFI.*.Tests.ps1' -Output Minimal

# Test specific module
Add-Type -Path bin\PcaiNative.dll
[PcaiNative.PcaiCore]::GetDiagnostics() | ConvertTo-Json
```

### Full Verification

```powershell
# Build everything
cd Native\pcai_core && cargo build --release
cd ..\PcaiNative && dotnet build -c Release

# Copy DLLs
# (use copy commands from Build Commands section)

# Run all tests
Invoke-Pester -Path 'Tests\Integration\FFI.*.Tests.ps1' -Output Detailed
```

---

## 10. Context Metadata

```yaml
project: PC_AI Native Acceleration Framework
location: C:\Users\david\PC_AI\Native
context_version: 1.0.0
last_updated: 2026-01-23
latest_commit: 9404eb9
current_phase: 3 (Performance Module Complete)
next_phase: 4 (System Module)

languages:
  - Rust (FFI libraries)
  - C# (.NET 8 P/Invoke)
  - PowerShell (Pester tests)

modules_complete:
  - pcai_core_lib (16 FFI, 26 unit tests)
  - pcai_search (21 FFI, 14 unit tests)
  - pcai_performance (25 FFI, 19 unit tests)

modules_planned:
  - pcai_system (PATH, logs, registry)

test_totals:
  ffi_integration: 62
  rust_unit: 59
  total: 121

reference_implementation: NukeNul (C:\Users\david\PC_AI\Native\NukeNul)
```

