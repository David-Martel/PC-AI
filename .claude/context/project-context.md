# NukeNul Project Context

**Last Updated**: 2026-01-23
**Status**: FULLY IMPLEMENTED AND TESTED

## Project Overview

**NukeNul** is a high-performance Windows reserved filename cleaner using a hybrid Rust/C# architecture.

### Purpose

Delete Windows reserved filenames that cannot be removed through normal file operations:

- `nul`, `con`, `prn`, `aux`
- `com1` through `com9`
- `lpt1` through `lpt9`

### Technology Stack

- **Rust DLL** (`nuker_core.dll`): Parallel file walking and Win32 API deletion
- **C# CLI** (`NukeNul.exe`): User interface with JSON output
- **.NET 8**: Framework-dependent build with Native AOT support
- **Key Rust Crates**: `ignore` (ripgrep's file walker), `windows-sys`, `widestring`

## Current State

### Build Status: COMPLETE

- Rust DLL builds successfully (optimized release mode)
- C# CLI builds successfully (.NET 8 win-x64)
- Integration tests passing with 100% deletion success rate

### Performance Metrics

- **12 files**: ~8ms execution time
- **Thread utilization**: Parallel (scales to CPU core count)
- **Walker**: ripgrep's `ignore` crate with work-stealing queue

### Build Outputs Location

```
C:\Users\david\PC_AI\Native\NukeNul\bin\Release\net8.0\win-x64\
  - NukeNul.exe    (C# CLI)
  - nuker_core.dll (Rust DLL)
```

## Project Structure

```
C:\Users\david\PC_AI\Native\NukeNul\
|-- nuker_core/                 # Rust DLL project
|   |-- Cargo.toml              # Rust dependencies and build config
|   |-- Cargo.lock              # Locked dependency versions
|   |-- src/
|   |   |-- lib.rs              # 327 lines - Core implementation
|   |-- target/release/         # Rust build output
|       |-- nuker_core.dll
|
|-- Program.cs                  # 243 lines - C# CLI application
|-- NukeNul.csproj              # .NET 8 project configuration
|-- build.ps1                   # Master build orchestration script
|-- test.ps1                    # Integration test script
|-- bin/Release/net8.0/win-x64/ # Final build outputs
```

**IMPORTANT**: C# files are in the ROOT directory, not in a subdirectory.

## Key Design Decisions

### 1. Hybrid Architecture (Rust + C#)

- **Rationale**: Rust provides safe low-level Win32 API access and parallel performance; C# provides familiar CLI patterns and JSON serialization
- **Alternative considered**: Pure PowerShell (too slow), Pure Rust CLI (less familiar to users)

### 2. ripgrep's `ignore` crate for File Walking

- **Rationale**: Battle-tested parallel walker with work-stealing for load balancing
- **Features used**: Multi-threaded traversal, .git directory filtering
- **Configuration**: `hidden(false)`, all gitignore settings disabled

### 3. Direct Win32 DeleteFileW API

- **Rationale**: Standard Rust `fs::remove_file` cannot delete reserved filenames
- **Implementation**: Extended-length path prefix (`\\?\`) bypasses Windows path normalization
- **Safety**: All unsafe blocks documented with safety invariants

### 4. JSON Output Format

- **Rationale**: Machine-readable for LLM/automation consumption
- **Implementation**: Source-generated JSON serialization for AOT compatibility
- **Fields**: `tool`, `target`, `timestamp`, `status`, `performance`, `results`

### 5. Native AOT Support

- **Rationale**: Fast startup for CLI tool, smaller dependency footprint
- **Configuration**: `PublishAot=true`, `TrimMode=full`, `IlcOptimizationPreference=Speed`

## Code Patterns

### FFI Interface (Rust side)

```rust
#[repr(C)]
pub struct ScanStats {
    pub files_scanned: u32,
    pub files_deleted: u32,
    pub errors: u32,
}

#[no_mangle]
pub extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats
```

### P/Invoke (C# side)

```csharp
[StructLayout(LayoutKind.Sequential)]
internal struct ScanStats {
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

[DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern ScanStats nuke_reserved_files(
    [MarshalAs(UnmanagedType.LPStr)] string rootPath);
```

### Reserved Filename Matching

- Case-insensitive comparison using `eq_ignore_ascii_case`
- Matches only exact filename (no extensions)
- 22 reserved names total

### Extended-Length Path Handling

```rust
// Regular path: C:\path -> \\?\C:\path
// UNC path: \\server\share -> \\?\UNC\server\share
let extended_path = format!("\\\\?\\{}", path_str);
```

## Build Commands

### Full Build

```powershell
cd C:\Users\david\PC_AI\Native\NukeNul
.\build.ps1
```

### Build Options

```powershell
.\build.ps1 -Publish                   # Self-contained executable
.\build.ps1 -Clean                     # Clean before build
.\build.ps1 -SkipRust                  # Skip Rust build
.\build.ps1 -SkipCSharp                # Skip C# build
```

### Manual Build Steps

```powershell
# Rust DLL
cd nuker_core
cargo build --release

# C# CLI
cd ..
copy nuker_core\target\release\nuker_core.dll .
dotnet build -c Release
```

## Test Commands

### Integration Tests

```powershell
.\test.ps1                             # Standard test (10 files)
.\test.ps1 -TestCount 100              # Stress test
.\test.ps1 -DeepNesting                # Nested directory test
.\test.ps1 -KeepTestDir                # Keep test artifacts
```

### Manual Testing

```powershell
cd bin\Release\net8.0\win-x64
.\NukeNul.exe .                        # Scan current directory
.\NukeNul.exe C:\path\to\scan          # Scan specific path
```

## JSON Output Format

### Success Response

```json
{
    "tool": "Nuke-Nul",
    "target": "C:\\path\\to\\scan",
    "timestamp": "2026-01-23T10:30:00Z",
    "status": "Success",
    "performance": {
        "mode": "Rust/Parallel",
        "threads": 22,
        "elapsed_ms": 8
    },
    "results": {
        "scanned": 150,
        "deleted": 12,
        "errors": 0
    }
}
```

### Error Response

```json
{
    "tool": "Nuke-Nul",
    "status": "Error",
    "message": "Target directory does not exist: C:\\nonexistent"
}
```

## Exit Codes

- `0`: Success (all files deleted, no errors)
- `1`: Invalid path or validation error
- `2`: DLL not found or load failure
- `3`: Partial success (some files deleted, some errors)
- `99`: Unexpected error

## Dependencies

### Rust (nuker_core)

```toml
[dependencies]
ignore = "0.4"              # Parallel file walker
widestring = "1.1"          # UTF-16 string conversion
windows-sys = "0.52"        # Win32 API bindings
libc = "0.2"                # C FFI types
```

### C# (NukeNul.csproj)

```xml
<PackageReference Include="System.Text.Json" Version="8.0.0" />
```

## Common Issues and Solutions

### Issue: DLL Not Found

- **Cause**: `nuker_core.dll` not in same directory as `NukeNul.exe`
- **Solution**: Run `.\build.ps1` which copies DLL to output directory

### Issue: Access Denied Errors

- **Cause**: File locked by another process
- **Solution**: Close applications using the file; check results.errors count

### Issue: Build Fails on Rust Side

- **Cause**: Missing Rust toolchain
- **Solution**: Install Rust from https://rustup.rs/

### Issue: Build Fails on C# Side

- **Cause**: Missing .NET 8 SDK
- **Solution**: Install from https://dotnet.microsoft.com/download

## Future Enhancements (Not Implemented)

1. **Dry-run mode**: List files without deleting
2. **Verbose output**: Per-file deletion logging
3. **Custom patterns**: User-defined filenames to delete
4. **Recursive depth limit**: Control scan depth
5. **Progress reporting**: Real-time scan progress

## Related Files

- `README.md` - User documentation
- `BUILD.md` - Detailed build instructions
- `QUICKSTART.md` - Quick setup guide
- `PROJECT_COMPLETE.md` - Completion summary
- `delete-nul-files.ps1` - Original PowerShell implementation (for comparison)

## Session Notes

### 2026-01-23 Context Save

- Project fully implemented and tested
- All tests passing with 100% deletion success
- Performance verified: 8ms for 12 files
- Build scripts verified to work with flat structure (C# in root)
- JSON output working with source-generated serialization
