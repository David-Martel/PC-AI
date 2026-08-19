# NukeNul Build and Test Guide

Complete guide for building, testing, and deploying the NukeNul hybrid Rust/C# project.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build System](#build-system)
4. [Testing](#testing)
5. [Manual Testing](#manual-testing)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

---

## Prerequisites

### Required Tools

1. **Rust Toolchain**
   ```powershell
   # Install Rust via rustup
   winget install Rustlang.Rustup

   # Or download from https://rustup.rs/

   # Verify installation
   cargo --version
   rustc --version
   ```

2. **.NET SDK 8.0+**
   ```powershell
   # Install .NET SDK
   winget install Microsoft.DotNet.SDK.8

   # Or download from https://dotnet.microsoft.com/download

   # Verify installation
   dotnet --version
   ```

3. **PowerShell 7+ (Recommended)**
   ```powershell
   # Install PowerShell 7
   winget install Microsoft.PowerShell

   # Verify installation
   pwsh --version
   ```

### System Requirements

- **OS**: Windows 10/11 (x64)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 500MB free space for build artifacts
- **CPU**: Multi-core recommended for parallel builds

---

## Quick Start

### 1. Clone or Navigate to Project

```powershell
cd C:\codedev\nukenul
```

### 2. Build the Project

```powershell
# Standard release build
.\build.ps1

# Clean build (removes all artifacts first)
.\build.ps1 -Clean

# Debug build
.\build.ps1 -Configuration Debug

# Self-contained executable (no .NET runtime required)
.\build.ps1 -Publish
```

### 3. Run Tests

```powershell
# Standard integration tests
.\test.ps1

# Stress test with 100 files
.\test.ps1 -TestCount 100

# Deep nesting test
.\test.ps1 -DeepNesting

# Keep test directory for inspection
.\test.ps1 -KeepTestDir
```

### 4. Use the Tool

```powershell
# Using the wrapper (auto-detects best version)
.\delete-nul-files-v2.ps1

# Direct execution
.\bin\Release\net8.0\win-x64\NukeNul.exe .

# Scan specific directory
.\bin\Release\net8.0\win-x64\NukeNul.exe "C:\Projects"
```

---

## Build System

### build.ps1 - Master Build Script

#### Command-Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Configuration` | String | `Release` | Build configuration (`Debug` or `Release`) |
| `-Publish` | Switch | `$false` | Create self-contained executable |
| `-Clean` | Switch | `$false` | Clean artifacts before building |
| `-SkipRust` | Switch | `$false` | Skip Rust build (use existing DLL) |
| `-SkipCSharp` | Switch | `$false` | Skip C# build (Rust only) |

#### Build Phases

**Phase 1: Pre-flight Checks**
- Validates project structure
- Checks for Rust toolchain (cargo)
- Checks for .NET SDK (dotnet)
- Reports versions and system info

**Phase 2: Clean (Optional)**
- Removes Rust artifacts (`cargo clean`)
- Removes C# artifacts (`dotnet clean`, bin/obj)
- Removes copied DLL files

**Phase 3: Build Rust DLL**
- Compiles `nuker_core` crate as cdylib
- Uses release profile (optimized) or debug profile
- Outputs `nuker_core.dll` to `target/release/` or `target/debug/`
- Copies DLL to C# project directory

**Phase 4: Build C# CLI**
- Compiles `NukeNul.csproj` with .NET 8.0
- Framework-dependent build (default) or self-contained (with `-Publish`)
- Ensures `nuker_core.dll` is in output directory
- Reports executable size and location

**Phase 5: Build Summary**
- Lists all built artifacts
- Provides next steps and usage instructions

#### Examples

```powershell
# Clean release build
.\build.ps1 -Clean -Configuration Release

# Quick rebuild (Rust only)
.\build.ps1 -SkipCSharp

# Quick rebuild (C# only, reuses existing DLL)
.\build.ps1 -SkipRust

# Portable executable for distribution
.\build.ps1 -Publish -Clean

# Debug build for troubleshooting
.\build.ps1 -Configuration Debug
```

---

## Testing

### test.ps1 - Integration Test Script

#### Command-Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Configuration` | String | `Release` | Build configuration to test |
| `-TestCount` | Int | `10` | Number of "nul" files to create (1-10000) |
| `-DeepNesting` | Switch | `$false` | Create nested directory structure |
| `-SkipBenchmark` | Switch | `$false` | Skip performance comparison |
| `-KeepTestDir` | Switch | `$false` | Don't clean up test directory |

#### Test Phases

**Phase 1: Pre-Test Validation**
- Checks if `NukeNul.exe` exists
- Checks if `nuker_core.dll` exists
- Verifies executable is runnable

**Phase 2: Create Test Environment**
- Creates temporary test directory in `%TEMP%`
- Creates directory structure (flat or nested)
- Reports directory count and structure

**Phase 3: Create "NUL" Files**
- Creates specified number of "nul" files using `\\?\` prefix
- Distributes files across directory structure
- Creates normal files for context (1 per 3 nul files)
- Verifies file creation using .NET APIs

**Phase 4: Run NukeNul.exe**
- Executes `NukeNul.exe` against test directory
- Measures execution time
- Captures and parses JSON output
- Reports scan statistics

**Phase 5: Verify Deletion**
- Checks if all "nul" files were deleted
- Verifies normal files were NOT deleted
- Reports success rate

**Phase 6: Performance Benchmark (Optional)**
- Recreates test files
- Runs original PowerShell script for comparison
- Calculates speedup factor
- Reports comparative performance

**Phase 7: Cleanup**
- Removes test directory (unless `-KeepTestDir` specified)
- Reports cleanup status

**Phase 8: Test Summary**
- Reports overall test results
- Displays statistics
- Exits with code 0 (success) or 1 (failure)

#### Examples

```powershell
# Standard test with 10 files
.\test.ps1

# Stress test with 1000 files and nested directories
.\test.ps1 -TestCount 1000 -DeepNesting

# Quick test without benchmark
.\test.ps1 -SkipBenchmark

# Debug test (keeps directory for inspection)
.\test.ps1 -TestCount 5 -KeepTestDir

# Test debug build
.\test.ps1 -Configuration Debug
```

---

## Manual Testing

### Creating Test "NUL" Files Manually

```powershell
# Create a test directory
$TestDir = "C:\Temp\NulTest"
New-Item -ItemType Directory -Path $TestDir -Force

# Create "nul" files using .NET (PowerShell can't do this directly)
$NulPath = Join-Path $TestDir "nul"
$ExtendedPath = "\\?\$NulPath"
$FileStream = [System.IO.File]::Create($ExtendedPath)
$FileStream.Close()

# Verify file exists
[System.IO.File]::Exists($ExtendedPath)  # Should return True

# Run NukeNul
.\NukeNul\bin\Release\net8.0\NukeNul.exe $TestDir

# Verify file is deleted
[System.IO.File]::Exists($ExtendedPath)  # Should return False
```

### Testing in Real Projects

```powershell
# Scan your actual project (READ-ONLY test first)
# Modify NukeNul to only report, not delete for safety:
# Comment out DeleteFileW call in src/lib.rs

# Then run:
.\NukeNul\bin\Release\net8.0\NukeNul.exe "C:\YourProject"

# Review output JSON to see what would be deleted

# Once confident, uncomment DeleteFileW and rebuild
.\build.ps1 -SkipCSharp

# Run again to actually delete
.\NukeNul\bin\Release\net8.0\NukeNul.exe "C:\YourProject"
```

### Verification Checklist

- [ ] Build completes without errors
- [ ] `nuker_core.dll` is copied to C# output directory
- [ ] `NukeNul.exe` runs without crashing
- [ ] JSON output is well-formed
- [ ] "nul" files are detected
- [ ] "nul" files are deleted
- [ ] Normal files are NOT deleted
- [ ] Performance is better than PowerShell version

---

## Deployment

### Option 1: Portable Executable (Recommended)

```powershell
# Build self-contained executable
.\build.ps1 -Publish -Clean

# Locate executable
$PublishDir = ".\NukeNul\bin\Release\net8.0\win-x64\publish"
Get-ChildItem $PublishDir

# Copy to PATH location
Copy-Item "$PublishDir\NukeNul.exe" "C:\Windows\System32\NukeNul.exe"

# Or to user binaries
$UserBin = "$env:LOCALAPPDATA\Microsoft\WindowsApps"
Copy-Item "$PublishDir\NukeNul.exe" "$UserBin\NukeNul.exe"

# Test from anywhere
NukeNul.exe --help
```

### Option 2: Framework-Dependent (Smaller Size)

```powershell
# Standard build
.\build.ps1

# Copy both EXE and DLL
$BuildDir = ".\NukeNul\bin\Release\net8.0"
$DestDir = "C:\Tools\NukeNul"

New-Item -ItemType Directory -Path $DestDir -Force
Copy-Item "$BuildDir\NukeNul.exe" $DestDir
Copy-Item "$BuildDir\nuker_core.dll" $DestDir

# Add to PATH
$env:PATH += ";$DestDir"

# Make permanent
[Environment]::SetEnvironmentVariable("PATH", "$env:PATH", "User")
```

### Option 3: Wrapper Script Deployment

```powershell
# Deploy wrapper script
Copy-Item ".\delete-nul-files-v2.ps1" "C:\Tools\delete-nul-files.ps1"

# Create alias in PowerShell profile
Add-Content $PROFILE @"
function Remove-NulFiles {
    param([string]`$Path = ".")
    & "C:\Tools\delete-nul-files.ps1" -SearchPath `$Path
}
"@

# Reload profile
. $PROFILE

# Use anywhere
Remove-NulFiles "C:\Projects"
```

---

## Troubleshooting

### Build Issues

#### "cargo: command not found"

**Problem**: Rust toolchain not installed or not in PATH

**Solution**:
```powershell
# Install Rust
winget install Rustlang.Rustup

# Or manually add to PATH
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
```

#### "dotnet: command not found"

**Problem**: .NET SDK not installed or not in PATH

**Solution**:
```powershell
# Install .NET SDK
winget install Microsoft.DotNet.SDK.8

# Verify installation
dotnet --list-sdks
```

#### "nuker_core.dll not found"

**Problem**: Rust DLL wasn't copied to C# output directory

**Solution**:
```powershell
# Rebuild with clean
.\build.ps1 -Clean

# Or manually copy
$RustDll = ".\nuker_core\target\release\nuker_core.dll"
$CSharpDir = ".\NukeNul\bin\Release\net8.0"
Copy-Item $RustDll $CSharpDir
```

### Runtime Issues

#### "Unable to load DLL 'nuker_core.dll'"

**Problem**: DLL not found or wrong architecture

**Solution**:
```powershell
# Check DLL exists
Test-Path ".\NukeNul\bin\Release\net8.0\nuker_core.dll"

# Rebuild both projects
.\build.ps1 -Clean

# Check architecture matches (x64)
dumpbin /headers ".\NukeNul\bin\Release\net8.0\nuker_core.dll"
```

#### "Access Denied" when deleting files

**Problem**: Files are locked or require elevation

**Solution**:
```powershell
# Run as Administrator
Start-Process pwsh -Verb RunAs -ArgumentList "-File", ".\build.ps1"

# Or check file locks
openfiles /query | Select-String "nul"
```

### Test Issues

#### "Failed to create test files"

**Problem**: Insufficient permissions or disk full

**Solution**:
```powershell
# Check available space
Get-PSDrive C | Select-Object Used,Free

# Use different test location
$env:TEMP = "D:\Temp"
.\test.ps1
```

#### "Performance benchmark slower than expected"

**Problem**: Small test size, disk caching, or system load

**Solution**:
```powershell
# Use larger test count
.\test.ps1 -TestCount 1000 -DeepNesting

# Run on cold cache
Clear-RecycleBin -Force
.\test.ps1
```

---

## Performance Tuning

### Rust Build Optimization

#### Enable Link-Time Optimization (LTO)

Edit `nuker_core/Cargo.toml`:

```toml
[profile.release]
lto = "fat"              # Full LTO for maximum optimization
codegen-units = 1        # Single codegen unit for better optimization
opt-level = 3            # Maximum optimization
strip = true             # Strip symbols for smaller binary
panic = "abort"          # Smaller code, faster execution
```

Rebuild:
```powershell
.\build.ps1 -Clean
```

#### CPU-Specific Optimization

```powershell
# Build for current CPU architecture
$env:RUSTFLAGS = "-C target-cpu=native"
.\build.ps1 -SkipCSharp

# Or edit .cargo/config.toml:
# [build]
# rustflags = ["-C", "target-cpu=native"]
```

### C# Build Optimization

#### Native AOT Compilation

Edit `NukeNul/NukeNul.csproj`:

```xml
<PropertyGroup>
  <PublishAot>true</PublishAot>
  <IlcOptimizationPreference>Speed</IlcOptimizationPreference>
  <IlcGenerateStackTraceData>false</IlcGenerateStackTraceData>
</PropertyGroup>
```

Build:
```powershell
.\build.ps1 -Publish -Clean
```

#### ReadyToRun (R2R) Images

```powershell
dotnet publish -c Release -r win-x64 `
  -p:PublishReadyToRun=true `
  -p:PublishSingleFile=true
```

### Parallel Walk Tuning

Edit `nuker_core/src/lib.rs` to tune thread count:

```rust
// Use specific thread count (default is CPU cores)
let walker = WalkBuilder::new(root_path)
    .threads(16)  // Force 16 threads
    .hidden(false)
    .build_parallel();
```

### Disk I/O Optimization

For network drives or slow disks:

```rust
// Reduce parallelism on slow I/O
let walker = WalkBuilder::new(root_path)
    .threads(4)  // Fewer threads for network drives
    .max_filesize(Some(1024 * 1024))  // Skip large files
    .build_parallel();
```

---

## Performance Benchmarks

### Expected Performance Ranges

| Test Size | PowerShell | NukeNul | Speedup |
|-----------|------------|---------|---------|
| 10 files  | 200-500ms  | 20-50ms | 5-10x   |
| 100 files | 2-5s       | 100-200ms | 15-25x |
| 1000 files | 20-60s    | 500ms-2s | 20-40x |
| 10000 files | 5-15min  | 5-10s   | 50-100x |

### Real-World Example

```
Target: C:\Projects (154,020 files scanned)
Results:
  Files Scanned:  154020
  Files Deleted:  12
  Errors:         0
Performance:
  Mode:           Rust/Parallel
  Threads:        16
  Elapsed:        847 ms

Comparison to PowerShell: ~67x faster
```

---

## Additional Resources

### Documentation
- [Rust ignore crate](https://docs.rs/ignore/)
- [Windows Wide Strings](https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file)
- [.NET P/Invoke Guide](https://docs.microsoft.com/en-us/dotnet/standard/native-interop/)

### Related Tools
- [Everything Search](https://www.voidtools.com/) - Fast file indexing
- [ripgrep](https://github.com/BurntSushi/ripgrep) - Fast search tool using same walker engine
- [fd](https://github.com/sharkdp/fd) - Fast alternative to `find`

---

## License

This project is provided as-is for personal and commercial use.

---

## Support

For issues, questions, or contributions, see the project repository.

