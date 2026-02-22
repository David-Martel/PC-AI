# NukeNul Build Instructions

## Primary Build Path (Recommended)

From repository root, use the unified orchestrator:

```powershell
.\Build.ps1 -Component nukenul
```

Use this document's remaining sections only for advanced/manual crate-level debugging.

## Prerequisites

1. **.NET 8 SDK** - Download from https://dotnet.microsoft.com/download/dotnet/8.0
2. **Rust toolchain** - Required to build the `nuker_core.dll`
3. **Windows x64** - This project targets Windows 64-bit

## Build Steps

### Step 1: Build the Rust DLL

```bash
# Navigate to the Rust project directory (if separate)
cd nuker_core

# Build in release mode for maximum performance
cargo build --release

# The DLL will be located at: target/release/nuker_core.dll
```

### Step 2: Copy the Rust DLL

```bash
# Copy the DLL to the C# project root
copy target\release\nuker_core.dll ..\NukeNul\nuker_core.dll
```

### Step 3: Build the C# CLI Application

```bash
# Navigate to the C# project directory
cd ..\NukeNul

# Restore dependencies
dotnet restore

# Build in Debug mode (for testing)
dotnet build -c Debug

# Build in Release mode
dotnet build -c Release
```

### Step 4: Publish as Native AOT Binary

```bash
# Publish as a self-contained native AOT executable
dotnet publish -c Release -r win-x64 --self-contained

# The executable will be located at:
# bin\Release\net8.0\win-x64\publish\NukeNul.exe
```

## Alternative: Quick Build Script (PowerShell)

```powershell
# build.ps1
param(
    [switch]$SkipRust
)

if (-not $SkipRust) {
    Write-Host "Building Rust DLL..." -ForegroundColor Cyan
    Push-Location nuker_core
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Rust build failed"
        exit 1
    }
    Pop-Location

    Write-Host "Copying DLL..." -ForegroundColor Cyan
    Copy-Item "nuker_core\target\release\nuker_core.dll" "nuker_core.dll" -Force
}

Write-Host "Publishing C# application..." -ForegroundColor Cyan
dotnet publish -c Release -r win-x64 --self-contained

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable location: bin\Release\net8.0\win-x64\publish\NukeNul.exe" -ForegroundColor Yellow
} else {
    Write-Error "C# build failed"
    exit 1
}
```

## Binary Locations

After building:

- **Debug build**: `bin\Debug\net8.0\NukeNul.exe`
- **Release build**: `bin\Release\net8.0\NukeNul.exe`
- **Published AOT binary**: `bin\Release\net8.0\win-x64\publish\NukeNul.exe`

## DLL Placement Requirements

The `nuker_core.dll` must be in the same directory as `NukeNul.exe`:

```
bin\Release\net8.0\win-x64\publish\
├── NukeNul.exe
└── nuker_core.dll  ← Must be here
```

The `.csproj` file is configured to automatically copy the DLL if it exists in the project root.

## Optimization Profiles

### Standard Release Build
- **Optimization**: Full
- **Size**: ~5-8 MB
- **Startup**: Fast
- **Use case**: General purpose

```bash
dotnet publish -c Release -r win-x64
```

### Size-Optimized Build
- **Optimization**: Size
- **Size**: ~3-5 MB
- **Startup**: Slightly slower
- **Use case**: Distribution, embedded systems

```bash
dotnet publish -c Release -r win-x64 /p:IlcOptimizationPreference=Size
```

### Speed-Optimized Build
- **Optimization**: Maximum speed
- **Size**: ~8-12 MB
- **Startup**: Fastest
- **Use case**: Performance-critical scenarios

```bash
dotnet publish -c Release -r win-x64 /p:IlcOptimizationPreference=Speed
```

## Verification

After building, verify the executable:

```bash
# Check file size
Get-Item bin\Release\net8.0\win-x64\publish\NukeNul.exe | Select-Object Length

# Test execution
.\bin\Release\net8.0\win-x64\publish\NukeNul.exe --help

# Test with current directory
.\bin\Release\net8.0\win-x64\publish\NukeNul.exe .
```

## Troubleshooting

### DLL Not Found Error

**Symptom**: `DllNotFoundException: Unable to load DLL 'nuker_core.dll'`

**Solutions**:
1. Verify DLL exists: `Test-Path nuker_core.dll`
2. Copy manually: `Copy-Item nuker_core.dll bin\Release\net8.0\win-x64\publish\`
3. Check architecture: Ensure DLL is 64-bit (`dumpbin /headers nuker_core.dll`)

### Native AOT Build Errors

**Symptom**: `ILC: error : ... is not compatible with native AOT`

**Solutions**:
1. Ensure .NET 8 SDK is installed: `dotnet --version`
2. Remove incompatible packages
3. Check `PublishAot` is set to `true` in `.csproj`

### Rust Build Failures

**Symptom**: `cargo build` fails with linking errors

**Solutions**:
1. Update Rust: `rustup update`
2. Install MSVC Build Tools: https://visualstudio.microsoft.com/downloads/
3. Verify toolchain: `rustc --version`

## Performance Benchmarking

```bash
# Scan a large directory and measure performance
Measure-Command { .\NukeNul.exe C:\LargeDirectory }

# Compare with PowerShell script
Measure-Command { .\delete-nul-files.ps1 -TargetPath C:\LargeDirectory }
```

## Distribution

To distribute the application:

1. Copy both files from `bin\Release\net8.0\win-x64\publish\`:
   - `NukeNul.exe`
   - `nuker_core.dll`

2. Both files must remain in the same directory

3. No .NET runtime installation required (native AOT = self-contained)

## Next Steps

- Run the application: See [README.md](README.md)
- Performance tuning: Adjust Rust thread count in `nuker_core`
- Integration: Parse JSON output for automation workflows
