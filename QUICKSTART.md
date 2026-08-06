# NukeNul Quick Start Guide

## 30-Second Setup

```bash
# 1. Build everything
.\build.ps1

# 2. Run on current directory
.\bin\Release\net8.0\win-x64\publish\NukeNul.exe .
```

## Common Commands

### Building

```powershell
# Full build (Rust + C#)
.\build.ps1

# C# only (if Rust DLL already built)
.\build.ps1 -SkipRust

# Debug build
.\build.ps1 -Profile Debug

# Build with verification tests
.\build.ps1 -Verify
```

### Running

```powershell
# Scan current directory
.\NukeNul.exe

# Scan specific path
.\NukeNul.exe C:\Path\To\Scan

# Scan and save JSON output
.\NukeNul.exe C:\temp > results.json

# Use published executable
.\bin\Release\net8.0\win-x64\publish\NukeNul.exe C:\LargeDirectory
```

### Parsing Results (PowerShell)

```powershell
# Capture and parse JSON
$result = .\NukeNul.exe . | ConvertFrom-Json

# Display summary
Write-Host "Status: $($result.status)"
Write-Host "Scanned: $($result.results.scanned) files"
Write-Host "Deleted: $($result.results.deleted) files"
Write-Host "Time: $($result.performance.elapsed_ms)ms"

# Check for errors
if ($result.results.errors -gt 0) {
    Write-Warning "Some files could not be deleted: $($result.results.errors)"
}

# Calculate scan rate
$rate = [Math]::Round($result.results.scanned / ($result.performance.elapsed_ms / 1000), 0)
Write-Host "Scan rate: $rate files/second"
```

## Directory Structure

After building:

```
nuke_nul/
├── NukeNul.exe              ← Compiled executable (if built)
├── nuker_core.dll           ← Rust DLL (required at runtime)
├── Program.cs               ← C# source code
├── NukeNul.csproj          ← Project configuration
├── build.ps1               ← Build script
├── bin/
│   └── Release/
│       └── net8.0/
│           └── win-x64/
│               └── publish/
│                   ├── NukeNul.exe      ← Native AOT binary
│                   └── nuker_core.dll   ← Runtime dependency
└── nuker_core/             ← Rust project
    ├── Cargo.toml
    ├── src/
    │   └── lib.rs
    └── target/
        └── release/
            └── nuker_core.dll
```

## File Locations Reference

| File | Development Location | Runtime Location | Purpose |
|------|---------------------|------------------|---------|
| `NukeNul.exe` | `bin\Release\net8.0\win-x64\publish\` | Same directory as DLL | Main executable |
| `nuker_core.dll` | `nuker_core\target\release\` | Same directory as EXE | Rust engine |

## Manual Build Steps

If the build script fails, build manually:

```powershell
# 1. Build Rust DLL
cd nuker_core
cargo build --release
cd ..

# 2. Copy DLL to project root
copy nuker_core\target\release\nuker_core.dll .

# 3. Build C# application
dotnet restore
dotnet build -c Release

# 4. Publish Native AOT
dotnet publish -c Release -r win-x64 --self-contained

# 5. Verify output
Test-Path bin\Release\net8.0\win-x64\publish\NukeNul.exe
Test-Path bin\Release\net8.0\win-x64\publish\nuker_core.dll
```

## Troubleshooting Quick Fixes

### DLL Not Found

```powershell
# Check if DLL exists in publish directory
Test-Path bin\Release\net8.0\win-x64\publish\nuker_core.dll

# If missing, copy manually
copy nuker_core.dll bin\Release\net8.0\win-x64\publish\
```

### Build Fails - .NET SDK Issue

```powershell
# Verify .NET 8 SDK
dotnet --list-sdks

# If not found, install from:
# https://dotnet.microsoft.com/download/dotnet/8.0
```

### Build Fails - Rust Issue

```powershell
# Verify Rust installation
rustc --version
cargo --version

# Update Rust
rustup update

# Rebuild Rust DLL
cd nuker_core
cargo clean
cargo build --release
```

### Runtime Error - DLL Architecture Mismatch

```powershell
# Verify DLL is 64-bit
# Should show "x64" in Machine field
dumpbin /headers nuker_core.dll | Select-String "machine"

# Rebuild Rust for x64 (should be default)
cd nuker_core
cargo build --release --target x86_64-pc-windows-msvc
```

## Performance Testing

```powershell
# Measure execution time
Measure-Command {
    .\bin\Release\net8.0\win-x64\publish\NukeNul.exe C:\LargeDirectory
}

# Parse and display performance metrics
$result = .\bin\Release\net8.0\win-x64\publish\NukeNul.exe C:\LargeDirectory | ConvertFrom-Json
$filesPerSecond = [Math]::Round($result.results.scanned / ($result.performance.elapsed_ms / 1000), 0)
Write-Host "Throughput: $filesPerSecond files/second"
Write-Host "CPU cores used: $($result.performance.threads)"
```

## Distribution Checklist

When distributing to other machines:

- [ ] Copy `NukeNul.exe` from publish directory
- [ ] Copy `nuker_core.dll` from publish directory
- [ ] Ensure both files are in the same folder
- [ ] No .NET runtime installation required (native AOT)
- [ ] Test on target machine before production use

## Common Use Cases

### Clean Build Artifacts

```powershell
# Remove all "nul" files from a project directory
.\NukeNul.exe C:\Projects\MyProject > cleanup-results.json

# Verify results
$result = Get-Content cleanup-results.json | ConvertFrom-Json
if ($result.results.deleted -gt 0) {
    Write-Host "Cleaned $($result.results.deleted) reserved files"
}
```

### Automated CI/CD Integration

```powershell
# Pre-build cleanup script
$scanResult = .\NukeNul.exe $env:BUILD_DIRECTORY | ConvertFrom-Json

if ($scanResult.results.deleted -gt 0) {
    Write-Warning "Removed $($scanResult.results.deleted) problematic files"
}

# Continue with build...
```

### Regular Maintenance Task

```powershell
# Schedule with Task Scheduler
$action = New-ScheduledTaskAction -Execute "C:\Tools\NukeNul.exe" -Argument "C:\Projects"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
Register-ScheduledTask -TaskName "CleanReservedFiles" -Action $action -Trigger $trigger
```

## Getting Help

- **Build issues**: See [BUILD.md](BUILD.md)
- **Usage details**: See [README.md](README.md)
- **Architecture**: See [NukeNul.md](NukeNul.md)

## Next Steps

1. ✅ Run `.\build.ps1` to compile
2. ✅ Test with `.\NukeNul.exe .`
3. ✅ Verify JSON output is valid
4. ✅ Distribute EXE + DLL together
5. ✅ Integrate into your workflow

---

**Pro Tip**: Add the publish directory to your PATH for system-wide access:

```powershell
$publishPath = Resolve-Path "bin\Release\net8.0\win-x64\publish"
$env:PATH += ";$publishPath"

# Now use from anywhere
NukeNul.exe C:\AnyDirectory
```
