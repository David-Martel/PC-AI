# NukeNul Quick Reference Card

One-page reference for common operations.

---

## Build Commands

```powershell
# Standard build
.\build.ps1

# Clean build
.\build.ps1 -Clean

# Self-contained executable
.\build.ps1 -Publish

# Debug build
.\build.ps1 -Configuration Debug

# Rebuild Rust only
.\build.ps1 -SkipCSharp

# Rebuild C# only
.\build.ps1 -SkipRust
```

---

## Test Commands

```powershell
# Standard test (10 files)
.\test.ps1

# Stress test (1000 files)
.\test.ps1 -TestCount 1000

# Deep nesting test
.\test.ps1 -DeepNesting

# Skip benchmark
.\test.ps1 -SkipBenchmark

# Keep test directory
.\test.ps1 -KeepTestDir

# Debug test
.\test.ps1 -Configuration Debug
```

---

## Usage Commands

```powershell
# Using wrapper (auto-detects best version)
.\delete-nul-files-v2.ps1

# Scan specific directory
.\delete-nul-files-v2.ps1 -SearchPath "C:\Projects"

# Force PowerShell version
.\delete-nul-files-v2.ps1 -UseOriginal

# Direct execution
.\NukeNul\bin\Release\net8.0\NukeNul.exe .

# Scan with full path
.\NukeNul\bin\Release\net8.0\NukeNul.exe "C:\Users\david\Projects"
```

---

## File Locations

| Component | Location |
|-----------|----------|
| Rust source | `nuker_core/src/lib.rs` |
| Rust config | `nuker_core/Cargo.toml` |
| C# source | `NukeNul/Program.cs` |
| C# project | `NukeNul/NukeNul.csproj` |
| Build script | `build.ps1` |
| Test script | `test.ps1` |
| Wrapper script | `delete-nul-files-v2.ps1` |
| Rust DLL (release) | `nuker_core/target/release/nuker_core.dll` |
| C# EXE (release) | `NukeNul/bin/Release/net8.0/NukeNul.exe` |
| Published EXE | `NukeNul/bin/Release/net8.0/win-x64/publish/NukeNul.exe` |

---

## Manual Test File Creation

```powershell
# Create test directory
$TestDir = "C:\Temp\NulTest"
New-Item -ItemType Directory -Path $TestDir -Force

# Create "nul" file (use .NET API)
$NulPath = Join-Path $TestDir "nul"
$ExtendedPath = "\\?\$NulPath"
$FileStream = [System.IO.File]::Create($ExtendedPath)
$FileStream.Close()

# Verify exists
[System.IO.File]::Exists($ExtendedPath)

# Delete with NukeNul
.\NukeNul\bin\Release\net8.0\NukeNul.exe $TestDir

# Verify deleted
[System.IO.File]::Exists($ExtendedPath)
```

---

## Deployment Options

### Option 1: Copy to PATH

```powershell
# Build self-contained
.\build.ps1 -Publish

# Copy to Windows system directory
Copy-Item ".\NukeNul\bin\Release\net8.0\win-x64\publish\NukeNul.exe" `
  "C:\Windows\System32\NukeNul.exe"

# Or user binaries
Copy-Item ".\NukeNul\bin\Release\net8.0\win-x64\publish\NukeNul.exe" `
  "$env:LOCALAPPDATA\Microsoft\WindowsApps\NukeNul.exe"
```

### Option 2: Add to PATH

```powershell
# Build standard
.\build.ps1

# Add to PATH
$ToolDir = "C:\Tools\NukeNul"
New-Item -ItemType Directory -Path $ToolDir -Force
Copy-Item ".\NukeNul\bin\Release\net8.0\NukeNul.exe" $ToolDir
Copy-Item ".\NukeNul\bin\Release\net8.0\nuker_core.dll" $ToolDir

# Add to user PATH
$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
[Environment]::SetEnvironmentVariable("PATH", "$CurrentPath;$ToolDir", "User")
```

### Option 3: PowerShell Alias

```powershell
# Add to PowerShell profile
Add-Content $PROFILE @"
function Remove-NulFiles {
    param([string]`$Path = ".")
    & "C:\Tools\NukeNul\NukeNul.exe" `$Path
}
Set-Alias nuke Remove-NulFiles
"@

# Reload profile
. $PROFILE

# Use anywhere
nuke "C:\Projects"
```

---

## Troubleshooting Quick Fixes

### Build Issues

```powershell
# Rust not found
winget install Rustlang.Rustup

# .NET not found
winget install Microsoft.DotNet.SDK.8

# DLL not found
.\build.ps1 -Clean

# Permission denied
Start-Process pwsh -Verb RunAs -ArgumentList "-File", ".\build.ps1"
```

### Runtime Issues

```powershell
# DLL load failed
Test-Path ".\NukeNul\bin\Release\net8.0\nuker_core.dll"
.\build.ps1 -Clean

# Access denied
# Run as Administrator or check file locks
openfiles /query | Select-String "nul"
```

---

## Performance Tips

### For Large Directories (>100k files)

```powershell
# Use release build (optimized)
.\build.ps1 -Configuration Release

# Monitor system resources
Get-Process NukeNul | Format-Table CPU,WS -AutoSize
```

### For Network Drives

```powershell
# Reduce thread count in src/lib.rs
# Change: .threads(16) -> .threads(4)
.\build.ps1 -Clean
```

### For SSDs

```powershell
# Use maximum optimization
# Add to Cargo.toml:
# [profile.release]
# lto = "fat"
# codegen-units = 1
.\build.ps1 -Clean
```

---

## Common JSON Output Examples

### Success (files found and deleted)

```json
{
  "Tool": "Nuke-Nul",
  "Target": "C:\\Projects",
  "Status": "Success",
  "Performance": {
    "Mode": "Rust/Parallel",
    "Threads": 16,
    "ElapsedMs": 847
  },
  "Results": {
    "Scanned": 154020,
    "Deleted": 12,
    "Errors": 0
  }
}
```

### Success (no files found)

```json
{
  "Tool": "Nuke-Nul",
  "Target": "C:\\Clean",
  "Status": "Success",
  "Performance": {
    "Mode": "Rust/Parallel",
    "Threads": 16,
    "ElapsedMs": 234
  },
  "Results": {
    "Scanned": 45000,
    "Deleted": 0,
    "Errors": 0
  }
}
```

### Partial errors

```json
{
  "Tool": "Nuke-Nul",
  "Target": "C:\\Locked",
  "Status": "Success",
  "Performance": {
    "Mode": "Rust/Parallel",
    "Threads": 16,
    "ElapsedMs": 567
  },
  "Results": {
    "Scanned": 23000,
    "Deleted": 8,
    "Errors": 2
  }
}
```

---

## Environment Variables

```powershell
# Force thread count
$env:RAYON_NUM_THREADS = "8"
.\NukeNul\bin\Release\net8.0\NukeNul.exe .

# Rust backtrace on errors
$env:RUST_BACKTRACE = "1"
.\NukeNul\bin\Release\net8.0\NukeNul.exe .

# Rust verbose output
$env:RUST_LOG = "debug"
.\NukeNul\bin\Release\net8.0\NukeNul.exe .
```

---

## Useful Aliases

```powershell
# Add to $PROFILE

# Quick build
function nb { .\build.ps1 @args }

# Quick test
function nt { .\test.ps1 @args }

# Quick clean build
function nbc { .\build.ps1 -Clean @args }

# Quick publish
function nbp { .\build.ps1 -Publish -Clean @args }

# Run NukeNul
function nuke {
    param([string]$Path = ".")
    .\NukeNul\bin\Release\net8.0\NukeNul.exe $Path
}

# Reload profile
. $PROFILE
```

---

## Version Checking

```powershell
# Check Rust version
cargo --version
rustc --version

# Check .NET version
dotnet --version
dotnet --list-sdks

# Check PowerShell version
$PSVersionTable.PSVersion

# Check Windows version
[System.Environment]::OSVersion.Version
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/build-and-test.yml
- name: Build and Test
  shell: pwsh
  run: |
    .\build.ps1 -Clean
    .\test.ps1 -TestCount 100
```

### Azure DevOps

```yaml
# azure-pipelines.yml
- task: PowerShell@2
  inputs:
    filePath: 'build.ps1'
    arguments: '-Clean -Publish'
```

### Jenkins

```groovy
// Jenkinsfile
stage('Build') {
    steps {
        pwsh './build.ps1 -Clean'
    }
}
stage('Test') {
    steps {
        pwsh './test.ps1 -TestCount 100'
    }
}
```

---

## Support and Resources

- **Documentation**: `BUILD_AND_TEST.md`
- **Architecture**: `NukeNul.md`
- **Rust Docs**: https://docs.rs/ignore/
- **Windows APIs**: https://docs.microsoft.com/en-us/windows/win32/fileio/

---

## License

MIT License - See LICENSE file for details.
