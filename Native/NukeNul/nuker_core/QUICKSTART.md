# Quick Start Guide

Get up and running with nuker_core in 5 minutes.

## Prerequisites

Install Rust if you haven't already:

```powershell
# Windows
winget install Rustlang.Rustup

# Or download from https://rustup.rs
```

Verify installation:
```powershell
rustc --version
cargo --version
```

## Build the DLL

### Option 1: Using the Build Script (Recommended)

```powershell
cd C:\Users\david\PC_AI\Native\NukeNul\nuker_core

# Simple release build
.\build.ps1

# Build with tests
.\build.ps1 -Test

# Build and copy to parent directory
.\build.ps1 -Copy

# Clean build with all features
.\build.ps1 -Clean -Profile release -Test -Copy
```

### Option 2: Using Cargo Directly

```powershell
cd C:\Users\david\PC_AI\Native\NukeNul\nuker_core

# Release build
cargo build --release

# Output: target\release\nuker_core.dll
```

## Test the DLL

### Quick Test (PowerShell)

```powershell
# Load the DLL and run test function
Add-Type @"
using System;
using System.Runtime.InteropServices;

public class NukerTest {
    [DllImport("target\\release\\nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint nuker_core_test();
}
"@

$result = [NukerTest]::nuker_core_test()
if ($result -eq 0xDEADBEEF) {
    Write-Host "✓ DLL works!" -ForegroundColor Green
} else {
    Write-Host "✗ DLL test failed" -ForegroundColor Red
}
```

### Full Test (Scan a Directory)

Create `test.ps1`:

```powershell
Add-Type @"
using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct ScanStats {
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

public class Nuker {
    [DllImport("target\\release\\nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ScanStats nuke_reserved_files(string rootPath);
}
"@

# Scan current directory (non-destructive if no "nul" files exist)
$stats = [Nuker]::nuke_reserved_files(".")

Write-Host "`nScan Results:" -ForegroundColor Cyan
Write-Host "Files Scanned: $($stats.FilesScanned)" -ForegroundColor White
Write-Host "Files Deleted: $($stats.FilesDeleted)" -ForegroundColor Yellow
Write-Host "Errors:        $($stats.Errors)" -ForegroundColor Red
```

Run it:
```powershell
.\test.ps1
```

## Create a Test File (Advanced)

Create a reserved filename for testing:

```powershell
# This uses PowerShell to create a "nul" file
# Standard commands like "touch nul" won't work!

# Create via .NET
[System.IO.File]::Create("\\?\$PWD\test_nul").Close()

# Verify it exists (will show in directory but can't be accessed normally)
Get-ChildItem | Where-Object { $_.Name -eq "nul" }

# Now run nuker_core to delete it
.\test.ps1
```

**WARNING**: Creating reserved filename files can cause issues. Only do this in a test directory.

## Integration Examples

### C# Console Application

Create `Program.cs`:

```csharp
using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct ScanStats
{
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

class Program
{
    [DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern ScanStats nuke_reserved_files(string rootPath);

    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: program.exe <directory>");
            return;
        }

        Console.WriteLine($"Scanning: {args[0]}");
        var sw = System.Diagnostics.Stopwatch.StartNew();

        ScanStats stats = nuke_reserved_files(args[0]);

        sw.Stop();

        Console.WriteLine($"\nResults:");
        Console.WriteLine($"  Scanned: {stats.FilesScanned:N0} files");
        Console.WriteLine($"  Deleted: {stats.FilesDeleted:N0} files");
        Console.WriteLine($"  Errors:  {stats.Errors:N0}");
        Console.WriteLine($"  Time:    {sw.ElapsedMilliseconds}ms");
    }
}
```

Build and run:
```powershell
# Copy DLL to project directory
Copy-Item target\release\nuker_core.dll .

# Compile C#
csc Program.cs

# Run
.\Program.exe C:\path\to\scan
```

### Python Script

Create `test.py`:

```python
from ctypes import CDLL, c_char_p, Structure, c_uint32
import sys

class ScanStats(Structure):
    _fields_ = [
        ("files_scanned", c_uint32),
        ("files_deleted", c_uint32),
        ("errors", c_uint32),
    ]

# Load DLL
nuker = CDLL("target/release/nuker_core.dll")
nuker.nuke_reserved_files.argtypes = [c_char_p]
nuker.nuke_reserved_files.restype = ScanStats

# Scan directory
path = sys.argv[1].encode('utf-8') if len(sys.argv) > 1 else b"."
print(f"Scanning: {path.decode('utf-8')}")

stats = nuker.nuke_reserved_files(path)

print(f"\nResults:")
print(f"  Scanned: {stats.files_scanned:,} files")
print(f"  Deleted: {stats.files_deleted:,} files")
print(f"  Errors:  {stats.errors:,}")
```

Run:
```powershell
python test.py C:\path\to\scan
```

## Performance Benchmarking

Create a large test directory:

```powershell
# Create test directory with many files
mkdir test_dir
cd test_dir

# Create 10,000 empty files
1..10000 | ForEach-Object {
    New-Item -ItemType File -Name "file_$_.txt" -Force | Out-Null
}

# Create a few "nul" files using extended-length paths
1..5 | ForEach-Object {
    [System.IO.File]::Create("\\?\$PWD\nul_$_").Close()
}

cd ..

# Benchmark the scan
Measure-Command {
    $stats = [Nuker]::nuke_reserved_files("test_dir")
    Write-Host "Scanned: $($stats.FilesScanned), Deleted: $($stats.FilesDeleted)"
}

# Cleanup
Remove-Item test_dir -Recurse -Force
```

Expected results:
- **10,000 files**: ~200ms on NVMe SSD
- **100,000 files**: ~2 seconds
- **1,000,000 files**: ~20 seconds

## Troubleshooting

### "DLL not found"
```powershell
# Ensure DLL is in the same directory as your executable
# Or add to PATH:
$env:PATH += ";$PWD\target\release"
```

### "BadImageFormatException"
```powershell
# Architecture mismatch (x86 vs x64)
# Rebuild DLL for correct architecture:
cargo build --release --target x86_64-pc-windows-msvc  # 64-bit
cargo build --release --target i686-pc-windows-msvc    # 32-bit
```

### "Access Denied" Errors
```powershell
# Run as Administrator for protected directories
Start-Process powershell -Verb RunAs -ArgumentList "-File test.ps1"
```

### Build Errors
```powershell
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release

# Check for missing dependencies
cargo tree
```

## Next Steps

1. **Read the full documentation**: See [README.md](README.md)
2. **Review platform considerations**: See [PLATFORM_CONSIDERATIONS.md](PLATFORM_CONSIDERATIONS.md)
3. **Explore build options**: See [BUILD.md](BUILD.md)
4. **Integrate into your project**: Copy DLL and use FFI interface

## Common Use Cases

### 1. Clean Up After Failed Git Operations

```powershell
# Sometimes Git fails to delete "nul" files on Windows
$stats = [Nuker]::nuke_reserved_files("C:\your\repo")
Write-Host "Cleaned up $($stats.FilesDeleted) reserved files"
```

### 2. Scan External Drives

```powershell
# Scan a USB drive or external HDD
$stats = [Nuker]::nuke_reserved_files("D:\")
```

### 3. Scheduled Cleanup Task

```powershell
# Create scheduled task to clean temp directories
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\scripts\nuke_cleanup.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
Register-ScheduledTask -TaskName "NukeReservedFiles" -Action $action -Trigger $trigger
```

## Safety Reminders

1. **Always test in a safe directory first**
2. **Backup important data before scanning**
3. **Review what will be deleted** (use dry-run if implemented)
4. **Be careful with system directories** (C:\Windows, etc.)
5. **Run as Administrator only when necessary**

## Getting Help

- **Build issues**: See [BUILD.md](BUILD.md) troubleshooting section
- **Runtime errors**: Check [PLATFORM_CONSIDERATIONS.md](PLATFORM_CONSIDERATIONS.md)
- **Performance tuning**: Review optimization sections in documentation

## Summary

You should now have:
- ✅ Built `nuker_core.dll`
- ✅ Tested the DLL loads correctly
- ✅ Run a sample scan
- ✅ Integrated into your preferred language (C#, Python, etc.)

Happy scanning! 🚀

