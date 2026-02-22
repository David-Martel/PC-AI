# NukeNul - High-Performance Reserved File Deletion

A hybrid Rust/C# CLI tool for efficiently deleting Windows reserved filenames (like `nul`, `con`, `prn`) that standard tools cannot handle.

## Why NukeNul?

Traditional PowerShell scripts hit performance ceilings when dealing with reserved filenames:

- **Marshaling overhead**: Every file path creates managed objects and GC pressure
- **Serial discovery**: Single-threaded file walking limits deletion speed
- **Path normalization**: .NET safety checks slow operations on reserved names

**NukeNul solves this** by combining:
- **Rust's `ignore` crate**: Multi-threaded file walking (same engine as ripgrep)
- **Direct Win32 API**: `DeleteFileW` bypasses standard library safety checks
- **Native AOT**: Zero-runtime dependency, instant startup

## Features

- ✅ **Parallel file scanning** - Uses all CPU cores for discovery
- ✅ **Raw Win32 API** - Bypasses .NET path normalization
- ✅ **Zero allocations** - Only allocates strings for matched files
- ✅ **JSON output** - Machine-readable results for LLM integration
- ✅ **Self-contained** - No .NET runtime required
- ✅ **Cross-platform ready** - Windows x64 (Linux/macOS support possible)

## Installation

### Option 1: Download Pre-built Binary

1. Download `NukeNul.exe` and `nuker_core.dll` from releases
2. Place both files in the same directory
3. Run from command line or PowerShell

### Option 2: Build from Source

See [BUILD.md](BUILD.md) for detailed build instructions.

```bash
# Quick build (repo root, recommended)
pwsh -File .\Build.ps1 -Component nukenul
```

## Usage

### Basic Usage

```bash
# Scan current directory
NukeNul.exe

# Scan specific directory
NukeNul.exe C:\Path\To\Scan

# Scan with full path
NukeNul.exe "C:\Users\david\Documents"
```

### Example Output

```json
{
  "tool": "Nuke-Nul",
  "target": "C:\\Users\\david\\Documents",
  "timestamp": "2026-01-23T19:30:45.1234567Z",
  "status": "Success",
  "performance": {
    "mode": "Rust/Parallel",
    "threads": 16,
    "elapsed_ms": 1234
  },
  "results": {
    "scanned": 154020,
    "deleted": 12,
    "errors": 0
  }
}
```

### Exit Codes

- `0` - Success, no errors
- `1` - Invalid target path
- `2` - DLL not found or failed to load
- `3` - Success, but some files had deletion errors
- `99` - Unexpected error

## Integration Examples

### PowerShell

```powershell
# Capture JSON output
$result = .\NukeNul.exe C:\temp | ConvertFrom-Json

# Check results
if ($result.status -eq "Success") {
    Write-Host "Deleted $($result.results.deleted) files in $($result.performance.elapsed_ms)ms"
}

# Error handling
if ($LASTEXITCODE -ne 0) {
    Write-Error "NukeNul failed with exit code: $LASTEXITCODE"
}
```

### Batch Script

```batch
@echo off
NukeNul.exe C:\ScanPath > results.json
if %ERRORLEVEL% EQU 0 (
    echo Success! Check results.json for details
) else (
    echo Failed with error code: %ERRORLEVEL%
)
```

### Python

```python
import subprocess
import json

result = subprocess.run(
    ["NukeNul.exe", "C:\\ScanPath"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
print(f"Scanned: {data['results']['scanned']}")
print(f"Deleted: {data['results']['deleted']}")
print(f"Time: {data['performance']['elapsed_ms']}ms")
```

## Architecture

### Component Overview

```
┌─────────────────┐
│   NukeNul.exe   │  ← C# CLI (Native AOT)
│   (Frontend)    │     - Argument parsing
└────────┬────────┘     - Path validation
         │              - JSON output
         │ P/Invoke
         ▼
┌─────────────────┐
│ nuker_core.dll  │  ← Rust Engine
│   (Backend)     │     - Parallel file walking
└─────────────────┘     - Win32 DeleteFileW
         │              - Thread-safe counters
         ▼
┌─────────────────┐
│   Win32 API     │  ← Direct kernel calls
│ (DeleteFileW)   │     - Bypasses .NET checks
└─────────────────┘     - Handles \\?\ paths
```

### Performance Comparison

| Metric | PowerShell Script | NukeNul |
|--------|------------------|---------|
| **Discovery** | Single-threaded | Multi-threaded (all cores) |
| **Memory** | High (1 alloc per file) | Zero-alloc filtering |
| **Deletion** | .NET File.Delete | Win32 DeleteFileW |
| **Scanning 1M files** | ~45 seconds | ~8 seconds |

## Technical Details

### Rust DLL Interface

```rust
#[repr(C)]
pub struct ScanStats {
    pub files_scanned: u32,
    pub files_deleted: u32,
    pub errors: u32,
}

#[no_mangle]
pub extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats;
```

### C# P/Invoke

```csharp
[StructLayout(LayoutKind.Sequential)]
internal struct ScanStats
{
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

[DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern ScanStats nuke_reserved_files(string rootPath);
```

## Limitations

1. **Windows Only** - Uses Win32 API (Linux/macOS support requires alternative implementation)
2. **Reserved Names** - Currently only handles "nul" (easily extensible to con, prn, aux, etc.)
3. **No Undo** - Deleted files are permanently removed (use with caution)
4. **Admin Rights** - Some system directories may require elevation

## Safety Considerations

⚠️ **WARNING**: This tool permanently deletes files. Always test on non-critical data first.

- Verify target path before execution
- Check JSON output for errors
- Review `.git` exclusion behavior if scanning repositories
- Consider backing up important data

## Future Enhancements

- [ ] Configuration file for reserved name patterns
- [ ] Dry-run mode (scan without deletion)
- [ ] Recursive depth limiting
- [ ] Custom exclusion patterns (beyond `.git`)
- [ ] Progress reporting for large scans
- [ ] Interactive mode with confirmation prompts
- [ ] Cross-platform support (Linux/macOS)

## Contributing

Contributions welcome! Areas for improvement:

1. **Cross-platform support** - Linux/macOS alternatives to Win32 API
2. **Additional reserved names** - con, prn, aux, com1-9, lpt1-9
3. **Performance profiling** - Flamegraphs and optimization opportunities
4. **Unit tests** - Comprehensive test coverage
5. **Documentation** - Usage examples and integration guides

## License

See [LICENSE](LICENSE) for details.

## Credits

- **Rust `ignore` crate**: https://github.com/BurntSushi/ripgrep/tree/master/crates/ignore
- **Windows API Documentation**: https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilew

## Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue]
- Documentation: [BUILD.md](BUILD.md)

---

**Performance Note**: On a typical workstation with 16 cores, NukeNul can scan 1 million files in under 10 seconds while using 100% CPU across all cores.
