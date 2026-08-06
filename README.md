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
- ✅ **Literal `$null` safety mode** - Deletes only real `$null` files, not device aliases
- ✅ **Composable match families** - reserved device names, literal `$null`, and
  path-mangle artifacts (`foo;C`) can be combined in a single pass
- ✅ **`--dry-run`** - Preview exactly what would be deleted (and why something
  would be skipped) without touching disk
- ✅ **Empty-directory cleanup** (`--include-dirs`) - Removes matching directories,
  but only if they are already empty; never recursive
- ✅ **Auditable output** - Every deleted/would-delete/skipped entry lists its full
  path, match family, and (for `$null`) size + a content preview

## ⚠️ Behavior change: `$null` zero-byte safety gate

As of this version, the `$null` family (`--dollar-null-only`) deletes **only
zero-byte** files by default. This matches the governing workspace policy:
*"Discovery hooks may delete only zero-byte matches inside the active
workspace; preserve and report non-empty or out-of-scope matches."*

Non-empty `$null` files are now **skipped** (not deleted) and reported in the
JSON output with their size and the first 200 bytes of their content, so an
operator can review them before deciding what to do. Pass `--allow-nonempty`
to opt into deleting `$null` files larger than zero bytes, restoring the
previous unconditional-delete behavior for that run.

Previously (pre-`--dry-run`/`--allow-nonempty`), `--dollar-null-only` deleted
`$null` files of any size unconditionally.

## Installation

### Option 1: Download Pre-built Binary

1. Download `NukeNul.exe` and `nuker_core.dll` from releases
2. Place both files in the same directory
3. Run from command line or PowerShell

### Option 2: Build from Source

See [BUILD.md](BUILD.md) for detailed build instructions.

```bash
# Quick build (repo root)
pwsh -File .\build.ps1
```

## Usage

### Basic Usage

```bash
# Scan current directory (default: reserved device names only)
NukeNul.exe

# Scan specific directory
NukeNul.exe C:\Path\To\Scan

# Scan with full path
NukeNul.exe "C:\Users\david\Documents"

# Delete only visible files named literal $null (zero-byte only by default)
NukeNul.exe --dollar-null-only "C:\Path\To\Scan"

# Preview a full sweep of every family, including empty directories
NukeNul.exe --dry-run --all --include-dirs "C:\Path\To\Scan"

# Combine families explicitly
NukeNul.exe --dollar-null-only --path-mangle "C:\Path\To\Scan"

# Delete non-empty $null files too (opt-in override)
NukeNul.exe --dollar-null-only --allow-nonempty "C:\Path\To\Scan"
```

### CLI Reference

Match families (composable - pass any combination; bare invocation defaults
to `--reserved`):

| Flag | Family | Notes |
|------|--------|-------|
| `--reserved` | Reserved device names | `nul`, `con`, `prn`, `aux`, `com1-9`, `lpt1-9`. Active by default when no other family flag is given. |
| `--dollar-null-only` | Literal `$null` | Zero-byte files deleted by default; see `--allow-nonempty`. Kept exclusive when passed alone, for backward compatibility. |
| `--path-mangle` | Shell path-mangle artifacts | Leaf names ending in `;<letter>` or `;<letter>:` (e.g. `foo;C`, `foo;C:`). Deliberately narrow - `;` is a legal filename character. |
| `--all` | All three families | Shorthand for `--reserved --dollar-null-only --path-mangle`. |

Modifiers:

| Flag | Effect |
|------|--------|
| `--include-dirs` | Also remove matching directories, but **only if they are empty**. Never recursive - `RemoveDirectoryW` itself refuses non-empty directories, and that refusal is the safety mechanism. |
| `--dry-run`, `-n` | Preview only. Walks and matches identically to a real run (including predicting whether a directory would be empty), but performs no deletion. Exit code 0. |
| `--allow-nonempty` | Allow deleting `$null` files larger than zero bytes (dollar-null family only). |

### Example Output

```json
{
  "tool": "Nuke-Nul",
  "target": "C:\\Users\\david\\Documents",
  "operation": "ReservedDeviceNames+LiteralDollarNull",
  "timestamp": "2026-01-23T19:30:45.1234567Z",
  "status": "Success",
  "dry_run": false,
  "performance": {
    "mode": "Rust/Parallel",
    "threads": 16,
    "elapsed_ms": 1234
  },
  "results": {
    "scanned": 154020,
    "deleted": 12,
    "would_delete": 0,
    "skipped": 1,
    "errors": 0,
    "deleted_entries": [
      {"path": "C:\\Users\\david\\Documents\\nul", "family": "reserved", "kind": "file"}
    ],
    "would_delete_entries": [],
    "skipped_entries": [
      {
        "path": "C:\\Users\\david\\Documents\\$null",
        "family": "dollar_null",
        "kind": "file",
        "reason": "non-empty $null file (877 bytes); rerun with --allow-nonempty to delete",
        "size": 877,
        "content_preview": "rg: no matches found"
      }
    ]
  }
}
```

Every `deleted_entries` / `would_delete_entries` / `skipped_entries` item
carries `path`, `family` (`reserved` | `dollar_null` | `path_mangle`), and
`kind` (`file` | `dir`). `reason`, `size`, and `content_preview` are present
only where relevant (skips always have a `reason`; `$null` files carry `size`,
and non-empty ones also carry `content_preview`).

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

The primary entry point is `nuke_files_ex`, which returns a heap-allocated
JSON string (the caller must free it with `nuke_free_string`):

```rust
#[repr(C)]
pub struct NukeOptions {
    pub match_reserved: u8,
    pub match_dollar_null: u8,
    pub match_path_mangle: u8,
    pub include_dirs: u8,
    pub dry_run: u8,
    pub allow_nonempty: u8,
}

#[no_mangle]
pub unsafe extern "C" fn nuke_files_ex(root_ptr: *const c_char, options: NukeOptions) -> *mut c_char;

#[no_mangle]
pub unsafe extern "C" fn nuke_free_string(ptr: *mut c_char);
```

The original single-mode functions remain exported with their original
`ScanStats`-returning signature for ABI compatibility with any other callers
of `nuker_core.dll` (NukeNul.exe itself no longer calls them):

```rust
#[repr(C)]
pub struct ScanStats {
    pub files_scanned: u32,
    pub files_deleted: u32,
    pub errors: u32,
}

#[no_mangle]
pub unsafe extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats;
#[no_mangle]
pub unsafe extern "C" fn nuke_dollar_null_files(root_ptr: *const c_char) -> ScanStats;
```

### C# P/Invoke

```csharp
[StructLayout(LayoutKind.Sequential)]
internal struct NukeOptions
{
    public byte MatchReserved;
    public byte MatchDollarNull;
    public byte MatchPathMangle;
    public byte IncludeDirs;
    public byte DryRun;
    public byte AllowNonempty;
}

[DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern IntPtr nuke_files_ex(
    [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
    NukeOptions options);

[DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern void nuke_free_string(IntPtr ptr);
```

## Limitations

1. **Windows Only** - Uses Win32 API (Linux/macOS support requires alternative implementation)
2. **No Undo** - Deleted files are permanently removed outside dry-run mode (use with caution)
3. **Admin Rights** - Some system directories may require elevation
4. **`--include-dirs` never recurses** - a matching directory is only removed if it is already empty; NukeNul will never empty out or recursively delete a directory tree

## Safety Considerations

⚠️ **WARNING**: This tool permanently deletes files. Always test on non-critical data first.

- Verify target path before execution
- Check JSON output for errors
- Review `.git` exclusion behavior if scanning repositories
- Consider backing up important data

## Future Enhancements

- [x] Dry-run mode (scan without deletion) - `--dry-run`/`-n`
- [x] Empty-directory cleanup - `--include-dirs`
- [x] Composable match families + auditable per-path JSON output
- [ ] Configuration file for reserved name patterns
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
