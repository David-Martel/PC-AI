# Platform-Specific Considerations for nuker_core

## Windows Architecture Deep Dive

### Win32 API: DeleteFileW

#### Why DeleteFileW Instead of std::fs::remove_file?

1. **Extended-Length Path Support**
   - Standard Rust APIs respect MAX_PATH (260 characters)
   - `DeleteFileW` with `\\?\` prefix supports up to 32,767 characters
   - Required for deeply nested directory structures

2. **Reserved Name Handling**
   - Rust's `std::fs` uses `DeleteFileA`/`DeleteFileW` internally but normalizes paths first
   - Path normalization treats "nul" as the null device, preventing deletion
   - Direct `DeleteFileW` with `\\?\` bypasses this normalization

3. **Performance**
   - One fewer layer of abstraction
   - No UTF-8 → UTF-16 → UTF-8 round-trip
   - Direct system call reduces latency

#### DeleteFileW Return Values

```rust
// Return value interpretation:
// Non-zero = Success
// Zero = Failure (use GetLastError() for details)

unsafe {
    if DeleteFileW(path_ptr) != 0 {
        // Success
    } else {
        // Call GetLastError() to determine cause:
        // ERROR_FILE_NOT_FOUND (2) - File doesn't exist
        // ERROR_ACCESS_DENIED (5) - Permission denied
        // ERROR_SHARING_VIOLATION (32) - File in use
        // ERROR_WRITE_PROTECT (19) - Write-protected media
    }
}
```

### Extended-Length Path Prefix (`\\?\`)

#### Path Formats

| Standard Path | Extended-Length Path | Notes |
|--------------|---------------------|-------|
| `C:\folder\file` | `\\?\C:\folder\file` | Standard absolute path |
| `\\server\share\file` | `\\?\UNC\server\share\file` | UNC network path |
| `folder\file` | Not supported | Relative paths cannot use `\\?\` |
| `C:\a\b\...\file` (>260) | `\\?\C:\a\b\...\file` | Long path support |

#### Important Limitations

1. **No Path Normalization**
   ```
   Standard: C:\folder\..\file → C:\file (normalized)
   Extended: \\?\C:\folder\..\file → FAILS (not normalized)
   ```

2. **Forward Slashes Not Allowed**
   ```
   Standard: C:/folder/file → Works (converted to backslash)
   Extended: \\?\C:/folder/file → FAILS (must use backslash)
   ```

3. **Case Sensitivity**
   - Windows file systems are case-insensitive but case-preserving
   - `\\?\` paths follow the same rules
   - NTFS extended attributes can enable case sensitivity (rare)

### UTF-16 Encoding

#### Why UTF-16 for Windows?

Windows uses UTF-16LE (Little Endian) for all Unicode APIs (the "W" suffix functions).

**Conversion Requirements**:
```rust
// Rust strings are UTF-8
let path_utf8 = "C:\\folder\\nul";

// Must convert to UTF-16 for Win32
let path_utf16: Vec<u16> = path_utf8.encode_utf16().collect();

// Must be null-terminated for C FFI
let path_utf16_cstr = U16CString::from_str(path_utf8)?;
```

**Performance Impact**:
- UTF-8 → UTF-16 conversion: ~100 ns per path
- Negligible compared to file system I/O (10-1000 µs)

### Thread Safety and Parallelism

#### Work-Stealing Queue (`ignore` crate)

The `ignore` crate uses a sophisticated work-stealing algorithm:

```
Thread 1: [Directory A] → [Subdirs A1, A2, A3] → Process A1
Thread 2: [Idle] → Steals [A2] from Thread 1 → Process A2
Thread 3: [Idle] → Steals [A3] from Thread 1 → Process A3
Thread 4: [Directory B] → [Subdirs B1, B2] → Process B1
```

**Benefits**:
- Automatic load balancing
- No manual work distribution
- Scales to CPU core count
- Minimal contention (lock-free queues)

#### Atomic Operations

```rust
// Ordering::Relaxed is sufficient for counters
// We only need eventual consistency, not strict ordering
scanned.fetch_add(1, Ordering::Relaxed);

// At scan completion, Ordering::Relaxed is also sufficient
// The thread synchronization from walker.run() provides happens-before guarantees
let total = scanned.load(Ordering::Relaxed);
```

**Why Relaxed Ordering is Safe**:
1. Counters are independent (no cross-counter dependencies)
2. Only read once at the end (no mid-scan consistency needed)
3. Thread join provides implicit memory barrier
4. Faster than SeqCst or Acquire/Release (~5-10ns vs ~20-30ns per operation)

### File System Considerations

#### NTFS-Specific Behavior

1. **Alternate Data Streams (ADS)**
   ```
   file.txt        ← Main stream
   file.txt:hidden ← Alternate stream
   ```
   - `DeleteFileW` only deletes the main stream
   - ADS are automatically deleted when the main stream is deleted
   - Reserved names can have ADS: `nul:stream`

2. **Hard Links**
   - Multiple directory entries can point to the same file data
   - `DeleteFileW` removes one link; file data persists until all links removed
   - Link count visible via `GetFileInformationByHandle`

3. **Reparse Points**
   - Symbolic links, mount points, junctions
   - `DeleteFileW` removes the reparse point, not the target
   - Important for avoiding accidental data loss

#### FAT32 Limitations

If scanning FAT32 volumes:
- No extended attributes
- 8.3 filename restrictions
- Reserved names still apply
- No alternate data streams
- Case-insensitive, not case-preserving (filenames uppercase)

#### ReFS Considerations

Windows Resilient File System (ReFS):
- Supports extended-length paths
- No 8.3 short names
- Block cloning (copy-on-write)
- `DeleteFileW` works identically to NTFS

### Windows Security Model

#### Access Control Lists (ACLs)

Required permissions for deletion:
```
File: DELETE permission (or WRITE_DAC to grant yourself DELETE)
Directory: FILE_DELETE_CHILD permission (allows deleting children)
```

**Privilege Escalation**:
```rust
// To delete files in protected directories:
// 1. Run as Administrator
// 2. Enable SeBackupPrivilege and SeRestorePrivilege
// 3. Use FILE_FLAG_BACKUP_SEMANTICS with CreateFile
```

#### User Account Control (UAC)

- Standard users: Can delete their own files
- Protected directories: `C:\Windows`, `C:\Program Files` require elevation
- Virtualization: UAC may redirect writes to `VirtualStore`

### Performance Optimization Strategies

#### 1. I/O Completion Ports (IOCP)

Not currently used, but could improve performance:
```rust
// Async file deletion using IOCP
// Allows overlapped I/O operations
// Useful for network drives or slow storage
```

#### 2. Directory Entry Caching

The `ignore` crate already implements:
- Bulk directory reads (`FindFirstFileW`/`FindNextFileW`)
- Pre-fetching directory entries
- Minimizing syscalls

#### 3. NTFS MFT Optimization

Master File Table (MFT) considerations:
- Sequential scans are fastest (MFT is B-tree ordered)
- Random access thrashes the MFT cache
- Parallel scanning can cause MFT contention (mitigated by work-stealing)

### Edge Cases and Gotchas

#### 1. Reserved Names with Extensions

```
nul.txt    ← Still treated as "nul" device
con.log    ← Still treated as "con" device
prn.doc    ← Still treated as "prn" device
```

Windows ignores everything after the reserved name. Our library checks the full filename, so these **will not** be detected/deleted unless you modify `RESERVED_NAMES` to include extension variants.

#### 2. Reserved Names in Directories

```
C:\nul\file.txt    ← "nul" is a directory name (allowed!)
C:\folder\nul      ← "nul" is a filename (problematic)
```

Directory names **can** be reserved names (Windows allows this). Only files with reserved names cause issues.

#### 3. Case Sensitivity Edge Cases

```
NUL    ← Reserved
nul    ← Reserved
Nul    ← Reserved
nUL    ← Reserved
```

All case variations are equivalent. Our `eq_ignore_ascii_case` handles this correctly.

#### 4. Network Paths

```
\\server\share\nul    → \\?\UNC\server\share\nul
```

UNC paths require special handling:
- Remove leading `\\`
- Add `UNC\` after `\\?\`
- Results in: `\\?\UNC\server\share\nul`

#### 5. Relative Paths

```rust
// Extended-length paths MUST be absolute
".\nul"         → ERROR (relative)
"C:\current\nul" → OK (absolute)

// Our library rejects relative paths in the conversion:
let extended_path = if path_str.starts_with("\\\\?\\") {
    path_str.to_string()  // Already extended
} else {
    format!("\\\\?\\{}", path_str)  // path_str must be absolute
}
```

#### 6. Trailing Backslashes

```
C:\folder\    → Directory
C:\folder     → Directory or file

// DeleteFileW behavior:
// - Fails on directories (use RemoveDirectoryW instead)
// - Trailing backslash always indicates directory
```

#### 7. Volume Mount Points

```
C:\MountedVolume\
```

The `ignore` crate follows mount points by default. This is **intentional** for our use case (we want to scan all accessible files). To exclude:

```rust
WalkBuilder::new(root)
    .filter_entry(|e| {
        // Check if entry is a reparse point (mount point/symlink)
        !is_reparse_point(e)
    })
```

### Memory Usage Patterns

#### Stack Usage
- Path buffers: ~32 KB per thread (MAX_PATH_WIDE * 2)
- Closure captures: ~1 KB per thread
- Thread stacks: 1-2 MB per thread (OS default)

**Total for 16 threads**: ~16-32 MB

#### Heap Usage
- `ignore` crate: ~5-10 MB for directory queue
- Wide string allocations: Only on matched files (~100 bytes per match)
- Atomic counters: 12 bytes total (shared across threads)

**Total**: ~10-50 MB depending on directory depth

### Compiler and Linker Optimizations

#### 1. Link-Time Optimization (LTO)

```toml
[profile.release]
lto = "fat"  # Enables cross-crate inlining
```

**Benefits**:
- Inlines `ignore` crate hot paths into our code
- Eliminates redundant bounds checks
- ~10-15% performance improvement

**Trade-offs**:
- Compile time: 30s → 2-3 minutes
- Required for maximum performance

#### 2. Code Generation Units

```toml
[profile.release]
codegen-units = 1  # Single compilation unit
```

**Benefits**:
- Better inter-procedural optimization
- Smaller binary (less code duplication)

**Trade-offs**:
- Cannot parallelize code generation
- Longer compile times

#### 3. Target CPU Features

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

**Enables**:
- AVX2 instructions (faster string operations)
- BMI2 (bit manipulation)
- SSE4.2 (faster comparisons)

**Trade-offs**:
- Binary not portable to older CPUs
- ~5-10% performance improvement on modern CPUs

### Debugging and Diagnostics

#### 1. Windows Debuggers

**WinDbg**:
```
!analyze -v          ← Automatic crash analysis
bp nuker_core!nuke_reserved_files  ← Set breakpoint
g                    ← Go
k                    ← Stack trace
```

**Visual Studio Debugger**:
- Attach to process
- Set breakpoint in `nuke_reserved_files`
- Step through with F10/F11

#### 2. Error Code Mapping

```rust
use windows_sys::Win32::Foundation::GetLastError;

unsafe {
    if DeleteFileW(path) == 0 {
        let error = GetLastError();
        match error {
            2 => "File not found",
            5 => "Access denied",
            32 => "File in use",
            // ... etc
        }
    }
}
```

#### 3. Performance Profiling

**Windows Performance Analyzer**:
```powershell
# Capture trace
wpr -start CPU -filemode

# Run your program
.\your_program.exe

# Stop and save trace
wpr -stop trace.etl

# Analyze in WPA
wpa trace.etl
```

### Security Considerations

#### 1. DLL Hijacking Prevention

Ensure DLL is loaded from trusted location:
```csharp
// C# - Use full path
[DllImport("C:\\TrustedLocation\\nuker_core.dll")]

// Or verify signature
var cert = X509Certificate.CreateFromSignedFile("nuker_core.dll");
```

#### 2. Path Injection Attacks

Our library is vulnerable to path injection if caller doesn't validate input:
```rust
// Attacker-controlled input:
let malicious_path = "C:\\Important\\Data";

// Caller must validate before calling:
nuke_reserved_files(malicious_path);  // Will delete files in Important\Data!
```

**Mitigation**: Caller must validate and sanitize paths.

#### 3. Race Conditions (TOCTOU)

Time-of-check to time-of-use race:
```
Thread 1: Checks if "nul" exists → Yes
[Context switch]
Thread 2: Another process deletes "nul"
[Context switch]
Thread 1: Attempts to delete "nul" → ERROR
```

**Impact**: Error counter increments, but operation is safe (no data loss).

### Future Improvements

#### 1. Async I/O
```rust
// Use tokio or async-std for async file operations
// Allows scanning while previous deletes are in flight
```

#### 2. Progress Callbacks
```rust
pub type ProgressCallback = extern "C" fn(u32, u32, u32);

pub extern "C" fn nuke_reserved_files_with_progress(
    root_ptr: *const c_char,
    callback: ProgressCallback,
) -> ScanStats;
```

#### 3. Configurable Reserved Names
```rust
pub extern "C" fn nuke_custom_names(
    root_ptr: *const c_char,
    names_ptr: *const *const c_char,
    names_count: usize,
) -> ScanStats;
```

#### 4. Dry-Run Mode
```rust
pub extern "C" fn scan_only(
    root_ptr: *const c_char,
) -> ScanStats;  // Returns count without deleting
```
