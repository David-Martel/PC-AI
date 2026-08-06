Analysis: Why delete-nul-files.ps1 hits a performance ceilingWhile your PowerShell script is well-optimized for a script (using Stack iteration and EnumerateFileSystemEntries), it faces three architectural bottlenecks that no amount of PowerShell tuning can fix:The Marshaling Tax: Every single file path found by .NET's EnumerateFiles is allocated as a managed string, wrapped in a PSObject, passed to the pipeline, and then unwrapped. For a disk with 1,000,000 files, that is 1,000,000 allocations and GC operations, even if only 5 files are "nul".Serial Discovery: Your file walker (the while loop) is single-threaded. You cannot delete files faster than this single thread can find them.Path Normalization: PowerShell and standard .NET libraries aggressively "normalize" paths. To delete "nul", you correctly use \\?\, but the underlying .NET framework still performs safety checks that slow down the operation.The Hybrid Solution: Nuke-NulWe will replace the PowerShell logic with a Rust DLL that acts as a "search-and-destroy" engine.Discovery: Uses the ignore crate (the same engine powering ripgrep) to perform a multi-threaded directory walk.Deletion: Uses raw Win32 API calls (DeleteFileW) to bypass standard library safety checks that often choke on "nul" files.Output: Returns a compact C-struct to C#, ensuring minimal data transfer overhead.1. Rust Core (src/lib.rs)Create a new library: cargo new --lib nuker_core.Edit Cargo.toml to add dependencies:Ini, TOML[package]
name = "nuker_core"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
ignore = "0.4"      # High-performance parallel file walker (ripgrep engine)
widestring = "1.0"  # Easy conversion to Windows Wide Strings (UTF-16)
windows-sys = { version = "0.52", features = ["Win32_Storage_FileSystem", "Win32_Foundation"] }
libc = "0.2"
Edit src/lib.rs:Rustuse std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU32, Ordering};
use ignore::WalkBuilder;
use widestring::U16CString;
use windows_sys::Win32::Storage::FileSystem::DeleteFileW;

/// C-Compatible struct to return stats to C#
#[repr(C)]
pub struct ScanStats {
    pub files_scanned: u32,
    pub files_deleted: u32,
    pub errors: u32,
}

/// The exported function callable from C#
#[no_mangle]
pub extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats {
    // 1. Safety check: Null pointer
    if root_ptr.is_null() {
        return ScanStats { files_scanned: 0, files_deleted: 0, errors: 1 };
    }

    // 2. Convert C-String (char*) to Rust &str
    let c_str = unsafe { CStr::from_ptr(root_ptr) };
    let root_path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ScanStats { files_scanned: 0, files_deleted: 0, errors: 1 },
    };

    // Atomic counters for thread-safe stat tracking
    let scanned = AtomicU32::new(0);
    let deleted = AtomicU32::new(0);
    let errs = AtomicU32::new(0);

    // 3. Configure the Walker
    // - threads: Automatically uses CPU core count
    // - hidden: false (we want to scan hidden folders, but ignore .git by default)
    // - git_global: false (don't respect user's global gitignore, just local)
    let walker = WalkBuilder::new(root_path)
        .hidden(false) // Scan hidden files
        .git_ignore(false) // Don't respect .gitignore files
        .filter_entry(|e| e.file_name() != ".git") // Explicitly skip .git folder
        .build_parallel();

    // 4. Run Parallel Walk
    walker.run(|| {
        let scanned = &scanned;
        let deleted = &deleted;
        let errs = &errs;
        
        Box::new(move |result| {
            if let Ok(entry) = result {
                // Cheap check: Is the file named "nul"?
                // We use OsStr comparison to avoid UTF-8 allocation
                if entry.file_name().eq_ignore_ascii_case("nul") {
                    
                    // Construct the "Long Path" manually: \\?\C:\Path\...\nul
                    // This bypasses Win32 path normalization.
                    let raw_path = entry.path().as_os_str().to_string_lossy();
                    let extended_path = format!("\\\\?\\{}", raw_path);

                    // Convert to UTF-16 (Wide String) for Win32 API
                    if let Ok(wpath) = U16CString::from_str(&extended_path) {
                        unsafe {
                            // Call DeleteFileW directly. 
                            // Non-zero return means success.
                            if DeleteFileW(wpath.as_ptr()) != 0 {
                                deleted.fetch_add(1, Ordering::Relaxed);
                            } else {
                                errs.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
                scanned.fetch_add(1, Ordering::Relaxed);
            }
            ignore::WalkState::Continue
        })
    });

    ScanStats {
        files_scanned: scanned.load(Ordering::Relaxed),
        files_deleted: deleted.load(Ordering::Relaxed),
        errors: errs.load(Ordering::Relaxed),
    }
}
2. C# Driver (Program.cs)This CLI tool serves as the user interface. It is lightweight, compiled AOT (native), and outputs JSON for your LLM.C#using System;
using System.Runtime.InteropServices;
using System.Text.Json; // Native JSON support in modern .NET

// Define the struct exactly as it is in Rust
[StructLayout(LayoutKind.Sequential)]
struct ScanStats
{
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

class Program
{
    // Import the Rust DLL
    [DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern ScanStats nuke_reserved_files(string rootPath);

    static void Main(string[] args)
    {
        string target = args.Length > 0 ? args[0] : ".";
        string fullPath = System.IO.Path.GetFullPath(target);

        // Metadata wrapper for LLM context
        var output = new
        {
            Tool = "Nuke-Nul",
            Target = fullPath,
            Timestamp = DateTime.UtcNow,
            Status = "Running",
            Performance = new { Mode = "Rust/Parallel", Threads = Environment.ProcessorCount },
            Results = (object)null
        };

        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            // --- THE CRITICAL CALL ---
            // Blocks here while Rust consumes 100% CPU on all cores
            ScanStats stats = nuke_reserved_files(fullPath);
            sw.Stop();

            // Update output object
            output = new
            {
                output.Tool,
                output.Target,
                output.Timestamp,
                Status = "Success",
                Performance = new { output.Performance.Mode, output.Performance.Threads, ElapsedMs = sw.ElapsedMilliseconds },
                Results = new 
                { 
                    Scanned = stats.FilesScanned, 
                    Deleted = stats.FilesDeleted, 
                    Errors = stats.Errors 
                }
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { Status = "Fatal Error", Message = ex.Message }));
            return;
        }

        // Output strict JSON for the LLM to parse
        Console.WriteLine(JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
    }
}
3. Performance & Architecture ComparisonFeatureOriginal PowerShell ScriptHybrid Rust/C# SolutionDiscoverySingle-threaded Stack<string>.Multi-threaded work-stealing queue (ignore crate).MemoryHigh. Allocates .NET strings & objects for every file.Zero-Allocation filtering. Strings are only allocated if a match is found.The "Nul" FixString manipulation ("\\?\" + $path) inside a loop.Native Wide-String passed directly to Kernel.ExclusionsManual HashSet lookup (O(1) but interpreted overhead).Compiled optimized filter (Sub-millisecond).LLM OutputUnstructured Console Text (Hard to parse).Strict JSON (Token efficient, machine readable).4. How to Build & RunBuild Rust:Bashcd nuker_core
cargo build --release
# Copy target/release/nuker_core.dll to the C# folder
Build C#:Bashdotnet publish -c Release -r win-x64 --self-contained
Run:Bash./NukeNul.exe "C:\Path\To\Scan"
5. LLM Data Context NoteBy outputting JSON, your local LLM (Ollama) receives this:JSON{
  "Tool": "Nuke-Nul",
  "Status": "Success",
  "Results": {
    "Scanned": 154020,
    "Deleted": 12,
    "Errors": 0
  }
}