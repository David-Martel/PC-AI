using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NukeNul;

/// <summary>
/// JSON source generation context for AOT compatibility
/// </summary>
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(ScanResult))]
[JsonSerializable(typeof(ErrorResult))]
[JsonSerializable(typeof(EngineOutput))]
internal partial class SourceGenerationContext : JsonSerializerContext
{
}

/// <summary>
/// C-compatible struct matching Rust's ScanStats layout (legacy, retained for reference).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct ScanStats
{
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
}

/// <summary>
/// C-compatible struct matching Rust's NukeOptions layout. Every field is a byte
/// (0 = false, non-zero = true) to avoid platform BOOL marshaling ambiguity.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NukeOptions
{
    public byte MatchReserved;
    public byte MatchDollarNull;
    public byte MatchPathMangle;
    public byte MatchDollarPrefix;
    public byte IncludeDirs;
    public byte DryRun;
    public byte AllowNonempty;
}

/// <summary>
/// JSON output structure for LLM-friendly machine-readable results
/// </summary>
internal sealed class ScanResult
{
    [JsonPropertyName("tool")]
    public string Tool { get; set; } = "Nuke-Nul";

    [JsonPropertyName("target")]
    public string Target { get; set; } = string.Empty;

    [JsonPropertyName("operation")]
    public string Operation { get; set; } = "ReservedDeviceNames";

    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    [JsonPropertyName("status")]
    public string Status { get; set; } = "Running";

    [JsonPropertyName("dry_run")]
    public bool DryRun { get; set; }

    [JsonPropertyName("performance")]
    public PerformanceInfo Performance { get; set; } = new();

    [JsonPropertyName("results")]
    public ResultsInfo? Results { get; set; }
}

internal sealed class PerformanceInfo
{
    [JsonPropertyName("mode")]
    public string Mode { get; set; } = "Rust/Parallel";

    [JsonPropertyName("threads")]
    public int Threads { get; set; } = Environment.ProcessorCount;

    [JsonPropertyName("elapsed_ms")]
    public long ElapsedMs { get; set; }
}

/// <summary>
/// Auditable results: counts plus the actual paths acted on, mirroring the
/// engine's JSON so nothing is lost translating Rust's payload into the
/// tool's top-level output shape.
/// </summary>
internal sealed class ResultsInfo
{
    [JsonPropertyName("scanned")]
    public uint Scanned { get; set; }

    [JsonPropertyName("deleted")]
    public int Deleted { get; set; }

    [JsonPropertyName("would_delete")]
    public int WouldDelete { get; set; }

    [JsonPropertyName("skipped")]
    public int Skipped { get; set; }

    [JsonPropertyName("errors")]
    public uint Errors { get; set; }

    [JsonPropertyName("deleted_entries")]
    public List<EngineActionEntry> DeletedEntries { get; set; } = new();

    [JsonPropertyName("would_delete_entries")]
    public List<EngineActionEntry> WouldDeleteEntries { get; set; } = new();

    [JsonPropertyName("skipped_entries")]
    public List<EngineActionEntry> SkippedEntries { get; set; } = new();
}

/// <summary>
/// Error message structure for JSON output
/// </summary>
internal sealed class ErrorResult
{
    [JsonPropertyName("tool")]
    public string Tool { get; set; } = "Nuke-Nul";

    [JsonPropertyName("status")]
    public string Status { get; set; } = "Error";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;
}

/// <summary>
/// Deserialization target for the JSON string returned by the Rust
/// <c>nuke_files_ex</c> FFI function. Field names match Rust's serde output
/// (snake_case) via explicit <see cref="JsonPropertyNameAttribute"/>.
/// </summary>
internal sealed class EngineOutput
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("dry_run")]
    public bool DryRun { get; set; }

    [JsonPropertyName("counts")]
    public EngineCounts Counts { get; set; } = new();

    [JsonPropertyName("deleted")]
    public List<EngineActionEntry> Deleted { get; set; } = new();

    [JsonPropertyName("would_delete")]
    public List<EngineActionEntry> WouldDelete { get; set; } = new();

    [JsonPropertyName("skipped")]
    public List<EngineActionEntry> Skipped { get; set; } = new();
}

internal sealed class EngineCounts
{
    [JsonPropertyName("scanned")]
    public uint Scanned { get; set; }

    [JsonPropertyName("deleted")]
    public int Deleted { get; set; }

    [JsonPropertyName("would_delete")]
    public int WouldDelete { get; set; }

    [JsonPropertyName("skipped")]
    public int Skipped { get; set; }

    [JsonPropertyName("errors")]
    public uint Errors { get; set; }
}

internal sealed class EngineActionEntry
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("family")]
    public string Family { get; set; } = string.Empty;

    [JsonPropertyName("kind")]
    public string Kind { get; set; } = string.Empty;

    [JsonPropertyName("reason")]
    public string? Reason { get; set; }

    [JsonPropertyName("size")]
    public ulong? Size { get; set; }

    [JsonPropertyName("content_preview")]
    public string? ContentPreview { get; set; }
}

internal static class NativeMethods
{
    private const string DllName = "nuker_core.dll";

    /// <summary>
    /// Runs a scan/delete pass and returns a heap-allocated, null-terminated
    /// UTF-8 JSON string describing the result. The caller MUST pass the
    /// returned pointer to <see cref="nuke_free_string"/> exactly once.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr nuke_files_ex(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        NukeOptions options);

    /// <summary>
    /// Frees a string previously returned by <see cref="nuke_files_ex"/>.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void nuke_free_string(IntPtr ptr);
}

internal static class Program
{
    /// <summary>
    /// Parsed CLI flags before family-default resolution.
    /// </summary>
    private sealed class ParsedArgs
    {
        public string TargetPath = ".";
        public bool Reserved;
        public bool DollarNull;
        public bool PathMangle;
        public bool DollarPrefix;
        public bool All;
        public bool IncludeDirs;
        public bool DryRun;
        public bool AllowNonempty;
    }

    private static int Main(string[] args)
    {
        if (Array.Exists(args, arg => arg is "--help" or "-h" or "/?"))
        {
            WriteHelp();
            return 0;
        }

        if (!TryParseArguments(args, out ParsedArgs parsed, out string? argumentError))
        {
            WriteError(argumentError!);
            return 1;
        }

        string targetPath = parsed.TargetPath;

        // Validate and resolve target path
        if (!ValidateTargetPath(ref targetPath, out string? errorMessage))
        {
            WriteError(errorMessage!);
            return 1;
        }

        // Resolve which match families are active. Bare invocation (no family
        // flag at all) preserves the original default: reserved-device names
        // only. --dollar-null-only alone keeps its historical "only" behavior
        // (reserved is NOT auto-included). Any explicit combination of
        // --reserved / --dollar-null-only / --path-mangle / --dollar-prefix
        // composes exactly the families named. --all is shorthand for all four.
        bool anyFamilyFlag = parsed.Reserved || parsed.DollarNull || parsed.PathMangle || parsed.DollarPrefix || parsed.All;
        bool finalReserved = parsed.All || parsed.Reserved || !anyFamilyFlag;
        bool finalDollarNull = parsed.All || parsed.DollarNull;
        bool finalPathMangle = parsed.All || parsed.PathMangle;
        bool finalDollarPrefix = parsed.All || parsed.DollarPrefix;

        var options = new NukeOptions
        {
            MatchReserved = (byte)(finalReserved ? 1 : 0),
            MatchDollarNull = (byte)(finalDollarNull ? 1 : 0),
            MatchPathMangle = (byte)(finalPathMangle ? 1 : 0),
            MatchDollarPrefix = (byte)(finalDollarPrefix ? 1 : 0),
            IncludeDirs = (byte)(parsed.IncludeDirs ? 1 : 0),
            DryRun = (byte)(parsed.DryRun ? 1 : 0),
            AllowNonempty = (byte)(parsed.AllowNonempty ? 1 : 0),
        };

        // Initialize result object
        var result = new ScanResult
        {
            Target = targetPath,
            Operation = BuildOperationLabel(finalReserved, finalDollarNull, finalPathMangle, finalDollarPrefix, parsed.IncludeDirs, parsed.AllowNonempty),
            Timestamp = DateTime.UtcNow,
            DryRun = parsed.DryRun,
        };

        // Verify DLL exists before attempting to call it
        if (!VerifyDllExists())
        {
            result.Status = "Fatal Error";
            WriteError("nuker_core.dll not found. Please ensure the Rust DLL is in the same directory as NukeNul.exe");
            return 2;
        }

        // Execute the Rust file scanning and deletion
        var stopwatch = Stopwatch.StartNew();
        IntPtr resultPtr = IntPtr.Zero;

        try
        {
            // Critical P/Invoke call - blocks while Rust uses all CPU cores
            resultPtr = NativeMethods.nuke_files_ex(targetPath, options);
            stopwatch.Stop();

            if (resultPtr == IntPtr.Zero)
            {
                result.Status = "Fatal Error";
                result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
                WriteError("nuke_files_ex returned a null pointer unexpectedly");
                return 99;
            }

            string json = Marshal.PtrToStringUTF8(resultPtr) ?? string.Empty;
            EngineOutput? engine = JsonSerializer.Deserialize(json, SourceGenerationContext.Default.EngineOutput);

            if (engine is null)
            {
                result.Status = "Fatal Error";
                result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
                WriteError("Failed to parse engine result JSON");
                return 99;
            }

            if (!string.Equals(engine.Status, "success", StringComparison.Ordinal))
            {
                result.Status = "Fatal Error";
                result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
                WriteError(engine.Message ?? "Unknown engine error");
                return 99;
            }

            // Update result with success data
            result.Status = "Success";
            result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
            result.Results = new ResultsInfo
            {
                Scanned = engine.Counts.Scanned,
                Deleted = engine.Counts.Deleted,
                WouldDelete = engine.Counts.WouldDelete,
                Skipped = engine.Counts.Skipped,
                Errors = engine.Counts.Errors,
                DeletedEntries = engine.Deleted,
                WouldDeleteEntries = engine.WouldDelete,
                SkippedEntries = engine.Skipped,
            };

            // Output JSON to stdout
            WriteJson(result);

            // Return exit code based on errors
            return engine.Counts.Errors > 0 ? 3 : 0;
        }
        catch (DllNotFoundException ex)
        {
            stopwatch.Stop();
            result.Status = "Fatal Error";
            result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
            WriteError($"Failed to load nuker_core.dll: {ex.Message}");
            return 2;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            result.Status = "Fatal Error";
            result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
            WriteError($"Unexpected error: {ex.Message}");
            return 99;
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
            {
                NativeMethods.nuke_free_string(resultPtr);
            }
        }
    }

    /// <summary>
    /// Parses CLI flags and the optional target directory.
    /// </summary>
    private static bool TryParseArguments(
        string[] args,
        out ParsedArgs parsed,
        out string? errorMessage)
    {
        parsed = new ParsedArgs();
        errorMessage = null;
        bool targetProvided = false;

        foreach (string arg in args)
        {
            switch (arg.ToLowerInvariant())
            {
                case "--dollar-null-only":
                    parsed.DollarNull = true;
                    break;
                case "--path-mangle":
                    parsed.PathMangle = true;
                    break;
                case "--dollar-prefix":
                    parsed.DollarPrefix = true;
                    break;
                case "--reserved":
                    parsed.Reserved = true;
                    break;
                case "--all":
                    parsed.All = true;
                    break;
                case "--include-dirs":
                    parsed.IncludeDirs = true;
                    break;
                case "--dry-run":
                case "-n":
                    parsed.DryRun = true;
                    break;
                case "--allow-nonempty":
                    parsed.AllowNonempty = true;
                    break;
                default:
                    if (arg.StartsWith("-", StringComparison.Ordinal))
                    {
                        errorMessage = $"Unknown option: {arg}";
                        return false;
                    }

                    if (targetProvided)
                    {
                        errorMessage = "Only one target directory may be specified.";
                        return false;
                    }

                    parsed.TargetPath = arg;
                    targetProvided = true;
                    break;
            }
        }

        return true;
    }

    /// <summary>
    /// Builds a human-readable operation label from the resolved options.
    /// </summary>
    private static string BuildOperationLabel(bool reserved, bool dollarNull, bool pathMangle, bool dollarPrefix, bool includeDirs, bool allowNonempty)
    {
        var families = new List<string>();
        if (reserved)
        {
            families.Add("ReservedDeviceNames");
        }

        if (dollarNull)
        {
            families.Add("LiteralDollarNull");
        }

        if (pathMangle)
        {
            families.Add("PathMangle");
        }

        if (dollarPrefix)
        {
            families.Add("DollarPrefix");
        }

        string label = families.Count > 0 ? string.Join("+", families) : "None";

        var modifiers = new List<string>();
        if (includeDirs)
        {
            modifiers.Add("IncludeDirs");
        }

        if (allowNonempty)
        {
            modifiers.Add("AllowNonempty");
        }

        return modifiers.Count > 0 ? $"{label} ({string.Join(",", modifiers)})" : label;
    }

    private static void WriteHelp()
    {
        Console.WriteLine(
            """
            NukeNul - Windows problematic-filename cleaner

            Usage:
              NukeNul.exe [options] [target-directory]

            Match families (composable - pass any combination):
              --reserved            Reserved device-name files (nul, con, prn, aux,
                                     com1-9, lpt1-9). Active by default when no other
                                     family flag is given.
              --dollar-null-only     Literal $null files/dirs. Zero-byte files are
                                     deleted by default; see --allow-nonempty.
              --path-mangle          Shell path-mangle artifacts: leaf names ending in
                                     ";<letter>" or ";<letter>:" (e.g. "foo;C").
              --dollar-prefix         Leading-"$" shell-variable artifacts other than
                                      $null (e.g. "$runDir", "$out", "$archiveDir"),
                                      produced when a PowerShell $variable name is
                                      stripped in a bash context. Zero-byte files are
                                      deleted by default; see --allow-nonempty.
              --all                  Shorthand for all four families above.

            Modifiers:
              --include-dirs         Also remove matching directories, but ONLY if they
                                      are empty (never recursive; RemoveDirectoryW itself
                                      refuses non-empty directories).
              --dry-run, -n           Preview only - walk and match, delete nothing.
              --allow-nonempty        Allow deleting non-zero-byte files matched by
                                      --dollar-null-only or --dollar-prefix (default is
                                      zero-byte-only for both).

            Examples:
              NukeNul.exe C:\Path\To\Scan
              NukeNul.exe --dry-run --all C:\Path\To\Scan
              NukeNul.exe --dollar-null-only --path-mangle --include-dirs C:\Path
              NukeNul.exe --dollar-prefix --include-dirs C:\Path

            Behavior change: --dollar-null-only now deletes ONLY zero-byte $null files
            by default (previously deleted files of any size). Pass --allow-nonempty to
            restore the old behavior for a given run. The same zero-byte-only default
            applies to the new --dollar-prefix family.
            """);
    }

    /// <summary>
    /// Validates and resolves the target path to an absolute path
    /// </summary>
    private static bool ValidateTargetPath(ref string targetPath, out string? errorMessage)
    {
        try
        {
            // Resolve to absolute path
            targetPath = Path.GetFullPath(targetPath);

            // Verify directory exists
            if (!Directory.Exists(targetPath))
            {
                errorMessage = $"Target directory does not exist: {targetPath}";
                return false;
            }

            errorMessage = null;
            return true;
        }
        catch (Exception ex)
        {
            errorMessage = $"Invalid target path: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Verifies that the Rust DLL exists in the expected location
    /// </summary>
    private static bool VerifyDllExists()
    {
        // Check in the same directory as the executable
        string exeDirectory = AppContext.BaseDirectory;
        string dllPath = Path.Combine(exeDirectory, "nuker_core.dll");
        return File.Exists(dllPath);
    }

    /// <summary>
    /// Writes a JSON object to stdout
    /// </summary>
    private static void WriteJson(ScanResult result)
    {
        string json = JsonSerializer.Serialize(result, SourceGenerationContext.Default.ScanResult);
        Console.WriteLine(json);
    }

    /// <summary>
    /// Writes an error message as JSON to stdout
    /// </summary>
    private static void WriteError(string message)
    {
        var errorResult = new ErrorResult
        {
            Tool = "Nuke-Nul",
            Status = "Error",
            Message = message
        };

        string json = JsonSerializer.Serialize(errorResult, SourceGenerationContext.Default.ErrorResult);
        Console.WriteLine(json);
    }
}
