using System;
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
internal partial class SourceGenerationContext : JsonSerializerContext
{
}

/// <summary>
/// C-compatible struct matching Rust's ScanStats layout
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct ScanStats
{
    public uint FilesScanned;
    public uint FilesDeleted;
    public uint Errors;
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

internal sealed class ResultsInfo
{
    [JsonPropertyName("scanned")]
    public uint Scanned { get; set; }

    [JsonPropertyName("deleted")]
    public uint Deleted { get; set; }

    [JsonPropertyName("errors")]
    public uint Errors { get; set; }
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

internal static class NativeMethods
{
    private const string DllName = "nuker_core.dll";

    /// <summary>
    /// Imports the Rust function that performs parallel file scanning and deletion
    /// </summary>
    /// <param name="rootPath">UTF-8 encoded root path to scan</param>
    /// <returns>Statistics struct containing scan results</returns>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern ScanStats nuke_reserved_files([MarshalAs(UnmanagedType.LPStr)] string rootPath);

    /// <summary>
    /// Deletes only ordinary files whose visible leaf is literal $null.
    /// </summary>
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    internal static extern ScanStats nuke_dollar_null_files(
        [MarshalAs(UnmanagedType.LPStr)] string rootPath);
}

internal static class Program
{

    private static int Main(string[] args)
    {
        if (Array.Exists(args, arg => arg is "--help" or "-h" or "/?"))
        {
            WriteHelp();
            return 0;
        }

        if (!TryParseArguments(args, out string targetPath, out bool dollarNullOnly,
                out string? argumentError))
        {
            WriteError(argumentError!);
            return 1;
        }

        // Validate and resolve target path
        if (!ValidateTargetPath(ref targetPath, out string? errorMessage))
        {
            WriteError(errorMessage!);
            return 1;
        }

        // Initialize result object
        var result = new ScanResult
        {
            Target = targetPath,
            Operation = dollarNullOnly ? "LiteralDollarNullOnly" : "ReservedDeviceNames",
            Timestamp = DateTime.UtcNow
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

        try
        {
            // Critical P/Invoke call - blocks while Rust uses all CPU cores
            ScanStats stats = dollarNullOnly
                ? NativeMethods.nuke_dollar_null_files(targetPath)
                : NativeMethods.nuke_reserved_files(targetPath);
            stopwatch.Stop();

            // Update result with success data
            result.Status = "Success";
            result.Performance.ElapsedMs = stopwatch.ElapsedMilliseconds;
            result.Results = new ResultsInfo
            {
                Scanned = stats.FilesScanned,
                Deleted = stats.FilesDeleted,
                Errors = stats.Errors
            };

            // Output JSON to stdout
            WriteJson(result);

            // Return exit code based on errors
            return stats.Errors > 0 ? 3 : 0;
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
    }

    /// <summary>
    /// Parses one optional target path and the narrow literal-$null mode.
    /// </summary>
    private static bool TryParseArguments(
        string[] args,
        out string targetPath,
        out bool dollarNullOnly,
        out string? errorMessage)
    {
        targetPath = ".";
        dollarNullOnly = false;
        errorMessage = null;
        bool targetProvided = false;

        foreach (string arg in args)
        {
            if (arg.Equals("--dollar-null-only", StringComparison.OrdinalIgnoreCase))
            {
                if (dollarNullOnly)
                {
                    errorMessage = "Option --dollar-null-only may be specified only once.";
                    return false;
                }

                dollarNullOnly = true;
                continue;
            }

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

            targetPath = arg;
            targetProvided = true;
        }

        return true;
    }

    private static void WriteHelp()
    {
        Console.WriteLine(
            """
            NukeNul - Windows problematic-filename cleaner

            Usage:
              NukeNul.exe [--dollar-null-only] [target-directory]

            Modes:
              default             Delete reserved device-name files.
              --dollar-null-only  Delete only real files named literal $null
                                  (case-insensitive; trailing dots/spaces allowed).
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
