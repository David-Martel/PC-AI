using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;

namespace PcaiNative;

/// <summary>
/// High-level wrapper for PCAI Core functionality.
/// </summary>
public static class PcaiCore
{
    // Magic number expected from pcai_core_test()
    // Value 0x50434149 spells "PCAI" in ASCII hex
    private const uint ExpectedMagicNumber = 0x5043_4149;

    // Thread-safe lazy initialization for availability check
    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            var result = NativeCore.pcai_core_test();
            return result == ExpectedMagicNumber;
        }
        catch
        {
            return false;
        }
    });

    // Thread-safe lazy initialization for version string
    private static readonly Lazy<string> _version = new(() =>
    {
        try
        {
            var ptr = NativeCore.pcai_core_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        }
        catch
        {
            return "unavailable";
        }
    });

    /// <summary>
    /// Gets whether the native library is available and functional.
    /// </summary>
    public static bool IsAvailable => _isAvailable.Value;

    /// <summary>
    /// Gets the native library version.
    /// </summary>
    public static string Version => _version.Value;

    /// <summary>
    /// Gets the number of logical CPU cores reported by the native library.
    /// </summary>
    public static uint CpuCount
    {
        get
        {
            try
            {
                return NativeCore.pcai_cpu_count();
            }
            catch
            {
                return (uint)Environment.ProcessorCount;
            }
        }
    }

    /// <summary>
    /// Gets a human-readable description for a status code.
    /// </summary>
    public static string GetStatusDescription(PcaiStatus status)
    {
        try
        {
            var ptr = NativeCore.pcai_status_description(status);
            return Marshal.PtrToStringUTF8(ptr) ?? status.ToString();
        }
        catch
        {
            return status.ToString();
        }
    }

    /// <summary>
    /// Tests string allocation round-trip with the native library.
    /// </summary>
    public static string? TestStringCopy(string input)
    {
        if (!IsAvailable) return null;

        using var handle = NativeCore.pcai_string_copy(input);
        if (handle.IsInvalid) return null;
        return handle.ToString();
    }

    /// <summary>
    /// Extracts JSON from a markdown-formatted string using high-performance native logic.
    /// </summary>
    public static string? ExtractJson(string input)
    {
        if (!IsAvailable) return null;

        using var handle = NativeCore.pcai_extract_json(input);
        if (handle.IsInvalid) return null;
        return handle.ToString();
    }

    /// <summary>
    /// Validates if a string is valid JSON using native logic.
    /// </summary>
    public static bool IsValidJson(string input)
    {
        if (!IsAvailable) return false;
        return NativeCore.pcai_is_valid_json(input);
    }

    /// <summary>
    /// Searches for files matching a glob pattern using native traversal.
    /// </summary>
    public static string? FindFiles(string rootPath, string pattern, ulong maxResults = 0)
    {
        return PcaiSearch.FindFilesJson(rootPath, pattern, (uint)maxResults);
    }

    /// <summary>
    /// Searches file contents using parallel native regex matching.
    /// </summary>
    public static string? SearchContent(string rootPath, string pattern, string? filePattern = null, ulong maxResults = 0, uint contextLines = 0)
    {
        return PcaiSearch.SearchContentJson(rootPath, pattern, filePattern, (uint)maxResults, contextLines);
    }

    /// <summary>
    /// Collects a directory manifest using native traversal.
    /// </summary>
    public static string? CollectDirectoryManifest(string rootPath, uint maxDepth = 0, ulong maxResults = 0)
    {
        return PcaiSearch.CollectDirectoryManifestJson(rootPath, maxDepth, (uint)maxResults);
    }

    /// <summary>
    /// Finds duplicate files using parallel native hashing.
    /// </summary>
    public static string? FindDuplicates(string rootPath, ulong minSize = 0, string? includePattern = null, string? excludePattern = null)
    {
        return PcaiSearch.FindDuplicatesJson(rootPath, minSize, includePattern, excludePattern);
    }

    /// <summary>
    /// Queries comprehensive system information natively.
    /// </summary>
    public static string? QuerySystemInfo()
    {
        return SystemModule.QuerySystemInfo();
    }

    /// <summary>
    /// Analyzes the system PATH environment variable natively for issues.
    /// </summary>
    public static PathAnalysisStats AnalyzePath()
    {
        return SystemModule.AnalyzePath();
    }

    /// <summary>
    /// Analyzes the system PATH environment variable and returns a detailed JSON report.
    /// </summary>
    public static string? AnalyzePathJson()
    {
        return SystemModule.AnalyzePathJson();
    }

    /// <summary>
    /// Gets structured hardware metrics natively.
    /// </summary>
    public static PcaiMetrics? GetResourceMetrics()
    {
        var json = QueryHardwareMetrics();
        if (string.IsNullOrEmpty(json)) return null;
        try { return JsonSerializer.Deserialize<PcaiMetrics>(json); }
        catch { return null; }
    }

    /// <summary>
    /// Queries hardware metrics JSON natively.
    /// </summary>
    public static string? QueryHardwareMetrics()
    {
        return PerformanceModule.QueryHardwareMetrics();
    }

    /// <summary>
    /// Estimates the number of tokens in a string for Gemma-like models natively.
    /// </summary>
    public static ulong EstimateTokens(string text)
    {
        if (!IsAvailable) return 0;
        return (ulong)NativeCore.pcai_estimate_tokens(text);
    }

    /// <summary>
    /// Checks if system resources are within safety limits (e.g. 80% load).
    /// </summary>
    public static bool CheckResourceSafety(float gpuLimit = 0.8f)
    {
        return PerformanceModule.CheckResourceSafety(gpuLimit);
    }

    /// <summary>
    /// Gets high-fidelity system telemetry as JSON using native core.
    /// </summary>
    public static string? GetSystemTelemetryJson()
    {
        return SystemModule.GetSystemTelemetryJson();
    }

    /// <summary>
    /// Gets WSL/VMM health status using direct native socket interrogation.
    /// </summary>
    public static string? GetVmmHealthJson()
    {
        return SystemModule.GetVmmHealthJson();
    }

    /// <summary>
    /// Gets the full consolidated diagnostic context for LLM ingestion.
    /// Eliminates multiple PowerShell calls and large JSON marshaling.
    /// </summary>
    public static string? QueryFullContextJson()
    {
        if (!IsAvailable) return null;
        var ptr = NativeCore.pcai_query_full_context_json();
        if (ptr == IntPtr.Zero) return null;
        try { return Marshal.PtrToStringUTF8(ptr); }
        finally { NativeCore.pcai_free_string(ptr); }
    }

    /// <summary>
    /// Assembles a prompt from a template and variables using native high-performance logic.
    /// Handles PowerShell objects by safely converting them to dictionaries before serialization.
    /// </summary>
    public static string? AssemblePrompt(string template, object variables)
    {
        if (!IsAvailable) return null;

        string jsonVars;
        try
        {
            // If it's a PowerShell object (PSObject or PSCustomObject), it often has cycles
            // and internal members that crash System.Text.Json. We convert to a safe dict.
            var safeVars = SafeConvertVariables(variables);

            var options = new JsonSerializerOptions
            {
                WriteIndented = false,
                ReferenceHandler = ReferenceHandler.IgnoreCycles,
                MaxDepth = 128
            };
            jsonVars = JsonSerializer.Serialize(safeVars, options);
        }
        catch (Exception ex)
        {
            jsonVars = $"{{\"error\": \"Serialization failed: {ex.Message}\"}}";
        }

        var buffer = NativeCore.pcai_query_prompt_assembly(template, jsonVars);
        try { return buffer.ToManagedString(); }
        finally { NativeCore.pcai_free_string_buffer(ref buffer); }
    }

    private static object? SafeConvertVariables(object? input, int depth = 0)
    {
        if (input == null || depth > 10) return null;

        var type = input.GetType();

        // Handle basic types directly
        if (type.IsPrimitive || input is string || input is decimal) return input;

        if (type.FullName == "System.Management.Automation.PSObject" ||
            type.FullName == "System.Management.Automation.PSCustomObject")
        {
            var dict = new Dictionary<string, object?>();
            try
            {
                var propsProperty = type.GetProperty("Properties");
                if (propsProperty != null)
                {
                    var props = propsProperty.GetValue(input) as System.Collections.IEnumerable;
                    if (props != null)
                    {
                        foreach (object prop in props)
                        {
                            var propType = prop.GetType();
                            var name = propType.GetProperty("Name")?.GetValue(prop) as string;
                            var val = propType.GetProperty("Value")?.GetValue(prop);

                            if (name != null)
                            {
                                // Skip internal/special properties to avoid bloat/cycles
                                if (name.StartsWith("_")) continue;
                                dict[name] = SafeConvertVariables(val, depth + 1);
                            }
                        }
                        return dict;
                    }
                }
            } catch { /* Fallback to standard serialization */ }
        }

        // Handle collections
        if (input is System.Collections.IEnumerable enumerable && !(input is string))
        {
            var list = new List<object?>();
            foreach (var item in enumerable)
            {
                list.Add(SafeConvertVariables(item, depth + 1));
            }
            return list;
        }

        return input;
    }

    /// <summary>
    /// Gets high-fidelity USB diagnostics using SetupAPI.
    /// </summary>
    public static string? GetUsbDeepDiagnostics()
    {
        if (!IsAvailable) return null;
        var ptr = NativeCore.pcai_get_usb_deep_diagnostics_json();
        if (ptr == IntPtr.Zero) return null;
        try { return Marshal.PtrToStringUTF8(ptr); }
        finally { NativeCore.pcai_free_string(ptr); }
    }

    /// <summary>
    /// Fast delete file or directory using the native FsModule.
    /// </summary>
    public static PcaiStatus DeleteFsItem(string path, bool recursive = false)
    {
        return FsModule.DeleteItem(path, recursive);
    }

    /// <summary>
    /// Gets network throughput and stats using IPHelper.
    /// </summary>
    public static string? GetNetworkThroughput()
    {
        return PerformanceModule.GetNetworkThroughput();
    }

    /// <summary>
    /// Gets detailed process history using Psapi.
    /// </summary>
    public static string? GetProcessHistory()
    {
        return PerformanceModule.GetProcessHistory();
    }

    /// <summary>
    /// Gets human-readable information for a Windows Device Manager problem code.
    /// </summary>
    public static CmProblemInfo? GetUsbProblemInfo(uint code)
    {
        if (!IsAvailable) return null;
        var ptr = NativeCore.pcai_get_usb_problem_info(code);
        if (ptr == IntPtr.Zero) return null;

        try
        {
            var json = Marshal.PtrToStringUTF8(ptr);
            if (string.IsNullOrEmpty(json)) return null;
            return JsonSerializer.Deserialize<CmProblemInfo>(json);
        }
        catch
        {
            return null;
        }
        finally
        {
            NativeCore.pcai_free_string(ptr);
        }
    }

    /// <summary>
    /// Gets a summarized dashboard snapshot for display or LLM status.
    /// Aggregates hardware, thermal, and performance data.
    /// </summary>
    public static string? GetDashboardSnapshotJson()
    {
        if (!IsAvailable) return null;

        var metrics = GetResourceMetrics();
        var telemetry = GetSystemTelemetryJson();
        var vmm = GetVmmHealthJson();

        var snapshot = new {
            Metrics = metrics,
            Telemetry = !string.IsNullOrEmpty(telemetry) ? JsonSerializer.Deserialize<JsonElement>(telemetry) : (object?)null,
            VmmHealth = !string.IsNullOrEmpty(vmm) ? JsonSerializer.Deserialize<JsonElement>(vmm) : (object?)null,
            Timestamp = DateTime.UtcNow,
            Version = Version
        };

        return JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>
    /// Gets disk usage statistics as JSON with detailed breakdown.
    /// </summary>
    public static string? GetDiskUsageJson(string? rootPath = null, uint topN = 10)
    {
        return FsModule.GetDiskUsageJson(rootPath ?? ".", (int)topN);
    }

    /// <summary>
    /// Gets diagnostic information about the native library.
    /// </summary>
    public static NativeDiagnostics GetDiagnostics()
    {
        return new NativeDiagnostics
        {
            IsAvailable = IsAvailable,
            Version = Version,
            CpuCount = CpuCount,
            Platform = Environment.Is64BitProcess ? "x64" : "x86",
            DotNetVersion = Environment.Version.ToString()
        };
    }
}
