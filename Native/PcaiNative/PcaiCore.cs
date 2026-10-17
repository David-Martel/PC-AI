using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PcaiNative;

/// <summary>
/// High-level wrapper for PCAI Core functionality.
/// </summary>
public static class PcaiCore
{
    // Magic number expected from pcai_core_test()
    // Value 0x50434149 spells "PCAI" in ASCII hex
    private const uint ExpectedMagicNumber = 0x5043_4149;

    /// <summary>
    /// Unconditional C ABI exports required for a <c>pcai_core_lib</c> artifact to be admitted
    /// as functional by <see cref="IsAvailable"/>. A partial or magic-only stub that omits any
    /// required entry point is rejected before P/Invoke operations execute.
    /// </summary>
    public static readonly IReadOnlyList<string> RequiredExports = new[]
    {
        "pcai_core_version",
        "pcai_core_test",
        "pcai_free_string",
        "pcai_free_string_buffer",
        "pcai_free_byte_buffer",
        "pcai_cpu_count",
        "pcai_status_description",
        "pcai_string_copy",
        "pcai_extract_json",
        "pcai_is_valid_json",
        "pcai_estimate_tokens",
        "pcai_find_files",
        "pcai_find_files_stats",
        "pcai_find_files_compact",
        "pcai_collect_directory_manifest",
        "pcai_collect_directory_manifest_stats",
        "pcai_search_content",
        "pcai_search_content_stats",
        "pcai_search_content_compact",
        "pcai_find_duplicates",
        "pcai_find_duplicates_stats",
        "pcai_query_system_info",
        "pcai_query_hardware_metrics",
        "pcai_check_resource_safety",
        "pcai_get_system_telemetry_json",
        "pcai_get_vmm_health_json",
        "pcai_query_full_context_json",
        "pcai_get_usb_deep_diagnostics_json",
        "pcai_get_pnp_devices_json",
        "pcai_get_pnp_problem_info",
        "pcai_get_disk_health_json",
        "pcai_sample_hardware_events_json",
        "pcai_get_network_throughput_json",
        "pcai_get_process_history_json",
        "pcai_get_process_lasso_snapshot_json",
        "pcai_get_usb_problem_info",
        "pcai_analyze_path",
        "pcai_analyze_path_json",
        "pcai_search_logs",
        "pcai_search_logs_json",
        "pcai_get_disk_usage",
        "pcai_get_disk_usage_json",
        "pcai_get_disk_usage_compact",
        "pcai_get_process_stats",
        "pcai_get_top_processes_json",
        "pcai_get_top_processes_compact",
        "pcai_get_memory_stats",
        "pcai_get_memory_stats_json",
        "pcai_fs_version",
        "pcai_replace_in_file",
        "pcai_replace_in_files",
        "pcai_delete_fs_item",
        "pcai_analyze_memory_pressure",
        "pcai_get_memory_pressure_json",
        "pcai_get_process_categories_json",
        "pcai_get_optimization_recommendations_json",
        "pcai_query_prompt_assembly",
    };

    /// <summary>
    /// Optional feature-gated C ABI exports declared in <see cref="NativeCore"/>
    /// (for example, <c>functiongemma</c>).
    /// </summary>
    public static readonly IReadOnlyList<string> OptionalFeatureExports = new[]
    {
        "pcai_build_router_dataset_jsonl",
    };

    /// <summary>
    /// Verifies that every symbol in <see cref="RequiredExports"/> is exported by <paramref name="libraryHandle"/>.
    /// </summary>
    public static bool VerifyRequiredExports(IntPtr libraryHandle, out IReadOnlyList<string> missingExports)
    {
        if (libraryHandle == IntPtr.Zero)
        {
            missingExports = RequiredExports;
            return false;
        }

        List<string>? missing = null;
        foreach (var exportName in RequiredExports)
        {
            if (!NativeLibrary.TryGetExport(libraryHandle, exportName, out var symbolAddress) ||
                symbolAddress == IntPtr.Zero)
            {
                missing ??= new List<string>();
                missing.Add(exportName);
            }
        }

        if (missing is { Count: > 0 })
        {
            missingExports = missing;
            return false;
        }

        missingExports = Array.Empty<string>();
        return true;
    }

    /// <summary>
    /// Loads <paramref name="libraryPath"/> and verifies that it exports all <see cref="RequiredExports"/>.
    /// </summary>
    public static bool VerifyLibraryExports(string libraryPath, out IReadOnlyList<string> missingExports)
    {
        if (string.IsNullOrWhiteSpace(libraryPath) || !File.Exists(libraryPath))
        {
            missingExports = RequiredExports;
            return false;
        }

        if (!NativeLibrary.TryLoad(libraryPath, out var handle))
        {
            missingExports = RequiredExports;
            return false;
        }

        try
        {
            return VerifyRequiredExports(handle, out missingExports);
        }
        finally
        {
            // This admission load owns one reference, independent of P/Invoke's binding.
            NativeLibrary.Free(handle);
        }
    }

    // Thread-safe lazy initialization for availability check
    private static readonly Lazy<bool> _isAvailable = new(EvaluateAvailabilityCore);

    private static bool EvaluateAvailabilityCore()
    {
        try
        {
            NativeResolver.EnsureRegistered(typeof(NativeCore).Assembly);

            if (NativeResolver.TryResolveLibraryPath(NativeCore.CoreLib, out var resolvedPath, out var provenance, typeof(NativeCore).Assembly))
            {
                if (string.IsNullOrEmpty(resolvedPath) || !VerifyLibraryExports(resolvedPath, out _))
                    return false;
            }
            else if (provenance is not null && provenance.StartsWith("env:", StringComparison.Ordinal))
            {
                return false;
            }
            else if (NativeLibrary.TryLoad(NativeCore.CoreLib, typeof(NativeCore).Assembly, null, out var defaultHandle))
            {
                try
                {
                    if (!VerifyRequiredExports(defaultHandle, out _))
                        return false;
                }
                finally
                {
                    NativeLibrary.Free(defaultHandle);
                }
            }
            else
            {
                return false;
            }

            if (NativeCore.pcai_core_test() != ExpectedMagicNumber)
                return false;

            using var probe = NativeCore.pcai_string_copy("pcai");
            return !probe.IsInvalid && string.Equals(probe.ToString(), "pcai", StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }

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
    /// Gets whether the native library is available, exports all required C ABI symbols, and passes magic/probe checks.
    /// Fails closed if an explicit environment variable override points to a missing or incomplete library.
    /// </summary>
    public static bool IsAvailable
    {
        get
        {
            if (!NativeResolver.TryResolveLibraryPath(NativeCore.CoreLib, out var resolvedPath, out var provenance, typeof(NativeCore).Assembly))
            {
                if (provenance is not null && provenance.StartsWith("env:", StringComparison.Ordinal))
                    return false;
            }
            else if (provenance is not null &&
                     provenance.StartsWith("env:", StringComparison.Ordinal) &&
                     (string.IsNullOrEmpty(resolvedPath) || !VerifyLibraryExports(resolvedPath, out _)))
            {
                return false;
            }

            return _isAvailable.Value;
        }
    }

    /// <summary>
    /// Gets the native library version, or <c>"unavailable"</c> when <see cref="IsAvailable"/> is <c>false</c>.
    /// </summary>
    public static string Version => IsAvailable ? _version.Value : "unavailable";

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
            }
            catch { /* Fallback to standard serialization */ }
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

        var snapshot = new
        {
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
        var available = IsAvailable;
        return new NativeDiagnostics
        {
            IsAvailable = available,
            Version = Version,
            CpuCount = CpuCount,
            Platform = Environment.Is64BitProcess ? "x64" : "x86",
            DotNetVersion = Environment.Version.ToString(),
            RuntimeIdentifier = NativeResolver.GetCurrentRuntimeIdentifier(),
            ResolvedLibraryPath = available ? NativeResolver.GetResolvedLibraryPath(NativeCore.CoreLib) : null,
            ProvenanceSource = available ? NativeResolver.GetResolvedProvenance(NativeCore.CoreLib) : null,
        };
    }
}
