using System.Runtime.InteropServices;

namespace PcaiNative;

/// <summary>
/// Unified P/Invoke declarations for pcai_core_lib.dll.
/// Consolidates all native functions from Core, Search, Performance, and System modules.
/// </summary>
internal static partial class NativeCore
{
    private const string CoreDll = "pcai_core_lib.dll";

    // ========================================================================
    // Basic Library Management
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr pcai_core_version();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint pcai_core_test();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr pcai_search_version();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void pcai_free_string(IntPtr buffer);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void pcai_free_string_buffer(ref PcaiStringBuffer buffer);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void pcai_free_byte_buffer(ref PcaiByteBuffer buffer);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint pcai_cpu_count();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr pcai_status_description(PcaiStatus status);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_string_copy([MarshalAs(UnmanagedType.LPUTF8Str)] string input);

    // ========================================================================
    // JSON & Utilities
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_extract_json([MarshalAs(UnmanagedType.LPUTF8Str)] string? input);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool pcai_is_valid_json([MarshalAs(UnmanagedType.LPUTF8Str)] string? input);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern UIntPtr pcai_estimate_tokens([MarshalAs(UnmanagedType.LPUTF8Str)] string? text);

    // ========================================================================
    // Search & Duplicate Detection
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_find_files(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern FileSearchStats pcai_find_files_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiByteBuffer pcai_find_files_compact(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_collect_directory_manifest(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        uint maxDepth,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern DirectoryManifestStats pcai_collect_directory_manifest_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        uint maxDepth,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_search_content(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        ulong maxResults,
        uint contextLines);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ContentSearchStats pcai_search_content_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiByteBuffer pcai_search_content_compact(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        ulong maxResults);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_find_duplicates(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        ulong minSize,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? includePattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? excludePattern);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern DuplicateStats pcai_find_duplicates_stats(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        ulong minSize,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? includePattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? excludePattern);

    // ========================================================================
    // System & Hardware Telemetry
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_gpu_utilization_json(uint deviceIndex);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_cuda_driver_version_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int pcai_gpu_count();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_gpu_info_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_driver_version();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_query_system_info();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_query_hardware_metrics();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int pcai_check_resource_safety(float gpuLimit);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_system_telemetry_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_vmm_health_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_query_full_context_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_usb_deep_diagnostics_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_pnp_devices_json([MarshalAs(UnmanagedType.LPUTF8Str)] string? classFilter);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_pnp_problem_info(uint code);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_disk_health_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_sample_hardware_events_json(uint days, uint maxEvents);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_network_throughput_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_process_history_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_process_lasso_snapshot_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? configPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? logPath,
        uint lookbackMinutes);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern SafeRustStringHandle pcai_get_usb_problem_info(uint code);

    // ========================================================================
    // System Analysis
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_normalize_path(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? path,
        uint style);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PathAnalysisStats pcai_analyze_path();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_analyze_path_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern LogSearchStats pcai_search_logs(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        [MarshalAs(UnmanagedType.U1)] bool caseSensitive,
        uint contextLines,
        uint maxMatches);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_search_logs_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        [MarshalAs(UnmanagedType.U1)] bool caseSensitive,
        uint contextLines,
        uint maxMatches);

    // ========================================================================
    // Performance Monitoring
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern DiskUsageStats pcai_get_disk_usage(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        uint topN);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_disk_usage_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        uint topN);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiByteBuffer pcai_get_disk_usage_compact(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? rootPath,
        uint topN);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ProcessStats pcai_get_process_stats();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_top_processes_json(
        uint topN,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? sortBy);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiByteBuffer pcai_get_top_processes_compact(
        uint topN,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? sortBy);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MemoryStats pcai_get_memory_stats();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_memory_stats_json();

    // ========================================================================
    // Filesystem Operations
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint pcai_fs_version();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStatus pcai_replace_in_file(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string filePath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string pattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string replacement,
        [MarshalAs(UnmanagedType.U1)] bool isRegex,
        [MarshalAs(UnmanagedType.U1)] bool backup);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_replace_in_files(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string rootPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? filePattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string contentPattern,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string replacement,
        [MarshalAs(UnmanagedType.U1)] bool isRegex,
        [MarshalAs(UnmanagedType.U1)] bool backup);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStatus pcai_delete_fs_item(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        [MarshalAs(UnmanagedType.U1)] bool recursive);

    // ========================================================================
    // Memory Optimizer
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MemoryPressureReport pcai_analyze_memory_pressure();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_memory_pressure_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_process_categories_json();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_get_optimization_recommendations_json();

    // ========================================================================
    // Prompt & LLM Ops
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_query_prompt_assembly(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? template,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? jsonVars);

    // ========================================================================
    // FunctionGemma Dataset Ops
    // ========================================================================

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    internal static extern PcaiStringBuffer pcai_build_router_dataset_jsonl(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? toolsPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? scenariosPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? outputJsonl,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? outputVectors,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? diagnosePrompt,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? chatPrompt,
        uint maxCases,
        [MarshalAs(UnmanagedType.U1)] bool includeToolCoverage);
}
