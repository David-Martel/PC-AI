using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PcaiNative
{
    /// <summary>
    /// Disk usage statistics returned by native functions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DiskUsageStats
    {
        public PcaiStatus Status;
        public ulong TotalSizeBytes;
        public ulong TotalFiles;
        public ulong TotalDirs;
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;
    }

    /// <summary>
    /// Process statistics returned by native functions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ProcessStats
    {
        public PcaiStatus Status;
        public uint TotalProcesses;
        public uint TotalThreads;
        public float SystemCpuUsage;
        public ulong SystemMemoryUsedBytes;
        public ulong SystemMemoryTotalBytes;
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;

        public double MemoryUsagePercent => SystemMemoryTotalBytes > 0
            ? (double)SystemMemoryUsedBytes / SystemMemoryTotalBytes * 100.0
            : 0.0;
    }

    /// <summary>
    /// Memory statistics returned by native functions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStats
    {
        public PcaiStatus Status;
        public ulong TotalMemoryBytes;
        public ulong UsedMemoryBytes;
        public ulong AvailableMemoryBytes;
        public ulong TotalSwapBytes;
        public ulong UsedSwapBytes;
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;

        public double MemoryUsagePercent => TotalMemoryBytes > 0
            ? (double)UsedMemoryBytes / TotalMemoryBytes * 100.0
            : 0.0;

        public double SwapUsagePercent => TotalSwapBytes > 0
            ? (double)UsedSwapBytes / TotalSwapBytes * 100.0
            : 0.0;
    }

    public sealed class ProcessEntry
    {
        [JsonPropertyName("pid")]
        public uint Pid { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("cpu_usage")]
        public double CpuUsage { get; set; }

        [JsonPropertyName("memory_bytes")]
        public ulong MemoryBytes { get; set; }

        [JsonPropertyName("exe_path")]
        public string? ExecutablePath { get; set; }

        [JsonPropertyName("status")]
        public string? Status { get; set; }
    }

    public sealed class TopProcessesResult
    {
        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("sort_by")]
        public string? SortBy { get; set; }

        [JsonPropertyName("top_n")]
        public uint TopN { get; set; }

        [JsonPropertyName("elapsed_ms")]
        public ulong ElapsedMs { get; set; }

        [JsonPropertyName("processes")]
        public ProcessEntry[] Processes { get; set; } = Array.Empty<ProcessEntry>();
    }

    public sealed class DiskUsageEntry
    {
        [JsonPropertyName("path")]
        public string? Path { get; set; }

        [JsonPropertyName("size_bytes")]
        public ulong SizeBytes { get; set; }

        [JsonPropertyName("size_formatted")]
        public string? SizeFormatted { get; set; }

        [JsonPropertyName("file_count")]
        public ulong FileCount { get; set; }
    }

    public sealed class DiskUsageReport
    {
        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("root_path")]
        public string? RootPath { get; set; }

        [JsonPropertyName("total_size_bytes")]
        public ulong TotalSizeBytes { get; set; }

        [JsonPropertyName("total_files")]
        public ulong TotalFiles { get; set; }

        [JsonPropertyName("top_entries")]
        public DiskUsageEntry[] TopEntries { get; set; } = Array.Empty<DiskUsageEntry>();
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct ProcessListCompactHeader
    {
        public PcaiStatus Status;
        public uint Reserved;
        public uint TotalProcesses;
        public uint TotalThreads;
        public float SystemCpuUsage;
        public uint CpuPadding0;
        public ulong SystemMemoryUsedBytes;
        public ulong SystemMemoryTotalBytes;
        public ulong ElapsedMs;
        public uint SortByOffset;
        public uint SortByLength;
        public ulong EntryCount;
        public ulong StringBytes;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct ProcessListCompactEntry
    {
        public uint Pid;
        public uint NameOffset;
        public uint NameLength;
        public uint StatusOffset;
        public uint StatusLength;
        public uint ExePathOffset;
        public uint ExePathLength;
        public float CpuUsage;
        public uint CpuPadding0;
        public ulong MemoryBytes;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct DiskUsageCompactHeader
    {
        public PcaiStatus Status;
        public uint Reserved;
        public ulong TotalSizeBytes;
        public ulong TotalFiles;
        public ulong TotalDirs;
        public ulong ElapsedMs;
        public uint RootPathOffset;
        public uint RootPathLength;
        public ulong EntryCount;
        public ulong StringBytes;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct DiskUsageCompactEntry
    {
        public uint PathOffset;
        public uint PathLength;
        public ulong SizeBytes;
        public ulong FileCount;
    }

    /// <summary>
    /// P/Invoke declarations for pcai_core_lib.dll (consolidated from performance module).
    /// Provides disk usage analysis, process monitoring, and memory statistics.
    /// </summary>
    public static class PerformanceModule
    {
        private static readonly Lazy<bool> _isAvailable = new(() =>
        {
            try { return NativeCore.pcai_core_test() == 0x50434149; }
            catch { return false; }
        });

        public static bool IsAvailable => _isAvailable.Value;

        private static bool TryReadStruct<T>(ReadOnlySpan<byte> buffer, ref int offset, out T value) where T : unmanaged
        {
            var size = Marshal.SizeOf<T>();
            if (offset < 0 || offset > buffer.Length - size)
            {
                value = default;
                return false;
            }

            value = MemoryMarshal.Read<T>(buffer.Slice(offset, size));
            offset += size;
            return true;
        }

        private static bool TrySlice(ReadOnlySpan<byte> buffer, int offset, int length, out ReadOnlySpan<byte> slice)
        {
            if (offset < 0 || length < 0 || offset > buffer.Length - length)
            {
                slice = default;
                return false;
            }

            slice = buffer.Slice(offset, length);
            return true;
        }

        private static bool TryReadUtf8String(ReadOnlySpan<byte> stringData, uint offset, uint length, out string value)
        {
            if (length == 0)
            {
                value = string.Empty;
                return true;
            }

            try
            {
                var start = checked((int)offset);
                var count = checked((int)length);
                if (start < 0 || count < 0 || start > stringData.Length - count)
                {
                    value = string.Empty;
                    return false;
                }

                value = System.Text.Encoding.UTF8.GetString(stringData.Slice(start, count));
                return true;
            }
            catch
            {
                value = string.Empty;
                return false;
            }
        }

        private static TopProcessesResult? ParseCompactTopProcesses(PcaiByteBuffer buffer)
        {
            var data = buffer.AsReadOnlySpan();
            if (data.Length < Marshal.SizeOf<ProcessListCompactHeader>())
            {
                return null;
            }

            var offset = 0;
            if (!TryReadStruct<ProcessListCompactHeader>(data, ref offset, out var header) || header.Status != PcaiStatus.Success)
            {
                return null;
            }

            var entryCount = checked((int)header.EntryCount);
            var entries = new ProcessListCompactEntry[entryCount];
            for (var i = 0; i < entries.Length; i++)
            {
                if (!TryReadStruct<ProcessListCompactEntry>(data, ref offset, out entries[i]))
                {
                    return null;
                }
            }

            if (!TrySlice(data, offset, checked((int)header.StringBytes), out var stringData))
            {
                return null;
            }

            if (!TryReadUtf8String(stringData, header.SortByOffset, header.SortByLength, out var sortBy))
            {
                return null;
            }

            var processes = new ProcessEntry[entries.Length];
            for (var i = 0; i < entries.Length; i++)
            {
                var entry = entries[i];
                if (!TryReadUtf8String(stringData, entry.NameOffset, entry.NameLength, out var name) ||
                    !TryReadUtf8String(stringData, entry.StatusOffset, entry.StatusLength, out var status) ||
                    !TryReadUtf8String(stringData, entry.ExePathOffset, entry.ExePathLength, out var exePath))
                {
                    return null;
                }

                processes[i] = new ProcessEntry
                {
                    Pid = entry.Pid,
                    Name = name,
                    CpuUsage = entry.CpuUsage,
                    MemoryBytes = entry.MemoryBytes,
                    ExecutablePath = string.IsNullOrEmpty(exePath) ? null : exePath,
                    Status = status
                };
            }

            return new TopProcessesResult
            {
                Status = header.Status.ToString(),
                SortBy = sortBy,
                TopN = (uint)entryCount,
                ElapsedMs = header.ElapsedMs,
                Processes = processes
            };
        }

        private static DiskUsageReport? ParseCompactDiskUsage(PcaiByteBuffer buffer)
        {
            var data = buffer.AsReadOnlySpan();
            if (data.Length < Marshal.SizeOf<DiskUsageCompactHeader>())
            {
                return null;
            }

            var offset = 0;
            if (!TryReadStruct<DiskUsageCompactHeader>(data, ref offset, out var header) || header.Status != PcaiStatus.Success)
            {
                return null;
            }

            var entryCount = checked((int)header.EntryCount);
            var entries = new DiskUsageCompactEntry[entryCount];
            for (var i = 0; i < entries.Length; i++)
            {
                if (!TryReadStruct<DiskUsageCompactEntry>(data, ref offset, out entries[i]))
                {
                    return null;
                }
            }

            if (!TrySlice(data, offset, checked((int)header.StringBytes), out var stringData))
            {
                return null;
            }

            if (!TryReadUtf8String(stringData, header.RootPathOffset, header.RootPathLength, out var rootPath))
            {
                return null;
            }

            var topEntries = new DiskUsageEntry[entries.Length];
            for (var i = 0; i < entries.Length; i++)
            {
                var entry = entries[i];
                if (!TryReadUtf8String(stringData, entry.PathOffset, entry.PathLength, out var path))
                {
                    return null;
                }

                topEntries[i] = new DiskUsageEntry
                {
                    Path = path,
                    SizeBytes = entry.SizeBytes,
                    SizeFormatted = FormatBytes(entry.SizeBytes),
                    FileCount = entry.FileCount
                };
            }

            return new DiskUsageReport
            {
                Status = header.Status.ToString(),
                RootPath = rootPath,
                TotalSizeBytes = header.TotalSizeBytes,
                TotalFiles = header.TotalFiles,
                TopEntries = topEntries
            };
        }

        // ====================================================================
        // Disk Usage Functions
        // ====================================================================


        /// <summary>
        /// Get disk usage statistics for a directory.
        /// </summary>
        /// <param name="rootPath">Path to analyze.</param>
        /// <param name="topN">Number of top subdirectories to include in breakdown.</param>
        /// <returns>Disk usage statistics.</returns>
        public static DiskUsageStats GetDiskUsage(string rootPath, uint topN = 10)
        {
            return NativeCore.pcai_get_disk_usage(rootPath, topN);
        }

        /// <summary>
        /// Get disk usage as JSON with detailed breakdown.
        /// </summary>
        /// <param name="rootPath">Path to analyze.</param>
        /// <param name="topN">Number of top subdirectories to include.</param>
        /// <returns>JSON string with usage details, or null on error.</returns>
        public static string? GetDiskUsageJson(string rootPath, uint topN = 10)
        {
            var buffer = NativeCore.pcai_get_disk_usage_json(rootPath, topN);
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        public static DiskUsageReport? GetDiskUsageReport(string rootPath, uint topN = 10)
        {
            var compactBuffer = NativeCore.pcai_get_disk_usage_compact(rootPath, topN);
            try
            {
                if (compactBuffer.IsValid)
                {
                    try
                    {
                        var compact = ParseCompactDiskUsage(compactBuffer);
                        if (compact is not null)
                        {
                            return compact;
                        }
                    }
                    catch
                    {
                    }
                }
            }
            finally
            {
                NativeCore.pcai_free_byte_buffer(ref compactBuffer);
            }

            var json = GetDiskUsageJson(rootPath, topN);
            if (string.IsNullOrWhiteSpace(json))
            {
                return null;
            }

            try
            {
                return JsonSerializer.Deserialize<DiskUsageReport>(json);
            }
            catch
            {
                return null;
            }
        }

        // ====================================================================
        // Process Functions
        // ====================================================================


        /// <summary>
        /// Get system-wide process statistics.
        /// </summary>
        /// <returns>Process statistics including counts and CPU/memory usage.</returns>
        public static ProcessStats GetProcessStats()
        {
            return NativeCore.pcai_get_process_stats();
        }

        /// <summary>
        /// Get top processes as JSON, sorted by memory or CPU.
        /// </summary>
        /// <param name="topN">Number of top processes to return.</param>
        /// <param name="sortBy">"memory" (default) or "cpu".</param>
        /// <returns>JSON string with process list, or null on error.</returns>
        public static string? GetTopProcessesJson(uint topN = 20, string sortBy = "memory")
        {
            var buffer = NativeCore.pcai_get_top_processes_json(topN, sortBy);
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        public static TopProcessesResult? GetTopProcesses(uint topN = 20, string sortBy = "memory")
        {
            var compactBuffer = NativeCore.pcai_get_top_processes_compact(topN, sortBy);
            try
            {
                if (compactBuffer.IsValid)
                {
                    try
                    {
                        var compact = ParseCompactTopProcesses(compactBuffer);
                        if (compact is not null)
                        {
                            return compact;
                        }
                    }
                    catch
                    {
                    }
                }
            }
            finally
            {
                NativeCore.pcai_free_byte_buffer(ref compactBuffer);
            }

            var json = GetTopProcessesJson(topN, sortBy);
            if (string.IsNullOrWhiteSpace(json))
            {
                return null;
            }

            try
            {
                return JsonSerializer.Deserialize<TopProcessesResult>(json);
            }
            catch
            {
                return null;
            }
        }

        // ====================================================================
        // Memory Functions
        // ====================================================================


        /// <summary>
        /// Get system memory statistics.
        /// </summary>
        /// <returns>Memory statistics including RAM and swap usage.</returns>
        public static MemoryStats GetMemoryStats()
        {
            return NativeCore.pcai_get_memory_stats();
        }

        /// <summary>
        /// Get memory statistics as JSON with detailed breakdown.
        /// </summary>
        /// <returns>JSON string with memory details, or null on error.</returns>
        public static string? GetMemoryStatsJson()
        {
            var buffer = NativeCore.pcai_get_memory_stats_json();
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Queries structured hardware metrics natively.
        /// </summary>
        public static PcaiMetrics? GetResourceMetrics()
        {
            // Placeholder - returns null until structured struct is ready
            return null;
        }

        /// <summary>
        /// Queries hardware metrics JSON natively.
        /// </summary>
        public static string? QueryHardwareMetrics()
        {
            var buffer = NativeCore.pcai_query_hardware_metrics();
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Gets network throughput and stats using IPHelper.
        /// </summary>
        public static string? GetNetworkThroughput()
        {
            if (!IsAvailable) return null;
            var ptr = NativeCore.pcai_get_network_throughput_json();
            if (ptr == IntPtr.Zero) return null;
            try
            {
                return Marshal.PtrToStringUTF8(ptr);
            }
            finally
            {
                NativeCore.pcai_free_string(ptr);
            }
        }

        /// <summary>
        /// Gets detailed process history using Psapi.
        /// </summary>
        public static string? GetProcessHistory()
        {
            if (!IsAvailable) return null;
            var ptr = NativeCore.pcai_get_process_history_json();
            if (ptr == IntPtr.Zero) return null;
            try
            {
                return Marshal.PtrToStringUTF8(ptr);
            }
            finally
            {
                NativeCore.pcai_free_string(ptr);
            }
        }

        /// <summary>
        /// Checks if system resources are within safety limits (e.g. 80% load).
        /// </summary>
        public static bool CheckResourceSafety(float gpuLimit = 0.8f)
        {
            return NativeCore.pcai_check_resource_safety(gpuLimit) != 0;
        }

        // ====================================================================
        // Utility Functions
        // ====================================================================


        /// <summary>
        /// Get the performance module version.
        /// </summary>
        /// <returns>Version encoded as 0xMMmmpp (major.minor.patch).</returns>
        public static uint GetVersion()
        {
            return 0x010000;
        }

        /// <summary>
        /// Test if the performance DLL is loaded correctly.
        /// </summary>
        /// <returns>True if the magic number matches.</returns>
        public static bool Test()
        {
            const uint expectedMagic = 0x50434149; // "PCAI"
            return NativeCore.pcai_core_test() == expectedMagic;
        }

        /// <summary>
        /// Format bytes as human-readable string.
        /// </summary>
        public static string FormatBytes(ulong bytes)
        {
            const ulong KB = 1024;
            const ulong MB = KB * 1024;
            const ulong GB = MB * 1024;
            const ulong TB = GB * 1024;

            return bytes switch
            {
                >= TB => $"{bytes / (double)TB:F2} TB",
                >= GB => $"{bytes / (double)GB:F2} GB",
                >= MB => $"{bytes / (double)MB:F2} MB",
                >= KB => $"{bytes / (double)KB:F2} KB",
                _ => $"{bytes} B"
            };
        }
    }
}
