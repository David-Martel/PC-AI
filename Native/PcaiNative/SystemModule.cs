using System;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace PcaiNative
{
    /// <summary>
    /// PATH analysis statistics returned by native functions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PathAnalysisStats
    {
        public PcaiStatus Status;
        public uint TotalEntries;
        public uint UniqueEntries;
        public uint DuplicateCount;
        public uint NonExistentCount;
        public uint EmptyCount;
        public uint TrailingSlashCount;
        public uint CrossDuplicateCount;
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;

        /// <summary>
        /// Health status based on analysis results.
        /// </summary>
        public string HealthStatus
        {
            get
            {
                if (DuplicateCount == 0 && NonExistentCount == 0 && EmptyCount == 0)
                    return "Healthy";
                if (CrossDuplicateCount > 0 || NonExistentCount > 5)
                    return "NeedsAttention";
                return "MinorIssues";
            }
        }
    }

    /// <summary>
    /// Log search statistics returned by native functions.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LogSearchStats
    {
        public PcaiStatus Status;
        public ulong FilesSearched;
        public ulong FilesWithMatches;
        public ulong TotalMatches;
        public ulong BytesSearched;
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;
    }

    /// <summary>
    /// P/Invoke declarations for pcai_core_lib.dll (consolidated from system module).
    /// Provides PATH environment analysis and log file searching.
    /// </summary>
    public static class SystemModule
    {
        private const uint ExpectedMagic = 0x50434149; // "PCAI"

        // Thread-safe lazy initialization for availability check
        private static readonly Lazy<bool> _isAvailable = new(() =>
        {
            try
            {
                return NativeCore.pcai_core_test() == ExpectedMagic;
            }
            catch
            {
                return false;
            }
        });

        /// <summary>
        /// Gets whether the native system library is available and functional.
        /// </summary>
        public static bool IsAvailable => _isAvailable.Value;

        // ====================================================================
        // PATH Analysis Functions
        // ====================================================================


        /// <summary>
        /// Analyze the PATH environment variable for issues.
        /// </summary>
        /// <returns>PATH analysis statistics, or error status if unavailable.</returns>
        public static PathAnalysisStats AnalyzePath()
        {
            if (!IsAvailable)
                return new PathAnalysisStats { Status = PcaiStatus.NotImplemented };

            return NativeCore.pcai_analyze_path();
        }

        /// <summary>
        /// Analyze PATH and return detailed JSON report.
        /// </summary>
        /// <returns>JSON string with comprehensive PATH analysis, or null on error.</returns>
        public static string? AnalyzePathJson()
        {
            if (!IsAvailable)
                return null;

            var buffer = NativeCore.pcai_analyze_path_json();
            try
            {
                var json = buffer.ToManagedString();
                if (string.IsNullOrEmpty(json)) return null;
                return json;
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        // ====================================================================
        // Log Search Functions
        // ====================================================================


        /// <summary>
        /// Search log files for a pattern.
        /// </summary>
        /// <param name="rootPath">Directory to search in.</param>
        /// <param name="pattern">Regex pattern to search for.</param>
        /// <param name="filePattern">Glob pattern for log files (e.g., "*.log"), or null for default.</param>
        /// <param name="caseSensitive">Whether search is case-sensitive.</param>
        /// <param name="contextLines">Number of context lines before/after matches.</param>
        /// <param name="maxMatches">Maximum number of matches to return.</param>
        /// <returns>Log search statistics, or error status if unavailable.</returns>
        public static LogSearchStats SearchLogs(
            string rootPath,
            string pattern,
            string? filePattern = "*.log",
            bool caseSensitive = false,
            uint contextLines = 2,
            uint maxMatches = 1000)
        {
            if (!IsAvailable)
                return new LogSearchStats { Status = PcaiStatus.NotImplemented };

            if (string.IsNullOrWhiteSpace(rootPath) || string.IsNullOrWhiteSpace(pattern))
                return new LogSearchStats { Status = PcaiStatus.InvalidArgument };

            return NativeCore.pcai_search_logs(rootPath, pattern, filePattern, caseSensitive, contextLines, maxMatches);
        }

        /// <summary>
        /// Search log files and return JSON results.
        /// </summary>
        /// <param name="rootPath">Directory to search in.</param>
        /// <param name="pattern">Regex pattern to search for.</param>
        /// <param name="filePattern">Glob pattern for log files (e.g., "*.log"), or null for default.</param>
        /// <param name="caseSensitive">Whether search is case-sensitive.</param>
        /// <param name="contextLines">Number of context lines before/after matches.</param>
        /// <param name="maxMatches">Maximum number of matches to return.</param>
        /// <returns>JSON string with search results, or null on error.</returns>
        public static string? SearchLogsJson(
            string rootPath,
            string pattern,
            string? filePattern = "*.log",
            bool caseSensitive = false,
            uint contextLines = 2,
            uint maxMatches = 1000)
        {
            if (!IsAvailable)
                return null;

            if (string.IsNullOrWhiteSpace(rootPath) || string.IsNullOrWhiteSpace(pattern))
                return null;

            var buffer = NativeCore.pcai_search_logs_json(rootPath, pattern, filePattern, caseSensitive, contextLines, maxMatches);
            try
            {
                var json = buffer.ToManagedString();
                if (string.IsNullOrEmpty(json)) return null;
                return json;
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        // ====================================================================
        // Utility Functions
        // ====================================================================

        /// <summary>
        /// Normalizes a path string for cross-platform compatibility natively.
        /// </summary>
        public static string? NormalizePath(string path, uint style = 0)
        {
            if (!IsAvailable) return null;

            var buffer = NativeCore.pcai_normalize_path(path, style);
            try
            {
                var json = buffer.ToManagedString();
                if (string.IsNullOrEmpty(json)) return null;
                return json;
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        /// <summary>
        /// Get the system module version.
        /// </summary>
        /// <returns>Version encoded as 0xMMmmpp (major.minor.patch).</returns>
        public static uint GetVersion()
        {
            return 0x010000; // Hardcoded or use PcaiCore.Version
        }

        /// <summary>
        /// Test if the system DLL is loaded correctly.
        /// </summary>
        /// <returns>True if the magic number matches.</returns>
        public static bool Test()
        {
            if (!IsAvailable) return false;
            return NativeCore.pcai_core_test() == 0x50434149; // "PCAI"
        }

        /// <summary>
        /// Gets high-fidelity system telemetry as JSON using native core.
        /// </summary>
        public static string? GetSystemTelemetryJson()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_system_telemetry_json();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets WSL/VMM health status using direct native socket interrogation.
        /// </summary>
        public static string? GetVmmHealthJson()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_vmm_health_json();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Queries comprehensive system information natively.
        /// </summary>
        public static string? QuerySystemInfo()
        {
            if (!IsAvailable) return null;

            var buffer = NativeCore.pcai_query_system_info();
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
        /// <summary>
        /// Execute a command and return output as JSON.
        /// </summary>
        public static string ExecuteCommand(string command, string args, int timeoutMs = 30000)
        {
            try
            {
                var startInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = command,
                    Arguments = args,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = new System.Diagnostics.Process { StartInfo = startInfo };
                var output = new System.Text.StringBuilder();
                var error = new System.Text.StringBuilder();

                process.OutputDataReceived += (s, e) => { if (e.Data != null) output.AppendLine(e.Data); };
                process.ErrorDataReceived += (s, e) => { if (e.Data != null) error.AppendLine(e.Data); };

                var sw = System.Diagnostics.Stopwatch.StartNew();
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                bool completed = process.WaitForExit(timeoutMs);
                if (!completed)
                {
                    process.Kill();
                    return JsonSerializer.Serialize(new
                    {
                        status = "Timeout",
                        elapsed_ms = sw.ElapsedMilliseconds,
                        output = output.ToString(),
                        error = error.ToString()
                    });
                }

                return JsonSerializer.Serialize(new
                {
                    status = process.ExitCode == 0 ? "Success" : "Error",
                    exit_code = process.ExitCode,
                    elapsed_ms = sw.ElapsedMilliseconds,
                    output = output.ToString(),
                    error = error.ToString()
                });
            }
            catch (Exception ex)
            {
                return JsonSerializer.Serialize(new
                {
                    status = "Exception",
                    error = ex.Message
                });
            }
        }
    }
}
