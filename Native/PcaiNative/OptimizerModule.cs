using System;
using System.Runtime.InteropServices;

namespace PcaiNative
{
    /// <summary>
    /// Memory pressure report returned by native analysis.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryPressureReport
    {
        public PcaiStatus Status;
        public byte PressureLevel;          // 0=low, 1=moderate, 2=high, 3=critical
        public ulong AvailableMB;
        public float CommittedPct;
        public ulong PoolNonpagedMB;
        public ulong PagesPerSec;
        public uint TopConsumerCount;       // Processes using >500MB
        public uint HandleLeakCount;        // Processes with >100K handles
        public uint OrphanTerminalCount;    // cmd/conhost without parent
        public ulong ElapsedMs;

        public bool IsSuccess => Status == PcaiStatus.Success;

        public string PressureLevelName => PressureLevel switch
        {
            0 => "Low",
            1 => "Moderate",
            2 => "High",
            3 => "Critical",
            _ => "Unknown"
        };
    }

    /// <summary>
    /// P/Invoke declarations and managed wrappers for the PCAI memory optimizer.
    /// Follows the CSharp_RustDLL paradigm: Rust FFI → C# P/Invoke → PowerShell.
    /// </summary>
    public static class OptimizerModule
    {
        private static readonly Lazy<bool> _isAvailable = new(() =>
        {
            try { return NativeCore.pcai_core_test() == 0x50434149; }
            catch { return false; }
        });

        public static bool IsAvailable => _isAvailable.Value;

        // ====================================================================
        // Memory Pressure Analysis
        // ====================================================================

        /// <summary>
        /// Analyze current memory pressure including paging, pool memory, handle leaks,
        /// and orphaned terminals.
        /// </summary>
        public static MemoryPressureReport AnalyzeMemoryPressure()
        {
            return NativeCore.pcai_analyze_memory_pressure();
        }

        /// <summary>
        /// Get memory pressure analysis as detailed JSON.
        /// </summary>
        public static string? GetMemoryPressureJson()
        {
            var buffer = NativeCore.pcai_get_memory_pressure_json();
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        // ====================================================================
        // Process Category Analysis
        // ====================================================================

        /// <summary>
        /// Classify all running processes into LLM-relevant categories
        /// (llm_agents, browsers, terminals, build_tools, system_services).
        /// Returns JSON with per-category totals.
        /// </summary>
        public static string? GetProcessCategoriesJson()
        {
            var buffer = NativeCore.pcai_get_process_categories_json();
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        // ====================================================================
        // Optimization Recommendations
        // ====================================================================

        /// <summary>
        /// Generate prioritized optimization recommendations based on current system state.
        /// Returns JSON array of recommendations with priority, category, estimated savings,
        /// and whether the action is safe to automate.
        /// </summary>
        public static string? GetOptimizationRecommendationsJson()
        {
            var buffer = NativeCore.pcai_get_optimization_recommendations_json();
            try
            {
                return buffer.ToManagedString();
            }
            finally
            {
                NativeCore.pcai_free_string_buffer(ref buffer);
            }
        }

        // ====================================================================
        // Convenience Methods
        // ====================================================================

        /// <summary>
        /// Quick check: is the system under critical memory pressure?
        /// </summary>
        public static bool IsCriticalPressure()
        {
            var report = AnalyzeMemoryPressure();
            return report.IsSuccess && report.PressureLevel >= 3;
        }

        /// <summary>
        /// Quick check: are there handle leaks detected?
        /// </summary>
        public static bool HasHandleLeaks()
        {
            var report = AnalyzeMemoryPressure();
            return report.IsSuccess && report.HandleLeakCount > 0;
        }

        /// <summary>
        /// Get the optimizer module version.
        /// </summary>
        public static uint GetVersion()
        {
            return 0x010000;
        }
    }
}
