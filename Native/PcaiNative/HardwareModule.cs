using System;
using System.Runtime.InteropServices;

namespace PcaiNative
{
    /// <summary>
    /// Specialized module for hardware-related native functions.
    /// Provides PnP interrogation, disk health, and event log sampling.
    /// </summary>
    public static class HardwareModule
    {
        /// <summary>
        /// Gets whether the native hardware library functions are available.
        /// </summary>
        public static bool IsAvailable => SystemModule.IsAvailable;

        /// <summary>
        /// Enumerates PnP devices based on an optional class filter.
        /// </summary>
        /// <param name="classFilter">Device class to filter for (e.g. "USB", "DiskDrive") or null.</param>
        /// <returns>
        /// JSON string containing a list of PnpDeviceDetail objects. Each object includes the following
        /// driver fields sourced from the Windows device registry:
        /// <list type="bullet">
        ///   <item><description><c>driver_version</c> — installed driver version string (e.g. "10.0.26100.3775"), or <c>null</c> if not present.</description></item>
        ///   <item><description><c>driver_date</c> — driver release date in ISO 8601 format (e.g. "2006-06-21"), or <c>null</c> if not present.</description></item>
        ///   <item><description><c>driver_provider</c> — publisher of the driver (e.g. "Microsoft"), or <c>null</c> if not present.</description></item>
        /// </list>
        /// Returns <c>null</c> if the native library is unavailable or the call fails.
        /// </returns>
        public static string? GetPnpDevicesJson(string? classFilter = null)
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_pnp_devices_json(classFilter);
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets detailed information about a PnP problem code.
        /// </summary>
        public static string? GetPnpProblemInfo(uint code)
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_pnp_problem_info(code);
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Queries native disk health and SMART status.
        /// </summary>
        public static string? GetDiskHealthJson()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_disk_health_json();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Samples hardware-related events from the Windows System Event Log.
        /// </summary>
        public static string? SampleHardwareEventsJson(uint days = 3, uint maxEvents = 50)
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_sample_hardware_events_json(days, maxEvents);
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets GPU utilization for a given device index natively.
        /// </summary>
        public static string? GetGpuUtilizationJson(uint deviceIndex)
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_gpu_utilization_json(deviceIndex);
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets CUDA driver version natively.
        /// </summary>
        public static string? GetCudaDriverVersionJson()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_cuda_driver_version_json();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets the total number of GPUs natively.
        /// </summary>
        public static int GetGpuCount()
        {
            if (!IsAvailable) return 0;
            return NativeCore.pcai_gpu_count();
        }

        /// <summary>
        /// Gets detailed inventory information of all GPUs natively.
        /// </summary>
        public static string? GetGpuInfoJson()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_gpu_info_json();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets the GPU driver version natively.
        /// </summary>
        public static string? GetDriverVersion()
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_driver_version();
            return ptr.ToManagedString();
        }

        /// <summary>
        /// Gets Process Lasso snapshot as JSON.
        /// </summary>
        public static string? GetProcessLassoSnapshotJson(string? configPath = null, string? logPath = null, uint lookbackMinutes = 60)
        {
            if (!IsAvailable) return null;
            using var ptr = NativeCore.pcai_get_process_lasso_snapshot_json(configPath, logPath, lookbackMinutes);
            return ptr.ToManagedString();
        }
    }
}
