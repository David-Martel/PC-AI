using System.Runtime.InteropServices;

namespace PcaiNative;

/// <summary>
/// Managed bridge for Process Lasso diagnostics backed by the native core library.
/// </summary>
public static class ProcessLassoModule
{
    /// <summary>
    /// Returns a JSON snapshot of the live Process Lasso configuration and recent log activity.
    /// </summary>
    public static string? GetSnapshotJson(
        string? configPath = null,
        string? logPath = null,
        uint lookbackMinutes = 60)
    {
        if (!PcaiCore.IsAvailable)
        {
            return null;
        }

        var ptr = NativeCore.pcai_get_process_lasso_snapshot_json(configPath, logPath, lookbackMinutes);
        if (ptr == IntPtr.Zero)
        {
            return null;
        }

        try
        {
            return Marshal.PtrToStringUTF8(ptr);
        }
        finally
        {
            NativeCore.pcai_free_string(ptr);
        }
    }
}
