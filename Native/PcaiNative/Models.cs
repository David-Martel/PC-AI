using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace PcaiNative;

/// <summary>
/// Status codes returned by PCAI native operations.
/// Must match the Rust PcaiStatus enum exactly.
/// </summary>
public enum PcaiStatus : uint
{
    Success = 0,
    InvalidArgument = 1,
    NullPointer = 2,
    InvalidUtf8 = 3,
    PathNotFound = 4,
    PermissionDenied = 5,
    IoError = 6,
    Cancelled = 7,
    Timeout = 8,
    InternalError = 9,
    NotImplemented = 10,
    OutOfMemory = 11,
    JsonError = 12,
    Unknown = 255
}

/// <summary>
/// Controls the amount of detail in the hardware report.
/// </summary>
public enum DiagnosticVerbosity
{
    Basic = 0,    // Only errors and critical summaries
    Normal = 1,   // Standard detailed info
    Full = 2      // Everything, including high-frequency metrics
}

/// <summary>
/// Generic result structure for operations that process items.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct PcaiResult
{
    public PcaiStatus Status;
    public ulong Processed;
    public ulong Matched;
    public ulong Errors;
    public ulong ElapsedMs;

    public readonly bool IsSuccess => Status == PcaiStatus.Success;
}

/// <summary>
/// String buffer for receiving string data from native code.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct PcaiStringBuffer
{
    public PcaiStatus Status;
    public IntPtr Data;
    public UIntPtr Length;

    public readonly bool IsValid => Status == PcaiStatus.Success && Data != IntPtr.Zero;

    /// <summary>
    /// Converts the native string buffer to a managed string.
    /// Uses UTF-8 encoding to match Rust string representation.
    /// </summary>
    public readonly string? ToManagedString()
    {
        if (!IsValid) return null;
        return Marshal.PtrToStringUTF8(Data, (int)(uint)Length);
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct PcaiByteBuffer
{
    public PcaiStatus Status;
    public IntPtr Data;
    public UIntPtr Length;

    public readonly bool IsValid => Status == PcaiStatus.Success && Data != IntPtr.Zero;

    public readonly byte[]? ToManagedBytes()
    {
        if (!IsValid) return null;

        var length = checked((int)(ulong)Length);
        if (length == 0) {
            return Array.Empty<byte>();
        }

        var bytes = new byte[length];
        Marshal.Copy(Data, bytes, 0, length);
        return bytes;
    }
}

/// <summary>
/// Diagnostic information about the native library.
/// </summary>
public sealed class NativeDiagnostics
{
    [JsonPropertyName("isAvailable")]
    public bool IsAvailable { get; init; }

    [JsonPropertyName("version")]
    public string Version { get; init; } = "";

    [JsonPropertyName("cpuCount")]
    public uint CpuCount { get; init; }

    [JsonPropertyName("platform")]
    public string Platform { get; init; } = "";

    [JsonPropertyName("dotNetVersion")]
    public string DotNetVersion { get; init; } = "";
}

/// <summary>
/// Information about a Windows Configuration Manager problem code.
/// </summary>
public sealed class CmProblemInfo
{
    [JsonPropertyName("code")]
    public uint Code { get; init; }

    [JsonPropertyName("short_description")]
    public string Description { get; init; } = "";

    [JsonPropertyName("help_summary")]
    public string Summary { get; init; } = "";

    [JsonPropertyName("help_url")]
    public string HelpUrl { get; init; } = "";
}

/// <summary>
/// Structured system resource metrics.
/// </summary>
public sealed class PcaiMetrics
{
    [JsonPropertyName("cpu_usage_perc")]
    public float CpuUsage { get; init; }

    [JsonPropertyName("memory_usage_bytes")]
    public ulong MemoryUsage { get; init; }

    [JsonPropertyName("total_memory_bytes")]
    public ulong TotalMemory { get; init; }

    [JsonPropertyName("gpu_usage_perc")]
    public float GpuUsage { get; init; }

    [JsonPropertyName("uptime_seconds")]
    public ulong Uptime { get; init; }
}
