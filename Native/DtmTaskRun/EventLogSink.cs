using System.Runtime.InteropServices;
using System.Security.Principal;
using Microsoft.Win32;

namespace DtmTaskRun;

/// <summary>Where failure events go. Abstracted so tests do not write to the real Application log.</summary>
internal interface IEventSink
{
    /// <summary>Records a task failure; returns false when the entry could not be written.</summary>
    bool WriteFailure(int eventId, string message);
}

/// <summary>
/// Writes to the Windows Application log under source <c>dtm-task-run</c> through the Win32
/// <c>ReportEvent</c> API, so there is no NuGet dependency and no <c>EventLog.SourceExists</c> call
/// (which throws <see cref="System.Security.SecurityException"/> for a limited token because it
/// enumerates the Security log).
/// </summary>
/// <remarks>
/// Event ids: 1001 task failed (nonzero exit), 1002 task timed out, 1003 launcher could not start the
/// child. Registering the source writes HKLM, so it needs an elevated token; run
/// <c>dtm-task-run.exe --install-eventlog-source</c> once. If the source is missing, an unelevated run
/// still lands the event under the same source name, but Event Viewer prefixes it with
/// "The description for Event ID ... cannot be found".
/// </remarks>
internal sealed class EventLogSink : IEventSink
{
    /// <summary>Event source name.</summary>
    public const string Source = "dtm-task-run";

    /// <summary>Failure event id for a nonzero exit.</summary>
    public const int IdFailed = 1001;

    /// <summary>Failure event id for a launcher timeout.</summary>
    public const int IdTimedOut = 1002;

    /// <summary>Failure event id for a spawn error.</summary>
    public const int IdSpawnFailed = 1003;

    private const string KeyPath = @"SYSTEM\CurrentControlSet\Services\EventLog\Application\" + Source;
    private const string MessageFile = @"%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\EventLogMessages.dll";
    private const ushort EventLogErrorType = 1;
    private const int MaxChars = 30_000;

    /// <summary>True when the source key already exists (readable by any user).</summary>
    public static bool IsSourceRegistered()
    {
        try
        {
            using var key = Registry.LocalMachine.OpenSubKey(KeyPath);
            return key is not null;
        }
        catch (Exception ex) when (ex is System.Security.SecurityException or UnauthorizedAccessException or IOException)
        {
            return false;
        }
    }

    /// <summary>
    /// Registers the source. Returns true if it exists afterwards; false (quietly) when the
    /// token is not elevated.
    /// </summary>
    public static bool TryRegisterSource()
    {
        if (IsSourceRegistered())
        {
            return true;
        }

        try
        {
            using var identity = WindowsIdentity.GetCurrent();
            if (!new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator))
            {
                return false;
            }

            using var key = Registry.LocalMachine.CreateSubKey(KeyPath);
            key.SetValue("EventMessageFile", MessageFile, RegistryValueKind.ExpandString);
            key.SetValue("TypesSupported", 7, RegistryValueKind.DWord);
            return true;
        }
        catch (Exception ex) when (ex is System.Security.SecurityException or UnauthorizedAccessException or IOException)
        {
            return false;
        }
    }

    /// <inheritdoc />
    public bool WriteFailure(int eventId, string message)
    {
        // Elevated runs create the source on first use; limited runs skip silently.
        _ = TryRegisterSource();

        var handle = RegisterEventSourceW(null, Source);
        if (handle == IntPtr.Zero)
        {
            return false;
        }

        try
        {
            var text = message.Length > MaxChars ? message[..MaxChars] : message;
            return ReportEventW(handle, EventLogErrorType, 0, (uint)eventId, IntPtr.Zero, 1, 0, [text], IntPtr.Zero);
        }
        finally
        {
            _ = DeregisterEventSource(handle);
        }
    }

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr RegisterEventSourceW(string? serverName, string sourceName);

    [DllImport("advapi32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool DeregisterEventSource(IntPtr handle);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool ReportEventW(
        IntPtr handle,
        ushort type,
        ushort category,
        uint eventId,
        IntPtr userSid,
        ushort numStrings,
        uint dataSize,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPWStr)] string[] strings,
        IntPtr rawData);
}
