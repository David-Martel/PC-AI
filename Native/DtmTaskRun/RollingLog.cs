using System.Text;

namespace DtmTaskRun;

/// <summary>
/// Append-only task log with the same rollover rule the scheduler agent used in
/// <c>Update-BwArchive.ps1</c>: when the file is larger than 256 KB (262144 bytes) it is
/// moved to <c>&lt;log&gt;.1</c>, replacing any previous <c>.1</c>, and a fresh file starts.
/// Lines look like <c>2026-10-20T16:50:09.1234567-04:00 message</c>.
/// </summary>
/// <remarks>
/// Every write opens, appends and closes with <c>FileShare.ReadWrite | Delete</c>. The task's
/// own script may append to the very same file (the Bitwarden wrapper does), so no handle is
/// held open between lines and neither writer can lock the other out. All failures are
/// swallowed: logging must never change a task's result.
/// </remarks>
internal sealed class RollingLog
{
    /// <summary>Rollover threshold in bytes, identical to the wrapper scripts.</summary>
    public const long MaxBytes = 262_144;

    private static readonly UTF8Encoding Utf8 = new(false);
    private readonly object _gate = new();
    private readonly Func<DateTimeOffset> _clock;

    /// <summary>Creates a log at <paramref name="path"/>.</summary>
    public RollingLog(string path, Func<DateTimeOffset>? clock = null)
    {
        Path = path;
        _clock = clock ?? (() => DateTimeOffset.Now);
    }

    /// <summary>Full path of the live log file.</summary>
    public string Path { get; }

    /// <summary>Appends one line, rolling the file over first when it is over the limit.</summary>
    public void WriteLine(string message)
    {
        lock (_gate)
        {
            try
            {
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
                RollIfNeeded();
                var line = _clock().ToString("o", System.Globalization.CultureInfo.InvariantCulture) + " " + message + "\r\n";
                var bytes = Utf8.GetBytes(line);
                using var fs = new FileStream(Path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);
                fs.Write(bytes, 0, bytes.Length);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Logging is best effort; a locked or full disk must not fail the task.
            }
        }
    }

    private void RollIfNeeded()
    {
        var info = new FileInfo(Path);
        if (info.Exists && info.Length > MaxBytes)
        {
            File.Move(Path, Path + ".1", overwrite: true);
        }
    }
}
