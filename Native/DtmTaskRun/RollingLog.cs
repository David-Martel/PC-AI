using System.Text;

namespace DtmTaskRun;

/// <summary>One log line and the moment it happened.</summary>
/// <param name="At">Timestamp written at the start of the line.</param>
/// <param name="Message">Line text (no newline).</param>
internal readonly record struct LogEntry(DateTimeOffset At, string Message);

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

    // Re-check the size at most this often (bytes written by us since the last check). The wrapper
    // script checks on every write, but it writes a handful of lines; a runner can write thousands,
    // and a stat per line is wasted work (and an antivirus scan trigger) for a limit this coarse.
    private const long RecheckEveryBytes = 16 * 1024;

    private static readonly UTF8Encoding Utf8 = new(false);
    private readonly object _gate = new();
    private readonly Func<DateTimeOffset> _clock;
    private bool _directoryReady;
    private long _bytesSinceCheck = long.MaxValue; // force a check before the first write

    /// <summary>Creates a log at <paramref name="path"/>.</summary>
    public RollingLog(string path, Func<DateTimeOffset>? clock = null)
    {
        Path = path;
        _clock = clock ?? (() => DateTimeOffset.Now);
    }

    /// <summary>Full path of the live log file.</summary>
    public string Path { get; }

    /// <summary>Appends one line, rolling the file over first when it is over the limit.</summary>
    public void WriteLine(string message) => WriteEntries([new LogEntry(_clock(), message)]);

    /// <summary>Current time on this log's clock (so deferred entries can be stamped when they happened).</summary>
    public DateTimeOffset Now() => _clock();

    /// <summary>Appends several entries, each with its own timestamp, in a single open/append/close.</summary>
    public void WriteEntries(IReadOnlyList<LogEntry> entries)
    {
        lock (_gate)
        {
            try
            {
                if (!_directoryReady)
                {
                    Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
                    _directoryReady = true;
                }

                if (_bytesSinceCheck >= RecheckEveryBytes)
                {
                    RollIfNeeded();
                    _bytesSinceCheck = 0;
                }

                var text = new StringBuilder();
                foreach (var entry in entries)
                {
                    text.Append(entry.At.ToString("o", System.Globalization.CultureInfo.InvariantCulture)).Append(' ').Append(entry.Message).Append("\r\n");
                }

                var bytes = Utf8.GetBytes(text.ToString());
                using var fs = new FileStream(Path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);
                fs.Write(bytes, 0, bytes.Length);
                _bytesSinceCheck += bytes.Length;
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
