using System.Diagnostics;
using System.Globalization;

namespace DtmTaskRun;

/// <summary>Result of an agent-bus notification attempt.</summary>
internal enum NotifyResult
{
    /// <summary>The finding was posted.</summary>
    Sent,

    /// <summary>A finding for this task was already posted within the rate-limit window.</summary>
    RateLimited,

    /// <summary>A recent attempt failed; backing off so a dead bus cannot slow every run.</summary>
    BackingOff,

    /// <summary>The agent-bus executable is not installed here.</summary>
    ExeMissing,

    /// <summary>The CLI failed, or did not answer within the timeout.</summary>
    Failed,
}

/// <summary>Posts a task-failure finding to the agent bus.</summary>
internal interface IBusNotifier
{
    /// <summary>Attempts to notify; never throws.</summary>
    NotifyResult TryNotify(string taskName, string fileStem, int exitCode, bool timedOut, string statusPath, string logPath, DateTimeOffset nowUtc);
}

/// <summary>
/// Runs <c>agent-bus.exe send ...</c> with a hard timeout and a one-message-per-task-per-hour limit.
/// </summary>
/// <remarks>
/// The message carries only the task id, exit code and file paths. It never includes stderr or the
/// command line: redaction is best effort and the bus is the one channel that leaves the machine.
/// The exit code is already decided and the status file already written when this runs, so a slow or
/// dead bus can delay the launcher's exit by at most <see cref="Timeout"/> but can never change it.
/// </remarks>
internal sealed class BusNotifier : IBusNotifier
{
    /// <summary>Hard cap on how long the CLI may run.</summary>
    public static readonly TimeSpan Timeout = TimeSpan.FromSeconds(5);

    /// <summary>One message per task per this interval.</summary>
    public static readonly TimeSpan MinInterval = TimeSpan.FromHours(1);

    /// <summary>After a failed attempt, do not retry for this long.</summary>
    public static readonly TimeSpan FailureBackoff = TimeSpan.FromMinutes(10);

    private readonly string _stateDir;
    private readonly string _busExe;
    private readonly string _host;
    private readonly Func<IReadOnlyList<string>, bool>? _sender;

    /// <summary>Creates a notifier. <paramref name="stateDir"/> holds the per-task timestamp files.</summary>
    /// <param name="stateDir">Directory for the <c>.notified</c> / <c>.notify-failed</c> timestamp files.</param>
    /// <param name="busExe">Path of <c>agent-bus.exe</c>.</param>
    /// <param name="host">Lower-case host name.</param>
    /// <param name="sender">Test seam: receives the CLI arguments instead of spawning <paramref name="busExe"/>.</param>
    public BusNotifier(string stateDir, string busExe, string host, Func<IReadOnlyList<string>, bool>? sender = null)
    {
        _stateDir = stateDir;
        _busExe = busExe;
        _host = host;
        _sender = sender;
    }

    /// <summary>The exact <c>agent-bus.exe</c> argument vector for a failure finding.</summary>
    internal static IReadOnlyList<string> BuildArgs(string host, string body) =>
    [
        "send", "--from-agent", "dtm-task-run", "--to-agent", "all", "--topic", "task-failures",
        "--schema", "finding", "--thread-id", host + "-tasks", "--tag", "host:" + host,
        "--tag", "repo:PC_AI", "--body", body,
    ];

    /// <summary>Builds the finding body (kept under the bus's 2000 character message limit).</summary>
    public static string BuildBody(string taskName, int exitCode, bool timedOut, string host, string statusPath, string logPath)
    {
        var hex = "0x" + unchecked((uint)exitCode).ToString("X", CultureInfo.InvariantCulture);
        var reason = timedOut ? $"timed out (killed, exit={exitCode} {hex})" : $"failed exit={exitCode} ({hex})";
        return $"FINDING: task {taskName} {reason} on {host}. status={statusPath} log={logPath} "
            + "(see Application log source dtm-task-run). SEVERITY: medium";
    }

    /// <inheritdoc />
    public NotifyResult TryNotify(string taskName, string fileStem, int exitCode, bool timedOut, string statusPath, string logPath, DateTimeOffset nowUtc)
    {
        var okStamp = Path.Combine(_stateDir, fileStem + ".notified");
        var failStamp = Path.Combine(_stateDir, fileStem + ".notify-failed");

        if (IsWithin(okStamp, nowUtc, MinInterval))
        {
            return NotifyResult.RateLimited;
        }

        if (IsWithin(failStamp, nowUtc, FailureBackoff))
        {
            return NotifyResult.BackingOff;
        }

        if (_sender is null && !File.Exists(_busExe))
        {
            return NotifyResult.ExeMissing;
        }

        var args = BuildArgs(_host, BuildBody(taskName, exitCode, timedOut, _host, statusPath, logPath));
        var ok = _sender is not null ? _sender(args) : Send(args);
        Touch(ok ? okStamp : failStamp, nowUtc);
        if (ok)
        {
            TryDelete(failStamp);
        }

        return ok ? NotifyResult.Sent : NotifyResult.Failed;
    }

    private bool Send(IReadOnlyList<string> args)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _busExe,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
        };
        foreach (var a in args)
        {
            psi.ArgumentList.Add(a);
        }

        try
        {
            using var p = Process.Start(psi);
            if (p is null)
            {
                return false;
            }

            p.StandardInput.Close();
            // Drain so a chatty CLI cannot block on a full pipe while we wait for it.
            var drain = Task.WhenAll(p.StandardOutput.ReadToEndAsync(), p.StandardError.ReadToEndAsync());
            if (!p.WaitForExit(Timeout))
            {
                p.Kill(entireProcessTree: true);
                return false;
            }

            _ = drain.Wait(TimeSpan.FromSeconds(1));
            return p.ExitCode == 0;
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
        {
            return false;
        }
    }

    private static bool IsWithin(string stamp, DateTimeOffset nowUtc, TimeSpan window)
    {
        try
        {
            if (!File.Exists(stamp))
            {
                return false;
            }

            var written = DateTimeOffset.Parse(File.ReadAllText(stamp).Trim(), CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind);
            return nowUtc - written < window && nowUtc >= written;
        }
        catch (Exception ex) when (ex is IOException or FormatException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static void Touch(string stamp, DateTimeOffset nowUtc)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(stamp)!);
            File.WriteAllText(stamp, nowUtc.ToString("o", CultureInfo.InvariantCulture));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Without a stamp the worst case is one extra message.
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // harmless
        }
    }
}
