using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace DtmTaskRun;

/// <summary>Everything the runner touches outside its own logic, so tests can redirect it.</summary>
internal sealed class RunnerContext
{
    /// <summary>Root for <c>logs\</c>, <c>status\</c> and <c>state\</c>.</summary>
    public required string BaseDir { get; init; }

    /// <summary>Lower-case machine name, used in JSON and bus messages.</summary>
    public required string Host { get; init; }

    /// <summary>UTC clock.</summary>
    public Func<DateTimeOffset> UtcNow { get; init; } = () => DateTimeOffset.UtcNow;

    /// <summary>Event-log destination.</summary>
    public required IEventSink Events { get; init; }

    /// <summary>Agent-bus destination (only used with <c>--notify-bus</c>).</summary>
    public required IBusNotifier Bus { get; init; }

    /// <summary>
    /// Production wiring. <c>DTM_TASKS_DIR</c> and <c>DTM_AGENT_BUS_EXE</c> override the defaults
    /// (<c>%LOCALAPPDATA%\dtm-tasks</c> and <c>%USERPROFILE%\bin\agent-bus.exe</c>); tests and dry runs use them.
    /// </summary>
    public static RunnerContext CreateDefault()
    {
        var baseDir = Environment.GetEnvironmentVariable("DTM_TASKS_DIR");
        if (string.IsNullOrWhiteSpace(baseDir))
        {
            baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "dtm-tasks");
        }

        var busExe = Environment.GetEnvironmentVariable("DTM_AGENT_BUS_EXE");
        if (string.IsNullOrWhiteSpace(busExe))
        {
            busExe = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "bin", "agent-bus.exe");
        }

        var host = Environment.MachineName.ToLowerInvariant();
        return new RunnerContext
        {
            BaseDir = baseDir,
            Host = host,
            Events = new EventLogSink(),
            Bus = new BusNotifier(Path.Combine(baseDir, "state"), busExe, host),
        };
    }
}

/// <summary>
/// Runs one child process hidden, captures its output, enforces an optional timeout and returns the
/// child's own exit code so Task Scheduler's LastTaskResult tells the truth.
/// </summary>
internal sealed class TaskRunner
{
    /// <summary>ERROR_TIMEOUT: returned when the launcher's <c>--timeout</c> killed the child.</summary>
    public const int TimeoutExitCode = 1460;

    /// <summary>ERROR_INVALID_PARAMETER: bad launcher command line.</summary>
    public const int UsageExitCode = 87;

    /// <summary>ERROR_INTERNAL_ERROR: the launcher itself failed after starting.</summary>
    public const int InternalErrorExitCode = 1359;

    /// <summary>Number of stderr lines kept for the status file and event log.</summary>
    public const int TailLines = 20;

    private static readonly TimeSpan PumpDrainWait = TimeSpan.FromSeconds(3);

    private readonly RunnerContext _ctx;

    /// <summary>Creates a runner.</summary>
    public TaskRunner(RunnerContext ctx) => _ctx = ctx;

    /// <summary>
    /// The start info for the child: no window, no shell, all three std streams redirected. Exposed
    /// so a test can assert the hidden-launch flags without spawning anything.
    /// </summary>
    internal static ProcessStartInfo BuildStartInfo(RunOptions o)
    {
        var psi = new ProcessStartInfo
        {
            FileName = o.Exe,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var a in o.Args)
        {
            psi.ArgumentList.Add(a);
        }

        psi.Environment["DTM_TASK_NAME"] = o.Name;
        return psi;
    }

    /// <summary>Runs the child and returns the exit code the launcher should exit with.</summary>
    public int Run(RunOptions o)
    {
        var stem = TaskNames.ToFileStem(o.Name);
        var logPath = Path.Combine(_ctx.BaseDir, "logs", stem + ".log");
        var statusPath = Path.Combine(_ctx.BaseDir, "status", stem + ".json");
        var log = new RollingLog(logPath);

        var cmdline = Redactor.RedactText(Redactor.JoinForDisplay(o.Exe, Redactor.RedactArgs(o.Args)));
        var start = _ctx.UtcNow();
        var status = new TaskStatus
        {
            Task = o.Name,
            State = "running",
            Pid = Environment.ProcessId,
            StartUtc = Iso(start),
            Cmdline = cmdline,
            Host = _ctx.Host,
            LogPath = logPath,
        };

        using var rec = new RunRecorder(
            log,
            $"dtm-task-run: start task={o.Name} pid={status.Pid} timeout={(o.TimeoutSeconds > 0 ? o.TimeoutSeconds + "s" : "none")} cmd={cmdline}",
            () => _ = StatusFile.TryWrite(statusPath, status));

        var watch = Stopwatch.StartNew();
        var tail = new TailBuffer(TailLines);
        var timedOut = false;
        long spawnMs = 0;
        int code;

        Process? child = null;
        JobObject? job = null;
        try
        {
            child = Process.Start(BuildStartInfo(o));
            if (child is null)
            {
                throw new InvalidOperationException("Process.Start returned no process");
            }

            job = JobObject.TryCreate();
            _ = job?.TryAssign(child);
            child.StandardInput.Close();
            spawnMs = watch.ElapsedMilliseconds;

            rec.Arm();
            var outPump = StartPump(child.StandardOutput, "out", rec, null);
            var errPump = StartPump(child.StandardError, "err", rec, tail);

            if (o.TimeoutSeconds > 0)
            {
                if (!child.WaitForExit(checked(o.TimeoutSeconds * 1000)))
                {
                    timedOut = true;
                    job?.Terminate(TimeoutExitCode);
                    TryKillTree(child);
                    _ = child.WaitForExit(5000);
                }
            }
            else
            {
                child.WaitForExit();
            }

            // A detached descendant can keep the pipes open after the child exits; do not wait forever.
            var deadline = DateTime.UtcNow + PumpDrainWait;
            foreach (var t in new[] { outPump, errPump })
            {
                var left = deadline - DateTime.UtcNow;
                if (left > TimeSpan.Zero)
                {
                    _ = t.Join(left);
                }
            }

            if (outPump.IsAlive || errPump.IsAlive)
            {
                rec.Write("dtm-task-run: output streams still open after exit (detached descendant holds the pipe); continuing");
            }

            code = timedOut ? TimeoutExitCode : child.ExitCode;
            if (!timedOut)
            {
                // Normal exit: let any deliberately detached daemon outlive the launcher.
                job?.ReleaseSurvivors();
            }
        }
        catch (Win32Exception ex)
        {
            // Could not even start the child (missing exe, bad image, access denied).
            code = ex.NativeErrorCode != 0 ? ex.NativeErrorCode : 1;
            tail.Add($"launcher could not start '{o.Exe}': {ex.Message}");
            rec.Write($"dtm-task-run: spawn failed code={code} reason={Redactor.RedactText(ex.Message)}");
            Finish(o, stem, status, statusPath, logPath, log, rec, watch, spawnMs, code, false, tail, EventLogSink.IdSpawnFailed);
            return code;
        }
        finally
        {
            job?.Dispose();
            child?.Dispose();
        }

        Finish(o, stem, status, statusPath, logPath, log, rec, watch, spawnMs, code, timedOut, tail, timedOut ? EventLogSink.IdTimedOut : EventLogSink.IdFailed);
        return code;
    }

    /// <summary>Formats the Application-log entry for a failed run.</summary>
    internal static string BuildEventMessage(TaskStatus s, int timeoutSeconds)
    {
        var sb = new StringBuilder();
        sb.Append("Scheduled task '").Append(s.Task).Append("' ")
          .Append(s.TimedOut ? "timed out and was killed" : "failed")
          .Append(" with exit code ").Append(s.ExitCode?.ToString(CultureInfo.InvariantCulture)).Append(" (").Append(s.ExitCodeHex).Append(')');
        if (s.TimedOut)
        {
            sb.Append(" after ").Append(timeoutSeconds.ToString(CultureInfo.InvariantCulture)).Append("s limit");
        }

        sb.AppendLine(".")
          .Append("Host: ").AppendLine(s.Host)
          .Append("Started (UTC): ").AppendLine(s.StartUtc)
          .Append("Duration ms: ").AppendLine(s.DurationMs?.ToString(CultureInfo.InvariantCulture))
          .Append("Command: ").AppendLine(s.Cmdline)
          .Append("Log: ").AppendLine(s.LogPath)
          .AppendLine("--- stderr tail (last " + TailLines.ToString(CultureInfo.InvariantCulture) + " lines, redacted) ---");
        foreach (var l in s.StderrTail)
        {
            sb.AppendLine(l);
        }

        return sb.ToString();
    }

    private void Finish(
        RunOptions o,
        string stem,
        TaskStatus status,
        string statusPath,
        string logPath,
        RollingLog log,
        RunRecorder rec,
        Stopwatch watch,
        long spawnMs,
        int code,
        bool timedOut,
        TailBuffer tail,
        int eventId)
    {
        rec.Disarm();
        var end = _ctx.UtcNow();
        status.State = "completed";
        status.EndUtc = Iso(end);
        status.DurationMs = watch.ElapsedMilliseconds;
        status.ExitCode = code;
        status.ExitCodeHex = "0x" + unchecked((uint)code).ToString("X", CultureInfo.InvariantCulture);
        status.TimedOut = timedOut;
        status.StderrTail = tail.ToArray();
        _ = StatusFile.TryWrite(statusPath, status);

        rec.WriteFinal($"dtm-task-run: end task={o.Name} exit={code} ({status.ExitCodeHex}) durationMs={status.DurationMs} spawnMs={spawnMs} timedOut={(timedOut ? "true" : "false")}");

        if (code == 0)
        {
            return;
        }

        var wrote = false;
        try
        {
            wrote = _ctx.Events.WriteFailure(eventId, BuildEventMessage(status, o.TimeoutSeconds));
        }
        catch (Exception ex) when (ex is Win32Exception or InvalidOperationException or UnauthorizedAccessException or IOException)
        {
            // Event logging must never change the task result.
        }

        log.WriteLine(wrote ? "dtm-task-run: event log entry written" : "dtm-task-run: event log entry skipped (source missing or not permitted)");

        if (o.NotifyBus)
        {
            var result = _ctx.Bus.TryNotify(o.Name, stem, code, timedOut, statusPath, logPath, end);
            log.WriteLine($"dtm-task-run: bus notify {result}");
        }
    }

    private static Thread StartPump(StreamReader reader, string tag, RunRecorder log, TailBuffer? tail)
    {
        var t = new Thread(() =>
        {
            try
            {
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    var safe = Redactor.RedactText(line);
                    log.Write($"[{tag}] {safe}");
                    tail?.Add(safe);
                }
            }
            catch (Exception ex) when (ex is IOException or ObjectDisposedException)
            {
                // The pipe closed underneath us (process tree killed); nothing left to read.
            }
        })
        {
            IsBackground = true,
            Name = "dtm-pump-" + tag,
        };
        t.Start();
        return t;
    }

    private static void TryKillTree(Process p)
    {
        try
        {
            p.Kill(entireProcessTree: true);
        }
        catch (Exception ex) when (ex is InvalidOperationException or Win32Exception or NotSupportedException)
        {
            // Already gone.
        }
    }

    private static string Iso(DateTimeOffset t) => t.UtcDateTime.ToString("o", CultureInfo.InvariantCulture);

    /// <summary>Fixed-size ring of the most recent lines; thread-safe.</summary>
    private sealed class TailBuffer
    {
        private readonly Queue<string> _lines = new();
        private readonly int _max;

        public TailBuffer(int max) => _max = max;

        public void Add(string line)
        {
            lock (_lines)
            {
                if (_lines.Count == _max)
                {
                    _ = _lines.Dequeue();
                }

                _lines.Enqueue(line);
            }
        }

        public string[] ToArray()
        {
            lock (_lines)
            {
                return [.. _lines];
            }
        }
    }
}
