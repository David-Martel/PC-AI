using System.Diagnostics;
using System.Text.Json;

namespace DtmTaskRun.Tests;

internal sealed class FakeEvents : IEventSink
{
    public List<(int Id, string Message)> Entries { get; } = [];

    public bool WriteFailure(int eventId, string message)
    {
        Entries.Add((eventId, message));
        return true;
    }
}

internal sealed class FakeBus : IBusNotifier
{
    public List<(string Task, int Exit, bool TimedOut)> Calls { get; } = [];

    public NotifyResult TryNotify(string taskName, string fileStem, int exitCode, bool timedOut, string statusPath, string logPath, DateTimeOffset nowUtc)
    {
        Calls.Add((taskName, exitCode, timedOut));
        return NotifyResult.Sent;
    }
}

/// <summary>Runs the real launcher core against real child processes (cmd.exe, ping.exe).</summary>
public sealed class TaskRunnerTests : IDisposable
{
    private readonly TempDir _dir = new();
    private readonly FakeEvents _events = new();
    private readonly FakeBus _bus = new();

    public void Dispose() => _dir.Dispose();

    private TaskRunner Runner() => new(new RunnerContext { BaseDir = _dir.Path, Host = "testhost", Events = _events, Bus = _bus });

    private static RunOptions Cmd(string name, string script, int timeout = 0, bool notify = false)
        => new(name, timeout, notify, "cmd.exe", ["/c", script]);

    private JsonElement Status(string stem)
    {
        var text = File.ReadAllText(Path.Combine(_dir.Path, "status", stem + ".json"));
        return JsonDocument.Parse(text).RootElement.Clone();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(255)]
    public void Propagates_the_childs_exit_code(int expected)
    {
        // Why: this is the bug being fixed. conhost --headless reported 0x0 whatever the child returned.
        var code = Runner().Run(Cmd("exit-code", $"exit {expected}"));

        Assert.Equal(expected, code);
        Assert.Equal(expected, Status("exit-code").GetProperty("exitCode").GetInt32());
    }

    [Fact]
    public void Preserves_ntstatus_style_codes_and_renders_them_as_unsigned_hex()
    {
        // Why: killed processes exit 0xC000013A; Task Scheduler shows it unsigned, so the status file must too.
        var code = Runner().Run(Cmd("ntstatus", "exit 3221225786"));

        Assert.Equal(unchecked((int)0xC000013A), code);
        Assert.Equal("0xC000013A", Status("ntstatus").GetProperty("exitCodeHex").GetString());
    }

    [Fact]
    public void Kills_the_whole_process_tree_on_timeout_and_returns_1460()
    {
        // Why: a hung task must be bounded and surfaced as ERROR_TIMEOUT, and its grandchildren must not linger.
        var started = DateTime.Now;
        var sw = Stopwatch.StartNew();

        var code = Runner().Run(Cmd("timeout", "ping -n 40 -l 1371 127.0.0.1 >nul", timeout: 1));

        sw.Stop();
        Assert.Equal(TaskRunner.TimeoutExitCode, code);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(15), $"took {sw.Elapsed}");
        Assert.True(Status("timeout").GetProperty("timedOut").GetBoolean());
        Thread.Sleep(500);
        Assert.DoesNotContain(Process.GetProcessesByName("PING"), p => SafeStart(p) >= started);
        Assert.Contains(_events.Entries, e => e.Id == EventLogSink.IdTimedOut);
    }

    [Fact]
    public void Writes_status_json_with_task_host_times_and_redacted_command()
    {
        // Why: the status contract (task/start/end/duration/exit/timedOut/cmdline/tail/host) is what agents and the health check read.
        var code = Runner().Run(Cmd("status check", "echo password=hunter2 1>&2 & exit 4"));

        var s = Status("status_check");
        Assert.Equal(4, code);
        Assert.Equal("status check", s.GetProperty("task").GetString());
        Assert.Equal("testhost", s.GetProperty("host").GetString());
        Assert.Equal("completed", s.GetProperty("state").GetString());
        Assert.False(s.GetProperty("timedOut").GetBoolean());
        Assert.True(s.GetProperty("durationMs").GetInt64() >= 0);
        Assert.True(DateTime.Parse(s.GetProperty("endUtc").GetString()!, null, System.Globalization.DateTimeStyles.RoundtripKind)
                    >= DateTime.Parse(s.GetProperty("startUtc").GetString()!, null, System.Globalization.DateTimeStyles.RoundtripKind));
        Assert.DoesNotContain("hunter2", s.GetProperty("cmdline").GetString(), StringComparison.Ordinal);
        Assert.DoesNotContain("hunter2", string.Join('\n', s.GetProperty("stderrTail").EnumerateArray().Select(e => e.GetString())), StringComparison.Ordinal);
    }

    [Fact]
    public void Keeps_only_the_last_twenty_stderr_lines()
    {
        // Why: the brief caps the tail at 20 lines so status files and event-log entries stay small.
        Runner().Run(Cmd("tail", "(for /L %i in (1,1,30) do @echo line%i 1>&2) & exit 2"));

        var tail = Status("tail").GetProperty("stderrTail").EnumerateArray().Select(e => e.GetString()!.Trim()).ToArray();
        Assert.Equal(20, tail.Length);
        Assert.Equal("line11", tail[0]);
        Assert.Equal("line30", tail[^1]);
    }

    [Fact]
    public void Redacts_secrets_in_the_log_event_and_status()
    {
        // Why: logs, the Application log and the status file persist text; none of them may carry a token.
        Runner().Run(RunOptionsWithSecretArg());

        var all = File.ReadAllText(Path.Combine(_dir.Path, "logs", "redact.log"))
                  + File.ReadAllText(Path.Combine(_dir.Path, "status", "redact.json"))
                  + string.Join('\n', _events.Entries.Select(e => e.Message));
        Assert.DoesNotContain("S3CRET-VALUE", all, StringComparison.Ordinal);
        Assert.DoesNotContain("hunter2", all, StringComparison.Ordinal);
    }

    private static RunOptions RunOptionsWithSecretArg()
        => new("redact", 0, false, "cmd.exe", ["/c", "echo token=S3CRET-VALUE 1>&2 & exit 9", "--password=hunter2"]);

    [Fact]
    public void Failure_writes_one_event_log_entry_with_task_exit_code_and_tail()
    {
        // Why: the owner wants failures to be visible in Event Viewer, with enough text to investigate.
        Runner().Run(Cmd("evt", "echo boom happened 1>&2 & exit 3"));

        var entry = Assert.Single(_events.Entries);
        Assert.Equal(EventLogSink.IdFailed, entry.Id);
        Assert.Contains("'evt'", entry.Message, StringComparison.Ordinal);
        Assert.Contains("exit code 3", entry.Message, StringComparison.Ordinal);
        Assert.Contains("boom happened", entry.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Success_writes_no_event_and_posts_nothing_to_the_bus()
    {
        // Why: only failures are interesting; a green run must stay silent or the signal drowns.
        Runner().Run(Cmd("quiet", "exit 0", notify: true));

        Assert.Empty(_events.Entries);
        Assert.Empty(_bus.Calls);
    }

    [Fact]
    public void Bus_is_notified_only_when_requested()
    {
        // Why: --notify-bus is opt-in per task; tasks without it must never talk to the bus.
        Runner().Run(Cmd("nobus", "exit 5", notify: false));
        Runner().Run(Cmd("withbus", "exit 5", notify: true));

        var call = Assert.Single(_bus.Calls);
        Assert.Equal(("withbus", 5, false), call);
    }

    [Fact]
    public void Missing_executable_returns_the_win32_error_and_logs_a_spawn_event()
    {
        // Why: a task pointing at a deleted exe must say so (ERROR_FILE_NOT_FOUND) instead of silently doing nothing.
        var code = Runner().Run(new RunOptions("nospawn", 0, false, @"C:\definitely\missing\tool.exe", []));

        Assert.Equal(2, code);
        Assert.Contains(_events.Entries, e => e.Id == EventLogSink.IdSpawnFailed);
        Assert.Equal(2, Status("nospawn").GetProperty("exitCode").GetInt32());
    }

    [Fact]
    public void Logs_child_stdout_and_stderr_with_stream_tags()
    {
        // Why: the per-task log is the first place to look; it must show which stream a line came from.
        Runner().Run(Cmd("streams", "echo to-out & echo to-err 1>&2"));

        var log = File.ReadAllText(Path.Combine(_dir.Path, "logs", "streams.log"));
        Assert.Contains("[out] to-out", log, StringComparison.Ordinal);
        Assert.Contains("[err] to-err", log, StringComparison.Ordinal);
        Assert.Contains("dtm-task-run: end task=streams exit=0", log, StringComparison.Ordinal);
    }

    [Fact]
    public void Start_info_hides_the_window_and_redirects_everything()
    {
        // Why: this is the "no visible window" guarantee; CreateNoWindow + UseShellExecute=false is what suppresses the console.
        var psi = TaskRunner.BuildStartInfo(new RunOptions("n", 0, false, "pwsh.exe", ["-File", @"C:\a b\x.ps1"]));

        Assert.True(psi.CreateNoWindow);
        Assert.False(psi.UseShellExecute);
        Assert.True(psi.RedirectStandardInput);
        Assert.True(psi.RedirectStandardOutput);
        Assert.True(psi.RedirectStandardError);
        Assert.Equal(["-File", @"C:\a b\x.ps1"], psi.ArgumentList);
        Assert.Equal("n", psi.Environment["DTM_TASK_NAME"]);
    }

    [Fact]
    public void Launcher_is_built_for_the_windows_gui_subsystem()
    {
        // Why: a console-subsystem launcher would itself flash a window; WinExe (subsystem 2) never allocates one.
        var dll = typeof(TaskRunner).Assembly.Location;
        var exe = Path.Combine(Path.GetDirectoryName(dll)!, "dtm-task-run.exe");
        Assert.True(File.Exists(exe), "apphost not found next to " + dll);

        using var fs = File.OpenRead(exe);
        using var br = new BinaryReader(fs);
        fs.Position = 0x3C;
        var peOffset = br.ReadInt32();
        fs.Position = peOffset + 4 + 20 + 68; // PE sig + COFF header + optional-header offset of Subsystem
        Assert.Equal(2, br.ReadUInt16());
    }

    private static DateTime SafeStart(Process p)
    {
        try
        {
            return p.StartTime;
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            return DateTime.MinValue;
        }
    }
}

public class JobObjectTests
{
    private static Process StartPing()
        => Process.Start(new ProcessStartInfo("ping.exe", "-n 60 127.0.0.1") { CreateNoWindow = true, UseShellExecute = false, RedirectStandardOutput = true })!;

    [Fact]
    public void Closing_the_job_kills_the_child_when_kill_on_close_is_armed()
    {
        // Why: if Task Scheduler kills the launcher, the child must die with it, not be orphaned and duplicated by the next trigger.
        using var p = StartPing();
        var job = JobObject.TryCreate();
        Assert.NotNull(job);
        Assert.True(job.TryAssign(p));

        job.Dispose();

        Assert.True(p.WaitForExit(5000), "child survived job close");
    }

    [Fact]
    public void Released_job_lets_a_detached_child_outlive_the_launcher()
    {
        // Why: LspmuxServer starts a daemon and exits; on a normal exit the daemon must survive, as it did under conhost.
        using var p = StartPing();
        try
        {
            var job = JobObject.TryCreate();
            Assert.NotNull(job);
            Assert.True(job.TryAssign(p));

            job.ReleaseSurvivors();
            job.Dispose();

            Assert.False(p.WaitForExit(1000), "child died although survivors were released");
        }
        finally
        {
            p.Kill(entireProcessTree: true);
        }
    }
}

public class BusNotifierTests
{
    private static BusNotifier Make(TempDir dir, List<IReadOnlyList<string>> sent, bool ok = true)
        => new(dir.Path, "ignored.exe", "testhost", args =>
        {
            sent.Add(args);
            return ok;
        });

    private static readonly DateTimeOffset T0 = new(2026, 10, 20, 12, 0, 0, TimeSpan.Zero);

    [Fact]
    public void Builds_the_documented_finding_command()
    {
        // Why: the bus rejects findings without FINDING:/SEVERITY:, and the thread/topic/tags are the agreed routing contract.
        var args = BusNotifier.BuildArgs("dtm-p1gen7", BusNotifier.BuildBody("T", 3, false, "dtm-p1gen7", "s.json", "l.log"));

        Assert.Equal(["send", "--from-agent", "dtm-task-run", "--to-agent", "all", "--topic", "task-failures", "--schema", "finding", "--thread-id", "dtm-p1gen7-tasks"], args.Take(11));
        Assert.Contains("host:dtm-p1gen7", args);
        var body = args[^1];
        Assert.StartsWith("FINDING: task T failed exit=3 (0x3)", body, StringComparison.Ordinal);
        Assert.EndsWith("SEVERITY: medium", body, StringComparison.Ordinal);
        Assert.True(body.Length < 2000);
    }

    [Fact]
    public void Rate_limits_to_one_message_per_task_per_hour()
    {
        // Why: a task failing every 5 minutes must not flood the bus; the brief caps it at 1 per task per hour.
        using var dir = new TempDir();
        var sent = new List<IReadOnlyList<string>>();
        var n = Make(dir, sent);

        Assert.Equal(NotifyResult.Sent, n.TryNotify("T", "T", 1, false, "s", "l", T0));
        Assert.Equal(NotifyResult.RateLimited, n.TryNotify("T", "T", 1, false, "s", "l", T0.AddMinutes(59)));
        Assert.Equal(NotifyResult.Sent, n.TryNotify("T", "T", 1, false, "s", "l", T0.AddMinutes(61)));
        Assert.Equal(2, sent.Count);
    }

    [Fact]
    public void Rate_limit_is_per_task()
    {
        // Why: one noisy task must not suppress the first report from a different task.
        using var dir = new TempDir();
        var sent = new List<IReadOnlyList<string>>();
        var n = Make(dir, sent);

        n.TryNotify("A", "A", 1, false, "s", "l", T0);
        Assert.Equal(NotifyResult.Sent, n.TryNotify("B", "B", 1, false, "s", "l", T0));
    }

    [Fact]
    public void Backs_off_after_a_failed_send_so_a_dead_bus_cannot_slow_every_run()
    {
        // Why: each attempt can cost up to 5 s; after a failure, wait before trying again.
        using var dir = new TempDir();
        var sent = new List<IReadOnlyList<string>>();
        var n = Make(dir, sent, ok: false);

        Assert.Equal(NotifyResult.Failed, n.TryNotify("T", "T", 1, false, "s", "l", T0));
        Assert.Equal(NotifyResult.BackingOff, n.TryNotify("T", "T", 1, false, "s", "l", T0.AddMinutes(5)));
        Assert.Equal(NotifyResult.Failed, n.TryNotify("T", "T", 1, false, "s", "l", T0.AddMinutes(11)));
        Assert.Equal(2, sent.Count);
    }

    [Fact]
    public void Message_never_contains_command_line_or_stderr()
    {
        // Why: the bus is the one channel that leaves the machine; redaction is best effort, so the body carries only ids and paths.
        var body = BusNotifier.BuildBody("T", 1, false, "h", @"C:\s\T.json", @"C:\l\T.log");

        Assert.DoesNotContain("stderr", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("cmdline", body, StringComparison.OrdinalIgnoreCase);
    }
}
