using System.Text.Json;

namespace DtmTaskRun.Tests;

/// <summary>Creates and removes a throwaway directory per test.</summary>
public sealed class TempDir : IDisposable
{
    public TempDir() => Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "dtm-task-run-tests-" + Guid.NewGuid().ToString("N"));

    public string Path { get; }

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, recursive: true);
        }
        catch (IOException)
        {
            // best effort
        }
    }
}

public class RollingLogTests
{
    [Fact]
    public void Appends_timestamped_lines_below_the_limit()
    {
        // Why: the log format must stay `<o-timestamp> <message>` like the wrapper scripts' logs.
        using var dir = new TempDir();
        var log = new RollingLog(System.IO.Path.Combine(dir.Path, "t.log"), () => new DateTimeOffset(2026, 10, 20, 16, 0, 0, TimeSpan.FromHours(-4)));

        log.WriteLine("one");
        log.WriteLine("two");

        Assert.Equal(["2026-10-20T16:00:00.0000000-04:00 one", "2026-10-20T16:00:00.0000000-04:00 two"], File.ReadAllLines(log.Path));
        Assert.False(File.Exists(log.Path + ".1"));
    }

    [Fact]
    public void Rolls_to_dot_one_when_over_256_KB_and_replaces_the_old_dot_one()
    {
        // Why: the owner asked for the same 256 KB rollover semantics as Update-BwArchive.ps1 (move to .1, overwrite).
        using var dir = new TempDir();
        Directory.CreateDirectory(dir.Path);
        var path = System.IO.Path.Combine(dir.Path, "t.log");
        File.WriteAllBytes(path, new byte[RollingLog.MaxBytes + 1]);
        File.WriteAllText(path + ".1", "stale previous generation");

        new RollingLog(path).WriteLine("fresh");

        Assert.Equal(RollingLog.MaxBytes + 1, new FileInfo(path + ".1").Length);
        Assert.Single(File.ReadAllLines(path));
        Assert.EndsWith("fresh", File.ReadAllLines(path)[0], StringComparison.Ordinal);
    }

    [Fact]
    public void Does_not_roll_at_exactly_the_limit()
    {
        // Why: the wrapper uses `-gt 262144`; a file of exactly 262144 bytes must not roll.
        using var dir = new TempDir();
        Directory.CreateDirectory(dir.Path);
        var path = System.IO.Path.Combine(dir.Path, "t.log");
        File.WriteAllBytes(path, new byte[RollingLog.MaxBytes]);

        new RollingLog(path).WriteLine("x");

        Assert.False(File.Exists(path + ".1"));
    }

    [Fact]
    public void Coexists_with_another_writer_holding_the_file_open()
    {
        // Why: Update-BwArchive.ps1 appends to the same file the launcher logs to; neither may lock the other out.
        using var dir = new TempDir();
        Directory.CreateDirectory(dir.Path);
        var path = System.IO.Path.Combine(dir.Path, "t.log");
        using var other = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);

        new RollingLog(path).WriteLine("from launcher");

        using var reader = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        Assert.Contains("from launcher", new StreamReader(reader).ReadToEnd(), StringComparison.Ordinal);
    }

    [Fact]
    public void Swallows_failures_so_logging_cannot_change_a_task_result()
    {
        // Why: an unwritable log directory must never raise out of the launcher.
        var log = new RollingLog(@"Z:\definitely\not\a\real\drive\t.log");
        log.WriteLine("ignored");
    }
}

public class StatusFileTests
{
    [Fact]
    public void Writes_camelCase_json_with_all_contract_fields()
    {
        // Why: Test-ScheduledTaskHealth.ps1 and agents parse these exact property names.
        using var dir = new TempDir();
        var path = System.IO.Path.Combine(dir.Path, "s", "t.json");
        var s = new TaskStatus
        {
            Task = "t", State = "completed", Pid = 5, StartUtc = "2026-10-20T20:00:00.0000000Z", EndUtc = "2026-10-20T20:00:01.0000000Z",
            DurationMs = 1000, ExitCode = 3, ExitCodeHex = "0x3", TimedOut = false, Cmdline = "x", StderrTail = ["e"], Host = "h", LogPath = "l",
        };

        Assert.True(StatusFile.TryWrite(path, s));

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        foreach (var name in new[] { "task", "startUtc", "endUtc", "durationMs", "exitCode", "timedOut", "cmdline", "stderrTail", "host", "state", "exitCodeHex", "logPath" })
        {
            Assert.True(root.TryGetProperty(name, out _), "missing property " + name);
        }

        Assert.Equal(3, root.GetProperty("exitCode").GetInt32());
        Assert.Equal("0x3", root.GetProperty("exitCodeHex").GetString());
    }

    [Fact]
    public void Overwrites_atomically_and_leaves_no_temp_files()
    {
        // Why: readers poll these files while the launcher rewrites them; a half-written file or stray .tmp is a bug.
        using var dir = new TempDir();
        var path = System.IO.Path.Combine(dir.Path, "t.json");

        StatusFile.TryWrite(path, new TaskStatus { Task = "first" });
        StatusFile.TryWrite(path, new TaskStatus { Task = "second" });

        Assert.Contains("second", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal([path], Directory.GetFiles(dir.Path));
    }

    [Fact]
    public void Reports_failure_instead_of_throwing_on_an_unwritable_path()
    {
        // Why: status is observability; losing it must not fail the task.
        Assert.False(StatusFile.TryWrite(@"Z:\nope\t.json", new TaskStatus()));
    }
}
