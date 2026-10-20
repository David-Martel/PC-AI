using System.Runtime.Versioning;

[assembly: SupportedOSPlatform("windows")]

namespace DtmTaskRun.Tests;

public class OptionsParserTests
{
    [Fact]
    public void Parses_name_timeout_notify_and_child()
    {
        // Why: the scheduled-task action line is exactly this shape; a parse regression breaks every migrated task.
        var r = OptionsParser.Parse(["--name", "Bitwarden-Archive-Refresh", "--timeout", "600", "--notify-bus", "--", "pwsh.exe", "-File", "x.ps1"]);

        Assert.Null(r.Error);
        Assert.Equal(Verb.Run, r.Verb);
        Assert.Equal("Bitwarden-Archive-Refresh", r.Run!.Name);
        Assert.Equal(600, r.Run.TimeoutSeconds);
        Assert.True(r.Run.NotifyBus);
        Assert.Equal("pwsh.exe", r.Run.Exe);
        Assert.Equal(["-File", "x.ps1"], r.Run.Args);
    }

    [Fact]
    public void Flags_after_the_separator_belong_to_the_child()
    {
        // Why: a child that has its own --timeout or --name must not be hijacked by the launcher.
        var r = OptionsParser.Parse(["--name", "t", "--", "tool.exe", "--timeout", "5", "--name", "x", "--notify-bus"]);

        Assert.Equal(0, r.Run!.TimeoutSeconds);
        Assert.False(r.Run.NotifyBus);
        Assert.Equal(["--timeout", "5", "--name", "x", "--notify-bus"], r.Run.Args);
    }

    [Theory]
    [InlineData("--name", "t", "tool.exe")]                          // no separator
    [InlineData("--", "tool.exe")]                                    // no name
    [InlineData("--name", "t", "--")]                                 // no child
    [InlineData("--name", "t", "--timeout", "abc", "--", "x.exe")]    // bad timeout
    [InlineData("--name", "t", "--timeout", "-5", "--", "x.exe")]     // negative timeout
    [InlineData("--name", "t", "--timeout", "999999", "--", "x.exe")] // absurd timeout
    [InlineData("--name", "t", "--bogus", "--", "x.exe")]             // unknown launcher option
    public void Rejects_malformed_command_lines(params string[] argv)
    {
        // Why: a malformed task action must fail loudly (exit 87) rather than run something unintended.
        var r = OptionsParser.Parse(argv);

        Assert.Null(r.Run);
        Assert.NotNull(r.Error);
    }

    [Theory]
    [InlineData("PC-AI Process Lasso Governor Watchdog", "PC-AI_Process_Lasso_Governor_Watchdog")]
    [InlineData("\\PowerShell\\ProfileLogSync", "_PowerShell_ProfileLogSync")]
    [InlineData("plain-name", "plain-name")]
    [InlineData("a:b*c?d", "a_b_c_d")]
    public void File_stem_maps_spaces_and_invalid_characters(string name, string expected)
    {
        // Why: the health script re-implements this mapping to find status files; both sides must agree.
        Assert.Equal(expected, TaskNames.ToFileStem(name));
    }

    [Fact]
    public void File_stem_never_produces_a_path_traversal()
    {
        // Why: --name is attacker-adjacent input (a task XML); it must not escape the logs directory.
        var stem = TaskNames.ToFileStem("..\\..\\evil");
        Assert.DoesNotContain('\\', stem);
        Assert.DoesNotContain('/', stem);
        Assert.False(stem.StartsWith("..", StringComparison.Ordinal));
    }
}

public class RedactorTests
{
    [Theory]
    [InlineData("token=abc123", "token=***")]
    [InlineData("--api-key=sk_live_x", "--api-key=***")]
    [InlineData("Password=hunter2", "Password=***")]
    [InlineData("client_secret:topsecret", "client_secret:***")]
    [InlineData("access_key=AKIA1234", "access_key=***")]
    public void Masks_name_value_arguments(string arg, string expected)
    {
        // Why: the brief requires token|password|secret|key= arguments never reach the status file or event log.
        Assert.Equal([expected], Redactor.RedactArgs([arg]));
    }

    [Fact]
    public void Masks_the_value_following_a_sensitive_flag()
    {
        // Why: `--token VALUE` (space form) is as common as `--token=VALUE` and must be masked too.
        var r = Redactor.RedactArgs(["-File", "x.ps1", "--token", "abc123", "-Password", "hunter2", "-Quiet"]);

        Assert.Equal(["-File", "x.ps1", "--token", "***", "-Password", "***", "-Quiet"], r);
    }

    [Fact]
    public void Leaves_ordinary_arguments_and_paths_alone()
    {
        // Why: over-redaction makes the status file useless for debugging; benign args must pass through.
        string[] args = ["-NonInteractive", "-File", @"C:\Users\david\.machine\Update-BwArchive.ps1", "-ReportPath", @"D:\pcai-relocation\x.json", "backup", "-Keep", "14"];

        Assert.Equal(args, Redactor.RedactArgs(args));
    }

    [Theory]
    [InlineData("calling api with token=abc123 now", "abc123")]
    [InlineData("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghijklmnop.qrstuvwxyz", "eyJhbGci")]
    [InlineData("Bearer abcDEF123456", "abcDEF123456")]
    [InlineData("{\"password\": \"hunter2\"}", "hunter2")]
    [InlineData("clone https://user:s3cr3t@example.com/repo", "s3cr3t")]
    // Token-shaped test data is assembled from pieces so secret scanners do not flag the source.
    [InlineData("auth gh" + "p_abcdefghijklmnopqrstuvwxyz0123456789", "gh" + "p_abcdef")]
    public void Masks_secrets_inside_free_text(string line, string secret)
    {
        // Why: stderr tails and log lines are free text; common token shapes must be masked there as well.
        var redacted = Redactor.RedactText(line);

        Assert.DoesNotContain(secret, redacted);
        Assert.Contains(Redactor.Mask, redacted);
    }

    [Fact]
    public void Leaves_plain_errors_readable()
    {
        // Why: the whole point of the stderr tail is to read the error; do not mask ordinary text.
        const string line = "The term 'Get-Foo' is not recognized as a name of a cmdlet, function, or script file.";
        Assert.Equal(line, Redactor.RedactText(line));
    }

    [Fact]
    public void Display_command_line_quotes_arguments_with_spaces()
    {
        // Why: the cmdline in status JSON must be copy-pasteable, so spaces stay quoted.
        Assert.Equal(@"pwsh.exe -File ""C:\a b\x.ps1""", Redactor.JoinForDisplay("pwsh.exe", ["-File", @"C:\a b\x.ps1"]));
    }
}
