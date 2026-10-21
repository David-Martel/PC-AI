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

public class RawCommandLineTests
{
    [Theory]
    [InlineData(@"""C:\x y\dtm-task-run.exe"" --name ""T"" -- cmd.exe /c """"C:\a b\job.cmd"" ""x y""""", new[] { "--name", "T", "--" }, @"cmd.exe /c """"C:\a b\job.cmd"" ""x y""""")]
    [InlineData(@"dtm-task-run.exe --name T --timeout 590 --notify-bus --   pwsh.exe -File ""C:\a b\x.ps1""  ", new[] { "--name", "T", "--timeout", "590", "--notify-bus", "--" }, @"pwsh.exe -File ""C:\a b\x.ps1""")]
    [InlineData(@"r.exe --name ""a\""b"" -- x.exe --", new[] { "--name", @"a""b", "--" }, "x.exe --")]
    public void Child_part_is_the_text_after_the_first_bare_separator(string raw, string[] parsedPrefix, string expected)
    {
        // Why: the child must get the scheduled task's own text, not argv re-quoted (see RawCommandLine).
        // Only the launcher's own tokens up to the separator are cross-checked against argv.
        var parsed = new List<string>(parsedPrefix) { "child-token-not-compared" };
        Assert.Equal(expected, RawCommandLine.ChildPart(raw, parsed));
    }

    [Theory]
    [InlineData("r.exe --name T x.exe")]    // no separator
    [InlineData("r.exe --name T --")]       // nothing after it
    [InlineData("r.exe --name U -- x.exe")] // tokens disagree with argv: fall back rather than guess
    [InlineData("")]
    public void Child_part_is_null_when_it_cannot_be_recovered(string raw)
    {
        // Why: a null makes the runner rebuild the line from argv, which is never worse than before.
        Assert.Null(RawCommandLine.ChildPart(raw, ["--name", "T", "--", "x.exe"]));
    }

    [Theory]
    [InlineData("pwsh.exe", new[] { "-File", @"C:\a b\x.ps1" }, @"pwsh.exe -File ""C:\a b\x.ps1""")]
    [InlineData(@"C:\Program Files\x.exe", new[] { "a" }, @"""C:\Program Files\x.exe"" a")]
    [InlineData("x.exe", new[] { @"say ""hi""", @"C:\dir\", "" }, @"x.exe ""say \""hi\"""" C:\dir\ """"")]
    [InlineData("x.exe", new[] { @"C:\a b\" }, @"x.exe ""C:\a b\\""")]
    public void Build_quotes_arguments_so_argv_parsing_reverses_it(string exe, string[] args, string expected)
    {
        // Why: the fallback path (no raw text) must still give the child exactly these argv elements.
        Assert.Equal(expected, RawCommandLine.Build(exe, args));
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
