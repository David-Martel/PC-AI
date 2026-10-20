namespace DtmTaskRun;

/// <summary>What the command line asked the launcher to do.</summary>
internal enum Verb
{
    /// <summary>Run a child process (the normal, scheduled-task path).</summary>
    Run,

    /// <summary>Print usage (only visible when started from a console).</summary>
    Help,

    /// <summary>Print the version.</summary>
    Version,

    /// <summary>Register the Application event-log source (needs an elevated token).</summary>
    InstallEventSource,
}

/// <summary>Validated arguments for <see cref="Verb.Run"/>.</summary>
/// <param name="Name">Raw task id as given; <see cref="TaskNames.ToFileStem"/> maps it to file names.</param>
/// <param name="TimeoutSeconds">0 means "no launcher timeout" (Task Scheduler's own limit still applies).</param>
/// <param name="NotifyBus">Post a finding to the agent bus on a nonzero exit.</param>
/// <param name="Exe">Child executable.</param>
/// <param name="Args">Child arguments, already split; re-quoted by ProcessStartInfo.ArgumentList.</param>
internal sealed record RunOptions(string Name, int TimeoutSeconds, bool NotifyBus, string Exe, IReadOnlyList<string> Args);

/// <summary>Outcome of parsing the command line.</summary>
internal sealed record ParseResult(Verb Verb, RunOptions? Run, string? Error);

/// <summary>
/// Parses <c>--name ID [--timeout SEC] [--notify-bus] -- EXE ARGS...</c>.
/// Everything after the first bare <c>--</c> belongs to the child verbatim, so child
/// flags such as <c>--timeout</c> or <c>-File</c> are never misread as launcher flags.
/// </summary>
internal static class OptionsParser
{
    /// <summary>Upper bound on --timeout; a day is already longer than any task here.</summary>
    internal const int MaxTimeoutSeconds = 86_400;

    /// <summary>Parses <paramref name="argv"/>.</summary>
    public static ParseResult Parse(IReadOnlyList<string> argv)
    {
        string? name = null;
        var timeout = 0;
        var notify = false;

        for (var i = 0; i < argv.Count; i++)
        {
            var a = argv[i];
            switch (a)
            {
                case "--":
                    return BuildRun(name, timeout, notify, argv, i + 1);
                case "--help" or "-h" or "-?" or "/?":
                    return new ParseResult(Verb.Help, null, null);
                case "--version":
                    return new ParseResult(Verb.Version, null, null);
                case "--install-eventlog-source":
                    return new ParseResult(Verb.InstallEventSource, null, null);
                case "--notify-bus":
                    notify = true;
                    break;
                case "--name":
                    if (i + 1 >= argv.Count)
                    {
                        return Fail("--name needs a value");
                    }

                    name = argv[++i];
                    break;
                case "--timeout":
                    if (i + 1 >= argv.Count
                        || !int.TryParse(argv[i + 1], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out timeout)
                        || timeout > MaxTimeoutSeconds)
                    {
                        return Fail($"--timeout needs whole seconds between 0 and {MaxTimeoutSeconds}");
                    }

                    i++;
                    break;
                default:
                    return Fail($"unknown option '{a}' (child arguments must follow a bare '--')");
            }
        }

        return Fail("missing '--' separator before the child command");
    }

    private static ParseResult BuildRun(string? name, int timeout, bool notify, IReadOnlyList<string> argv, int childStart)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return Fail("--name is required");
        }

        if (childStart >= argv.Count || string.IsNullOrWhiteSpace(argv[childStart]))
        {
            return Fail("no child executable after '--'");
        }

        var args = new List<string>(argv.Count - childStart - 1);
        for (var i = childStart + 1; i < argv.Count; i++)
        {
            args.Add(argv[i]);
        }

        return new ParseResult(Verb.Run, new RunOptions(name.Trim(), timeout, notify, argv[childStart], args), null);
    }

    private static ParseResult Fail(string message) => new(Verb.Run, null, message);
}

/// <summary>Maps a task id to the file names every component shares.</summary>
internal static class TaskNames
{
    /// <summary>
    /// File stem for a task id: every character that is invalid in a Windows file name,
    /// plus space, becomes <c>_</c>. So <c>PC-AI Process Lasso Governor Watchdog</c> is
    /// <c>PC-AI_Process_Lasso_Governor_Watchdog</c>. The health check reimplements this rule.
    /// </summary>
    public static string ToFileStem(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.Trim().ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (chars[i] == ' ' || Array.IndexOf(invalid, chars[i]) >= 0)
            {
                chars[i] = '_';
            }
        }

        var stem = new string(chars).Trim('.');
        return stem.Length == 0 ? "_" : stem;
    }
}
