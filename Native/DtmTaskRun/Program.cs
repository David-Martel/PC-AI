using System.Reflection;
using System.Runtime.InteropServices;

namespace DtmTaskRun;

/// <summary>
/// Entry point of <c>dtm-task-run.exe</c>, a Windows-subsystem (no console) launcher for scheduled tasks.
/// </summary>
internal static class Program
{
    private const string Usage =
        "dtm-task-run --name <TaskId> [--timeout <sec>] [--notify-bus] -- <exe> [args...]\r\n"
        + "  --name       task id; names the log (logs\\<id>.log) and status (status\\<id>.json) files\r\n"
        + "  --timeout    kill the child's process tree after <sec> seconds and exit 1460\r\n"
        + "  --notify-bus post a finding to the agent bus on a nonzero exit (max 1/task/hour)\r\n"
        + "  --install-eventlog-source   register the Application log source (needs elevation)\r\n"
        + "Exit code is the child's exit code; 1460 on timeout; 87 for bad arguments.\r\n";

    private const int AttachParentProcess = -1;

    private static int Main(string[] args)
    {
        var parsed = OptionsParser.Parse(args);
        if (parsed.Run is not null)
        {
            // Hand the child the text that followed '--', not argv re-quoted (RawCommandLine explains why).
            parsed = parsed with { Run = parsed.Run with { CommandLine = RawCommandLine.ChildPart(RawCommandLine.Current(), args) } };
        }

        switch (parsed.Verb)
        {
            case Verb.Help:
                Print(Usage);
                return 0;
            case Verb.Version:
                Print(Assembly.GetExecutingAssembly().GetName().Version + "\r\n");
                return 0;
            case Verb.InstallEventSource:
                var ok = EventLogSink.TryRegisterSource();
                Print(ok ? "event source 'dtm-task-run' is registered\r\n" : "could not register the event source; run from an elevated shell\r\n");
                return ok ? 0 : 5;
        }

        if (parsed.Run is null)
        {
            Print("dtm-task-run: " + parsed.Error + "\r\n\r\n" + Usage);
            return TaskRunner.UsageExitCode;
        }

        try
        {
            return new TaskRunner(RunnerContext.CreateDefault()).Run(parsed.Run);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // Last resort. Say so in the task log, and return a code that is not any child's.
            new RollingLog(Path.Combine(RunnerContext.CreateDefault().BaseDir, "logs", TaskNames.ToFileStem(parsed.Run.Name) + ".log"))
                .WriteLine("dtm-task-run: internal error " + ex.GetType().Name + ": " + Redactor.RedactText(ex.Message));
            return TaskRunner.InternalErrorExitCode;
        }
    }

    // A WinExe has no console of its own. For the interactive verbs only, borrow the parent's
    // console so --help / --version / --install-eventlog-source are readable from a shell.
    // The scheduled-task path never reaches this, so no console is ever allocated there.
    private static void Print(string text)
    {
        if (AttachConsole(AttachParentProcess))
        {
            Console.Out.Write(text);
            Console.Out.Flush();
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AttachConsole(int processId);
}
