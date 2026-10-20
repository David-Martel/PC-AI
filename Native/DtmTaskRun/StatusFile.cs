using System.Text.Json;
using System.Text.Json.Serialization;

namespace DtmTaskRun;

/// <summary>
/// The per-task status record written to <c>%LOCALAPPDATA%\dtm-tasks\status\&lt;stem&gt;.json</c>.
/// Property names are the contract with <c>Tools/Test-ScheduledTaskHealth.ps1</c>.
/// </summary>
internal sealed class TaskStatus
{
    /// <summary>Bump when a field changes meaning.</summary>
    public int SchemaVersion { get; set; } = 1;

    /// <summary>Raw task id as passed in <c>--name</c>.</summary>
    public string Task { get; set; } = "";

    /// <summary><c>running</c> while the child lives, <c>completed</c> once it has exited. A file stuck at
    /// <c>running</c> with no live task means the launcher itself was killed (Task Scheduler stop, logoff).</summary>
    public string State { get; set; } = "running";

    /// <summary>Launcher process id (lets a reader tell a live run from a dead one).</summary>
    public int Pid { get; set; }

    /// <summary>UTC start time, ISO 8601.</summary>
    public string StartUtc { get; set; } = "";

    /// <summary>UTC end time, null while running.</summary>
    public string? EndUtc { get; set; }

    /// <summary>Wall time in milliseconds, null while running.</summary>
    public long? DurationMs { get; set; }

    /// <summary>Exit code the launcher itself exited with (the child's code, 1460 on timeout), null while running.</summary>
    public int? ExitCode { get; set; }

    /// <summary>The same code as Task Scheduler's LastTaskResult would show, e.g. <c>0x3</c> or <c>0xC000013A</c>.</summary>
    public string? ExitCodeHex { get; set; }

    /// <summary>True when the launcher's own <c>--timeout</c> fired and the process tree was killed.</summary>
    public bool TimedOut { get; set; }

    /// <summary>Redacted command line.</summary>
    public string Cmdline { get; set; } = "";

    /// <summary>Last 20 stderr lines, redacted.</summary>
    public string[] StderrTail { get; set; } = [];

    /// <summary>Lower-case machine name.</summary>
    public string Host { get; set; } = "";

    /// <summary>Path of the task log.</summary>
    public string LogPath { get; set; } = "";
}

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(TaskStatus))]
internal sealed partial class StatusJsonContext : JsonSerializerContext;

/// <summary>Writes <see cref="TaskStatus"/> atomically.</summary>
internal static class StatusFile
{
    // Relaxed escaping keeps quotes and ampersands in command lines readable instead of " / &.
    private static readonly StatusJsonContext Context = new(new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Encoder =System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    });

    /// <summary>Serialises to JSON text.</summary>
    public static string Serialize(TaskStatus status) => JsonSerializer.Serialize(status, Context.TaskStatus);

    /// <summary>
    /// Writes <paramref name="status"/> to <paramref name="path"/> via a temp file in the same
    /// directory and an overwriting move, so a reader never sees a half-written file.
    /// Failures are swallowed (status is observability, not control flow).
    /// </summary>
    public static bool TryWrite(string path, TaskStatus status)
    {
        var tmp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(tmp, Serialize(status), new System.Text.UTF8Encoding(false));
            File.Move(tmp, path, overwrite: true);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            try
            {
                File.Delete(tmp);
            }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
            {
                // nothing more to do
            }

            return false;
        }
    }
}
