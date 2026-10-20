namespace DtmTaskRun;

/// <summary>
/// Defers the "start" log line and the "running" status file until they are worth writing.
/// </summary>
/// <remarks>
/// Most runs are over in well under a second. Writing a start line and a running status for each
/// of them costs two extra file creations (each one an antivirus scan on this host) for records
/// that would be replaced milliseconds later. So the start records are emitted only when the child
/// has produced output, or is still alive after <see cref="Delay"/>, or at the end (as one batched
/// append together with the end line). A launcher killed inside that first moment leaves the
/// previous run's status in place, which is acceptable for a sub-second window.
/// </remarks>
internal sealed class RunRecorder : IDisposable
{
    /// <summary>How long a run may be silent before its start records are written anyway.</summary>
    public static readonly TimeSpan Delay = TimeSpan.FromMilliseconds(1500);

    private readonly object _gate = new();
    private readonly RollingLog _log;
    private readonly string _startLine;
    private readonly DateTimeOffset _startedAt;
    private readonly Action _writeRunningStatus;
    private Timer? _timer;
    private bool _emitted;
    private bool _closed;

    /// <summary>Creates a recorder; <paramref name="writeRunningStatus"/> persists the "running" status.</summary>
    public RunRecorder(RollingLog log, string startLine, Action writeRunningStatus)
    {
        _log = log;
        _startLine = startLine;
        _startedAt = log.Now();
        _writeRunningStatus = writeRunningStatus;
    }

    /// <summary>Arms the timer that emits the start records for long-running children.</summary>
    public void Arm() => _timer = new Timer(_ => EmitCore(fromTimer: true), null, Delay, Timeout.InfiniteTimeSpan);

    /// <summary>
    /// Called as soon as the child has exited, before the final status is built, so a late timer tick
    /// can never write a half-updated "running" status over the final one.
    /// </summary>
    public void Disarm()
    {
        lock (_gate)
        {
            _closed = true;
            _timer?.Dispose();
            _timer = null;
        }
    }

    /// <summary>Writes a line, emitting the start records first so ordering stays correct.</summary>
    public void Write(string message)
    {
        EmitCore(fromTimer: false);
        _log.WriteLine(message);
    }

    private void EmitCore(bool fromTimer)
    {
        lock (_gate)
        {
            if (_emitted || (fromTimer && _closed))
            {
                return;
            }

            _emitted = true;
            _log.WriteEntries([new LogEntry(_startedAt, _startLine)]);
            if (!_closed)
            {
                _writeRunningStatus();
            }
        }
    }

    /// <summary>Writes the final line; if the start records were never emitted they go out in the same append.</summary>
    public void WriteFinal(string message)
    {
        lock (_gate)
        {
            _timer?.Dispose();
            _timer = null;
            if (_emitted)
            {
                _log.WriteLine(message);
            }
            else
            {
                _emitted = true;
                _log.WriteEntries([new LogEntry(_startedAt, _startLine), new LogEntry(_log.Now(), message)]);
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        lock (_gate)
        {
            _timer?.Dispose();
            _timer = null;
        }
    }
}
