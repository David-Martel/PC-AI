using System.Management.Automation;
using System.Management.Automation.Runspaces;
using System.Text.Json;

namespace PcaiNative;

/// <summary>
/// Provides a persistent PowerShell Runspace for low-latency tool execution.
/// Avoids the overhead of starting a new process for every command.
/// </summary>
public sealed class PowerShellHost : IDisposable
{
    private readonly Runspace _runspace;
    private readonly PowerShell _ps;

    public PowerShellHost()
    {
        var iss = InitialSessionState.CreateDefault2();
        // Allow loading modules from the project
        iss.ExecutionPolicy = Microsoft.PowerShell.ExecutionPolicy.RemoteSigned;

        _runspace = RunspaceFactory.CreateRunspace(iss);
        _runspace.Open();

        _ps = PowerShell.Create();
        _ps.Runspace = _runspace;
    }

    /// <summary>
    /// Executes a PowerShell command/script and returns the result as a JSON string.
    /// </summary>
    public Task<string> ExecuteAsync(string script) => ExecuteAsync(script, timeout: null);

    /// <summary>
    /// Executes a PowerShell command/script with an optional timeout.
    /// </summary>
    public async Task<string> ExecuteAsync(string script, TimeSpan? timeout)
    {
        _ps.Commands.Clear();
        _ps.AddScript(script);

        try
        {
            var invokeTask = _ps.InvokeAsync();
            if (timeout.HasValue)
            {
                var winner = await Task.WhenAny(invokeTask, Task.Delay(timeout.Value));
                if (winner != invokeTask)
                {
                    _ps.Stop();
                    return "{\"error\": \"PowerShell execution timed out.\"}";
                }
            }

            var results = await invokeTask;

            if (_ps.HadErrors)
            {
                var errors = string.Join("; ", _ps.Streams.Error.Select(e => e.ToString()));
                return $"{{\"error\": \"PowerShell execution failed: {errors}\"}}";
            }

            if (results == null || results.Count == 0)
            {
                return "{}";
            }

            // If the last command was | ConvertTo-Json, we just return the string
            // Otherwise we serialize the results
            var lastResult = results.Last().BaseObject;
            if (lastResult is string s && (s.Trim().StartsWith("{") || s.Trim().StartsWith("[")))
            {
                return s;
            }

            return JsonSerializer.Serialize(results.Select(r => r.BaseObject));
        }
        catch (Exception ex)
        {
            return $"{{\"error\": \"Host execution failed: {ex.Message}\"}}";
        }
    }

    public void Dispose()
    {
        _ps.Dispose();
        _runspace.Dispose();
    }
}
