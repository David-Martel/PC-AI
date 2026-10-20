using System.Diagnostics;

namespace PcaiNative.Tests;

internal sealed record BoundedTestProcessResult(int ExitCode, string StandardOutput, string StandardError);

/// <summary>Runs owned test children with concurrent pipe draining and bounded cleanup.</summary>
internal static class BoundedTestProcess
{
    // Disposable-test cleanup allowance; never a production process or request timeout.
    private static readonly TimeSpan CleanupTimeout = TimeSpan.FromSeconds(5);

    internal static async Task<BoundedTestProcessResult> RunAsync(ProcessStartInfo startInfo, TimeSpan timeout)
    {
        ArgumentNullException.ThrowIfNull(startInfo);
        if (timeout <= TimeSpan.Zero || timeout == Timeout.InfiniteTimeSpan)
            throw new ArgumentOutOfRangeException(nameof(timeout), "A finite positive deadline is required.");
        if (startInfo.UseShellExecute || !startInfo.RedirectStandardOutput || !startInfo.RedirectStandardError)
            throw new ArgumentException("Test processes require redirected stdout/stderr and UseShellExecute=false.", nameof(startInfo));

        using var process = new Process { StartInfo = startInfo };
        using var reads = new CancellationTokenSource();
        if (!process.Start()) throw new InvalidOperationException("Test process did not start.");
        int processId = process.Id;
        Task<string> stdout = process.StandardOutput.ReadToEndAsync(reads.Token);
        Task<string> stderr = process.StandardError.ReadToEndAsync(reads.Token);
        Task completion = Task.WhenAll(process.WaitForExitAsync(), stdout, stderr);
        try
        {
            // The same deadline covers process exit AND EOF on both redirected streams.
            await completion.WaitAsync(timeout).ConfigureAwait(false);
            return new(process.ExitCode, await stdout.ConfigureAwait(false), await stderr.ConfigureAwait(false));
        }
        catch (Exception failure)
        {
            Exception? cleanupFailure = await CleanupAsync(process, reads, completion).ConfigureAwait(false);
            if (failure is TimeoutException)
                throw new TimeoutException($"Test process {processId} exceeded {timeout}; " +
                    (cleanupFailure is null ? "parent exited/reaped; redirected reads canceled." : "bounded cleanup failed."),
                    cleanupFailure is null ? failure : new AggregateException(failure, cleanupFailure));
            if (cleanupFailure is not null)
                throw new AggregateException($"Test process {processId} failed and bounded cleanup failed.", failure, cleanupFailure);
            throw;
        }
    }

    private static async Task<Exception?> CleanupAsync(Process process, CancellationTokenSource reads, Task completion)
    {
        Exception? failure = null;
        try
        {
            // Best effort only while the parent still exists. We cannot identify an
            // already-reparented descendant after its original parent has exited.
            if (!process.HasExited)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) when (process.HasExited) { }
            }
            await process.WaitForExitAsync().WaitAsync(CleanupTimeout).ConfigureAwait(false);
        }
        catch (Exception error) { failure = error; }
        finally { reads.Cancel(); }
        try { await completion.WaitAsync(CleanupTimeout).ConfigureAwait(false); }
        catch (OperationCanceledException) { }
        catch (Exception error)
        {
            // A completed read/exit failure is the original error, not a cleanup
            // failure; inability to finish within the cleanup budget is additional.
            if (error is TimeoutException)
                failure = failure is null ? error : new AggregateException(failure, error);
        }
        return failure;
    }
}
