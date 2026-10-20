using System.Diagnostics;
using System.Text;

namespace PcaiNative.Tests;

public sealed class BoundedTestProcessTests
{
    private static ProcessStartInfo PowerShell(string script)
    {
        var info = new ProcessStartInfo("pwsh")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (string argument in new[] { "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                     Convert.ToBase64String(Encoding.Unicode.GetBytes(script)) })
            info.ArgumentList.Add(argument);
        return info;
    }

    // Protects concurrent pipe draining; detects ReadToEnd-before-exit/other-pipe deadlocks.
    // Needs installed pwsh, no native library/hardware; breadcrumb: BoundedTestProcess.RunAsync.
    [Fact]
    public async Task CapturesExactLargeOutputOnBothPipesWithoutDeadlock()
    {
        var result = await BoundedTestProcess.RunAsync(
            PowerShell("""
                Add-Type -TypeDefinition 'using System; using System.Threading.Tasks; public static class PipeWriter { public static void Run() { Task.WaitAll(Task.Run(() => Console.Out.Write(new string((char)79, 262144))), Task.Run(() => Console.Error.Write(new string((char)69, 262144)))); } }'
                [PipeWriter]::Run()
                """),
            TimeSpan.FromSeconds(30));
        Assert.Equal(0, result.ExitCode);
        Assert.Equal(new string('O', 262144), result.StandardOutput);
        Assert.Equal(new string('E', 262144), result.StandardError);
    }

    // Protects native-tool failure evidence; detects converting nonzero exit into success.
    // Needs pwsh only; breadcrumb: BoundedTestProcessResult and RunAsync return path.
    [Fact]
    public async Task PreservesNonzeroExitAndBothDiagnosticStreams()
    {
        var result = await BoundedTestProcess.RunAsync(
            PowerShell("[Console]::Out.Write('stdout-sentinel'); [Console]::Error.Write('stderr-sentinel'); exit 23"),
            TimeSpan.FromSeconds(30));
        Assert.Equal(23, result.ExitCode);
        Assert.Equal("stdout-sentinel", result.StandardOutput);
        Assert.Equal("stderr-sentinel", result.StandardError);
    }

    // Protects bounded owned-process cleanup; detects an ineffective deadline or surviving child.
    // Needs pwsh and private temporary file; breadcrumb: RunAsync timeout/kill/reap path.
    [Fact]
    public async Task HangingChildTimesOutAndKnownProcessHasExited()
    {
        string pidFile = Path.Combine(Path.GetTempPath(), "pcai-child-pid-" + Guid.NewGuid().ToString("N"));
        var elapsed = Stopwatch.StartNew();
        try
        {
            string escaped = pidFile.Replace("'", "''");
            var error = await Assert.ThrowsAsync<TimeoutException>(() => BoundedTestProcess.RunAsync(
                PowerShell($"[IO.File]::WriteAllText('{escaped}', [string]$PID); Start-Sleep -Seconds 120"),
                TimeSpan.FromSeconds(3)));
            Assert.DoesNotContain("cleanup failed", error.Message, StringComparison.OrdinalIgnoreCase);
            Assert.True(elapsed.Elapsed < TimeSpan.FromSeconds(20), $"Cleanup exceeded test budget: {elapsed.Elapsed}");
            Assert.True(File.Exists(pidFile), "Child did not publish its PID before the deadline.");
            int childId = int.Parse(File.ReadAllText(pidFile));
            try
            {
                using var child = Process.GetProcessById(childId);
                Assert.True(child.HasExited, $"Timed-out owned child {childId} remains alive.");
            }
            catch (ArgumentException) { /* PID no longer exists: expected successful reap. */ }
        }
        finally
        {
            if (File.Exists(pidFile)) File.Delete(pidFile);
        }
    }
}
