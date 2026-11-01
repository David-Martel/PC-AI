using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Xunit;

namespace PcaiNative.Tests;

/// <summary>Exercises resolver admission through public diagnostics in fresh processes.</summary>
/// <remarks>
/// Synthetic full-symbol fixtures qualify export inventory, magic/string round-trip
/// and CPU/status selection only. Uncalled placeholders implement no production ABI.
/// </remarks>
public sealed class NativeResolverPartialCoreAdmissionTests
{
    // Group: all eight PublicDiagnosticsAdmitOnlyCompleteCoreSymbols rows below.
    // Protects: full required-export admission before native CPU/status calls or caching.
    // Detects: rejected partial cores returning their sentinel through public diagnostics,
    //          resolver publication before verification, leaked rejected Windows modules,
    //          or loss of the specific missing-export exception.
    // Needs: rustc, native linker, pwsh, fresh child processes and owned temporary files;
    //        no live core, model, GPU, network request or hardware workload.
    // Breadcrumb: Native/PcaiNative/NativeResolver.cs::Resolve/VerifyLoadedCoreExports;
    //             Native/PcaiNative/PcaiCore.cs::CpuCount/GetDiagnostics/GetStatusDescription.
    [Theory]
    [InlineData("cpu", false)]
    [InlineData("diagnostics", false)]
    [InlineData("status", false)]
    [InlineData("resolver", false)]
    [InlineData("cpu", true)]
    [InlineData("diagnostics", true)]
    [InlineData("status", true)]
    [InlineData("resolver", true)]
    public async Task PublicDiagnosticsAdmitOnlyCompleteCoreSymbols(string mode, bool fullSymbols)
    {
        var tempRoot = Path.GetFullPath(Path.GetTempPath());
        var directory = Path.GetFullPath(Path.Combine(tempRoot, "pcai-resolver-admission-" + Guid.NewGuid().ToString("N")));
        Directory.CreateDirectory(directory);
        Exception? testFailure = null;
        try
        {
            var sourcePath = Path.Combine(directory, "owned_fixture.rs");
            var libraryPath = Path.Combine(directory, NativeResolver.GetExpectedFileName("pcai_core_lib"));
            await File.WriteAllTextAsync(sourcePath, BuildOwnedFixture(fullSymbols));
            var compiler = StartInfo("rustc");
            foreach (var value in new[] { "--edition", "2021", "--crate-type", "cdylib", sourcePath, "-o", libraryPath })
                compiler.ArgumentList.Add(value);
            var compiled = await BoundedTestProcess.RunAsync(compiler, TimeSpan.FromSeconds(60));
            Assert.True(compiled.ExitCode == 0,
                $"Owned Rust fixture failed ({compiled.ExitCode}): {compiled.StandardError}\n{compiled.StandardOutput}");

            var child = StartInfo("pwsh");
            child.Environment["PCAI_CORE_LIB_PATH"] = libraryPath;
            foreach (var name in new[] { "PCAI_TEST_CORE_DLL", "PCAI_NATIVE_DIR", "PCAI_NATIVE_BUNDLE_ROOT" })
                child.Environment.Remove(name);
            var assembly = typeof(PcaiCore).Assembly.Location.Replace("'", "''");
            var script = $$"""
                $ErrorActionPreference = 'Stop'
                Add-Type -Path '{{assembly}}'
                [PcaiNative.NativeResolver]::EnsureRegistered()
                $cpu = $null; $status = $null; $available = $null; $errorText = $null
                try {
                    switch ('{{mode}}') {
                        'cpu' { $cpu = [PcaiNative.PcaiCore]::CpuCount }
                        'diagnostics' {
                            $diagnostics = [PcaiNative.PcaiCore]::GetDiagnostics()
                            $cpu = $diagnostics.CpuCount; $available = $diagnostics.IsAvailable
                        }
                        'status' { $status = [PcaiNative.PcaiCore]::GetStatusDescription([PcaiNative.PcaiStatus]::Success) }
                        'resolver' {
                            $callback = [PcaiNative.NativeResolver].GetMethod('Resolve', [Reflection.BindingFlags]'NonPublic,Static')
                            $null = $callback.Invoke($null, [object[]]@('pcai_core_lib', [PcaiNative.PcaiCore].Assembly, $null))
                        }
                    }
                } catch {
                    $failure = $_.Exception.GetBaseException()
                    $errorText = $failure.GetType().Name + ': ' + $failure.Message
                }
                $records = [PcaiNative.NativeResolver].GetField('ResolvedPaths', [Reflection.BindingFlags]'NonPublic,Static').GetValue($null)
                $loaded = $null
                if ($IsWindows) {
                    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class OwnedModuleProbe { [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr GetModuleHandleW(string path); }'
                    $loaded = [OwnedModuleProbe]::GetModuleHandleW($env:PCAI_CORE_LIB_PATH) -ne [IntPtr]::Zero
                }
                [pscustomobject]@{
                    Cpu = $cpu; Status = $status; Available = $available; ErrorText = $errorText
                    RecordCount = $records.Count; ModuleLoaded = $loaded; ProcessorCount = [Environment]::ProcessorCount
                } | ConvertTo-Json -Compress
                """;
            foreach (var value in new[] { "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script })
                child.ArgumentList.Add(value);
            var result = await BoundedTestProcess.RunAsync(child, TimeSpan.FromSeconds(30));
            Assert.True(result.ExitCode == 0, $"Owned child failed ({result.ExitCode}): {result.StandardError}\n{result.StandardOutput}");
            var json = result.StandardOutput.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .LastOrDefault(line => line.TrimStart().StartsWith("{", StringComparison.Ordinal));
            Assert.False(string.IsNullOrWhiteSpace(json), $"No diagnostic record: {result.StandardOutput}");
            using var document = JsonDocument.Parse(json!);
            var record = document.RootElement;

            Assert.Equal(fullSymbols ? 1 : 0, record.GetProperty("RecordCount").GetInt32());
            if (OperatingSystem.IsWindows())
                Assert.Equal(fullSymbols, record.GetProperty("ModuleLoaded").GetBoolean());
            if (mode is "cpu" or "diagnostics")
                Assert.Equal(fullSymbols ? 424242U : record.GetProperty("ProcessorCount").GetUInt32(),
                    record.GetProperty("Cpu").GetUInt32());
            if (mode == "diagnostics")
                Assert.Equal(fullSymbols, record.GetProperty("Available").GetBoolean());
            if (mode == "status")
                Assert.Equal(fullSymbols ? "OWNED-FULL-CORE" : PcaiStatus.Success.ToString(), record.GetProperty("Status").GetString());
            if (mode == "resolver" && !fullSymbols)
                Assert.Equal($"EntryPointNotFoundException: Native library '{libraryPath}' (env:PCAI_CORE_LIB_PATH) is missing required export 'pcai_free_string'.",
                    record.GetProperty("ErrorText").GetString());
            else
                Assert.Equal(JsonValueKind.Null, record.GetProperty("ErrorText").ValueKind);
        }
        catch (Exception error)
        {
            testFailure = error;
            throw;
        }
        finally
        {
            // BoundedTestProcess has reaped every owned child before fixture removal.
            try { RemoveOwnedFixture(directory, tempRoot); }
            catch (Exception cleanupError) when (testFailure is not null)
            {
                throw new AggregateException("Native admission test and owned fixture cleanup both failed.", testFailure, cleanupError);
            }
        }
    }

    private static void RemoveOwnedFixture(string directory, string tempRoot)
    {
        var resolved = Path.GetFullPath(directory);
        var containedRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(tempRoot)) + Path.DirectorySeparatorChar;
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!resolved.StartsWith(containedRoot, comparison) ||
            !Path.GetFileName(resolved).StartsWith("pcai-resolver-admission-", StringComparison.Ordinal))
            throw new IOException("Refusing cleanup outside the captured temporary fixture root.");
        var root = new DirectoryInfo(resolved);
        if (!root.Exists) return;
        if ((root.Attributes & FileAttributes.ReparsePoint) != 0)
            throw new IOException("Refusing cleanup of a replaced or linked fixture root.");
        Directory.Delete(resolved, recursive: true);
    }

    private static ProcessStartInfo StartInfo(string executable) => new()
    {
        FileName = executable,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };

    private static string BuildOwnedFixture(bool fullSymbols)
    {
        const string partial = """
            use std::ffi::c_char;
            #[no_mangle] pub extern "C" fn pcai_core_test() -> u32 { 0x50434149 }
            #[no_mangle] pub extern "C" fn pcai_core_version() -> *const u8 { b"owned-fixture\0".as_ptr() }
            #[no_mangle] pub extern "C" fn pcai_cpu_count() -> u32 { 424242 }
            #[no_mangle] pub extern "C" fn pcai_status_description(_status: i32) -> *const u8 { b"OWNED-FULL-CORE\0".as_ptr() }
            """;
        if (!fullSymbols) return partial;
        const string stringRoundTrip = """
            #[no_mangle] pub unsafe extern "C" fn pcai_string_copy(input: *const c_char) -> *mut c_char {
                if input.is_null() { return std::ptr::null_mut(); }
                std::ffi::CStr::from_ptr(input).to_owned().into_raw()
            }
            #[no_mangle] pub unsafe extern "C" fn pcai_free_string(input: *mut c_char) {
                if !input.is_null() { drop(std::ffi::CString::from_raw(input)); }
            }
            """;
        var implemented = new HashSet<string>(StringComparer.Ordinal)
        {
            "pcai_core_test", "pcai_core_version", "pcai_cpu_count", "pcai_status_description",
            "pcai_string_copy", "pcai_free_string",
        };
        // Derive inventory-only placeholders from the maintained production list.
        // No second required-export list or actual native workload is introduced.
        var placeholders = PcaiCore.RequiredExports.Where(name => !implemented.Contains(name))
            .Select(name => $"#[no_mangle] pub extern \"C\" fn {name}() -> usize {{ 0 }}");
        return partial + "\n" + stringRoundTrip + "\n" + string.Join("\n", placeholders) + "\n";
    }
}
