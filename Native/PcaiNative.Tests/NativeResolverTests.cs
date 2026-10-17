using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace PcaiNative.Tests;

/// <summary>
/// Serializes tests that read or mutate process-wide native library override environment variables
/// (<c>PCAI_CORE_LIB_PATH</c>, <c>PCAI_TEST_CORE_DLL</c>, <c>PCAI_NATIVE_DIR</c>).
/// </summary>
[CollectionDefinition("NativeEnvironment", DisableParallelization = true)]
public sealed class NativeEnvironmentTestCollection
{
}

/// <summary>
/// Unit tests for <see cref="NativeResolver"/> cross-platform naming, RID mapping,
/// candidate path resolution, required-export admission, and fail-closed override handling.
/// </summary>
[Collection("NativeEnvironment")]
public sealed class NativeResolverTests
{
    [Theory]
    [InlineData("pcai_core_lib", "pcai_core_lib")]
    [InlineData("pcai_core_lib.dll", "pcai_core_lib")]
    [InlineData("libpcai_core_lib.so", "pcai_core_lib")]
    [InlineData("libpcai_core_lib.dylib", "pcai_core_lib")]
    [InlineData("PCAI_INFERENCE.DLL", "pcai_inference")]
    [InlineData("libpcai_media.so", "pcai_media")]
    [InlineData("", "")]
    [InlineData("   ", "")]
    [InlineData(null, "")]
    public void NormalizeLibraryNameStripsPlatformPrefixesAndExtensions(string? rawName, string expected)
    {
        Assert.Equal(expected, NativeResolver.NormalizeLibraryName(rawName));
    }

    [Fact]
    public void GetExpectedFileNameFormatsPlatformConventions()
    {
        Assert.Equal("pcai_core_lib.dll", NativeResolver.GetExpectedFileName("pcai_core_lib", OSPlatform.Windows));
        Assert.Equal("libpcai_core_lib.so", NativeResolver.GetExpectedFileName("pcai_core_lib.dll", OSPlatform.Linux));
        Assert.Equal("libpcai_core_lib.dylib", NativeResolver.GetExpectedFileName("libpcai_core_lib.so", OSPlatform.OSX));
    }

    [Fact]
    public void GetRuntimeIdentifierFormatsOsAndArchitecturePairs()
    {
        Assert.Equal("win-x64", NativeResolver.GetRuntimeIdentifier(OSPlatform.Windows, Architecture.X64));
        Assert.Equal("win-arm64", NativeResolver.GetRuntimeIdentifier(OSPlatform.Windows, Architecture.Arm64));
        Assert.Equal("linux-x64", NativeResolver.GetRuntimeIdentifier(OSPlatform.Linux, Architecture.X64));
        Assert.Equal("linux-arm64", NativeResolver.GetRuntimeIdentifier(OSPlatform.Linux, Architecture.Arm64));
        Assert.Equal("osx-x64", NativeResolver.GetRuntimeIdentifier(OSPlatform.OSX, Architecture.X64));
        Assert.Equal("osx-arm64", NativeResolver.GetRuntimeIdentifier(OSPlatform.OSX, Architecture.Arm64));

        var currentRid = NativeResolver.GetCurrentRuntimeIdentifier();
        Assert.Matches("^(win|linux|osx)-(x64|arm64|x86|arm)$", currentRid);
    }

    [Fact]
    public void TryResolveLibraryPathHonorsExplicitOverrideAndFailsClosedOnMissingOverride()
    {
        var prevCorePath = Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH");
        var prevTestDll = Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL");
        var prevNativeDir = Environment.GetEnvironmentVariable("PCAI_NATIVE_DIR");

        var tempDir = Path.Combine(Path.GetTempPath(), "pcai-resolver-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            var expectedFileName = NativeResolver.GetExpectedFileName("pcai_core_lib");
            var fakeArtifact = Path.Combine(tempDir, expectedFileName);
            File.WriteAllText(fakeArtifact, "stub");

            Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", fakeArtifact);
            Environment.SetEnvironmentVariable("PCAI_TEST_CORE_DLL", null);
            Environment.SetEnvironmentVariable("PCAI_NATIVE_DIR", null);

            Assert.True(NativeResolver.TryResolveLibraryPath("pcai_core_lib", out var resolved, out var provenance));
            Assert.Equal(Path.GetFullPath(fakeArtifact), resolved);
            Assert.Equal("env:PCAI_CORE_LIB_PATH", provenance);

            // An un-loadable stub at PCAI_CORE_LIB_PATH must fail closed in PcaiCore.IsAvailable.
            Assert.False(PcaiCore.IsAvailable);
            Assert.Equal("unavailable", PcaiCore.Version);

            // Missing explicit override must fail closed instead of falling back to another directory.
            var missingPath = Path.Combine(tempDir, "nonexistent-" + expectedFileName);
            Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", missingPath);
            Assert.False(NativeResolver.TryResolveLibraryPath("pcai_core_lib", out var missingResolved, out var missingProvenance));
            Assert.Null(missingResolved);
            Assert.Equal("env:PCAI_CORE_LIB_PATH:missing", missingProvenance);
            Assert.False(PcaiCore.IsAvailable);
            Assert.Equal("unavailable", PcaiCore.Version);

            // Missing PCAI_NATIVE_DIR must also fail closed.
            Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", null);
            Environment.SetEnvironmentVariable("PCAI_NATIVE_DIR", Path.Combine(tempDir, "missing-native-dir"));
            Assert.False(NativeResolver.TryResolveLibraryPath("pcai_core_lib", out var missingDirResolved, out var missingDirProvenance));
            Assert.Null(missingDirResolved);
            Assert.Equal("env:PCAI_NATIVE_DIR:missing", missingDirProvenance);
            Assert.False(PcaiCore.IsAvailable);
            Assert.Equal("unavailable", PcaiCore.Version);

            // Unknown library names must be rejected.
            Assert.False(NativeResolver.TryResolveLibraryPath("unknown_lib", out _, out _));
        }
        finally
        {
            Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", prevCorePath);
            Environment.SetEnvironmentVariable("PCAI_TEST_CORE_DLL", prevTestDll);
            Environment.SetEnvironmentVariable("PCAI_NATIVE_DIR", prevNativeDir);
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void RequiredExportsCoverAllNativeCoreDllImports()
    {
        var nativeCoreType = typeof(PcaiCore).Assembly.GetType("PcaiNative.NativeCore", throwOnError: true)!;
        var declaredExports = nativeCoreType
            .GetMethods(BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            .Where(m => m.GetCustomAttribute<DllImportAttribute>() is { Value: "pcai_core_lib" })
            .Select(m => m.GetCustomAttribute<DllImportAttribute>()!.EntryPoint is { Length: > 0 } ep ? ep : m.Name)
            .ToHashSet(StringComparer.Ordinal);

        Assert.NotEmpty(declaredExports);

        var classifiedExports = PcaiCore.RequiredExports
            .Concat(PcaiCore.OptionalFeatureExports)
            .ToHashSet(StringComparer.Ordinal);

        var unclassified = declaredExports.Except(classifiedExports).OrderBy(x => x).ToArray();
        var stale = classifiedExports.Except(declaredExports).OrderBy(x => x).ToArray();

        Assert.True(unclassified.Length == 0,
            $"NativeCore declares pcai_core_lib P/Invoke symbols not classified in PcaiCore.RequiredExports / OptionalFeatureExports: {string.Join(", ", unclassified)}");
        Assert.True(stale.Length == 0,
            $"PcaiCore lists exports no longer declared in NativeCore: {string.Join(", ", stale)}");
        Assert.Empty(PcaiCore.RequiredExports.Intersect(PcaiCore.OptionalFeatureExports, StringComparer.Ordinal));

        Assert.False(PcaiCore.VerifyRequiredExports(IntPtr.Zero, out var nullMissing));
        Assert.Equal(PcaiCore.RequiredExports.Count, nullMissing.Count);
    }

    [Fact]
    public void MagicOnlyNativeLibraryIsRejectedByRequiredExportAdmission()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "pcai-magic-only-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            var expectedFileName = NativeResolver.GetExpectedFileName("pcai_core_lib");
            var magicRustSource = Path.Combine(tempDir, "magic_only.rs");
            var magicLibPath = Path.Combine(tempDir, expectedFileName);

            // Compile a real loadable native shared library that exports ONLY pcai_core_test
            // (returning the valid 0x50434149 magic number) and pcai_core_version, omitting
            // pcai_string_copy and the remaining required C ABI exports.
            File.WriteAllText(magicRustSource, """
                #[no_mangle]
                pub extern "C" fn pcai_core_test() -> u32 {
                    0x5043_4149
                }

                #[no_mangle]
                pub extern "C" fn pcai_core_version() -> *const u8 {
                    b"0.1.0-magic-only\0".as_ptr()
                }
                """);

            var rustc = new ProcessStartInfo
            {
                FileName = "rustc",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            rustc.ArgumentList.Add("--crate-type");
            rustc.ArgumentList.Add("cdylib");
            rustc.ArgumentList.Add(magicRustSource);
            rustc.ArgumentList.Add("-o");
            rustc.ArgumentList.Add(magicLibPath);

            using (var compileProc = Process.Start(rustc))
            {
                Assert.NotNull(compileProc);
                var compileOut = compileProc!.StandardOutput.ReadToEnd();
                var compileErr = compileProc.StandardError.ReadToEnd();
                Assert.True(compileProc.WaitForExit(60000), "rustc timed out compiling magic-only stub.");
                Assert.True(compileProc.ExitCode == 0, $"rustc failed ({compileProc.ExitCode}): {compileErr}\n{compileOut}");
            }

            Assert.True(File.Exists(magicLibPath));

            // 1. Direct export admission check must reject the magic-only shared library and identify missing exports.
            Assert.False(PcaiCore.VerifyLibraryExports(magicLibPath, out var missingExports));
            Assert.Contains("pcai_string_copy", missingExports);
            Assert.Contains("pcai_extract_json", missingExports);
            Assert.Contains("pcai_estimate_tokens", missingExports);
            Assert.DoesNotContain("pcai_core_test", missingExports);
            Assert.DoesNotContain("pcai_core_version", missingExports);

            // 2. If a real pcai_core_lib artifact is configured in the test environment, verify it passes all 57 required exports.
            var configuredRealCore = Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH")
                ?? Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL");
            if (!string.IsNullOrWhiteSpace(configuredRealCore) && File.Exists(configuredRealCore))
            {
                Assert.True(PcaiCore.VerifyLibraryExports(configuredRealCore, out var realMissing),
                    $"Real pcai_core_lib failed export verification; missing: {string.Join(", ", realMissing)}");
                Assert.Empty(realMissing);
            }

            // 3. In a fresh process where PCAI_CORE_LIB_PATH points to the magic-only library,
            // PcaiCore.IsAvailable (and all sibling module availability properties) must be false,
            // and PcaiCore.TestStringCopy must return null cleanly without throwing EntryPointNotFoundException.
            var wrapperAssemblyPath = typeof(NativeResolver).Assembly.Location;
            Assert.True(File.Exists(wrapperAssemblyPath));

            var psi = new ProcessStartInfo
            {
                FileName = "pwsh",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.Environment["PCAI_CORE_LIB_PATH"] = magicLibPath;
            psi.Environment.Remove("PCAI_TEST_CORE_DLL");
            psi.Environment.Remove("PCAI_NATIVE_DIR");
            psi.ArgumentList.Add("-NoLogo");
            psi.ArgumentList.Add("-NoProfile");
            psi.ArgumentList.Add("-Command");
            psi.ArgumentList.Add($@"
$ErrorActionPreference = 'Stop'
Add-Type -Path '{wrapperAssemblyPath.Replace("'", "''")}'
$coreAvail = [PcaiNative.PcaiCore]::IsAvailable
$searchAvail = [PcaiNative.PcaiSearch]::IsAvailable
$sysAvail = [PcaiNative.SystemModule]::IsAvailable
$perfAvail = [PcaiNative.PerformanceModule]::IsAvailable
$optAvail = [PcaiNative.OptimizerModule]::IsAvailable
$fsAvail = [PcaiNative.FsModule]::IsAvailable
$ver = [PcaiNative.PcaiCore]::Version
$copyResult = [PcaiNative.PcaiCore]::TestStringCopy('hello')
$diag = [PcaiNative.PcaiCore]::GetDiagnostics()
[pscustomobject]@{{
    CoreAvailable = [bool]$coreAvail
    SearchAvailable = [bool]$searchAvail
    SystemAvailable = [bool]$sysAvail
    PerformanceAvailable = [bool]$perfAvail
    OptimizerAvailable = [bool]$optAvail
    FsAvailable = [bool]$fsAvail
    Version = $ver
    CopyIsNull = ($null -eq $copyResult)
    ResolvedPath = $diag.ResolvedLibraryPath
    Provenance = $diag.ProvenanceSource
}} | ConvertTo-Json -Compress
");

            using var child = Process.Start(psi);
            Assert.NotNull(child);
            var stdout = child!.StandardOutput.ReadToEnd();
            var stderr = child.StandardError.ReadToEnd();
            Assert.True(child.WaitForExit(30000), "Child pwsh process timed out.");
            Assert.True(child.ExitCode == 0, $"Child pwsh failed ({child.ExitCode}): {stderr}\n{stdout}");

            var jsonLine = stdout
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .LastOrDefault(line => line.TrimStart().StartsWith("{", StringComparison.Ordinal));
            Assert.False(string.IsNullOrWhiteSpace(jsonLine), $"Expected JSON output from child process, got: {stdout}");

            using var doc = JsonDocument.Parse(jsonLine!);
            var root = doc.RootElement;
            Assert.False(root.GetProperty("CoreAvailable").GetBoolean());
            Assert.False(root.GetProperty("SearchAvailable").GetBoolean());
            Assert.False(root.GetProperty("SystemAvailable").GetBoolean());
            Assert.False(root.GetProperty("PerformanceAvailable").GetBoolean());
            Assert.False(root.GetProperty("OptimizerAvailable").GetBoolean());
            Assert.False(root.GetProperty("FsAvailable").GetBoolean());
            Assert.Equal("unavailable", root.GetProperty("Version").GetString());
            Assert.True(root.GetProperty("CopyIsNull").GetBoolean());
            Assert.Equal(JsonValueKind.Null, root.GetProperty("ResolvedPath").ValueKind);
            Assert.Equal(JsonValueKind.Null, root.GetProperty("Provenance").ValueKind);
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }
}
