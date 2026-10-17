using System;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;

namespace PcaiNative.Tests;

/// <summary>
/// Unit tests for <see cref="NativeResolver"/> cross-platform naming, RID mapping,
/// candidate path resolution, and fail-closed override handling.
/// </summary>
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

            // Missing explicit override must fail closed instead of falling back to another directory.
            var missingPath = Path.Combine(tempDir, "nonexistent-" + expectedFileName);
            Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", missingPath);
            Assert.False(NativeResolver.TryResolveLibraryPath("pcai_core_lib", out var missingResolved, out var missingProvenance));
            Assert.Null(missingResolved);
            Assert.Equal("env:PCAI_CORE_LIB_PATH:missing", missingProvenance);

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
}
