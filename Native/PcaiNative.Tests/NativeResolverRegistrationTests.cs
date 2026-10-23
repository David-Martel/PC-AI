using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace PcaiNative.Tests;

[Collection("NativeEnvironment")]
public class NativeResolverRegistrationTests
{
    [Fact]
    public void RegistrationIsPublishedForEachAssemblyAndRepeatedCallsAreIdempotent()
    {
        var first = new AssemblyLoadContext("resolver-owned-first", isCollectible: true);
        var second = new AssemblyLoadContext("resolver-owned-second", isCollectible: true);
        try
        {
            foreach (var context in new[] { first, second })
            {
                var assembly = context.LoadFromAssemblyPath(typeof(PcaiCore).Assembly.Location);
                NativeResolver.EnsureRegistered(assembly);
                NativeResolver.EnsureRegistered(assembly);
                Assert.Throws<InvalidOperationException>(() =>
                    NativeLibrary.SetDllImportResolver(assembly, (_, _, _) => IntPtr.Zero));
            }
        }
        finally
        {
            first.Unload();
            second.Unload();
        }
    }

    [Fact]
    public void FailedRegistrationDoesNotPublishAFalseSuccess()
    {
        // Keep resolver state isolated from the actual native integration fixture.
        var context = new AssemblyLoadContext("resolver-failure-fixture", isCollectible: true);
        try
        {
            var assembly = context.LoadFromAssemblyPath(typeof(PcaiCore).Assembly.Location);
            NativeLibrary.SetDllImportResolver(assembly, (_, _, _) => IntPtr.Zero);
            var resolver = assembly.GetType("PcaiNative.NativeResolver", throwOnError: true)!;
            var register = resolver.GetMethod("EnsureRegistered", new[] { typeof(Assembly) })!;
            // Both attempts must report the real registration conflict. Previously
            // the first failure left the published flag set and the second succeeded.
            for (var attempt = 0; attempt < 2; attempt++)
            {
                var error = Assert.Throws<TargetInvocationException>(() => register.Invoke(null, new object[] { assembly }));
                Assert.IsType<InvalidOperationException>(error.InnerException);
            }
        }
        finally
        {
            context.Unload();
        }
    }

    [Fact]
    public void CandidateDiscoveryDoesNotPublishASuccessfulLoadRecord()
    {
        using var fixture = new BundleFixture();
        File.WriteAllText(Path.Combine(fixture.Root, NativeResolver.GetExpectedFileName("pcai_core_lib")), "inert marker; never loaded");
        var context = new AssemblyLoadContext("resolver-discovery-only", isCollectible: true);
        try
        {
            var assembly = context.LoadFromAssemblyPath(typeof(PcaiCore).Assembly.Location);
            var resolver = assembly.GetType("PcaiNative.NativeResolver", throwOnError: true)!;
            object?[] discovery = { "pcai_core_lib", null, null, assembly };
            Assert.True((bool)resolver.GetMethod("TryResolveLibraryPath")!.Invoke(null, discovery)!);
            Assert.NotNull(discovery[1]);
            object?[] recorded = { "pcai_core_lib", null };
            Assert.False((bool)resolver.GetMethod("TryGetLoadedLibraryPath")!.Invoke(null, recorded)!);
            Assert.Null(recorded[1]);
        }
        finally { context.Unload(); }
    }

    [Theory]
    [InlineData("pcai_core_lib")]
    [InlineData("pcai_inference")]
    [InlineData("pcai_media")]
    public void ExplicitBundleSelectsPlatformFileAndNeverFallsBack(string logicalName)
    {
        using var fixture = new BundleFixture();
        var expected = Path.Combine(fixture.Root, NativeResolver.GetExpectedFileName(logicalName));
        File.WriteAllText(expected, "inert file marker; never loaded");
        Assert.True(NativeResolver.TryResolveLibraryPath(logicalName, out var actual, out var provenance));
        Assert.Equal(expected, actual);
        Assert.Equal("env:PCAI_NATIVE_BUNDLE_ROOT", provenance);
        File.Delete(expected);
        Assert.False(NativeResolver.TryResolveLibraryPath(logicalName, out actual, out provenance));
        Assert.Null(actual);
        Assert.Equal("env:PCAI_NATIVE_BUNDLE_ROOT:missing", provenance);
    }

    [Fact]
    public void ExplicitBundleRejectsAConflictingCoreOverride()
    {
        using var fixture = new BundleFixture();
        var expected = Path.Combine(fixture.Root, NativeResolver.GetExpectedFileName("pcai_core_lib"));
        File.WriteAllText(expected, "inert marker");
        Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", expected);
        Assert.True(NativeResolver.TryResolveLibraryPath("pcai_core_lib", out var actual, out _));
        Assert.Equal(expected, actual);
        Environment.SetEnvironmentVariable("PCAI_CORE_LIB_PATH", Path.Combine(fixture.Root, "foreign-core"));
        Assert.Throws<DllNotFoundException>(() => NativeResolver.TryResolveLibraryPath("pcai_core_lib", out _, out _));
    }

    private sealed class BundleFixture : IDisposable
    {
        private readonly Dictionary<string, string?> _saved = new();
        public string Root { get; } = Path.Combine(Path.GetTempPath(), "pcai-resolver-bundle-" + Guid.NewGuid().ToString("N"));

        public BundleFixture()
        {
            Directory.CreateDirectory(Root);
            foreach (var name in new[] { "PCAI_NATIVE_BUNDLE_ROOT", "PCAI_CORE_LIB_PATH", "PCAI_TEST_CORE_DLL", "PCAI_NATIVE_DIR" })
            {
                _saved[name] = Environment.GetEnvironmentVariable(name);
                Environment.SetEnvironmentVariable(name, null);
            }
            Environment.SetEnvironmentVariable("PCAI_NATIVE_BUNDLE_ROOT", Root);
        }

        public void Dispose()
        {
            foreach (var (name, value) in _saved) Environment.SetEnvironmentVariable(name, value);
            Directory.Delete(Root, recursive: true);
        }
    }
}
