using System.Runtime.InteropServices;

namespace PcaiNative.Tests;

/// <summary>
/// Exercises the managed JSON validator against an explicitly selected Rust DLL.
/// </summary>
public class JsonNativeIntegrationTests
{
    // Native integration is opt-in so metadata tests remain portable. An explicit
    // path prevents an installed/stale DLL from accidentally satisfying the test.
    private static readonly Lazy<IntPtr> CoreLibrary = new(() =>
    {
        var path = Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL");
        Assert.False(string.IsNullOrWhiteSpace(path));
        Assert.True(Path.IsPathFullyQualified(path!), "PCAI_TEST_CORE_DLL must be an absolute path.");
        var handle = NativeLibrary.Load(path!);
        NativeLibrary.SetDllImportResolver(typeof(PcaiCore).Assembly,
            (name, _, _) => name == "pcai_core_lib.dll" ? handle : IntPtr.Zero);
        return handle;
    });

    [NativeCoreTheory]
    [Trait("Category", "NativeIntegration")]
    [InlineData("{}", true)]
    [InlineData("[1,true,null]", true)]
    [InlineData("null", true)]
    [InlineData("false", true)]
    [InlineData("\"caf\u00e9\"", true)]
    [InlineData("{broken", false)]
    [InlineData("{}garbage", false)]
    [InlineData("", false)]
    [InlineData("   ", false)]
    [InlineData(null, false)]
    public void IsValidJsonMatchesNativeParser(string? input, bool expected)
    {
        _ = CoreLibrary.Value;
        Assert.True(PcaiCore.IsAvailable, "The selected Rust DLL must load and return the core magic number.");
        Assert.Equal(expected, PcaiCore.IsValidJson(input!));
    }
}

/// <summary>
/// Requires an explicit native fixture; absent fixtures report skipped tests.
/// </summary>
public sealed class NativeCoreTheoryAttribute : TheoryAttribute
{
    public NativeCoreTheoryAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL")))
            Skip = "Set PCAI_TEST_CORE_DLL to the absolute path of a freshly built pcai_core_lib DLL.";
    }
}
