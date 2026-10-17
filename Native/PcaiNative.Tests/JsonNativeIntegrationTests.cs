using System.IO;

namespace PcaiNative.Tests;

/// <summary>
/// Exercises the managed PcaiCore P/Invoke boundary through the production <see cref="NativeResolver"/>
/// against an explicitly selected native library (<c>PCAI_TEST_CORE_DLL</c> or <c>PCAI_CORE_LIB_PATH</c>).
/// </summary>
public class JsonNativeIntegrationTests
{
    private static string RequireExplicitCorePath()
    {
        var path = Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH");
        if (string.IsNullOrWhiteSpace(path))
            path = Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL");

        Assert.False(string.IsNullOrWhiteSpace(path));
        Assert.True(Path.IsPathFullyQualified(path!), "Explicit native core path must be an absolute path.");
        Assert.True(File.Exists(path!), $"Explicit native core path does not exist: {path}");
        return Path.GetFullPath(path!);
    }

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
    public void IsValidJsonMatchesNativeParserThroughProductionResolver(string? input, bool expected)
    {
        var expectedPath = RequireExplicitCorePath();
        Assert.True(PcaiCore.IsAvailable, "The selected native core library must load via NativeResolver and return the magic number.");
        Assert.Equal(expectedPath, NativeResolver.GetResolvedLibraryPath("pcai_core_lib"));
        Assert.Equal(expected, PcaiCore.IsValidJson(input!));
    }

    [NativeCoreTheory]
    [Trait("Category", "NativeIntegration")]
    [InlineData("plain ascii")]
    [InlineData("caf\u00e9 \u4f60\u597d \ud83d\ude80")]
    [InlineData("{\"nested\": [1, 2, 3], \"label\": \"\u03bb\"}")]
    public void StringCopyAndExtractJsonRoundTripUtf8AndFreeBuffers(string sample)
    {
        _ = RequireExplicitCorePath();
        Assert.True(PcaiCore.IsAvailable);

        var copied = PcaiCore.TestStringCopy(sample);
        Assert.Equal(sample, copied);

        const string markdown = "Here is payload:\n```json\n{\"status\":\"ok\",\"value\":42}\n```\nDone.";
        var extracted = PcaiCore.ExtractJson(markdown);
        Assert.Equal("{\"status\":\"ok\",\"value\":42}", extracted);

        var diagnostics = PcaiCore.GetDiagnostics();
        Assert.True(diagnostics.IsAvailable);
        Assert.Equal(NativeResolver.GetCurrentRuntimeIdentifier(), diagnostics.RuntimeIdentifier);
        Assert.False(string.IsNullOrWhiteSpace(diagnostics.ResolvedLibraryPath));
        Assert.False(string.IsNullOrWhiteSpace(diagnostics.ProvenanceSource));
    }

    [NativeCoreTheory]
    [Trait("Category", "NativeIntegration")]
    [InlineData("", 0UL)]
    [InlineData("Hello world", 4UL)]
    [InlineData("This is a test.", 9UL)]
    [InlineData("\u4f60\u597d\u4e16\u754c", 4UL)]
    [InlineData("internationalization", 6UL)]
    public void EstimateTokensMatchesNativeSpecification(string input, ulong expected)
    {
        _ = RequireExplicitCorePath();
        Assert.True(PcaiCore.IsAvailable);
        Assert.Equal(expected, PcaiCore.EstimateTokens(input));
    }
}

/// <summary>
/// Requires an explicit native fixture; absent fixtures report skipped tests.
/// </summary>
public sealed class NativeCoreTheoryAttribute : TheoryAttribute
{
    public NativeCoreTheoryAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PCAI_TEST_CORE_DLL")) &&
            string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PCAI_CORE_LIB_PATH")))
        {
            Skip = "Set PCAI_TEST_CORE_DLL or PCAI_CORE_LIB_PATH to the absolute path of a freshly built pcai_core_lib native library.";
        }
    }
}

