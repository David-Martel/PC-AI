using System.Runtime.InteropServices;

namespace PcaiNative.Tests;

/// <summary>Checks optional-feature errors against an explicitly selected CPU-only DLL.</summary>
public class MediaFeatureIntegrationTests
{
    [CpuMediaFact]
    [Trait("Category", "MediaNativeIntegration")]
    public void CpuLibraryReportsUnsupportedUpscalingWithoutWritingOutputs()
    {
        var bundle = Environment.GetEnvironmentVariable("PCAI_NATIVE_BUNDLE_ROOT");
        Assert.False(string.IsNullOrWhiteSpace(bundle));
        var library = Path.Combine(bundle!, "pcai_media.dll");
        var handle = NativeLibrary.Load(library);
        try
        {
            Assert.False(NativeLibrary.TryGetExport(handle, "pcai_media_upscale_image", out _),
                "This fixture requires a real CPU DLL without the optional upscale feature.");
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
        var output = Path.Combine(Path.GetTempPath(), "pcai-media-feature-" + Guid.NewGuid(), "output.png");
        var error = MediaModule.UpscaleImage("missing-model.onnx", "missing-image.png", output);
        Assert.Contains("does not include the upscale feature", error);
        Assert.False(Directory.Exists(Path.GetDirectoryName(output)));
        Assert.False(File.Exists(output));
    }
}

/// <summary>Requires a fresh CPU DLL fixture rather than an installed native library.</summary>
public sealed class CpuMediaFactAttribute : FactAttribute
{
    public CpuMediaFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("PCAI_TEST_CPU_MEDIA") != "1")
            Skip = "Set PCAI_TEST_CPU_MEDIA=1 and PCAI_NATIVE_BUNDLE_ROOT to a freshly built CPU-only media bundle.";
    }
}
