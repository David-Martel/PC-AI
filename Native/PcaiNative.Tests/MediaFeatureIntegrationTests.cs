using System.Runtime.InteropServices;

namespace PcaiNative.Tests;

/// <summary>Checks optional-feature errors against an explicitly selected CPU-only DLL.</summary>
[Collection("NativeEnvironment")]
public class MediaFeatureIntegrationTests
{
    [CpuMediaFact]
    [Trait("Category", "MediaNativeIntegration")]
    public void CpuLibraryReportsUnsupportedUpscalingWithoutWritingOutputs()
    {
        var selected = Environment.GetEnvironmentVariable("PCAI_TEST_MEDIA_DLL");
        Assert.False(string.IsNullOrWhiteSpace(selected));
        var library = Path.GetFullPath(selected!);
        Assert.True(File.Exists(library), "The explicitly admitted CPU media artifact must exist.");
        Assert.Equal(NativeResolver.GetExpectedFileName("pcai_media"), Path.GetFileName(library));
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
        var previousBundle = Environment.GetEnvironmentVariable("PCAI_NATIVE_BUNDLE_ROOT");
        var previousDirectory = Environment.GetEnvironmentVariable("PCAI_NATIVE_DIR");
        try
        {
            // Select only this real media artifact without imposing a different
            // bundle/core selection on the other serialized native controls.
            Environment.SetEnvironmentVariable("PCAI_NATIVE_BUNDLE_ROOT", null);
            Environment.SetEnvironmentVariable("PCAI_NATIVE_DIR", Path.GetDirectoryName(library));
            var output = Path.Combine(Path.GetTempPath(), "pcai-media-feature-" + Guid.NewGuid(), "output.png");
            var error = MediaModule.UpscaleImage("missing-model.onnx", "missing-image.png", output);
            Assert.Contains("does not include the upscale feature", error);
            Assert.True(NativeResolver.TryGetLoadedLibraryPath("pcai_media", out var loaded));
            Assert.True(string.Equals(library, loaded,
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal),
                "The managed wrapper must bind the explicitly admitted CPU media artifact.");
            Assert.False(Directory.Exists(Path.GetDirectoryName(output)));
            Assert.False(File.Exists(output));
        }
        finally
        {
            Environment.SetEnvironmentVariable("PCAI_NATIVE_DIR", previousDirectory);
            Environment.SetEnvironmentVariable("PCAI_NATIVE_BUNDLE_ROOT", previousBundle);
        }
    }
}

/// <summary>Requires a fresh CPU DLL fixture rather than an installed native library.</summary>
public sealed class CpuMediaFactAttribute : FactAttribute
{
    public CpuMediaFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("PCAI_TEST_CPU_MEDIA") != "1")
            Skip = "Set PCAI_TEST_CPU_MEDIA=1 and PCAI_TEST_MEDIA_DLL to a freshly built CPU-only media artifact.";
    }
}
