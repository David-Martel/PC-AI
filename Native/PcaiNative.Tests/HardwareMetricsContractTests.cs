using System.Text.Json;

namespace PcaiNative.Tests;

public sealed class HardwareMetricsContractTests
{
    [Fact]
    public void WarmupPreservesUnavailableCpuMeasurements()
    {
        const string json = """
            {"cpu_usage":[null,null],"avg_load":null,"cpu_sample_status":"WarmingUp",
             "cpu_sample_age_milliseconds":7,"temps":[]}
            """;

        var metrics = Assert.IsType<PcaiMetrics>(JsonSerializer.Deserialize<PcaiMetrics>(json));

        Assert.Null(metrics.CpuUsage);
        Assert.Equal(new float?[] { null, null }, metrics.CpuUsagePerCore);
        Assert.Equal("WarmingUp", metrics.CpuSampleStatus);
        Assert.Equal((UInt128)7, metrics.CpuSampleAgeMilliseconds);
        Assert.Empty(Assert.IsType<PcaiComponentTemperature[]>(metrics.Temperatures));
        AssertLegacyMeasurementsUnknown(metrics);
    }

    [Fact]
    public void SampledZeroIsAnObservedValueRatherThanUnknown()
    {
        const string json = """
            {"cpu_usage":[0,0],"avg_load":0,"cpu_sample_status":"Sampled",
             "cpu_sample_age_milliseconds":0,"temps":[]}
            """;

        var metrics = Assert.IsType<PcaiMetrics>(JsonSerializer.Deserialize<PcaiMetrics>(json));

        Assert.Equal(0f, metrics.CpuUsage);
        Assert.Equal(new float?[] { 0f, 0f }, metrics.CpuUsagePerCore);
        Assert.Equal("Sampled", metrics.CpuSampleStatus);
        Assert.Equal((UInt128)0, metrics.CpuSampleAgeMilliseconds);
        AssertLegacyMeasurementsUnknown(metrics);
    }

    [Fact]
    public void MissingMetricsRemainUnknownInsteadOfDefaultingToZero()
    {
        var metrics = Assert.IsType<PcaiMetrics>(JsonSerializer.Deserialize<PcaiMetrics>("{}"));

        Assert.Null(metrics.CpuUsage);
        Assert.Null(metrics.CpuUsagePerCore);
        Assert.Null(metrics.CpuSampleStatus);
        Assert.Null(metrics.CpuSampleAgeMilliseconds);
        Assert.Null(metrics.Temperatures);
        AssertLegacyMeasurementsUnknown(metrics);
    }

    [Fact]
    public void TemperaturesPreserveLabelsAndUnavailableAndRealReadings()
    {
        const string json = """
            {"cpu_usage":[null,25.5],"avg_load":25.5,"cpu_sample_status":"Sampled",
             "cpu_sample_age_milliseconds":18446744073709551616,
             "temps":[{"label":"Unavailable sensor","celsius":null},
                      {"label":"CPU package","celsius":42.75},
                      {"label":"Zero-degree sensor","celsius":0}]}
            """;

        var metrics = Assert.IsType<PcaiMetrics>(JsonSerializer.Deserialize<PcaiMetrics>(json));

        Assert.Equal(25.5f, metrics.CpuUsage);
        Assert.Equal(new float?[] { null, 25.5f }, metrics.CpuUsagePerCore);
        Assert.Equal(UInt128.Parse("18446744073709551616"), metrics.CpuSampleAgeMilliseconds);
        var temperatures = Assert.IsType<PcaiComponentTemperature[]>(metrics.Temperatures);
        Assert.Collection(temperatures,
            unavailable => { Assert.Equal("Unavailable sensor", unavailable.Label); Assert.Null(unavailable.Celsius); },
            observed => { Assert.Equal("CPU package", observed.Label); Assert.Equal(42.75f, observed.Celsius); },
            zero => { Assert.Equal("Zero-degree sensor", zero.Label); Assert.Equal(0f, zero.Celsius); });

        using var roundTrip = JsonDocument.Parse(JsonSerializer.Serialize(metrics));
        Assert.Equal(25.5f, roundTrip.RootElement.GetProperty("avg_load").GetSingle());
        Assert.False(roundTrip.RootElement.TryGetProperty("cpu_usage_perc", out _));
        Assert.Equal(JsonValueKind.Null, roundTrip.RootElement.GetProperty("temps")[0].GetProperty("celsius").ValueKind);
        AssertLegacyMeasurementsUnknown(metrics);
    }

    [Fact]
    public void SuppliedLegacyMeasurementsRetainTheirValues()
    {
        const string json = """
            {"memory_usage_bytes":1024,"total_memory_bytes":4096,"gpu_usage_perc":0,"uptime_seconds":0}
            """;

        var metrics = Assert.IsType<PcaiMetrics>(JsonSerializer.Deserialize<PcaiMetrics>(json));

        Assert.Equal((ulong)1024, metrics.MemoryUsage);
        Assert.Equal((ulong)4096, metrics.TotalMemory);
        Assert.Equal(0f, metrics.GpuUsage);
        Assert.Equal((ulong)0, metrics.Uptime);
        Assert.Null(metrics.CpuUsage);
    }

    private static void AssertLegacyMeasurementsUnknown(PcaiMetrics metrics)
    {
        Assert.Null(metrics.MemoryUsage);
        Assert.Null(metrics.TotalMemory);
        Assert.Null(metrics.GpuUsage);
        Assert.Null(metrics.Uptime);
    }
}
