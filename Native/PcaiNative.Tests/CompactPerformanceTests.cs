using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace PcaiNative.Tests;

public class CompactPerformanceTests
{
    [Fact]
    public void ProcessEntryMatchesNativeCAbi()
    {
        Assert.Equal(72, Marshal.SizeOf<ProcessListCompactHeader>());
        Assert.Equal(48, Marshal.SizeOf<ProcessListCompactEntry>());
        Assert.Equal(40, Marshal.OffsetOf<ProcessListCompactEntry>(nameof(ProcessListCompactEntry.MemoryBytes)).ToInt32());
    }

    [Fact]
    public void DirectParserReadsNativeAlignedEntryWithoutJsonFallback()
    {
        var strings = Encoding.UTF8.GetBytes("cpucaféRunningC:/selected.exe");
        var bytes = new byte[72 + 48 + strings.Length];
        Write(bytes, 0, (int)PcaiStatus.Success);
        Write(bytes, 48, 0u);
        Write(bytes, 52, 3u);
        Write(bytes, 56, 1ul);
        Write(bytes, 64, (ulong)strings.Length);
        Write(bytes, 72, 12345u);
        Write(bytes, 76, 3u);
        Write(bytes, 80, 5u);
        Write(bytes, 84, 8u);
        Write(bytes, 88, 7u);
        Write(bytes, 92, 15u);
        Write(bytes, 96, 15u);
        Write(bytes, 100, 37.5f);
        // Deliberately distinct padding detects a wrong memory offset.
        Write(bytes, 108, 0x55667788u);
        Write(bytes, 112, 0x1122334455667788ul);
        strings.CopyTo(bytes, 120);
        var parsed = Assert.IsType<TopProcessesResult>(Parse("ParseCompactTopProcesses", bytes));
        var row = Assert.Single(parsed.Processes);
        Assert.Equal("cpu", parsed.SortBy);
        Assert.Equal(12345u, row.Pid);
        Assert.Equal("café", row.Name);
        Assert.Equal("Running", row.Status);
        Assert.Equal("C:/selected.exe", row.ExecutablePath);
        Assert.Equal(37.5f, row.CpuUsage);
        Assert.Equal(0x1122334455667788ul, row.MemoryBytes);
    }

    [Theory]
    [InlineData("ParseCompactTopProcesses", 72, 56, 64)]
    [InlineData("ParseCompactDiskUsage", 64, 48, 56)]
    public void DirectParserRejectsUnboundedEntryCountBeforeAllocation(string parser, int size, int entryOffset, int stringOffset)
    {
        var bytes = new byte[size];
        Write(bytes, entryOffset, (ulong)int.MaxValue);
        Write(bytes, stringOffset, 0ul);
        Assert.Null(Parse(parser, bytes));
    }

    [Theory]
    [InlineData("ParseCompactTopProcesses", 72, 64)]
    [InlineData("ParseCompactDiskUsage", 64, 56)]
    public void DirectParserRejectsUnboundedStringCount(string parser, int size, int stringOffset)
    {
        var bytes = new byte[size];
        Write(bytes, stringOffset, ulong.MaxValue);
        Assert.Null(Parse(parser, bytes));
    }

    [Theory]
    [InlineData("ParseCompactTopProcesses", 72)]
    [InlineData("ParseCompactDiskUsage", 64)]
    public void DirectParserRejectsUnclaimedTrailingBytes(string parser, int headerSize)
    {
        Assert.Null(Parse(parser, new byte[headerSize + 1]));
    }

    private static object? Parse(string name, byte[] bytes)
    {
        var allocation = Marshal.AllocHGlobal(bytes.Length);
        try
        {
            Marshal.Copy(bytes, 0, allocation, bytes.Length);
            var buffer = new PcaiByteBuffer { Status = PcaiStatus.Success, Data = allocation, Length = (UIntPtr)(uint)bytes.Length };
            var parser = typeof(PerformanceModule).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(parser);
            return parser!.Invoke(null, new object[] { buffer });
        }
        finally { Marshal.FreeHGlobal(allocation); }
    }

    private static void Write<T>(byte[] bytes, int offset, T value) where T : struct =>
        MemoryMarshal.Write(bytes.AsSpan(offset), in value);
}
