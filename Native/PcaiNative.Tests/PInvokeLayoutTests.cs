using System.Reflection;
using System.Runtime.InteropServices;

namespace PcaiNative.Tests;

/// <summary>
/// Pins the managed memory layout of every C# struct that mirrors a Rust <c>#[repr(C)]</c>
/// struct crossing the pcai_inference, pcai_media or pcai_core_lib FFI boundary.
///
/// Protects: the by-value struct returns and out-parameters used by the native DLLs. On Win64 a
/// struct larger than 8 bytes is returned through a caller-allocated hidden pointer sized from the
/// MANAGED declaration, so a managed struct that is shorter than the Rust one lets the native side
/// write past the buffer (stack or heap corruption). A reordered or retyped field silently
/// misreads every value after it.
///
/// Detects: a field added, removed, reordered or retyped on the C# side without the matching
/// Rust change (or the reverse), because the expected sizes and offsets below were derived by
/// hand from the Rust definitions. The Rust half of the contract lives in
/// <c>Native/pcai_core/pcai_inference/src/ffi/mod.rs</c> (<c>test_async_result_repr_c</c>);
/// both sides must be edited together.
///
/// Scope: the numbers assume the 64-bit Windows ABI that PC-AI ships for. The tests do nothing
/// in a 32-bit process, where the layouts legitimately differ.
/// </summary>
public class PInvokeLayoutTests
{
    // Each row: managed type, expected Marshal.SizeOf, "Field:offset" list covering every field.
    // Source of truth for the numbers is the Rust struct named in the trailing comment.
    public static TheoryData<Type, int, string> MirroredStructs => new()
    {
        // pcai_inference/src/ffi/mod.rs PcaiAsyncResult
        { typeof(InferenceModule.PcaiAsyncResult), 24,
          "Status:0,Text:8,TokensGenerated:16,PromptTokensEstimated:20" },

        // pcai_media/src/ffi/mod.rs PcaiMediaAsyncResult
        { typeof(PcaiMediaAsyncResult), 16, "Status:0,Text:8" },

        // pcai_core_lib/src/result.rs PcaiResult
        { typeof(PcaiResult), 40, "Status:0,Processed:8,Matched:16,Errors:24,ElapsedMs:32" },

        // pcai_core_lib/src/string.rs PcaiStringBuffer and PcaiByteBuffer
        { typeof(PcaiStringBuffer), 24, "Status:0,Data:8,Length:16" },
        { typeof(PcaiByteBuffer), 24, "Status:0,Data:8,Length:16" },

        // pcai_core_lib/src/system/path.rs PathAnalysisStats
        { typeof(PathAnalysisStats), 40,
          "Status:0,TotalEntries:4,UniqueEntries:8,DuplicateCount:12,NonExistentCount:16," +
          "EmptyCount:20,TrailingSlashCount:24,CrossDuplicateCount:28,ElapsedMs:32" },

        // pcai_core_lib/src/system/logs.rs LogSearchStats
        { typeof(LogSearchStats), 48,
          "Status:0,FilesSearched:8,FilesWithMatches:16,TotalMatches:24,BytesSearched:32,ElapsedMs:40" },

        // pcai_core_lib/src/performance/optimizer.rs MemoryPressureReport
        { typeof(MemoryPressureReport), 64,
          "Status:0,PressureLevel:4,AvailableMB:8,CommittedPct:16,PoolNonpagedMB:24,PagesPerSec:32," +
          "TopConsumerCount:40,HandleLeakCount:44,OrphanTerminalCount:48,ElapsedMs:56" },

        // pcai_core_lib/src/search/duplicates.rs DuplicateStats
        { typeof(DuplicateStats), 48,
          "Status:0,FilesScanned:8,DuplicateGroups:16,DuplicateFiles:24,WastedBytes:32,ElapsedMs:40" },

        // pcai_core_lib/src/search/files.rs FileSearchStats and DirectoryManifestStats
        { typeof(FileSearchStats), 40,
          "Status:0,FilesScanned:8,FilesMatched:16,TotalSize:24,ElapsedMs:32" },
        { typeof(DirectoryManifestStats), 48,
          "Status:0,EntriesReturned:8,FileCount:16,DirectoryCount:24,TotalSize:32,ElapsedMs:40" },

        // pcai_core_lib/src/search/content.rs ContentSearchStats
        { typeof(ContentSearchStats), 40,
          "Status:0,FilesScanned:8,FilesMatched:16,TotalMatches:24,ElapsedMs:32" },

        // pcai_core_lib/src/search/files.rs FileSearchCompactHeader and FileSearchCompactEntry
        { typeof(FileSearchCompactHeader), 64,
          "Status:0,Reserved:4,FilesScanned:8,FilesMatched:16,TotalSize:24,ElapsedMs:32,EntryCount:40," +
          "StringBytes:48,Truncated:56,Padding0:57,Padding1:58,Padding2:59,Padding3:60,Padding4:61," +
          "Padding5:62,Padding6:63" },
        { typeof(FileSearchCompactEntry), 32,
          "PathOffset:0,PathLength:4,Size:8,Modified:16,ReadOnly:24,Padding0:25,Padding1:26,Padding2:27," +
          "Padding3:28,Padding4:29,Padding5:30,Padding6:31" },

        // pcai_core_lib/src/search/content.rs ContentSearchCompactHeader and ContentSearchCompactEntry
        { typeof(ContentSearchCompactHeader), 64,
          "Status:0,Reserved:4,FilesScanned:8,FilesMatched:16,TotalMatches:24,ElapsedMs:32,EntryCount:40," +
          "StringBytes:48,Truncated:56,Padding0:57,Padding1:58,Padding2:59,Padding3:60,Padding4:61," +
          "Padding5:62,Padding6:63" },
        { typeof(ContentSearchCompactEntry), 24,
          "PathOffset:0,PathLength:4,LineOffset:8,LineLength:12,LineNumber:16" },

        // pcai_core_lib/src/performance/disk.rs DiskUsageStats and the compact header/entry
        { typeof(DiskUsageStats), 40,
          "Status:0,TotalSizeBytes:8,TotalFiles:16,TotalDirs:24,ElapsedMs:32" },
        { typeof(DiskUsageCompactHeader), 64,
          "Status:0,Reserved:4,TotalSizeBytes:8,TotalFiles:16,TotalDirs:24,ElapsedMs:32," +
          "RootPathOffset:40,RootPathLength:44,EntryCount:48,StringBytes:56" },
        { typeof(DiskUsageCompactEntry), 24,
          "PathOffset:0,PathLength:4,SizeBytes:8,FileCount:16" },

        // pcai_core_lib/src/performance/process.rs ProcessStats and the compact header/entry.
        // The Rust entry has implicit padding at bytes 36..40, so MemoryBytes sits at 40.
        { typeof(ProcessStats), 40,
          "Status:0,TotalProcesses:4,TotalThreads:8,SystemCpuUsage:12,SystemMemoryUsedBytes:16," +
          "SystemMemoryTotalBytes:24,ElapsedMs:32" },
        { typeof(ProcessListCompactHeader), 72,
          "Status:0,Reserved:4,TotalProcesses:8,TotalThreads:12,SystemCpuUsage:16,CpuPadding0:20," +
          "SystemMemoryUsedBytes:24,SystemMemoryTotalBytes:32,ElapsedMs:40,SortByOffset:48," +
          "SortByLength:52,EntryCount:56,StringBytes:64" },
        { typeof(ProcessListCompactEntry), 48,
          "Pid:0,NameOffset:4,NameLength:8,StatusOffset:12,StatusLength:16,ExePathOffset:20," +
          "ExePathLength:24,CpuUsage:28,CpuPadding0:32,MemoryBytes:40" },

        // pcai_core_lib/src/performance/memory.rs MemoryStats
        { typeof(MemoryStats), 56,
          "Status:0,TotalMemoryBytes:8,UsedMemoryBytes:16,AvailableMemoryBytes:24,TotalSwapBytes:32," +
          "UsedSwapBytes:40,ElapsedMs:48" },
    };

    [Theory]
    [MemberData(nameof(MirroredStructs))]
    public void MirroredStructMatchesRustReprCLayout(Type type, int expectedSize, string expectedOffsets)
    {
        if (!Environment.Is64BitProcess)
        {
            return; // Layouts below are the x64 ABI; see the class comment.
        }

        Assert.Equal(expectedSize, Marshal.SizeOf(type));

        var expected = expectedOffsets
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(pair => pair.Split(':'))
            .ToDictionary(parts => parts[0], parts => int.Parse(parts[1]));

        // Every declared field must be pinned, so a newly added managed field cannot slip past.
        var declared = type
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Select(f => f.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(expected.Keys.OrderBy(n => n, StringComparer.Ordinal).ToArray(), declared);

        foreach (var (field, offset) in expected)
        {
            Assert.True(
                offset == (int)Marshal.OffsetOf(type, field),
                $"{type.Name}.{field}: expected offset {offset}, actual {(int)Marshal.OffsetOf(type, field)}");
        }
    }

    [Fact]
    public void PcaiStatusIsFourBytesLikeTheRustReprCEnum()
    {
        // Protects: every status-bearing struct above. Rust's #[repr(C)] enum PcaiStatus is a
        // C int (4 bytes); the managed enum must be exactly that wide or all later offsets shift.
        Assert.Equal(4, Marshal.SizeOf(Enum.GetUnderlyingType(typeof(PcaiStatus))));
    }

    [Fact]
    public void PollResultSurfacesTokenCountsFromTheNativeStruct()
    {
        // Protects: the managed AsyncPollResult shape that carries TokensGenerated and
        // PromptTokensEstimated out of PcaiAsyncResult. Detects the counters being dropped again
        // when the struct is next edited.
        var result = new InferenceModule.AsyncPollResult(
            InferenceModule.AsyncRequestStatus.Complete, "ok", 7, 3);

        Assert.Equal(7u, result.TokensGenerated);
        Assert.Equal(3u, result.PromptTokensEstimated);
    }
}
