using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PcaiNative;

// ============================================================================
// FFI Structures - Must match Rust exactly
// ============================================================================

/// <summary>
/// Statistics returned by duplicate detection operations.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DuplicateStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong DuplicateGroups;
    public ulong DuplicateFiles;
    public ulong WastedBytes;
    public ulong ElapsedMs;

    public readonly bool IsSuccess => Status == PcaiStatus.Success;
}

/// <summary>
/// Statistics returned by file search operations.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct FileSearchStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalSize;
    public ulong ElapsedMs;

    public readonly bool IsSuccess => Status == PcaiStatus.Success;
}

/// <summary>
/// Statistics returned by directory manifest operations.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DirectoryManifestStats
{
    public PcaiStatus Status;
    public ulong EntriesReturned;
    public ulong FileCount;
    public ulong DirectoryCount;
    public ulong TotalSize;
    public ulong ElapsedMs;

    public readonly bool IsSuccess => Status == PcaiStatus.Success;
}

/// <summary>
/// Statistics returned by content search operations.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct ContentSearchStats
{
    public PcaiStatus Status;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalMatches;
    public ulong ElapsedMs;

    public readonly bool IsSuccess => Status == PcaiStatus.Success;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct FileSearchCompactHeader
{
    public PcaiStatus Status;
    public uint Reserved;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalSize;
    public ulong ElapsedMs;
    public ulong EntryCount;
    public ulong StringBytes;
    public byte Truncated;
    public byte Padding0;
    public byte Padding1;
    public byte Padding2;
    public byte Padding3;
    public byte Padding4;
    public byte Padding5;
    public byte Padding6;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct FileSearchCompactEntry
{
    public uint PathOffset;
    public uint PathLength;
    public ulong Size;
    public ulong Modified;
    public byte ReadOnly;
    public byte Padding0;
    public byte Padding1;
    public byte Padding2;
    public byte Padding3;
    public byte Padding4;
    public byte Padding5;
    public byte Padding6;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct ContentSearchCompactHeader
{
    public PcaiStatus Status;
    public uint Reserved;
    public ulong FilesScanned;
    public ulong FilesMatched;
    public ulong TotalMatches;
    public ulong ElapsedMs;
    public ulong EntryCount;
    public ulong StringBytes;
    public byte Truncated;
    public byte Padding0;
    public byte Padding1;
    public byte Padding2;
    public byte Padding3;
    public byte Padding4;
    public byte Padding5;
    public byte Padding6;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct ContentSearchCompactEntry
{
    public uint PathOffset;
    public uint PathLength;
    public uint LineOffset;
    public uint LineLength;
    public ulong LineNumber;
}

// ============================================================================
// JSON Result Classes - For deserializing full results
// ============================================================================

/// <summary>
/// A group of duplicate files sharing the same hash.
/// </summary>
public sealed class DuplicateGroup
{
    [JsonPropertyName("hash")]
    public string Hash { get; set; } = "";

    [JsonPropertyName("size")]
    public ulong Size { get; set; }

    [JsonPropertyName("paths")]
    public List<string> Paths { get; set; } = new();

    /// <summary>
    /// Number of duplicate files (excluding the original).
    /// </summary>
    public int DuplicateCount => Math.Max(0, Paths.Count - 1);

    /// <summary>
    /// Total bytes wasted by duplicates.
    /// </summary>
    public ulong WastedBytes => (ulong)DuplicateCount * Size;
}

/// <summary>
/// Complete result of a duplicate detection operation.
/// </summary>
public sealed class DuplicateResult
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("files_scanned")]
    public ulong FilesScanned { get; set; }

    [JsonPropertyName("duplicate_groups")]
    public ulong DuplicateGroupCount { get; set; }

    [JsonPropertyName("duplicate_files")]
    public ulong DuplicateFiles { get; set; }

    [JsonPropertyName("wasted_bytes")]
    public ulong WastedBytes { get; set; }

    [JsonPropertyName("elapsed_ms")]
    public ulong ElapsedMs { get; set; }

    [JsonPropertyName("groups")]
    public List<DuplicateGroup> Groups { get; set; } = new();

    public bool IsSuccess => Status == "Success";
}

/// <summary>
/// Information about a found file.
/// </summary>
public sealed class FoundFile
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("size")]
    public ulong Size { get; set; }

    [JsonPropertyName("modified")]
    public ulong Modified { get; set; }

    [JsonPropertyName("readonly")]
    public bool ReadOnly { get; set; }
}

/// <summary>
/// Information about a manifest entry.
/// </summary>
public sealed class DirectoryManifestEntry
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("relative_path")]
    public string RelativePath { get; set; } = "";

    [JsonPropertyName("entry_type")]
    public string EntryType { get; set; } = "";

    [JsonPropertyName("extension")]
    public string Extension { get; set; } = "";

    [JsonPropertyName("depth")]
    public uint Depth { get; set; }

    [JsonPropertyName("size")]
    public ulong Size { get; set; }

    [JsonPropertyName("modified")]
    public ulong Modified { get; set; }

    [JsonPropertyName("readonly")]
    public bool ReadOnly { get; set; }
}

/// <summary>
/// Complete result of a file search operation.
/// </summary>
public sealed class FileSearchResult
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("pattern")]
    public string Pattern { get; set; } = "";

    [JsonPropertyName("files_scanned")]
    public ulong FilesScanned { get; set; }

    [JsonPropertyName("files_matched")]
    public ulong FilesMatched { get; set; }

    [JsonPropertyName("total_size")]
    public ulong TotalSize { get; set; }

    [JsonPropertyName("elapsed_ms")]
    public ulong ElapsedMs { get; set; }

    [JsonPropertyName("files")]
    public List<FoundFile> Files { get; set; } = new();

    [JsonPropertyName("truncated")]
    public bool Truncated { get; set; }

    public bool IsSuccess => Status == "Success";
}

/// <summary>
/// Complete result of a directory manifest operation.
/// </summary>
public sealed class DirectoryManifestResult
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("root_path")]
    public string RootPath { get; set; } = "";

    [JsonPropertyName("max_depth")]
    public uint MaxDepth { get; set; }

    [JsonPropertyName("entries_returned")]
    public ulong EntriesReturned { get; set; }

    [JsonPropertyName("file_count")]
    public ulong FileCount { get; set; }

    [JsonPropertyName("directory_count")]
    public ulong DirectoryCount { get; set; }

    [JsonPropertyName("total_size")]
    public ulong TotalSize { get; set; }

    [JsonPropertyName("elapsed_ms")]
    public ulong ElapsedMs { get; set; }

    [JsonPropertyName("entries")]
    public List<DirectoryManifestEntry> Entries { get; set; } = new();

    [JsonPropertyName("truncated")]
    public bool Truncated { get; set; }

    public bool IsSuccess => Status == "Success";
}

/// <summary>
/// A single match within a file.
/// </summary>
public sealed class ContentMatch
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = "";

    [JsonPropertyName("line_number")]
    public ulong LineNumber { get; set; }

    [JsonPropertyName("line")]
    public string Line { get; set; } = "";

    [JsonPropertyName("before")]
    public List<string> Before { get; set; } = new();

    [JsonPropertyName("after")]
    public List<string> After { get; set; } = new();
}

/// <summary>
/// Complete result of a content search operation.
/// </summary>
public sealed class ContentSearchResult
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("pattern")]
    public string Pattern { get; set; } = "";

    [JsonPropertyName("file_pattern")]
    public string? FilePattern { get; set; }

    [JsonPropertyName("files_scanned")]
    public ulong FilesScanned { get; set; }

    [JsonPropertyName("files_matched")]
    public ulong FilesMatched { get; set; }

    [JsonPropertyName("total_matches")]
    public ulong TotalMatches { get; set; }

    [JsonPropertyName("elapsed_ms")]
    public ulong ElapsedMs { get; set; }

    [JsonPropertyName("matches")]
    public List<ContentMatch> Matches { get; set; } = new();

    [JsonPropertyName("truncated")]
    public bool Truncated { get; set; }

    public bool IsSuccess => Status == "Success";
}


// ============================================================================
// High-Level Wrapper
// ============================================================================

/// <summary>
/// High-level wrapper for PCAI Search functionality.
/// Provides type-safe, exception-safe access to native search operations.
/// </summary>
public static class PcaiSearch
{
    // Thread-safe lazy initialization for availability check
    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            using var version = NativeCore.pcai_core_version();
            return !version.IsInvalid;
        }
        catch
        {
            return false;
        }
    });

    // Thread-safe lazy initialization for version string
    private static readonly Lazy<string> _version = new(() =>
    {
        try
        {
            using var ptr = NativeCore.pcai_core_version();
            return ptr.ToManagedString() ?? "unknown";
        }
        catch
        {
            return "unavailable";
        }
    });

    /// <summary>
    /// Gets whether the native search library is available and functional.
    /// </summary>
    public static bool IsAvailable => _isAvailable.Value;

    /// <summary>
    /// Gets the native search library version.
    /// </summary>
    public static string Version => _version.Value;

    private static bool TryReadStruct<T>(ReadOnlySpan<byte> buffer, ref int offset, out T value) where T : unmanaged
    {
        var size = Marshal.SizeOf<T>();
        if (offset < 0 || offset > buffer.Length - size) {
            value = default;
            return false;
        }

        value = MemoryMarshal.Read<T>(buffer.Slice(offset, size));
        offset += size;
        return true;
    }

    private static bool TryReadUtf8String(ReadOnlySpan<byte> stringData, uint offset, uint length, out string value)
    {
        if (length == 0) {
            value = string.Empty;
            return true;
        }

        try
        {
            var start = checked((int)offset);
            var count = checked((int)length);
            if (start < 0 || count < 0 || start > stringData.Length - count) {
                value = string.Empty;
                return false;
            }

            value = Encoding.UTF8.GetString(stringData.Slice(start, count));
            return true;
        }
        catch (ArgumentException)
        {
            value = string.Empty;
            return false;
        }
        catch (OverflowException)
        {
            value = string.Empty;
            return false;
        }
    }

    private static bool TrySlice(ReadOnlySpan<byte> buffer, int offset, int length, out ReadOnlySpan<byte> slice)
    {
        if (offset < 0 || length < 0 || offset > buffer.Length - length) {
            slice = default;
            return false;
        }

        slice = buffer.Slice(offset, length);
        return true;
    }

    private static FileSearchResult? ParseCompactFileSearch(PcaiByteBuffer buffer, string pattern)
    {
        var bytes = buffer.ToManagedBytes();
        if (bytes is null || bytes.Length < Marshal.SizeOf<FileSearchCompactHeader>()) {
            return null;
        }

        var data = bytes.AsSpan();
        var offset = 0;
        if (!TryReadStruct<FileSearchCompactHeader>(data, ref offset, out var header) || header.Status != PcaiStatus.Success) {
            return null;
        }

        var entryCount = checked((int)header.EntryCount);
        var compactEntries = new FileSearchCompactEntry[entryCount];
        for (var i = 0; i < compactEntries.Length; i++) {
            if (!TryReadStruct<FileSearchCompactEntry>(data, ref offset, out compactEntries[i])) {
                return null;
            }
        }

        if (!TrySlice(data, offset, checked((int)header.StringBytes), out var stringData)) {
            return null;
        }

        var entries = new List<FoundFile>(compactEntries.Length);
        foreach (var entry in compactEntries) {
            if (!TryReadUtf8String(stringData, entry.PathOffset, entry.PathLength, out var path)) {
                return null;
            }

            entries.Add(new FoundFile {
                Path = path,
                Size = entry.Size,
                Modified = entry.Modified,
                ReadOnly = entry.ReadOnly != 0
            });
        }

        return new FileSearchResult {
            Status = header.Status.ToString(),
            Pattern = pattern,
            FilesScanned = header.FilesScanned,
            FilesMatched = header.FilesMatched,
            TotalSize = header.TotalSize,
            ElapsedMs = header.ElapsedMs,
            Files = entries,
            Truncated = header.Truncated != 0
        };
    }

    private static ContentSearchResult? ParseCompactContentSearch(PcaiByteBuffer buffer, string pattern, string? filePattern)
    {
        var bytes = buffer.ToManagedBytes();
        if (bytes is null || bytes.Length < Marshal.SizeOf<ContentSearchCompactHeader>()) {
            return null;
        }

        var data = bytes.AsSpan();
        var offset = 0;
        if (!TryReadStruct<ContentSearchCompactHeader>(data, ref offset, out var header) || header.Status != PcaiStatus.Success) {
            return null;
        }

        var entryCount = checked((int)header.EntryCount);
        var entries = new ContentMatch[entryCount];
        var compactEntries = new ContentSearchCompactEntry[entries.Length];

        for (var i = 0; i < compactEntries.Length; i++) {
            if (!TryReadStruct<ContentSearchCompactEntry>(data, ref offset, out compactEntries[i])) {
                return null;
            }
        }

        if (!TrySlice(data, offset, checked((int)header.StringBytes), out var stringData)) {
            return null;
        }

        for (var i = 0; i < compactEntries.Length; i++) {
            var entry = compactEntries[i];
            if (!TryReadUtf8String(stringData, entry.PathOffset, entry.PathLength, out var path) ||
                !TryReadUtf8String(stringData, entry.LineOffset, entry.LineLength, out var line)) {
                return null;
            }

            entries[i] = new ContentMatch {
                Path = path,
                LineNumber = entry.LineNumber,
                Line = line,
                Before = new List<string>(),
                After = new List<string>()
            };
        }

        return new ContentSearchResult {
            Status = header.Status.ToString(),
            Pattern = pattern,
            FilePattern = filePattern,
            FilesScanned = header.FilesScanned,
            FilesMatched = header.FilesMatched,
            TotalMatches = header.TotalMatches,
            ElapsedMs = header.ElapsedMs,
            Matches = new List<ContentMatch>(entries),
            Truncated = header.Truncated != 0
        };
    }

    // =========================================================================
    // Duplicate Detection
    // =========================================================================

    /// <summary>
    /// Finds duplicate files in a directory using parallel SHA-256 hashing.
    /// </summary>
    /// <param name="rootPath">Directory to search</param>
    /// <param name="minSize">Minimum file size in bytes (0 = all files)</param>
    /// <param name="includePattern">Glob pattern for files to include (null = all)</param>
    /// <param name="excludePattern">Glob pattern for files to exclude (null = none)</param>
    /// <returns>Full result with duplicate groups, or null if unavailable</returns>
    public static DuplicateResult? FindDuplicates(
        string? rootPath = null,
        ulong minSize = 0,
        string? includePattern = null,
        string? excludePattern = null)
    {
        if (!IsAvailable) return null;

        var buffer = NativeCore.pcai_find_duplicates(rootPath, minSize, includePattern, excludePattern);
        try
        {
            var json = buffer.ToManagedString();
            if (string.IsNullOrEmpty(json)) return null;

            return JsonSerializer.Deserialize<DuplicateResult>(json);
        }
        finally
        {
            NativeCore.pcai_free_string_buffer(ref buffer);
        }
    }

    /// <summary>
    /// Gets statistics about duplicates without the full file list.
    /// </summary>
    public static DuplicateStats FindDuplicatesStats(
        string? rootPath = null,
        ulong minSize = 0,
        string? includePattern = null,
        string? excludePattern = null)
    {
        if (!IsAvailable)
            return new DuplicateStats { Status = PcaiStatus.NotImplemented };

        return NativeCore.pcai_find_duplicates_stats(rootPath, minSize, includePattern, excludePattern);
    }

    // =========================================================================
    // File Search
    // =========================================================================

    /// <summary>
    /// Searches for files matching a glob pattern.
    /// </summary>
    /// <param name="pattern">Glob pattern (e.g., "*.txt", "**/*.rs")</param>
    /// <param name="rootPath">Directory to search (null = current directory)</param>
    /// <param name="maxResults">Maximum results (0 = unlimited)</param>
    /// <returns>Full result with file list, or null if unavailable</returns>
    public static FileSearchResult? FindFiles(
        string pattern,
        string? rootPath = null,
        ulong maxResults = 0)
    {
        if (!IsAvailable) return null;

        var compactBuffer = NativeCore.pcai_find_files_compact(rootPath, pattern, maxResults);
        try
        {
            if (compactBuffer.IsValid) {
                try
                {
                    var parsed = ParseCompactFileSearch(compactBuffer, pattern);
                    if (parsed is not null) {
                        return parsed;
                    }
                }
                catch (Exception)
                {
                    // Fall through to the JSON path if the compact payload is malformed.
                }
            }
        }
        finally
        {
            NativeCore.pcai_free_byte_buffer(ref compactBuffer);
        }

        var buffer = NativeCore.pcai_find_files(rootPath, pattern, maxResults);
        try
        {
            var json = buffer.ToManagedString();
            if (string.IsNullOrEmpty(json)) return null;

            return JsonSerializer.Deserialize<FileSearchResult>(json);
        }
        finally
        {
            NativeCore.pcai_free_string_buffer(ref buffer);
        }
    }

    /// <summary>
    /// Gets statistics about a file search without the full file list.
    /// </summary>
    public static FileSearchStats FindFilesStats(
        string pattern,
        string? rootPath = null,
        ulong maxResults = 0)
    {
        if (!IsAvailable)
            return new FileSearchStats { Status = PcaiStatus.NotImplemented };

        return NativeCore.pcai_find_files_stats(rootPath, pattern, maxResults);
    }

    /// <summary>
    /// Collects a directory manifest using native traversal.
    /// </summary>
    public static DirectoryManifestResult? CollectDirectoryManifest(
        string? rootPath = null,
        uint maxDepth = 0,
        ulong maxResults = 0)
    {
        if (!IsAvailable) return null;

        var buffer = NativeCore.pcai_collect_directory_manifest(rootPath, maxDepth, maxResults);
        try
        {
            var json = buffer.ToManagedString();
            if (string.IsNullOrEmpty(json)) return null;

            return JsonSerializer.Deserialize<DirectoryManifestResult>(json);
        }
        finally
        {
            NativeCore.pcai_free_string_buffer(ref buffer);
        }
    }

    /// <summary>
    /// Gets directory manifest statistics without the full entry list.
    /// </summary>
    public static DirectoryManifestStats CollectDirectoryManifestStats(
        string? rootPath = null,
        uint maxDepth = 0,
        ulong maxResults = 0)
    {
        if (!IsAvailable)
            return new DirectoryManifestStats { Status = PcaiStatus.NotImplemented };

        return NativeCore.pcai_collect_directory_manifest_stats(rootPath, maxDepth, maxResults);
    }

    // =========================================================================
    // Content Search
    // =========================================================================

    /// <summary>
    /// Searches file contents for a regex pattern.
    /// </summary>
    /// <param name="pattern">Regex pattern to search for</param>
    /// <param name="rootPath">Directory to search (null = current directory)</param>
    /// <param name="filePattern">Glob pattern for files to search (null = text files)</param>
    /// <param name="maxResults">Maximum matches (0 = unlimited)</param>
    /// <param name="contextLines">Lines of context around matches</param>
    /// <returns>Full result with matches, or null if unavailable</returns>
    public static ContentSearchResult? SearchContent(
        string pattern,
        string? rootPath = null,
        string? filePattern = null,
        ulong maxResults = 0,
        uint contextLines = 0)
    {
        if (!IsAvailable) return null;

        if (contextLines == 0) {
            var compactBuffer = NativeCore.pcai_search_content_compact(rootPath, pattern, filePattern, maxResults);
            try
            {
                if (compactBuffer.IsValid) {
                    try
                    {
                        var parsed = ParseCompactContentSearch(compactBuffer, pattern, filePattern);
                        if (parsed is not null) {
                            return parsed;
                        }
                    }
                    catch (Exception)
                    {
                        // Fall through to the JSON path if the compact payload is malformed.
                    }
                }
            }
            finally
            {
                NativeCore.pcai_free_byte_buffer(ref compactBuffer);
            }
        }

        var buffer = NativeCore.pcai_search_content(rootPath, pattern, filePattern, maxResults, contextLines);
        try
        {
            var json = buffer.ToManagedString();
            if (string.IsNullOrEmpty(json)) return null;

            return JsonSerializer.Deserialize<ContentSearchResult>(json);
        }
        finally
        {
            NativeCore.pcai_free_string_buffer(ref buffer);
        }
    }

    /// <summary>
    /// Gets statistics about a content search without the full match list.
    /// </summary>
    public static ContentSearchStats SearchContentStats(
        string pattern,
        string? rootPath = null,
        string? filePattern = null,
        ulong maxResults = 0)
    {
        if (!IsAvailable)
            return new ContentSearchStats { Status = PcaiStatus.NotImplemented };

        return NativeCore.pcai_search_content_stats(rootPath, pattern, filePattern, maxResults);
    }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// <summary>
    /// Gets diagnostic information about the search module.
    /// </summary>
    public static string GetDiagnostics()
    {
        if (!IsAvailable) return "Search module not available.";
        using var versionPtr = NativeCore.pcai_search_version();
        var versionStr = versionPtr.ToManagedString() ?? "unknown";
        return $"Search Module: Functional, Root: {NativeCore.pcai_fs_version()}, Version: {versionStr}";
    }

    public static string? FindFilesJson(string? rootPath, string pattern, uint maxResults = 0)
    {
        if (!IsAvailable) return null;
        var buffer = NativeCore.pcai_find_files(rootPath, pattern, maxResults);
        try { return buffer.ToManagedString(); }
        finally { NativeCore.pcai_free_string_buffer(ref buffer); }
    }

    public static string? SearchContentJson(string? rootPath, string pattern, string? filePattern = null, uint maxResults = 0, uint contextLines = 0)
    {
        if (!IsAvailable) return null;
        var buffer = NativeCore.pcai_search_content(rootPath, pattern, filePattern, maxResults, contextLines);
        try { return buffer.ToManagedString(); }
        finally { NativeCore.pcai_free_string_buffer(ref buffer); }
    }

    public static string? CollectDirectoryManifestJson(string? rootPath, uint maxDepth = 0, uint maxResults = 0)
    {
        if (!IsAvailable) return null;
        var buffer = NativeCore.pcai_collect_directory_manifest(rootPath, maxDepth, maxResults);
        try { return buffer.ToManagedString(); }
        finally { NativeCore.pcai_free_string_buffer(ref buffer); }
    }

    public static string? FindDuplicatesJson(string? rootPath, ulong minSize = 0, string? includePattern = null, string? excludePattern = null)
    {
        if (!IsAvailable) return null;
        var buffer = NativeCore.pcai_find_duplicates(rootPath, minSize, includePattern, excludePattern);
        try { return buffer.ToManagedString(); }
        finally { NativeCore.pcai_free_string_buffer(ref buffer); }
    }
}

/// <summary>
/// Diagnostic information about the search module.
/// </summary>
public sealed class SearchDiagnostics
{
    [JsonPropertyName("isAvailable")]
    public bool IsAvailable { get; init; }

    [JsonPropertyName("version")]
    public string Version { get; init; } = "";

    [JsonPropertyName("coreAvailable")]
    public bool CoreAvailable { get; init; }

    [JsonPropertyName("coreVersion")]
    public string CoreVersion { get; init; } = "";

    public string ToJson() => JsonSerializer.Serialize(this, new JsonSerializerOptions
    {
        WriteIndented = true
    });
}
