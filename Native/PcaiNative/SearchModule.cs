using System.Runtime.InteropServices;
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
            var version = NativeCore.pcai_core_version();
            return version != IntPtr.Zero;
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
            var ptr = NativeCore.pcai_core_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
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
        return $"Search Module: Functional, Root: {NativeCore.pcai_fs_version()}";
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
