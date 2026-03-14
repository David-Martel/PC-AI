#Requires -Version 5.1
<#
.SYNOPSIS
    .NET parallel processing helpers for CPU-bound operations
#>

function Invoke-ParallelFileHash {
    <#
    .SYNOPSIS
        Computes file hashes in parallel using .NET Parallel.ForEach
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string[]]$FilePaths,

        [Parameter()]
        [ValidateSet('SHA256', 'SHA1', 'MD5', 'SHA384', 'SHA512')]
        [string]$Algorithm = 'SHA256',

        [Parameter()]
        [int]$MaxDegreeOfParallelism = [Environment]::ProcessorCount
    )

    Initialize-PcaiDotNetHashHelpers
    $effectiveParallelism = if ($MaxDegreeOfParallelism -gt 0) { $MaxDegreeOfParallelism } else { [Environment]::ProcessorCount }
    return [PcAi.Acceleration.ParallelHashHelper]::ComputeFileHashes($FilePaths, $Algorithm, $effectiveParallelism)
}

function Initialize-PcaiDotNetHashHelpers {
    if ('PcAi.Acceleration.ParallelHashHelper' -as [type]) {
        return
    }

    $typeDefinition = @"
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace PcAi.Acceleration
{
    public sealed class PowerShellHashResult
    {
        public string Path { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Hash { get; set; }
        public string Algorithm { get; set; } = string.Empty;
        public long Size { get; set; }
        public long SizeBytes => Size;
        public double SizeMB => Math.Round(Size / 1024d / 1024d, 2);
        public bool Success { get; set; }
        public string Error { get; set; }
    }

    public static class ParallelHashHelper
    {
        private sealed class HashRequestCacheEntry
        {
            public string[] Paths { get; init; } = Array.Empty<string>();
            public long[] Lengths { get; init; } = Array.Empty<long>();
            public long[] LastWriteTicks { get; init; } = Array.Empty<long>();
            public PowerShellHashResult[] Results { get; init; } = Array.Empty<PowerShellHashResult>();
            public DateTime LastAccessUtc { get; set; } = DateTime.UtcNow;
        }

        private static readonly object CacheSync = new();
        private static readonly Dictionary<string, HashRequestCacheEntry> RequestCache = new(StringComparer.OrdinalIgnoreCase);
        private const int MaxRequestCacheEntries = 32;

        private static string BuildRequestKey(string[] filePaths, string algorithmName)
        {
            var builder = new StringBuilder(algorithmName.Length + (filePaths.Length * 64));
            builder.Append(algorithmName);
            builder.Append('|');
            foreach (var filePath in filePaths)
            {
                builder.Append(filePath);
                builder.Append('\n');
            }

            return builder.ToString();
        }

        private static bool TryGetCachedResults(string[] filePaths, string algorithmName, out PowerShellHashResult[] results)
        {
            var cacheKey = BuildRequestKey(filePaths, algorithmName);
            lock (CacheSync)
            {
                if (!RequestCache.TryGetValue(cacheKey, out var entry))
                {
                    results = Array.Empty<PowerShellHashResult>();
                    return false;
                }

                if (entry.Paths.Length != filePaths.Length)
                {
                    RequestCache.Remove(cacheKey);
                    results = Array.Empty<PowerShellHashResult>();
                    return false;
                }

                for (var i = 0; i < filePaths.Length; i++)
                {
                    if (!string.Equals(entry.Paths[i], filePaths[i], StringComparison.OrdinalIgnoreCase))
                    {
                        RequestCache.Remove(cacheKey);
                        results = Array.Empty<PowerShellHashResult>();
                        return false;
                    }

                    var fileInfo = new FileInfo(filePaths[i]);
                    if (!fileInfo.Exists ||
                        fileInfo.Length != entry.Lengths[i] ||
                        fileInfo.LastWriteTimeUtc.Ticks != entry.LastWriteTicks[i])
                    {
                        RequestCache.Remove(cacheKey);
                        results = Array.Empty<PowerShellHashResult>();
                        return false;
                    }
                }

                entry.LastAccessUtc = DateTime.UtcNow;
                results = entry.Results;
                return true;
            }
        }

        private static void StoreCachedResults(string[] filePaths, string algorithmName, PowerShellHashResult[] results)
        {
            var cacheKey = BuildRequestKey(filePaths, algorithmName);
            var lengths = new long[filePaths.Length];
            var lastWriteTicks = new long[filePaths.Length];

            for (var i = 0; i < filePaths.Length; i++)
            {
                var fileInfo = new FileInfo(filePaths[i]);
                lengths[i] = fileInfo.Exists ? fileInfo.Length : 0;
                lastWriteTicks[i] = fileInfo.Exists ? fileInfo.LastWriteTimeUtc.Ticks : 0;
            }

            lock (CacheSync)
            {
                if (RequestCache.Count >= MaxRequestCacheEntries)
                {
                    var oldest = RequestCache.OrderBy(kvp => kvp.Value.LastAccessUtc).FirstOrDefault();
                    if (!string.IsNullOrEmpty(oldest.Key))
                    {
                        RequestCache.Remove(oldest.Key);
                    }
                }

                RequestCache[cacheKey] = new HashRequestCacheEntry
                {
                    Paths = filePaths.ToArray(),
                    Lengths = lengths,
                    LastWriteTicks = lastWriteTicks,
                    Results = results,
                    LastAccessUtc = DateTime.UtcNow
                };
            }
        }

        private static string ComputeHashHex(string filePath, string algorithmName)
        {
            using var stream = new FileStream(
                filePath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                1024 * 1024,
                FileOptions.SequentialScan);

            byte[] hashBytes = algorithmName.ToUpperInvariant() switch
            {
                "SHA256" => SHA256.HashData(stream),
                "SHA1" => SHA1.HashData(stream),
                "MD5" => MD5.HashData(stream),
                "SHA384" => SHA384.HashData(stream),
                "SHA512" => SHA512.HashData(stream),
                _ => throw new InvalidOperationException($"Unsupported hash algorithm: {algorithmName}")
            };

            return Convert.ToHexString(hashBytes);
        }

        private static PowerShellHashResult ProcessSingleFile(string filePath, string algorithmName)
        {
            if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            {
                return null;
            }

            try
            {
                var fileInfo = new FileInfo(filePath);
                return new PowerShellHashResult
                {
                    Path = filePath,
                    Name = fileInfo.Name,
                    Hash = ComputeHashHex(filePath, algorithmName),
                    Algorithm = algorithmName,
                    Size = fileInfo.Length,
                    Success = true,
                    Error = null
                };
            }
            catch (Exception ex)
            {
                return new PowerShellHashResult
                {
                    Path = filePath ?? string.Empty,
                    Name = Path.GetFileName(filePath ?? string.Empty),
                    Hash = null,
                    Algorithm = algorithmName,
                    Size = 0,
                    Success = false,
                    Error = ex.Message
                };
            }
        }

        public static PowerShellHashResult[] ComputeFileHashes(string[] filePaths, string algorithmName, int maxDegreeOfParallelism)
        {
            if (filePaths == null) throw new ArgumentNullException(nameof(filePaths));
            if (string.IsNullOrWhiteSpace(algorithmName)) throw new ArgumentException("Algorithm is required.", nameof(algorithmName));
            if (filePaths.Length == 0) return Array.Empty<PowerShellHashResult>();
            if (TryGetCachedResults(filePaths, algorithmName, out var cachedResults))
            {
                return cachedResults;
            }

            var results = new PowerShellHashResult[filePaths.Length];
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Environment.ProcessorCount
            };

            var sequentialThreshold = Math.Max(256, options.MaxDegreeOfParallelism * 32);
            if (filePaths.Length <= sequentialThreshold)
            {
                for (var i = 0; i < filePaths.Length; i++)
                {
                    results[i] = ProcessSingleFile(filePaths[i], algorithmName);
                }
            }
            else
            {
                Parallel.For(0, filePaths.Length, options, i =>
                {
                    results[i] = ProcessSingleFile(filePaths[i], algorithmName);
                });
            }

            var finalized = results.Where(static result => result != null).ToArray();
            StoreCachedResults(filePaths, algorithmName, finalized);
            return finalized;
        }
    }
}
"@

    Add-Type -TypeDefinition $typeDefinition -Language CSharp
}

function Invoke-ParallelFileOperation {
    <#
    .SYNOPSIS
        Executes a scriptblock in parallel across multiple files
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string[]]$FilePaths,

        [Parameter(Mandatory)]
        [scriptblock]$ScriptBlock,

        [Parameter()]
        [int]$MaxDegreeOfParallelism = [Environment]::ProcessorCount
    )

    $results = [System.Collections.Concurrent.ConcurrentBag[object]]::new()

    $parallelOptions = [System.Threading.Tasks.ParallelOptions]::new()
    $parallelOptions.MaxDegreeOfParallelism = $MaxDegreeOfParallelism

    [System.Threading.Tasks.Parallel]::ForEach(
        $FilePaths,
        $parallelOptions,
        [Action[string]] {
            param($filePath)
            try {
                $result = & $ScriptBlock -FilePath $filePath
                if ($result) {
                    $results.Add($result)
                }
            }
            catch {
                # Skip errors in parallel processing
            }
        }
    )

    return $results.ToArray()
}

function Get-OptimalParallelism {
    <#
    .SYNOPSIS
        Determines optimal parallelism based on workload
    #>
    [CmdletBinding()]
    param(
        [Parameter()]
        [int]$ItemCount = 100,

        [Parameter()]
        [ValidateSet('CPU', 'IO', 'Mixed')]
        [string]$WorkloadType = 'Mixed'
    )

    $cpuCount = [Environment]::ProcessorCount

    switch ($WorkloadType) {
        'CPU' {
            # CPU-bound: use processor count
            return $cpuCount
        }
        'IO' {
            # I/O-bound: can use more threads
            return [Math]::Min($cpuCount * 2, $ItemCount)
        }
        'Mixed' {
            # Mixed: balance between CPU and I/O
            return [Math]::Min([Math]::Ceiling($cpuCount * 1.5), $ItemCount)
        }
    }
}
