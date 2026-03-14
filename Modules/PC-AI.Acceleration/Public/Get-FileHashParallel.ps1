#Requires -Version 7.0
<#
.SYNOPSIS
    Computes file hashes in parallel using a compiled .NET helper

.DESCRIPTION
    Enumerates candidate files in PowerShell, then delegates the hot hashing path
    to a compiled .NET helper. This avoids runspace overhead and is substantially
    faster and more reliable than invoking PowerShell cmdlets inside worker threads.

.PARAMETER Path
    File path(s) or directory to hash

.PARAMETER Algorithm
    Hash algorithm: SHA256 (default), SHA1, MD5, SHA384, SHA512

.PARAMETER Recurse
    Recurse into subdirectories

.PARAMETER Include
    File patterns to include

.PARAMETER ThrottleLimit
    Maximum concurrent operations (default: CPU count)

.EXAMPLE
    Get-FileHashParallel -Path "C:\Downloads" -Recurse
    Hashes all files in Downloads recursively

.EXAMPLE
    Get-FileHashParallel -Path "D:\Backups" -Include "*.zip" -Algorithm SHA512
    Hashes only zip files with SHA512

.OUTPUTS
    PSCustomObject[] with hash results
#>
function Get-FileHashParallel {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipeline)]
        [string[]]$Path,

        [Parameter()]
        [ValidateSet('SHA256', 'SHA1', 'MD5', 'SHA384', 'SHA512')]
        [string]$Algorithm = 'SHA256',

        [Parameter()]
        [switch]$Recurse,

        [Parameter()]
        [string[]]$Include,

        [Parameter()]
        [int]$ThrottleLimit = [Environment]::ProcessorCount,

        [Parameter()]
        [int64]$MinimumSize = 0,

        [Parameter()]
        [int64]$MaximumSize = [long]::MaxValue
    )

    begin {
        $allFiles = [System.Collections.Generic.List[string]]::new()
        $totalBytes = [int64]0
    }

    process {
        foreach ($p in $Path) {
            if ([System.IO.Directory]::Exists($p)) {
                $params = @{
                    Path        = $p
                    File        = $true
                    ErrorAction = 'SilentlyContinue'
                }

                # -Include requires -Recurse to work properly
                if ($Include) {
                    $params.Recurse = $true
                    $params.Include = $Include

                    # If Recurse wasn't explicitly requested, limit depth to 0
                    if (-not $Recurse) {
                        $params.Depth = 0
                    }
                }
                elseif ($Recurse) {
                    $params.Recurse = $true
                }

                Get-ChildItem @params | Where-Object {
                    $_.Length -ge $MinimumSize -and $_.Length -le $MaximumSize
                } | ForEach-Object {
                    $allFiles.Add($_.FullName)
                    $totalBytes += $_.Length
                }
            }
            elseif ([System.IO.File]::Exists($p)) {
                $fileInfo = [System.IO.FileInfo]::new($p)
                if ($fileInfo.Length -ge $MinimumSize -and $fileInfo.Length -le $MaximumSize) {
                    # Check if file matches Include patterns
                    $matchesInclude = $true
                    if ($Include) {
                        $matchesInclude = $false
                        foreach ($pattern in $Include) {
                            if ($fileInfo.Name -like $pattern) {
                                $matchesInclude = $true
                                break
                            }
                        }
                    }
                    if ($matchesInclude) {
                        $allFiles.Add($p)
                        $totalBytes += $fileInfo.Length
                    }
                }
            }
        }
    }

    end {
        if ($allFiles.Count -eq 0) {
            Write-Warning "No files found matching criteria"
            return @()
        }

        Write-Verbose "Hashing $($allFiles.Count) files with $ThrottleLimit concurrent threads"

        $startTime = Get-Date
        $useSequentialFastPath = $allFiles.Count -le 32 -and $totalBytes -le 4MB
        $pcaiPerfPath = Get-PcaiPerfToolPath
        $useRustHashPath = $Algorithm -eq 'SHA256' -and $allFiles.Count -ge 4096 -and $pcaiPerfPath -and $env:PCAI_PREFER_RUST_HASHER -eq '1'
        $pendingPaths = $allFiles.ToArray()

        if ($useRustHashPath) {
            $results = @(Get-FileHashWithPcaiPerf -ToolPath $pcaiPerfPath -FilePaths $pendingPaths -Algorithm $Algorithm)
            if ($results.Count -eq 0) {
                $useRustHashPath = $false
            }
        }

        if (-not $results -and -not $useRustHashPath -and $useSequentialFastPath) {
            Write-Verbose "Using sequential hash fast path for small workload ($($allFiles.Count) files, $([Math]::Round($totalBytes / 1MB, 2)) MB)"
            $results = foreach ($filePath in $pendingPaths) {
                try {
                    $hashResult = Get-FileHash -Path $filePath -Algorithm $Algorithm -ErrorAction Stop
                    $fileInfo = [System.IO.FileInfo]::new($filePath)
                    [PSCustomObject]@{
                        Path      = $filePath
                        Name      = $fileInfo.Name
                        Hash      = $hashResult.Hash
                        Algorithm = $Algorithm
                        SizeBytes = $fileInfo.Length
                        SizeMB    = [Math]::Round($fileInfo.Length / 1MB, 2)
                        Success   = $true
                        Error     = $null
                    }
                }
                catch {
                    [PSCustomObject]@{
                        Path      = $filePath
                        Name      = [System.IO.Path]::GetFileName($filePath)
                        Hash      = $null
                        Algorithm = $Algorithm
                        SizeBytes = 0
                        SizeMB    = 0
                        Success   = $false
                        Error     = $_.Exception.Message
                    }
                }
            }
        }
        elseif (-not $results -and -not $useRustHashPath) {
            $results = @(Invoke-ParallelFileHash -FilePaths $pendingPaths -Algorithm $Algorithm -MaxDegreeOfParallelism $ThrottleLimit)
        }

        if ($VerbosePreference -ne 'SilentlyContinue') {
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            $successCount = 0
            $totalSize = [int64]0

            foreach ($result in $results) {
                if ($result.Success) {
                    $successCount++
                    $sizeValue = if ($null -ne $result.SizeBytes) { $result.SizeBytes } else { $result.Size }
                    $totalSize += [int64]$sizeValue
                }
            }

            Write-Verbose "Hashed $successCount files ($([Math]::Round($totalSize / 1MB, 2)) MB) in $([Math]::Round($duration, 2)) seconds"
            if ($duration -gt 0) {
                Write-Verbose "Throughput: $([Math]::Round(($totalSize / 1MB) / $duration, 2)) MB/s"
            }
        }

        return $results
    }
}

function Get-FileHashWithPcaiPerf {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ToolPath,
        [Parameter(Mandatory)]
        [string[]]$FilePaths,
        [Parameter(Mandatory)]
        [string]$Algorithm
    )

    try {
        if (Get-Command Invoke-PcaiPerfWorkerRequest -ErrorAction SilentlyContinue) {
            $result = Invoke-PcaiPerfWorkerRequest -ToolPath $ToolPath -Command 'hash-list' -Payload @{
                algorithm = $Algorithm
                paths     = $FilePaths
            }
            if ($result) {
                return @($result)
            }
        }

        $json = & $ToolPath 'hash-list' '--algorithm' $Algorithm @FilePaths 2>$null
        if (-not $json) {
            return @()
        }

        return @($json | ConvertFrom-Json)
    } catch {
        Write-Verbose "pcai-perf hash-list failed: $_"
        return @()
    }
}
