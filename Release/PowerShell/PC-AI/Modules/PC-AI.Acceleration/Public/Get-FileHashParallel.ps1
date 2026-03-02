#Requires -Version 7.0
<#
.SYNOPSIS
    Computes file hashes in parallel using PowerShell 7+ native parallelism

.DESCRIPTION
    Uses ForEach-Object -Parallel (PS7+) to compute hashes of multiple files
    concurrently. Significantly faster than sequential hashing for multiple files.

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
    }

    process {
        foreach ($p in $Path) {
            if (Test-Path $p -PathType Container) {
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
                }
            }
            elseif (Test-Path $p -PathType Leaf) {
                $fileInfo = Get-Item $p
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

        $results = $allFiles | ForEach-Object -Parallel {
            $filePath = $_
            $algo = $using:Algorithm

            try {
                $fileInfo = Get-Item $filePath -ErrorAction Stop
                $hash = Get-FileHash -Path $filePath -Algorithm $algo -ErrorAction Stop

                [PSCustomObject]@{
                    Path      = $filePath
                    Name      = $fileInfo.Name
                    Hash      = $hash.Hash
                    Algorithm = $algo
                    SizeBytes = $fileInfo.Length
                    SizeMB    = [Math]::Round($fileInfo.Length / 1MB, 2)
                    Success   = $true
                    Error     = $null
                }
            }
            catch {
                [PSCustomObject]@{
                    Path      = $filePath
                    Name      = (Split-Path $filePath -Leaf)
                    Hash      = $null
                    Algorithm = $algo
                    SizeBytes = 0
                    SizeMB    = 0
                    Success   = $false
                    Error     = $_.Exception.Message
                }
            }
        } -ThrottleLimit $ThrottleLimit

        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        $successCount = ($results | Where-Object Success).Count
        $totalSize = ($results | Where-Object Success | Measure-Object -Property SizeBytes -Sum).Sum

        Write-Verbose "Hashed $successCount files ($([Math]::Round($totalSize / 1MB, 2)) MB) in $([Math]::Round($duration, 2)) seconds"
        Write-Verbose "Throughput: $([Math]::Round(($totalSize / 1MB) / $duration, 2)) MB/s"

        return $results
    }
}
