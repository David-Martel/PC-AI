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

    # Use thread-safe collection
    $results = [System.Collections.Concurrent.ConcurrentBag[PSCustomObject]]::new()

    $parallelOptions = [System.Threading.Tasks.ParallelOptions]::new()
    $parallelOptions.MaxDegreeOfParallelism = $MaxDegreeOfParallelism

    [System.Threading.Tasks.Parallel]::ForEach(
        $FilePaths,
        $parallelOptions,
        [Action[string]] {
            param($filePath)
            try {
                if (Test-Path $filePath -PathType Leaf) {
                    $hash = Get-FileHash -Path $filePath -Algorithm $Algorithm -ErrorAction Stop
                    $fileInfo = Get-Item $filePath
                    $results.Add([PSCustomObject]@{
                        Path      = $filePath
                        Hash      = $hash.Hash
                        Algorithm = $Algorithm
                        Size      = $fileInfo.Length
                        Success   = $true
                        Error     = $null
                    })
                }
            }
            catch {
                $results.Add([PSCustomObject]@{
                    Path      = $filePath
                    Hash      = $null
                    Algorithm = $Algorithm
                    Size      = 0
                    Success   = $false
                    Error     = $_.Exception.Message
                })
            }
        }
    )

    return $results.ToArray()
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
