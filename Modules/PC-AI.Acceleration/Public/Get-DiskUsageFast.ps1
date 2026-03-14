#Requires -Version 7.0
<#
.SYNOPSIS
    Fast disk usage analysis using dust or parallel enumeration

.DESCRIPTION
    Analyzes disk usage using dust (Rust du alternative) when available,
    with fallback to parallel directory size calculation.

.PARAMETER Path
    Path to analyze

.PARAMETER Depth
    Maximum depth to display (default: 3)

.PARAMETER Top
    Show only top N directories by size

.PARAMETER MinSize
    Minimum size to display (e.g., "1MB", "100KB")

.PARAMETER SortBy
    Sort by: size (default), name, count

.EXAMPLE
    Get-DiskUsageFast -Path "C:\Users" -Depth 2 -Top 10
    Shows top 10 largest directories under Users

.EXAMPLE
    Get-DiskUsageFast -Path "D:\Projects" -MinSize 100MB
    Shows directories over 100MB

.OUTPUTS
    PSCustomObject[] with directory size information
#>
function Get-DiskUsageFast {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Path,

        [Parameter()]
        [int]$Depth = 3,

        [Parameter()]
        [int]$Top = 0,

        [Parameter()]
        [string]$MinSize = '0',

        [Parameter()]
        [ValidateSet('size', 'name', 'count')]
        [string]$SortBy = 'size',

        [Parameter()]
        [int]$ThrottleLimit = [Environment]::ProcessorCount
    )

    # Parse MinSize
    $minBytes = Convert-SizeToBytes -Size $MinSize

    $preferRustCli = $env:PCAI_PREFER_RUST_CLI -eq '1'
    $pcaiPerfPath = Get-PcaiPerfToolPath
    $dustPath = Get-RustToolPath -ToolName 'dust'
    $useDust = $null -ne $dustPath -and (Test-Path $dustPath)
    $nativeType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type

    $finalizeResults = {
        param([object[]]$InputRows)
        $rows = [System.Collections.Generic.List[object]]::new()
        foreach ($row in @($InputRows)) {
            if ($null -eq $row) {
                continue
            }
            if ([int64]$row.SizeBytes -lt $minBytes) {
                continue
            }
            $rows.Add($row)
        }

        if ($rows.Count -eq 0) {
            return @()
        }

        $sorted = switch ($SortBy) {
            'name'  { $rows | Sort-Object Path }
            'count' { $rows | Sort-Object FileCount -Descending }
            default { $rows | Sort-Object SizeBytes -Descending }
        }

        if ($Top -gt 0) {
            return @($sorted | Select-Object -First $Top)
        }

        return @($sorted)
    }

    if ($preferRustCli -and $pcaiPerfPath -and $Top -gt 0) {
        if ($Depth -gt 1) {
            Write-Verbose "Rust disk usage ignores Depth; returning top entries only."
        }
        $rustResults = Get-DiskUsageWithPcaiPerf -Path $Path -Top $Top -ToolPath $pcaiPerfPath
        if ($null -ne $rustResults) {
            return @(& $finalizeResults $rustResults)
        }
    }

    if ($nativeType -and [PcaiNative.PcaiCore]::IsAvailable -and $Top -gt 0) {
        if ($Depth -gt 1) {
            Write-Verbose "Native disk usage ignores Depth; returning top entries only."
        }
        $nativeResults = Get-DiskUsageWithNative -Path $Path -Top $Top
        if ($null -ne $nativeResults) {
            return @(& $finalizeResults $nativeResults)
        }
    }

    if ($useDust) {
        return @(& $finalizeResults (Get-DiskUsageWithDust -Path $Path -Depth $Depth -DustPath $dustPath))
    }
    else {
        Write-Verbose "dust not available, using parallel enumeration"
        return @(& $finalizeResults (Get-DiskUsageParallel -Path $Path -Depth $Depth -ThrottleLimit $ThrottleLimit))
    }
}

function Get-DiskUsageWithPcaiPerf {
    [CmdletBinding()]
    param(
        [string]$Path,
        [int]$Top,
        [Parameter(Mandatory)]
        [string]$ToolPath
    )

    try {
        $json = & $ToolPath 'disk' '--path' $Path '--top' $Top 2>$null
        if (-not $json) {
            return $null
        }

        return @(ConvertFrom-Json -InputObject $json)
    } catch {
        return $null
    }
}

function Convert-PcaiNativeDiskRows {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [object[]]$Rows,
        [Parameter(Mandatory)]
        [scriptblock]$Mapper
    )

    $results = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $Rows) {
        if ($null -eq $row) {
            continue
        }
        $results.Add((& $Mapper $row))
    }

    return @($results)
}

<#
.SYNOPSIS
    Retrieve disk usage using the native PCAI performance module.

.PARAMETER Path
    Path to analyze.

.PARAMETER Top
    Number of top directories to return.
#>
function Get-DiskUsageWithNative {
    [CmdletBinding()]
    param(
        [string]$Path,
        [int]$Top
    )

    try {
        $preferCompact = $env:PCAI_PREFER_COMPACT_TRANSPORT -eq '1'
        $performanceType = ([System.Management.Automation.PSTypeName]'PcaiNative.PerformanceModule').Type
        $typedMethod = if ($performanceType) {
            $performanceType.GetMethod('GetDiskUsageReport', [System.Reflection.BindingFlags]'Public,Static', $null, [Type[]]@([string], [uint32]), $null)
        } else {
            $null
        }
        $jsonMethod = if ($performanceType) {
            $performanceType.GetMethod('GetDiskUsageJson', [System.Reflection.BindingFlags]'Public,Static', $null, [Type[]]@([string], [uint32]), $null)
        } else {
            $null
        }

        $json = $null
        if ($jsonMethod -and -not $preferCompact) {
            $json = [PcaiNative.PerformanceModule]::GetDiskUsageJson($Path, [uint32]$Top)
        }
        if (-not $json) {
            $typedResult = $null
            if ($typedMethod) {
                $typedResult = [PcaiNative.PerformanceModule]::GetDiskUsageReport($Path, [uint32]$Top)
            }

            if ($typedResult -and $typedResult.TopEntries) {
                return @(Convert-PcaiNativeDiskRows -Rows @($typedResult.TopEntries) -Mapper {
                        param($row)
                        [PSCustomObject]@{
                            Path      = $row.Path
                            SizeBytes = [int64]$row.SizeBytes
                            SizeMB    = [Math]::Round([double]$row.SizeBytes / 1MB, 2)
                            SizeGB    = [Math]::Round([double]$row.SizeBytes / 1GB, 2)
                            SizeHuman = $row.SizeFormatted
                            FileCount = [int64]$row.FileCount
                            Tool      = 'pcai_native'
                        }
                    })
            }

            if ($jsonMethod -and $preferCompact) {
                $json = [PcaiNative.PerformanceModule]::GetDiskUsageJson($Path, [uint32]$Top)
            }
        }

        if (-not $json) {
            return $null
        }

        $result = ConvertFrom-Json -InputObject $json
        if (-not $result.top_entries) {
            return @()
        }

        return @(Convert-PcaiNativeDiskRows -Rows @($result.top_entries) -Mapper {
                param($row)
                [PSCustomObject]@{
                    Path      = $row.path
                    SizeBytes = [int64]$row.size_bytes
                    SizeMB    = [Math]::Round([double]$row.size_bytes / 1MB, 2)
                    SizeGB    = [Math]::Round([double]$row.size_bytes / 1GB, 2)
                    SizeHuman = $row.size_formatted
                    FileCount = [int64]$row.file_count
                    Tool      = 'pcai_native'
                }
            })
    }
    catch {
        return $null
    }
}

function Convert-SizeToBytes {
    [CmdletBinding()]
    param([string]$Size)

    if ($Size -match '^(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)?$') {
        $value = [double]$Matches[1]
        $unit = $Matches[2]

        switch ($unit) {
            'KB' { return [int64]($value * 1KB) }
            'MB' { return [int64]($value * 1MB) }
            'GB' { return [int64]($value * 1GB) }
            'TB' { return [int64]($value * 1TB) }
            default { return [int64]$value }
        }
    }

    return [int64]0
}

function Get-DiskUsageWithDust {
    [CmdletBinding()]
    param(
        [string]$Path,
        [int]$Depth,
        [string]$DustPath
    )

    $args = @(
        '-d', $Depth.ToString(),
        '-b',  # Use bytes for accurate parsing
        $Path
    )

    try {
        $output = & $DustPath @args 2>&1
        $results = @()

        foreach ($line in $output) {
            # dust output format: "  SIZE │ PATH"
            if ($line -match '^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B?)?\s*[│|]\s*(.+)$') {
                $sizeStr = $Matches[1]
                $unit = $Matches[2]
                $dirPath = $Matches[3].Trim()

                $sizeBytes = switch -Regex ($unit) {
                    'K' { [int64]($sizeStr) * 1KB }
                    'M' { [int64]($sizeStr) * 1MB }
                    'G' { [int64]($sizeStr) * 1GB }
                    'T' { [int64]($sizeStr) * 1TB }
                    default { [int64]$sizeStr }
                }

                $results += [PSCustomObject]@{
                    Path      = $dirPath
                    SizeBytes = $sizeBytes
                    SizeMB    = [Math]::Round($sizeBytes / 1MB, 2)
                    SizeGB    = [Math]::Round($sizeBytes / 1GB, 2)
                    SizeHuman = Format-ByteSize -Bytes $sizeBytes
                    Tool      = 'dust'
                }
            }
        }

        return $results
    }
    catch {
        Write-Warning "dust failed: $_"
        return @()
    }
}

function Get-DiskUsageParallel {
    [CmdletBinding()]
    param(
        [string]$Path,
        [int]$Depth,
        [int]$ThrottleLimit
    )

    # Get directories to analyze
    $directories = @(Get-Item $Path)

    if ($Depth -gt 0) {
        $directories += Get-ChildItem -Path $Path -Directory -Recurse -Depth ($Depth - 1) -ErrorAction SilentlyContinue
    }

    # Parallel size calculation
    $results = $directories | ForEach-Object -Parallel {
        $dir = $_

        try {
            $files = Get-ChildItem -Path $dir.FullName -File -Recurse -ErrorAction SilentlyContinue
            $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
            $fileCount = ($files | Measure-Object).Count

            [PSCustomObject]@{
                Path      = $dir.FullName
                SizeBytes = [int64]$totalSize
                SizeMB    = [Math]::Round($totalSize / 1MB, 2)
                SizeGB    = [Math]::Round($totalSize / 1GB, 2)
                FileCount = $fileCount
                Tool      = 'PowerShell'
            }
        }
        catch {
            $null
        }
    } -ThrottleLimit $ThrottleLimit | Where-Object { $_ }

    # Add human-readable size
    $results | ForEach-Object {
        $_ | Add-Member -NotePropertyName 'SizeHuman' -NotePropertyValue (Format-ByteSize -Bytes $_.SizeBytes) -Force
    }

    return $results
}

function Format-ByteSize {
    [CmdletBinding()]
    param([int64]$Bytes)

    if ($Bytes -ge 1TB) { return "{0:N2} TB" -f ($Bytes / 1TB) }
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
    return "$Bytes B"
}

function Sort-DiskUsageResults {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [PSCustomObject[]]$InputObject,
        [string]$SortBy
    )

    begin { $items = @() }
    process { $items += $InputObject }
    end {
        switch ($SortBy) {
            'size'  { $items | Sort-Object SizeBytes -Descending }
            'name'  { $items | Sort-Object Path }
            'count' { $items | Sort-Object FileCount -Descending }
            default { $items | Sort-Object SizeBytes -Descending }
        }
    }
}

function Select-TopResults {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [PSCustomObject[]]$InputObject,
        [int]$Top
    )

    begin { $items = @() }
    process { $items += $InputObject }
    end {
        if ($Top -gt 0) {
            $items | Select-Object -First $Top
        }
        else {
            $items
        }
    }
}
