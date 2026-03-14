#Requires -Version 7.0
<#
.SYNOPSIS
    Fast duplicate file detection using parallel hashing and fd

.DESCRIPTION
    Finds duplicate files using a multi-phase approach:
    1. Prefer the native PCAI duplicate engine when available
    2. Otherwise enumerate files with fd (if available)
    3. Group by file size (quick filter)
    4. Compute hashes with a compiled .NET parallel helper
    5. Group by hash to identify duplicates

.PARAMETER Path
    Root path to search for duplicates

.PARAMETER Recurse
    Recurse into subdirectories

.PARAMETER MinimumSize
    Minimum file size to consider (default: 1KB)

.PARAMETER Include
    File patterns to include

.PARAMETER Exclude
    Patterns to exclude

.PARAMETER Algorithm
    Hash algorithm (default: SHA256)

.PARAMETER ThrottleLimit
    Maximum concurrent hash operations

.EXAMPLE
    Find-DuplicatesFast -Path "D:\Downloads" -Recurse
    Finds all duplicates in Downloads

.EXAMPLE
    Find-DuplicatesFast -Path "C:\Photos" -Include "*.jpg","*.png" -MinimumSize 100KB
    Finds duplicate images over 100KB

.OUTPUTS
    PSCustomObject[] with duplicate groups
#>
function Find-DuplicatesFast {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Path,

        [Parameter()]
        [switch]$Recurse,

        [Parameter()]
        [int64]$MinimumSize = 1KB,

        [Parameter()]
        [int64]$MaximumSize = [long]::MaxValue,

        [Parameter()]
        [string[]]$Include,

        [Parameter()]
        [string[]]$Exclude,

        [Parameter()]
        [ValidateSet('SHA256', 'SHA1', 'MD5')]
        [string]$Algorithm = 'SHA256',

        [Parameter()]
        [int]$ThrottleLimit = [Environment]::ProcessorCount,

        [Parameter()]
        [switch]$ShowProgress,

        [Parameter()]
        [switch]$DisableNative
    )

    $useNativeDuplicates =
        -not $DisableNative -and
        $Recurse -and
        $Algorithm -eq 'SHA256' -and
        (Get-Command Invoke-PcaiNativeDuplicates -ErrorAction SilentlyContinue) -and
        (Test-PcaiNativeAvailable)

    if ($useNativeDuplicates) {
        $nativeInclude = if ($Include.Count -le 1) { $Include | Select-Object -First 1 } else { $null }
        $nativeExclude = if ($Exclude.Count -le 1) { $Exclude | Select-Object -First 1 } else { $null }

        if (($Include.Count -le 1) -and ($Exclude.Count -le 1)) {
            $nativeResult = Invoke-PcaiNativeDuplicates -Path $Path -MinimumSize $MinimumSize -IncludePattern $nativeInclude -ExcludePattern $nativeExclude
            if ($nativeResult -and $nativeResult.IsSuccess) {
                return @(ConvertFrom-PcaiDuplicateResult -NativeResult $nativeResult -Algorithm $Algorithm -MaximumSize $MaximumSize)
            }
        }
    }

    return @(Invoke-PowerShellDuplicateScan @PSBoundParameters)
}

function Invoke-PowerShellDuplicateScan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$Recurse,
        [int64]$MinimumSize = 1KB,
        [int64]$MaximumSize = [long]::MaxValue,
        [string[]]$Include,
        [string[]]$Exclude,
        [ValidateSet('SHA256', 'SHA1', 'MD5')]
        [string]$Algorithm = 'SHA256',
        [int]$ThrottleLimit = [Environment]::ProcessorCount,
        [switch]$ShowProgress,
        [switch]$DisableNative
    )

    $startTime = Get-Date
    Write-Host "[*] Phase 1: Enumerating files..." -ForegroundColor Cyan

    # Use fd for fast enumeration if available
    $fdPath = Get-RustToolPath -ToolName 'fd'
    $useFd = $null -ne $fdPath -and (Test-Path $fdPath)

    if ($useFd) {
        Write-Verbose "Using fd for file enumeration"
        $files = Find-WithFdForDuplicates -Path $Path -Recurse:$Recurse -Include $Include -Exclude $Exclude -FdPath $fdPath
    }
    else {
        Write-Verbose "Using Get-ChildItem for file enumeration"
        $params = @{
            Path        = $Path
            File        = $true
            Recurse     = $Recurse
            ErrorAction = 'SilentlyContinue'
        }
        if ($Include) { $params.Include = $Include }
        $files = Get-ChildItem @params
    }

    # Apply size filter
    $files = $files | Where-Object {
        $_.Length -ge $MinimumSize -and $_.Length -le $MaximumSize
    }

    $totalFiles = ($files | Measure-Object).Count
    Write-Host "[*] Found $totalFiles files to analyze" -ForegroundColor Cyan

    if ($totalFiles -eq 0) {
        Write-Warning "No files found matching criteria"
        return @()
    }

    # Phase 2: Group by size (quick duplicate candidate identification)
    Write-Host "[*] Phase 2: Grouping by file size..." -ForegroundColor Cyan

    $sizeGroups = $files | Group-Object -Property Length | Where-Object { $_.Count -gt 1 }
    $candidateCount = ($sizeGroups | ForEach-Object { $_.Group } | Measure-Object).Count

    Write-Host "[*] Found $candidateCount files in $(($sizeGroups | Measure-Object).Count) size groups (potential duplicates)" -ForegroundColor Cyan

    if ($candidateCount -eq 0) {
        Write-Host "[+] No potential duplicates found" -ForegroundColor Green
        return @()
    }

    # Phase 3: Parallel hash computation
    Write-Host "[*] Phase 3: Computing hashes in parallel (throttle: $ThrottleLimit)..." -ForegroundColor Cyan

    $candidateFiles = $sizeGroups | ForEach-Object { $_.Group.FullName }
    $hashResults = @(Invoke-ParallelFileHash -FilePaths $candidateFiles -Algorithm $Algorithm -MaxDegreeOfParallelism $ThrottleLimit |
        Where-Object { $_.Success } |
        ForEach-Object {
            [PSCustomObject]@{
                Path = $_.Path
                Hash = $_.Hash
                Size = $_.Size
            }
        })

    # Phase 4: Group by hash to find actual duplicates
    Write-Host "[*] Phase 4: Identifying duplicates..." -ForegroundColor Cyan

    $duplicateGroups = $hashResults |
        Group-Object -Property Hash |
        Where-Object { $_.Count -gt 1 }

    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds

    # Build results
    $results = @()
    $totalWasted = 0

    foreach ($group in $duplicateGroups) {
        $files = $group.Group | Sort-Object Path
        $wastedSpace = $files[0].Size * ($group.Count - 1)
        $totalWasted += $wastedSpace

        $results += [PSCustomObject]@{
            Hash        = $group.Name
            Algorithm   = $Algorithm
            FileSize    = $files[0].Size
            FileSizeMB  = [Math]::Round($files[0].Size / 1MB, 2)
            Count       = $group.Count
            WastedBytes = $wastedSpace
            WastedMB    = [Math]::Round($wastedSpace / 1MB, 2)
            Files       = $files.Path
            Original    = $files[0].Path
            Duplicates  = $files[1..($files.Count - 1)].Path
        }
    }

    # Summary
    Write-Host ""
    Write-Host "=== Duplicate Analysis Complete ===" -ForegroundColor Green
    Write-Host "  Total files scanned: $totalFiles"
    Write-Host "  Duplicate groups found: $(($results | Measure-Object).Count)"
    Write-Host "  Total duplicate files: $(($results | Measure-Object -Property Count -Sum).Sum - ($results | Measure-Object).Count)"
    Write-Host "  Wasted space: $([Math]::Round($totalWasted / 1MB, 2)) MB ($([Math]::Round($totalWasted / 1GB, 2)) GB)"
    Write-Host "  Duration: $([Math]::Round($duration, 2)) seconds"
    Write-Host ""

    return $results | Sort-Object WastedBytes -Descending
}

function ConvertFrom-PcaiDuplicateResult {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]$NativeResult,
        [Parameter(Mandatory)][string]$Algorithm,
        [Parameter(Mandatory)][int64]$MaximumSize
    )

    $results = [System.Collections.Generic.List[object]]::new()
    foreach ($group in @($NativeResult.Groups)) {
        if (-not $group) { continue }
        if ([int64]$group.Size -gt $MaximumSize) { continue }

        $files = @($group.Paths | Sort-Object)
        if ($files.Count -lt 2) { continue }

        $results.Add([PSCustomObject]@{
            Hash        = $group.Hash
            Algorithm   = $Algorithm
            FileSize    = [int64]$group.Size
            FileSizeMB  = [Math]::Round(([double]$group.Size / 1MB), 2)
            Count       = $files.Count
            WastedBytes = [int64]$group.WastedBytes
            WastedMB    = [Math]::Round(([double]$group.WastedBytes / 1MB), 2)
            Files       = $files
            Original    = $files[0]
            Duplicates  = if ($files.Count -gt 1) { @($files | Select-Object -Skip 1) } else { @() }
            Provider    = 'PcaiNative'
            DurationMs  = [int64]$NativeResult.ElapsedMs
        }) | Out-Null
    }

    Write-Host ""
    Write-Host "=== Duplicate Analysis Complete ===" -ForegroundColor Green
    Write-Host "  Total files scanned: $($NativeResult.FilesScanned)"
    Write-Host "  Duplicate groups found: $($results.Count)"
    Write-Host "  Total duplicate files: $($NativeResult.DuplicateFiles)"
    Write-Host "  Wasted space: $([Math]::Round(([double]$NativeResult.WastedBytes / 1MB), 2)) MB ($([Math]::Round(([double]$NativeResult.WastedBytes / 1GB), 2)) GB)"
    Write-Host "  Duration: $([Math]::Round(([double]$NativeResult.ElapsedMs / 1000), 2)) seconds"
    Write-Host ""

    return @($results | Sort-Object WastedBytes -Descending)
}

function Find-WithFdForDuplicates {
    [CmdletBinding()]
    param(
        [string]$Path,
        [switch]$Recurse,
        [string[]]$Include,
        [string[]]$Exclude,
        [string]$FdPath
    )

    $args = @('.', '-t', 'f', '-a')  # Pattern first, then type=file, absolute paths

    if (-not $Recurse) {
        $args += '-d'
        $args += '1'
    }

    foreach ($inc in $Include) {
        $args += '-e'
        $args += $inc.TrimStart('*.')
    }

    foreach ($exc in $Exclude) {
        $args += '-E'
        $args += $exc
    }

    $args += $Path

    try {
        $output = & $FdPath @args 2>&1
        $results = @()
        foreach ($line in $output) {
            if ($line -and (Test-Path $line -ErrorAction SilentlyContinue)) {
                $results += Get-Item $line -ErrorAction SilentlyContinue
            }
        }
        return $results
    }
    catch {
        Write-Warning "fd enumeration failed, falling back to Get-ChildItem"
        return Get-ChildItem -Path $Path -File -Recurse -ErrorAction SilentlyContinue
    }
}
