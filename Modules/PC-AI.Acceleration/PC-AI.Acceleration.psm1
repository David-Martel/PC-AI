#Requires -Version 7.0
<#
.SYNOPSIS
    PC-AI Acceleration Module - Rust and .NET performance optimizations

.DESCRIPTION
    Provides high-performance alternatives to standard PowerShell operations
    using Rust CLI tools (ripgrep, fd, procs) and .NET parallel processing.
#>

# Module-level tool cache
$script:RustToolCache = @{}
$script:ToolPaths = @{
    rg       = $null
    fd       = $null
    bat      = $null
    procs    = $null
    'pcai-perf' = $null
    tokei    = $null
    sd       = $null
    eza      = $null
    hyperfine = $null
}
$script:PcaiQueryCache = [PSCustomObject]@{
    MaxEntries = 128
    Entries    = [System.Collections.Specialized.OrderedDictionary]::new()
}

# Initialize tool detection on module load
$script:SearchPaths = @(
    "$env:USERPROFILE\.cargo\bin"
    "$env:USERPROFILE\bin"
    "$env:USERPROFILE\.local\bin"
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    "C:\Program Files\ripgrep"
)

function Get-PcaiCacheKey {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Category,
        [hashtable]$Parameters = @{}
    )

    if (-not $Parameters -or $Parameters.Count -eq 0) {
        return "$Category::"
    }

    $keys = [System.Collections.Generic.List[string]]::new()
    foreach ($key in $Parameters.Keys) {
        $keys.Add([string]$key)
    }
    $keys.Sort([System.StringComparer]::Ordinal)

    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append($Category)
    [void]$builder.Append('::')

    for ($index = 0; $index -lt $keys.Count; $index++) {
        if ($index -gt 0) {
            [void]$builder.Append(';')
        }

        $key = $keys[$index]
        $value = $Parameters[$key]
        [void]$builder.Append($key)
        [void]$builder.Append('=')
        if ($null -eq $value) {
            [void]$builder.Append('<null>')
        } else {
            [void]$builder.Append([string]$value)
        }
    }

    return $builder.ToString()
}

function Get-PcaiCachedValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Key,
        [int]$TtlSeconds = 15
    )

    if (-not $script:PcaiQueryCache.Entries.Contains($Key)) {
        return $null
    }

    $entry = $script:PcaiQueryCache.Entries[$Key]
    if ($TtlSeconds -gt 0 -and ((Get-Date) - $entry.Created).TotalSeconds -gt $TtlSeconds) {
        $script:PcaiQueryCache.Entries.Remove($Key)
        return $null
    }

    $entry.LastAccessed = Get-Date
    return $entry.Value
}

function Set-PcaiCachedValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Key,
        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value
    )

    $now = Get-Date
    if ($script:PcaiQueryCache.Entries.Contains($Key)) {
        $script:PcaiQueryCache.Entries.Remove($Key)
    }

    $script:PcaiQueryCache.Entries.Add($Key, [PSCustomObject]@{
            Value        = $Value
            Created      = $now
            LastAccessed = $now
        })

    while ($script:PcaiQueryCache.Entries.Count -gt $script:PcaiQueryCache.MaxEntries) {
        $oldest = $null
        foreach ($candidate in $script:PcaiQueryCache.Entries.GetEnumerator()) {
            if (-not $oldest -or $candidate.Value.LastAccessed -lt $oldest.Value.LastAccessed) {
                $oldest = $candidate
            }
        }

        if ($oldest) {
            $script:PcaiQueryCache.Entries.Remove($oldest.Key)
        } else {
            break
        }
    }

    return $Value
}

function Resolve-PcaiPath {
    [CmdletBinding()]
    param(
        [string]$Path = '.'
    )

    $resolved = Resolve-Path -Path $Path -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($resolved) {
        return $resolved.Path
    }

    return $Path
}

function Get-PcaiRelativeDepth {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$BasePath,
        [Parameter(Mandatory)]
        [string]$CandidatePath
    )

    $normalizedBase = $BasePath.TrimEnd('\', '/')
    $normalizedCandidate = $CandidatePath.TrimEnd('\', '/')
    if (-not $normalizedCandidate.StartsWith($normalizedBase, [System.StringComparison]::OrdinalIgnoreCase)) {
        return [int]::MaxValue
    }

    $relative = $normalizedCandidate.Substring($normalizedBase.Length).TrimStart('\', '/')
    if ([string]::IsNullOrWhiteSpace($relative)) {
        return 0
    }

    return ([regex]::Matches($relative, '[\\/]').Count)
}

function New-PcaiPathItem {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [string]$Type = 'file'
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    if ($Type -in @('directory', 'd')) {
        return [System.IO.DirectoryInfo]::new($Path)
    }

    return [System.IO.FileInfo]::new($Path)
}

# Dot-source function files. Keep this import path lean because module parse and
# script discovery still dominate cold-start time.
$PublicPath = Join-Path $PSScriptRoot 'Public'
$PrivatePath = Join-Path $PSScriptRoot 'Private'

foreach ($scriptDirectory in @($PrivatePath, $PublicPath)) {
    if (-not (Test-Path -LiteralPath $scriptDirectory -PathType Container)) {
        continue
    }

    foreach ($scriptFile in (Get-ChildItem -LiteralPath $scriptDirectory -Filter '*.ps1' -File -ErrorAction SilentlyContinue | Sort-Object -Property Name)) {
        . $scriptFile.FullName
    }
}

# Tool and native discovery stay lazy. The public entrypoints cache the first
# successful lookup and only initialize native support when a native-backed
# command is actually used.

Export-ModuleMember -Function @(
    'Get-RustToolStatus'
    'Test-RustToolAvailable'
    'Search-LogsFast'
    'Find-FilesFast'
    'Get-ProcessesFast'
    'Get-FileHashParallel'
    'Find-DuplicatesFast'
    'Get-DiskUsageFast'
    'Search-ContentFast'
    'Measure-CommandPerformance'
    'Compare-ToolPerformance'
    'Get-PcaiCapabilities'
    'Get-UnifiedHardwareReportJson'
    'Initialize-PcaiNative'
    'Test-PcaiNativeAvailable'
    'Get-PcaiNativeStatus'
    'Invoke-PcaiNativeDuplicates'
    'Invoke-PcaiNativeFileSearch'
    'Invoke-PcaiNativeContentSearch'
    'Invoke-PcaiNativeDirectoryManifest'
    'Invoke-PcaiNativeSystemInfo'
    'Test-PcaiResourceSafety'
    'Invoke-PcaiNativeUnifiedHardwareReport'
    'Invoke-PcaiNativeEstimateTokens'
)
