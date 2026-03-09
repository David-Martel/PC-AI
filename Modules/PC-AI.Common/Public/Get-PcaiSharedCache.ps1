#Requires -PSEdition Core

if (-not (Get-Variable -Name 'PcaiSharedCache' -Scope Global -ErrorAction SilentlyContinue) -or
    -not $global:PcaiSharedCache -or
    -not ($global:PcaiSharedCache -is [System.Collections.IDictionary]) -or
    -not $global:PcaiSharedCache.Contains('MaxEntries') -or
    -not $global:PcaiSharedCache.Contains('Entries') -or
    -not ($global:PcaiSharedCache.Entries -is [System.Collections.Specialized.OrderedDictionary])) {
    $global:PcaiSharedCache = @{
        MaxEntries = 256
        Entries    = [System.Collections.Specialized.OrderedDictionary]::new()
    }
}

$script:PcaiSharedCache = $global:PcaiSharedCache

function Copy-PcaiCacheValue {
    [CmdletBinding()]
    param(
        [Parameter()]
        [AllowNull()]
        [object]$Value
    )

    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [string] -or
        $Value -is [ValueType] -or
        $Value -is [datetime] -or
        $Value -is [timespan] -or
        $Value -is [guid]) {
        return $Value
    }

    if ($Value -is [System.Collections.IDictionary]) {
        $copy = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $copy[$key] = Copy-PcaiCacheValue -Value $Value[$key]
        }
        return $copy
    }

    if ($Value -is [System.Management.Automation.PSCustomObject] -or
        ($Value -is [psobject] -and -not ($Value -is [System.Collections.IEnumerable]))) {
        $copy = [ordered]@{}
        foreach ($prop in $Value.PSObject.Properties) {
            if ($prop.MemberType -notin @(
                    [System.Management.Automation.PSMemberTypes]::NoteProperty,
                    [System.Management.Automation.PSMemberTypes]::Property,
                    [System.Management.Automation.PSMemberTypes]::AliasProperty
                )) {
                continue
            }
            $copy[$prop.Name] = Copy-PcaiCacheValue -Value $prop.Value
        }
        return [PSCustomObject]$copy
    }

    if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [string])) {
        return @($Value | ForEach-Object { Copy-PcaiCacheValue -Value $_ })
    }

    return $Value
}

function Remove-StalePcaiCacheEntries {
    [CmdletBinding()]
    param()

    while ($script:PcaiSharedCache.Entries.Count -gt $script:PcaiSharedCache.MaxEntries) {
        $oldest = $null
        foreach ($candidate in $script:PcaiSharedCache.Entries.GetEnumerator()) {
            if (-not $oldest -or $candidate.Value.LastAccessedUtc -lt $oldest.Value.LastAccessedUtc) {
                $oldest = $candidate
            }
        }

        if (-not $oldest) {
            break
        }

        $script:PcaiSharedCache.Entries.Remove($oldest.Key)
    }
}

function Get-PcaiDependencyStamp {
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter()]
        [AllowEmptyCollection()]
        [object[]]$InputObject
    )

    if (-not $InputObject -or $InputObject.Count -eq 0) {
        return '<none>'
    }

    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($item in $InputObject) {
        if ($null -eq $item) {
            $parts.Add('<null>')
            continue
        }

        switch ($item) {
            { $_ -is [System.IO.FileSystemInfo] } {
                $parts.Add(('{0}|{1}|{2}|{3}' -f $_.FullName, $_.Exists, $_.Length, $_.LastWriteTimeUtc.Ticks))
                continue
            }
            { $_ -is [string] } {
                $literal = [string]$item
                if ([string]::IsNullOrWhiteSpace($literal)) {
                    $parts.Add('<empty>')
                } elseif (Test-Path -LiteralPath $literal) {
                    $resolvedItem = Get-Item -LiteralPath $literal -ErrorAction SilentlyContinue
                    if ($resolvedItem) {
                        $parts.Add(('{0}|{1}|{2}|{3}' -f $resolvedItem.FullName, $resolvedItem.Exists, $resolvedItem.Length, $resolvedItem.LastWriteTimeUtc.Ticks))
                    } else {
                        $parts.Add("missing::$literal")
                    }
                } else {
                    $parts.Add("missing::$literal")
                }
                continue
            }
            default {
                $parts.Add(($item | Out-String).Trim())
            }
        }
    }

    $payload = ($parts | Sort-Object) -join "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
        return [Convert]::ToHexString($sha.ComputeHash($bytes))
    }
    finally {
        $sha.Dispose()
    }
}

function Get-PcaiSharedCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter()]
        [int]$TtlSeconds = 0,

        [Parameter()]
        [string]$DependencyStamp
    )

    $fullKey = '{0}::{1}' -f $Namespace, $Key
    if (-not $script:PcaiSharedCache.Entries.Contains($fullKey)) {
        return $null
    }

    $entry = $script:PcaiSharedCache.Entries[$fullKey]
    $now = [datetime]::UtcNow
    if ($TtlSeconds -gt 0 -and ($now - $entry.CreatedUtc).TotalSeconds -gt $TtlSeconds) {
        $script:PcaiSharedCache.Entries.Remove($fullKey)
        return $null
    }

    if ($PSBoundParameters.ContainsKey('DependencyStamp') -and $DependencyStamp -ne $entry.DependencyStamp) {
        $script:PcaiSharedCache.Entries.Remove($fullKey)
        return $null
    }

    $entry.LastAccessedUtc = $now
    return (Copy-PcaiCacheValue -Value $entry.Value)
}

function Set-PcaiSharedCacheEntry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Namespace,

        [Parameter(Mandatory)]
        [string]$Key,

        [Parameter(Mandatory)]
        [AllowNull()]
        [object]$Value,

        [Parameter()]
        [string]$DependencyStamp
    )

    $fullKey = '{0}::{1}' -f $Namespace, $Key
    $now = [datetime]::UtcNow
    if ($script:PcaiSharedCache.Entries.Contains($fullKey)) {
        $script:PcaiSharedCache.Entries.Remove($fullKey)
    }

    $script:PcaiSharedCache.Entries.Add($fullKey, [PSCustomObject]@{
            Value           = Copy-PcaiCacheValue -Value $Value
            DependencyStamp = $DependencyStamp
            CreatedUtc      = $now
            LastAccessedUtc = $now
        })

    Remove-StalePcaiCacheEntries
    return (Copy-PcaiCacheValue -Value $Value)
}

function Clear-PcaiSharedCache {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$Namespace
    )

    if ([string]::IsNullOrWhiteSpace($Namespace)) {
        $script:PcaiSharedCache.Entries.Clear()
        return
    }

    foreach ($key in @($script:PcaiSharedCache.Entries.Keys)) {
        if ($key -like "$Namespace::*") {
            $script:PcaiSharedCache.Entries.Remove($key)
        }
    }
}
