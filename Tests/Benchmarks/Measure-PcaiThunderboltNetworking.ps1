#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateRange(1, 100)]
    [int]$Iterations = 3,

    [Parameter()]
    [string]$InterfaceAlias
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-WindowsPowerShellJson {
    param(
        [Parameter(Mandatory)]
        [string]$Script
    )

    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    $output = & powershell.exe -NoLogo -NoProfile -EncodedCommand $encoded 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'Windows PowerShell network probe failed.'
    }

    $text = ($output -join [Environment]::NewLine).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    return $text | ConvertFrom-Json -Depth 12
}

function Measure-Step {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$Action
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $value = & $Action
    $stopwatch.Stop()

    return [pscustomobject]@{
        DurationMs = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 2)
        Value      = $value
    }
}

if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
    $InterfaceAlias = Invoke-WindowsPowerShellJson -Script @"
Get-NetAdapter -IncludeHidden -ErrorAction SilentlyContinue |
    Where-Object {
        `$_.Status -eq 'Up' -and (
            `$_.InterfaceDescription -match 'USB4' -or
            `$_.InterfaceDescription -match 'Thunderbolt' -or
            `$_.InterfaceDescription -match 'P2P'
        )
    } |
    Sort-Object ifIndex |
    Select-Object -First 1 -ExpandProperty Name |
    ConvertTo-Json -Depth 2
"@
}

$measurements = for ($iteration = 1; $iteration -le $Iterations; $iteration++) {
    $adapterStep = Measure-Step {
        Invoke-WindowsPowerShellJson -Script @"
Get-NetAdapter -Name '$InterfaceAlias' -ErrorAction Stop |
    Select-Object Name, InterfaceDescription, Status, MacAddress, LinkSpeed, ifIndex |
    ConvertTo-Json -Depth 6
"@
    }

    $ifIndex = $adapterStep.Value.ifIndex

    $ipStep = Measure-Step {
        Invoke-WindowsPowerShellJson -Script @"
Get-NetIPAddress -InterfaceIndex $ifIndex -ErrorAction SilentlyContinue |
    Select-Object IPAddress, AddressFamily, PrefixLength |
    ConvertTo-Json -Depth 6
"@
    }

    $neighborStep = Measure-Step {
        Invoke-WindowsPowerShellJson -Script @"
Get-NetNeighbor -InterfaceIndex $ifIndex -ErrorAction SilentlyContinue |
    Where-Object {
        `$_.IPAddress -and
        `$_.State -notin @(0, '0', 'Unreachable', 'Invalid') -and
        `$_.IPAddress -notin @('255.255.255.255', '169.254.255.255') -and
        `$_.IPAddress -notmatch '^(22[4-9]|23\d)\.' -and
        `$_.IPAddress -notlike 'ff*' -and
        `$_.LinkLayerAddress -and
        `$_.LinkLayerAddress -notin @('00-00-00-00-00-00', '00:00:00:00:00:00', 'FF-FF-FF-FF-FF-FF', 'ff-ff-ff-ff-ff-ff', 'ff:ff:ff:ff:ff:ff')
    } |
    Select-Object IPAddress, LinkLayerAddress, State |
    ConvertTo-Json -Depth 6
"@
    }

    $arpStep = Measure-Step {
        @(arp -a)
    }

    $totalDuration = [Math]::Round(($adapterStep.DurationMs + $ipStep.DurationMs + $neighborStep.DurationMs + $arpStep.DurationMs), 2)

    [pscustomobject]@{
        iteration   = $iteration
        durationMs  = $totalDuration
        phases      = [pscustomobject]@{
            adapterMs   = $adapterStep.DurationMs
            ipConfigMs  = $ipStep.DurationMs
            neighborMs  = $neighborStep.DurationMs
            arpMs       = $arpStep.DurationMs
        }
        localLink   = $adapterStep.Value
        addressCount = @($ipStep.Value).Count
        neighborCount = @($neighborStep.Value).Count
    }
}

[pscustomobject]@{
    timestampUtc = [DateTime]::UtcNow.ToString('o')
    interfaceAlias = $InterfaceAlias
    iterations = $Iterations
    measurements = @($measurements)
} | ConvertTo-Json -Depth 8
