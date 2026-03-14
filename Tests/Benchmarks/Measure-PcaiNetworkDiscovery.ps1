#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateRange(1, 100)]
    [int]$Iterations = 3,

    [Parameter()]
    [string[]]$ComputerNameCandidates = @(
        'DESKTOP-86FC90B',
        'dtm-supernuc',
        'dtm-work',
        'dtm-work-mobo'
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$driversManifest = Join-Path $repoRoot 'Modules\PC-AI.Drivers\PC-AI.Drivers.psd1'
Import-Module $driversManifest -Force

function Measure-Step {
    param(
        [Parameter(Mandatory)]
        [scriptblock]$Action
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $value = & $Action
    $stopwatch.Stop()

    [pscustomobject]@{
        DurationMs = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 2)
        Value      = $value
    }
}

$measurements = for ($iteration = 1; $iteration -le $Iterations; $iteration++) {
    $snapshotStep = Measure-Step {
        Get-NetworkDiscoverySnapshot
    }

    $peerStep = Measure-Step {
        Find-ThunderboltPeer -ComputerNameCandidates $ComputerNameCandidates
    }

    $thunderboltStep = Measure-Step {
        Get-ThunderboltNetworkStatus
    }

    [pscustomobject]@{
        iteration         = $iteration
        durationMs        = [Math]::Round(($snapshotStep.DurationMs + $peerStep.DurationMs + $thunderboltStep.DurationMs), 2)
        phases            = [pscustomobject]@{
            snapshotMs    = $snapshotStep.DurationMs
            peerProbeMs   = $peerStep.DurationMs
            thunderboltMs = $thunderboltStep.DurationMs
        }
        adapterCount      = @($snapshotStep.Value.Adapters).Count
        thunderboltCount  = @($thunderboltStep.Value).Count
        peerCandidateCount = @($peerStep.Value.Neighbors).Count
    }
}

[pscustomobject]@{
    timestampUtc            = [DateTime]::UtcNow.ToString('o')
    iterations              = $Iterations
    computerNameCandidates  = @($ComputerNameCandidates)
    measurements            = @($measurements)
} | ConvertTo-Json -Depth 8
