#Requires -Version 5.1
<#
.SYNOPSIS
    One semantic health sweep of this host (scheduled every 15 min as an S4U task).

.DESCRIPTION
    Replays the agent-bus spool, runs the detectors, updates incidents, runs bounded playbooks
    and writes the ledger. Exit 0 when the sweep ran (incidents are reported through the ledger
    and agent-bus, not through this exit code); exit 2 when the sweep itself could not run, so
    the scheduled-task health check sees a broken sweeper. Docs: docs/reliability/fleet-ledger.md.

.PARAMETER PolicyPath
    Host policy JSON. Default: <FLEET_RELIABILITY_HOME or %ProgramData%\fleet-reliability>\policy.json.

.PARAMETER NoPlaybooks
    Observe only; never remediate.

.PARAMETER Detector
    Run only these detectors (task, disk, postgres, process, cert, runner, workflow).

.EXAMPLE
    pwsh -NoProfile -File Invoke-FleetHealthSweep.ps1 -NoPlaybooks
#>
[CmdletBinding()]
param(
    [string]$PolicyPath,
    [switch]$NoPlaybooks,
    [string[]]$Detector
)
$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $p = @{ NoPlaybooks = $NoPlaybooks }
    if ($PolicyPath) { $p.PolicyPath = $PolicyPath }
    if ($Detector) { $p.Detector = $Detector }
    $s = Invoke-FleetHealthSweep @p
    Write-Output ("sweep: {0} observations, {1} bad, opened {2}, resolved {3}, open {4}, playbooks [{5}], detector errors [{6}]" -f `
            $s.observations, $s.bad, $s.opened, $s.resolved, $s.open_incidents, ($s.playbooks -join ', '), ($s.detector_errors -join '; '))
    exit 0
}
catch {
    Write-Error "fleet health sweep failed: $($_.Exception.Message)"
    exit 2
}
