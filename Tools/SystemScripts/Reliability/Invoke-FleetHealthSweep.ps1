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
    Observe and record, but never remediate.

.PARAMETER Detector
    Run only these detectors (task, disk, postgres, process, cert, runner, workflow).

.PARAMETER DryRun
    Run the detectors and print their observations with no side effect: no ledger, state, spool,
    bus message, Discord message or remediation. `--DryRun` also works.

.PARAMETER Help
    Print this help and exit. `-h`, `-?` and `--help` also work.

.EXAMPLE
    pwsh -NoProfile -File Invoke-FleetHealthSweep.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$PolicyPath,
    [switch]$NoPlaybooks,
    [string[]]$Detector,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)
$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($CliArgs -contains '--help') { $Help = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' }) }
if ($CliArgs -contains '--DryRun') { $DryRun = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' }) }
if ($Help) {
    $m = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    Write-Output $m.Groups[1].Value.Trim()
    exit 0
}
if ($CliArgs.Count -gt 0) { Write-Error "Unknown argument(s): $($CliArgs -join ', ')"; exit 2 }

$ErrorActionPreference = 'Stop'
try {
    Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force
    $null = Set-FleetNormalPriority   # tasks start BelowNormal; a starved monitor goes quiet under load
    $p = @{}
    if ($PolicyPath) { $p.PolicyPath = $PolicyPath }
    if ($Detector) { $p.Detector = $Detector }
    if ($DryRun) {
        $obs = @(Invoke-FleetHealthSweep @p -DryRun)
        Write-Output ("DRY RUN: {0} observations, {1} not ok (nothing recorded, sent or remediated)" -f $obs.Count, @($obs | Where-Object { @('ok', 'skipped') -notcontains $_.Status }).Count)
        $obs | Where-Object { @('ok', 'skipped') -notcontains $_.Status } | ForEach-Object { Write-Output ("  [{0}] {1}/{2} ({3}): {4}" -f $_.Status, $_.Source, $_.Name, $_.Check, $_.Message) }
        exit 0
    }
    $s = Invoke-FleetHealthSweep @p -NoPlaybooks:$NoPlaybooks
    Write-Output ("sweep: {0} observations, {1} bad, opened {2}, resolved {3}, open {4}, playbooks [{5}], detector errors [{6}]" -f `
            $s.observations, $s.bad, $s.opened, $s.resolved, $s.open_incidents, ($s.playbooks -join ', '), ($s.detector_errors -join '; '))
    exit 0
}
catch {
    Write-Error "fleet health sweep failed: $($_.Exception.Message)"
    exit 2
}
