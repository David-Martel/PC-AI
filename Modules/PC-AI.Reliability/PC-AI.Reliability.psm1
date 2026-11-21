#Requires -Version 5.1
<#
.SYNOPSIS
    PC-AI.Reliability: fleet failure ledger, semantic health sweeper, bounded reconciliation.

.DESCRIPTION
    Records every failure or anomaly as a fleet-ledger/1 JSONL event, forwards incidents to
    agent-bus (topic and thread fleet-failures) with an offline spool, alerts once per incident,
    and runs bounded playbooks. Design, contract and operations: docs/reliability/fleet-ledger.md.
#>

$script:ModuleRoot = $PSScriptRoot
# Deferred to a follow-up PR (docs/reliability/fleet-ledger.md, "Deferred"): automatic remediation and the
# privileged setup steps (ACL hardening, token provisioning) refuse to run. The ledger, spool, incidents,
# detectors, alerts and digest are complete and on. Only the unit tests switch this off, to keep the deferred
# code's regression tests running; production code never does.
$script:FleetDeferred = @{ remediation = $true; privileged_setup = $true }
foreach ($dir in @('Private', 'Public')) {
    $path = Join-Path $PSScriptRoot $dir
    if (Test-Path -LiteralPath $path) {
        Get-ChildItem -LiteralPath $path -Filter '*.ps1' | Sort-Object Name | ForEach-Object { . $_.FullName }
    }
}

Export-ModuleMember -Function @(
    'Get-FleetPolicy', 'Get-FleetReliabilityHome', 'Get-FleetRealProfile', 'Set-FleetNormalPriority',
    'New-FleetLedgerEvent', 'Write-FleetLedgerEvent', 'Send-FleetBusEvent', 'Invoke-FleetSpoolReplay',
    'Get-FleetDiscordToken', 'Set-FleetDiscordToken', 'Send-FleetDiscordEvent', 'Set-FleetHomeAcl', 'Get-FleetSecretDirAclViolation',
    'New-FleetObservation', 'Submit-FleetObservation', 'Get-FleetIncidentState', 'Resolve-FleetUnobservedIncidents',
    'Get-FleetTaskObservations', 'Get-FleetDiskObservations', 'Get-FleetDiskTrend', 'Get-FleetRunnerObservations',
    'Get-FleetWorkflowObservations', 'Get-FleetPostgresObservations', 'Get-FleetOrphanProcessObservations', 'Get-FleetCertObservations',
    'Invoke-FleetPlaybook', 'Test-FleetCacheDirAllowed', 'Invoke-FleetCachePrune',
    'Invoke-FleetTask', 'Invoke-FleetHealthSweep', 'Send-FleetDigest', 'Read-FleetLedger'
)
