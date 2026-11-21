@{
    RootModule        = 'PC-AI.Reliability.psm1'
    ModuleVersion     = '1.0.0'
    GUID              = '3fa80ddd-a14c-4acc-9241-668beb655542'
    Author            = 'PC_AI Framework'
    CompanyName       = 'PC_AI'
    Copyright         = '(c) 2026 PC_AI Framework. All rights reserved.'
    Description       = 'Fleet failure ledger (fleet-ledger/1 JSONL + agent-bus), semantic health sweeper, incident dedup, bounded reconciliation playbooks and daily digest.'
    PowerShellVersion = '5.1'
    FunctionsToExport = @(
        'Get-FleetPolicy', 'Get-FleetReliabilityHome', 'Get-FleetRealProfile',
        'New-FleetLedgerEvent', 'Write-FleetLedgerEvent', 'Send-FleetBusEvent', 'Invoke-FleetSpoolReplay',
        'Get-FleetDiscordToken', 'Set-FleetDiscordToken', 'Send-FleetDiscordEvent',
        'New-FleetObservation', 'Submit-FleetObservation', 'Get-FleetIncidentState', 'Resolve-FleetUnobservedIncidents',
        'Get-FleetTaskObservations', 'Get-FleetDiskObservations', 'Get-FleetDiskTrend', 'Get-FleetRunnerObservations',
        'Get-FleetWorkflowObservations', 'Get-FleetPostgresObservations', 'Get-FleetOrphanProcessObservations', 'Get-FleetCertObservations',
        'Invoke-FleetPlaybook', 'Test-FleetCacheDirAllowed', 'Invoke-FleetCachePrune',
        'Invoke-FleetTask', 'Invoke-FleetHealthSweep', 'Send-FleetDigest', 'Read-FleetLedger'
    )
    CmdletsToExport   = @()
    VariablesToExport = @()
    AliasesToExport   = @()
    PrivateData       = @{ PSData = @{ Tags = @('reliability', 'monitoring', 'agent-bus', 'scheduled-tasks') } }
}
