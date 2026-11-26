#Requires -Version 7.0

<#
.SYNOPSIS
Reports retirement of the former ASUS fleet NFS share.
.DESCRIPTION
Status, DryRun and WhatIf report retirement without contacting the network.
The legacy parameters remain accepted so existing callers receive an explicit
disposition. Apply requests to mount or unmount, and task-template requests,
fail before invoking any native tool or changing a drive or scheduled task.
.EXAMPLE
./Mount-VigilFleetShare.ps1 -Action Mount -DryRun
#>
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', 'ProbeTimeoutMilliseconds', Justification = 'Accept legacy timeout arguments without starting a probe.')]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSReviewUnusedParameter', 'NativeTimeoutSeconds', Justification = 'Accept legacy timeout arguments without starting a native process.')]
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [ValidateSet('Status', 'Mount', 'Unmount')][string]$Action = 'Status',
    [ValidatePattern('^[D-Zd-z]:?$')][string]$Drive = 'N:',
    [switch]$Apply,
    [switch]$DryRun,
    [ValidateRange(100, 3000)][int]$ProbeTimeoutMilliseconds = 750,
    [ValidateRange(1, 30)][int]$NativeTimeoutSeconds = 8,
    [switch]$ShowTaskTemplate,
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments = @()
)

if ($ShowHelp -or '--help' -in $RemainingArguments) {
    Get-Help $PSCommandPath -Detailed
    return
}
if ($RemainingArguments.Count -gt 0) {
    throw 'Unrecognized positional argument. Use -h for help.'
}

$retirement = 'The ASUS fleet NFS share is retired.'
if ($ShowTaskTemplate -or ($Apply -and $Action -ne 'Status' -and -not $DryRun -and -not $WhatIfPreference)) {
    throw $retirement
}

[pscustomobject]@{
    Action = $Action
    Drive = $Drive.TrimEnd(':').ToUpperInvariant() + ':'
    State = 'Retired'
    Endpoint = $null
    Native = $null
    Reason = $retirement
}
