#Requires -Version 7.0
<#
.SYNOPSIS
    Plans legacy Windows Thunderbolt settings or applies guarded peer tuning.
.DESCRIPTION
    Compatibility bridge for PowerShell 7 (pwsh), including Windows peers. The
    default is a plan; it never deletes addresses, changes network categories or
    enables WinRM. Apply delegates to the maintained Optimize path, whose shared
    IPv4 guard verifies peer identity, address/route conflicts and preservation.
    Unrelated addresses must be resolved explicitly before static apply. Linux peers use the maintained
    LinuxPeer controller and a qualified peer profile instead.
.EXAMPLE
    ./Initialize-ThunderboltLink.ps1 -InterfaceAlias 'Ethernet 11' -DryRun
.EXAMPLE
    ./Initialize-ThunderboltLink.ps1 -InterfaceAlias 'Ethernet 11' -MetricOnly -Apply
.EXAMPLE
    ./Initialize-ThunderboltLink.ps1 --help
#>
[CmdletBinding(SupportsShouldProcess, PositionalBinding = $false)]
param(
    [string]$InterfaceAlias,
    [string]$IPv4Address,
    [ValidateRange(8, 30)][int]$PrefixLength = 30,
    [ValidateRange(1, 9999)][int]$InterfaceMetric = 15,
    [ValidateRange(1280, 65535)][int]$MtuBytes = 62000,
    [switch]$SetPrivateProfile,
    [switch]$EnablePsRemoting,
    [switch]$MetricOnly,
    [switch]$Apply,
    [switch]$DryRun,
    [ValidateSet('Local', 'WindowsPeer')][string]$CompatibilityRole = 'Local',
    [Alias('h', 'help')][switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments)][string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($ShowHelp -or $CliArgs -contains '--help') {
    $helpScript = if ($CompatibilityRole -eq 'WindowsPeer') { 'Bootstrap-ThunderboltPeerRemote.ps1' } else { 'Initialize-ThunderboltLink.ps1' }
    @"
Usage: pwsh -File $helpScript [-InterfaceAlias <exact alias>] [-DryRun]
       pwsh -File $helpScript -InterfaceAlias <exact alias> -MetricOnly -Apply [-WhatIf]
       pwsh -File $helpScript -InterfaceAlias <exact alias> -IPv4Address <peer IPv4> -PrefixLength 30 -Apply
       pwsh -File $helpScript -h | --help
PowerShell 7 is required. Ordinary invocation plans only. Static IPv4 Apply uses
the maintained driver's live identity/conflict/preservation guards. Private
network category and WinRM requests remain visible and require separate policy.
Use Invoke-ThunderboltNetworking.ps1 -Mode LinuxPeer for a qualified Linux peer;
Windows peers require pwsh; this bridge never starts Windows PowerShell or WinRM.
"@
    return
}
if ($CliArgs) {
    throw 'Use named parameters for legacy Thunderbolt settings; unrecognized positional arguments are not applied.'
}

if ($MetricOnly -and ($PSBoundParameters.ContainsKey('IPv4Address') -or $PSBoundParameters.ContainsKey('PrefixLength'))) {
    throw 'MetricOnly cannot be combined with requested IPv4Address or PrefixLength. No requested static setting will be silently discarded.'
}
$suggestedAddress = if ($CompatibilityRole -eq 'WindowsPeer') { '172.31.240.2' } else { '172.31.240.1' }
$addressIntent = $null
if (-not $MetricOnly) {
    $addressIntent = if ($PSBoundParameters.ContainsKey('IPv4Address')) { $IPv4Address } else { $suggestedAddress }
    $parsedAddress = $null
    if (-not [Net.IPAddress]::TryParse($addressIntent, [ref]$parsedAddress) -or
        $parsedAddress.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
        throw 'IPv4Address must be a valid IPv4 address. Use MetricOnly to request tuning without static addressing.'
    }
    $addressIntent = $parsedAddress.ToString()
}

$entrypoint = Join-Path $PSScriptRoot 'Invoke-ThunderboltNetworking.ps1'
$blockers = [Collections.Generic.List[string]]::new()
if ([string]::IsNullOrWhiteSpace($InterfaceAlias)) {
    $blockers.Add('Apply requires one explicit InterfaceAlias; the maintained optimizer verifies its exact peer identity.')
}
if ($SetPrivateProfile) {
    $blockers.Add('Private network profile changes require a separately owned host security policy; this bridge will not apply SetPrivateProfile.')
}
if ($EnablePsRemoting) {
    $blockers.Add('WinRM enablement requires a separately owned host security policy; this bridge will not apply EnablePsRemoting.')
}
$plan = [pscustomobject]@{
    State                      = 'Planned'
    Applied                    = $false
    CompatibilityRole          = $CompatibilityRole
    InterfaceAlias             = $InterfaceAlias
    MetricOnly                 = [bool]$MetricOnly
    InterfaceMetric            = $InterfaceMetric
    MtuBytes                   = $MtuBytes
    IPv4Address                = $addressIntent
    PrefixLength               = if ($MetricOnly) { $null } else { $PrefixLength }
    SuggestedCompatibilityIPv4 = $suggestedAddress
    SetPrivateProfile          = [bool]$SetPrivateProfile
    EnablePsRemoting           = [bool]$EnablePsRemoting
    ApplyBlockers              = @($blockers)
    StaticIPv4Safety           = 'Apply performs fresh peer identity, IPv4 address/route conflict and preservation checks through the maintained driver. Resolve existing unrelated addresses explicitly; a plan does not establish live safety.'
    MaintainedEntrypoint       = $entrypoint
    PowerShellRequirement      = 'PowerShell 7 (pwsh); no Windows PowerShell subprocess is started.'
}
if (-not $Apply -or $DryRun -or $WhatIfPreference) {
    return $plan
}
if ($blockers.Count -gt 0) {
    throw "Legacy Thunderbolt apply refused. $($blockers -join ' ')"
}
if (-not $PSCmdlet.ShouldProcess($InterfaceAlias, 'Apply requested peer settings through the maintained Thunderbolt optimizer and IPv4 preservation guard')) {
    return $plan
}

# Central Optimize owns native exits, exact peer identity and IPv4 preservation.
# Remoting and profile policy remain blocked above.
$delegateArguments = @{
    Mode            = 'Optimize'
    InterfaceAlias  = $InterfaceAlias
    InterfaceMetric = $InterfaceMetric
    MtuBytes        = $MtuBytes
    Apply           = $true
    Confirm         = $false
}
if (-not $MetricOnly) {
    $delegateArguments.IPv4Address = $addressIntent
    $delegateArguments.PrefixLength = $PrefixLength
}
$status = @(& $entrypoint @delegateArguments)
if ($status.Count -ne 1 -or $status[0].InterfaceAlias -ne $InterfaceAlias) {
    throw 'The maintained Thunderbolt optimizer did not return one verified result for the selected interface.'
}
$plan.State = 'Applied'
$plan.Applied = $true
$plan | Add-Member -NotePropertyName VerifiedStatus -NotePropertyValue $status[0]
return $plan
