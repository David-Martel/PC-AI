#Requires -Version 7.0
<#
.SYNOPSIS
    Read-only PATH and binary-location audit for one Windows host.

.DESCRIPTION
    Checks the user and machine PATH in the registry against machine-path.policy.json: dead,
    duplicate, forbidden, relative and %USERPROFILE%-relative entries, REG_SZ values that need
    expansion, length, order. Then checks the managed bin directories for shadowed or mismatched
    commands, order-dependent resolution, SHA-suffixed variants, broken links and dead wrappers.

    Exit code: 0 no errors or warnings, 1 warnings, 2 errors. Nothing is written except with -OutFile.
    Hashing is on by default for names that appear in more than one place (it reads only those files).
    Never executes a probed binary, so agent-bus-http and agent-bus-mcp are safe.

.PARAMETER Format
    Human (default) or Json.

.PARAMETER Timing
    Adds lookup timings: the fast resolver, Get-Command for a present and a missing command, and the
    PowerShell module auto-discovery share of the missing-command cost.

.PARAMETER Baseline
    Emits the resolution baseline (winner, size, SHA-256 and all candidates for each policy command).

.PARAMETER OutFile
    Writes the Json result to this file as well.

.EXAMPLE
    pwsh -NoProfile -File Test-MachinePath.ps1 -Timing

.LINK
    Docs/MACHINE_PATH_POLICY.md
#>
[CmdletBinding()]
param(
    [ValidateSet('Human', 'Json')][string]$Format = 'Human',
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'machine-path.policy.json'),
    [string]$UserHome,
    [switch]$Timing,
    [switch]$Baseline,
    [switch]$NoHash,
    [string]$OutFile
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'MachinePath.psm1') -Force

$ctx = Get-MachinePathContext -UserHome $UserHome
$policy = Read-MachinePathPolicy -Path $PolicyPath -Context $ctx
$user = Get-MachinePathRegistry -Scope User
$machine = Get-MachinePathRegistry -Scope Machine

$findings = [System.Collections.Generic.List[object]]::new()
$findings.AddRange(@(Test-MachinePathEntries -UserEntries $user.Entries -MachineEntries $machine.Entries -UserKind $user.Kind -MachineKind $machine.Kind -Policy $policy -Context $ctx))

$userDirs = @($user.Entries | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $ctx } | Where-Object { [IO.Directory]::Exists($_) } | Select-Object -Unique)
$machineDirs = @($machine.Entries | ForEach-Object { Resolve-MachinePathEntry -Entry $_ -Context $ctx } | Where-Object { [IO.Directory]::Exists($_) } | Select-Object -Unique)
$findings.AddRange(@(Test-MachinePathShadows -UserDirectories $userDirs -MachineDirectories $machineDirs -Policy $policy -Hash:(-not $NoHash)))
$findings.AddRange(@(Test-MachinePathExtensionless -Directories $policy.ManagedBinDirs -AllDirectories @($userDirs + $machineDirs | Select-Object -Unique)))
$findings.AddRange(@(Test-MachinePathWrappers -Directories $policy.ManagedBinDirs -Context $ctx -AgentHome $(if ($env:USERPROFILE -ne $ctx.UserHome) { $env:USERPROFILE })))

$variants = @(foreach ($d in $policy.ManagedBinDirs) { Get-MachineVariantFile -Directory $d })
foreach ($v in $variants) { $findings.Add((New-MachinePathFinding Info 'PATH-VARIANT' '' $v 'Versioned or temporary copy of a binary next to the canonical one.')) }

$ext = @(($env:PATHEXT -split ';') | ForEach-Object { $_.ToUpperInvariant() })
$pathExt = [Environment]::GetEnvironmentVariable('PATHEXT', 'User'); if (-not $pathExt) { $pathExt = [Environment]::GetEnvironmentVariable('PATHEXT', 'Machine') }
foreach ($need in $policy.RequiredPathExt) {
    if ($pathExt -and (($pathExt -split ';') | ForEach-Object { $_.ToUpperInvariant() }) -notcontains $need) {
        $findings.Add((New-MachinePathFinding Error 'PATHEXT-MISSING' '' $need 'Required extension is missing from PATHEXT.'))
    }
}

foreach ($e in $policy.RequiredUserEnv.PSObject.Properties) {
    $actual = [Environment]::GetEnvironmentVariable($e.Name, 'User')
    if ($actual -ne $e.Value) {
        $findings.Add((New-MachinePathFinding Warn 'ENV-REQUIRED' 'User' $e.Name "Expected '$($e.Value)', found '$actual'. Without it the dotnet SDK appends an agent-home .dotnet	ools directory to the user PATH on first run."))
    }
}

$dirStats = foreach ($d in @($userDirs + $machineDirs)) {
    [pscustomobject]@{ Dir = $d; Files = @([IO.Directory]::EnumerateFileSystemEntries($d)).Count }
}

$result = [ordered]@{
    Host       = $env:COMPUTERNAME
    UserHome   = $ctx.UserHome
    CargoHome  = $ctx.CargoHome
    Process    = [ordered]@{ UserProfile = $env:USERPROFILE; AgentHomeRedirected = ($env:USERPROFILE -ne $ctx.UserHome) }
    Path       = [ordered]@{
        UserEntries = $user.Entries.Count; UserKind = $user.Kind; UserLength = $user.Raw.Length
        MachineEntries = $machine.Entries.Count; MachineKind = $machine.Kind; MachineLength = $machine.Raw.Length
        ProcessEntries = @($env:Path -split ';' | Where-Object { $_ }).Count
    }
    LargestDirs = @($dirStats | Sort-Object Files -Descending | Select-Object -First 5)
    Counts     = [ordered]@{
        Error = @($findings | Where-Object Severity -eq 'Error').Count
        Warn  = @($findings | Where-Object Severity -eq 'Warn').Count
        Info  = @($findings | Where-Object Severity -eq 'Info').Count
    }
    Findings   = @($findings)
}

if ($Timing) {
    $dirs = @($env:Path -split ';' | Where-Object { $_ })
    $sw = [Diagnostics.Stopwatch]::StartNew()
    foreach ($i in 1..20) { [void](Resolve-MachineCommand -Name 'definitelymissingcmd123' -Directories $dirs) }
    $fastMiss = $sw.Elapsed.TotalMilliseconds / 20
    $sw.Restart(); foreach ($i in 1..20) { [void](Resolve-MachineCommand -Name 'git' -Directories $dirs) }
    $fastHit = $sw.Elapsed.TotalMilliseconds / 20
    $gcMiss = (Measure-Command { Get-Command definitelymissingcmd123 -ErrorAction SilentlyContinue | Out-Null }).TotalMilliseconds
    $gcHit = (Measure-Command { Get-Command git -ErrorAction SilentlyContinue | Out-Null }).TotalMilliseconds
    $saved = $PSModuleAutoLoadingPreference
    $PSModuleAutoLoadingPreference = 'None'
    $gcMissNoAuto = (Measure-Command { Get-Command definitelymissingcmd123 -ErrorAction SilentlyContinue | Out-Null }).TotalMilliseconds
    $PSModuleAutoLoadingPreference = $saved
    $result.Timing = [ordered]@{
        FastResolverMissMs = [math]::Round($fastMiss, 1); FastResolverHitMs = [math]::Round($fastHit, 1)
        GetCommandMissMs = [math]::Round($gcMiss); GetCommandHitMs = [math]::Round($gcHit)
        GetCommandMissNoModuleAutoloadMs = [math]::Round($gcMissNoAuto)
        PSModulePathEntries = @($env:PSModulePath -split ';').Count
    }
}
if ($Baseline) {
    $all = @($userDirs + $machineDirs | Select-Object -Unique)
    $result.Baseline = @(Get-MachineResolutionBaseline -Commands $policy.BaselineCommands -Directories $all -NoHash:$NoHash)
}

if ($OutFile) { $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutFile -Encoding utf8NoBOM }
if ($Format -eq 'Json') { $result | ConvertTo-Json -Depth 6 }
else {
    "{0}: user {1} entries ({2} chars, {3}), machine {4} entries ({5} chars, {6}); agent-home redirected: {7}" -f $result.Host, $user.Entries.Count, $user.Raw.Length, $user.Kind, $machine.Entries.Count, $machine.Raw.Length, $machine.Kind, $result.Process.AgentHomeRedirected
    "findings: {0} error, {1} warn, {2} info" -f $result.Counts.Error, $result.Counts.Warn, $result.Counts.Info
    $showInfo = $VerbosePreference -ne 'SilentlyContinue'
    $findings | Where-Object { $_.Severity -ne 'Info' -or $showInfo } |
        Sort-Object @{e = { @{Error = 0; Warn = 1; Info = 2 }[$_.Severity] } }, Code |
        ForEach-Object { '[{0}] {1} {2} {3} :: {4}' -f $_.Severity.ToUpper(), $_.Code, $_.Scope, $_.Subject, $_.Message }
    if (-not $showInfo) { $findings | Where-Object Severity -eq 'Info' | Group-Object Code | ForEach-Object { 'info {0} x{1} (use -Verbose to list)' -f $_.Name, $_.Count } }
    if ($Timing) { $result.Timing.GetEnumerator() | ForEach-Object { "timing {0} = {1}" -f $_.Key, $_.Value } }
}
if ($result.Counts.Error -gt 0) { exit 2 } elseif ($result.Counts.Warn -gt 0) { exit 1 } else { exit 0 }
