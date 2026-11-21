#Requires -Version 5.1
<#
.SYNOPSIS
    Registers the fleet reliability sweeper (and optionally the daily digest) as S4U tasks.

.DESCRIPTION
    S4U ("run whether the user is logged on or not", no stored password) because Interactive tasks
    never fire on a host nobody is logged on to. The tasks run as the current user so agent-bus
    finds its remote configuration in the real profile.

    The fleet-reliability home (%ProgramData%\fleet-reliability) gets an explicit ACL: SYSTEM,
    Administrators and the installing user, inheritance removed. An existing task with the same
    name is exported to <home>\backups\<task>.<timestamp>.xml before it is replaced, and the
    revert commands are printed. Nothing else on the host is changed. Use -WhatIf to preview.

.PARAMETER SweepIntervalMinutes
    How often the sweep runs. Default 15.

.PARAMETER Digest
    Also register the daily digest at -DigestTime (local time). Use -FleetDigest on exactly one host.

.EXAMPLE
    pwsh -NoProfile -File Install-FleetReliabilityTasks.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [int]$SweepIntervalMinutes = 15,
    [switch]$Digest,
    [switch]$FleetDigest,
    [string]$DigestTime = '07:30',
    [string]$TaskFolder = '\fleet-reliability\'
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '..\..\..\Modules\PC-AI.Reliability\PC-AI.Reliability.psd1') -Force

$pwsh = (Get-Process -Id $PID).Path
if ($pwsh -notmatch '(?i)pwsh(\.exe)?$') {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'PowerShell 7 (pwsh) is required for the sweeper (Test-ScheduledTaskHealth.ps1 needs it).' }
    $pwsh = $cmd.Source
}
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$homeDir = Get-FleetReliabilityHome
$backupDir = Join-Path $homeDir 'backups'

if ($PSCmdlet.ShouldProcess($homeDir, 'create with ACL SYSTEM/Administrators/' + $user)) {
    foreach ($d in @($homeDir, $backupDir)) { if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null } }
    & icacls.exe $homeDir /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' "${user}:(OI)(CI)F" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "icacls failed on $homeDir (exit $LASTEXITCODE)" }
}

$scripts = Join-Path $PSScriptRoot '.'
$tasks = @(
    @{
        Name     = 'health-sweep'
        Args     = ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}"' -f (Join-Path $scripts 'Invoke-FleetHealthSweep.ps1'))
        Trigger  = (New-ScheduledTaskTrigger -Once -At ((Get-Date).Date.AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes $SweepIntervalMinutes))
        Limit    = (New-TimeSpan -Minutes 20)
    }
)
if ($Digest -or $FleetDigest) {
    $dArgs = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}"' -f (Join-Path $scripts 'Send-FleetDigest.ps1')
    if ($FleetDigest) { $dArgs += ' -Fleet' }
    $tasks += @{ Name = 'daily-digest'; Args = $dArgs; Trigger = (New-ScheduledTaskTrigger -Daily -At $DigestTime); Limit = (New-TimeSpan -Minutes 10) }
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$reverts = @()
foreach ($t in $tasks) {
    $existing = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $t.Name -ErrorAction SilentlyContinue
    if ($existing -and $PSCmdlet.ShouldProcess("$TaskFolder$($t.Name)", 'export existing definition')) {
        $bak = Join-Path $backupDir ("{0}.{1}.xml" -f $t.Name, $stamp)
        Export-ScheduledTask -TaskPath $TaskFolder -TaskName $t.Name | Set-Content -LiteralPath $bak -Encoding Unicode
        $reverts += "Register-ScheduledTask -TaskPath '$TaskFolder' -TaskName '$($t.Name)' -Xml (Get-Content -Raw '$bak') -Force"
    }
    else {
        $reverts += "Unregister-ScheduledTask -TaskPath '$TaskFolder' -TaskName '$($t.Name)' -Confirm:`$false"
    }
    $action = New-ScheduledTaskAction -Execute $pwsh -Argument $t.Args
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit $t.Limit -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    if ($PSCmdlet.ShouldProcess("$TaskFolder$($t.Name)", "register S4U task as $user")) {
        Register-ScheduledTask -TaskPath $TaskFolder -TaskName $t.Name -Action $action -Trigger $t.Trigger -Principal $principal -Settings $settings -Force | Out-Null
    }
}
Write-Output 'Revert:'
$reverts | ForEach-Object { Write-Output "  $_" }
