#Requires -Version 5.1
<#
.SYNOPSIS
Shrinks the boot delay on the cloud-cache-disk VHDX automount task.

.DESCRIPTION
AutoMount_VHDX_cloud-cache-disk fires at boot with a startup delay before it
attaches T:\vm\cloud-cache-disk.vhdx. Cloud sync clients autostart from Run keys
at logon, so the mount must win that race. On 2026-09-07 the margin was only
~70s (mount 09:13:29, logon 09:14:39).

Google Drive re-creates its own Run key on every start and so cannot be fully
gated behind the launcher; shrinking this delay is what actually widens the
margin for it.

⚠ CORRECTED 2026-09-09 - DO NOT RUN THIS WITHOUT READING THIS PARAGRAPH.
An earlier revision justified the shrink with "the VHDX lives on internal NVMe,
so a long fixed delay buys nothing". That premise was WRONG. cloud-cache-disk.vhdx
was on D:\, an 8 TB WD_BLACK SN850X in an ACASIS TBU405Pro Thunderbolt enclosure -
external, hot-plugged, and frequently not attached at all. Thunderbolt enumeration
does not complete within 5 s of a boot trigger, so the shrink made the race worse,
not better. Measured: ExitCode 51 (missing VHDX) on 8 boots between 2026-09-01 and
2026-09-09, zero before that.
The delay has been restored to PT1M30S and the mounter now takes -WaitForVhdSeconds,
a bounded poll for the backing file, which is the correct fix for a late-enumerating
disk. Only shrink this delay again once the VHDX genuinely lives on internal storage
(T:), and say so with evidence - `Get-Disk | Select BusType,Location` on the disk that
actually hosts the file, not an assumption.

Both the trigger delay and the script's own -StartupDelaySeconds argument are
updated together - changing only one leaves the other still sleeping.

.PARAMETER DelaySeconds
New startup delay in seconds. Default 90. A 5-second delay is what originally re-broke the mount, so going lower must be deliberate.

.PARAMETER DryRun
Non-mutating preview. The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Set-VhdxMountDelay.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = 'AutoMount_VHDX_cloud-cache-disk',
    # Default was 5 until 2026-09-09. That value came from the incorrect premise
    # corrected in the help above, and running this script with the old default
    # silently re-broke the mount. The default is now the safe value actually in
    # use; pass -DelaySeconds explicitly to go lower, deliberately.
    [ValidateRange(0, 600)]
    [int]$DelaySeconds = 90,
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') { $Help = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' }) }
if (@($CliArgs) -contains '--DryRun') { $DryRun = $true; $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' }) }
if ($Help) {
    $m = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($m.Success) { $m.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if ($DryRun) { $WhatIfPreference = $true }

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) { throw "Task not found: $TaskName" }

$newDelay = 'PT{0}S' -f $DelaySeconds
Write-Host "Task: $TaskName" -ForegroundColor Cyan

foreach ($tr in $task.Triggers) { Write-Host ("  trigger delay now : {0}" -f $tr.Delay) }
$oldArgs = $task.Actions[0].Arguments
Write-Host ("  action delay now  : {0}" -f ([regex]::Match($oldArgs, '-StartupDelaySeconds\s+(\d+)').Groups[1].Value))
Write-Host ("  target            : {0} / -StartupDelaySeconds {1}" -f $newDelay, $DelaySeconds)

$newArgs = [regex]::Replace($oldArgs, '(-StartupDelaySeconds\s+)\d+', ('${1}' + $DelaySeconds))
if ($newArgs -eq $oldArgs -and $oldArgs -notmatch '-StartupDelaySeconds') {
    Write-Warning '  action has no -StartupDelaySeconds argument; only the trigger will change.'
}

foreach ($tr in $task.Triggers) { $tr.Delay = $newDelay }
$action = New-ScheduledTaskAction -Execute $task.Actions[0].Execute -Argument $newArgs

if ($PSCmdlet.ShouldProcess($TaskName, "Set delay to ${DelaySeconds}s")) {
    Set-ScheduledTask -TaskName $TaskName -Trigger $task.Triggers -Action $action | Out-Null
    $after = Get-ScheduledTask -TaskName $TaskName
    Write-Host '  --- after ---' -ForegroundColor Green
    foreach ($tr in $after.Triggers) { Write-Host ("  trigger delay : {0}" -f $tr.Delay) }
    Write-Host ("  action        : {0}" -f ([regex]::Match($after.Actions[0].Arguments, '-StartupDelaySeconds\s+\d+').Value))
    Write-Host ("  state         : {0}" -f $after.State)
}
