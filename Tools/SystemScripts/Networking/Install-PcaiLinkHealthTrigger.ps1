#Requires -Version 7.2
<#
.SYNOPSIS
Installs, updates or removes the startup and reconnect triggers for the PC-AI health checks.
.DESCRIPTION
Copies the checks into a content-addressed directory and points the triggers at that copy, so a
checkout that changes branch never changes what a trigger runs.

Windows, per user (no elevation needed for the task itself):
  \PC-AI\LinkHealth      logon (+45 s), USB4 arrival/removal, network connect/disconnect,
                         resume, and every 30 minutes. Runs Invoke-PcaiLinkHealth.ps1 (Observe).
  \PC-AI\ToolingHealth   logon (+2 min), network connect, and every 4 hours, when
                         Invoke-PcaiToolingHealth.ps1 is installed next to this script's checks.
Windows, -System (elevated; the owner's opt-in step):
  \PC-AI\LinkHealthSystem  boot (+1 min) plus the same events, as SYSTEM, running the check in
                           Apply mode from an admin-owned copy. It still changes nothing unless
                           the admin-owned auto-apply policy enables it.
Linux, per user (systemd --user; needs linger for boot-time runs):
  pcai-link-health.timer + .service, pcai-link-health-monitor.service (ip monitor), and
  pcai-tooling-health.timer + .service.

Every replaced task definition or unit file is saved first; -Uninstall removes what this script
installed and, with -Restore, re-registers the saved definitions.
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -DryRun
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -RunNow
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Networking/Install-PcaiLinkHealthTrigger.ps1 -Uninstall
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('LinkHealth', 'ToolingHealth')][string[]]$Component = @('LinkHealth', 'ToolingHealth'),
    [string]$SourceRoot,
    [string]$InstallRoot,
    [switch]$System,
    [switch]$Uninstall,
    [switch]$Restore,
    [switch]$RunNow,
    [switch]$DryRun,
    [Alias('h')][switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:PcaiTaskFolder = '\PC-AI\'
$script:PcaiAdminSids = @('S-1-5-32-544', 'S-1-5-18')

# Event sources verified on dtm-p1gen9 on 2026-11-21 against real USB4NET plug history:
#  - Kernel-PnP/Configuration 400 (configured) and 410 (started) record a USB4NET adapter arriving;
#  - Kernel-PnP/Device Management 1010 records "surprise removed ... missing on the bus" on unplug;
#  - NetworkProfile/Operational 10000/10001 record connect/disconnect, including the USB4 link's
#    "Unidentified network" (the existing \PC-AI tasks already use these);
#  - Power-Troubleshooter 1 records resume from sleep.
# Tcpip 4201/4202 never appeared for these adapters, so they are not used.
$script:PcaiLinkEventQuery = @'
<QueryList>
  <Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational"><Select Path="Microsoft-Windows-NetworkProfile/Operational">*[System[(EventID=10000 or EventID=10001)]]</Select></Query>
  <Query Id="1" Path="Microsoft-Windows-Kernel-PnP/Configuration"><Select Path="Microsoft-Windows-Kernel-PnP/Configuration">*[System[(EventID=400 or EventID=410)]]</Select></Query>
  <Query Id="2" Path="Microsoft-Windows-Kernel-PnP/Device Management"><Select Path="Microsoft-Windows-Kernel-PnP/Device Management">*[System[(EventID=1010)]]</Select></Query>
  <Query Id="3" Path="System"><Select Path="System">*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and (EventID=1)]]</Select></Query>
</QueryList>
'@
$script:PcaiToolingEventQuery = @'
<QueryList>
  <Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational"><Select Path="Microsoft-Windows-NetworkProfile/Operational">*[System[(EventID=10000)]]</Select></Query>
</QueryList>
'@

function ConvertTo-PcaiXmlText {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Text)
    return [Security.SecurityElement]::Escape($Text)
}

function Get-PcaiHealthFile {
    <# The files a component needs, relative to the repository root. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root, [string[]]$Components)
    $files = [Collections.Generic.List[object]]::new()
    if ('LinkHealth' -in $Components) {
        $files.Add([pscustomobject]@{ Component = 'LinkHealth'; Source = Join-Path $Root 'Tools/SystemScripts/Networking/Invoke-PcaiLinkHealth.ps1'; Name = 'Invoke-PcaiLinkHealth.ps1' })
        if (-not $IsWindows) {
            $files.Add([pscustomobject]@{ Component = 'LinkHealth'; Source = Join-Path $Root 'Tools/SystemScripts/Networking/pcai-link-health-monitor.sh'; Name = 'pcai-link-health-monitor.sh' })
        }
    }
    if ('ToolingHealth' -in $Components) {
        $files.Add([pscustomobject]@{ Component = 'ToolingHealth'; Source = Join-Path $Root 'Tools/SystemScripts/Machine/Invoke-PcaiToolingHealth.ps1'; Name = 'Invoke-PcaiToolingHealth.ps1' })
    }
    return $files.ToArray()
}

function Get-PcaiContentVersion {
    <# First 12 hex digits of SHA-256 over the names and bytes of the files, in name order. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object[]]$File)
    $sha = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    foreach ($f in @($File | Sort-Object Name)) {
        $sha.AppendData([Text.Encoding]::UTF8.GetBytes($f.Name + "`n"))
        $sha.AppendData([IO.File]::ReadAllBytes($f.Source))
    }
    return [Convert]::ToHexString($sha.GetHashAndReset()).Substring(0, 12).ToLowerInvariant()
}

function New-PcaiProtectedDirectory {
    <# Admin-owned directory: Administrators and SYSTEM full control, Users read and execute. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { $null = New-Item -ItemType Directory -Path $Path -Force }
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
    foreach ($sid in $script:PcaiAdminSids) {
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]$sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
    }
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]'S-1-5-32-545', 'ReadAndExecute', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function New-PcaiTaskXml {
    <#
    .SYNOPSIS
    Builds a Task Scheduler 1.4 definition. Pure function, so the triggers can be tested.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$Command,
        [Parameter(Mandatory)][string]$Arguments,
        [Parameter(Mandatory)][ValidateSet('User', 'System')][string]$Principal,
        [string]$UserId,
        [string]$LogonDelay,
        [string]$BootDelay,
        [string]$EventQuery,
        [string]$EventDelay = 'PT15S',
        [string]$RepeatInterval,
        [string]$ExecutionTimeLimit = 'PT5M'
    )
    $t = [Collections.Generic.List[string]]::new()
    if ($BootDelay) { $t.Add("<BootTrigger><Enabled>true</Enabled><Delay>$BootDelay</Delay></BootTrigger>") }
    if ($LogonDelay) { $t.Add("<LogonTrigger><Enabled>true</Enabled><UserId>$(ConvertTo-PcaiXmlText $UserId)</UserId><Delay>$LogonDelay</Delay></LogonTrigger>") }
    if ($EventQuery) { $t.Add("<EventTrigger><Enabled>true</Enabled><Delay>$EventDelay</Delay><Subscription>$(ConvertTo-PcaiXmlText $EventQuery)</Subscription></EventTrigger>") }
    if ($RepeatInterval) {
        $start = [DateTime]::Now.Date.ToString('yyyy-MM-ddTHH:mm:ss')
        $t.Add("<TimeTrigger><Enabled>true</Enabled><StartBoundary>$start</StartBoundary><Repetition><Interval>$RepeatInterval</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition></TimeTrigger>")
    }
    $principalXml = if ($Principal -eq 'System') {
        '<Principal id="Author"><UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel></Principal>'
    } else {
        "<Principal id=`"Author`"><UserId>$(ConvertTo-PcaiXmlText $UserId)</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>"
    }
    return @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Author>PC-AI</Author><Description>$(ConvertTo-PcaiXmlText $Description)</Description></RegistrationInfo>
  <Triggers>$($t -join '')</Triggers>
  <Principals>$principalXml</Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <ExecutionTimeLimit>$ExecutionTimeLimit</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author"><Exec><Command>$(ConvertTo-PcaiXmlText $Command)</Command><Arguments>$(ConvertTo-PcaiXmlText $Arguments)</Arguments></Exec></Actions>
</Task>
"@
}

function Get-PcaiTaskSpec {
    <# The task definitions this installer owns. Pure function. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$BinDir,
        [Parameter(Mandatory)][string]$Pwsh,
        [Parameter(Mandatory)][string]$UserId,
        [string[]]$Components,
        [switch]$SystemTask
    )
    $common = "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File"
    $specs = [Collections.Generic.List[object]]::new()
    if ($SystemTask) {
        if ('LinkHealth' -in $Components) {
            $specs.Add([pscustomobject]@{ Name = 'LinkHealthSystem'; Xml = New-PcaiTaskXml -Principal System -Command $Pwsh `
                        -Arguments "$common `"$(Join-Path $BinDir 'Invoke-PcaiLinkHealth.ps1')`" -Mode Apply -Trigger Event -NoBus" `
                        -Description 'PC-AI USB4 link health, boot and reconnect, SYSTEM. Applies the recorded route plan only when the admin-owned auto-apply policy enables it.' `
                        -BootDelay 'PT1M' -EventQuery $script:PcaiLinkEventQuery -RepeatInterval 'PT30M' })
        }
        return $specs.ToArray()
    }
    if ('LinkHealth' -in $Components) {
        $specs.Add([pscustomobject]@{ Name = 'LinkHealth'; Xml = New-PcaiTaskXml -Principal User -UserId $UserId -Command $Pwsh `
                    -Arguments "$common `"$(Join-Path $BinDir 'Invoke-PcaiLinkHealth.ps1')`" -Trigger Event" `
                    -Description 'PC-AI USB4 link health (read-only): logon, USB4 arrival/removal, network change, resume, every 30 minutes.' `
                    -LogonDelay 'PT45S' -EventQuery $script:PcaiLinkEventQuery -RepeatInterval 'PT30M' })
    }
    if ('ToolingHealth' -in $Components) {
        $specs.Add([pscustomobject]@{ Name = 'ToolingHealth'; Xml = New-PcaiTaskXml -Principal User -UserId $UserId -Command $Pwsh `
                    -Arguments "$common `"$(Join-Path $BinDir 'Invoke-PcaiToolingHealth.ps1')`" -Trigger Event" `
                    -Description 'PC-AI profile and tooling health: profile chain, tool shims, fast-forward of clean default-branch checkouts, stale worktree and PR report.' `
                    -LogonDelay 'PT2M' -EventQuery $script:PcaiToolingEventQuery -EventDelay 'PT1M' -RepeatInterval 'PT4H' -ExecutionTimeLimit 'PT15M' })
    }
    return $specs.ToArray()
}

function Get-PcaiSystemdUnit {
    <# The systemd --user units this installer owns. Pure function. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$BinDir, [Parameter(Mandatory)][string]$Pwsh, [string[]]$Components)
    $units = [ordered]@{}
    if ('LinkHealth' -in $Components) {
        $units['pcai-link-health.service'] = @"
[Unit]
Description=PC-AI USB4/Thunderbolt link health (read-only)
Documentation=https://github.com/David-Martel/PC-AI/blob/main/docs/usb4/link-health.md

[Service]
Type=oneshot
ExecStart=$Pwsh -NoLogo -NoProfile -NonInteractive -File $BinDir/Invoke-PcaiLinkHealth.ps1 -Trigger Systemd
TimeoutStartSec=300
# Exit 2 means the check ran and found something to report; it is not a unit failure.
SuccessExitStatus=2
"@
        $units['pcai-link-health.timer'] = @'
[Unit]
Description=Run PC-AI link health at user-manager start and every 30 minutes

[Timer]
OnStartupSec=60
OnUnitActiveSec=30min
Unit=pcai-link-health.service

[Install]
WantedBy=timers.target
'@
        $units['pcai-link-health-monitor.service'] = @"
[Unit]
Description=Start PC-AI link health when a Thunderbolt network interface changes

[Service]
Type=simple
ExecStart=/usr/bin/env bash $BinDir/pcai-link-health-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"@
    }
    if ('ToolingHealth' -in $Components) {
        $units['pcai-tooling-health.service'] = @"
[Unit]
Description=PC-AI profile and tooling health

[Service]
Type=oneshot
ExecStart=$Pwsh -NoLogo -NoProfile -NonInteractive -File $BinDir/Invoke-PcaiToolingHealth.ps1 -Trigger Systemd
TimeoutStartSec=900
SuccessExitStatus=2
"@
        $units['pcai-tooling-health.timer'] = @'
[Unit]
Description=Run PC-AI tooling health at user-manager start and every 4 hours

[Timer]
OnStartupSec=2min
OnUnitActiveSec=4h
Unit=pcai-tooling-health.service

[Install]
WantedBy=timers.target
'@
    }
    return $units
}

function Install-PcaiHealthWindows {
    [CmdletBinding()]
    param([string]$Root, [string]$Target, [string[]]$Components, [switch]$SystemTask, [switch]$Remove, [switch]$RestoreSaved, [switch]$StartNow, [switch]$IsDryRun)
    $userId = "$env:USERDOMAIN\$env:USERNAME"
    $pwsh = Join-Path $PSHOME 'pwsh.exe'
    $backupDir = Join-Path $Target 'backup'
    if ($Remove) {
        $names = if ($SystemTask) { @('LinkHealthSystem') } else { @($Components) }
        foreach ($n in $names) {
            $existing = Get-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $n -ErrorAction SilentlyContinue
            if (-not $existing) { Write-Output "absent: $($script:PcaiTaskFolder)$n"; continue }
            if ($IsDryRun) { Write-Output "would remove: $($script:PcaiTaskFolder)$n"; continue }
            Unregister-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $n -Confirm:$false
            Write-Output "removed: $($script:PcaiTaskFolder)$n"
            if ($RestoreSaved) {
                $saved = Get-ChildItem -LiteralPath $backupDir -Filter "$n.*.xml" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
                if ($saved) {
                    Register-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $n -Xml (Get-Content -LiteralPath $saved.FullName -Raw) | Out-Null
                    Write-Output "restored: $($script:PcaiTaskFolder)$n from $($saved.Name)"
                }
            }
        }
        return
    }
    $files = @(Get-PcaiHealthFile -Root $Root -Components $Components | Where-Object { Test-Path -LiteralPath $_.Source })
    $present = @($files | ForEach-Object Component | Sort-Object -Unique)
    foreach ($c in $Components) { if ($c -notin $present) { Write-Output "skipped component $c`: its script is not in $Root" } }
    if (-not $files) { throw "No health scripts found under $Root" }
    $version = Get-PcaiContentVersion -File $files
    $bin = Join-Path $Target $version
    $specs = Get-PcaiTaskSpec -BinDir $bin -Pwsh $pwsh -UserId $userId -Components $present -SystemTask:$SystemTask
    if ($IsDryRun) {
        Write-Output "would install $($files.Count) file(s) to $bin"
        foreach ($s in $specs) { Write-Output "would register $($script:PcaiTaskFolder)$($s.Name)" }
        return
    }
    if ($SystemTask) {
        # The SYSTEM task trusts everything under %ProgramData%\pcai-usb4\link-health. Protect that
        # directory itself, not only bin\, and refuse a pcai-usb4 or link-health that another
        # account created first (it could otherwise swap the scripts the task runs).
        $linkRoot = Split-Path -Parent $Target
        $usb4Root = Split-Path -Parent $linkRoot
        foreach ($dir in @($usb4Root, $linkRoot)) {
            if (-not (Test-Path -LiteralPath $dir)) { continue }
            $item = Get-Item -LiteralPath $dir -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "$dir is a reparse point; review and remove it" }
            $owner = (Get-Acl -LiteralPath $dir).GetOwner([Security.Principal.SecurityIdentifier]).Value
            if ($owner -notin $script:PcaiAdminSids) { throw "$dir is owned by $owner, not Administrators or SYSTEM; review it, then: icacls `"$dir`" /setowner *S-1-5-32-544" }
        }
        if (-not (Test-Path -LiteralPath $usb4Root)) { New-PcaiProtectedDirectory -Path $usb4Root }
        New-PcaiProtectedDirectory -Path $linkRoot
        New-PcaiProtectedDirectory -Path $Target
    }
    $null = New-Item -ItemType Directory -Force -Path $bin, $backupDir
    foreach ($f in $files) { Copy-Item -LiteralPath $f.Source -Destination (Join-Path $bin $f.Name) -Force }
    $commit = $null
    try { $commit = (& git -C $Root rev-parse HEAD 2>$null) } catch { $commit = $null }
    [ordered]@{ version = $version; installedAtUtc = [DateTime]::UtcNow.ToString('o'); sourceRoot = $Root; sourceCommit = $commit
        files = @($files | ForEach-Object { [ordered]@{ name = $_.Name; sha256 = (Get-FileHash -LiteralPath (Join-Path $bin $_.Name)).Hash.ToLowerInvariant() } })
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $bin 'manifest.json')
    foreach ($s in $specs) {
        $existing = Get-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $s.Name -ErrorAction SilentlyContinue
        if ($existing) {
            Export-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $s.Name |
                Set-Content -LiteralPath (Join-Path $backupDir ("{0}.{1}.xml" -f $s.Name, [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')))
        }
        Register-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $s.Name -Xml $s.Xml -Force | Out-Null
        Write-Output "registered $($script:PcaiTaskFolder)$($s.Name) -> $bin"
        if ($StartNow) { Start-ScheduledTask -TaskPath $script:PcaiTaskFolder -TaskName $s.Name; Write-Output "started $($s.Name)" }
    }
}

function Install-PcaiHealthLinux {
    [CmdletBinding()]
    param([string]$Root, [string]$Target, [string[]]$Components, [switch]$Remove, [switch]$RestoreSaved, [switch]$StartNow, [switch]$IsDryRun)
    $unitDir = Join-Path $HOME '.config/systemd/user'
    $backupDir = Join-Path $Target 'backup'
    $pwsh = Join-Path $PSHOME 'pwsh'
    $allUnits = @(Get-PcaiSystemdUnit -BinDir '/x' -Pwsh $pwsh -Components $Components).Keys
    if ($Remove) {
        foreach ($u in $allUnits) {
            if ($IsDryRun) { Write-Output "would disable and remove $u"; continue }
            & systemctl --user disable --now $u 2>$null
            Remove-Item -LiteralPath (Join-Path $unitDir $u) -Force -ErrorAction SilentlyContinue
            Write-Output "removed $u"
            if ($RestoreSaved) {
                $saved = Get-ChildItem -LiteralPath $backupDir -Filter "$u.*" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
                if ($saved) { Copy-Item -LiteralPath $saved.FullName -Destination (Join-Path $unitDir $u); Write-Output "restored $u from $($saved.Name)" }
            }
        }
        if (-not $IsDryRun) { & systemctl --user daemon-reload }
        return
    }
    $files = @(Get-PcaiHealthFile -Root $Root -Components $Components | Where-Object { Test-Path -LiteralPath $_.Source })
    if (-not $files) { throw "No health scripts found under $Root" }
    $present = @($files | ForEach-Object Component | Sort-Object -Unique)
    $version = Get-PcaiContentVersion -File $files
    $bin = Join-Path $Target $version
    $units = Get-PcaiSystemdUnit -BinDir $bin -Pwsh $pwsh -Components $present
    if ($IsDryRun) {
        Write-Output "would install $($files.Count) file(s) to $bin"
        foreach ($u in $units.Keys) { Write-Output "would write and enable $u" }
        return
    }
    $null = New-Item -ItemType Directory -Force -Path $bin, $backupDir, $unitDir
    foreach ($f in $files) {
        $dest = Join-Path $bin $f.Name
        # Repository checkouts on Windows may carry CRLF; bash and systemd need LF.
        $text = (Get-Content -LiteralPath $f.Source -Raw) -replace "`r`n", "`n"
        [IO.File]::WriteAllText($dest, $text, [Text.UTF8Encoding]::new($false))
        if ($dest -like '*.sh') { & chmod 0755 $dest }
    }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    foreach ($u in $units.Keys) {
        $path = Join-Path $unitDir $u
        if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination (Join-Path $backupDir "$u.$stamp") }
        [IO.File]::WriteAllText($path, ($units[$u] -replace "`r`n", "`n") + "`n", [Text.UTF8Encoding]::new($false))
        Write-Output "wrote $path"
    }
    & systemctl --user daemon-reload
    foreach ($u in @($units.Keys | Where-Object { $_ -like '*.timer' -or $_ -eq 'pcai-link-health-monitor.service' })) {
        & systemctl --user enable --now $u
        if ($u -like '*.service') { & systemctl --user restart $u }
        Write-Output "enabled $u"
    }
    if ($StartNow) {
        foreach ($u in @($units.Keys | Where-Object { $_ -like '*health.service' })) { & systemctl --user start --no-block $u; Write-Output "started $u" }
    }
    $linger = (& loginctl show-user $env:USER -p Linger 2>$null)
    if ($linger -ne 'Linger=yes') { Write-Output "NOTE: linger is off, so units start at login, not boot. Owner: sudo loginctl enable-linger $env:USER" }
}

if ($MyInvocation.InvocationName -ne '.') {
    if ($Help) { Get-Help $PSCommandPath -Detailed; return }
    if (-not $SourceRoot) { $SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path }
    if ($IsWindows) {
        if ($System) {
            $elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
            if (-not $elevated -and -not $DryRun) { throw '-System registers a SYSTEM task and needs an elevated PowerShell.' }
            if (-not $InstallRoot) { $InstallRoot = Join-Path $env:ProgramData 'pcai-usb4\link-health\bin' }
        } elseif (-not $InstallRoot) { $InstallRoot = Join-Path $env:LOCALAPPDATA 'PC-AI\health-bin' }
        Install-PcaiHealthWindows -Root $SourceRoot -Target $InstallRoot -Components $Component -SystemTask:$System -Remove:$Uninstall -RestoreSaved:$Restore -StartNow:$RunNow -IsDryRun:$DryRun
    } elseif ($IsLinux) {
        if ($System) { throw '-System is Windows-only; on Linux the auto-apply path is not implemented (route-plan emits Windows commands).' }
        if (-not $InstallRoot) { $InstallRoot = Join-Path $HOME '.local/share/pcai/health-bin' }
        Install-PcaiHealthLinux -Root $SourceRoot -Target $InstallRoot -Components $Component -Remove:$Uninstall -RestoreSaved:$Restore -StartNow:$RunNow -IsDryRun:$DryRun
    } else {
        throw 'Unsupported platform'
    }
}
