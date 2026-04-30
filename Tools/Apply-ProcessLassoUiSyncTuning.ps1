#Requires -Version 7.0
<#
.SYNOPSIS
Applies Process Lasso policy for boot, sync, shell, and input responsiveness.

.DESCRIPTION
Updates the Process Lasso INI configuration so ProBalance and SmartTrim do not
restrain boot-critical mount processes, shell/input processes, Lenovo/input
support processes, or Process Lasso helper processes. It also applies
conservative default CPU and I/O priorities: input/shell processes are kept
responsive, while sync/build/archive/Docker/WSL background work is de-elevated.
It also normalizes browser/video-call GPU boosts so display composition is not
competing with elevated foreground app GPU rules during dock/eGPU churn.
By default the script writes a timestamped backup before changing the live
config.

.PARAMETER ConfigPath
Path to the Process Lasso INI file.

.PARAMETER ReportPath
Optional JSON report path describing planned or applied changes.

.PARAMETER DryRun
Preview changes and write the optional report without modifying the INI file.
The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$ConfigPath = 'C:\ProgramData\ProcessLasso\config\prolasso.ini',

    [string]$ReportPath,

    [switch]$DryRun,

    [Alias('h', '?')]
    [switch]$Help,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Process Lasso config not found: $ConfigPath"
}

$mountStartupExclusions = @(
    'powershell.exe',
    'pwsh.exe',
    'rclone.exe',
    'wsl.exe',
    'wslhost.exe',
    'vmmemWSL',
    'Docker Desktop.exe',
    'GoogleDriveFS.exe',
    'OneDrive*',
    'OneDrive.exe',
    'OneDrive.Sync.Service.exe',
    'FileSyncHelper.exe'
)

$uiShellExclusions = @(
    'audiodg.exe',
    'bitsumsessionagent.exe',
    'ctfmon.exe',
    'dwm.exe',
    'explorer.exe',
    'fontdrvhost.exe',
    'Lenovo.Modern.ImController.exe',
    'LenovoAccessoriesAndDisplayControlCenterService.exe',
    'LenovoGoCentral1.exe',
    'LenovoVantageService.exe',
    'LenovoVantage-(GenericMessagingAddin).exe',
    'LenovoVantage-(LenovoServiceBridgeAddin).exe',
    'LenovoVantage-(VantageCoreAddin).exe',
    'LockApp.exe',
    'RuntimeBroker.exe',
    'SearchHost.exe',
    'ShellExperienceHost.exe',
    'sihost.exe',
    'StartMenuExperienceHost.exe',
    'SynRpcServer.exe',
    'TabTip.exe',
    'taskhostw.exe',
    'TextInputHost.exe',
    'UserOOBEBroker.exe',
    'winlogon.exe',
    'WinStore.App.exe'
)

$touchpadVendorExclusions = @(
    'Sensel*.exe',
    'SNSL*.exe',
    'SynRpcServer.exe',
    'SynTP*.exe',
    'Synaptics*.exe',
    'ELAN*.exe',
    'Elan*.exe',
    'ETD*.exe'
)

$processLassoHelperExclusions = @(
    'ProcessGovernor.exe',
    'ProcessLasso.exe',
    'ProcessLassoLauncher.exe',
    'ProcessLassoService.exe'
)

$interactiveSmartTrimExclusions = @(
    'bitwarden.exe',
    'brave.exe',
    'chatgpt.exe',
    'chrome.exe',
    'claude.exe',
    'code.exe',
    'codex.exe',
    'signal.exe',
    'windowsterminal.exe',
    'zoom.exe'
)

$interactivePriorityDefaults = [ordered]@{
    'audiodg.exe'                                      = 'above normal'
    'ctfmon.exe'                                      = 'above normal'
    'dwm.exe'                                         = 'above normal'
    'explorer.exe'                                    = 'above normal'
    'sihost.exe'                                      = 'above normal'
    'ShellExperienceHost.exe'                         = 'above normal'
    'StartMenuExperienceHost.exe'                     = 'above normal'
    'TabTip.exe'                                      = 'above normal'
    'TextInputHost.exe'                               = 'above normal'
    'SynRpcServer.exe'                                = 'above normal'
    'Sensel*.exe'                                     = 'above normal'
    'SNSL*.exe'                                       = 'above normal'
    'SynTP*.exe'                                      = 'above normal'
    'Synaptics*.exe'                                  = 'above normal'
    'ELAN*.exe'                                       = 'above normal'
    'ETD*.exe'                                        = 'above normal'
    'Lenovo.Modern.ImController.exe'                  = 'above normal'
    'LenovoAccessoriesAndDisplayControlCenterService.exe' = 'above normal'
    'LenovoVantageService.exe'                        = 'above normal'
}

$backgroundPriorityDefaults = [ordered]@{
    'OneDrive.exe'                    = 'below normal'
    'OneDrive.Sync.Service.exe'       = 'below normal'
    'FileSyncHelper.exe'              = 'below normal'
    'GoogleDriveFS.exe'               = 'below normal'
    'Dropbox.exe'                     = 'below normal'
    'iCloudDrive.exe'                 = 'below normal'
    'iCloudCKKS.exe'                  = 'below normal'
    'ProtonDrive.exe'                 = 'below normal'
    'rclone.exe'                      = 'below normal'
    'Docker Desktop.exe'              = 'below normal'
    'com.docker.backend.exe'          = 'below normal'
    'com.docker.build.exe'            = 'below normal'
    'docker-agent.exe'                = 'below normal'
    'docker-sandbox.exe'              = 'below normal'
    'wsl.exe'                         = 'below normal'
    'wslhost.exe'                     = 'below normal'
    'wslservice.exe'                  = 'below normal'
    'vmmemWSL'                        = 'below normal'
    'redis-server.exe'                = 'below normal'
    'redis-service.exe'               = 'below normal'
    '7z.exe'                          = 'below normal'
    'robocopy.exe'                    = 'below normal'
    'cargo.exe'                       = 'below normal'
    'rustc.exe'                       = 'below normal'
    'sccache.exe'                     = 'below normal'
    'sccache-dist.exe'                = 'below normal'
    'link.exe'                        = 'below normal'
    'cl.exe'                          = 'below normal'
    'node.exe'                        = 'below normal'
    'npm.exe'                         = 'below normal'
    'npx.exe'                         = 'below normal'
    'winget.exe'                      = 'below normal'
    'git-cluster-analyzer.exe'        = 'below normal'
    'NVIDIA Broadcast.exe'            = 'below normal'
    'NVIDIA Overlay.exe'              = 'below normal'
    'NVIDIA Share.exe'                = 'below normal'
    'NVIDIA App.exe'                  = 'below normal'
    'PresentMon_x64.exe'              = 'below normal'
    'PresentMonService.exe'           = 'below normal'
    'nvfvsdksvc_x64.exe'              = 'below normal'
    'disp.exe'                        = 'below normal'
    'DPMCrashHandler.exe'             = 'below normal'
    'DPMService.exe'                  = 'below normal'
    'Dell.TechHub*.exe'               = 'below normal'
    'Dell.Update.SubAgent.exe'        = 'below normal'
    'Dell.CoreServices.Client.exe'    = 'below normal'
    'Dell.UCA.Manager.exe'            = 'below normal'
    'ServiceShell.exe'                = 'below normal'
    'HPPrinterHealthMonitor.exe'      = 'below normal'
    'HPPrintScanDoctorService.exe'    = 'below normal'
}

$interactiveIoDefaults = [ordered]@{
    'audiodg.exe'                                      = '3'
    'ctfmon.exe'                                      = '3'
    'dwm.exe'                                         = '3'
    'explorer.exe'                                    = '3'
    'sihost.exe'                                      = '3'
    'ShellExperienceHost.exe'                         = '3'
    'StartMenuExperienceHost.exe'                     = '3'
    'TabTip.exe'                                      = '3'
    'TextInputHost.exe'                               = '3'
    'SynRpcServer.exe'                                = '3'
    'Sensel*.exe'                                     = '3'
    'SNSL*.exe'                                       = '3'
    'SynTP*.exe'                                      = '3'
    'Synaptics*.exe'                                  = '3'
    'ELAN*.exe'                                       = '3'
    'ETD*.exe'                                        = '3'
    'Lenovo.Modern.ImController.exe'                  = '3'
    'LenovoAccessoriesAndDisplayControlCenterService.exe' = '3'
    'LenovoVantageService.exe'                        = '3'
}

$backgroundIoDefaults = [ordered]@{}
foreach ($processName in $backgroundPriorityDefaults.Keys) {
    $backgroundIoDefaults[$processName] = '1'
}

$gpuPriorityDefaults = [ordered]@{
    'chrome.exe' = '2'
    'brave.exe'  = '2'
    'zoom.exe'   = '2'
}

$proBalanceExclusions = $uiShellExclusions + $touchpadVendorExclusions + $mountStartupExclusions + $processLassoHelperExclusions
$smartTrimExclusions = $proBalanceExclusions + $interactiveSmartTrimExclusions
$loggingFlags = [ordered]@{
    IncludeCommandLines            = 'true'
    LogProcessExecutions          = 'true'
    LogProcessTerminations        = 'true'
    LogEfficiencyMode             = 'true'
    LogCPUSets                    = 'true'
    LogDefaultPriorityAdjustments = 'true'
    LogDefaultAffinityAdjustments = 'true'
    LogProBalanceBegin            = 'true'
    LogProBalanceEnd              = 'true'
    LogPowerProfileChanges        = 'true'
    LogSmartTrim                  = 'true'
    LogCPULimiter                 = 'true'
}

function Read-ProcessLassoIni {
    param([Parameter(Mandatory)] [string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $encoding = [System.Text.UnicodeEncoding]::new($false, $true)
        $text = $encoding.GetString($bytes)
    } elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $encoding = [System.Text.UnicodeEncoding]::new($true, $true)
        $text = $encoding.GetString($bytes)
    } elseif ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $encoding = [System.Text.UTF8Encoding]::new($true)
        $text = $encoding.GetString($bytes)
    } else {
        $encoding = [System.Text.UnicodeEncoding]::new($false, $true)
        $text = $encoding.GetString($bytes)
    }

    $text = $text.TrimEnd([char]0xFEFF)
    $newline = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($text -split "`r?`n")) {
        [void]$lines.Add($line)
    }

    [pscustomobject]@{
        Lines    = $lines
        Encoding = $encoding
        Newline  = $newline
    }
}

function Write-ProcessLassoIni {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [object]$Lines,
        [Parameter(Mandatory)] [System.Text.Encoding]$Encoding,
        [Parameter(Mandatory)] [string]$Newline
    )

    $text = [string]::Join($Newline, $Lines)
    [System.IO.File]::WriteAllText($Path, $text, $Encoding)
}

function Get-IniSectionBounds {
    param(
        [Parameter(Mandatory)] [object]$Lines,
        [Parameter(Mandatory)] [string]$Section
    )

    $start = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i].Trim() -ieq "[$Section]") {
            $start = $i
            break
        }
    }

    if ($start -lt 0) {
        return [pscustomobject]@{ Start = -1; End = -1 }
    }

    $end = $Lines.Count
    for ($i = $start + 1; $i -lt $Lines.Count; $i++) {
        $trimmed = $Lines[$i].Trim()
        if ($trimmed.StartsWith('[') -and $trimmed.EndsWith(']')) {
            $end = $i
            break
        }
    }

    [pscustomobject]@{ Start = $start; End = $end }
}

function Get-IniValue {
    param(
        [Parameter(Mandatory)] [object]$Lines,
        [Parameter(Mandatory)] [string]$Section,
        [Parameter(Mandatory)] [string]$Key
    )

    $bounds = Get-IniSectionBounds -Lines $Lines -Section $Section
    if ($bounds.Start -lt 0) {
        return ''
    }

    for ($i = $bounds.Start + 1; $i -lt $bounds.End; $i++) {
        if ($Lines[$i] -match ('^\s*' + [regex]::Escape($Key) + '\s*=(.*)$')) {
            return $Matches[1].Trim()
        }
    }

    return ''
}

function Set-IniValue {
    param(
        [Parameter(Mandatory)] [object]$Lines,
        [Parameter(Mandatory)] [string]$Section,
        [Parameter(Mandatory)] [string]$Key,
        [Parameter(Mandatory)] [string]$Value
    )

    $bounds = Get-IniSectionBounds -Lines $Lines -Section $Section
    if ($bounds.Start -lt 0) {
        if ($Lines.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($Lines[$Lines.Count - 1])) {
            [void]$Lines.Add('')
        }
        [void]$Lines.Add("[$Section]")
        [void]$Lines.Add("$Key=$Value")
        return
    }

    for ($i = $bounds.Start + 1; $i -lt $bounds.End; $i++) {
        if ($Lines[$i] -match ('^\s*' + [regex]::Escape($Key) + '\s*=')) {
            $Lines[$i] = "$Key=$Value"
            return
        }
    }

    $Lines.Insert($bounds.End, "$Key=$Value")
}

function Merge-CsvValue {
    param(
        [AllowNull()] [string]$Existing,
        [string[]]$Additions
    )

    $seen = [ordered]@{}
    foreach ($item in (($Existing -split ',') + $Additions)) {
        $name = $item.Trim()
        $key = $name.ToLowerInvariant()
        if ($name.Length -gt 0 -and -not $seen.Contains($key)) {
            $seen[$key] = $name
        }
    }

    return ($seen.Values -join ',')
}

function Merge-PairCsvValue {
    param(
        [AllowNull()] [string]$Existing,
        [Parameter(Mandatory)] [hashtable]$Additions
    )

    $orderedKeys = [System.Collections.Generic.List[string]]::new()
    $valuesByKey = [ordered]@{}
    $displayNameByKey = [ordered]@{}

    $parts = @($Existing -split ',' | ForEach-Object { $_.Trim() })
    for ($i = 0; $i -lt $parts.Count; $i += 2) {
        if ($i + 1 -ge $parts.Count) {
            break
        }

        $name = $parts[$i]
        $value = $parts[$i + 1]
        if ([string]::IsNullOrWhiteSpace($name) -or [string]::IsNullOrWhiteSpace($value)) {
            continue
        }

        $key = $name.ToLowerInvariant()
        if (-not $valuesByKey.Contains($key)) {
            [void]$orderedKeys.Add($key)
            $displayNameByKey[$key] = $name
        }
        $valuesByKey[$key] = $value
    }

    foreach ($entry in $Additions.GetEnumerator()) {
        $name = [string]$entry.Key
        $key = $name.ToLowerInvariant()
        if (-not $valuesByKey.Contains($key)) {
            [void]$orderedKeys.Add($key)
            $displayNameByKey[$key] = $name
        }
        $valuesByKey[$key] = [string]$entry.Value
    }

    $merged = [System.Collections.Generic.List[string]]::new()
    foreach ($key in $orderedKeys) {
        [void]$merged.Add([string]$displayNameByKey[$key])
        [void]$merged.Add([string]$valuesByKey[$key])
    }

    return ($merged -join ',')
}

function Add-Change {
    param(
        [Parameter(Mandatory)] [object]$Changes,
        [Parameter(Mandatory)] [string]$Section,
        [Parameter(Mandatory)] [string]$Key,
        [AllowNull()] [string]$Before,
        [AllowNull()] [string]$After
    )

    [void]$Changes.Add([pscustomobject]@{
            Section = $Section
            Key     = $Key
            Changed = ($Before -ne $After)
            Before  = $Before
            After   = $After
        })
}

$ini = Read-ProcessLassoIni -Path $ConfigPath
$lines = $ini.Lines
$changes = [System.Collections.Generic.List[object]]::new()

$oocBefore = Get-IniValue -Lines $lines -Section 'OutOfControlProcessRestraint' -Key 'OocExclusions'
$oocAfter = Merge-CsvValue -Existing $oocBefore -Additions $proBalanceExclusions
Set-IniValue -Lines $lines -Section 'OutOfControlProcessRestraint' -Key 'OocExclusions' -Value $oocAfter
Add-Change -Changes $changes -Section 'OutOfControlProcessRestraint' -Key 'OocExclusions' -Before $oocBefore -After $oocAfter

$smartTrimBefore = Get-IniValue -Lines $lines -Section 'MemoryManagement' -Key 'SmartTrimExclusions'
$smartTrimAfter = Merge-CsvValue -Existing $smartTrimBefore -Additions $smartTrimExclusions
Set-IniValue -Lines $lines -Section 'MemoryManagement' -Key 'SmartTrimExclusions' -Value $smartTrimAfter
Add-Change -Changes $changes -Section 'MemoryManagement' -Key 'SmartTrimExclusions' -Before $smartTrimBefore -After $smartTrimAfter

foreach ($flag in $loggingFlags.GetEnumerator()) {
    $before = Get-IniValue -Lines $lines -Section 'Logging' -Key $flag.Key
    Set-IniValue -Lines $lines -Section 'Logging' -Key $flag.Key -Value $flag.Value
    Add-Change -Changes $changes -Section 'Logging' -Key $flag.Key -Before $before -After $flag.Value
}

$priorityDefaults = [ordered]@{}
foreach ($entry in $interactivePriorityDefaults.GetEnumerator()) {
    $priorityDefaults[$entry.Key] = $entry.Value
}
foreach ($entry in $backgroundPriorityDefaults.GetEnumerator()) {
    $priorityDefaults[$entry.Key] = $entry.Value
}
$defaultPrioritiesBefore = Get-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultPriorities'
$defaultPrioritiesAfter = Merge-PairCsvValue -Existing $defaultPrioritiesBefore -Additions $priorityDefaults
Set-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultPriorities' -Value $defaultPrioritiesAfter
Add-Change -Changes $changes -Section 'ProcessDefaults' -Key 'DefaultPriorities' -Before $defaultPrioritiesBefore -After $defaultPrioritiesAfter

$ioDefaults = [ordered]@{}
foreach ($entry in $interactiveIoDefaults.GetEnumerator()) {
    $ioDefaults[$entry.Key] = $entry.Value
}
foreach ($entry in $backgroundIoDefaults.GetEnumerator()) {
    $ioDefaults[$entry.Key] = $entry.Value
}
$defaultIoBefore = Get-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultIOPriorities'
$defaultIoAfter = Merge-PairCsvValue -Existing $defaultIoBefore -Additions $ioDefaults
Set-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultIOPriorities' -Value $defaultIoAfter
Add-Change -Changes $changes -Section 'ProcessDefaults' -Key 'DefaultIOPriorities' -Before $defaultIoBefore -After $defaultIoAfter

$defaultGpuBefore = Get-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultGPUPriorities'
$defaultGpuAfter = Merge-PairCsvValue -Existing $defaultGpuBefore -Additions $gpuPriorityDefaults
Set-IniValue -Lines $lines -Section 'ProcessDefaults' -Key 'DefaultGPUPriorities' -Value $defaultGpuAfter
Add-Change -Changes $changes -Section 'ProcessDefaults' -Key 'DefaultGPUPriorities' -Before $defaultGpuBefore -After $defaultGpuAfter

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupPath = Join-Path (Split-Path -Parent $ConfigPath) "prolasso.ini.bak-$stamp-boot-safety"
$wouldApply = $changes.Where({ $_.Changed }).Count -gt 0
$dryRunActive = [bool]($DryRun -or $WhatIfPreference)

if ($wouldApply -and -not $dryRunActive) {
    if ($PSCmdlet.ShouldProcess($ConfigPath, 'Apply Process Lasso boot-safety exclusions and logging flags')) {
        Copy-Item -LiteralPath $ConfigPath -Destination $backupPath -Force
        Write-ProcessLassoIni -Path $ConfigPath -Lines $lines -Encoding $ini.Encoding -Newline $ini.Newline
    }
} elseif ($wouldApply -and $dryRunActive) {
    $backupPath = $null
}

$result = [pscustomobject]@{
    ConfigPath          = $ConfigPath
    BackupPath          = $backupPath
    DryRun              = $dryRunActive
    Changed             = $wouldApply
    GeneratedAt         = (Get-Date).ToString('o')
    ExpectedExclusions  = $proBalanceExclusions
    InteractivePriorityDefaults = $interactivePriorityDefaults
    BackgroundPriorityDefaults  = $backgroundPriorityDefaults
    InteractiveIoDefaults       = $interactiveIoDefaults
    BackgroundIoDefaults        = $backgroundIoDefaults
    GpuPriorityDefaults         = $gpuPriorityDefaults
    LoggingFlags        = $loggingFlags
    VerificationChanges = @($changes)
}

if ($ReportPath) {
    $reportDirectory = Split-Path -Parent $ReportPath
    if ($reportDirectory) {
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
    }
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}

$changes |
    Select-Object Section, Key, Changed,
        @{ Name = 'After'; Expression = {
                if ($_.After -and $_.After.Length -gt 120) { $_.After.Substring(0, 117) + '...' } else { $_.After }
            } } |
    Format-Table -AutoSize

$result
