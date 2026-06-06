#Requires -Version 7.0
<#
.SYNOPSIS
    READ-ONLY diagnostics collector for input-stack freezes (Shift key / trackpad /
    fingerprint) on DTM-P1GEN7 (Lenovo ThinkPad P1 Gen 7, Windows 11). Reproduces the
    2026-05-30 investigation so it can be re-run any time freezes recur.

.DESCRIPTION
    Collects (and only reads) the data points that mattered in the investigation:
      - System identity / OS build / uptime
      - Input/HID/biometric device inventory + ANY non-OK device (all classes)
      - I2C / Serial-IO / SMBus / GPIO bus controllers
      - Accessibility flags (StickyKeys/FilterKeys/ToggleKeys) with decode
      - Event-log triage: error/critical summary, Kernel-Power 41, nvlddmkm,
        WHEA-Logger, BTHUSB, real GPU TDR (Display 4101/4109), Service Control
        Manager failures (7000/7024/7034/7043), plus 90-day KP41 / 6008 counts
      - Crash-dump config + presence of dumps
      - Active power scheme + USB selective suspend AC/DC index
      - Startup load (Win32_StartupCommand) - surfaces login-storm contention
      - Process Lasso running state + key ProBalance/affinity config
      - Optional: summarize an HWiNFO sensor CSV if present

    Writes a human-readable .txt transcript always, and a structured .json with -AsJson.
    Nothing is mutated. Some queries read better when elevated; the script detects and
    warns rather than failing.

.PARAMETER OutputPath
    Directory for report files. Default: C:\codedev\PC_AI\Logs\input-diagnostics

.PARAMETER DaysBack
    Event-log lookback window for detailed sections. Default: 7

.PARAMETER AsJson
    Also emit a machine-readable JSON report alongside the .txt.

.EXAMPLE
    pwsh -File .\Invoke-InputStackDiagnostics.ps1

.EXAMPLE
    pwsh -File .\Invoke-InputStackDiagnostics.ps1 -DaysBack 14 -AsJson

.NOTES
    Author: input-stack investigation (Claude Code) - 2026-05-30. Strictly read-only.
#>
[CmdletBinding()]
param(
    [string]$OutputPath = 'C:\codedev\PC_AI\Logs\input-diagnostics',
    [int]$DaysBack = 7,
    [switch]$AsJson
)

$ErrorActionPreference = 'SilentlyContinue'
$now    = Get-Date
$stamp  = $now.ToString('yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$txt    = Join-Path $OutputPath "input-diagnostics-$stamp.txt"
$report = [ordered]@{}
$sb     = [System.Text.StringBuilder]::new()

function Section($title) {
    [void]$sb.AppendLine(''); [void]$sb.AppendLine("===== $title =====")
}
function Line($text) { [void]$sb.AppendLine([string]$text) }
function Dump($obj)  { [void]$sb.AppendLine(($obj | Format-Table -AutoSize -Wrap | Out-String).TrimEnd()) }

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
            ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)

Line "Input-Stack Diagnostics  |  $($now.ToString('o'))  |  Elevated=$elevated  |  DaysBack=$DaysBack"

# 1. System identity ---------------------------------------------------
Section 'SYSTEM IDENTITY'
try {
    $cs  = Get-CimInstance Win32_ComputerSystem
    $bios= Get-CimInstance Win32_BIOS
    $os  = Get-CimInstance Win32_OperatingSystem
    $up  = $now - $os.LastBootUpTime
    $sys = [ordered]@{
        Manufacturer = $cs.Manufacturer; Model = $cs.Model; Family = $cs.SystemFamily
        RAM_GB = [math]::Round($cs.TotalPhysicalMemory/1GB,1)
        BIOS = $bios.SMBIOSBIOSVersion; OS = $os.Caption; Build = $os.BuildNumber
        LastBoot = $os.LastBootUpTime; UptimeHours = [math]::Round($up.TotalHours,1)
    }
    $report.System = $sys
    Dump ([pscustomobject]$sys)
} catch { Line "  (failed: $_)" }

# 2. Input / HID / biometric devices ----------------------------------
Section 'INPUT / HID / BIOMETRIC DEVICES'
$dev = Get-PnpDevice -PresentOnly -Class Keyboard,Mouse,HIDClass,Biometric |
       Select-Object Status, Class, FriendlyName, InstanceId | Sort-Object Class, FriendlyName
$report.InputDevices = $dev
Dump ($dev | Select-Object Status, Class, FriendlyName)

Section 'ANY NON-OK DEVICE (all classes)'
$bad = Get-PnpDevice -PresentOnly | Where-Object Status -ne 'OK' |
       Select-Object Status, Class, FriendlyName
$report.NonOkDevices = $bad
Dump $bad

Section 'BUS CONTROLLERS (I2C / Serial-IO / SMBus / GPIO)'
$bus = Get-PnpDevice -PresentOnly -Class System |
       Where-Object FriendlyName -match 'I2C|Serial|SMBus|GPIO' |
       Select-Object Status, FriendlyName
$report.BusControllers = $bus
Dump $bus

# 3. Accessibility flags ----------------------------------------------
Section 'ACCESSIBILITY FLAGS (Shift-freeze culprits)'
function Decode-Flags($val) {
    $v = [int]$val
    $on   = if ($v -band 0x01) { 'ON' } else { 'off' }
    $hk   = if ($v -band 0x04) { 'hotkey ARMED' } else { 'hotkey disabled' }
    "$on, $hk (Flags=$v)"
}
$acc = [ordered]@{}
foreach ($leaf in 'StickyKeys','Keyboard Response','ToggleKeys') {
    $f = (Get-ItemProperty "HKCU:\Control Panel\Accessibility\$leaf" -Name Flags).Flags
    $acc[$leaf] = $f
    Line ("  {0,-18} {1}" -f $leaf, (Decode-Flags $f))
}
$kr = Get-ItemProperty 'HKCU:\Control Panel\Accessibility\Keyboard Response'
Line ("  FilterKeys SlowKeys wait (DelayBeforeAcceptance)={0}ms  Bounce={1}ms" -f $kr.DelayBeforeAcceptance, $kr.BounceTime)
$report.Accessibility = $acc

# 4. Event-log triage --------------------------------------------------
Section "SYSTEM LOG ERRORS/CRITICAL (last $DaysBack d) grouped"
$errs = Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2; StartTime=$now.AddDays(-$DaysBack)} |
        Group-Object ProviderName, Id | Sort-Object Count -Descending |
        Select-Object Count, Name
$report.ErrorSummary = $errs
Dump $errs

function Detail($provider, $ids, $label) {
    Section "$label"
    $ht = @{LogName='System'; ProviderName=$provider; StartTime=$now.AddDays(-$DaysBack)}
    if ($ids) { $ht.Id = $ids }
    $ev = Get-WinEvent -FilterHashtable $ht |
          Select-Object TimeCreated, Id, @{n='Msg';e={ ($_.Message -split "`n")[0] }}
    Dump $ev
    return $ev
}
$report.KernelPower41 = Detail 'Microsoft-Windows-Kernel-Power' 41 'KERNEL-POWER 41 (unexpected shutdown / hard hang)'
$report.Nvidia        = Detail 'nvlddmkm' $null 'NVIDIA DRIVER (nvlddmkm)'
$report.Whea          = Detail 'Microsoft-Windows-WHEA-Logger' $null 'WHEA-LOGGER (hardware errors)'
$report.Bluetooth     = Detail 'BTHUSB' $null 'BLUETOOTH (BTHUSB)'
$report.GpuTdr        = Detail 'Display' @(4101,4109) 'REAL GPU TDR (Display 4101/4109) - blank = none'
$report.ScmFailures   = Detail 'Service Control Manager' @(7000,7024,7034,7043) 'SERVICE CONTROL MANAGER FAILURES'

Section 'ACUTE vs CHRONIC (90-day counts)'
$kp90 = (Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=41; StartTime=$now.AddDays(-90)} | Measure-Object).Count
$ev90 = (Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='EventLog'; Id=6008; StartTime=$now.AddDays(-90)} | Measure-Object).Count
$report.KP41_90d = $kp90; $report.Unexpected6008_90d = $ev90
Line "  Kernel-Power 41 (90d): $kp90    EventLog 6008 unexpected-shutdown (90d): $ev90"

# 5. Crash-dump config -------------------------------------------------
Section 'CRASH DUMP CONFIG'
$cc = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' |
      Select-Object CrashDumpEnabled, AutoReboot
$mini = (Get-ChildItem 'C:\Windows\Minidump' -ErrorAction SilentlyContinue | Measure-Object).Count
$memdmp = Test-Path 'C:\Windows\MEMORY.DMP'
$report.CrashControl = @{ CrashDumpEnabled = $cc.CrashDumpEnabled; AutoReboot = $cc.AutoReboot; MinidumpCount = $mini; MemoryDmp = $memdmp }
Line "  CrashDumpEnabled=$($cc.CrashDumpEnabled) (0x7=Automatic[recommended >32GB], 0x3=small, 0x2=kernel, 0x0=none)"
Line "  AutoReboot=$($cc.AutoReboot)  Minidumps=$mini  MEMORY.DMP=$memdmp"

# 6. Power config ------------------------------------------------------
Section 'POWER SCHEME + USB SELECTIVE SUSPEND'
$scheme = (powercfg /getactivescheme) -join ' '
$usbq = powercfg /q SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226
$acIdx = ($usbq | Select-String 'Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)').Matches.Groups[1].Value
$dcIdx = ($usbq | Select-String 'Current DC Power Setting Index:\s*(0x[0-9a-fA-F]+)').Matches.Groups[1].Value
$report.Power = @{ ActiveScheme = $scheme; UsbSuspendAC = $acIdx; UsbSuspendDC = $dcIdx }
Line "  $scheme"
Line "  USB selective suspend: AC=$acIdx DC=$dcIdx  (0x0=disabled, 0x1=enabled)"

# 7. Startup load ------------------------------------------------------
Section 'STARTUP LOAD (login-storm contention)'
$startup = Get-CimInstance Win32_StartupCommand | Select-Object Name, Location, Command
$report.StartupCount = ($startup | Measure-Object).Count
$report.Startup = $startup
Line "  Total autostart entries: $($report.StartupCount)"
Dump ($startup | Select-Object Name, Location)

# 8. Process Lasso -----------------------------------------------------
Section 'PROCESS LASSO'
$pl = Get-Process ProcessLasso -ErrorAction SilentlyContinue
$plRunning = [bool]$pl
Line "  ProcessLasso running: $plRunning"
$plIni = 'C:\ProgramData\ProcessLasso\config\prolasso.ini'
if (Test-Path $plIni) {
    $keys = 'TotalProcessorUsageBeforeRestraint','RestrainByAffinity','ForcedMode',
            'ProcessThrottles','CPULimitRules','CPUSets','DefaultAffinitiesEx',
            'EfficiencyMode','OocExclusions','StartWithPowerPlan'
    $cfg = Get-Content $plIni
    foreach ($k in $keys) {
        $m = $cfg | Select-String "^$k=" | Select-Object -First 1
        if ($m) { Line "  $($m.Line)" }
    }
    $report.ProcessLasso = @{ Running = $plRunning; ConfigFound = $true }
} else {
    Line "  (prolasso.ini not found at $plIni)"
    $report.ProcessLasso = @{ Running = $plRunning; ConfigFound = $false }
}

# 9. Optional HWiNFO sensor CSV ---------------------------------------
Section 'THERMAL (HWiNFO CSV, if present)'
$csv = Get-ChildItem 'C:\codedev\PC_AI\Logs\hwinfo\*.csv' -ErrorAction SilentlyContinue |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($csv) {
    Line "  Newest HWiNFO log: $($csv.FullName)"
    try {
        $data = Import-Csv $csv.FullName
        $tempCols = $data[0].PSObject.Properties.Name | Where-Object { $_ -match 'Temperature|Temp' -and $_ -match 'CPU|GPU|Core|Package' }
        foreach ($c in $tempCols) {
            $max = ($data.$c | ForEach-Object { [double]($_ -replace '[^\d.]','') } | Measure-Object -Maximum).Maximum
            Line ("    max {0} = {1}" -f $c, $max)
        }
    } catch { Line "  (could not parse CSV: $_)" }
} else {
    Line "  No HWiNFO CSV found. Run Start-LoadCapture.ps1 during a heavy ML session to capture temps/DPC."
}

# --- write outputs ----------------------------------------------------
Set-Content -Path $txt -Value $sb.ToString() -Encoding utf8
if ($AsJson) {
    $json = Join-Path $OutputPath "input-diagnostics-$stamp.json"
    $report | ConvertTo-Json -Depth 6 | Set-Content -Path $json -Encoding utf8
}

# --- findings summary -------------------------------------------------
Write-Host "==== FINDINGS SUMMARY ====" -ForegroundColor Cyan
Write-Host ("Non-OK devices       : {0}" -f (($report.NonOkDevices | Measure-Object).Count))
Write-Host ("Accessibility hotkeys : StickyKeys/FilterKeys/ToggleKeys Flags = {0}/{1}/{2}" -f $acc['StickyKeys'], $acc['Keyboard Response'], $acc['ToggleKeys'])
Write-Host ("USB selective suspend : AC=$acIdx DC=$dcIdx (0x0 desired)")
Write-Host ("Crash dump            : CrashDumpEnabled=$($cc.CrashDumpEnabled) (0x7 desired), Minidumps=$mini")
Write-Host ("Kernel-Power 41 (90d) : $kp90   (clustered = acute incident; spread = chronic)")
Write-Host ("Startup autostarts    : $($report.StartupCount)")
Write-Host ("Process Lasso running : $plRunning")
Write-Host ""
Write-Host "Full report: $txt" -ForegroundColor Green
if ($AsJson) { Write-Host "JSON report: $json" -ForegroundColor Green }
