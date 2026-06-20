#Requires -Version 7.0
<#
.SYNOPSIS
    Exports a read-only haptic touchpad repro evidence bundle.

.DESCRIPTION
    Collects machine-local evidence for Sensel haptic touchpad debugging:
    firmware reconciliation, PnP device topology, signed-driver versions,
    Precision Touchpad settings, relevant services/processes, recent input and
    driver-framework events, NVIDIA dual-GPU state when the checker is present,
    and the existing symptom ledger.

    This command does not alter device, registry, driver, service, firmware, or
    ETW state.  Pair its output with Start-HapticTouchpadTrace.ps1 ETL captures.

.PARAMETER OutDir
    Bundle output directory. Default: Reports\haptic-touchpad\bundle-<timestamp>.

.PARAMETER SinceHours
    Event-log lookback window. Default: 48 hours.

.EXAMPLE
    pwsh -File .\Export-HapticTouchpadReproBundle.ps1 -SinceHours 72
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [ValidateRange(1, 720)] [int]$SinceHours = 48
)

$ErrorActionPreference = 'Stop'

if (-not $OutDir) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $OutDir = Join-Path $repo ("Reports\haptic-touchpad\bundle-{0}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

function Save-Json {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [object]$Value,
        [int]$Depth = 8
    )
    $path = Join-Path $OutDir $Name
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -Path $path -Encoding UTF8
    return $path
}

function Invoke-CaptureCommand {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$FileName,
        [Parameter(Mandatory)] [string[]]$Arguments
    )
    $path = Join-Path $OutDir $Name
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FileName
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add($arg) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    @(
        ">>> $FileName $($Arguments -join ' ')",
        "exit=$($p.ExitCode)",
        '--- stdout ---',
        $stdout,
        '--- stderr ---',
        $stderr
    ) | Set-Content -Path $path -Encoding UTF8
    return $path
}

$since = (Get-Date).AddHours(-$SinceHours)

$deviceQuery = {
    Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object {
            $_.InstanceId -match 'SNSL002D|LEN032A|VEN_8086&DEV_7E78|ELAS|ETDHSA|VID_2C2F' -or
            $_.FriendlyName -match 'Touch|TrackPoint|ELAN|I2C|Sensel|HID Sensor|Input Configuration'
        } |
        Select-Object Class, FriendlyName, InstanceId, Status, Problem
}

$devices = @(& $deviceQuery)
Save-Json -Name 'input-devices.json' -Value $devices | Out-Null

$drivers = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
    Where-Object {
        $_.DeviceID -match 'SNSL002D|LEN032A|VEN_8086&DEV_7E78|ELAS|ETDHSA|VID_2C2F' -or
        $_.DeviceName -match 'Touch|TrackPoint|ELAN|I2C|Sensel|HID'
    } |
    Select-Object DeviceName, DeviceID, DriverProviderName, DriverVersion, DriverDate, InfName, Manufacturer, IsSigned)
Save-Json -Name 'signed-drivers.json' -Value $drivers | Out-Null

$precision = @()
foreach ($key in 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\PrecisionTouchPad', 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\PrecisionTouchPad') {
    if (Test-Path -LiteralPath $key) {
        $precision += [pscustomobject]@{
            Path = $key
            Values = Get-ItemProperty -Path $key | Select-Object * -ExcludeProperty PSPath, PSParentPath, PSChildName, PSDrive, PSProvider
        }
    }
}
Save-Json -Name 'precision-touchpad-settings.json' -Value $precision | Out-Null

$power = @()
try {
    $power = @(Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceName -match 'SNSL002D|7E78|LEN032A|ELAS' } |
        Select-Object InstanceName, Enable)
} catch {}
Save-Json -Name 'device-powerdown-state.json' -Value $power | Out-Null

$services = @(Get-Service | Where-Object {
        $_.Name -match 'EPD|ELAN|Sensel|Syn|Hid|Wudf|Lenovo|Vantage|ImController' -or
        $_.DisplayName -match 'EPD|ELAN|Sensel|Syn|Hid|Wudf|Lenovo|Vantage|ImController|TrackPoint'
    } | Select-Object Name, DisplayName, Status, StartType)
Save-Json -Name 'services.json' -Value $services | Out-Null

$processes = @(Get-Process | Where-Object {
        $_.ProcessName -match 'elan|epd|sensel|syn|hid|wudf|lenovo|vantage|imcontroller|touch|track|dwm|explorer|ctfmon'
    } | Select-Object ProcessName, Id, Path, CPU, WorkingSet64)
Save-Json -Name 'processes.json' -Value $processes | Out-Null

$logs = @(
    'System',
    'Application',
    'Microsoft-Windows-DriverFrameworks-UserMode/Operational',
    'Microsoft-Windows-hidcfu/Operational'
)
foreach ($log in $logs) {
    try {
        $events = @(Get-WinEvent -FilterHashtable @{ LogName = $log; StartTime = $since } -ErrorAction SilentlyContinue |
            Where-Object { $_.Message -match 'SNSL|ELAS|LEN032A|HID|I2C|touch|TrackPoint|EPD|haptic|Sensel|NVIDIA|nvlddmkm|WUDF' -or $_.ProviderName -match 'HID|I2C|WDF|Kernel-PnP|nvlddmkm' } |
            Select-Object -First 200 TimeCreated, Id, ProviderName, LevelDisplayName, Message)
        Save-Json -Name ("events-{0}.json" -f ($log -replace '[\\\/]', '_')) -Value $events -Depth 6 | Out-Null
    } catch {
        Save-Json -Name ("events-{0}.json" -f ($log -replace '[\\\/]', '_')) -Value @([pscustomobject]@{ Error = $_.Exception.Message }) | Out-Null
    }
}

Invoke-CaptureCommand -Name 'pnputil-sensel.txt' -FileName 'pnputil.exe' -Arguments @('/enum-devices', '/instanceid', 'ACPI\SNSL002D\4&39979B3E&0', '/properties') | Out-Null
Invoke-CaptureCommand -Name 'pnputil-trackpoint.txt' -FileName 'pnputil.exe' -Arguments @('/enum-devices', '/instanceid', 'ACPI\LEN032A\4&76D3D92&0', '/properties') | Out-Null
Invoke-CaptureCommand -Name 'pnputil-i2c-7e78.txt' -FileName 'pnputil.exe' -Arguments @('/enum-devices', '/instanceid', 'PCI\VEN_8086&DEV_7E78&SUBSYS_223417AA&REV_20\3&11583659&1&A8', '/properties') | Out-Null
Invoke-CaptureCommand -Name 'logman-input-providers.txt' -FileName 'logman.exe' -Arguments @('query', 'providers') | Out-Null

$firmwareDir = Join-Path $OutDir 'firmware'
try {
    & (Join-Path $PSScriptRoot 'Get-SenselFirmwareState.ps1') -OutDir $firmwareDir | Out-Null
} catch {
    Save-Json -Name 'firmware-error.json' -Value @([pscustomobject]@{ Error = $_.Exception.Message }) | Out-Null
}

$nvidiaScript = Join-Path $PSScriptRoot 'Test-NvidiaDualGpuDriverHealth.ps1'
if (Test-Path -LiteralPath $nvidiaScript) {
    try {
        $nvidiaJson = & $nvidiaScript -AsJson
        $nvidiaJson | Set-Content -Path (Join-Path $OutDir 'nvidia-dual-gpu.json') -Encoding UTF8
    } catch {
        Save-Json -Name 'nvidia-dual-gpu-error.json' -Value @([pscustomobject]@{ Error = $_.Exception.Message }) | Out-Null
    }
}

$ledger = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')) 'Reports\input-glitch-watch\symptom-log.jsonl'
if (Test-Path -LiteralPath $ledger) {
    Copy-Item -LiteralPath $ledger -Destination (Join-Path $OutDir 'symptom-log.jsonl') -Force
}

$manifest = [ordered]@{
    Timestamp = (Get-Date).ToString('o')
    Machine = $env:COMPUTERNAME
    OutDir = $OutDir
    SinceHours = $SinceHours
    Files = @(Get-ChildItem -LiteralPath $OutDir -Recurse -File | Select-Object FullName, Length, LastWriteTime)
    RecommendedNext = @(
        'Run Start-HapticTouchpadTrace.ps1 during a live repro and place the ETL in this bundle.',
        'Run Watch-HapticTouchpadInput.ps1 concurrently if the symptom is press/button stickiness.',
        'Compare haptic-enabled and haptic-disabled captures only after baseline ETW evidence exists.'
    )
}
Save-Json -Name 'manifest.json' -Value $manifest -Depth 6 | Out-Null

$readme = @(
    '# Haptic Touchpad Repro Bundle',
    '',
    "- Generated: $($manifest.Timestamp)",
    "- Lookback hours: $SinceHours",
    "- Bundle path: $OutDir",
    '',
    'Primary files:',
    '',
    '- firmware\sensel-firmware-state.json',
    '- input-devices.json',
    '- signed-drivers.json',
    '- precision-touchpad-settings.json',
    '- device-powerdown-state.json',
    '- events-System.json',
    '- events-Microsoft-Windows-DriverFrameworks-UserMode_Operational.json',
    '- nvidia-dual-gpu.json when available',
    '',
    'Next capture:',
    '',
    '```powershell',
    'pwsh -File .\Tools\InputDiagnostics\Start-HapticTouchpadTrace.ps1 -DurationSeconds 90 -IncludeHidi2cWpp -Note "TrackPoint palm haptic repro"',
    'pwsh -File .\Tools\InputDiagnostics\Watch-HapticTouchpadInput.ps1 -Seconds 90 -Note "same repro window"',
    '```'
)
$readme | Set-Content -Path (Join-Path $OutDir 'README.md') -Encoding UTF8

Write-Host "Haptic touchpad repro bundle written: $OutDir" -ForegroundColor Green
