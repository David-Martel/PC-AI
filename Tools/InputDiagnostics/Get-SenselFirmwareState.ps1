#Requires -Version 7.0
<#
.SYNOPSIS
    Reconciles Sensel haptic touchpad firmware state from Windows and Lenovo Vantage.

.DESCRIPTION
    Collects read-only evidence for the ThinkPad P1 Gen 7 Sensel haptic touchpad
    firmware package.  The tool compares Lenovo Vantage metadata, staged package
    files, firmware INF contents, Windows firmware PnP state, signed-driver state,
    and hidcfu firmware-update events.

    It does not install, remove, or stage firmware.  The output is intended to
    answer whether Vantage's "AlreadyInstalled" state agrees with Windows'
    firmware-device view before any firmware action is considered.

.PARAMETER OutDir
    Report output directory. Default: Reports\haptic-touchpad\<timestamp>.

.PARAMETER AsJson
    Emit the JSON report to stdout in addition to writing it to disk.

.EXAMPLE
    pwsh -File .\Get-SenselFirmwareState.ps1 -AsJson

.NOTES
    Target firmware resource: UEFI\RES_{e3074a9c-a8f2-4ec6-8b7a-4124b1b3c134}
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [switch]$AsJson
)

$ErrorActionPreference = 'Stop'

if (-not $OutDir) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $OutDir = Join-Path $repo ("Reports\haptic-touchpad\firmware-{0}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$firmwareGuid = 'e3074a9c-a8f2-4ec6-8b7a-4124b1b3c134'
$firmwarePnP = "UEFI\RES_{$firmwareGuid}"
$vantageSession = 'C:\ProgramData\Lenovo\Vantage\AddinData\LenovoSystemUpdateAddin\session'
$packageRoot = Join-Path $vantageSession 'Repository\n48gb01w'
$infPath = Join-Path $packageRoot 'SenselTrackpad.inf'
$xmlPath = Join-Path $packageRoot 'n48gb01w_2_.xml'

function ConvertFrom-FirmwareDword {
    param([string]$HexText)
    if (-not $HexText) { return $null }
    $trimmed = $HexText.Trim() -replace '^0x', ''
    if ($trimmed -notmatch '^[0-9a-fA-F]+$') { return $null }
    $value = [Convert]::ToUInt32($trimmed, 16)
    $major = ($value -shr 24) -band 0xff
    $minor = ($value -shr 16) -band 0xff
    $build = $value -band 0xffff
    [pscustomobject]@{
        Hex     = ('0x{0:x8}' -f $value)
        Decimal = $value
        Version = "$major.$minor.$build"
    }
}

function Read-TextFileSafe {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        return Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    }
    return $null
}

function Invoke-CaptureCommand {
    param(
        [Parameter(Mandatory)] [string]$FileName,
        [Parameter(Mandatory)] [string[]]$Arguments
    )
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
    [pscustomobject]@{
        FileName = $FileName
        Arguments = $Arguments
        ExitCode = $p.ExitCode
        StdOut = $stdout
        StdErr = $stderr
    }
}

$inf = [ordered]@{ Path = $infPath; Present = Test-Path -LiteralPath $infPath }
if ($inf.Present) {
    $rawInf = Read-TextFileSafe -Path $infPath
    $driverVer = [regex]::Match($rawInf, '(?m)^\s*DriverVer\s*=\s*(.+)$').Groups[1].Value.Trim()
    $firmwareVersion = [regex]::Match($rawInf, '(?m)^\s*HKR,,FirmwareVersion,[^,]*,(0x[0-9a-fA-F]+)').Groups[1].Value.Trim()
    $firmwareFile = [regex]::Match($rawInf, '(?m)^\s*HKR,,FirmwareFilename,,(.+)$').Groups[1].Value.Trim()
    $desc = [regex]::Match($rawInf, '(?m)^\s*FirmwareDesc\s*=\s*"(.+)"').Groups[1].Value.Trim()
    $inf.DriverVer = $driverVer
    $inf.FirmwareVersion = ConvertFrom-FirmwareDword -HexText $firmwareVersion
    $inf.FirmwareFilename = $firmwareFile
    $inf.Description = $desc
}

$package = [ordered]@{ XmlPath = $xmlPath; Present = Test-Path -LiteralPath $xmlPath }
if ($package.Present) {
    try {
        [xml]$xml = Get-Content -LiteralPath $xmlPath -Raw
        $package.Id = $xml.Package.id
        $package.Name = $xml.Package.name
        $package.Version = $xml.Package.version
        $package.ReleaseDate = $xml.Package.ReleaseDate
        $package.Title = $xml.Package.Title.Desc.'#text'
        $package.DetectPnPId = $xml.Package.DetectVersion._PnPID.'#cdata-section'
        $package.DetectInstallHardwareIds = @($xml.Package.DetectInstall._Firmware.HardwareIDs.'#cdata-section')
        $package.DetectInstallVersion = $xml.Package.DetectInstall._Firmware.Version
    } catch {
        $package.ParseError = $_.Exception.Message
    }
}

$vantage = [ordered]@{
    SessionRoot = $vantageSession
    PackageRoot = $packageRoot
    PackageFiles = @()
    UpdateHistoryMatches = @()
    ProblematicUpdatesMatches = @()
    AvailableUpdatesMatches = @()
}

if (Test-Path -LiteralPath $packageRoot) {
    $vantage.PackageFiles = @(Get-ChildItem -LiteralPath $packageRoot -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object FullName, Length, LastWriteTime)
}

$historyPath = Join-Path $vantageSession 'update_history.txt'
if (Test-Path -LiteralPath $historyPath) {
    $vantage.UpdateHistoryMatches = @(Select-String -LiteralPath $historyPath -Pattern 'n48gb|Sensel|Forcepad' -ErrorAction SilentlyContinue |
        ForEach-Object { [pscustomobject]@{ Path = $_.Path; LineNumber = $_.LineNumber; Line = $_.Line } })
}

$problemPath = Join-Path $vantageSession 'ProblematicUpdates.xml'
if (Test-Path -LiteralPath $problemPath) {
    $vantage.ProblematicUpdatesMatches = @(Select-String -LiteralPath $problemPath -Pattern 'n48gb|Sensel|Forcepad' -ErrorAction SilentlyContinue |
        ForEach-Object { [pscustomobject]@{ Path = $_.Path; LineNumber = $_.LineNumber; Line = $_.Line } })
}

foreach ($jsonName in 'available_updates.json', 'aggregated_device_updates.json', 'updates.json') {
    $jsonPath = Join-Path $vantageSession $jsonName
    if (-not (Test-Path -LiteralPath $jsonPath)) { continue }
    $matches = Select-String -LiteralPath $jsonPath -Pattern 'n48gb|Sensel Forcepad|UEFI\\RES_\{e3074a9c' -ErrorAction SilentlyContinue |
        Select-Object -First 20
    foreach ($match in $matches) {
        $vantage.AvailableUpdatesMatches += [pscustomobject]@{
            Path = $match.Path
            LineNumber = $match.LineNumber
            Line = if ($match.Line.Length -gt 800) { $match.Line.Substring(0, 800) + '...' } else { $match.Line }
        }
    }
}

$firmwareDevice = [ordered]@{
    ResourceId = $firmwarePnP
    PnpDevice = $null
    PnputilInstance = $null
    PnputilFirmwareClassMatches = @()
    SignedDrivers = @()
}

try {
    $firmwareDevice.PnpDevice = @(Get-PnpDevice -Class Firmware -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -like "$firmwarePnP*" -or $_.InstanceId -like "*$firmwareGuid*" } |
        Select-Object Class, FriendlyName, InstanceId, Status, Problem)
} catch {}

$pnputilInstance = Invoke-CaptureCommand -FileName 'pnputil.exe' -Arguments @('/enum-devices', '/instanceid', "$firmwarePnP\0", '/properties')
$firmwareDevice.PnputilInstance = $pnputilInstance

$pnputilAll = Invoke-CaptureCommand -FileName 'pnputil.exe' -Arguments @('/enum-devices', '/class', 'Firmware')
$firmwareDevice.PnputilFirmwareClassMatches = @(
    ($pnputilAll.StdOut -split "`r?`n") |
        Select-String -Pattern @([regex]::Escape($firmwareGuid), 'Sensel', 'Trackpad') |
        ForEach-Object { $_.Line }
)

try {
    $firmwareDevice.SignedDrivers = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
        Where-Object { $_.DeviceID -like "$firmwarePnP*" -or $_.DeviceID -like "*$firmwareGuid*" -or $_.DeviceName -match 'Sensel|Trackpad|Firmware' } |
        Select-Object DeviceName, DeviceID, DriverProviderName, DriverVersion, DriverDate, InfName, Manufacturer, IsSigned)
} catch {}

$hidcfuEvents = @()
try {
    $hidcfuEvents = @(Get-WinEvent -FilterHashtable @{
            LogName = 'Microsoft-Windows-hidcfu/Operational'
            StartTime = (Get-Date).AddDays(-180)
        } -ErrorAction SilentlyContinue |
        Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message)
} catch {}

$setupEvents = @()
try {
    $setupEvents = @(Get-WinEvent -FilterHashtable @{
            LogName = 'Setup'
            StartTime = (Get-Date).AddDays(-180)
        } -ErrorAction SilentlyContinue |
        Where-Object { $_.Message -match 'n48gb|Sensel|Forcepad|e3074a9c|Trackpad' } |
        Select-Object -First 80 TimeCreated, Id, ProviderName, LevelDisplayName, Message)
} catch {}

$report = [ordered]@{
    Timestamp = (Get-Date).ToString('o')
    Machine = $env:COMPUTERNAME
    OutDir = $OutDir
    Target = [ordered]@{
        FirmwareGuid = $firmwareGuid
        FirmwarePnPId = $firmwarePnP
        ExpectedPackageId = 'n48gb01w'
        ExpectedFirmwareVersionDword = '0x01040002'
    }
    FirmwareInf = $inf
    LenovoVantage = $vantage
    WindowsFirmwareDevice = $firmwareDevice
    HidCfuEvents = $hidcfuEvents
    SetupEvents = $setupEvents
    Interpretation = @(
        'Vantage package metadata and Windows firmware PnP state should agree before any firmware action.',
        'If Vantage says AlreadyInstalled but firmware PnP cannot expose the UEFI resource, collect this report for Lenovo/Sensel escalation.',
        'Empty hidcfu logs do not prove firmware absence; they only mean no recorded HID CFU events in the queried window.'
    )
}

$jsonPath = Join-Path $OutDir 'sensel-firmware-state.json'
$report | ConvertTo-Json -Depth 10 | Set-Content -Path $jsonPath -Encoding UTF8

$summaryPath = Join-Path $OutDir 'README.md'
@"
# Sensel Firmware State

- Generated: $($report.Timestamp)
- Target: `$firmwarePnP`
- JSON: `sensel-firmware-state.json`

Key checks:

- Firmware INF present: $($inf.Present)
- Firmware INF version: $($inf.FirmwareVersion.Version)
- Vantage package present: $($package.Present)
- Vantage package version: $($package.Version)
- PnP firmware device count: $(@($firmwareDevice.PnpDevice).Count)
- hidcfu events: $(@($hidcfuEvents).Count)

Use this report before installing or reinstalling `n48gb01w`.
"@ | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "Sensel firmware report written: $jsonPath" -ForegroundColor Green
if ($AsJson) {
    $report | ConvertTo-Json -Depth 10
}
