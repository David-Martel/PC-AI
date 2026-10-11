#Requires -Version 7.0
<#
.SYNOPSIS
Collects a sanitized, read-only keyboard diagnostic snapshot.
.DESCRIPTION
Returns schema 1.0 with independently reported probes. Status is ok, empty,
unavailable, error, or timeout. Data is always an array; errors contain only a
type, never exception messages. No serials, device instance IDs, command lines,
typed text, event messages or remap values are included. No settings are changed.
CIM calls use a five-second operation timeout. Events use a bounded thread job.
.PARAMETER OutputPath
Optional new local JSON file. Its parent must already exist. Existing files and
Windows reserved leaves are rejected. Default output is a structured object.
.PARAMETER SinceMinutes
Event lookback, from 1 to 1440 minutes. At most 100 matching events are returned.
.PARAMETER DryRun
Returns the planned probe names without executing probes or writing files.
.PARAMETER Help
Displays usage without executing probes or writing files. Aliases: -h, --help.
.EXAMPLE
.\Get-KeyboardDiagnosticSnapshot.ps1 -OutputPath C:\Temp\keyboard-snapshot.json
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$OutputPath,
    [ValidateRange(1, 1440)][int]$SinceMinutes = 60,
    [switch]$DryRun,
    [Alias('h')][switch]$Help,
    [Parameter(ValueFromRemainingArguments)][string[]]$RemainingArguments
)
if ($RemainingArguments -contains '--help') { $Help = $true }
elseif ($RemainingArguments.Count) { throw 'Unsupported argument. Use -h for help.' }

function Invoke-KeyboardSnapshotProbe {
    param([string]$Name, [scriptblock]$Action)
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $data = @(& $Action)
        $status = if ($data.Count) { 'ok' } else { 'empty' }
        [pscustomobject]@{ Name = $Name; Status = $status; Data = $data; ErrorType = $null; DurationMs = $timer.ElapsedMilliseconds }
    } catch {
        $status = if ($_.Exception -is [System.TimeoutException]) { 'timeout' }
        elseif ($_.Exception -is [System.NotSupportedException] -or $_.Exception -is [System.Management.Automation.CommandNotFoundException]) { 'unavailable' }
        else { 'error' }
        [pscustomobject]@{ Name = $Name; Status = $status; Data = @(); ErrorType = $_.Exception.GetType().FullName; DurationMs = $timer.ElapsedMilliseconds }
    }
}

function Get-KeyboardSnapshotMachine {
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem -Property Manufacturer, Model -OperationTimeoutSec 5 -ErrorAction Stop
    $bios = Get-CimInstance -ClassName Win32_BIOS -Property Manufacturer, SMBIOSBIOSVersion, ReleaseDate -OperationTimeoutSec 5 -ErrorAction Stop
    $os = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop
    [pscustomobject]@{
        Manufacturer = $computer.Manufacturer; Model = $computer.Model
        BiosManufacturer = $bios.Manufacturer; BiosVersion = $bios.SMBIOSBIOSVersion
        BiosReleaseDate = $bios.ReleaseDate; OsBuild = $os.CurrentBuildNumber
        OsDisplayVersion = $os.DisplayVersion; OsUbr = $os.UBR
    }
}

function Get-KeyboardSnapshotProcess {
    foreach ($process in @(Get-Process -ErrorAction Stop | Where-Object {
        $_.ProcessName -match '^(PowerToys(?:\..*)?|AutoHotkey(?:32|64|U32|U64)?|kanata|kmonad|SharpKeys|ctfmon|TextInputHost|dwm|explorer|ProcessLasso|ProcessGovernor|logioptionsplus(?:_agent|_updater|_appbroker)?|Lenovo\.Modern\.ImController|LenovoGoCentral1|LenovoAccessoriesAndDisplayControlCenterService)$'
    })) {
        $priority = $null; $priorityStatus = 'ok'
        try { $priority = $process.PriorityClass.ToString() } catch { $priorityStatus = 'unavailable' }
        [pscustomobject]@{ Name = $process.ProcessName; Priority = $priority; PriorityStatus = $priorityStatus }
    }
}

function Get-KeyboardSnapshotDeviceToken {
    param([string]$InstanceId)
    if (-not $InstanceId) { return $null }
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $algorithm.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($InstanceId.ToUpperInvariant()))
        'kbd-' + [System.BitConverter]::ToString($digest).Replace('-', '').Substring(0, 16).ToLowerInvariant()
    } finally { $algorithm.Dispose() }
}

function Get-KeyboardSnapshotDevice {
    $index = 0
    foreach ($device in @(Get-CimInstance -ClassName Win32_PnPEntity -Filter "PNPClass = 'Keyboard'" -Property PNPClass, PNPDeviceID, Status, Service, ConfigManagerErrorCode -OperationTimeoutSec 5 -ErrorAction Stop)) {
        $index++
        [pscustomobject]@{ Index = $index; DeviceToken = Get-KeyboardSnapshotDeviceToken $device.PNPDeviceID; Class = $device.PNPClass; Status = $device.Status; Service = $device.Service; ProblemCode = $device.ConfigManagerErrorCode }
    }
}

function Get-KeyboardSnapshotDriver {
    $index = 0
    foreach ($device in @(Get-CimInstance -ClassName Win32_PnPEntity -Filter "PNPClass = 'Keyboard'" -Property PNPDeviceID -OperationTimeoutSec 5 -ErrorAction Stop)) {
        $index++
        try {
            $instance = Get-ItemProperty -LiteralPath ('HKLM:\SYSTEM\CurrentControlSet\Enum\' + $device.PNPDeviceID) -ErrorAction Stop
            if ($instance.Driver -notmatch '^\{[0-9A-Fa-f-]{36}\}\\[0-9]{4}$') { throw [System.NotSupportedException]::new('Active driver reference unavailable') }
            $driver = Get-ItemProperty -LiteralPath ('HKLM:\SYSTEM\CurrentControlSet\Control\Class\' + $instance.Driver) -ErrorAction Stop
            [pscustomobject]@{ Index = $index; DeviceToken = Get-KeyboardSnapshotDeviceToken $device.PNPDeviceID; Status = 'ok'; Source = 'ActiveDeviceRegistry'; Provider = $driver.ProviderName; DriverVersion = $driver.DriverVersion; DriverDate = $driver.DriverDate; InfName = $driver.InfPath; SignatureStatus = 'not-queried'; ErrorType = $null }
        } catch {
            [pscustomobject]@{ Index = $index; DeviceToken = Get-KeyboardSnapshotDeviceToken $device.PNPDeviceID; Status = 'unavailable'; Source = 'ActiveDeviceRegistry'; Provider = $null; DriverVersion = $null; DriverDate = $null; InfName = $null; SignatureStatus = 'not-queried'; ErrorType = $_.Exception.GetType().FullName }
        }
    }
}

function Get-KeyboardSnapshotFilter {
    $classPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4D36E96B-E325-11CE-BFC1-08002BE10318}'
    $class = Get-ItemProperty -LiteralPath $classPath -ErrorAction Stop
    [pscustomobject]@{ Scope = 'Class'; Index = $null; Status = 'ok'; Upper = @($class.UpperFilters | Where-Object { $_ }); Lower = @($class.LowerFilters | Where-Object { $_ }) }
    $index = 0
    foreach ($device in @(Get-CimInstance -ClassName Win32_PnPEntity -Filter "PNPClass = 'Keyboard'" -Property PNPDeviceID -OperationTimeoutSec 5 -ErrorAction Stop)) {
        $index++
        try {
            # Instance IDs may contain serials. They are used locally but never emitted.
            $entry = Get-ItemProperty -LiteralPath ('HKLM:\SYSTEM\CurrentControlSet\Enum\' + $device.PNPDeviceID) -ErrorAction Stop
            [pscustomobject]@{ Scope = 'Instance'; Index = $index; DeviceToken = Get-KeyboardSnapshotDeviceToken $device.PNPDeviceID; Status = 'ok'; Upper = @($entry.UpperFilters | Where-Object { $_ }); Lower = @($entry.LowerFilters | Where-Object { $_ }) }
        } catch {
            [pscustomobject]@{ Scope = 'Instance'; Index = $index; DeviceToken = Get-KeyboardSnapshotDeviceToken $device.PNPDeviceID; Status = 'unavailable'; Upper = $null; Lower = $null }
        }
    }
}

function Get-KeyboardSnapshotAccessibility {
    if (-not $IsWindows) { throw [System.NotSupportedException]::new('Windows required') }
    if (-not ('PcaiKeyboardSnapshot.Native' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
namespace PcaiKeyboardSnapshot {
    public static class Native {
        [StructLayout(LayoutKind.Sequential)]
        public struct Flags { public uint Size; public uint Value; }
        [StructLayout(LayoutKind.Sequential)]
        public struct Filter { public uint Size; public uint Value; public uint Wait; public uint Delay; public uint Repeat; public uint Bounce; }
        [DllImport("user32.dll", EntryPoint="SystemParametersInfoW", SetLastError=true)]
        private static extern bool GetFlags(uint action, uint size, ref Flags value, uint flags);
        [DllImport("user32.dll", EntryPoint="SystemParametersInfoW", SetLastError=true)]
        private static extern bool GetFilter(uint action, uint size, ref Filter value, uint flags);
        public static uint ReadSticky() { var value = new Flags { Size = 8 }; if (!GetFlags(0x003A, value.Size, ref value, 0)) throw new Win32Exception(Marshal.GetLastWin32Error()); return value.Value; }
        public static uint ReadToggle() { var value = new Flags { Size = 8 }; if (!GetFlags(0x0034, value.Size, ref value, 0)) throw new Win32Exception(Marshal.GetLastWin32Error()); return value.Value; }
        public static Filter ReadFilter() { var value = new Filter { Size = 24 }; if (!GetFilter(0x0032, value.Size, ref value, 0)) throw new Win32Exception(Marshal.GetLastWin32Error()); return value; }
    }
}
'@ -ErrorAction Stop
    }
    $filter = [PcaiKeyboardSnapshot.Native]::ReadFilter()
    $sticky = [PcaiKeyboardSnapshot.Native]::ReadSticky()
    $toggle = [PcaiKeyboardSnapshot.Native]::ReadToggle()
    [pscustomobject]@{
        FilterKeys = [pscustomobject]@{ Flags = $filter.Value; Enabled = [bool]($filter.Value -band 1); WaitMs = $filter.Wait; DelayMs = $filter.Delay; RepeatMs = $filter.Repeat; BounceMs = $filter.Bounce }
        StickyKeys = [pscustomobject]@{ Flags = $sticky; Enabled = [bool]($sticky -band 1) }
        ToggleKeys = [pscustomobject]@{ Flags = $toggle; Enabled = [bool]($toggle -band 1) }
    }
}

function Get-KeyboardSnapshotPowerToys {
    $root = Join-Path $env:LOCALAPPDATA 'Microsoft/PowerToys'
    $settingsPath = Join-Path $root 'settings.json'
    if (-not (Test-Path -LiteralPath $settingsPath)) { throw [System.NotSupportedException]::new('Settings unavailable') }
    $settings = Get-Content -LiteralPath $settingsPath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    $enabled = $null
    if ($settings.enabled -is [System.Collections.IDictionary]) {
        foreach ($key in @('Keyboard Manager', 'KeyboardManager')) {
            if ($settings.enabled.Contains($key)) { $enabled = [bool]$settings.enabled[$key] }
        }
    }
    $manager = Join-Path $root 'Keyboard Manager'
    $profile = 'default'; $selectionStatus = 'default-assumed'
    $managerSettingsPath = Join-Path $manager 'settings.json'
    if (Test-Path -LiteralPath $managerSettingsPath) {
        $managerSettings = Get-Content -LiteralPath $managerSettingsPath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable -ErrorAction Stop
        $active = $managerSettings.properties.activeConfiguration.value
        if ($active) {
            if ($active -notmatch '^[a-zA-Z0-9_-]+$') { throw [System.NotSupportedException]::new('Profile name unavailable') }
            $profile = $active; $selectionStatus = 'configured'
        }
    }
    $profilePath = Join-Path $manager ($profile + '.json')
    if (-not (Test-Path -LiteralPath $profilePath)) {
        [pscustomobject]@{ Enabled = $enabled; ProfileSelection = $selectionStatus; RemapStatus = 'unavailable'; KeyRemapCount = $null; ShortcutRemapCount = $null; TextRemapCount = $null }
        return
    }
    $map = Get-Content -LiteralPath $profilePath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    if ($map -isnot [System.Collections.IDictionary] -or -not (@('remapKeys', 'remapShortcuts', 'remapKeysToText', 'remapShortcutsToText') | Where-Object { $map.Contains($_) })) {
        throw [System.NotSupportedException]::new('Mapping schema unavailable')
    }
    $keyCount = 0; $shortcutCount = 0; $textCount = 0
    foreach ($section in @('remapKeys', 'remapShortcuts', 'remapKeysToText', 'remapShortcutsToText')) {
        if (-not $map.Contains($section)) { continue }
        if ($map[$section] -isnot [System.Collections.IDictionary]) {
            throw [System.NotSupportedException]::new('Mapping section schema unavailable')
        }
        foreach ($value in $map[$section].Values) {
            if ($value -isnot [array]) {
                throw [System.NotSupportedException]::new('Mapping collection schema unavailable')
            }
            foreach ($mapping in $value) {
                if ($mapping -isnot [System.Collections.IDictionary]) {
                    throw [System.NotSupportedException]::new('Mapping entry schema unavailable')
                }
            }
            if ($section -like '*ToText') { $textCount += $value.Count }
            elseif ($section -eq 'remapKeys') { $keyCount += $value.Count }
            else { $shortcutCount += $value.Count }
        }
    }
    [pscustomobject]@{ Enabled = $enabled; ProfileSelection = $selectionStatus; RemapStatus = 'ok'; KeyRemapCount = $keyCount; ShortcutRemapCount = $shortcutCount; TextRemapCount = $textCount }
}

function Get-KeyboardSnapshotEvent {
    param([int]$Minutes)
    $job = Start-ThreadJob -ArgumentList $Minutes -ScriptBlock {
        param($Lookback)
        try {
            Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = (Get-Date).AddMinutes(-$Lookback); ProviderName = @('Microsoft-Windows-Kernel-PnP', 'Microsoft-Windows-DriverFrameworks-UserMode', 'Microsoft-Windows-WHEA-Logger', 'Display', 'Microsoft-Windows-Kernel-Power') } -MaxEvents 100 -ErrorAction Stop |
                Select-Object @{ Name = 'Provider'; Expression = { $_.ProviderName } }, Id, Level, @{ Name = 'TimeUtc'; Expression = { $_.TimeCreated.ToUniversalTime().ToString('o') } }
        } catch {
            if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') { throw }
        }
    }
    try {
        if (-not (Wait-Job -Job $job -Timeout 8)) { throw [System.TimeoutException]::new('Event query timed out') }
        Receive-Job -Job $job -ErrorAction Stop
    } finally {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-KeyboardSnapshotReport {
    param([int]$Minutes = 60)
    $probes = @(
        Invoke-KeyboardSnapshotProbe 'Machine' { Get-KeyboardSnapshotMachine }
        Invoke-KeyboardSnapshotProbe 'Processes' { Get-KeyboardSnapshotProcess }
        Invoke-KeyboardSnapshotProbe 'Keyboards' { Get-KeyboardSnapshotDevice }
        Invoke-KeyboardSnapshotProbe 'Drivers' { Get-KeyboardSnapshotDriver }
        Invoke-KeyboardSnapshotProbe 'Filters' { Get-KeyboardSnapshotFilter }
        Invoke-KeyboardSnapshotProbe 'Accessibility' { Get-KeyboardSnapshotAccessibility }
        Invoke-KeyboardSnapshotProbe 'PowerToys' { Get-KeyboardSnapshotPowerToys }
        Invoke-KeyboardSnapshotProbe 'Events' { Get-KeyboardSnapshotEvent -Minutes $Minutes }
    )
    [pscustomobject]@{ SchemaVersion = '1.0'; CapturedAtUtc = [datetime]::UtcNow.ToString('o'); Mode = 'Snapshot'; SinceMinutes = $Minutes; Probes = $probes }
}

function Get-KeyboardDiagnosticSnapshot {
param([string]$OutputPath, [int]$SinceMinutes = 60, [switch]$DryRun, [switch]$Help)
if ($Help) {
    'Get-KeyboardDiagnosticSnapshot.ps1 [-SinceMinutes 60] [-OutputPath <new-local-json-file>] [-DryRun] [-h|--help]'
    return
}
if ($DryRun) {
    [pscustomobject]@{ SchemaVersion = '1.0'; Mode = 'DryRun'; Probes = @('Machine', 'Processes', 'Keyboards', 'Drivers', 'Filters', 'Accessibility', 'PowerToys', 'Events'); WritesRequested = [bool]$OutputPath }
    return
}
if ($OutputPath) {
    $fullPath = [System.IO.Path]::GetFullPath($OutputPath)
    $leaf = [System.IO.Path]::GetFileName($fullPath)
    if ($fullPath.StartsWith('\\') -or $leaf.TrimEnd('.', ' ') -eq '$null' -or $leaf -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.| |$)' -or $leaf.Contains(':')) { throw 'OutputPath must name a non-reserved local JSON file.' }
    if ([System.IO.Path]::GetExtension($fullPath) -ne '.json') { throw 'OutputPath must use .json.' }
    if (-not [System.IO.Directory]::Exists([System.IO.Path]::GetDirectoryName($fullPath))) { throw 'OutputPath parent must already exist.' }
    if ([System.IO.File]::Exists($fullPath)) { throw 'OutputPath already exists.' }
}
$report = Invoke-KeyboardSnapshotReport -Minutes $SinceMinutes
if ($OutputPath) {
    $json = $report | ConvertTo-Json -Depth 12
    # CreateNew closes the race between validation and creation; never overwrite.
    $stream = [System.IO.File]::Open($fullPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($json)
        $stream.Write($bytes, 0, $bytes.Length)
    } finally { $stream.Dispose() }
}
$report
}

Get-KeyboardDiagnosticSnapshot -OutputPath $OutputPath -SinceMinutes $SinceMinutes -DryRun:$DryRun -Help:$Help
