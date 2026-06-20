#Requires -Version 7.0
<#
.SYNOPSIS
    Disables reversible Sensel touchpad / Intel I2C power-down paths.

.DESCRIPTION
    Applies the targeted touchpad fix for the ThinkPad P1 Gen 7 input-stickiness
    investigation.  The previous live snapshots showed the Sensel touchpad path
    (SNSL002D) and Intel Serial IO I2C controller (7E78) with
    MSPower_DeviceEnable.Enable = true, meaning Windows is allowed to power the
    devices down.  That is a plausible cause for intermittent pointer or finger
    press stickiness after idle, sleep, dock/eGPU churn, or high system load.

    Later haptic-touchpad captures also showed repeated WUDFHost timeouts and
    WUDFRd load warnings for the nearby Elliptic human-presence device
    (ACPI\VEN_ELAS&DEV_B41A), while that device still allowed power-down.  Use
    -IncludeHumanPresenceSensor to apply the same reversible power-down hardening
    to that sensor path.  It is opt-in because it may affect presence-detection
    behavior such as walk-away lock or wake-on-approach.

    Forward mode:
      - Writes a timestamped JSON backup under .\backups unless -BackupFile is set.
      - Sets MSPower_DeviceEnable.Enable = false for SNSL002D and 7E78 entries.
      - Sets EnhancedPowerManagementEnabled = 0 on matching Enum device keys.
      - Sets SelectiveSuspendEnabled and AllowIdleIrpInD3 to 0 when already present.

    Revert mode restores the WMI and registry values captured in the backup.

    This script intentionally does not reset devices by default.  Use
    -RestartDevices only when you accept a brief touchpad/I2C interruption.

.PARAMETER Revert
    Restore values from the newest touchpad-power backup, or from -BackupFile.

.PARAMETER BackupFile
    Backup JSON to write in forward mode, or to read in -Revert mode.

.PARAMETER BackupDir
    Directory for generated backup JSON files.

.PARAMETER RestartDevices
    Restart matching PnP devices after applying changes.

.PARAMETER IncludeHumanPresenceSensor
    Also target the Elliptic human-presence sensor path ACPI\VEN_ELAS&DEV_B41A.

.EXAMPLE
    pwsh -File .\Repair-TouchpadPowerManagement.ps1 -WhatIf

.EXAMPLE
    pwsh -File .\Repair-TouchpadPowerManagement.ps1

.EXAMPLE
    pwsh -File .\Repair-TouchpadPowerManagement.ps1 -IncludeHumanPresenceSensor

.EXAMPLE
    pwsh -File .\Repair-TouchpadPowerManagement.ps1 -Revert

.NOTES
    Requires elevation because it writes HKLM Enum keys and root\wmi power state.
    Full effect may require sign out/in or reboot if the driver has cached state.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
param(
    [switch]$Revert,
    [string]$BackupFile,
    [string]$BackupDir = (Join-Path $PSScriptRoot 'backups'),
    [switch]$RestartDevices,
    [switch]$IncludeHumanPresenceSensor
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$BaseTargetPatterns = @('SNSL002D', 'VEN_8086&DEV_7E78')
$HumanPresenceTargetPatterns = @('VEN_ELAS&DEV_B41A')
$TargetPatterns = if ($IncludeHumanPresenceSensor) {
    @($BaseTargetPatterns + $HumanPresenceTargetPatterns)
} else {
    @($BaseTargetPatterns)
}
$RegistryValueNames = @(
    'EnhancedPowerManagementEnabled',
    'SelectiveSuspendEnabled',
    'AllowIdleIrpInD3'
)

function Test-IsElevated {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function ConvertTo-EnumInstanceId {
    param([Parameter(Mandatory)] [string]$InstanceName)
    $candidate = $InstanceName
    if ($candidate -match '(.+)_\d+$') {
        $candidate = $Matches[1]
    }
    return $candidate
}

function Test-IsTargetInstance {
    param([Parameter(Mandatory)] [string]$Text)
    foreach ($pattern in $TargetPatterns) {
        if ($Text -like "*$pattern*") {
            return $true
        }
    }
    return $false
}

function Get-TargetPowerEntries {
    @(Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable -ErrorAction Stop |
        Where-Object { Test-IsTargetInstance -Text $_.InstanceName } |
        Sort-Object InstanceName)
}

function Get-TargetEnumInstanceIds {
    $ids = [System.Collections.Generic.List[string]]::new()

    foreach ($entry in Get-TargetPowerEntries) {
        $id = ConvertTo-EnumInstanceId -InstanceName $entry.InstanceName
        if ($id -and -not $ids.Contains($id)) {
            $ids.Add($id)
        }
    }

    try {
        Get-CimInstance -ClassName Win32_PnPEntity -ErrorAction Stop |
            Where-Object { $_.PNPDeviceID -and (Test-IsTargetInstance -Text $_.PNPDeviceID) } |
            ForEach-Object {
                if (-not $ids.Contains($_.PNPDeviceID)) {
                    $ids.Add($_.PNPDeviceID)
                }
            }
    } catch {
        Write-Verbose "Win32_PnPEntity target discovery failed: $_"
    }

    @($ids | Sort-Object -Unique)
}

function Get-RegistryValueState {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Name
    )

    $prop = Get-ItemProperty -Path $Path -Name $Name -ErrorAction SilentlyContinue
    $exists = $null -ne $prop -and $null -ne $prop.$Name
    [ordered]@{
        Name          = $Name
        ExistedBefore = $exists
        PriorValue    = if ($exists) { $prop.$Name } else { $null }
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)] [object]$InputObject,
        [Parameter(Mandatory)] [string]$Path
    )
    $InputObject | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding UTF8
}

function Resolve-BackupForRevert {
    if ($BackupFile) {
        return $BackupFile
    }

    $candidate = Get-ChildItem -Path $BackupDir -Filter 'touchpad-power-*.json' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $candidate) {
        throw "No touchpad-power backup files found in '$BackupDir'. Specify -BackupFile."
    }

    return $candidate.FullName
}

if (-not (Test-IsElevated)) {
    throw "This script must run as Administrator. Re-run from an elevated PowerShell 7 prompt."
}

if (-not (Test-Path -LiteralPath $BackupDir)) {
    if ($WhatIfPreference) {
        Write-Host "[WhatIf] Backup directory would be created: $BackupDir" -ForegroundColor Cyan
    } else {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    }
}

if ($Revert) {
    $resolvedBackup = Resolve-BackupForRevert
    if (-not (Test-Path -LiteralPath $resolvedBackup)) {
        throw "Backup file not found: $resolvedBackup"
    }

    Write-Host "Reverting touchpad power-management state from: $resolvedBackup" -ForegroundColor Cyan
    $backup = Get-Content -LiteralPath $resolvedBackup -Raw | ConvertFrom-Json

    foreach ($entry in @($backup.PowerEntries)) {
        $target = Get-CimInstance -Namespace root\wmi -ClassName MSPower_DeviceEnable -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceName -eq $entry.InstanceName } |
            Select-Object -First 1

        if (-not $target) {
            Write-Warning "Power entry no longer exists: $($entry.InstanceName)"
            continue
        }

        $prior = [bool]$entry.PriorEnable
        if ($PSCmdlet.ShouldProcess($entry.InstanceName, "Restore MSPower_DeviceEnable.Enable = $prior")) {
            Set-CimInstance -InputObject $target -Property @{ Enable = $prior } | Out-Null
        }
        Write-Host "  WMI $($entry.InstanceName) -> Enable=$prior"
    }

    foreach ($device in @($backup.RegistryDevices)) {
        $devicePath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($device.InstanceId)\Device Parameters"
        if (-not (Test-Path -LiteralPath $devicePath)) {
            Write-Warning "Device Parameters key no longer exists: $devicePath"
            continue
        }

        foreach ($value in @($device.Values)) {
            if ($value.ExistedBefore) {
                if ($PSCmdlet.ShouldProcess("$devicePath\$($value.Name)", "Restore DWord = $($value.PriorValue)")) {
                    Set-ItemProperty -Path $devicePath -Name $value.Name -Value ([int]$value.PriorValue) -Type DWord
                }
                Write-Host "  $($device.InstanceId) $($value.Name) -> $($value.PriorValue)"
            } else {
                if ($PSCmdlet.ShouldProcess("$devicePath\$($value.Name)", 'Remove value created by this script')) {
                    Remove-ItemProperty -Path $devicePath -Name $value.Name -ErrorAction SilentlyContinue
                }
                Write-Host "  $($device.InstanceId) $($value.Name) removed"
            }
        }
    }

    Write-Host "Revert complete. Sign out/in or reboot if a driver cached the old state." -ForegroundColor Green
    return
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $BackupFile) {
    $BackupFile = Join-Path $BackupDir "touchpad-power-$stamp.json"
}

$powerEntries = [System.Collections.Generic.List[object]]::new()
foreach ($entry in Get-TargetPowerEntries) {
    $powerEntries.Add([ordered]@{
        InstanceName = $entry.InstanceName
        PriorEnable  = [bool]$entry.Enable
    })
}

$registryDevices = [System.Collections.Generic.List[object]]::new()
foreach ($id in Get-TargetEnumInstanceIds) {
    $devicePath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$id\Device Parameters"
    if (-not (Test-Path -LiteralPath $devicePath)) {
        Write-Verbose "No Device Parameters key for $id"
        continue
    }

    $values = [System.Collections.Generic.List[object]]::new()
    foreach ($name in $RegistryValueNames) {
        $values.Add((Get-RegistryValueState -Path $devicePath -Name $name))
    }

    $registryDevices.Add([ordered]@{
        InstanceId = $id
        Values     = $values
    })
}

$backup = [ordered]@{
    Timestamp                  = (Get-Date).ToString('o')
    Machine                    = $env:COMPUTERNAME
    Script                     = 'Repair-TouchpadPowerManagement.ps1'
    IncludeHumanPresenceSensor = [bool]$IncludeHumanPresenceSensor
    TargetPatterns             = $TargetPatterns
    PowerEntries               = $powerEntries
    RegistryDevices            = $registryDevices
}

if (-not $WhatIfPreference) {
    Write-JsonFile -InputObject $backup -Path $BackupFile
    Write-Host "Backup written: $BackupFile" -ForegroundColor Cyan
} else {
    Write-Host "[WhatIf] Backup would be written: $BackupFile" -ForegroundColor Cyan
}

foreach ($entry in Get-TargetPowerEntries) {
    $beforeEnable = [bool]$entry.Enable
    if ($PSCmdlet.ShouldProcess($entry.InstanceName, 'Set MSPower_DeviceEnable.Enable = false')) {
        Set-CimInstance -InputObject $entry -Property @{ Enable = $false } | Out-Null
    }
    Write-Host "  WMI $($entry.InstanceName) Enable $beforeEnable -> false"
}

foreach ($device in $registryDevices) {
    $devicePath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($device.InstanceId)\Device Parameters"
    Write-Host "  Registry $($device.InstanceId)" -ForegroundColor Yellow

    foreach ($value in $device.Values) {
        $shouldWrite = $value.Name -eq 'EnhancedPowerManagementEnabled' -or $value.ExistedBefore
        if (-not $shouldWrite) {
            Write-Host "    $($value.Name): absent; left absent"
            continue
        }

        if ($PSCmdlet.ShouldProcess("$devicePath\$($value.Name)", 'Set DWord = 0')) {
            Set-ItemProperty -Path $devicePath -Name $value.Name -Value 0 -Type DWord
        }
        $before = if ($value.ExistedBefore) { $value.PriorValue } else { '(not set)' }
        Write-Host "    $($value.Name): $before -> 0"
    }
}

if ($RestartDevices) {
    foreach ($id in Get-TargetEnumInstanceIds) {
        if ($PSCmdlet.ShouldProcess($id, 'Restart PnP device with pnputil')) {
            pnputil /restart-device "$id" | Out-Host
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "pnputil failed restarting $id with exit code $LASTEXITCODE"
            }
        }
    }
} else {
    Write-Host "Device restart skipped. Use -RestartDevices or reboot/sign out if the symptom persists." -ForegroundColor Yellow
}

if ($WhatIfPreference) {
    Write-Host "Touchpad power-management fix preview complete." -ForegroundColor Green
} else {
    Write-Host "Touchpad power-management fix applied." -ForegroundColor Green
}
