# Apply-PowerFix.ps1
# Disables EnhancedPowerManagement on the I2C HID parent for the Sensel touchpad.
# This stops aggressive D-state cycling that causes skip/pause/freeze on resume.
#
# Fully reversible: -Restore reapplies the captured pre-fix value.
[CmdletBinding()]
param(
    [switch]$Restore
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$instances = @(
    'ACPI\SNSL002D\4&39979B3E&0'
    # NOTE: deliberately NOT touching I2C controller (would affect audio TIAS2781 on same bus)
    # NOTE: deliberately NOT touching child HID collections (they inherit from parent)
)
$rollbackFile = Join-Path $root 'powerfix-rollback.json'

function Get-DeviceParamPath {
    param([string]$instanceId)
    "Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\$instanceId\Device Parameters"
}

if ($Restore) {
    Write-Host "==== RESTORE MODE ===="
    if (-not (Test-Path $rollbackFile)) {
        Write-Error "No rollback file found at $rollbackFile"
        return
    }
    $rb = Get-Content $rollbackFile -Raw | ConvertFrom-Json
    foreach ($entry in $rb) {
        $path = Get-DeviceParamPath $entry.InstanceId
        Write-Host "Restoring $($entry.InstanceId)..."
        foreach ($prop in $entry.PSObject.Properties) {
            if ($prop.Name -in 'InstanceId','CapturedAt') { continue }
            $name = $prop.Name
            $val  = $prop.Value
            if ($null -eq $val) {
                # Was unset originally; remove if present now
                Remove-ItemProperty -Path $path -Name $name -ErrorAction SilentlyContinue
                Write-Host "  $name -> (removed, was unset originally)"
            } else {
                Set-ItemProperty -Path $path -Name $name -Value $val -Type DWord
                Write-Host "  $name -> $val"
            }
        }
        # Restart device to apply
        & pnputil /restart-device "$($entry.InstanceId)" | Out-Null
        Write-Host "  Device restarted"
    }
    Write-Host "==== RESTORE COMPLETE ===="
    return
}

# Capture current state for rollback
Write-Host "==== APPLY MODE ===="
Write-Host "Capturing rollback state..."
$rollback = foreach ($i in $instances) {
    $path = Get-DeviceParamPath $i
    if (-not (Test-Path $path)) {
        Write-Warning "  $i : Device Parameters key does not exist; skipping"
        continue
    }
    $epm = Get-ItemProperty -Path $path -Name 'EnhancedPowerManagementEnabled' -ErrorAction SilentlyContinue
    $ssEn = Get-ItemProperty -Path $path -Name 'SelectiveSuspendEnabled' -ErrorAction SilentlyContinue

    # Get LastWriteTime of the key for forensics
    $regKey = Get-Item -Path $path -ErrorAction SilentlyContinue
    $lastWrite = $null
    if ($regKey) {
        $sigType  = [Microsoft.Win32.RegistryKey]
        $hkey     = [Microsoft.Win32.Registry]::LocalMachine
        $subPath  = "SYSTEM\CurrentControlSet\Enum\$i\Device Parameters"
        try {
            $sub = $hkey.OpenSubKey($subPath)
            if ($sub) {
                $hk = [Microsoft.Win32.RegistryKey].GetField('hkey','NonPublic,Instance').GetValue($sub)
                # not portable; skip
            }
        } catch { }
    }

    [pscustomobject]@{
        InstanceId = $i
        EnhancedPowerManagementEnabled = if ($epm) { $epm.EnhancedPowerManagementEnabled } else { $null }
        SelectiveSuspendEnabled        = if ($ssEn) { $ssEn.SelectiveSuspendEnabled } else { $null }
        CapturedAt = (Get-Date).ToString('o')
    }
}

$rollback | ConvertTo-Json -Depth 4 | Set-Content $rollbackFile
Write-Host "Rollback captured to $rollbackFile"
foreach ($r in $rollback) {
    Write-Host "  $($r.InstanceId): EnhancedPowerManagementEnabled=$($r.EnhancedPowerManagementEnabled) SelectiveSuspendEnabled=$($r.SelectiveSuspendEnabled)"
}

# Apply fix
Write-Host "`nApplying fix..."
foreach ($i in $instances) {
    $path = Get-DeviceParamPath $i
    if (-not (Test-Path $path)) { continue }

    Set-ItemProperty -Path $path -Name 'EnhancedPowerManagementEnabled' -Value 0 -Type DWord
    Write-Host "  $i : EnhancedPowerManagementEnabled -> 0"

    # Verify
    $verify = Get-ItemProperty -Path $path -Name 'EnhancedPowerManagementEnabled'
    if ($verify.EnhancedPowerManagementEnabled -ne 0) {
        Write-Warning "  $i : VERIFY FAILED - value is $($verify.EnhancedPowerManagementEnabled)"
    } else {
        Write-Host "  $i : verified = 0"
    }
}

# Restart the I2C HID device to apply
Write-Host "`nRestarting devices to apply..."
foreach ($i in $instances) {
    Write-Host "  pnputil /restart-device `"$i`""
    $output = & pnputil /restart-device "$i" 2>&1
    $output | ForEach-Object { Write-Host "    $_" }
}

Write-Host "`n==== APPLY COMPLETE ===="
Write-Host "Test the touchpad now. To restore: & '$PSCommandPath' -Restore"
