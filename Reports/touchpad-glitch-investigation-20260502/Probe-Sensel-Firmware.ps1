# Probe Sensel touchpad firmware version + Lenovo Vantage state
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$out  = Join-Path $root '21-sensel-firmware-probe.txt'
"# Sensel firmware probe $(Get-Date -Format o)" | Set-Content $out

# 1. UEFI capsule firmware history (n48et.inf and other firmware capsules)
"`n## 1. UEFI Firmware capsule history (Get-WindowsDriver Class=Firmware)" | Add-Content $out
try {
    $fw = Get-WindowsDriver -Online -ErrorAction SilentlyContinue | Where-Object {
        $_.ClassName -eq 'Firmware' -or $_.OriginalFileName -match 'firmware|capsule|sensel|n48et|trackpad'
    }
    if ($fw) {
        $fw | Format-Table Driver, OriginalFileName, ProviderName, ClassName, Date, Version -AutoSize -Wrap | Out-String | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 2. Try to read Sensel HID feature reports for firmware version
"`n## 2. Sensel HID Feature Report probe (firmware version via HID API)" | Add-Content $out
$probeCsharp = @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class HidProbe {
    [DllImport("hid.dll")] static extern void HidD_GetHidGuid(out Guid guid);
    [DllImport("hid.dll")] [return:MarshalAs(UnmanagedType.U1)]
    static extern bool HidD_GetAttributes(IntPtr h, ref HIDD_ATTRIBUTES a);
    [DllImport("hid.dll")] [return:MarshalAs(UnmanagedType.U1)]
    static extern bool HidD_GetProductString(IntPtr h, byte[] buf, int len);
    [DllImport("hid.dll")] [return:MarshalAs(UnmanagedType.U1)]
    static extern bool HidD_GetManufacturerString(IntPtr h, byte[] buf, int len);
    [DllImport("hid.dll")] [return:MarshalAs(UnmanagedType.U1)]
    static extern bool HidD_GetSerialNumberString(IntPtr h, byte[] buf, int len);

    [StructLayout(LayoutKind.Sequential)]
    public struct HIDD_ATTRIBUTES {
        public int Size;
        public ushort VendorID;
        public ushort ProductID;
        public ushort VersionNumber;
    }

    public static string ProbeAll() {
        var sb = new System.Text.StringBuilder();
        Guid g; HidD_GetHidGuid(out g);
        sb.AppendLine("HID guid: " + g);
        // Note: opening a HID device for control transfers requires SetupDi enumeration -
        // simpler: just report HID guid; deeper probe requires P/Invoke into setupapi.
        return sb.ToString();
    }
}
'@
try {
    Add-Type -TypeDefinition $probeCsharp -ErrorAction SilentlyContinue 2>&1 | Out-Null
    $hidGuid = [Guid]::Empty
    "(C# probe registered. To get firmware version, query HID feature report 0xX from device — varies per Sensel firmware.)" | Add-Content $out
} catch {
    "(C# probe registration failed: $($_.Exception.Message))" | Add-Content $out
}

# 3. Lenovo Vantage / Commercial Vantage installation
"`n## 3. Lenovo Vantage / Commercial Vantage / System Interface Foundation versions" | Add-Content $out
$apps = @(
    'Lenovo Vantage Service',
    'Lenovo Vantage',
    'Lenovo Commercial Vantage',
    'System Interface Foundation',
    'Lenovo System Interface Foundation',
    'ImController',
    'Sensel Haptic Touchpad'
)
$installed = Get-Package -ErrorAction SilentlyContinue | Where-Object {
    $name = $_.Name
    $apps | ForEach-Object { if ($name -like "*$_*") { return $true } }
}
if ($installed) {
    $installed | Sort-Object Name | Format-Table Name, Version, ProviderName -AutoSize | Out-String | Add-Content $out
}
# Also AppX
$appx = Get-AppxPackage -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match 'Lenovo|Sensel|HapticTouchpad|Vantage|CommercialVantage'
}
"`n### AppX packages" | Add-Content $out
if ($appx) {
    $appx | Format-Table Name, PackageFullName, Version, Publisher -AutoSize -Wrap | Out-String | Add-Content $out
}

# 4. Lenovo update history
"`n## 4. Lenovo Setup logs / Vantage update history" | Add-Content $out
$lenovoLogs = @(
    "$env:ProgramData\Lenovo\ImController\Logs",
    "$env:ProgramData\Lenovo\Vantage\Logs",
    "$env:ProgramData\Lenovo\SystemUpdate\Logs",
    "$env:LOCALAPPDATA\Lenovo\Vantage\Logs"
)
foreach ($d in $lenovoLogs) {
    if (Test-Path $d) {
        "Found: $d" | Add-Content $out
        Get-ChildItem $d -Recurse -File -Filter *.log -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 10 |
            Format-Table FullName, LastWriteTime, Length -AutoSize -Wrap | Out-String | Add-Content $out
    }
}

# 5. Lenovo Vantage update events
"`n## 5. Lenovo / firmware events in Setup log last 30d" | Add-Content $out
try {
    $cutoff = (Get-Date).AddDays(-30)
    $setupEvts = Get-WinEvent -LogName Setup -ErrorAction SilentlyContinue -MaxEvents 2000 |
        Where-Object { $_.TimeCreated -gt $cutoff -and $_.Message -match 'Lenovo|Sensel|firmware|n48et|capsule|touchpad' }
    if ($setupEvts) {
        "Found $($setupEvts.Count) events" | Add-Content $out
        $setupEvts | Sort-Object TimeCreated | ForEach-Object {
            $msg = ($_.Message -split [Environment]::NewLine | Select-Object -First 1)
            "$($_.TimeCreated) [Id $($_.Id)] $msg"
        } | Select-Object -First 50 | Add-Content $out
    } else {
        "(no Lenovo/firmware events in Setup log last 30d)" | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 6. n48et firmware bound version
"`n## 6. n48et firmware (Lenovo touchpad firmware capsule) - currently bound" | Add-Content $out
try {
    $fwDevices = Get-PnpDevice -Class Firmware -ErrorAction SilentlyContinue
    foreach ($d in $fwDevices) {
        $props = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName `
            'DEVPKEY_Device_DriverDate','DEVPKEY_Device_DriverVersion','DEVPKEY_Device_DriverInfPath','DEVPKEY_Device_HardwareIds' -ErrorAction SilentlyContinue
        $info = "$($d.FriendlyName) | $($d.InstanceId)"
        $props | ForEach-Object { $info += " | $($_.KeyName)=$($_.Data)" }
        $info | Add-Content $out
    }
} catch {
    "ERROR: $($_.Exception.Message)" | Add-Content $out
}

# 7. UEFI/BIOS firmware version
"`n## 7. System BIOS/UEFI version" | Add-Content $out
$bios = Get-CimInstance Win32_BIOS -ErrorAction SilentlyContinue
if ($bios) {
    "Manufacturer : $($bios.Manufacturer)" | Add-Content $out
    "Name         : $($bios.Name)" | Add-Content $out
    "Version      : $($bios.Version)" | Add-Content $out
    "SMBIOSVersion: $($bios.SMBIOSBIOSVersion)" | Add-Content $out
    "ReleaseDate  : $($bios.ReleaseDate)" | Add-Content $out
}

"`n# Done $(Get-Date -Format o)" | Add-Content $out
Write-Host "Wrote $out"
