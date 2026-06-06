[CmdletBinding()]
param(
    [int]$SinceHours = 24
)

$ErrorActionPreference = 'Continue'
$reportRoot = Split-Path -Parent $PSCommandPath
$start = (Get-Date).AddHours(-$SinceHours)

function Save-Text {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(ValueFromPipeline)] $InputObject
    )
    process { $InputObject }
    end { }
}

$eventFilter = @{ LogName = @('System','Application'); StartTime = $start; Level = 1,2,3 }
$eventProviders = @(
    'Microsoft-Windows-HttpEvent',
    'Bonjour Service',
    'Microsoft-Windows-DistributedCOM',
    'Netwaw18',
    'Microsoft-Windows-Kernel-Power',
    'Microsoft-Windows-Kernel-PnP',
    'Microsoft-Windows-DriverFrameworks-UserMode',
    'Microsoft-Windows-FilterManager',
    'disk',
    'Application Error',
    'Windows Error Reporting'
)

Get-WinEvent -FilterHashtable $eventFilter -ErrorAction SilentlyContinue |
    Where-Object { $eventProviders -contains $_.ProviderName } |
    Sort-Object TimeCreated -Descending |
    Select-Object -First 160 TimeCreated,ProviderName,Id,LevelDisplayName,Message |
    ConvertTo-Json -Depth 6 |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'focused-events-24h.json')

@(
    'PCI\VEN_10DE&DEV_28B8&SUBSYS_223417AA&REV_A1\4&218D282A&0&0008',
    'ROOT\NET\0001'
) | ForEach-Object {
    $safe = ($_ -replace '[\\/:*?"<>|]', '_')
    pnputil /enum-devices /instanceid $_ |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot "pnputil-device-$safe.txt")
}

Get-CimInstance Win32_PnPSignedDriver |
    Where-Object { $_.DeviceClass -in @('DISPLAY','NET') -or $_.DeviceName -like '*NVIDIA*' -or $_.DeviceName -like '*AnyConnect*' -or $_.DeviceName -like '*Cisco*' } |
    Select-Object DeviceName,DeviceClass,Manufacturer,DriverProviderName,DriverVersion,DriverDate,InfName,IsSigned,DeviceID |
    Sort-Object DeviceClass,DeviceName |
    ConvertTo-Json -Depth 5 |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'signed-drivers-display-net.json')

pnputil /enum-drivers /class Display |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'pnputil-drivers-display.txt')

pnputil /enum-drivers /class Net |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'pnputil-drivers-net.txt')

Get-Process |
    Where-Object { $_.ProcessName -match 'iCloud|OneDrive|Dropbox|GoogleDrive|DriveFS' } |
    Select-Object ProcessName,Id,Path,StartTime |
    Format-List |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'sync-provider-processes.txt')

Get-Service |
    Where-Object { $_.Name -match 'Bonjour|Cisco|vpn|iCloud|OneDrive|Google|Dropbox|NVIDIA|nv' -or $_.DisplayName -match 'Bonjour|Cisco|VPN|iCloud|OneDrive|Google|Dropbox|NVIDIA' } |
    Select-Object Name,DisplayName,Status,StartType |
    Sort-Object Name |
    Format-Table -AutoSize |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'relevant-services.txt')

schtasks /query /fo LIST /v |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'scheduled-tasks-full-refresh.txt')
