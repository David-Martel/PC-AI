[CmdletBinding()]
param()

$reportRoot = Split-Path -Parent $PSCommandPath

$patterns = 'Windows Admin|Admin Center|ServerManagement|Gateway|Web Management|WinRM|WMSvc|Management'

Get-Service |
    Where-Object { $_.Name -match $patterns -or $_.DisplayName -match $patterns } |
    Select-Object Name,DisplayName,Status,StartType |
    Sort-Object Name |
    Format-Table -AutoSize |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'http50443-services-candidates.txt')

$uninstallRoots = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall'
)

$apps = foreach ($root in $uninstallRoots) {
    Get-ChildItem -LiteralPath $root -ErrorAction SilentlyContinue | ForEach-Object {
        $item = Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue
        if ($item.DisplayName -match $patterns -or $item.Publisher -match $patterns -or $item.InstallLocation -match $patterns) {
            [PSCustomObject]@{
                DisplayName = $item.DisplayName
                DisplayVersion = $item.DisplayVersion
                Publisher = $item.Publisher
                InstallLocation = $item.InstallLocation
                UninstallString = $item.UninstallString
                RegistryKey = $_.Name
            }
        }
    }
}

$apps |
    Sort-Object DisplayName |
    ConvertTo-Json -Depth 5 |
    Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot 'http50443-installed-app-candidates.json')

$stores = @(
    'Cert:\LocalMachine\Windows Web Management',
    'Cert:\CurrentUser\Windows Web Management',
    'Cert:\LocalMachine\My',
    'Cert:\CurrentUser\My'
)

foreach ($store in $stores) {
    if (Test-Path -LiteralPath $store) {
        Get-ChildItem -LiteralPath $store -ErrorAction SilentlyContinue |
            Select-Object Subject,Issuer,Thumbprint,NotBefore,NotAfter,FriendlyName,EnhancedKeyUsageList |
            ConvertTo-Json -Depth 5 |
            Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot (('cert-' + ($store -replace '[:\\ ]','_') + '.json')))
    }
}

@('127.0.0.1:50443','[::1]:50443') | ForEach-Object {
    $safe = $_ -replace '[\\/:*?"<>|]','_'
    netsh http show sslcert ipport=$_ |
        Out-File -Encoding utf8 -LiteralPath (Join-Path $reportRoot "netsh-http-sslcert-$safe.txt")
}
