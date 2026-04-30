# Fix Network Drive Timeouts for WSL
# Sets ProviderFlags=1 to prevent slow WSL startup

$drives = Get-ChildItem 'HKCU:\Network' -ErrorAction SilentlyContinue
if ($drives) {
    foreach ($drive in $drives) {
        try {
            New-ItemProperty -Path $drive.PSPath -Name 'ProviderFlags' -Value 1 -PropertyType DWORD -Force -ErrorAction Stop | Out-Null
            Write-Host "Set ProviderFlags=1 for drive $($drive.PSChildName)"
        } catch {
            Write-Host "Note: $($drive.PSChildName) - $_"
        }
    }
} else {
    Write-Host 'No network drives found in HKCU:\Network - no changes needed'
}
