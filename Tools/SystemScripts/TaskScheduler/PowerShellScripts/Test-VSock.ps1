# Test-VSock.ps1
try {
    $ModulePath = Join-Path ([System.Environment]::GetFolderPath("MyDocuments")) "PowerShell\Modules\WslExtensions\WslExtensions.psm1"
    Import-Module $ModulePath -Force -ErrorAction Stop
    Write-Host "Module imported successfully."

    # Test Type Definition
    if ([WslExtensions.VSock]) {
        Write-Host "VSock Type found."
    }

    # Test Port ID Generation
    $guid = [WslExtensions.VSock]::GetServiceId(5000)
    Write-Host "Service ID for port 5000: $guid"

    # Test Listener Creation (Bind only, don't block)
    $listener = [WslExtensions.VSock]::CreateListener(5001)
    Write-Host "Listener created on 5001."
    $listener.Close()
    Write-Host "Listener closed."

} catch {
    Write-Error $_
}
