# Quick check of virtualization status
Write-Host "Docker Status:" -ForegroundColor Cyan
docker ps 2>&1 | Select-Object -First 3

Write-Host "`nWSL Distributions:" -ForegroundColor Cyan
wsl -l -v

Write-Host "`nHyper-V Status:" -ForegroundColor Cyan
Get-WindowsOptionalFeature -Online -FeatureName *Hyper* | Select-Object FeatureName, State

Write-Host "`nVirtualization Platform:" -ForegroundColor Cyan
Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform | Select-Object FeatureName, State

Write-Host "`nWSL Feature:" -ForegroundColor Cyan
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux | Select-Object FeatureName, State
