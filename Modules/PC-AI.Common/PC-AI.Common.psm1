# PC-AI Common Module
. "$PSScriptRoot\Module-Common.ps1"

$publicPath = Join-Path $PSScriptRoot 'Public'
if (Test-Path $publicPath) {
    $sharedCacheHelper = Join-Path $publicPath 'Get-PcaiSharedCache.ps1'
    if (Test-Path $sharedCacheHelper) {
        . $sharedCacheHelper
    }
}
if (Test-Path $publicPath) {
    Get-ChildItem -Path $publicPath -Filter '*.ps1' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne 'Get-PcaiSharedCache.ps1' } |
        ForEach-Object {
        . $_.FullName
    }
}

Export-ModuleMember -Function @(
    'Write-Success',
    'Write-Warning',
    'Write-Error',
    'Write-Info',
    'Write-Header',
    'Write-SubHeader',
    'Write-Bullet',
    'Initialize-PcaiNative',
    'Get-PcaiAccelerationProbe',
    'Get-PcaiDirectCoreProbe',
    'Get-PcaiDirectTokenEstimate',
    'Get-PcaiDependencyStamp',
    'Get-PcaiSharedCacheEntry',
    'Set-PcaiSharedCacheEntry',
    'Clear-PcaiSharedCache',
    'Resolve-PcaiRepoRoot',
    'Get-PcaiRuntimeConfig',
    'Get-ScriptMetadata'
)
